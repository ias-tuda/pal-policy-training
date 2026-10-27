"""Plan S11's hand-off: up, out, then home (`pick-place-v1-objects-plan.md`, *S11*, BS11's hand-off).

Once an arm has placed its cube, a position controller takes it over for the rest of the episode:
the TCP goes straight up `HANDOFF_LIFT`, then horizontally out to the arm's own side (|y| >=
`HANDOFF_OUT_Y`, the robot frame), then the joints go home at the slew; the jaw opens. S9's read
saw the eval's straight return home knock 1-18 towers per 64 (with a 10 cm lift first), and on two
arms the column above A has to be empty before the right hand comes down on it.

The controller is a resolved-rate one, batched on the simulator's own kinematics: each control
step it asks for a TCP velocity -- `HANDOFF_SPEED` along the leg, a proportional hold on the other
axes and on the hand's orientation -- and turns it into a joint step with the arm's geometric
Jacobian (joint anchors and axes from the simulator, as `ToolSpeedCappedJointPositionAction`
reads them) by damped least squares, scaled down as a whole until every joint is within its slew.
The step is added to the arm's last commanded target and sent through the action term exactly as a
policy's target would be -- the slew, plan S7-4's T band and C clip, the jaw servo all apply -- so
what the controller can do is what a deployer driving the same targets can do. No IK, no CPU work,
so JOINT can run the left's hand-off inside training on every env (`HANDOFF_TRAIN=left`).

Who starts it: with an arm's ``trigger`` the term watches it itself (training) -- the jaw opened
after the lift (plan S7-4's J, the arm's release latch) and the TCP at least `HANDOFF_CLEAR` over
the placed cube's top for `HANDOFF_CLEAR_STEPS` steps, or the episode reaching ``timeout_s`` --
as `scripts/eval_pick_place.py`'s ``--handoff`` does; without one, `start_handoff` starts it
(the eval's own trigger).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_box_minus

from pal_mjlab.tasks.manipulation.mdp.actions import (
  RateLimitedJointPositionAction,
  RateLimitedJointPositionActionCfg,
)

#: The legs of the route (the plan's numbers): straight up this far from where the hand was, m...
HANDOFF_LIFT = 0.10
#: ... then out until the TCP's |y| in the robot frame is at least this, m (the arm's own side).
HANDOFF_OUT_Y = 0.20
#: TCP speed asked for along a leg, m/s (the slew may hold it lower); `eval_pick_place.LIFT_SPEED`
#: is 0.2 for its IK lift.
HANDOFF_SPEED = 0.15
#: Gains holding the axes a leg does not move along, 1/s, and the hand's orientation, 1/s.
HANDOFF_POS_GAIN = 2.0
HANDOFF_ROT_GAIN = 2.0
#: Damping of the least-squares step (m, rad): keeps the step bounded near a singularity.
HANDOFF_DLS = 0.05
#: A leg that has not arrived in this long hands over to the next, s (a joint at its range clip or
#: T's band can stall a leg; going home is always possible).
HANDOFF_LEG_TIMEOUT_S = 3.0
#: The leg counts as arrived this close, m.
HANDOFF_TOL = 0.005
#: The trigger, as the eval's (`eval_pick_place.HANDOFF_CLEAR` / `HANDOFF_STEPS`).
HANDOFF_CLEAR = 0.03
HANDOFF_CLEAR_STEPS = 5

#: Plan S9.3's park poses (`HANDOFF_PARK`): where the last leg ends, as offsets (rad) on the arm's 7
#: joints from its home. ``home`` is every run before S9.3. ``elbow`` bends arm 4 a further 0.2 rad:
#: the TCP 11 cm higher, and >= 7.6 cm from a standing tower anywhere on WC at a table of 0.827-0.847 m,
#: where the home pose's fingers reach 4.8 cm into one (S9.3 step 0 item 3,
#: `results/S9.3-wide/step0/`).
HANDOFF_PARK_POSES: dict[str, tuple[float, ...] | None] = {
  "home": None,
  "elbow": (0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0),
}

#: The phases, per env and arm.
POLICY, UP, OUT, HOME = 0, 1, 2, 3

#: Attribute on the env: per arm name, the envs whose next episode starts with that arm already
#: handed off and home (B,) -- set by a reset event (plan S11's BI-JM), read by the term's reset,
#: which mjlab runs after the reset events.
START_HOME = "_pp_handoff_start_home"
#: Attribute on the env: plan S11.2's hold starts (`bimanual.hold_start`), set by a reset event --
#: ``mask`` (B,) the envs that start from a snapshot, ``target`` (B, n) the action term's last target and
#: ``servo`` (B, n_servo) its jaw servos there, ``clock`` (B,) the episode step to resume at and
#: ``goal_offset`` (B, 2) T's offset from the env origin then. The term's reset puts the targets back (it runs
#: after the events); the clock and T are set on the first step after the reset, because mjlab zeroes
#: ``episode_length_buf`` and the command manager redraws T after every term's reset.
START_STATE = "_pp_handoff_start_state"


@dataclass(frozen=True)
class HandoffTriggerCfg:
  """What an arm's own trigger watches (training): its release latch's parameters, the cube it
  placed, and the episode time at which it hands off regardless."""

  command_name: str
  object_name: str
  grasp_sensor_name: str
  support_sensor_name: str
  surface_cfg: SceneEntityCfg
  jaw_cfg: SceneEntityCfg
  jaw_open: float
  timeout_s: float


@dataclass(frozen=True)
class HandoffArmCfg:
  """One arm the controller can take over."""

  name: str
  arm_joints: tuple[str, ...]
  """The arm's 7 actuator names (exact, arm 1-7), in the action term's joint naming."""
  jaw_joint: str
  tcp_site: str
  side: float
  """+1 when the arm's own side is +y (the left arm), -1 when it is -y (the right)."""
  trigger: HandoffTriggerCfg | None = None
  """The arm's own trigger (training); None: `start_handoff` starts it (the eval)."""
  park: tuple[float, ...] | None = None
  """Offsets (rad) on the 7 arm joints from home where the last leg ends (`HANDOFF_PARK_POSES`);
  None: home."""


class HandoffJointPositionAction(RateLimitedJointPositionAction):
  """`RateLimitedJointPositionAction` with plan S11's up-out-home controller on named arms.

  While an arm is handed off (``phase`` > `POLICY`), its 7 joints and its jaw get the controller's
  targets instead of the policy's: the policy's action columns for them are replaced before the
  term processes them (absolute targets: ``(q* - offset) / scale``), so the slew and plan S7-4's
  bounds act on them as on any target. The policy keeps its other columns. ``phase`` and
  ``handed_step`` (the episode step it started, -1 if not) are per env and arm, cleared at a
  reset.
  """

  cfg: HandoffJointPositionActionCfg

  def __init__(self, cfg: HandoffJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    names = [n.split("/")[-1] for n in self._target_names]
    entity = self._entity
    self._arms = []
    for arm in cfg.arms:
      if arm.trigger is not None:
        # Not a term parameter, so no manager resolves them; resolving is idempotent.
        arm.trigger.surface_cfg.resolve(env.scene)
        arm.trigger.jaw_cfg.resolve(env.scene)
      missing = [j for j in (*arm.arm_joints, arm.jaw_joint) if j not in names]
      if missing:
        raise ValueError(f"hand-off arm {arm.name!r}: {missing} not in {names}")
      if arm.park is not None and len(arm.park) != len(arm.arm_joints):
        raise ValueError(f"hand-off arm {arm.name!r}: park has {len(arm.park)} offsets")
      idx = torch.tensor(
        [names.index(j) for j in arm.arm_joints], device=self.device, dtype=torch.long
      )
      self._arms.append(
        {
          "cfg": arm,
          "idx": idx,
          "jaw": names.index(arm.jaw_joint),
          "joints": entity.indexing.joint_ids[self._target_ids[idx]],
          "site": entity.site_names.index(arm.tcp_site),
          "park": None
          if arm.park is None
          else torch.tensor(arm.park, device=self.device, dtype=torch.float),
        }
      )
    n, k = self.num_envs, len(self._arms)
    self.phase = torch.zeros(n, k, dtype=torch.long, device=self.device)
    self.handed_step = torch.full((n, k), -1, dtype=torch.long, device=self.device)
    #: Whether an arm's hand-off was started by its trigger's time-out rather than its release and lift (plan S11.2:
    #: a timed-out hand-off pays no release, leave or success level).
    self.by_timeout = torch.zeros(n, k, dtype=torch.bool, device=self.device)
    self._leg_steps = torch.zeros(n, k, dtype=torch.long, device=self.device)
    self._clear_steps = torch.zeros(n, k, dtype=torch.long, device=self.device)
    self._hold_pos = torch.zeros(n, k, 3, device=self.device)
    self._hold_quat = torch.zeros(n, k, 4, device=self.device)
    self._hold_quat[..., 0] = 1.0
    self._leg_timeout = round(HANDOFF_LEG_TIMEOUT_S / env.step_dt)
    self._dt = env.step_dt

  def arm_index(self, arm: int | str) -> int:
    if isinstance(arm, int):
      return arm
    for k, a in enumerate(self._arms):
      if a["cfg"].name == arm:
        return k
    raise KeyError(f"no hand-off arm {arm!r}")

  def _tcp(self, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The arm's TCP in the env frame (the robot frame: ``base_link`` at the origin) and its world
    orientation. (B, 3), (B, 4)"""
    site = self._arms[k]["site"]
    origins = self._env.scene.env_origins
    return (
      self._entity.data.site_pos_w[:, site] - origins,
      self._entity.data.site_quat_w[:, site],
    )

  def start_handoff(self, arm: int | str, mask: torch.Tensor) -> None:
    """Hand ``arm`` over to the controller in the envs of ``mask`` (B,) that have not been: the
    route starts from where the TCP is now."""
    k = self.arm_index(arm)
    fresh = mask & (self.phase[:, k] == POLICY)
    if not bool(fresh.any()):
      return
    pos, quat = self._tcp(k)
    self._hold_pos[:, k] = torch.where(fresh.unsqueeze(-1), pos, self._hold_pos[:, k])
    self._hold_quat[:, k] = torch.where(
      fresh.unsqueeze(-1), quat, self._hold_quat[:, k]
    )
    self.phase[:, k] = torch.where(fresh, UP, self.phase[:, k])
    self._leg_steps[:, k] = torch.where(fresh, 0, self._leg_steps[:, k])
    self.handed_step[:, k] = torch.where(
      fresh, self._env.episode_length_buf, self.handed_step[:, k]
    )

  def set_home(self, arm: int | str, env_ids: torch.Tensor) -> None:
    """Put ``arm`` straight into the last leg for ``env_ids`` -- already handed off, at home (plan
    S11's BI-JM: episodes that start with A placed and the left arm gone)."""
    k = self.arm_index(arm)
    self.phase[env_ids, k] = HOME
    self.handed_step[env_ids, k] = 0

  def _auto_trigger(self) -> None:
    from pal_mjlab.tasks.manipulation.mdp.observations import _release_latch
    from pal_mjlab.tasks.manipulation.mdp.shapes import object_half_height

    env = self._env
    for k, arm in enumerate(self._arms):
      t = arm["cfg"].trigger
      if t is None:
        continue
      latch = _release_latch(
        env,
        t.command_name,
        t.object_name,
        t.grasp_sensor_name,
        t.support_sensor_name,
        t.surface_cfg,
        t.jaw_cfg,
        t.jaw_open,
      )
      obj = env.scene[t.object_name]
      top = obj.data.root_link_pos_w[:, 2] + object_half_height(env, t.object_name)
      tcp = self._entity.data.site_pos_w[:, arm["site"]]
      clear = latch.released & (tcp[:, 2] - top >= HANDOFF_CLEAR)
      self._clear_steps[:, k] = torch.where(
        clear, self._clear_steps[:, k] + 1, torch.zeros_like(self._clear_steps[:, k])
      )
      by_lift = self._clear_steps[:, k] >= HANDOFF_CLEAR_STEPS
      by_time = env.episode_length_buf.float() * env.step_dt >= t.timeout_s
      fresh = (by_lift | by_time) & (self.phase[:, k] == POLICY)
      self.by_timeout[:, k] |= fresh & ~by_lift
      self.start_handoff(k, by_lift | by_time)

  def _jacobian(self, k: int) -> torch.Tensor:
    """The arm's geometric Jacobian at its TCP, world frame: rows (v, w), columns arm 1-7. (B, 6, 7)"""
    data = self._env.sim.data
    joints = self._arms[k]["joints"]
    anchor = data.xanchor[:, joints]
    axis = data.xaxis[:, joints]
    tcp = self._entity.data.site_pos_w[:, self._arms[k]["site"]].unsqueeze(1)
    jv = torch.linalg.cross(axis, tcp - anchor, dim=-1)
    return torch.cat([jv, axis], dim=-1).transpose(1, 2)

  def _route(self, actions: torch.Tensor) -> torch.Tensor:
    out = actions.clone()
    for k, arm in enumerate(self._arms):
      phase = self.phase[:, k]
      active = phase > POLICY
      if not bool(active.any()):
        continue
      side = float(arm["cfg"].side)
      pos, quat = self._tcp(k)
      hold = self._hold_pos[:, k]
      lift_z = hold[:, 2] + HANDOFF_LIFT
      self._leg_steps[:, k] += active.long()
      stalled = self._leg_steps[:, k] >= self._leg_timeout
      up_done = (phase == UP) & ((pos[:, 2] >= lift_z - HANDOFF_TOL) | stalled)
      phase = torch.where(up_done, OUT, phase)
      self._leg_steps[:, k] = torch.where(up_done, 0, self._leg_steps[:, k])
      stalled = self._leg_steps[:, k] >= self._leg_timeout
      out_done = (phase == OUT) & ((side * pos[:, 1] >= HANDOFF_OUT_Y) | stalled)
      phase = torch.where(out_done, HOME, phase)
      self.phase[:, k] = phase

      # The TCP velocity each leg asks for: along the leg at HANDOFF_SPEED, the other axes held.
      g = HANDOFF_POS_GAIN
      v = torch.zeros_like(pos)
      up = phase == UP
      v[:, 0] = g * (hold[:, 0] - pos[:, 0])
      v[:, 1] = torch.where(
        up,
        g * (hold[:, 1] - pos[:, 1]),
        torch.full_like(pos[:, 1], side * HANDOFF_SPEED),
      )
      v[:, 2] = torch.where(
        up, torch.full_like(pos[:, 2], HANDOFF_SPEED), g * (lift_z - pos[:, 2])
      )
      w = HANDOFF_ROT_GAIN * quat_box_minus(self._hold_quat[:, k], quat)
      twist = torch.cat([v, w], dim=-1).unsqueeze(-1)  # (B, 6, 1)
      jac = self._jacobian(k)
      jjt = jac @ jac.transpose(1, 2)
      damp = (HANDOFF_DLS**2) * torch.eye(6, device=self.device).expand_as(jjt)
      dq = (jac.transpose(1, 2) @ torch.linalg.solve(jjt + damp, twist)).squeeze(-1)
      dq = dq * self._dt
      idx = arm["idx"]
      rate = self._rate[:, idx]
      shrink = (
        (rate / dq.abs().clamp_min(1e-9)).amin(dim=-1, keepdim=True).clamp(max=1.0)
      )
      q_route = self._last_target[:, idx] + dq * shrink
      home_q = self._default_target()[:, idx]
      if arm["park"] is not None:
        home_q = home_q + arm["park"]
      q_star = torch.where((phase == HOME).unsqueeze(-1), home_q, q_route)
      a_arm = (q_star - self._offset_of(idx)) / self._scale_of(idx)
      out[:, idx] = torch.where(active.unsqueeze(-1), a_arm, out[:, idx])
      jaw = arm["jaw"]
      open_jaw = (0.0 - self._offset_of(jaw)) / self._scale_of(jaw)
      out[:, jaw] = torch.where(active, open_jaw, out[:, jaw])
    return out

  def _offset_of(self, idx):
    return (
      self._offset[:, idx] if isinstance(self._offset, torch.Tensor) else self._offset
    )

  def _scale_of(self, idx):
    return self._scale[:, idx] if isinstance(self._scale, torch.Tensor) else self._scale

  def process_actions(self, actions: torch.Tensor) -> None:
    self._resume_started()
    if any(a["cfg"].trigger is not None for a in self._arms):
      self._auto_trigger()
    super().process_actions(self._route(actions))

  def _resume_started(self) -> None:
    """The first step after a hold start's reset: the episode clock and T as they were in the snapshot."""
    pending = getattr(self, "_pending", None)
    if pending is None:
      return
    ids, clock, offsets, command = pending
    self._pending = None
    self._env.episode_length_buf[ids] = clock
    self.handed_step[ids] = torch.where(
      self.handed_step[ids] >= 0,
      clock.unsqueeze(-1).expand_as(self.handed_step[ids]),
      -1,
    )
    self._env.command_manager.get_term(command).set_offsets(ids, offsets)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      env_ids = slice(None)
    self.phase[env_ids] = POLICY
    self.handed_step[env_ids] = -1
    self.by_timeout[env_ids] = False
    self._leg_steps[env_ids] = 0
    self._clear_steps[env_ids] = 0
    state = getattr(self._env, START_STATE, None)
    if state is not None:
      ids = torch.arange(self.num_envs, device=self.device)[env_ids]
      ids = ids[state["mask"][ids]]
      if len(ids):
        self._last_target[ids] = state["target"][ids]
        if self._servo_idx is not None:
          self._servo[ids] = state["servo"][ids]
        self._pending = (
          ids,
          state["clock"][ids],
          state["goal_offset"][ids],
          state["command"],
        )
    start = getattr(self._env, START_HOME, None) or {}
    for k, arm in enumerate(self._arms):
      home = start.get(arm["cfg"].name)
      if home is None:
        continue
      ids = torch.arange(self.num_envs, device=self.device)[env_ids]
      ids = ids[home[ids]]
      self.phase[ids, k] = HOME
      self.handed_step[ids, k] = 0


@dataclass(kw_only=True)
class HandoffJointPositionActionCfg(RateLimitedJointPositionActionCfg):
  """`RateLimitedJointPositionActionCfg` plus the arms plan S11's hand-off can take over."""

  arms: tuple[HandoffArmCfg, ...]

  def build(self, env: ManagerBasedRlEnv) -> HandoffJointPositionAction:
    return HandoffJointPositionAction(self, env)


def with_handoff(old, arms: tuple[HandoffArmCfg, ...]) -> HandoffJointPositionActionCfg:
  """``old`` -- the absolute rate-limited action term, as `make_tabletop_env_cfg` builds it --
  with the hand-off controller on ``arms``."""
  if type(old) is not RateLimitedJointPositionActionCfg:
    raise ValueError(
      "the hand-off is built on the absolute rate-limited action without a tool speed cap; got "
      f"{type(old).__name__} (ACTION=delta, TOOL_SPEED or GATE set?)"
    )
  fields = {f.name: getattr(old, f.name) for f in dataclasses.fields(old) if f.init}
  return HandoffJointPositionActionCfg(**fields, arms=arms)
