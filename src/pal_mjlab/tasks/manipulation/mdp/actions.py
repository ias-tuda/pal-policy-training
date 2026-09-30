"""Action terms mjlab does not provide.

Ported verbatim from Claudius's `origin/fix-stack-exploration-collapse` (8068e64) by S5.1
item 4 of the pick-and-place plan, where it carries the arms' hardware speed limits.

mjlab's ``JointPositionAction`` turns an action into an absolute joint target in one step:
``default + scale * action``, clipped. For an arm that is fine -- the target moves as far as
the policy asks and the servo follows. For the jaw it is not, because the jaw's whole travel
is 0.95 rad and the policy is free to command either end of it on consecutive control steps.
Nothing on the robot can do that, and nothing in the solver enjoys being asked.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.utils.lab_api.string import (
  resolve_matching_names,
  resolve_matching_names_values,
)


class RateLimitedJointPositionAction(JointPositionAction):
  """Joint position targets with a per-step slew limit on named joints.

  The target still comes from ``default + scale * action`` and is still absolute, so the
  deployment contract is unchanged: the ONNX emits the same vector and the deployer applies
  it the same way. What changes is that the *commanded* target may only move so far per
  control step, so a saturated action ramps to the stop over several steps instead of
  arriving there in one.
  """

  cfg: RateLimitedJointPositionActionCfg

  def __init__(self, cfg: RateLimitedJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    # inf on every joint the cfg does not name: unnamed joints keep mjlab's behaviour.
    self._rate = torch.full(
      (self.num_envs, self.action_dim), float("inf"), device=self.device
    )
    if cfg.rate_limit:
      index_list, _, value_list = resolve_matching_names_values(
        cfg.rate_limit, self._target_names
      )
      if not index_list:
        raise ValueError(
          f"rate_limit matched no actuator; patterns={list(cfg.rate_limit)}, "
          f"targets={self._target_names}"
        )
      self._rate[:, index_list] = torch.tensor(value_list, device=self.device)
    # The last target actually commanded, which the next one is measured against. Seeded
    # with the default pose, which is what `_offset` holds under `use_default_offset`.
    self._last_target = self._default_target()
    self._range_idx, self._range = self._resolve_range_clip(cfg, env)
    self._band_idx, self._band = self._resolve_effort_band(cfg, env)
    #: Per env, whether T's band / C's range clip moved the target at the last step. For the
    #: plan's check that neither binds during a scripted carry, and for the eval.
    self.band_hit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.range_hit = torch.zeros_like(self.band_hit)
    self._servo_idx, self._servo = self._resolve_servo(cfg, env)

  @property
  def commanded_target(self) -> torch.Tensor:
    """The target this term last commanded: after the slew and the bounds, before the jaw
    servo. What a deployer sends, and what plan S7-8's ``CMD_OBS`` shows the actor. (B, A)"""
    return self._last_target

  def _indices(self, patterns: tuple[str, ...], what: str) -> torch.Tensor:
    index_list, _ = resolve_matching_names(patterns, self._target_names)
    if not index_list:
      raise ValueError(
        f"{what} matched no actuator; patterns={list(patterns)}, "
        f"targets={self._target_names}"
      )
    return torch.tensor(index_list, device=self.device, dtype=torch.long)

  def _resolve_range_clip(self, cfg, env):
    """Plan S7-4's C: the named joints' targets are clipped to their joint range. (idx, (B, n, 2))"""
    if not cfg.range_clip_joints:
      return None, None
    idx = self._indices(cfg.range_clip_joints, "range_clip_joints")
    limits = self._entity.data.joint_pos_limits[:, self._target_ids[idx]].clone()
    if not torch.isfinite(limits).all():
      raise ValueError("range_clip_joints names a joint without a finite range")
    return idx, limits

  def _resolve_effort_band(self, cfg, env):
    """Plan S7-4's T: the band the named joints' targets are kept in, as (idx, (n, 3)) rows of
    (tau_cmd / K, D / K, K).

    The target q* is kept where the position actuator's own PD output stays within the cap,

        |K (q* - q) - D qdot| <= tau_cmd,   i.e.   q* in q + (D / K) qdot +/- tau_cmd / K,

    so the PD can never command more than tau_cmd and a target cannot run away past what the
    capped actuator can deliver (S7-3's press). The plan wrote the band as q +/- tau_cmd / K,
    the K term alone; on G20 that is not the PD's output while the arm moves -- a joint moving
    at qdot needs K (q* - q) = tau_g + D qdot, D qdot of which the D term takes back -- and it
    capped every move at (tau_cmd - tau_g) / D, ~0.2 rad/s on the elbow: it bound on every step
    of the scripted approach (S7-4 prep 3, 2026-09-28). Centred on (D / K) qdot it binds only
    where the capped actuator would saturate anyway.

    tau_cmd is the joint's effort limit as the compiled model carries it -- the joint's
    ``actuatorfrcrange`` where it is limited (the lowered cap, `_cap_left_arm_effort`), else
    its actuator's ``forcerange`` -- and K, D the actuator's nominal gains: what a deployer
    knows, not a DR draw.
    """
    if not cfg.effort_band_joints:
      return None, None
    idx = self._indices(cfg.effort_band_joints, "effort_band_joints")
    mj = env.sim.mj_model
    ctrl_of = {}
    for actuator in self._entity.actuators:
      for name, ctrl in zip(
        actuator.target_names, actuator.global_ctrl_ids.tolist(), strict=True
      ):
        ctrl_of[name.split("/")[-1]] = ctrl
    joint_ids = self._entity.indexing.joint_ids[self._target_ids[idx]].tolist()
    half = []
    for i, jid in zip(idx.tolist(), joint_ids, strict=True):
      name = self._target_names[i].split("/")[-1]
      ctrl = ctrl_of[name]
      if mj.jnt_actfrclimited[jid]:
        tau = float(mj.jnt_actfrcrange[jid, 1])
      elif mj.actuator_forcelimited[ctrl]:
        tau = float(mj.actuator_forcerange[ctrl, 1])
      else:
        raise ValueError(f"effort_band_joints: {name} has no effort limit")
      k = float(mj.actuator_gainprm[ctrl, 0])
      d = -float(mj.actuator_biasprm[ctrl, 2])
      half.append((tau / k, d / k, k))
    return idx, torch.tensor(half, device=self.device)

  def _resolve_servo(self, cfg, env):
    """Plan S7-6's jaw servo model (Claudius's 71e1147, `BinaryJawAction`'s servo): the
    target written to MuJoCo for the named joints is a rate-limited first-order lag of the
    commanded target, not the command itself. (idx, (B, n) state) or (None, None).

    The real gripper follows its command with a steady 0.13-0.17 s lag during a stroke, peaks
    at 1.3-1.7 rad/s, keeps moving ~0.1 s after the command stops and creeps into the open stop
    (Claudius's recording of ``/pal_policy/desired_state`` against the measured jaw, four
    strokes). One pole with a saturating velocity reproduces all four. In sim without it the
    jaw is a ~1e-4 kg m^2 finger behind a stiff PD that reaches any target inside one control
    step, so reversing the jaw costs nothing -- on the robot a reversed stroke never completes.

    Advanced per physics step (`apply_actions`), not per control step: it is a property of the
    hardware, so it should not depend on the decimation. ``tau`` and the speed limit are drawn
    per env at each reset. The command -- `_processed_actions`, what the slew, the bounds, the
    trace and a deployer see -- is unchanged: on hardware the lag comes from the gripper, not
    from the node sending it commands.
    """
    if not cfg.servo_joints:
      return None, None
    idx = self._indices(cfg.servo_joints, "servo_joints")
    self._physics_dt = env.physics_dt
    shape = (self.num_envs, len(idx))
    self._servo_tau = torch.empty(shape, device=self.device)
    self._servo_max_vel = torch.empty(shape, device=self.device)
    self._sample_servo(slice(None))
    return idx, self._default_target()[:, idx].clone()

  def _sample_servo(self, env_ids: torch.Tensor | slice) -> None:
    lo, hi = self.cfg.servo_tau
    self._servo_tau[env_ids] = torch.empty_like(self._servo_tau[env_ids]).uniform_(
      lo, hi
    )
    lo, hi = self.cfg.servo_max_vel
    self._servo_max_vel[env_ids] = torch.empty_like(
      self._servo_max_vel[env_ids]
    ).uniform_(lo, hi)

  def servo_target(self) -> torch.Tensor:
    """The joint targets as they reach MuJoCo: the command with the servo's joints replaced by
    the servo's state. `_processed_actions` itself when there is no servo. (B, A)"""
    if self._servo_idx is None:
      return self._processed_actions
    target = self._processed_actions.clone()
    target[:, self._servo_idx] = self._servo
    return target

  def apply_actions(self) -> None:
    if self._servo_idx is None:
      super().apply_actions()
      return
    # Under a ramp the jaw settles exactly tau behind the command; after the command stops it
    # decays with tau; it never exceeds max_vel.
    err = self._processed_actions[:, self._servo_idx] - self._servo
    travel = self._servo_max_vel * self._physics_dt
    self._servo = self._servo + (err * (self._physics_dt / self._servo_tau)).clamp(
      -travel, travel
    )
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    self._entity.set_joint_position_target(
      self.servo_target() - encoder_bias, joint_ids=self._target_ids
    )

  def _bound(self) -> None:
    """Plan S7-4's bounds on the final target, after the slew and any tool cap: T's band
    about the measured position, then C's joint range."""
    target = self._processed_actions
    if self._band_idx is not None:
      joints = self._target_ids[self._band_idx]
      centre = (
        self._entity.data.joint_pos[:, joints]
        + self._band[:, 1] * self._entity.data.joint_vel[:, joints]
      )
      half = self._band[:, 0]
      before = target[:, self._band_idx]
      after = torch.clamp(before, centre - half, centre + half)
      self.band_hit = (after != before).any(dim=-1)
      target[:, self._band_idx] = after
    if self._range_idx is not None:
      before = target[:, self._range_idx]
      after = torch.clamp(before, self._range[..., 0], self._range[..., 1])
      self.range_hit = (after != before).any(dim=-1)
      target[:, self._range_idx] = after
    self._processed_actions = target

  def _shape_step(self, previous: torch.Tensor) -> None:
    """Hook between the slew and the bounds, on `_processed_actions`. None here."""

  def _default_target(self) -> torch.Tensor:
    if isinstance(self._offset, torch.Tensor):
      return self._offset.clone()
    return torch.full(
      (self.num_envs, self.action_dim), float(self._offset), device=self.device
    )

  def process_actions(self, actions: torch.Tensor) -> None:
    previous = self._last_target.clone()
    super().process_actions(actions)  # scale, offset, then clip
    delta = (self._processed_actions - previous).clamp(-self._rate, self._rate)
    self._processed_actions = previous + delta
    self._shape_step(previous)
    self._bound()
    self._last_target = self._processed_actions.clone()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    # The episode restarts from the default pose, so the slew reference must too --
    # otherwise the first step of a new episode is measured against the last step of the
    # old one and the limit is silently skipped exactly where the state jumps most.
    if env_ids is None:
      env_ids = slice(None)
    self._last_target[env_ids] = self._default_target()[env_ids]
    if self._servo_idx is not None:
      # The servo restarts where the command does, with a fresh draw of its constants.
      self._servo[env_ids] = self._default_target()[env_ids][:, self._servo_idx]
      self._sample_servo(env_ids)


@dataclass(kw_only=True)
class RateLimitedJointPositionActionCfg(JointPositionActionCfg):
  rate_limit: dict[str, float] | None = None
  """Max change in commanded target per control step, in rad, keyed by actuator regex.

  Joints not matched are unlimited. This is a property of what the hardware can execute, not
  a tuning knob: a target the real servo cannot reach within one control period is a target
  that only ever existed in the solver.
  """

  range_clip_joints: tuple[str, ...] | None = None
  """Plan S7-4's C: actuator regexes whose target is clipped to the joint's range, after the
  slew. mjlab's position actuators are ``ctrllimited=False``, so a target past a limit pushes
  the joint into its stop at up to full effort (S7-3's elbow run-away). None for none."""

  effort_band_joints: tuple[str, ...] | None = None
  """Plan S7-4's T: actuator regexes whose target is kept where the actuator's PD output
  stays within the effort cap, q* in q + (D / K) qdot +/- tau_cmd / K, after the slew
  (`RateLimitedJointPositionAction._resolve_effort_band`). None for none. Like the slew, a
  deployer must apply both bounds to reproduce training."""

  servo_joints: tuple[str, ...] | None = None
  """Plan S7-6's jaw servo model: actuator regexes whose written target lags the command
  (`RateLimitedJointPositionAction._resolve_servo`). None for none, which is every run up to
  S7-5. Unlike the slew and the bounds, a deployer must *not* apply it: the gripper does."""

  servo_tau: tuple[float, float] | None = None
  """Per-env range of the servo's time constant, s. Required with ``servo_joints``."""

  servo_max_vel: tuple[float, float] | None = None
  """Per-env range of the servo's speed limit, rad/s. Required with ``servo_joints``."""

  def __post_init__(self) -> None:
    super().__post_init__()
    given = [
      x is not None for x in (self.servo_joints, self.servo_tau, self.servo_max_vel)
    ]
    if any(given) and not all(given):
      raise ValueError(
        "servo_joints, servo_tau and servo_max_vel are one model and must be given "
        f"together; got servo_joints={self.servo_joints}, servo_tau={self.servo_tau}, "
        f"servo_max_vel={self.servo_max_vel}"
      )
    if self.servo_tau is not None and not 0.0 < self.servo_tau[0] <= self.servo_tau[1]:
      raise ValueError(
        f"servo_tau must be 0 < min <= max seconds; got {self.servo_tau}"
      )
    if (
      self.servo_max_vel is not None
      and not 0.0 < self.servo_max_vel[0] <= self.servo_max_vel[1]
    ):
      raise ValueError(
        f"servo_max_vel must be 0 < min <= max rad/s; got {self.servo_max_vel}"
      )

  def build(self, env: ManagerBasedRlEnv) -> RateLimitedJointPositionAction:
    return RateLimitedJointPositionAction(self, env)


class DeltaRateLimitedJointPositionAction(RateLimitedJointPositionAction):
  """Plan S7-8's DEL: the action moves the commanded target, it does not set it.

  ``target_t = clip(target_{t-1} + step_max * clip(a, -1, 1), box)`` with ``step_max`` the
  ``rate_limit`` of each joint (the jaw's too) and ``box`` the box the absolute form would span,
  ``default +/- scale`` (the jaw: its open-to-closed clip). |a| = 1 is full speed, so the whole
  action range is usable and no separate slew is needed. This is the delta form of the stack task
  and of Claudius's ``DeltaJointPositionAction`` (``origin/faster-moves``): behind a slew clamp an
  absolute action saturates the clamp and the policy cannot see it.

  The integrator's state is the commanded target after plan S7-4's bounds (`_bound`: T's effort
  band about the measured position, C's joint range), not the unbounded sum, so it cannot wind up
  past a bound. It starts from the default pose at every reset, like the slew reference. The
  policy needs to see its state: ``CMD_OBS`` (`mdp.commanded_target_rel`). A deployer must keep
  the integrator and apply the same clip -- the ONNX output is a velocity command in units of
  ``step_max`` per control step.
  """

  cfg: DeltaRateLimitedJointPositionActionCfg

  def __init__(
    self, cfg: DeltaRateLimitedJointPositionActionCfg, env: ManagerBasedRlEnv
  ):
    super().__init__(cfg=cfg, env=env)
    if not torch.isfinite(self._rate).all():
      raise ValueError(
        "a delta action needs a per-step limit on every joint (it is the action's full "
        f"scale); rate_limit={cfg.rate_limit}, targets={self._target_names}"
      )
    lo = self._offset - self._scale
    hi = self._offset + self._scale
    if cfg.clip is not None:
      lo = torch.maximum(lo, self._clip[..., 0])
      hi = torch.minimum(hi, self._clip[..., 1])
    self._box_lo, self._box_hi = lo, hi

  def process_actions(self, actions: torch.Tensor) -> None:
    previous = self._last_target
    self._raw_actions[:] = actions
    step = self._raw_actions.clamp(-1.0, 1.0) * self._rate
    self._processed_actions = torch.clamp(previous + step, self._box_lo, self._box_hi)
    self._shape_step(previous)
    self._bound()
    self._last_target = self._processed_actions.clone()


@dataclass(kw_only=True)
class DeltaRateLimitedJointPositionActionCfg(RateLimitedJointPositionActionCfg):
  """``rate_limit`` is the largest change of the target per control step, per joint, rad (the
  action's full scale, so every joint needs one); ``scale`` and ``clip`` give the box the
  target is held inside, exactly as they do the absolute form's action range."""

  def build(self, env: ManagerBasedRlEnv) -> DeltaRateLimitedJointPositionAction:
    return DeltaRateLimitedJointPositionAction(self, env)


class ToolSpeedCappedJointPositionAction(RateLimitedJointPositionAction):
  """`RateLimitedJointPositionAction` with a bound on the tool's speed (plan S7-2, module V).

  Halving every joint's slew (S7 speed bound A) cut the held cube's peak speed by only 15-30 %:
  several joints add up at the tool. This bounds the tool itself. After the per-joint slew,
  the capped joints' step in commanded target, dq, is scaled down uniformly -- its direction
  kept -- until

      sum_i |dq_i| * r_i <= v_max * control_dt,

  with r_i the distance from joint i's axis to the tool point. A revolute joint turning by
  dq_i moves the tool by |dq_i| * r_i, so the sum bounds how far one control step's command
  moves the tool, from the joint anchors and axes alone -- no Jacobian solve.

  v_max is ``speed_limit``, and ``slow_speed_limit`` while the object is held and either
  within ``slow_zone`` of its resting height on the surface (lifting off, setting down) or
  within ``slow_zone`` of the goal horizontally (arriving over it).

  Like the slew, this lives in the action term: a deployment must apply the same bound to
  reproduce what was trained. The bound uses the robot's own kinematics (anchors, axes, tool
  point) plus "held", the object's pose and the goal; on hardware "held" has no sensor and
  needs a proxy (the jaw-angle rule of the sim2sim adapter).
  """

  cfg: ToolSpeedCappedJointPositionActionCfg

  def __init__(
    self, cfg: ToolSpeedCappedJointPositionActionCfg, env: ManagerBasedRlEnv
  ):
    super().__init__(cfg=cfg, env=env)
    index_list, _ = resolve_matching_names(cfg.capped_joints, self._target_names)
    if not index_list:
      raise ValueError(
        f"capped_joints matched no actuator; patterns={list(cfg.capped_joints)}, "
        f"targets={self._target_names}"
      )
    self._capped = torch.tensor(index_list, device=self.device, dtype=torch.long)
    # Global joint ids, for the anchors and axes MuJoCo computes per joint.
    self._capped_joints = self._entity.indexing.joint_ids[
      self._target_ids[self._capped]
    ]
    self._tool = self._entity.site_names.index(cfg.tool_site)
    #: The tool speed bound applied at the last step, per env (m/s).
    self.speed_limit = torch.full((self.num_envs,), cfg.speed_limit, device=self.device)

  def _lever_arms(self) -> torch.Tensor:
    """Distance from each capped joint's axis to the tool point. (B, n)"""
    data = self._env.sim.data
    anchor = data.xanchor[:, self._capped_joints]
    axis = data.xaxis[:, self._capped_joints]
    tool = self._entity.data.site_pos_w[:, self._tool].unsqueeze(1)
    return torch.norm(torch.linalg.cross(tool - anchor, axis, dim=-1), dim=-1)

  def _speed_limits(self) -> torch.Tensor:
    """``slow_speed_limit`` in the slow zone, ``speed_limit`` elsewhere. (B,)"""
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    from pal_mjlab.tasks.manipulation.mdp.events import surface_top_z
    from pal_mjlab.tasks.manipulation.mdp.observations import _grasped_bool

    env, cfg = self._env, self.cfg
    obj = env.scene[cfg.object_name]
    pos = obj.data.root_link_pos_w
    half_z = env.sim.model.geom_size[:, obj.indexing.geom_ids[0], 2]
    height = pos[:, 2] - surface_top_z(env, SceneEntityCfg(cfg.surface_name)) - half_z
    goal = env.command_manager.get_term(cfg.command_name).command
    near = (height < cfg.slow_zone) | (
      torch.norm((pos - goal)[:, :2], dim=-1) < cfg.slow_zone
    )
    slow = _grasped_bool(env, cfg.grasp_sensor_name) & near
    return torch.where(
      slow,
      torch.full_like(height, cfg.slow_speed_limit),
      torch.full_like(height, cfg.speed_limit),
    )

  def _shape_step(self, previous: torch.Tensor) -> None:
    # After scale, offset, clip and the per-joint slew; before plan S7-4's bounds.
    target = self._processed_actions
    dq = target[:, self._capped] - previous[:, self._capped]
    travel = (dq.abs() * self._lever_arms()).sum(dim=-1)  # bound on the tool's move, m
    self.speed_limit = self._speed_limits()
    allowed = self.speed_limit * self.cfg.control_dt
    shrink = (allowed / travel.clamp_min(1e-9)).clamp(max=1.0)
    target[:, self._capped] = previous[:, self._capped] + dq * shrink.unsqueeze(-1)
    self._processed_actions = target


@dataclass(kw_only=True)
class ToolSpeedCappedJointPositionActionCfg(RateLimitedJointPositionActionCfg):
  capped_joints: tuple[str, ...]
  """Actuator regexes of the joints whose step the tool bound scales: the arm, not the jaw."""

  tool_site: str
  """The tool point whose speed is bounded (the grasp point)."""

  speed_limit: float
  """Tool speed bound outside the slow zone, m/s."""

  slow_speed_limit: float
  """Tool speed bound in the slow zone, m/s."""

  slow_zone: float
  """Distance (m) to the object's resting height, or horizontally to the goal, inside which a
  held object gets ``slow_speed_limit``."""

  control_dt: float
  """Control period, s: the bound is a speed times this."""

  object_name: str
  grasp_sensor_name: str
  surface_name: str
  command_name: str

  def build(self, env: ManagerBasedRlEnv) -> ToolSpeedCappedJointPositionAction:
    return ToolSpeedCappedJointPositionAction(self, env)
