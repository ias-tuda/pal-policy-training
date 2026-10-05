"""Plan S7-11: open the jaw before the grasp (`PREGRASP`).

Siwei, 2026-10-05: the deployed WC-c s1 @ 5500 places well, but its misses on the robot are the
jaw not fully open on the way in -- a finger meets the cube and pushes it away. The same policy
does it in MuJoCo (plan S7-11 section 1): half of its 12 reference episodes push the cube
>= 1 cm before the grasp, the jaw starts closing at full speed 2-4 cm above the grasp height,
and it is half closed > 10 cm out. Nothing in the reward priced either before `was_lifted`.

This module holds what the codes read (`pick_place/pregrasp.py` builds them):

* `PregraspState` -- per env and episode: where the cube spawned, its largest horizontal
  displacement before the first grasp (frozen there), and the step of that first grasp;
* `in_grasp_window` -- the cube within the grasp window of the TCP, from a cube position in
  the world (true or as the actor observed it);
* the reward terms NP (`pregrasp_push_penalty`), JO (`pregrasp_jaw_closed`) and YG
  (`pregrasp_grasp_alignment`);
* the action G / OVR (`PregraspGateJointPositionAction`): the jaw's commanded target capped
  while the *observed* cube is outside the window, before the grasp.

"Before the grasp" is: not held now (both pads, `observations._grasped_bool`) and not yet
`was_lifted` -- a miss reopens the window, a carry does not.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from pal_mjlab.tasks.manipulation.mdp.actions import (
  RateLimitedJointPositionAction,
  RateLimitedJointPositionActionCfg,
)
from pal_mjlab.tasks.manipulation.mdp.observations import (
  _base_pose,
  _grasped_bool,
  _object_pose_w,
  _was_lifted_bool,
)
from pal_mjlab.tasks.manipulation.mdp.stack import fold_quarter

_ROBOT = SceneEntityCfg("robot")


@dataclass(frozen=True)
class GraspWindow:
  """Where the cube's centre must be in the TCP frame to count as "in the grasp", m.

  From WC-c s1 @ 5500's `grasp_calibration.json` (12 MuJoCo reference grasps, cube in TCP at
  the jaw's onset: x -0.7..0.6 cm, y -0.9..1.9 cm, z 3.4..6.7 cm, median z 4.9 cm), widened as
  the plan says: x +/-2 cm, y +/-2.5 cm, z 3-7.5 cm."""

  x: float = 0.02
  y: float = 0.025
  z: tuple[float, float] = (0.03, 0.075)


GRASP_WINDOW = GraspWindow()


def cube_in_tcp(
  env: ManagerBasedRlEnv, pos_w: torch.Tensor, tcp_cfg: SceneEntityCfg
) -> torch.Tensor:
  """A world point in the TCP site's frame. (B, 3)"""
  robot: Entity = env.scene[tcp_cfg.name]
  tcp_pos = robot.data.site_pos_w[:, tcp_cfg.site_ids].squeeze(1)
  tcp_quat = robot.data.site_quat_w[:, tcp_cfg.site_ids].squeeze(1)
  return quat_apply_inverse(tcp_quat, pos_w - tcp_pos)


def in_grasp_window(
  env: ManagerBasedRlEnv,
  pos_w: torch.Tensor,
  tcp_cfg: SceneEntityCfg,
  window: GraspWindow = GRASP_WINDOW,
) -> torch.Tensor:
  """The world point ``pos_w`` is inside ``window`` of the TCP. (B,)"""
  p = cube_in_tcp(env, pos_w, tcp_cfg)
  return (
    (p[:, 0].abs() <= window.x)
    & (p[:, 1].abs() <= window.y)
    & (p[:, 2] >= window.z[0])
    & (p[:, 2] <= window.z[1])
  )


def observed_object_pos_w(
  env: ManagerBasedRlEnv,
  object_name: str,
  term_name: str = "object_pos",
  group: str = "actor",
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """The cube's position as the actor last observed it -- tracker lag, per-episode bias and
  jitter included -- back in the world frame. (B, 3)

  Read out of the observation manager's last output: the vector the policy acted on at this
  step, which is what the deployer has. Before any observation exists (inside the first
  reset) it falls back to the tracker's pose, or the live one."""
  om = getattr(env, "observation_manager", None)
  buf = None if om is None else getattr(om, "_obs_buffer", None)
  robot: Entity = env.scene[asset_cfg.name]
  if buf is not None and isinstance(buf.get(group), torch.Tensor):
    offset = 0
    for name, dims in zip(
      om.active_terms[group], om.group_obs_term_dim[group], strict=True
    ):
      width = math.prod(dims)
      if name == term_name:
        if width != 3:
          raise ValueError(
            f"{group}/{term_name} is {dims} wide; the pregrasp gate reads a 3-D position"
          )
        pos_b = buf[group][:, offset : offset + 3]
        base_pos, base_quat = _base_pose(robot)
        return base_pos + quat_apply(base_quat, pos_b)
      offset += width
    raise ValueError(f"the {group!r} group has no {term_name!r} term")
  pos_w, _ = _object_pose_w(env, object_name, None)
  return pos_w


# -- Per-episode state ------------------------------------------------------------------------

_STATE = "_pp_pregrasp"


class PregraspState:
  """Per env: the cube's spawn xy, its largest horizontal displacement before the first grasp,
  and the episode step of that grasp (-1 until then).

  Updated on read, like `observations._was_lifted_bool`: every reader in one step sees the same
  physics state, and the updates are idempotent (a max, a latch). The spawn is captured at the
  first read after a reset (`reset_pregrasp` only marks the env), because derived positions are
  not refreshed until the env's `forward()` after the reset events."""

  def __init__(self, env: ManagerBasedRlEnv):
    n, dev = env.num_envs, env.device
    self.spawn_xy = torch.zeros(n, 2, device=dev)
    self.push = torch.zeros(n, device=dev)
    self.grasp_step = torch.full((n,), -1, dtype=torch.long, device=dev)
    self.pending = torch.ones(n, dtype=torch.bool, device=dev)

  @property
  def grasped_once(self) -> torch.Tensor:
    return self.grasp_step >= 0

  def update(
    self, env: ManagerBasedRlEnv, object_name: str, grasp_sensor_name: str
  ) -> None:
    obj: Entity = env.scene[object_name]
    xy = obj.data.root_link_pos_w[:, :2]
    if bool(self.pending.any()):
      self.spawn_xy[self.pending] = xy[self.pending]
      self.pending[:] = False
    before = ~self.grasped_once
    d = torch.norm(xy - self.spawn_xy, dim=-1)
    self.push = torch.where(before, torch.maximum(self.push, d), self.push)
    onset = before & _grasped_bool(env, grasp_sensor_name)
    self.grasp_step = torch.where(
      onset, env.episode_length_buf.to(self.grasp_step.dtype), self.grasp_step
    )

  def reset(self, env_ids: torch.Tensor | slice) -> None:
    self.push[env_ids] = 0.0
    self.grasp_step[env_ids] = -1
    self.pending[env_ids] = True


def pregrasp_state(
  env: ManagerBasedRlEnv, object_name: str, grasp_sensor_name: str
) -> PregraspState:
  """The env's `PregraspState`, created on first use and brought up to this step."""
  state = getattr(env, _STATE, None)
  if state is None or state.push.shape[0] != env.num_envs:
    state = PregraspState(env)
    setattr(env, _STATE, state)
  state.update(env, object_name, grasp_sensor_name)
  return state


def reset_pregrasp(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
  """Reset event: clear the env's latch and mark its spawn to be read at the next step."""
  state = getattr(env, _STATE, None)
  if state is None or state.push.shape[0] != env.num_envs:
    state = PregraspState(env)
    setattr(env, _STATE, state)
  state.reset(slice(None) if env_ids is None else env_ids)


def _before_grasp(
  env: ManagerBasedRlEnv, grasp_sensor_name: str, support_sensor_name: str
) -> torch.Tensor:
  """Not held now and not yet lifted. (B,)"""
  return ~_grasped_bool(env, grasp_sensor_name) & ~_was_lifted_bool(
    env, grasp_sensor_name, support_sensor_name
  )


# -- Rewards ----------------------------------------------------------------------------------


def pregrasp_push_penalty(
  env: ManagerBasedRlEnv,
  object_name: str,
  grasp_sensor_name: str,
  deadband: float,
  span: float,
) -> torch.Tensor:
  """NP: clip((d - deadband) / span, 0, 1), d the cube's largest horizontal displacement from
  its spawn before the first grasp (frozen at it), every step to the episode's end. Whatever
  pushed it -- finger, palm or wrist. The shape of S10's ``box_moved``. (B,)"""
  d = pregrasp_state(env, object_name, grasp_sensor_name).push
  return ((d - deadband) / span).clamp(0.0, 1.0)


def pregrasp_jaw_closed(
  env: ManagerBasedRlEnv,
  object_name: str,
  grasp_sensor_name: str,
  support_sensor_name: str,
  jaw_cfg: SceneEntityCfg,
  tcp_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """JO: the jaw's measured angle (0 = open) while before the grasp and the cube (true pose) is
  outside the grasp window. Inside the window it is free to close. (B,)"""
  robot: Entity = env.scene[jaw_cfg.name]
  q = robot.data.joint_pos[:, jaw_cfg.joint_ids][:, 0].clamp_min(0.0)
  obj: Entity = env.scene[object_name]
  outside = ~in_grasp_window(env, obj.data.root_link_pos_w, tcp_cfg)
  active = _before_grasp(env, grasp_sensor_name, support_sensor_name) & outside
  return torch.where(active, q, torch.zeros_like(q))


def grasp_relative_yaw(
  env: ManagerBasedRlEnv, object_name: str, tcp_cfg: SceneEntityCfg
) -> torch.Tensor:
  """The angle between the hand and the cube about the vertical, folded mod 90 deg into
  [0, 45 deg]: the TCP's x axis and the cube's x axis, both projected on the table. (B,)"""
  robot: Entity = env.scene[tcp_cfg.name]
  tcp_quat = robot.data.site_quat_w[:, tcp_cfg.site_ids].squeeze(1)
  obj: Entity = env.scene[object_name]
  ex = torch.zeros_like(tcp_quat[:, :3])
  ex[:, 0] = 1.0
  hand = quat_apply(tcp_quat, ex)
  cube = quat_apply(obj.data.root_link_quat_w, ex)
  diff = torch.atan2(hand[:, 1], hand[:, 0]) - torch.atan2(cube[:, 1], cube[:, 0])
  return fold_quarter(diff)


def pregrasp_grasp_alignment(
  env: ManagerBasedRlEnv,
  object_name: str,
  grasp_sensor_name: str,
  tcp_cfg: SceneEntityCfg,
  yaw_std: float,
) -> torch.Tensor:
  """YG: once, at the step both pads first touch the cube, exp(-(rel / std)^2) of the hand's
  yaw against the cube's (`grasp_relative_yaw`). 0 at every other step. (B,)"""
  state = pregrasp_state(env, object_name, grasp_sensor_name)
  onset = state.grasp_step == env.episode_length_buf.to(state.grasp_step.dtype)
  rel = grasp_relative_yaw(env, object_name, tcp_cfg)
  grade = torch.exp(-((rel / yaw_std) ** 2))
  return torch.where(onset, grade, torch.zeros_like(grade))


# -- The gate (G) and the deployer's override (OVR) ------------------------------------------


class PregraspGateJointPositionAction(RateLimitedJointPositionAction):
  """Plan S7-11's G (``cap`` 0.03 rad) and OVR (``cap`` 0, the deployer's patch, eval only):
  before the grasp, while the cube *as the actor observed it* is outside the grasp window, the
  jaw's commanded target is at most ``cap`` -- open.

  In the slew hook (`_shape_step`, after the slew, before plan S7-4's bounds), so the ONNX is
  unchanged and ``CMD_OBS=cmd`` shows the actor the capped target, as on the robot. The window is
  judged from the observed cube (FoundationPose's lag, bias and jitter) and the TCP from the
  arm's kinematics -- what a deployer has. "Held" reads the grasp sensor here; a deployer needs
  the jaw-angle proxy of the sim2sim adapter.

  Where the S9 lock kept the jaw from *opening* (and was circumvented by stripping the cube),
  this keeps it from *closing early*; its failure mode is a stall -- a cube observed outside the
  window that never enters it. `capped` counts it.
  """

  cfg: PregraspGateJointPositionActionCfg

  def __init__(self, cfg: PregraspGateJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    idx = self._indices(cfg.gate_joints, "gate_joints")
    if len(idx) != 1:
      raise ValueError(
        f"gate_joints must match the one jaw actuator; matched {idx.tolist()}"
      )
    self._gate_idx = int(idx[0])
    self._tcp = SceneEntityCfg(cfg.entity_name, site_names=(cfg.tcp_site,))
    self._tcp.resolve(env.scene)
    #: Per env, whether the gate was armed at the last step (before the grasp, cube observed
    #: outside the window), and whether it lowered the policy's target.
    self.armed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.capped = torch.zeros_like(self.armed)

  def gate_mask(self) -> torch.Tensor:
    env, cfg = self._env, self.cfg
    seen = observed_object_pos_w(env, cfg.object_name)
    outside = ~in_grasp_window(env, seen, self._tcp)
    return _before_grasp(env, cfg.grasp_sensor_name, cfg.support_sensor_name) & outside

  def _shape_step(self, previous: torch.Tensor) -> None:
    self.armed = self.gate_mask()
    i = self._gate_idx
    target = self._processed_actions
    capped = target[:, i].clamp(max=self.cfg.cap)
    self.capped = self.armed & (capped < target[:, i])
    target[:, i] = torch.where(self.armed, capped, target[:, i])
    self._processed_actions = target

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      env_ids = slice(None)
    self.armed[env_ids] = False
    self.capped[env_ids] = False


@dataclass(kw_only=True)
class PregraspGateJointPositionActionCfg(RateLimitedJointPositionActionCfg):
  """`RateLimitedJointPositionActionCfg` plus plan S7-11's pregrasp gate."""

  gate_joints: tuple[str, ...]
  """Actuator regex of the jaw."""

  cap: float
  """Largest commanded jaw angle while the gate is armed, rad (0 = open)."""

  tcp_site: str
  object_name: str
  grasp_sensor_name: str
  support_sensor_name: str

  def build(self, env: ManagerBasedRlEnv) -> PregraspGateJointPositionAction:
    return PregraspGateJointPositionAction(self, env)


def gate_cfg_from(
  old: RateLimitedJointPositionActionCfg, **gate
) -> PregraspGateJointPositionActionCfg:
  """The absolute rate-limited action ``old`` with the pregrasp gate."""
  if type(old) is not RateLimitedJointPositionActionCfg:
    raise ValueError(
      "the pregrasp gate is built on the absolute rate-limited action without a tool speed "
      f"cap or the stack lock; got {type(old).__name__} (ACTION=delta, TOOL_SPEED or GATE set?)"
    )
  fields = {f.name: getattr(old, f.name) for f in dataclasses.fields(old) if f.init}
  return PregraspGateJointPositionActionCfg(**fields, **gate)
