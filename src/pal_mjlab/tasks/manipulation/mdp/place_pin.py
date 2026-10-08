"""Plan §14 (`pick-place-insert-plan.md`): pick the cross pin up and put it down at a goal point, in any pose.

The pin (`scene.get_pin_spec`, two crossed box plates, the body's origin at its centre, its axis the body z) starts
standing on its foot or lying on its side, the "x" pose (resting on two plate tips, the plates at 45 deg to the
table). It counts as placed when it rests on the table -- standing or lying, not held, still -- with its centre
within 2 cm (xy) of the goal. Everything that depends on the pin's pose goes through the pin's exact geometry:

* `pin_extent_z` -- the lowest and highest points of the two plates along world z, from their 16 corners: a standing
  pin's centre is 45 mm over its lowest point, a lying "x" pin's 14.1 mm, a "+" pin's 16 mm;
* `pin_axis_canonical_w` -- the pin's axis with its sign fixed (the pin is end-for-end symmetric and the tracker may
  flip it): pointing up for a pin within 60 deg of vertical (plan §12's `pin_tilt` convention), otherwise the sign
  with a non-negative x in `base_link` (ties to +y). ``standing`` is that 60 deg test;
* `PinPlaceCommand` -- the goal point on the table surface (``goal_surface``), drawn from the area as the cube's goal
  is (`SurfaceOffsetCommand`, at least `TARGET_MIN_SEPARATION` from the pin), and the **pose-following** goal its
  ``command`` returns: (goal x, goal y, table top + the pin centre's height over its lowest point *in its current
  pose*). So every generic at-goal test (`observations._at_goal` with an (xy, z) tolerance, M1 / J / M6 / M8 and the
  eval's ``placed``) reads "centre within xy of the goal, lowest point within z of the table", and M2's distance is
  the one plan §14.2 names, from (c_x, c_y, z_min) to the goal point. `observations._release_latch` takes its resting
  height from ``rest_centre_z`` (the same number), so M6's height at the let-go is the lowest point's over the table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
)

from pal_mjlab.tasks.manipulation import scene
from pal_mjlab.tasks.manipulation.mdp.box import (
  _corrupted,
  tracker_yaw_bias,
  yaw_relative_to_base,
)
from pal_mjlab.tasks.manipulation.mdp.commands import (
  SurfaceOffsetCommand,
  SurfaceOffsetCommandCfg,
)
from pal_mjlab.tasks.manipulation.mdp.events import (
  _surface_centre_xy,
  in_keepout,
  surface_top_z,
)
from pal_mjlab.tasks.manipulation.mdp.observations import (
  _base_pose,
  _grasped_bool,
  _object_pose_w,
  _placed_bool,
  _to_base_frame,
  _was_lifted_bool,
)
from pal_mjlab.tasks.manipulation.mdp.rewards import object_placed_graded

_ROBOT = SceneEntityCfg("robot")
_TABLE = SceneEntityCfg("table")

#: A pin whose axis is within this of vertical counts as standing, else lying (plan §14.2's sign rule), as the
#: axis's |z| component: cos 60 deg.
STANDING_AXIS_Z = 0.5


def _corners_b(device: torch.device) -> torch.Tensor:
  """The 16 corners of the pin's two plates in its body frame (`scene.get_pin_spec`'s layout). (16, 3)"""
  halves = (
    (scene.PIN_SPAN / 2, scene.PIN_THICKNESS / 2, scene.PIN_LENGTH / 2),
    (scene.PIN_THICKNESS / 2, scene.PIN_SPAN / 2, scene.PIN_LENGTH / 2),
  )
  signs = torch.tensor(
    [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
    dtype=torch.float32,
    device=device,
  )
  return torch.cat([signs * torch.tensor(h, device=device) for h in halves], dim=0)


def pin_extent_z(quat_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """How far the pin's lowest and highest points lie below and above its centre along world z, m, for the
  orientation ``quat_w`` (B, 4): (below, above), both >= 0. Standing: 0.045 / 0.045; lying "x": 0.0141 each. (B,)"""
  corners = _corners_b(quat_w.device)
  n = quat_w.shape[0]
  z = quat_apply(
    quat_w.unsqueeze(1).expand(n, corners.shape[0], 4).reshape(-1, 4),
    corners.unsqueeze(0).expand(n, -1, -1).reshape(-1, 3),
  )[:, 2].reshape(n, -1)
  return -z.amin(-1), z.amax(-1)


def pin_axis_w(quat_w: torch.Tensor) -> torch.Tensor:
  """The pin's axis (its body z) in the world, as the orientation gives it (either sign). (B, 3)"""
  ez = torch.zeros(quat_w.shape[0], 3, device=quat_w.device)
  ez[:, 2] = 1.0
  return quat_apply(quat_w, ez)


def pin_standing(quat_w: torch.Tensor) -> torch.Tensor:
  """Whether the pin's axis is within 60 deg of vertical (`STANDING_AXIS_Z`). (B,)"""
  return pin_axis_w(quat_w)[:, 2].abs() >= STANDING_AXIS_Z


def pin_axis_canonical_w(env: ManagerBasedRlEnv, quat_w: torch.Tensor) -> torch.Tensor:
  """The pin's axis with plan §14.2's sign: up for a standing pin (`pin_standing`), otherwise with x >= 0 in
  `base_link`, a zero x taking +y. (B, 3)"""
  axis = pin_axis_w(quat_w)
  _, base_q = _base_pose(env.scene[_ROBOT.name])
  axis_b = quat_apply(quat_inv(base_q), axis)
  tie = axis_b[:, 0].abs() < 1e-6
  lying_neg = torch.where(tie, axis_b[:, 1] < 0.0, axis_b[:, 0] < 0.0)
  standing = axis[:, 2].abs() >= STANDING_AXIS_Z
  flip = torch.where(standing, axis[:, 2] < 0.0, lying_neg)
  return torch.where(flip.unsqueeze(-1), -axis, axis)


def pin_psi(env: ManagerBasedRlEnv, quat_w: torch.Tensor) -> torch.Tensor:
  """Plan §14.2's psi, the cross's rotation about the pin's own axis, rad: a standing pin's yaw relative to
  `base_link` (plan §12's, `box.yaw_relative_to_base`); a lying pin's roll -- the angle from world up to its first
  plate's span (body x) about the canonical axis (`pin_axis_canonical_w`): 0 (mod 90 deg) in the "+" pose, 45 deg in
  the "x" pose. (B,)"""
  standing = pin_standing(quat_w)
  yaw = yaw_relative_to_base(env, quat_w)
  axis = pin_axis_canonical_w(env, quat_w)
  ex = torch.zeros_like(axis)
  ex[:, 0] = 1.0
  span = quat_apply(quat_w, ex)
  up = torch.zeros_like(axis)
  up[:, 2] = 1.0
  side = torch.cross(axis, up, dim=-1)
  roll = torch.atan2((span * side).sum(-1), (span * up).sum(-1))
  return torch.where(standing, yaw, roll)


@dataclass
class PinRestState:
  """Where and how the pin lies relative to the table and the goal, per env (`pin_rest_state`); m, rad, m/s, rad/s."""

  z_min: torch.Tensor
  """The pin's lowest point over the table's top. (B,)"""
  z_top: torch.Tensor
  """Its highest point over the table's top. (B,)"""
  standing: torch.Tensor
  """`pin_standing`. (B,)"""
  tilt: torch.Tensor
  """The axis's angle from vertical. (B,)"""
  goal_dxy: torch.Tensor
  """The centre's horizontal distance from the goal point. (B,)"""
  speed: torch.Tensor
  angular_speed: torch.Tensor

  def rested(
    self,
    grasped: torch.Tensor,
    radius: float,
    height: float,
    speed: float,
    angular_speed: float,
  ) -> torch.Tensor:
    """Plan §14.2's ``rested``: not grasped, the lowest point within ``height`` of the table, the centre within
    ``radius`` of the goal, slower than ``speed`` and turning slower than ``angular_speed``; any pose. (B,)"""
    return (
      ~grasped
      & (self.z_min.abs() < height)
      & (self.goal_dxy < radius)
      & (self.speed < speed)
      & (self.angular_speed < angular_speed)
    )


def pin_rest_state(
  env: ManagerBasedRlEnv,
  object_name: str,
  command_name: str,
  surface_cfg: SceneEntityCfg = _TABLE,
) -> PinRestState:
  """`PinRestState` of the pin ``object_name`` against the goal of ``command_name`` (a `PinPlaceCommand`)."""
  pin: Entity = env.scene[object_name]
  pos, quat = pin.data.root_link_pos_w, pin.data.root_link_quat_w
  below, above = pin_extent_z(quat)
  top = surface_top_z(env, surface_cfg)
  goal = env.command_manager.get_term(command_name).goal_surface
  return PinRestState(
    z_min=pos[:, 2] - below - top,
    z_top=pos[:, 2] + above - top,
    standing=pin_standing(quat),
    tilt=torch.acos(pin_axis_w(quat)[:, 2].abs().clamp(max=1.0)),
    goal_dxy=torch.norm(pos[:, :2] - goal[:, :2], dim=-1),
    speed=torch.norm(pin.data.root_link_lin_vel_w, dim=-1),
    angular_speed=torch.norm(pin.data.root_link_ang_vel_w, dim=-1),
  )


# -- The goal --------------------------------------------------------------------------------------


class PinPlaceCommand(SurfaceOffsetCommand):
  """Plan §14's goal (the module docstring): a point on the table drawn as the cube's area goal, outside
  ``goal_keepout``; `command` follows the pin's pose so that the at-goal tests read the lowest point against the
  table. ``target_pos`` (metrics, debug drawing) is refreshed to it every step."""

  cfg: PinPlaceCommandCfg

  @property
  def goal_surface(self) -> torch.Tensor:
    """The goal point on the table's top surface, world. (B, 3)"""
    from pal_mjlab.tasks.manipulation.mdp.events import _surface_centre_xy

    centre = _surface_centre_xy(self._env, self._surface_cfg)
    top = surface_top_z(self._env, self._surface_cfg)
    return torch.cat([centre + self.offsets, top.unsqueeze(-1)], dim=-1)

  def rest_centre_z(self) -> torch.Tensor:
    """The height the pin's centre has when the pin rests on the table in its current pose. (B,)"""
    below, _ = pin_extent_z(self.object.data.root_link_quat_w)
    return surface_top_z(self._env, self._surface_cfg) + below

  @property
  def command(self) -> torch.Tensor:
    goal = self.goal_surface.clone()
    goal[:, 2] = self.rest_centre_z()
    return goal

  def _draw_clear_of_object(self, env_ids: torch.Tensor) -> torch.Tensor:
    offsets = super()._draw_clear_of_object(env_ids)
    keepout = self.cfg.goal_keepout
    if keepout is None:
      return offsets
    centre = _surface_centre_xy(self._env, self._surface_cfg)[env_ids]
    origin = self._env.scene.env_origins[env_ids, :2]
    pin = self._object_xy(env_ids)
    for _ in range(self.MAX_REDRAWS):
      bad = in_keepout(centre + offsets - origin, keepout) | (
        torch.norm(centre + offsets - pin, dim=-1) < self.cfg.min_object_distance
      )
      if not bool(bad.any()):
        break
      offsets = torch.where(bad.unsqueeze(-1), self._draw(len(env_ids)), offsets)
    return offsets

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    self.target_pos[env_ids] = self.command[env_ids]

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    super()._update_command(env_ids)
    self.target_pos = self.command


@dataclass(kw_only=True)
class PinPlaceCommandCfg(SurfaceOffsetCommandCfg):
  goal_keepout: tuple[float, float, float, float] | None = None
  """A box (x lo, x hi, y lo, y hi, the robot frame) the goal point is redrawn out of (as the pin's distance is);
  None for none."""

  #: M6's yaw grade has no reference here: the pin may end in any pose.
  yaw_reference: str | None = None

  def build(self, env: ManagerBasedRlEnv) -> PinPlaceCommand:
    return PinPlaceCommand(self, env)


# -- The start -------------------------------------------------------------------------------------


#: The lying start's roll: the "x" pose, both plates at 45 deg to the table (plan §14 *The lying pin*).
LYING_ROLL = math.pi / 4


def lying_quat(axis_yaw: torch.Tensor) -> torch.Tensor:
  """A lying pin in the "x" pose with its axis at ``axis_yaw`` (rad, world) in the horizontal plane: the body
  turned `LYING_ROLL` about its axis, tipped 90 deg about y (the axis to +x), then yawed. (B, 4)"""
  zero = torch.zeros_like(axis_yaw)
  roll = quat_from_euler_xyz(zero, zero, torch.full_like(axis_yaw, LYING_ROLL))
  tip = quat_from_euler_xyz(zero, torch.full_like(axis_yaw, math.pi / 2), zero)
  yaw = quat_from_euler_xyz(zero, zero, axis_yaw)
  return quat_mul(yaw, quat_mul(tip, roll))


def reset_pin_start(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  asset_cfg: SceneEntityCfg,
  surface_cfg: SceneEntityCfg,
  lie_probability: float,
  area_x: tuple[float, float],
  area_y: tuple[float, float],
  drop: float = 0.0,
) -> None:
  """Plan §14's start: after `events.reset_objects_on_surface` has stood the pin on its foot (outside the start
  hand's keepout), lay a ``lie_probability`` share of the envs' pins down -- the "x" pose, the axis at a yaw
  uniform in [0, pi), the centre uniform on the area (``area_x`` x ``area_y`` about the surface centre; the keepout
  is for standing pins only), its lowest points ``drop`` over the table."""
  n = len(env_ids)
  if n == 0 or lie_probability <= 0.0:
    return
  device = env.device
  lie = torch.rand(n, device=device) < lie_probability
  if not bool(lie.any()):
    return
  ids = env_ids[lie]
  m = len(ids)
  pin: Entity = env.scene[asset_cfg.name]
  centre = _surface_centre_xy(env, surface_cfg)[ids]
  top = surface_top_z(env, surface_cfg)[ids]
  lo = torch.tensor([area_x[0], area_y[0]], device=device)
  hi = torch.tensor([area_x[1], area_y[1]], device=device)
  xy = centre + sample_uniform(lo, hi, (m, 2), device=device)
  quat = lying_quat(sample_uniform(0.0, math.pi, (m,), device=device))
  below, _ = pin_extent_z(quat)
  pose = torch.cat([xy, (top + below + drop).unsqueeze(-1), quat], dim=-1)
  pin.write_root_link_pose_to_sim(pose, env_ids=ids)
  pin.write_root_link_velocity_to_sim(torch.zeros(m, 6, device=device), env_ids=ids)


# -- Observations ------------------------------------------------------------------------------------


def place_goal_b(
  env: ManagerBasedRlEnv, command_name: str, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """The goal point on the table surface in `base_link` (plan §14.2's target slot), as given: no tracker. (B, 3)"""
  term = env.command_manager.get_term(command_name)
  return _to_base_frame(env.scene[asset_cfg.name], term.goal_surface)


def no_yaw(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The goal's yaw slot: the constant (1, 0) -- the goal has none (plan §14.2). (B, 2)"""
  out = torch.zeros(env.num_envs, 2, device=env.device)
  out[:, 0] = 1.0
  return out


def pin_psi_b(
  env: ManagerBasedRlEnv,
  object_name: str,
  bias_key: str | None = None,
  noise: float = 0.0,
  lag: tuple[int, int] | None = None,
) -> torch.Tensor:
  """Plan §14.2's object yaw slot: (cos 4 psi, sin 4 psi) of `pin_psi`, from the pin's tracker (``lag``), with the
  episode's yaw bias and a per-step jitter while the actor is corrupted, as `box.box_yaw_b` -- which it equals for a
  standing pin. (B, 2)"""
  _, quat_w = _object_pose_w(env, object_name, lag)
  psi = pin_psi(env, quat_w)
  if _corrupted(env):
    if bias_key is not None:
      psi = psi + tracker_yaw_bias(env, bias_key)
    if noise > 0.0:
      psi = psi + (2.0 * torch.rand_like(psi) - 1.0) * noise
  return torch.stack([torch.cos(4.0 * psi), torch.sin(4.0 * psi)], dim=-1)


def pin_axis_b(
  env: ManagerBasedRlEnv,
  object_name: str,
  noise: float = 0.0,
  lag: tuple[int, int] | None = None,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Plan §14.2's tilt slot: x, y in `base_link` of `pin_axis_canonical_w` -- plan §12's `pin_tilt_b` for a
  standing pin (~0 upright), the axis direction (length ~1) for a lying one. Tracker and jitter as §12's. (B, 2)"""
  _, quat_w = _object_pose_w(env, object_name, lag)
  axis = pin_axis_canonical_w(env, quat_w)
  _, base_q = _base_pose(env.scene[asset_cfg.name])
  xy = quat_apply(quat_inv(base_q), axis)[:, :2]
  if noise > 0.0 and _corrupted(env):
    xy = xy + (2.0 * torch.rand_like(xy) - 1.0) * noise
  return xy


def pin_rested(
  env: ManagerBasedRlEnv,
  object_name: str,
  command_name: str,
  grasp_sensor_name: str,
  radius: float,
  height: float,
  speed: float,
  angular_speed: float,
) -> torch.Tensor:
  """1 while the pin is ``rested`` (`PinRestState.rested`), else 0: the critic's input in plan §12's ``seated``
  slot. (B, 1)"""
  st = pin_rest_state(env, object_name, command_name)
  ok = st.rested(
    _grasped_bool(env, grasp_sensor_name), radius, height, speed, angular_speed
  )
  return ok.float().unsqueeze(-1)


# -- Rewards ----------------------------------------------------------------------------------------


def _g(x: torch.Tensor, std: float) -> torch.Tensor:
  return torch.exp(-torch.square(x / std))


def pin_grasp_point_w(
  env: ManagerBasedRlEnv, object_name: str, above_top: float, above_top_lying: float
) -> torch.Tensor:
  """Where the tool point grasps the pin: a standing pin ``above_top`` past its upper end along its up-pointing
  axis (plan §12's `insert.grasp_point_w`); a lying pin straight over its centre, ``above_top_lying`` over its
  highest point. (B, 3)"""
  pin: Entity = env.scene[object_name]
  pos, quat = pin.data.root_link_pos_w, pin.data.root_link_quat_w
  axis = pin_axis_w(quat)
  axis = torch.where(axis[:, 2:3] < 0.0, -axis, axis)
  standing = pos + axis * (0.5 * scene.PIN_LENGTH + above_top)
  _, above = pin_extent_z(quat)
  lying = pos.clone()
  lying[:, 2] = pos[:, 2] + above + above_top_lying
  return torch.where(pin_standing(quat).unsqueeze(-1), standing, lying)


def _tcp_w(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  return robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)


def pin_reach(
  env: ManagerBasedRlEnv,
  object_name: str,
  above_top: float,
  above_top_lying: float,
  reaching_std: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """R-C-strict's ``reach`` to the pose's grasp point (`pin_grasp_point_w`); plan §12's `insert.insert_reach` for a
  standing pin. (B,)"""
  d2 = torch.sum(
    torch.square(
      _tcp_w(env, asset_cfg)
      - pin_grasp_point_w(env, object_name, above_top, above_top_lying)
    ),
    dim=-1,
  )
  return torch.exp(-d2 / reaching_std**2)


def jaw_axis_angle(
  env: ManagerBasedRlEnv, object_name: str, pads_cfg: SceneEntityCfg
) -> torch.Tensor:
  """The angle between the jaw's closing direction (pad to pad, horizontal part) and the horizontal normal to the
  pin's axis, rad in [0, pi/2]: 0 when the jaw closes across a lying pin, pi/2 along it. (B,)"""
  robot: Entity = env.scene[pads_cfg.name]
  pads = robot.data.site_pos_w[:, pads_cfg.site_ids]
  close = pads[:, 1, :2] - pads[:, 0, :2]
  pin: Entity = env.scene[object_name]
  axis = pin_axis_w(pin.data.root_link_quat_w)[:, :2]
  normal = torch.stack([-axis[:, 1], axis[:, 0]], dim=-1)
  cos = (close * normal).sum(-1).abs() / (
    torch.norm(close, dim=-1) * torch.norm(normal, dim=-1)
  ).clamp_min(1e-9)
  return torch.acos(cos.clamp(max=1.0))


def pin_jaw_yaw(
  env: ManagerBasedRlEnv,
  object_name: str,
  pads_cfg: SceneEntityCfg,
  asset_cfg: SceneEntityCfg,
  above_top: float,
  above_top_lying: float,
  angle_std: float,
  dist_std: float,
) -> torch.Tensor:
  """Plan §14.2's jaw-yaw term, for a lying pin only: G(`jaw_axis_angle`, angle_std) x G(tool point to the grasp
  point, dist_std). (B,)"""
  pin: Entity = env.scene[object_name]
  lying = ~pin_standing(pin.data.root_link_quat_w)
  d = torch.norm(
    _tcp_w(env, asset_cfg)
    - pin_grasp_point_w(env, object_name, above_top, above_top_lying),
    dim=-1,
  )
  angle = jaw_axis_angle(env, object_name, pads_cfg)
  return _g(angle, angle_std) * _g(d, dist_std) * lying.float()


def pin_lift_faded(
  env: ManagerBasedRlEnv,
  grasp_sensor_name: str,
  object_name: str,
  command_name: str,
  surface_cfg: SceneEntityCfg,
  max_height: float,
  fade_std: float,
) -> torch.Tensor:
  """M3 for the pin: the lowest point's height over the table as a fraction of ``max_height`` (capped) while held,
  x (1 - G(distance to the pose-following goal, fade_std)) -- R-C's `lift_height` and M3's fade measured from the
  lowest point, so a lying and a standing pin are lifted alike and the lift fades out where the pin is set
  down. (B,)"""
  pin: Entity = env.scene[object_name]
  below, _ = pin_extent_z(pin.data.root_link_quat_w)
  above = pin.data.root_link_pos_w[:, 2] - below - surface_top_z(env, surface_cfg)
  lift = above.clamp(0.0, max_height) / max_height
  command = env.command_manager.get_term(command_name)
  d = torch.norm(command.command - pin.data.root_link_pos_w, dim=-1)
  held = _grasped_bool(env, grasp_sensor_name).float()
  return lift * held * (1.0 - _g(d, fade_std))


def pin_placed_graded(
  env: ManagerBasedRlEnv,
  angular_speed: float,
  **kwargs,
) -> torch.Tensor:
  """M1's per-step ``placed`` for the pin: `rewards.object_placed_graded` (at the goal by the pose-following
  command, released by J, at rest) while the pin also turns slower than ``angular_speed`` rad/s (plan §14.2's
  ``rested``). (B,)"""
  pin: Entity = env.scene[kwargs["object_name"]]
  still = torch.norm(pin.data.root_link_ang_vel_w, dim=-1) < angular_speed
  return object_placed_graded(env, **kwargs) * still.float()


def pin_retreat_height(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  grasp_sensor_name: str,
  support_sensor_name: str,
  jaw_cfg: SceneEntityCfg,
  jaw_open: float,
  asset_cfg: SceneEntityCfg,
  clear_height: float,
  threshold: tuple[float, float],
  velocity_threshold: float = 0.05,
) -> torch.Tensor:
  """M8 for the pin: `rewards.retreat_height` with the tool point measured over the pin's highest point in its
  current pose (the cube's reads its centre + half-height, a standing pin's top). (B,)"""
  from pal_mjlab.tasks.manipulation.mdp.observations import _pad_contact_any

  placed = _placed_bool(
    env,
    command_name,
    object_name,
    grasp_sensor_name,
    threshold,
    velocity_threshold,
    support_sensor_name,
    jaw_cfg,
    jaw_open,
  )
  armed = placed & ~_pad_contact_any(env, grasp_sensor_name)
  pin: Entity = env.scene[object_name]
  _, above = pin_extent_z(pin.data.root_link_quat_w)
  top = pin.data.root_link_pos_w[:, 2] + above
  over = _tcp_w(env, asset_cfg)[:, 2] - top
  return (over / clear_height).clamp(0.0, 1.0) * armed.float()


def grasp_closure(
  env: ManagerBasedRlEnv,
  object_name: str,
  jaw_cfg: SceneEntityCfg,
  asset_cfg: SceneEntityCfg,
  grasp_sensor_name: str,
  support_sensor_name: str,
  above_top: float,
  above_top_lying: float,
  q_touch: float,
  q_touch_lying: float,
  xy_tol: float,
  z_tol: float,
) -> torch.Tensor:
  """Plan §14.2's GC (code PS-SG): clip(q_jaw / q_touch, 0, 1) -- q_touch the pose's first-touch jaw angle --
  while the tool point is within ``xy_tol`` (horizontal) and ``z_tol`` (vertical) of the pose's grasp point, the
  pin not yet grasped and not yet lifted this episode: it pays closing only where closing grasps. (B,)"""
  pin: Entity = env.scene[object_name]
  standing = pin_standing(pin.data.root_link_quat_w)
  d = _tcp_w(env, asset_cfg) - pin_grasp_point_w(
    env, object_name, above_top, above_top_lying
  )
  near = (torch.norm(d[:, :2], dim=-1) < xy_tol) & (d[:, 2].abs() < z_tol)
  q = env.scene[jaw_cfg.name].data.joint_pos[:, jaw_cfg.joint_ids][:, 0]
  touch = torch.where(
    standing, torch.full_like(q, q_touch), torch.full_like(q, q_touch_lying)
  )
  closed = (q / touch).clamp(0.0, 1.0)
  gate = (
    near
    & ~_grasped_bool(env, grasp_sensor_name)
    & ~_was_lifted_bool(env, grasp_sensor_name, support_sensor_name)
  )
  return closed * gate.float()
