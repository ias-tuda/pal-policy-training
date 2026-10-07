"""Plan S10 (`pick-place-v1-objects-plan.md`): put the cube into an open box.

The box is a free entity (`scene.get_box_spec`: a floor and four walls, its origin at the
centre of its bottom face) put down on the table at every reset (`reset_box_on_surface`), and
the goal is its floor: `BoxGoalCommand` is the floor's top centre plus the cube's half-height,
recomputed every step so that a pushed box takes its goal with it. "In the box"
(`object_in_box_bool`) is the cube's centre inside the inner walls and its bottom below the rim.

The actor sees the goal through the box's own tracker (`box_goal_position_tracked_b`): the
box's pose as FoundationPose reports it, lagged like the cube's, and both poses carry a tracker
bias split into a part shared by every object in the episode (the camera) and a part of each
object's own (`reset_tracker_bias`; plan S9/S10: 6 + 4 mm). The critic keeps the live goal.

Plan S10.1 (the box on the wide area, at any yaw): `reset_box_in_area` draws the box from an area
of the table at least a distance from the cube, at a yaw in +/- 45 deg; with ``BOX_OBS=pose`` the
actor sees the box itself -- its tracked position (`object_position_biased_b` on the box) and its
yaw as (cos 4 psi, sin 4 psi) (`box_yaw_b`, 4-fold symmetric like the box) -- instead of the goal
point; and M6's yaw grade is taken against the box (`BoxGoalCommandCfg.yaw_reference`).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.dr.geom import _recompute_geom_bounds
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.manipulation.mdp.commands import LiftingCommand, LiftingCommandCfg
from mjlab.utils.lab_api.math import (
  euler_xyz_from_quat,
  quat_apply,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
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
  _to_base_frame,
)
from pal_mjlab.tasks.manipulation.mdp.shapes import object_half_height

_ROBOT = SceneEntityCfg("robot")


def _free_pose(entity: Entity) -> tuple[torch.Tensor, torch.Tensor]:
  """A free body's world pose from its joint's ``qpos`` -- what a reset event wrote, valid
  before the next ``forward()`` (the derived ``xpos`` is stale until then). (B, 3), (B, 4)"""
  adr = entity.data.indexing.free_joint_q_adr
  q = entity.data.data.qpos[:, adr]
  return q[:, :3], q[:, 3:7]


def box_floor_top_w(
  env: ManagerBasedRlEnv, box_name: str, floor: float
) -> torch.Tensor:
  """World position of the box floor's top centre (the ``{box}_floor`` site), from the
  free joint. (B, 3)"""
  pos, quat = _free_pose(env.scene[box_name])
  up = torch.zeros_like(pos)
  up[:, 2] = floor
  return pos + quat_apply(quat, up)


def _wall_ids(env: ManagerBasedRlEnv, box_name: str, walls: tuple[str, ...]):
  box: Entity = env.scene[box_name]
  local = [box.geom_names.index(w) for w in walls]
  return box.indexing.geom_ids[local]


def box_inner_half(
  env: ManagerBasedRlEnv, box_name: str, walls: tuple[str, ...]
) -> torch.Tensor:
  """Half the box's inner width, m, per env: the +x wall's inner face from the box's centre
  (`scene.get_box_spec`: walls +x, -x, +y, -y; BX-DR draws it per world). (B,)"""
  gid = _wall_ids(env, box_name, walls)[0]
  inner = env.sim.model.geom_pos[:, gid, 0] - env.sim.model.geom_size[:, gid, 0]
  return inner.expand(env.num_envs)


def box_rim_z(
  env: ManagerBasedRlEnv, box_name: str, walls: tuple[str, ...]
) -> torch.Tensor:
  """World z of the rim, the highest wall top, per env. (B,)"""
  return surface_top_z(env, SceneEntityCfg(box_name, geom_names=walls))


def object_in_box_bool(
  env: ManagerBasedRlEnv,
  object_name: str,
  box_name: str,
  walls: tuple[str, ...],
) -> torch.Tensor:
  """Plan S10's ``in_box``: the object's centre inside the inner walls (in the box's own yawed
  frame) and its bottom below the rim. (B,)

  The plan's words are "centre below the rim", but a 5 cm cube resting on the 5 mm floor of a
  2.5 cm box has its centre exactly at the rim; the bottom is what has to be below it."""
  obj: Entity = env.scene[object_name]
  box: Entity = env.scene[box_name]
  rel = obj.data.root_link_pos_w - box.data.root_link_pos_w
  q = box.data.root_link_quat_w
  q_inv = torch.cat([q[:, :1], -q[:, 1:]], dim=-1)
  local = quat_apply(q_inv, rel)
  half = box_inner_half(env, box_name, walls)
  inside = (local[:, 0].abs() < half) & (local[:, 1].abs() < half)
  bottom = obj.data.root_link_pos_w[:, 2] - object_half_height(env, object_name)
  return inside & (bottom < box_rim_z(env, box_name, walls))


class BoxGoalCommand(LiftingCommand):
  """The goal of plan S10: the box floor's top centre plus the object's half-height -- where the
  object's centre is when it rests on the floor. Recomputed every step from the box's pose, so
  a knocked box carries its goal; logs how far the box has moved since the reset
  (``box_displacement``, m, horizontal)."""

  cfg: BoxGoalCommandCfg

  def __init__(self, cfg: BoxGoalCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    #: The box's world xy when the episode's goal was drawn. (B, 2)
    self.box_start_xy = torch.zeros(self.num_envs, 2, device=self.device)
    self.metrics["box_displacement"] = torch.zeros(self.num_envs, device=self.device)

  def _goal(self) -> torch.Tensor:
    floor = box_floor_top_w(self._env, self.cfg.box_name, self.cfg.floor)
    floor[:, 2] += object_half_height(self._env, self.cfg.entity_name)
    return floor

  def box_displacement(self) -> torch.Tensor:
    """Horizontal distance the box has moved since the reset, m. (B,)"""
    pos, _ = _free_pose(self._env.scene[self.cfg.box_name])
    return torch.norm(pos[:, :2] - self.box_start_xy, dim=-1)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    pos, _ = _free_pose(self._env.scene[self.cfg.box_name])
    self.box_start_xy[env_ids] = pos[env_ids, :2]
    self.target_pos[env_ids] = self._goal()[env_ids]

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    super()._update_command(env_ids)
    self.target_pos = self._goal()
    self.metrics["box_displacement"] = self.box_displacement()


@dataclass(kw_only=True)
class BoxGoalCommandCfg(LiftingCommandCfg):
  box_name: str
  """The box entity."""

  floor: float
  """Thickness of the box floor, m (`scene.BOX_FLOOR`): its top is this far above the box's
  origin, the centre of its bottom face."""

  yaw_reference: str | None = None
  """Plan S10.1: the entity M6's yaw grade measures the object's yaw against (the box, which
  stands at any yaw there), read by the release latch (`observations._release_latch`); None
  measures it against the world's axes, as every S10 / DEP1 run did (box at 0 +/- 5 deg)."""

  # Placement belongs to a reset event, not to the command -- as for `SurfaceOffsetCommandCfg`.
  object_pose_range: LiftingCommandCfg.ObjectPoseRangeCfg | None = None

  def build(self, env: ManagerBasedRlEnv) -> BoxGoalCommand:
    return BoxGoalCommand(self, env)


def reset_box_on_surface(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  asset_cfg: SceneEntityCfg,
  surface_cfg: SceneEntityCfg,
  offset: tuple[float, float],
  half_range: tuple[float, float],
  yaw: float,
) -> None:
  """Put the box down on the surface: its bottom on the top face, its centre at ``offset``
  +/- ``half_range`` (x, y) from the surface centre -- table-relative, as the cube's spawn and
  baseline B's target box -- at a yaw drawn in +/- ``yaw``, at rest."""
  n = len(env_ids)
  device = env.device
  box: Entity = env.scene[asset_cfg.name]
  centre = _surface_centre_xy(env, surface_cfg)[env_ids]
  top = surface_top_z(env, surface_cfg)[env_ids]
  lo = torch.tensor(
    [offset[0] - half_range[0], offset[1] - half_range[1]], device=device
  )
  hi = torch.tensor(
    [offset[0] + half_range[0], offset[1] + half_range[1]], device=device
  )
  xy = centre + sample_uniform(lo, hi, (n, 2), device=device)
  zeros = torch.zeros(n, device=device)
  quat = quat_from_euler_xyz(
    zeros, zeros, sample_uniform(-yaw, yaw, (n,), device=device)
  )
  pose = torch.cat([xy, top.unsqueeze(-1), quat], dim=-1)
  box.write_root_link_pose_to_sim(pose, env_ids=env_ids)
  box.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids=env_ids)


#: Rounds of redrawing in `reset_box_in_area` before the stragglers take the area's farthest corner.
BOX_AREA_REDRAWS = 32


def reset_box_in_area(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  asset_cfg: SceneEntityCfg,
  surface_cfg: SceneEntityCfg,
  object_name: str,
  area_x: tuple[float, float],
  area_y: tuple[float, float],
  yaw: float,
  min_distance: float,
  keepout: tuple[float, float, float, float] | None = None,
) -> None:
  """Plan S10.1: put the box down on the surface, its centre drawn uniformly from the area
  (``area_x`` x ``area_y``, relative to the surface centre -- table-relative, so DR-scene's table
  shift moves it), at least ``min_distance`` (xy, centre to centre) from the object, at a yaw drawn
  in +/- ``yaw``, at rest.

  Runs after the object's reset event and reads the object from its free joint's ``qpos`` (the
  derived ``xpos`` is stale until the next ``forward()``, S7-8). Vectorised rejection as
  `SurfaceOffsetCommand._draw_clear_of_object`: each round redraws only the envs still too close;
  after `BOX_AREA_REDRAWS` rounds the stragglers take the area's corner farthest from the object,
  which is at least ``min_distance`` away when the object was drawn where one is
  (`events.reset_objects_on_surface`'s ``goal_area``). Plan §12's ``keepout`` (x lo, x hi, y lo, y hi about the env
  origin, `events.reset_objects_on_surface`'s) is redrawn the same way; a corner straggler is not checked
  against it."""
  n = len(env_ids)
  device = env.device
  box: Entity = env.scene[asset_cfg.name]
  centre = _surface_centre_xy(env, surface_cfg)[env_ids]
  top = surface_top_z(env, surface_cfg)[env_ids]
  cube, _ = _free_pose(env.scene[object_name])
  cube = cube[env_ids, :2]
  lo = torch.tensor([area_x[0], area_y[0]], device=device)
  hi = torch.tensor([area_x[1], area_y[1]], device=device)
  offsets = sample_uniform(lo, hi, (n, 2), device=device)
  origin = env.scene.env_origins[env_ids, :2]

  def rejected(off: torch.Tensor) -> torch.Tensor:
    bad = torch.norm(centre + off - cube, dim=-1) < min_distance
    if keepout is not None:
      bad |= in_keepout(centre + off - origin, keepout)
    return bad

  for _ in range(BOX_AREA_REDRAWS):
    too_close = rejected(offsets)
    if not bool(too_close.any()):
      break
    redraw = sample_uniform(lo, hi, (n, 2), device=device)
    offsets = torch.where(too_close.unsqueeze(-1), redraw, offsets)
  too_close = torch.norm(centre + offsets - cube, dim=-1) < min_distance
  if bool(too_close.any()):
    corners = torch.tensor(
      [
        [area_x[0], area_y[0]],
        [area_x[0], area_y[1]],
        [area_x[1], area_y[0]],
        [area_x[1], area_y[1]],
      ],
      device=device,
    )
    far = torch.norm(centre.unsqueeze(1) + corners - cube.unsqueeze(1), dim=-1).argmax(
      -1
    )
    offsets = torch.where(too_close.unsqueeze(-1), corners[far], offsets)
  zeros = torch.zeros(n, device=device)
  quat = quat_from_euler_xyz(
    zeros, zeros, sample_uniform(-yaw, yaw, (n,), device=device)
  )
  pose = torch.cat([centre + offsets, top.unsqueeze(-1), quat], dim=-1)
  box.write_root_link_pose_to_sim(pose, env_ids=env_ids)
  box.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids=env_ids)


@requires_model_fields("geom_size", "geom_pos", "geom_rbound", "geom_aabb")
def set_box_geometry(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg,
  walls: tuple[str, ...],
  floor_name: str,
  wall: float,
  floor: float,
  inner_range: tuple[float, float],
  height_range: tuple[float, float],
  feet: tuple[str, ...] = (),
  foot_inset: float = 0.0,
) -> None:
  """Startup event, plan S10's BX-DR: draw each world's box inner width and wall height
  (above the floor) uniformly, and rebuild the floor and the four walls (`scene.get_box_spec`'s
  layout: walls +x, -x, +y, -y, the y walls across the outer width) -- sizes, positions and the
  bounds the broadphase reads. Wall and floor thickness stay; ``feet`` (+x+y, +x-y, -x+y, -x-y,
  `scene.box_foot_geom_names`) move to ``foot_inset`` inside the new outer corners. The mass is the
  DR of the box's mass, a separate event."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  n = len(env_ids)
  device = env.device
  box: Entity = env.scene[asset_cfg.name]
  px, mx, py, my = _wall_ids(env, asset_cfg.name, walls)
  floor_id = box.indexing.geom_ids[box.geom_names.index(floor_name)]
  half_in = 0.5 * sample_uniform(inner_range[0], inner_range[1], (n,), device=device)
  h = sample_uniform(height_range[0], height_range[1], (n,), device=device)
  half_out = half_in + wall
  size, pos = env.sim.model.geom_size, env.sim.model.geom_pos
  wall_z = floor + h / 2
  for gid, sign in ((px, 1.0), (mx, -1.0)):
    pos[env_ids, gid, 0] = sign * (half_in + wall / 2)
    pos[env_ids, gid, 2] = wall_z
    size[env_ids, gid, 1] = half_in
    size[env_ids, gid, 2] = h / 2
  for gid, sign in ((py, 1.0), (my, -1.0)):
    pos[env_ids, gid, 1] = sign * (half_in + wall / 2)
    pos[env_ids, gid, 2] = wall_z
    size[env_ids, gid, 0] = half_out
    size[env_ids, gid, 2] = h / 2
  size[env_ids, floor_id, 0] = half_out
  size[env_ids, floor_id, 1] = half_out
  if feet:
    c = half_out - foot_inset
    for name, (sx, sy) in zip(feet, ((1, 1), (1, -1), (-1, 1), (-1, -1)), strict=True):
      gid = box.indexing.geom_ids[box.geom_names.index(name)]
      pos[env_ids, gid, 0] = sx * c
      pos[env_ids, gid, 1] = sy * c
  _recompute_geom_bounds(env, env_ids.to(dtype=torch.int), asset_cfg)


#: Attribute on the env holding the tracker biases of the current episode, per env.
_TRACKER_BIAS = "_pal_tracker_bias"


def _tracker_bias(env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
  bias = getattr(env, _TRACKER_BIAS, None)
  if bias is None:
    bias = {}
    setattr(env, _TRACKER_BIAS, bias)
  return bias


def tracker_bias(env: ManagerBasedRlEnv, key: str) -> torch.Tensor:
  """The position bias the tracker puts on object ``key`` this episode: the shared part plus
  the object's own, in the base frame, m; zero before the first reset. (B, 3)"""
  bias = _tracker_bias(env)
  zero = torch.zeros(env.num_envs, 3, device=env.device)
  return bias.get("shared", zero) + bias.get(key, zero)


def reset_tracker_bias(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  shared: float,
  own: float,
  keys: tuple[str, ...],
) -> None:
  """Draw this episode's tracker biases (plan S9/S10's split): one shared by every tracked
  object -- the camera's -- uniform in +/- ``shared`` per axis, and one per object in ``keys``,
  uniform in +/- ``own``. A fresh draw per reset, not an accumulation (the "abs" lesson of
  mjlab's additive-bias noise, CLAUDE.md)."""
  bias = _tracker_bias(env)
  n = len(env_ids)
  for key, half in (("shared", shared), *((k, own) for k in keys)):
    if key not in bias:
      bias[key] = torch.zeros(env.num_envs, 3, device=env.device)
    bias[key][env_ids] = sample_uniform(-half, half, (n, 3), device=env.device)


def _corrupted(env: ManagerBasedRlEnv) -> bool:
  """Whether the actor's observations are corrupted (training), not the play / eval override
  that turns its noise off: the bias is part of that noise."""
  return bool(env.cfg.observations["actor"].enable_corruption)


def object_position_biased_b(
  env: ManagerBasedRlEnv,
  object_name: str,
  bias_key: str,
  asset_cfg: SceneEntityCfg = _ROBOT,
  lag: tuple[int, int] | None = None,
) -> torch.Tensor:
  """`observations.object_position_b` (tracked ``lag`` steps late) plus the tracker's bias on
  ``bias_key`` (`tracker_bias`) while the actor is corrupted. (B, 3)"""
  robot: Entity = env.scene[asset_cfg.name]
  pos_w, _ = _object_pose_w(env, object_name, lag)
  pos_b = _to_base_frame(robot, pos_w)
  if _corrupted(env):
    pos_b = pos_b + tracker_bias(env, bias_key)
  return pos_b


def box_goal_position_tracked_b(
  env: ManagerBasedRlEnv,
  box_name: str,
  object_name: str,
  floor: float,
  bias_key: str,
  asset_cfg: SceneEntityCfg = _ROBOT,
  lag: tuple[int, int] | None = None,
) -> torch.Tensor:
  """The goal as the actor sees it on the robot: the box floor's top centre plus the object's
  half-height, from the box's pose as its tracker reports it (``lag`` steps late; None for the
  live pose), plus the box's tracker bias while the actor is corrupted, in the base frame.
  (B, 3)"""
  robot: Entity = env.scene[asset_cfg.name]
  pos_w, quat_w = _object_pose_w(env, box_name, lag)
  up = torch.zeros_like(pos_w)
  up[:, 2] = floor
  goal_w = pos_w + quat_apply(quat_w, up)
  goal_w[:, 2] += object_half_height(env, object_name)
  goal_b = _to_base_frame(robot, goal_w)
  if _corrupted(env):
    goal_b = goal_b + tracker_bias(env, bias_key)
  return goal_b


#: Attribute on the env holding the per-episode yaw biases of the tracked objects, rad, per env.
_TRACKER_YAW_BIAS = "_pal_tracker_yaw_bias"


def reset_tracker_yaw_bias(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  half: float,
  keys: tuple[str, ...],
) -> None:
  """Plan S10.1: draw this episode's tracker yaw bias for each object in ``keys``, uniform in
  +/- ``half`` rad -- the box's own draw, as baseline B's cube yaw bias (0.03 rad). A fresh draw
  per reset, not an accumulation."""
  bias = getattr(env, _TRACKER_YAW_BIAS, None)
  if bias is None:
    bias = {}
    setattr(env, _TRACKER_YAW_BIAS, bias)
  for key in keys:
    if key not in bias:
      bias[key] = torch.zeros(env.num_envs, device=env.device)
    bias[key][env_ids] = sample_uniform(-half, half, (len(env_ids),), device=env.device)


def tracker_yaw_bias(env: ManagerBasedRlEnv, key: str) -> torch.Tensor:
  """The yaw bias the tracker puts on object ``key`` this episode, rad; zero before the first
  reset. (B,)"""
  bias = getattr(env, _TRACKER_YAW_BIAS, {})
  return bias.get(key, torch.zeros(env.num_envs, device=env.device))


def yaw_relative_to_base(env: ManagerBasedRlEnv, quat_w: torch.Tensor) -> torch.Tensor:
  """The yaw of a world orientation relative to the robot's `base_link`, rad. (B,)"""
  _, base_q = _base_pose(env.scene[_ROBOT.name])
  _, _, yaw = euler_xyz_from_quat(quat_mul(quat_inv(base_q), quat_w))
  return yaw


def box_yaw_b(
  env: ManagerBasedRlEnv,
  box_name: str,
  bias_key: str | None = None,
  noise: float = 0.0,
  fold: int = 4,
  lag: tuple[int, int] | None = None,
) -> torch.Tensor:
  """Plan S10.1's ``box_yaw``: the box's yaw psi relative to the robot base, as (cos fold psi,
  sin fold psi) -- the same for psi and psi + 90 deg, as FoundationPose on the 4-fold symmetric box
  may report either, and continuous across +/- 45 deg. From the box's tracker (``lag`` steps late;
  None for the live pose); while the actor is corrupted psi carries the episode's yaw bias on
  ``bias_key`` (`reset_tracker_yaw_bias`) and a per-step jitter uniform in +/- ``noise`` rad, both on
  the angle, not on the encoding. (B, 2)"""
  _, quat_w = _object_pose_w(env, box_name, lag)
  psi = yaw_relative_to_base(env, quat_w)
  if _corrupted(env):
    if bias_key is not None:
      psi = psi + tracker_yaw_bias(env, bias_key)
    if noise > 0.0:
      psi = psi + (2.0 * torch.rand_like(psi) - 1.0) * noise
  return torch.stack([torch.cos(fold * psi), torch.sin(fold * psi)], dim=-1)


def box_displacement_penalty(
  env: ManagerBasedRlEnv, command_name: str, scale: float
) -> torch.Tensor:
  """How far the box has been pushed since the reset, as a fraction of ``scale`` (m), capped
  at 1 (plan S10: "pushing the box costs"). Weight it negative. (B,)"""
  command = env.command_manager.get_term(command_name)
  return (command.box_displacement() / scale).clamp(0.0, 1.0)


def contact_while_held(
  env: ManagerBasedRlEnv, grasp_sensor_name: str, sensor_names: tuple[str, ...]
) -> torch.Tensor:
  """1.0 while the object is held and any of the sensors reports a contact -- plan S10's
  BX-RIM: the hand or the held cube on a box wall. Weight it negative. (B,)"""
  hit = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  for name in sensor_names:
    found = env.scene[name].data.found
    assert found is not None, f"sensor {name!r} does not provide 'found'"
    hit |= found.bool().any(dim=-1)
  return (hit & _grasped_bool(env, grasp_sensor_name)).float()
