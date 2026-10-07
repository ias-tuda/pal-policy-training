"""Plan §12 (`pick-place-insert-plan.md`): push the cross pin through the socket block, let go.

The pin (`scene.get_pin_spec`, the manipulated object, its origin at its centre) is picked from the table and
pushed through the block (`scene.get_socket_block_spec`, origin at the centre of its bottom face on the hole
axis) until its bottom stands on the table. Everything that measures the insertion goes through
`insert_state`, in the **block's own frame**, so that plan §13's held block (in the air, at a tilt) reads the
same quantities:

* the pin's axis is its body z, taken pointing *up the block* (the pin is end-for-end symmetric: whichever end
  is lower in the block is its bottom), its bottom and top `object_half_height` along it from the centre;
* ``height`` -- the pin bottom's height above the block's bottom face, ``depth`` = block height - height (how
  far the bottom is below the hole's entry), ``offset`` -- the bottom's distance from the hole axis,
  ``axis_offset`` -- where the pin's axis crosses the block's top plane, from the hole axis, ``tilt`` -- the
  pin's axis against the block's;
* ``seated`` (plan §12): the bottom within `SEATED_TOL` of the block's bottom face, inside the bore, the pin
  within `SEATED_TILT` of the block's axis. On the table that is the pin standing on the table through the
  block; the table itself is not part of the test (§13's ``through``: the bottom at or below the block's
  bottom face, `InsertState.through`).

`InsertGoalCommand` is the pin centre's seated position, recomputed every step from the block (a pushed block
takes its goal with it), so M1 / J / M6 / M8's at-goal terms read "seated" through their (xy, z) tolerance
(`pick_place.insert`). The terms here are §12's additions: the approach to the entry (`insert_approach`,
replacing M2's goal point), alignment A, depth D, upright U, M3's lift faded by the approach, and the stage
ladder ``LAD-IN`` (`insert_ladder`, S11.2's structure).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.manipulation.mdp.commands import LiftingCommand, LiftingCommandCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv

from pal_mjlab.tasks.manipulation.mdp.box import _corrupted, _free_pose
from pal_mjlab.tasks.manipulation.mdp.observations import (
  _base_pose,
  _grasped_bool,
  _jaw_open_bool,
  _object_pose_w,
  _pad_contact_any,
  _was_lifted_bool,
)
from pal_mjlab.tasks.manipulation.mdp.shapes import object_half_height

_ROBOT = SceneEntityCfg("robot")

#: Plan §12's ``seated``: the pin's bottom within this of the block's bottom face (m), tilted less than
#: `SEATED_TILT` against the block's axis.
SEATED_TOL = 0.003
SEATED_TILT = math.radians(5.0)
#: An axis this close to horizontal in the block's frame (its z component) has no "top-plane crossing".
_FLAT = 0.2


@dataclass
class InsertState:
  """Where the pin is relative to the block, per env (`insert_state`); lengths m, angles rad."""

  axis_w: torch.Tensor
  """The pin's axis, pointing up the block, world. (B, 3)"""
  bottom_w: torch.Tensor
  """The pin's bottom centre, world. (B, 3)"""
  bottom_b: torch.Tensor
  """The pin's bottom centre in the block's frame (origin at its bottom face's centre). (B, 3)"""
  axis_b: torch.Tensor
  """The pin's axis in the block's frame. (B, 3)"""
  block_height: float
  hole_radius: float

  @property
  def height(self) -> torch.Tensor:
    """The pin bottom's height above the block's bottom face. (B,)"""
    return self.bottom_b[:, 2]

  @property
  def depth(self) -> torch.Tensor:
    """How far the pin's bottom is below the hole's entry (the block's top face); negative above it. (B,)"""
    return self.block_height - self.bottom_b[:, 2]

  @property
  def offset(self) -> torch.Tensor:
    """The pin bottom's distance from the hole's axis. (B,)"""
    return torch.norm(self.bottom_b[:, :2], dim=-1)

  @property
  def axis_offset(self) -> torch.Tensor:
    """Where the pin's axis crosses the block's top plane, from the hole's axis; inf for an axis within
    `_FLAT` of the plane. (B,)"""
    uz = self.axis_b[:, 2]
    t = (self.block_height - self.bottom_b[:, 2]) / uz.clamp_min(_FLAT)
    cross = self.bottom_b[:, :2] + self.axis_b[:, :2] * t.unsqueeze(-1)
    out = torch.norm(cross, dim=-1)
    return torch.where(uz > _FLAT, out, torch.full_like(out, math.inf))

  @property
  def tilt(self) -> torch.Tensor:
    """The pin's axis against the block's. (B,)"""
    return torch.acos(self.axis_b[:, 2].clamp(-1.0, 1.0))

  @property
  def approach(self) -> torch.Tensor:
    """Distance of the pin's bottom from the hole's entry, counting only the height *above* it: below the
    entry it is the offset alone, so pushing the pin in never moves it away. (B,)"""
    above = (self.bottom_b[:, 2] - self.block_height).clamp_min(0.0)
    return torch.sqrt(torch.square(self.offset) + torch.square(above))

  def inside(self) -> torch.Tensor:
    """The pin's bottom within the bore's radius of its axis. (B,)"""
    return self.offset < self.hole_radius

  def seated(self, tol: float = SEATED_TOL, tilt: float = SEATED_TILT) -> torch.Tensor:
    """Plan §12's seated: bottom within ``tol`` of the block's bottom face (above or below), inside the bore,
    tilted less than ``tilt``. (B,)"""
    return (self.height.abs() < tol) & self.inside() & (self.tilt < tilt)

  def through(self) -> torch.Tensor:
    """Plan §13's through: the bottom at or below the block's bottom face, inside the bore. (B,)"""
    return (self.height <= 0.0) & self.inside()


def insert_state(
  env: ManagerBasedRlEnv,
  object_name: str,
  block_name: str,
  block_height: float,
  hole_d: float,
) -> InsertState:
  """`InsertState` of the pin ``object_name`` in the block ``block_name`` (`scene.get_socket_block_spec`, its
  height and hole), from the live poses."""
  pin: Entity = env.scene[object_name]
  block: Entity = env.scene[block_name]
  half = object_half_height(env, object_name).unsqueeze(-1)
  pq, pp = pin.data.root_link_quat_w, pin.data.root_link_pos_w
  bq, bp = block.data.root_link_quat_w, block.data.root_link_pos_w
  ez = torch.zeros_like(pp)
  ez[:, 2] = 1.0
  axis = quat_apply(pq, ez)
  block_up = quat_apply(bq, ez)
  sign = torch.where((axis * block_up).sum(-1, keepdim=True) < 0.0, -1.0, 1.0)
  axis = axis * sign
  bottom = pp - axis * half
  inv = quat_inv(bq)
  return InsertState(
    axis_w=axis,
    bottom_w=bottom,
    bottom_b=quat_apply(inv, bottom - bp),
    axis_b=quat_apply(inv, axis),
    block_height=float(block_height),
    hole_radius=0.5 * float(hole_d),
  )


# -- The goal --------------------------------------------------------------------------------------


class InsertGoalCommand(LiftingCommand):
  """Plan §12's goal: the pin centre's seated position -- the block's origin plus half the pin's length along
  the block's axis --, recomputed every step from the block's pose; logs how far the block has moved since the
  reset (``box_displacement``, m, horizontal: S10's metric and `box.box_displacement_penalty`'s reading)."""

  cfg: InsertGoalCommandCfg

  def __init__(self, cfg: InsertGoalCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    #: The block's world xy when the episode's goal was drawn. (B, 2)
    self.box_start_xy = torch.zeros(self.num_envs, 2, device=self.device)
    self.metrics["box_displacement"] = torch.zeros(self.num_envs, device=self.device)

  def _goal(self) -> torch.Tensor:
    pos, quat = _free_pose(self._env.scene[self.cfg.block_name])
    up = torch.zeros_like(pos)
    up[:, 2] = object_half_height(self._env, self.cfg.entity_name)
    return pos + quat_apply(quat, up)

  def box_displacement(self) -> torch.Tensor:
    """Horizontal distance the block has moved since the reset, m. (B,)"""
    pos, _ = _free_pose(self._env.scene[self.cfg.block_name])
    return torch.norm(pos[:, :2] - self.box_start_xy, dim=-1)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    pos, _ = _free_pose(self._env.scene[self.cfg.block_name])
    self.box_start_xy[env_ids] = pos[env_ids, :2]
    self.target_pos[env_ids] = self._goal()[env_ids]

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    super()._update_command(env_ids)
    self.target_pos = self._goal()
    self.metrics["box_displacement"] = self.box_displacement()


@dataclass(kw_only=True)
class InsertGoalCommandCfg(LiftingCommandCfg):
  block_name: str
  """The socket block entity."""

  #: M6's yaw grade has no reference here (a cross in a round hole needs no turning).
  yaw_reference: str | None = None

  # Placement belongs to a reset event, not to the command -- as for `SurfaceOffsetCommandCfg`.
  object_pose_range: LiftingCommandCfg.ObjectPoseRangeCfg | None = None

  def build(self, env: ManagerBasedRlEnv) -> InsertGoalCommand:
    return InsertGoalCommand(self, env)


# -- Observations ------------------------------------------------------------------------------------


def pin_tilt_b(
  env: ManagerBasedRlEnv,
  object_name: str,
  noise: float = 0.0,
  lag: tuple[int, int] | None = None,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Plan §12's ``pin_tilt``: x, y in `base_link` of the pin's up-pointing axis (end-for-end symmetric: the
  axis with z >= 0), 0 when it stands upright. From the pin's tracker (``lag`` steps late, the tracker
  `object_pos` reads; None for the live pose), plus a per-step jitter uniform in +/- ``noise`` while the actor is
  corrupted. (B, 2)"""
  _, quat_w = _object_pose_w(env, object_name, lag)
  ez = torch.zeros(quat_w.shape[0], 3, device=quat_w.device)
  ez[:, 2] = 1.0
  axis = quat_apply(quat_w, ez)
  axis = torch.where(axis[:, 2:3] < 0.0, -axis, axis)
  _, base_q = _base_pose(env.scene[asset_cfg.name])
  xy = quat_apply(quat_inv(base_q), axis)[:, :2]
  if noise > 0.0 and _corrupted(env):
    xy = xy + (2.0 * torch.rand_like(xy) - 1.0) * noise
  return xy


def pin_seated(
  env: ManagerBasedRlEnv,
  object_name: str,
  block_name: str,
  block_height: float,
  hole_d: float,
) -> torch.Tensor:
  """1 while the pin is seated (`InsertState.seated`), else 0; a critic input. (B, 1)"""
  st = insert_state(env, object_name, block_name, block_height, hole_d)
  return st.seated().float().unsqueeze(-1)


# -- Rewards ----------------------------------------------------------------------------------------


def _g(x: torch.Tensor, std: float) -> torch.Tensor:
  return torch.exp(-torch.square(x / std))


def grasp_point_w(
  env: ManagerBasedRlEnv, object_name: str, above_top: float
) -> torch.Tensor:
  """Where the tool point grasps the pin: ``above_top`` past its upper end, along its up-pointing axis. (B, 3)"""
  pin: Entity = env.scene[object_name]
  ez = torch.zeros_like(pin.data.root_link_pos_w)
  ez[:, 2] = 1.0
  axis = quat_apply(pin.data.root_link_quat_w, ez)
  axis = torch.where(axis[:, 2:3] < 0.0, -axis, axis)
  half = object_half_height(env, object_name).unsqueeze(-1)
  return pin.data.root_link_pos_w + axis * (half + above_top)


def insert_reach(
  env: ManagerBasedRlEnv,
  object_name: str,
  above_top: float,
  reaching_std: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """R-C-strict's ``reach`` (`rewards.reach_only`) to the pin's grasp point (`grasp_point_w`) instead of its
  centre: the hand must take the pin by its top for the fingers to clear the block when it is seated (plan §12
  step 0 item 2). (B,)"""
  robot: Entity = env.scene[asset_cfg.name]
  tcp = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  d2 = torch.sum(torch.square(tcp - grasp_point_w(env, object_name, above_top)), dim=-1)
  return torch.exp(-d2 / reaching_std**2)


def insert_approach(
  env: ManagerBasedRlEnv,
  object_name: str,
  block_name: str,
  block_height: float,
  hole_d: float,
  grasp_sensor_name: str,
  support_sensor_name: str,
  std: float,
) -> torch.Tensor:
  """M2's goal terms for the insert: G(`InsertState.approach`, std) once the pin has been lifted (the
  `was_lifted` latch) -- towards the hole's entry from above, and flat once inside it. (B,)"""
  st = insert_state(env, object_name, block_name, block_height, hole_d)
  lifted = _was_lifted_bool(env, grasp_sensor_name, support_sensor_name)
  return _g(st.approach, std) * lifted.float()


def insert_lift_faded(
  env: ManagerBasedRlEnv,
  object_name: str,
  block_name: str,
  block_height: float,
  hole_d: float,
  grasp_sensor_name: str,
  surface_cfg: SceneEntityCfg,
  max_height: float,
  fade_std: float,
) -> torch.Tensor:
  """M3 for the insert: R-C's bounded `lift_height` x (1 - G(`InsertState.approach`, fade_std)) -- faded out at
  the hole's entry and inside it, so pushing the pin down loses no lift income. (B,)"""
  from pal_mjlab.tasks.manipulation.mdp.rewards import lift_height

  st = insert_state(env, object_name, block_name, block_height, hole_d)
  lift = lift_height(env, grasp_sensor_name, object_name, surface_cfg, max_height)
  return lift * (1.0 - _g(st.approach, fade_std))


def _held_lifted(
  env: ManagerBasedRlEnv, grasp_sensor_name: str, support_sensor_name: str
) -> torch.Tensor:
  return _grasped_bool(env, grasp_sensor_name) & _was_lifted_bool(
    env, grasp_sensor_name, support_sensor_name
  )


def insert_align(
  env: ManagerBasedRlEnv,
  object_name: str,
  block_name: str,
  block_height: float,
  hole_d: float,
  grasp_sensor_name: str,
  support_sensor_name: str,
  std: float,
  window: float,
) -> torch.Tensor:
  """Plan §12's A: G(pin bottom's distance from the hole axis, std) while the pin is held after the lift with
  its bottom no more than ``window`` above the hole's entry (and anywhere below it). (B,)"""
  st = insert_state(env, object_name, block_name, block_height, hole_d)
  gate = _held_lifted(env, grasp_sensor_name, support_sensor_name) & (
    st.depth > -window
  )
  return _g(st.offset, std) * gate.float()


def insert_depth(
  env: ManagerBasedRlEnv,
  object_name: str,
  block_name: str,
  block_height: float,
  hole_d: float,
  grasp_sensor_name: str,
  support_sensor_name: str,
  axis_gate: float,
) -> torch.Tensor:
  """Plan §12's D: clip(depth / block height, 0, 1) while the pin is held after the lift and its axis crosses
  the block's top within ``axis_gate`` of the hole's axis -- so a pin lowered beside the block earns nothing. (B,)"""
  st = insert_state(env, object_name, block_name, block_height, hole_d)
  gate = _held_lifted(env, grasp_sensor_name, support_sensor_name) & (
    st.axis_offset < axis_gate
  )
  return (st.depth / block_height).clamp(0.0, 1.0) * gate.float()


def insert_upright(
  env: ManagerBasedRlEnv,
  object_name: str,
  block_name: str,
  block_height: float,
  hole_d: float,
  grasp_sensor_name: str,
  support_sensor_name: str,
  std: float,
  near: float,
) -> torch.Tensor:
  """Plan §12's U: G(pin tilt against the block, std) while the pin is held after the lift with its bottom
  within ``near`` of the hole's entry (the top face's centre). (B,)"""
  st = insert_state(env, object_name, block_name, block_height, hole_d)
  entry = st.bottom_b.clone()
  entry[:, 2] -= block_height
  gate = _held_lifted(env, grasp_sensor_name, support_sensor_name) & (
    torch.norm(entry, dim=-1) < near
  )
  return _g(st.tilt, std) * gate.float()


# -- LAD-IN: the stage ladder (plan §12's IN-LAD; S11.2's structure) -------------------------------


#: The ladder's weight: as S11.2's (`ladder.LADDER_SCALE`): a floor gap of 1 is 6 per step, >= 2 x the largest
#: per-step penalty a policy pays while inserting (`block_moved` 2, saturating; AH10 and the pad contact 1).
LADDER_SCALE = 6.0
#: Floors L0 ... L5 and the shaping inside a level (<= `LADDER_SHAPE`, below the gap to the next floor).
LADDER_FLOORS = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0)
LADDER_SHAPE = 0.45
#: L2's door: the pin's bottom at most this far above the entry and this close to the hole's axis, m.
LADDER_OVER_WINDOW = 0.03
LADDER_OVER_OFFSET = 0.005
#: L5's retreat: the tool point's height over the pin's top, saturating here, m (M8 for the insert).
LADDER_RETREAT = 0.10
#: Env attribute keeping each env's level, for the metrics.
_LEVEL = "_pp_insert_ladder_level"


def insert_ladder(
  env: ManagerBasedRlEnv,
  object_name: str,
  block_name: str,
  block_height: float,
  hole_d: float,
  tcp_cfg: SceneEntityCfg,
  grasp_sensor_name: str,
  support_sensor_name: str,
  surface_cfg: SceneEntityCfg,
  jaw_cfg: SceneEntityCfg,
  jaw_open: float,
  jaw_hold: float,
  above_top: float,
  approach_std: float,
  align_std: float,
  upright_std: float,
  axis_gate: float,
) -> torch.Tensor:
  """``LAD-IN`` (plan §12's IN-LAD), each step the highest level whose condition holds (B,), times nothing
  -- the term's weight is `LADDER_SCALE`:

  ======  =============================================================  ======================================
  level   condition                                                      reward
  ======  =============================================================  ======================================
  L0      default                                                        0.9 (reach + grasp + 2 lift) / 4
  L1      held, lifted once                                              1 + 0.45 G(approach)
  L2      L1 and the bottom <= 3 cm over the entry, <= 5 mm off the axis  2 + 0.45 (A + U) / 2
  L3      L1 and the bottom below the entry, the axis through the hole   3 + 0.45 depth / height
  L4      seated, held                                                    4 + 0.45 open(jaw from hold to J)
  L5      seated, released (J: jaw open, pads off), lifted once           6 + 0.45 retreat / 10 cm
  ======  =============================================================  ======================================
  """
  from pal_mjlab.tasks.manipulation.mdp.rewards import lift_height

  st = insert_state(env, object_name, block_name, block_height, hole_d)
  grasped = _grasped_bool(env, grasp_sensor_name)
  lifted = _was_lifted_bool(env, grasp_sensor_name, support_sensor_name)
  held = grasped & lifted
  seated = st.seated()
  reach = insert_reach(env, object_name, above_top, 0.2, tcp_cfg)
  lift = lift_height(env, grasp_sensor_name, object_name, surface_cfg, 0.10)
  levels = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
  reward = 0.9 * (reach + grasped.float() + 2.0 * lift) / 4.0
  floors = LADDER_FLOORS
  over = (
    held
    & (st.depth > -LADDER_OVER_WINDOW)
    & (st.depth <= 0.0)
    & (st.offset < LADDER_OVER_OFFSET)
  )
  inside = held & (st.depth > 0.0) & (st.axis_offset < axis_gate)
  q = env.scene[jaw_cfg.name].data.joint_pos[:, jaw_cfg.joint_ids][:, 0]
  opened = ((jaw_hold - q) / (jaw_hold - jaw_open)).clamp(0.0, 1.0)
  released = (
    _jaw_open_bool(env, jaw_cfg, jaw_open)
    & ~_pad_contact_any(env, grasp_sensor_name)
    & lifted
  )
  robot: Entity = env.scene[tcp_cfg.name]
  tcp_z = robot.data.site_pos_w[:, tcp_cfg.site_ids].squeeze(1)[:, 2]
  top_z = (
    st.bottom_w + st.axis_w * 2.0 * object_half_height(env, object_name).unsqueeze(-1)
  )[:, 2]
  retreat = ((tcp_z - top_z) / LADDER_RETREAT).clamp(0.0, 1.0)
  rungs = (
    (held, floors[1] + LADDER_SHAPE * _g(st.approach, approach_std)),
    (
      over,
      floors[2]
      + LADDER_SHAPE * 0.5 * (_g(st.offset, align_std) + _g(st.tilt, upright_std)),
    ),
    (
      inside,
      floors[3] + LADDER_SHAPE * (st.depth / block_height).clamp(0.0, 1.0),
    ),
    (held & seated, floors[4] + LADDER_SHAPE * opened),
    (seated & released, floors[5] + LADDER_SHAPE * retreat),
  )
  for k, (cond, value) in enumerate(rungs, start=1):
    levels = torch.where(cond, torch.full_like(levels, k), levels)
    reward = torch.where(cond, value, reward)
  setattr(env, _LEVEL, levels)
  return reward


def insert_ladder_level_is(env: ManagerBasedRlEnv, level: int) -> torch.Tensor:
  """1 while the ladder is at ``level`` (as last computed by the reward), a metric. (B,)"""
  levels = getattr(env, _LEVEL, None)
  if levels is None:
    return torch.zeros(env.num_envs, device=env.device)
  return (levels == level).float()


@dataclasses.dataclass(frozen=True)
class InsertGeometry:
  """The block's numbers every insert term needs (`scene.get_socket_block_spec`), m."""

  block_height: float
  hole_d: float

  def params(self) -> dict:
    return {"block_height": self.block_height, "hole_d": self.hole_d}
