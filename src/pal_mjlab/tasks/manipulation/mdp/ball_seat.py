"""Plan §15 (`pick-place-insert-plan.md` §15): drop a ball into the chamfer of plan §12's socket block -- the seat.

A ball wider than the Ø 38 hole cannot enter it; it comes to rest on the entry chamfer's cone, centred on the hole's
axis, its centre `seat_centre_z` above the block's bottom face (plan §15.0's seat table). Everything that measures
where the ball is against the seat goes through `seat_state`, in the **block's own frame** (as `insert.insert_state`):

* ``offset`` -- the ball centre's distance from the hole's axis; ``dz`` -- its height over the seated centre;
  ``bottom_over_top`` -- the ball's bottom over the block's top face;
* ``seated`` (plan §15.2's ``seated_b``, with the grasp and the speed): within `SEAT_XY_TOL` of the axis and
  `SEAT_Z_TOL` of the seat height.

`BallSeatCommand` is the seated centre, recomputed every step from the block (a pushed block takes its goal with it),
so M1 / J / M8's (xy, z) at-goal tests read "seated" through `SEAT_TOL`; its ``rest_centre_z`` makes the release
latch measure a let-go's height over the seat (`observations._release_latch`). The terms here are §15's additions:
the approach to the drop point with the clearance gate (`seat_approach`), the over-the-seat grade ``A_B``
(`seat_over_grade`), and M1's release gated on the let-go envelope (`seat_release`).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.manipulation.mdp.commands import LiftingCommand, LiftingCommandCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv

from pal_mjlab.tasks.manipulation.mdp.box import _free_pose
from pal_mjlab.tasks.manipulation.mdp.observations import (
  _grasped_bool,
  _release_latch,
  _was_lifted_bool,
)
from pal_mjlab.tasks.manipulation.mdp.rewards import released_at_goal_graded
from pal_mjlab.tasks.manipulation.mdp.shapes import BySize, object_half_height

#: Plan §15.2's ``seated_b``: the ball's centre within this of the hole's axis (m) and of the seat height (m).
SEAT_XY_TOL = 0.005
SEAT_Z_TOL = 0.004
SEAT_TOL = (SEAT_XY_TOL, SEAT_Z_TOL)
#: ... and slower than this (m/s).
SEAT_SPEED = 0.01


@dataclass
class SeatState:
  """Where the ball is relative to the block's seat, per env (`seat_state`); m."""

  centre_b: torch.Tensor
  """The ball's centre in the block's frame (origin at its bottom face's centre on the hole's axis). (B, 3)"""
  radius: torch.Tensor
  """The ball's radius. (B,)"""
  seat_z: torch.Tensor
  """The seated centre's height over the block's bottom face for this radius. (B,)"""
  block_height: float

  @property
  def offset(self) -> torch.Tensor:
    """The ball centre's distance from the hole's axis. (B,)"""
    return torch.norm(self.centre_b[:, :2], dim=-1)

  @property
  def dz(self) -> torch.Tensor:
    """The ball centre's height over the seated centre. (B,)"""
    return self.centre_b[:, 2] - self.seat_z

  @property
  def bottom_over_top(self) -> torch.Tensor:
    """The ball's bottom over the block's top face. (B,)"""
    return self.centre_b[:, 2] - self.radius - self.block_height

  def seated(self, xy: float = SEAT_XY_TOL, z: float = SEAT_Z_TOL) -> torch.Tensor:
    """In the seat: within ``xy`` of the axis and ``z`` of the seat height (grasp and speed not read). (B,)"""
    return (self.offset < xy) & (self.dz.abs() < z)


def seat_centre_z(radius: torch.Tensor, seat: BySize) -> torch.Tensor:
  """The seated ball's centre over the block's bottom face for ``radius`` (m), from the measured ``seat``
  (`BySize` over the radius). (B,)"""
  return seat.at(radius)


def seat_state(
  env: ManagerBasedRlEnv,
  object_name: str,
  block_name: str,
  block_height: float,
  seat: BySize,
) -> SeatState:
  """`SeatState` of the ball ``object_name`` on the block ``block_name``, from the live poses."""
  ball: Entity = env.scene[object_name]
  bp, bq = _free_pose(env.scene[block_name])
  radius = object_half_height(env, object_name)
  return SeatState(
    centre_b=quat_apply(quat_inv(bq), ball.data.root_link_pos_w - bp),
    radius=radius,
    seat_z=seat_centre_z(radius, seat),
    block_height=float(block_height),
  )


def seated_b(
  env: ManagerBasedRlEnv,
  object_name: str,
  block_name: str,
  block_height: float,
  seat: BySize,
  grasp_sensor_name: str,
  speed: float = SEAT_SPEED,
) -> torch.Tensor:
  """Plan §15.2's ``seated_b``: the ball not grasped, in the seat (`SeatState.seated`), slower than ``speed``. (B,)"""
  st = seat_state(env, object_name, block_name, block_height, seat)
  v = torch.norm(env.scene[object_name].data.root_link_lin_vel_w, dim=-1)
  return st.seated() & ~_grasped_bool(env, grasp_sensor_name) & (v < speed)


def ball_seated_obs(
  env: ManagerBasedRlEnv,
  object_name: str,
  block_name: str,
  block_height: float,
  seat: BySize,
  grasp_sensor_name: str,
) -> torch.Tensor:
  """1 while `seated_b`, else 0: the critic's input in plan §12's ``seated`` slot. (B, 1)"""
  ok = seated_b(env, object_name, block_name, block_height, seat, grasp_sensor_name)
  return ok.float().unsqueeze(-1)


def no_tilt(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Plan §12's pin-tilt slot for a ball: the constant (0, 0) -- a ball has no axis (plan §15.2). (B, 2)"""
  return torch.zeros(env.num_envs, 2, device=env.device)


# -- The goal --------------------------------------------------------------------------------------


class BallSeatCommand(LiftingCommand):
  """Plan §15's goal: the seated ball's centre -- the block's origin plus `seat_centre_z` along the block's axis --,
  recomputed every step from the block's pose; logs how far the block has moved since the reset (``box_displacement``,
  m, horizontal: `box.box_displacement_penalty`'s reading). ``rest_centre_z`` is the goal's height, so that the release
  latch measures a let-go's height over the seat, not over the table."""

  cfg: BallSeatCommandCfg

  def __init__(self, cfg: BallSeatCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    #: The block's world xy when the episode's goal was drawn. (B, 2)
    self.box_start_xy = torch.zeros(self.num_envs, 2, device=self.device)
    self.metrics["box_displacement"] = torch.zeros(self.num_envs, device=self.device)

  def _goal(self, up: float = 0.0) -> torch.Tensor:
    pos, quat = _free_pose(self._env.scene[self.cfg.block_name])
    off = torch.zeros_like(pos)
    radius = object_half_height(self._env, self.cfg.entity_name)
    off[:, 2] = seat_centre_z(radius, self.cfg.seat) + up
    return pos + quat_apply(quat, off)

  def drop_point(self) -> torch.Tensor:
    """Where the ball's centre is let go from: `cfg.drop_height` above the seated centre on the hole's axis. (B, 3)"""
    return self._goal(self.cfg.drop_height)

  def rest_centre_z(self) -> torch.Tensor:
    """The seated centre's world height, the release latch's resting height. (B,)"""
    return self._goal()[:, 2]

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
class BallSeatCommandCfg(LiftingCommandCfg):
  block_name: str
  """The socket block entity."""

  seat: BySize
  """The seated centre's height over the block's bottom face by the ball's radius (`seat_centre_z`)."""

  drop_height: float
  """The drop point's height over the seated centre, m (plan §15.2's ``h_drop``)."""

  #: M6's yaw grade has no reference here (a ball has no yaw).
  yaw_reference: str | None = None

  # Placement belongs to a reset event, not to the command -- as for `SurfaceOffsetCommandCfg`.
  object_pose_range: LiftingCommandCfg.ObjectPoseRangeCfg | None = None

  def build(self, env: ManagerBasedRlEnv) -> BallSeatCommand:
    return BallSeatCommand(self, env)


# -- Rewards ----------------------------------------------------------------------------------------


def _g(x: torch.Tensor, std: float) -> torch.Tensor:
  return torch.exp(-torch.square(x / std))


def _drop_frame(
  env: ManagerBasedRlEnv, command_name: str, object_name: str, block_name: str
) -> torch.Tensor:
  """The ball's centre relative to the drop point, in the block's frame. (B, 3)"""
  term = env.command_manager.get_term(command_name)
  _, bq = _free_pose(env.scene[block_name])
  ball = env.scene[object_name].data.root_link_pos_w
  return quat_apply(quat_inv(bq), ball - term.drop_point())


def seat_approach(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  block_name: str,
  block_height: float,
  seat: BySize,
  grasp_sensor_name: str,
  support_sensor_name: str,
  std: float,
  gate_radius: float,
  gate_clearance: float,
  seat_radius: float,
) -> torch.Tensor:
  """Plan §15.2's M2 goal terms: G(distance of the ball's centre from the drop point, std) once the ball has been
  lifted (the `was_lifted` latch), counting only the height *above* the drop point -- below it the axis offset
  alone, so a ball let go into the seat keeps the income --, and the clearance gate: within ``gate_radius`` (xy) of
  the axis the term pays only while the ball's bottom is more than ``gate_clearance`` over the block's top, or the
  ball is inside the seat's cone (centre within ``seat_radius`` of the axis), so a held ball is not dragged into the
  block's side. (B,)"""
  rel = _drop_frame(env, command_name, object_name, block_name)
  st = seat_state(env, object_name, block_name, block_height, seat)
  dist = torch.sqrt(torch.square(st.offset) + torch.square(rel[:, 2].clamp_min(0.0)))
  lifted = _was_lifted_bool(env, grasp_sensor_name, support_sensor_name)
  blocked = (
    (st.offset < gate_radius)
    & (st.bottom_over_top <= gate_clearance)
    & (st.offset > seat_radius)
  )
  return _g(dist, std) * (lifted & ~blocked).float()


def seat_over_grade(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  block_name: str,
  block_height: float,
  seat: BySize,
  grasp_sensor_name: str,
  support_sensor_name: str,
  xy_std: float,
  speed_std: float,
  above: float,
  below: float,
) -> torch.Tensor:
  """Plan §15.2's ``A_B``: G(the ball centre's distance from the axis, xy_std) x G(the ball's speed, speed_std) while
  the ball is held after the lift with its centre between ``below`` under and ``above`` over the drop point. (B,)"""
  rel = _drop_frame(env, command_name, object_name, block_name)
  st = seat_state(env, object_name, block_name, block_height, seat)
  speed = torch.norm(env.scene[object_name].data.root_link_lin_vel_w, dim=-1)
  held = _grasped_bool(env, grasp_sensor_name) & _was_lifted_bool(
    env, grasp_sensor_name, support_sensor_name
  )
  window = (rel[:, 2] <= above) & (rel[:, 2] >= -below)
  return _g(st.offset, xy_std) * _g(speed, speed_std) * (held & window).float()


def let_go_in_envelope(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  grasp_sensor_name: str,
  support_sensor_name: str,
  surface_cfg: SceneEntityCfg,
  jaw_cfg: SceneEntityCfg,
  jaw_open,
  block_height: float,
  seat: BySize,
  envelope: tuple[float, float, float],
) -> torch.Tensor:
  """Whether the latest J-release's let-go (`observations._release_latch`) was inside plan §15.2's envelope
  ``(xy, bottom, speed)``: the ball's centre within xy of the axis, its bottom at most ``bottom`` over the block's
  top, slower than ``speed``. The latch's height is the centre's over the seated centre (`BallSeatCommand`). (B,)"""
  latch = _release_latch(
    env,
    command_name,
    object_name,
    grasp_sensor_name,
    support_sensor_name,
    surface_cfg,
    jaw_cfg,
    jaw_open,
  )
  radius = object_half_height(env, object_name)
  bottom = latch.height + seat_centre_z(radius, seat) - radius - block_height
  xy, top, speed = envelope
  return (
    latch.released & (latch.dxy <= xy) & (bottom <= top) & (latch.rel_speed <= speed)
  )


def seat_release(
  env: ManagerBasedRlEnv,
  block_height: float,
  seat: BySize,
  envelope: tuple[float, float, float],
  **graded,
) -> torch.Tensor:
  """Plan §15.2's M1 ``release``: `rewards.released_at_goal_graded` (the ball at the seat, J-released after the lift,
  x M6's grade) paid only if the let-go was inside the envelope (`let_go_in_envelope`). (B,)"""
  ok = let_go_in_envelope(
    env,
    graded["command_name"],
    graded["object_name"],
    graded["grasp_sensor_name"],
    graded["support_sensor_name"],
    graded["surface_cfg"],
    graded["jaw_cfg"],
    graded["jaw_open"],
    block_height,
    seat,
    envelope,
  )
  return released_at_goal_graded(env, **graded) * ok.float()
