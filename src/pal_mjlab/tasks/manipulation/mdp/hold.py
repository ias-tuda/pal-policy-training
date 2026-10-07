"""Plan §13 (`pick-place-insert-plan.md`): the left's side grasp of the socket block and its hold at H.

L-H picks the 56 mm block off the table with the hand **horizontal** -- its approach axis (the TCP
site's +z, the jaw's pointing direction) level, the jaw closing on two opposite side faces -- so that
the block's top, and the hole in it, stay free for the right hand's pin; it then holds the block level
and still at H. Everything here reads the block's live pose; the actor's view of it is the tracked one
(S10.1's `box_yaw_b` for its yaw, `object_up_b` for its tilt).

The block of L-H has its origin at its **centre** (`pick_place.hold.held_block_spec`), as every picked
object here, so `shapes.object_half_height`, the spawn, the lift and the goal treat it as a cube.
"""

from __future__ import annotations

import math

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_inv,
)

from pal_mjlab.tasks.manipulation.mdp.observations import (
  _ROBOT,
  _base_pose,
  _grasped_bool,
  _object_pose_w,
)


def hand_axes_w(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> tuple[torch.Tensor, torch.Tensor]:
  """The hand's approach axis (the TCP site's +z) and its jaw axis (+x), world frame. (B, 3) each."""
  robot: Entity = env.scene[asset_cfg.name]
  rot = matrix_from_quat(robot.data.site_quat_w[:, asset_cfg.site_ids].squeeze(1))
  return rot[..., :, 2], rot[..., :, 0]


def approach_elevation(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """|Angle of the approach axis above or below the horizontal|, rad: 0 for a side grasp, pi/2 for a
  top-down one. (B,)"""
  approach, _ = hand_axes_w(env, asset_cfg)
  return torch.asin(approach[:, 2].clamp(-1.0, 1.0)).abs()


def side_grade(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, std: float
) -> torch.Tensor:
  """G(approach elevation, ``std``): 1 for a horizontal hand, ~0 for a top-down one. (B,)"""
  return torch.exp(-torch.square(approach_elevation(env, asset_cfg) / std))


def object_up_w(env: ManagerBasedRlEnv, object_name: str) -> torch.Tensor:
  """The object's +z axis in the world. (B, 3)"""
  obj: Entity = env.scene[object_name]
  return matrix_from_quat(obj.data.root_link_quat_w)[..., :, 2]


def axis_tilt(env: ManagerBasedRlEnv, object_name: str) -> torch.Tensor:
  """Angle between the object's own +z and the world's, rad: the hole's tilt for the socket block (not
  `observations.object_tilt`, the most vertical of the three axes, which reads 0 for a block on its side).
  (B,)"""
  return torch.acos(object_up_w(env, object_name)[:, 2].clamp(-1.0, 1.0))


def _base_dir_w(
  env: ManagerBasedRlEnv, direction_b: tuple[float, float, float]
) -> torch.Tensor:
  """A direction given in the robot base frame, in the world. (B, 3)"""
  robot: Entity = env.scene["robot"]
  _, base_quat = _base_pose(robot)
  d = torch.tensor(direction_b, device=env.device, dtype=torch.float32).expand(
    env.num_envs, 3
  )
  return quat_apply(base_quat, d)


def side_grasp_targets_w(
  env: ManagerBasedRlEnv,
  object_name: str,
  approach_b: tuple[float, float, float],
  pad_height: float,
  half_height: float,
  pad_offset: float,
  face_cone: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """The side-grasp TCP target and approach direction for each of the block's four side faces, and which faces
  count. (B, 4, 3), (B, 4, 3), (B, 4)

  A face counts when its outward normal is within ``face_cone`` (rad) of -``approach_b`` (the base-frame
  direction the hand should point in, e.g. +x: the hand comes from the robot's side of the block): at a block
  yaw near 45 deg two faces do, and step 0 found only the other one reachable over part of the area. The pads go
  to the block's axis at ``pad_height`` above its bottom (``half_height`` below its centre), the TCP
  ``pad_offset`` behind the axis, outside the face (L-H passes the jaw's pad reach plus how far short of the axis
  the pads should close, `pick_place.hold.GRASP_DEPTH`)."""
  obj: Entity = env.scene[object_name]
  rot = matrix_from_quat(obj.data.root_link_quat_w)
  ax, ay, up = rot[..., :, 0], rot[..., :, 1], rot[..., :, 2]
  want = -_base_dir_w(env, approach_b)  # the outward normal we want
  normals = torch.stack([ax, -ax, ay, -ay], dim=1)  # (B, 4, 3)
  normals = normals - (normals * up.unsqueeze(1)).sum(-1, keepdim=True) * up.unsqueeze(
    1
  )
  normals = normals / normals.norm(dim=-1, keepdim=True).clamp_min(1e-6)
  ok = torch.einsum("bkd,bd->bk", normals, want) >= math.cos(face_cone)
  pads = obj.data.root_link_pos_w + (pad_height - half_height) * up
  return pads.unsqueeze(1) + pad_offset * normals, -normals, ok


def side_reach(
  env: ManagerBasedRlEnv,
  object_name: str,
  asset_cfg: SceneEntityCfg,
  approach_b: tuple[float, float, float],
  pad_height: float,
  half_height: float,
  pad_offset: float,
  std: float,
  cone_std: float,
  face_cone: float,
) -> torch.Tensor:
  """RCTP's reach on a side-grasp point instead of the object's centre, half of it paid for pointing the hand at
  the face: max over the faces that count (`side_grasp_targets_w`) of
  G(|TCP - target|, std) x (0.5 + 0.5 G(angle(approach, wanted), cone_std)). (B,)"""
  robot: Entity = env.scene[asset_cfg.name]
  tcp = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  target, want, ok = side_grasp_targets_w(
    env, object_name, approach_b, pad_height, half_height, pad_offset, face_cone
  )
  approach, _ = hand_axes_w(env, asset_cfg)
  angle = torch.acos(torch.einsum("bkd,bd->bk", want, approach).clamp(-1.0, 1.0))
  near = torch.exp(-torch.sum(torch.square(tcp.unsqueeze(1) - target), dim=-1) / std**2)
  value = near * (0.5 + 0.5 * torch.exp(-torch.square(angle / cone_std)))
  return torch.where(ok, value, torch.zeros_like(value)).amax(dim=1)


def side_graded(
  env: ManagerBasedRlEnv,
  inner,
  side_cfg: SceneEntityCfg,
  side_std: float,
  **params,
) -> torch.Tensor:
  """``inner(env, **params)`` x `side_grade`: a term that pays only as much as the grasp is a side one.
  Wraps RCTP's grasp, faded lift and M2's goal terms so a top-down grasp earns none of them."""
  return inner(env, **params) * side_grade(env, side_cfg, side_std)


def held_at_goal(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  grasp_sensor_name: str,
  side_cfg: SceneEntityCfg,
  approach_b: tuple[float, float, float],
  distance: float,
  tilt: float,
  speed: float,
  elevation: float,
  cone: float,
) -> torch.Tensor:
  """1 while the block is held still and level at the goal in a side grasp: grasped, within
  ``distance`` of the goal, tilted less than ``tilt`` (rad), slower than ``speed`` (m/s), the hand's
  approach within ``elevation`` of the horizontal and within ``cone`` of ``approach_b`` (rad). (B,)"""
  obj: Entity = env.scene[object_name]
  command = env.command_manager.get_term(command_name)
  near = torch.norm(command.command - obj.data.root_link_pos_w, dim=-1) < distance
  level = axis_tilt(env, object_name) < tilt
  still = torch.norm(obj.data.root_link_lin_vel_w, dim=-1) < speed
  approach, _ = hand_axes_w(env, side_cfg)
  side = approach_elevation(env, side_cfg) < elevation
  aimed = (approach * _base_dir_w(env, approach_b)).sum(-1) > math.cos(cone)
  held = _grasped_bool(env, grasp_sensor_name)
  return (held & near & level & still & side & aimed).float()


def level_near_goal(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  grasp_sensor_name: str,
  near: float,
  std: float,
) -> torch.Tensor:
  """G(block tilt, ``std``) while it is held within ``near`` of the goal. (B,)"""
  obj: Entity = env.scene[object_name]
  command = env.command_manager.get_term(command_name)
  close = torch.norm(command.command - obj.data.root_link_pos_w, dim=-1) < near
  gate = (_grasped_bool(env, grasp_sensor_name) & close).float()
  return torch.exp(-torch.square(axis_tilt(env, object_name) / std)) * gate


# -- Observations ---------------------------------------------------------------------------------


def object_up_b(
  env: ManagerBasedRlEnv,
  object_name: str,
  asset_cfg: SceneEntityCfg = _ROBOT,
  lag: tuple[int, int] | None = None,
) -> torch.Tensor:
  """x and y of the object's +z axis in the base frame: (0, 0) when it stands level. (B, 2)"""
  robot: Entity = env.scene[asset_cfg.name]
  _, quat_w = _object_pose_w(env, object_name, lag)
  up_w = matrix_from_quat(quat_w)[..., :, 2]
  up_b = quat_apply(quat_inv(_base_pose(robot)[1]), up_w)
  return up_b[:, :2]


def object_point_biased_b(
  env: ManagerBasedRlEnv,
  object_name: str,
  local: tuple[float, float, float],
  bias_key: str,
  asset_cfg: SceneEntityCfg = _ROBOT,
  lag: tuple[int, int] | None = None,
) -> torch.Tensor:
  """`box.object_position_biased_b` of a point fixed in the object (``local``, its body frame) instead of its origin:
  plan §13's chain shows the left the socket block's centre (L-H's object, trained with the origin there) while the
  block's origin is its bottom face's (R-I's). (B, 3)"""
  from pal_mjlab.tasks.manipulation.mdp.box import _corrupted, tracker_bias
  from pal_mjlab.tasks.manipulation.mdp.observations import _to_base_frame

  robot: Entity = env.scene[asset_cfg.name]
  pos_w, quat_w = _object_pose_w(env, object_name, lag)
  off = torch.tensor(local, device=env.device, dtype=torch.float32).expand(
    pos_w.shape[0], 3
  )
  pos_b = _to_base_frame(robot, pos_w + quat_apply(quat_w, off))
  if _corrupted(env):
    pos_b = pos_b + tracker_bias(env, bias_key)
  return pos_b


def side_grade_obs(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, std: float
) -> torch.Tensor:
  """`side_grade`, for the critic. (B, 1)"""
  return side_grade(env, asset_cfg, std).unsqueeze(-1)


def held_at_goal_obs(env: ManagerBasedRlEnv, **params) -> torch.Tensor:
  """`held_at_goal`, for the critic. (B, 1)"""
  return held_at_goal(env, **params).unsqueeze(-1)


# -- L-HP: pushes on the held block's top ---------------------------------------------------------

#: Attribute on the env holding the push state (`push_object_top`).
_PUSH = "_s13_push"


class _PushState:
  def __init__(self, n: int, device) -> None:
    self.next_start = torch.zeros(n, device=device)
    self.end = torch.zeros(n, device=device)
    self.force = torch.zeros(n, 3, device=device)
    self.active = torch.zeros(n, dtype=torch.bool, device=device)


def push_state(env: ManagerBasedRlEnv) -> _PushState:
  state = getattr(env, _PUSH, None)
  if state is None:
    state = _PushState(env.num_envs, env.device)
    setattr(env, _PUSH, state)
  return state


def push_object_top(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  object_name: str,
  start_s: float,
  down_n: tuple[float, float],
  side_n: float,
  duration_s: tuple[float, float],
  gap_s: tuple[float, float],
  top: float,
) -> None:
  """Step event (L-HP): from ``start_s`` into the episode, pushes on the object's top face -- a force
  ``down_n`` (N, uniform) straight down plus up to ``side_n`` (N) sideways in a random direction, held
  for ``duration_s`` (s, uniform), the next one ``gap_s`` after it ends -- applied at the top face's
  centre, ``top`` above the object's origin (so a sideways push also turns it). Written into
  ``xfrc_applied`` every step while it lasts, zeroed after. A stand-in for the right hand's press."""
  del env_ids
  state = push_state(env)
  obj: Entity = env.scene[object_name]
  t = env.episode_length_buf.float() * env.step_dt
  fresh = env.episode_length_buf <= 1
  state.active &= ~fresh
  state.next_start = torch.where(fresh, torch.full_like(t, start_s), state.next_start)
  ending = state.active & (t >= state.end)
  state.active &= ~ending
  if ending.any():
    gap = torch.empty_like(t).uniform_(*gap_s)
    state.next_start = torch.where(ending, t + gap, state.next_start)
  starting = ~state.active & (t >= state.next_start)
  if starting.any():
    n = env.num_envs
    down = torch.empty(n, device=env.device).uniform_(*down_n)
    phi = torch.empty(n, device=env.device).uniform_(0.0, 2.0 * math.pi)
    side = torch.empty(n, device=env.device).uniform_(0.0, side_n)
    force = torch.stack([side * torch.cos(phi), side * torch.sin(phi), -down], dim=-1)
    state.force = torch.where(starting.unsqueeze(-1), force, state.force)
    dur = torch.empty_like(t).uniform_(*duration_s)
    state.end = torch.where(starting, t + dur, state.end)
    state.active |= starting
  force = state.force * state.active.unsqueeze(-1).float()
  up = matrix_from_quat(obj.data.root_link_quat_w)[..., :, 2]
  torque = torch.cross(top * up, force, dim=-1)
  obj.data.write_external_wrench(force.unsqueeze(1), torque.unsqueeze(1), body_ids=[0])


def push_active(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Whether a push is on, per env (for the eval and the critic). (B,)"""
  return push_state(env).active
