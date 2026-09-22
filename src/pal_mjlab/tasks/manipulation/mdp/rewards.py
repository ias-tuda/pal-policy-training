"""Reward terms mjlab does not provide.

mjlab already ships the shaping that gets an arm to an object and the object to a goal
(``staged_position_reward``, ``bring_object_reward``). What it has no notion of is the
*grasp* and the *release* -- its lift task succeeds by proximity alone. A stacking task is
defined by letting go at the right moment, so those terms live here.

Every term is a pure function of the scene plus the commanded goal. None reaches into
another manager's internals.
"""

from __future__ import annotations

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from pal_mjlab.tasks.manipulation.mdp.observations import (
  _grasped_bool,
  _on_surface_bool,
  _placed_bool,
  _was_lifted_bool,
)

_ROBOT = SceneEntityCfg("robot")


def grasped(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """1.0 while both pads hold the object."""
  return _grasped_bool(env, sensor_name).float()


def object_lifted(
  env: ManagerBasedRlEnv,
  grasp_sensor_name: str,
  support_sensor_name: str,
) -> torch.Tensor:
  """1.0 once the object is held *and* has left the support surface.

  Both halves matter: without the grasp term the reward pays for knocking the object off
  the table, and without the support term it pays for merely touching it.
  """
  return (
    _grasped_bool(env, grasp_sensor_name) & ~_on_surface_bool(env, support_sensor_name)
  ).float()


def reach_only(
  env: ManagerBasedRlEnv,
  object_name: str,
  reaching_std: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """The reaching half of mjlab's `staged_position_reward`, without the goal factor.

  `staged_position_reward` returns `G(d_r) * (1 + G(d_g))`, and that second factor pays for
  the object being near the goal however it got there -- so on a table-height goal it is one
  of the terms a sliding policy collects. This is the same Gaussian on tool-to-object
  distance with the factor dropped.
  """
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  reach_error = torch.sum(torch.square(ee_pos_w - obj.data.root_link_pos_w), dim=-1)
  return torch.exp(-reach_error / reaching_std**2)


def released_at_goal(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  grasp_sensor_name: str,
  threshold: float = 0.05,
  support_sensor_name: str | None = None,
) -> torch.Tensor:
  """1.0 when the object is at the goal and no longer held.

  The shaping counterpart to :func:`~.terminations.object_placed`: it pays as soon as the
  jaw opens in the right place, without waiting for the object to settle, which is what
  stops the policy from converging on holding the object at the goal forever.

  `¬grasped` is also true of an object that was never grasped, so on a table-height goal
  this pays a slide. Passing `support_sensor_name` additionally requires the `was_lifted`
  latch; leaving it None is the pre-C2 behaviour.
  """
  obj: Entity = env.scene[object_name]
  command = env.command_manager.get_term(command_name)
  at_goal = torch.norm(command.command - obj.data.root_link_pos_w, dim=-1) < threshold
  released = at_goal & ~_grasped_bool(env, grasp_sensor_name)
  if support_sensor_name is not None:
    released &= _was_lifted_bool(env, grasp_sensor_name, support_sensor_name)
  return released.float()


def object_placed(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  grasp_sensor_name: str,
  threshold: float = 0.05,
  velocity_threshold: float = 0.05,
  support_sensor_name: str | None = None,
) -> torch.Tensor:
  """1.0 on full success: at the goal, released, and come to rest.

  With `support_sensor_name`, additionally requires that the object was lifted at some
  point in the episode -- see `_placed_bool`.
  """
  return _placed_bool(
    env,
    command_name,
    object_name,
    grasp_sensor_name,
    threshold,
    velocity_threshold,
    support_sensor_name,
  ).float()


def entity_lin_vel_l2(env: ManagerBasedRlEnv, entity_name: str) -> torch.Tensor:
  """Squared planar speed of an entity, for penalising scene disturbance."""
  entity: Entity = env.scene[entity_name]
  return torch.sum(torch.square(entity.data.root_link_lin_vel_w[:, :2]), dim=-1)


def joint_pos_limits(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Amount by which the commanded joints exceed their soft position limits."""
  robot: Entity = env.scene[asset_cfg.name]
  joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
  limits = robot.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
  below = (limits[..., 0] - joint_pos).clamp_min(0.0)
  above = (joint_pos - limits[..., 1]).clamp_min(0.0)
  return torch.sum(below + above, dim=-1)


def lift_height(
  env: ManagerBasedRlEnv,
  grasp_sensor_name: str,
  object_name: str,
  surface_cfg: SceneEntityCfg,
  max_height: float = 0.10,
) -> torch.Tensor:
  """How far the object has been lifted, as a fraction of ``max_height``, while held.

  The bounded, gated counterpart to :func:`object_lifted`, which is binary. Bounded because
  an unbounded height reward pays more for throwing the object than for placing it; gated on
  the grasp because otherwise it pays for the object being anywhere high, including mid-fall
  after being knocked. The surface height is read from physics, so the term is still right
  when the table is randomised.
  """
  from pal_mjlab.tasks.manipulation.mdp.events import surface_top_z

  obj: Entity = env.scene[object_name]
  above = obj.data.root_link_pos_w[:, 2] - surface_top_z(env, surface_cfg)
  height = above.clamp(0.0, max_height)
  return (height / max_height) * _grasped_bool(env, grasp_sensor_name).float()


def bring_object_when_lifted(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  grasp_sensor_name: str,
  support_sensor_name: str,
  std: float = 0.15,
) -> torch.Tensor:
  """Distance-to-goal shaping that only pays once the object is actually off the table.

  mjlab's ``bring_object_reward`` pays for the object being near the goal however it got
  there, which on a tabletop task includes sliding it. Gating on "held and not supported"
  makes the ordering of the stages monotone: the policy cannot collect the transport reward
  without having completed the lift.
  """
  obj: Entity = env.scene[object_name]
  command = env.command_manager.get_term(command_name)
  distance = torch.norm(command.command - obj.data.root_link_pos_w, dim=-1)
  lifted = object_lifted(env, grasp_sensor_name, support_sensor_name)
  return torch.exp(-torch.square(distance / std)) * lifted


#: Corners of a unit cube, in the object's own frame. Scaled by the object's half-size to
#: give the eight keypoints of its oriented bounding box.
_UNIT_CORNERS = torch.tensor(
  [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)]
)


def _object_corners(env: ManagerBasedRlEnv, object_name: str) -> torch.Tensor:
  """The eight corners of the object's bounding box, in its own frame. (B, 8, 3)

  Read from ``geom_size`` rather than from a constant, so a task that randomises the cube's
  size gets the right keypoints for free.
  """
  obj: Entity = env.scene[object_name]
  half = env.sim.model.geom_size[:, obj.indexing.geom_ids[0], :]  # (B, 3)
  return _UNIT_CORNERS.to(half.device) * half.unsqueeze(1)


def keypoint_distance_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  kernel_a: float = 30.0,
  kernel_b: float = 2.0,
) -> torch.Tensor:
  """Allshire et al.'s keypoint reward: eight bounding-box corners, logistic kernel.

  ``K(x) = (exp(a x) + b + exp(-a x))^-1``, averaged over the eight corners of the object's
  oriented bounding box against the same corners at the goal, and normalised so a perfect
  match scores 1. From *Transferring Dexterous Manipulation from GPU Simulation to a Remote
  Real-World TriFinger* (2022), which uses ``a = 30``, ``b = 2``; that kernel's half-width is
  about 4 cm, the right scale against this task's 2.5 cm success threshold.

  Why corners rather than the centre: a keypoint set measures position and orientation as one
  quantity, in metres, with no separate rotation term to weight against the position one. v0
  does not score the placed cube's orientation -- the goal is a point, so the target corners
  are an upright cube at it -- which makes this a reward-*shape* comparison today. It is the
  term that makes scoring orientation free when v1 wants it.

  The plan sketched this variant with a Gaussian kernel. The published kernel is used
  instead: the variant exists to test the published idea against the framework's distance
  reward, and changing the kernel would test something else.
  """
  obj: Entity = env.scene[object_name]
  goal = env.command_manager.get_term(command_name).command  # (B, 3)
  corners = _object_corners(env, object_name)  # (B, 8, 3)

  from mjlab.utils.lab_api.math import quat_apply

  quat = obj.data.root_link_quat_w.unsqueeze(1).expand(-1, corners.shape[1], -1)
  current = obj.data.root_link_pos_w.unsqueeze(1) + quat_apply(quat, corners)
  target = goal.unsqueeze(1) + corners

  distance = torch.norm(current - target, dim=-1)  # (B, 8)
  kernel = 1.0 / (
    torch.exp(kernel_a * distance) + kernel_b + torch.exp(-kernel_a * distance)
  )
  return kernel.mean(dim=-1) * (2.0 + kernel_b)


def time_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
  """1.0 every step, for a constant per-step cost. Weight it negative."""
  return torch.ones(env.num_envs, device=env.device)


def contact_penalty(
  env: ManagerBasedRlEnv, sensor_name: str, force_threshold: float = 10.0
) -> torch.Tensor:
  """1.0 while a contact sensor reports more than ``force_threshold``. Weight it negative.

  The same predicate mjlab terminates on, used as a cost instead: leaning on the table is a
  cheap way to stabilise a grasp in simulation and not one that survives contact with a real
  table.
  """
  from mjlab.tasks.manipulation.mdp import illegal_contact

  return illegal_contact(env, sensor_name, force_threshold).float()
