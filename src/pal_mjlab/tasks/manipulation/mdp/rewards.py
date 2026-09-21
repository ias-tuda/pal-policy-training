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


def released_at_goal(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  grasp_sensor_name: str,
  threshold: float = 0.05,
) -> torch.Tensor:
  """1.0 when the object is at the goal and no longer held.

  The shaping counterpart to :func:`~.terminations.object_placed`: it pays as soon as the
  jaw opens in the right place, without waiting for the object to settle, which is what
  stops the policy from converging on holding the object at the goal forever.
  """
  obj: Entity = env.scene[object_name]
  command = env.command_manager.get_term(command_name)
  at_goal = torch.norm(command.command - obj.data.root_link_pos_w, dim=-1) < threshold
  return (at_goal & ~_grasped_bool(env, grasp_sensor_name)).float()


def object_placed(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  grasp_sensor_name: str,
  threshold: float = 0.05,
  velocity_threshold: float = 0.05,
) -> torch.Tensor:
  """1.0 on full success: at the goal, released, and come to rest."""
  return _placed_bool(
    env,
    command_name,
    object_name,
    grasp_sensor_name,
    threshold,
    velocity_threshold,
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
