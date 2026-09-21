"""Observation terms mjlab does not provide.

mjlab's manipulation observations are written for a fixed-base arm, where the world frame
and the robot base frame coincide. A humanoid's base is somewhere else, so anything the
policy should see needs an explicit transform into the base frame -- that is what most of
this module is.

The term *names* used by the env cfg are part of the ONNX/deployment contract; see
``tabletop_env_cfg``. Changing what these functions compute is fine; changing the shapes
they return is not.
"""

from __future__ import annotations

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import euler_xyz_from_quat, quat_apply, quat_inv, quat_mul

_ROBOT = SceneEntityCfg("robot")


def _to_base_frame(robot: Entity, pos_w: torch.Tensor) -> torch.Tensor:
  return quat_apply(
    quat_inv(robot.data.root_link_quat_w), pos_w - robot.data.root_link_pos_w
  )


def ee_position_b(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Grasp-point position in the robot base frame. (B, 3)"""
  robot: Entity = env.scene[asset_cfg.name]
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  return _to_base_frame(robot, ee_pos_w)


def object_position_b(
  env: ManagerBasedRlEnv, object_name: str, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Object position in the robot base frame. (B, 3)"""
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  return _to_base_frame(robot, obj.data.root_link_pos_w)


def object_yaw_b(
  env: ManagerBasedRlEnv, object_name: str, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Object yaw relative to the robot base, as ``[cos, sin]``. (B, 2)

  Encoded as a unit vector rather than an angle so the observation stays continuous across
  the +/-pi wrap, which a raw yaw does not.
  """
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  quat_b = quat_mul(quat_inv(robot.data.root_link_quat_w), obj.data.root_link_quat_w)
  _, _, yaw = euler_xyz_from_quat(quat_b)
  return torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)


def goal_position_b(
  env: ManagerBasedRlEnv, command_name: str, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Commanded goal position in the robot base frame. (B, 3)"""
  robot: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_term(command_name)
  return _to_base_frame(robot, command.command)


def is_grasped(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """1.0 when *both* pads report contact with the object. (B, 1)

  This reads the contact sensor, not pad-to-object proximity. Proximity reports a grasp for
  an object merely sitting between open jaws, which makes every downstream "is it held"
  gate meaningless.
  """
  return _grasped_bool(env, sensor_name).float().unsqueeze(-1)


def _grasped_bool(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None, f"sensor {sensor_name!r} does not provide 'found'"
  # found is (B, P) over the pad geoms: a grasp needs every pad in contact, not any.
  return found.bool().all(dim=-1)


def is_on_surface(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """1.0 while the object still rests on the support surface. (B, 1)"""
  return _on_surface_bool(env, sensor_name).float().unsqueeze(-1)


def _on_surface_bool(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None, f"sensor {sensor_name!r} does not provide 'found'"
  return found.bool().any(dim=-1)


def _placed_bool(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  grasp_sensor_name: str,
  threshold: float,
  velocity_threshold: float,
) -> torch.Tensor:
  """At the goal, released, and at rest. Shared by the success reward and termination."""
  obj: Entity = env.scene[object_name]
  command = env.command_manager.get_term(command_name)
  at_goal = torch.norm(command.command - obj.data.root_link_pos_w, dim=-1) < threshold
  released = ~_grasped_bool(env, grasp_sensor_name)
  settled = torch.norm(obj.data.root_link_lin_vel_w, dim=-1) < velocity_threshold
  return at_goal & released & settled
