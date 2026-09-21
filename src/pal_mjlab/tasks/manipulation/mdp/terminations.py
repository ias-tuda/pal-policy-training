"""Termination terms mjlab does not provide."""

from __future__ import annotations

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv

from pal_mjlab.tasks.manipulation.mdp.observations import _placed_bool


def object_placed(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  grasp_sensor_name: str,
  threshold: float = 0.05,
  velocity_threshold: float = 0.05,
) -> torch.Tensor:
  """Success: the object is at the goal, has been let go, and has come to rest.

  Deliberately stateless. A dwell timer would need somewhere to live, and the only place a
  manager-based env offers is a command term -- which is how the previous framework ended
  up with a command that owned the goal, the grasp test and the success latch all at once.
  "Released and settled" is also the truer condition for a stack: a cube held motionless at
  the goal is not stacked, and a cube dropped from above is at the goal only briefly.
  """
  return _placed_bool(
    env, command_name, object_name, grasp_sensor_name, threshold, velocity_threshold
  )


def entity_below_height(
  env: ManagerBasedRlEnv, entity_name: str, min_height: float
) -> torch.Tensor:
  """Failure: an entity has dropped below ``min_height`` -- knocked off the table."""
  entity: Entity = env.scene[entity_name]
  return entity.data.root_link_pos_w[:, 2] < min_height
