"""Termination terms mjlab does not provide."""

from __future__ import annotations

from collections.abc import Sequence

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
  support_sensor_name: str | None = None,
) -> torch.Tensor:
  """Success: the object is at the goal, has been let go, and has come to rest.

  Deliberately stateless. A dwell timer would need somewhere to live, and the only place a
  manager-based env offers is a command term -- which is how the previous framework ended
  up with a command that owned the goal, the grasp test and the success latch all at once.
  "Released and settled" is also the truer condition for a stack: a cube held motionless at
  the goal is not stacked, and a cube dropped from above is at the goal only briefly.

  With `support_sensor_name`, additionally requires that the object was lifted at some
  point this episode -- see `_placed_bool`. Gating only the success *reward* is not enough:
  ending the episode early is itself worth the remaining time to a policy.
  """
  return _placed_bool(
    env,
    command_name,
    object_name,
    grasp_sensor_name,
    threshold,
    velocity_threshold,
    support_sensor_name,
  )


def entity_below_height(
  env: ManagerBasedRlEnv, entity_name: str, min_height: float
) -> torch.Tensor:
  """Failure: an entity has dropped below ``min_height`` -- knocked off the table."""
  entity: Entity = env.scene[entity_name]
  return entity.data.root_link_pos_w[:, 2] < min_height


def non_finite_state(env: ManagerBasedRlEnv) -> torch.Tensor:
  """True for any world whose physics state has gone non-finite.

  Contact-rich GPU simulation diverges occasionally, and `mujoco_warp` has no equivalent of
  MuJoCo's automatic reset on a bad `qacc`: a world that blows up stays blown up and feeds
  NaN to the learner, which raises and takes the whole run with it. Measured on this task
  at four scales, the rate is roughly **one per 0.5M world-steps** and does not depend on
  `num_envs` -- 246k, 737k and 1.1M world-steps to the first failure at 1024, 2048 and 256
  environments. Over a 20k-iteration run that is a thousand-odd events, so surviving them is
  not optional.

  Terminating recycles the world: mjlab resets terminated environments *before* it computes
  observations (`manager_based_rl_env.py`, step order), so the learner sees the fresh state
  rather than the NaN. Rewards are computed *before* that reset and are handled separately --
  see `_finite` in the reward builder.

  Counted as a time-out rather than a failure. The policy did not do anything wrong, and
  teaching it that this state is terminal-bad would be teaching it about the solver.
  """
  return ~torch.isfinite(env.sim.data.qpos).all(dim=-1) | ~torch.isfinite(
    env.sim.data.qvel
  ).all(dim=-1)


def entity_out_of_reach(
  env: ManagerBasedRlEnv,
  entity_names: Sequence[str],
  max_distance: float,
  reference_body: str,
  robot_name: str = "robot",
) -> torch.Tensor:
  """Failure: some entity is further from the robot, horizontally, than it can reach. (B,)

  A failure rather than a ``time_out``: pushing the cube out of reach is something the
  policy did, and the critic must not bootstrap past it. Ported from Claudius's 82c4450
  (S5.1 item 8).
  """
  robot: Entity = env.scene[robot_name]
  ref_id = robot.body_names.index(reference_body)
  base_xy = robot.data.body_link_pos_w[:, ref_id, :2]
  out: torch.Tensor | None = None
  for name in entity_names:
    entity: Entity = env.scene[name]
    distance = torch.norm(entity.data.root_link_pos_w[:, :2] - base_xy, dim=-1)
    term = distance > max_distance
    out = term if out is None else (out | term)
  assert out is not None, "entity_names must name at least one entity"
  return out
