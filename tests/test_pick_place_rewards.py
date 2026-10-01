"""Every reward variant builds, steps, and produces finite numbers.

The arms of the S5 comparison are only comparable if they differ where they are meant to and
nowhere else, so this checks both: that each variant carries the common regularisers and the
terms its design calls for, and that stepping the environment under it evaluates every term
without a NaN. A reward term that throws or returns NaN on the first step is a wasted queue
slot, and the cheapest place to find out is here.
"""

import numpy as np
import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import (
  BASELINES,
  make_pick_place_env_cfg,
)
from pal_mjlab.tasks.manipulation.pick_place.rewards import (
  DEFAULT_EPISODE_LENGTH_S,
  EPISODE_LENGTH_S,
  REWARD_VARIANTS,
  STACK_ONLY_ARMS,
  build_rewards,
)
from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS_LEFT_ARM

#: Carried by every arm, so that no comparison between arms is confounded by them.
REGULARISERS = (
  "action_rate_l2",
  "joint_pos_limits",
  "joint_vel_hinge",
  "table_contact",
)


def _cfg(variant: str, num_envs: int = 2):
  # Plan S9's RG / SA read the base cube: they are built on the stack goal, which needs B's box.
  stack = (
    {"goal": "stack", "baseline": BASELINES["B"]} if variant in STACK_ONLY_ARMS else {}
  )
  cfg = make_pick_place_env_cfg(
    KANGAROO_GRIPPERS_LEFT_ARM, gripper_name="left", reward=variant, **stack
  )
  cfg.scene.num_envs = num_envs
  return cfg


@pytest.mark.parametrize("variant", sorted(REWARD_VARIANTS))
def test_every_variant_carries_the_regularisers(variant):
  rewards = _cfg(variant).rewards
  for name in REGULARISERS:
    assert name in rewards, f"{variant} is missing {name}"


@pytest.mark.parametrize("variant", sorted(REWARD_VARIANTS))
def test_every_variant_steps_with_finite_rewards(variant):
  torch.manual_seed(0)
  env = ManagerBasedRlEnv(cfg=_cfg(variant), device=get_test_device())
  env.reset()
  for _ in range(5):
    env.step(2 * torch.rand(env.action_space.shape, device=env.device) - 1)
  sums = {
    name: float(env.reward_manager._episode_sums[name].mean())
    for name in env.reward_manager.active_terms
  }
  env.close()
  bad = [name for name, value in sums.items() if not np.isfinite(value)]
  assert not bad, f"{variant}: non-finite reward terms {bad}"
  # The approach term has to be alive from the first step, or nothing else can ever fire.
  assert sums["reach"] > 0.0


def test_r_b_is_the_sparse_arm():
  """R-B exists to ask whether the contact terms are load-bearing; it must not have them."""
  rewards = _cfg("R-B").rewards
  for name in ("grasp", "lift", "release"):
    assert name not in rewards
  assert "success" in rewards


def test_r_c_gates_the_lift_and_the_transport():
  rewards = _cfg("R-C").rewards
  assert rewards["lift"].func.__name__ == "lift_height"
  assert rewards["goal"].func.__name__ == "bring_object_when_lifted"
  assert rewards["lift"].params["max_height"] == 0.10


def test_r_d_replaces_only_the_goal_term():
  """The keypoint variant is a swap of one term, or it is not a controlled comparison."""
  baseline, keypoints = _cfg("R-A").rewards, _cfg("R-D").rewards
  assert set(baseline) == set(keypoints)
  differing = [k for k in baseline if baseline[k].func is not keypoints[k].func]
  assert differing == ["goal"]
  assert keypoints["goal"].func.__name__ == "keypoint_distance_reward"


def test_r_e_pays_for_time_and_runs_shorter():
  rewards = _cfg("R-E").rewards
  assert rewards["alive"].weight < 0.0
  assert rewards["success"].weight > _cfg("R-A").rewards["success"].weight
  assert EPISODE_LENGTH_S["R-E"] < DEFAULT_EPISODE_LENGTH_S
  assert _cfg("R-E").episode_length_s == EPISODE_LENGTH_S["R-E"]


def test_the_curriculum_ramps_the_velocity_penalty():
  curriculum = _cfg("R-A").curriculum
  stages = curriculum["joint_vel_hinge_weight"].params["stages"]
  weights = [stage["weight"] for stage in stages]
  assert weights == sorted(weights, reverse=True), "the penalty has to get stronger"
  assert (
    curriculum["joint_vel_hinge_weight"].params["reward_name"] in _cfg("R-A").rewards
  )


def test_an_unknown_variant_is_an_error():
  with pytest.raises(ValueError, match="R-Z"):
    build_rewards("R-Z", None)  # type: ignore[arg-type]
