"""R-C-strict pays nothing for a cube that reached the goal without being picked up.

C0 showed every arm succeeding with `Episode_Reward/grasp` at exactly 0: the goal sits at
the cube's resting height and `_placed_bool` counts `released = ¬grasped`, so a cube that
was never grasped is "placed" and sliding it 20 cm solves the task. R-C gates only its
`goal` term on being lifted, which leaves `reach`'s `(1 + G(d_g))` factor, `release` and
`success` all payable by a slide.

The test the plan asks for: put the cube on the goal, never lift it, and assert that
`release`, `success` and the success termination are all zero -- then set the latch and
assert they fire, so that a term returning zero because it is *broken* cannot pass.
"""

import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from pal_mjlab.tasks.manipulation.mdp.observations import _WAS_LIFTED, _placed_bool
from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import (
  OBJECT,
  make_pick_place_env_cfg,
)
from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS_LEFT_ARM
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import SUPPORT_SENSOR

NUM_ENVS = 2
#: Terms a slid cube must not earn under R-C-strict.
GATED = ("release", "success")


def _env(reward: str) -> ManagerBasedRlEnv:
  cfg = make_pick_place_env_cfg(
    KANGAROO_GRIPPERS_LEFT_ARM, gripper_name="left", reward=reward
  )
  cfg.scene.num_envs = NUM_ENVS
  return ManagerBasedRlEnv(cfg=cfg, device=get_test_device())


def _idle(env: ManagerBasedRlEnv, steps: int = 1) -> None:
  action = torch.zeros(env.action_space.shape, device=env.device)
  for _ in range(steps):
    env.step(action)


def _slide_cube_to_the_goal(env: ManagerBasedRlEnv) -> None:
  """Put the cube on the goal, at rest, having never left the table.

  This is a slide in every way the reward can observe: the object is at the target, it is
  not grasped, it is on the table, and it was never off it.

  The steps are not padding. Contact sensors are empty until the first step after a reset,
  and a teleport does not refresh them either -- reading `object_table_contact` without
  stepping reports "not on the surface" for a cube visibly resting on it, which would make
  `object_lifted` collapse to `grasped` and set the latch for the wrong reason. Stepping
  after the teleport also means the gate is exercised through the real step path.
  """
  _idle(env, steps=3)
  obj = env.scene[OBJECT]
  goal = env.command_manager.get_term("goal_pose").command
  pose = obj.data.root_link_pose_w.clone()
  pose[:, :3] = goal
  obj.write_root_link_pose_to_sim(pose)
  obj.write_root_link_velocity_to_sim(torch.zeros_like(obj.data.root_link_vel_w))
  _idle(env)


def _terms(env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
  """Every reward term's raw value at the current state, unweighted."""
  return {
    name: cfg.func(env, **cfg.params)
    for name, cfg in zip(
      env.reward_manager.active_terms, env.reward_manager._term_cfgs, strict=True
    )
  }


def test_a_slid_cube_earns_nothing_from_the_gated_terms():
  env = _env("R-C-strict")
  env.reset()
  _slide_cube_to_the_goal(env)
  values = _terms(env)

  assert not bool(getattr(env, _WAS_LIFTED).any()), "the cube was never lifted"
  for name in GATED:
    assert torch.all(values[name] == 0.0), f"{name} paid for a slide: {values[name]}"
  env.close()


def test_a_slid_cube_does_not_terminate():
  """The termination matters as much as the reward: ending early is worth time to a policy."""
  env = _env("R-C-strict")
  env.reset()
  _slide_cube_to_the_goal(env)
  term = env.termination_manager._term_cfgs[
    env.termination_manager.active_terms.index("success")
  ]
  assert not bool(term.func(env, **term.params).any())
  env.close()


def test_the_same_cube_does_pay_once_the_latch_is_set():
  """The positive control. A term that returns zero because it is broken also 'passes'."""
  env = _env("R-C-strict")
  env.reset()
  _slide_cube_to_the_goal(env)
  # Exactly what a real lift earlier in the episode would have left behind.
  getattr(env, _WAS_LIFTED)[:] = True
  values = _terms(env)
  for name in GATED:
    assert torch.all(values[name] > 0.0), f"{name} is gated on more than was_lifted"
  env.close()


def test_reach_drops_the_goal_factor():
  """`staged_position_reward` is `G(d_r)*(1 + G(d_g))`; strict keeps only `G(d_r)`.

  Both terms are evaluated against the *same* environment rather than one each, so the
  comparison cannot be confounded by two sims drifting apart -- and so that R-C's ungated
  success termination cannot reset its own env out from under the measurement.
  """
  env = _env("R-C-strict")
  env.reset()
  _slide_cube_to_the_goal(env)

  loose_cfg = make_pick_place_env_cfg(
    KANGAROO_GRIPPERS_LEFT_ARM, gripper_name="left", reward="R-C"
  ).rewards["reach"]
  # A term cfg taken from an unbuilt cfg still names its entities by regex; the managers
  # resolve those to indices at build time, so do the same here before calling it.
  loose_cfg.params["asset_cfg"].resolve(env.scene)
  loose_reach = loose_cfg.func(env, **loose_cfg.params)
  strict_reach = _terms(env)["reach"]

  assert torch.all(strict_reach < loose_reach), "the goal factor is still being paid"
  # What remains is a pure reaching kernel, so it is bounded by 1.
  assert torch.all(strict_reach <= 1.0)
  env.close()


def test_the_latch_clears_between_episodes():
  """A latch that never falls makes every episode after the first count a slide as a pick."""
  env = _env("R-C-strict")
  env.reset()
  _slide_cube_to_the_goal(
    env
  )  # also creates the latch, which is allocated on first read
  getattr(env, _WAS_LIFTED)[:] = True
  env.reset()
  assert not bool(getattr(env, _WAS_LIFTED).any())
  env.close()


@pytest.mark.parametrize("reward", ["R-A", "R-C"])
def test_the_other_arms_are_untouched(reward):
  """The gate is opt-in: C1's arm has to keep behaving exactly as C1 measured it."""
  cfg = make_pick_place_env_cfg(
    KANGAROO_GRIPPERS_LEFT_ARM, gripper_name="left", reward=reward
  )
  assert "support_sensor_name" not in cfg.terminations["success"].params
  assert "support_sensor_name" not in cfg.rewards["release"].params
  # The latch reset is present on every arm (the eval needs it) but does nothing here,
  # because on an ungated arm nothing reads the latch during training.
  assert "reset_was_lifted" in cfg.events


def test_the_latch_resets_on_an_ungated_arm_too():
  """The eval reads `placed ∧ was_lifted` on every arm, so every arm must reset the latch.

  Regression: the reset event was originally added only for `REQUIRE_LIFTED` arms, because
  only they read the latch during training. Evaluating R-A would then latch True on the
  first lift of the run and report every later episode as a pick.
  """
  env = _env("R-A")
  env.reset()
  _slide_cube_to_the_goal(env)
  # Read it the way the eval does, through `_placed_bool`, which allocates it.
  _placed_bool(
    env, "goal_pose", OBJECT, "left_pad_contact", 0.025, 0.05, SUPPORT_SENSOR
  )
  getattr(env, _WAS_LIFTED)[:] = True
  env.reset()
  assert not bool(getattr(env, _WAS_LIFTED).any())
  env.close()
