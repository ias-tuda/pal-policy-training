"""A world whose physics goes non-finite is recycled instead of killing the run.

Contact-rich GPU simulation diverges, and `mujoco_warp` has no equivalent of MuJoCo's
automatic reset on a bad `qacc`, so a blown-up world stays blown up and feeds NaN to the
learner. Measured on the pick-and-place task at four scales on the IAS cluster, the rate is
roughly one per 0.5M world-steps and does not depend on `num_envs`: jobs 157178 and 157179
died at iteration 10 (1024 envs, 246k world-steps) and 15 (2048 envs, 737k), and 157123 at
180 (256 envs, 1.1M). A 20k-iteration run is a thousand-odd events, so what matters is not
whether divergence happens but whether the run survives it.

The divergence itself is rare and GPU-only, so these tests do not wait for one -- they
inject it, which is also the only way to assert the recovery deterministically. The guard is one term plus one config field, and most of the work is already mjlab's:
`RewardManager.compute` sanitises every reward term unconditionally
(`reward_manager.py:129`), and `ObservationGroupCfg.nan_policy` sanitises observations once
it is switched on from its `"disabled"` default. What neither does is *recycle* the world,
which is the part that matters -- a sanitised diverged world returns zeros forever.
"""

import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import (
  make_pick_place_env_cfg,
)
from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS_LEFT_ARM

NUM_ENVS = 4
POISONED = 1


def _env(guard: bool = True, num_envs: int = NUM_ENVS) -> ManagerBasedRlEnv:
  cfg = make_pick_place_env_cfg(KANGAROO_GRIPPERS_LEFT_ARM, gripper_name="left")
  cfg.scene.num_envs = num_envs
  if not guard:
    # Both halves, or this is not the pre-guard state: the sanitiser alone hides the NaN
    # from the observations while leaving the world diverged forever.
    del cfg.terminations["non_finite_state"]
    for group in cfg.observations.values():
      group.nan_policy = "disabled"
  return ManagerBasedRlEnv(cfg=cfg, device=get_test_device())


def _poison(env: ManagerBasedRlEnv) -> None:
  """Make one world diverge, the way a solver failure does: velocities go non-finite."""
  env.sim.data.qvel[POISONED, :] = float("nan")


def _step(env: ManagerBasedRlEnv):
  torch.manual_seed(0)
  return env.step(torch.zeros(env.action_space.shape, device=env.device))


def test_the_guard_is_on_by_default():
  cfg = make_pick_place_env_cfg(KANGAROO_GRIPPERS_LEFT_ARM, gripper_name="left")
  assert "non_finite_state" in cfg.terminations
  # A time-out, not a failure: the policy did not cause it, so it must not be trained on it.
  assert cfg.terminations["non_finite_state"].time_out


def test_a_diverged_world_is_terminated_and_recycled():
  env = _env()
  env.reset()
  _step(env)
  _poison(env)
  obs, _, _, truncated, _ = _step(env)

  assert bool(truncated[POISONED]), "the diverged world should have been truncated"
  assert not bool(truncated.sum() > 1), "only the poisoned world should have ended"
  assert torch.isfinite(obs["actor"]).all(), (
    "observations must be finite after the reset"
  )
  assert torch.isfinite(env.sim.data.qvel[POISONED]).all(), (
    "the reset must clear the NaN"
  )

  # And the world keeps running afterwards rather than being permanently poisoned.
  obs, _, _, _, _ = _step(env)
  assert torch.isfinite(obs["actor"]).all()
  env.close()


def test_the_rewards_of_a_diverged_step_are_finite():
  """The half the termination cannot cover.

  mjlab's step order is terminations -> rewards -> events -> `_reset_idx` -> observations,
  so the reward for the step that diverged is computed from the diverged state. rsl_rl's
  `check_nan` checks rewards too, so an unwrapped term kills the run one step after the
  termination saved the observations.
  """
  env = _env()
  env.reset()
  _step(env)
  _poison(env)
  _, rewards, _, _, _ = _step(env)

  assert torch.isfinite(rewards).all(), f"non-finite rewards {rewards}"
  bad = [
    name
    for name in env.reward_manager.active_terms
    if not torch.isfinite(env.reward_manager._episode_sums[name]).all()
  ]
  assert not bad, f"non-finite reward terms {bad}"
  env.close()


def test_without_the_guard_the_nan_is_exactly_what_kills_a_run():
  """The positive control: the failure mode is real, and the guard is what removes it.

  Without this, the tests above would pass just as well on an environment that never
  diverges in the first place -- which is how the run that produced job 157178's traceback
  would have looked to a test suite.
  """
  env = _env(guard=False)
  env.reset()
  _step(env)
  _poison(env)
  obs, _, _, _, _ = _step(env)
  assert not torch.isfinite(obs["actor"]).all(), (
    "poisoning did not reach the observations, so the guard tests prove nothing"
  )
  env.close()


@pytest.mark.parametrize("variant", ["R-A", "R-C", "R-D"])
def test_observations_are_sanitised_on_every_variant(variant):
  """mjlab's own backstop, which defaults to off and is useless left there.

  It covers the residual case the termination cannot: a state finite in qpos/qvel that
  still produces a non-finite observation, e.g. a velocity or sensor reading that
  overflows on the way out.
  """
  cfg = make_pick_place_env_cfg(
    KANGAROO_GRIPPERS_LEFT_ARM, gripper_name="left", reward=variant
  )
  for name, group in cfg.observations.items():
    assert group.nan_policy == "sanitize", f"{variant}/{name} left at mjlab's default"
    # Per-term checking costs a host sync per term per step and answers a question
    # `non_finite_state` has already answered.
    assert not group.nan_check_per_term
