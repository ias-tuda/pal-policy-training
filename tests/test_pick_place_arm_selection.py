"""D10: one task id, and the reward arm and DR rung come from the environment.

The alternative was a task id per (arm, rung) -- thirty ids, and every recorded result
depending on which one was spelled. The registry builds its cfg at import, so a tyro
`--env.*` flag cannot reach a constructor argument that decides which reward *terms exist*;
an environment variable read at registration can.

What these check is the property that makes the hook safe to use on a cluster: an unknown
name is a loud error, never a silent fall back to R-A. A job that trained the baseline under
a run named `R-C-strict` would be worse than a job that failed.
"""

import pytest
from pal_mjlab.tasks.manipulation.pick_place.kangaroo.env_cfg import (
  DEFAULT_DR,
  DEFAULT_PPO,
  DEFAULT_REWARD,
  DR_ENV_VAR,
  PPO_ENV_VAR,
  REWARD_ENV_VAR,
  kangaroo_pick_place_env_cfg,
  kangaroo_pick_place_rl_cfg,
)
from pal_mjlab.tasks.manipulation.pick_place.rewards import (
  NO_SUCCESS_TERMINATION,
  REQUIRE_LIFTED,
  REWARD_VARIANTS,
  STACK_ONLY_ARMS,
)
from pal_mjlab.tasks.manipulation.rl_cfg import PPO_RECIPES, manipulation_ppo_runner_cfg
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import DR_LEVELS


def test_the_default_is_c1s_configuration(monkeypatch):
  """No variables set must reproduce exactly what C1 is running."""
  monkeypatch.delenv(REWARD_ENV_VAR, raising=False)
  monkeypatch.delenv(DR_ENV_VAR, raising=False)
  cfg = kangaroo_pick_place_env_cfg()
  assert DEFAULT_REWARD == "R-A" and DEFAULT_DR == "DR0"
  assert cfg.rewards["reach"].func.__name__ == "staged_position_reward"
  assert "support_sensor_name" not in cfg.terminations["success"].params


@pytest.mark.parametrize("arm", sorted(REWARD_VARIANTS))
def test_every_registered_arm_can_be_selected(arm, monkeypatch):
  monkeypatch.setenv(REWARD_ENV_VAR, arm)
  if (
    arm in STACK_ONLY_ARMS
  ):  # plan S9's RG / SA read the base cube: the stack goal, in B's box
    monkeypatch.setenv("GOAL", "stack")
    monkeypatch.setenv("BASELINE", "B")
  cfg = kangaroo_pick_place_env_cfg()
  # Every arm resets the latch -- the eval reads it on all of them -- but only the strict
  # arms gate their success termination on it, and the S5.2 M1 arms have none at all.
  assert "reset_was_lifted" in cfg.events
  if arm in NO_SUCCESS_TERMINATION:
    assert "success" not in cfg.terminations
    return
  strict = arm in REQUIRE_LIFTED
  assert ("support_sensor_name" in cfg.terminations["success"].params) is strict


@pytest.mark.parametrize("rung", DR_LEVELS)
def test_every_registered_rung_can_be_selected(rung, monkeypatch):
  monkeypatch.setenv(DR_ENV_VAR, rung)
  assert kangaroo_pick_place_env_cfg() is not None


@pytest.mark.parametrize(
  ("var", "value"),
  [(REWARD_ENV_VAR, "R-Q"), (DR_ENV_VAR, "DR-nope"), (PPO_ENV_VAR, "PPO-Z")],
)
def test_an_unknown_name_is_loud(var, value, monkeypatch):
  """Silently training R-A under a name that claims otherwise is the failure mode."""
  monkeypatch.setenv(var, value)
  with pytest.raises(ValueError, match=value):
    kangaroo_pick_place_env_cfg()
    kangaroo_pick_place_rl_cfg()


def test_the_error_names_the_valid_options(monkeypatch):
  """A cluster job's log is the only place this message will ever be read."""
  monkeypatch.setenv(REWARD_ENV_VAR, "typo")
  with pytest.raises(ValueError) as excinfo:
    kangaroo_pick_place_env_cfg()
  for arm in REWARD_VARIANTS:
    assert arm in str(excinfo.value)


def test_the_default_ppo_is_c1r_c2rs_recipe(monkeypatch):
  """S5.1 item 9: Claudius's exploration settings, and nothing else changed."""
  monkeypatch.delenv(PPO_ENV_VAR, raising=False)
  cfg = kangaroo_pick_place_rl_cfg()
  assert DEFAULT_PPO == "PPO-stack"
  assert cfg.actor.distribution_cfg == {
    "class_name": "GaussianDistribution",
    "init_std": 0.35,
    "std_type": "log",
    "std_range": (1.0e-3, 0.5),
  }
  assert cfg.algorithm.entropy_coef == 0.001
  assert cfg.algorithm.learning_rate == 3.0e-4
  assert cfg.num_steps_per_env == 32
  # The rest is PPO-A's: same networks, same PPO, same checkpointing.
  base = manipulation_ppo_runner_cfg("kangaroo_pick_place")
  assert cfg.actor.hidden_dims == base.actor.hidden_dims
  assert cfg.critic == base.critic
  assert cfg.algorithm.schedule == base.algorithm.schedule == "adaptive"
  assert cfg.algorithm.gamma == base.algorithm.gamma
  assert cfg.save_interval == base.save_interval


def test_ppo_a_gives_back_what_c0_to_c2_ran(monkeypatch):
  monkeypatch.setenv(PPO_ENV_VAR, "PPO-A")
  assert kangaroo_pick_place_rl_cfg() == manipulation_ppo_runner_cfg(
    "kangaroo_pick_place"
  )


def test_the_recipe_does_not_leak_into_the_shared_base():
  """`stack_ppo_runner_cfg` edits a fresh base cfg; lift and stack must not see it."""
  PPO_RECIPES["PPO-stack"]("x")
  assert manipulation_ppo_runner_cfg("x").actor.distribution_cfg["init_std"] == 1.0
