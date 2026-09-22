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
  DEFAULT_REWARD,
  DR_ENV_VAR,
  REWARD_ENV_VAR,
  kangaroo_pick_place_env_cfg,
)
from pal_mjlab.tasks.manipulation.pick_place.rewards import (
  REQUIRE_LIFTED,
  REWARD_VARIANTS,
)
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import DR_LEVELS


def test_the_default_is_c1s_configuration(monkeypatch):
  """No variables set must reproduce exactly what C1 is running."""
  monkeypatch.delenv(REWARD_ENV_VAR, raising=False)
  monkeypatch.delenv(DR_ENV_VAR, raising=False)
  cfg = kangaroo_pick_place_env_cfg()
  assert DEFAULT_REWARD == "R-A" and DEFAULT_DR == "DR0"
  assert cfg.rewards["reach"].func.__name__ == "staged_position_reward"
  assert "reset_was_lifted" not in cfg.events


@pytest.mark.parametrize("arm", sorted(REWARD_VARIANTS))
def test_every_registered_arm_can_be_selected(arm, monkeypatch):
  monkeypatch.setenv(REWARD_ENV_VAR, arm)
  cfg = kangaroo_pick_place_env_cfg()
  # The strict arm is the one that brings the latch, and it must bring all of it.
  strict = arm in REQUIRE_LIFTED
  assert ("reset_was_lifted" in cfg.events) is strict
  assert ("support_sensor_name" in cfg.terminations["success"].params) is strict


@pytest.mark.parametrize("rung", DR_LEVELS)
def test_every_registered_rung_can_be_selected(rung, monkeypatch):
  monkeypatch.setenv(DR_ENV_VAR, rung)
  assert kangaroo_pick_place_env_cfg() is not None


@pytest.mark.parametrize(
  ("var", "value"), [(REWARD_ENV_VAR, "R-Q"), (DR_ENV_VAR, "DR-nope")]
)
def test_an_unknown_name_is_loud(var, value, monkeypatch):
  """Silently training R-A under a name that claims otherwise is the failure mode."""
  monkeypatch.setenv(var, value)
  with pytest.raises(ValueError, match=value):
    kangaroo_pick_place_env_cfg()


def test_the_error_names_the_valid_options(monkeypatch):
  """A cluster job's log is the only place this message will ever be read."""
  monkeypatch.setenv(REWARD_ENV_VAR, "typo")
  with pytest.raises(ValueError) as excinfo:
    kangaroo_pick_place_env_cfg()
  for arm in REWARD_VARIANTS:
    assert arm in str(excinfo.value)
