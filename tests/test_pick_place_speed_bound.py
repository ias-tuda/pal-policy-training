"""Plan S7, speed bound A (FA): the arm's slew limit at a fraction of the URDF rate.

S5.2's full fix places by carrying the cube at up to ~1.2 m/s (its 1 k probe and the 3 k
eval's peak cube speed). FA halves every limited joint's top speed through the action term,
``action_slew_rate_scale`` = 0.5, selected like the reward arm by an environment variable
(`SLEW`, D10's mechanism). These check the three things a run depends on: the limit really
is scaled in the action term, a bad value is refused rather than silently trained at 1.0,
and the task is still doable at the lower limit -- the scripted pick-and-place, whose
interpolation stretches to whatever limit the action term reports.
"""

import importlib.util
import math
import pathlib
import sys

import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from pal_mjlab.tasks.manipulation.pick_place.kangaroo.env_cfg import (
  DEFAULT_SLEW,
  SLEW_ENV_VAR,
  kangaroo_pick_place_env_cfg,
)
from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import (
  make_pick_place_env_cfg,
)
from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS_LEFT_ARM
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import CONTROL_DT

_SCRIPT = pathlib.Path(__file__).parent.parent / "scripts/scripted_pick_place.py"
_SPEC = importlib.util.spec_from_file_location("scripted_pick_place_fa", _SCRIPT)
assert _SPEC and _SPEC.loader
scripted = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = scripted
_SPEC.loader.exec_module(scripted)

FA = 0.5
FULL_FIX = "R-C-strict+M1+M2+M3"
#: The URDF speeds (`robots/kangaroo.py`, S5.1 item 4), rad/s.
URDF = {1: 1.95, 2: 1.95, 3: 2.35, 4: 2.35, 5: 2.35, 6: 2.35, 7: 2.35}


def _cfg(scale: float = FA):
  return make_pick_place_env_cfg(
    KANGAROO_GRIPPERS_LEFT_ARM, gripper_name="left", action_slew_rate_scale=scale
  )


@pytest.fixture(scope="module")
def env():
  cfg = _cfg()
  cfg.scene.num_envs = 1
  e = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  e.reset()
  yield e
  e.close()


def _names(term) -> list[str]:
  return [n.split("/", 1)[-1] for n in term.target_names]


def test_the_default_is_the_urdf_rate():
  """The disabled path: without the factor, the limits are exactly what S5.1 set."""
  full, default = make_pick_place_env_cfg(KANGAROO_GRIPPERS_LEFT_ARM), _cfg(1.0)
  assert full.actions["joint_pos"].rate_limit == default.actions["joint_pos"].rate_limit


def test_fa_halves_every_limited_joint_in_the_cfg():
  full = _cfg(1.0).actions["joint_pos"].rate_limit
  half = _cfg(FA).actions["joint_pos"].rate_limit
  assert full.keys() == half.keys()
  for pattern, limit in full.items():
    assert half[pattern] == pytest.approx(FA * limit)


def test_the_action_term_clamps_at_the_scaled_limit(env):
  term = env.action_manager.get_term("joint_pos")
  rate = dict(zip(_names(term), term._rate[0].tolist(), strict=True))
  for i, speed in URDF.items():
    assert rate[f"arm_left_{i}_joint"] == pytest.approx(FA * speed * CONTROL_DT)
  assert rate["gripper_left_finger_joint"] == float("inf")  # the jaw stays unlimited


def test_a_saturated_action_ramps_at_half_the_urdf_speed(env):
  term = env.action_manager.get_term("joint_pos")
  env.reset()
  default = term._default_target()[0].clone()
  ones = torch.ones(1, len(_names(term)), device=env.device)
  targets = []
  for _ in range(3):
    env.step(ones)
    targets.append(term._processed_actions[0].clone())
  steps = torch.diff(torch.stack([default, *targets]), dim=0)
  arm = torch.isfinite(term._rate[0])
  expected = term._rate[0][arm].expand(3, -1)
  assert torch.allclose(steps[:, arm].abs(), expected, atol=1e-6)


@pytest.mark.parametrize("scale", [0.0, -0.5, 1.5, math.nan])
def test_a_scale_outside_zero_to_one_is_refused(scale):
  with pytest.raises(ValueError, match="action_slew_rate_scale"):
    _cfg(scale)


def test_the_environment_variable_selects_it(monkeypatch):
  monkeypatch.delenv(SLEW_ENV_VAR, raising=False)
  unset = kangaroo_pick_place_env_cfg().actions["joint_pos"].rate_limit
  assert unset == _cfg(DEFAULT_SLEW).actions["joint_pos"].rate_limit
  monkeypatch.setenv(SLEW_ENV_VAR, "0.5")
  assert (
    kangaroo_pick_place_env_cfg().actions["joint_pos"].rate_limit
    == _cfg(FA).actions["joint_pos"].rate_limit
  )


@pytest.mark.parametrize("raw", ["fast", "0", "1.5", "nan", ""])
def test_a_bad_value_is_loud(raw, monkeypatch):
  """A typo must not silently train FA at the URDF rate under a run named slew0.5."""
  monkeypatch.setenv(SLEW_ENV_VAR, raw)
  with pytest.raises(ValueError, match=SLEW_ENV_VAR):
    kangaroo_pick_place_env_cfg()


# -- The task is still doable at the lower limit ------------------------------------------


@pytest.fixture(scope="module")
def fa_episodes() -> list[dict]:
  return scripted.rollout(
    num_envs=10,
    device=get_test_device(),
    seed=0,
    reward=FULL_FIX,
    verbose=False,
    action_slew_rate_scale=FA,
  )


def test_the_scripted_pick_and_place_succeeds_under_fa(fa_episodes):
  """S5.0's gate, at FA: placed and lifted in >= 9 of the centre + 8 corners + 1 random."""
  placed = [e["env"] for e in fa_episodes if e["placed_and_lifted"]]
  assert len(placed) >= 9, fa_episodes


def test_the_script_keeps_to_the_lower_limit(fa_episodes):
  """The interpolation stretched to the halved limit: no step asked for more than it, and
  nothing was clipped, so the result is not bought with commands the policy cannot issue."""
  assert fa_episodes[0]["over_slew_env_steps"] == 0
  assert fa_episodes[0]["clipped_env_steps"] == 0
