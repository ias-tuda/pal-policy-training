"""S5.2: the release/place arms are what the plan says, pay nothing for a slide, and can be
finished -- and the eval counts their finishes.

Built on the c1a base, R-C-strict (plan S5.2, revision 2026-09-24 10:27). Four arms add
modules on top of it (see `pick_place/rewards.py`): M1 per-step dominance, M2 no-cliff goal,
M3 lift fade, M4 terminal bonus. The per-step ordering is `test_pick_place_s52_ordering.py`.
"""

import importlib.util
import pathlib
import sys

import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from pal_mjlab.tasks.manipulation.mdp.observations import _WAS_LIFTED
from pal_mjlab.tasks.manipulation.pick_place.kangaroo.env_cfg import (
  REWARD_ENV_VAR,
  kangaroo_pick_place_env_cfg,
)
from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import (
  OBJECT,
  make_pick_place_env_cfg,
)
from pal_mjlab.tasks.manipulation.pick_place.rewards import (
  M1_PLACED_WEIGHT,
  M1_RELEASE_WEIGHT,
  M2_GOAL_WEIGHT,
  M2_NEAR_STD,
  M2_NEAR_WEIGHT,
  M4_SUCCESS_WEIGHT,
  NO_SUCCESS_TERMINATION,
  REQUIRE_LIFTED,
  S52_ARMS,
  S52_BASE,
)
from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS_LEFT_ARM

_SCRIPT = pathlib.Path(__file__).parent.parent / "scripts/scripted_pick_place.py"
_SPEC = importlib.util.spec_from_file_location("scripted_pick_place_s52r", _SCRIPT)
assert _SPEC and _SPEC.loader
scripted = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = scripted
_SPEC.loader.exec_module(scripted)

ARMS = (S52_BASE, *S52_ARMS)
TASK_TERMS = (
  "reach",
  "grasp",
  "lift",
  "goal",
  "goal_near",
  "release",
  "placed",
  "success",
)
#: Everything that must require the `was_lifted` latch (plan S5.2, design rule 3).
GATED = ("goal", "goal_near", "release", "placed", "success")


def _cfg(arm: str):
  return make_pick_place_env_cfg(
    KANGAROO_GRIPPERS_LEFT_ARM, gripper_name="left", reward=arm
  )


@pytest.mark.parametrize("arm", sorted(S52_ARMS))
def test_every_arm_is_selectable_by_name(arm, monkeypatch):
  """D10: the arms are reached through the same REWARD= hook as every other."""
  monkeypatch.setenv(REWARD_ENV_VAR, arm)
  cfg = kangaroo_pick_place_env_cfg()
  assert "trunk_clearance" in cfg.rewards and "action_rate_l2" in cfg.rewards


@pytest.mark.parametrize("arm", ARMS)
def test_the_modules_are_what_the_plan_says(arm):
  rewards, terms = _cfg(arm).rewards, _cfg(arm).terminations
  # The base's grasp and transport side is untouched except where a module says otherwise.
  assert rewards["reach"].func.__name__ == "reach_only"
  assert rewards["grasp"].weight == 1.0
  expect_lift = "lift_height_faded" if "M3" in arm else "lift_height"
  assert rewards["lift"].func.__name__ == expect_lift
  if "M1" in arm:
    assert "success" not in terms and "success" not in rewards
    assert rewards["release"].weight == M1_RELEASE_WEIGHT
    assert rewards["placed"].weight == M1_PLACED_WEIGHT
    assert arm in NO_SUCCESS_TERMINATION
  else:
    assert "support_sensor_name" in terms["success"].params  # gated termination
    assert arm in REQUIRE_LIFTED
  if "M2" in arm:
    assert rewards["goal"].func.__name__ == "bring_object_after_lift"
    assert rewards["goal"].weight == M2_GOAL_WEIGHT
    assert rewards["goal_near"].weight == M2_NEAR_WEIGHT
    assert rewards["goal_near"].params["std"] == M2_NEAR_STD
  else:
    assert rewards["goal"].func.__name__ == "bring_object_when_lifted"
    assert "goal_near" not in rewards
  if "M4" in arm:
    assert rewards["success"].weight == M4_SUCCESS_WEIGHT


@pytest.mark.parametrize("arm", ARMS)
def test_every_release_placed_and_success_term_requires_the_latch(arm):
  rewards = _cfg(arm).rewards
  for name in ("release", "placed", "success", "goal_near"):
    if name in rewards:
      assert rewards[name].params.get("support_sensor_name"), (
        f"{arm}: {name} is ungated"
      )


# -- A slid cube --------------------------------------------------------------------------


def _resolved_terms(env, arm: str) -> dict:
  rewards = _cfg(arm).rewards
  for cfg in rewards.values():
    for value in cfg.params.values():
      if isinstance(value, SceneEntityCfg):
        value.resolve(env.scene)
  return {n: c for n, c in rewards.items() if n in TASK_TERMS}


@pytest.fixture(scope="module")
def slid_env():
  """The cube pushed onto the target along the table, never lifted: at rest on the goal, not
  held, on the table, latch unset. Stepped after the teleport, since contact sensors are
  empty until a step (see `test_pick_place_strict_gate.py`)."""
  cfg = _cfg(f"{S52_BASE}+M1+M2+M3")
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  env.reset()
  zero = torch.zeros(env.action_space.shape, device=env.device)
  for _ in range(3):
    env.step(zero)
  obj = env.scene[OBJECT]
  pose = obj.data.root_link_pose_w.clone()
  pose[:, :3] = env.command_manager.get_term("goal_pose").command
  obj.write_root_link_pose_to_sim(pose)
  obj.write_root_link_velocity_to_sim(torch.zeros_like(obj.data.root_link_vel_w))
  env.step(zero)
  yield env
  env.close()


@pytest.mark.parametrize("arm", ARMS)
def test_a_slid_cube_earns_nothing_past_reach(arm, slid_env):
  assert not bool(getattr(slid_env, _WAS_LIFTED).any()), "the cube was never lifted"
  for name, cfg in _resolved_terms(slid_env, arm).items():
    value = cfg.func(slid_env, **cfg.params)
    if name == "reach":
      assert torch.all(value > 0.0)
    else:
      assert torch.all(value == 0.0), f"{arm}: {name} paid a slide: {value}"


@pytest.mark.parametrize("arm", ARMS)
def test_the_same_cube_pays_once_the_latch_is_set(arm, slid_env):
  """Positive control: a gated term that returns zero because it is broken also 'passes'."""
  latch = getattr(slid_env, _WAS_LIFTED)
  saved = latch.clone()
  latch[:] = True
  try:
    terms = _resolved_terms(slid_env, arm)
    for name in GATED:
      if name in terms and name != "goal":  # R-C's goal also needs "off the table now"
        value = terms[name].func(slid_env, **terms[name].params)
        assert torch.all(value > 0.0), f"{arm}: {name} is gated on more than was_lifted"
  finally:
    latch[:] = saved


# -- The scripted pick-and-place under every arm ------------------------------------------


@pytest.mark.parametrize("arm", ARMS)
def test_the_scripted_pick_and_place_finishes_under_every_arm(arm):
  """Every arm must be finishable, and `eval_pick_place.py`'s Tally -- which the rollout uses
  -- must count `placed ∧ was_lifted` for it: through `observe` when the placed state
  persists (no success termination, M1) and through `terminal` when it ends the episode."""
  placed_after: list[bool] = []
  seen = {"placed": False}

  def hook(phase, env):
    if "placed" not in env.reward_manager.active_terms:
      return
    value = float(
      env.reward_manager._step_reward[
        0, env.reward_manager.active_terms.index("placed")
      ]
    )
    if value > 0:
      seen["placed"] = True
    if seen["placed"]:
      placed_after.append(value > 0)

  rows = scripted.rollout(
    num_envs=2, device=get_test_device(), reward=arm, verbose=False, hook=hook
  )
  assert all(r["placed_and_lifted"] for r in rows), rows
  assert all(r["steps_to_place"] >= 0 for r in rows), rows
  if arm in NO_SUCCESS_TERMINATION:
    assert not any(r["ended_by"] for r in rows), "an M1 arm terminated"
    assert placed_after and sum(placed_after) / len(placed_after) > 0.9
  else:
    assert all(r["ended_by"] == "success" for r in rows), rows
