"""Plan S7, speed bound B (FB): a squared hinge on the held cube's speed, the control for FA.

FB is the full fix plus one reward term, module M5: `held_object_speed_hinge`, weight
`M5_SPEED_WEIGHT`, on the cube's linear speed above 0.4 m/s while it is in the jaw. The plan
fixes the weight by one condition -- a carry at ~1.2 m/s, the full fix's measured peak, must
cost more per step than one step of `placed` -- which these check along with the term
itself: zero unless held, zero below the threshold, squared above it.
"""

import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from pal_mjlab.tasks.manipulation import mdp
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
  M5_SPEED_THRESHOLD,
  M5_SPEED_WEIGHT,
  NO_SUCCESS_TERMINATION,
  S7_ARMS,
)
from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS_LEFT_ARM

F = "R-C-strict+M1+M2+M3"
FB = f"{F}+M5"
GRASP_SENSOR = KANGAROO_GRIPPERS_LEFT_ARM.gripper("left").contact_sensor_name


def _cfg(arm: str):
  return make_pick_place_env_cfg(
    KANGAROO_GRIPPERS_LEFT_ARM, gripper_name="left", reward=arm
  )


def test_fb_is_the_full_fix_plus_one_term():
  f, fb = _cfg(F), _cfg(FB)
  assert set(fb.rewards) - set(f.rewards) == {"held_speed"}
  for name, term in f.rewards.items():
    assert fb.rewards[name].func is term.func and fb.rewards[name].weight == term.weight
  assert FB in S7_ARMS and FB in NO_SUCCESS_TERMINATION
  assert "success" not in fb.terminations  # it is still M1: no success termination


def test_fb_is_selectable_by_name(monkeypatch):
  monkeypatch.setenv(REWARD_ENV_VAR, FB)
  assert "held_speed" in kangaroo_pick_place_env_cfg().rewards


def test_a_carry_at_the_full_fix_speed_costs_more_than_a_placed_step():
  """The plan's condition on the weight: ~1.2 m/s held costs more per step than `placed`
  pays. At the slew-bound-A speed (~0.6 m/s) it is small, so FB does not forbid carrying."""
  cost = lambda v: -M5_SPEED_WEIGHT * max(0.0, v - M5_SPEED_THRESHOLD) ** 2  # noqa: E731
  assert cost(1.2) > M1_PLACED_WEIGHT
  assert cost(0.6) < 0.1 * M1_PLACED_WEIGHT
  assert cost(M5_SPEED_THRESHOLD) == 0.0


@pytest.fixture(scope="module")
def env():
  cfg = _cfg(FB)
  cfg.scene.num_envs = 3
  e = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  e.reset()
  yield e
  e.close()


def test_the_term_prices_only_a_held_cube_above_the_threshold(env, monkeypatch):
  """Env 0 held at 1.2 m/s, env 1 held at 0.3 m/s, env 2 at 1.2 m/s but not held. The
  grasp is patched in: a real grasp needs the jaw physically closed on the cube, and what is
  under test here is the arithmetic and the gating, not the sensor."""
  obj = env.scene[OBJECT]
  vel = torch.zeros_like(obj.data.root_link_vel_w)
  vel[:, 0] = torch.tensor([1.2, 0.3, 1.2], device=env.device)
  obj.write_root_link_velocity_to_sim(vel)
  # `root_link_lin_vel_w` is derived from `cvel`, which only a forward pass recomputes.
  env.sim.forward()
  held = torch.tensor([True, True, False], device=env.device)
  monkeypatch.setattr(mdp.rewards, "_grasped_bool", lambda *_: held)
  value = mdp.held_object_speed_hinge(env, OBJECT, GRASP_SENSOR, M5_SPEED_THRESHOLD)
  expected = torch.tensor([(1.2 - 0.4) ** 2, 0.0, 0.0], device=env.device)
  assert torch.allclose(value, expected, atol=1e-5), value
