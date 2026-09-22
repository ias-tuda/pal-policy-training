"""Every rung of the domain-randomisation ladder randomises what it claims to.

S2's exit criterion: build the environment at each level and assert that the randomised
fields differ across environments, and that at DR0 they do not. The test is written against
the *compiled model*, not against the config, because a DR term that is present in the event
dict and silently does nothing is exactly the failure this is here to catch -- and there is
precedent for it in this repo (see CLAUDE.md on geoms re-enabled after compile).

Four environments each, on the test device. The levels are cumulative, so each row of
``EXPECTED`` is the full set of fields that should vary at that rung.
"""

import mujoco
import numpy as np
import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import (
  make_pick_place_env_cfg,
)
from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS_LEFT_ARM
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import DR_LEVELS

NUM_ENVS = 4

#: Which quantity each rung is supposed to start varying. Cumulative: a rung varies its own
#: row and every row above it.
ADDS = {
  "DR0": (),
  "DR-object": ("object friction", "object mass"),
  "DR-contact": ("pad friction",),
  "DR-robot": ("actuator gains", "joint friction"),
  "DR-obs": ("encoder bias",),
  "DR-scene": ("table height",),
}


def _expected(level: str) -> set[str]:
  return {
    field for rung in DR_LEVELS[: DR_LEVELS.index(level) + 1] for field in ADDS[rung]
  }


def _spread_across_envs(values: torch.Tensor | np.ndarray) -> float:
  """Largest per-element range over the environment axis.

  Not the spread of the whole tensor: actuators legitimately differ from each other by three
  orders of magnitude, and taking a global max-minus-min would report the leg drive next to
  the wrist as randomisation.
  """
  array = values.numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
  array = array.reshape(array.shape[0], -1)
  return float((array.max(axis=0) - array.min(axis=0)).max())


def _measure(env: ManagerBasedRlEnv) -> dict[str, float]:
  model, mj_model = env.sim.model, env.sim.mj_model

  def geom(name: str) -> int:
    return mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)

  def body(name: str) -> int:
    return mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)

  return {
    "object friction": _spread_across_envs(
      model.geom_friction[:, geom("object/object_geom"), 0]
    ),
    "object mass": _spread_across_envs(model.body_mass[:, body("object/object")]),
    "pad friction": _spread_across_envs(
      model.geom_friction[:, geom("robot/gripper_left_fingertip_left_collision"), 0]
    ),
    "actuator gains": _spread_across_envs(model.actuator_gainprm[:, :, 0]),
    "joint friction": _spread_across_envs(model.dof_frictionloss),
    "encoder bias": _spread_across_envs(env.scene["robot"].data.encoder_bias),
    "table height": _spread_across_envs(model.body_pos[:, body("table/table"), 2]),
  }


@pytest.fixture(scope="module")
def measured() -> dict[str, dict[str, float]]:
  """One env per rung, measured once: building six of them is the expensive part."""
  out = {}
  for level in DR_LEVELS:
    torch.manual_seed(0)
    cfg = make_pick_place_env_cfg(
      KANGAROO_GRIPPERS_LEFT_ARM, gripper_name="left", dr=level
    )
    cfg.scene.num_envs = NUM_ENVS
    env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
    env.reset()
    out[level] = _measure(env)
    env.close()
  return out


@pytest.mark.parametrize("level", DR_LEVELS)
def test_each_rung_randomises_exactly_what_it_claims(level, measured):
  expected = _expected(level)
  for field, spread in measured[level].items():
    if field in expected:
      assert spread > 0.0, f"{level}: {field} is identical across envs"
    else:
      assert spread == 0.0, f"{level}: {field} varies but {level} does not claim it"


def test_dr0_randomises_nothing(measured):
  assert all(spread == 0.0 for spread in measured["DR0"].values())


def test_the_ladder_is_cumulative(measured):
  """A rung keeps everything the rung below it randomised; that is what makes it a ladder."""
  for lower, upper in zip(DR_LEVELS, DR_LEVELS[1:], strict=False):
    for field, spread in measured[lower].items():
      if spread > 0.0:
        assert measured[upper][field] > 0.0, f"{upper} dropped {field} from {lower}"


def test_ranges_land_where_the_config_says():
  """Spot-check two absolute ranges, so a units slip shows up as a failure not a curve."""
  from pal_mjlab.tasks.manipulation import tabletop_env_cfg as base

  torch.manual_seed(0)
  cfg = make_pick_place_env_cfg(
    KANGAROO_GRIPPERS_LEFT_ARM, gripper_name="left", dr="DR-object"
  )
  cfg.scene.num_envs = 64
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  env.reset()
  mj_model = env.sim.mj_model
  body = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "object/object")
  geom = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "object/object_geom")

  mass = env.sim.model.body_mass[:, body].numpy()
  friction = env.sim.model.geom_friction[:, geom, 0].numpy()
  env.close()

  lo, hi = base._DR_OBJECT_MASS_KG
  assert lo <= mass.min() and mass.max() <= hi
  lo, hi = base._DR_OBJECT_FRICTION
  assert lo <= friction.min() and friction.max() <= hi


def test_an_unknown_level_is_an_error():
  with pytest.raises(ValueError, match="DR-everything"):
    make_pick_place_env_cfg(
      KANGAROO_GRIPPERS_LEFT_ARM, gripper_name="left", dr="DR-everything"
    )


def _pick_place_env(level: str, num_envs: int, drop_events: tuple[str, ...] = ()):
  torch.manual_seed(0)
  cfg = make_pick_place_env_cfg(
    KANGAROO_GRIPPERS_LEFT_ARM, gripper_name="left", dr=level
  )
  cfg.scene.num_envs = num_envs
  for name in drop_events:
    cfg.events.pop(name)
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  env.reset()
  return env


def _table_top_z(env: ManagerBasedRlEnv) -> np.ndarray:
  geom = mujoco.mj_name2id(
    env.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "table/table_top"
  )
  centre = env.sim.data.geom_xpos[:, geom, 2].numpy()
  return centre + env.sim.model.geom_size[:, geom, 2].numpy()


def test_dr_scene_goal_matches_the_fixed_goal():
  """Swapping the command at DR-scene must not move the goal when the table does not move."""
  from pal_mjlab.tasks.manipulation.pick_place import pick_place_env_cfg as cfg_mod

  env = _pick_place_env("DR-scene", num_envs=2, drop_events=("table_height",))
  goal = env.command_manager.get_term("goal_pose").command.numpy()
  env.close()

  expected = np.array([*cfg_mod.TARGET_XY, cfg_mod.TARGET_Z])
  assert goal == pytest.approx(np.tile(expected, (2, 1)), abs=1e-6)


def test_dr_scene_goal_tracks_the_table():
  """With the table moving, the goal has to move with it or the task becomes unwinnable.

  A 2 cm table offset against a 2.5 cm success threshold leaves nothing for the xy error,
  which is why this rung uses a goal derived from the surface rather than a constant z.
  """
  from pal_mjlab.tasks.manipulation.pick_place import pick_place_env_cfg as cfg_mod

  env = _pick_place_env("DR-scene", num_envs=8)
  goal = env.command_manager.get_term("goal_pose").command.numpy()
  top_z = _table_top_z(env)
  env.close()

  assert top_z.max() - top_z.min() > 0.0, "the table did not move; nothing was tested"
  assert goal[:, 0] == pytest.approx(cfg_mod.TARGET_XY[0], abs=1e-6)
  assert goal[:, 1] == pytest.approx(cfg_mod.TARGET_XY[1], abs=1e-6)
  assert goal[:, 2] == pytest.approx(top_z + 0.025, abs=1e-6)
