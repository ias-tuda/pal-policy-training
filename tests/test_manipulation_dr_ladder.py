"""Every rung of the domain-randomisation ladder randomises what it claims to.

S2's exit criterion: build the environment at each level and assert that the randomised
fields differ across environments, and that at DR0 they do not. The test is written against
the *compiled model*, not against the config, because a DR term that is present in the event
dict and silently does nothing is exactly the failure this is here to catch -- and there is
precedent for it in this repo (see CLAUDE.md on geoms re-enabled after compile).

Four environments each, on the test device. The levels are cumulative, so each row of
``EXPECTED`` is the full set of fields that should vary at that rung.
"""

import re

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
  "DR-robot": ("actuator gains", "joint friction", "joint damping"),
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
    "joint damping": _spread_across_envs(model.dof_damping),
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


def _dofs(env: ManagerBasedRlEnv, pattern: str) -> list[int]:
  mj_model = env.sim.mj_model
  out = []
  for j in range(mj_model.njnt):
    name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
    if re.fullmatch(pattern, name.split("/", 1)[-1]):
      out.append(int(mj_model.jnt_dofadr[j]))
  assert out, pattern
  return out


def test_dr_robot_leaves_the_parked_arm_alone():
  """Joint friction/damping vary on the driven arm and jaw only (S5.1 item +).

  mjlab resolves string-keyed ranges against every joint of the entity, so without
  `tabletop_env_cfg._on_joints` the right arm -- outside a one-arm task's action space --
  would be randomised too.
  """
  from pal_mjlab.tasks.manipulation import tabletop_env_cfg as base

  env = _pick_place_env("DR-robot", num_envs=64)
  model = env.sim.model
  friction, damping = model.dof_frictionloss.numpy(), model.dof_damping.numpy()
  dofs_of = {
    p: _dofs(env, p)
    for p in (
      r"arm_left_[1-7]_joint",
      r"gripper_left_finger_joint",
      r"arm_right_[1-7]_joint",
      r"gripper_right_finger_joint",
    )
  }
  env.close()

  for group, pattern in (
    ("arm", r"arm_left_[1-7]_joint"),
    ("jaw", r"gripper_left_finger_joint"),
  ):
    dofs = dofs_of[pattern]
    key = r"arm_.*_[1-7]_joint" if group == "arm" else r"gripper_.*_finger_joint"
    for values, ranges in (
      (friction, base._DR_ROBOT_JOINT_FRICTION),
      (damping, base._DR_ROBOT_JOINT_DAMPING),
    ):
      lo, hi = ranges[key]
      assert _spread_across_envs(values[:, dofs]) > 0.0, pattern
      assert lo <= values[:, dofs].min() and values[:, dofs].max() <= hi, pattern
  for pattern in (r"arm_right_[1-7]_joint", r"gripper_right_finger_joint"):
    dofs = dofs_of[pattern]
    assert _spread_across_envs(friction[:, dofs]) == 0.0, pattern
    assert _spread_across_envs(damping[:, dofs]) == 0.0, pattern


def test_dr_obs_gives_the_actor_a_tracker_error_model_and_the_critic_the_truth():
  """S5.1 item +: per-episode bias, per-step jitter and 0-2 steps of lag, actor only."""
  from mjlab.utils.noise import NoiseModelWithAdditiveBiasCfg

  for level in DR_LEVELS:
    cfg = make_pick_place_env_cfg(
      KANGAROO_GRIPPERS_LEFT_ARM, gripper_name="left", dr=level
    )
    actor, critic = cfg.observations["actor"].terms, cfg.observations["critic"].terms
    on = DR_LEVELS.index(level) >= DR_LEVELS.index("DR-obs")
    for name in ("object_pos", "object_yaw"):
      assert isinstance(actor[name].noise, NoiseModelWithAdditiveBiasCfg) is on, level
      assert (actor[name].delay_max_lag, actor[name].delay_min_lag) == (
        (2, 0) if on else (0, 0)
      ), level
      assert critic[name].delay_max_lag == 0, level
      assert not isinstance(critic[name].noise, NoiseModelWithAdditiveBiasCfg), level


def test_dr_obs_error_is_bounded_and_biased():
  """The actor's object position differs from the critic's by at most bias + jitter,
  and by a per-env constant, which is what makes it a bias and not noise."""
  from pal_mjlab.tasks.manipulation import tabletop_env_cfg as base

  env = _pick_place_env("DR-obs", num_envs=16)
  zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
  # Many episodes first: the bias must be redrawn at each reset, not accumulated. mjlab's
  # default "add" bias operation random-walks it (see `tabletop_env_cfg`, DR-obs).
  for _ in range(20):
    env.reset()
  errors = []
  for _ in range(10):
    obs = env.step(zero)[0]
    # Past the 2-step lag, and the cube is resting, so lag contributes nothing here.
    actor, critic = (_object_pos(env, obs, g) for g in ("actor", "critic"))
    errors.append(actor - critic)
  env.close()
  err = torch.stack(errors[3:]).numpy()  # (T, B, 3)
  bound = base._DR_OBS_OBJECT_POS_BIAS + base._DR_OBS_OBJECT_POS_NOISE + 1e-4
  assert np.abs(err).max() <= bound
  mean = err.mean(axis=0)
  assert _spread_across_envs(mean) > 2 * base._DR_OBS_OBJECT_POS_NOISE, "no bias"


def _object_pos(env: ManagerBasedRlEnv, obs, group: str) -> torch.Tensor:
  """Slice `object_pos` out of one concatenated observation group."""
  manager = env.observation_manager
  group_obs = obs[group]
  names = manager.active_terms[group]
  dims = manager.group_obs_term_dim[group]
  start = 0
  for name, dim in zip(names, dims, strict=True):
    width = int(np.prod(dim))
    if name == "object_pos":
      return group_obs[:, start : start + width]
    start += width
  raise KeyError("object_pos")
