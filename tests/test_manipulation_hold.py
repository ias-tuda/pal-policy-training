"""The two halves of S4: who commands the joints the policy does not, and who carries the arm.

`hold_nominal_targets` writes the configured default to every joint target at reset, because
`Entity.clear_state` zeroes them and only the action joints get one written back. Gravity
compensation on the torso and arms is what lets PAL's own impedance gains hold the result --
they need 16 Nm at a stiffness of 40 otherwise. Both are decisions with a blast radius, so
both are pinned here.
"""

import mujoco
import pytest
from mjlab.scene import Scene, SceneCfg
from mjlab.terrains import TerrainEntityCfg
from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import (
  get_kangaroo_grippers_robot_cfg,
  get_kangaroo_grippers_tabletop_robot_cfg,
)
from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import make_tabletop_env_cfg

_COMPENSATED = ("pelvis_1_link", "pelvis_2_link", "arm_", "gripper_")
_NOT_COMPENSATED = ("base_link", "leg_")


def _compile(entity_cfg) -> mujoco.MjModel:
  return Scene(
    SceneCfg(
      num_envs=1,
      env_spacing=2.5,
      entities={"robot": entity_cfg},
      terrain=TerrainEntityCfg(terrain_type="plane"),
    ),
    "cpu",
  ).compile()


def _gravcomp_by_body(model: mujoco.MjModel) -> dict[str, float]:
  return {
    (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").split("/", 1)[
      -1
    ]: float(model.body_gravcomp[b])
    for b in range(model.nbody)
  }


@pytest.fixture(scope="module")
def tabletop() -> mujoco.MjModel:
  return _compile(get_kangaroo_grippers_tabletop_robot_cfg())


def test_torso_and_arms_are_gravity_compensated(tabletop):
  gravcomp = _gravcomp_by_body(tabletop)
  compensated = {n for n, v in gravcomp.items() if v}
  assert compensated, "no body carries gravity compensation"
  for name in compensated:
    assert name.startswith(_COMPENSATED), name


def test_the_legs_and_the_welded_base_are_not(tabletop):
  """The legs keep their real weight: foot contact and the leg four-bar must not change."""
  for name, value in _gravcomp_by_body(tabletop).items():
    if name.startswith(_NOT_COMPENSATED):
      assert value == 0.0, name


def test_gravcomp_can_be_switched_off_for_comparison():
  model = _compile(get_kangaroo_grippers_tabletop_robot_cfg(gravcomp=False))
  assert not model.body_gravcomp.any()


def test_the_locomotion_variant_is_untouched():
  """`Mjlab-Velocity-*` and `Mjlab-Reaching-*` share this cfg and must not inherit any of it."""
  model = _compile(get_kangaroo_grippers_robot_cfg())
  assert not model.body_gravcomp.any()


def test_tabletop_env_holds_the_uncommanded_joints_by_default():
  cfg = make_tabletop_env_cfg(KANGAROO_GRIPPERS, gripper_name="left")
  term = cfg.events["hold_nominal_targets"]
  assert term.mode == "reset"
  # It has to run after the object placement's sibling `reset_to_default_pose`, which is
  # what writes the qpos this then commands.
  order = list(cfg.events)
  assert order.index("reset_to_default_pose") < order.index("hold_nominal_targets")


def test_the_hold_event_can_be_switched_off_for_comparison():
  cfg = make_tabletop_env_cfg(
    KANGAROO_GRIPPERS, gripper_name="left", hold_nominal=False
  )
  assert "hold_nominal_targets" not in cfg.events
