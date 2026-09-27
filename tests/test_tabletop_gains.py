"""Plan S8.0: the tabletop gain codes build what they claim, and G0 is still today's robot.

S8 trains the pick-and-place policy on stiff pelvis/arm gains *instead of* gravity
compensation. A code that kept a piece of the compensation -- a body `gravcomp`, the
`actgravcomp` routing, a joint `actfrcrange` -- or that stiffened the left arm but not the
right arm and the pelvis would make the comparison against FR meaningless; and G0 has to stay
exactly the plant every run before S8 trained on. Both are pinned here, on the compiled model
rather than the config.
"""

import dataclasses
import math
import re

import mujoco
import numpy as np
import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scene import Scene, SceneCfg
from mjlab.terrains import TerrainEntityCfg
from pal_mjlab.robots.pal_kangaroo import kangaroo_constants as kc
from pal_mjlab.tasks.manipulation.pick_place.kangaroo.env_cfg import (
  GAINS_ENV_VAR,
  SPAWN_ROBOTS,
  kangaroo_pick_place_env_cfg,
)
from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import (
  make_pick_place_env_cfg,
)
from pal_mjlab.tasks.manipulation.robots import (
  KANGAROO_GRIPPERS_LEFT_ARM,
  kangaroo_grippers_left_arm,
)

STIFF = ("G15", "G20", "G20s")

#: Which block of a code drives which joint. Right arm = left arm.
_BLOCKS = {
  r"arm_(left|right)_[12]_joint": "s_plus",
  r"arm_(left|right)_[345]_joint": "s_minus",
  r"arm_(left|right)_[67]_joint": "xs",
  r"pelvis_1_joint": "pelvis_1",
  r"pelvis_2_joint": "pelvis_2",
}

#: `impedance_control_v37` (pal-policy-deployment, 2025-11-18): the deployed gain set G20 is
#: meant to equal. stiffness_factor / damping_factor of pelvis_1/2 + arm 1-2, and of arm 3-4.
_V37_S_PLUS = (998.7900689301515, 63.58495063434878)
_V37_S_MINUS = (523.5351393467378, 33.32928212468764)


def _compile(gains: str, gravcomp: bool | None = None) -> mujoco.MjModel:
  return Scene(
    SceneCfg(
      num_envs=1,
      env_spacing=2.5,
      entities={
        "robot": kc.get_kangaroo_grippers_tabletop_robot_cfg(
          gravcomp=gravcomp, gains=gains
        )
      },
      terrain=TerrainEntityCfg(terrain_type="plane"),
    ),
    "cpu",
  ).compile()


def _joint(model: mujoco.MjModel, j: int) -> str:
  return (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "").split("/")[-1]


def _jid(model: mujoco.MjModel, name: str) -> int:
  return next(j for j in range(model.njnt) if _joint(model, j) == name)


def _actuated(model: mujoco.MjModel) -> dict[str, int]:
  """Joint name -> the actuator driving it."""
  return {_joint(model, model.actuator_trnid[a, 0]): a for a in range(model.nu)}


def _block(name: str) -> str | None:
  return next((b for rx, b in _BLOCKS.items() if re.fullmatch(rx, name)), None)


@pytest.fixture(scope="module")
def models() -> dict[str, mujoco.MjModel]:
  return {g: _compile(g) for g in kc.TABLETOP_GAINS}


def test_g0_is_the_robot_every_run_before_s8_trained_on(models):
  """The pre-S8 construction, compiled, against G0: actuators and compensation identical."""
  before = Scene(
    SceneCfg(
      num_envs=1,
      env_spacing=2.5,
      entities={
        "robot": kc._make_robot_cfg("grippers_tabletop", kc.MANIPULATION_INIT_STATE)
      },
      terrain=TerrainEntityCfg(terrain_type="plane"),
    ),
    "cpu",
  ).compile()
  g0 = models["G0"]
  for field in (
    "actuator_gainprm",
    "actuator_biasprm",
    "actuator_forcerange",
    "actuator_trnid",
    "dof_armature",
    "body_gravcomp",
    "jnt_actgravcomp",
    "jnt_actfrclimited",
    "jnt_actfrcrange",
  ):
    np.testing.assert_array_equal(getattr(g0, field), getattr(before, field), field)
  # And the builder the stiff codes go through reproduces G0's articulation exactly.
  assert (
    kc._tabletop_articulation(kc.TABLETOP_GAINS["G0"])
    == kc.KANGAROO_GRIPPERS_TABLETOP_ARTICULATION
  )


def test_g0_carries_the_sysid_gains_and_the_compensation(models):
  g0, code = models["G0"], kc.TABLETOP_GAINS["G0"]
  assert code.gravcomp
  for name, a in _actuated(g0).items():
    block = _block(name)
    if block is None:
      continue
    params = getattr(code, block)
    assert g0.actuator_gainprm[a, 0] == pytest.approx(params["stiffness"]), name
    assert -g0.actuator_biasprm[a, 2] == pytest.approx(params["damping"]), name
  assert int(g0.jnt_actgravcomp.sum()) == 16
  assert (g0.body_gravcomp > 0).any()


@pytest.mark.parametrize("gains", STIFF)
def test_a_stiff_code_has_no_gravity_compensation_anywhere(gains, models):
  model = models[gains]
  assert not kc.TABLETOP_GAINS[gains].gravcomp
  assert not model.body_gravcomp.any()
  assert not model.jnt_actgravcomp.any()
  assert not model.jnt_actfrclimited.any()


@pytest.mark.parametrize("gains", STIFF)
def test_a_stiff_code_carries_its_gains_on_both_arms_and_the_pelvis(gains, models):
  model, g0, code = models[gains], models["G0"], kc.TABLETOP_GAINS[gains]
  covered = set()
  actuated_g0 = _actuated(g0)
  for name, a in _actuated(model).items():
    block = _block(name)
    if block is None:
      # Legs and jaw: exactly G0's.
      b = actuated_g0[name]
      np.testing.assert_array_equal(model.actuator_gainprm[a], g0.actuator_gainprm[b])
      np.testing.assert_array_equal(model.actuator_biasprm[a], g0.actuator_biasprm[b])
      np.testing.assert_array_equal(
        model.actuator_forcerange[a], g0.actuator_forcerange[b]
      )
      continue
    params = getattr(code, block)
    dof = model.jnt_dofadr[model.actuator_trnid[a, 0]]
    assert model.actuator_gainprm[a, 0] == pytest.approx(params["stiffness"]), name
    assert -model.actuator_biasprm[a, 1] == pytest.approx(params["stiffness"]), name
    assert -model.actuator_biasprm[a, 2] == pytest.approx(params["damping"]), name
    assert tuple(model.actuator_forcerange[a]) == (
      -params["effort_limit"],
      params["effort_limit"],
    ), name
    # Effort and armature are the SysID blocks' own in every code; only K and D move.
    b = actuated_g0[name]
    assert tuple(model.actuator_forcerange[a]) == tuple(g0.actuator_forcerange[b])
    assert model.dof_armature[dof] == g0.dof_armature[g0.jnt_dofadr[_jid(g0, name)]]
    # Clearly stiffer than G0 -- G15's pelvis_2 (S4's x15 of PAL's old 20) is the least,
    # at 6x the SysID 49.94.
    assert model.actuator_gainprm[a, 0] > 5 * g0.actuator_gainprm[b, 0], name
    covered.add(name)
  expected = {f"arm_{s}_{i}_joint" for s in ("left", "right") for i in range(1, 8)}
  assert covered == expected | {"pelvis_1_joint", "pelvis_2_joint"}


def test_g20_is_impedance_control_v37_and_g20s_only_lowers_the_damping():
  g20, g20s = kc.TABLETOP_GAINS["G20"], kc.TABLETOP_GAINS["G20s"]
  for block, (k, d) in (
    (g20.s_plus, _V37_S_PLUS),
    (g20.pelvis_1, _V37_S_PLUS),
    (g20.pelvis_2, _V37_S_PLUS),
    (g20.s_minus, _V37_S_MINUS),
  ):
    assert block["stiffness"] == pytest.approx(k, rel=1e-12)
    assert block["damping"] == pytest.approx(d, rel=1e-12)
  for name in ("s_plus", "s_minus", "xs", "pelvis_1", "pelvis_2"):
    a, b = getattr(g20, name), getattr(g20s, name)
    assert b["stiffness"] == a["stiffness"]
    assert b["damping"] == pytest.approx(a["damping"] / math.sqrt(20))
  # D / K: the lag behind a moving target. G0 and G20 share it; G20s cuts it by sqrt(20).
  g0 = kc.TABLETOP_GAINS["G0"]
  assert g20.s_plus["damping"] / g20.s_plus["stiffness"] == pytest.approx(
    g0.s_plus["damping"] / g0.s_plus["stiffness"], rel=1e-3
  )


def test_an_unknown_gain_code_is_loud(monkeypatch):
  with pytest.raises(ValueError, match="G99"):
    kc.get_kangaroo_grippers_tabletop_robot_cfg(gains="G99")
  with pytest.raises(ValueError, match="G99"):
    kangaroo_grippers_left_arm("G99")
  monkeypatch.setenv(GAINS_ENV_VAR, "G99")
  with pytest.raises(ValueError) as excinfo:
    kangaroo_pick_place_env_cfg()
  for code in kc.TABLETOP_GAINS:
    assert code in str(excinfo.value)


@pytest.mark.parametrize("gains", tuple(kc.TABLETOP_GAINS))
def test_the_gains_variable_selects_the_plant(gains, monkeypatch):
  monkeypatch.setenv(GAINS_ENV_VAR, gains)
  robot = kangaroo_pick_place_env_cfg().scene.entities["robot"]
  expected = (
    kc.KANGAROO_GRIPPERS_TABLETOP_ARTICULATION
    if gains == "G0"
    else kc._tabletop_articulation(kc.TABLETOP_GAINS[gains])
  )
  assert robot.articulation == expected


def test_unset_is_g0(monkeypatch):
  monkeypatch.delenv(GAINS_ENV_VAR, raising=False)
  robot = kangaroo_pick_place_env_cfg().scene.entities["robot"]
  assert robot.articulation == kc.KANGAROO_GRIPPERS_TABLETOP_ARTICULATION


@pytest.mark.parametrize(
  ("gains", "spawn"), [("G0", "W2"), ("G20", "S71"), ("G20", "W1"), ("G20", "W2")]
)
def test_the_gain_code_stacks_on_the_spawn_boxs_arm(gains, spawn, monkeypatch):
  """Plan S7-3 is S7-2 with `GAINS=G20`: under baseline B the plant must be the code's -- no
  compensation left in the compiled scene -- and the action space the spawn box's, so a W box
  keeps P2's wide scales."""
  monkeypatch.setenv(GAINS_ENV_VAR, gains)
  monkeypatch.setenv("SPAWN", spawn)
  monkeypatch.setenv("BASELINE", "B")
  cfg = kangaroo_pick_place_env_cfg()
  expected = (
    kc.KANGAROO_GRIPPERS_TABLETOP_ARTICULATION
    if gains == "G0"
    else kc._tabletop_articulation(kc.TABLETOP_GAINS[gains])
  )
  assert cfg.scene.entities["robot"].articulation == expected
  assert cfg.actions["joint_pos"].scale == SPAWN_ROBOTS[spawn].action_scale
  model = Scene(cfg.scene, "cpu").compile()
  compensated = kc.TABLETOP_GAINS[gains].gravcomp
  assert bool(model.body_gravcomp.any()) == compensated
  assert bool(model.jnt_actgravcomp.any()) == compensated


def test_a_gain_code_refuses_an_arm_on_another_robot():
  """The codes rebuild the tabletop robot; an arm built on any other would be swapped for it
  without a word."""
  other = dataclasses.replace(
    KANGAROO_GRIPPERS_LEFT_ARM, entity_cfg_fn=kc.get_kangaroo_grippers_robot_cfg
  )
  with pytest.raises(ValueError, match="tabletop robot only"):
    kangaroo_grippers_left_arm("G20", other)


def test_dr_robot_scales_around_the_codes_gains():
  """DR-robot's +/-20 % multiplies the compiled defaults, i.e. the code's K, not G0's.

  Every pelvis/arm actuator is checked, not only the driven ones: mjlab's `dr.pd_gains`
  selects by `actuator_ids`, which the rung's `SceneEntityCfg(joint_names=...)` leaves at
  `slice(None)`, so it randomises all 30 actuators -- the parked arm, the pelvis and the legs
  included (measured 2026-09-25, plan S8.0; `tabletop_env_cfg.apply_dr_level`).
  """
  torch.manual_seed(0)
  cfg = make_pick_place_env_cfg(
    kangaroo_grippers_left_arm("G20"), gripper_name="left", dr="DR-robot"
  )
  cfg.scene.num_envs = 8
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  env.reset()
  mj_model = env.sim.mj_model
  kp = env.sim.model.actuator_gainprm[:, :, 0].cpu().numpy()
  env.close()
  code = kc.TABLETOP_GAINS["G20"]
  for name, a in _actuated(mj_model).items():
    block = _block(name)
    if block is None:
      continue
    ratio = kp[:, a] / getattr(code, block)["stiffness"]
    assert ratio.min() >= 0.8 - 1e-6 and ratio.max() <= 1.2 + 1e-6, name
    assert np.ptp(ratio) > 0.0, name
