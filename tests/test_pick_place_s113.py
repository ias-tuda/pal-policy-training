"""Plan S11.3 (`pick-place-s11-bimanual-plan.md`, *SEQ on the high table*): the SEQ chain's two codes and
the two-arm env on a table raised by ``TABLE_DZ=22`` mm (0.815 -> 0.837 m), the robot cell's height.

S11.3 resumes L-T s2 @ 8998 and R-ST-W s1 @ 7998 with ``TABLE_DZ=22`` and nothing else. What is checked,
on CPU: unset and ``TABLE_DZ=0`` are each code's cfg exactly (L-T, R-ST-W and the two-arm env that
`eval_chain.py` / `eval_bimanual.py` build); 22 moves only the table entity and the fall-off heights
written from its top; bad values are refused; and at the raised table A and B start on the raised top,
R-ST-W's base and L-T's goal sit at T on the raised top, and both arms' start poses are clear of it
(the clearances are printed for step 0 item 2).
"""

from __future__ import annotations

import dataclasses
import re

import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from pal_mjlab.tasks.manipulation import mdp, scene
from pal_mjlab.tasks.manipulation.pick_place import arm_table as at
from pal_mjlab.tasks.manipulation.pick_place import bimanual as bi
from pal_mjlab.tasks.manipulation.pick_place.kangaroo.env_cfg import (
  kangaroo_pick_place_env_cfg,
)
from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import T11_XY
from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import TABLE

RCTP = "R-C-strict+M1+M2+M3+M6+J+M8"
#: BS11 (`test_pick_place_s11`): the selectors L-T s2 (161302) and R-ST-W s1 (161305) trained with.
BS11 = {
  "DR": "DR-scene",
  "PPO": "PPO-stack",
  "SLEW": "stack",
  "BASELINE": "B",
  "SPAWN": "W2",
  "TOOL_SPEED": "none",
  "GAINS": "G20",
  "EFFORT": "CTP",
  "LAG": "FP+JS",
  "ACTION": "abs",
  "EPISODE_S": "10",
  "CRITIC_OBS": "q",
  "ARM_TABLE": "AH10",
  "ARM_HOME": "mirror",
  "TRACKER": "split",
  "TARGET": "T11",
}
CODES = {
  "L-T": {"REWARD": RCTP + "+YG"},
  "R-ST-W": {"REWARD": RCTP + "+SA", "GOAL": "stack", "TARGET": "T11W"},
  # The chain eval's env (`eval_chain.py`, `ARMS=both`): BS11 is fixed inside it.
  "chain": {
    "REWARD": "S11-BI",
    "ARMS": "both",
    "HANDOFF_TRAIN": "left",
    "START": "scratch",
    "HANDOFF_ROUTE": "up-out",
    "EPISODE_S": "30",
    "TARGET": None,
  },
}
_ALL = {
  *BS11,
  "REWARD",
  "GOAL",
  "GATE",
  "ARMS",
  "HANDOFF_TRAIN",
  "START",
  "HANDOFF_ROUTE",
  "SHAPE",
  "OCC",
  "OBJ_OBS",
  "CMD_OBS",
  "TABLE_DZ",
}
DZ = 0.022
TABLE_CFG = SceneEntityCfg(TABLE)


def _set_env(monkeypatch, code: str, **changes) -> None:
  for var in _ALL:
    monkeypatch.delenv(var, raising=False)
  for var, value in {**BS11, **CODES[code], **changes}.items():
    if value is None:
      monkeypatch.delenv(var, raising=False)
    else:
      monkeypatch.setenv(var, value)


def _cfg(monkeypatch, code: str, num_envs: int = 4, **changes):
  _set_env(monkeypatch, code, **changes)
  cfg = kangaroo_pick_place_env_cfg()
  cfg.scene.num_envs = num_envs
  cfg.seed = 0
  return cfg


def _flat(obj, prefix=""):
  """Every leaf of a cfg as path -> repr, so two cfgs can be compared field by field."""
  out = {}
  if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
    for f in dataclasses.fields(obj):
      out.update(_flat(getattr(obj, f.name), f"{prefix}.{f.name}"))
  elif isinstance(obj, dict):
    for k, v in obj.items():
      out.update(_flat(v, f"{prefix}[{k!r}]"))
  elif isinstance(obj, (list, tuple)):
    for i, v in enumerate(obj):
      out.update(_flat(v, f"{prefix}[{i}]"))
  else:
    # A closure's repr carries its address, which differs between any two builds.
    out[prefix] = re.sub(r" at 0x[0-9a-f]+", "", repr(obj))
  return out


def _diff(a, b):
  fa, fb = _flat(a), _flat(b)
  return {
    k: (fa.get(k), fb.get(k)) for k in fa.keys() | fb.keys() if fa.get(k) != fb.get(k)
  }


# -- The cfg -----------------------------------------------------------------------------


@pytest.mark.parametrize("code", list(CODES))
def test_unset_and_zero_are_the_trained_cfg_exactly(monkeypatch, code):
  base = _cfg(monkeypatch, code)
  zero = _cfg(monkeypatch, code, TABLE_DZ="0")
  assert _diff(base, zero) == {}
  assert base.scene.entities[TABLE].init_state.pos[2] == pytest.approx(
    scene.ROBOT_CELL_TABLE_TOP_Z - scene.TABLE_THICKNESS / 2
  )


@pytest.mark.parametrize("code", list(CODES))
def test_22_moves_the_table_and_only_what_is_written_from_its_top(monkeypatch, code):
  base = _cfg(monkeypatch, code)
  high = _cfg(monkeypatch, code, TABLE_DZ="22")
  assert high.scene.entities[TABLE].init_state.pos[2] == pytest.approx(
    scene.ROBOT_CELL_TABLE_TOP_Z + DZ - scene.TABLE_THICKNESS / 2
  )
  diff = _diff(base, high)
  assert diff, "TABLE_DZ=22 changed nothing"
  for path, (a, b) in diff.items():
    # The table's position, the table spec closure's height, and the fall-off terminations' heights.
    allowed = f".scene.entities[{TABLE!r}]" in path or (
      "terminations" in path and "min_height" in path
    )
    assert allowed, (path, a, b)
  heights = {p: (float(a), float(b)) for p, (a, b) in diff.items() if "min_height" in p}
  assert heights, "no fall-off height moved"
  for p, (a, b) in heights.items():
    assert b - a == pytest.approx(DZ), p
  if code == "chain":
    # Both cubes' fall-off heights moved (A's and B's).
    assert any("cube_b_dropped" in p for p in heights)


@pytest.mark.parametrize("bad", ["220", "-51", "nan", "2.2cm", "inf"])
def test_bad_offsets_are_refused(monkeypatch, bad):
  with pytest.raises(ValueError, match="TABLE_DZ"):
    _cfg(monkeypatch, "L-T", TABLE_DZ=bad)
  with pytest.raises(ValueError, match="TABLE_DZ"):
    _cfg(monkeypatch, "chain", TABLE_DZ=bad)


# -- The envs, on the raised table ----------------------------------------------------------


def _build(code: str, dz: str, num_envs: int = 8):
  mp = pytest.MonkeyPatch()
  cfg = _cfg(mp, code, num_envs=num_envs, TABLE_DZ=dz)
  cfg.observations["actor"].enable_corruption = False
  cfg.seed = 7
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  env.reset()
  return env, mp


@pytest.fixture(scope="module")
def envs():
  """Each code at 0 and 22 mm, the same seed, the actor's noise off."""
  out = {(c, dz): _build(c, dz) for c in CODES for dz in ("0", "22")}
  yield {k: v[0] for k, v in out.items()}
  for env, mp in out.values():
    env.close()
    mp.undo()


def _hold(env, steps: int) -> None:
  with torch.inference_mode():
    for _ in range(steps):
      env.step(torch.zeros(env.action_space.shape, device=env.device))


def _arms(code: str):
  """(label, arm 1-6 cfg, arm-table sensor) per arm the env watches."""
  out = [
    (
      "left",
      SceneEntityCfg(
        "robot", body_names=(at.arm_body_regex(KANGAROO_GRIPPERS.gripper("left")),)
      ),
      at.ARM_TABLE_SENSOR,
    )
  ]
  if code == "chain":
    out.append(
      (
        "right",
        SceneEntityCfg(
          "robot", body_names=(at.arm_body_regex(KANGAROO_GRIPPERS.gripper("right")),)
        ),
        "right_" + at.ARM_TABLE_SENSOR,
      )
    )
  return out


@pytest.mark.parametrize("code", list(CODES))
def test_the_table_top_is_raised(envs, code):
  low = mdp.surface_top_z(envs[(code, "0")], TABLE_CFG)
  high = mdp.surface_top_z(envs[(code, "22")], TABLE_CFG)
  # DR-scene moves each table by up to +/-1 cm about its nominal top; the same seed draws the same.
  assert torch.allclose(high - low, torch.full_like(low, DZ), atol=1e-5)
  assert ((high - (scene.ROBOT_CELL_TABLE_TOP_Z + DZ)).abs() <= 0.0101).all()


@pytest.mark.parametrize("code", list(CODES))
def test_the_start_pose_is_clear_of_the_raised_table(envs, code):
  """Step 0 item 2: arm 1-6 at the start pose (`ARM_HOME=mirror`) over the raised table, per arm. Read
  at the reset pose; printed so the step-0 record has the number. Inside AH's margin is recorded, not
  failed (the start pose is the robot's); touching the table is a failure."""
  for dz in ("0", "22"):
    env = envs[(code, dz)]
    top = mdp.surface_top_z(env, TABLE_CFG)
    for label, arm, sensor in _arms(code):
      arm.resolve(env.scene)
      clearance = mdp.arm_table_clearance(env, arm, TABLE_CFG)
      finite = clearance[torch.isfinite(clearance)]
      print(
        f"{code} TABLE_DZ={dz} {label}: table top {top.min().item():.4f}-{top.max().item():.4f} m, "
        f"lowest arm 1-6 clearance {finite.min().item() * 1000:.1f} mm "
        f"(AH margin {at.CLEARANCE_MARGIN * 1000:.0f} mm), per link min "
        f"{[round(v * 1000, 1) for v in clearance.amin(dim=0).tolist()]} mm"
      )
      assert (finite > 0.0).all()
      force = torch.norm(env.scene[sensor].data.force_history, dim=-1)
      assert (force <= 1.0).all()


@pytest.mark.parametrize("code", list(CODES))
def test_the_cubes_start_on_the_raised_top(envs, code):
  env = envs[(code, "22")]
  _hold(env, 10)
  top = mdp.surface_top_z(env, TABLE_CFG)
  names = {"L-T": ("object",), "R-ST-W": ("object", "base_object")}.get(
    code, ("object", bi.CUBE_B)
  )
  for name in names:
    z = env.scene[name].data.root_link_pos_w[:, 2]
    half = mdp.object_half_height(env, name)
    assert ((z - half - top).abs() < 3e-3).all(), (name, z - half - top)


def test_l_t_goal_and_r_st_w_base_sit_at_t_on_the_raised_top(envs):
  env = envs[("L-T", "22")]
  goal = env.command_manager.get_term("goal_pose").command - env.scene.env_origins
  top = mdp.surface_top_z(env, TABLE_CFG)
  assert torch.allclose(goal[:, 2] + env.scene.env_origins[:, 2], top + 0.025, atol=1e-4)
  assert float((goal[:, 0] - T11_XY[0]).abs().max()) <= 0.01 + 1e-6
  env = envs[("R-ST-W", "22")]
  base = env.scene["base_object"].data.root_link_pos_w - env.scene.env_origins
  assert float((base[:, 0] - T11_XY[0]).abs().max()) <= 0.025 + 1e-3
  assert float((base[:, 1] - T11_XY[1]).abs().max()) <= 0.025 + 1e-3


@pytest.mark.parametrize("code", ["L-T", "R-ST-W"])
def test_the_observed_cube_and_goal_rise_by_the_offset(envs, code):
  # Past the tracker's 8-step lag, settled.
  low_env, high_env = envs[(code, "0")], envs[(code, "22")]
  for env in (low_env, high_env):
    _hold(env, 40)
  cube = [mdp.object_position_b(e, "object") for e in (low_env, high_env)]
  assert torch.allclose(
    cube[1][:, 2] - cube[0][:, 2], torch.full_like(cube[0][:, 2], DZ), atol=2e-3
  )
  assert torch.allclose(cube[1][:, :2], cube[0][:, :2], atol=2e-3)
