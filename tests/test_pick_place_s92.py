"""Plan S9.2 (`pick-place-v1-objects-plan.md`, *the one-arm stacker on the robot cell's table height*):
SC-FT's stack env on a table raised by ``TABLE_DZ=22`` mm (0.815 -> 0.837 m), and ST-H22-JW's wide jaw
servo (``LAG=FP+JSW``, S7-11).

What is checked, on CPU: unset and ``TABLE_DZ=0`` are SC-FT's cfg exactly; 22 moves only the table entity
and the fall-off heights written from its top (the red cube's and the base's); ``FP+JSW`` changes only the
jaw servo's ranges, and draws servos slower than S7-6's in an env; at the raised table both cubes start on
the raised top, the base sits in T11W, and the stack goal rises by the offset under the same draws; the
start pose is clear of the raised table (printed for step 0 item 3).
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
from pal_mjlab.tasks.manipulation import tabletop_env_cfg as tt
from pal_mjlab.tasks.manipulation.pick_place import arm_table as at
from pal_mjlab.tasks.manipulation.pick_place.kangaroo.env_cfg import (
  kangaroo_pick_place_env_cfg,
)
from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import T11_XY
from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import TABLE

#: SC-FT's selectors (IAS 162274-162275, `siwei-s9-1-stack` @ 533c92d; plan S9.2).
SC_FT = {
  "REWARD": "R-C-strict+M1+M2+M3+M6+J+M8",
  "DR": "DR-scene",
  "PPO": "PPO-stack",
  "SLEW": "stack",
  "BASELINE": "B",
  "SPAWN": "W2",
  "TARGET": "T11W",
  "GOAL": "stack",
  "GATE": "none",
  "GAINS": "G20",
  "EFFORT": "CTP",
  "LAG": "FP+JS",
  "ACTION": "abs",
  "CMD_OBS": "cmd",
  "EPISODE_S": "10",
  "CRITIC_OBS": "q",
  "ARM_TABLE": "AH10",
  "TRACKER": "split",
  "ARM_HOME": "mirror",
  "TOOL_SPEED": "none",
}
CODES = {"ST": {}, "ST-JW": {"LAG": "FP+JSW"}}
_ALL = {
  *SC_FT,
  "BASE_OBS",
  "CONTACT",
  "ARMS",
  "HANDOFF_TRAIN",
  "START",
  "HANDOFF_ROUTE",
  "SHAPE",
  "OCC",
  "OBJ_OBS",
  "TABLE_DZ",
}
DZ = 0.022
TABLE_CFG = SceneEntityCfg(TABLE)


def _set_env(monkeypatch, code: str, **changes) -> None:
  for var in _ALL:
    monkeypatch.delenv(var, raising=False)
  for var, value in {**SC_FT, **CODES[code], **changes}.items():
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
    allowed = f".scene.entities[{TABLE!r}]" in path or (
      "terminations" in path and "min_height" in path
    )
    assert allowed, (path, a, b)
  heights = {p: (float(a), float(b)) for p, (a, b) in diff.items() if "min_height" in p}
  # The red cube's and the base's fall-off heights (`object_dropped`, `base_dropped`).
  assert len(heights) == 2 and any("base" in p for p in heights), heights
  for p, (a, b) in heights.items():
    assert b - a == pytest.approx(DZ), p


@pytest.mark.parametrize("bad", ["220", "-51", "nan", "2.2cm", "inf"])
def test_bad_offsets_are_refused(monkeypatch, bad):
  with pytest.raises(ValueError, match="TABLE_DZ"):
    _cfg(monkeypatch, "ST", TABLE_DZ=bad)


def test_fp_jsw_only_widens_the_servo(monkeypatch):
  """ST-H22-JW is ST-H22 with the servo's slow end at S7-11's stress (a): nothing else moves."""
  for dz in (None, "22"):
    base = _cfg(monkeypatch, "ST", TABLE_DZ=dz)
    wide = _cfg(monkeypatch, "ST-JW", TABLE_DZ=dz)
    assert tt.JAW_SERVO_TAU_WIDE == (0.11, 0.38)
    assert tt.JAW_SERVO_MAX_VEL_WIDE == (0.625, 1.75)
    assert tt.JAW_SERVO_TAU_WIDE[0] == tt.JAW_SERVO_TAU[0]
    assert tt.JAW_SERVO_TAU_WIDE[1] == pytest.approx(2 * tt.JAW_SERVO_TAU[1])
    assert tt.JAW_SERVO_MAX_VEL_WIDE[0] == pytest.approx(0.5 * tt.JAW_SERVO_MAX_VEL[0])
    assert tt.JAW_SERVO_MAX_VEL_WIDE[1] == tt.JAW_SERVO_MAX_VEL[1]
    old, new = base.actions["joint_pos"], wide.actions["joint_pos"]
    assert (old.servo_tau, old.servo_max_vel) == (
      tt.JAW_SERVO_TAU,
      tt.JAW_SERVO_MAX_VEL,
    )
    assert (new.servo_tau, new.servo_max_vel) == (
      tt.JAW_SERVO_TAU_WIDE,
      tt.JAW_SERVO_MAX_VEL_WIDE,
    )
    diff = _diff(base, wide)
    assert diff and all(".actions['joint_pos'].servo_" in p for p in diff), sorted(diff)


def test_fp_jsw_draws_slow_servos_in_an_env(monkeypatch):
  cfg = _cfg(monkeypatch, "ST-JW", num_envs=64, TABLE_DZ="22")
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  try:
    env.reset()
    term = env.action_manager.get_term("joint_pos")
    tau, vel = term._servo_tau[:, 0], term._servo_max_vel[:, 0]
    assert ((tau >= 0.11) & (tau <= 0.38)).all()
    assert ((vel >= 0.625) & (vel <= 1.75)).all()
    # Not stuck at S7-6's range: a fair share of envs is slower than anything SC-FT trained on.
    assert (tau > 0.19).float().mean() > 0.4 and (vel < 1.25).float().mean() > 0.2
  finally:
    env.close()


# -- The env, on the raised table -----------------------------------------------------------


def _build(dz: str, num_envs: int = 16):
  mp = pytest.MonkeyPatch()
  cfg = _cfg(mp, "ST", num_envs=num_envs, TABLE_DZ=dz)
  cfg.observations["actor"].enable_corruption = False
  cfg.seed = 7
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  env.reset()
  return env, mp


@pytest.fixture(scope="module")
def envs():
  """SC-FT's env at 0 and 22 mm, the same seed, the actor's noise off."""
  out = {dz: _build(dz) for dz in ("0", "22")}
  yield {k: v[0] for k, v in out.items()}
  # Undo in reverse: each patch recorded the variables the earlier ones had set.
  for env, mp in reversed(list(out.values())):
    env.close()
    mp.undo()


def _hold(env, steps: int) -> None:
  with torch.inference_mode():
    for _ in range(steps):
      env.step(torch.zeros(env.action_space.shape, device=env.device))


def test_the_table_top_is_raised(envs):
  low = mdp.surface_top_z(envs["0"], TABLE_CFG)
  high = mdp.surface_top_z(envs["22"], TABLE_CFG)
  # DR-scene moves each table by up to +/-1 cm about its nominal top; the same seed draws the same.
  assert torch.allclose(high - low, torch.full_like(low, DZ), atol=1e-5)
  assert ((high - (scene.ROBOT_CELL_TABLE_TOP_Z + DZ)).abs() <= 0.0101).all()


def test_the_start_pose_is_clear_of_the_raised_table(envs):
  """Step 0 item 3: arm 1-6 at the start pose (`ARM_HOME=mirror`) over the raised table. Printed so the
  step-0 record has the number; inside AH's margin is recorded, not failed (the start pose is the
  robot's); touching the table is a failure."""
  arm = SceneEntityCfg(
    "robot", body_names=(at.arm_body_regex(KANGAROO_GRIPPERS.gripper("left")),)
  )
  for dz, env in envs.items():
    top = mdp.surface_top_z(env, TABLE_CFG)
    arm.resolve(env.scene)
    clearance = mdp.arm_table_clearance(env, arm, TABLE_CFG)
    finite = clearance[torch.isfinite(clearance)]
    print(
      f"SC-FT TABLE_DZ={dz}: table top {top.min().item():.4f}-{top.max().item():.4f} m, "
      f"lowest arm 1-6 clearance {finite.min().item() * 1000:.1f} mm "
      f"(AH margin {at.CLEARANCE_MARGIN * 1000:.0f} mm), per link min "
      f"{[round(v * 1000, 1) for v in clearance.amin(dim=0).tolist()]} mm"
    )
    assert (finite > 0.0).all()
    force = torch.norm(env.scene[at.ARM_TABLE_SENSOR].data.force_history, dim=-1)
    assert (force <= 1.0).all()


def test_both_cubes_start_on_the_raised_top_and_the_base_in_t11w(envs):
  env = envs["22"]
  _hold(env, 10)
  top = mdp.surface_top_z(env, TABLE_CFG)
  for name in ("object", "base_object"):
    z = env.scene[name].data.root_link_pos_w[:, 2]
    half = mdp.object_half_height(env, name)
    assert ((z - half - top).abs() < 3e-3).all(), (name, z - half - top)
  base = env.scene["base_object"].data.root_link_pos_w - env.scene.env_origins
  assert float((base[:, 0] - T11_XY[0]).abs().max()) <= 0.025 + 1e-3
  assert float((base[:, 1] - T11_XY[1]).abs().max()) <= 0.025 + 1e-3


def test_the_stack_goal_and_the_observed_cube_rise_by_the_offset(envs):
  low_env, high_env = envs["0"], envs["22"]
  # Past the tracker's 8-step lag, settled; the same seed draws the same spawns.
  for env in (low_env, high_env):
    _hold(env, 40)
  goals = [
    e.command_manager.get_term("goal_pose").command - e.scene.env_origins
    for e in (low_env, high_env)
  ]
  assert torch.allclose(
    goals[1][:, 2] - goals[0][:, 2], torch.full_like(goals[0][:, 2], DZ), atol=2e-3
  )
  assert torch.allclose(goals[1][:, :2], goals[0][:, :2], atol=2e-3)
  # The goal is the base's top: one base height above the raised table (+ the red cube's half).
  top = mdp.surface_top_z(high_env, TABLE_CFG)
  base_z = high_env.scene["base_object"].data.root_link_pos_w[:, 2]
  half = mdp.object_half_height(high_env, "base_object")
  assert ((base_z - half - top).abs() < 3e-3).all()
  cube = [mdp.object_position_b(e, "object") for e in (low_env, high_env)]
  assert torch.allclose(
    cube[1][:, 2] - cube[0][:, 2], torch.full_like(cube[0][:, 2], DZ), atol=2e-3
  )
  assert torch.allclose(cube[1][:, :2], cube[0][:, :2], atol=2e-3)
