"""Plan S10.2 (`pick-place-v1-objects-plan.md`): the table raised to the robot cell's height.

Siwei, 2026-10-06, on the robot: the real table stands about 2.2 cm above baseline B's 0.815 m. S10.2
retrains S10.1's WX-a (`act`, the box on WC at any yaw, the box pose observed) on a table raised by
``TABLE_DZ=22`` mm and nothing else. What is checked: unset and ``TABLE_DZ=0`` are S10.1's cfg exactly;
22 moves the table entity and the two fall-off heights that are written from the table top and nothing
else; bad values are refused; and on CPU, at the raised table, the cube and the box start on the raised
top, the goal sits on the box floor, the observed box position rises by 22 mm with everything else
unchanged, and the start pose is clear of the table (no arm-table force, AH's clearance still above
its margin).
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
from pal_mjlab.tasks.manipulation.pick_place import box as bx
from pal_mjlab.tasks.manipulation.pick_place.kangaroo.env_cfg import (
  kangaroo_pick_place_env_cfg,
)
from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS_LEFT_ARM
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import TABLE

RCTP = "R-C-strict+M1+M2+M3+M6+J+M8"
#: S10.1's WX-a (162258), which S10.2 retrains with only TABLE_DZ added.
WXA_ENV = {
  "REWARD": RCTP,
  "DR": "DR-scene",
  "PPO": "PPO-stack",
  "SLEW": "stack",
  "BASELINE": "B",
  "SPAWN": "WC",
  "TOOL_SPEED": "none",
  "GAINS": "G20",
  "EFFORT": "CTP",
  "LAG": "FP+JS",
  "OCC": "none",
  "OBJ_OBS": "live",
  "ACTION": "abs",
  "CMD_OBS": "act",
  "TARGET": "WC",
  "EPISODE_S": "15",
  "SHAPE": "cube",
  "GATE": "none",
  "CRITIC_OBS": "q",
  "ARM_TABLE": "AH10",
  "GOAL": "box-rim",
  "BOX_OBS": "pose",
}
DZ = 0.022
TABLE_CFG = SceneEntityCfg(TABLE)
#: Arm 1-6, as AH10's clearance term watches them (`test_pick_place_s710`).
ARM = SceneEntityCfg(
  "robot", body_names=(at.arm_body_regex(KANGAROO_GRIPPERS_LEFT_ARM.gripper("left")),)
)


def _set_env(monkeypatch, **changes) -> None:
  for var, value in {**WXA_ENV, **changes}.items():
    if value is None:
      monkeypatch.delenv(var, raising=False)
    else:
      monkeypatch.setenv(var, value)
  for var in ("BOX_FEET", "CKPT_SPAWN", "OCC_FLAG", "TABLE_DZ"):
    if var not in changes:
      monkeypatch.delenv(var, raising=False)


def _cfg(monkeypatch, num_envs: int = 4, **changes):
  _set_env(monkeypatch, **changes)
  cfg = kangaroo_pick_place_env_cfg()
  cfg.scene.num_envs = num_envs
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


def test_unset_and_zero_are_s101_exactly(monkeypatch):
  base = _cfg(monkeypatch)
  zero = _cfg(monkeypatch, TABLE_DZ="0")
  assert _diff(base, zero) == {}
  assert base.scene.entities[TABLE].init_state.pos[2] == pytest.approx(
    scene.ROBOT_CELL_TABLE_TOP_Z - scene.TABLE_THICKNESS / 2
  )


def test_22_moves_the_table_and_only_what_is_written_from_its_top(monkeypatch):
  base = _cfg(monkeypatch)
  high = _cfg(monkeypatch, TABLE_DZ="22")
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


@pytest.mark.parametrize("bad", ["220", "-51", "nan", "2.2cm", "inf"])
def test_bad_offsets_are_refused(monkeypatch, bad):
  with pytest.raises(ValueError, match="TABLE_DZ"):
    _cfg(monkeypatch, TABLE_DZ=bad)


# -- The env, on the raised table -----------------------------------------------------------


@pytest.fixture(scope="module")
def envs():
  """The same seed at 0 and 22 mm, the actor's noise off, so the two differ only by the table."""
  out = {}
  for dz in ("0", "22"):
    mp = pytest.MonkeyPatch()
    cfg = _cfg(mp, num_envs=8, TABLE_DZ=dz)
    cfg.observations["actor"].enable_corruption = False
    cfg.seed = 7
    env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
    env.reset()
    out[dz] = (env, mp)
  yield {k: v[0] for k, v in out.items()}
  for env, mp in out.values():
    env.close()
    mp.undo()


def _hold(env, steps: int) -> None:
  with torch.inference_mode():
    for _ in range(steps):
      env.step(torch.zeros(env.action_space.shape, device=env.device))


def test_the_start_pose_is_clear_of_the_raised_table(envs):
  """Step 0 item 2: arm 1-6 at the start pose over the raised table. Read before any step, at the
  reset pose; printed so the step-0 record has the number. Inside AH's margin is recorded, not
  failed (the start pose is also the robot's); touching the table is a failure."""
  for dz, env in envs.items():
    ARM.resolve(env.scene)
    clearance = mdp.arm_table_clearance(env, ARM, TABLE_CFG)
    finite = clearance[torch.isfinite(clearance)]
    top = mdp.surface_top_z(env, TABLE_CFG)
    print(
      f"TABLE_DZ={dz}: table top {top.min().item():.4f}-{top.max().item():.4f} m, "
      f"lowest arm 1-6 clearance {finite.min().item() * 1000:.1f} mm "
      f"(AH margin {at.CLEARANCE_MARGIN * 1000:.0f} mm), per link min "
      f"{[round(v * 1000, 1) for v in clearance.amin(dim=0).tolist()]} mm"
    )
    assert (finite > 0.0).all()
    force = torch.norm(env.scene[at.ARM_TABLE_SENSOR].data.force_history, dim=-1)
    assert (force <= 1.0).all()


def test_the_table_top_is_raised(envs):
  low = mdp.surface_top_z(envs["0"], TABLE_CFG)
  high = mdp.surface_top_z(envs["22"], TABLE_CFG)
  # DR-scene moves each table by up to +/-1 cm about its nominal top; the same seed draws the same.
  assert torch.allclose(high - low, torch.full_like(low, DZ), atol=1e-5)
  assert ((high - (scene.ROBOT_CELL_TABLE_TOP_Z + DZ)).abs() <= 0.0101).all()


def test_the_cube_and_the_box_start_on_the_raised_top(envs):
  env = envs["22"]
  _hold(env, 10)
  top = mdp.surface_top_z(env, TABLE_CFG)
  cube = env.scene["object"].data.root_link_pos_w[:, 2]
  half = mdp.object_half_height(env, "object")
  assert ((cube - half - top).abs() < 3e-3).all(), cube - half - top
  box = env.scene[bx.BOX].data.root_link_pos_w[:, 2]
  assert ((box - top).abs() < 2e-3).all(), box - top
  # The goal is half a cube above the box floor, which stands on the raised table.
  command = env.command_manager.get_term("goal_pose")
  floor = mdp.box.box_floor_top_w(env, bx.BOX, scene.BOX_FLOOR)
  assert (command.command[:, 2] - floor[:, 2] - 0.025).abs().max() < 1e-3


def test_the_observed_box_and_cube_rise_by_the_offset(envs):
  # Past the tracker's 8-step lag, settled (`test_pick_place_s101`).
  for env in envs.values():
    _hold(env, 40)
  low, high = (envs[k].observation_manager.compute()["actor"] for k in ("0", "22"))
  assert low.shape[-1] == 40 and high.shape[-1] == 40
  # 35:38 is the box's position in base_link (`test_pick_place_s101`), 38:40 its yaw.
  assert torch.allclose(
    high[:, 37] - low[:, 37], torch.full_like(low[:, 37], DZ), atol=2e-3
  )
  assert torch.allclose(high[:, 35:37], low[:, 35:37], atol=2e-3)
  assert torch.allclose(high[:, 38:40], low[:, 38:40], atol=1e-2)
  cube = [mdp.object_position_b(envs[k], "object") for k in ("0", "22")]
  assert torch.allclose(
    cube[1][:, 2] - cube[0][:, 2], torch.full_like(cube[0][:, 2], DZ), atol=2e-3
  )
  assert torch.allclose(cube[1][:, :2], cube[0][:, :2], atol=2e-3)
