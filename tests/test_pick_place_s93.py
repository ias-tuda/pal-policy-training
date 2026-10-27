"""Plan S9.3 (`pick-place-v1-objects-plan.md`, *the one-arm stacker on the wide area WC*): S9.2's stacker
with both cubes drawn from WC (``SPAWN=WC TARGET=WC GOAL=stack``), the base at least
`STACK_MIN_SEPARATION` from the red cube, 15 s episodes, `base_moved` on the base wherever it stands.

What is checked, on CPU: the WC stack differs from S9.2's cfg only where the plan says (the spawn, the
base's reset event, the arm's action scales, the episode); the refused combinations; in an env both
cubes start on the raised top inside WC, apart, at any yaw, and the goal sits on the base; a pushed base
is priced; a cube slid onto a far base earns nothing past reach (L4); and the reward rises strictly
along a scripted stack onto a base across the area (L5).
"""

from __future__ import annotations

import collections
import importlib.util
import math
import pathlib
import sys

import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz
from pal_mjlab.tasks.manipulation import mdp, scene
from pal_mjlab.tasks.manipulation import tabletop_env_cfg as tt
from pal_mjlab.tasks.manipulation.mdp.observations import _WAS_LIFTED
from pal_mjlab.tasks.manipulation.pick_place import pick_place_env_cfg as pp
from pal_mjlab.tasks.manipulation.pick_place.kangaroo.env_cfg import (
  kangaroo_pick_place_env_cfg,
)
from pal_mjlab.tasks.manipulation.robots import (
  KANGAROO_GRIPPERS_LEFT_ARM_WC,
  kangaroo_grippers_left_arm,
)

#: S9.2's selectors (IAS 163001-163006, `siwei-s9-2-table` @ e160d67) -- ST-H22-R's; JW adds FP+JSW.
S92 = {
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
  "TABLE_DZ": "22",
}
#: Plan S9.3's change: both cubes from WC, the WC arm, 15 s.
WIDE = {"SPAWN": "WC", "TARGET": "WC", "EPISODE_S": "15"}
CODES = {"WST-R": {}, "WST-JW": {"LAG": "FP+JSW"}, "WST-S": {"LAG": "FP+JSW"}}
_ALL = {
  *S92,
  "BASE_OBS",
  "CONTACT",
  "ARMS",
  "HANDOFF_TRAIN",
  "START",
  "HANDOFF_ROUTE",
  "SHAPE",
  "OCC",
  "OBJ_OBS",
  "CKPT_SPAWN",
  "BOX_OBS",
}
WC = pp.SPAWN_BOXES["WC"]
TABLE_CFG = SceneEntityCfg("table")


def _set_env(mp, **values) -> None:
  for var in _ALL:
    mp.delenv(var, raising=False)
  for var, value in values.items():
    if value is not None:
      mp.setenv(var, value)


def _cfg(mp, code: str | None = "WST-R", num_envs: int = 4, **changes):
  _set_env(
    mp, **{**S92, **({} if code is None else {**WIDE, **CODES[code]}), **changes}
  )
  cfg = kangaroo_pick_place_env_cfg()
  cfg.scene.num_envs = num_envs
  cfg.seed = 0
  return cfg


def _summary(cfg) -> dict:
  return {
    "rewards": {n: (t.func.__name__, t.weight) for n, t in cfg.rewards.items()},
    "terminations": sorted(cfg.terminations),
    "events": list(cfg.events),
    "actor": list(cfg.observations["actor"].terms),
    "critic": list(cfg.observations["critic"].terms),
    "entities": sorted(cfg.scene.entities),
  }


# -- The cfg -----------------------------------------------------------------------------------


def test_the_wide_stack_changes_only_the_spawn_the_base_event_the_arm_and_the_episode(
  monkeypatch,
):
  old, new = _cfg(monkeypatch, None), _cfg(monkeypatch, "WST-R")
  assert _summary(old) == _summary(new)
  assert new.episode_length_s == 15.0 and old.episode_length_s == 10.0
  base = new.events["reset_base_object"]
  assert base.func is mdp.reset_box_in_area
  cx, cy = scene.TABLE_POS[0], scene.TABLE_POS[1]
  assert base.params["area_x"] == pytest.approx((WC.x[0] - cx, WC.x[1] - cx))
  assert base.params["area_y"] == pytest.approx((WC.y[0] - cy, WC.y[1] - cy))
  assert base.params["yaw"] == pytest.approx(math.pi / 4)
  assert base.params["min_distance"] == pp.STACK_MIN_SEPARATION == 0.12
  assert base.params["redraws"] == 64 and base.params["on_half_height"]
  red = new.events["reset_objects"].params
  assert [c.name for c in red["asset_cfgs"]] == ["object"]
  assert red["pose_range"]["x"] == pytest.approx((WC.x[0] - cx, WC.x[1] - cx))
  assert red["pose_range"]["y"] == pytest.approx((WC.y[0] - cy, WC.y[1] - cy))
  assert red.get("goal_area") is None
  # The base is priced by its displacement (S11's `base_moved`), as at T11W.
  assert new.rewards["base_moved"].weight == pp.BASE_MOVED_WEIGHT
  # The WC arm: arm 1 and arm 4 wider (S7-8 Part B).
  scales = new.actions["joint_pos"].scale
  assert scales != old.actions["joint_pos"].scale


def test_the_codes_differ_only_in_the_jaw_servo(monkeypatch):
  r, jw = _cfg(monkeypatch, "WST-R"), _cfg(monkeypatch, "WST-JW")
  assert (r.actions["joint_pos"].servo_tau, jw.actions["joint_pos"].servo_tau) == (
    tt.JAW_SERVO_TAU,
    tt.JAW_SERVO_TAU_WIDE,
  )


@pytest.mark.parametrize(
  ("changes", "match"),
  [
    ({"SPAWN": "W2"}, "SPAWN must be the same area"),
    ({"SPAWN": "WCK", "TARGET": "WCK"}, "too small"),
  ],
)
def test_combinations_that_are_not_built_raise(monkeypatch, changes, match):
  with pytest.raises(ValueError, match=match):
    _cfg(monkeypatch, "WST-R", **changes)


# -- The env -----------------------------------------------------------------------------------


def _hold(env, steps: int) -> None:
  with torch.inference_mode():
    for _ in range(steps):
      env.step(torch.zeros(env.action_space.shape, device=env.device))


def _yaw(q: torch.Tensor) -> torch.Tensor:
  w, x, y, z = q.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


@pytest.fixture(scope="module")
def wide_env():
  mp = pytest.MonkeyPatch()
  cfg = _cfg(mp, "WST-JW", num_envs=128)
  cfg.observations["actor"].enable_corruption = False
  cfg.seed = 3
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  env.reset()
  yield env
  env.close()
  mp.undo()


def test_both_cubes_start_in_wc_apart_at_any_yaw_on_the_raised_top(wide_env):
  env = wide_env
  cx, cy = scene.TABLE_POS[0], scene.TABLE_POS[1]
  yaws, dists = [], []
  for episode in range(3):
    if episode:
      with torch.inference_mode():
        env.reset()
    _hold(env, 15)
    centre = mdp.events._surface_centre_xy(env, TABLE_CFG)
    top = mdp.surface_top_z(env, TABLE_CFG)
    pos = {}
    for name in ("object", "base_object"):
      data = env.scene[name].data
      rel = data.root_link_pos_w[:, :2] - centre
      assert (rel[:, 0] >= WC.x[0] - cx - 1e-3).all() and (
        rel[:, 0] <= WC.x[1] - cx + 1e-3
      ).all(), name
      assert (rel[:, 1] >= WC.y[0] - cy - 1e-3).all() and (
        rel[:, 1] <= WC.y[1] - cy + 1e-3
      ).all(), name
      half = mdp.object_half_height(env, name)
      # A few cubes keep hopping: mujoco_warp's box-box collider gives one cube in ~100 a recurring
      # spurious deep table contact (S9.2's T11W draw too; S9.3 step 0). The rest rest on the top.
      resting = (data.root_link_pos_w[:, 2] - half - top).abs() < 3e-3
      assert resting.float().mean() >= 0.97, (name, (~resting).nonzero().flatten())
      y = _yaw(data.root_link_quat_w)
      assert y.abs().max() <= math.pi / 4 + 1e-3
      yaws.append(y)
      pos[name] = data.root_link_pos_w[:, :2]
    d = torch.norm(pos["object"] - pos["base_object"], dim=-1)
    assert d.min() >= pp.STACK_MIN_SEPARATION - 2e-3, d.min()
    dists.append(d)
    goal = env.command_manager.get_term("goal_pose").command
    base = env.scene["base_object"].data.root_link_pos_w
    assert torch.allclose(goal[:, :2], base[:, :2], atol=1e-3)
    assert ((goal[:, 2] - base[:, 2]) - 0.05).abs().max() < 1e-3
  yaws, dists = torch.cat(yaws), torch.cat(dists)
  assert (yaws > math.radians(30)).any() and (yaws < -math.radians(30)).any()
  # The plan's sampled carries: 12-35 cm, median ~17 cm.
  assert 0.14 < dists.median() < 0.21 and dists.max() > 0.25, (
    dists.median(),
    dists.max(),
  )
  assert env.observation_manager.compute()["actor"].shape[-1] == 38


def test_a_pushed_base_is_priced(wide_env):
  env = wide_env
  with torch.inference_mode():
    env.reset()
  _hold(env, 2)
  term = env.reward_manager.get_term_cfg("base_moved")
  assert float(term.func(env, **term.params).abs().max()) < 0.05
  base = env.scene["base_object"]
  pose = base.data.root_link_pose_w.clone()
  pose[:, 0] += 0.03
  base.write_root_link_pose_to_sim(pose)
  _hold(env, 2)
  value = term.func(env, **term.params)
  assert torch.allclose(value, torch.ones_like(value), atol=0.05), value


# -- L4 / L5 on WC -----------------------------------------------------------------------------

GATED_TERMS = (
  "grasp",
  "lift",
  "goal",
  "goal_near",
  "release",
  "placed",
  "retreat",
  "pad_contact",
)


def _term(env, name: str) -> torch.Tensor:
  cfg = env.reward_manager.get_term_cfg(name)
  return cfg.func(env, **cfg.params)


def test_a_cube_slid_onto_a_far_base_earns_nothing_past_reach(wide_env):
  env = wide_env
  with torch.inference_mode():
    env.reset()
  _hold(env, 1)
  obj = env.scene["object"]
  goal = env.command_manager.get_term("goal_pose").command
  pose = obj.data.root_link_pose_w.clone()
  pose[:, :3] = goal
  zeros = torch.zeros(env.num_envs, device=env.device)
  pose[:, 3:] = quat_from_euler_xyz(zeros, zeros, zeros)
  obj.write_root_link_pose_to_sim(pose)
  obj.write_root_link_velocity_to_sim(torch.zeros(env.num_envs, 6, device=env.device))
  _hold(env, 1)
  assert not bool(getattr(env, _WAS_LIFTED).any()), "the cube was never lifted"
  assert torch.all(_term(env, "reach") > 0.0)
  for name in GATED_TERMS:
    if name in env.reward_manager.active_terms:
      value = _term(env, name)
      assert torch.all(value == 0.0), f"{name} paid a slide: {value}"


_SCRIPT = (
  pathlib.Path(__file__).resolve().parents[1] / "scripts" / "scripted_pick_place.py"
)
_SPEC = importlib.util.spec_from_file_location("scripted_pick_place_s93", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
scripted = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = scripted
_SPEC.loader.exec_module(scripted)

ORDER = ("reach", "carry", "over_target", "held_on_target", "placed")
VERTICAL_LOWERING = tuple(
  step
  for phase in scripted.PHASES
  for step in (
    (("lower", "retreat", True, 30), ("lower_down", "place", True, 40))
    if phase[0] == "lower"
    else (phase,)
  )
)
TASK_TERMS = ("reach", *GATED_TERMS)
CTP = pp.EFFORT_BOUNDS["CTP"]
REWARD = S92["REWARD"]


def _wc_robot():
  return kangaroo_grippers_left_arm(
    "G20", KANGAROO_GRIPPERS_LEFT_ARM_WC, arm_effort_scale=CTP.effort_scale
  )


#: Red cube inside W2, base 15 cm away to the right and forward, outside S9.2's T11W draw (the
#: scripted planner finds no contact-free place pose for bases much farther out, e.g. (0.52, 0.08)
#: from (0.45, 0.22): a planner limit, S7-8 step 0).
SPAWN = ((0.42, 0.24), 0.0)
BASE = ((0.48, 0.10), 0.0)


@pytest.fixture(scope="module")
def wide_stack_episode():
  """One scripted stack (no lock) on a base drawn far across WC, every task term recorded at every
  step for env 0, weighted."""
  rec = collections.defaultdict(list)
  terms: dict = {}

  def hook(phase, env):
    if not terms:
      terms.update(
        {
          n: c
          for n, c in zip(
            env.reward_manager.active_terms,
            env.reward_manager._term_cfgs,
            strict=True,
          )
          if n in TASK_TERMS
        }
      )
    rec[phase].append(
      {n: float(c.weight * c.func(env, **c.params)[0]) for n, c in terms.items()}
    )

  rows = scripted.rollout(
    device=get_test_device(),
    verbose=False,
    robot=_wc_robot(),
    baseline=pp.BASELINES["B"],
    spawn_box=WC,
    target_area=WC,
    dr="DR-scene",
    effort=CTP,
    latency=tt.LATENCY_MODELS["JS"],
    stack_speed=True,
    episode_length_s=70.0,
    goal="stack",
    num_envs=1,
    reward=REWARD,
    spawns=[SPAWN],
    base_spawns=[BASE],
    hook=hook,
    phases=VERTICAL_LOWERING,
  )
  assert rows[0]["placed_and_lifted"], rows[0]
  return rec


def test_the_reward_rises_strictly_along_a_stack_across_wc(wide_stack_episode):
  rec = wide_stack_episode
  total = lambda step: sum(step.values())  # noqa: E731
  placed = [t for t in rec["open"] + rec["retreat"] if t["placed"] > 0]
  assert placed, "the cube never counted as placed on the base"
  totals = {
    "reach": total(rec["descend"][-1]),
    "carry": total(rec["lift"][-1]),
    "over_target": total(rec["steady"][-1]),
    "held_on_target": total(rec["hold"][-1]),
    "placed": total(placed[-1]),
  }
  values = [totals[name] for name in ORDER]
  assert all(a < b for a, b in zip(values, values[1:], strict=False)), totals


# -- The eval's park pose and return contact ----------------------------------------------------


def test_the_park_pose_ends_the_route_with_the_elbow_bent(monkeypatch):
  cfg = _cfg(
    monkeypatch, "WST-R", num_envs=2, HANDOFF_ROUTE="up-out", HANDOFF_PARK="elbow"
  )
  assert cfg.actions["joint_pos"].arms[0].park == mdp.HANDOFF_PARK_POSES["elbow"]
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  try:
    env.reset()
    term = env.action_manager.get_term("joint_pos")
    term.set_home("left", torch.arange(2, device=env.device))
    _hold(env, 150)
    robot = env.scene["robot"]
    names = list(robot.joint_names)
    q = robot.data.joint_pos
    home = robot.data.default_joint_pos
    for i in range(1, 8):
      j = names.index(f"arm_left_{i}_joint")
      want = 0.2 if i == 4 else 0.0
      assert torch.allclose(
        q[:, j] - home[:, j], torch.full_like(q[:, j], want), atol=0.03
      ), i
  finally:
    env.close()
  with pytest.raises(ValueError, match="HANDOFF_PARK"):
    _cfg(monkeypatch, "WST-R", HANDOFF_PARK="elbow")


def test_the_return_contact_sees_a_cube_in_the_hand(wide_env):
  spec = importlib.util.spec_from_file_location(
    "eval_pick_place_s93",
    pathlib.Path(__file__).resolve().parents[1] / "scripts" / "eval_pick_place.py",
  )
  ev = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(ev)
  env = wide_env
  with torch.inference_mode():
    env.reset()
  _hold(env, 2)
  contacts = ev._CubeHandContacts(env)
  assert contacts.ok
  assert not bool(contacts.touching(env).any())
  # Positive control: the red cube put where the left fingers are, in half the envs.
  robot = env.scene["robot"]
  tip = robot.body_names.index("gripper_left_fingertip_left_link")
  obj = env.scene["object"]
  pose = obj.data.root_link_pose_w.clone()
  half = torch.arange(env.num_envs, device=env.device) % 2 == 0
  pose[half, :3] = robot.data.body_link_pos_w[half, tip]
  obj.write_root_link_pose_to_sim(pose)
  # Read the contact list of a forward pass, before a step pushes the overlapping cube out.
  env.sim.forward()
  hit = contacts.touching(env)
  assert hit[half].float().mean() > 0.9 and not bool(hit[~half].any()), hit


# -- Decision 0005: mujoco_warp box_box's bogus deep contacts --------------------------------------


def _box_box_patched() -> bool:
  import mujoco_warp._src.collision_primitive as cp

  return "Local patch (oppsie, 2026-10-08)" in pathlib.Path(cp.__file__).read_text()


@pytest.mark.parametrize("yaw_deg", [39.1, 39.9, 30.0])
def test_box_box_gives_no_contact_deeper_than_the_boxes(yaw_deg):
  """A 5 cm cube resting flat 0.1 mm into a table box: mujoco_warp 3.11's box_box adds contacts 45-615 mm deep at
  39.1 / 39.9 deg (CPU MuJoCo gives the 4 at 0.1 mm). With `mujoco_warp_box_box_depth.patch` (decision 0005) none
  is deeper than the two boxes' smallest half-sizes together; without it the test is an expected failure."""
  import mujoco
  import mujoco_warp as mjw
  import numpy as np
  import warp as wp

  xml = """<mujoco><option><flag nativeccd="disable"/></option><worldbody>
    <body name="table" pos="0 0 0" mocap="true"><geom name="top" type="box" size="0.05 0.05 0.01"/></body>
    <body name="cube" pos="0 0 1"><freejoint/><geom type="box" size="0.025 0.025 0.025" mass="0.05"/></body>
    </worldbody></mujoco>"""
  m = mujoco.MjModel.from_xml_string(xml)
  d = mujoco.MjData(m)
  a = math.radians(yaw_deg)
  d.qpos[:7] = (
    0.0,
    0.0,
    0.01 + 0.025 - 1e-4,
    math.cos(a / 2),
    0.0,
    0.0,
    math.sin(a / 2),
  )
  with wp.ScopedDevice(get_test_device()):
    mw = mjw.put_model(m)
    dw = mjw.put_data(m, d, nconmax=64, njmax=256)
    mjw.forward(mw, dw)
    n = int(dw.nacon.numpy()[0])
    dist = dw.contact.dist.numpy()[:n]
  mujoco.mj_forward(m, d)
  cpu = np.array([d.contact.dist[j] for j in range(d.ncon)])
  if not _box_box_patched() and dist.min() < -0.035:
    pytest.xfail(
      "mujoco_warp_box_box_depth.patch not applied (decision 0005); bogus contact %.3f m"
      % dist.min()
    )
  assert dist.min() >= -0.035, dist
  assert len(cpu) == 4 and np.allclose(
    np.sort(dist[dist < 0.0]), np.sort(cpu), atol=1e-5
  ), (dist, cpu)
