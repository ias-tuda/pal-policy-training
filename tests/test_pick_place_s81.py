"""Plan S8.1 (`pick-place-v1-objects-plan.md`, *the ball on the wide area WC, on the robot cell's table, in
the deploy contract*): DEP1's BALL-c (``SHAPE=ball-dr`` on ``BS2``) moved to WC (``SPAWN=WC TARGET=WC``), the
0.837 m table (``TABLE_DZ=22``), the wide jaw servo (``LAG=FP+JSW``) and 15 s (30 s for WB-L), and the
reward module RO (`mdp.open_after_release`, WB-RO only).

What is checked, on CPU: the S8.1 cfg differs from DEP1 BALL-c's only where the plan says (step 0 item 1);
the five codes differ only in the episode and RO; in an env the ball starts on the raised top inside WC,
the goal at least 10 cm away at the surface + the drawn radius, J's threshold follows the radius, the
actor is 38-D with the yaw slot the constant (1, 0) and the critic 41-D; a ball slid onto the goal earns
nothing past reach (L4); RO is zero before the place and while held and orders placed-and-half-open <
placed-and-open; the reward rises strictly along a scripted pick-and-place across WC (L5).
"""

from __future__ import annotations

import collections
import importlib.util
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
from pal_mjlab.tasks.manipulation.mdp.shapes import resolve
from pal_mjlab.tasks.manipulation.pick_place import pick_place_env_cfg as pp
from pal_mjlab.tasks.manipulation.pick_place import rewards as rw
from pal_mjlab.tasks.manipulation.pick_place.kangaroo.env_cfg import (
  kangaroo_pick_place_env_cfg,
)
from pal_mjlab.tasks.manipulation.robots import (
  KANGAROO_GRIPPERS_LEFT_ARM_WC,
  kangaroo_grippers_left_arm,
)

RCTP = f"{rw.S52_BASE}+M1+M2+M3+M6+J+M8"
#: DEP1 BALL-c's selectors (IAS 161291-161292, `siwei-dep1` @ 648c0c9; its run name and `submit_dep1.sh`).
BALL_C = {
  "REWARD": RCTP,
  "DR": "DR-scene",
  "PPO": "PPO-stack",
  "SLEW": "stack",
  "BASELINE": "B",
  "SPAWN": "W2",
  "GAINS": "G20",
  "EFFORT": "CTP",
  "LAG": "FP+JS",
  "ACTION": "abs",
  "CMD_OBS": "cmd",
  "EPISODE_S": "10",
  "CRITIC_OBS": "q",
  "ARM_TABLE": "AH10",
  "TOOL_SPEED": "none",
  "SHAPE": "ball-dr",
  "GOAL": "table",
  "GATE": "none",
}
#: Plan S8.1's base: WC, the raised table, the wide servo, 15 s.
S81 = {
  "SPAWN": "WC",
  "TARGET": "WC",
  "TABLE_DZ": "22",
  "LAG": "FP+JSW",
  "EPISODE_S": "15",
}
CODES = {
  "WB-R": {},
  "WB-JW": {},
  "WB-S": {},
  "WB-L": {"EPISODE_S": "30"},
  "WB-RO": {"REWARD": f"{RCTP}+RO"},
}
_ALL = {
  *BALL_C,
  *S81,
  "BASE_OBS",
  "CONTACT",
  "ARMS",
  "HANDOFF_TRAIN",
  "START",
  "HANDOFF_ROUTE",
  "HANDOFF_PARK",
  "OCC",
  "OBJ_OBS",
  "CKPT_SPAWN",
  "BOX_OBS",
  "TRACKER",
  "ARM_HOME",
}
WC = pp.SPAWN_BOXES["WC"]
TABLE_CFG = SceneEntityCfg("table")


def _set_env(mp, **values) -> None:
  for var in _ALL:
    mp.delenv(var, raising=False)
  for var, value in values.items():
    if value is not None:
      mp.setenv(var, value)


def _cfg(mp, code: str | None = "WB-S", num_envs: int = 4, **changes):
  _set_env(
    mp, **{**BALL_C, **({} if code is None else {**S81, **CODES[code]}), **changes}
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


def test_s81_changes_only_the_area_the_table_the_servo_and_the_episode(monkeypatch):
  old, new = _cfg(monkeypatch, None), _cfg(monkeypatch, "WB-S")
  assert _summary(old) == _summary(new)
  assert (old.episode_length_s, new.episode_length_s) == (10.0, 15.0)
  a_old, a_new = old.actions["joint_pos"], new.actions["joint_pos"]
  assert (a_old.servo_tau, a_new.servo_tau) == (
    tt.JAW_SERVO_TAU,
    tt.JAW_SERVO_TAU_WIDE,
  )
  assert a_old.scale != a_new.scale  # the WC arm: arm 1 and arm 4 wider
  dz = (
    new.scene.entities["table"].init_state.pos[2]
    - old.scene.entities["table"].init_state.pos[2]
  )
  assert dz == pytest.approx(0.022)
  red = new.events["reset_objects"].params
  cx, cy = scene.TABLE_POS[0], scene.TABLE_POS[1]
  assert red["pose_range"]["x"] == pytest.approx((WC.x[0] - cx, WC.x[1] - cx))
  assert red["pose_range"]["y"] == pytest.approx((WC.y[0] - cy, WC.y[1] - cy))
  assert red["goal_area"]["distance"] == pp.TARGET_MIN_SEPARATION == 0.10
  # J's threshold is the ball's, per radius (S8 step 0), on every J term.
  for name in ("release", "placed", "retreat", "pad_contact"):
    assert new.rewards[name].params["jaw_open"] == rw.BALL_JAW_OPEN_BY_RADIUS, name


def test_the_codes_differ_only_in_the_episode_and_ro(monkeypatch):
  base = _summary(_cfg(monkeypatch, "WB-S"))
  for code in ("WB-R", "WB-JW"):
    assert _summary(_cfg(monkeypatch, code)) == base, code
  wbl = _cfg(monkeypatch, "WB-L")
  assert _summary(wbl) == base and wbl.episode_length_s == 30.0
  ro = _summary(_cfg(monkeypatch, "WB-RO"))
  extra = {k: v for k, v in ro["rewards"].items() if k not in base["rewards"]}
  assert extra == {"open_after_release": ("open_after_release", rw.RO_WEIGHT)}
  assert {**ro, "rewards": base["rewards"]} == base


def test_ro_needs_j_and_m8_before_it():
  with pytest.raises(ValueError, match="RO needs J and M8"):
    rw._s52_arm(("M1", "M2", "M3", "M6", "J", "RO"))


# -- The env -----------------------------------------------------------------------------------


def _hold(env, steps: int) -> None:
  with torch.inference_mode():
    for _ in range(steps):
      env.step(torch.zeros(env.action_space.shape, device=env.device))


def _term(env, name: str) -> torch.Tensor:
  cfg = env.reward_manager.get_term_cfg(name)
  return cfg.func(env, **cfg.params)


@pytest.fixture(scope="module")
def ball_env():
  mp = pytest.MonkeyPatch()
  cfg = _cfg(mp, "WB-RO", num_envs=96)
  cfg.observations["actor"].enable_corruption = False
  cfg.seed = 5
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  env.reset()
  yield env
  env.close()
  mp.undo()


def test_the_ball_starts_in_wc_on_the_raised_top_and_the_goal_is_10cm_away(ball_env):
  env = ball_env
  cx, cy = scene.TABLE_POS[0], scene.TABLE_POS[1]
  radii, dists = [], []
  for episode in range(2):
    if episode:
      with torch.inference_mode():
        env.reset()
    _hold(env, 10)
    centre = mdp.events._surface_centre_xy(env, TABLE_CFG)
    top = mdp.surface_top_z(env, TABLE_CFG)
    data = env.scene["object"].data
    rel = data.root_link_pos_w[:, :2] - centre
    # A ball can roll a few mm in 10 steps of the arm settling; allow it.
    assert (rel[:, 0] >= WC.x[0] - cx - 5e-3).all()
    assert (rel[:, 0] <= WC.x[1] - cx + 5e-3).all()
    assert (rel[:, 1] >= WC.y[0] - cy - 5e-3).all()
    assert (rel[:, 1] <= WC.y[1] - cy + 5e-3).all()
    r = mdp.object_half_height(env, "object")
    assert ((data.root_link_pos_w[:, 2] - r - top).abs() < 3e-3).all()
    # The table is the raised one: its top ~0.837 m (DR-scene's +/-1 cm on top).
    assert (top > 0.837 - 0.012).all() and (top < 0.837 + 0.012).all(), top
    goal = env.command_manager.get_term("goal_pose").command
    assert ((goal[:, 2] - top - r).abs() < 1e-3).all()
    d = torch.norm(goal[:, :2] - data.root_link_pos_w[:, :2], dim=-1)
    assert d.min() >= pp.TARGET_MIN_SEPARATION - 6e-3, d.min()
    radii.append(r)
    dists.append(d)
    # J's threshold follows the radius: larger ball, smaller threshold.
    thr = resolve(env, rw.BALL_JAW_OPEN_BY_RADIUS)
    order = torch.argsort(r)
    assert torch.all(thr[order][1:] <= thr[order][:-1] + 1e-6)
  radii = torch.cat(radii)
  assert radii.min() < 0.0285 and radii.max() > 0.0315, (radii.min(), radii.max())
  obs = env.observation_manager.compute()
  assert obs["actor"].shape[-1] == 38 and obs["critic"].shape[-1] == 41
  assert torch.isfinite(obs["actor"]).all() and torch.isfinite(obs["critic"]).all()


def test_a_ball_slid_onto_the_goal_earns_nothing_past_reach(ball_env):
  env = ball_env
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
  assert not bool(getattr(env, _WAS_LIFTED).any()), "the ball was never lifted"
  assert torch.all(_term(env, "reach") > 0.0)
  for name in (
    "grasp",
    "lift",
    "goal",
    "goal_near",
    "release",
    "placed",
    "retreat",
    "pad_contact",
    "open_after_release",
  ):
    if name in env.reward_manager.active_terms:
      value = _term(env, name)
      assert torch.all(value == 0.0), f"{name} paid a slide: {value}"


def test_ro_pays_by_how_far_past_the_threshold_the_jaw_has_opened(ball_env):
  """With the ball resting on the goal and the was-lifted latch set (as after a J-release), RO is
  0 at J's threshold for that radius, ~0.5 halfway to fully open, ~1 fully open; and 0 again with the
  latch cleared (not placed)."""
  env = ball_env
  with torch.inference_mode():
    env.reset()
  _hold(env, 2)
  obj = env.scene["object"]
  goal = env.command_manager.get_term("goal_pose").command
  pose = obj.data.root_link_pose_w.clone()
  pose[:, :3] = goal
  obj.write_root_link_pose_to_sim(pose)
  obj.write_root_link_velocity_to_sim(torch.zeros(env.num_envs, 6, device=env.device))
  _hold(env, 1)
  was_lifted = getattr(env, _WAS_LIFTED)
  saved = was_lifted.clone()
  robot = env.scene["robot"]
  jaw = env.reward_manager.get_term_cfg("open_after_release").params["jaw_cfg"]
  jid = jaw.joint_ids
  q0 = robot.data.joint_pos.clone()
  thr = resolve(env, rw.BALL_JAW_OPEN_BY_RADIUS)
  values = {}
  try:
    was_lifted[:] = True
    for label, frac in (
      ("closed", 1.2),
      ("threshold", 1.0),
      ("half", 0.5),
      ("open", 0.0),
    ):
      q = q0[:, jid].clone()
      q[:] = (thr * frac).unsqueeze(-1)
      # qpos only, no forward pass: RO reads the jaw angle; the contacts stay the last step's.
      robot.write_joint_position_to_sim(q, joint_ids=jid)
      values[label] = _term(env, "open_after_release")
    was_lifted[:] = False
    values["not_lifted"] = _term(env, "open_after_release")
  finally:
    was_lifted[:] = saved
    robot.write_joint_position_to_sim(q0)
  # Envs whose ball sits still on the goal and is not touched by the hand (home pose: always).
  ok = values["open"] > 0
  assert ok.float().mean() > 0.9, ok.float().mean()
  assert torch.all(values["closed"] == 0.0) and torch.all(values["threshold"] == 0.0)
  assert torch.allclose(values["half"][ok], torch.full_like(values["half"][ok], 0.5))
  assert torch.allclose(values["open"][ok], torch.ones_like(values["open"][ok]))
  assert torch.all(values["not_lifted"] == 0.0)


def test_ro_is_smaller_than_a_full_retreat_and_than_placed():
  assert rw.RO_WEIGHT < rw.M8_RETREAT_WEIGHT < 8.0


# -- L5 on WC with the ball ---------------------------------------------------------------------

_SCRIPT = (
  pathlib.Path(__file__).resolve().parents[1] / "scripts" / "scripted_pick_place.py"
)
_SPEC = importlib.util.spec_from_file_location("scripted_pick_place_s81", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
scripted = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = scripted
_SPEC.loader.exec_module(scripted)

ORDER = ("reach", "carry", "over_target", "held_on_target", "placed")
TASK_TERMS = (
  "reach",
  "grasp",
  "lift",
  "goal",
  "goal_near",
  "release",
  "placed",
  "retreat",
  "pad_contact",
  "open_after_release",
)
CTP = pp.EFFORT_BOUNDS["CTP"]
#: A ball carried ~14 cm to the right across WC (`targets` is the planner's, not env-frame: (0.52, 0.08)
#: put the goal at (0.59, -0.13), outside the planner's contact-free IK).
SPAWN = ((0.45, 0.22), 0.0)
TARGET = (0.46, 0.08)


@pytest.fixture(scope="module")
def wide_ball_episode():
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
    robot=kangaroo_grippers_left_arm(
      "G20", KANGAROO_GRIPPERS_LEFT_ARM_WC, arm_effort_scale=CTP.effort_scale
    ),
    baseline=pp.BASELINES["B"],
    spawn_box=WC,
    target_area=WC,
    dr="DR-scene",
    effort=CTP,
    latency=tt.LATENCY_MODELS["JS"],
    stack_speed=True,
    episode_length_s=70.0,
    num_envs=1,
    reward=f"{RCTP}+RO",
    shape=pp.OBJECT_SHAPES["ball"],
    ball_offset=rw.EQ_OFFSET,
    spawns=[SPAWN],
    targets=[TARGET],
    hook=hook,
  )
  assert rows[0]["placed_and_lifted"], rows[0]
  return rec


def test_the_reward_rises_strictly_along_a_ball_pick_and_place_across_wc(
  wide_ball_episode,
):
  rec = wide_ball_episode
  total = lambda step: sum(step.values())  # noqa: E731
  after = [t for p in ("open", "retreat") for t in rec[p]]
  placed = [t for t in after if t["placed"] > 0]
  assert placed, f"the ball never counted as placed; phases {list(rec)}"
  totals = {
    "reach": total(rec["descend"][-1]),
    "carry": total(rec["lift"][-1]),
    "over_target": total(rec["steady"][-1]),
    "held_on_target": total(rec["hold"][-1]),
    "placed": total(placed[-1]),
  }
  values = [totals[name] for name in ORDER]
  assert all(a < b for a, b in zip(values, values[1:], strict=False)), totals
  # RO pays only once placed, and placed-open-lifted earns more than placed-open.
  before = [t for p in ("descend", "lift", "steady", "hold") for t in rec[p]]
  assert all(t["open_after_release"] == 0.0 for t in before)
  assert max(t["open_after_release"] for t in placed) > 0.5
