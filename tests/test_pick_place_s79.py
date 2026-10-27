"""Plan S7-9 (designed 2026-10-01 13:20, Siwei: "可以先在桌面任务解决"): placing precision at the
stack speed, on the table task. Four codes on S7-8's ABS-a, each resumed from ABS-a's 3 k checkpoint
against the unchanged control P0:

* PQ  -- ``CRITIC_OBS=q``: the critic also observes M6's latched grade and the release latch; the
  actor is unchanged, and a resume widens the loaded critic with zero weights for the two inputs;
* PXY -- ``REWARD=...+PXY``: M6's xy width 0.02 -> 0.01 m, M2's ``goal_near`` 0.03 -> 0.01 m;
* PT  -- ``REWARD=...+PT``: M6's graded part x G_10deg(tilt) x G_0.05(speed) at the let-go;
* PV  -- ``TOOL_SPEED=PV``: 0.05 m/s while the held cube is within 5 cm of the target
  horizontally, no other tool bound.

What is checked here: unset is S7-8's ABS-a exactly; each code changes only what it names (so an
ABS-a checkpoint resumes under every one of them); the PXY lowering path stays monotone (L5); the
latch records the tilt at the pads' let-go and PT's grade uses it; the critic's two inputs are the
grade the reward pays; the critic widening preserves the loaded value function and its Adam
moments; PV's slow zone is the goal's only.
"""

import dataclasses
import math

import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from pal_mjlab.tasks.manipulation import mdp
from pal_mjlab.tasks.manipulation import tabletop_env_cfg as tt
from pal_mjlab.tasks.manipulation.mdp import observations as obs_mod
from pal_mjlab.tasks.manipulation.mdp.rewards import release_quality_of
from pal_mjlab.tasks.manipulation.pick_place import pick_place_env_cfg as pp
from pal_mjlab.tasks.manipulation.pick_place import rewards as rw
from pal_mjlab.tasks.manipulation.pick_place.kangaroo import env_cfg as kenv
from pal_mjlab.tasks.manipulation.robots import (
  KANGAROO_GRIPPERS_LEFT_ARM_WIDE,
  kangaroo_grippers_left_arm,
)
from pal_mjlab.tasks.manipulation.runner import widen_critic_input
from rsl_rl.models import MLPModel
from tensordict import TensorDict

RCTP = f"{rw.S52_BASE}+M1+M2+M3+M6+J+M8"
PXY = f"{RCTP}+PXY"
PT = f"{RCTP}+PT"
CTP = pp.EFFORT_BOUNDS["CTP"]
#: S7-8's ABS-a at plan S7-9's 10 s, as `results/S7-9/submit_s79.sh` exports it.
ABS_A_ENV = {
  "REWARD": RCTP,
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
  "TARGET": "B",
  "EPISODE_S": "10",
}
UNSET = ("CMD_OBS", "SHAPE", "OCC", "OBJ_OBS", "OCC_FLAG", "CRITIC_OBS")
FIELDS = ("actions", "observations", "commands", "events", "terminations", "rewards")


def _env_vars(monkeypatch, **changes):
  for var, value in {**ABS_A_ENV, **changes}.items():
    if value is None:
      monkeypatch.delenv(var, raising=False)
    else:
      monkeypatch.setenv(var, value)
  for var in UNSET:
    if var not in changes:
      monkeypatch.delenv(var, raising=False)


def _cfg(reward=RCTP, tool="none", critic=False, num_envs=None):
  """S7-8's ABS-a (RCTP, G20, B, W2, DR-scene, CTP, FP+JS, stack speeds, 10 s) with one S7-9 code."""
  cfg = pp.make_pick_place_env_cfg(
    kangaroo_grippers_left_arm(
      "G20", KANGAROO_GRIPPERS_LEFT_ARM_WIDE, arm_effort_scale=CTP.effort_scale
    ),
    gripper_name="left",
    reward=reward,
    baseline=pp.BASELINES["B"],
    spawn=pp.SPAWN_BOXES["W2"],
    dr="DR-scene",
    effort=CTP,
    latency=tt.LATENCY_MODELS["FP+JS"],
    episode_length_s=10.0,
    stack_speed=True,
    tool_speed_cap=pp.TOOL_SPEED_CAPS[tool],
    critic_release=critic,
  )
  if num_envs is not None:
    cfg.scene.num_envs = num_envs
  return cfg


def _diff(a, b) -> set[str]:
  return {f for f in FIELDS if getattr(a, f) != getattr(b, f)}


# -- Configuration --------------------------------------------------------------------------


def test_unset_is_s78_abs_a(monkeypatch):
  _env_vars(monkeypatch)
  got = kenv.kangaroo_pick_place_env_cfg()
  assert not _diff(got, _cfg())
  assert got.episode_length_s == 10.0
  assert not kenv.kangaroo_pick_place_rl_cfg().widen_critic_on_resume
  _env_vars(monkeypatch, CRITIC_OBS="base")
  assert not _diff(kenv.kangaroo_pick_place_env_cfg(), got)


def test_the_selectors_are_read_and_validated(monkeypatch):
  _env_vars(monkeypatch, CRITIC_OBS="q", REWARD=PT, TOOL_SPEED="PV")
  cfg = kenv.kangaroo_pick_place_env_cfg()
  assert not _diff(cfg, _cfg(PT, "PV", critic=True))
  assert kenv.kangaroo_pick_place_rl_cfg().widen_critic_on_resume
  _env_vars(monkeypatch, CRITIC_OBS="value")
  with pytest.raises(ValueError, match="CRITIC_OBS"):
    kenv.kangaroo_pick_place_env_cfg()
  # PQ shows M6's grade: an arm without it is refused.
  _env_vars(monkeypatch, CRITIC_OBS="q", REWARD=f"{rw.S52_BASE}+M1+M2+M3")
  with pytest.raises(ValueError, match="M6"):
    kenv.kangaroo_pick_place_env_cfg()


@pytest.mark.parametrize(
  "code",
  [
    (PXY, "none", False),
    (PT, "none", False),
    (RCTP, "PV", False),
    (RCTP, "none", True),
  ],
)
def test_each_code_changes_only_its_own_part(code):
  """So an ABS-a checkpoint resumes under every code: none touches the actor or the action's
  shape. PQ adds two critic terms, PXY / PT change reward parameters, PV the action term's class."""
  base, cfg = _cfg(), _cfg(*code)
  want = {
    PXY: {"rewards"},
    PT: {"rewards"},
  }.get(code[0], {"actions"} if code[1] == "PV" else {"observations"})
  assert _diff(cfg, base) == want
  assert cfg.observations["actor"] == base.observations["actor"]


def test_pxy_and_pt_change_only_the_grade_widths():
  base, pxy, pt = (_cfg(r).rewards for r in (RCTP, PXY, PT))
  assert set(base) == set(pxy) == set(pt)
  for name in base:
    b, x, t = base[name], pxy[name], pt[name]
    assert b.func is x.func is t.func and b.weight == x.weight == t.weight, name
    if name in ("release", "placed"):
      assert x.params == {**b.params, "dist_std": 0.01}
      assert t.params == {**b.params, "tilt_std": math.radians(10), "speed_std": 0.05}
    elif name == "goal_near":
      assert x.params == {**b.params, "std": 0.01} and t.params == b.params
    else:
      assert b.params == x.params == t.params, name


def test_the_modules_need_m6_and_m2_before_them():
  for mods, match in (
    (("M1", "M2", "M3", "PXY"), "PXY"),
    (("M1", "M3", "M6", "PXY"), "PXY"),
    (("M1", "M2", "M3", "PT", "M6"), "PT"),
  ):
    with pytest.raises(ValueError, match=match):
      rw._s52_arm(mods)


def test_the_pxy_lowering_path_stays_monotone():
  """L5 for PXY: straight down over the target, `goal` (7, sigma 0.15) + `goal_near` (3, sigma
  0.01) + the faded lift (2 x min(h / 0.1, 1) x (1 - G_0.05(h))) must fall with the height -- M2's
  weight 7 was chosen against a dip at goal_near's 0.03; at 0.01 it stays monotone, nearly flat
  between 3 and 5 cm (the 0.001 steps below)."""

  def f(h, near):
    return (
      rw.M2_GOAL_WEIGHT * math.exp(-((h / rw.BRINGING_STD) ** 2))
      + rw.M2_NEAR_WEIGHT * math.exp(-((h / near) ** 2))
      + 2.0
      * min(h / rw.LIFT_HEIGHT, 1.0)
      * (1 - math.exp(-((h / rw.M3_FADE_STD) ** 2)))
    )

  for near in (rw.M2_NEAR_STD, rw.PXY_NEAR_STD):
    path = [f(h / 1000, near) for h in range(0, 101)]
    assert all(a > b for a, b in zip(path, path[1:], strict=False)), near


def test_the_pt_grade_is_m6_times_tilt_and_speed_above_the_floor():
  dxy, h = torch.tensor([0.0, 0.01, 0.0, 0.0]), torch.tensor([0.0, 0.0, 0.02, 0.0])
  tilt = torch.tensor([0.0, math.radians(10), math.radians(22), 0.0])
  speed = torch.tensor([0.0, 0.05, 0.11, 0.0])
  m6 = rw.M6_DIST_STD, rw.M6_HEIGHT_FREE, rw.M6_HEIGHT_STD, rw.M6_FLOOR
  q0 = release_quality_of(dxy, h, *m6)
  q = release_quality_of(dxy, h, *m6, tilt, rw.PT_TILT_STD, speed, rw.PT_SPEED_STD)
  g = torch.exp(-((tilt / rw.PT_TILT_STD) ** 2) - (speed / rw.PT_SPEED_STD) ** 2)
  assert torch.allclose(q, rw.M6_FLOOR + (q0 - rw.M6_FLOOR) * g)
  assert float(q[0]) == pytest.approx(1.0) and float(q[3]) == pytest.approx(1.0)
  # S7-8's median let-go (22 deg, 0.11 m/s) is graded at the floor; one at the bounds, e^-2.
  assert float(q[2]) == pytest.approx(rw.M6_FLOOR, abs=1e-3)
  assert float(q[1] - rw.M6_FLOOR) == pytest.approx(
    float(q0[1] - rw.M6_FLOOR) * math.exp(-2), rel=1e-5
  )


def test_pq_appends_the_placed_terms_grade_to_the_critic_only():
  for reward in (RCTP, PT):
    base, cfg = _cfg(reward), _cfg(reward, critic=True)
    got, was = cfg.observations["critic"].terms, base.observations["critic"].terms
    assert list(got) == [*was, "release_grade", "release_latched"]
    assert all(got[n] == was[n] for n in was)
    placed = cfg.rewards["placed"].params
    for name in ("release_grade", "release_latched"):
      params = got[name].params
      assert params == {k: v for k, v in placed.items() if k in params}
      assert "dist_std" in params and "threshold" not in params
    assert ("tilt_std" in got["release_grade"].params) == (reward == PT)


def test_pv_is_the_goal_slow_zone_only_with_no_carry_cap():
  cap = pp.TOOL_SPEED_CAPS["PV"]
  assert (cap.speed, cap.slow_speed, cap.slow_zone) == (math.inf, 0.05, 0.05)
  assert not cap.slow_near_surface and pp.TOOL_SPEED_CAPS["V"].slow_near_surface
  base, act = _cfg().actions["joint_pos"], _cfg(tool="PV").actions["joint_pos"]
  assert type(act).__name__ == "ToolSpeedCappedJointPositionActionCfg"
  shared = {f.name for f in dataclasses.fields(base)}
  assert {k: getattr(act, k) for k in shared} == {k: getattr(base, k) for k in shared}
  assert act.speed_limit == math.inf and act.slow_speed_limit == 0.05
  assert act.slow_near_surface is False


# -- In the simulator -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def env():
  e = ManagerBasedRlEnv(
    cfg=_cfg(PT, "PV", critic=True, num_envs=2), device=get_test_device()
  )
  e.reset()
  yield e
  e.close()


def _put_cube(env, xy_off, z_off, tilt):
  """The cube at the goal + (xy_off, z_off) in every env, tilted ``tilt`` about world x, at rest."""
  obj = env.scene[pp.OBJECT]
  goal = env.command_manager.get_term(tt.COMMAND).command
  pose = obj.data.root_link_pose_w.clone()
  pose[:, 0] = goal[:, 0] + xy_off[0]
  pose[:, 1] = goal[:, 1] + xy_off[1]
  pose[:, 2] = goal[:, 2] + z_off
  pose[:, 3:] = torch.tensor(
    [math.cos(tilt / 2), math.sin(tilt / 2), 0.0, 0.0], device=env.device
  )
  ids = torch.arange(env.num_envs, device=env.device)
  obj.write_root_link_pose_to_sim(pose, env_ids=ids)
  obj.write_root_link_velocity_to_sim(
    torch.zeros(env.num_envs, 6, device=env.device), env_ids=ids
  )
  env.sim.forward()


def test_the_tilt_is_the_angle_to_the_nearest_face_normal(env):
  for deg, want in ((0, 0), (20, 20), (45, 45), (70, 20), (90, 0)):
    _put_cube(env, (0.0, 0.0), 0.05, math.radians(deg))
    got = obs_mod.object_tilt(env, pp.OBJECT)
    assert torch.allclose(got, torch.full_like(got, math.radians(want)), atol=1e-4), deg


def test_the_latch_takes_the_tilt_at_the_pads_let_go_and_the_critic_sees_the_grade(
  env, monkeypatch
):
  """Held, then the pads let go tilted 20 deg, then the jaw opens with the cube flat again: the
  release is the jaw opening (J), graded with the let-go's tilt; the critic's two inputs are that
  grade and 1, and the reset clears both."""
  state = {"grasped": True, "jaw_open": False}
  ones = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
  monkeypatch.setattr(obs_mod, "_grasped_bool", lambda e, s: ones & state["grasped"])
  monkeypatch.setattr(obs_mod, "_was_lifted_bool", lambda e, g, s: ones)
  monkeypatch.setattr(
    obs_mod, "_jaw_open_bool", lambda e, c, o: ones & state["jaw_open"]
  )
  terms = env.observation_manager._group_obs_term_cfgs["critic"]
  names = env.observation_manager._group_obs_term_names["critic"]
  grade_cfg = terms[names.index("release_grade")]
  placed = env.reward_manager.get_term_cfg("placed").params

  def read(step):
    env.common_step_counter = step
    return (
      float(mdp.release_grade(env, **grade_cfg.params)[0, 0]),
      float(mdp.release_latched(env, **grade_cfg.params)[0, 0]),
    )

  mdp.reset_release_latch(env, torch.arange(env.num_envs, device=env.device))
  base = env.common_step_counter + 10
  _put_cube(env, (0.01, 0.0), 0.0, 0.0)
  assert read(base) == (pytest.approx(rw.M6_FLOOR), 0.0)  # held
  state["grasped"] = False
  _put_cube(env, (0.01, 0.0), 0.0, math.radians(20))
  assert read(base + 1) == (
    pytest.approx(rw.M6_FLOOR),
    0.0,
  )  # the pads let go, jaw closed
  state["jaw_open"] = True
  _put_cube(env, (0.0, 0.0), 0.0, 0.0)
  q, released = read(base + 2)  # the jaw opens: committed with the let-go's numbers
  latch = getattr(env, obs_mod._RELEASE)
  assert released == 1.0 and bool(latch.released[0])
  assert float(latch.tilt[0]) == pytest.approx(math.radians(20), abs=1e-4)
  assert float(latch.dxy[0]) == pytest.approx(0.01, abs=1e-4)
  want = release_quality_of(
    latch.dxy, latch.height, placed["dist_std"], placed["height_free"],
    placed["height_std"], placed["floor"], latch.tilt, placed["tilt_std"],
    latch.rel_speed, placed["speed_std"],
  )  # fmt: skip
  assert q == pytest.approx(float(want[0]))
  assert rw.M6_FLOOR < q < rw.M6_FLOOR + 0.75 * math.exp(-4) * math.exp(-0.25) + 1e-3
  # The critic term reads the same latch the reward does.
  assert q == pytest.approx(float(mdp.release_quality(env, **grade_cfg.params)[0]))
  mdp.reset_release_latch(env, torch.tensor([0], device=env.device))
  assert float(latch.tilt[0]) == 0.0 and not bool(latch.released[0])
  assert float(latch.tilt[1]) > 0.3
  env.common_step_counter = base + 3


def test_pv_slows_only_a_held_cube_near_the_goal(env, monkeypatch):
  term = env.action_manager.get_term("joint_pos")
  held = {"v": True}
  monkeypatch.setattr(
    obs_mod,
    "_grasped_bool",
    lambda e, s: torch.full(
      (e.num_envs,), held["v"], dtype=torch.bool, device=e.device
    ),
  )
  # 3 cm off the goal and 10 cm up: slow. 10 cm off the goal at the table: free (V would slow it).
  for off, z, want in (((0.03, 0.0), 0.10, 0.05), ((0.10, 0.0), 0.0, math.inf)):
    _put_cube(env, off, z, 0.0)
    assert torch.equal(term._speed_limits(), torch.full((2,), want, device=env.device))
  held["v"] = False
  _put_cube(env, (0.0, 0.0), 0.0, 0.0)
  assert torch.isinf(term._speed_limits()).all()


# -- The critic widening of a resume ---------------------------------------------------------------


def _critic(width: int) -> MLPModel:
  obs = TensorDict({"critic": torch.zeros(4, width)}, batch_size=[4])
  return MLPModel(obs, {"critic": ["critic"]}, "critic", 1, (16, 8), "elu", True)


def test_the_widened_critic_is_the_saved_one_and_its_moments_are_kept():
  torch.manual_seed(0)
  old, new = _critic(10), _critic(12)
  old.train()
  old.obs_normalizer.update(3.0 + 2.0 * torch.randn(256, 10))
  opt = torch.optim.Adam(old.parameters(), lr=1e-3)
  x = torch.randn(32, 10)
  old(TensorDict({"critic": x}, batch_size=[32])).pow(2).mean().backward()
  opt.step()
  loaded = {
    "critic_state_dict": {k: v.clone() for k, v in old.state_dict().items()},
    "optimizer_state_dict": opt.state_dict(),
  }
  new_opt = torch.optim.Adam(new.parameters(), lr=1e-3)
  assert widen_critic_input(loaded, new, list(new.parameters())) == 2
  new.load_state_dict(loaded["critic_state_dict"])
  new_opt.load_state_dict(loaded["optimizer_state_dict"])
  old.eval()
  new.eval()
  x_new = torch.cat([x, torch.rand(32, 2)], dim=1)  # anything in the new inputs
  with torch.no_grad():
    a = old(TensorDict({"critic": x}, batch_size=[32]))
    b = new(TensorDict({"critic": x_new}, batch_size=[32]))
  assert torch.allclose(a, b, atol=1e-6)
  first = new.mlp[0].weight
  assert torch.equal(first[:, 10:], torch.zeros(16, 2))
  st = new_opt.state[first]
  assert torch.equal(st["exp_avg"][:, 10:], torch.zeros(16, 2))
  assert torch.equal(st["exp_avg"][:, :10], opt.state[old.mlp[0].weight]["exp_avg"])
  # Already wide: nothing to do. Narrower, or another layer changed: refused.
  assert widen_critic_input(loaded, new, list(new.parameters())) == 0
  with pytest.raises(ValueError):
    widen_critic_input(
      {"critic_state_dict": new.state_dict(), "optimizer_state_dict": {"state": {}}},
      old,
      list(old.parameters()),
    )
