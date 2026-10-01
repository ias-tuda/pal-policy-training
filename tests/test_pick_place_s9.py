"""Plan S9 (Siwei, 2026-10-01 22:04-22:20): stack the cube on a second one, and let go only when aligned
and slow.

What is checked here (``pick-place-v1-objects-plan.md``, *S9 revised again*), on the baseline the plan
names -- `BS`: S7-8's ABS-a plant (G20, baseline B, CTP, DR-scene, LAG=FP+JS, the stack task's speeds,
the raw last action observed), 10 s:

* ``GOAL`` / ``GATE`` unset build S8's cfg exactly; bad names and the combinations that are not built
  (a ball, a wide target area, occlusion, the lock without the stack, RG / SA on a table goal, the
  lock on a delta action) raise;
* the base cube is spawned in B's target box at yaw 0 +/-5 deg, the cube in W2 >= 10 cm from it, the
  goal is the cube resting on the base and follows it; the actor still sees 38 numbers, its goal
  through a tracker error, the critic the live goal; the base is priced and its fall ends the episode;
* the five release quantities and ``OK`` on cubes put there by hand; the lock holds a jaw that asks
  to open while armed, and frees it otherwise;
* L4: a cube slid onto the base pays nothing past reach; RG floors a release that was not ``OK``;
* L5 and L1 (the scripted stack through the lock, wrist turned to the base): the per-step reward
  rises along the task for ST, ST-R and ST-GA, and a let-go 2 cm off the base is refused by the lock.
"""

import collections
import importlib.util
import math
import pathlib
import sys

import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz
from pal_mjlab.tasks.manipulation import mdp
from pal_mjlab.tasks.manipulation import tabletop_env_cfg as tt
from pal_mjlab.tasks.manipulation.mdp.observations import _RELEASE, _WAS_LIFTED
from pal_mjlab.tasks.manipulation.mdp.stack import OK_DXY, OK_SPEED, OK_TILT, OK_YAW
from pal_mjlab.tasks.manipulation.pick_place import pick_place_env_cfg as pp
from pal_mjlab.tasks.manipulation.pick_place import rewards as rw
from pal_mjlab.tasks.manipulation.pick_place.kangaroo import env_cfg as kenv
from pal_mjlab.tasks.manipulation.pick_place.rewards import S52_BASE
from pal_mjlab.tasks.manipulation.robots import (
  KANGAROO_GRIPPERS_LEFT_ARM_WIDE,
  kangaroo_grippers_left_arm,
)

RCTP = f"{S52_BASE}+M1+M2+M3+M6+J+M8"
RCTP_RG = f"{RCTP}+RG"
RCTP_SA = f"{RCTP}+SA"
CTP = pp.EFFORT_BOUNDS["CTP"]
#: `BS` (plan §0) = S7-8's ABS-a, as `pick_place.sbatch` exports it.
BS_ENV = {
  "REWARD": RCTP,
  "DR": "DR-scene",
  "SLEW": "stack",
  "BASELINE": "B",
  "SPAWN": "W2",
  "GAINS": "G20",
  "EFFORT": "CTP",
  "LAG": "FP+JS",
  "ACTION": "abs",
  "EPISODE_S": "10",
}


def _env_vars(monkeypatch, **changes):
  for var in ("CMD_OBS", "SHAPE", "GOAL", "GATE", "TARGET", "OCC"):
    monkeypatch.delenv(var, raising=False)
  for var, value in {**BS_ENV, **changes}.items():
    if value is None:
      monkeypatch.delenv(var, raising=False)
    else:
      monkeypatch.setenv(var, value)


def _robot():
  return kangaroo_grippers_left_arm(
    "G20", KANGAROO_GRIPPERS_LEFT_ARM_WIDE, arm_effort_scale=CTP.effort_scale
  )


def _cfg(num_envs: int = 4, reward: str = RCTP, goal="stack", gate="stack", **kw):
  cfg = pp.make_pick_place_env_cfg(
    _robot(),
    gripper_name="left",
    reward=reward,
    baseline=pp.BASELINES["B"],
    spawn=pp.SPAWN_BOXES["W2"],
    dr="DR-scene",
    effort=CTP,
    latency=tt.LATENCY_MODELS["FP+JS"],
    stack_speed=True,
    episode_length_s=10.0,
    goal=goal,
    gate=gate,
    **kw,
  )
  cfg.scene.num_envs = num_envs
  return cfg


def _summary(cfg) -> dict:
  return {
    "action": type(cfg.actions["joint_pos"]).__name__,
    "command": type(cfg.commands["goal_pose"]).__name__,
    "rewards": sorted(cfg.rewards),
    "events": list(cfg.events),
    "terminations": sorted(cfg.terminations),
    "entities": sorted(cfg.scene.entities),
    "actor": {n: repr(t.noise) for n, t in cfg.observations["actor"].terms.items()},
    "critic": sorted(cfg.observations["critic"].terms),
  }


# -- Selectors ---------------------------------------------------------------------------------


def test_goal_and_gate_unset_build_the_table_task_exactly(monkeypatch):
  _env_vars(monkeypatch)
  unset = _summary(kenv.kangaroo_pick_place_env_cfg())
  _env_vars(monkeypatch, GOAL="table", GATE="none")
  explicit = _summary(kenv.kangaroo_pick_place_env_cfg())
  assert unset == explicit
  assert unset["action"] == "RateLimitedJointPositionActionCfg"
  assert unset["command"] == "SurfaceOffsetCommandCfg"
  assert "base_object" not in unset["entities"]
  assert "reset_base_object" not in unset["events"]


def test_bad_goal_and_gate_names_raise(monkeypatch):
  for var in ("GOAL", "GATE"):
    _env_vars(monkeypatch, **{var: "tower"})
    with pytest.raises(ValueError, match=var):
      kenv.kangaroo_pick_place_env_cfg()


@pytest.mark.parametrize(
  "changes, match",
  [
    ({"GOAL": "stack", "SHAPE": "ball"}, "SHAPE must be cube"),
    ({"GOAL": "stack", "TARGET": "WC", "SPAWN": "WC"}, "TARGET must be B"),
    ({"GOAL": "stack", "OCC": "arm"}, "occlusion"),
    ({"GOAL": "stack", "BASELINE": "S71", "SPAWN": "S71"}, "target box"),
    ({"GATE": "stack"}, "needs GOAL=stack"),
    ({"REWARD": RCTP_RG}, "needs GOAL=stack"),
    ({"REWARD": RCTP_SA}, "needs GOAL=stack"),
    ({"GOAL": "stack", "GATE": "stack", "ACTION": "delta"}, "absolute"),
  ],
)
def test_combinations_that_are_not_built_raise(monkeypatch, changes, match):
  _env_vars(monkeypatch, **changes)
  with pytest.raises(ValueError, match=match):
    kenv.kangaroo_pick_place_env_cfg()


def test_the_stack_codes_build(monkeypatch):
  for reward, gate in (
    (RCTP, None),
    (RCTP, "stack"),
    (RCTP_RG, None),
    (RCTP_SA, "stack"),
  ):
    _env_vars(monkeypatch, REWARD=reward, GOAL="stack", GATE=gate)
    s = _summary(kenv.kangaroo_pick_place_env_cfg())
    assert s["command"] == "EntityOffsetCommandCfg"
    assert s["action"] == (
      "StackGateJointPositionActionCfg" if gate else "RateLimitedJointPositionActionCfg"
    )
    assert "base_disturbance" in s["rewards"] and "base_dropped" in s["terminations"]
    assert ("stack_align" in s["rewards"]) == (reward == RCTP_SA)
    # The actor's goal goes through a tracker; the critic's does not, and it has no extra term.
    assert "0.004" in s["actor"]["target_pos"] and "abs" in s["actor"]["target_pos"]
    assert len(s["actor"]) == 8 and "grasped" in s["critic"]


def test_rg_and_sa_need_m6_and_m1():
  with pytest.raises(ValueError, match="RG"):
    rw._s52_arm(("M1", "RG", "M6"))
  with pytest.raises(ValueError, match="SA"):
    rw._s52_arm(("M3", "SA"))
  assert rw.STACK_ONLY_ARMS == {RCTP_RG, RCTP_SA}
  assert {RCTP_RG, RCTP_SA} <= rw.JAW_RELEASE and {RCTP_RG, RCTP_SA} <= (
    rw.NO_SUCCESS_TERMINATION
  )


# -- The scene ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stack_env():
  env = ManagerBasedRlEnv(
    cfg=_cfg(num_envs=16, reward=RCTP_SA), device=get_test_device()
  )
  env.reset()
  zero = torch.zeros(env.action_space.shape, device=env.device)
  for _ in range(3):
    env.step(zero)
  yield env
  env.close()


def _table_centre(env):
  table = env.scene["table"]
  return env.sim.data.geom_xpos[:, table.indexing.geom_ids[0], :2]


def test_the_base_spawns_in_bs_target_box_square_and_the_goal_sits_on_it(stack_env):
  env = stack_env
  base = env.scene["base_object"].data
  rel = base.root_link_pos_w[:, :2] - _table_centre(env)
  cx, cy = pp.scene.TABLE_POS[0], pp.scene.TABLE_POS[1]
  (tx, ty), h = pp.BASELINES["B"].target_xy, pp.BASELINES["B"].target_half_xy
  assert torch.all((rel[:, 0] - (tx - cx)).abs() <= h + 1e-4), rel
  assert torch.all((rel[:, 1] - (ty - cy)).abs() <= h + 1e-4), rel
  top = mdp.surface_top_z(env, SceneEntityCfg("table"))
  assert torch.allclose(
    base.root_link_pos_w[:, 2] - top, torch.full_like(top, 0.025), atol=2e-3
  )
  _, base_yaw = mdp.stack._tilt_and_yaw(base.root_link_quat_w)
  assert torch.all(mdp.fold_quarter(base_yaw) <= pp.STACK_BASE_YAW + 1e-3)
  goal = env.command_manager.get_term("goal_pose").command
  d = goal - base.root_link_pos_w
  assert torch.allclose(
    d, torch.tensor([0.0, 0.0, 0.05], device=d.device).expand_as(d), atol=1e-4
  )
  cube = env.scene["object"].data.root_link_pos_w
  assert torch.all(torch.norm((cube - base.root_link_pos_w)[:, :2], dim=-1) >= 0.10)


def test_the_actor_is_38_d_and_its_goal_is_off_by_the_trackers_bias(stack_env):
  env = stack_env
  assert env.observation_manager.group_obs_dim["actor"] == (38,)
  assert env.observation_manager.group_obs_dim["critic"] == (39,)
  cfg = env.cfg.observations
  assert cfg["critic"].terms["target_pos"].noise is None
  noise = cfg["actor"].terms["target_pos"].noise
  assert noise.bias_noise_cfg.n_max == pp.STACK_TARGET_BIAS
  assert noise.bias_noise_cfg.operation == "abs"


def _put_cube(env, dxy=0.0, dz=0.0, tilt=0.0, yaw=0.0, vx=0.0):
  """The red cube on the base: ``dxy`` (m, +x) off its goal, ``dz`` up, tilted by ``tilt`` about x
  and turned by ``yaw`` against the base, moving at ``vx``; one step makes it visible."""
  obj, base = env.scene["object"], env.scene["base_object"]
  goal = env.command_manager.get_term("goal_pose").command
  _, base_yaw = mdp.stack._tilt_and_yaw(base.data.root_link_quat_w)
  pose = obj.data.root_link_pose_w.clone()
  pose[:, :3] = goal + torch.tensor([dxy, 0.0, dz], device=env.device)
  quat = quat_from_euler_xyz(
    torch.full_like(base_yaw, tilt), torch.zeros_like(base_yaw), base_yaw + yaw
  )
  pose[:, 3:7] = quat
  obj.write_root_link_pose_to_sim(pose)
  vel = torch.zeros_like(obj.data.root_link_vel_w)
  vel[:, 0] = vx
  obj.write_root_link_velocity_to_sim(vel)
  env.sim.forward()


@pytest.mark.parametrize(
  "kw, ok",
  [
    ({}, True),
    ({"dxy": OK_DXY * 0.8}, True),
    ({"dxy": 0.02}, False),
    ({"dz": 0.02}, False),
    ({"tilt": math.radians(20)}, False),
    ({"yaw": math.radians(20)}, False),
    ({"yaw": math.radians(88)}, True),  # a cube is square every 90 deg
    ({"vx": 0.1}, False),
  ],
)
def test_ok_is_inside_all_five_bounds(stack_env, kw, ok):
  env = stack_env
  _put_cube(env, **kw)
  state = mdp.stack_state(env, "object", "base_object", "goal_pose")
  assert bool(state.ok.all()) == ok, state
  if not ok:
    assert not bool(state.ok.any()), state


def test_tilt_and_yaw_read_back_what_was_set(stack_env):
  env = stack_env
  _put_cube(env, tilt=math.radians(7), yaw=math.radians(6), vx=0.03)
  s = mdp.stack_state(env, "object", "base_object", "goal_pose")
  assert torch.allclose(s.tilt, torch.full_like(s.tilt, math.radians(7)), atol=2e-3)
  assert torch.allclose(s.yaw, torch.full_like(s.yaw, math.radians(6)), atol=2e-2)
  assert torch.allclose(s.speed, torch.full_like(s.speed, 0.03), atol=1e-4)
  assert OK_TILT == OK_YAW == math.radians(10) and OK_SPEED == 0.05


def test_the_lock_holds_an_opening_jaw_only_while_armed(stack_env):
  env = stack_env
  term = env.action_manager.get_term("joint_pos")
  assert isinstance(term, mdp.StackGateJointPositionAction)
  jaw = term._gate_idx
  n = env.num_envs
  armed = torch.arange(n, device=env.device) % 2 == 0
  saved = term.lock_mask
  term.lock_mask = lambda: armed
  try:
    term._last_target[:, jaw] = 0.40  # a held jaw
    action = torch.zeros(env.action_space.shape, device=env.device)
    action[:, jaw] = -1.0  # all the way open
    term.process_actions(action)
    target = term._processed_actions[:, jaw]
    assert torch.all(target[armed] == 0.40), target
    assert torch.all(target[~armed] < 0.40), target  # slewed towards open, 0.03 a step
    assert torch.equal(term.locked, armed)
    action[:, jaw] = 1.0  # closing is never held back
    term._last_target[:, jaw] = 0.40
    term.process_actions(action)
    assert torch.all(term._processed_actions[:, jaw] > 0.40)
    assert not bool(term.locked.any())
  finally:
    term.lock_mask = saved
    env.reset()


def test_the_lock_is_not_armed_for_a_cube_never_lifted(stack_env):
  env = stack_env
  env.reset()
  term = env.action_manager.get_term("joint_pos")
  assert not bool(term.lock_mask().any())


# -- L4: a slid cube; RG's floor ---------------------------------------------------------------

GATED_TERMS = (
  "grasp",
  "lift",
  "goal",
  "goal_near",
  "release",
  "placed",
  "retreat",
  "pad_contact",
  "stack_align",
)


def _term(env, name: str) -> torch.Tensor:
  cfg = env.reward_manager.get_term_cfg(name)
  return cfg.func(env, **cfg.params)


def test_a_cube_slid_onto_the_base_earns_nothing_past_reach(stack_env):
  env = stack_env
  env.reset()
  zero = torch.zeros(env.action_space.shape, device=env.device)
  env.step(zero)
  _put_cube(env)
  env.step(zero)
  assert not bool(getattr(env, _WAS_LIFTED).any()), "the cube was never lifted"
  assert torch.all(_term(env, "reach") > 0.0)
  for name in GATED_TERMS:
    value = _term(env, name)
    assert torch.all(value == 0.0), f"{name} paid a slide: {value}"


def test_rg_floors_a_release_that_was_not_ok():
  """With the latch as a release on the base would leave it (2 mm off, 1 mm up), RG's grade is
  M6's only if ``OK`` held at the let-go, and the floor otherwise (positive control: the same
  latch with ``rel_ok``). Read from the grade itself: a cube put on the base by hand can meet the
  hand at its home pose (see `test_the_home_pose_hangs_the_fingers_over_the_tower`)."""
  env = ManagerBasedRlEnv(
    cfg=_cfg(num_envs=2, reward=RCTP_RG, gate="none"), device=get_test_device()
  )
  try:
    env.reset()
    env.step(torch.zeros(env.action_space.shape, device=env.device))
    params = dict(env.reward_manager.get_term_cfg("release").params)
    assert params["require_ok"] is True
    params = {k: v for k, v in params.items() if k != "threshold"}
    _ = mdp.release_quality(env, **params)  # builds the latch
    latch = getattr(env, _RELEASE)
    latch.released[:] = True
    latch.dxy = torch.full_like(latch.dxy, 0.002)
    latch.height = torch.full_like(latch.height, 0.001)
    latch.rel_ok = torch.tensor([True, False], device=env.device)
    q = mdp.release_quality(env, **params)
    assert q[0].item() > 0.9 and q[1].item() == pytest.approx(rw.M6_FLOOR), q
    params["require_ok"] = False
    q = mdp.release_quality(env, **params)
    assert torch.all(q > 0.9), q
  finally:
    env.close()


def test_the_home_pose_hangs_the_fingers_over_the_tower(stack_env):
  """A finding, recorded as a test so it is not forgotten (S9 step 0, 2026-10-01): at the arm's
  home pose the TCP is 4-15 cm beside and 4-8 cm above the stack goal, so the open fingertips
  hang at the height of a stacked cube's top face. The eval's hand-off, which returns home, has
  to lift clear first (`--handoff-lift`) -- and so does the robot's."""
  env = stack_env
  env.reset()
  env.step(torch.zeros(env.action_space.shape, device=env.device))
  robot = env.scene["robot"]
  tcp = robot.data.site_pos_w[:, robot.site_names.index("tcp_left")]
  d = tcp - env.command_manager.get_term("goal_pose").command
  assert torch.all(d[:, 2].abs() < 0.08) and torch.all(d[:, :2].norm(dim=-1) < 0.15), d


# -- L1 / L5: the scripted stack through the lock ----------------------------------------------

_SCRIPT = pathlib.Path(__file__).parent.parent / "scripts/scripted_pick_place.py"
_SPEC = importlib.util.spec_from_file_location("scripted_pick_place_s9", _SCRIPT)
assert _SPEC and _SPEC.loader
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
TASK_TERMS = (
  "reach",
  "grasp",
  "lift",
  "goal",
  "goal_near",
  "release",
  "placed",
  "retreat",
  "stack_align",
)


def _rollout(**kw):
  return scripted.rollout(
    device=get_test_device(),
    verbose=False,
    robot=_robot(),
    baseline=pp.BASELINES["B"],
    spawn_box=pp.SPAWN_BOXES["W2"],
    dr="DR-scene",
    effort=CTP,
    latency=tt.LATENCY_MODELS["JS"],
    stack_speed=True,
    episode_length_s=70.0,
    goal="stack",
    **kw,
  )


@pytest.fixture(scope="module")
def stack_episode():
  """One scripted stack under ST-GA (the lock, RCTP + SA), every task term of ST (RCTP), ST-R
  (RCTP + RG) and ST-GA recorded at every step for env 0, weighted: the states are reached
  physically, through the lock."""
  rec = {arm: collections.defaultdict(list) for arm in (RCTP, RCTP_RG, RCTP_SA)}
  terms: dict = {}

  def hook(phase, env):
    if not terms:
      for arm in rec:
        built = _cfg(num_envs=1, reward=arm).rewards
        for cfg in built.values():
          for value in cfg.params.values():
            if isinstance(value, SceneEntityCfg):
              value.resolve(env.scene)
        terms[arm] = {n: c for n, c in built.items() if n in TASK_TERMS}
    for arm, arm_terms in terms.items():
      rec[arm][phase].append(
        {n: float(c.weight * c.func(env, **c.params)[0]) for n, c in arm_terms.items()}
      )
    rec["locked"] = rec.get("locked", 0) + int(
      env.action_manager.get_term("joint_pos").locked[0]
    )

  rows = _rollout(
    num_envs=1, reward=RCTP_SA, gate="stack", hook=hook, phases=VERTICAL_LOWERING
  )
  assert rows[0]["placed_and_lifted"], rows[0]
  return rec


def _totals(rec) -> dict[str, float]:
  total = lambda step: sum(step.values())  # noqa: E731
  placed = [t for t in rec["open"] + rec["retreat"] if t["placed"] > 0]
  assert placed, "the cube never counted as placed on the base"
  return {
    "reach": total(rec["descend"][-1]),
    "carry": total(rec["lift"][-1]),
    "over_target": total(rec["steady"][-1]),
    "held_on_target": total(rec["hold"][-1]),
    "placed": total(placed[-1]),
  }


@pytest.mark.parametrize("arm", [RCTP, RCTP_RG, RCTP_SA])
def test_the_reward_rises_strictly_along_the_stack(stack_episode, arm):
  totals = _totals(stack_episode[arm])
  values = [totals[name] for name in ORDER]
  assert all(a < b for a, b in zip(values, values[1:], strict=False)), totals


def test_the_scripted_let_go_is_ok_so_rg_pays_the_full_grade(stack_episode):
  rg = [t for t in stack_episode[RCTP_RG]["retreat"] if t["placed"] > 0]
  assert rg and rg[-1]["placed"] > 0.8 * rw.M1_PLACED_WEIGHT, rg[-1:]


def test_sa_pays_on_the_base_and_not_on_the_way(stack_episode):
  sa = stack_episode[RCTP_SA]
  assert max(t["stack_align"] for t in sa["hold"]) > 0.3
  for phase in ("approach", "descend", "close", "lift", "carry"):
    # Held 5 cm or more over the goal: the height factor alone is exp(-25).
    assert all(t["stack_align"] < 1e-3 for t in sa[phase]), phase


def test_the_lock_refuses_a_let_go_two_centimetres_off():
  """Positive control for the lock: the script aims the release 2 cm off the base's centre (env
  0), and on it (env 1). The off one is held shut -- the jaw never opens over the base, so it is
  never placed -- the aligned one is placed."""
  rows = _rollout(
    num_envs=2,
    reward=RCTP,
    gate="stack",
    corners_first=False,
    release_offsets=[(0.02, 0.0), (0.0, 0.0)],
  )
  off, on = rows
  assert off["gate_locked_steps"] > 0 and not off["placed_and_lifted"], off
  assert on["placed_and_lifted"], on
