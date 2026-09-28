"""Plan S7-4 (revised 2026-09-28 21:33): a release is the jaw opening (J), a retreat after it
(M8), the arm's effort bound (C, TP), V at 0.5 m/s and M7 revised (S).

What the plan asks before GPU time, on states the scripted pick-and-place reaches through the
policy's action interface on S7-4's plant (G20, W2's arm, baseline B, TP's lowered limit):

* J: a "release" with the jaw still near the hold angle is not paid; open-and-lift beats it per
  step and in discounted return over a 500-step episode;
* M8: place + retreat earns more than place + stay; place + stay pressing earns less than both;
  a retreat that knocks the cube 3 cm off earns less than a clean one;
* P: the hinge's weight, calibrated -- a press at the cap costs more per step than the `placed`
  income it could protect, and a clean scripted pick-and-place pays < 1 % of its return;
* C / T: the arm's target stays in the joint range and within q +/- tau_cmd / K;
* S: a set-down earns the bonus on top of M6's grade; a release while lowering keeps M6's grade
  (M7 floored it) and earns no bonus.

The knock is a stand-in: the cube is moved 3 cm, at rest, as the retreat starts (the script
cannot sweep the hand through it reliably). The press is S7-3's (report section 8.4): after the
jaw opens, the elbow's target is walked 0.5 rad further down from the place pose
(`scripted.PRESS_WAYPOINT`), so the open fingertips push on the table around the placed cube at
up to T's cap. (Lowering the place *pose* does not do it: the script's re-plan refuses a pose in
the table, and with the jaw open nothing loads the arm.)
"""

import collections
import importlib.util
import pathlib
import sys

import numpy as np
import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from pal_mjlab.tasks.manipulation import mdp
from pal_mjlab.tasks.manipulation.mdp.observations import (
  _RELEASE,
  JAW_OPEN_BELOW,
  _grasped_bool,
)
from pal_mjlab.tasks.manipulation.pick_place import pick_place_env_cfg as pp
from pal_mjlab.tasks.manipulation.pick_place.kangaroo import env_cfg as kenv
from pal_mjlab.tasks.manipulation.pick_place.rewards import (
  JAW_OPEN,
  JAW_RELEASE,
  M1_PLACED_WEIGHT,
  M6_FLOOR,
  M8_CLEAR_HEIGHT,
  M8_CONTACT_WEIGHT,
  M8_RETREAT_WEIGHT,
  NO_SUCCESS_TERMINATION,
  P_THRESHOLD,
  P_WEIGHT,
  S52_BASE,
  S_BONUS_WEIGHT,
  S_DWELL_STEPS,
  _s52_arm,
)
from pal_mjlab.tasks.manipulation.rl_cfg import stack_ppo_runner_cfg
from pal_mjlab.tasks.manipulation.robots import (
  KANGAROO_GRIPPERS_LEFT_ARM_WIDE,
  kangaroo_grippers_left_arm,
)

_SCRIPT = pathlib.Path(__file__).parent.parent / "scripts/scripted_pick_place.py"
_SPEC = importlib.util.spec_from_file_location("scripted_pick_place_s74", _SCRIPT)
assert _SPEC and _SPEC.loader
scripted = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = scripted
_SPEC.loader.exec_module(scripted)

_EVAL = pathlib.Path(__file__).parent.parent / "scripts/eval_pick_place.py"
_ESPEC = importlib.util.spec_from_file_location("eval_pick_place_s74", _EVAL)
assert _ESPEC and _ESPEC.loader
ev = importlib.util.module_from_spec(_ESPEC)
sys.modules[_ESPEC.name] = ev
_ESPEC.loader.exec_module(ev)

FR = f"{S52_BASE}+M1+M2+M3+M6"
R = f"{FR}+J+M8"
RS = f"{R}+S"
B = pp.BASELINES["B"]
W2 = pp.SPAWN_BOXES["W2"]
CTP = pp.EFFORT_BOUNDS["CTP"]
SLEW = 0.5
GAMMA = stack_ppo_runner_cfg("x").algorithm.gamma
EPISODE_STEPS = 500
#: Terms left out of the per-step totals: the action-rate and joint-velocity regularisers price
#: the script's own motion, which is not what is compared.
EXCLUDED = ("action_rate_l2", "joint_vel_hinge")
#: The script's carry height, m (as the corners check at B's raised table), and its episode, s.
CARRY_HEIGHT = 0.05
SCRIPT_EPISODE_S = 40.0
#: W2's centre, yaw 0: the spawn every scenario starts from.
SPAWN = ((0.5 * sum(W2.x), 0.5 * sum(W2.y)), 0.0)
#: A jaw command a little short of the first touch of the 5 cm cube (~0.21 rad): the pads let
#: go, the jaw stays above `JAW_OPEN` -- S7-1's FR s2 "release" (plan S7-4, revision 1).
HOLD_CLOSED_JAW = 0.18
KNOCK = 0.03


def _robot():
  return kangaroo_grippers_left_arm(
    "G20", KANGAROO_GRIPPERS_LEFT_ARM_WIDE, arm_effort_scale=CTP.effort_scale
  )


def _cfg(arm: str, effort=CTP, tool_speed=None):
  return pp.make_pick_place_env_cfg(
    _robot() if effort.effort_scale else kangaroo_grippers_left_arm("G20"),
    gripper_name="left",
    reward=arm,
    action_slew_rate_scale=SLEW,
    baseline=B,
    spawn=W2,
    dr="DR-scene",
    tool_speed_cap=tool_speed,
    effort=effort,
  )


# -- Configuration --------------------------------------------------------------------------


def test_j_and_m8_on_fr():
  r, fr = _cfg(R), _cfg(FR)
  assert r.rewards.keys() == fr.rewards.keys() | {"retreat", "pad_contact"}
  for name in ("release", "placed"):
    extra = {
      k: v
      for k, v in r.rewards[name].params.items()
      if k not in fr.rewards[name].params
    }
    assert set(extra) == {"jaw_cfg", "jaw_open"}, name
    assert extra["jaw_open"] == JAW_OPEN == JAW_OPEN_BELOW == 0.15
    assert extra["jaw_cfg"].joint_names == ("gripper_left_finger_joint",)
    assert r.rewards[name].func is fr.rewards[name].func
    assert r.rewards[name].weight == fr.rewards[name].weight
  retreat = r.rewards["retreat"]
  assert retreat.func is mdp.retreat_height and retreat.weight == M8_RETREAT_WEIGHT
  assert retreat.params["clear_height"] == M8_CLEAR_HEIGHT == 0.05
  contact = r.rewards["pad_contact"]
  assert contact.func is mdp.pad_contact_after_release
  assert contact.weight == M8_CONTACT_WEIGHT < 0
  assert R in NO_SUCCESS_TERMINATION and R in JAW_RELEASE and FR not in JAW_RELEASE


def test_s_adds_the_set_down_bonus_and_keeps_m6():
  rs, r = _cfg(RS), _cfg(R)
  assert rs.rewards.keys() == r.rewards.keys() | {"set_down"}
  for name in ("release", "placed"):
    assert rs.rewards[name].params == r.rewards[name].params, "S keeps M6's grade"
  bonus = rs.rewards["set_down"]
  assert bonus.func is mdp.set_down_placed and bonus.weight == S_BONUS_WEIGHT
  assert bonus.params["set_down_steps"] == S_DWELL_STEPS == 2


def test_module_order_is_checked():
  for mods, match in (
    (("M1", "M2", "M3", "M6", "M8"), "M8 needs J"),
    (("M2", "J"), "J redefines"),
    (("M1", "M6", "J", "S", "M7"), "M7 pays|pick one"),
    (("M1", "J", "S"), "S needs M6 and J"),
  ):
    with pytest.raises(ValueError, match=match):
      _s52_arm(mods)


def test_the_effort_codes():
  assert set(pp.EFFORT_BOUNDS) == {"none", "C", "CTP"}
  arm = ("arm_left_[1-7]_joint",)
  for code, clip, band, hinge in (
    ("none", None, None, False),
    ("C", arm, None, False),
    ("CTP", arm, arm, True),
  ):
    cfg = _cfg(FR, effort=pp.EFFORT_BOUNDS[code])
    action = cfg.actions["joint_pos"]
    assert action.range_clip_joints == clip, code
    assert action.effort_band_joints == band, code
    assert ("arm_effort" in cfg.rewards) is hinge, code
  assert CTP.effort_scale == 0.7
  p = _cfg(FR).rewards["arm_effort"]
  assert p.func is mdp.arm_effort_hinge and p.weight == P_WEIGHT
  assert p.params["threshold"] == P_THRESHOLD == 0.8
  assert p.params["asset_cfg"].joint_names == arm


def test_the_hinge_weight_is_calibrated():
  """A press at the cap costs more per step than the `placed` income it could protect, with
  the plan's thresholds 28 / 14 N.m above the ~11 N.m gravity load."""
  at_cap = -P_WEIGHT * (1.0 - P_THRESHOLD) ** 2
  assert at_cap > M1_PLACED_WEIGHT * 1.0, (at_cap, M1_PLACED_WEIGHT)
  assert 0.7 * 50 * P_THRESHOLD == pytest.approx(28.0)
  assert 0.7 * 25 * P_THRESHOLD == pytest.approx(14.0)
  assert 0.7 * 25 * P_THRESHOLD > 10.9


def test_v_at_half_a_metre_per_second():
  v = pp.TOOL_SPEED_CAPS["V0.5"]
  assert (v.speed, v.slow_speed, v.slow_zone) == (0.5, 0.1, 0.05)
  assert pp.TOOL_SPEED_CAPS["V"].speed == 0.3, "S7-2 / S7-3's V is unchanged"


def test_effort_is_read_and_validated(monkeypatch):
  monkeypatch.setenv("GAINS", "G20")
  monkeypatch.setenv("EFFORT", "CTP")
  cfg = kenv.kangaroo_pick_place_env_cfg()
  assert cfg.actions["joint_pos"].effort_band_joints is not None
  assert "arm_effort" in cfg.rewards
  monkeypatch.setenv("EFFORT", "TP")
  with pytest.raises(ValueError, match="EFFORT"):
    kenv.kangaroo_pick_place_env_cfg()
  monkeypatch.delenv("EFFORT")
  cfg = kenv.kangaroo_pick_place_env_cfg()
  assert cfg.actions["joint_pos"].effort_band_joints is None


# -- The plant and the action term ------------------------------------------------------------


def _env(effort, num_envs=2):
  cfg = _cfg(FR, effort=effort)
  cfg.scene.num_envs = num_envs
  return ManagerBasedRlEnv(cfg=cfg, device=get_test_device())


@pytest.fixture(scope="module")
def ctp_env():
  env = _env(CTP)
  yield env
  env.close()


def test_t_lowers_the_left_arm_limit_on_the_model(ctp_env):
  import mujoco

  mj = ctp_env.sim.mj_model
  seen = {}
  for j in range(mj.njnt):
    name = (mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, j) or "").split("/")[-1]
    if name.startswith("arm_"):
      seen[name] = (bool(mj.jnt_actfrclimited[j]), float(mj.jnt_actfrcrange[j, 1]))
  for i in range(1, 8):
    assert seen[f"arm_left_{i}_joint"] == (True, 35.0 if i <= 2 else 17.5)
    # The parked arm is untouched: G20 has no gravity compensation, so no joint limit at all.
    assert not seen[f"arm_right_{i}_joint"][0]


def test_t_band_is_the_pd_output_within_the_cap(ctp_env):
  """Half-width tau_cmd / K, centred D / K x qdot ahead of the measured position."""
  term = ctp_env.action_manager.get_term("joint_pos")
  k = {1: 998.79, 3: 523.54}  # G20: SysID x 20 (kangaroo_constants)
  want = [35.0 / k[1]] * 2 + [17.5 / k[3]] * 5
  assert term._band[:, 0].tolist() == pytest.approx(want, rel=1e-3)
  assert term._band[:, 1].tolist() == pytest.approx([0.0637] * 7, rel=1e-2)


@pytest.mark.parametrize("code", ["C", "CTP"])
def test_a_run_away_target_is_bounded(code):
  """S7-3's run-away: the policy walks the elbow target past its lower limit at the slew rate.
  An action far outside [-1, 1] (the policy's output is not clipped) drives every arm target
  down; C holds it at the joint range, T where the PD output stays within the cap -- checked
  against the state the bound saw, before the step."""
  env = _env(pp.EFFORT_BOUNDS[code], num_envs=1)
  try:
    env.reset()
    term = env.action_manager.get_term("joint_pos")
    robot = env.scene["robot"]
    arm = term._range_idx
    joints = term._target_ids[arm]
    lo = term._range[0, :, 0]
    action = torch.zeros(env.action_space.shape, device=env.device)
    action[:, arm] = -50.0
    with torch.inference_mode():
      for _ in range(150):
        q = robot.data.joint_pos[0, joints].clone()
        qd = robot.data.joint_vel[0, joints].clone()
        env.step(action)
        target = term._processed_actions[0, arm]
        assert (target >= lo - 1e-6).all(), (target, lo)
        if term._band is not None:
          tau_over_k, d_over_k = term._band[:, 0], term._band[:, 1]
          pd = (target - q) - d_over_k * qd  # the PD output over K
          assert (pd.abs() <= tau_over_k + 1e-5).all(), (pd, tau_over_k)
    # Positive control: the bound under test did bind.
    hit = term.range_hit if code == "C" else term.band_hit
    assert bool(hit.any()), code
  finally:
    env.close()


# -- Scripted episodes ----------------------------------------------------------------------


class _Recorder:
  """Per step and env: R's and RS's terms, weighted, the jaw angle, the grasp, the placed
  predicate without J (the control), the TCP's height over the cube and the release latch."""

  def __init__(self):
    self.terms: dict = {}
    self.rec = collections.defaultdict(lambda: collections.defaultdict(list))
    self.grasp_sensor = _robot().gripper("left").contact_sensor_name

  def __call__(self, phase, env):
    if not self.terms:
      for arm in (R, RS):
        rewards = _cfg(arm).rewards
        for cfg in rewards.values():
          for value in cfg.params.values():
            if isinstance(value, SceneEntityCfg):
              value.resolve(env.scene)
        self.terms[arm] = {n: c for n, c in rewards.items() if n not in EXCLUDED}
      self.control = dict(self.terms[R]["placed"].params)
      for k in ("jaw_cfg", "jaw_open", "surface_cfg", "dist_std", "height_free"):
        self.control.pop(k, None)
      for k in ("height_std", "floor", "set_down_steps"):
        self.control.pop(k, None)
    values = {
      arm: {n: (c.weight * c.func(env, **c.params)).tolist() for n, c in ts.items()}
      for arm, ts in self.terms.items()
    }
    robot = env.scene["robot"]
    jaw = robot.data.joint_pos[:, robot.joint_names.index(scripted.JAW_JOINT)]
    placed_contact = mdp.object_placed_reward(env, **self.control).tolist()
    latch = getattr(env, _RELEASE)
    tcp = robot.data.site_pos_w[:, robot.site_names.index("tcp_left")]
    obj = env.scene[pp.OBJECT]
    top = obj.data.root_link_pos_w[:, 2] + 0.025
    grasped = _grasped_bool(env, self.grasp_sensor).tolist()
    for i in range(env.num_envs):
      self.rec[i][phase].append(
        {
          "values": {a: {n: v[i] for n, v in vs.items()} for a, vs in values.items()},
          "jaw": float(jaw[i]),
          "grasped": grasped[i],
          "placed_contact": placed_contact[i],
          "released": bool(latch.released[i]),
          "latch_dxy": float(latch.dxy[i]),
          "latch_height": float(latch.height[i]),
          "tcp_above": float(tcp[i, 2] - top[i]),
        }
      )


def _flat(rec_env: dict) -> list[dict]:
  return [s | {"phase": ph} for ph, steps in rec_env.items() for s in steps]


def _total(step: dict, arm: str = R) -> float:
  return sum(step["values"][arm].values())


def _run(num_envs, hook, release_height, phases, **kw):
  saved = scripted.CLEAR_HEIGHT, scripted.RELEASE_HEIGHT
  scripted.CLEAR_HEIGHT, scripted.RELEASE_HEIGHT = CARRY_HEIGHT, release_height
  try:
    return scripted.rollout(
      num_envs=num_envs,
      device=get_test_device(),
      reward=R,
      verbose=False,
      hook=hook,
      action_slew_rate_scale=SLEW,
      robot=_robot(),
      baseline=B,
      spawn_box=W2,
      effort=CTP,
      spawns=[SPAWN] * num_envs,
      targets=[B.target_xy] * num_envs,
      episode_length_s=SCRIPT_EPISODE_S,
      phases=phases,
      **kw,
    )
  finally:
    scripted.CLEAR_HEIGHT, scripted.RELEASE_HEIGHT = saved


_UP_TO_HOLD = scripted.PHASES[: [p[0] for p in scripted.PHASES].index("hold") + 1]


@pytest.fixture(scope="module")
def clean_and_knocked():
  """Env 0 sets the cube down on the target (pressed 2 mm into the table, so it rests there
  held), opens the jaw and lifts clear. Env 1 does the same, and its cube is moved `KNOCK`
  horizontally, at rest, once the hand has lifted (at the start of `settle_more`): moved while
  the open fingers still stand around it, it is pushed straight back by a fingertip."""
  rec = _Recorder()
  moved = {"done": False}

  def hook(phase, env):
    if phase == "settle_more" and not moved["done"]:
      obj = env.scene[pp.OBJECT]
      ids = torch.tensor([1], device=env.device)
      pose = obj.data.root_link_pose_w[1:2].clone()
      pose[0, 1] += KNOCK
      obj.write_root_link_pose_to_sim(pose, env_ids=ids)
      obj.write_root_link_velocity_to_sim(
        torch.zeros(1, 6, device=env.device), env_ids=ids
      )
      moved["done"] = True
    rec(phase, env)

  phases = scripted.PHASES + (("settle_more", "retreat", False, 60),)
  rows = _run(2, hook, -0.002, phases)
  assert moved["done"]
  return rows, rec.rec


@pytest.fixture(scope="module")
def stay_open():
  """Sets the cube down, opens the jaw fully, and leaves the open hand where it is."""
  rec = _Recorder()
  phases = _UP_TO_HOLD + (
    ("open", "place", False, 30),
    ("stay", "place", False, 60),
    ("settle", "place", False, 40),
  )
  return _run(1, rec, -0.002, phases), rec.rec


@pytest.fixture(scope="module")
def hold_closed():
  """Sets the cube down and opens the jaw only to `HOLD_CLOSED_JAW`: the pads let go, the jaw
  never opens, the hand stays -- the release S7-2/S7-3's policies learned."""
  rec = _Recorder()
  frac = HOLD_CLOSED_JAW / _robot().gripper("left").closed_pos
  phases = _UP_TO_HOLD + (
    ("open", "place", frac, 30),
    ("stay", "place", frac, 60),
    ("settle", "place", frac, 40),
  )
  return _run(1, rec, -0.002, phases), rec.rec


@pytest.fixture(scope="module")
def press():
  """Sets the cube down, opens the jaw, then walks the elbow's target down past the place pose
  and holds it there: the open fingertips push on the table around the placed cube."""
  rec = _Recorder()
  phases = _UP_TO_HOLD + (
    ("open", "place", False, 30),
    ("press", scripted.PRESS_WAYPOINT, False, 30),
    ("stay", scripted.PRESS_WAYPOINT, False, 60),
    ("settle", scripted.PRESS_WAYPOINT, False, 40),
  )
  return _run(1, rec, -0.002, phases), rec.rec


#: M7's lowering release (tests/test_pick_place_m7.py): the jaw opens as the arm, lagging its
#: command, is still coming down -- a release just above the table while the cube moves.
LOWER_GRIP = 0.9
LOWER_TO = -0.02


@pytest.fixture(scope="module")
def lowering():
  rec = _Recorder()
  phases = scripted.PHASES[:6] + (
    ("lower", "place", LOWER_GRIP, 8),
    ("open", "place", False, 30),
    ("retreat", "retreat", False, 40),
    ("settle", "retreat", False, 60),
  )
  return _run(1, rec, LOWER_TO, phases), rec.rec


def _j_release(steps: list[dict]) -> int:
  for k, s in enumerate(steps):
    if s["released"]:
      return k
  raise AssertionError("no J-release in the episode")


def _return(steps: list[dict], arm: str = R) -> float:
  """Discounted return from the J-release over the rest of a 500-step episode, the release at
  step 50, the last recorded value held to the end."""
  seq = [_total(s, arm) for s in steps[_j_release(steps) :]]
  seq = (seq + [seq[-1]] * EPISODE_STEPS)[: EPISODE_STEPS - 50]
  return float((GAMMA ** np.arange(len(seq))) @ np.asarray(seq))


def test_the_scenarios_reach_their_states(
  clean_and_knocked, stay_open, hold_closed, press
):
  clean = _flat(clean_and_knocked[1][0])
  assert clean[-1]["jaw"] < JAW_OPEN and clean[-1]["tcp_above"] > 0.03
  assert clean[-1]["values"][R]["placed"] > 0
  knocked = _flat(clean_and_knocked[1][1])
  assert knocked[-1]["values"][R]["placed"] == 0
  stay = _flat(stay_open[1][0])
  assert stay[-1]["jaw"] < JAW_OPEN and stay[-1]["tcp_above"] < 0.01
  assert stay[-1]["values"][R]["placed"] > 0
  held = _flat(hold_closed[1][0])
  assert held[-1]["jaw"] >= JAW_OPEN and not held[-1]["grasped"], held[-1]
  pressed = _flat(press[1][0])
  assert pressed[-1]["jaw"] < JAW_OPEN
  assert pressed[-1]["values"][R]["table_contact"] < 0, "the hand presses on the table"


def test_a_hold_closed_release_is_not_paid(hold_closed):
  s = _flat(hold_closed[1][0])[-1]
  # Positive control: the pads' let-go counts as placed without J.
  assert s["placed_contact"] > 0, s
  assert not s["released"]
  assert s["values"][R]["placed"] == 0 and s["values"][R]["release"] == 0
  assert s["values"][R]["retreat"] == 0


def test_open_and_lift_beats_stay_closed(clean_and_knocked, hold_closed):
  clean = _flat(clean_and_knocked[1][0])
  held = _flat(hold_closed[1][0])
  assert _total(clean[-1]) > _total(held[-1]) + 5.0
  # The held run has no J-release; its return is taken from the step the clean one opened.
  k = _j_release(clean)
  seq_h = [_total(s) for s in held[k:]]
  seq_h = (seq_h + [seq_h[-1]] * EPISODE_STEPS)[: EPISODE_STEPS - 50]
  ret_h = float((GAMMA ** np.arange(len(seq_h))) @ np.asarray(seq_h))
  assert _return(clean) > ret_h


def test_retreat_beats_stay_and_stay_beats_press(clean_and_knocked, stay_open, press):
  clean = _flat(clean_and_knocked[1][0])
  stay = _flat(stay_open[1][0])
  pressed = _flat(press[1][0])
  assert _total(clean[-1]) > _total(stay[-1]) > _total(pressed[-1])
  # The script lifts to its carry height, ~4 cm over the cube's top: most of `h_clear`.
  frac = min(clean[-1]["tcp_above"] / M8_CLEAR_HEIGHT, 1.0)
  assert clean[-1]["values"][R]["retreat"] == pytest.approx(M8_RETREAT_WEIGHT * frac)
  assert frac > 0.6
  assert stay[-1]["values"][R]["retreat"] == 0
  assert _return(clean) > _return(stay) > _return(pressed)


def test_a_knocked_retreat_earns_less(clean_and_knocked):
  clean = _flat(clean_and_knocked[1][0])
  knocked = _flat(clean_and_knocked[1][1])
  assert _total(clean[-1]) > _total(knocked[-1]) + 5.0
  assert _return(clean) > _return(knocked)


def test_the_press_is_priced_at_the_cap(press):
  """What the hinge charged the press, per step, against the placed income it could protect."""
  pressed = _flat(press[1][0])
  worst = min(s["values"][R]["arm_effort"] for s in pressed[_j_release(pressed) :])
  print(f"press: worst per-step effort hinge {worst:.2f}")
  assert worst < 0


def test_a_clean_pick_and_place_pays_under_one_percent(clean_and_knocked):
  steps = _flat(clean_and_knocked[1][0])
  hinge = sum(s["values"][R]["arm_effort"] for s in steps)
  income = sum(v for s in steps for v in s["values"][R].values() if v > 0)
  print(f"clean: hinge {hinge:.3f} over {len(steps)} steps, income {income:.1f}")
  assert -hinge < 0.01 * income, (hinge, income)


def test_the_set_down_earns_the_bonus(clean_and_knocked):
  s = _flat(clean_and_knocked[1][0])[-1]
  assert s["values"][RS]["set_down"] == pytest.approx(S_BONUS_WEIGHT)


def test_a_lowering_release_keeps_m6_and_earns_no_bonus(lowering):
  steps = _flat(lowering[1][0])
  s = steps[-1]
  assert s["released"] and s["values"][RS]["set_down"] == 0
  # M7 floored this release; S keeps M6's grade of it (its placed term above the floor's).
  assert s["values"][RS]["placed"] > M1_PLACED_WEIGHT * (M6_FLOOR + 0.2), s
  assert s["latch_height"] < 0.01


# -- The eval's hand-off (scripts/eval_pick_place.py) ----------------------------------------


def test_eval_constants_match():
  assert ev.JAW_OPEN == JAW_OPEN
  assert ev.HANDOFF_CLEAR == 0.03 and ev.HANDOFF_STEPS == 5
  assert ev.HANDOFF_TIMEOUT_S == 8.0


def _feed(t, n, **kw):
  base = dict(
    jaw_q=torch.tensor([0.3]),
    grasped=torch.tensor([False]),
    lifted=torch.tensor([False]),
    placed_j=torch.tensor([False]),
    tcp_above=torch.tensor([0.0]),
    obj_xy=torch.zeros(1, 2),
    goal_dxy=torch.tensor([0.0]),
    arm_home_err=torch.tensor([0.5]),
  )
  base.update(
    {k: torch.tensor([v]) if not torch.is_tensor(v) else v for k, v in kw.items()}
  )
  return [t.step(**base) for _ in range(n)]


def test_the_handoff_waits_for_the_jaw_release_and_the_lift():
  t = ev.RetreatTally(1, "cpu", 0.02, handoff=True)
  # The jaw open at the start with the TCP high is not a release: nothing was lifted.
  assert not any(bool(h) for h in _feed(t, 10, jaw_q=0.0, tcp_above=0.1))
  _feed(t, 5, jaw_q=0.3, grasped=True, lifted=True)  # carry
  _feed(t, 3, jaw_q=0.3, grasped=False)  # the pads let go; the jaw still near the hold
  assert not bool(t.released[0])
  _feed(t, 2, jaw_q=0.1, placed_j=True, tcp_above=0.0)  # the jaw opens: J-release
  assert bool(t.released[0])
  hands = _feed(t, 5, jaw_q=0.1, placed_j=True, tcp_above=0.04)
  assert [bool(h) for h in hands] == [False] * 4 + [True]
  assert int(t.hand_by[0]) == 1
  _feed(t, 3, jaw_q=0.0, placed_j=True, tcp_above=0.08, arm_home_err=0.01)
  t.finish(torch.tensor([0]), [True])
  row = t.done[0]
  assert row["handoff_by"] == "lift" and row["lifted_clear"] and row["success_retreat"]
  assert row["time_to_home_s"] == pytest.approx(0.02)


def test_the_handoff_falls_back_at_eight_seconds():
  t = ev.RetreatTally(1, "cpu", 0.02, handoff=True)
  hands = _feed(t, 401, jaw_q=0.3)
  assert not bool(hands[399]) and bool(hands[400])
  t.finish(torch.tensor([0]), [False])
  assert t.done[0]["handoff_by"] == "timeout" and not t.done[0]["success_retreat"]


def test_no_handoff_without_the_flag():
  t = ev.RetreatTally(1, "cpu", 0.02, handoff=False)
  assert not any(bool(h) for h in _feed(t, 450, jaw_q=0.3))


def test_eval_parses_the_handoff_flag():
  args = ev.parse_args(["--checkpoint", "x.pt", "--handoff"])
  assert args.handoff
