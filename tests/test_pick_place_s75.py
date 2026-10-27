"""Plan S7-5 (designed 2026-09-29 12:08, Siwei): open the jaw at the goal (O) and the goal
tolerance split into xy and z (Z), on S7-4's RCTP.

What the plan asks before GPU time (prep 2), on S7-4's plant (G20, W2's arm, baseline B, TP's
lowered limit, DR-scene):

* O pays only inside its gate -- lifted this episode, not yet J-released, xy < 2 cm and
  |dz| < 2 cm of the goal, any tilt -- and grows as the jaw opens from 0.40 (the plan's 0.30,
  raised on prep 3's trace) towards 0.15 rad;
* a half-open hover earns less per step than placing (analytically, and on the scripted
  pick-and-place: the pads let go at the target, the jaw stays at 0.18 rad);
* Z's `placed` on a cube tilted on an edge at the target is true once it is released and has
  fallen flat and settled.

Why: RCTP s2 (S7-4 report) parks the cube on the target and holds it to the time-out with the
jaw closed; nothing pays for starting to open, and a release from its tilted, rocking hold rarely
ends within 2.5 cm in 3-D.
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
from pal_mjlab.tasks.manipulation import mdp
from pal_mjlab.tasks.manipulation.mdp.observations import (
  _RELEASE,
  _WAS_LIFTED,
  _at_goal,
  _grasped_bool,
  _placed_bool,
)
from pal_mjlab.tasks.manipulation.pick_place import pick_place_env_cfg as pp
from pal_mjlab.tasks.manipulation.pick_place.rewards import (
  GOAL_TOL_XYZ,
  JAW_OPEN,
  JAW_RELEASE,
  M1_PLACED_WEIGHT,
  M1_RELEASE_WEIGHT,
  M6_FLOOR,
  NO_SUCCESS_TERMINATION,
  O_JAW_HOLD,
  O_WEIGHT,
  S52_BASE,
  Z_GOAL_TOL,
  _s52_arm,
)
from pal_mjlab.tasks.manipulation.robots import (
  KANGAROO_GRIPPERS_LEFT_ARM_WIDE,
  kangaroo_grippers_left_arm,
)
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import SUPPORT_SENSOR

_SCRIPT = pathlib.Path(__file__).parent.parent / "scripts/scripted_pick_place.py"
_SPEC = importlib.util.spec_from_file_location("scripted_pick_place_s75", _SCRIPT)
assert _SPEC and _SPEC.loader
scripted = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = scripted
_SPEC.loader.exec_module(scripted)

_EVAL = pathlib.Path(__file__).parent.parent / "scripts/eval_pick_place.py"
_ESPEC = importlib.util.spec_from_file_location("eval_pick_place_s75", _EVAL)
assert _ESPEC and _ESPEC.loader
ev = importlib.util.module_from_spec(_ESPEC)
sys.modules[_ESPEC.name] = ev
_ESPEC.loader.exec_module(ev)

R = f"{S52_BASE}+M1+M2+M3+M6+J+M8"
OZ = f"{R}+O+Z"
B = pp.BASELINES["B"]
W2 = pp.SPAWN_BOXES["W2"]
CTP = pp.EFFORT_BOUNDS["CTP"]
SLEW = 0.5
#: Terms left out of the per-step totals, as in S7-4's tests: they price the script's motion.
EXCLUDED = ("action_rate_l2", "joint_vel_hinge")
CARRY_HEIGHT = 0.05
SCRIPT_EPISODE_S = 40.0
SPAWN = ((0.5 * sum(W2.x), 0.5 * sum(W2.y)), 0.0)
#: S7-4's hold-closed release: the pads let go, the jaw stays above `JAW_OPEN`.
HALF_OPEN_JAW = 0.18
HALF = 0.025  # the cube's half-width, m


def _robot():
  return kangaroo_grippers_left_arm(
    "G20", KANGAROO_GRIPPERS_LEFT_ARM_WIDE, arm_effort_scale=CTP.effort_scale
  )


def _cfg(arm: str, num_envs: int | None = None):
  cfg = pp.make_pick_place_env_cfg(
    _robot(),
    gripper_name="left",
    reward=arm,
    action_slew_rate_scale=SLEW,
    baseline=B,
    spawn=W2,
    dr="DR-scene",
    effort=CTP,
  )
  if num_envs is not None:
    cfg.scene.num_envs = num_envs
  return cfg


def _resolved(env, arm: str) -> dict:
  rewards = _cfg(arm).rewards
  for cfg in rewards.values():
    for value in cfg.params.values():
      if isinstance(value, SceneEntityCfg):
        value.resolve(env.scene)
  return rewards


# -- Configuration --------------------------------------------------------------------------


def test_oz_is_rctp_plus_o_with_z_on_every_at_goal_term():
  oz, r = _cfg(OZ), _cfg(R)
  assert oz.rewards.keys() == r.rewards.keys() | {"open_at_goal"}
  for name, term in oz.rewards.items():
    if name == "open_at_goal":
      continue
    base = r.rewards[name]
    assert term.func is base.func and term.weight == base.weight, name
    if name in ("release", "placed", "retreat"):
      assert base.params["threshold"] == pp.GOAL_THRESHOLD
      assert term.params["threshold"] == Z_GOAL_TOL, name
      assert {k: v for k, v in term.params.items() if k != "threshold"} == {
        k: v for k, v in base.params.items() if k != "threshold"
      }
    else:
      assert term.params == base.params, name
  o = oz.rewards["open_at_goal"]
  assert o.func is mdp.open_at_goal and o.weight == O_WEIGHT == 1.0
  assert o.params["gate"] == Z_GOAL_TOL == (0.02, 0.02)
  assert o.params["jaw_hold"] == O_JAW_HOLD == 0.40
  assert o.params["jaw_open"] == JAW_OPEN == 0.15
  assert OZ in JAW_RELEASE and OZ in GOAL_TOL_XYZ and R not in GOAL_TOL_XYZ
  assert OZ in NO_SUCCESS_TERMINATION and "success" not in oz.terminations


def test_module_order_is_checked():
  with pytest.raises(ValueError, match="O needs J"):
    _s52_arm(("M1", "M2", "M3", "M6", "O"))
  with pytest.raises(ValueError, match="comes last"):
    _s52_arm(("M1", "M2", "M3", "M6", "J", "Z", "M8"))
  with pytest.raises(ValueError, match="needs M1"):
    _s52_arm(("M3", "Z"))
  _s52_arm(("M1", "M2", "M3", "M6", "J", "M8", "S", "O", "Z"))  # S's term gets Z too


def test_eval_constants_match():
  assert ev.GOAL_TOL_XYZ == Z_GOAL_TOL


def test_a_half_open_hover_earns_less_than_the_floor_of_placing():
  """The most O can pay per step is below what `placed` + `release` pay at M6's floor, and a
  hover with the pads just off a 5 cm cube (~0.21 rad) earns 0.76."""
  assert O_WEIGHT < (M1_PLACED_WEIGHT + M1_RELEASE_WEIGHT) * M6_FLOOR
  assert O_WEIGHT * (O_JAW_HOLD - 0.21) / (O_JAW_HOLD - JAW_OPEN) == pytest.approx(0.76)


def test_at_goal_is_3d_for_a_float_and_xy_z_for_a_pair():
  goal = torch.zeros(6, 3)
  pos = torch.tensor(
    [
      [0.018, 0.0, 0.0],  # xy 1.8 cm: both
      [0.022, 0.0, 0.0],  # xy 2.2 cm: 3-D only -- Z is tighter in xy
      [0.005, 0.0, 0.018],  # a cube on its corner at the target: both (3-D 1.87 cm)
      [0.015, 0.0, 0.019],  # 3-D 2.42 cm: both
      [0.019, 0.0, 0.019],  # 3-D 2.69 cm: Z only -- looser in z
      [0.0, 0.0, -0.021],  # 2.1 cm below: neither (3-D yes: 2.1 < 2.5)
    ]
  )
  assert _at_goal(goal, pos, pp.GOAL_THRESHOLD).tolist() == [1, 1, 1, 1, 0, 1]
  assert _at_goal(goal, pos, Z_GOAL_TOL).tolist() == [1, 0, 1, 1, 1, 0]
  assert _at_goal(goal, pos, list(Z_GOAL_TOL)).tolist() == [1, 0, 1, 1, 1, 0]


# -- O's gate, on set states ----------------------------------------------------------------

#: (lifted, dx, dz, jaw, released) per env, and O's expected value. Tilt is free: env 5 sits
#: with its centre 1.8 cm up (a cube on its corner).
GATE_CASES = (
  ((True, 0.0, 0.0, 0.40, False), 0.0),  # at the hold angle
  ((True, 0.0, 0.0, 0.275, False), 0.5),  # half-way
  ((True, 0.0, 0.0, 0.10, False), 1.0),  # open, capped
  ((False, 0.0, 0.0, 0.10, False), 0.0),  # never lifted (a slide)
  ((True, 0.025, 0.0, 0.10, False), 0.0),  # 2.5 cm off in xy
  ((True, 0.0, 0.018, 0.10, False), 1.0),  # 1.8 cm up
  ((True, 0.0, 0.0, 0.10, True), 0.0),  # after the J-release
  ((True, 0.0, 0.025, 0.10, False), 0.0),  # 2.5 cm up
)


def test_o_pays_only_in_the_gate_and_before_the_release():
  n = len(GATE_CASES)
  env = ManagerBasedRlEnv(cfg=_cfg(OZ, num_envs=n), device=get_test_device())
  try:
    env.reset()
    zero = torch.zeros(env.action_space.shape, device=env.device)
    for _ in range(3):
      env.step(zero)
    term = _resolved(env, OZ)["open_at_goal"]
    robot, obj = env.scene["robot"], env.scene[pp.OBJECT]
    goal = env.command_manager.get_term("goal_pose").command
    lifted = torch.tensor([c[0][0] for c in GATE_CASES], device=env.device)
    pose = obj.data.root_link_pose_w.clone()
    pose[:, :3] = goal
    pose[:, 0] += torch.tensor([c[0][1] for c in GATE_CASES], device=env.device)
    pose[:, 2] += torch.tensor([c[0][2] for c in GATE_CASES], device=env.device)
    obj.write_root_link_pose_to_sim(pose)
    jaw_ids = term.params["jaw_cfg"].joint_ids
    jaw = torch.tensor([[c[0][3]] for c in GATE_CASES], device=env.device)
    robot.write_joint_position_to_sim(jaw, joint_ids=jaw_ids)
    env.sim.forward()
    setattr(env, _WAS_LIFTED, lifted.clone())
    first = term.func(env, **term.params)  # builds the latch at this step
    latch = getattr(env, _RELEASE)
    latch.released[:] = torch.tensor([c[0][4] for c in GATE_CASES], device=env.device)
    value = term.func(env, **term.params)
    assert value.tolist() == pytest.approx([c[1] for c in GATE_CASES], abs=1e-3)
    # Before the latch was edited only env 6's release differs.
    assert first[6] == pytest.approx(1.0, abs=1e-3)
  finally:
    env.close()


# -- Z on a cube that falls flat ------------------------------------------------------------

#: A cube let go tilted this far about an edge, resting on that edge at the target. It stands
#: HALF (cos t + sin t) - HALF up at release (0.7 cm at 20 deg), and falls flat.
TILTS_DEG = (10.0, 20.0, 30.0)


def test_z_places_a_cube_released_tilted_on_an_edge_once_it_has_settled():
  n = len(TILTS_DEG)
  env = ManagerBasedRlEnv(cfg=_cfg(OZ, num_envs=n), device=get_test_device())
  try:
    env.reset()
    zero = torch.zeros(env.action_space.shape, device=env.device)
    for _ in range(40):  # the arm goes home, the jaw opens: nothing touches the cube
      env.step(zero)
    obj = env.scene[pp.OBJECT]
    goal = env.command_manager.get_term("goal_pose").command.clone()
    t = torch.tensor([math.radians(d) for d in TILTS_DEG], device=env.device)
    pose = obj.data.root_link_pose_w.clone()
    pose[:, :3] = goal
    pose[:, 2] += HALF * (torch.cos(t) + torch.sin(t)) - HALF
    pose[:, 3:] = torch.stack(
      [torch.cos(t / 2), torch.zeros_like(t), torch.sin(t / 2), torch.zeros_like(t)], -1
    )
    obj.write_root_link_pose_to_sim(pose)
    obj.write_root_link_velocity_to_sim(torch.zeros_like(obj.data.root_link_vel_w))
    setattr(env, _WAS_LIFTED, torch.ones(n, dtype=torch.bool, device=env.device))
    env.step(zero)
    term = _resolved(env, OZ)["placed"].params
    grasp = term["grasp_sensor_name"]
    jaw = {"jaw_cfg": term["jaw_cfg"], "jaw_open": term["jaw_open"]}
    at = obj.data.root_link_pos_w - goal
    # O's gate holds on the edge, before it falls.
    assert _at_goal(goal, obj.data.root_link_pos_w, Z_GOAL_TOL).all(), at
    for _ in range(75):  # 1.5 s
      env.step(zero)
    pos, vel = obj.data.root_link_pos_w, obj.data.root_link_lin_vel_w
    common = (env, "goal_pose", pp.OBJECT, grasp)
    placed_z = _placed_bool(*common, Z_GOAL_TOL, 0.05, SUPPORT_SENSOR, **jaw)
    shift = torch.norm((pos - goal)[:, :2], dim=-1)
    print("xy shift after falling flat (m):", [round(float(s), 4) for s in shift])
    print("dz after settling (m):", [round(float(d), 4) for d in (pos - goal)[:, 2]])
    assert (torch.norm(vel, dim=-1) < 0.05).all()
    assert (ev.tilt(env) < math.radians(2)).all(), "fell flat"
    assert not _grasped_bool(env, grasp).any()
    assert placed_z.all(), (shift, pos - goal)
  finally:
    env.close()


# -- O on the scripted pick-and-place -------------------------------------------------------


class _Recorder:
  """Per step: OZ's terms, weighted, the jaw angle, the grasp and the J-release latch."""

  def __init__(self):
    self.terms: dict = {}
    self.rec = collections.defaultdict(list)
    self.grasp_sensor = _robot().gripper("left").contact_sensor_name

  def __call__(self, phase, env):
    if not self.terms:
      self.terms = {n: c for n, c in _resolved(env, OZ).items() if n not in EXCLUDED}
    values = {
      n: float(c.weight * c.func(env, **c.params)[0]) for n, c in self.terms.items()
    }
    robot = env.scene["robot"]
    jaw = robot.data.joint_pos[0, robot.joint_names.index(scripted.JAW_JOINT)]
    self.rec[phase].append(
      {
        "values": values,
        "jaw": float(jaw),
        "grasped": bool(_grasped_bool(env, self.grasp_sensor)[0]),
        "released": bool(getattr(env, _RELEASE).released[0]),
      }
    )


def _run(phases):
  rec = _Recorder()
  saved = scripted.CLEAR_HEIGHT, scripted.RELEASE_HEIGHT
  scripted.CLEAR_HEIGHT, scripted.RELEASE_HEIGHT = CARRY_HEIGHT, -0.002
  try:
    scripted.rollout(
      num_envs=1,
      device=get_test_device(),
      reward=OZ,
      verbose=False,
      hook=rec,
      action_slew_rate_scale=SLEW,
      robot=_robot(),
      baseline=B,
      spawn_box=W2,
      effort=CTP,
      spawns=[SPAWN],
      targets=[B.target_xy],
      episode_length_s=SCRIPT_EPISODE_S,
      phases=phases,
    )
  finally:
    scripted.CLEAR_HEIGHT, scripted.RELEASE_HEIGHT = saved
  return rec.rec


_UP_TO_HOLD = scripted.PHASES[: [p[0] for p in scripted.PHASES].index("hold") + 1]


@pytest.fixture(scope="module")
def clean():
  """Sets the cube down on the target, opens the jaw and lifts clear."""
  return _run(scripted.PHASES + (("settle_more", "retreat", False, 40),))


@pytest.fixture(scope="module")
def half_open():
  """Sets the cube down and opens the jaw only to `HALF_OPEN_JAW`: the pads let go, the jaw
  stays above `JAW_OPEN` and the hand stays -- a hover O pays, J does not."""
  frac = HALF_OPEN_JAW / _robot().gripper("left").closed_pos
  return _run(
    _UP_TO_HOLD
    + (
      ("open", "place", frac, 30),
      ("stay", "place", frac, 60),
      ("settle", "place", frac, 40),
    )
  )


def _total(step: dict) -> float:
  return sum(step["values"].values())


def test_o_is_zero_while_carrying_and_small_while_holding_closed(clean):
  for phase in ("carry", "steady"):
    assert all(s["values"]["open_at_goal"] == 0.0 for s in clean[phase]), phase
  # Holding the cube on the target, jaw at the script's hold angle: O's formula, and well
  # under half of what opening pays.
  held = clean["hold"][-1]
  assert held["grasped"] and held["jaw"] > 0.27, held
  want = O_WEIGHT * max(O_JAW_HOLD - held["jaw"], 0.0) / (O_JAW_HOLD - JAW_OPEN)
  assert held["values"]["open_at_goal"] == pytest.approx(want, abs=1e-3)
  assert held["values"]["open_at_goal"] < 0.5 * O_WEIGHT


def test_o_rises_as_the_jaw_opens_and_stops_at_the_release(clean):
  steps = clean["open"] + clean["retreat"] + clean["settle"] + clean["settle_more"]
  k = next(i for i, s in enumerate(steps) if s["released"])
  before = [s["values"]["open_at_goal"] for s in steps[:k]]
  assert max(before) > 0.5 * O_WEIGHT, before
  assert all(s["values"]["open_at_goal"] == 0.0 for s in steps[k:])
  assert steps[-1]["values"]["placed"] > 0


def test_a_half_open_hover_earns_less_per_step_than_placing(clean, half_open):
  hover = half_open["settle"][-1]
  assert not hover["grasped"] and not hover["released"], hover
  assert hover["jaw"] >= JAW_OPEN
  o = hover["values"]["open_at_goal"]
  want = O_WEIGHT * (O_JAW_HOLD - hover["jaw"]) / (O_JAW_HOLD - JAW_OPEN)
  assert o == pytest.approx(want, abs=1e-3) and o > 0.5
  assert hover["values"]["placed"] == 0 and hover["values"]["release"] == 0
  placed = clean["settle_more"][-1]
  print(f"per step: hover {_total(hover):.2f} (O {o:.2f}), placed {_total(placed):.2f}")
  assert _total(placed) > _total(hover) + 5.0
