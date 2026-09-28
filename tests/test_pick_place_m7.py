"""Plan S7-2, module M7: M6's release grade paid only for a set-down.

B-S = baseline B + ``R-C-strict+M1+M2+M3+M6+M7``. FR (M6) lets go at the target but, over
training, while the cube still moves (0.10 -> 0.44 m/s, S7 report section 5.5): M6 grades where
and how high the release is and gives the first centimetre of height free. M7 pays the grade
only when, at the step the grasp opens, the cube rests on the table, moves slower than
0.05 m/s, and has been held there, still and within 1.5 cm of the target, for 3 control steps
since a touchdown within 1.5 cm (`mdp.set_down_passed`, `observations._release_latch`); any
other release is graded at the floor. `held_slide` prices dragging a held cube on the table.

What the plan asks before GPU time, on states the scripted pick-and-place reaches through the
policy's action interface under baseline B (a held cube cannot be teleported into the jaw):

* a set-down that dwells beats a release while lowering and a release 6 cm short, per step and
  in discounted return (gamma 0.99) over B's 10 s episode -- crediting the others with the
  steps they save;
* a cube put down short and dragged to the target is not paid, and the drag is priced.

The returns put the set-down's release at 1 s into a 500-step episode (the trained policies
place in ~0.8 s; the script, slow and closed-loop on purpose, takes ~8 s and runs in a longer
episode), so the comparison has the tail a policy would have.
"""

import collections
import importlib.util
import pathlib
import sys
import types

import numpy as np
import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import pytest
import torch
from conftest import get_test_device
from mjlab.managers.scene_entity_config import SceneEntityCfg
from pal_mjlab.tasks.manipulation import mdp
from pal_mjlab.tasks.manipulation.mdp.observations import (
  _RELEASE,
  SET_DOWN_RADIUS,
  STILL_SPEED,
  _grasped_bool,
  _on_surface_bool,
)
from pal_mjlab.tasks.manipulation.mdp.rewards import release_quality_of
from pal_mjlab.tasks.manipulation.pick_place import pick_place_env_cfg as pp
from pal_mjlab.tasks.manipulation.pick_place.rewards import (
  M6_DIST_STD,
  M6_FLOOR,
  M6_HEIGHT_FREE,
  M6_HEIGHT_STD,
  M7_DWELL_STEPS,
  M7_SLIDE_WEIGHT,
  NO_SUCCESS_TERMINATION,
  S52_BASE,
  _s52_arm,
)
from pal_mjlab.tasks.manipulation.rl_cfg import stack_ppo_runner_cfg
from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS_LEFT_ARM
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import CONTROL_DT, SUPPORT_SENSOR

_SCRIPT = pathlib.Path(__file__).parent.parent / "scripts/scripted_pick_place.py"
_SPEC = importlib.util.spec_from_file_location("scripted_pick_place_m7", _SCRIPT)
assert _SPEC and _SPEC.loader
scripted = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = scripted
_SPEC.loader.exec_module(scripted)

FR = f"{S52_BASE}+M1+M2+M3+M6"
FRS = f"{S52_BASE}+M1+M2+M3+M6+M7"
B = pp.BASELINES["B"]
SLEW = 0.5
TERMS = (
  "reach",
  "grasp",
  "lift",
  "goal",
  "goal_near",
  "release",
  "placed",
  "held_slide",
)
GAMMA = stack_ppo_runner_cfg("x").algorithm.gamma
#: B's episode, in control steps: 10 s at 50 Hz.
EPISODE_STEPS = round(B.episode_length_s / CONTROL_DT)
#: Where the set-down lets go in the returns: 1 s in, as a trained policy does.
RELEASE_AT = 50
#: The plan's estimate of what letting go early saves: 3-5 control steps (plan S7, M6).
SAVED_STEPS = 5
SHORT = 0.06
DRAG = 0.05
#: The script's carry height, m, and the episode it runs in, s (see the module docstring).
CARRY_HEIGHT = 0.05
SCRIPT_EPISODE_S = 40.0
SUBSTEP_TOL = 1e-3


def _q(dxy: float, height: float, height_free: float) -> float:
  return float(
    release_quality_of(
      torch.tensor(dxy),
      torch.tensor(height),
      M6_DIST_STD,
      height_free,
      M6_HEIGHT_STD,
      M6_FLOOR,
    )
  )


# -- The arm ------------------------------------------------------------------------------


def _cfg(arm: str):
  return pp.make_pick_place_env_cfg(
    KANGAROO_GRIPPERS_LEFT_ARM,
    gripper_name="left",
    reward=arm,
    action_slew_rate_scale=SLEW,
    baseline=B,
  )


def test_m7_is_fr_with_the_set_down_and_the_slide_price():
  frs, fr = _cfg(FRS), _cfg(FR)
  assert frs.rewards.keys() == fr.rewards.keys() | {"held_slide"}
  for name in ("release", "placed"):
    extra = {
      k: v
      for k, v in frs.rewards[name].params.items()
      if fr.rewards[name].params.get(k) != v
    }
    assert extra == {"height_free": 0.0, "set_down_steps": M7_DWELL_STEPS}, name
    assert frs.rewards[name].func is fr.rewards[name].func
    assert frs.rewards[name].weight == fr.rewards[name].weight
  slide = frs.rewards["held_slide"]
  assert slide.func is mdp.held_object_slide_speed and slide.weight == M7_SLIDE_WEIGHT
  assert FRS in NO_SUCCESS_TERMINATION and "success" not in frs.terminations
  assert M7_DWELL_STEPS == 3 and SET_DOWN_RADIUS == 0.015 and STILL_SPEED == 0.05


def test_m7_needs_m6_before_it():
  with pytest.raises(ValueError, match="needs M6"):
    _s52_arm(("M1", "M2", "M3", "M7"))
  with pytest.raises(ValueError, match="needs M6"):
    _s52_arm(("M1", "M2", "M3", "M7", "M6"))


def test_the_set_down_rule():
  """`set_down_passed` on hand-built latches: every condition is necessary."""

  def latch(**kw):
    base = dict(released=True, rel_on_table=True, rel_speed=0.01, rel_dwell=3)
    base.update(kw)
    return types.SimpleNamespace(**{k: torch.tensor([v]) for k, v in base.items()})

  assert bool(mdp.rewards.set_down_passed(latch(), 3))
  assert not bool(mdp.rewards.set_down_passed(latch(released=False), 3))
  assert not bool(mdp.rewards.set_down_passed(latch(rel_on_table=False), 3))
  assert not bool(mdp.rewards.set_down_passed(latch(rel_speed=0.06), 3))
  assert not bool(mdp.rewards.set_down_passed(latch(rel_dwell=2), 3))


# -- Scripted episodes --------------------------------------------------------------------


class _Recorder:
  """Per step and env: the task terms of FR and of B-S, weighted, with the grasp and table
  contact, the cube's speed and geometry, and the release latch."""

  def __init__(self):
    self.terms: dict = {}
    self.rec = collections.defaultdict(lambda: collections.defaultdict(list))
    self.grasp_sensor = KANGAROO_GRIPPERS_LEFT_ARM.gripper("left").contact_sensor_name

  def __call__(self, phase, env):
    if not self.terms:
      for arm in (FR, FRS):
        rewards = _cfg(arm).rewards
        for cfg in rewards.values():
          for value in cfg.params.values():
            if isinstance(value, SceneEntityCfg):
              value.resolve(env.scene)
        self.terms[arm] = {n: c for n, c in rewards.items() if n in TERMS}
    values = {
      arm: {n: (c.weight * c.func(env, **c.params)).tolist() for n, c in ts.items()}
      for arm, ts in self.terms.items()
    }
    latch = getattr(env, _RELEASE)
    obj = env.scene[pp.OBJECT]
    goal = env.command_manager.get_term("goal_pose").command
    pos = obj.data.root_link_pos_w
    dxy = torch.norm((pos - goal)[:, :2], dim=-1).tolist()
    dz = (pos[:, 2] - goal[:, 2]).tolist()
    speed = torch.norm(obj.data.root_link_lin_vel_w, dim=-1).tolist()
    grasped = _grasped_bool(env, self.grasp_sensor).tolist()
    on_table = _on_surface_bool(env, SUPPORT_SENSOR).tolist()
    for i in range(env.num_envs):
      self.rec[i][phase].append(
        {
          "values": {a: {n: v[i] for n, v in vs.items()} for a, vs in values.items()},
          "grasped": grasped[i],
          "on_table": on_table[i],
          "speed": speed[i],
          "dxy": dxy[i],
          "dz": dz[i],
          "released": bool(latch.released[i]),
          "latch_dxy": float(latch.dxy[i]),
          "latch_height": float(latch.height[i]),
          "rel_on_table": bool(latch.rel_on_table[i]),
          "rel_speed": float(latch.rel_speed[i]),
          "rel_dwell": int(latch.rel_dwell[i]),
        }
      )


def _flat(rec_env: dict) -> list[dict]:
  return [s | {"phase": ph} for ph, steps in rec_env.items() for s in steps]


def _total(state: dict, arm: str) -> float:
  return sum(state["values"][arm].values())


def _releases(steps: list[dict]) -> list[int]:
  lifted, out = False, []
  for k in range(1, len(steps)):
    lifted |= steps[k - 1]["grasped"] and not steps[k - 1]["on_table"]
    if lifted and steps[k - 1]["grasped"] and not steps[k]["grasped"]:
      out.append(k)
  return out


def _passed(step: dict) -> bool:
  return (
    step["released"]
    and step["rel_on_table"]
    and step["rel_speed"] < STILL_SPEED
    and step["rel_dwell"] >= M7_DWELL_STEPS
  )


def _grade(step: dict, arm: str) -> float:
  """The grade the latch holds for ``arm``, from the recorded latch."""
  if not step["released"]:
    return M6_FLOOR
  if arm == FRS:
    q = _q(step["latch_dxy"], step["latch_height"], 0.0)
    return q if _passed(step) else M6_FLOOR
  return _q(step["latch_dxy"], step["latch_height"], M6_HEIGHT_FREE)


def _short_offset(length: float) -> tuple[float, float]:
  d = np.asarray(pp.SPAWN_XY) - np.asarray(B.target_xy)
  d = length * d / np.linalg.norm(d)
  return float(d[0]), float(d[1])


def _run(num_envs, hook, release_height, **kw):
  saved = scripted.CLEAR_HEIGHT, scripted.RELEASE_HEIGHT
  scripted.CLEAR_HEIGHT, scripted.RELEASE_HEIGHT = CARRY_HEIGHT, release_height
  try:
    return scripted.rollout(
      num_envs=num_envs,
      device=get_test_device(),
      reward=FRS,
      verbose=False,
      hook=hook,
      action_slew_rate_scale=SLEW,
      baseline=B,
      spawns=[(pp.SPAWN_XY, 0.0)] * num_envs,
      targets=[B.target_xy] * num_envs,
      episode_length_s=SCRIPT_EPISODE_S,
      **kw,
    )
  finally:
    scripted.CLEAR_HEIGHT, scripted.RELEASE_HEIGHT = saved


@pytest.fixture(scope="module")
def pair():
  """Env 0 sets the cube down on the target and holds it there before opening -- the script
  presses 2 mm into the table (`RELEASE_HEIGHT` -0.002) so the held cube rests on it. Env 1 sets
  it down `SHORT` before the target; once its arm has retreated, its cube is moved onto the
  target at rest -- the slide that brings a short release in -- so both end placed."""
  rec = _Recorder()
  moved = {"done": False}

  def hook(phase, env):
    if phase == "settle" and not moved["done"]:
      obj = env.scene[pp.OBJECT]
      ids = torch.tensor([1], device=env.device)
      pose = obj.data.root_link_pose_w.clone()
      pose[1, :3] = env.command_manager.get_term("goal_pose").command[1]
      obj.write_root_link_pose_to_sim(pose[1:2], env_ids=ids)
      obj.write_root_link_velocity_to_sim(
        torch.zeros(1, 6, device=env.device), env_ids=ids
      )
      moved["done"] = True
    rec(phase, env)

  rows = _run(
    2,
    hook,
    -0.002,
    release_offsets=[(0.0, 0.0), _short_offset(SHORT)],
    phases=scripted.PHASES + (("settle_more", "retreat", False, 40),),
  )
  assert moved["done"]
  return rows, rec.rec


#: `lowering`: the jaw's command while lowering (0.9 of the full close, 0.70 rad: it still holds
#: through a fast descent -- at 0.5-0.7 the cube slips out 2-3 cm up), and the place pose it
#: lowers to, 2 cm *below* the resting height, so the arm is still coming down when the jaw
#: lets go. Tuned on the script (2026-09-26): this lets go 0.9 cm from the target, 0.2 cm
#: above rest, at 0.34 m/s -- FR's drift (0.3-0.6 cm, <= 0.7 cm up, 0.10-0.44 m/s).
LOWER_GRIP = 0.9
LOWER_TO = -0.02


@pytest.fixture(scope="module")
def lowering():
  """The arm lowers the held cube onto the target in 8 control steps and the jaw opens as the
  command arrives, while the arm, lagging it, is still coming down: the cube is let go just
  above the table and moving -- FR's drift."""
  rec = _Recorder()
  phases = scripted.PHASES[:6] + (
    ("lower", "place", LOWER_GRIP, 8),
    ("open", "place", False, 30),
    ("retreat", "retreat", False, 40),
    ("settle", "retreat", False, 60),
  )
  rows = _run(1, rec, LOWER_TO, phases=phases)
  return rows, rec.rec


@pytest.fixture(scope="module")
def drag():
  """The cube set down on the table `DRAG` short of the target, then dragged along it to the
  target while held, held still there, and let go."""
  rec = _Recorder()
  phases = scripted.PHASES[: [p[0] for p in scripted.PHASES].index("hold") + 1] + (
    ("drag", scripted.DRAG_WAYPOINT, True, 60),
    ("dwell", scripted.DRAG_WAYPOINT, True, 20),
    ("open", scripted.DRAG_WAYPOINT, False, 30),
    ("settle", scripted.DRAG_WAYPOINT, False, 40),
  )
  rows = _run(1, rec, -0.002, release_offsets=[_short_offset(DRAG)], phases=phases)
  return rows, rec.rec


def test_the_set_down_passes_and_is_graded(pair):
  steps = _flat(pair[1][0])
  rel = _releases(steps)
  assert len(rel) == 1, rel
  s = steps[rel[0]]
  assert s["rel_on_table"] and s["rel_speed"] < STILL_SPEED, s
  assert s["rel_dwell"] >= M7_DWELL_STEPS, s["rel_dwell"]
  assert _passed(s)
  assert s["latch_dxy"] == pytest.approx(s["dxy"], abs=SUBSTEP_TOL)
  assert _grade(steps[-1], FRS) > 0.8


def test_a_release_short_of_the_target_gets_the_floor(pair):
  """Set down on the table 6 cm short: on the table and still, but its touchdown was 6 cm out,
  so it never builds a dwell -- and M6's own grade of 6 cm is the floor anyway."""
  steps = _flat(pair[1][1])
  s = steps[_releases(steps)[0]]
  assert s["rel_on_table"] and s["rel_dwell"] == 0, s
  assert abs(s["dxy"] - SHORT) < 0.02
  assert _grade(steps[-1], FRS) == pytest.approx(M6_FLOOR)


def test_a_release_while_lowering_gets_the_floor_where_m6_paid_it(lowering):
  rows, rec = lowering
  steps = _flat(rec[0])
  s = steps[_releases(steps)[0]]
  assert not _passed(s), s
  assert not s["rel_on_table"] and s["rel_speed"] > 0.2, s
  assert s["dxy"] < SET_DOWN_RADIUS, "let go over the target"
  assert s["latch_height"] < M6_HEIGHT_FREE, "inside M6's free centimetre"
  # Positive control: M6 grades the same release well above the floor.
  assert _grade(steps[-1], FR) > M6_FLOOR + 0.2, _grade(steps[-1], FR)
  assert _grade(steps[-1], FRS) == pytest.approx(M6_FLOOR)
  assert rows[0]["placed_and_lifted"], "the cube still lands on the target"


def test_a_drag_is_not_paid_and_is_priced(drag):
  rows, rec = drag
  steps = _flat(rec[0])
  rel = _releases(steps)
  s = steps[rel[-1]]
  assert s["dxy"] < SET_DOWN_RADIUS, f"the drag ended {s['dxy']:.3f} m out"
  assert s["rel_on_table"] and s["rel_speed"] < STILL_SPEED, s
  assert s["rel_dwell"] == 0, "a dwell built on a touchdown 5 cm out"
  assert _grade(steps[-1], FRS) == pytest.approx(M6_FLOOR)
  # Positive control: M6 would pay this release, set down on the target and still.
  assert _grade(steps[-1], FR) > 0.6
  slide = [x["values"][FRS]["held_slide"] for x in rec[0]["drag"]]
  assert sum(slide) < -0.05, f"the drag cost {sum(slide):.3f}"
  assert all(v <= 0.0 for x in steps for v in [x["values"][FRS]["held_slide"]])
  assert rows[0]["placed_and_lifted"]


def _settled(rec_env: dict) -> dict:
  s = _flat(rec_env)[-1]
  assert s["values"][FRS]["placed"] > 0 and s["speed"] < 0.05, s
  return s


def test_after_settling_the_set_down_earns_more_per_step(pair, lowering):
  down = _total(_settled(pair[1][0]), FRS)
  short = _total(_settled(pair[1][1]), FRS)
  low = _total(_settled(lowering[1][0]), FRS)
  assert down > short + 2.0, (down, short)
  assert down > low + 2.0, (down, low)


def _returns(down_env: dict, alt_settled: float, arm: str, saved: int):
  """Discounted returns over B's episode of the set-down and of an alternative that lets go
  ``saved`` steps sooner and is then worth ``alt_settled`` per step -- the M6 test's
  construction, with the set-down letting go `RELEASE_AT` steps in."""
  steps = _flat(down_env)
  t_rel = _releases(steps)[0]
  seq = [_total(s, arm) for s in steps[t_rel - RELEASE_AT :]]
  seq = (seq + [seq[-1]] * EPISODE_STEPS)[:EPISODE_STEPS]
  alt = seq[: RELEASE_AT - saved] + [alt_settled] * (EPISODE_STEPS - RELEASE_AT + saved)
  g = GAMMA ** np.arange(EPISODE_STEPS)
  return float(g @ np.asarray(seq)), float(g @ np.asarray(alt))


@pytest.mark.parametrize("saved", range(SAVED_STEPS + 1))
@pytest.mark.parametrize("alt", ["short", "lowering"])
def test_the_set_down_has_the_higher_discounted_return(pair, lowering, saved, alt):
  settled = _settled(pair[1][1] if alt == "short" else lowering[1][0])
  down, other = _returns(pair[1][0], _total(settled, FRS), FRS, saved)
  assert down > other, f"{alt}, saving {saved} steps: {down:.1f} vs {other:.1f}"


def test_the_break_even_is_far_beyond_the_steps_saved(pair, lowering):
  """How many steps a release while lowering would have to save to pay: many under B-S, and
  more than under FR, where M6 paid it."""

  def break_even(arm):
    settled = _total(_settled(lowering[1][0]), arm)
    for k in range(RELEASE_AT + 1):
      down, other = _returns(pair[1][0], settled, arm, k)
      if other >= down:
        return k
    return RELEASE_AT

  frs, fr = break_even(FRS), break_even(FR)
  assert frs > 4 * SAVED_STEPS, frs
  assert frs > fr, (frs, fr)
