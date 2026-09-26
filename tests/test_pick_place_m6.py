"""Plan S7, module M6: the full fix's `release` and `placed` graded by where the cube was let go.

FR = R-C-strict+M1+M2+M3+M6, trained at FA's slew (0.5). The grade of the latest release after
`was_lifted` is q = 0.25 + 0.75 * G_0.02(d_xy) * G_0.01(max(h - 0.01, 0)), with d_xy the
cube's horizontal distance to the target and h its height above rest at the step the grasp
opened (`pick_place/rewards.py`, `mdp/observations._release_latch`). `goal` and `goal_near`
are not graded.

Why: S5.2's full fix places 64/64 on three seeds, but two of them let go 5-6 cm short and let
the cube slide or fly in, which `placed` counts; under M1 every step saved is worth ~22, so the
early release was the per-step optimum (S5.2 report section 6). What the plan asks of M6
before GPU time, on states the scripted pick-and-place reaches through the policy's action
interface at slew 0.5 (a held cube cannot be teleported into the jaw):

* the S5.2 per-step ordering still holds on the set-down path;
* a set-down beats a release 6 cm short: more per step once both have settled, and a higher
  discounted return (gamma 0.99) -- even crediting the short release with the steps it saves;
* repeated grasp/release cannot push q above a single good release: the latest one counts;
* a slid cube earns nothing past reach;
* the scripted pick-and-place still reaches `placed`.
"""

import collections
import importlib.util
import math
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
  _WAS_LIFTED,
  _grasped_bool,
  _on_surface_bool,
)
from pal_mjlab.tasks.manipulation.mdp.rewards import release_quality_of
from pal_mjlab.tasks.manipulation.pick_place import pick_place_env_cfg as pp
from pal_mjlab.tasks.manipulation.pick_place.rewards import (
  M1_PLACED_WEIGHT,
  M1_RELEASE_WEIGHT,
  M6_DIST_STD,
  M6_FLOOR,
  M6_HEIGHT_FREE,
  M6_HEIGHT_STD,
  NO_SUCCESS_TERMINATION,
  REWARD_VARIANTS,
  S52_BASE,
  _s52_arm,
)
from pal_mjlab.tasks.manipulation.rl_cfg import stack_ppo_runner_cfg
from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS_LEFT_ARM
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import CONTROL_DT, SUPPORT_SENSOR

_SCRIPT = pathlib.Path(__file__).parent.parent / "scripts/scripted_pick_place.py"
_SPEC = importlib.util.spec_from_file_location("scripted_pick_place_m6", _SCRIPT)
assert _SPEC and _SPEC.loader
scripted = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = scripted
_SPEC.loader.exec_module(scripted)

FR = f"{S52_BASE}+M1+M2+M3+M6"
FULL = f"{S52_BASE}+M1+M2+M3"
#: FR trains at speed bound A's slew, so its tests run there too.
SLEW = 0.5
#: How far short of the target the S5.2 full-fix seeds 2-3 let go (S5.2 report section 6).
SHORT = 0.06
#: c1a's hover height, the carry height of the S5.2 ordering test.
CARRY_HEIGHT = 0.07
TASK_TERMS = ("reach", "grasp", "lift", "goal", "goal_near", "release", "placed")
ORDER = ("reach", "carry", "over_target", "held_on_target", "released", "placed")
GAMMA = stack_ppo_runner_cfg("x").algorithm.gamma
#: Control steps in a training episode (20 s at 50 Hz); a finished place keeps paying to here.
EPISODE_STEPS = round(pp.DEFAULT_EPISODE_LENGTH_S / CONTROL_DT)
#: The plan's estimate of what letting go short saves: 3-5 control steps (plan S7, M6).
SAVED_STEPS = 5
#: The latch reads the object where the reward terms see it, which is one 2 ms physics substep
#: behind where a hook sees it after the step (mjlab re-runs the kinematics at the end of
#: `step`); 1 mm covers that at up to 0.5 m/s. Measured: 7.01 vs 7.09 mm at a set-down.
SUBSTEP_TOL = 1e-3


def _q(dxy: float, height: float) -> float:
  return float(
    release_quality_of(
      torch.tensor(dxy),
      torch.tensor(height),
      M6_DIST_STD,
      M6_HEIGHT_FREE,
      M6_HEIGHT_STD,
      M6_FLOOR,
    )
  )


def _cfg(arm: str, num_envs: int | None = None):
  cfg = pp.make_pick_place_env_cfg(
    KANGAROO_GRIPPERS_LEFT_ARM,
    gripper_name="left",
    reward=arm,
    action_slew_rate_scale=SLEW,
  )
  if num_envs is not None:
    cfg.scene.num_envs = num_envs
  return cfg


def _task_terms(env, arm: str) -> dict:
  rewards = _cfg(arm).rewards
  for cfg in rewards.values():
    for value in cfg.params.values():
      if isinstance(value, SceneEntityCfg):
        value.resolve(env.scene)
  return {name: cfg for name, cfg in rewards.items() if name in TASK_TERMS}


# -- The grade ----------------------------------------------------------------------------


def test_the_grade_is_the_plans_formula():
  """The plan's numbers: a set-down like S5.2 seed 1's (0.8 cm off, 0.4 cm up) is ~0.9, a
  release 6 cm short is at the floor, and the first centimetre of height is free."""
  assert _q(0.008, 0.004) == pytest.approx(0.25 + 0.75 * math.exp(-0.16), abs=1e-6)
  assert _q(0.008, 0.004) > 0.88
  assert _q(SHORT, 0.0) == pytest.approx(M6_FLOOR, abs=1e-3)
  assert _q(0.0, 0.0) == pytest.approx(1.0) and _q(0.0, 0.01) == pytest.approx(1.0)
  assert _q(0.0, 0.02) == pytest.approx(0.25 + 0.75 * math.exp(-1.0), abs=1e-6)
  dxy = [_q(d, 0.0) for d in (0.0, 0.01, 0.02, 0.04, 0.08)]
  up = [_q(0.0, h) for h in (0.01, 0.015, 0.02, 0.03, 0.05)]
  assert all(a > b for a, b in zip(dxy, dxy[1:], strict=False))
  assert all(a > b for a, b in zip(up, up[1:], strict=False))
  assert min(dxy + up) >= M6_FLOOR


# -- The arm ------------------------------------------------------------------------------


def test_fr_is_the_full_fix_with_release_and_placed_graded():
  fr, full = _cfg(FR), _cfg(FULL)
  assert fr.rewards.keys() == full.rewards.keys()
  for name, term in fr.rewards.items():
    if name in ("release", "placed"):
      assert term.weight == full.rewards[name].weight
      extra = set(term.params) - set(full.rewards[name].params)
      assert extra == {"surface_cfg", "dist_std", "height_free", "height_std", "floor"}
      assert term.params["support_sensor_name"]  # still gated on was_lifted
    else:  # goal and goal_near in particular are not graded
      assert term.func is full.rewards[name].func, name
      assert term.weight == full.rewards[name].weight, name
  assert fr.rewards["release"].func is mdp.released_at_goal_graded
  assert fr.rewards["placed"].func is mdp.object_placed_graded
  assert fr.rewards["release"].weight == M1_RELEASE_WEIGHT
  assert fr.rewards["placed"].weight == M1_PLACED_WEIGHT
  assert FR in NO_SUCCESS_TERMINATION and "success" not in fr.terminations


def test_m6_needs_m1_before_it():
  with pytest.raises(ValueError, match="needs M1"):
    _s52_arm(("M2", "M3", "M6"))
  with pytest.raises(ValueError, match="needs M1"):
    _s52_arm(("M6", "M1"))


@pytest.mark.parametrize("arm", sorted(REWARD_VARIANTS))
def test_every_arm_resets_the_release_latch(arm):
  """Registered unconditionally, as `reset_was_lifted` is: an eval or test that reads the
  latch on any arm must not be graded by the previous episode's release."""
  cfg = pp.make_pick_place_env_cfg(
    KANGAROO_GRIPPERS_LEFT_ARM, gripper_name="left", reward=arm
  )
  assert cfg.events["reset_release_latch"].func is mdp.reset_release_latch
  assert cfg.events["reset_release_latch"].mode == "reset"


# -- A slid cube --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def slid_env():
  """The cube pushed onto the target along the table, never lifted: at rest on the goal, not
  held, on the table. Stepped after the teleport, since contact sensors are empty until a
  step."""
  env = ManagerBasedRlEnv(cfg=_cfg(FR, num_envs=2), device=get_test_device())
  env.reset()
  zero = torch.zeros(env.action_space.shape, device=env.device)
  for _ in range(3):
    env.step(zero)
  obj = env.scene[pp.OBJECT]
  pose = obj.data.root_link_pose_w.clone()
  pose[:, :3] = env.command_manager.get_term("goal_pose").command
  obj.write_root_link_pose_to_sim(pose)
  obj.write_root_link_velocity_to_sim(torch.zeros_like(obj.data.root_link_vel_w))
  env.step(zero)
  yield env
  env.close()


def test_a_slid_cube_earns_nothing_past_reach(slid_env):
  assert not bool(getattr(slid_env, _WAS_LIFTED).any()), "the cube was never lifted"
  assert not bool(getattr(slid_env, _RELEASE).released.any()), "nor let go"
  for name, cfg in _task_terms(slid_env, FR).items():
    value = cfg.func(slid_env, **cfg.params)
    if name == "reach":
      assert torch.all(value > 0.0)
    else:
      assert torch.all(value == 0.0), f"{name} paid a slide: {value}"


def test_the_same_cube_pays_its_grade_once_lifted_and_let_go(slid_env):
  """Positive control: a term that returns zero because it is broken also 'passes' the slide
  test. With the latches set as a release 5 mm off and 2 mm up would set them, `release` and
  `placed` pay exactly that release's grade."""
  was_lifted = getattr(slid_env, _WAS_LIFTED)
  latch = getattr(slid_env, _RELEASE)
  saved = (was_lifted.clone(), latch.released.clone(), latch.dxy, latch.height)
  was_lifted[:] = True
  latch.released[:] = True
  latch.dxy = torch.full_like(latch.dxy, 0.005)
  latch.height = torch.full_like(latch.height, 0.002)
  try:
    terms = _task_terms(slid_env, FR)
    for name in ("release", "placed"):
      value = terms[name].func(slid_env, **terms[name].params)
      assert torch.allclose(value, torch.full_like(value, _q(0.005, 0.002))), name
  finally:
    was_lifted[:] = saved[0]
    latch.released[:] = saved[1]
    latch.dxy, latch.height = saved[2], saved[3]


def test_the_reset_event_clears_the_latch(slid_env):
  latch = getattr(slid_env, _RELEASE)
  saved = (latch.grasped.clone(), latch.released.clone(), latch.dxy, latch.height)
  latch.grasped[:] = True
  latch.released[:] = True
  latch.dxy = torch.full_like(latch.dxy, 0.03)
  latch.height = torch.full_like(latch.height, 0.02)
  try:
    mdp.reset_release_latch(slid_env, torch.tensor([1], device=slid_env.device))
    assert bool(latch.released[0]) and not bool(latch.released[1])
    assert bool(latch.grasped[0]) and not bool(latch.grasped[1])
    assert float(latch.dxy[1]) == 0.0 and float(latch.height[1]) == 0.0
  finally:
    latch.grasped[:] = saved[0]
    latch.released[:] = saved[1]
    latch.dxy, latch.height = saved[2], saved[3]


# -- A set-down and a release 6 cm short, from the same spawn -----------------------------


def _short_offset() -> tuple[float, float]:
  """`SHORT` back along the carry, from the target toward the spawn: where the cube is when
  a policy that lets go early lets go."""
  d = np.asarray(pp.SPAWN_XY) - np.asarray(pp.TARGET_XY)
  d = SHORT * d / np.linalg.norm(d)
  return float(d[0]), float(d[1])


class _Recorder:
  """Per step and env: every weighted task term of FR and of the full fix, the grasp and
  table contact, the cube's speed and geometry, and the release latch."""

  def __init__(self):
    self.terms: dict = {}
    self.rec = collections.defaultdict(lambda: collections.defaultdict(list))
    self.table = None
    self.grasp_sensor = KANGAROO_GRIPPERS_LEFT_ARM.gripper("left").contact_sensor_name

  def geometry(self, env):
    """The cube's horizontal distance to the goal and its height above rest, from the scene
    itself: an independent check on what the latch records."""
    obj = env.scene[pp.OBJECT]
    goal = env.command_manager.get_term("goal_pose").command
    pos = obj.data.root_link_pos_w
    half_z = env.sim.model.geom_size[:, obj.indexing.geom_ids[0], 2]
    rest = mdp.surface_top_z(env, self.table) + half_z
    return torch.norm((pos - goal)[:, :2], dim=-1), pos[:, 2] - rest

  def __call__(self, phase, env):
    if not self.terms:
      self.terms.update({arm: _task_terms(env, arm) for arm in (FR, FULL)})
      self.table = SceneEntityCfg(pp.TABLE)
      self.table.resolve(env.scene)
    values = {
      arm: {n: (c.weight * c.func(env, **c.params)).tolist() for n, c in ts.items()}
      for arm, ts in self.terms.items()
    }
    latch = getattr(env, _RELEASE)
    dxy, height = self.geometry(env)
    grasped = _grasped_bool(env, self.grasp_sensor).tolist()
    on_table = _on_surface_bool(env, SUPPORT_SENSOR).tolist()
    speed = torch.norm(env.scene[pp.OBJECT].data.root_link_lin_vel_w, dim=-1).tolist()
    for i in range(env.num_envs):
      self.rec[i][phase].append(
        {
          "values": {
            arm: {n: v[i] for n, v in vs.items()} for arm, vs in values.items()
          },
          "grasped": grasped[i],
          "on_table": on_table[i],
          "speed": speed[i],
          "dxy": float(dxy[i]),
          "height": float(height[i]),
          "released": bool(latch.released[i]),
          "latch_dxy": float(latch.dxy[i]),
          "latch_height": float(latch.height[i]),
        }
      )


def _latched_q(step: dict) -> float:
  return _q(step["latch_dxy"], step["latch_height"]) if step["released"] else M6_FLOOR


def _flat(rec_env: dict) -> list[dict]:
  """One env's steps in episode order, each tagged with its phase."""
  return [s | {"phase": ph} for ph, steps in rec_env.items() for s in steps]


def _total(state: dict, arm: str) -> float:
  return sum(state["values"][arm].values())


@pytest.fixture(scope="module")
def pair():
  """One scripted episode per env from the spawn centre under FR at slew 0.5, carried at c1a's
  hover height: env 0 sets the cube down on the target, env 1 sets it down `SHORT` before
  it. Once env 1's arm has retreated, its cube is moved onto the target at rest -- the slide
  or flight that brings a short-released cube in, which the script cannot reproduce but a
  policy does (S5.2 seeds 2-3) -- so both end placed and differ in the release alone."""
  rec = _Recorder()
  moved = {"done": False}

  def hook(phase, env):
    if phase == "settle" and not moved["done"]:
      obj = env.scene[pp.OBJECT]
      pose = obj.data.root_link_pose_w.clone()
      pose[1, :3] = env.command_manager.get_term("goal_pose").command[1]
      obj.write_root_link_pose_to_sim(
        pose[1:2], env_ids=torch.tensor([1], device=env.device)
      )
      obj.write_root_link_velocity_to_sim(
        torch.zeros(1, 6, device=env.device),
        env_ids=torch.tensor([1], device=env.device),
      )
      moved["done"] = True
    rec(phase, env)

  saved = scripted.CLEAR_HEIGHT
  scripted.CLEAR_HEIGHT = CARRY_HEIGHT
  phases = scripted.PHASES + (("settle_more", "retreat", False, 40),)
  try:
    rows = scripted.rollout(
      num_envs=2,
      device=get_test_device(),
      reward=FR,
      verbose=False,
      hook=hook,
      action_slew_rate_scale=SLEW,
      spawns=[(pp.SPAWN_XY, 0.0)] * 2,
      release_offsets=[(0.0, 0.0), _short_offset()],
      phases=phases,
    )
  finally:
    scripted.CLEAR_HEIGHT = saved
  assert moved["done"]
  return rows, rec.rec


def _releases(steps: list[dict]) -> list[int]:
  """Indices of the steps the grasp opened on after the cube was first lifted."""
  lifted = False
  out = []
  for k in range(1, len(steps)):
    lifted |= steps[k - 1]["grasped"] and not steps[k - 1]["on_table"]
    if lifted and steps[k - 1]["grasped"] and not steps[k]["grasped"]:
      out.append(k)
  return out


def test_the_latch_records_where_each_cube_was_let_go(pair):
  _, rec = pair
  for env_i, want in ((0, 0.0), (1, SHORT)):
    steps = _flat(rec[env_i])
    rel = _releases(steps)
    assert len(rel) == 1, f"env {env_i}: releases at {rel}"
    s = steps[rel[0]]
    assert s["released"] and not steps[rel[0] - 1]["released"]
    # What the latch holds is the scene at that step, measured independently -- to within the
    # one physics substep the reward terms' view trails the end of the step by.
    assert s["latch_dxy"] == pytest.approx(s["dxy"], abs=SUBSTEP_TOL)
    assert s["latch_height"] == pytest.approx(s["height"], abs=SUBSTEP_TOL)
    assert abs(s["dxy"] - want) < 0.02, f"env {env_i} let go {s['dxy']:.3f} m out"
    assert s["height"] < 0.01, "the script sets the cube down, it does not drop it"
  q_down, q_short = (_latched_q(_flat(rec[i])[-1]) for i in (0, 1))
  assert q_down > 0.6 and q_short < 0.3, (q_down, q_short)


def _set_down_states(rec_env: dict) -> dict:
  opened = [s for s in rec_env["open"] if s["values"][FR]["release"] > 0]
  assert opened, "the jaw opened without a release being paid"
  settling = [s for s in opened if s["values"][FR]["placed"] == 0]
  return {
    "reach": rec_env["descend"][-1],
    "carry": rec_env["lift"][-1],
    "over_target": rec_env["steady"][-1],
    "held_on_target": rec_env["hold"][-1],
    "released": (settling or opened)[0],
    "placed": rec_env["open"][-1],
  }


def test_the_states_are_the_ones_named(pair):
  s = _set_down_states(pair[1][0])
  assert not s["reach"]["grasped"]
  for name in ("carry", "over_target"):
    assert s[name]["grasped"] and not s[name]["on_table"], name
  # Held at the target at table height. At slew 0.5 the script's place pose (3 mm over rest,
  # `RELEASE_HEIGHT`) can leave the cube just clear of the table rather than on it, which the
  # S5.2 test at slew 1.0 never saw; with M2's latch-gated goal the two pay the same.
  held = s["held_on_target"]
  assert held["grasped"] and held["dxy"] < 0.02, held["dxy"]
  assert held["on_table"] or held["height"] < 0.005, held["height"]
  assert not s["released"]["grasped"] and s["released"]["values"][FR]["release"] > 0
  assert s["placed"]["values"][FR]["placed"] > 0


def test_the_s52_ordering_still_holds_on_the_set_down_path(pair):
  """reach < carry < over the target < held on the target < released < placed, as for the
  full fix (S5.2 design rule), now with `release` and `placed` graded."""
  s = _set_down_states(pair[1][0])
  values = [_total(s[name], FR) for name in ORDER]
  assert all(a < b for a, b in zip(values, values[1:], strict=False)), dict(
    zip(ORDER, values, strict=True)
  )


def test_the_lowering_path_does_not_dip(pair):
  """M6 changes nothing before the release, so the lowering path over the target is the full
  fix's: monotone in height, and below holding the cube on the target.

  Monotone in *height*: since the fda9492 model (plan S7-2 prep 1) the scripted lowering
  drifts ~0.5 mm a step sideways, ~3 cm from the target, until `hold` corrects it -- another
  IK branch under arm 5's new range. `goal_near` (3 G_0.03 of the distance) falls ~0.07 per
  mm there, so the total dips by that drift and nothing else (measured: -0.037 a step, with
  and without S7-2's settle after `approach`). So: every other term is monotone along the
  lowering, and `goal_near` falls only on a step whose horizontal distance grew."""
  rec_env = pair[1][0]
  steps = rec_env["lower"]
  rest = [_total(s, FR) - s["values"][FR]["goal_near"] for s in steps]
  worst = min(b - a for a, b in zip(rest, rest[1:], strict=False))
  assert worst > -0.01, f"the lowering path dips by {worst:.3f} besides goal_near"
  for a, b in zip(steps, steps[1:], strict=False):
    if b["values"][FR]["goal_near"] < a["values"][FR]["goal_near"] - 1e-3:
      assert b["dxy"] > a["dxy"], "goal_near fell without the cube drifting sideways"
  path = [_total(s, FR) for s in steps]
  assert _total(_set_down_states(rec_env)["held_on_target"], FR) > max(path)
  assert path == pytest.approx([_total(s, FULL) for s in steps])


def _settled(rec_env: dict) -> dict:
  """The last step: the cube placed and at rest, the arm retreated."""
  s = _flat(rec_env)[-1]
  assert s["values"][FR]["placed"] > 0 and s["speed"] < 0.05
  return s


def test_after_settling_the_set_down_earns_more_per_step(pair):
  _, rec = pair
  down, short = _settled(rec[0]), _settled(rec[1])
  # Both are placed; the short one's cube even sits closer to the goal (it was moved there).
  assert down["dxy"] >= short["dxy"]
  assert _total(down, FR) > _total(short, FR) + 2.0, (
    _total(down, FR),
    _total(short, FR),
  )
  # Without M6 the two are worth the same per step: nothing tells them apart.
  assert abs(_total(down, FULL) - _total(short, FULL)) < 1.0


def _returns(pair, arm: str, saved: int) -> tuple[float, float]:
  """Discounted returns (gamma 0.99) over a training episode of the set-down and of a release
  short that saves ``saved`` control steps.

  The set-down is env 0's own sequence, extended to the episode's end by its last (settled)
  step. The short release is credited with everything that could favour it: the set-down's
  own income up to ``saved`` steps before the set-down lets go, then the settled placed state
  of env 1 from that step on -- as if the cube had slid onto the target the instant it was
  released.
  """
  _, rec = pair
  down = [_total(s, arm) for s in _flat(rec[0])]
  t_rel = _releases(_flat(rec[0]))[0]
  placed_short = _total(_settled(rec[1]), arm)
  down = (down + [down[-1]] * EPISODE_STEPS)[:EPISODE_STEPS]
  short = down[: t_rel - saved] + [placed_short] * (EPISODE_STEPS - t_rel + saved)
  g = GAMMA ** np.arange(EPISODE_STEPS)
  return float(g @ np.asarray(down)), float(g @ np.asarray(short))


@pytest.mark.parametrize("saved", range(SAVED_STEPS + 1))
def test_the_set_down_has_the_higher_discounted_return(pair, saved):
  down, short = _returns(pair, FR, saved)
  assert down > short, f"saving {saved} steps: set-down {down:.1f} vs short {short:.1f}"


@pytest.mark.parametrize("saved", range(1, SAVED_STEPS + 1))
def test_without_m6_the_short_release_wins(pair, saved):
  """Positive control: the same comparison under the full fix, which is S5.2's incentive to let
  go early -- so the test above is measuring what M6 changes, not the construction."""
  down, short = _returns(pair, FULL, saved)
  assert short > down, f"saving {saved} steps: set-down {down:.1f} vs short {short:.1f}"


def test_the_break_even_is_far_beyond_the_steps_saved(pair):
  """How many steps a short release would have to save to pay under FR. The plan's estimate
  of the saving is 3-5 steps; the margin should be several times that."""
  t_rel = _releases(_flat(pair[1][0]))[0]
  even = [
    k for k in range(t_rel) if _returns(pair, FR, k)[1] >= _returns(pair, FR, k)[0]
  ]
  saved = even[0] if even else t_rel
  assert saved > 4 * SAVED_STEPS, saved


def test_the_scripted_pick_and_place_reaches_placed_under_fr(pair):
  rows, rec = pair
  assert rows[0]["placed_and_lifted"], rows[0]
  assert not rows[0]["ended_by"], "FR has no success termination"
  assert rows[0]["over_slew_env_steps"] == 0 and rows[0]["clipped_env_steps"] == 0
  after = _flat(rec[0])
  first = next(i for i, s in enumerate(after) if s["values"][FR]["placed"] > 0)
  paid = [s["values"][FR]["placed"] > 0 for s in after[first:]]
  assert sum(paid) / len(paid) > 0.9


# -- Letting go again ---------------------------------------------------------------------

#: A good set-down, then take it again, lift it, set it down again -- and then take it again,
#: lift it and drop it from 10 cm.
_REPEAT = scripted.PHASES[: [p[0] for p in scripted.PHASES].index("open") + 1] + (
  ("regrasp", "place", True, 30),
  ("relift", "retreat", True, 40),
  ("relower", "place", True, 40),
  ("reopen", "place", False, 30),
  ("regrasp2", "place", True, 30),
  ("relift2", "retreat", True, 40),
  ("drop", "retreat", False, 30),
  ("settle", "retreat", False, 40),
)


@pytest.fixture(scope="module")
def repeat():
  rec = _Recorder()
  rows = scripted.rollout(
    num_envs=1,
    device=get_test_device(),
    reward=FR,
    verbose=False,
    hook=rec,
    action_slew_rate_scale=SLEW,
    spawns=[(pp.SPAWN_XY, 0.0)],
    phases=_REPEAT,
  )
  return rows, _flat(rec.rec[0])


def test_every_release_overwrites_the_last(repeat):
  _, steps = repeat
  rel = _releases(steps)
  assert [steps[k]["phase"] for k in rel] == ["open", "reopen", "drop"], rel
  for k in rel:
    assert steps[k]["latch_dxy"] == pytest.approx(steps[k]["dxy"], abs=SUBSTEP_TOL)
    assert steps[k]["latch_height"] == pytest.approx(
      steps[k]["height"], abs=SUBSTEP_TOL
    )
  # Held again, the latch keeps the previous release; the next one replaces it.
  for a, b in zip(rel, rel[1:] + [len(steps)], strict=True):
    for s in steps[a:b]:
      assert (s["latch_dxy"], s["latch_height"]) == (
        steps[a]["latch_dxy"],
        steps[a]["latch_height"],
      )


def test_repeating_the_release_cannot_raise_the_grade(repeat):
  """Two good set-downs grade like the second alone, and a drop after them grades like the
  drop: nothing is banked, so re-grasping cannot farm q."""
  _, steps = repeat
  first, second, drop = (_latched_q(steps[k]) for k in _releases(steps))
  assert first > 0.6 and second > 0.6, (first, second)
  assert drop < 0.3 and steps[_releases(steps)[2]]["height"] > 0.05, drop
  # What the terms pay per unit, each step, is the latest release's grade and nothing more.
  for s in steps:
    q_now = _latched_q(s)
    assert s["values"][FR]["release"] <= M1_RELEASE_WEIGHT * q_now + 1e-5
    assert s["values"][FR]["placed"] <= M1_PLACED_WEIGHT * q_now + 1e-5
  assert max(first, second, drop) <= 1.0
