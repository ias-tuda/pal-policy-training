"""S5.2: the per-step reward ordering of every arm, at physically reached states.

The plan's design rule (S5.2): along the task, the per-step reward must be strictly increasing,

    reach < carry < over the target < held on the target < released < placed.

The states are reached by the scripted pick-and-place (`scripts/scripted_pick_place.py`), not
teleported: a held cube cannot be teleported into the jaw, and only a real closing jaw makes
the grasp and table-contact sensors report what a policy would see. The carry height is set
to 7 cm, the height every c1a R-C-family run hovered at, so "over the target" is the state
the policies actually converged to (`s5-1-report.md` section 6). Every arm's task terms are
evaluated at the same state of the same episode, so the arms differ only in their reward.

Only the full fix (base + M1 + M2 + M3) is designed to satisfy the whole ordering. The other
arms test whether *part* of it is enough, so for them this pins exactly which inequalities
hold and which fail by design -- the cliff at "held on the target" is the thing M2 removes.
"""

import collections
import importlib.util
import pathlib
import sys

import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import pytest
from conftest import get_test_device
from mjlab.managers.scene_entity_config import SceneEntityCfg
from pal_mjlab.tasks.manipulation.mdp.observations import (
  _grasped_bool,
  _on_surface_bool,
)
from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import (
  make_pick_place_env_cfg,
)
from pal_mjlab.tasks.manipulation.pick_place.rewards import (
  M4_SUCCESS_WEIGHT,
  S52_ARMS,
  S52_BASE,
)
from pal_mjlab.tasks.manipulation.rl_cfg import stack_ppo_runner_cfg
from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS_LEFT_ARM
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import SUPPORT_SENSOR

_SCRIPT = pathlib.Path(__file__).parent.parent / "scripts/scripted_pick_place.py"
_SPEC = importlib.util.spec_from_file_location("scripted_pick_place_s52", _SCRIPT)
assert _SPEC and _SPEC.loader
scripted = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = scripted
_SPEC.loader.exec_module(scripted)

FULL = f"{S52_BASE}+M1+M2+M3"
ARMS = (S52_BASE, *S52_ARMS)
#: The terms the ordering is about; the regularisers and trunk_clearance are identical across
#: arms and depend on the script's own motion, not on the state.
TASK_TERMS = (
  "reach",
  "grasp",
  "lift",
  "goal",
  "goal_near",
  "release",
  "placed",
  "success",
)
#: c1a's hover height; see the module docstring.
CARRY_HEIGHT = 0.07
ORDER = ("reach", "carry", "over_target", "held_on_target", "released", "placed")


def _task_terms(env, arm: str) -> dict:
  rewards = make_pick_place_env_cfg(
    KANGAROO_GRIPPERS_LEFT_ARM, gripper_name="left", reward=arm
  ).rewards
  for cfg in rewards.values():
    for value in cfg.params.values():
      if isinstance(value, SceneEntityCfg):
        value.resolve(env.scene)
  return {name: cfg for name, cfg in rewards.items() if name in TASK_TERMS}


#: The script's phases with `lower` split in two: `lower` re-plans the place and moves the held
#: cube over it at carry height (to `retreat`, which the re-plan puts straight above the place
#: pose), then `lower_down` descends onto it. Since the fda9492 model (plan S7-2 prep 1) the
#: one-phase `lower` also moves the cube ~0.5 mm a step sideways, ~3 cm from the target, while
#: it descends -- the re-plan's correction, interpolated in joint space -- and the goal terms
#: fall with that drift; the design rule is about a descent over the target.
VERTICAL_LOWERING = tuple(
  step
  for phase in scripted.PHASES
  for step in (
    (("lower", "retreat", True, 30), ("lower_down", "place", True, 40))
    if phase[0] == "lower"
    else (phase,)
  )
)


@pytest.fixture(scope="module")
def episode():
  """One scripted episode under the full arm (no success termination, so it runs through
  `placed` and on), recording every arm's weighted task terms for env 0 at every step."""
  terms: dict = {}
  rec = collections.defaultdict(list)
  grasp_sensor = KANGAROO_GRIPPERS_LEFT_ARM.gripper("left").contact_sensor_name

  def hook(phase, env):
    if not terms:
      terms.update({arm: _task_terms(env, arm) for arm in ARMS})
    values = {
      arm: {n: float(c.weight * c.func(env, **c.params)[0]) for n, c in ts.items()}
      for arm, ts in terms.items()
    }
    obj = env.scene["object"]
    rec[phase].append(
      {
        "values": values,
        "grasped": bool(_grasped_bool(env, grasp_sensor)[0]),
        "on_table": bool(_on_surface_bool(env, SUPPORT_SENSOR)[0]),
        "speed": float(obj.data.root_link_lin_vel_w[0].norm()),
      }
    )

  saved = scripted.CLEAR_HEIGHT
  scripted.CLEAR_HEIGHT = CARRY_HEIGHT
  try:
    rows = scripted.rollout(
      num_envs=1,
      device=get_test_device(),
      reward=FULL,
      verbose=False,
      hook=hook,
      phases=VERTICAL_LOWERING,
    )
  finally:
    scripted.CLEAR_HEIGHT = saved
  assert rows[0]["placed_and_lifted"], rows
  return rec


def _states(rec) -> dict:
  released = [
    s
    for s in rec["open"]
    if s["values"][FULL]["release"] > 0 and s["values"][FULL]["placed"] == 0
  ]
  assert released, "no released-but-settling step was observed after the jaw opened"
  return {
    "reach": rec["descend"][-1],
    "carry": rec["lift"][-1],
    "over_target": rec["steady"][-1],
    "held_on_target": rec["hold"][-1],
    "released": released[0],
    "placed": rec["open"][-1],
  }


def _total(state, arm: str) -> float:
  return sum(state["values"][arm].values())


def test_the_states_are_the_ones_named(episode):
  """Guard on the construction: a mislabelled state would make every ordering meaningless."""
  s = _states(episode)
  assert not s["reach"]["grasped"]
  for name in ("carry", "over_target"):
    assert s[name]["grasped"] and not s[name]["on_table"], name
  # Held, and resting on the table at the target: the state the R-C cliff is about.
  assert s["held_on_target"]["grasped"] and s["held_on_target"]["on_table"]
  assert not s["released"]["grasped"] and s["released"]["speed"] > 0.05
  assert s["placed"]["values"][FULL]["placed"] > 0


def test_the_full_fix_is_strictly_increasing(episode):
  s = _states(episode)
  values = [_total(s[name], FULL) for name in ORDER]
  assert all(a < b for a, b in zip(values, values[1:], strict=False)), dict(
    zip(ORDER, values, strict=True)
  )


def test_the_full_fix_pays_more_all_the_way_down(episode):
  """No dip on the lowering path over the target (design rule 2). At M2's planned goal weight
  of 4 the reward dips at ~5 cm below its 7.5 cm value; the weight is 7 for this."""
  # The `lower_down` phase only: a straight descent onto the re-planned place pose (see
  # `VERTICAL_LOWERING`). `hold` includes the script's place corrections, which lift the cube
  # 2 cm and put it back down -- motion the script makes, not the path a policy lowers on.
  path = [_total(s, FULL) for s in episode["lower_down"]]
  worst = min(b - a for a, b in zip(path, path[1:], strict=False))
  assert worst > -0.01, f"the lowering path dips by {worst:.3f}"
  assert path[-1] > path[0] + 2.0
  assert _total(_states(episode)["held_on_target"], FULL) > max(path)


@pytest.mark.parametrize("arm", ARMS)
def test_every_arm_pays_more_for_each_stage_up_to_the_target(arm, episode):
  """The grasp / transport part is the base's and is monotone on every arm."""
  s = _states(episode)
  reach, carry, over = (_total(s[n], arm) for n in ("reach", "carry", "over_target"))
  assert reach < carry < over


@pytest.mark.parametrize("arm", ARMS)
def test_the_arms_without_m2_keep_the_cliff(arm, episode):
  """The documented failure M2 exists for: without it, resting the cube on the target while
  holding it pays less than hovering above it, because the goal term stops at table contact."""
  s = _states(episode)
  over, held = _total(s["over_target"], arm), _total(s["held_on_target"], arm)
  if "M2" in arm:
    assert held > over
  else:
    assert held < over


@pytest.mark.parametrize("arm", [a for a in ARMS if "M1" in a])
def test_m1_makes_the_placed_state_out_earn_the_hover(arm, episode):
  """Per-step dominance: whatever the path, the finished state pays more per step than the
  hover the c1a policies settled in, and releasing pays more than keeping hold."""
  s = _states(episode)
  assert _total(s["placed"], arm) > _total(s["over_target"], arm)
  assert _total(s["released"], arm) > _total(s["held_on_target"], arm)


@pytest.mark.parametrize("arm", [a for a in ARMS if "M3" in a])
def test_m3_fades_the_lift_income_at_the_target(arm, episode):
  s = _states(episode)
  assert s["held_on_target"]["values"][arm]["lift"] < 0.05
  assert s["carry"]["values"][arm]["lift"] > 1.0  # far from the target: unfaded


def test_the_m4_bonus_beats_the_discounted_hover(episode):
  """Terminate-on-success is safe only if the one-off bonus exceeds what hovering forever is
  worth: r_hover / (1 - gamma) under the training discount."""
  arm = f"{S52_BASE}+M3+M4"
  gamma = stack_ppo_runner_cfg("x").algorithm.gamma
  hover = _total(_states(episode)["over_target"], arm)
  assert M4_SUCCESS_WEIGHT > hover / (1.0 - gamma)


def test_placed_keeps_paying_after_the_placement(episode):
  """With no success termination (M1), the placed state is income, not an ending: it must keep
  paying while the arm retreats and the episode goes on."""
  after = episode["open"] + episode["retreat"] + episode["settle"]
  first = next(i for i, s in enumerate(after) if s["values"][FULL]["placed"] > 0)
  paid = [s["values"][FULL]["placed"] > 0 for s in after[first:]]
  assert sum(paid) / len(paid) > 0.9
