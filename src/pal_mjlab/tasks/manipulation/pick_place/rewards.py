"""The reward variants S5 compares, selected by name.

One function per variant, each returning the whole ``rewards`` dict. Selected by name in the
env cfg so that a wandb run's config says which reward produced a curve -- the comparison is
between named arms, not between remembered edits.

``R-A`` is the framework's own reward and the only one v0 is scheduled to train: C1 is a
baseline, and the plan runs the variants only after it has established that the task learns
at all. The rest are implemented rather than sketched because the terms they need are the
real work, and because a variant designed after seeing C1's curve is a variant fitted to it.

Common to every variant (``_regularisers``): action rate, joint limits, a joint-velocity
hinge on mjlab's three-stage curriculum, and a penalty for leaning on the table. Those are
not the thing under comparison, so they do not vary between arms.

Literature behind the shapes, read before they were written:

* **R-C** -- staged, gated bonuses: IsaacGymEnvs' ``FrankaCubeStack`` (Makoviychuk et al.,
  *Isaac Gym*, 2021). Corroborated by Lin et al., *Sim-to-Real RL for Vision-Based Dexterous
  Manipulation on Humanoids* (arXiv:2502.20396), which gates a bimanual handover on a stage
  variable and gives "a bonus whose scale increases as stage progresses".
* **R-D** -- eight bounding-box keypoints under a logistic kernel: Allshire et al.,
  *Transferring Dexterous Manipulation from GPU Simulation to a Remote Real-World TriFinger*
  (2022), ``K(x) = (e^{ax} + b + e^{-ax})^-1`` with ``a = 30``, ``b = 2``.
* **R-B** -- the sparse arm: mjlab's own ``lift_cube`` succeeds on proximity alone, so the
  question is whether the contact-based terms are load-bearing here or merely present.
"""

from __future__ import annotations

import dataclasses
import math

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from pal_mjlab.tasks.manipulation import mdp
from pal_mjlab.tasks.manipulation.mdp.shapes import BySize, PerShape
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import (
  COMMAND,
  OBJECT,
  SUPPORT_SENSOR,
  TABLE,
)


@dataclasses.dataclass(frozen=True)
class RewardContext:
  """Everything a variant needs to name a term, and nothing about the robot."""

  tcp: SceneEntityCfg
  action_joints: SceneEntityCfg
  grasp_sensor: str
  surface_sensor: str
  threshold: float
  jaw: SceneEntityCfg | None = None
  """The jaw's actuated joint, for plan S7-4's J (a release is the jaw opening)."""
  jaw_open: float | BySize | PerShape = 0.15
  """J's threshold (`JAW_OPEN`), rad: below it, with the pads off, the jaw is open. A `PerShape`
  when the object's shape changes where the jaw first touches it (plan S8, L7)."""
  pads: SceneEntityCfg | None = None
  """The two pad sites, for plan S8's EQ (the height the pads close at)."""

  @property
  def grasp_params(self) -> dict:
    return {
      "command_name": COMMAND,
      "object_name": OBJECT,
      "grasp_sensor_name": self.grasp_sensor,
      "threshold": self.threshold,
    }


#: How high the object has to come off the table for the staged lift term to saturate.
LIFT_HEIGHT = 0.10

#: Kernel width of the goal-distance terms, in metres. The framework's default.
BRINGING_STD = 0.15

#: Per-step cost for R-E, and the success bonus it buys back.
TIME_COST = 0.01
R_E_SUCCESS_WEIGHT = 20.0


def _regularisers(ctx: RewardContext) -> dict[str, RewardTermCfg]:
  """The terms every arm carries, so that no comparison is confounded by them."""
  return {
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    "joint_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits, weight=-10.0, params={"asset_cfg": ctx.action_joints}
    ),
    # Curriculum-weighted; see `reward_curriculum_cfg`. Starts almost free so that early
    # exploration is not punished for moving quickly.
    "joint_vel_hinge": RewardTermCfg(
      func=mdp.joint_velocity_hinge_penalty,
      weight=-0.01,
      params={"max_vel": 0.5, "asset_cfg": ctx.action_joints},
    ),
    "table_contact": RewardTermCfg(
      func=mdp.contact_penalty,
      weight=-0.5,
      params={"sensor_name": ctx.surface_sensor, "force_threshold": 10.0},
    ),
  }


def reward_curriculum_cfg() -> dict[str, CurriculumTermCfg]:
  """mjlab's three-stage ramp on the joint-velocity hinge, in env steps.

  Copied from mjlab's own lift-cube recipe rather than retuned: the point of v0 is to find
  out what the environment does, and a re-tuned optimiser makes that harder to attribute.
  """
  return {
    "joint_vel_hinge_weight": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "joint_vel_hinge",
        "stages": [
          {"step": 0, "weight": -0.01},
          {"step": 500 * 24, "weight": -0.1},
          {"step": 1000 * 24, "weight": -1.0},
        ],
      },
    ),
  }


def _reach_grasp_lift(ctx: RewardContext) -> dict[str, RewardTermCfg]:
  """The framework's approach-and-hold terms, shared by R-A, R-D and R-E."""
  return {
    "reach": RewardTermCfg(
      func=mdp.staged_position_reward,
      weight=1.0,
      params={
        "command_name": COMMAND,
        "object_name": OBJECT,
        "reaching_std": 0.2,
        "bringing_std": 0.2,
        "asset_cfg": ctx.tcp,
      },
    ),
    "grasp": RewardTermCfg(
      func=mdp.grasped, weight=1.0, params={"sensor_name": ctx.grasp_sensor}
    ),
    "lift": RewardTermCfg(
      func=mdp.object_lifted,
      weight=2.0,
      params={
        "grasp_sensor_name": ctx.grasp_sensor,
        "support_sensor_name": SUPPORT_SENSOR,
      },
    ),
  }


def _release_and_success(
  ctx: RewardContext, success_weight: float = 10.0
) -> dict[str, RewardTermCfg]:
  return {
    "release": RewardTermCfg(
      func=mdp.released_at_goal, weight=2.0, params=ctx.grasp_params
    ),
    "success": RewardTermCfg(
      func=mdp.object_placed_reward, weight=success_weight, params=ctx.grasp_params
    ),
  }


def r_a_framework(ctx: RewardContext) -> dict[str, RewardTermCfg]:
  """What ``stack`` uses, and what the tabletop base cfg builds. The baseline arm."""
  return {
    **_reach_grasp_lift(ctx),
    "goal": RewardTermCfg(
      func=mdp.bring_object_reward,
      weight=4.0,
      params={"command_name": COMMAND, "object_name": OBJECT, "std": BRINGING_STD},
    ),
    **_release_and_success(ctx),
    **_regularisers(ctx),
  }


def r_b_sparse(ctx: RewardContext) -> dict[str, RewardTermCfg]:
  """Reach, goal and success only: can the policy find the grasp without being paid for it?

  mjlab's own lift-cube succeeds by proximity, with no contact term at all. If R-B matches
  R-A then the grasp, lift and release terms are decoration and the framework is carrying
  three terms it does not need; if it fails, they are load-bearing and that is worth knowing
  before the ladder makes contact harder.
  """
  return {
    "reach": _reach_grasp_lift(ctx)["reach"],
    "goal": RewardTermCfg(
      func=mdp.bring_object_reward,
      weight=4.0,
      params={"command_name": COMMAND, "object_name": OBJECT, "std": BRINGING_STD},
    ),
    "success": _release_and_success(ctx)["success"],
    **_regularisers(ctx),
  }


def r_c_phase_gated(ctx: RewardContext) -> dict[str, RewardTermCfg]:
  """Monotone stages: reach, grasp, bounded lift, transport, release.

  Two changes against R-A, both gates. ``lift`` becomes a bounded height fraction rather
  than a binary, so the policy is paid for lifting *more* up to 10 cm and not for lifting
  *higher* beyond it -- an unbounded height term pays better for throwing than for placing.
  ``goal`` only pays once the object is off the table, which removes the shortcut of sliding
  it there and makes the stage ordering monotone.
  """
  staged = _reach_grasp_lift(ctx)
  staged["lift"] = RewardTermCfg(
    func=mdp.lift_height,
    weight=2.0,
    params={
      "grasp_sensor_name": ctx.grasp_sensor,
      "object_name": OBJECT,
      "surface_cfg": SceneEntityCfg(TABLE),
      "max_height": LIFT_HEIGHT,
    },
  )
  return {
    **staged,
    "goal": RewardTermCfg(
      func=mdp.bring_object_when_lifted,
      weight=4.0,
      params={
        "command_name": COMMAND,
        "object_name": OBJECT,
        "grasp_sensor_name": ctx.grasp_sensor,
        "support_sensor_name": SUPPORT_SENSOR,
        "std": BRINGING_STD,
      },
    ),
    **_release_and_success(ctx),
    **_regularisers(ctx),
  }


def r_c_strict(ctx: RewardContext) -> dict[str, RewardTermCfg]:
  """R-C with the slide path closed everywhere, not just on the `goal` term.

  R-C gates only `goal` on the object being lifted, which leaves three terms that a policy
  can collect by pushing the cube along the table into the target:

  ===========  ==========================================================================
  ``reach``    ``staged_position_reward`` = ``G(d_r)*(1 + G(d_g))``; the ``G(d_g)`` factor
               is ungated, so it pays for the object being near the goal however it got
               there.
  ``release``  ``released_at_goal`` is ``at_goal ∧ ¬grasped``, and a cube never grasped is
               trivially "released".
  ``success``  ``_placed_bool`` uses the same ``¬grasped``, for both the 10-weight reward
               and the termination that ends the episode.
  ===========  ==========================================================================

  So: ``reach`` drops to the reaching Gaussian alone, and ``release``, ``success`` and the
  success *termination* all additionally require the `was_lifted` episode latch. What
  remains payable without a lift is the approach itself, which is the point -- the policy
  still gets a gradient towards the cube from step one.

  The gate on the termination matters as much as the one on the reward: without it a slid
  cube ends the episode, and the policy collects the remaining-time saving even with the
  bonus withheld.

  **This is the arm most exposed to Isaac Lab #204** (reach saturates, `lifting_object`
  stays exactly 0), because nothing pays for closing the jaw until `lift_height` does. The
  plan's pre-agreed fix is a `+1·grasped` term or `lift` weight 30-50 -- as a follow-up run,
  not an edit to a running one.
  """
  staged = r_c_phase_gated(ctx)
  staged["reach"] = RewardTermCfg(
    func=mdp.reach_only,
    weight=1.0,
    params={"object_name": OBJECT, "reaching_std": 0.2, "asset_cfg": ctx.tcp},
  )
  lifted = {**ctx.grasp_params, "support_sensor_name": SUPPORT_SENSOR}
  staged["release"] = RewardTermCfg(
    func=mdp.released_at_goal, weight=2.0, params=lifted
  )
  staged["success"] = RewardTermCfg(
    func=mdp.object_placed_reward, weight=10.0, params=lifted
  )
  return staged


def r_d_keypoints(ctx: RewardContext) -> dict[str, RewardTermCfg]:
  """R-A with the centre-distance goal term replaced by eight-corner keypoints."""
  return {
    **_reach_grasp_lift(ctx),
    "goal": RewardTermCfg(
      func=mdp.keypoint_distance_reward,
      weight=4.0,
      params={"command_name": COMMAND, "object_name": OBJECT},
    ),
    **_release_and_success(ctx),
    **_regularisers(ctx),
  }


def r_e_time_penalty(ctx: RewardContext) -> dict[str, RewardTermCfg]:
  """R-A plus a small per-step cost and a larger terminal bonus.

  The classic anti-dawdling pair: without a cost for time, an episode spent hovering beside
  the cube scores the same as one spent placing it late. Runs with a shorter
  ``episode_length_s``; see `make_pick_place_env_cfg`.
  """
  return {
    **r_a_framework(ctx),
    **_release_and_success(ctx, success_weight=R_E_SUCCESS_WEIGHT),
    "alive": RewardTermCfg(func=mdp.time_penalty, weight=-TIME_COST),
  }


#: The arms of the S5 comparison. `R-A` is the baseline every other is measured against.
REWARD_VARIANTS = {
  "R-A": r_a_framework,
  "R-B": r_b_sparse,
  "R-C": r_c_phase_gated,
  "R-C-strict": r_c_strict,
  "R-D": r_d_keypoints,
  "R-E": r_e_time_penalty,
}

# -- S5.2: release / place modules on the c1a base ---------------------------------------
#
# c1a (`s5-1-report.md` section 6) solved picking on every arm and placing on none, and chose
# R-C-strict as the base: every S5.2 arm keeps its reach / grasp / lift / transport terms and
# varies only the release / place part (plan S5.2, revision 2026-09-24 10:27). The modules:
#
# * M1 per-step dominance -- no success termination in training; `release` (gated) weight 4;
#   a per-step `placed` term (gated) weight 8, so the finished state out-earns holding.
# * M2 no-cliff goal -- `goal` gated on the `was_lifted` latch instead of "off the table right
#   now", plus a narrow `goal_near` term for the last centimetres. The plan's weight on the
#   broad term was 4; it is 7 because at 4 the lowering path over the target has a dip at 5 cm
#   (6.71 there vs 6.91 at 7.5 cm) -- a hover optimum of its own. 7 is the smallest integer
#   that makes the lowering path monotone (analytic check, s5-2 report; test
#   `test_pick_place_s52_ordering.py`).
# * M3 lift fade -- `lift_height` x (1 - G_0.05(d_g)).
# * M4 terminal bonus -- keep the success termination, success weight 1000 (gated).
#
# Every release / placed / success term, and the success termination, requires `was_lifted`
# (R-C-strict already gates them; the modules keep it that way).

#: The c1a arm S5.2 builds on.
S52_BASE = "R-C-strict"
M1_RELEASE_WEIGHT = 4.0
M1_PLACED_WEIGHT = 8.0
M2_GOAL_WEIGHT = 7.0
M2_NEAR_WEIGHT = 3.0
M2_NEAR_STD = 0.03
M3_FADE_STD = 0.05
M4_SUCCESS_WEIGHT = 1000.0


def _gated(ctx: RewardContext) -> dict:
  return {**ctx.grasp_params, "support_sensor_name": SUPPORT_SENSOR}


def _m1(rewards: dict[str, RewardTermCfg], ctx: RewardContext) -> None:
  rewards.pop("success", None)
  rewards["release"] = RewardTermCfg(
    func=mdp.released_at_goal, weight=M1_RELEASE_WEIGHT, params=_gated(ctx)
  )
  rewards["placed"] = RewardTermCfg(
    func=mdp.object_placed_reward, weight=M1_PLACED_WEIGHT, params=_gated(ctx)
  )


def _m2(rewards: dict[str, RewardTermCfg], ctx: RewardContext) -> None:
  params = {
    "command_name": COMMAND,
    "object_name": OBJECT,
    "grasp_sensor_name": ctx.grasp_sensor,
    "support_sensor_name": SUPPORT_SENSOR,
  }
  rewards["goal"] = RewardTermCfg(
    func=mdp.bring_object_after_lift,
    weight=M2_GOAL_WEIGHT,
    params={**params, "std": BRINGING_STD},
  )
  rewards["goal_near"] = RewardTermCfg(
    func=mdp.bring_object_after_lift,
    weight=M2_NEAR_WEIGHT,
    params={**params, "std": M2_NEAR_STD},
  )


def _m3(rewards: dict[str, RewardTermCfg], ctx: RewardContext) -> None:
  lift = rewards["lift"]
  assert lift.func is mdp.lift_height, "M3 is defined on R-C's bounded lift_height"
  rewards["lift"] = RewardTermCfg(
    func=mdp.lift_height_faded,
    weight=lift.weight,
    params={**lift.params, "command_name": COMMAND, "fade_std": M3_FADE_STD},
  )


def _m4(rewards: dict[str, RewardTermCfg], ctx: RewardContext) -> None:
  rewards["success"] = RewardTermCfg(
    func=mdp.object_placed_reward, weight=M4_SUCCESS_WEIGHT, params=_gated(ctx)
  )


# -- S7: speed bound B, the control for the slew bound A ------------------------------------
#
# M5 prices the held cube's speed above 0.4 m/s, squared (plan S7, FB). The full fix carries
# at up to ~1.2 m/s (S5.2 report §5-6); the weight makes that cost more per step than one
# step of `placed` (8): 16 x (1.2 - 0.4)^2 = 10.24. At 0.6 m/s it is 0.64, so a carry at the
# slew-bound-A speed (TCP ~0.6 m/s) is nearly free.
M5_SPEED_THRESHOLD = 0.4  # m/s
M5_SPEED_WEIGHT = -16.0


def _m5(rewards: dict[str, RewardTermCfg], ctx: RewardContext) -> None:
  rewards["held_speed"] = RewardTermCfg(
    func=mdp.held_object_speed_hinge,
    weight=M5_SPEED_WEIGHT,
    params={
      "object_name": OBJECT,
      "grasp_sensor_name": ctx.grasp_sensor,
      "threshold": M5_SPEED_THRESHOLD,
    },
  )


# -- S7: M6, release quality (plan S7, added 2026-09-25) ----------------------------------
#
# Under M1 a cube let go 5-6 cm short that slides or flies onto the target counts as placed
# and starts earning `placed` a few steps sooner than one set down on it -- the per-step
# optimum two of three S5.2 full-fix seeds converged to (S5.2 report section 6). M6 grades
# M1's `release` and `placed` by where the latest release after `was_lifted` happened:
# q = 0.25 + 0.75 * G_0.02(d_xy) * G_0.01(max(h - 0.01, 0)), d_xy the cube's horizontal
# distance to the target and h its height above rest, at the step the grasp opened. A set-down
# like seed 1's (0.8 cm off, 0.4 cm up) gives q ~ 0.89, one 6 cm short ~ 0.25; `goal` and
# `goal_near` are not graded. The floor keeps placing worth something early in training.
M6_FLOOR = 0.25
M6_DIST_STD = 0.02  # m
M6_HEIGHT_FREE = 0.01  # m
M6_HEIGHT_STD = 0.01  # m


def _m6(rewards: dict[str, RewardTermCfg], ctx: RewardContext) -> None:
  quality = {
    "surface_cfg": SceneEntityCfg(TABLE),
    "dist_std": M6_DIST_STD,
    "height_free": M6_HEIGHT_FREE,
    "height_std": M6_HEIGHT_STD,
    "floor": M6_FLOOR,
  }
  graded = {"release": mdp.released_at_goal_graded, "placed": mdp.object_placed_graded}
  for name, func in graded.items():
    term = rewards.get(name)
    assert term is not None and "support_sensor_name" in term.params, (
      "M6 grades M1's gated release and placed terms; apply M1 first"
    )
    rewards[name] = RewardTermCfg(
      func=func, weight=term.weight, params={**term.params, **quality}
    )


# -- S7-2: M7, the set-down (plan S7-2, decided by Siwei 2026-09-25) ------------------------
#
# FR (M6) lets go at the target, but increasingly while the cube still moves: 0.10 -> 0.44
# m/s at release over training (S7 report section 5.5). M6 grades where and how high the
# release is and gives the first centimetre of height free. M7 pays M6's grade only for a
# set-down: at the step the grasp opens the cube rests on the table, moves slower than
# 0.05 m/s, and has been held there, still and within 1.5 cm of the target, for
# `M7_DWELL_STEPS` control steps since a touchdown within that radius (`mdp.set_down_passed`,
# `observations._release_latch`); any other release is graded at the floor. The table contact
# replaces M6's free centimetre of height, so `height_free` goes to 0. The touchdown rule is
# the guard against dragging the cube along the table into the target and letting go of it at
# rest; `held_slide` prices the drag itself (`M7_SLIDE_WEIGHT` per m/s of the held cube's
# horizontal speed on the table, after the first lift).
#
# The plan leaves the dwell at 2-3 steps; 3 (60 ms) is taken, the stricter end.
M7_DWELL_STEPS = 3
M7_SLIDE_WEIGHT = -10.0


def _m7(rewards: dict[str, RewardTermCfg], ctx: RewardContext) -> None:
  for name in ("release", "placed"):
    term = rewards.get(name)
    assert term is not None and "floor" in term.params, (
      "M7 pays M6's grade for a set-down only; apply M6 first"
    )
    rewards[name] = RewardTermCfg(
      func=term.func,
      weight=term.weight,
      params={**term.params, "height_free": 0.0, "set_down_steps": M7_DWELL_STEPS},
    )
  rewards["held_slide"] = RewardTermCfg(
    func=mdp.held_object_slide_speed,
    weight=M7_SLIDE_WEIGHT,
    params={
      "object_name": OBJECT,
      "grasp_sensor_name": ctx.grasp_sensor,
      "support_sensor_name": SUPPORT_SENSOR,
    },
  )


# -- S7-4: J (a release is the jaw opening), M8 (the retreat), S (M7 revised) --------------
#
# Plan S7-4, revised 2026-09-28 21:33 (Siwei). S7-2's videos and a per-step probe of S7-1's FR s2
# showed "released" (= the pads no longer both touch the cube) firing with the jaw still at the
# hold angle (~0.30 rad; first touch of a 5 cm cube ~0.21). J makes a release the jaw's measured
# angle falling below `JAW_OPEN` after `was_lifted`: M1's `release` and `placed` (and M6's grade,
# through the release latch) need it. `grasped` stays the critic's input and the lift gate.
#
# M8 pays the tool point's height above the placed cube's top, up to `M8_CLEAR_HEIGHT`, while
# the cube is placed (J) and no pad touches it (`mdp.retreat_height`), and prices pad-cube
# contact after the release (`mdp.pad_contact_after_release`). `placed` keeps paying through the
# retreat, so knocking the cube off loses both. Weights: the retreat at full clear is a quarter
# of one step of `placed` (8), enough to make lifting clear beat resting the open hand on the
# table and small against losing `placed` + `release` (~11.5 per step) by disturbing the cube.
#
# S keeps M6's grade for every J-release and adds a per-step bonus while the cube is placed after
# a set-down: held on the table, still, within 1.5 cm of the target for `S_DWELL_STEPS` steps
# since a touchdown there (`mdp.set_down_placed`). M7 as run floored every other release and was
# not learned (0-5 % set-downs on B-S, S7-2 / S7-3); the plan's dwell here is 2 steps.
JAW_OPEN = 0.15  # rad; `mdp.observations.JAW_OPEN_BELOW`
#: J's threshold for plan S8's ball (L7: it is size-dependent). The plan's proposal is the ball's touch angle minus
#: 0.05 rad (the cube's is 0.06 below its ~0.21). The gripper bench (`scripts/gripper_bench.py --stimulus touch`,
#: `../results/S8-ball/bench/`, G20, the jaw ramped to closed over 5 s) first touches a 60 mm ball at 0.2752 rad and
#: holds it at 0.309; the pad-site gap curve (`--stimulus kinematics`: 79.2 mm open, 60.7 mm at 0.275 rad) moves that
#: 0.012 rad per mm of diameter, so BALL-DR's 55-65 mm balls are touched at 0.336 / 0.275 / 0.211 rad. A fixed
#: threshold would sit above the big ball's hold angle, and "released" would fire with the jaw still closed on it (the
#: pads merely slipping off) -- what J was made to stop -- so the threshold follows the radius.
BALL_JAW_OPEN_BY_RADIUS = BySize(
  radii=(0.0275, 0.03, 0.0325), values=(0.286, 0.225, 0.161)
)
#: ... and at the nominal radius (BALL, BALL-EQ, the eval's prints and tests).
BALL_JAW_OPEN = 0.225
M8_CLEAR_HEIGHT = 0.05  # m (plan S7-4 revision 2: was 8 cm)
M8_RETREAT_WEIGHT = 2.0
M8_CONTACT_WEIGHT = -1.0
S_DWELL_STEPS = 2
S_BONUS_WEIGHT = 4.0


def _jaw_params(ctx: RewardContext) -> dict:
  assert ctx.jaw is not None, "plan S7-4's J needs RewardContext.jaw"
  return {"jaw_cfg": ctx.jaw, "jaw_open": ctx.jaw_open}


def _j(rewards: dict[str, RewardTermCfg], ctx: RewardContext) -> None:
  for name in ("release", "placed"):
    term = rewards.get(name)
    assert term is not None and "support_sensor_name" in term.params, (
      "J redefines M1's gated release and placed terms; apply M1 first"
    )
    rewards[name] = RewardTermCfg(
      func=term.func, weight=term.weight, params={**term.params, **_jaw_params(ctx)}
    )


def _latched(ctx: RewardContext) -> dict:
  return {
    **_gated(ctx),
    "surface_cfg": SceneEntityCfg(TABLE),
    **_jaw_params(ctx),
  }


def _m8(rewards: dict[str, RewardTermCfg], ctx: RewardContext) -> None:
  gated = {**_gated(ctx), **_jaw_params(ctx)}
  rewards["retreat"] = RewardTermCfg(
    func=mdp.retreat_height,
    weight=M8_RETREAT_WEIGHT,
    params={**gated, "asset_cfg": ctx.tcp, "clear_height": M8_CLEAR_HEIGHT},
  )
  latched = _latched(ctx)
  latched.pop("threshold")
  rewards["pad_contact"] = RewardTermCfg(
    func=mdp.pad_contact_after_release, weight=M8_CONTACT_WEIGHT, params=latched
  )


def _s(rewards: dict[str, RewardTermCfg], ctx: RewardContext) -> None:
  rewards["set_down"] = RewardTermCfg(
    func=mdp.set_down_placed,
    weight=S_BONUS_WEIGHT,
    params={**_latched(ctx), "set_down_steps": S_DWELL_STEPS},
  )


# -- S7-4: P, the effort hinge (with T: the `EFFORT` code CTP) ------------------------------
#
# -w * sum_i max(|tau_i| / tau_cmd,i - 0.8, 0)^2 over the seven driven-arm joints, tau_cmd the
# lowered limit T puts on the model (35 / 17.5 N.m): thresholds 28 / 14 N.m, above the ~11 N.m
# gravity load at the home pose, so holding the arm up is never priced. With T the excess is at
# most 0.2 per joint, 0.04 squared. Calibrated as the plan asks: a full-effort press on one joint
# held after the release must cost more per step than the `placed` income it could protect
# (8 x q <= 8): w x 0.04 > 8, w > 200; with 25 % margin, 250 (10 per step at the cap). The
# other half of the calibration -- a clean carry costs < 1 % of its return -- is checked by
# `tests/test_pick_place_s74.py` on the scripted pick-and-place.
P_THRESHOLD = 0.8
P_WEIGHT = -250.0


def effort_hinge_reward(arm: SceneEntityCfg) -> RewardTermCfg:
  return RewardTermCfg(
    func=mdp.arm_effort_hinge,
    weight=P_WEIGHT,
    params={"asset_cfg": arm, "threshold": P_THRESHOLD},
  )


# -- S7-5: O (open at the goal) and Z (the goal tolerance split into xy and z) --------------
#
# Plan S7-5 (designed 2026-09-29 12:08, Siwei). RCTP s2 (S7-4) carried the cube to the target and
# held it there to the time-out, jaw closed (placed 5 %): nothing paid for starting to open. O
# pays the jaw's opening from the hold angle `O_JAW_HOLD` towards `JAW_OPEN`, at most `O_WEIGHT`
# per step, while the lifted, not yet J-released cube is at the goal by `Z_GOAL_TOL`
# (`mdp.open_at_goal`); a half-open hover (the pads come off a 5 cm cube at ~0.21 rad) earns
# 0.76 per step, well under `placed` (8 x M6's grade, >= 2 at the floor, + `release`).
#
# `O_JAW_HOLD` is 0.40 rad, not the plan's 0.30 (Siwei, 2026-09-29, on prep 3's trace): RCTP s2
# squeezes the cube at the goal to a median 0.37 rad (p90 0.40; RCTP s1 0.30), so from 0.30 O
# would pay nothing for the first part of s2's opening -- the state O exists for. A 0.30 hold
# at the goal now earns 0.4 per step.
#
# Z makes "at the goal" xy < 2 cm and |dz| < 2 cm (was 3-D < 2.5 cm) in every term that tests
# it after M1: `release`, `placed`, M8's `retreat` and S's `set_down`. Tighter in xy, looser in
# z, so a cube resting tilted on an edge (+1.0 cm) or a corner (+1.8 cm) is at the goal. The plan
# names `placed` (M1, M8's arming); M1's `release` is included so the two M1 terms keep one
# definition -- otherwise `release` would pay where `placed` cannot. Z must be the last module,
# so that it sees every such term. O's gate is Z's tolerance with or without Z.
O_WEIGHT = 1.0
O_JAW_HOLD = 0.40  # rad
Z_GOAL_TOL = (0.02, 0.02)  # m: (xy, |dz|)


def _o(rewards: dict[str, RewardTermCfg], ctx: RewardContext) -> None:
  if ctx.jaw_open != JAW_OPEN:
    raise ValueError(
      f"O pays the jaw's opening from the cube's hold angle ({O_JAW_HOLD} rad) to {JAW_OPEN}; "
      f"this object's J threshold is {ctx.jaw_open}, and its hold angle is not calibrated (plan S8)"
    )
  params = _latched(ctx)
  params.pop("threshold")
  rewards["open_at_goal"] = RewardTermCfg(
    func=mdp.open_at_goal,
    weight=O_WEIGHT,
    params={**params, "jaw_hold": O_JAW_HOLD, "gate": Z_GOAL_TOL},
  )


#: The terms whose "at the goal" Z redefines, by their function.
_AT_GOAL_TERMS = (
  mdp.released_at_goal,
  mdp.released_at_goal_graded,
  mdp.object_placed_reward,
  mdp.object_placed_graded,
  mdp.retreat_height,
  mdp.set_down_placed,
)


def _z(rewards: dict[str, RewardTermCfg], ctx: RewardContext) -> None:
  hit = [n for n, t in rewards.items() if t.func in _AT_GOAL_TERMS]
  assert {"release", "placed"} <= set(hit), (
    "Z redefines M1's at-goal test; apply M1 first"
  )
  for name in hit:
    term = rewards[name]
    assert "threshold" in term.params, name
    rewards[name] = RewardTermCfg(
      func=term.func,
      weight=term.weight,
      params={**term.params, "threshold": Z_GOAL_TOL},
    )


# -- S8: EQ, the height the pads close at (plan S8, BALL-EQ) ---------------------------------
#
# Plan S8 (designed 2026-09-30 22:30-23:00, Siwei). Parallel pads touch a ball at two points on
# the line between them; a pad centre above or below the ball's equator squeezes the ball out
# (step 0 measures the slip against the height, `scripts/gripper_bench.py --stimulus ballheight`).
# EQ pays G(pad-centre height - ball-centre height - EQ_OFFSET, EQ_STD) while the pads are around the
# ball -- within `EQ_NEAR` of it horizontally -- and it has not been lifted yet (`was_lifted`), so
# the last centimetres of the approach, and the close, are steered to the height that holds.
# Weight 1: the order of one `grasp` step, well under `lift`, `goal` and `placed`, so hovering
# at the right height is never worth more than carrying.
#
# EQ_OFFSET is the pad-site midpoint's height above the ball's centre at which it is paid: -10 mm, the
# middle of what holds, not the plan's 0 (the equator). The gripper bench (`scripts/gripper_bench.py
# --stimulus ballheight`, `../results/S8-ball/bench/`, G20, the arm shaken at the stack task's speeds for
# 30 s) loses no ball with the pad midpoint from 20 mm below the ball's centre to its centre, and loses
# every one from 5 mm above it (the contact normals tilt up and the jaw squeezes the ball out); 20 mm
# below, the ball sits 4 mm into the palm. At the equator the retained force is lowest (48 N against
# 62 N at -15 mm) and the ball creeps most (17 mm against 14 mm in 30 s). The scripted pick-and-place
# confirms it: with the pad midpoint at the equator the ball was let go 1.5-8.6 cm above its resting
# height in 8 of the 11 grasped corner episodes (placed 4 / 12); at -10 mm it is placed 60 / 64, none let go > 3 cm up.
EQ_WEIGHT = 1.0
EQ_STD = 0.005  # m
EQ_NEAR = 0.03  # m, horizontal
EQ_OFFSET = -0.010  # m


def _eq(rewards: dict[str, RewardTermCfg], ctx: RewardContext) -> None:
  assert ctx.pads is not None, "plan S8's EQ needs RewardContext.pads"
  rewards["pad_height"] = RewardTermCfg(
    func=mdp.grasp_height_match,
    weight=EQ_WEIGHT,
    params={
      "object_name": OBJECT,
      "grasp_sensor_name": ctx.grasp_sensor,
      "support_sensor_name": SUPPORT_SENSOR,
      "pad_cfg": ctx.pads,
      "offset": EQ_OFFSET,
      "std": EQ_STD,
      "near": EQ_NEAR,
    },
  )


# -- S9: RG (pay the release grade only for an OK let-go) and SA (a gradient towards OK) --------
#
# Plan S9 revised again (Siwei, 2026-10-01 22:04-22:20): stack, and let go only when aligned and
# slow. The release condition OK (`mdp.stack`) is within 1 cm of the goal on the base, at most 1 cm
# above it, tilted and turned against the base by at most 10 deg, slower than 5 cm/s. Three ways
# of enforcing it are compared: the lock (`GATE=stack`, the action term), RG and SA.
#
# RG pays M6's grade on `release` and `placed` only when OK held at the let-go
# (`release_quality(require_ok=True)`, the release latch's ``rel_ok``); any other release is graded
# at the floor (0.25), so a sloppy let-go that still lands on the base keeps `placed` x 0.25.
#
# SA pays prod_i G(x_i / bound_i) over the five OK quantities while the cube is held after the lift
# within `SA_NEAR` of the goal horizontally (`mdp.stack_alignment`): 1 per step at a perfect hold on
# the base. Weight 1, the order of one `grasp` step, and checked by the L5 ordering test: a hold on
# the base must still earn less per step than the placed state.
#: The base cube's entity name (the old `stack` task's; `pick_place_env_cfg.GOAL=stack`).
BASE_OBJECT = "base_object"
SA_WEIGHT = 1.0
SA_NEAR = 0.03  # m, horizontal


def _rg(rewards: dict[str, RewardTermCfg], ctx: RewardContext) -> None:
  for name in ("release", "placed"):
    term = rewards.get(name)
    assert term is not None and "floor" in term.params, (
      "RG pays M6's grade for an OK let-go only; apply M6 first"
    )
    rewards[name] = RewardTermCfg(
      func=term.func, weight=term.weight, params={**term.params, "require_ok": True}
    )


def _sa(rewards: dict[str, RewardTermCfg], ctx: RewardContext) -> None:
  rewards["stack_align"] = RewardTermCfg(
    func=mdp.stack_alignment,
    weight=SA_WEIGHT,
    params={
      "command_name": COMMAND,
      "object_name": OBJECT,
      "reference_name": BASE_OBJECT,
      "grasp_sensor_name": ctx.grasp_sensor,
      "support_sensor_name": SUPPORT_SENSOR,
      "near": SA_NEAR,
    },
  )


# -- S7-9: placing precision at the stack speed (PXY, PT; plan S7-9, designed 2026-10-01 13:20) --
#
# At the stack speeds ABS-a lets go 1.5-2.5 cm off the target, tilted 21-28 deg and at 0.07-0.19
# m/s, and between 3 k and 5 k the release drifts out while M6's grade halves (S7-8 eval). The
# table forgives it; a cube on a cube (S9) or in a box (S10) does not. Two reward modules, each
# resumed from ABS-a's 3 k checkpoint against the unchanged control (P0):
#
# * PXY narrows the grade: M6's xy width 0.02 -> 0.01 m and M2's ``goal_near`` 0.03 -> 0.01 m.
#   The lowering path over the target stays monotone (analytic check in
#   `tests/test_pick_place_s79.py`), but is nearly flat between 3 and 5 cm up (+0.01).
# * PT grades the let-go's tilt and speed as well: M6's graded part x G_10deg(tilt) x
#   G_0.05(speed), both at the pads' let-go (`observations._release_latch`). Inside M6's
#   ``(1 - floor)`` part, so a poor release keeps the floor (`mdp.release_quality_of`).
#
# Plan S7-9's other two codes are not reward modules: PQ is a critic input (`CRITIC_OBS=q`) and PV
# a tool speed cap (`TOOL_SPEED=PV`).
PXY_DIST_STD = 0.01  # m, M6's xy width (was M6_DIST_STD)
PXY_NEAR_STD = 0.01  # m, M2's goal_near width (was M2_NEAR_STD)
PT_TILT_STD = math.radians(10.0)  # rad
PT_SPEED_STD = 0.05  # m/s, `observations.STILL_SPEED`


def _graded_terms(rewards: dict[str, RewardTermCfg], module: str) -> list[str]:
  names = [n for n in ("release", "placed") if n in rewards]
  assert names and all("floor" in rewards[n].params for n in names), (
    f"{module} changes M6's grade; apply M6 first"
  )
  return names


def _pxy(rewards: dict[str, RewardTermCfg], ctx: RewardContext) -> None:
  for name in _graded_terms(rewards, "PXY"):
    term = rewards[name]
    rewards[name] = RewardTermCfg(
      func=term.func,
      weight=term.weight,
      params={**term.params, "dist_std": PXY_DIST_STD},
    )
  near = rewards.get("goal_near")
  assert near is not None, "PXY narrows M2's goal_near; apply M2 first"
  rewards["goal_near"] = RewardTermCfg(
    func=near.func, weight=near.weight, params={**near.params, "std": PXY_NEAR_STD}
  )


def _pt(rewards: dict[str, RewardTermCfg], ctx: RewardContext) -> None:
  for name in _graded_terms(rewards, "PT"):
    term = rewards[name]
    rewards[name] = RewardTermCfg(
      func=term.func,
      weight=term.weight,
      params={**term.params, "tilt_std": PT_TILT_STD, "speed_std": PT_SPEED_STD},
    )


# -- S11: YG, the yaw grade on the let-go (plan S11's L-T: A's yaw) -----------------------------
#
# Plan S11 (Siwei 2026-10-02): the left arm puts cube A down at T for the right to stack on, so A's
# yaw matters to the right (its SA pays alignment against A). L-T grades it as S10's BX graded the
# cube into the box: M6's graded part x G(|yaw mod 90 deg|, 10 deg) at the let-go
# (`observations.yaw_error_mod_90`, the release latch's ``yaw``). In S10 it put 50-73 % of let-gos
# within 10 deg at no cost in success (S10 report §7).
YG_YAW_STD = math.radians(10.0)


def _yg(rewards: dict[str, RewardTermCfg], ctx: RewardContext) -> None:
  for name in _graded_terms(rewards, "YG"):
    term = rewards[name]
    rewards[name] = RewardTermCfg(
      func=term.func, weight=term.weight, params={**term.params, "yaw_std": YG_YAW_STD}
    )


# -- S9.1: SAW, SA with widths a let-go can reach (plan S9.1's SC-SAw) ---------------------------
#
# SA was dead (L15): it paid 0.0000 per episode in DEP1 STSA-c's eval sums and ~2e-5 in R-ST's
# training logs, because its Gaussians have the `OK` bounds as widths and the policies let go far
# outside them (1.1-1.4 cm, tilt 14-22 deg, ~0.1 m/s). SAW widens them to 2 cm (xy), 2 cm
# (height), 20 deg (tilt) and 0.10 m/s (speed) and drops the yaw (the base's yaw is not observed
# in SC-SAw). Its weight is set so that it pays >= 1 % of the return (plan S9.1 step 0 item 4,
# measured on DEP1 STSA-c s1 @ 3000's eval sums) while a hold over the base still earns less per
# step than the placed state (the L5 ordering test).
SAW_WIDTHS = (0.02, 0.02, math.radians(20.0), None, 0.10)
SAW_WEIGHT = 1.0


def _saw(rewards: dict[str, RewardTermCfg], ctx: RewardContext) -> None:
  _sa(rewards, ctx)
  term = rewards["stack_align"]
  rewards["stack_align"] = RewardTermCfg(
    func=term.func,
    weight=SAW_WEIGHT,
    params={**term.params, "widths": SAW_WIDTHS},
  )


_MODULES = {
  "M1": _m1,
  "M2": _m2,
  "M3": _m3,
  "M4": _m4,
  "M5": _m5,
  "M6": _m6,
  "M7": _m7,
  "J": _j,
  "M8": _m8,
  "S": _s,
  "O": _o,
  "Z": _z,
  "EQ": _eq,
  "RG": _rg,
  "SA": _sa,
  "PXY": _pxy,
  "PT": _pt,
  "YG": _yg,
  "SAW": _saw,
}


def _s52_arm(modules: tuple[str, ...]):
  if "M1" in modules and "M4" in modules:
    raise ValueError("M1 removes the success termination that M4 keeps; pick one")
  if "M6" in modules and "M1" not in modules[: modules.index("M6")]:
    raise ValueError("M6 grades M1's release and placed terms; it needs M1 before it")
  if "M7" in modules and "M6" not in modules[: modules.index("M7")]:
    raise ValueError("M7 pays M6's grade for a set-down only; it needs M6 before it")
  if "J" in modules and "M1" not in modules[: modules.index("J")]:
    raise ValueError("J redefines M1's release and placed; it needs M1 before it")
  for name, needs in (("M8", ("J",)), ("S", ("M6", "J"))):
    if name in modules and not all(n in modules[: modules.index(name)] for n in needs):
      raise ValueError(f"{name} needs {' and '.join(needs)} before it")
  if "M7" in modules and "S" in modules:
    raise ValueError("S is M7 revised; pick one")
  if "O" in modules and "J" not in modules[: modules.index("O")]:
    raise ValueError("O needs J before it")
  if "Z" in modules and ("M1" not in modules or modules[-1] != "Z"):
    raise ValueError(
      "Z redefines every at-goal term after M1; it needs M1 and comes last"
    )
  if "EQ" in modules and "M1" not in modules:
    raise ValueError("EQ is a term beside M1's; it needs M1 in the arm")
  if "RG" in modules and "M6" not in modules[: modules.index("RG")]:
    raise ValueError("RG pays M6's grade for an OK let-go only; it needs M6 before it")
  for name in ("SA", "SAW"):
    if name in modules and "M1" not in modules:
      raise ValueError(f"{name} is a term beside M1's; it needs M1 in the arm")
  if "SA" in modules and "SAW" in modules:
    raise ValueError("SAW is SA with other widths; pick one")
  for name, needs in (("PXY", ("M2", "M6")), ("PT", ("M6",)), ("YG", ("M6",))):
    if name in modules and not all(n in modules[: modules.index(name)] for n in needs):
      raise ValueError(f"{name} needs {' and '.join(needs)} before it")

  def build(ctx: RewardContext) -> dict[str, RewardTermCfg]:
    rewards = REWARD_VARIANTS[S52_BASE](ctx)
    for name in modules:
      _MODULES[name](rewards, ctx)
    return rewards

  build.__doc__ = f"S5.2: {S52_BASE} + {' + '.join(modules)}."
  return build


#: The S5.2 arms (plan S5.2, revision 10:27), by the modules on top of the base.
S52_ARMS = {
  f"{S52_BASE}+M1": ("M1",),
  f"{S52_BASE}+M1+M3": ("M1", "M3"),
  f"{S52_BASE}+M1+M2+M3": ("M1", "M2", "M3"),
  f"{S52_BASE}+M3+M4": ("M3", "M4"),
}
REWARD_VARIANTS.update({name: _s52_arm(mods) for name, mods in S52_ARMS.items()})

#: The S7 arms: FB is the full fix plus M5, FR the full fix plus M6 (plan S7). FA is the full
#: fix itself, with the slew bound set through the action term (`SLEW`), not through the
#: reward -- and so is FR's.
S7_ARMS = {
  f"{S52_BASE}+M1+M2+M3+M5": ("M1", "M2", "M3", "M5"),
  f"{S52_BASE}+M1+M2+M3+M6": ("M1", "M2", "M3", "M6"),
}
REWARD_VARIANTS.update({name: _s52_arm(mods) for name, mods in S7_ARMS.items()})
#: The S7-2 arm: FR plus M7, for the B-S and B-SV runs (plan S7-2).
S72_ARMS = {
  f"{S52_BASE}+M1+M2+M3+M6+M7": ("M1", "M2", "M3", "M6", "M7"),
}
REWARD_VARIANTS.update({name: _s52_arm(mods) for name, mods in S72_ARMS.items()})
#: The S7-4 arms (plan S7-4, revised 2026-09-28 21:33): FR + J + M8 for every code, + S for
#: R-CTPS. The effort bound (C, TP) and V are not reward modules: `EFFORT`, `TOOL_SPEED`.
S74_ARMS = {
  f"{S52_BASE}+M1+M2+M3+M6+J+M8": ("M1", "M2", "M3", "M6", "J", "M8"),
  f"{S52_BASE}+M1+M2+M3+M6+J+M8+S": ("M1", "M2", "M3", "M6", "J", "M8", "S"),
}
REWARD_VARIANTS.update({name: _s52_arm(mods) for name, mods in S74_ARMS.items()})
#: The S7-5 arm (plan S7-5): S7-4's RCTP reward + O + Z. The control, RCTP, is S7-4's first arm.
S75_ARMS = {
  f"{S52_BASE}+M1+M2+M3+M6+J+M8+O+Z": ("M1", "M2", "M3", "M6", "J", "M8", "O", "Z"),
}
REWARD_VARIANTS.update({name: _s52_arm(mods) for name, mods in S75_ARMS.items()})
#: The S8 arm (plan S8's BALL-EQ): S7-4's RCTP reward + EQ, the pads' closing height. The other
#: S8 codes (BALL, BALL-FT, BALL-DR, MIX) are RCTP itself on another object (`SHAPE`).
S8_ARMS = {
  f"{S52_BASE}+M1+M2+M3+M6+J+M8+EQ": ("M1", "M2", "M3", "M6", "J", "M8", "EQ"),
}
REWARD_VARIANTS.update({name: _s52_arm(mods) for name, mods in S8_ARMS.items()})
#: The S9 arms (plan S9 revised again, 2026-10-01 22:04-22:20): S7-4's RCTP + RG (ST-R) and RCTP + SA
#: (ST-GA, with the lock). ST, ST-G and ST-GFT are RCTP itself on the stack goal (`GOAL`, `GATE`).
S9_ARMS = {
  f"{S52_BASE}+M1+M2+M3+M6+J+M8+RG": ("M1", "M2", "M3", "M6", "J", "M8", "RG"),
  f"{S52_BASE}+M1+M2+M3+M6+J+M8+SA": ("M1", "M2", "M3", "M6", "J", "M8", "SA"),
}
REWARD_VARIANTS.update({name: _s52_arm(mods) for name, mods in S9_ARMS.items()})
#: The S7-9 arms (plan S7-9): S7-4's RCTP reward + one precision module. P0 (the control) and the
#: other two codes, PQ (`CRITIC_OBS`) and PV (`TOOL_SPEED`), run RCTP itself.
S79_ARMS = {
  f"{S52_BASE}+M1+M2+M3+M6+J+M8+PXY": ("M1", "M2", "M3", "M6", "J", "M8", "PXY"),
  f"{S52_BASE}+M1+M2+M3+M6+J+M8+PT": ("M1", "M2", "M3", "M6", "J", "M8", "PT"),
}
REWARD_VARIANTS.update({name: _s52_arm(mods) for name, mods in S79_ARMS.items()})
#: The S11 arm (plan S11's L-T): S7-4's RCTP + YG, A's yaw graded at the let-go. R-ST / R-ST-W run
#: S9's RCTP + SA on a stack goal at T (`TARGET=T11*`); JOINT builds its own (`pick_place/bimanual.py`).
S11_ARMS = {
  f"{S52_BASE}+M1+M2+M3+M6+J+M8+YG": ("M1", "M2", "M3", "M6", "J", "M8", "YG"),
}
REWARD_VARIANTS.update({name: _s52_arm(mods) for name, mods in S11_ARMS.items()})
#: The S9.1 arm (plan S9.1's SC-SAw): S7-4's RCTP + SAW. SC and SC-FT run RCTP itself on the stack goal
#: at T; SC-Y runs S11's RCTP + YG there with the base's yaw observed (`BASE_OBS=pose`).
S91_ARMS = {
  f"{S52_BASE}+M1+M2+M3+M6+J+M8+SAW": ("M1", "M2", "M3", "M6", "J", "M8", "SAW"),
}
REWARD_VARIANTS.update({name: _s52_arm(mods) for name, mods in S91_ARMS.items()})
_MODULE_ARMS = {
  **S52_ARMS,
  **S7_ARMS,
  **S72_ARMS,
  **S74_ARMS,
  **S75_ARMS,
  **S8_ARMS,
  **S9_ARMS,
  **S79_ARMS,
  **S11_ARMS,
  **S91_ARMS,
}
#: Arms whose terms read the base cube (RG, SA, SAW): they need plan S9's `GOAL=stack`.
STACK_ONLY_ARMS = frozenset(
  n for n, m in _MODULE_ARMS.items() if {"RG", "SA", "SAW"} & set(m)
)
#: Arms that grade the let-go's yaw (YG): on a stack goal, against the base (plan S9.1's SC-Y).
YAW_GRADED_ARMS = frozenset(n for n, m in _MODULE_ARMS.items() if "YG" in m)

#: Arms whose release is the jaw opening (plan S7-4's J): the eval and the scripts read it.
JAW_RELEASE = frozenset(n for n, m in _MODULE_ARMS.items() if "J" in m)
#: Arms whose "at the goal" is plan S7-5's (xy, z) tolerance `Z_GOAL_TOL`.
GOAL_TOL_XYZ = frozenset(n for n, m in _MODULE_ARMS.items() if "Z" in m)

#: Arms whose success *termination* must also require the `was_lifted` latch. The reward
#: dict cannot carry this on its own -- the termination is built by the tabletop factory --
#: so `make_pick_place_env_cfg` reads this set.
REQUIRE_LIFTED = frozenset(
  {"R-C-strict", *(n for n, m in _MODULE_ARMS.items() if "M1" not in m)}
)

#: Arms that train with no success termination at all (M1): the placed state keeps paying
#: per step instead of ending the episode.
NO_SUCCESS_TERMINATION = frozenset(n for n, m in _MODULE_ARMS.items() if "M1" in m)

#: Episode length per variant, where it differs. R-E buys its time penalty a shorter horizon.
EPISODE_LENGTH_S = {"R-E": 12.0}
DEFAULT_EPISODE_LENGTH_S = 20.0


def build_rewards(name: str, ctx: RewardContext) -> dict[str, RewardTermCfg]:
  if name not in REWARD_VARIANTS:
    raise ValueError(f"unknown reward variant {name!r}; have {tuple(REWARD_VARIANTS)}")
  return REWARD_VARIANTS[name](ctx)
