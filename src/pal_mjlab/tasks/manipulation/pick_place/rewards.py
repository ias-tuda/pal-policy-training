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

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from pal_mjlab.tasks.manipulation import mdp
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


_MODULES = {"M1": _m1, "M2": _m2, "M3": _m3, "M4": _m4}


def _s52_arm(modules: tuple[str, ...]):
  if "M1" in modules and "M4" in modules:
    raise ValueError("M1 removes the success termination that M4 keeps; pick one")

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

#: Arms whose success *termination* must also require the `was_lifted` latch. The reward
#: dict cannot carry this on its own -- the termination is built by the tabletop factory --
#: so `make_pick_place_env_cfg` reads this set.
REQUIRE_LIFTED = frozenset(
  {"R-C-strict", *(n for n, m in S52_ARMS.items() if "M1" not in m)}
)

#: Arms that train with no success termination at all (M1): the placed state keeps paying
#: per step instead of ending the episode.
NO_SUCCESS_TERMINATION = frozenset(n for n, m in S52_ARMS.items() if "M1" in m)

#: Episode length per variant, where it differs. R-E buys its time penalty a shorter horizon.
EPISODE_LENGTH_S = {"R-E": 12.0}
DEFAULT_EPISODE_LENGTH_S = 20.0


def build_rewards(name: str, ctx: RewardContext) -> dict[str, RewardTermCfg]:
  if name not in REWARD_VARIANTS:
    raise ValueError(f"unknown reward variant {name!r}; have {tuple(REWARD_VARIANTS)}")
  return REWARD_VARIANTS[name](ctx)
