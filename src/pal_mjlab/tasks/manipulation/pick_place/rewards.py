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
  guard_reward_terms,
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
  "R-D": r_d_keypoints,
  "R-E": r_e_time_penalty,
}

#: Episode length per variant, where it differs. R-E buys its time penalty a shorter horizon.
EPISODE_LENGTH_S = {"R-E": 12.0}
DEFAULT_EPISODE_LENGTH_S = 20.0


def build_rewards(name: str, ctx: RewardContext) -> dict[str, RewardTermCfg]:
  if name not in REWARD_VARIANTS:
    raise ValueError(f"unknown reward variant {name!r}; have {tuple(REWARD_VARIANTS)}")
  # Every arm is guarded, so a divergence costs one zero-scored step in one world rather
  # than the whole run. It replaces `.func` with a memoised wrapper, so terms that were the
  # same function before still are -- see `guard_reward_terms`.
  return guard_reward_terms(REWARD_VARIANTS[name](ctx))
