"""The MDP term namespace for PAL manipulation tasks.

One import surface for three layers, most general first:

* ``mjlab.envs.mdp`` -- generic terms (``joint_pos_rel``, ``last_action``, ``time_out``,
  ``action_rate_l2``, the reset events, the domain-randomisation helpers).
* ``mjlab.tasks.manipulation.mdp`` -- mjlab's native manipulation terms. These are the
  default choice: prefer adding a term here only after checking it is genuinely absent.
* This package -- the gaps: a goal defined relative to another entity, table-aware
  multi-object placement, and everything about grasping and releasing, which mjlab's
  proximity-based lift task has no notion of.

Exports are explicit. A star-import chain makes the namespace whatever the modules happen
to leave lying around -- including their own imports -- and that is not an interface.
"""

from mjlab.envs.mdp import *  # noqa: F401, F403
from mjlab.tasks.manipulation.mdp import (
  bring_object_reward,
  ee_to_object_distance,
  ee_velocity,
  illegal_contact,
  joint_velocity_hinge_penalty,
  object_to_goal_distance,
  reward_curriculum,
  staged_position_reward,
  target_position,
)

from .actions import (
  RateLimitedJointPositionAction,
  RateLimitedJointPositionActionCfg,
  ToolSpeedCappedJointPositionAction,
  ToolSpeedCappedJointPositionActionCfg,
)
from .commands import (
  EntityOffsetCommand,
  EntityOffsetCommandCfg,
  SurfaceOffsetCommand,
  SurfaceOffsetCommandCfg,
)
from .events import (
  hold_nominal_targets,
  reset_objects_on_surface,
  reset_release_latch,
  reset_was_lifted,
  surface_top_z,
)
from .observations import (
  ee_orientation_b,
  ee_position_b,
  goal_position_b,
  is_grasped,
  is_on_surface,
  object_position_b,
  object_yaw_b,
)
from .occlusion import (
  CubeSensor,
  OccCfg,
  object_position_occ,
  object_seen,
  object_unseen_steps,
  object_yaw_occ,
)
from .rewards import (
  arm_effort_hinge,
  bring_object_after_lift,
  bring_object_when_lifted,
  contact_penalty,
  entity_lin_vel_l2,
  grasped,
  held_object_slide_speed,
  held_object_speed_hinge,
  joint_pos_limits,
  keepout_box_penalty,
  keypoint_distance_reward,
  lift_height,
  lift_height_faded,
  object_lifted,
  object_placed_graded,
  open_at_goal,
  pad_contact_after_release,
  reach_only,
  release_quality,
  released_at_goal,
  released_at_goal_graded,
  retreat_height,
  set_down_placed,
  time_penalty,
)
from .rewards import object_placed as object_placed_reward
from .terminations import (
  entity_below_height,
  entity_out_of_reach,
  non_finite_state,
  object_placed,
)

__all__ = [
  "CubeSensor",
  "OccCfg",
  "object_position_occ",
  "object_seen",
  "object_unseen_steps",
  "object_yaw_occ",
  "RateLimitedJointPositionAction",
  "RateLimitedJointPositionActionCfg",
  "ToolSpeedCappedJointPositionAction",
  "ToolSpeedCappedJointPositionActionCfg",
  "entity_out_of_reach",
  "keepout_box_penalty",
  # mjlab manipulation terms, re-exported so a task needs one namespace.
  "bring_object_reward",
  "ee_to_object_distance",
  "ee_velocity",
  "illegal_contact",
  "joint_velocity_hinge_penalty",
  "object_to_goal_distance",
  "reward_curriculum",
  "staged_position_reward",
  "target_position",
  # Commands.
  "EntityOffsetCommand",
  "EntityOffsetCommandCfg",
  "SurfaceOffsetCommand",
  "SurfaceOffsetCommandCfg",
  # Events.
  "hold_nominal_targets",
  "reset_objects_on_surface",
  "surface_top_z",
  # Observations.
  "ee_orientation_b",
  "ee_position_b",
  "goal_position_b",
  "is_grasped",
  "is_on_surface",
  "object_position_b",
  "object_yaw_b",
  # Rewards.
  "arm_effort_hinge",
  "bring_object_after_lift",
  "bring_object_when_lifted",
  "contact_penalty",
  "entity_lin_vel_l2",
  "grasped",
  "keypoint_distance_reward",
  "lift_height",
  "held_object_slide_speed",
  "held_object_speed_hinge",
  "lift_height_faded",
  "time_penalty",
  "joint_pos_limits",
  "object_lifted",
  "object_placed_graded",
  "open_at_goal",
  "object_placed_reward",
  "release_quality",
  "released_at_goal",
  "released_at_goal_graded",
  "pad_contact_after_release",
  "retreat_height",
  "set_down_placed",
  # Terminations.
  "entity_below_height",
  "non_finite_state",
  "reach_only",
  "reset_release_latch",
  "reset_was_lifted",
  "object_placed",
]
