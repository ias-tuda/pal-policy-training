"""Stack one block on top of another.

The task that proves the framework is shared rather than a single-task wrapper. Everything
that makes stacking *stacking* is expressed in configuration:

* a second cube, passed to the base cfg as another entry in ``objects``;
* a goal that tracks that cube instead of being sampled
  (:class:`~pal_mjlab.tasks.manipulation.mdp.commands.EntityOffsetCommand`);
* a reward for letting go in the right place, and a success condition that requires it.

No new command subclass, no new scene, no edits to the base cfg. The difference against
``lift`` is one command and three terms.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from pal_mjlab.tasks.manipulation import mdp, scene
from pal_mjlab.tasks.manipulation.robot_cfg import ManipulationRobotCfg
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import (
  COMMAND,
  OBJECT,
  apply_play_overrides,
  make_tabletop_env_cfg,
)

#: The cube that stays put and gets stacked upon.
BASE_OBJECT = "base_object"

#: Placement tolerance. Half the cube width: beyond this the cube is not on the one below.
_GOAL_THRESHOLD = 0.025

#: Cubes must spawn far enough apart that the jaw can reach one without fouling the other.
_MIN_SEPARATION = 0.12


def make_stack_env_cfg(
  robot: ManipulationRobotCfg,
  gripper_name: str | None = None,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_tabletop_env_cfg(
    robot,
    gripper_name=gripper_name,
    objects={
      OBJECT: scene.get_cube_cfg(name=OBJECT, rgba=(0.8, 0.2, 0.2, 1.0)),
      BASE_OBJECT: scene.get_cube_cfg(name=BASE_OBJECT, rgba=(0.2, 0.35, 0.8, 1.0)),
    },
    command_cfg=mdp.EntityOffsetCommandCfg(
      entity_name=OBJECT,
      reference_name=BASE_OBJECT,
      # Resting height is computed from both cubes' half-heights, so the goal stays right
      # if either cube is resized or randomised.
      stack_on_top=True,
      resampling_time_range=(1e9, 1e9),  # One goal per episode.
      success_threshold=_GOAL_THRESHOLD,
      debug_vis=True,
    ),
    object_pose_range={"x": (-0.15, 0.15), "y": (-0.25, 0.25), "yaw": (-3.14, 3.14)},
    min_separation=_MIN_SEPARATION,
  )

  gripper = robot.gripper(gripper_name)
  grasp_sensor = gripper.contact_sensor_name

  # Release shaping. Without it the policy converges on holding the cube at the goal
  # forever: every distance reward is already maximal there, and letting go can only lose
  # points.
  cfg.rewards["release"] = RewardTermCfg(
    func=mdp.released_at_goal,
    weight=2.0,
    params={
      "command_name": COMMAND,
      "object_name": OBJECT,
      "grasp_sensor_name": grasp_sensor,
      "threshold": _GOAL_THRESHOLD,
    },
  )
  cfg.rewards["success"] = RewardTermCfg(
    func=mdp.object_placed_reward,
    weight=10.0,
    params={
      "command_name": COMMAND,
      "object_name": OBJECT,
      "grasp_sensor_name": grasp_sensor,
      "threshold": _GOAL_THRESHOLD,
    },
  )
  # The base cube is scenery the policy must work around, not through.
  cfg.rewards["base_disturbance"] = RewardTermCfg(
    func=mdp.entity_lin_vel_l2, weight=-0.5, params={"entity_name": BASE_OBJECT}
  )

  cfg.terminations["success"] = TerminationTermCfg(
    func=mdp.object_placed,
    params={
      "command_name": COMMAND,
      "object_name": OBJECT,
      "grasp_sensor_name": grasp_sensor,
      "threshold": _GOAL_THRESHOLD,
    },
  )
  cfg.terminations["base_dropped"] = TerminationTermCfg(
    func=mdp.entity_below_height,
    params={"entity_name": BASE_OBJECT, "min_height": scene.TABLE_TOP_Z - 0.2},
  )

  if play:
    apply_play_overrides(cfg)
  return cfg
