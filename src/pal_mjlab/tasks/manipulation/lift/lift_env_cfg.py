"""Lift an object to a commanded point above the table.

The reference task: it adds nothing to the base configuration beyond a goal sampling range.
That is the point -- if a task needs to reach into ``make_tabletop_env_cfg``'s internals to
work, the base cfg is missing an argument.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.manipulation.mdp import LiftingCommandCfg

from pal_mjlab.tasks.manipulation.robot_cfg import ManipulationRobotCfg
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import (
  OBJECT,
  apply_play_overrides,
  make_tabletop_env_cfg,
)


def make_lift_env_cfg(
  robot: ManipulationRobotCfg,
  gripper_name: str | None = None,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_tabletop_env_cfg(
    robot,
    gripper_name=gripper_name,
    command_cfg=LiftingCommandCfg(
      entity_name=OBJECT,
      difficulty="dynamic",
      # Goal: 15-35 cm above the table, within the reachable band in front of the robot.
      target_position_range=LiftingCommandCfg.TargetPositionRangeCfg(
        x=(0.40, 0.60), y=(-0.25, 0.25), z=(0.90, 1.10)
      ),
      object_pose_range=None,  # Placement is the reset event's job.
      resampling_time_range=(1e9, 1e9),
      debug_vis=True,
    ),
    # Spawn anywhere on the table top, clear of the edges.
    object_pose_range={"x": (-0.15, 0.15), "y": (-0.25, 0.25), "yaw": (-3.14, 3.14)},
  )
  if play:
    apply_play_overrides(cfg)
  return cfg
