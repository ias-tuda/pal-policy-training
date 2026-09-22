"""Pick a cube off the table and put it down somewhere else on the table.

Stack with the second cube replaced by a fixed point: one arm, one object, a goal that does
not move, and the same release-and-let-go shaping that makes ``stack`` a placing task rather
than a carrying one. The difference against ``stack`` is one command and one fewer object.

The two positions are not free parameters. They come out of the S1 reach map
(``../docs/reports/pick-place-v0/s1-report.md``): the left jaw can take a cube from above on
353 of the table's 1200 2 cm cells, a band along the near edge, and these are the two points
inside it that are 20 cm apart with the most clearance to its edge. A target chosen off that
map is an unlearnable task, not a hard one.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.manipulation.mdp import LiftingCommandCfg

from pal_mjlab.tasks.manipulation import mdp, scene
from pal_mjlab.tasks.manipulation.robot_cfg import ManipulationRobotCfg
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import (
  COMMAND,
  OBJECT,
  apply_dr_level,
  apply_play_overrides,
  make_tabletop_env_cfg,
)

#: Where the cube starts and where it has to end up, in the env frame. Measured, see the
#: module docstring: clearance to the edge of the graspable region is 14.0 cm at the spawn
#: and 4.5 cm at the target, and they are 20.0 cm apart.
SPAWN_XY = (0.42, 0.27)
TARGET_XY = (0.42, 0.07)

#: Goal height: the cube's centre when it is resting on the table. Not the height the grasp
#: point goes to -- the pads reach 25.6 mm past ``tcp_left``, so a top-down grasp puts the
#: TCP 11 mm higher than this (S1.3). Anything that drives the TCP at the goal needs to know.
TARGET_Z = scene.TABLE_TOP_Z + 0.025

#: "Lightly noised", per the plan's v0 scope: the cube is where the policy was told it is,
#: give or take 2 cm and 10 degrees. At this size the whole spawn box sits inside the 14 cm
#: clearance, so every sampled pose is graspable.
SPAWN_JITTER_XY = 0.02
SPAWN_JITTER_YAW = 0.1745  # 10 degrees

#: Placement tolerance, and the same number the success termination uses. Half the cube
#: width: beyond this the cube is not on the target.
GOAL_THRESHOLD = 0.025


def _object_pose_range(spawn_xy: tuple[float, float]) -> dict[str, tuple[float, float]]:
  """Spawn box for ``mdp.reset_objects_on_surface``, which samples about the table centre.

  The event adds the table geom's own world XY to whatever range it is given, so that table
  height randomisation and env origins are picked up for free. That makes its range a
  *relative* one, and a spawn point measured in the env frame has to be converted.
  """
  dx = spawn_xy[0] - scene.TABLE_POS[0]
  dy = spawn_xy[1] - scene.TABLE_POS[1]
  return {
    "x": (dx - SPAWN_JITTER_XY, dx + SPAWN_JITTER_XY),
    "y": (dy - SPAWN_JITTER_XY, dy + SPAWN_JITTER_XY),
    "yaw": (-SPAWN_JITTER_YAW, SPAWN_JITTER_YAW),
  }


def make_pick_place_env_cfg(
  robot: ManipulationRobotCfg,
  gripper_name: str | None = None,
  target_xy: tuple[float, float] = TARGET_XY,
  spawn_xy: tuple[float, float] = SPAWN_XY,
  dr: str = "DR0",
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Pick-and-place on a fixed target.

  Args:
    robot: the robot description; the only robot-specific input.
    gripper_name: which gripper the task uses. Defaults to the robot's first.
    target_xy: where the cube has to end up, in the env frame.
    spawn_xy: centre of the cube's spawn box, in the env frame.
    dr: domain-randomisation level, by name. See ``tabletop_env_cfg.DR_LEVELS``.
    play: apply the play overrides (no observation noise, no timeout).
  """
  cfg = make_tabletop_env_cfg(
    robot,
    gripper_name=gripper_name,
    command_cfg=LiftingCommandCfg(
      entity_name=OBJECT,
      # A degenerate "dynamic" range rather than difficulty="fixed": mjlab's fixed mode
      # hardcodes (0.4, 0, 0.3), which is 46 cm below this table. Keeping the target a
      # sampled quantity with zero width is also what makes v1 -- a target the policy has
      # to read -- a change to two numbers rather than a change of command term.
      difficulty="dynamic",
      target_position_range=LiftingCommandCfg.TargetPositionRangeCfg(
        x=(target_xy[0], target_xy[0]),
        y=(target_xy[1], target_xy[1]),
        z=(TARGET_Z, TARGET_Z),
      ),
      object_pose_range=None,  # Placement is the reset event's job; it knows about the table.
      resampling_time_range=(1e9, 1e9),  # One goal per episode.
      success_threshold=GOAL_THRESHOLD,
      debug_vis=True,
    ),
    object_pose_range=_object_pose_range(spawn_xy),
  )

  gripper = robot.gripper(gripper_name)
  grasp_sensor = gripper.contact_sensor_name

  # Release shaping, exactly as in `stack`. Without it the policy converges on holding the
  # cube at the goal forever: every distance reward is already maximal there, and letting go
  # can only lose points.
  cfg.rewards["release"] = RewardTermCfg(
    func=mdp.released_at_goal,
    weight=2.0,
    params={
      "command_name": COMMAND,
      "object_name": OBJECT,
      "grasp_sensor_name": grasp_sensor,
      "threshold": GOAL_THRESHOLD,
    },
  )
  cfg.rewards["success"] = RewardTermCfg(
    func=mdp.object_placed_reward,
    weight=10.0,
    params={
      "command_name": COMMAND,
      "object_name": OBJECT,
      "grasp_sensor_name": grasp_sensor,
      "threshold": GOAL_THRESHOLD,
    },
  )
  cfg.terminations["success"] = TerminationTermCfg(
    func=mdp.object_placed,
    params={
      "command_name": COMMAND,
      "object_name": OBJECT,
      "grasp_sensor_name": grasp_sensor,
      "threshold": GOAL_THRESHOLD,
    },
  )

  apply_dr_level(cfg, dr)
  if play:
    apply_play_overrides(cfg)
  return cfg
