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

import dataclasses
import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.manipulation.mdp import LiftingCommandCfg

from pal_mjlab.tasks.manipulation import mdp, scene
from pal_mjlab.tasks.manipulation.pick_place.rewards import (
  DEFAULT_EPISODE_LENGTH_S,
  EPISODE_LENGTH_S,
  NO_SUCCESS_TERMINATION,
  REQUIRE_LIFTED,
  RewardContext,
  build_rewards,
  reward_curriculum_cfg,
)
from pal_mjlab.tasks.manipulation.robot_cfg import ManipulationRobotCfg
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import (
  COMMAND,
  DR_LEVELS_THAT_MOVE_THE_TABLE,
  OBJECT,
  ROBOT,
  SUPPORT_SENSOR,
  TABLE,
  ToolSpeedCapCfg,
  apply_play_overrides,
  make_tabletop_env_cfg,
  trunk_clearance_rewards,
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


@dataclasses.dataclass(frozen=True)
class SpawnBox:
  """Where the cube starts: a box in the env frame and a yaw range (plan S7-2, `SPAWN`)."""

  x: tuple[float, float]
  y: tuple[float, float]
  yaw: float
  """Half-range of the cube's yaw, rad. A cube is 4-fold symmetric, so pi/4 is every yaw."""

  widen_at_dr_scene: bool = True
  """Whether `DR-scene` widens the box by its +/-3 cm. The small box S7-1 trained on is
  widened; a W box is already the region to train on and is not (plan S7-2)."""


#: The spawn boxes by name (plan S7 P2, approved by Siwei 2026-09-25 17:49). ``S71`` is the
#: +/-2 cm, +/-10 deg box every run up to S7-1 trained on. ``W1`` is +/-5 cm about the same
#: centre and ``W2`` the widest box the arm grasps through its action space with a 4 cm margin
#: (S7 report section 6), both at every yaw; they need the wider action scales of
#: `robots.KANGAROO_GRIPPERS_LEFT_ARM_WIDE`, and `kangaroo/env_cfg.py` pairs them.
SPAWN_BOXES = {
  "S71": SpawnBox(
    x=(SPAWN_XY[0] - SPAWN_JITTER_XY, SPAWN_XY[0] + SPAWN_JITTER_XY),
    y=(SPAWN_XY[1] - SPAWN_JITTER_XY, SPAWN_XY[1] + SPAWN_JITTER_XY),
    yaw=SPAWN_JITTER_YAW,
  ),
  "W1": SpawnBox(
    x=(0.37, 0.47), y=(0.22, 0.32), yaw=math.pi / 4, widen_at_dr_scene=False
  ),
  "W2": SpawnBox(
    x=(0.34, 0.46), y=(0.20, 0.34), yaw=math.pi / 4, widen_at_dr_scene=False
  ),
}


@dataclasses.dataclass(frozen=True)
class Baseline:
  """The environment a run starts from, apart from its reward, DR rung, spawn and speed bounds.

  Plan S7-2's baseline B changes five things against S7-1, each a field here, so that B can
  be taken apart setting by setting against S7-1's (``S71``) when it does not do what FR did.
  """

  table_top_z: float
  """Height of the table's top face, m."""

  episode_length_s: float | None
  """Episode length, s; None keeps the reward variant's own (`rewards.EPISODE_LENGTH_S`)."""

  target_xy: tuple[float, float]
  """Where the cube has to end up, in the env frame for the table at its nominal place: the
  target, or the centre of the target box."""

  target_half_xy: float
  """Half-width in x and y of the box the target is drawn from once per episode, relative to
  the table, m. 0 is the fixed target every run up to S7-1 had."""

  observation_set: str
  """`tabletop_env_cfg.OBSERVATION_SETS`."""

  jaw_slew_rate: float | None
  """Largest change of the commanded jaw target per control step, rad; None for none."""


#: Baselines by name (plan S7-2, decided by Siwei 2026-09-25 18:41). ``B``: the deployment
#: cell's table height, 10 s episodes, a target drawn per episode from a table-relative box,
#: the 38-D deployable actor (`tabletop_env_cfg._deployable_observations`) and the jaw at PAL's
#: 2 rad/s.
#:
#: B's target box is (0.46, 0.08) +/-2 cm, not the plan's proposed (0.42, 0.07) +/-3 cm (Siwei,
#: 2026-09-26, on the scripted corners check the plan made the arbiter of the box). At the
#: 0.815 m table the arm cannot place near the torso with its upper arm clear: over the plan's
#: box plus DR-scene's +/-2 cm table shift, the script's planner found 11-18 % of targets
#: unplannable from the S71 spawn (66 % for a cube at +10 deg), and even the point (0.42, 0.07)
#: 2-8 %. Shrinking the box cannot help when its centre is on the edge; at (0.46, 0.08) +/-2 cm
#: it is 0-2 % on every spawn and yaw mapped, and the spawn-target clearance to W2 stays 10 cm.
#: `artifacts/s7-2/ikmap/`; `tests/test_pick_place_s72_corners.py` is the rollout check.
BASELINES = {
  "S71": Baseline(
    table_top_z=scene.TABLE_TOP_Z,
    episode_length_s=None,
    target_xy=TARGET_XY,
    target_half_xy=0.0,
    observation_set="S71",
    jaw_slew_rate=None,
  ),
  "B": Baseline(
    table_top_z=scene.ROBOT_CELL_TABLE_TOP_Z,
    episode_length_s=10.0,
    target_xy=(0.46, 0.08),
    target_half_xy=0.02,
    observation_set="B",
    jaw_slew_rate=0.04,
  ),
}


#: Tool speed caps by name (plan S7-2, module V; the values confirmed by Siwei 2026-09-25):
#: 0.3 m/s, and 0.1 m/s while the held cube is within 5 cm of the table or of the target.
#: ISO 10218-1's reduced speed is 0.25 m/s, for reference; the trained policies peak at ~0.9.
TOOL_SPEED_CAPS: dict[str, ToolSpeedCapCfg | None] = {
  "none": None,
  "V": ToolSpeedCapCfg(speed=0.3, slow_speed=0.1, slow_zone=0.05),
}


def _object_pose_range(spawn: SpawnBox) -> dict[str, tuple[float, float]]:
  """Spawn box for ``mdp.reset_objects_on_surface``, which samples about the table centre.

  The event adds the table geom's own world XY to whatever range it is given, so that table
  height randomisation and env origins are picked up for free. That makes its range a
  *relative* one, and a spawn box measured in the env frame has to be converted.
  """
  cx, cy = scene.TABLE_POS[0], scene.TABLE_POS[1]
  return {
    "x": (spawn.x[0] - cx, spawn.x[1] - cx),
    "y": (spawn.y[0] - cy, spawn.y[1] - cy),
    "yaw": (-spawn.yaw, spawn.yaw),
  }


def _command_cfg(
  target_xy: tuple[float, float],
  dr: str,
  table_top_z: float = scene.TABLE_TOP_Z,
  target_half_xy: float = 0.0,
):
  """The goal: a point on the table, expressed in whichever way the DR level allows.

  Below `DR-scene` the table does not move, so a fixed goal is a degenerate "dynamic" range
  -- a sampled point of zero width. That is not `difficulty="fixed"`, which hardcodes
  (0.4, 0, 0.3) and is 46 cm below this table; and keeping it a sampled quantity is what
  makes v1 -- a target the policy has to read -- a change to two numbers.

  At `DR-scene` the table height varies per env, and a constant goal z is then wrong by up
  to 2 cm against a 2.5 cm success threshold. The goal becomes an offset from the table's
  own surface instead, read out of physics -- the same two quantities the spawn event reads,
  so the goal and the spawn agree about where the table is. When the table does not move the
  two commands produce the same point; `test_dr_scene_goal_matches_the_fixed_goal` checks it.

  A target box (``target_half_xy`` > 0, plan S7-2's baseline B) is always the surface-offset
  goal, drawn once per episode relative to the table, at every rung.
  """
  if dr in DR_LEVELS_THAT_MOVE_THE_TABLE or target_half_xy > 0.0:
    return mdp.SurfaceOffsetCommandCfg(
      entity_name=OBJECT,
      surface_name=TABLE,
      offset=(target_xy[0] - scene.TABLE_POS[0], target_xy[1] - scene.TABLE_POS[1]),
      offset_range=(target_half_xy, target_half_xy),
      resampling_time_range=(1e9, 1e9),
      success_threshold=GOAL_THRESHOLD,
      debug_vis=True,
    )
  target_z = table_top_z + 0.025
  return LiftingCommandCfg(
    entity_name=OBJECT,
    difficulty="dynamic",
    target_position_range=LiftingCommandCfg.TargetPositionRangeCfg(
      x=(target_xy[0], target_xy[0]),
      y=(target_xy[1], target_xy[1]),
      z=(target_z, target_z),
    ),
    object_pose_range=None,  # Placement is the reset event's job; it knows about the table.
    resampling_time_range=(1e9, 1e9),  # One goal per episode.
    success_threshold=GOAL_THRESHOLD,
    debug_vis=True,
  )


def make_pick_place_env_cfg(
  robot: ManipulationRobotCfg,
  gripper_name: str | None = None,
  target_xy: tuple[float, float] | None = None,
  spawn: SpawnBox = SPAWN_BOXES["S71"],
  dr: str = "DR0",
  reward: str = "R-A",
  play: bool = False,
  action_slew_rate_scale: float = 1.0,
  baseline: Baseline = BASELINES["S71"],
  tool_speed_cap: ToolSpeedCapCfg | None = None,
) -> ManagerBasedRlEnvCfg:
  """Pick-and-place on a target on the table.

  Args:
    robot: the robot description; the only robot-specific input.
    gripper_name: which gripper the task uses. Defaults to the robot's first.
    target_xy: where the cube has to end up, in the env frame -- the centre of the target
      box when the baseline has one. None takes the baseline's.
    spawn: where the cube starts. See `SPAWN_BOXES`.
    dr: domain-randomisation level, by name. See ``tabletop_env_cfg.DR_LEVELS``.
    reward: reward variant, by name. See ``rewards.REWARD_VARIANTS``. R-A is the framework's
      own reward and the baseline every other arm is measured against.
    play: apply the play overrides (no observation noise, no timeout).
    action_slew_rate_scale: factor on the arm's URDF slew limit; 0.5 is plan S7's speed
      bound A. See `make_tabletop_env_cfg`.
    baseline: table height, episode length, target box, observations and jaw slew. See
      `BASELINES`; ``S71`` is every run up to S7-1.
    tool_speed_cap: plan S7-2's module V, or None. See `tabletop_env_cfg.ToolSpeedCapCfg`.
  """
  if target_xy is None:
    target_xy = baseline.target_xy
  cfg = make_tabletop_env_cfg(
    robot,
    gripper_name=gripper_name,
    command_cfg=_command_cfg(
      target_xy, dr, baseline.table_top_z, baseline.target_half_xy
    ),
    object_pose_range=_object_pose_range(spawn),
    dr_level=dr,
    episode_length_s=baseline.episode_length_s
    or EPISODE_LENGTH_S.get(reward, DEFAULT_EPISODE_LENGTH_S),
    action_slew_rate_scale=action_slew_rate_scale,
    table_top_z=baseline.table_top_z,
    observation_set=baseline.observation_set,
    jaw_slew_rate=baseline.jaw_slew_rate,
    tool_speed_cap=tool_speed_cap,
    dr_spawn_widening=spawn.widen_at_dr_scene,
  )

  gripper = robot.gripper(gripper_name)
  grasp_sensor = gripper.contact_sensor_name

  # The whole reward dict comes from the named variant, replacing the base cfg's. Replacing
  # rather than patching is what makes the arms of the comparison comparable: two variants
  # cannot differ by a term one of them forgot to remove.
  cfg.rewards = build_rewards(
    reward,
    RewardContext(
      tcp=SceneEntityCfg(ROBOT, site_names=(gripper.tcp_site,)),
      action_joints=SceneEntityCfg(ROBOT, joint_names=robot.action_joint_regex),
      grasp_sensor=grasp_sensor,
      surface_sensor=gripper.surface_sensor_name,
      threshold=GOAL_THRESHOLD,
    ),
  )
  # Not part of any arm's comparison: every variant carries the same trunk penalty.
  cfg.rewards.update(trunk_clearance_rewards(robot))
  cfg.curriculum = reward_curriculum_cfg()

  success_params = {
    "command_name": COMMAND,
    "object_name": OBJECT,
    "grasp_sensor_name": grasp_sensor,
    "threshold": GOAL_THRESHOLD,
  }
  # Unconditionally, not only for the arms that gate on it. The latch is allocated by the
  # first thing that reads it, and `scripts/eval_pick_place.py` reads it on *every* arm so
  # that `placed ∧ was_lifted` is measured the same way for all of them. Without a reset
  # here that evaluation would latch True after the first lift and stay there for the rest
  # of the run. The term is a no-op when nothing ever reads the latch, which is the case
  # during training on every arm outside `REQUIRE_LIFTED`.
  cfg.events["reset_was_lifted"] = EventTermCfg(func=mdp.reset_was_lifted, mode="reset")
  # The same for M6's release latch (plan S7), for the same reason: registered on every arm,
  # a no-op where nothing reads the latch, so an eval or a test that reads it on any arm is
  # never graded by the previous episode's release.
  cfg.events["reset_release_latch"] = EventTermCfg(
    func=mdp.reset_release_latch, mode="reset"
  )
  if reward in REQUIRE_LIFTED:
    # Gating the reward without gating the termination would still let a slid cube end the
    # episode early, which is worth remaining-time to the policy even with the bonus
    # withheld. `SUPPORT_SENSOR` is what turns `_placed_bool` into `placed ∧ was_lifted`.
    success_params["support_sensor_name"] = SUPPORT_SENSOR
  # S5.2 module M1: no success termination -- the placed state pays per step instead, so
  # finishing never ends the income (plan S5.2, design rule 2).
  if reward not in NO_SUCCESS_TERMINATION:
    cfg.terminations["success"] = TerminationTermCfg(
      func=mdp.object_placed, params=success_params
    )

  if play:
    apply_play_overrides(cfg)
  return cfg
