"""The robot-agnostic base every tabletop manipulation task starts from.

``make_tabletop_env_cfg(robot)`` returns a complete, runnable environment: a robot at a
table, one or more free objects on it, an action space restricted to the arms and jaws, and
the observation/reward/termination terms any pick-and-place-shaped task needs. Nothing here
names a body part -- everything robot-specific is read off the :class:`ManipulationRobotCfg`.

Which manager owns what:

* **Events** place the objects (``mdp.reset_objects_on_surface``) and randomise friction.
* **Commands** produce only a goal. Nothing else.
* **Sensors** answer "is it held" (pads vs object) and "is it still on the table"
  (object vs table). These are two different questions and two different sensors.
* **Rewards/terminations** are pure functions of the scene plus the goal.

A task builds on top by *adding*, not by rewriting strings into this one::

    cfg = make_tabletop_env_cfg(KANGAROO_GRIPPERS, objects={...}, command_cfg=...)
    cfg.rewards["release"] = RewardTermCfg(func=mdp.released_at_goal, weight=2.0)

If a task needs to reach into this function's internals to work, it is missing an argument.

Deployment note: the exported ONNX carries ``observation_names`` in the order the actor
group declares them, and ``pal_policy_inferer``'s ``getObservationTermDim()`` is a hardcoded
whitelist that today knows only ``joint_pos``, ``joint_vel``, ``actions``, ``command`` and
the IMU terms. The object-state terms below have no entry there yet, so a policy trained
with them cannot be driven on hardware until the C++ node is extended. Keep these term
*names and dimensions* stable so that extension stays mechanical.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Mapping

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.manipulation.mdp import LiftingCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import NoiseModelWithAdditiveBiasCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from pal_mjlab.tasks.manipulation import mdp, scene
from pal_mjlab.tasks.manipulation.robot_cfg import GripperCfg, ManipulationRobotCfg

#: Scene entity names. Terms refer to these, never to a task-specific name.
ROBOT = "robot"
TABLE = "table"
OBJECT = "object"

#: The command term name.
COMMAND = "goal_pose"

#: Sensor name for object-on-table support.
SUPPORT_SENSOR = "object_table_contact"


def make_grasp_sensor(gripper: GripperCfg, object_name: str) -> ContactSensorCfg:
  """Pads-vs-object contact: the ground-truth grasp signal."""
  return ContactSensorCfg(
    name=gripper.contact_sensor_name,
    primary=ContactMatch(mode="geom", pattern=gripper.grasp_geom_regex, entity=ROBOT),
    secondary=ContactMatch(mode="body", pattern=object_name, entity=object_name),
    fields=("found", "force"),
    reduce="none",
    num_slots=2,
    history_length=1,
  )


def make_support_sensor(
  object_name: str, surface_name: str = TABLE
) -> ContactSensorCfg:
  """Object-vs-table contact: whether the object is still supported.

  Distinct from the grasp sensor. Conflating the two is what previously made the ``lift``
  and ``goal`` rewards evaluate to identically zero, silently.
  """
  return ContactSensorCfg(
    name=SUPPORT_SENSOR,
    primary=ContactMatch(mode="body", pattern=object_name, entity=object_name),
    secondary=ContactMatch(mode="body", pattern=surface_name, entity=surface_name),
    fields=("found",),
    reduce="none",
    num_slots=1,
    history_length=1,
  )


def make_hand_ground_sensor(gripper: GripperCfg) -> ContactSensorCfg:
  """Hand-vs-floor contact, so a task can penalise scraping the ground."""
  return ContactSensorCfg(
    name=gripper.ground_sensor_name,
    primary=ContactMatch(mode="subtree", pattern=gripper.hand_body, entity=ROBOT),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )


def make_hand_surface_sensor(
  gripper: GripperCfg, surface_name: str = TABLE
) -> ContactSensorCfg:
  """Hand-vs-table contact, so a task can penalise leaning on the work surface.

  Distinct from the ground sensor above, which watches the floor. Scraping or resting the
  jaw on the table is a cheap way for a simulated policy to stabilise a grasp, and it is the
  first thing that stops working on a real table.
  """
  return ContactSensorCfg(
    name=gripper.surface_sensor_name,
    primary=ContactMatch(mode="subtree", pattern=gripper.hand_body, entity=ROBOT),
    secondary=ContactMatch(mode="body", pattern=surface_name, entity=surface_name),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )


#: Physics step and control decimation: 2 ms x 10 = 50 Hz control. Named because the arm slew
#: limit is a speed times the control period; see the sim cfg below for why 2 ms.
SIM_TIMESTEP = 0.002
DECIMATION = 10
CONTROL_DT = SIM_TIMESTEP * DECIMATION

#: What the policy observes, by name (plan S7-2). ``S71`` is every run up to S7-1: 36-D, with
#: `grasped` in the actor. ``B`` is baseline B's actor (Claudius's `origin/faster-moves`
#: dc498d9): 38-D, only what the robot can measure -- see `_deployable_observations`.
OBSERVATION_SETS = ("S71", "B")

#: Baseline B's joint-encoder model: a per-episode calibration offset on the joint positions,
#: rad, and 0-1 control steps of joint-state latency. Claudius's `JOINT_POS_BIAS` and
#: `JOINT_DELAY_*_LAG`.
_B_JOINT_POS_BIAS = 0.01
_B_JOINT_LAG = (0, 1)


#: Plan S7-6: the real robot's two latencies, ported from Claudius's `origin/faster-moves`
#: (1d484c4 tracker, 71e1147 jaw servo; measured on the robot 2026-09-29).
#:
#: The cube is tracked by FoundationPose, 100-150 ms end to end (capture, inference, network,
#: ROS) = 5-8 control steps at `CONTROL_DT`, against the 0-2 steps (`_DR_OBS_OBJECT_LAG`, on
#: the actor's cube terms under DR-obs and baseline B) every run up to S7-5 trained with. Drawn per env per episode, on
#: the cube's pose (`mdp._ObjectTracker`), replacing the term-level 0-2 on the actor's cube
#: terms rather than adding to it.
VISION_TRACKER_LAG = (5, 8)  # control steps
#: The jaw servo: a first-order lag with a saturating speed between the jaw command and the
#: joint (`mdp.RateLimitedJointPositionAction._resolve_servo`). Recorded strokes: 0.13-0.17 s
#: steady lag, 1.3-1.7 rad/s peak. The two are not separable from one logged stroke (a pure
#: 0.14 s lag fits the lag samples too), so both are modelled and both ranges are wide --
#: Claudius's reading, kept as is. Per env, per episode.
JAW_SERVO_TAU = (0.11, 0.19)  # s
JAW_SERVO_MAX_VEL = (1.25, 1.75)  # rad/s


@dataclasses.dataclass(frozen=True)
class LatencyModelCfg:
  """Plan S7-6's latency model: which of the robot's measured latencies training sees."""

  tracker_lag: tuple[int, int] | None
  """The cube tracker's lag, control steps, on the cube pose; None keeps the observation
  set's own term-level lag (0-2 under DR-obs, every run up to S7-5)."""

  jaw_servo: bool
  """The jaw servo model (`JAW_SERVO_TAU`, `JAW_SERVO_MAX_VEL`) between command and joint."""


#: Latency models by name (`LAG`). ``none`` is every run up to S7-5; ``FP`` the FoundationPose
#: tracker alone, ``JS`` the jaw servo alone, ``FP+JS`` both (plan S7-6).
LATENCY_MODELS = {
  "none": LatencyModelCfg(tracker_lag=None, jaw_servo=False),
  "FP": LatencyModelCfg(tracker_lag=VISION_TRACKER_LAG, jaw_servo=False),
  "JS": LatencyModelCfg(tracker_lag=None, jaw_servo=True),
  "FP+JS": LatencyModelCfg(tracker_lag=VISION_TRACKER_LAG, jaw_servo=True),
}

#: The actor's terms that read the tracked object, and so move onto the tracker's lag.
_TRACKED_OBJECT_TERMS = ("object_pos", "object_yaw")


def _apply_tracker_lag(cfg: ManagerBasedRlEnvCfg, lag: tuple[int, int]) -> None:
  """Put the actor's cube terms on one shared tracker ``lag`` steps late, and the critic's on
  the live pose.

  After `apply_dr_level`, which may already have replaced the actor's cube terms (DR-obs). The
  terms are *replaced*, not edited: in the ``S71`` set the actor and critic share term objects.
  The term-level delay goes to zero on both, so the lag is the tracker's alone; the noise model
  (per-episode bias, per-step jitter) is kept, and is applied to the lagged pose, as on the
  robot.
  """
  for group_name, group in cfg.observations.items():
    for name in _TRACKED_OBJECT_TERMS:
      if name not in group.terms:
        raise ValueError(
          f"tracker lag asked for, but the {group_name!r} group has no {name!r} term"
        )
      term = group.terms[name]
      group.terms[name] = dataclasses.replace(
        term,
        params={**term.params, "lag": lag if group_name == "actor" else None},
        delay_min_lag=0,
        delay_max_lag=0,
      )


@dataclasses.dataclass(frozen=True)
class ToolSpeedCapCfg:
  """Plan S7-2's module V: a bound on the tool's speed in the action term. See
  `mdp.ToolSpeedCappedJointPositionAction`."""

  speed: float
  """Tool speed bound, m/s."""

  slow_speed: float
  """Tool speed bound in the slow zone, m/s."""

  slow_zone: float = 0.05
  """The slow zone: a held object within this distance of its resting height on the table,
  or horizontally of the goal, m."""


def _deployable_observations(
  action_joints: SceneEntityCfg,
  arm_joints: SceneEntityCfg,
  tcp: SceneEntityCfg,
  grasp_sensor: str,
) -> dict[str, ObservationGroupCfg]:
  """Baseline B's observation groups (plan S7-2): an actor fed only what the robot measures.

  Against ``S71``: the jaw's velocity is out (noise-dominated, and the jaw's position and the
  last action already say what it is doing); the TCP's orientation is in; `grasped` -- a
  MuJoCo contact sensor with no counterpart on the robot -- moves to the critic. The joint
  readings get a per-episode encoder offset and 0-1 steps of latency, and the object-tracker
  error model of `DR-obs` is on at every rung. The bias noise uses ``operation="abs"`` (see
  `apply_dr_level`, DR-obs): with mjlab's default "add" it random-walks across episodes.

  The critic is the actor's terms without noise *or* latency -- `enable_corruption=False`
  switches noise off but not delay -- plus `grasped` as privileged information. It runs only in
  simulation and is not exported.
  """

  def tracker(bias: float, noise: float) -> NoiseModelWithAdditiveBiasCfg:
    return NoiseModelWithAdditiveBiasCfg(
      noise_cfg=Unoise(n_min=-noise, n_max=noise),
      bias_noise_cfg=Unoise(n_min=-bias, n_max=bias, operation="abs"),
    )

  joint_lag = {"delay_min_lag": _B_JOINT_LAG[0], "delay_max_lag": _B_JOINT_LAG[1]}
  object_lag = {
    "delay_min_lag": _DR_OBS_OBJECT_LAG[0],
    "delay_max_lag": _DR_OBS_OBJECT_LAG[1],
  }
  actor_terms = {
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"asset_cfg": action_joints},
      noise=tracker(_B_JOINT_POS_BIAS, 0.002),
      **joint_lag,
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      params={"asset_cfg": arm_joints},
      noise=Unoise(n_min=-0.2, n_max=0.2),
      **joint_lag,
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "tcp_pos": ObservationTermCfg(func=mdp.ee_position_b, params={"asset_cfg": tcp}),
    "tcp_quat": ObservationTermCfg(
      func=mdp.ee_orientation_b, params={"asset_cfg": tcp}
    ),
    "object_pos": ObservationTermCfg(
      func=mdp.object_position_b,
      params={"object_name": OBJECT},
      noise=tracker(_DR_OBS_OBJECT_POS_BIAS, _DR_OBS_OBJECT_POS_NOISE),
      **object_lag,
    ),
    "object_yaw": ObservationTermCfg(
      func=mdp.object_yaw_b,
      params={"object_name": OBJECT},
      noise=tracker(_DR_OBS_OBJECT_YAW_BIAS, _DR_OBS_OBJECT_YAW_NOISE),
      **object_lag,
    ),
    "target_pos": ObservationTermCfg(
      func=mdp.goal_position_b, params={"command_name": COMMAND}
    ),
  }
  critic_terms = {
    name: dataclasses.replace(term, delay_min_lag=0, delay_max_lag=0)
    for name, term in actor_terms.items()
  }
  critic_terms["grasped"] = ObservationTermCfg(
    func=mdp.is_grasped, params={"sensor_name": grasp_sensor}
  )
  return {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
    "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
  }


def make_tabletop_env_cfg(
  robot: ManipulationRobotCfg,
  gripper_name: str | None = None,
  objects: Mapping[str, EntityCfg] | None = None,
  command_cfg: CommandTermCfg | None = None,
  object_pose_range: Mapping[str, tuple[float, float]] | None = None,
  min_separation: float = 0.0,
  episode_length_s: float = 20.0,
  hold_nominal: bool = True,
  dr_level: str | None = None,
  guard_divergence: bool = True,
  trunk_clearance_penalty: float = 1.0,
  action_slew_rate_scale: float = 1.0,
  table_top_z: float = scene.TABLE_TOP_Z,
  observation_set: str = "S71",
  jaw_slew_rate: float | None = None,
  tool_speed_cap: ToolSpeedCapCfg | None = None,
  dr_spawn_widening: bool = True,
  arm_range_clip: bool = False,
  arm_effort_band: bool = False,
  latency: LatencyModelCfg = LATENCY_MODELS["none"],
) -> ManagerBasedRlEnvCfg:
  """A complete tabletop manipulation environment for ``robot``.

  Args:
    robot: the robot description; the only robot-specific input.
    gripper_name: which gripper the task uses. Defaults to the robot's first.
    objects: entity key -> cfg, in placement order. The first is the manipulated object and
      must be keyed ``OBJECT``. Defaults to a single graspable cube.
    command_cfg: the goal. Defaults to a uniformly sampled point above the table.
    object_pose_range: XY/yaw spawn range on the table, relative to the table centre.
    min_separation: minimum XY gap enforced between spawned objects.
    episode_length_s: episode timeout.
    hold_nominal: command the joints the policy does not drive to their defaults instead of
      to zero. Off is the pre-S4 behaviour and is kept only so the two can be compared; it
      collapses the legs within 0.2 s.
    dr_level: domain-randomisation rung by name, or None to leave the base cfg's own
      randomisation alone. See `DR_LEVELS`. Spelled `dr_level` rather than `dr` only because
      this module imports mjlab's `dr` term namespace; the task factories expose it as `dr`.
    guard_divergence: terminate any world whose physics goes non-finite, and sanitise
      observations as a backstop.
      On by default because contact-rich GPU sim diverges at a measured ~1 per 0.5M
      world-steps here and `mujoco_warp` does not reset a diverged world by itself; off
      reproduces the pre-guard behaviour, in which the first divergence kills the run.
    trunk_clearance_penalty: weight of ``trunk_clearance`` (see `trunk_clearance_rewards`),
      for a robot with a ``trunk_keepout``; 0 turns it off.
    action_slew_rate_scale: a factor on the robot's `joint_velocity_limits` slew: 1.0 is the
      URDF rate, 0.5 halves every limited joint's top speed (plan S7, speed bound A). Must be
      in (0, 1] -- the limits are hardware ceilings, so a factor above one would train
      against an actuator that does not exist -- and 1.0 on a robot without limits.
    table_top_z: height of the table's top face, m. `scene.TABLE_TOP_Z` (0.76) for every run
      up to S7-1; `scene.ROBOT_CELL_TABLE_TOP_Z` for plan S7-2's baseline B.
    observation_set: what the actor and critic observe, by name; see `OBSERVATION_SETS`.
    jaw_slew_rate: largest change of the commanded jaw target per control step, rad, or None
      for none. Plan S7-2's baseline B: 0.04 = PAL's 2 rad/s. Not scaled by
      ``action_slew_rate_scale``, which is the arm's.
    tool_speed_cap: plan S7-2's module V, or None. See `ToolSpeedCapCfg`.
    dr_spawn_widening: whether `DR-scene` widens ``object_pose_range`` (it does for the
      +/-2 cm box every run up to S7-1 used). Off for a spawn box that is already the whole
      region to train on (plan S7-2's W boxes).
    arm_range_clip: plan S7-4's C -- the arm's (not the jaw's) targets clipped to the joint
      range in the action term, after the slew and any tool cap.
    arm_effort_band: plan S7-4's T in the action term -- the arm's targets kept within
      q +/- tau_cmd / K, tau_cmd the effort limit the robot's model carries (lowered by the
      robot cfg, `robots.kangaroo_grippers_left_arm(arm_effort_scale=...)`).
    latency: plan S7-6's latency model -- the cube tracker's lag and the jaw servo. See
      `LATENCY_MODELS`; ``none`` is every run up to S7-5.
  """
  if observation_set not in OBSERVATION_SETS:
    raise ValueError(
      f"unknown observation set {observation_set!r}; have {OBSERVATION_SETS}"
    )
  gripper = robot.gripper(gripper_name)
  if objects is None:
    objects = {OBJECT: scene.get_cube_cfg(name=OBJECT)}
  if OBJECT not in objects:
    raise ValueError(
      f"objects must contain the manipulated object under key {OBJECT!r}"
    )
  if object_pose_range is None:
    object_pose_range = {"x": (-0.15, 0.15), "y": (-0.25, 0.25), "yaw": (-3.14, 3.14)}

  tcp = SceneEntityCfg(ROBOT, site_names=(gripper.tcp_site,))
  action_joints = SceneEntityCfg(ROBOT, joint_names=robot.action_joint_regex)
  pad_geoms = SceneEntityCfg(ROBOT, geom_names=(gripper.pad_geom_regex,))
  grasp_sensor = gripper.contact_sensor_name

  # -- Observations ---------------------------------------------------------------
  # Proprioception first, then object state. Names are part of the deployment contract
  # (see the module docstring); do not rename without updating the C++ node.
  actor_terms = {
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"asset_cfg": action_joints},
      noise=Unoise(n_min=-0.002, n_max=0.002),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      params={"asset_cfg": action_joints},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "tcp_pos": ObservationTermCfg(func=mdp.ee_position_b, params={"asset_cfg": tcp}),
    "object_pos": ObservationTermCfg(
      func=mdp.object_position_b,
      params={"object_name": OBJECT},
      noise=Unoise(n_min=-0.005, n_max=0.005),
    ),
    "object_yaw": ObservationTermCfg(
      func=mdp.object_yaw_b,
      params={"object_name": OBJECT},
      noise=Unoise(n_min=-0.02, n_max=0.02),
    ),
    "target_pos": ObservationTermCfg(
      func=mdp.goal_position_b, params={"command_name": COMMAND}
    ),
    "grasped": ObservationTermCfg(
      func=mdp.is_grasped, params={"sensor_name": grasp_sensor}
    ),
  }
  observations = {
    "actor": ObservationGroupCfg(dict(actor_terms), enable_corruption=True),
    "critic": ObservationGroupCfg(dict(actor_terms), enable_corruption=False),
  }
  # The arm's joints: the action joints but the jaw.
  arm_regex = tuple(
    p for p in robot.action_joint_regex if not re.fullmatch(p, gripper.finger_joint)
  )
  if observation_set == "B":
    observations = _deployable_observations(
      action_joints,
      SceneEntityCfg(ROBOT, joint_names=arm_regex),
      tcp,
      grasp_sensor,
    )

  # -- Actions --------------------------------------------------------------------
  # With `joint_velocity_limits` the commanded target may move at most limit * CONTROL_DT
  # per control step (S5.1 item 4): still an absolute target, `default + scale * action`, so
  # the deployment contract is unchanged, but a saturated action ramps instead of jumping.
  # Joints without a limit -- the jaw, unless ``jaw_slew_rate`` gives it one -- behave exactly
  # as `JointPositionActionCfg`.
  joint_pos_kwargs = dict(
    entity_name=ROBOT,
    actuator_names=robot.action_joint_regex,
    scale=robot.action_scale,
    clip=robot.action_clip,
    use_default_offset=True,
  )
  # `action_slew_rate_scale` scales that limit (plan S7, speed bound A). It is a property of
  # the action term, so a deployment must apply the same per-step clamp to reproduce it.
  if not 0.0 < action_slew_rate_scale <= 1.0:
    raise ValueError(
      f"action_slew_rate_scale={action_slew_rate_scale} is outside (0, 1]: the velocity "
      "limits are hardware ceilings and can only be tightened"
    )
  if robot.joint_velocity_limits is None and action_slew_rate_scale != 1.0:
    raise ValueError(
      f"action_slew_rate_scale={action_slew_rate_scale}, but the robot has no "
      "joint_velocity_limits for it to scale"
    )
  rate_limit = {
    pattern: limit * CONTROL_DT * action_slew_rate_scale
    for pattern, limit in (robot.joint_velocity_limits or {}).items()
  }
  if jaw_slew_rate is not None:
    rate_limit[re.escape(gripper.finger_joint)] = jaw_slew_rate
  # Plan S7-4's bounds on the arm's final target (C, T); `mdp.RateLimitedJointPositionAction`.
  bounds = {}
  if arm_range_clip:
    bounds["range_clip_joints"] = arm_regex
  if arm_effort_band:
    bounds["effort_band_joints"] = arm_regex
  if latency.jaw_servo:
    bounds["servo_joints"] = (re.escape(gripper.finger_joint),)
    bounds["servo_tau"] = JAW_SERVO_TAU
    bounds["servo_max_vel"] = JAW_SERVO_MAX_VEL
  if bounds and not rate_limit:
    raise ValueError(
      "the arm bounds and the jaw servo live in the rate-limited action term; no slew limit"
    )
  joint_pos_action: ActionTermCfg = (
    mdp.RateLimitedJointPositionActionCfg(
      **joint_pos_kwargs, rate_limit=rate_limit, **bounds
    )
    if rate_limit
    else JointPositionActionCfg(**joint_pos_kwargs)
  )
  if tool_speed_cap is not None:
    joint_pos_action = mdp.ToolSpeedCappedJointPositionActionCfg(
      **joint_pos_kwargs,
      **bounds,
      rate_limit=rate_limit,
      capped_joints=arm_regex,
      tool_site=gripper.tcp_site,
      speed_limit=tool_speed_cap.speed,
      slow_speed_limit=tool_speed_cap.slow_speed,
      slow_zone=tool_speed_cap.slow_zone,
      control_dt=CONTROL_DT,
      object_name=OBJECT,
      grasp_sensor_name=grasp_sensor,
      surface_name=TABLE,
      command_name=COMMAND,
    )
  actions: dict[str, ActionTermCfg] = {"joint_pos": joint_pos_action}

  # -- Commands -------------------------------------------------------------------
  if command_cfg is None:
    command_cfg = LiftingCommandCfg(
      entity_name=OBJECT,
      difficulty="dynamic",
      object_pose_range=None,  # Placement is the event's job.
      resampling_time_range=(1e9, 1e9),  # One goal per episode.
      debug_vis=True,
    )
  commands: dict[str, CommandTermCfg] = {COMMAND: command_cfg}

  # -- Events ---------------------------------------------------------------------
  events = {
    # Every joint back to its configured default first. This must cover the whole robot,
    # not just the commanded joints: the legs are a closed four-bar held by equality
    # constraints, and any joint the reset does not write keeps the model's qpos0 of zero.
    # For Kangaroo that put the telescopic leg at length 0 -- below its 0.132 m mechanical
    # minimum -- leaving the robot standing on collapsed legs.
    "reset_to_default_pose": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg(ROBOT, joint_names=(".*",)),
      },
    ),
    # Then tell the actuators to hold that pose. Without this the reset zeroes every joint
    # target and only the action joints get one written back, so the legs, the telescopic
    # drive and the pelvis are commanded to 0 for the whole episode -- see
    # mdp.hold_nominal_targets.
    "hold_nominal_targets": EventTermCfg(
      func=mdp.hold_nominal_targets, mode="reset", params={}
    ),
    # Only then jitter the commanded joints. The legs are deliberately excluded: perturbing
    # them independently would violate the four-bar mimic.
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.02, 0.02),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": action_joints,
      },
    ),
    "reset_objects": EventTermCfg(
      func=mdp.reset_objects_on_surface,
      mode="reset",
      params={
        "asset_cfgs": [SceneEntityCfg(name) for name in objects],
        "surface_cfg": SceneEntityCfg(TABLE),
        "pose_range": dict(object_pose_range),
        "min_separation": min_separation,
      },
    ),
    "pad_friction_slide": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": pad_geoms,
        "operation": "abs",
        "distribution": "uniform",
        "axes": [0],
        "ranges": (0.6, 1.2),
      },
    ),
    "pad_friction_spin": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": pad_geoms,
        "operation": "abs",
        "distribution": "log_uniform",
        "axes": [1],
        "ranges": (5e-3, 5e-2),
      },
    ),
  }
  if not hold_nominal:
    del events["hold_nominal_targets"]
  if not robot.fixed_base:
    events["reset_base"] = EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={"pose_range": {}, "velocity_range": {}},
    )

  # -- Rewards --------------------------------------------------------------------
  # staged_position_reward is mjlab's workhorse: reaching * (1 + bringing), so approach is
  # rewarded before the object moves and the bring bonus cannot be collected without it.
  rewards = {
    "reach": RewardTermCfg(
      func=mdp.staged_position_reward,
      weight=1.0,
      params={
        "command_name": COMMAND,
        "object_name": OBJECT,
        "reaching_std": 0.2,
        "bringing_std": 0.2,
        "asset_cfg": tcp,
      },
    ),
    "grasp": RewardTermCfg(
      func=mdp.grasped, weight=1.0, params={"sensor_name": grasp_sensor}
    ),
    "lift": RewardTermCfg(
      func=mdp.object_lifted,
      weight=2.0,
      params={
        "grasp_sensor_name": grasp_sensor,
        "support_sensor_name": SUPPORT_SENSOR,
      },
    ),
    "goal": RewardTermCfg(
      func=mdp.bring_object_reward,
      weight=4.0,
      params={"command_name": COMMAND, "object_name": OBJECT, "std": 0.15},
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    "joint_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits, weight=-10.0, params={"asset_cfg": action_joints}
    ),
    **trunk_clearance_rewards(robot, trunk_clearance_penalty),
  }

  # -- Terminations ---------------------------------------------------------------
  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "object_dropped": TerminationTermCfg(
      func=mdp.entity_below_height,
      params={"entity_name": OBJECT, "min_height": table_top_z - 0.2},
    ),
  }
  if robot.reach is not None:
    terminations["out_of_reach"] = TerminationTermCfg(
      func=mdp.entity_out_of_reach,
      params={
        "entity_names": tuple(objects),
        "max_distance": robot.reach.max_distance,
        "reference_body": robot.reach.reference_body,
        "robot_name": ROBOT,
      },
    )
  if guard_divergence:
    # A failure, not a truncation -- the plan's D9. Both are defensible: a failure pushes
    # the policy away from the configurations that blow up, at the cost of a shaping term
    # whose size is whatever V(s') happened to be; a truncation injects no value signal but
    # bootstraps from a state that is either corrupted or, here, freshly reset -- the value
    # of a new episode, not of this one's continuation. D9 chose failure because V(s')=0 is
    # at least a consistent defined choice, and nothing measured since overturns it: the
    # rate is 0.056-0.074% of episodes (jobs 157183/157184), so the choice moves ~0.06% of
    # transitions either way.
    terminations["non_finite_state"] = TerminationTermCfg(
      func=mdp.non_finite_state, time_out=False
    )
    # The backstop, for a state that is finite in qpos/qvel but produces a non-finite
    # observation anyway -- a velocity or sensor reading that overflows on the way out.
    # Rewards need no equivalent: mjlab's RewardManager.compute already sanitises every
    # term unconditionally (reward_manager.py:129).
    for group in observations.values():
      group.nan_policy = "sanitize"
      # One check on the concatenated group rather than one per term. Per-term checking
      # forces a host sync per term per step, and the term is not the interesting answer
      # anyway -- `non_finite_state` already says the physics diverged.
      group.nan_check_per_term = False

  sensors = [
    make_grasp_sensor(gripper, OBJECT),
    make_support_sensor(OBJECT),
    make_hand_ground_sensor(gripper),
    make_hand_surface_sensor(gripper),
  ]

  entities: dict[str, EntityCfg] = {
    ROBOT: robot.build_entity_cfg(),
    TABLE: scene.get_table_cfg(table_top_z),
  }
  entities.update(objects)

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities=entities,
      sensors=tuple(sensors),
      num_envs=1,
      # Zero, not the locomotion default. mjlab simulates the envs as batched worlds with
      # identical geometry -- every world's table is at the same place -- but a scene with a
      # terrain still lays `env_origins` out on a grid of `env_spacing`, which is what a
      # locomotion task needs to spread its robots over the terrain. Nothing here honours
      # that grid: `reset_objects_on_surface` reads the table out of physics, so the cube
      # lands on the table, while `LiftingCommand` adds `env_origins` to its target, so the
      # goal walks off into space. At the default 2.5 the goal sits 1.6-1.9 m from the cube
      # in every env but the first, and no policy can reach it. Measured; see
      # `test_the_goal_is_reachable_from_the_object`.
      env_spacing=0.0,
      extent=2.5,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum={},
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name=ROBOT,
      body_name=robot.viewer_body,
      distance=1.8,
      elevation=-15.0,
      azimuth=140.0,
    ),
    sim=SimulationCfg(
      # Contact-rich profile from mjlab's lift_cube, with headroom raised past its 55/600:
      # Kangaroo adds two leg closed chains and the gripper four-bar.
      nconmax=200,
      njmax=2000,
      mujoco=MujocoCfg(
        # 2 ms, matching the locomotion tasks. The gripper four-bar is a stiff closed
        # linkage whose passive joints carry only 1e-4 armature; at 5 ms they diverge
        # (measured: |qvel| past 900 rad/s and NaN within ~40 control steps under random
        # actions). Decimation is raised to keep the control rate at 50 Hz. That was measured
        # before decision 0002 raised the tabletop overlay's passive armature to 5e-3; whether
        # a longer step is now viable is untested.
        timestep=SIM_TIMESTEP,
        iterations=10,
        ls_iterations=20,
        impratio=10,
        cone="elliptic",
      ),
    ),
    decimation=DECIMATION,
    episode_length_s=episode_length_s,
  )
  if dr_level is not None:
    apply_dr_level(cfg, dr_level, action_joints, spawn_widening=dr_spawn_widening)
  if latency.tracker_lag is not None:
    _apply_tracker_lag(cfg, latency.tracker_lag)
  return cfg


def trunk_clearance_rewards(
  robot: ManipulationRobotCfg, weight: float = 1.0
) -> dict[str, RewardTermCfg]:
  """``trunk_clearance``: a penalty for arm links approaching the trunk's keep-out box.

  S5.1 item 8 (Claudius's 82c4450). The trunk colliders stop an arm at the chest; this
  prices the approach, so the policy does not learn to lean an elbow on it. Weight 1.0 is
  his: ~10 % of what the top of his stack reward pays per step. A separate function because
  a task that replaces the reward dict wholesale (pick-and-place's variants) has to add it
  back, and every variant must carry it the same way.
  """
  keepout = robot.trunk_keepout
  if keepout is None or not weight:
    return {}
  return {
    "trunk_clearance": RewardTermCfg(
      func=mdp.keepout_box_penalty,
      weight=-abs(weight),
      params={
        "center": keepout.center,
        "half_extents": keepout.half_extents,
        "margin": keepout.margin,
        "reference_body": keepout.reference_body,
        "asset_cfg": SceneEntityCfg(ROBOT, body_names=keepout.body_name_regex),
      },
    )
  }


#: The domain-randomisation ladder (S2 of the pick-and-place plan), lowest rung first. Each
#: rung *adds* to the one below, so a task states its level and the config records it instead
#: of a report claiming it. `None` -- the default -- leaves the base cfg exactly as it is,
#: which is what lift and stack have always run; note that the base cfg is **not** DR0, since
#: it randomises pad friction at startup and jitters the commanded joints at reset.
#:
#: What each rung adds:
#:
#: * ``DR0`` -- nothing in the model varies: removes the two things the base cfg randomises
#:   (pad friction, reset jitter). The actor's base observation noise -- joint pos +/-0.002
#:   rad, vel +/-0.2 rad/s, object pos +/-5 mm, yaw +/-0.02 rad -- stays on at every rung.
#: * ``DR-object`` -- the object's physics: sliding friction and mass.
#: * ``DR-contact`` -- the gripper side: the base cfg's pad friction, restored.
#: * ``DR-robot`` -- actuation: PD gains, joint friction and joint damping on the driven
#:   joints, and a +/-0.02 rad jitter on their reset pose.
#: * ``DR-obs`` -- perception: an object-tracker error model (per-episode bias, per-step
#:   jitter, 0-2 steps of lag) and an encoder bias.
#: * ``DR-scene`` -- layout: the table's pose (x, y +/-2 cm, height +/-1 cm) and a wider
#:   spawn box.
#:
#: v0 trains at DR0 and DR-object only; the rest are defined so that climbing is a change of
#: one string. Two caveats are recorded rather than hidden:
#:
#: * **DR-scene moves the table, so a goal with a hardcoded position stops being right.** A
#:   2 cm table offset eats 2 cm of a 2.5 cm success threshold. A task that uses this rung
#:   needs a goal that reads the surface out of physics -- `mdp.SurfaceOffsetCommand` is that
#:   goal, and `pick_place_env_cfg._command_cfg` is one way to switch to it. Both it and the
#:   spawn (`reset_objects_on_surface`) read the table's world XY as well as its height, so
#:   the target stays fixed *relative to the table* and moves +/-2 cm relative to the robot.
#: * **Randomising the cube's size is deliberately not here**, even though `dr.geom_size`
#:   works: the cube's resting height is a function of its size, so it breaks a hardcoded
#:   goal z the same way the table does. It would be safe to add for a task already using
#:   `mdp.SurfaceOffsetCommand`, which reads the object's half-height, and unsafe for one
#:   that is not -- and a rung that is conditionally correct is worse than a missing one.
DR_LEVELS = ("DR0", "DR-object", "DR-contact", "DR-robot", "DR-obs", "DR-scene")

#: Rungs at and above this one move the table; see the caveat on `DR_LEVELS`.
DR_LEVELS_THAT_MOVE_THE_TABLE = ("DR-scene",)

#: DR-object: the cube's sliding friction, and its mass in kg -- 30 g to 100 g brackets a
#: light plastic cube and a heavy wooden one, and the point of the rung is that the policy
#: cannot assume the 50 g it was built with.
#:
#: The mass goes through `dr.pseudo_inertia`, not `dr.body_mass`, on mjlab's own instruction:
#: `body_mass` changes the mass and leaves `body_inertia` alone, which models a point mass
#: bolted to the COM rather than a denser cube, and it warns as much at runtime. A 100 g cube
#: carrying the rotational inertia of a 50 g one is exactly the quantity a grasp has to
#: resist, so the approximation is not one to make here. `alpha` is a log density scale --
#: mass and inertia both scale by exp(2*alpha) -- so the kg range is converted rather than
#: written out, and the conversion is what `test_ranges_land_where_the_config_says` checks.
_DR_OBJECT_FRICTION = (0.5, 1.2)
_DR_OBJECT_MASS_KG = (0.03, 0.10)
_CUBE_NOMINAL_MASS_KG = 0.05  # scene.get_cube_spec's default
_DR_OBJECT_ALPHA = tuple(
  math.log(mass / _CUBE_NOMINAL_MASS_KG) / 2 for mass in _DR_OBJECT_MASS_KG
)

#: DR-robot: +/-20% on the PD gains of the driven joints, and absolute joint-level friction
#: and viscous damping per joint group, keyed by joint regex.
#:
#: The friction and damping ranges are Claudius's (`origin/fix-stack-exploration-collapse`,
#: 860554a, `joint_friction_range` / `joint_damping_range` in his `robots/kangaroo.py`),
#: folded in by S5.1 of the pick-and-place plan. The arms carry no `frictionloss` and no
#: joint `damping` in the model at all -- their damping lives in the actuator kd -- so both
#: are absolute and are new behaviour rather than a perturbation: 1.0 N.m of stiction is 2 %
#: of the arm's 50 N.m effort limit, 0.05 N.m.s/rad is ~5 % of an arm kd of 1.0. The jaw
#: ships with `damping="0.03"` (`kangaroo_grippers.xml`), which its range brackets, and gets
#: three orders of magnitude less stiction than an arm, since anything larger stalls it.
_DR_ROBOT_GAIN_SCALE = (0.8, 1.2)
#: DR-robot's reset jitter on the driven joints, rad: Claudius's value, aligned by plan S7
#: (decided 2026-09-24). The base cfg has the same +/-0.02; DR0 switches it off and this rung
#: puts it back.
_DR_ROBOT_RESET_JITTER = 0.02
_DR_ROBOT_JOINT_FRICTION = {
  r"arm_.*_[1-7]_joint": (0.0, 1.0),
  r"gripper_.*_finger_joint": (0.0, 0.02),
}
_DR_ROBOT_JOINT_DAMPING = {
  r"arm_.*_[1-7]_joint": (0.0, 0.05),
  r"gripper_.*_finger_joint": (0.02, 0.04),
}

#: DR-obs: what the policy is allowed to be wrong about. The object pose gets a tracker
#: error model rather than wider zero-mean noise: a per-episode bias, a small per-step
#: jitter, and 0-2 control steps (0-40 ms) of lag sampled per env. The numbers are
#: Claudius's (860554a, `OBJECT_*` in his `tabletop_env_cfg.py`), sized for a marker-based
#: tracker with headroom, and folded in by S5.1. The point of the bias is that it cannot be
#: averaged away over a few frames, which zero-mean noise resampled every step can -- the
#: base cfg's +/-5 mm, and the +/-15 mm this rung used before, both left the policy
#: effectively reading ground truth. Plus an encoder bias, also a per-episode constant.
_DR_OBS_OBJECT_POS_BIAS = 0.008  # m
_DR_OBS_OBJECT_POS_NOISE = 0.002  # m
_DR_OBS_OBJECT_YAW_BIAS = 0.03  # rad
_DR_OBS_OBJECT_YAW_NOISE = 0.01  # rad
_DR_OBS_OBJECT_LAG = (0, 2)  # control steps
_DR_OBS_ENCODER_BIAS = 0.005

#: DR-scene: the table's pose, and how much wider the spawn box gets. The pose is Claudius's
#: (x, y +/-2 cm, z +/-1 cm), aligned by plan S7 (decided 2026-09-24); the height was +/-2 cm
#: before. The spawn widening is additive rather than a factor, because the tasks' nominal
#: spawn ranges differ by an order of magnitude (2 cm for pick-and-place, 15 cm for lift) and
#: a factor would push the larger ones off the table.
_DR_SCENE_TABLE_XY = 0.02
_DR_SCENE_TABLE_HEIGHT = 0.01
_DR_SCENE_EXTRA_SPAWN = 0.03


def _on_joints(
  ranges: Mapping[str, tuple[float, float]], joint_regex: tuple[str, ...]
) -> dict[str, tuple[float, float]]:
  """Restrict joint-regex keyed DR ranges to the joints matched by ``joint_regex``.

  mjlab matches a string-keyed ``ranges`` against *every* joint of the entity and ignores
  the event's ``asset_cfg.joint_names`` (`_randomize_with_string_ranges` in
  `mjlab/envs/mdp/dr/_core.py`), so ``arm_.*`` would also randomise an arm the task has
  parked outside its action space. Each key becomes "matches the group *and* one of the
  action regexes": the lookahead asserts the group pattern over the whole name, then the
  alternation must match it too. A group with no action joint still raises in mjlab, which
  is the loud failure wanted.
  """
  either = "|".join(f"(?:{r})" for r in joint_regex)
  return {rf"(?=(?:{group})$)(?:{either})": rng for group, rng in ranges.items()}


def _dr_index(level: str) -> int:
  if level not in DR_LEVELS:
    raise ValueError(f"unknown DR level {level!r}; have {DR_LEVELS}")
  return DR_LEVELS.index(level)


def apply_dr_level(
  cfg: ManagerBasedRlEnvCfg,
  level: str,
  action_joints: SceneEntityCfg,
  object_name: str = OBJECT,
  spawn_widening: bool = True,
) -> ManagerBasedRlEnvCfg:
  """Set the environment's domain randomisation to a named rung, in place.

  Cumulative: every rung includes the ones below it. See `DR_LEVELS`. ``spawn_widening=False``
  keeps `DR-scene`'s table pose but not its wider spawn box (plan S7-2's W boxes are already
  the region to train on).
  """
  rung = _dr_index(level)

  # DR0 and up: the base cfg's own randomisation is off unless a rung puts it back.
  if rung < _dr_index("DR-contact"):
    for term in ("pad_friction_slide", "pad_friction_spin"):
      cfg.events.pop(term, None)
  cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)

  if rung >= _dr_index("DR-object"):
    object_geom = SceneEntityCfg(object_name, geom_names=(f"{object_name}_geom",))
    cfg.events["object_friction"] = EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": object_geom,
        "operation": "abs",
        "distribution": "uniform",
        "axes": [0],  # sliding only; torsional friction is the pads' business.
        "ranges": _DR_OBJECT_FRICTION,
      },
    )
    cfg.events["object_mass"] = EventTermCfg(
      mode="startup",
      func=dr.pseudo_inertia,
      params={
        "asset_cfg": SceneEntityCfg(object_name, body_names=(object_name,)),
        "alpha_range": _DR_OBJECT_ALPHA,
        "distribution": "uniform",
      },
    )

  if rung >= _dr_index("DR-robot"):
    cfg.events["reset_robot_joints"].params["position_range"] = (
      -_DR_ROBOT_RESET_JITTER,
      _DR_ROBOT_RESET_JITTER,
    )
    cfg.events["actuator_gains"] = EventTermCfg(
      mode="startup",
      func=dr.pd_gains,
      params={
        "asset_cfg": action_joints,
        "kp_range": _DR_ROBOT_GAIN_SCALE,
        "kd_range": _DR_ROBOT_GAIN_SCALE,
        "operation": "scale",
      },
    )
    for name, func, ranges in (
      ("joint_friction", dr.joint_friction, _DR_ROBOT_JOINT_FRICTION),
      ("joint_damping", dr.joint_damping, _DR_ROBOT_JOINT_DAMPING),
    ):
      cfg.events[name] = EventTermCfg(
        mode="startup",
        func=func,
        params={
          "asset_cfg": action_joints,
          "operation": "abs",
          "distribution": "uniform",
          "ranges": _on_joints(ranges, tuple(action_joints.joint_names)),
        },
      )
    # Action delay is *not* here, and cannot be: mjlab's delay buffer is configured on the
    # actuator (`delay_min_lag`/`delay_max_lag` and friends, `mjlab/actuator/actuator.py:67-85`),
    # so it is a property of the robot cfg rather than something an event can switch on. Per-env
    # lags can then be randomised through `Actuator.set_lags`. Adding it means a robot variant,
    # which is why this rung stops short of it.

  if rung >= _dr_index("DR-obs"):
    # The actor and critic groups share their term objects (`make_tabletop_env_cfg` copies the
    # dict, not the terms), so the actor's terms are *replaced*, not edited. That matters for
    # the lag: `enable_corruption=False` switches off a group's noise but not its delay, so an
    # edited shared term would feed the critic stale object poses too. The critic runs only in
    # sim and keeps the true pose.
    terms = cfg.observations["actor"].terms
    for name, bias, noise in (
      ("object_pos", _DR_OBS_OBJECT_POS_BIAS, _DR_OBS_OBJECT_POS_NOISE),
      ("object_yaw", _DR_OBS_OBJECT_YAW_BIAS, _DR_OBS_OBJECT_YAW_NOISE),
    ):
      terms[name] = dataclasses.replace(
        terms[name],
        noise=NoiseModelWithAdditiveBiasCfg(
          noise_cfg=Unoise(n_min=-noise, n_max=noise),
          # "abs", not the default "add": mjlab 1.6.0's `NoiseModelWithAdditiveBias.reset`
          # computes `bias = bias_noise_cfg.apply(bias)`, so with "add" every reset adds a
          # fresh draw to the old bias and it random-walks across episodes (measured: 15 mm
          # against an 8 mm bound after the first reset alone). "abs" replaces it.
          bias_noise_cfg=Unoise(n_min=-bias, n_max=bias, operation="abs"),
        ),
        delay_min_lag=_DR_OBS_OBJECT_LAG[0],
        delay_max_lag=_DR_OBS_OBJECT_LAG[1],
      )
    cfg.events["encoder_bias"] = EventTermCfg(
      mode="reset",
      func=dr.encoder_bias,
      params={
        "asset_cfg": action_joints,
        "bias_range": (-_DR_OBS_ENCODER_BIAS, _DR_OBS_ENCODER_BIAS),
      },
    )

  if rung >= _dr_index("DR-scene"):
    cfg.events["table_pose"] = EventTermCfg(
      mode="startup",
      func=dr.body_pos,
      params={
        "asset_cfg": SceneEntityCfg(TABLE, body_names=(TABLE,)),
        "operation": "add",
        "distribution": "uniform",
        "axes": [0, 1, 2],
        "ranges": {
          0: (-_DR_SCENE_TABLE_XY, _DR_SCENE_TABLE_XY),
          1: (-_DR_SCENE_TABLE_XY, _DR_SCENE_TABLE_XY),
          2: (-_DR_SCENE_TABLE_HEIGHT, _DR_SCENE_TABLE_HEIGHT),
        },
      },
    )
    # The spawn box widens about its own centre. `reset_objects_on_surface` reads the table's
    # position and height out of physics, so the cube still lands on the surface, at the same
    # offset from the table's centre, wherever the table is.
    if spawn_widening:
      pose_range = cfg.events["reset_objects"].params["pose_range"]
      for axis in ("x", "y"):
        lo, hi = pose_range[axis]
        pose_range[axis] = (lo - _DR_SCENE_EXTRA_SPAWN, hi + _DR_SCENE_EXTRA_SPAWN)

  return cfg


def apply_play_overrides(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Turn an env cfg into its ``--play`` variant: no corruption, no timeout, no curriculum."""
  cfg.episode_length_s = int(1e9)
  cfg.observations["actor"].enable_corruption = False
  cfg.curriculum = {}
  return cfg
