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
    primary=ContactMatch(mode="geom", pattern=gripper.pad_geom_regex, entity=ROBOT),
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


def make_tabletop_env_cfg(
  robot: ManipulationRobotCfg,
  gripper_name: str | None = None,
  objects: Mapping[str, EntityCfg] | None = None,
  command_cfg: CommandTermCfg | None = None,
  object_pose_range: Mapping[str, tuple[float, float]] | None = None,
  min_separation: float = 0.0,
  episode_length_s: float = 20.0,
  hold_nominal: bool = True,
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
  """
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

  # -- Actions --------------------------------------------------------------------
  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name=ROBOT,
      actuator_names=robot.action_joint_regex,
      scale=robot.action_scale,
      use_default_offset=True,
    )
  }

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
  }

  # -- Terminations ---------------------------------------------------------------
  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "object_dropped": TerminationTermCfg(
      func=mdp.entity_below_height,
      params={"entity_name": OBJECT, "min_height": scene.TABLE_TOP_Z - 0.2},
    ),
  }

  sensors = [
    make_grasp_sensor(gripper, OBJECT),
    make_support_sensor(OBJECT),
    make_hand_ground_sensor(gripper),
  ]

  entities: dict[str, EntityCfg] = {
    ROBOT: robot.build_entity_cfg(),
    TABLE: scene.get_table_cfg(),
  }
  entities.update(objects)

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities=entities,
      sensors=tuple(sensors),
      num_envs=1,
      env_spacing=2.5,
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
        # actions). Decimation is raised to keep the control rate at 50 Hz.
        timestep=0.002,
        iterations=10,
        ls_iterations=20,
        impratio=10,
        cone="elliptic",
      ),
    ),
    decimation=10,
    episode_length_s=episode_length_s,
  )


def apply_play_overrides(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Turn an env cfg into its ``--play`` variant: no corruption, no timeout, no curriculum."""
  cfg.episode_length_s = int(1e9)
  cfg.observations["actor"].enable_corruption = False
  cfg.curriculum = {}
  return cfg
