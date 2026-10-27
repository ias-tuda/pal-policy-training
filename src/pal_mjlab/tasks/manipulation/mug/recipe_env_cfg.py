"""The base the mug, coaster and kugel tasks build on: the grasp ladder on the deployed plant.

On ``faster-moves`` these tasks sat on Claudius's standing recipe
(``kangaroo_standing_recipe_env_cfg``: V36 arm gains, delta actions, ``BinaryJaw``). That stack
is gone; the robot, its actions and the sim2real chain are now the pick-and-place controller the
real robot runs, WC-c: ``SPAWN=WC TARGET=WC CMD_OBS=cmd DR=DR-scene SLEW=stack BASELINE=B
GAINS=G20 EFFORT=CTP LAG=FP+JSW ACTION=abs`` (``pick_place/kangaroo/env_cfg.py``). Only the
rewards, the actor's object terms and the sensors they read are the recipe's: the ladder in
``mdp.grasp_ladder``, with the old names (``reach``, ``align``, ``secure``, ``grasp``, ``lift``,
``goal``) so that the task cfgs on top find the terms they edit.

What did not come across: the per-episode arm assignment (``ARM_SELECT``) and with it the
bimanual / upper-body variants, and the tracker's dropout and hold (``observations._object_pose_w``
has the lag only).
"""

from __future__ import annotations

from collections.abc import Mapping

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from pal_mjlab.tasks.manipulation import mdp, mug_scene
from pal_mjlab.tasks.manipulation.mdp.occlusion import (  # noqa: F401 -- re-exported
  CAMERA_VIEW_MARGIN,
  LEFT_ARM_SIGHT_SEGMENTS,
  TORSO_FRONT_CAMERA_BODY,
  TORSO_FRONT_CAMERA_POS,
)
from pal_mjlab.tasks.manipulation.pick_place.arm_table import make_arm_table_sensor
from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import (
  BASELINES,
  EFFORT_BOUNDS,
)
from pal_mjlab.tasks.manipulation.pick_place.rewards import effort_hinge_reward
from pal_mjlab.tasks.manipulation.robots import (
  KANGAROO_GRIPPERS_LEFT_ARM_WC,
  kangaroo_grippers_left_arm,
)
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import (
  COMMAND,
  LATENCY_MODELS,
  OBJECT,
  ROBOT,
  SUPPORT_SENSOR,
  make_tabletop_env_cfg,
  trunk_clearance_rewards,
)

#: The deployed controller (WC-c), by its plan codes.
GAINS = "G20"
EFFORT = "CTP"
BASELINE = "B"
DR = "DR-scene"
LAG = "FP+JSW"

#: Spawn range of ``-Wide``, about the table centre: x 0.40-0.60 in the env frame, the same
#: area as ``pick_place``'s ``SPAWN_BOXES["WC"]``.
WIDE_OBJECT_POSE_RANGE = {"x": (-0.15, 0.05), "y": (-0.04, 0.26), "yaw": (-3.14, 3.14)}

#: Discount of ``-Patient``: at 50 Hz 0.99 looks ~2 s ahead, too short for a slow arm. Set it in
#: the runner cfg (``rl_cfg.PPO_RECIPES["PPO-stack"]`` with ``gamma`` replaced).
PATIENT_GAMMA = 0.995

#: Each jaw face of the left gripper, any of its geoms, against the object. A grasp is both
#: (``mdp.grasp_ladder``); the pick-and-place grasp sensor watches the fingertip pads alone.
FACE_SENSORS = (("left_face_a_contact", "left_face_b_contact"),)
_FACE_GEOMS = (
  "gripper_left_(fingertip|inner_finger)_left_collision",
  "gripper_left_(fingertip|inner_finger)_right_collision",
)

#: Arm-table contact ends the episode (``-NoTable``).
ARM_TABLE_SENSOR = "arm_table_contact"


def kangaroo_standing_recipe_env_cfg(
  play: bool = False,
  objects: Mapping[str, EntityCfg] | None = None,
  object_width: float = mug_scene.MUG_WIDTH,
  object_pose_range: Mapping[str, tuple[float, float]] = WIDE_OBJECT_POSE_RANGE,
  min_separation: float = 0.0,
  command_cfg: CommandTermCfg | None = None,
  episode_length_s: float = 20.0,
  object_tilt_limit: float | None = None,
) -> ManagerBasedRlEnvCfg:
  """The grasp ladder for ``objects[OBJECT]`` on the WC-c robot, actions and sim2real chain."""
  effort = EFFORT_BOUNDS[EFFORT]
  baseline = BASELINES[BASELINE]
  robot = kangaroo_grippers_left_arm(
    GAINS, KANGAROO_GRIPPERS_LEFT_ARM_WC, arm_effort_scale=effort.effort_scale
  )
  gripper = robot.gripper("left")
  cfg = make_tabletop_env_cfg(
    robot,
    gripper_name="left",
    objects=objects,
    command_cfg=command_cfg,
    object_pose_range=object_pose_range,
    min_separation=min_separation,
    episode_length_s=episode_length_s,
    dr_level=DR,
    table_top_z=baseline.table_top_z,
    observation_set=baseline.observation_set,
    jaw_slew_rate=baseline.jaw_slew_rate,
    dr_spawn_widening=False,
    arm_range_clip=effort.range_clip,
    arm_effort_band=effort.effort_scale is not None,
    latency=LATENCY_MODELS[LAG],
    stack_speed=True,
    action_mode="abs",
    command_obs=True,
  )

  cfg.scene.sensors = (
    *cfg.scene.sensors,
    *(
      ContactSensorCfg(
        name=name,
        primary=ContactMatch(mode="geom", pattern=pattern, entity=ROBOT),
        secondary=ContactMatch(mode="body", pattern=OBJECT, entity=OBJECT),
        fields=("found",),
        reduce="none",
        num_slots=2,
        history_length=1,
      )
      for name, pattern in zip(FACE_SENSORS[0], _FACE_GEOMS, strict=True)
    ),
    make_arm_table_sensor(gripper),
  )

  pads = SceneEntityCfg(ROBOT, site_names=gripper.pad_sites, preserve_order=True)
  arm = tuple(
    p for p in robot.action_joint_regex if p != gripper.finger_joint
  )  # the finger joint is its own entry
  _set_ladder_observations(cfg, pads)
  cfg.rewards = {
    **{k: cfg.rewards[k] for k in ("action_rate_l2", "joint_pos_limits")},
    **_ladder_rewards(pads, object_width),
    "arm_effort": effort_hinge_reward(SceneEntityCfg(ROBOT, joint_names=arm)),
    **trunk_clearance_rewards(robot),
  }
  cfg.terminations["table_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": ARM_TABLE_SENSOR, "force_threshold": 0.0},
  )
  if object_tilt_limit is not None:
    cfg.terminations["object_flipped"] = TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": object_tilt_limit, "asset_cfg": SceneEntityCfg(OBJECT)},
    )
  if play:
    cfg.episode_length_s = int(1e9)
  return cfg


def _ladder_rewards(
  pads: SceneEntityCfg, object_width: float
) -> dict[str, RewardTermCfg]:
  """The recipe's ladder; weights never decrease up it, so no rung is a loss to take."""
  return {
    "reach": RewardTermCfg(
      func=mdp.staged_position_reward_jaw,
      weight=1.0,
      params={
        "command_name": COMMAND,
        "object_name": OBJECT,
        "reaching_std": 0.4,
        "bringing_std": 0.2,
        "grasp_cfg": pads,
      },
    ),
    "align": RewardTermCfg(
      func=mdp.jaw_aligned_at_object,
      weight=2.0,
      params={"object_name": OBJECT, "grasp_cfg": pads, "object_width": object_width},
    ),
    "secure": RewardTermCfg(
      func=mdp.jaw_closing_on_object,
      weight=2.0,
      params={"object_name": OBJECT, "grasp_cfg": pads, "object_width": object_width},
    ),
    "grasp": RewardTermCfg(
      func=mdp.grasped_pair, weight=3.0, params={"sensor_names": FACE_SENSORS}
    ),
    "lift": RewardTermCfg(
      func=mdp.object_lifted_pair,
      weight=3.0,
      params={
        "grasp_sensor_names": FACE_SENSORS,
        "support_sensor_name": SUPPORT_SENSOR,
      },
    ),
    "goal": RewardTermCfg(
      func=mdp.bring_object_while_held_pair,
      weight=4.0,
      params={
        "command_name": COMMAND,
        "object_name": OBJECT,
        "grasp_sensor_names": FACE_SENSORS,
        "std": 0.15,
      },
    ),
  }


def _set_ladder_observations(cfg: ManagerBasedRlEnvCfg, pads: SceneEntityCfg) -> None:
  """The recipe's jaw-relative object terms, appended to both groups.

  ``jaw_to_object`` and ``object_to_goal`` carry the lag of the group's own ``object_pos`` term
  (the WC-c tracker lag in the actor, none in the critic); ``object_yaw_to_jaw`` the noise of its
  ``object_yaw``. The cube's base-frame ``object_pos`` / ``object_yaw`` stay.
  """
  for group in cfg.observations.values():
    lag = group.terms["object_pos"].params.get("lag")
    yaw = group.terms.get("object_yaw")
    group.terms["jaw_to_object"] = ObservationTermCfg(
      func=mdp.jaw_to_object_b,
      params={"object_name": OBJECT, "grasp_cfg": pads, "lag": lag},
      noise=group.terms["object_pos"].noise,
    )
    group.terms["object_yaw_to_jaw"] = ObservationTermCfg(
      func=mdp.object_yaw_to_jaw,
      params={"object_name": OBJECT, "grasp_cfg": pads, "lag": lag},
      noise=None if yaw is None else yaw.noise,
    )
    group.terms["object_to_goal"] = ObservationTermCfg(
      func=mdp.object_to_goal_b,
      params={"object_name": OBJECT, "command_name": COMMAND, "lag": lag},
    )
