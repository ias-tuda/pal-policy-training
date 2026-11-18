"""KANGAROO lower-body velocity policy with scripted arms (loco-manipulation stage P1).

Built on `pal_kangaroo_flat_env_cfg` (velocity task, flat terrain). What changes:

- robot: the tabletop grippers model under the G20 gains, free base -- the same plant
  WC-c pick-place was trained on (`pick_place/kangaroo/env_cfg.py`, GAINS=G20), so that
  WC-c can later drive the arms of this policy (stage P2) without a plant change;
- actions: the policy drives legs + pelvis only (14 targets, as
  `Mjlab-Velocity-Flat-Pal-Kangaroo-Lower-Body`); arms and jaws are driven by
  `mdp.ScriptedArmJointPositionAction` (random minimum-jerk targets with a curriculum);
- payload: 0-1 kg point mass on each hand (`arm_*_7_link`), drawn at startup;
- commands: 35 % standing envs, because the manipulation half of the demo stands;
- posture reward: legs + pelvis only (the policy cannot hold the arms at a pose).
"""

import dataclasses
import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from pal_mjlab.robots import (
  KANGAROO_LOWER_BODY_ACTION_SCALE,
  KANGAROO_LOWER_BODY_ACTUATOR_NAMES,
  get_kangaroo_grippers_tabletop_robot_cfg,
)
from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import MANIPULATION_INIT_STATE
from pal_mjlab.tasks.locomanip import mdp
from pal_mjlab.tasks.velocity.kangaroo.env_cfgs import pal_kangaroo_flat_env_cfg

#: Actuated legs + pelvis (femur/knee are four-bar driven and excluded, as in
#: REGEX_ALL_ACTUATED_JOINTS).
REGEX_LOWER_BODY_ACTUATED = r"^(leg_.*_(1|2|3|4|5|length)_joint|pelvis_[12]_joint)$"

#: Sampling half-widths around `_MANIPULATION_ARM_HOME_POSE` at curriculum level 1.
#: Shoulder/elbow wide enough to cover a reach to the table and a lift; wrist smaller.
ARM_AMPLITUDE = {
  r"arm_.*_[1-4]_joint": 0.8,
  r"arm_.*_5_joint": 0.8,
  r"arm_.*_[67]_joint": 0.6,
}
#: The WC-c jaw values (`manipulation/robots/kangaroo.py` _JAW_OPEN / _JAW_CLOSED).
JAW_OPEN_CLOSED = (0.0, 0.78)

#: Env steps from level 0.1 to 1 (24 steps/iteration -> ~4000 iterations).
ARM_LEVEL_RAMP_STEPS = 4000 * 24


def pal_kangaroo_locomanip_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = pal_kangaroo_flat_env_cfg(play=play)

  robot = get_kangaroo_grippers_tabletop_robot_cfg(gains="G20")
  robot = dataclasses.replace(robot, init_state=MANIPULATION_INIT_STATE)
  cfg.scene.entities = {"robot": robot}

  # -- Actions: policy = legs + pelvis, arms + jaws scripted.
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = KANGAROO_LOWER_BODY_ACTION_SCALE
  joint_pos_action.actuator_names = KANGAROO_LOWER_BODY_ACTUATOR_NAMES
  cfg.actions["arms"] = mdp.ScriptedArmJointPositionActionCfg(
    entity_name="robot",
    actuator_names=(r"arm_.*_[1-7]_joint", r"gripper_.*_finger_joint"),
    amplitude=ARM_AMPLITUDE,
    binary={r"gripper_.*_finger_joint": JAW_OPEN_CLOSED},
    segment_duration_s=(0.8, 3.0),
    hold_probability=0.3,
    level_start=0.1,
    level_ramp_steps=0 if play else ARM_LEVEL_RAMP_STEPS,
  )

  # -- Commands: the demo stands while it manipulates.
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.rel_standing_envs = 0.35

  # -- Events: hand payload. A point mass at the hand COM (dr.body_mass leaves inertia).
  cfg.events["hand_payload"] = EventTermCfg(
    mode="startup",
    func=dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(r"arm_.*_7_link",)),
      "operation": "add",
      "ranges": (0.0, 1.0),
    },
  )

  # -- Rewards: posture over what the policy controls.
  pose = cfg.rewards["pose"].params
  pose["asset_cfg"].joint_names = (REGEX_LOWER_BODY_ACTUATED,)
  pose["std_standing"] = {REGEX_LOWER_BODY_ACTUATED: 0.05}
  for key in ("std_walking", "std_running"):
    pose[key] = {k: v for k, v in pose[key].items() if not k.startswith("arm_")}
  cfg.rewards["action_rate_l2"].weight = -0.2
  cfg.rewards["upright"].weight = 1.5

  # Lower-body IMU noise, as `pal_kangaroo_lower_body_flat_env_cfg` (microstrain IMU).
  cfg.observations["actor"].terms["imu_projected_gravity"].noise = Unoise(
    n_min=-0.025, n_max=0.025
  )
  cfg.observations["actor"].terms["base_lin_acc"].noise = Unoise(n_min=-1.0, n_max=1.0)

  return cfg


def pal_kangaroo_locomanip_lowspeed_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """P1.1: `pal_kangaroo_locomanip_flat_env_cfg` trained for precise walk_to.

  The P1 policy at 4500 iterations stalls ~10 cm short of a walk_to target under a pure
  P-controller and drifts up to 24 cm while standing (locomanip_docs/report.md §P1): it
  ignores small twist commands. Three changes, all from existing PAL configs:
  - the twist command becomes `mdp.DualBandVelocityCommandCfg`, with 40 % of moving
    commands sampled inside |v| < 0.3 m/s, |wz| < 0.3 rad/s;
  - linear/angular tracking std tightened to sqrt(0.1) / sqrt(0.25), as
    `pal_kangaroo_rough_env_cfg` does "so the reward stays discriminative at low command
    speeds";
  - air-time reward from 0.05 m/s commands on (PAL stable loco v639), not 0.5 m/s.
  """
  cfg = pal_kangaroo_locomanip_flat_env_cfg(play=play)
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  kwargs = {f.name: getattr(twist, f.name) for f in dataclasses.fields(twist) if f.init}
  kwargs.update(init_velocity_prob=0.0, rel_forward_envs=0.0)
  cfg.commands["twist"] = mdp.DualBandVelocityCommandCfg(
    **kwargs,
    rel_inside_low=0.4,
    min_lin_vel_x=0.3,
    min_lin_vel_y=0.3,
    min_ang_vel_z=0.3,
  )
  cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.1)
  cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.25)
  # Step for small commands too. The base config rewards air time only above 0.5 m/s, so
  # below that the policy shuffles. Values of PAL's shipped lower-body walking policy
  # (github.com/pal-robotics/pal_policy_deployer, models.yaml stable/lower_body_loco
  # 639.0.0, run B12A2): 0.2-0.45 s air time, from 0.05 m/s on.
  air = cfg.rewards["air_time"].params
  air["threshold_min"], air["threshold_max"], air["command_threshold"] = 0.2, 0.45, 0.05
  return cfg
