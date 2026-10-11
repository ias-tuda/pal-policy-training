"""Plan §13.1's chain (`pick-place-insert-plan.md` §13 *Structure*, SEQ-H): both arms, the left running L-H, the right R-I
through the mirror map, in one env -- for `scripts/eval_chain_insert.py` only (nothing trains here).

``ARMS=both S13=CHAIN`` with `BS12`'s plant selectors (G20, CTP, DR-scene, FP+JS, AH10, TABLE_DZ). The scene:

* the robot with both arms in the action space (16-D), the left at the side-grasp scales, the right at WC's
  (`robots.both_arms_scales`), the right's home mirrored;
* the socket block (`scene.get_socket_block_spec`, origin at its bottom face) on the left's area (L-H's
  `HOLD_SPAWN_*`, any yaw), the pin (`scene.get_pin_spec`, origin at its centre) on the right's -- mirror(R-I's
  `PIN_AREA_*`) -- out of the right start pose's finger keepout (§12's `HOME_HAND_KEEPOUT` mirrored);
* H as L-H's goal (the block's centre 10 cm + half its height over the table at (0.45, 0.00));
* one corrupted observation group ``actor`` with every term either policy reads, with its training noise, bias
  and lag: both arms' joints and last actions, both TCPs, the block's centre (L-H's object), yaw (4-fold) and up
  axis, H, the pin's position, yaw (4-fold) and tilt, the block's origin (R-I's target). One shared tracker
  bias (one camera) plus each object's own, as both skills trained with;
* the sensors the reads need: each hand's grasp on its part, both parts against the table, arm-arm (S11's HH),
  arm 1-6 on the table per arm, each hand on the table.

No reward is paid (the regularisers only, so the managers have a term); the episode ends at its time-out or when a
part falls off the table.
"""

from __future__ import annotations

import dataclasses

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from pal_mjlab.tasks.manipulation import mdp, scene
from pal_mjlab.tasks.manipulation.pick_place import bimanual as bi
from pal_mjlab.tasks.manipulation.pick_place import hold as H
from pal_mjlab.tasks.manipulation.pick_place import insert_held as IH
from pal_mjlab.tasks.manipulation.pick_place.arm_table import (
  ArmTableCfg,
  apply_arm_table,
)
from pal_mjlab.tasks.manipulation.pick_place.box import (
  BOX_YAW_BIAS,
  BOX_YAW_NOISE,
  TRACKER_OWN_BIAS,
  TRACKER_SHARED_BIAS,
)
from pal_mjlab.tasks.manipulation.pick_place.insert import (
  BLOCK,
  HOME_HAND_KEEPOUT,
  PIN_TILT_NOISE,
)
from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import (
  BASELINES,
  EFFORT_BOUNDS,
  SPAWN_BOXES,
  Baseline,
  EffortBound,
  make_pick_place_env_cfg,
)
from pal_mjlab.tasks.manipulation.robots import (
  KANGAROO_GRIPPERS_LEFT_ARM_SIDE,
  KANGAROO_GRIPPERS_LEFT_ARM_WC,
  both_arms_scales,
  kangaroo_grippers_both_arms,
)
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import (
  _DR_OBS_OBJECT_POS_NOISE,
  COMMAND,
  LATENCY_MODELS,
  OBJECT,
  ROBOT,
  TABLE,
  LatencyModelCfg,
  make_grasp_sensor,
  make_hand_surface_sensor,
)

#: The chain's episode (plan §13 *Reads and bars*: 30 s).
EPISODE_S = 30.0
#: The block's support sensor (the pin's is the env's own, `SUPPORT_SENSOR`).
BLOCK_SUPPORT = "block_table_contact"
#: The least xy distance between the block's and the pin's centres at the reset.
PART_MIN_SEPARATION = 0.14
#: The right start pose's finger keepout: §12's (left) mirrored through y = 0.
RIGHT_HOME_KEEPOUT = (
  HOME_HAND_KEEPOUT[0],
  HOME_HAND_KEEPOUT[1],
  -HOME_HAND_KEEPOUT[3],
  -HOME_HAND_KEEPOUT[2],
)
#: The terms of the ``actor`` group, in order (`eval_chain_insert.ChainAdapter` slices them by name).
TERMS = (
  "joint_pos",
  "joint_vel",
  "actions",
  "tcp_pos",
  "tcp_quat",
  "tcp_pos_right",
  "tcp_quat_right",
  "block_centre",
  "block_yaw",
  "hold_goal",
  "block_up",
  "pin_pos",
  "pin_yaw",
  "block_pos",
  "pin_tilt",
)


def make_insert_chain_env_cfg(
  gains: str = "G20",
  effort: EffortBound = EFFORT_BOUNDS["CTP"],
  dr_level: str = "DR-scene",
  latency: LatencyModelCfg = LATENCY_MODELS["FP+JS"],
  baseline: Baseline = BASELINES["B"],
  episode_length_s: float = EPISODE_S,
  arm_table: ArmTableCfg | None = None,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """The chain's env (module docstring)."""
  robot = kangaroo_grippers_both_arms(
    gains,
    arm_effort_scale=effort.effort_scale,
    base=both_arms_scales(
      KANGAROO_GRIPPERS_LEFT_ARM_SIDE, KANGAROO_GRIPPERS_LEFT_ARM_WC
    ),
  )
  left, right = robot.gripper("left"), robot.gripper("right")
  cfg = make_pick_place_env_cfg(
    robot,
    gripper_name="left",
    spawn=SPAWN_BOXES["WC"],
    dr=dr_level,
    reward="R-C-strict+M1+M2+M3+M6+J+M8",
    play=False,
    baseline=baseline,
    effort=effort,
    latency=latency,
    episode_length_s=episode_length_s,
    stack_speed=True,
    arm_table=arm_table,
    tracker="split",
  )
  top_z = baseline.table_top_z
  half = scene.BLOCK_HEIGHT / 2
  cx, cy = scene.TABLE_POS[0], scene.TABLE_POS[1]

  # -- Scene. -----------------------------------------------------------------------------------------
  cfg.scene.entities = {
    **cfg.scene.entities,
    TABLE: scene.get_table_cfg(top_z, contype=scene.INSERT_TABLE_CONTYPE),
    OBJECT: scene.get_pin_cfg(name=OBJECT),
    BLOCK: scene.get_socket_block_cfg(name=BLOCK),
  }
  sensors = [s for s in cfg.scene.sensors if s.name != left.contact_sensor_name]
  cfg.scene.sensors = (
    *sensors,
    make_grasp_sensor(left, BLOCK),
    make_grasp_sensor(right, OBJECT),
    bi._support_sensor(BLOCK, BLOCK_SUPPORT),
    make_hand_surface_sensor(right),
    bi._arm_arm_sensor(),
  )
  apply_arm_table(cfg, right, arm_table, prefix="right_")

  # -- Events: the pin on the right's area, then the block on the left's. -----------------------------
  pin_reset = cfg.events["reset_objects"]
  pin_reset.params["pose_range"] = {
    "x": (IH.PIN_AREA_X[0] - cx, IH.PIN_AREA_X[1] - cx),
    "y": (-IH.PIN_AREA_Y[1] - cy, -IH.PIN_AREA_Y[0] - cy),
    "yaw": pin_reset.params["pose_range"]["yaw"],
  }
  pin_reset.params.pop("goal_area", None)
  pin_reset.params["keepout"] = RIGHT_HOME_KEEPOUT
  events = {}
  for name, term in cfg.events.items():
    events[name] = term
    if name == "reset_objects":
      events["reset_block"] = EventTermCfg(
        func=mdp.reset_box_in_area,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg(BLOCK),
          "surface_cfg": SceneEntityCfg(TABLE),
          "object_name": OBJECT,
          "area_x": (H.HOLD_SPAWN_X[0] - cx, H.HOLD_SPAWN_X[1] - cx),
          "area_y": (H.HOLD_SPAWN_Y[0] - cy, H.HOLD_SPAWN_Y[1] - cy),
          "yaw": 3.141592653589793 / 4,
          "min_distance": PART_MIN_SEPARATION,
        },
      )
  cfg.events = events
  for name in ("object_friction", "object_mass"):
    cfg.events.pop(
      name, None
    )  # the pin's own friction stands (§12); the parts' masses are nominal here
  cfg.events["reset_tracker_bias"] = EventTermCfg(
    func=mdp.reset_tracker_bias,
    mode="reset",
    params={
      "shared": TRACKER_SHARED_BIAS,
      "own": TRACKER_OWN_BIAS,
      "keys": (OBJECT, BLOCK),
    },
  )
  cfg.events["reset_tracker_yaw_bias"] = EventTermCfg(
    func=mdp.reset_tracker_yaw_bias,
    mode="reset",
    params={"half": BOX_YAW_BIAS, "keys": (OBJECT, BLOCK)},
  )

  # -- H. ------------------------------------------------------------------------------------------------
  cfg.commands[COMMAND] = mdp.SurfaceOffsetCommandCfg(
    entity_name=BLOCK,
    surface_name=TABLE,
    offset=H.HOLD_XY,
    offset_range=(0.0, 0.0),
    frame="env",
    height=H.HOLD_DZ,
    resampling_time_range=(1e9, 1e9),
    success_threshold=H.HELD_DISTANCE,
    debug_vis=False,
  )

  # -- Observations. -------------------------------------------------------------------------------------
  a = cfg.observations["actor"].terms
  lag = a["object_pos"].params.get("lag")
  pos_noise = Unoise(n_min=-_DR_OBS_OBJECT_POS_NOISE, n_max=_DR_OBS_OBJECT_POS_NOISE)
  tcp_r = SceneEntityCfg(ROBOT, site_names=(right.tcp_site,))
  terms = {
    "joint_pos": a["joint_pos"],
    "joint_vel": a["joint_vel"],
    "actions": a["actions"],
    "tcp_pos": a["tcp_pos"],
    "tcp_quat": a["tcp_quat"],
    "tcp_pos_right": dataclasses.replace(a["tcp_pos"], params={"asset_cfg": tcp_r}),
    "tcp_quat_right": dataclasses.replace(a["tcp_quat"], params={"asset_cfg": tcp_r}),
    "block_centre": ObservationTermCfg(
      func=mdp.object_point_biased_b,
      params={
        "object_name": BLOCK,
        "local": (0.0, 0.0, half),
        "bias_key": BLOCK,
        "lag": lag,
      },
      noise=pos_noise,
    ),
    "block_yaw": ObservationTermCfg(
      func=mdp.box_yaw_b,
      params={"box_name": BLOCK, "bias_key": BLOCK, "noise": BOX_YAW_NOISE, "lag": lag},
    ),
    "hold_goal": ObservationTermCfg(
      func=mdp.goal_position_b, params={"command_name": COMMAND}
    ),
    "block_up": ObservationTermCfg(
      func=mdp.object_up_b,
      params={"object_name": BLOCK, "lag": lag},
      noise=Unoise(n_min=-H.UP_NOISE, n_max=H.UP_NOISE),
    ),
    "pin_pos": ObservationTermCfg(
      func=mdp.object_position_biased_b,
      params={"object_name": OBJECT, "bias_key": OBJECT, "lag": lag},
      noise=pos_noise,
    ),
    "pin_yaw": ObservationTermCfg(
      func=mdp.box_yaw_b,
      params={
        "box_name": OBJECT,
        "bias_key": OBJECT,
        "noise": BOX_YAW_NOISE,
        "lag": lag,
      },
    ),
    "block_pos": ObservationTermCfg(
      func=mdp.object_position_biased_b,
      params={"object_name": BLOCK, "bias_key": BLOCK, "lag": lag},
      noise=pos_noise,
    ),
    "pin_tilt": ObservationTermCfg(
      func=mdp.pin_tilt_b,
      params={"object_name": OBJECT, "noise": PIN_TILT_NOISE, "lag": lag},
    ),
  }
  assert tuple(terms) == TERMS
  cfg.observations = {
    "actor": ObservationGroupCfg(terms, enable_corruption=True),
    "critic": ObservationGroupCfg(
      {k: dataclasses.replace(v, noise=None) for k, v in terms.items()},
      enable_corruption=False,
    ),
  }

  # -- No reward (the managers need a term); ends at the time-out or a part off the table. --------------
  cfg.rewards = {"action_rate_l2": cfg.rewards["action_rate_l2"]}
  cfg.curriculum = {}
  keep = {
    k: v for k, v in cfg.terminations.items() if k in ("time_out", "non_finite_state")
  }
  keep["object_dropped"] = TerminationTermCfg(
    func=mdp.entity_below_height,
    params={"entity_name": OBJECT, "min_height": top_z - 0.2},
  )
  keep["block_dropped"] = TerminationTermCfg(
    func=mdp.entity_below_height,
    params={"entity_name": BLOCK, "min_height": top_z - 0.2},
  )
  cfg.terminations = keep
  if play:
    from pal_mjlab.tasks.manipulation.tabletop_env_cfg import apply_play_overrides

    apply_play_overrides(cfg)
  return cfg
