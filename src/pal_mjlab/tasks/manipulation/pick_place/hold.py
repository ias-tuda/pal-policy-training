"""Plan §13.1's left skill (`pick-place-insert-plan.md`): L-H picks the socket block off the table with a
side grasp and holds it level and still at H; L-HP does the same against pushes on the block's top.

``S13=L-H`` / ``S13=L-HP`` (`kangaroo/env_cfg.py`) on top of `BS12`'s one-arm cfg. Against that cfg:

* **the object** is the socket block (`scene.get_socket_block_spec`, the v2 part of plan §0.2) with its
  origin moved to its centre (`held_block_spec`), so the spawn, the lift, the goal and the tracker treat
  it as they treat the cube; the deployer adds half the block's height along its axis to FoundationPose's
  pose (the CAD origin is the bottom face's centre);
* **the spawn** is `HOLD_SPAWN`, the part of `WC` the hand reaches horizontally (step 0 item 1), any yaw;
* **the goal** is H (`HOLD_XY`, `HOLD_DZ`): the block's bottom 10 cm over the table at (0.45, 0.00) in the
  robot frame, its centre ``HOLD_DZ + half`` over the table (`mdp.SurfaceOffsetCommandCfg.height`);
* **the reward** is RCTP's reach / grasp / faded lift / M2 goal with every one of them paid only for a
  side grasp (`mdp.side_graded`, G(approach elevation, `SIDE_STD`)), the reach on the side-grasp point of
  whichever face within 61 deg of facing the robot is nearer (`mdp.side_reach`), M2's narrow term at 1 cm, a level grade near H and the per-step `held` (×8: within
  1 cm of H, tilt < 3 deg, slower than 2 cm/s, hand horizontal and pointing forward). No release terms:
  the left never lets go in this task;
* **the actor** sees the block's yaw as (cos 4 psi, sin 4 psi) through its tracker with S10.1's yaw bias and
  jitter (`mdp.box_yaw_b`, in the cube's yaw slot) and its up axis appended (40-D on `BS2`); the critic adds
  `grasped`, the side grade and `held`;
* ``L-HP`` adds `mdp.push_object_top` (0-10 N down, 0-2 N sideways, 0.5-1.5 s, from 4 s on).

The hand comes from the robot's side of the block, pointing forward (`HOLD_APPROACH` = +x): plan D7 had it
come from +y, which the arm reaches horizontally on 34 % of WC's (cell, yaw) pairs against 67 % from the
front (step 0 item 1, `results/S13-insert/step0/`); see the plan's *§13.1 as implemented*.
"""

from __future__ import annotations

import dataclasses
import math

import mujoco
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from pal_mjlab.tasks.manipulation import mdp, scene
from pal_mjlab.tasks.manipulation.pick_place.box import BOX_YAW_BIAS, BOX_YAW_NOISE
from pal_mjlab.tasks.manipulation.robot_cfg import GripperCfg
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import (
  COMMAND,
  OBJECT,
  ROBOT,
  TABLE,
)

#: H, the block's presentation pose (plan §13 *The hold and H*): its bottom centre at `HOLD_XY` in the
#: robot frame (= the env frame) and `HOLD_DZ` above the table top, level, its faces on the robot's axes.
HOLD_XY = (0.45, 0.00)
HOLD_DZ = 0.10
#: The direction the hand points in at a side grasp and at H, robot frame: forward (step 0 item 1).
HOLD_APPROACH = (1.0, 0.0, 0.0)
#: The pads' height above the block's bottom at the grasp, 8 mm under its top. Step 0 item 1 planned most cells at
#: 40 mm, but there the palm's hull is 1.7 mm over the table (`results/S13-insert/step0/dummy_hand_40.json`) and the
#: scripted grasp at 40 mm scraped it in 7 of 8 envs; at 48 mm it is ~10 mm clear.
PAD_HEIGHT = 0.048
#: How far short of the block's axis the pads close, toward the hand: at 0 the jaw's inner links (the four-bar's,
#: next to the palm) sit 1.5 mm behind the block's back face (`results/S13-insert/step0/dummy_hand_40.json`), and the
#: scripted grasp pushed the free block 18 mm ahead of the hand before the jaw closed (28 mm after); 10 mm leaves them
#: ~11 mm clear and the pads still on the faces (their pads span ~+/-15 mm along the approach).
GRASP_DEPTH = 0.010
#: How far the pads reach past the TCP along the approach axis with the jaw open (S1.3, re-measured on
#: the tabletop model 2026-10-07: 25.57 mm).
PAD_OFFSET = 0.0256

#: The side grade's width: G(approach elevation, this). 15 deg (the plan's grasp grade).
SIDE_STD = math.radians(15.0)
#: The reach: Gaussian width on the TCP's distance to the side-grasp point (RCTP's 0.2 m) and on the
#: hand's angle to the face (half the reach is paid for pointing at it).
REACH_STD = 0.2
REACH_CONE_STD = 0.5
#: Which faces the reach may take: those whose outward normal is within this of the robot's direction (-x). Step 0
#: item 1 grasped at approaches up to 60 deg off +x (block yaw 30-45 deg on the far / left part of the area).
REACH_FACE_CONE = math.radians(61.0)
#: M2's narrow goal term at H: 1 cm (the plan's `goal_near` at 1 cm; placing tasks use 3 cm).
NEAR_STD = 0.01
#: The level grade near H: G(tilt, 3 deg) while held within 5 cm of H, weight 2.
LEVEL_STD = math.radians(3.0)
LEVEL_NEAR = 0.05
LEVEL_WEIGHT = 2.0
#: `held` (per step, the plan's ×8): within 1 cm of H, tilt < 3 deg, slower than 2 cm/s, the hand within
#: 25 deg of horizontal and within 30 deg of `HOLD_APPROACH`.
HELD_WEIGHT = 8.0
HELD_DISTANCE = 0.01
HELD_TILT = math.radians(3.0)
HELD_SPEED = 0.02
HELD_ELEVATION = math.radians(25.0)
HELD_CONE = math.radians(30.0)

#: The block's spawn: the part of `WC` (x 0.40-0.60, y -0.04-0.26) the hand reaches horizontally at every
#: yaw (step 0 item 1: x 0.40-0.55, y >= 0.08), any yaw. Table-relative as every spawn box.
HOLD_SPAWN_X = (0.40, 0.55)
HOLD_SPAWN_Y = (0.08, 0.26)

#: The actor's tracker noise on the block's up axis (per step, on each of its two components).
UP_NOISE = 0.02

#: L-HP's pushes (plan §13.1): 0-10 N down plus 0-2 N sideways on the block's top, 0.5-1.5 s at a time,
#: from 4 s on, 1-3 s apart.
PUSH_START_S = 4.0
PUSH_DOWN_N = (0.0, 10.0)
PUSH_SIDE_N = 2.0
PUSH_DURATION_S = (0.5, 1.5)
PUSH_GAP_S = (1.0, 3.0)


#: L-H2 (plan §13.1 phase 1.1; Siwei 2026-10-08 "Fix + resume"): what L-H's reward missed. Its policies carried the
#: block to ~5 cm off H (+y 3-5 cm) tilted 21-47 deg and swinging, the hand rolled 9-43 deg (`results/S13-insert/read/`),
#: and held at H on 0 / 64 at every checkpoint: the side grade looked at the approach axis alone, the level grade counted
#: only within 5 cm of H and M2's narrow term was at 1 cm, so nothing paid between "near" (the 15 cm goal term, ~0.9 at
#: 5 cm) and H. L-H2 grades the hand's roll as well (G(roll, `ROLL_STD`) in the side grade), pays a wide level grade
#: whenever the block is grasped and off the table (`LEVEL_LIFTED_*`), and keeps M2's own 3 cm term as ``goal_mid``.
ROLL_STD = math.radians(15.0)
LEVEL_LIFTED_STD = math.radians(20.0)
LEVEL_LIFTED_CLEAR = 0.02
LEVEL_LIFTED_WEIGHT = 2.0
#: L-H3 (phase 1.1): L-H2 put the block within 1 cm of H on 96-97 % of the last 2 s with the hand level (roll 0.3 deg,
#: pitch -0.9 deg), but it grasps the block on the table with the hand pitched ~21 deg down and the pads clamp it, so
#: levelling the hand tilts the block ~18 deg (`results/S13-insert/read/diag_hold.py`, L-H2 @ 6500). L-H3 grades the
#: goal terms by the block's tilt (`LEVEL_GOAL_TILT_STD`) and the hand's roll, and only loosely by its elevation
#: (`LEVEL_GOAL_ELEVATION_STD`; `held` allows 25 deg): a pitched hand holding a level block earns them.
LEVEL_GOAL_TILT_STD = math.radians(10.0)
LEVEL_GOAL_ELEVATION_STD = math.radians(30.0)
LEVEL_GOAL_TERMS = ("goal", "goal_mid", "goal_near")


@dataclasses.dataclass(frozen=True)
class HoldTask:
  """An ``S13`` code's left skill."""

  push: bool = False
  """L-HP: pushes on the block's top (`mdp.push_object_top`)."""
  shaped: bool = False
  """L-H2: the roll in the side grade, the level grade while lifted, M2's 3 cm goal term (`ROLL_STD` ...)."""
  level_goal: bool = False
  """L-H3 (with ``shaped``): the goal terms graded by the block's tilt instead of the hand's elevation."""


HOLD_CODES = {
  "L-H": HoldTask(),
  "L-HP": HoldTask(push=True),
  "L-H2": HoldTask(shaped=True),
  "L-H3": HoldTask(shaped=True, level_goal=True),
}


def held_block_spec(name: str = OBJECT) -> mujoco.MjSpec:
  """`scene.get_socket_block_spec` with the body's origin at the block's centre (every geom and site
  moved down by half its height): the convention of every picked object here."""
  spec = scene.get_socket_block_spec(name=name)
  half = scene.BLOCK_HEIGHT / 2
  body = spec.body(name)
  for geom in body.geoms:
    geom.pos = (geom.pos[0], geom.pos[1], geom.pos[2] - half)
  for site in body.sites:
    site.pos = (site.pos[0], site.pos[1], site.pos[2] - half)
  return spec


def held_block_cfg(name: str = OBJECT) -> EntityCfg:
  return EntityCfg(spec_fn=lambda: held_block_spec(name))


def block_geom_names(name: str = OBJECT) -> tuple[str, ...]:
  """Every geom of the block, for `DR-object`'s friction."""
  return tuple(f"{name}_{box[0]}" for box in scene.socket_block_boxes())


def _side_cfg(gripper: GripperCfg) -> SceneEntityCfg:
  return SceneEntityCfg(ROBOT, site_names=(gripper.tcp_site,))


def _hold_rewards(
  cfg: ManagerBasedRlEnvCfg, gripper: GripperCfg, task: HoldTask
) -> None:
  """RCTP's dict turned into L-H's (see the module docstring), L-H2's with ``task.shaped``. Every term not named
  here is dropped except the shared penalties (regularisers, table contact, effort hinge, trunk, arm-table)."""
  r = cfg.rewards
  side = _side_cfg(gripper)
  half = scene.BLOCK_HEIGHT / 2
  keep = {
    name: term
    for name, term in r.items()
    if name
    in (
      "action_rate_l2",
      "joint_pos_limits",
      "joint_vel_hinge",
      "table_contact",
      "arm_effort",
    )
    or name.startswith(("trunk", "arm_table"))
  }
  missing = {"grasp", "lift", "goal", "goal_near"} - set(r)
  if missing:
    raise ValueError(
      f"L-H is built on RCTP (M2 and M3); the reward has no {sorted(missing)}"
    )

  def graded(term: RewardTermCfg, **over) -> RewardTermCfg:
    roll = {"roll_std": ROLL_STD} if task.shaped else {}
    return RewardTermCfg(
      func=mdp.side_roll_graded if task.shaped else mdp.side_graded,
      weight=term.weight,
      params={
        "inner": term.func,
        "side_cfg": side,
        "side_std": SIDE_STD,
        **roll,
        **term.params,
        **over,
      },
    )

  goal_params = r["goal"].params
  cfg.rewards = {
    "reach": RewardTermCfg(
      func=mdp.side_reach,
      weight=1.0,
      params={
        "object_name": OBJECT,
        "asset_cfg": side,
        "approach_b": HOLD_APPROACH,
        "pad_height": PAD_HEIGHT,
        "half_height": half,
        "pad_offset": PAD_OFFSET + GRASP_DEPTH,
        "std": REACH_STD,
        "cone_std": REACH_CONE_STD,
        "face_cone": REACH_FACE_CONE,
      },
    ),
    "grasp": graded(r["grasp"]),
    "lift": graded(r["lift"]),
    "goal": graded(r["goal"]),
    "goal_near": graded(r["goal_near"], std=NEAR_STD),
    "level": RewardTermCfg(
      func=mdp.level_near_goal,
      weight=LEVEL_WEIGHT,
      params={
        "command_name": COMMAND,
        "object_name": OBJECT,
        "grasp_sensor_name": goal_params["grasp_sensor_name"],
        "near": LEVEL_NEAR,
        "std": LEVEL_STD,
      },
    ),
    "held": RewardTermCfg(
      func=mdp.held_at_goal, weight=HELD_WEIGHT, params=held_params(gripper)
    ),
    **keep,
  }
  if task.shaped:
    cfg.rewards["goal_mid"] = graded(r["goal_near"])
    cfg.rewards["level_lifted"] = RewardTermCfg(
      func=mdp.level_lifted,
      weight=LEVEL_LIFTED_WEIGHT,
      params={
        "object_name": OBJECT,
        "grasp_sensor_name": goal_params["grasp_sensor_name"],
        "surface_cfg": SceneEntityCfg(TABLE),
        "half_height": half,
        "clear": LEVEL_LIFTED_CLEAR,
        "std": LEVEL_LIFTED_STD,
      },
    )
  if task.level_goal:
    for name in LEVEL_GOAL_TERMS:
      term = cfg.rewards[name]
      inner = {
        k: v
        for k, v in term.params.items()
        if k not in ("inner", "side_cfg", "side_std", "roll_std")
      }
      cfg.rewards[name] = RewardTermCfg(
        func=mdp.level_block_graded,
        weight=term.weight,
        params={
          "inner": term.params["inner"],
          "level_object": OBJECT,
          "side_cfg": side,
          "tilt_std": LEVEL_GOAL_TILT_STD,
          "roll_std": ROLL_STD,
          "elevation_std": LEVEL_GOAL_ELEVATION_STD,
          **inner,
        },
      )


def held_params(gripper: GripperCfg) -> dict:
  """`mdp.held_at_goal`'s parameters for L-H (also the eval's "held at H")."""
  return {
    "command_name": COMMAND,
    "object_name": OBJECT,
    "grasp_sensor_name": gripper.contact_sensor_name,
    "side_cfg": _side_cfg(gripper),
    "approach_b": HOLD_APPROACH,
    "distance": HELD_DISTANCE,
    "tilt": HELD_TILT,
    "speed": HELD_SPEED,
    "elevation": HELD_ELEVATION,
    "cone": HELD_CONE,
  }


def _hold_observations(
  cfg: ManagerBasedRlEnvCfg, gripper: GripperCfg, task: HoldTask
) -> None:
  """The actor's block yaw as (cos 4 psi, sin 4 psi) and its up axis appended (40-D); the critic the same,
  live, plus the side grade and `held`."""
  actor = cfg.observations["actor"].terms
  lag = actor["object_pos"].params.get("lag")
  actor["object_yaw"] = dataclasses.replace(
    actor["object_yaw"],
    func=mdp.box_yaw_b,
    params={"box_name": OBJECT, "bias_key": OBJECT, "noise": BOX_YAW_NOISE, "lag": lag},
    noise=None,
  )
  actor["object_up"] = ObservationTermCfg(
    func=mdp.object_up_b,
    params={"object_name": OBJECT, "lag": lag},
    noise=Unoise(n_min=-UP_NOISE, n_max=UP_NOISE),
  )
  critic = cfg.observations["critic"].terms
  critic["object_yaw"] = dataclasses.replace(
    critic["object_yaw"], func=mdp.box_yaw_b, params={"box_name": OBJECT}
  )
  extra = {
    "object_up": ObservationTermCfg(
      func=mdp.object_up_b, params={"object_name": OBJECT}
    ),
    "side_grade": ObservationTermCfg(
      func=mdp.side_roll_grade_obs,
      params={"asset_cfg": _side_cfg(gripper), "std": SIDE_STD, "roll_std": ROLL_STD},
    )
    if task.shaped
    else ObservationTermCfg(
      func=mdp.side_grade_obs, params={"asset_cfg": _side_cfg(gripper), "std": SIDE_STD}
    ),
    "held": ObservationTermCfg(func=mdp.held_at_goal_obs, params=held_params(gripper)),
  }
  # The actor's order (the critic's first terms mirror it), then the critic's own.
  order = [*actor]
  cfg.observations["critic"].terms = {
    **{k: critic.get(k) or extra[k] for k in order},
    **{k: v for k, v in critic.items() if k not in order},
    **{k: v for k, v in extra.items() if k not in order},
  }


def apply_hold(
  cfg: ManagerBasedRlEnvCfg, gripper: GripperCfg, task: HoldTask
) -> ManagerBasedRlEnvCfg:
  """Turn a `BS12` one-arm cfg (built with the cube, RCTP, `TARGET=B`) into L-H / L-HP (module
  docstring). Raises on a cfg it was not built for."""
  if "placed" not in cfg.rewards or "goal_near" not in cfg.rewards:
    raise ValueError("L-H is built on RCTP (R-C-strict+M1+M2+M3+...)")
  if "release_grade" in cfg.observations["critic"].terms:
    raise ValueError("L-H has no release: CRITIC_OBS=q (PQ) does not apply; unset it")
  # The insert tasks' table (decision of plan §12, `scene.INSERT_TABLE_CONTYPE`): only the block's outer ring
  # stands on it, its inner boxes do not touch it.
  table = cfg.scene.entities[TABLE]
  top_z = table.init_state.pos[2] + scene.TABLE_THICKNESS / 2
  cfg.scene.entities = {
    **cfg.scene.entities,
    TABLE: scene.get_table_cfg(top_z, contype=scene.INSERT_TABLE_CONTYPE),
    OBJECT: held_block_cfg(OBJECT),
  }
  cx, cy = scene.TABLE_POS[0], scene.TABLE_POS[1]
  reset = cfg.events["reset_objects"]
  reset.params["pose_range"] = {
    "x": (HOLD_SPAWN_X[0] - cx, HOLD_SPAWN_X[1] - cx),
    "y": (HOLD_SPAWN_Y[0] - cy, HOLD_SPAWN_Y[1] - cy),
    "yaw": (-math.pi / 4, math.pi / 4),
  }
  reset.params.pop("goal_area", None)
  friction = cfg.events.get("object_friction")
  if friction is not None:
    friction.params["asset_cfg"] = SceneEntityCfg(
      OBJECT, geom_names=block_geom_names(OBJECT)
    )
  cfg.commands[COMMAND] = mdp.SurfaceOffsetCommandCfg(
    entity_name=OBJECT,
    surface_name=TABLE,
    offset=HOLD_XY,
    offset_range=(0.0, 0.0),
    frame="env",
    height=HOLD_DZ,
    resampling_time_range=(1e9, 1e9),
    success_threshold=HELD_DISTANCE,
    debug_vis=True,
  )
  _hold_rewards(cfg, gripper, task)
  _hold_observations(cfg, gripper, task)
  cfg.events["reset_tracker_yaw_bias"] = EventTermCfg(
    func=mdp.reset_tracker_yaw_bias,
    mode="reset",
    params={"half": BOX_YAW_BIAS, "keys": (OBJECT,)},
  )
  cfg.terminations.pop("success", None)
  if task.push:
    cfg.events["push_block"] = EventTermCfg(
      func=mdp.push_object_top,
      mode="step",
      params={
        "object_name": OBJECT,
        "start_s": PUSH_START_S,
        "down_n": PUSH_DOWN_N,
        "side_n": PUSH_SIDE_N,
        "duration_s": PUSH_DURATION_S,
        "gap_s": PUSH_GAP_S,
        "top": scene.BLOCK_HEIGHT / 2,
      },
    )
  return cfg
