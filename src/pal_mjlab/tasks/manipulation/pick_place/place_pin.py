"""Plan §14 (`pick-place-insert-plan.md` §14): ``GOAL=place`` -- pick the cross pin up anywhere on the wide pin area
`WP`, standing or lying, and put it down at a goal point drawn from `WP`, in any pose.

Built on `BS12` (plan §0.3) with the pin (`SHAPE=pin`) and RCTP (``R-C-strict+M1+M2+M3+M6+J+M8``, or ``...+GC`` for
PS-SG); §12's insert with the block, the hole and the upright taken out. Every change is made here
(`apply_place_goal`), so ``GOAL=place`` changes the same things whatever else is set:

* **Start** (`PlaceGoal.lie_probability`, ``PIN_START`` / ``PIN_LIE_P``): `events.reset_objects_on_surface` stands
  the pin on its foot on `WP` at any yaw, its centre outside `START_KEEPOUT` (the start pose's left and right
  fingers hang below a standing pin's top; plan §12's `insert.HOME_HAND_KEEPOUT` grown for `WP` and the DR-scene
  table); `mdp.reset_pin_start` then lays the given share down in the "x" pose, anywhere on `WP` (a lying pin passes
  under the fingers), the axis at any yaw.
* **Goal.** `mdp.PinPlaceCommand`: a point on the table drawn from `WP` at least `TARGET_MIN_SEPARATION` from the
  pin, outside `GOAL_KEEPOUT` (the right hand's); its ``command`` follows the pin's pose (`mdp.place_pin`), so M1 / J / M8's (xy, z) at-goal tests read "the
  centre within `PLACE_RADIUS` of the goal, the lowest point within z of the table" and M2's ``goal`` /
  ``goal_near`` pull (c_x, c_y, z_min) to the goal point (plan §14.2).
* **Reward.** RCTP with ``reach`` to the pose's grasp point (`mdp.pin_reach`: §12's for a standing pin, over the
  centre for a lying one) + the jaw-yaw term for a lying pin (`mdp.pin_jaw_yaw`), M3's lift from the lowest point
  (`mdp.pin_lift_faded`), ``release`` at the goal touching the table (`RELEASE_TOL`), per-step ``placed`` while the
  pin rests there (`PLACED_TOL`, `REST_SPEED`, `REST_ANGULAR_SPEED`), M6's grade = G(speed at the let-go,
  `M6_SPEED_STD`) x G(lowest point's height at the let-go, `M6_HEIGHT_STD`), M8's retreat to `RETREAT_CLEAR` over
  the pin's top (`mdp.pin_retreat_height`). GC (`mdp.grasp_closure`) with ``REWARD=...+GC``. No ``pin_down``
  termination: a knocked pin is still a pin to pick.
* **Observations (actor 42-D, §12's layout).** ``object_pos`` / ``object_yaw`` the pin's (tracked, biased; the yaw
  slot `mdp.pin_psi_b`: the yaw standing, the roll lying), ``target_pos`` the goal point on the table
  (`mdp.place_goal_b`), ``goal_yaw`` the constant (1, 0) in §12's block-yaw slot, ``pin_axis`` (`mdp.pin_axis_b`) in
  §12's tilt slot. A standing pin is observed exactly as in §12. The critic: the same live, + ``goal_yaw``,
  ``pin_axis`` and ``pin_rested`` in §12's three appended slots (46 with PQ).
"""

from __future__ import annotations

import dataclasses
import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from pal_mjlab.tasks.manipulation import mdp, scene
from pal_mjlab.tasks.manipulation.pick_place.box import (
  BOX_YAW_BIAS,
  BOX_YAW_NOISE,
  TRACKER_OWN_BIAS,
  TRACKER_SHARED_BIAS,
)
from pal_mjlab.tasks.manipulation.pick_place.insert import (
  GRASP_ABOVE_TOP,
  INSERT_REWARD,
  PIN_TILT_NOISE,
)
from pal_mjlab.tasks.manipulation.robot_cfg import GripperCfg
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import (
  _DR_OBS_OBJECT_POS_NOISE,
  COMMAND,
  OBJECT,
  TABLE,
)

#: RCTP, the reward ``GOAL=place`` edits (§12's), and PS-SG's RCTP + GC.
PLACE_REWARD = INSERT_REWARD
PLACE_REWARD_GC = f"{INSERT_REWARD}+GC"
PLACE_REWARDS = (PLACE_REWARD, PLACE_REWARD_GC)
#: Plan §14's success radius on the pin's centre (D14-3), m.
PLACE_RADIUS = 0.02
#: M1's ``release`` at-goal test (xy, z_min over the table) at the step the jaw opens: touching the table.
RELEASE_TOL = (PLACE_RADIUS, 0.005)
#: ``placed`` / ``rested``: the lowest point within 3 mm of the table, slower than `REST_SPEED` (m/s) and turning
#: slower than `REST_ANGULAR_SPEED` (rad/s).
PLACED_TOL = (PLACE_RADIUS, 0.003)
REST_SPEED = 0.02
REST_ANGULAR_SPEED = 0.5
#: M6 for the pin: the let-go's speed (m/s) and lowest-point height (m) graded; xy only gated (the release's at-goal
#: test), the tilt not graded (any pose counts).
M6_SPEED_STD = 0.05
M6_HEIGHT_STD = 0.005
M6_DIST_STD = 1.0
#: The tool point over a lying pin's highest point at the grasp, m (plan §14.2's ``GRASP_ABOVE_TOP_LYING``).
GRASP_ABOVE_TOP_LYING = 0.020
#: The jaw-yaw term (lying pins): weight, the angle's width, the tool point's distance width.
JAW_YAW_WEIGHT, JAW_YAW_STD, JAW_YAW_DIST_STD = 1.0, math.radians(20.0), 0.03
#: GC (PS-SG; its weight `rewards.GC_WEIGHT`): the first-touch jaw angles (rad; standing: §12.1's bench,
#: 0.546-0.587), the gate (m).
GC_Q_TOUCH = 0.55
GC_Q_TOUCH_LYING = 0.55
GC_XY_TOL, GC_Z_TOL = 0.015, 0.01
#: M8's retreat: to 10 cm over the pin's top.
RETREAT_CLEAR = 0.10
#: M3's fade width.
LIFT_FADE_STD = 0.05
#: Where a standing pin's centre may not start, and a goal may not be drawn (x lo, x hi, y lo, y hi, the robot frame):
#: within 1 cm of the start pose's fingers at the highest DR-scene table (0.847 m), on a 1 cm grid, the pin's
#: circumradius (16.5 mm) and 90 mm height included -- step 0 item 4 (`results/S14-pin-place/step0/home_clearance_wp.py`,
#: 2026-10-08): the left hand x 0.32-0.48, y 0.07-0.21 (plan §12's `insert.HOME_HAND_KEEPOUT`, measured at 0.837 m, is
#: x 0.34-0.47, y 0.07-0.18), the right hand (§12's default home) x 0.33-0.47, y -0.14 to -0.09; half a cell added.
#: A lying pin (28 mm tall) clears both by >= 39 mm everywhere and may start under them. The hand-off's elbow park
#: keeps the left hand >= 85 mm from a standing pin anywhere; the right hand stays where it started all episode, so a
#: goal there would have the pin set down against its fingers.
LEFT_HAND_KEEPOUT = (0.315, 0.485, 0.065, 0.215)
RIGHT_HAND_KEEPOUT = (0.325, 0.475, -0.20, -0.085)
START_KEEPOUT = (LEFT_HAND_KEEPOUT, RIGHT_HAND_KEEPOUT)
GOAL_KEEPOUT = RIGHT_HAND_KEEPOUT
#: The share of lying starts for ``PIN_START=mix`` (D14-6; 50 / 50 proposed).
PIN_LIE_P = 0.5
#: ``PIN_START``: the lying share each name gives (mix: `PIN_LIE_P` unless ``PIN_LIE_P`` is set).
PIN_STARTS = {"mix": PIN_LIE_P, "stand": 0.0, "lie": 1.0}


@dataclasses.dataclass(frozen=True)
class PlaceGoal:
  """What ``GOAL=place`` builds (`apply_place_goal`)."""

  lie_probability: float = PIN_LIE_P
  """The share of episodes whose pin starts lying (``PIN_START`` / ``PIN_LIE_P``)."""
  above_top_lying: float = GRASP_ABOVE_TOP_LYING
  start_keepout: tuple[tuple[float, float, float, float], ...] = START_KEEPOUT
  goal_keepout: tuple[float, float, float, float] | None = GOAL_KEEPOUT
  jaw_yaw_weight: float = JAW_YAW_WEIGHT
  """The jaw-yaw term's weight; 0 drops it (D14-4 may)."""


#: The ``GOAL`` place codes.
PLACE_GOALS: dict[str, PlaceGoal] = {"place": PlaceGoal()}


def apply_place_goal(
  cfg: ManagerBasedRlEnvCfg,
  gripper: GripperCfg,
  code: PlaceGoal,
  area: tuple[tuple[float, float], tuple[float, float]],
  threshold: float,
  pads_cfg: SceneEntityCfg,
) -> None:
  """Turn a built RCTP pick-and-place cfg with the pin (`SHAPE=pin`) on a target area into plan §14's place, in
  place (the module docstring). ``area`` ((x lo, x hi), (y lo, y hi), env frame for the table at its nominal place)
  is where the pin and the goal are drawn; ``threshold`` the goal tolerance the command logs success with; ``pads``
  the jaw's two pad sites (the jaw-yaw term's closing direction)."""
  missing = [
    t
    for t in ("reach", "grasp", "lift", "goal", "release", "placed", "retreat")
    if t not in cfg.rewards
  ]
  if missing:
    raise ValueError(
      f"GOAL=place edits {PLACE_REWARD}'s task terms; this reward lacks {missing}"
    )
  table_cx, table_cy = scene.TABLE_POS[0], scene.TABLE_POS[1]
  (x0, x1), (y0, y1) = area
  rel_x, rel_y = (x0 - table_cx, x1 - table_cx), (y0 - table_cy, y1 - table_cy)

  # -- Start and goal. ----------------------------------------------------------------------------
  cfg.events["reset_objects"].params["keepout"] = code.start_keepout
  start = EventTermCfg(
    func=mdp.reset_pin_start,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg(OBJECT),
      "surface_cfg": SceneEntityCfg(TABLE),
      "lie_probability": code.lie_probability,
      "area_x": rel_x,
      "area_y": rel_y,
    },
  )
  # Right after `reset_objects` (the dict keeps insertion order), before anything that reads the pin's pose.
  events = {}
  for name, term in cfg.events.items():
    events[name] = term
    if name == "reset_objects":
      events["reset_pin_start"] = start
  cfg.events = events
  cfg.events["reset_tracker_bias"] = EventTermCfg(
    func=mdp.reset_tracker_bias,
    mode="reset",
    params={"shared": TRACKER_SHARED_BIAS, "own": TRACKER_OWN_BIAS, "keys": (OBJECT,)},
  )
  cfg.events["reset_tracker_yaw_bias"] = EventTermCfg(
    func=mdp.reset_tracker_yaw_bias,
    mode="reset",
    params={"half": BOX_YAW_BIAS, "keys": (OBJECT,)},
  )
  old = cfg.commands[COMMAND]
  if not isinstance(old, mdp.SurfaceOffsetCommandCfg) or old.min_object_distance <= 0:
    raise ValueError("GOAL=place draws its goal from a target area (TARGET=WP)")
  cfg.commands[COMMAND] = mdp.PinPlaceCommandCfg(
    entity_name=OBJECT,
    surface_name=old.surface_name,
    offset=old.offset,
    offset_range=old.offset_range,
    min_object_distance=old.min_object_distance,
    goal_keepout=code.goal_keepout,
    resampling_time_range=(1e9, 1e9),
    success_threshold=threshold,
    debug_vis=True,
  )

  # -- Observations. ---------------------------------------------------------------------------
  actor = cfg.observations["actor"].terms
  obj_term = actor["object_pos"]
  if obj_term.func is not mdp.object_position_b:
    raise ValueError(
      "GOAL=place replaces the actor's object_pos term; it is "
      f"{obj_term.func.__name__} (occlusion is not built for the pin)"
    )
  lag = obj_term.params.get("lag")
  pos_noise = Unoise(n_min=-_DR_OBS_OBJECT_POS_NOISE, n_max=_DR_OBS_OBJECT_POS_NOISE)
  actor["object_pos"] = dataclasses.replace(
    obj_term,
    func=mdp.object_position_biased_b,
    params={**obj_term.params, "bias_key": OBJECT},
    noise=pos_noise,
  )
  actor["object_yaw"] = dataclasses.replace(
    actor["object_yaw"],
    func=mdp.pin_psi_b,
    params={
      "object_name": OBJECT,
      "bias_key": OBJECT,
      "noise": BOX_YAW_NOISE,
      "lag": lag,
    },
    noise=None,
  )
  actor["target_pos"] = ObservationTermCfg(
    func=mdp.place_goal_b, params={"command_name": COMMAND}
  )
  actor["goal_yaw"] = ObservationTermCfg(func=mdp.no_yaw)
  actor["pin_axis"] = ObservationTermCfg(
    func=mdp.pin_axis_b,
    params={"object_name": OBJECT, "noise": PIN_TILT_NOISE, "lag": lag},
  )
  critic = cfg.observations["critic"].terms
  critic["object_yaw"] = ObservationTermCfg(
    func=mdp.pin_psi_b, params={"object_name": OBJECT}
  )
  critic["target_pos"] = ObservationTermCfg(
    func=mdp.place_goal_b, params={"command_name": COMMAND}
  )

  # -- Rewards. ---------------------------------------------------------------------------------
  rw = cfg.rewards
  tcp = rw["reach"].params["asset_cfg"]
  tops = {"above_top": GRASP_ABOVE_TOP, "above_top_lying": code.above_top_lying}
  rw["reach"] = RewardTermCfg(
    func=mdp.pin_reach,
    weight=rw["reach"].weight,
    params={
      "object_name": OBJECT,
      **tops,
      "reaching_std": rw["reach"].params["reaching_std"],
      "asset_cfg": tcp,
    },
  )
  if code.jaw_yaw_weight:
    rw["jaw_yaw"] = RewardTermCfg(
      func=mdp.pin_jaw_yaw,
      weight=code.jaw_yaw_weight,
      params={
        "object_name": OBJECT,
        "pads_cfg": pads_cfg,
        "asset_cfg": tcp,
        **tops,
        "angle_std": JAW_YAW_STD,
        "dist_std": JAW_YAW_DIST_STD,
      },
    )
  lift = rw["lift"]
  rw["lift"] = RewardTermCfg(
    func=mdp.pin_lift_faded,
    weight=lift.weight,
    params={
      "grasp_sensor_name": gripper.contact_sensor_name,
      "object_name": OBJECT,
      "command_name": COMMAND,
      "surface_cfg": lift.params["surface_cfg"],
      "max_height": lift.params["max_height"],
      "fade_std": LIFT_FADE_STD,
    },
  )
  m6 = {
    "dist_std": M6_DIST_STD,
    "height_free": 0.0,
    "height_std": M6_HEIGHT_STD,
    "speed_std": M6_SPEED_STD,
  }
  release = rw["release"]
  rw["release"] = RewardTermCfg(
    func=release.func,
    weight=release.weight,
    params={**release.params, **m6, "threshold": RELEASE_TOL},
  )
  placed = rw["placed"]
  placed_params = {
    **placed.params,
    **m6,
    "threshold": PLACED_TOL,
    "velocity_threshold": REST_SPEED,
  }
  rw["placed"] = RewardTermCfg(
    func=mdp.pin_placed_graded,
    weight=placed.weight,
    params={**placed_params, "angular_speed": REST_ANGULAR_SPEED},
  )
  retreat = rw["retreat"].params
  rw["retreat"] = RewardTermCfg(
    func=mdp.pin_retreat_height,
    weight=rw["retreat"].weight,
    params={
      "command_name": retreat["command_name"],
      "object_name": retreat["object_name"],
      "grasp_sensor_name": retreat["grasp_sensor_name"],
      "support_sensor_name": retreat["support_sensor_name"],
      "jaw_cfg": retreat["jaw_cfg"],
      "jaw_open": retreat["jaw_open"],
      "asset_cfg": retreat["asset_cfg"],
      "clear_height": RETREAT_CLEAR,
      "threshold": PLACED_TOL,
      "velocity_threshold": REST_SPEED,
    },
  )
  if "grasp_closure" in rw:
    gc = rw["grasp_closure"]
    rw["grasp_closure"] = RewardTermCfg(
      func=gc.func,
      weight=gc.weight,
      params={
        **gc.params,
        "object_name": OBJECT,
        **tops,
        "q_touch": GC_Q_TOUCH,
        "q_touch_lying": GC_Q_TOUCH_LYING,
        "xy_tol": GC_XY_TOL,
        "z_tol": GC_Z_TOL,
      },
    )


def append_critic_place(cfg: ManagerBasedRlEnvCfg, grasp_sensor: str) -> None:
  """The critic's place inputs in §12's three appended slots, after PQ's: the goal's (1, 0) yaw, the pin's live
  canonical axis, and ``rested`` (`mdp.pin_rested`)."""
  critic = cfg.observations["critic"]
  critic.terms = {
    **critic.terms,
    "goal_yaw": ObservationTermCfg(func=mdp.no_yaw),
    "pin_axis": ObservationTermCfg(func=mdp.pin_axis_b, params={"object_name": OBJECT}),
    "pin_rested": ObservationTermCfg(
      func=mdp.pin_rested,
      params={
        "object_name": OBJECT,
        "command_name": COMMAND,
        "grasp_sensor_name": grasp_sensor,
        "radius": PLACE_RADIUS,
        "height": PLACED_TOL[1],
        "speed": REST_SPEED,
        "angular_speed": REST_ANGULAR_SPEED,
      },
    ),
  }
