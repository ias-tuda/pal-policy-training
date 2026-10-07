"""Plan §12 (`pick-place-insert-plan.md`, §12 / §12.1): ``GOAL=insert`` -- push the cross pin through the socket block.

The pin (`SHAPE=pin`, `scene.get_pin_spec`) is picked from the wide area `WC` at any yaw and pushed through the
socket block (`scene.get_socket_block_spec`), drawn from the same area at any yaw, until its bottom stands on the
table; then the jaw opens and the hand leaves. Built on `BS12` (plan §0.3): S10.1's box-on-WC machinery with the
block in the box's place, on RCTP (``R-C-strict+M1+M2+M3+M6+J+M8``, the only reward this goal edits). Every change
is made here, so a ``GOAL`` code changes the same things whatever else is set:

* **Scene.** The block (`BLOCK`), drawn by `mdp.reset_box_in_area` at least `PIN_MIN_SEPARATION` from the pin at a
  yaw in +/- 45 deg (the pin is drawn only where such a block fits, `pick_place_env_cfg`'s ``goal_area``); the table
  rebuilt with `scene.INSERT_TABLE_CONTYPE`, so only the block's outer ring stands on it (`scene`'s collision bits).
* **Goal.** `mdp.InsertGoalCommand`: the pin centre's seated position, following the block. M1 / J / M6 / M8's
  at-goal test is (xy, z) = `AT_GOAL_TOL` on the pin's centre -- with the bottom within 3 mm of the floor and the
  centre within 8 mm of the axis the pin is in the Ø 38 bore (the eval reads plan §12's exact ``seated``,
  `mdp.InsertState`). M6's grade is the pin's tilt at the let-go, G(tilt, 5 deg), paid only for a let-go with the
  pin seated (its bottom within 3 mm of the floor: `M6_HEIGHT`), the floor otherwise (plan §12 *Reward*).
* **Reward (IN, IN-T, IN-NA).** RCTP with ``reach`` to the pin's grasp point (`GRASP_ABOVE_TOP` over its top:
  the fingers clear the block when it is seated, step 0 item 2), ``goal`` / ``goal_near`` on the approach to the
  hole's entry (`mdp.insert_approach`, M2's weights and widths), M3's lift faded by that approach, M8's retreat to
  10 cm, plus A (`ALIGN_WEIGHT`), D (`DEPTH_WEIGHT`), U (`UPRIGHT_WEIGHT`) and S10's ``block_moved``. IN-NA drops A;
  IN-T is IN in the tight Ø 35 hole. L5's ordering is checked in `tests/test_pick_place_s12.py`.
* **Reward (IN-LAD).** RCTP's task terms replaced by the stage ladder `mdp.insert_ladder` (weight
  `mdp.insert.LADDER_SCALE`); ``release`` / ``placed`` stay at weight 0 for PQ's critic inputs and the eval;
  penalties (M8's pad contact, ``block_moved``, AH10, the effort hinge, the regularisers) as IN.
* **Observations (actor 42-D on `BS2`).** S10.1's ``BOX_OBS=pose`` layout with the pin in the cube's slots and the
  block in the box's: ``object_pos`` the pin's centre and ``object_yaw`` its (cos 4 psi, sin 4 psi), through the
  pin's tracker; ``target_pos`` / ``box_yaw`` the block's; ``pin_tilt`` (`mdp.pin_tilt_b`) appended last. Both
  poses carry S10's 6 mm shared + 4 mm own bias and a yaw bias. The critic: the same live, + the block's yaw, the
  pin's tilt and ``seated``, after PQ's inputs (a box checkpoint widens by appended columns only).
"""

from __future__ import annotations

import dataclasses
import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from pal_mjlab.tasks.manipulation import mdp, scene
from pal_mjlab.tasks.manipulation.mdp.insert import LADDER_SCALE
from pal_mjlab.tasks.manipulation.pick_place.box import (
  BOX_YAW_BIAS,
  BOX_YAW_NOISE,
  TRACKER_OWN_BIAS,
  TRACKER_SHARED_BIAS,
)
from pal_mjlab.tasks.manipulation.robot_cfg import GripperCfg
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import (
  _DR_OBS_OBJECT_POS_NOISE,
  COMMAND,
  OBJECT,
  ROBOT,
  SUPPORT_SENSOR,
  TABLE,
)

#: The socket block's scene entity name.
BLOCK = "block"
#: The block's yaw on the area: every orientation of the 4-fold symmetric block, rad.
BLOCK_AREA_YAW = math.pi / 4
#: The least xy distance between the pin's and the block's centres, m. Plan §12.1 proposed 12 cm; step 0 item 4
#: (`scripts/s12_insert_step0.py separation`, devbox 2026-10-07: the scripted grasp next to the block, 8 directions x
#: block yaw 0 / 45 x pin yaw 0 / 45) touched the block in 4 / 32 grasps at 8 cm, 1 / 32 at 10, 0 / 32 at 12 and 14 cm;
#: S10.1's rule, the least clean distance + 2 cm, gives 14.
PIN_MIN_SEPARATION = 0.14
#: Where neither the pin's nor the block's centre may be drawn (x lo, x hi, y lo, y hi, m, the robot frame): under the
#: start pose's left fingers, which hang 78-90 mm over the 0.837 m table at x 0.36-0.45, y 0.09-0.16 (step 0,
#: `results/S12-insert/step0/home_hand.py`) -- below a standing pin's 90 mm top, which they hit at the reset, and
#: below a seated pin's, which the hand-off's return home hits -- plus the pin's 16 mm reach and 4 mm. 13 % of WC.
HOME_HAND_KEEPOUT = (0.34, 0.47, 0.07, 0.18)
#: The tool point's height over the pin's top at the grasp, m: the fingertips reach 44 mm below the tool point at
#: the pin's hold angle, so 15.5 mm over the top keeps them 5 mm over the block's top when the pin is seated
#: (step 0 item 2, `results/S12-insert/step0/geometry/seated_clearance.json`); 20 mm leaves 2.5 mm of margin.
GRASP_ABOVE_TOP = 0.020
#: The at-goal (xy, |dz|) tolerance on the pin's centre that stands for seated in M1 / J / M6 / M8, m.
AT_GOAL_TOL = (0.008, 0.003)
#: M6's grade for the insert: the let-go's tilt (rad), the xy not graded, the let-go seated (height above
#: the seated centre free to `M6_HEIGHT`, then falling with that width).
M6_TILT_STD = math.radians(5.0)
M6_DIST_STD = 1.0
M6_HEIGHT = 0.003
#: M2's widths and weights on the approach (`mdp.insert_approach`), and M3's fade width.
APPROACH_STD, APPROACH_WEIGHT = 0.15, 7.0
NEAR_STD, NEAR_WEIGHT = 0.03, 3.0
LIFT_FADE_STD = 0.05
#: A, D, U (plan §12 *Reward*). Weights: D 4 -- the plan's "~ 2 x goal_near's peak" (6) would leave a seated,
#: still-held pin (12 + A + U + D = 21 per step) within 1 of a seated pin let go at a 3 deg tilt (q 0.78:
#: 10 + 12 q + 2 = 21.4); at 4 the let-go out-earns the hold by >= 3 (L5, `tests/test_pick_place_s12.py`).
ALIGN_WEIGHT, ALIGN_STD, ALIGN_WINDOW = 2.0, 0.003, 0.03
DEPTH_WEIGHT, DEPTH_AXIS_GATE = 4.0, 0.005
UPRIGHT_WEIGHT, UPRIGHT_STD, UPRIGHT_NEAR = 1.0, math.radians(8.0), 0.05
#: M8's retreat for the insert: to 10 cm over the pin's top (the pin stands 34 mm proud, L9).
RETREAT_CLEAR = 0.10
#: The actor's pin tilt jitter, per component (~1 deg), and the pin's yaw bias and jitter (the box's).
PIN_TILT_NOISE = 0.02
#: LAD-IN's L4 shaping opens the jaw from the pin's hold angle (the bench: 0.56 rad face on, 0.60 at 45 deg).
INSERT_JAW_HOLD = 0.60
#: Sensors of the block against the hand (the `arm_*_7_link` subtree) and against the pin, for the eval.
BLOCK_HAND_SENSOR = "block_hand_contact"
BLOCK_PIN_SENSOR = "block_pin_contact"
#: RCTP, the reward every insert code edits.
INSERT_REWARD = "R-C-strict+M1+M2+M3+M6+J+M8"
#: RCTP's task terms, which the insert rewrites or (IN-LAD) replaces.
_TASK_TERMS = (
  "reach",
  "grasp",
  "lift",
  "goal",
  "goal_near",
  "release",
  "placed",
  "retreat",
  "pad_contact",
)


@dataclasses.dataclass(frozen=True)
class InsertGoal:
  """What a ``GOAL`` insert code builds (`apply_insert_goal`)."""

  hole_d: float = scene.BLOCK_HOLE_D
  chamfer: tuple[float, float] = scene.BLOCK_CHAMFER
  align_weight: float = ALIGN_WEIGHT
  """A's weight; 0 drops it (IN-NA)."""
  ladder: bool = False
  """IN-LAD: the task terms are the stage ladder `mdp.insert_ladder`."""

  @property
  def geometry(self) -> dict:
    """The block numbers every insert term reads."""
    return {"block_height": scene.BLOCK_HEIGHT, "hole_d": self.hole_d}


#: The ``GOAL`` insert codes (plan §12.1's batch: IN, IN-FT -- IN resumed --, IN-T, IN-NA, IN-LAD).
INSERT_GOALS: dict[str, InsertGoal] = {
  "insert": InsertGoal(),
  "insert-t": InsertGoal(
    hole_d=scene.BLOCK_HOLE_D_TIGHT, chamfer=scene.BLOCK_CHAMFER_TIGHT
  ),
  "insert-na": InsertGoal(align_weight=0.0),
  "insert-lad": InsertGoal(ladder=True),
}


def _contact_sensor(name: str, secondary: ContactMatch) -> ContactSensorCfg:
  return ContactSensorCfg(
    name=name,
    primary=ContactMatch(mode="body", pattern=BLOCK, entity=BLOCK),
    secondary=secondary,
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=1,
  )


def apply_insert_goal(
  cfg: ManagerBasedRlEnvCfg,
  gripper: GripperCfg,
  code: InsertGoal,
  area: tuple[tuple[float, float], tuple[float, float]],
  table_top_z: float,
  threshold: float,
) -> None:
  """Turn a built RCTP pick-and-place cfg with the pin (`SHAPE=pin`) into plan §12's insert, in place (the module
  docstring). ``area`` ((x lo, x hi), (y lo, y hi), env frame for the table at its nominal place) is where the block
  is drawn; ``table_top_z`` the baseline's table; ``threshold`` the goal tolerance the command logs success with."""
  missing = [t for t in _TASK_TERMS if t not in cfg.rewards]
  if missing:
    raise ValueError(
      f"GOAL=insert edits {INSERT_REWARD}'s task terms; this reward lacks {missing}"
    )
  geo = code.geometry
  scene_cfg = cfg.scene
  scene_cfg.entities = {
    **scene_cfg.entities,
    TABLE: scene.get_table_cfg(table_top_z, contype=scene.INSERT_TABLE_CONTYPE),
    BLOCK: scene.get_socket_block_cfg(
      name=BLOCK, hole_d=code.hole_d, chamfer=code.chamfer
    ),
  }
  table_cx, table_cy = scene.TABLE_POS[0], scene.TABLE_POS[1]
  (x0, x1), (y0, y1) = area
  # After `reset_objects` (the dict keeps insertion order): it reads where the pin now is.
  cfg.events["reset_block"] = EventTermCfg(
    func=mdp.reset_box_in_area,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg(BLOCK),
      "surface_cfg": SceneEntityCfg(TABLE),
      "object_name": OBJECT,
      "area_x": (x0 - table_cx, x1 - table_cx),
      "area_y": (y0 - table_cy, y1 - table_cy),
      "yaw": BLOCK_AREA_YAW,
      "min_distance": PIN_MIN_SEPARATION,
      "keepout": HOME_HAND_KEEPOUT,
    },
  )
  cfg.events["reset_objects"].params["keepout"] = HOME_HAND_KEEPOUT
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
  cfg.commands[COMMAND] = mdp.InsertGoalCommandCfg(
    entity_name=OBJECT,
    block_name=BLOCK,
    difficulty="dynamic",
    resampling_time_range=(1e9, 1e9),
    success_threshold=threshold,
    debug_vis=True,
  )

  # -- Observations. ---------------------------------------------------------------------------
  actor = cfg.observations["actor"].terms
  obj_term = actor["object_pos"]
  if obj_term.func is not mdp.object_position_b:
    raise ValueError(
      "GOAL=insert replaces the actor's object_pos term; it is "
      f"{obj_term.func.__name__} (occlusion is not built for the insert)"
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
    func=mdp.box_yaw_b,
    params={
      "box_name": OBJECT,
      "bias_key": OBJECT,
      "noise": BOX_YAW_NOISE,
      "lag": lag,
    },
    noise=None,
  )
  actor["target_pos"] = ObservationTermCfg(
    func=mdp.object_position_biased_b,
    params={"object_name": BLOCK, "bias_key": BLOCK, "lag": lag},
    noise=pos_noise,
  )
  actor["box_yaw"] = ObservationTermCfg(
    func=mdp.box_yaw_b,
    params={"box_name": BLOCK, "bias_key": BLOCK, "noise": BOX_YAW_NOISE, "lag": lag},
  )
  actor["pin_tilt"] = ObservationTermCfg(
    func=mdp.pin_tilt_b,
    params={"object_name": OBJECT, "noise": PIN_TILT_NOISE, "lag": lag},
  )
  critic = cfg.observations["critic"].terms
  critic["object_yaw"] = ObservationTermCfg(
    func=mdp.box_yaw_b, params={"box_name": OBJECT}
  )
  critic["target_pos"] = ObservationTermCfg(
    func=mdp.object_position_b, params={"object_name": BLOCK}
  )

  # -- Rewards. ---------------------------------------------------------------------------------
  rw = cfg.rewards
  lifted = {
    "grasp_sensor_name": gripper.contact_sensor_name,
    "support_sensor_name": SUPPORT_SENSOR,
  }
  pin = {"object_name": OBJECT, "block_name": BLOCK, **geo}
  m6 = {
    "threshold": AT_GOAL_TOL,
    "dist_std": M6_DIST_STD,
    "height_free": M6_HEIGHT,
    "height_std": M6_HEIGHT,
    "tilt_std": M6_TILT_STD,
  }
  for name in ("release", "placed"):
    term = rw[name]
    rw[name] = RewardTermCfg(
      func=term.func,
      weight=0.0 if code.ladder else term.weight,
      params={**term.params, **m6},
    )
  tcp = rw["reach"].params["asset_cfg"]
  if code.ladder:
    jaw = rw["release"].params
    for name in ("reach", "grasp", "lift", "goal", "goal_near", "retreat"):
      rw.pop(name)
    rw["ladder"] = RewardTermCfg(
      func=mdp.insert_ladder,
      weight=LADDER_SCALE,
      params={
        **pin,
        **lifted,
        "tcp_cfg": tcp,
        "surface_cfg": SceneEntityCfg(TABLE),
        "jaw_cfg": jaw["jaw_cfg"],
        "jaw_open": jaw["jaw_open"],
        "jaw_hold": INSERT_JAW_HOLD,
        "above_top": GRASP_ABOVE_TOP,
        "approach_std": APPROACH_STD,
        "align_std": ALIGN_STD,
        "upright_std": UPRIGHT_STD,
        "axis_gate": DEPTH_AXIS_GATE,
      },
    )
  else:
    rw["reach"] = RewardTermCfg(
      func=mdp.insert_reach,
      weight=rw["reach"].weight,
      params={
        "object_name": OBJECT,
        "above_top": GRASP_ABOVE_TOP,
        "reaching_std": rw["reach"].params["reaching_std"],
        "asset_cfg": tcp,
      },
    )
    lift = rw["lift"]
    rw["lift"] = RewardTermCfg(
      func=mdp.insert_lift_faded,
      weight=lift.weight,
      params={
        **pin,
        "grasp_sensor_name": gripper.contact_sensor_name,
        "surface_cfg": lift.params["surface_cfg"],
        "max_height": lift.params["max_height"],
        "fade_std": LIFT_FADE_STD,
      },
    )
    for name, std, weight in (
      ("goal", APPROACH_STD, APPROACH_WEIGHT),
      ("goal_near", NEAR_STD, NEAR_WEIGHT),
    ):
      rw[name] = RewardTermCfg(
        func=mdp.insert_approach,
        weight=weight,
        params={**pin, **lifted, "std": std},
      )
    retreat = rw["retreat"]
    rw["retreat"] = RewardTermCfg(
      func=retreat.func,
      weight=retreat.weight,
      params={
        **retreat.params,
        "threshold": AT_GOAL_TOL,
        "clear_height": RETREAT_CLEAR,
      },
    )
    if code.align_weight:
      rw["align"] = RewardTermCfg(
        func=mdp.insert_align,
        weight=code.align_weight,
        params={**pin, **lifted, "std": ALIGN_STD, "window": ALIGN_WINDOW},
      )
    rw["depth"] = RewardTermCfg(
      func=mdp.insert_depth,
      weight=DEPTH_WEIGHT,
      params={**pin, **lifted, "axis_gate": DEPTH_AXIS_GATE},
    )
    rw["upright"] = RewardTermCfg(
      func=mdp.insert_upright,
      weight=UPRIGHT_WEIGHT,
      params={**pin, **lifted, "std": UPRIGHT_STD, "near": UPRIGHT_NEAR},
    )
  rw["block_moved"] = RewardTermCfg(
    func=mdp.box_displacement_penalty,
    weight=-2.0,
    params={"command_name": COMMAND, "scale": 0.02},
  )

  cfg.scene.sensors = (
    *cfg.scene.sensors,
    _contact_sensor(
      BLOCK_HAND_SENSOR,
      ContactMatch(mode="subtree", pattern=gripper.hand_body, entity=ROBOT),
    ),
    _contact_sensor(
      BLOCK_PIN_SENSOR, ContactMatch(mode="body", pattern=OBJECT, entity=OBJECT)
    ),
  )
  cfg.terminations["block_dropped"] = TerminationTermCfg(
    func=mdp.entity_below_height,
    params={"entity_name": BLOCK, "min_height": table_top_z - 0.1},
  )


def append_critic_insert(cfg: ManagerBasedRlEnvCfg, code: InsertGoal) -> None:
  """The critic's insert inputs, appended after every other critic term -- PQ's included -- so that a resume from
  a box checkpoint widens the critic by appended columns only (`runner.widen_input`): the block's live yaw (cos 4
  psi, sin 4 psi; the box's slot), the pin's live tilt, and ``seated``."""
  critic = cfg.observations["critic"]
  critic.terms = {
    **critic.terms,
    "box_yaw": ObservationTermCfg(func=mdp.box_yaw_b, params={"box_name": BLOCK}),
    "pin_tilt": ObservationTermCfg(func=mdp.pin_tilt_b, params={"object_name": OBJECT}),
    "pin_seated": ObservationTermCfg(
      func=mdp.pin_seated,
      params={"object_name": OBJECT, "block_name": BLOCK, **code.geometry},
    ),
  }
