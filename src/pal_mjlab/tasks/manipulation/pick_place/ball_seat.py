"""Plan §15 (`pick-place-insert-plan.md` §15): ``GOAL=ball-seat`` -- pick a Ø 50 mm ball anywhere on `WC` and drop it
into the entry chamfer of plan §12's socket block (anywhere on `WC`, any yaw), so that it comes to rest on the hole's
axis; let go, lift clear.

Built on `BS12` (plan §0.3) with the ball (``SHAPE=ball50`` / ``ball50-dr``) and RCTP (``R-C-strict+M1+M2+M3+M6+J+M8``,
the only reward this goal edits); §12's block and its machinery unchanged. Every change is made here
(`apply_ball_seat_goal`), so ``GOAL=ball-seat`` changes the same things whatever else is set:

* **Scene.** §12's block (`insert.BLOCK`, same geoms, friction, mass), drawn by `mdp.reset_box_in_area` from the area at
  any yaw, at least `BALL_MIN_SEPARATION` from the ball, its centre outside `BLOCK_KEEPOUT` (the start-pose left hand:
  a ball seated on the block is 89 mm tall, the fingers hang 78-90 mm over the table); the ball anywhere on the area
  (it is 50 mm tall: it passes under the fingers). The table with `scene.INSERT_TABLE_CONTYPE` (§12's).
* **Goal.** `mdp.BallSeatCommand`: the seated ball's centre, following the block; M1 / J / M8's (xy, z) at-goal tests
  read ``seated`` through `mdp.ball_seat.SEAT_TOL`; the release latch measures a let-go's height over the seat.
* **Reward.** RCTP with M2's ``goal`` / ``goal_near`` pulling the ball's centre to the drop point `DROP_HEIGHT` over the
  seat with §13.1's clearance gate (`mdp.seat_approach`); the over-the-seat grade ``A_B`` (`mdp.seat_over_grade`);
  ``release`` paid only for a let-go inside `ENVELOPE` (`mdp.seat_release`); ``placed`` per step while ``seated_b``
  (`SEAT_SPEED`) after a J-release; M6's grade G(xy at the let-go, `M6_DIST_STD`) x G(speed, `M6_SPEED_STD`), the height
  free inside the envelope; M8's retreat to `RETREAT_CLEAR` over the ball's top; §12's ``block_moved``. No termination
  when the ball rolls off the block (§14's rule); §12's ``block_dropped`` stays.
* **Observations (actor 42-D, §12's layout).** ``object_pos`` the ball's centre (tracked, biased), ``object_yaw`` the
  constant (1, 0) (S8), ``target_pos`` / ``box_yaw`` the block's position and (cos 4 psi, sin 4 psi) (§12's slots),
  ``pin_tilt`` the constant (0, 0). The critic: the same live, + the block's yaw, the (0, 0) slot and ``seated_b`` in
  §12's three appended slots (46 with PQ), so §12's and S10.1's checkpoints load as IN and IN-FT did.
"""

from __future__ import annotations

import dataclasses

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from pal_mjlab.tasks.manipulation import mdp, scene
from pal_mjlab.tasks.manipulation.mdp.ball_seat import SEAT_SPEED, SEAT_TOL
from pal_mjlab.tasks.manipulation.mdp.shapes import BySize
from pal_mjlab.tasks.manipulation.pick_place.box import (
  BOX_YAW_BIAS,
  BOX_YAW_NOISE,
  TRACKER_OWN_BIAS,
  TRACKER_SHARED_BIAS,
)
from pal_mjlab.tasks.manipulation.pick_place.insert import (
  BLOCK,
  BLOCK_AREA_YAW,
  BLOCK_HAND_SENSOR,
  BLOCK_PIN_SENSOR,
  INSERT_REWARD,
  _contact_sensor,
)
from pal_mjlab.tasks.manipulation.pick_place.place_pin import LEFT_HAND_KEEPOUT
from pal_mjlab.tasks.manipulation.robot_cfg import GripperCfg
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import (
  _DR_OBS_OBJECT_POS_NOISE,
  COMMAND,
  OBJECT,
  ROBOT,
  SUPPORT_SENSOR,
  TABLE,
)

#: RCTP, the reward ``GOAL=ball-seat`` edits.
BALL_SEAT_REWARD = INSERT_REWARD
#: The ball (D15-1 open: the plan assumes Ø 50 mm, 50 g); PB-DR's radius range (Ø 48-52 mm).
BALL_SEAT_RADIUS = 0.025
BALL_SEAT_RADIUS_RANGE = (0.024, 0.026)
#: The seated centre's height over the block's bottom face by the ball's radius, m (`mdp.seat_centre_z`). Plan §15.0's
#: CPU seat table (MuJoCo 3.15, §12's 60 boxes; `results/S15-ball-seat/preview/ball_cup_r*.json`'s ``rest_z``):
#: 60.67 / 62.58 / 64.38 mm at r 24 / 25 / 26 mm -- 1.3 mm under the analytic cone's (the 16-gon sits lower). Step 0
#: item 1 re-measures it on mujoco_warp (`tests/test_pick_place_s15.py`).
SEAT = BySize(radii=(0.024, 0.025, 0.026), values=(0.06067, 0.06258, 0.06438))
#: The drop point's height over the seated centre, m (plan §15.2's ``h_drop``; step 0 item 4 sets it).
DROP_HEIGHT = 0.025
#: The least xy distance between the ball's and the block's centres, m (§12.1's pin: 14 cm; step 0 item 6).
BALL_MIN_SEPARATION = 0.14
#: Where the block's centre may not be drawn: §14's left start-hand keepout (measured at 0.847 m, step 0 item 4 there).
BLOCK_KEEPOUT = LEFT_HAND_KEEPOUT
#: M1's release envelope at the let-go (xy of the centre from the axis, the ball's bottom over the block's top, speed):
#: plan §15.0's safe cell (seated 8 / 8 up to 20 mm off at <= 5 cm/s, the height up to 3 cm), m, m, m/s.
ENVELOPE = (0.015, 0.030, 0.05)
#: M6's grade: G(xy at the let-go, `M6_DIST_STD`) x G(speed, `M6_SPEED_STD`); the height free up to the envelope's
#: (the centre `M6_HEIGHT_FREE` over the seat: bottom 3 cm over the top for r 25), falling with `M6_HEIGHT_STD` past it.
M6_DIST_STD = 0.010
M6_SPEED_STD = 0.05
M6_HEIGHT_FREE = 0.048
M6_HEIGHT_STD = 0.02
#: M2 on the drop point (RCTP's widths and weights) and the clearance gate (§13.1's ``over_block_gated``): within
#: `GATE_RADIUS` of the axis only with the bottom `GATE_CLEARANCE` over the top, unless inside the seat's cone
#: (`SEAT_CONE_RADIUS`: a centre 10 mm off the axis has its bottom on the chamfer).
APPROACH_STD, APPROACH_WEIGHT = 0.15, 7.0
NEAR_STD, NEAR_WEIGHT = 0.03, 3.0
GATE_RADIUS, GATE_CLEARANCE, SEAT_CONE_RADIUS = 0.06, 0.005, 0.010
#: ``A_B`` (plan §15.2): weight, widths (xy m, speed m/s), and the window about the drop point (m over / under).
OVER_WEIGHT, OVER_XY_STD, OVER_SPEED_STD = 2.0, 0.008, 0.05
OVER_ABOVE, OVER_BELOW = 0.04, 0.01
#: M8's retreat: to 10 cm over the ball's top.
RETREAT_CLEAR = 0.10
#: RCTP's terms the seat rewrites.
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
class BallSeatGoal:
  """What ``GOAL=ball-seat`` builds (`apply_ball_seat_goal`)."""

  seat: BySize = SEAT
  drop_height: float = DROP_HEIGHT
  envelope: tuple[float, float, float] = ENVELOPE
  over_weight: float = OVER_WEIGHT
  """``A_B``'s weight; 0 drops it."""

  @property
  def geometry(self) -> dict:
    """The block numbers every seat term reads."""
    return {"block_height": scene.BLOCK_HEIGHT, "seat": self.seat}


#: The ``GOAL`` seat codes (plan §15.4: PB-BX, PB-IN, PB-DR, PB-S all run ``ball-seat``; PB-DR's ball is its SHAPE).
BALL_SEAT_GOALS: dict[str, BallSeatGoal] = {"ball-seat": BallSeatGoal()}


def apply_ball_seat_goal(
  cfg: ManagerBasedRlEnvCfg,
  gripper: GripperCfg,
  code: BallSeatGoal,
  area: tuple[tuple[float, float], tuple[float, float]],
  table_top_z: float,
  threshold: float,
) -> None:
  """Turn a built RCTP pick-and-place cfg with the ball on a target area into plan §15's ball seat, in place (the module
  docstring). ``area`` ((x lo, x hi), (y lo, y hi), env frame for the table at its nominal place) is where the block is
  drawn; ``table_top_z`` the baseline's table; ``threshold`` the goal tolerance the command logs success with."""
  missing = [t for t in _TASK_TERMS if t not in cfg.rewards]
  if missing:
    raise ValueError(
      f"GOAL=ball-seat edits {BALL_SEAT_REWARD}'s task terms; this reward lacks {missing}"
    )
  geo = code.geometry
  scene_cfg = cfg.scene
  scene_cfg.entities = {
    **scene_cfg.entities,
    TABLE: scene.get_table_cfg(table_top_z, contype=scene.INSERT_TABLE_CONTYPE),
    BLOCK: scene.get_socket_block_cfg(name=BLOCK),
  }
  table_cx, table_cy = scene.TABLE_POS[0], scene.TABLE_POS[1]
  (x0, x1), (y0, y1) = area
  # After `reset_objects` (and the shape's draw before it): it reads where the ball now is.
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
      "min_distance": BALL_MIN_SEPARATION,
      "keepout": BLOCK_KEEPOUT,
    },
  )
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
    params={"half": BOX_YAW_BIAS, "keys": (BLOCK,)},
  )
  cfg.commands[COMMAND] = mdp.BallSeatCommandCfg(
    entity_name=OBJECT,
    block_name=BLOCK,
    seat=code.seat,
    drop_height=code.drop_height,
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
      "GOAL=ball-seat replaces the actor's object_pos term; it is "
      f"{obj_term.func.__name__} (occlusion is not built for the seat)"
    )
  lag = obj_term.params.get("lag")
  pos_noise = Unoise(n_min=-_DR_OBS_OBJECT_POS_NOISE, n_max=_DR_OBS_OBJECT_POS_NOISE)
  actor["object_pos"] = dataclasses.replace(
    obj_term,
    func=mdp.object_position_biased_b,
    params={**obj_term.params, "bias_key": OBJECT},
    noise=pos_noise,
  )
  actor["object_yaw"] = ObservationTermCfg(func=mdp.no_yaw)
  actor["target_pos"] = ObservationTermCfg(
    func=mdp.object_position_biased_b,
    params={"object_name": BLOCK, "bias_key": BLOCK, "lag": lag},
    noise=pos_noise,
  )
  actor["box_yaw"] = ObservationTermCfg(
    func=mdp.box_yaw_b,
    params={"box_name": BLOCK, "bias_key": BLOCK, "noise": BOX_YAW_NOISE, "lag": lag},
  )
  actor["pin_tilt"] = ObservationTermCfg(func=mdp.no_tilt)
  critic = cfg.observations["critic"].terms
  critic["object_yaw"] = ObservationTermCfg(func=mdp.no_yaw)
  critic["target_pos"] = ObservationTermCfg(
    func=mdp.object_position_b, params={"object_name": BLOCK}
  )

  # -- Rewards. ---------------------------------------------------------------------------------
  rw = cfg.rewards
  lifted = {
    "grasp_sensor_name": gripper.contact_sensor_name,
    "support_sensor_name": SUPPORT_SENSOR,
  }
  ball = {"command_name": COMMAND, "object_name": OBJECT, "block_name": BLOCK, **geo}
  m6 = {
    "threshold": SEAT_TOL,
    "dist_std": M6_DIST_STD,
    "height_free": M6_HEIGHT_FREE,
    "height_std": M6_HEIGHT_STD,
    "speed_std": M6_SPEED_STD,
  }
  release = rw["release"]
  rw["release"] = RewardTermCfg(
    func=mdp.seat_release,
    weight=release.weight,
    params={**release.params, **m6, **geo, "envelope": code.envelope},
  )
  placed = rw["placed"]
  rw["placed"] = RewardTermCfg(
    func=placed.func,
    weight=placed.weight,
    params={**placed.params, **m6, "velocity_threshold": SEAT_SPEED},
  )
  for name, std, weight in (
    ("goal", APPROACH_STD, APPROACH_WEIGHT),
    ("goal_near", NEAR_STD, NEAR_WEIGHT),
  ):
    rw[name] = RewardTermCfg(
      func=mdp.seat_approach,
      weight=weight,
      params={
        **ball,
        **lifted,
        "std": std,
        "gate_radius": GATE_RADIUS,
        "gate_clearance": GATE_CLEARANCE,
        "seat_radius": SEAT_CONE_RADIUS,
      },
    )
  if code.over_weight:
    rw["over_seat"] = RewardTermCfg(
      func=mdp.seat_over_grade,
      weight=code.over_weight,
      params={
        **ball,
        **lifted,
        "xy_std": OVER_XY_STD,
        "speed_std": OVER_SPEED_STD,
        "above": OVER_ABOVE,
        "below": OVER_BELOW,
      },
    )
  retreat = rw["retreat"]
  rw["retreat"] = RewardTermCfg(
    func=retreat.func,
    weight=retreat.weight,
    params={
      **retreat.params,
      "threshold": SEAT_TOL,
      "velocity_threshold": SEAT_SPEED,
      "clear_height": RETREAT_CLEAR,
    },
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


def append_critic_ball_seat(
  cfg: ManagerBasedRlEnvCfg, code: BallSeatGoal, grasp_sensor: str
) -> None:
  """The critic's seat inputs in §12's three appended slots, after PQ's: the block's live yaw (cos 4 psi, sin 4 psi),
  the constant (0, 0) in the pin's tilt slot, and ``seated_b`` (`mdp.ball_seated_obs`) in ``seated``'s."""
  critic = cfg.observations["critic"]
  critic.terms = {
    **critic.terms,
    "box_yaw": ObservationTermCfg(func=mdp.box_yaw_b, params={"box_name": BLOCK}),
    "pin_tilt": ObservationTermCfg(func=mdp.no_tilt),
    "pin_seated": ObservationTermCfg(
      func=mdp.ball_seated_obs,
      params={
        "object_name": OBJECT,
        "block_name": BLOCK,
        **code.geometry,
        "grasp_sensor_name": grasp_sensor,
      },
    ),
  }
