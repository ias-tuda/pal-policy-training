"""Put the Kugel in the mug, on the standing grasp recipe.

The hollow mug (``scene.get_hollow_mug_spec``) stands upright on the table, handle any
way round, beside a 40 mm wooden ball, the Kugel. The left hand picks the Kugel up,
carries it over the mug's 60.6 mm opening, lets go so it drops in, and takes the hand
away. The goal sits over the opening, the Kugel's underside 3 cm above the rim, and
follows the mug if it is pushed. See ``mdp.kugel_rewards`` for the ladder above the grasp.

The actor sees the Kugel and the goal relative to the jaw, as the lean recipe does, but
not the Kugel's yaw: a ball has none worth tracking.

Rather too few terminations than too many: none is added to the recipe's, and its
arm-table one is taken out. A Kugel dropped beside the mug lies there to be picked up
again, and a tipped mug costs ``KUGEL_MUG_TIPPED_PENALTY`` a second for as long as it
lies. The arm may touch the table -- the 40 mm Kugel puts the fingers within millimetres
of it -- and pays only for pressing into it, by the newton (``TABLE_FORCE_*``).
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from pal_mjlab.tasks.manipulation import mdp, scene
from pal_mjlab.tasks.manipulation.coaster.coaster_env_cfg import (
  COASTER_POSE_RANGE,
  COASTER_RELEASING_REWARD,
)
from pal_mjlab.tasks.manipulation.grasp.kangaroo.env_cfg import (
  PATIENT_GAMMA,
  kangaroo_standing_recipe_env_cfg,
)
from pal_mjlab.tasks.manipulation.mug.mug_env_cfg import _grasp_cfg, _grasp_sensors
from pal_mjlab.tasks.manipulation.stack.recipe_stack_env_cfg import (
  CLEAR_DISTANCE,
  PLACE_REWARDS,
)
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import COMMAND, OBJECT

MUG = "mug"

#: Centre to centre between the Kugel and the mug at spawn: the handle's 65 mm reach,
#: the Kugel's 20 mm radius, and room for a finger between them.
KUGEL_MIN_SEPARATION = 0.12

#: Where the goal sits over the mug centre: the Kugel's underside ``KUGEL_DROP_HEIGHT``
#: above the rim. Dropped from 5 cm it went in up to 30 mm off the axis.
KUGEL_DROP_HEIGHT = 0.03
KUGEL_GOAL_OFFSET_Z = scene.MUG_HALF_HEIGHT + scene.KUGEL_RADIUS + KUGEL_DROP_HEIGHT

#: Seconds of its full reward ``lift`` pays before it stops: holding on at the goal must
#: not pay for good (``mdp.paid_within_budget``).
KUGEL_LIFT_BUDGET = 3.0
#: ``kugel_drop_progress`` to the goal on a 15 cm kernel, and the last centimetres onto
#: the axis on a 2 cm one, as the coaster's carry and centre.
KUGEL_CARRY_REWARD = 60.0
KUGEL_CARRY_STD = 0.15
KUGEL_CENTRE_REWARD = 30.0
KUGEL_CENTRE_STD = 0.02
#: In the mug, let go: per second, as ``stacked``; and with the hand away, as ``clear``.
KUGEL_IN_MUG_REWARD = PLACE_REWARDS["stacked"]
KUGEL_CLEAR_REWARD = PLACE_REWARDS["clear"]
#: Per second the mug lies tipped. Above what the grasp ladder pays for holding the Kugel
#: once ``lift`` is spent, so knocking the mug over never pays.
KUGEL_MUG_TIPPED_PENALTY = -10.0
#: The mug's speed, as the coaster's: the goal follows it, but shoving it is no way there.
KUGEL_MUG_DISTURBANCE_WEIGHT = -0.1
#: Per second any part of the hand touches the mug, with ``hand_off_mug``. In 73-78% of
#: the tip-overs under s123 model_2000 and s456 model_3300 the hand touched the mug in the
#: half second before, the Kugel still in the jaw in 69-76%; the Kugel or the forearm
#: alone hardly ever. ``mug_tipped`` only prices the mug once it lies, so brushing the rim
#: on the way in cost nothing. Twice the old flat arm-table price.
KUGEL_HAND_MUG_PENALTY = -10.0

#: The ball tracker on the robot, from the last 8 mug-ball recordings: ``ball_1`` published
#: at 5.1-6.3 Hz, 167 ms apart (every 5th frame at 30 fps), each pose ~235 ms old when it
#: arrives, gaps of 0.4-1.9 s with the ball in the jaw or the mug. The recipe trained on
#: 100-160 ms and a pose every step, so the policy saw the ball 0.24-0.4 s late against
#: 0.1-0.16 s; at the 9 cm/s the arm moved at release, that is ~2 cm. With ``real_tracker``,
#: in control steps of 20 ms: published every 120-200 ms, 180-280 ms old, lost for up to 2 s.
KUGEL_TRACKER_LAG = (9, 14)
KUGEL_TRACKER_HOLD = (6, 10)
KUGEL_TRACKER_DROPOUT = (0.01, 5, 100)

#: Arm-table net force the arm may put on the table for free, and the cost per newton
#: over it, per second. Under model_700 of the first runs, with the arm-table termination
#: off, contact is 8.6% of steps at a median 14 N, 90% under 29 N, 0.3% over 100 N: 20 N
#: lets resting and brushing pass. 30 N then costs 5/s, the recipe's flat price for any
#: contact; a 200 N crash 90/s for its steps.
TABLE_FORCE_FREE = 20.0
TABLE_FORCE_WEIGHT = -0.5
#: Newtons over the free force a step is charged at most: one solver spike is not a crash.
TABLE_FORCE_CAP = 200.0

#: The hand's subtree against the mug, for ``clear``.
HAND_MUG_SENSOR = "hand_mug_contact"


def mug_kugel_env_cfg(
  play: bool = False, hand_off_mug: bool = False, real_tracker: bool = False
) -> ManagerBasedRlEnvCfg:
  """Pick the Kugel up, drop it into the upright mug, take the hand away.

  ``hand_off_mug`` prices every step the hand touches the mug
  (``KUGEL_HAND_MUG_PENALTY``); ``real_tracker`` shows the actor the Kugel as the robot's
  tracker reports it (``KUGEL_TRACKER_*``).
  """
  cfg = kangaroo_standing_recipe_env_cfg(
    play=play,
    objects={
      OBJECT: scene.get_kugel_cfg(name=OBJECT),
      MUG: scene.get_hollow_mug_cfg(name=MUG),
    },
    object_width=scene.KUGEL_WIDTH,
    object_pose_range=COASTER_POSE_RANGE,
    min_separation=KUGEL_MIN_SEPARATION,
    command_cfg=mdp.EntityOffsetCommandCfg(
      entity_name=OBJECT,
      reference_name=MUG,
      offset=(0.0, 0.0, KUGEL_GOAL_OFFSET_Z),
      resampling_time_range=(1e9, 1e9),
      success_threshold=0.02,
      debug_vis=True,
    ),
  )
  sensors = _grasp_sensors(cfg)
  grasp_cfg = _grasp_cfg(cfg)

  # A ball has no yaw the tracker could give, nor one the grasp depends on.
  actor = cfg.observations["actor"]
  actor.terms = {k: t for k, t in actor.terms.items() if k != "object_yaw_to_jaw"}
  if real_tracker:
    for k in ("jaw_to_object", "object_to_goal"):
      actor.terms[k] = replace(
        actor.terms[k],
        params={
          **actor.terms[k].params,
          "lag": KUGEL_TRACKER_LAG,
          "dropout": KUGEL_TRACKER_DROPOUT,
          "hold": KUGEL_TRACKER_HOLD,
        },
      )

  (hand,) = [s for s in cfg.scene.sensors if s.name.endswith("_hand_ground_contact")]
  cfg.scene.sensors = (
    *cfg.scene.sensors,
    ContactSensorCfg(
      name=HAND_MUG_SENSOR,
      primary=hand.primary,
      secondary=ContactMatch(mode="body", pattern=MUG, entity=MUG),
      fields=("found",),
      reduce="none",
      num_slots=1,
      history_length=1,
    ),
  )

  _table_contact_by_force(cfg)

  r = cfg.rewards
  r["lift"] = RewardTermCfg(
    func=mdp.paid_within_budget,
    weight=r["lift"].weight,
    params={"func": r["lift"].func, "budget": KUGEL_LIFT_BUDGET, **r["lift"].params},
  )
  drop = {
    "command_name": COMMAND,
    "object_name": OBJECT,
    "mug_name": MUG,
    "grasp_sensor_names": sensors,
    "gamma": PATIENT_GAMMA,
  }
  r["carry"] = RewardTermCfg(
    func=mdp.kugel_drop_progress,
    weight=KUGEL_CARRY_REWARD,
    params={**drop, "std": KUGEL_CARRY_STD},
  )
  r["centre"] = RewardTermCfg(
    func=mdp.kugel_drop_progress,
    weight=KUGEL_CENTRE_REWARD,
    params={**drop, "std": KUGEL_CENTRE_STD, "xy_only": True},
  )
  r["releasing"] = RewardTermCfg(
    func=mdp.kugel_releasing_over_mug,
    weight=COASTER_RELEASING_REWARD,
    params={"object_name": OBJECT, "mug_name": MUG, "grasp_cfg": grasp_cfg},
  )
  in_mug = {"object_name": OBJECT, "mug_name": MUG, "grasp_sensor_names": sensors}
  r["in_mug"] = RewardTermCfg(
    func=mdp.kugel_in_mug, weight=KUGEL_IN_MUG_REWARD, params=in_mug
  )
  r["clear"] = RewardTermCfg(
    func=mdp.kugel_in_mug,
    weight=KUGEL_CLEAR_REWARD,
    params={
      **in_mug,
      "grasp_cfg": grasp_cfg,
      "hand_sensor_name": HAND_MUG_SENSOR,
      "clearance": CLEAR_DISTANCE,
    },
  )
  r["mug_tipped"] = RewardTermCfg(
    func=mdp.mug_tipped, weight=KUGEL_MUG_TIPPED_PENALTY, params={"mug_name": MUG}
  )
  r["mug_disturbance"] = RewardTermCfg(
    func=mdp.entity_lin_vel_l2,
    weight=KUGEL_MUG_DISTURBANCE_WEIGHT,
    params={"entity_name": MUG},
  )
  if hand_off_mug:
    r["hand_mug"] = RewardTermCfg(
      func=mdp.self_collision,
      weight=KUGEL_HAND_MUG_PENALTY,
      params={"sensor_names": [HAND_MUG_SENSOR]},
    )
  cfg.metrics.update(mdp.success_metrics("in_mug"))
  return cfg


def _table_contact_by_force(cfg: ManagerBasedRlEnvCfg) -> None:
  """The recipe's arm-table contact as a price on the force, not an end to the episode.

  The arm-table sensor reports its net force over every physics substep of a control
  step, and ``table_contact`` pays ``TABLE_FORCE_WEIGHT`` per newton over
  ``TABLE_FORCE_FREE`` instead of a flat cost for any touch.
  """
  (arm,) = [s for s in cfg.scene.sensors if s.name == "arm_table_contact"]
  cfg.scene.sensors = tuple(
    s
    if s is not arm
    else ContactSensorCfg(
      name=arm.name,
      primary=arm.primary,
      secondary=arm.secondary,
      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
      history_length=cfg.decimation,
    )
    for s in cfg.scene.sensors
  )
  cfg.terminations.pop("table_contact")
  cfg.rewards["table_contact"] = RewardTermCfg(
    func=mdp.contact_force_excess,
    weight=TABLE_FORCE_WEIGHT,
    params={
      "sensor_name": arm.name,
      "threshold": TABLE_FORCE_FREE,
      "cap": TABLE_FORCE_CAP,
    },
  )
