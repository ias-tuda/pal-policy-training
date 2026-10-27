"""Put the mug on the coaster, on the standing grasp recipe.

The mug stands upright on the table, handle any way round, beside a free coaster
(``scene.get_coaster_spec``). The left hand picks it up, sets it down on the coaster,
lets go and takes the hand away. It is the stack's ``-Place`` ladder with the mug for the
red cube and the coaster for the blue one: the lift goal sits where the mug stands on the
coaster and follows the coaster if it is pushed, and the rungs above the carry are
``recipe_stack_env_cfg.PLACE_REWARDS`` -- held on the coaster and centred, let go on it,
and the hand withdrawn.

The mug comes through the tracker as in the other mug tasks, whole orientation included,
and the actor sees the vector to the goal on the coaster. Tipping the mug past
``TURN_TIP_LIMIT`` ends the episode, as does dropping it or moving the mug or the coaster
out of the camera's view.
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from pal_mjlab.tasks.manipulation import mdp, scene
from pal_mjlab.tasks.manipulation.grasp.kangaroo.env_cfg import (
  PATIENT_GAMMA,
  WIDE_OBJECT_POSE_RANGE,
  kangaroo_standing_recipe_env_cfg,
)
from pal_mjlab.tasks.manipulation.mug.mug_env_cfg import (
  TURN_TIP_LIMIT,
  UPRIGHT_X_RANGE,
  VISIBLE_BOX,
  _add_hand_object_sensor,
  _grasp_sensors,
  _keep_visible,
  _set_mug_observations,
)
from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS
from pal_mjlab.tasks.manipulation.stack.recipe_stack_env_cfg import (
  kangaroo_recipe_stack_env_cfg,
  set_place_rewards,
)
from pal_mjlab.tasks.manipulation.stack.stack_env_cfg import (
  BASE_OBJECT,
  STACK_CONTACT_SENSOR,
)
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import (
  OBJECT,
  SUPPORT_SENSOR,
  TABLE,
)

COASTER = "coaster"

#: Where the mug and the coaster spawn, relative to the table centre: the upright mug's
#: spawn (``UPRIGHT_X_RANGE``, clear of the resting fingertips), over the wide spawn's y.
COASTER_POSE_RANGE = {**WIDE_OBJECT_POSE_RANGE, "x": UPRIGHT_X_RANGE}

#: Centre to centre between the mug and the coaster at spawn: the mug body (32.5 mm) clear
#: of the coaster's corner (71 mm) whichever way the coaster is turned. Any further and
#: the 12 cm deep spawn leaves the sampler few layouts to find.
COASTER_MIN_SEPARATION = 0.11

#: Penalty on the coaster's speed: the goal follows it, but shoving it is no way to get
#: there.
COASTER_DISTURBANCE_WEIGHT = -0.1

#: ``-Release`` (``coaster_rewards``): the jaw letting go of the placed mug, ``0.5`` of
#: it open on the mug and all of it withdrawn, and how long holding it at the goal pays
#: once it has been set down -- in all, this episode. Holding on pays the grasp ladder
#: (~8) after that; open on the mug ~10 more, withdrawn and off it ``stacked`` (60) and
#: the release (20), clear of it ``clear`` (20) on top.
COASTER_RELEASING_REWARD = 20.0
COASTER_HOLD_TIME = 1.0
#: ``-Release``: ``lift`` drops to ``COASTER_LIFT_AFTER_BUDGET`` once it has paid this many
#: seconds of its full reward this episode, however far off the goal:
#: ``coaster_rewards.paid_until_placed``.
COASTER_LIFT_BUDGET = 3.0
#: The share of ``lift`` paid once ``COASTER_LIFT_BUDGET`` is spent. A quarter (to keep
#: pulling the mug toward the goal) cut success from 13% to 3% within 600 iterations of
#: a fine-tune: it paid holding the mug in the air again.
COASTER_LIFT_AFTER_BUDGET = 0.0
#: ``-Release``: ``set_down`` centred to 2 cm, not the cube's 1 cm. The coaster is 10 cm
#: square and the mug body 6.5 cm across: 2 cm off centre it still stands wholly on it,
#: and at 1 cm the policy's median 1.6 cm earned 8% of the rung.
COASTER_SET_DOWN_XY_STD = 0.02
#: ``-Replay``: the share of episodes started from a state the policy reached, the mug
#: just set down on the coaster in the jaw (``coaster_rewards.reset_from_placed_states``),
#: and how many such states are kept, the newest.
COASTER_REPLAY_PROBABILITY = 0.3
#: ``-Release``: what the mug tipped over on the table or pushed out of view costs, once
#: (``mdp.termination_fired``). About the lift budget's worth: the episode's end alone
#: cost ~20 in the approach, and 84% of the tip-overs came before any grasp.
COASTER_KNOCK_PENALTY = -50.0
#: ``-Release``: carrying the mug to the coaster as shaping (``coaster_carry_progress``),
#: ``weight`` what a carry from ``phi = 0`` to on the goal pays, discounted at the
#: runner's gamma; ``std`` the distance kernel, the spawn ~11-25 cm off.
COASTER_CARRY_REWARD = 60.0
COASTER_CARRY_STD = 0.15
#: ``-Release``: the same shaping on a 2 cm kernel, ``centre``: the last centimetres onto
#: the goal. ``set_down``'s 2 cm Gaussian pays 6% at the 3.3 cm the policy set the mug
#: down off centre (``mug-coaster-table837`` 7100, 194 of 512 envs), short of the latch's
#: 2 cm, and ``carry``'s 15 cm kernel pays 0.2 of its weight for all of the last 3.3 cm.
#: ``exp(-d / std)`` is steepest at the goal; as shaping, hovering near it earns nothing.
COASTER_CENTRE_REWARD = 30.0
COASTER_CENTRE_STD = 0.02
COASTER_REPLAY_CAPACITY = 8192


def mug_coaster_env_cfg(
  play: bool = False, release: bool = False, replay: bool = False
) -> ManagerBasedRlEnvCfg:
  """Pick the upright mug up, set it down on the coaster, let go, take the hand away.

  ``release`` (``-Release``) swaps the stack's top rungs for ``coaster_rewards``: the
  set-down holds through the contact's flicker, holding the mug at the goal stops paying
  ``COASTER_HOLD_TIME`` after it is set down (``lift`` also after paying
  ``COASTER_LIFT_BUDGET``), the release pays as the jaw opens and
  withdraws (``COASTER_RELEASING_REWARD``), and placed means no jaw face on the mug. The
  tip ends the episode only for a mug lying on the table out of the jaw
  (``object_tipped_free``): past ``TURN_TIP_LIMIT`` at any instant, it ended set-downs
  that wobbled and would have settled, and held tilts the hand could still right, so every
  set-down near the coaster risked the episode's remaining pay.
  ``replay`` (``-Replay``, with ``release``) starts ``COASTER_REPLAY_PROBABILITY`` of the
  training episodes from a set-down state the policy reached.
  """
  assert release or not replay, "-Replay replays the -Release latch"
  grasp_sensors = (KANGAROO_GRIPPERS.gripper("left").contact_sensor_names,)
  cfg = kangaroo_standing_recipe_env_cfg(
    play=play,
    objects={
      OBJECT: scene.get_mug_cfg(name=OBJECT),
      COASTER: scene.get_coaster_cfg(name=COASTER),
    },
    object_width=scene.MUG_WIDTH,
    object_pose_range=COASTER_POSE_RANGE,
    min_separation=COASTER_MIN_SEPARATION,
    object_tilt_limit=TURN_TIP_LIMIT,
    command_cfg=mdp.EntityOffsetCommandCfg(
      entity_name=OBJECT,
      reference_name=COASTER,
      stack_on_top=True,
      resampling_time_range=(1e9, 1e9),
      success_threshold=0.02,
      placement_grasp_sensor_names=grasp_sensors,
      debug_vis=True,
    ),
  )
  sensors = _grasp_sensors(cfg)
  _set_mug_observations(cfg, observe_goal=True)

  # On the coaster means touching it, not only at the right place: position alone also
  # counts a mug held there.
  cfg.scene.sensors = (
    *cfg.scene.sensors,
    ContactSensorCfg(
      name=STACK_CONTACT_SENSOR,
      primary=ContactMatch(mode="body", pattern=OBJECT, entity=OBJECT),
      secondary=ContactMatch(mode="body", pattern=COASTER, entity=COASTER),
      fields=("found",),
      reduce="none",
      num_slots=1,
      history_length=1,
    ),
  )
  set_place_rewards(cfg, sensors, object_width=scene.MUG_WIDTH)
  if release:
    _set_release_rewards(cfg)
    cfg.terminations["object_flipped"] = TerminationTermCfg(
      func=mdp.object_tipped_free,
      params={
        "object_name": OBJECT,
        "grasp_sensor_names": sensors,
        "support_sensor_name": SUPPORT_SENSOR,
        "limit_angle": TURN_TIP_LIMIT,
      },
    )
  if replay and not play:
    _set_replay(cfg, COASTER)
  cfg.metrics.update(mdp.success_metrics("stacked"))
  cfg.rewards["coaster_disturbance"] = RewardTermCfg(
    func=mdp.entity_lin_vel_l2,
    weight=COASTER_DISTURBANCE_WEIGHT,
    params={"entity_name": COASTER},
  )

  cfg.terminations["mug_dropped"] = TerminationTermCfg(
    func=mdp.object_dropped_free,
    params={
      "object_name": OBJECT,
      "grasp_sensor_names": sensors,
      "support_sensor_name": SUPPORT_SENSOR,
    },
  )
  _keep_visible(cfg)
  if release:
    _add_carry(cfg, sensors)
    cfg.rewards["knocked"] = RewardTermCfg(
      func=mdp.termination_fired,
      weight=COASTER_KNOCK_PENALTY,
      params={"term_names": ("object_flipped", "out_of_view")},
    )
  cfg.terminations["coaster_out_of_view"] = TerminationTermCfg(
    func=mdp.entity_outside_surface_box,
    params={
      "entity_name": COASTER,
      "surface_cfg": SceneEntityCfg(TABLE),
      "box": VISIBLE_BOX,
    },
  )
  return cfg


def cube_stack_release_env_cfg(
  play: bool = False, replay: bool = False
) -> ManagerBasedRlEnvCfg:
  """The red cube stacked on the blue one with this task's ``-Release`` ladder on the
  stack's ``-Place`` (``recipe_stack_env_cfg``); here, as this module builds on that one.

  The stack's first runs rested 82% of the cubes on the base in the jaw and let go of
  4-7%, as the mug did here before ``-Release``: holding at the goal paid to the end of
  the episode. The mug's tip and view terminations stay with the mug; a cube rests on the
  base on any face. ``replay`` (``-Replay``) starts ``COASTER_REPLAY_PROBABILITY`` of the
  training episodes from a set-down the policy reached.
  """
  cfg = kangaroo_recipe_stack_env_cfg(play=play, place=True)
  _set_release_rewards(cfg, object_width=scene.CUBE_WIDTH)
  _add_carry(cfg, _grasp_sensors(cfg), stand_tilt=math.pi)
  if replay and not play:
    _set_replay(cfg, BASE_OBJECT)
  cfg.metrics.update(mdp.success_metrics("stacked"))
  return cfg


def _set_release_rewards(
  cfg: ManagerBasedRlEnvCfg, object_width: float = scene.MUG_WIDTH
) -> None:
  """``-Release``'s top rungs, on the place ladder ``set_place_rewards`` built;
  ``object_width`` is what the jaw has to open past to let go."""
  r = cfg.rewards
  hand = {
    "hand_sensor_name": _add_hand_object_sensor(cfg),
    "object_width": object_width,
  }
  stacked = {**r["stacked"].params, **hand}
  r["stacked"] = RewardTermCfg(
    func=mdp.mug_placed_on_coaster, weight=r["stacked"].weight, params=stacked
  )
  r["clear"] = RewardTermCfg(
    func=mdp.mug_placed_on_coaster,
    weight=r["clear"].weight,
    params={**r["clear"].params, **hand},
  )
  r.pop("release")
  r["releasing"] = RewardTermCfg(
    func=mdp.mug_releasing_on_coaster,
    weight=COASTER_RELEASING_REWARD,
    params=stacked,
  )
  r["set_down"].params["xy_std"] = COASTER_SET_DOWN_XY_STD
  for name in ("lift", "set_down"):
    r[name] = RewardTermCfg(
      func=mdp.paid_until_placed,
      weight=r[name].weight,
      params={
        "func": r[name].func,
        "placed_term": "stacked",
        "hold_time": COASTER_HOLD_TIME,
        **(
          {"budget": COASTER_LIFT_BUDGET, "after_budget": COASTER_LIFT_AFTER_BUDGET}
          if name == "lift"
          else {}
        ),
        **r[name].params,
      },
    )


def _add_carry(
  cfg: ManagerBasedRlEnvCfg, sensors, stand_tilt: float | None = None
) -> None:
  """``-Release``'s carry to the goal and its last centimetres, as shaping
  (``coaster_carry_progress``); ``stand_tilt`` how far off upright the object still
  stands on the base, the mug's by default."""
  cfg.rewards["carry"] = RewardTermCfg(
    func=mdp.coaster_carry_progress,
    weight=COASTER_CARRY_REWARD,
    params={
      "command_name": cfg.rewards["set_down"].params["command_name"],
      "object_name": OBJECT,
      "grasp_sensor_names": sensors,
      "stack_contact_sensor_name": STACK_CONTACT_SENSOR,
      "gamma": PATIENT_GAMMA,
      "std": COASTER_CARRY_STD,
      **({} if stand_tilt is None else {"stand_tilt": stand_tilt}),
    },
  )
  cfg.rewards["centre"] = RewardTermCfg(
    func=mdp.coaster_carry_progress,
    weight=COASTER_CENTRE_REWARD,
    params={**cfg.rewards["carry"].params, "std": COASTER_CENTRE_STD},
  )


def _set_replay(cfg: ManagerBasedRlEnvCfg, base_name: str) -> None:
  """Keep the set-down states ``stacked`` latches, and start episodes from them;
  ``base_name`` is what the object is set down on."""
  table = SceneEntityCfg(TABLE)
  cfg.rewards["stacked"].params.update(
    replay_capacity=COASTER_REPLAY_CAPACITY, surface_cfg=table
  )
  # Last, after the settled stance: it overwrites the whole scene that one wrote.
  cfg.events["replay_placed"] = EventTermCfg(
    func=mdp.reset_from_placed_states,
    mode="reset",
    params={
      "placed_term": "stacked",
      "object_names": (OBJECT, base_name),
      "surface_cfg": table,
      "probability": COASTER_REPLAY_PROBABILITY,
    },
  )
