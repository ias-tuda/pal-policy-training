"""Mug tasks on the grasp ladder: stand it up, turn it round.

Both start from ``kangaroo_standing_recipe_env_cfg`` -- the grasp ladder on the deployed
WC-c controller (``mug.recipe_env_cfg``) -- with the cube swapped for ``scene.get_mug_spec``, the
Isaac Lab talos env's ``Mug018``. The grasp ladder carries over as it is, scored on the
mug's body centre with the jaw closing to the body's width. The actor sees the mug's
whole orientation (``object_rot``), where the cube tasks keep only its yaw: which way up
the mug is and where its handle points is the task. It comes through the same tracker,
with the same latency and noise as the cube's pose.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.noise import NoiseModelWithAdditiveBiasCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from pal_mjlab.tasks.manipulation import mdp
from pal_mjlab.tasks.manipulation import mug_scene as scene
from pal_mjlab.tasks.manipulation.mug.recipe_env_cfg import (
  CAMERA_VIEW_MARGIN,
  LEFT_ARM_SIGHT_SEGMENTS,
  PATIENT_GAMMA,
  TORSO_FRONT_CAMERA_BODY,
  TORSO_FRONT_CAMERA_POS,
  WIDE_OBJECT_POSE_RANGE,
  kangaroo_standing_recipe_env_cfg,
)
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import (
  OBJECT,
  SUPPORT_SENSOR,
  TABLE,
)

#: Tracker noise on the mug's orientation, per 6D component: the cube yaw's under DR-scene
#: (``tabletop_env_cfg._DR_OBS_OBJECT_YAW_*``), which for a small error is about the same angle.
OBJECT_YAW_NOISE = 0.01
OBJECT_YAW_BIAS = 0.03

#: Tilt from upright within which a mug counts as standing.
STAND_TILT = math.radians(10.0)

#: MugTurn: the handle heading to reach (pointing back at the robot, which faces +x) and
#: how close counts. The mug spawns with the handle ``TURN_SPAWN_OFFSET`` off that heading,
#: to either side, so the turn is 30-120 deg one way or the other.
TURN_TARGET_YAW = math.pi
TURN_TOLERANCE = math.radians(25.0)
TURN_SPAWN_OFFSET = (math.radians(30.0), math.radians(120.0))
#: ``-Near``: the spawn offset cut to what one grasp turns. ``-Release-Regrasp`` 3700,
#: rolled out on 512 mugs, turned 41-67% of those spawned 30-50 deg off, 29-46% at
#: 50-70, and on the side the left arm turns against ~0% past 70.
TURN_NEAR_SPAWN_OFFSET = (math.radians(30.0), math.radians(70.0))
#: A turned mug has to stand, but a little less squarely than a stood-up one.
TURN_TILT = math.radians(15.0)
#: Tipping the mug past this ends a MugTurn or MugHandover episode: it is not coming
#: back up.
TURN_TIP_LIMIT = math.radians(45.0)

#: Where an upright mug may spawn, relative to the table centre. Its 123 mm top is level
#: with the resting jaws, which hang ~123 mm over the table at x -0.09 to -0.16, so the
#: wide cube spawn's back edge (-0.15) put mugs into the hand at reset: with zero action
#: they were shoved up to 15 cm in the first second. From -0.04 the body clears the
#: fingertips with the handle pointing back at them (65 mm), and the front edge moves up
#: by as much, so the spawn stays 12 cm deep.
UPRIGHT_X_RANGE = (-0.04, 0.08)

#: MugUpright ``-Facing``: the lying mug's top -- the axis heading ``reset_mug_on_surface``
#: samples -- points back at the robot, within this. ``-Overhand`` 6300, rolled out on
#: 512 lying mugs: top pointing 90-180 deg and -180 to -135 deg off away from the robot,
#: it stood 88-92% of them, every hold fingers down; pointing -90 to +45 deg, 0-5%, held
#: fingers up 84-100% -- the left arm tips it up only toward itself.
UPRIGHT_FACING_YAW = (math.radians(90.0), math.radians(225.0))

#: MugUpright ``-Away``: ``UPRIGHT_FACING_YAW`` turned half round, the top pointing away
#: from the robot, so the mug turns up toward the body. Stood up from the facing
#: headings, the forearm reaches over the mug to its far side: at ``-Facing`` 7600 the
#: arm lay on the torso camera's line of sight to the mug in 81% of the tipping steps.
UPRIGHT_AWAY_YAW = (math.radians(-90.0), math.radians(45.0))

#: MugUpright ``-View``: what the left arm on the torso camera's line of sight to the
#: mug costs a second (``mdp.camera_view_blocked``), the grasp task's ``-View`` weight.
#: ``-Away`` 7600 -> 8800 never stood a mug: the facing headings stay, and the arm is
#: asked to keep off the line instead -- 81% of the tipping steps at ``-Facing`` 7600.
#: Tipping takes ~2 s, so blocking it throughout costs ~20 against ``stood_released``'s
#: 60 a second. The line stops 3 cm short of the mug's near side.
UPRIGHT_VIEW_PENALTY = 10.0

#: The part of the table the real camera sees, relative to the table centre: the upright
#: and turn spawns (``WIDE_OBJECT_POSE_RANGE``, ``UPRIGHT_X_RANGE``) and 3 cm round them.
#: A mug that leaves it ends the episode: on the robot it would be lost to the tracker.
VISIBLE_BOX = {"x": (-0.18, 0.11), "y": (-0.07, 0.29)}

#: Per-step rewards of the rungs above the grasp ladder, each paying more than everything
#: below it that it gives up.
UPRIGHT_HELD_REWARD = 10.0
STOOD_REWARD = 20.0
STOOD_RELEASED_REWARD = 35.0

#: ``-Lower``: ``mug_lowered_upright`` between holding the mug upright and standing it,
#: with the rungs above raised to stay ahead of it. Held upright high up pays ~20 a step
#: (grasp ladder, lift, ``upright_held``), just above the table ~28, standing in the jaw
#: ~45 (``lowered`` keeps paying there), and let go with the jaw clear 60.
LOWERED_REWARD = 8.0
LOWER_STOOD_REWARD = 30.0
LOWER_STOOD_RELEASED_REWARD = 60.0
#: ``-Release``, on top of ``-Lower``: holding the mug set down pays only for
#: ``STOOD_HOLD_TIME`` (``stood`` and ``lowered``), and the jaw letting go of it pays
#: ``STOOD_RELEASING_REWARD`` -- half once open, all once withdrawn. Held standing pays
#: ~44 a step until then and the grasp ladder's ~6 after; opening the jaw ~17, withdrawing
#: ~30, and clear of the mug ~90. The set-down holds until the mug is
#: ``STOOD_PICKUP_HEIGHT`` above its standing pose, not just off the table: the release
#: rocks it a few mm off it.
STOOD_HOLD_TIME = 1.0
STOOD_RELEASING_REWARD = 30.0
STOOD_PICKUP_HEIGHT = 0.02
TURNING_REWARD = 10.0
TURNED_REWARD = 35.0
#: Upright and turn: the arm back at its start pose once the task is done
#: (``mdp.arm_home_once_done``), within ``HOME_STD`` rad over its joints. The arm ends
#: the task 2.8-4.2 rad (q10-q90) off its start pose, so the kernel is that wide: at
#: 0.5 rad it paid exp(-60) there and gave the policy nothing to climb.
HOME_REWARD = 15.0
HOME_STD = 3.0
#: Upright: the arm held still once the task is done (``mdp.arm_still_once_done``),
#: within ``STILL_STD`` rad/s over its joints. Its arm after the release moved ~0.1 rad/s
#: a joint (turn's 0.02), which pays ~1/3 of it.
STILL_REWARD = 15.0
STILL_STD = 0.3
#: Upright: and with the jaw raised ``RAISED_HEIGHT`` over the table
#: (``mdp.arm_raised_once_done``, in place of ``still``): the fingertips ~4 cm over the
#: 12.3 cm mug's rim. They rest 9 cm up at the start pose.
RAISED_HEIGHT = 0.16
#: ``-Release``: the jaw letting go of the turned mug, between the turning (held) and the
#: turned (let go and clear), so the release is paid all the way through.
TURN_RELEASING_REWARD = 20.0
#: ``-Stage``: the rungs above the grasp as one progress shaping term
#: (``mug_upright_progress``, ``mug_turn_progress``), each weight the shaping a whole run
#: from no progress to all of it pays -- under two seconds of the end state's per-step
#: reward. Discounted at the runner's gamma, as the shaping has to be to leave the
#: optimal policy as it is.
UPRIGHT_PROGRESS_REWARD = 100.0
TURN_PROGRESS_REWARD = 100.0
PROGRESS_TERM = "progress"



def _mug_env_cfg(
  play: bool,
  pose_range: Mapping[str, tuple[float, float]],
  lying: bool,
  observe_goal: bool,
  **task_kwargs,
) -> ManagerBasedRlEnvCfg:
  """The recipe with the mug for the cube, placed upright or ``lying``.

  ``pose_range``'s yaw is the handle heading for an upright mug and the body axis
  heading for a lying one. ``observe_goal`` keeps the cube-to-goal vector in the actor;
  without it the goal command still runs, for the reach rung, but the actor never sees it.
  """
  cfg = kangaroo_standing_recipe_env_cfg(
    play=play,
    objects={OBJECT: scene.get_mug_cfg(name=OBJECT)},
    object_width=scene.MUG_WIDTH,
    **task_kwargs,
  )
  # Same slot as the cube placement it replaces.
  cfg.events["reset_objects"] = EventTermCfg(
    func=mdp.reset_mug_on_surface,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg(OBJECT),
      "surface_cfg": SceneEntityCfg(TABLE),
      "pose_range": dict(pose_range),
      "lying": lying,
    },
  )
  _set_mug_observations(cfg, observe_goal)
  return cfg


def _set_mug_observations(cfg: ManagerBasedRlEnvCfg, observe_goal: bool) -> None:
  """The mug's orientation in place of the cube's yaw relative to the jaw.

  Through the same tracker as the position terms, so it carries the same latency and
  dropouts in the actor and reads the simulator in the critic; per axis component the
  noise is the cube yaw's, which for a small error is about the same angle.
  """
  for name, group in cfg.observations.items():
    src = group.terms["jaw_to_object"]
    rot = ObservationTermCfg(
      func=mdp.object_rot6d_b,
      params={
        "object_name": OBJECT,
        "lag": src.params.get("lag"),
      },
      noise=NoiseModelWithAdditiveBiasCfg(
        noise_cfg=Unoise(n_min=-OBJECT_YAW_NOISE, n_max=OBJECT_YAW_NOISE),
        bias_noise_cfg=Unoise(n_min=-OBJECT_YAW_BIAS, n_max=OBJECT_YAW_BIAS),
      ),
    )
    terms = {}
    for key, term in group.terms.items():
      if name == "actor" and key == "object_yaw_to_jaw":
        terms["object_rot"] = rot
        continue
      if name == "actor" and key == "object_to_goal" and not observe_goal:
        continue
      terms[key] = term
    if name != "actor":
      terms["object_rot"] = replace(rot, noise=None)
    group.terms = terms


def _home_once_done(cfg: ManagerBasedRlEnvCfg, done_term: str) -> None:
  """Pay the driven arm back at its start pose while ``done_term`` pays: once the mug is
  let go nothing else says where the arm goes."""
  arm = cfg.actions["joint_pos"].actuator_names
  cfg.rewards["home"] = RewardTermCfg(
    func=mdp.arm_home_once_done,
    weight=HOME_REWARD,
    params={
      "done_term": done_term,
      "asset_cfg": SceneEntityCfg("robot", joint_names=arm),
      "std": HOME_STD,
    },
  )


def _still_once_done(cfg: ManagerBasedRlEnvCfg, done_term: str) -> None:
  """Pay the driven arm held still while ``done_term`` pays."""
  arm = cfg.actions["joint_pos"].actuator_names
  cfg.rewards["still"] = RewardTermCfg(
    func=mdp.arm_still_once_done,
    weight=STILL_REWARD,
    params={
      "done_term": done_term,
      "asset_cfg": SceneEntityCfg("robot", joint_names=arm),
      "std": STILL_STD,
    },
  )


def _raised_once_done(cfg: ManagerBasedRlEnvCfg, done_term: str) -> None:
  """Pay the driven arm's jaw raised over the table, and held there, while ``done_term``
  pays."""
  arm = cfg.actions["joint_pos"].actuator_names
  cfg.rewards["raised"] = RewardTermCfg(
    func=mdp.arm_raised_once_done,
    weight=STILL_REWARD,
    params={
      "done_term": done_term,
      "asset_cfg": SceneEntityCfg("robot", joint_names=arm),
      "grasp_cfg": cfg.rewards["align"].params["grasp_cfg"],
      "surface_cfg": SceneEntityCfg(TABLE),
      "height": RAISED_HEIGHT,
      "std": STILL_STD,
    },
  )


def _keep_visible(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.terminations["out_of_view"] = TerminationTermCfg(
    func=mdp.entity_outside_surface_box,
    params={
      "entity_name": OBJECT,
      "surface_cfg": SceneEntityCfg(TABLE),
      "box": VISIBLE_BOX,
    },
  )


#: ``-HandsOff``: the hand, any part of it, against the mug.
HAND_OBJECT_SENSOR = "hand_object_contact"


def _add_hand_object_sensor(cfg: ManagerBasedRlEnvCfg) -> str:
  """Add ``HAND_OBJECT_SENSOR``, the hand's subtree against the mug: the pad sensors see
  only the pads, and a finger can rest on the rim with its side. The hand is the
  subtree the single-arm recipe's hand-ground sensor watches."""
  (hand,) = [s for s in cfg.scene.sensors if s.name.endswith("_hand_ground_contact")]
  cfg.scene.sensors = (
    *cfg.scene.sensors,
    ContactSensorCfg(
      name=HAND_OBJECT_SENSOR,
      primary=hand.primary,
      secondary=ContactMatch(mode="body", pattern=OBJECT, entity=OBJECT),
      fields=("found",),
      reduce="none",
      num_slots=1,
      history_length=1,
    ),
  )
  return HAND_OBJECT_SENSOR


def _tcp_cfg(cfg: ManagerBasedRlEnvCfg) -> SceneEntityCfg:
  return cfg.observations["critic"].terms["tcp_pos"].params["asset_cfg"]


def _grasp_sensors(cfg: ManagerBasedRlEnvCfg):
  return cfg.rewards["grasp"].params["sensor_names"]


def _grasp_cfg(cfg: ManagerBasedRlEnvCfg) -> SceneEntityCfg:
  return cfg.rewards["align"].params["grasp_cfg"]


def mug_upright_env_cfg(
  play: bool = False,
  lower: bool = False,
  release: bool = False,
  stage: bool = False,
  overhand: bool = False,
  facing: bool = False,
  away: bool = False,
  view: bool = False,
  still: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Pick up a mug lying on its side and stand it upright on the table, let go.

  Anywhere on the table: the task is the turn, not a place. The ladder above the grasp:
  lifted clear of the table, how upright it is held (``UPRIGHT_HELD_REWARD``), set down
  upright on the table in the jaw after a lift (``STOOD_REWARD``), and that with the jaw
  then off it and clear (``STOOD_RELEASED_REWARD``). The lift task's lift-to-goal is
  dropped: its goal hangs in the air, where this task does not want the mug. Dropping the
  mug ends the episode, and so does moving it out of the camera's view (``VISIBLE_BOX``).

  ``lower`` (``-Lower``) pays for bringing
  the upright mug back down to the table (``LOWERED_REWARD``): without it the first runs
  held the mug upright in the air and never set it down. ``release`` (``-Release``, with
  ``lower``) pays the jaw letting go of the stood mug and stops paying for holding it
  there (``STOOD_HOLD_TIME``): ``-Lower`` policies held it until the episode ended.

  ``stage`` (``-Stage``) replaces the rungs above the grasp, lift included, with
  ``mug_upright_progress``: the hold timer refilled at every re-lift, and the policies
  went back to holding the mug upright in the air. Only standing it free after the jaw
  set it down (``STOOD_RELEASED_REWARD``) still pays per step.

  ``overhand`` (``-Overhand``, with ``lower``) pays holding the mug upright and lowering
  it only with the fingers level or down (``mug_rewards._overhand_gate``): turned up the
  other way the hand is under the mug and can never set it down, and half the
  ``-Lower-Release`` policy's holds were made so. ``facing`` (``-Facing``, with
  ``overhand``) spawns the mug with its top toward the robot (``UPRIGHT_FACING_YAW``):
  from the other half of the headings the left arm stood it up only fingers up. It gates
  the lift too (``mug_lifted_overhand``). ``away`` (``-Away``, with ``overhand``) is
  ``facing`` with the top pointing away instead (``UPRIGHT_AWAY_YAW``): the mug turns up
  toward the body, and the arm stays out of the torso camera's view of it. ``view``
  (``-View``) prices the left arm on the camera's line of sight to the mug
  (``UPRIGHT_VIEW_PENALTY``).

  ``still`` (``-Still``) pays the arm held still once the mug stands released, as before
  ``raised``: the table837 ``-Facing`` fine-tune that stood 69% of the mugs trained on it,
  the ``raised`` one from it 57%.
  """
  assert lower or not release, "-Release builds on -Lower"
  assert lower or not overhand, "-Overhand builds on -Lower"
  assert overhand or not facing, "-Facing builds on -Overhand"
  assert overhand or not away, "-Away builds on -Overhand"
  assert not (facing and away), "-Facing and -Away are opposite headings"
  assert not (stage and (lower or release)), "-Stage replaces -Lower and -Release"
  cfg = _mug_env_cfg(
    play,
    pose_range=(
      {
        **WIDE_OBJECT_POSE_RANGE,
        "yaw": UPRIGHT_AWAY_YAW if away else UPRIGHT_FACING_YAW,
      }
      if facing or away
      else WIDE_OBJECT_POSE_RANGE
    ),
    lying=True,
    observe_goal=False,
  )
  sensors = _grasp_sensors(cfg)
  cfg.rewards["lift"] = RewardTermCfg(
    func=mdp.object_lifted_pair,
    weight=3.0,
    params={
      "grasp_sensor_names": sensors,
      "support_sensor_name": SUPPORT_SENSOR,
    },
  )
  cfg.rewards["upright_held"] = RewardTermCfg(
    func=mdp.mug_upright_held,
    weight=UPRIGHT_HELD_REWARD,
    params={
      "object_name": OBJECT,
      "grasp_sensor_names": sensors,
      "support_sensor_name": SUPPORT_SENSOR,
    },
  )
  stood = {
    "object_name": OBJECT,
    "grasp_sensor_names": sensors,
    "support_sensor_name": SUPPORT_SENSOR,
    "grasp_cfg": _grasp_cfg(cfg),
    "max_tilt": STAND_TILT,
  }
  set_down = {}
  if release:
    set_down = {
      "surface_cfg": SceneEntityCfg(TABLE),
      "pickup_height": STOOD_PICKUP_HEIGHT,
    }
    stood.update(set_down)
  cfg.rewards["stood"] = RewardTermCfg(
    func=mdp.mug_stood_upright,
    weight=LOWER_STOOD_REWARD if lower else STOOD_REWARD,
    params={
      **stood,
      "released": False,
      **({"hold_time": STOOD_HOLD_TIME} if release else {}),
    },
  )
  cfg.rewards["stood_released"] = RewardTermCfg(
    func=mdp.mug_stood_upright,
    weight=LOWER_STOOD_RELEASED_REWARD if lower else STOOD_RELEASED_REWARD,
    params={**stood, "released": True},
  )
  if stage:
    for name in ("lift", "upright_held", "stood"):
      cfg.rewards.pop(name)
    cfg.rewards["stood_released"].params.update(
      surface_cfg=SceneEntityCfg(TABLE), pickup_height=STOOD_PICKUP_HEIGHT
    )
    _add_progress(
      cfg,
      mdp.mug_upright_progress,
      UPRIGHT_PROGRESS_REWARD,
      {
        "object_name": OBJECT,
        "grasp_sensor_names": sensors,
        "support_sensor_name": SUPPORT_SENSOR,
        "grasp_cfg": _grasp_cfg(cfg),
        "surface_cfg": SceneEntityCfg(TABLE),
        "stand_tilt": STAND_TILT,
      },
    )
  if lower:
    cfg.rewards["lowered"] = RewardTermCfg(
      func=mdp.mug_lowered_upright,
      weight=LOWERED_REWARD,
      params={
        "object_name": OBJECT,
        "grasp_sensor_names": sensors,
        "support_sensor_name": SUPPORT_SENSOR,
        "surface_cfg": SceneEntityCfg(TABLE),
        **(
          {
            "stand_tilt": STAND_TILT,
            "pickup_height": STOOD_PICKUP_HEIGHT,
            "hold_time": STOOD_HOLD_TIME,
          }
          if release
          else {}
        ),
      },
    )
  if overhand:
    gate = {"grasp_cfg": _grasp_cfg(cfg), "tcp_cfg": _tcp_cfg(cfg)}
    cfg.rewards["upright_held"].params.update(gate)
    cfg.rewards["lowered"].params.update(gate)
    if facing or away:
      lift = cfg.rewards["lift"]
      cfg.rewards["lift"] = RewardTermCfg(
        func=mdp.mug_lifted_overhand,
        weight=lift.weight,
        params={**lift.params, **gate},
      )
  if release:
    cfg.rewards["stood_releasing"] = RewardTermCfg(
      func=mdp.mug_stood_releasing,
      weight=STOOD_RELEASING_REWARD,
      params=stood,
    )
  if view:
    cfg.rewards["camera_view_blocked"] = RewardTermCfg(
      func=mdp.object_view_blocked,
      weight=-UPRIGHT_VIEW_PENALTY,
      params={
        "camera_body": TORSO_FRONT_CAMERA_BODY,
        "camera_pos": TORSO_FRONT_CAMERA_POS,
        "segments": LEFT_ARM_SIGHT_SEGMENTS,
        "object_name": OBJECT,
        "stop_short": scene.MUG_RADIUS + 0.03,
        "margin": CAMERA_VIEW_MARGIN,
      },
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
  cfg.metrics.update(mdp.success_metrics("stood_released"))
  if still:
    _still_once_done(cfg, "stood_released")
  else:
    _raised_once_done(cfg, "stood_released")
  return cfg


def mug_turn_env_cfg(
  play: bool = False,
  release: bool = False,
  regrasp: bool = False,
  stage: bool = False,
  hands_off: bool = False,
  near: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Grasp a mug whose handle points away, turn it until the handle points back, let go.

  Left arm. The turn has to be made holding the mug: ``TURNING_REWARD`` pays the handle
  heading only while the mug is in the jaw, and ``TURNED_REWARD`` only once it was
  grasped this episode, standing, handle within ``TURN_TOLERANCE`` of the robot, jaw off
  and clear. Tipping the mug over ends the episode, and so does moving it out of the
  camera's view (``VISIBLE_BOX``). With ``release``, ``TURN_RELEASING_REWARD`` pays the
  jaw opening on the mug set down turned and withdrawing from it. With ``regrasp`` the
  turning keeps paying the turn made in the jaw after it lets go (``mug_turning_kept``),
  so the turn can be made in several grasps.

  ``stage`` (``-Stage``) replaces the turning and the release rungs with
  ``mug_turn_progress``: the turning paid the heading the mug spawned with, most of its
  weight for a grasp that turned nothing. Only ``TURNED_REWARD`` still pays per step.
  The mug may be turned in as many grasps as it takes.

  ``hands_off`` (``-HandsOff``) pays the turned mug and the release's withdrawal only
  with no part of the hand on the mug (``HAND_OBJECT_SENSOR``): ``-Release-Regrasp``
  policies set it down turned and then kept a finger on its rim to the end. ``near``
  (``-Near``) spawns it ``TURN_NEAR_SPAWN_OFFSET`` off, what one grasp turns.
  """
  assert not (stage and (release or regrasp)), "-Stage replaces -Release and -Regrasp"
  assert not (stage and hands_off), "-HandsOff is for the per-step turned rungs"
  spawn = TURN_NEAR_SPAWN_OFFSET if near else TURN_SPAWN_OFFSET
  cfg = _mug_env_cfg(
    play,
    pose_range={
      **WIDE_OBJECT_POSE_RANGE,
      "x": UPRIGHT_X_RANGE,
      "yaw": (TURN_TARGET_YAW - spawn[1], TURN_TARGET_YAW - spawn[0]),
    },
    lying=False,
    observe_goal=False,
    object_tilt_limit=TURN_TIP_LIMIT,
  )
  cfg.events["reset_objects"].params["yaw_mirror"] = TURN_TARGET_YAW
  sensors = _grasp_sensors(cfg)
  cfg.rewards.pop("lift")
  cfg.rewards["turning"] = RewardTermCfg(
    func=mdp.mug_turning_kept if regrasp else mdp.mug_turning_held,
    weight=TURNING_REWARD,
    params={
      "object_name": OBJECT,
      "grasp_sensor_names": sensors,
      "target_yaw": TURN_TARGET_YAW,
    },
  )
  cfg.rewards["turned"] = RewardTermCfg(
    func=mdp.mug_turned,
    weight=TURNED_REWARD,
    params={
      "object_name": OBJECT,
      "grasp_sensor_names": sensors,
      "support_sensor_name": SUPPORT_SENSOR,
      "grasp_cfg": _grasp_cfg(cfg),
      "target_yaw": TURN_TARGET_YAW,
      "tolerance": TURN_TOLERANCE,
      "max_tilt": TURN_TILT,
      **({"hand_sensor_name": _add_hand_object_sensor(cfg)} if hands_off else {}),
    },
  )
  if release:
    cfg.rewards["releasing"] = RewardTermCfg(
      func=mdp.mug_turned_releasing,
      weight=TURN_RELEASING_REWARD,
      params=cfg.rewards["turned"].params,
    )
  if stage:
    cfg.rewards.pop("turning")
    turned = dict(cfg.rewards["turned"].params)
    cfg.rewards["turned"].params["progress_term"] = PROGRESS_TERM
    _add_progress(
      cfg,
      mdp.mug_turn_progress,
      TURN_PROGRESS_REWARD,
      {
        **{k: v for k, v in turned.items() if k != "max_tilt"},
        "stand_tilt": TURN_TILT,
        "max_tilt": TURN_TIP_LIMIT,
      },
    )
  _keep_visible(cfg)
  cfg.metrics.update(mdp.success_metrics("turned"))
  _home_once_done(cfg, "turned")
  return cfg


def _add_progress(cfg: ManagerBasedRlEnvCfg, func, weight: float, params: dict) -> None:
  """Add a progress shaping term, discounted at ``PATIENT_GAMMA``, and its latches to
  the critic (``mug_progress_latches``)."""
  cfg.rewards[PROGRESS_TERM] = RewardTermCfg(
    func=func, weight=weight, params={**params, "gamma": PATIENT_GAMMA}
  )
  for name, group in cfg.observations.items():
    if name != "actor":
      group.terms["progress_latches"] = ObservationTermCfg(
        func=mdp.mug_progress_latches, params={"term_name": PROGRESS_TERM}
      )
