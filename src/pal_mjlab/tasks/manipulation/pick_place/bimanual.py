"""Plan S11's JOINT (`ARMS=both`): one policy drives both arms -- the left places the blue cube A
at T, the right stacks the red cube B on it (`pick-place-v1-objects-plan.md`, *S11*, BI-J / BI-JM).

The cfg is built from the parts the one-arm codes use, so that a JOINT run differs from SEQ's L-T
and R-ST by the coupling and nothing else:

* the **left half** is L-T exactly -- `make_pick_place_env_cfg` with `TARGET=T11`, `TRACKER=split`,
  RCTP + YG, PQ, AH10 -- built on the two-arm robot (`robots.kangaroo_grippers_both_arms`: 16-D
  action, both arms at W2's scales, T on both, the right home mirrored), so the action term, the
  regularisers, the effort hinge and the trunk penalty already span both arms;
* the **right half** is R-ST's reward arm (RCTP + SA) built for the right gripper and renamed onto
  cube B and its goal on top of A (`COMMAND_B`, an `EntityOffsetCommand` that follows A), with the
  right's own sensors, AH10 and PQ pair;
* the **couplings** (the plan's three): the right's stack terms pay only while A is placed
  (`mdp.when_base_placed`); the left's hand-off runs inside training (`HANDOFF_TRAIN=left`,
  `mdp.handoff`); the arms are kept apart (HH: `mdp.arm_arm_clearance_penalty` and a contact
  sensor between the two forearm-to-fingertip subtrees).

The actor is 73-D: both arms' joint positions (16) and velocities (14), the last action (16), both
TCPs' positions and orientations (14), A's and B's tracked positions and yaws (10) and T (3). The
critic adds both grasps, both PQ pairs, "A placed" and whether the left is handed off.

``START=mixed`` (BI-JM): half the episodes begin with A already placed at T -- +/- 1 cm, yaw +/- 20
deg, the left's ``was_lifted`` and release latched at that pose -- and the left arm handed off at
home, so the stack is practised from the first iteration.

Plan S11.2 (`pick-place-s11-bimanual-plan.md`) adds, each default-off:

* ``reward="S12-BI"``: the task terms of both halves give way to two stage-replacing ladders and a
  one-off cost for pushing A, coupled through the A_latched latch (`mdp.ladder`); HH's clearance margin
  stops at the left's hand-off; the critic sees A_latched; metrics log each level's share of the steps.
* ``handoff_train="both"``: the right hands off on its own trigger in training too (J-release and the
  TCP 3 cm over B's top for 5 steps), so the ladder's R5 is reachable.
* ``start="hold"``: `HOLD_SHARE` of the episodes start from a snapshot of the SEQ chain with B in the
  right hand 2-6 cm over A (`hold_start`, the bank from `scripts/s112_hold_bank.py`), `PLACED_SHARE` with
  A placed and B on the table (BI-JM's start), the rest from scratch.
* ``right_target="gated"`` (BI-F's actor): an actor term ``right_target`` after the 73 -- the wait point
  before A_latched, A's tracked top after (`mdp.right_target_gated`) -- for the factored policy's right
  half (`tasks.manipulation.bimanual_policy`).
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import re

import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, sample_uniform

from pal_mjlab.tasks.manipulation import mdp, scene
from pal_mjlab.tasks.manipulation.mdp.handoff import (
  START_HOME,
  HandoffArmCfg,
  HandoffTriggerCfg,
  with_handoff,
)
from pal_mjlab.tasks.manipulation.pick_place.arm_table import (
  ArmTableCfg,
  apply_arm_table,
)
from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import (
  BASE_RGBA,
  BASELINES,
  EFFORT_BOUNDS,
  GOAL_THRESHOLD,
  ROBOT_TARGETS,
  SPAWN_BOXES,
  Baseline,
  EffortBound,
  SpawnBox,
  _object_pose_range,
  make_pick_place_env_cfg,
)
from pal_mjlab.tasks.manipulation.pick_place.rewards import (
  BASE_OBJECT,
  JAW_OPEN,
  RewardContext,
  build_rewards,
)
from pal_mjlab.tasks.manipulation.robots import kangaroo_grippers_both_arms
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import (
  COMMAND,
  DECIMATION,
  LATENCY_MODELS,
  OBJECT,
  ROBOT,
  SUPPORT_SENSOR,
  TABLE,
  LatencyModelCfg,
  make_grasp_sensor,
  make_hand_ground_sensor,
  make_hand_surface_sensor,
)

#: Cube B, the right's: its entity, its goal on top of A, and its contact with the table.
CUBE_B = "cube_b"
COMMAND_B = "goal_b"
SUPPORT_SENSOR_B = "cube_b_table_contact"
#: The arm-arm contact sensor (HH).
ARM_ARM_SENSOR = "arm_arm_contact"
#: The colours: A blue (the base), B red (the carried one), as S9's two cubes.
CUBE_B_RGBA = (0.8, 0.2, 0.2, 1.0)

#: The two halves' reward arms: L-T's and R-ST's (`pick_place.rewards`).
LEFT_REWARD = "R-C-strict+M1+M2+M3+M6+J+M8+YG"
RIGHT_REWARD = "R-C-strict+M1+M2+M3+M6+J+M8+SA"
#: The right's terms that wait for A (coupling 1).
GATED_RIGHT = ("goal", "goal_near", "release", "placed", "stack_align", "retreat")
#: The terms both halves would carry: kept once, over both arms (the left's ctx names them).
SHARED = ("action_rate_l2", "joint_pos_limits", "joint_vel_hinge")

#: B's spawn box: W2's mirror image through y = 0 (the right's own W2).
W2_MIRROR = SpawnBox(
  x=SPAWN_BOXES["W2"].x,
  y=(-SPAWN_BOXES["W2"].y[1], -SPAWN_BOXES["W2"].y[0]),
  yaw=SPAWN_BOXES["W2"].yaw,
  widen_at_dr_scene=False,
)

#: HH (coupling 3): the clearance penalty starts `HH_MARGIN` before contact and reaches 1 there,
#: weight `HH_CLEARANCE_WEIGHT`; contact above `HH_CONTACT_FORCE` costs `HH_CONTACT_WEIGHT` per step.
HH_MARGIN = 0.05
HH_CLEARANCE_WEIGHT = -5.0
HH_CONTACT_WEIGHT = -10.0
HH_CONTACT_FORCE = 1.0

#: The left's hand-off in training: the eval's trigger, or this far into the episode (s).
LEFT_HANDOFF_TIMEOUT_S = 9.0
#: BI-JM's placed-A start: the share of episodes, A about T (+/- m, the T draw's own +/- 1 cm keeps
#: |A - T| inside the 2.5 cm "at T"), its yaw (+/- rad; the plan's placeholder before step 0's
#: L-T measurement).
MIXED_SHARE = 0.5
MIXED_A_HALF_XY = 0.01
MIXED_A_YAW = math.radians(20.0)

#: The JOINT episode, s (plan: 15 s until the overlapped script sets it).
EPISODE_S = 15.0
#: ``START`` names.
STARTS = ("scratch", "mixed", "hold")
#: ``REWARD`` names (plan S11.2): S11's gated sum, or the ladders.
REWARDS = ("S11-BI", "S12-BI")
#: ``HANDOFF_TRAIN`` names: which arms hand off on their own trigger in training.
HANDOFF_TRAINS = ("none", "left", "both")
#: Plan S11.2's hold starts: the shares of the episodes that start with B in the right hand over A (a SEQ
#: snapshot) and with A placed and B on the table; the rest start from scratch.
HOLD_SHARE = 0.3
PLACED_SHARE = 0.2
#: `hold_start`'s per-episode label (`_pp_start_kind`), for the metrics.
START_KINDS = {"scratch": 0, "placed": 1, "hold": 2}
#: The terms `S12-BI` replaces by the ladders: each half's task terms (the left's, and the right's with
#: `right_`), and SA.
LADDER_REPLACES = ("reach", "grasp", "lift", "goal", "goal_near", "release", "placed", "retreat")


def _rename(value, mapping: dict[str, str]):
  """A reward parameter with its entity / command / sensor names moved onto the right half's."""
  if isinstance(value, str):
    return mapping.get(value, value)
  if isinstance(value, SceneEntityCfg) and value.name in mapping:
    return dataclasses.replace(value, name=mapping[value.name])
  return value


def _right_rewards(robot, right) -> dict[str, RewardTermCfg]:
  """R-ST's reward arm for the right gripper on cube B, its stack goal on A. Renamed, not rebuilt:
  the same builders as every one-arm arm, so the halves cannot drift apart."""
  ctx = RewardContext(
    tcp=SceneEntityCfg(ROBOT, site_names=(right.tcp_site,)),
    action_joints=SceneEntityCfg(
      ROBOT, joint_names=(r"arm_right_[1-7]_joint", re.escape(right.finger_joint))
    ),
    grasp_sensor=right.contact_sensor_name,
    surface_sensor=right.surface_sensor_name,
    threshold=GOAL_THRESHOLD,
    jaw=SceneEntityCfg(ROBOT, joint_names=(re.escape(right.finger_joint),)),
    jaw_open=JAW_OPEN,
    pads=SceneEntityCfg(ROBOT, site_names=right.pad_sites),
  )
  mapping = {
    OBJECT: CUBE_B,
    BASE_OBJECT: OBJECT,
    COMMAND: COMMAND_B,
    SUPPORT_SENSOR: SUPPORT_SENSOR_B,
  }
  out = {}
  for name, term in build_rewards(RIGHT_REWARD, ctx).items():
    if name in SHARED:
      continue
    params = {k: _rename(v, mapping) for k, v in term.params.items()}
    out[name] = RewardTermCfg(func=term.func, weight=term.weight, params=params)
  return out


def _gate_params(left_placed: RewardTermCfg) -> dict:
  """`mdp.when_base_placed`'s ``gate_*`` parameters: the left's "placed" predicate (L-T's J and
  `was_lifted`, at T within the goal tolerance, still)."""
  p = left_placed.params
  return {
    "gate_command_name": p["command_name"],
    "gate_object_name": p["object_name"],
    "gate_grasp_sensor_name": p["grasp_sensor_name"],
    "gate_support_sensor_name": p["support_sensor_name"],
    "gate_jaw_cfg": p["jaw_cfg"],
    "gate_jaw_open": p["jaw_open"],
    "gate_threshold": p["threshold"],
    "gate_velocity_threshold": p.get("velocity_threshold", 0.05),
  }


def _pq_terms(placed: RewardTermCfg, prefix: str) -> dict[str, ObservationTermCfg]:
  wanted = inspect.signature(mdp.release_quality).parameters
  params = {k: v for k, v in placed.params.items() if k in wanted}
  return {
    f"{prefix}release_grade": ObservationTermCfg(func=mdp.release_grade, params=params),
    f"{prefix}release_latched": ObservationTermCfg(
      func=mdp.release_latched, params=params
    ),
  }


def _support_sensor(object_name: str, name: str) -> ContactSensorCfg:
  return ContactSensorCfg(
    name=name,
    primary=ContactMatch(mode="body", pattern=object_name, entity=object_name),
    secondary=ContactMatch(mode="body", pattern=TABLE, entity=TABLE),
    fields=("found",),
    reduce="none",
    num_slots=1,
    history_length=1,
  )


def _arm_arm_sensor() -> ContactSensorCfg:
  """The left arm from its forearm (link 4) to the fingertips against the right's, the whole
  control step, the strongest contact (HH's contact term)."""
  return ContactSensorCfg(
    name=ARM_ARM_SENSOR,
    primary=ContactMatch(mode="subtree", pattern="arm_left_4_link", entity=ROBOT),
    secondary=ContactMatch(mode="subtree", pattern="arm_right_4_link", entity=ROBOT),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=DECIMATION,
  )


def place_a_at_t(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  share: float,
  half_xy: float,
  yaw: float,
  arm: str,
  command_name: str,
  object_name: str,
  grasp_sensor_name: str,
  support_sensor_name: str,
  surface_cfg: SceneEntityCfg,
  jaw_cfg: SceneEntityCfg,
  jaw_open: float,
) -> None:
  """Reset event, BI-JM's mixed start: in a ``share`` of ``env_ids`` put cube A down at T (+/-
  ``half_xy``, yaw +/- ``yaw``, the robot frame), latch the left's `was_lifted` and its release
  there (the grade of a let-go at that pose), and start the left arm handed off at home (the
  action term reads `handoff.START_HOME` at its reset, which runs after this event). The release
  latch's parameters are the left's ``placed`` term's, flat, so the event manager resolves them."""
  pick = torch.rand(len(env_ids), device=env.device) < share
  _start_home(env, arm, env_ids, pick)
  ids = env_ids[pick]
  if len(ids) == 0:
    return
  _put_a_at_t(
    env, ids, half_xy, yaw, None, None,
    (command_name, object_name, grasp_sensor_name, support_sensor_name, surface_cfg, jaw_cfg, jaw_open),
  )  # fmt: skip


def _start_home(env: ManagerBasedRlEnv, arm: str, env_ids: torch.Tensor, pick: torch.Tensor) -> None:
  """Mark ``env_ids[pick]`` to start with ``arm`` handed off at home (`handoff.START_HOME`), the rest not."""
  start = getattr(env, START_HOME, None)
  if start is None:
    start = {}
    setattr(env, START_HOME, start)
  if arm not in start:
    start[arm] = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  start[arm][env_ids] = pick


def _put_a_at_t(
  env: ManagerBasedRlEnv,
  ids: torch.Tensor,
  half_xy: float,
  yaw: float,
  xy: torch.Tensor | None,
  quat: torch.Tensor | None,
  latch_params: tuple,
) -> None:
  """Put cube A down on the table -- at ``xy`` (env frame, (N, 2)) with ``quat``, or about T (+/- ``half_xy``,
  yaw +/- ``yaw``) -- and latch the left's `was_lifted` and its release there, as a let-go at that pose."""
  from pal_mjlab.tasks.manipulation.mdp.events import surface_top_z
  from pal_mjlab.tasks.manipulation.mdp.observations import (
    _WAS_LIFTED,
    _release_latch,
    _remember_latch,
    latch_attr,
    yaw_error_mod_90,
  )
  from pal_mjlab.tasks.manipulation.mdp.shapes import object_half_height

  (
    command_name,
    object_name,
    grasp_sensor_name,
    support_sensor_name,
    surface_cfg,
    jaw_cfg,
    jaw_open,
  ) = latch_params
  device = env.device
  m = len(ids)
  tx, ty = ROBOT_TARGETS["T11"].xy
  origin = env.scene.env_origins[ids, :2]
  t_xy = torch.tensor([tx, ty], device=device)
  if xy is None:
    xy = t_xy + sample_uniform(-half_xy, half_xy, (m, 2), device=device)
  xy = origin + xy
  top = surface_top_z(env, surface_cfg)[ids]
  z = top + object_half_height(env, object_name, ids)
  if quat is None:
    zeros = torch.zeros(m, device=device)
    quat = quat_from_euler_xyz(
      zeros, zeros, sample_uniform(-yaw, yaw, (m,), device=device)
    )
  cube = env.scene[object_name]
  cube.write_root_link_pose_to_sim(
    torch.cat([xy, z.unsqueeze(-1), quat], dim=-1), env_ids=ids
  )
  cube.write_root_link_velocity_to_sim(torch.zeros(m, 6, device=device), env_ids=ids)

  attr = latch_attr(_WAS_LIFTED, grasp_sensor_name)
  lifted = getattr(env, attr, None)
  if lifted is None:
    lifted = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    setattr(env, attr, lifted)
    _remember_latch(env, attr)
  lifted[ids] = True
  latch = _release_latch(
    env,
    command_name,
    object_name,
    grasp_sensor_name,
    support_sensor_name,
    surface_cfg,
    jaw_cfg,
    jaw_open,
  )
  # Let go where it stands: on the table, still, 0 high; dxy against T's centre (the drawn T is
  # unknown until the command resamples, after the reset events).
  latch.released[ids] = True
  latch.grasped[ids] = False
  latch.dxy[ids] = torch.norm(xy - origin - t_xy, dim=-1)
  latch.height[ids] = 0.0
  latch.rel_on_table[ids] = True
  latch.rel_speed[ids] = 0.0
  latch.yaw[ids] = yaw_error_mod_90(quat)
  if latch.jaw is not None:
    latch.jaw_closed[ids] = False
    latch.pend[ids] = False


_BANK_CACHE = "_pp_hold_bank"


def load_hold_bank(env: ManagerBasedRlEnv, path: str) -> dict:
  """The SEQ-chain snapshot bank (`scripts/s112_hold_bank.py`) on the env's device, checked against this env's
  joint and action-target names. Cached on the env."""
  bank = getattr(env, _BANK_CACHE, None)
  if bank is not None:
    return bank
  raw = torch.load(path, map_location=env.device, weights_only=False)
  robot = env.scene[ROBOT]
  term = env.action_manager.get_term("joint_pos")
  if list(raw["joint_names"]) != list(robot.joint_names):
    raise ValueError(f"{path}: joint names differ from this env's robot")
  targets = [n.split("/")[-1] for n in term.target_names]
  if list(raw["target_names"]) != targets:
    raise ValueError(f"{path}: action targets {raw['target_names']} != {targets}")
  if len(raw["step"]) == 0:
    raise ValueError(f"{path}: an empty bank")
  setattr(env, _BANK_CACHE, raw)
  return raw


def hold_start(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  bank_path: str,
  hold_share: float,
  placed_share: float,
  half_xy: float,
  yaw: float,
  target_command_name: str,
  command_name: str,
  object_name: str,
  grasp_sensor_name: str,
  support_sensor_name: str,
  surface_cfg: SceneEntityCfg,
  jaw_cfg: SceneEntityCfg,
  jaw_open: float,
) -> None:
  """Reset event, plan S11.2's ``START=hold``: of ``env_ids``, ``hold_share`` start from a snapshot of the SEQ
  chain -- the robot's joints, A, B (in the right hand, 2-6 cm over A) and the action term's targets and jaw
  servos as saved, the left handed off and going home, the episode clock and T as in the snapshot
  (`handoff.START_STATE`) -- ``placed_share`` with A placed at T and the left home (BI-JM's start), the rest
  from scratch. A is put on this episode's table (DR-scene moves it); B keeps its place in the hand. The
  latch parameters are the left's ``placed`` term's. Runs after every other reset event."""
  from pal_mjlab.tasks.manipulation.mdp.handoff import START_STATE

  device = env.device
  n = len(env_ids)
  u = torch.rand(n, device=device)
  hold = u < hold_share
  placed = (u >= hold_share) & (u < hold_share + placed_share)
  kind = getattr(env, "_pp_start_kind", None)
  if kind is None:
    kind = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    setattr(env, "_pp_start_kind", kind)
  kind[env_ids] = torch.where(
    hold, START_KINDS["hold"], torch.where(placed, START_KINDS["placed"], START_KINDS["scratch"])
  )
  _start_home(env, "left", env_ids, hold | placed)
  latch_params = (command_name, object_name, grasp_sensor_name, support_sensor_name, surface_cfg, jaw_cfg, jaw_open)
  if bool(placed.any()):
    _put_a_at_t(env, env_ids[placed], half_xy, yaw, None, None, latch_params)

  state = getattr(env, START_STATE, None)
  term = env.action_manager.get_term("joint_pos")
  if state is None:
    servo = term._servo if term._servo_idx is not None else torch.zeros(env.num_envs, 0, device=device)
    state = {
      "mask": torch.zeros(env.num_envs, dtype=torch.bool, device=device),
      "target": torch.zeros_like(term._last_target),
      "servo": torch.zeros_like(servo),
      "clock": torch.zeros(env.num_envs, dtype=torch.long, device=device),
      "goal_offset": torch.zeros(env.num_envs, 2, device=device),
      "command": target_command_name,
    }
    setattr(env, START_STATE, state)
  state["mask"][env_ids] = hold
  ids = env_ids[hold]
  if len(ids) == 0:
    return
  bank = load_hold_bank(env, bank_path)
  rows = torch.randint(0, len(bank["step"]), (len(ids),), device=device)
  robot = env.scene[ROBOT]
  robot.write_joint_state_to_sim(bank["q"][rows], bank["qd"][rows], env_ids=ids)
  origin = env.scene.env_origins[ids]
  _put_a_at_t(env, ids, 0.0, 0.0, bank["a_pos"][rows, :2], bank["a_quat"][rows], latch_params)
  b = env.scene[CUBE_B]
  b.write_root_link_pose_to_sim(
    torch.cat([origin + bank["b_pos"][rows], bank["b_quat"][rows]], dim=-1), env_ids=ids
  )
  b.write_root_link_velocity_to_sim(bank["b_vel"][rows], env_ids=ids)
  state["target"][ids] = bank["target"][rows]
  if state["servo"].shape[1]:
    state["servo"][ids] = bank["servo"][rows]
  state["clock"][ids] = bank["step"][rows]
  state["goal_offset"][ids] = bank["goal_offset"][rows]


def make_bimanual_env_cfg(
  gains: str = "G20",
  effort: EffortBound = EFFORT_BOUNDS["CTP"],
  dr_level: str = "DR-scene",
  latency: LatencyModelCfg = LATENCY_MODELS["FP+JS"],
  baseline: Baseline = BASELINES["B"],
  episode_length_s: float = EPISODE_S,
  arm_table: ArmTableCfg | None = None,
  handoff_train: str = "left",
  start: str = "scratch",
  critic_release: bool = True,
  play: bool = False,
  reward: str = "S11-BI",
  hold_bank: str | None = None,
  right_target: str = "plain",
) -> ManagerBasedRlEnvCfg:
  """Plan S11's JOINT cfg (BI-J: ``start="scratch"``, BI-JM: ``"mixed"``) and plan S11.2's codes; see the
  module docstring. ``handoff_train`` is ``HANDOFF_TRAIN``: ``left`` the left's up-out-home hand-off on its
  own trigger in training (the eval starts the right's), ``both`` the right's too, ``none`` the controller is
  there for the eval to start. ``hold_bank`` is the snapshot file ``start="hold"`` draws from."""
  if start not in STARTS:
    raise ValueError(f"unknown start {start!r}; have {STARTS}")
  if reward not in REWARDS:
    raise ValueError(f"unknown reward {reward!r}; have {REWARDS}")
  if handoff_train not in HANDOFF_TRAINS:
    raise ValueError(f"unknown handoff_train {handoff_train!r}; have {HANDOFF_TRAINS}")
  if right_target not in ("plain", "gated"):
    raise ValueError(f"unknown right_target {right_target!r}; have ('plain', 'gated')")
  if (start == "hold") != (hold_bank is not None):
    raise ValueError("START=hold needs a snapshot bank (HOLD_BANK), and only it does")
  if not critic_release:
    raise ValueError("JOINT's critic carries both PQ pairs (BS11); CRITIC_OBS=q")
  robot = kangaroo_grippers_both_arms(gains, arm_effort_scale=effort.effort_scale)
  left, right = robot.gripper("left"), robot.gripper("right")
  cfg = make_pick_place_env_cfg(
    robot,
    gripper_name="left",
    spawn=SPAWN_BOXES["W2"],
    dr=dr_level,
    reward=LEFT_REWARD,
    play=False,
    baseline=baseline,
    effort=effort,
    latency=latency,
    episode_length_s=episode_length_s,
    stack_speed=True,
    arm_table=arm_table,
    critic_release=True,
    robot_target=ROBOT_TARGETS["T11"],
    tracker="split",
  )

  # -- Cube B: entity, spawn (mirror(W2), table-relative like W2), DR, sensors, goal. ----------
  cfg.scene.entities = {
    **cfg.scene.entities,
    OBJECT: scene.get_cube_cfg(name=OBJECT, rgba=BASE_RGBA),
    CUBE_B: scene.get_cube_cfg(name=CUBE_B, rgba=CUBE_B_RGBA),
  }
  events = {}
  for name, term in cfg.events.items():
    events[name] = term
    if name == "reset_objects":
      events["reset_cube_b"] = EventTermCfg(
        func=mdp.reset_objects_on_surface,
        mode="reset",
        params={
          "asset_cfgs": [SceneEntityCfg(CUBE_B)],
          "surface_cfg": SceneEntityCfg(TABLE),
          "pose_range": _object_pose_range(W2_MIRROR),
        },
      )
  cfg.events = events
  for name in ("object_friction", "object_mass"):
    term = cfg.events.get(name)
    if term is None:
      continue
    asset = term.params["asset_cfg"]
    cfg.events[f"cube_b_{name.split('_')[1]}"] = dataclasses.replace(
      term,
      params={
        **term.params,
        "asset_cfg": SceneEntityCfg(
          CUBE_B,
          geom_names=(f"{CUBE_B}_geom",) if asset.geom_names else None,
          body_names=(CUBE_B,) if asset.body_names else None,
        ),
      },
    )
  cfg.events["reset_tracker_bias"].params["keys"] = (OBJECT, CUBE_B)
  cfg.scene.sensors = (
    *cfg.scene.sensors,
    make_grasp_sensor(right, CUBE_B),
    _support_sensor(CUBE_B, SUPPORT_SENSOR_B),
    make_hand_ground_sensor(right),
    make_hand_surface_sensor(right),
    _arm_arm_sensor(),
  )
  cfg.commands[COMMAND_B] = mdp.EntityOffsetCommandCfg(
    entity_name=CUBE_B,
    reference_name=OBJECT,
    stack_on_top=True,
    resampling_time_range=(1e9, 1e9),
    success_threshold=GOAL_THRESHOLD,
    debug_vis=True,
  )
  cfg.terminations["cube_b_dropped"] = dataclasses.replace(
    cfg.terminations["object_dropped"],
    params={**cfg.terminations["object_dropped"].params, "entity_name": CUBE_B},
  )
  if "out_of_reach" in cfg.terminations:
    reach = cfg.terminations["out_of_reach"]
    reach.params["entity_names"] = (OBJECT, CUBE_B)

  # -- Observations: the 73-D actor and its critic. -------------------------------------------
  actor = cfg.observations["actor"]
  critic = cfg.observations["critic"]
  a = actor.terms
  tcp_r = SceneEntityCfg(ROBOT, site_names=(right.tcp_site,))
  b_pos = dataclasses.replace(
    a["object_pos"],
    params={**a["object_pos"].params, "object_name": CUBE_B, "bias_key": CUBE_B},
  )
  b_yaw = dataclasses.replace(
    a["object_yaw"], params={**a["object_yaw"].params, "object_name": CUBE_B}
  )
  order = (
    "joint_pos",
    "joint_vel",
    "actions",
    "tcp_pos",
    "tcp_quat",
    "tcp_pos_right",
    "tcp_quat_right",
    "object_pos",
    "object_yaw",
    "object_b_pos",
    "object_b_yaw",
    "target_pos",
  )
  extra = {
    "tcp_pos_right": dataclasses.replace(a["tcp_pos"], params={"asset_cfg": tcp_r}),
    "tcp_quat_right": dataclasses.replace(a["tcp_quat"], params={"asset_cfg": tcp_r}),
    "object_b_pos": b_pos,
    "object_b_yaw": b_yaw,
  }
  actor.terms = {name: a.get(name) or extra[name] for name in order}
  c = critic.terms
  live = {
    "tcp_pos_right": dataclasses.replace(c["tcp_pos"], params={"asset_cfg": tcp_r}),
    "tcp_quat_right": dataclasses.replace(c["tcp_quat"], params={"asset_cfg": tcp_r}),
    "object_b_pos": dataclasses.replace(
      c["object_pos"], params={**c["object_pos"].params, "object_name": CUBE_B}
    ),
    "object_b_yaw": dataclasses.replace(
      c["object_yaw"], params={**c["object_yaw"].params, "object_name": CUBE_B}
    ),
  }
  critic_terms = {name: c.get(name) or live[name] for name in order}
  critic_terms["grasped"] = c["grasped"]
  critic_terms["grasped_right"] = ObservationTermCfg(
    func=mdp.is_grasped, params={"sensor_name": right.contact_sensor_name}
  )

  # -- Rewards: the left half is L-T's; the right half R-ST's, gated on A; HH; AH10 right. ------
  right_rewards = _right_rewards(robot, right)
  gate = _gate_params(cfg.rewards["placed"])
  critic_terms.update(_pq_terms(cfg.rewards["placed"], ""))
  critic_terms.update(_pq_terms(right_rewards["placed"], "right_"))
  critic_terms["base_placed"] = ObservationTermCfg(
    func=mdp.base_placed_obs,
    params={k.removeprefix("gate_"): v for k, v in gate.items()},
  )
  critic_terms["left_handed_off"] = ObservationTermCfg(
    func=mdp.handoff_active, params={"arm": "left"}
  )
  if reward == "S12-BI":
    critic_terms["a_latched"] = ObservationTermCfg(
      func=mdp.a_latched, params={"gate": _gate_dict(gate)}
    )
  critic.terms = critic_terms
  if right_target == "gated":
    actor.terms["right_target"] = ObservationTermCfg(
      func=mdp.right_target_gated,
      params={
        "base_name": OBJECT,
        "object_name": CUBE_B,
        "bias_key": OBJECT,
        "lag": a["object_pos"].params.get("lag"),
        "target_command_name": COMMAND,
        "gate": _gate_dict(gate),
      },
      noise=a["object_pos"].noise,
    )
  for name, term in right_rewards.items():
    if name in GATED_RIGHT:
      term = RewardTermCfg(
        func=mdp.when_base_placed,
        weight=term.weight,
        params={**term.params, "inner": term.func, **gate},
      )
    cfg.rewards[f"right_{name}"] = term
  apply_arm_table(cfg, right, arm_table, prefix="right_")
  cfg.rewards["arm_arm_clearance"] = RewardTermCfg(
    func=mdp.arm_arm_clearance_penalty,
    weight=HH_CLEARANCE_WEIGHT,
    params={"margin": HH_MARGIN},
  )
  cfg.rewards["arm_arm_contact"] = RewardTermCfg(
    func=mdp.contact_penalty,
    weight=HH_CONTACT_WEIGHT,
    params={"sensor_name": ARM_ARM_SENSOR, "force_threshold": HH_CONTACT_FORCE},
  )
  left_placed = cfg.rewards["placed"]
  right_placed = right_rewards["placed"]
  if reward == "S12-BI":
    _apply_ladders(cfg, left, right, left_placed, gate)

  # -- The hand-off controller on both arms; the left's own trigger in training (and the right's). --
  p = left_placed.params
  trigger = _trigger(p, LEFT_HANDOFF_TIMEOUT_S)
  # The right's has no time-out: in training it hands off once it has let go and lifted clear, or never.
  right_trigger = _trigger(right_placed.params, math.inf)
  cfg.actions["joint_pos"] = with_handoff(
    cfg.actions["joint_pos"],
    arms=(
      handoff_arm(left, +1.0, trigger if handoff_train != "none" else None),
      handoff_arm(right, -1.0, right_trigger if handoff_train == "both" else None),
    ),
  )
  if start == "mixed":
    cfg.events["start_a_placed"] = EventTermCfg(
      func=place_a_at_t,
      mode="reset",
      params={
        "share": MIXED_SHARE,
        "half_xy": MIXED_A_HALF_XY,
        "yaw": MIXED_A_YAW,
        "arm": "left",
        **{
          k: p[k]
          for k in (
            "command_name",
            "object_name",
            "grasp_sensor_name",
            "support_sensor_name",
            "surface_cfg",
            "jaw_cfg",
            "jaw_open",
          )
        },
      },
    )
  if start == "hold":
    cfg.events["start_hold"] = EventTermCfg(
      func=hold_start,
      mode="reset",
      params={
        "bank_path": hold_bank,
        "hold_share": HOLD_SHARE,
        "placed_share": PLACED_SHARE,
        "half_xy": MIXED_A_HALF_XY,
        "yaw": MIXED_A_YAW,
        "target_command_name": COMMAND,
        **{
          k: p[k]
          for k in (
            "command_name",
            "object_name",
            "grasp_sensor_name",
            "support_sensor_name",
            "surface_cfg",
            "jaw_cfg",
            "jaw_open",
          )
        },
      },
    )
    for kind, k in START_KINDS.items():
      cfg.metrics[f"start_{kind}"] = MetricsTermCfg(
        func=mdp.start_kind_is, params={"kind": k}, reduce="last"
      )
      if reward == "S12-BI":
        cfg.metrics[f"tower_{kind}"] = MetricsTermCfg(
          func=mdp.tower_of_start, params={"kind": k}, reduce="last"
        )
  if play:
    from pal_mjlab.tasks.manipulation.tabletop_env_cfg import apply_play_overrides

    apply_play_overrides(cfg)
  return cfg


def _gate_dict(gate: dict) -> dict:
  """`_gate_params`' ``gate_*`` keys as `mdp.base_placed_bool`'s own (a nested dict: the latch functions take it
  whole)."""
  return {k.removeprefix("gate_"): v for k, v in gate.items()}


def _trigger(p: dict, timeout_s: float) -> HandoffTriggerCfg:
  """An arm's training hand-off trigger from its ``placed`` term's parameters."""
  return HandoffTriggerCfg(
    command_name=p["command_name"],
    object_name=p["object_name"],
    grasp_sensor_name=p["grasp_sensor_name"],
    support_sensor_name=p["support_sensor_name"],
    surface_cfg=p["surface_cfg"],
    jaw_cfg=p["jaw_cfg"],
    jaw_open=p["jaw_open"],
    timeout_s=timeout_s,
  )


def _apply_ladders(cfg, left, right, left_placed: RewardTermCfg, gate: dict) -> None:
  """Plan S11.2's ``S12-BI`` on a JOINT cfg built with S11-BI: both halves' task terms (`LADDER_REPLACES`, the
  right's SA) give way to the two ladders and A's one-off displacement cost (`mdp.ladder`); HH's clearance
  margin stops at the left's hand-off; metrics for each level's share of the steps. The regularisers, P,
  the trunk and arm-table terms, the hand-table contacts, M8's pad contact after a release and HH's
  contact stay as they are."""
  from pal_mjlab.tasks.manipulation.mdp import ladder

  drop = {*LADDER_REPLACES, *(f"right_{n}" for n in LADDER_REPLACES), "right_stack_align"}
  missing = sorted(n for n in drop if n not in cfg.rewards)
  if missing:
    raise ValueError(f"S12-BI replaces terms this cfg does not have: {missing}")
  for name in drop:
    del cfg.rewards[name]
  g = _gate_dict(gate)
  lp = left_placed.params
  wanted = inspect.signature(mdp.release_quality).parameters
  grade = {k: v for k, v in lp.items() if k in wanted}
  table = SceneEntityCfg(TABLE)
  cfg.rewards["left_ladder"] = RewardTermCfg(
    func=mdp.left_ladder,
    weight=ladder.LADDER_SCALE,
    params={
      "object_name": OBJECT,
      "command_name": COMMAND,
      "tcp_cfg": SceneEntityCfg(ROBOT, site_names=(left.tcp_site,)),
      "grasp_sensor_name": left.contact_sensor_name,
      "support_sensor_name": SUPPORT_SENSOR,
      "surface_cfg": table,
      "jaw_cfg": SceneEntityCfg(ROBOT, joint_names=(re.escape(left.finger_joint),)),
      "jaw_open": JAW_OPEN,
      "grade": grade,
      "gate": g,
    },
  )
  cfg.rewards["right_ladder"] = RewardTermCfg(
    func=mdp.right_ladder,
    weight=ladder.LADDER_SCALE,
    params={
      "object_name": CUBE_B,
      "base_name": OBJECT,
      "command_name": COMMAND_B,
      "tcp_cfg": SceneEntityCfg(ROBOT, site_names=(right.tcp_site,)),
      "grasp_sensor_name": right.contact_sensor_name,
      "support_sensor_name": SUPPORT_SENSOR_B,
      "surface_cfg": table,
      "jaw_cfg": SceneEntityCfg(ROBOT, joint_names=(re.escape(right.finger_joint),)),
      "jaw_open": JAW_OPEN,
      "target_command_name": COMMAND,
      "gate": g,
    },
  )
  cfg.rewards["a_displaced"] = RewardTermCfg(
    func=mdp.a_displaced, weight=ladder.A_MOVED_WEIGHT, params={"gate": g}
  )
  cfg.rewards["arm_arm_clearance"] = RewardTermCfg(
    func=mdp.arm_arm_clearance_before_left_handoff,
    weight=HH_CLEARANCE_WEIGHT,
    params={"margin": HH_MARGIN},
  )
  # The latches clear before any start event can set them.
  cfg.events = {
    "reset_ladder_state": EventTermCfg(func=mdp.reset_ladder_state, mode="reset"),
    **cfg.events,
  }
  for arm, levels in (("right", range(6)), ("left", range(4))):
    for k in levels:
      cfg.metrics[f"{arm}_level_{k}"] = MetricsTermCfg(
        func=mdp.ladder_level_is, params={"arm": arm, "level": k}
      )
  cfg.metrics["a_latched_at_end"] = MetricsTermCfg(
    func=mdp.a_latched_flag,
    params={"gate": g},
    reduce="last",
  )
  cfg.metrics["tower_at_end"] = MetricsTermCfg(
    func=mdp.ladder_level_is, params={"arm": "right", "level": 5}, reduce="last"
  )


def handoff_arm(gripper, side: float, trigger) -> HandoffArmCfg:
  """A gripper's arm for the hand-off controller (`mdp.handoff`)."""
  s = gripper.name
  return HandoffArmCfg(
    name=s,
    arm_joints=tuple(f"arm_{s}_{i}_joint" for i in range(1, 8)),
    jaw_joint=gripper.finger_joint,
    tcp_site=gripper.tcp_site,
    side=side,
    trigger=trigger,
  )
