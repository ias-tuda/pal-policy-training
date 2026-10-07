"""Plan S11.2's reward ``S12-BI``: stage-replacing ladders for JOINT (`pick-place-s11-bimanual-plan.md`, *S11.2*).

S11-BI added the right's stack terms and gated them, every step, on "A placed" (`bimanual.when_base_placed`).
Step 0's ledger (S11.2 item 1, `scripts/s112_step0.py ledger`) found the gate dropping in 63 of 64 SEQ-chain
episodes once the right started, almost always because A moved when B came down on it (`a_still`): the
right's income and the left's `placed` went to 0 for those steps. ``S12-BI`` replaces both halves' task
terms by a ladder each, and couples them through a latch instead of a per-step gate:

* **A_latched** turns true the first time the old gate holds (`base_placed_bool`) and stays true for the
  episode. Its reference is A's pose then; A pushed more than `A_MOVED` from it costs once
  (`a_displaced`, weighted -w_A) and the reference moves with it, so a further push is a new event.
* **The right's ladder** (`right_ladder`), each step the highest level whose condition holds; the shaping
  inside a level is scaled below the gap to the next floor, so any higher level beats any lower one:

  ======  ==========================================================  =================================
  level   condition                                                   reward
  ======  ==========================================================  =================================
  R0      default                                                     0.9 norm(reach + grasp + 2 lift)
  R1      B held and lifted, A not latched                            1 + 0.5 G(TCP to the wait point)
  R2      B held, A latched                                           2 + 0.45 (near + align) / 2
  R3      B on A's current top (xy 1.5 cm, z 1 cm, tilt, yaw 20 deg)  4 + 0.45 open(jaw)
  R4      R3, the jaw open (J) and no right pad on B                  6 + 0.45 retreat
  R5      R4, B still, A at T, both arms handed off                   8
  ======  ==========================================================  =================================

  R3-R5 need A latched too. "near" and "align" are relative to A's *current* pose (`goal_b` follows A), so
  a nudged A does not drop the right a level.
* **The left's ladder** (`left_ladder`): L0 fetch, L1 carry, L2 at T ready to open, L3 placed and leaving
  (L-T's M6 + YG grade of the let-go and M8's retreat) -- **frozen** at its value on the step the left's
  hand-off starts, so nothing the right does afterwards can take the left's income away.

Each ladder is one reward term (weight `LADDER_SCALE`); the level per env is kept on the env for the
metrics (`ladder_level_is`). Everything here is per step and stateless but for the latches, which
`reset_ladder_state` clears at a reset (it must come before any start event that sets them).
"""

from __future__ import annotations

import math

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

#: The ladders' weight (plan S11.2 decision 1: every gap between adjacent levels >= 2 x the largest per-step
#: penalty the right takes while stacking, as step 0's ledger measures it). Ledger (SEQ chain, 64 episodes, S11-BI,
#: 11 160 steps with the right driven and not handed off; reward per step / dt): the right-side penalties sum to
#: p99 5.6 (HH clearance 5.0 in 24 % of the steps, the effort hinge up to 12.1 in 1 %). The stacking levels
#: R2 -> R5 are 2 floors apart, so x 6 makes each gap 12 >= 2 x 5.6 (and >= the largest effort spike).
LADDER_SCALE = 6.0
#: w_A, the one-off cost of pushing A off its latched pose: one level gap (decision 1), 2 floors x `LADDER_SCALE`.
A_MOVED_WEIGHT = -2.0 * LADDER_SCALE
#: A pushed this far from its reference is one displacement event, m.
A_MOVED = 0.01
#: The right's floors and its shaping per level (R0 ... R5).
RIGHT_FLOORS = (0.0, 1.0, 2.0, 4.0, 6.0, 8.0)
#: The left's floors (L0 ... L3).
LEFT_FLOORS = (0.0, 1.0, 2.0, 3.0)
SHAPE = 0.45
#: R1's wait point: T + this, robot frame, m -- 10 cm over T and 10 cm to the right's side, out of the left's path.
WAIT_OFFSET = (0.0, -0.10, 0.10)
WAIT_STD = 0.10
#: R2's kernels: B to A's top (m), tilt and yaw (rad), wide.
NEAR_STD = 0.05
ALIGN_STD = math.radians(30.0)
#: R3's door: B on A's top within these (m, m, rad, rad).
STACK_DXY = 0.015
STACK_DZ = 0.01
STACK_TILT = math.radians(20.0)
STACK_YAW = math.radians(20.0)
#: S7-5's O shape: the jaw's opening from the hold angle to J's threshold, rad.
OPEN_HOLD = 0.40
#: R4's retreat: the TCP's height over B's top, saturating here, m (RGB-Stacking's leave level, ~10 cm).
RETREAT_HEIGHT = 0.10
#: B still, m/s (the gate's `velocity_threshold`).
STILL = 0.05
#: L2's door: A at T within these (m, m).
AT_T_DXY = 0.015
AT_T_DZ = 0.01
#: L1's kernel, A to T, m.
CARRY_STD = 0.05

#: Env attributes.
_A_LATCHED = "_pp_a_latched"
_A_REF = "_pp_a_ref"
_A_STEP = "_pp_a_latch_step"
_LEFT_FROZEN = "_pp_left_frozen"
_RIGHT_FROZEN = "_pp_right_frozen"
_LEFT_VALUE = "_pp_left_frozen_value"
_LEVEL = "_pp_ladder_level"


def _buf(env: ManagerBasedRlEnv, name: str, like: torch.Tensor) -> torch.Tensor:
  value = getattr(env, name, None)
  if value is None:
    value = torch.zeros_like(like)
    setattr(env, name, value)
  return value


_ROBOT = SceneEntityCfg("robot")


def _resolve(env: ManagerBasedRlEnv, params: dict) -> dict:
  """Resolve the entity cfgs inside a nested parameter dict (``gate``, ``grade``): the managers resolve
  top-level ones only. Every call -- resolving is idempotent and cheap, and a cache keyed on the object would
  outlive it (a freed cfg's ``id`` is reused)."""
  for v in params.values():
    if isinstance(v, SceneEntityCfg):
      v.resolve(env.scene)
  return params


def _g(x: torch.Tensor, std: float) -> torch.Tensor:
  return torch.exp(-torch.square(x / std))


def _open(
  env: ManagerBasedRlEnv, jaw_cfg: SceneEntityCfg, jaw_open: float
) -> torch.Tensor:
  q = env.scene[jaw_cfg.name].data.joint_pos[:, jaw_cfg.joint_ids][:, 0]
  return ((OPEN_HOLD - q) / (OPEN_HOLD - jaw_open)).clamp(0.0, 1.0)


def _phase(env: ManagerBasedRlEnv, arm: str) -> torch.Tensor:
  term = env.action_manager.get_term("joint_pos")
  return term.phase[:, term.arm_index(arm)]


def _fetch(
  env: ManagerBasedRlEnv,
  object_name: str,
  tcp_cfg: SceneEntityCfg,
  grasp_sensor_name: str,
  surface_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """0.9 x the fetch stage's terms -- R-C-strict's reach, grasp and bounded lift (weights 1, 1, 2) -- over
  their maximum: [0, 0.9]. (B,)"""
  from pal_mjlab.tasks.manipulation.mdp.observations import _grasped_bool
  from pal_mjlab.tasks.manipulation.mdp.rewards import lift_height, reach_only

  reach = reach_only(env, object_name, 0.2, tcp_cfg)
  grasp = _grasped_bool(env, grasp_sensor_name).float()
  lift = lift_height(env, grasp_sensor_name, object_name, surface_cfg, 0.10)
  return 0.9 * (reach + grasp + 2.0 * lift) / 4.0


def update_a_latch(
  env: ManagerBasedRlEnv, gate: dict
) -> tuple[torch.Tensor, torch.Tensor]:
  """A_latched and the displacement event of this step: (latched (B,), moved (B,)). Updated once per env step
  (the first reader of the step; `common_step_counter` guards a second read), so every term sees the same
  latch. ``gate`` is `bimanual.base_placed_bool`'s parameters."""
  from pal_mjlab.tasks.manipulation.mdp.bimanual import base_placed_bool

  a = env.scene[gate["object_name"]].data.root_link_pos_w
  latched = _buf(
    env, _A_LATCHED, torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  )
  ref = _buf(env, _A_REF, a)
  step = getattr(env, _A_STEP, None)
  moved_attr = "_pp_a_moved"
  if step == env.common_step_counter and hasattr(env, moved_attr):
    return latched, getattr(env, moved_attr)
  placed = base_placed_bool(env, **_resolve(env, gate))
  fresh = placed & ~latched
  ref[fresh] = a[fresh]
  latched |= fresh
  moved = latched & (torch.norm(a - ref, dim=-1) > A_MOVED)
  ref[moved] = a[moved]
  setattr(env, moved_attr, moved)
  setattr(env, _A_STEP, env.common_step_counter)
  return latched, moved


def a_latched(env: ManagerBasedRlEnv, gate: dict) -> torch.Tensor:
  """1 once A has been placed this episode (the latch), as a critic input. (B, 1)"""
  latched, _ = update_a_latch(env, gate)
  return latched.float().unsqueeze(-1)


def a_latched_flag(env: ManagerBasedRlEnv, gate: dict) -> torch.Tensor:
  """`a_latched` as a metric. (B,)"""
  latched, _ = update_a_latch(env, gate)
  return latched.float()


def a_displaced(env: ManagerBasedRlEnv, gate: dict) -> torch.Tensor:
  """1 on the step A is pushed `A_MOVED` off its latched reference (one event; the reference then follows A),
  else 0. Weight it -w_A. (B,)"""
  _, moved = update_a_latch(env, gate)
  return moved.float()


def _stack_geometry(
  env: ManagerBasedRlEnv, object_name: str, base_name: str, command_name: str
) -> dict[str, torch.Tensor]:
  from pal_mjlab.tasks.manipulation.mdp.stack import stack_state

  st = stack_state(env, object_name, base_name, command_name)
  goal = env.command_manager.get_term(command_name).command
  dz = env.scene[object_name].data.root_link_pos_w[:, 2] - goal[:, 2]
  on = (
    (st.dxy <= STACK_DXY)
    & (dz.abs() <= STACK_DZ)
    & (st.tilt <= STACK_TILT)
    & (st.yaw <= STACK_YAW)
  )
  return {"st": st, "dz": dz, "on": on}


def right_levels(
  env: ManagerBasedRlEnv,
  object_name: str,
  base_name: str,
  command_name: str,
  tcp_cfg: SceneEntityCfg,
  grasp_sensor_name: str,
  support_sensor_name: str,
  surface_cfg: SceneEntityCfg,
  jaw_cfg: SceneEntityCfg,
  jaw_open: float,
  target_command_name: str,
  gate: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
  """The right's (level (B,) long, reward (B,)) of the ladder in the module docstring."""
  from pal_mjlab.tasks.manipulation.mdp.observations import (
    _at_goal,
    _grasped_bool,
    _pad_contact_any,
    _was_lifted_bool,
  )

  latched, _ = update_a_latch(env, gate)
  robot = env.scene[tcp_cfg.name]
  tcp = robot.data.site_pos_w[:, tcp_cfg.site_ids].squeeze(1)
  b = env.scene[object_name]
  held = _grasped_bool(env, grasp_sensor_name)
  lifted = _was_lifted_bool(env, grasp_sensor_name, support_sensor_name)
  geo = _stack_geometry(env, object_name, base_name, command_name)
  st = geo["st"]
  jaw_q = robot.data.joint_pos[:, jaw_cfg.joint_ids][:, 0]
  released = (jaw_q < jaw_open) & ~_pad_contact_any(env, grasp_sensor_name)
  above = tcp[:, 2] - (b.data.root_link_pos_w[:, 2] + 0.025)
  t_cmd = env.command_manager.get_term(target_command_name).command
  a_at_t = _at_goal(t_cmd, env.scene[base_name].data.root_link_pos_w, gate["threshold"])
  handed = (_phase(env, "left") > 0) & (_phase(env, "right") > 0)

  r5 = latched & geo["on"] & released & (st.speed < STILL) & a_at_t & handed
  r4 = latched & geo["on"] & released
  r3 = latched & geo["on"]
  r2 = latched & held & lifted
  r1 = ~latched & held & lifted
  level = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
  for k, cond in ((1, r1), (2, r2), (3, r3), (4, r4), (5, r5)):
    level = torch.where(cond, k, level)

  wait = t_cmd + torch.tensor(WAIT_OFFSET, device=env.device)
  shaping = torch.stack(
    [
      _fetch(env, object_name, tcp_cfg, grasp_sensor_name, surface_cfg),
      0.5 * _g(torch.norm(tcp - wait, dim=-1), WAIT_STD),
      SHAPE
      * 0.5
      * (
        _g(torch.norm(torch.stack([st.dxy, geo["dz"]], -1), dim=-1), NEAR_STD)
        + _g(st.tilt, ALIGN_STD) * _g(st.yaw, ALIGN_STD)
      ),
      SHAPE * _open(env, jaw_cfg, jaw_open),
      SHAPE * (above / RETREAT_HEIGHT).clamp(0.0, 1.0),
      torch.zeros(env.num_envs, device=env.device),
    ],
    dim=-1,
  )
  floors = torch.tensor(RIGHT_FLOORS, device=env.device)
  reward = floors[level] + shaping.gather(1, level.unsqueeze(-1)).squeeze(-1)
  return level, reward


def right_ladder(env: ManagerBasedRlEnv, **params) -> torch.Tensor:
  """The right's ladder (`right_levels`); keeps the level per env for the metrics. A hand-off started by the right's
  time-out (decision 5: a safety net) freezes the ladder at the level and value reached then, so the jaw it opens
  pays no release, leave or success. (B,)"""
  level, reward = right_levels(env, **params)
  term = env.action_manager.get_term("joint_pos")
  timed_out = term.by_timeout[:, term.arm_index("right")]
  frozen = _buf(
    env, _RIGHT_FROZEN, torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  )
  value = _buf(env, "_pp_right_frozen_value", reward)
  frozen_level = _buf(env, "_pp_right_frozen_level", level)
  fresh = timed_out & ~frozen
  value[fresh] = reward[fresh]
  frozen_level[fresh] = level[fresh]
  frozen |= fresh
  _levels(env)["right"] = torch.where(frozen, frozen_level, level)
  return torch.where(frozen, value, reward)


def left_levels(
  env: ManagerBasedRlEnv,
  object_name: str,
  command_name: str,
  tcp_cfg: SceneEntityCfg,
  grasp_sensor_name: str,
  support_sensor_name: str,
  surface_cfg: SceneEntityCfg,
  jaw_cfg: SceneEntityCfg,
  jaw_open: float,
  grade: dict,
  gate: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
  """The left's (level, reward) before its freeze: L0 fetch, L1 carry (A held and lifted; G(A to T)), L2 at T
  ready to open (A within `AT_T_DXY` / `AT_T_DZ` of T, lifted; O), L3 placed (the gate) and leaving
  (M8's retreat and the let-go's M6 + YG grade, ``grade`` = the L-T ``placed`` term's parameters)."""
  from pal_mjlab.tasks.manipulation.mdp.bimanual import base_placed_bool
  from pal_mjlab.tasks.manipulation.mdp.observations import (
    _grasped_bool,
    _was_lifted_bool,
  )
  from pal_mjlab.tasks.manipulation.mdp.rewards import release_quality

  robot = env.scene[tcp_cfg.name]
  tcp = robot.data.site_pos_w[:, tcp_cfg.site_ids].squeeze(1)
  a = env.scene[object_name].data.root_link_pos_w
  t_cmd = env.command_manager.get_term(command_name).command
  d = a - t_cmd
  dxy = torch.norm(d[:, :2], dim=-1)
  held = _grasped_bool(env, grasp_sensor_name)
  lifted = _was_lifted_bool(env, grasp_sensor_name, support_sensor_name)
  placed = base_placed_bool(env, **_resolve(env, gate))
  at_t = lifted & (dxy <= AT_T_DXY) & (d[:, 2].abs() <= AT_T_DZ)
  level = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
  for k, cond in ((1, held & lifted), (2, at_t), (3, placed)):
    level = torch.where(cond, k, level)
  above = tcp[:, 2] - (a[:, 2] + 0.025)
  q = release_quality(env, **_resolve(env, grade))
  floor = grade.get("floor", 0.25)
  shaping = torch.stack(
    [
      _fetch(env, object_name, tcp_cfg, grasp_sensor_name, surface_cfg),
      SHAPE * _g(torch.norm(d, dim=-1), CARRY_STD),
      SHAPE * _open(env, jaw_cfg, jaw_open),
      SHAPE
      * 0.5
      * (
        (above / RETREAT_HEIGHT).clamp(0.0, 1.0)
        + ((q - floor) / (1.0 - floor)).clamp(0.0, 1.0)
      ),
    ],
    dim=-1,
  )
  floors = torch.tensor(LEFT_FLOORS, device=env.device)
  reward = floors[level] + shaping.gather(1, level.unsqueeze(-1)).squeeze(-1)
  return level, reward


def left_ladder(env: ManagerBasedRlEnv, **params) -> torch.Tensor:
  """The left's ladder, frozen at its value on the step the left's hand-off starts (phase > policy). (B,)"""
  level, reward = left_levels(env, **params)
  frozen = _buf(
    env, _LEFT_FROZEN, torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  )
  value = _buf(env, _LEFT_VALUE, reward)
  frozen_level = _buf(env, "_pp_left_frozen_level", level)
  fresh = (_phase(env, "left") > 0) & ~frozen
  value[fresh] = reward[fresh]
  frozen_level[fresh] = level[fresh]
  frozen |= fresh
  _levels(env)["left"] = torch.where(frozen, frozen_level, level)
  return torch.where(frozen, value, reward)


def _levels(env: ManagerBasedRlEnv) -> dict:
  levels = getattr(env, _LEVEL, None)
  if levels is None:
    levels = {}
    setattr(env, _LEVEL, levels)
  return levels


def ladder_level_is(env: ManagerBasedRlEnv, arm: str, level: int) -> torch.Tensor:
  """1 while ``arm``'s ladder is at ``level`` (as last computed by the reward), a metric: its episode mean is
  the share of steps at that level. (B,)"""
  lv = _levels(env).get(arm)
  if lv is None:
    return torch.zeros(env.num_envs, device=env.device)
  return (lv == level).float()


def arm_arm_clearance_before_left_handoff(
  env: ManagerBasedRlEnv, margin: float
) -> torch.Tensor:
  """S11's HH clearance penalty until the left's hand-off starts, 0 after (plan S11.2: the contact term only,
  so the margin cannot hold the right above the tower). (B,)"""
  from pal_mjlab.tasks.manipulation.mdp.bimanual import arm_arm_clearance_penalty

  return arm_arm_clearance_penalty(env, margin) * (_phase(env, "left") == 0).float()


def reset_ladder_state(env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> None:
  """Reset event: clear A_latched, its reference and the left's freeze. Must run before any start event that
  sets them (`bimanual.hold_start`)."""
  for name in (_A_LATCHED, _LEFT_FROZEN, _RIGHT_FROZEN):
    buf = getattr(env, name, None)
    if buf is not None:
      buf[env_ids] = False
  setattr(env, _A_STEP, None)


def right_target_gated(
  env: ManagerBasedRlEnv,
  base_name: str,
  object_name: str,
  bias_key: str,
  lag: tuple[int, int] | None,
  target_command_name: str,
  gate: dict,
  asset_cfg: SceneEntityCfg = _ROBOT,
  left_home: bool = False,
) -> torch.Tensor:
  """Plan S11.2's BI-F: the right's goal as the actor sees it, gated on A_latched -- the wait point (T +
  `WAIT_OFFSET`) before, A's tracked top after (R-ST's own `target_pos`, `stack_goal_position_tracked_b`),
  base frame. Without the gate R-ST would set B down on the empty T. ``left_home`` (plan S11.4's
  ``RIGHT_GATE=clear``): the wait point until A_latched and the left's hand-off on its home leg. (B, 3)"""
  from pal_mjlab.tasks.manipulation.mdp.bimanual import stack_goal_position_tracked_b
  from pal_mjlab.tasks.manipulation.mdp.observations import _to_base_frame

  latched, _ = update_a_latch(env, gate)
  if left_home:
    from pal_mjlab.tasks.manipulation.mdp.bimanual import handoff_home_bool

    latched = latched & handoff_home_bool(env, "left")
  stack = stack_goal_position_tracked_b(
    env, base_name, object_name, bias_key, asset_cfg, lag
  )
  wait_w = env.command_manager.get_term(target_command_name).command + torch.tensor(
    WAIT_OFFSET, device=env.device
  )
  wait = _to_base_frame(env.scene[asset_cfg.name], wait_w)
  return torch.where(latched.unsqueeze(-1), stack, wait)


def start_kind_is(env: ManagerBasedRlEnv, kind: int) -> torch.Tensor:
  """1 when this episode started as ``kind`` (`bimanual.START_KINDS`: 0 scratch, 1 A placed, 2 hold), a metric.
  (B,)"""
  k = getattr(env, "_pp_start_kind", None)
  if k is None:
    return torch.full((env.num_envs,), float(kind == 0), device=env.device)
  return (k == kind).float()


def tower_standing(
  env: ManagerBasedRlEnv, gate: dict, right_gate: dict
) -> torch.Tensor:
  """1 while the tower stands with both hands handed off -- A placed at T (the left's gate), B placed on A (the
  right's J-placed predicate on its stack goal, `eval_bimanual`'s right gate) and both arms past the policy phase --
  whatever the reward; a metric (``reduce="last"``: the share of episodes that end with it). (B,)"""
  from pal_mjlab.tasks.manipulation.mdp.bimanual import base_placed_bool

  a = base_placed_bool(env, **_resolve(env, gate))
  b = base_placed_bool(env, **_resolve(env, right_gate))
  return (a & b & (_phase(env, "left") > 0) & (_phase(env, "right") > 0)).float()


def tower_of_start(
  env: ManagerBasedRlEnv, kind: int, gate: dict, right_gate: dict
) -> torch.Tensor:
  """`tower_standing` in an episode that started as ``kind`` (hold starts reported apart). (B,)"""
  return tower_standing(env, gate, right_gate) * start_kind_is(env, kind)
