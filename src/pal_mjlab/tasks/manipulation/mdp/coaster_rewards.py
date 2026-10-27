"""Reward terms of MugCoaster ``-Release``: set the mug down on the coaster, let go, leave.

The stack's ``-Place`` ladder (``object_held_on_base``, ``release_on_base``,
``object_stacked_set_down``) carried over to the mug as it was, and the first policies
held the mug on the coaster to the end. Rolled out at ``mug-coaster-scratch`` 600 on 512
envs: held within 2 cm of the goal 54% of the steps, and only 1% of the envs ever opened
the jaw there. Three things kept it holding on:

- the lift to the goal paid ~20 a step for holding the mug there, for good;
- the set-down latch cleared whenever the mug-coaster contact dropped out for a step --
  24228 times, 99% of them with the mug still at the goal -- so a release found nothing
  to pay it;
- ``stacked`` asked only for no two-face grasp, which the jaw's contacts flicker out of
  while it holds on: 41% of the envs were paid it without opening. No pad touching is
  not enough either: with the jaw shut to 62 mm round the 65 mm mug, the pad sensors
  read nothing for steps at a time, and neither does the hand: rolled out again, ``stacked``
  paid 16 steps an env to the jaw parked shut round the mug, out of contact.

``_PlaceLatch`` keeps the set-down through the contact flicker and counts how long the
mug has stood placed this episode; ``paid_until_placed`` stops the holding rungs once it
has for ``hold_time``, a total that a re-lift does not refill, and ``lift`` once it has paid
``COASTER_LIFT_BUDGET`` in all; ``mug_releasing_on_coaster``
pays the jaw opening and withdrawing; ``mug_placed_on_coaster`` pays the mug standing
there with no part of the hand on it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from pal_mjlab.tasks.manipulation.mdp.events import surface_top_z
from pal_mjlab.tasks.manipulation.mdp.grasp_ladder import (
  _grasped_any_bool,
  _jaw_frame,
  _on_base,
)
from pal_mjlab.tasks.manipulation.mdp.mug_rewards import (
  MUG_WIDTH,
  _hand_touching,
  _ProgressShaping,
  _release_progress,
  mug_tilt_cos,
)

#: How far the jaw midpoint withdraws from the mug before it counts as let go even shut.
_LET_GO_CLEARANCE = 0.08


class _PlaceLatch:
  """Set down on the base in the jaw, at the goal -- and for how many steps in all.

  The mug rests on the base while it touches it or sits within ``rest_z`` of the goal
  height: the contact flag drops out for steps at a time with the mug on the coaster. The
  set-down latches the first step the jaw holds it resting within ``xy_threshold`` of the
  goal, and clears only once it is ``pickup_z`` above the goal or ``keep_xy`` off it --
  picked up again or pushed off. ``placed_steps`` counts the steps latched this episode,
  and never goes back down.
  """

  def __init__(self, env: ManagerBasedRlEnv):
    self.set_down = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.placed_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    # Set by ``reset_from_placed_states`` for the envs it started set down: the reward
    # terms reset after the events, and would clear the set-down it restored.
    self.restored = torch.zeros_like(self.set_down)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self.set_down[ids] = self.restored[ids]
    self.restored[ids] = False
    self.placed_steps[ids] = 0

  def update(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    object_name: str,
    grasp_sensor_names: Sequence[Sequence[str]],
    stack_contact_sensor_name: str,
    xy_threshold: float,
    rest_z: float,
    pickup_z: float,
    keep_xy: float,
  ) -> torch.Tensor:
    """Advance the latch; returns where the mug rests at the goal. (B,)"""
    touching, xy, z = _on_base(
      env, command_name, object_name, stack_contact_sensor_name
    )
    held = _grasped_any_bool(env, grasp_sensor_names)
    at_goal = (touching | (z < rest_z)) & (xy < xy_threshold)
    keep = (z < pickup_z) & (xy < keep_xy)
    self.set_down = (self.set_down | (held & at_goal)) & keep
    self.placed_steps += self.set_down.long()
    return at_goal

  def holding_paid(self, env: ManagerBasedRlEnv, hold_time: float) -> torch.Tensor:
    """False once the mug has stood placed for ``hold_time`` in all this episode. (B,)"""
    return self.placed_steps < round(hold_time / env.step_dt)


class mug_placed_on_coaster:
  """1.0 while the mug, set down on the coaster in the jaw, stands there let go: no
  part of the hand on it (``hand_sensor_name``) and the jaw open past it or withdrawn
  (``_release_progress`` at least ``0.5``); with ``clearance``, the jaw midpoint also that
  far from it. (B,)

  ``_PlaceLatch``. The whole hand and an open jaw, not just no grasp between both pads:
  holding on, the contacts flicker out for steps at a time.

  With ``replay_capacity`` in its params it also keeps the scene as it stood the step
  the jaw first set the mug down, for ``reset_from_placed_states`` to start episodes from.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self.latch = _PlaceLatch(env)
    capacity = (getattr(cfg, "params", None) or {}).get("replay_capacity")
    self.replay = None if capacity is None else _StateReplay(env, capacity)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.latch.reset(env_ids)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    object_name: str,
    grasp_sensor_names: Sequence[Sequence[str]],
    stack_contact_sensor_name: str,
    grasp_cfg: SceneEntityCfg,
    hand_sensor_name: str,
    clearance: float | None = None,
    xy_threshold: float = 0.02,
    rest_z: float = 0.005,
    pickup_z: float = 0.02,
    keep_xy: float = 0.05,
    hold: float = 0.03,
    open_margin: float = 0.005,
    replay_capacity: int | None = None,
    surface_cfg: SceneEntityCfg | None = None,
    object_width: float = MUG_WIDTH,
  ) -> torch.Tensor:
    del replay_capacity  # Read in __init__.
    was_set_down = self.latch.set_down.clone()
    at_goal = self.latch.update(
      env,
      command_name,
      object_name,
      grasp_sensor_names,
      stack_contact_sensor_name,
      xy_threshold,
      rest_z,
      pickup_z,
      keep_xy,
    )
    obj: Entity = env.scene[object_name]
    let_go = (
      _release_progress(
        env,
        grasp_cfg,
        obj,
        _LET_GO_CLEARANCE,
        hold,
        open_margin,
        object_width=object_width,
      )
      >= 0.5
    )
    if self.replay is not None:
      assert surface_cfg is not None, "replay_capacity needs surface_cfg"
      self.replay.capture(env, self.latch.set_down & ~was_set_down, surface_cfg)
    ok = self.latch.set_down & at_goal & let_go & ~_hand_touching(env, hand_sensor_name)
    if clearance is not None:
      mid, _, _ = _jaw_frame(env.scene[grasp_cfg.name], grasp_cfg)
      ok &= torch.norm(mid[:, 0] - obj.data.root_link_pos_w, dim=-1) > clearance
    return ok.float()


class mug_releasing_on_coaster(mug_placed_on_coaster):
  """How far the jaw has let go of the mug set down on the coaster: ``0.5`` open on it,
  rising to ``1`` withdrawn with the hand off it (``mug_rewards._release_progress``), from
  the step the jaw no longer holds it between both faces. (B,)"""

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    object_name: str,
    grasp_sensor_names: Sequence[Sequence[str]],
    stack_contact_sensor_name: str,
    grasp_cfg: SceneEntityCfg,
    hand_sensor_name: str,
    clearance: float = 0.08,
    hold: float = 0.03,
    open_margin: float = 0.005,
    xy_threshold: float = 0.02,
    rest_z: float = 0.005,
    pickup_z: float = 0.02,
    keep_xy: float = 0.05,
    replay_capacity: int | None = None,
    surface_cfg: SceneEntityCfg | None = None,
    object_width: float = MUG_WIDTH,
  ) -> torch.Tensor:
    del replay_capacity, surface_cfg  # The placed term keeps the replay.
    at_goal = self.latch.update(
      env,
      command_name,
      object_name,
      grasp_sensor_names,
      stack_contact_sensor_name,
      xy_threshold,
      rest_z,
      pickup_z,
      keep_xy,
    )
    free = self.latch.set_down & at_goal & ~_grasped_any_bool(env, grasp_sensor_names)
    obj: Entity = env.scene[object_name]
    return free.float() * _release_progress(
      env, grasp_cfg, obj, clearance, hold, open_margin, hand_sensor_name, object_width
    )


class paid_until_placed:
  """``func``'s reward until the mug has stood placed for ``hold_time`` this episode, and,
  with ``budget``, until it has paid ``budget`` seconds' worth of its full reward. (B,)

  The placement is ``placed_term``'s ``_PlaceLatch``, read as of the last step that term
  ran. For the rungs that pay holding the mug at the goal: without an end to them, holding
  on paid as much as letting go, for good.

  ``budget`` because the placed clock alone left a way round it: ``lift``'s 15 cm wide
  bell pays 80% of it 7 cm off the goal, where the mug never rests and the clock never
  starts. Fine-tuned from ``mug-coaster-scratch`` 800, the policy moved there within 300
  iterations -- position error 3.7 to 7.4 cm, at the goal 38% to 1% of the steps, ``lift``
  2.2 to 9.4 -- and stopped setting the mug down (placed in 64% of the episodes, then
  8%). A total on what the term pays caps hovering at any distance.

  ``after_budget`` is the share it pays once the budget is spent. ``lift`` is the only
  rung that pulls the carried mug toward the goal, and cut to nothing it left none: from
  scratch the budget ran out 5 to 10 s into the carry, the mug was at the goal 13% of the
  steps where it had been at 50%, and 71% of the mugs dropped in the air, 7 cm off the goal,
  had spent it. A quarter of it hovering pays 4/s at most, against ``stacked``'s 60.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self.paid = torch.zeros(env.num_envs, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.paid[slice(None) if env_ids is None else env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    func: Callable[..., torch.Tensor],
    placed_term: str,
    hold_time: float,
    budget: float | None = None,
    after_budget: float = 0.0,
    **params,
  ) -> torch.Tensor:
    latch: _PlaceLatch = env.reward_manager.get_term_cfg(placed_term).func.latch
    reward = func(env, **params) * latch.holding_paid(env, hold_time).float()
    if budget is not None:
      within = self.paid < budget
      self.paid += reward * within.float() * env.step_dt
      reward = torch.where(within, reward, reward * after_budget)
    return reward


# -- Replay of set-down states (``-Replay``) ---------------------------------------------
# The release is the last of seven stages, and before ``-Release`` 1% of the envs ever
# opened the jaw at the goal: PPO saw it too seldom to learn it, and every fix to the
# rungs below cost another few thousand iterations to get back there. Some episodes
# start instead from a state the policy itself reached, the mug just set down on the
# coaster in the jaw, so the release is practised from the first iteration it is reached.


class _StateReplay:
  """A ring buffer of whole-scene states (``qpos``, ``qvel``) and the table height each
  was taken over. Every env shares the one origin on the floating-base robot."""

  def __init__(self, env: ManagerBasedRlEnv, capacity: int):
    data = env.sim.data
    self.qpos = torch.zeros(capacity, data.qpos.shape[-1], device=env.device)
    self.qvel = torch.zeros(capacity, data.qvel.shape[-1], device=env.device)
    self.table_z = torch.zeros(capacity, device=env.device)
    self.capacity = capacity
    self.count = 0
    self.head = 0

  def capture(
    self, env: ManagerBasedRlEnv, mask: torch.Tensor, surface_cfg: SceneEntityCfg
  ) -> None:
    ids = mask.nonzero().squeeze(-1)[: self.capacity]
    n = len(ids)
    if n == 0:
      return
    slots = (self.head + torch.arange(n, device=ids.device)) % self.capacity
    self.qpos[slots] = env.sim.data.qpos[ids]
    self.qvel[slots] = env.sim.data.qvel[ids]
    self.table_z[slots] = surface_top_z(env, surface_cfg)[ids]
    self.head = (self.head + n) % self.capacity
    self.count = min(self.count + n, self.capacity)


def reset_from_placed_states(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  placed_term: str,
  object_names: Sequence[str],
  surface_cfg: SceneEntityCfg,
  probability: float,
  min_states: int = 256,
  candidates: int = 16,
  max_table_dz: float = 0.005,
) -> None:
  """Start ``probability`` of the resetting envs from a replayed set-down state. Last
  among the reset events, after the settled stance it overwrites.

  The table height is drawn per env at startup, +/-1.5 cm: of ``candidates`` stored
  states the one taken over the nearest table height is used, if within
  ``max_table_dz``, with ``object_names`` (the mug, the coaster) moved by the difference;
  the robot stands on the floor and keeps its pose. Every ``_PlaceLatch`` of the reward
  terms then starts set down, as the state was when it was taken.
  """
  term = env.reward_manager.get_term_cfg(placed_term).func
  replay: _StateReplay | None = term.replay
  if replay is None or replay.count < min_states or env_ids is None:
    return
  device = env.device
  ids = env_ids[torch.rand(len(env_ids), device=device) < probability]
  if len(ids) == 0:
    return
  pick = torch.randint(0, replay.count, (len(ids), candidates), device=device)
  table_z = surface_top_z(env, surface_cfg)[ids]
  dz = table_z.unsqueeze(-1) - replay.table_z[pick]
  best = dz.abs().argmin(dim=-1)
  rows = torch.arange(len(ids), device=device)
  slot, dz = pick[rows, best], dz[rows, best]
  ok = dz.abs() < max_table_dz
  ids, slot, dz = ids[ok], slot[ok], dz[ok]
  if len(ids) == 0:
    return
  data = env.sim.data
  data.qpos[ids] = replay.qpos[slot]
  data.qvel[ids] = replay.qvel[slot]
  env.sim.forward()
  for name in object_names:
    obj: Entity = env.scene[name]
    pose = obj.data.root_link_pose_w[ids].clone()
    pose[:, 2] += dz
    obj.write_root_link_pose_to_sim(pose, env_ids=ids)
  env.sim.forward()
  rewards = env.reward_manager
  for name in rewards.active_terms:
    latch = getattr(rewards.get_term_cfg(name).func, "latch", None)
    if isinstance(latch, _PlaceLatch):
      latch.restored[ids] = True


def termination_fired(
  env: ManagerBasedRlEnv, term_names: Sequence[str]
) -> torch.Tensor:
  """``1 / step_dt`` on the step any of the termination terms ``term_names`` ends the
  episode, else 0: the weight is what the event costs, not a per-second rate. (B,)

  Ending the episode is no deterrent by itself where the approach pays little: at gamma
  0.995 the return looks ~4 s ahead, and the grasp ladder pays under 8 a second, so
  pushing the mug over cost the coaster policy ~20 -- and 84% of its tip-overs were a
  finger levering the tall mug over before it was ever grasped.
  """
  terms = env.termination_manager
  fired = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  for name in term_names:
    fired |= terms.get_term(name)
  return fired.float() / env.step_dt


class coaster_carry_progress(_ProgressShaping):
  """Carrying the mug to the coaster, as potential-based shaping. (B,)

  ``phi = exp(-d / std)`` of the mug's distance ``d`` to the goal, while the jaw holds it
  or it stands on the coaster (so letting go there keeps ``phi``), else 0. ``lift`` pays
  only its budget's 3 s and ``set_down`` only on the coaster, and between them nothing
  did: deterministic rollouts dropped the mug in the air in 22% of the episodes, ~7 cm off
  the goal. Here every centimetre carried pays, hovering costs ``(1 - gamma) * phi`` a
  step, and a drop pays back ``phi``: ~0.6 of ``weight`` 7 cm off. Reads only the state,
  so the critic needs no latches and a checkpoint fine-tunes onto it.
  """

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    object_name: str,
    grasp_sensor_names: Sequence[Sequence[str]],
    stack_contact_sensor_name: str,
    gamma: float,
    std: float = 0.15,
    stand_tilt: float = 0.35,
  ) -> torch.Tensor:
    obj: Entity = env.scene[object_name]
    command = env.command_manager.get_term(command_name)
    d = torch.norm(command.command - obj.data.root_link_pos_w, dim=-1)
    held = _grasped_any_bool(env, grasp_sensor_names)
    touching, _, _ = _on_base(env, command_name, object_name, stack_contact_sensor_name)
    standing = touching & (mug_tilt_cos(obj) > math.cos(stand_tilt))
    phi = (held | standing).float() * torch.exp(-d / std)
    return self._shape(env, phi, gamma)
