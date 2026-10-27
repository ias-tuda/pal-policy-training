"""Reward terms of the mug tasks: stand it up, turn its handle round, hand it over.

The mug is ``mug_scene.get_mug_spec``: its local z is the body axis and its local x points
at the handle. The grasp ladder of the cube tasks carries over unchanged -- it scores the
jaw on the body centre, which a jaw closing across the body straddles -- and these terms
are the rungs above it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

from pal_mjlab.tasks.manipulation.mdp.events import surface_top_z
from pal_mjlab.tasks.manipulation.mdp.grasp_ladder import (
  _grasped_any_bool,
  _grasped_pair_bool,
  _jaw_frame,
  object_lifted_pair,
)
from pal_mjlab.tasks.manipulation.mdp.metrics import reward_term_active
from pal_mjlab.tasks.manipulation.mdp.observations import _on_surface_bool
from pal_mjlab.tasks.manipulation.mug_scene import (
  MUG_HALF_HEIGHT,
  MUG_HANDLE_BOXES,
  MUG_WIDTH,
)

#: The handle's outer bar, in the mug frame, and the jaw aperture that holds it.
_HANDLE_BAR = MUG_HANDLE_BOXES[2][1]
_HANDLE_WIDTH = 2 * MUG_HANDLE_BOXES[2][0][1]


def _axis_w(obj: Entity, axis: tuple[float, float, float]) -> torch.Tensor:
  quat = obj.data.root_link_quat_w
  return quat_apply(
    quat, torch.tensor(axis, device=quat.device).expand(quat.shape[0], 3)
  )


def mug_tilt_cos(obj: Entity) -> torch.Tensor:
  """Cosine of the mug body axis's angle from vertical: 1 upright, 0 lying. (B,)"""
  return _axis_w(obj, (0.0, 0.0, 1.0))[:, 2]


def _stand_cos(obj: Entity, either_end: bool) -> torch.Tensor:
  """``mug_tilt_cos``, or with ``either_end`` its absolute value: upside down -- the
  opening on the table -- counts as standing too. The body is a plain cylinder, so it
  stands as well on either end. (B,)"""
  cos = mug_tilt_cos(obj)
  return cos.abs() if either_end else cos


def _upright_gate(obj: Entity, max_tilt: float | None) -> torch.Tensor | float:
  """1 upright, falling linearly in the tilt's cosine to 0 at ``max_tilt``; 1 throughout
  without one. (B,)"""
  if max_tilt is None:
    return 1.0
  cos_limit = math.cos(max_tilt)
  return ((mug_tilt_cos(obj) - cos_limit) / (1.0 - cos_limit)).clamp(0.0, 1.0)


def handle_heading_error(obj: Entity, target_yaw: float) -> torch.Tensor:
  """Angle between the handle's heading in the world xy plane and ``target_yaw``. (B,)"""
  x = _axis_w(obj, (1.0, 0.0, 0.0))
  yaw = torch.atan2(x[:, 1], x[:, 0])
  return torch.remainder(yaw - target_yaw + torch.pi, 2 * torch.pi).sub(torch.pi).abs()


def _jaw_clear(
  env: ManagerBasedRlEnv, grasp_cfg: SceneEntityCfg, obj: Entity, clearance: float
) -> torch.Tensor:
  """True where every jaw is at least ``clearance`` from the mug centre. (B,)"""
  mid, _, _ = _jaw_frame(env.scene[grasp_cfg.name], grasp_cfg)
  dist = torch.norm(mid - obj.data.root_link_pos_w.unsqueeze(1), dim=-1)
  return (dist > clearance).all(dim=-1)


def _hand_touching(
  env: ManagerBasedRlEnv, hand_sensor_name: str | None
) -> torch.Tensor:
  """True where the hand touches the mug anywhere, pads, fingers or palm; False
  throughout without a sensor. (B,)"""
  if hand_sensor_name is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  return _on_surface_bool(env, hand_sensor_name)


def _release_progress(
  env: ManagerBasedRlEnv,
  grasp_cfg: SceneEntityCfg,
  obj: Entity,
  clearance: float,
  hold: float,
  open_margin: float,
  hand_sensor_name: str | None = None,
  object_width: float = MUG_WIDTH,
) -> torch.Tensor:
  """How far the jaw has let go of the mug: ``0.5`` once it opens ``open_margin`` wider
  than the mug (``object_width``), rising to ``1`` as it withdraws from ``hold`` to
  ``clearance``. (B,)

  With ``hand_sensor_name`` the withdrawal counts only with the hand off the mug: the jaw
  midpoint can be ``clearance`` from the mug centre with a finger still on its rim.
  """
  mid, _, aperture = _jaw_frame(env.scene[grasp_cfg.name], grasp_cfg)
  dist = torch.norm(mid - obj.data.root_link_pos_w.unsqueeze(1), dim=-1).amin(dim=-1)
  away = ((dist - hold) / (clearance - hold)).clamp(0.0, 1.0)
  away = away * (~_hand_touching(env, hand_sensor_name)).float()
  opening = torch.sigmoid((aperture.amin(dim=-1) - object_width - open_margin) / 0.003)
  return torch.maximum(opening * (0.5 + 0.5 * away), away)


#: ``_overhand_gate``: the holding jaw's approach axis (wrist to fingertips) pointing at
#: most this far up in z pays in full, and from ``_OVERHAND_UP`` up nothing.
_OVERHAND_LEVEL = -0.2
_OVERHAND_UP = 0.2


def _overhand_gate(
  env: ManagerBasedRlEnv,
  grasp_cfg: SceneEntityCfg | None,
  tcp_cfg: SceneEntityCfg | None,
  grasp_sensor_names: Sequence[Sequence[str]],
) -> torch.Tensor | float:
  """1 where a holding jaw's fingers point level or down, 0 where they point up. (B,)

  Turned upright with the fingers up, the hand is under the mug and the forearm below
  it, and the mug cannot be set down without the hand going through the table: rolled
  out at ``-Lower-Release`` 5599, the 47% of upright holds made that way never stood the
  mug on the table, against 64% of those made fingers down. Approach z falls linearly
  from 1 at ``_OVERHAND_LEVEL`` to 0 at ``_OVERHAND_UP``; the best of the holding jaws.
  1 throughout without ``tcp_cfg``.
  """
  if tcp_cfg is None:
    return 1.0
  robot = env.scene[grasp_cfg.name]
  mid, _, _ = _jaw_frame(robot, grasp_cfg)
  approach = mid - robot.data.site_pos_w[:, tcp_cfg.site_ids]
  up = approach[..., 2] / approach.norm(dim=-1).clamp_min(1e-6)  # (B, G)
  gate = ((_OVERHAND_UP - up) / (_OVERHAND_UP - _OVERHAND_LEVEL)).clamp(0.0, 1.0)
  held = torch.stack([_grasped_pair_bool(env, pair) for pair in grasp_sensor_names], dim=-1)
  return (gate * held.float()).amax(dim=-1)


def mug_lifted_overhand(
  env: ManagerBasedRlEnv,
  grasp_sensor_names: Sequence[Sequence[str]],
  support_sensor_name: str,
  grasp_cfg: SceneEntityCfg,
  tcp_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """``object_lifted_pair``, only for a hold it can set the mug down from
  (``_overhand_gate``). (B,)

  With the held and lowered rungs gated alone, a fingers-up hold kept the lift and the
  grasp ladder, ~10 a step, and the ``-Overhand`` policy made it on 40% of its spawns.
  """
  lifted = object_lifted_pair(env, grasp_sensor_names, support_sensor_name)
  return lifted * _overhand_gate(env, grasp_cfg, tcp_cfg, grasp_sensor_names)


def mug_upright_held(
  env: ManagerBasedRlEnv,
  object_name: str,
  grasp_sensor_names: Sequence[Sequence[str]],
  support_sensor_name: str,
  either_end: bool = False,
  grasp_cfg: SceneEntityCfg | None = None,
  tcp_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  """How upright a mug held clear of the table is, ``((1 + cos tilt) / 2)^2``. (B,)

  The dense rung between lifting a lying mug and standing it up: it pays for turning it
  in the air, so the policy need not find the whole turn before anything pays. With
  ``tcp_cfg`` (``-Overhand``) only for a hold it can set the mug down from
  (``_overhand_gate``).
  """
  obj: Entity = env.scene[object_name]
  lifted = _grasped_any_bool(env, grasp_sensor_names) & ~_on_surface_bool(
    env, support_sensor_name
  )
  upright = ((1.0 + _stand_cos(obj, either_end)) / 2).square()
  overhand = _overhand_gate(env, grasp_cfg, tcp_cfg, grasp_sensor_names)
  return lifted.float() * upright * overhand


#: ``_SetDownLatch`` with a ``pickup_height``: within this of its standing pose a mug
#: counts as on the table though its contact drops out, and tipped past this it is no
#: longer set down.
_REST_HEIGHT = 0.01
_SET_DOWN_TIP = math.radians(45.0)


class _SetDownLatch:
  """Picked up this episode, then set down standing in the jaw -- and how long ago.

  ``was_lifted`` latches the first step the mug is held clear of the table, so tipping it
  up with a push does not count. ``set_down`` latches the first step after that the jaw
  holds it standing on the table, and ``steps`` counts control steps since. Without
  ``pickup_height`` the mug is on the table only while its contact sensor says so, and
  the latch clears whenever it does not. Opening the jaw rocks a standing mug off the
  table for a few steps, ~3 mm and 7 mm at most, measured, which cleared that latch: a
  clean release then paid nothing (2 of 119). With ``pickup_height`` the mug also counts
  as on the table within ``_REST_HEIGHT`` of its standing pose, and the latch clears only
  once it is ``pickup_height`` above that pose -- picked up again -- or tips past
  ``_SET_DOWN_TIP``.
  """

  def __init__(self, env: ManagerBasedRlEnv):
    self.was_lifted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.set_down = torch.zeros_like(self.was_lifted)
    self.steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self.was_lifted[ids] = False
    self.set_down[ids] = False
    self.steps[ids] = 0

  def update(
    self,
    env: ManagerBasedRlEnv,
    obj: Entity,
    held: torch.Tensor,
    support_sensor_name: str,
    max_tilt: float,
    either_end: bool,
    surface_cfg: SceneEntityCfg | None,
    pickup_height: float | None,
  ) -> torch.Tensor:
    """Advance the latch; returns where the mug stands upright on the table. (B,)"""
    on_table = _on_surface_bool(env, support_sensor_name)
    self.was_lifted |= held & ~on_table
    cos = _stand_cos(obj, either_end)
    if pickup_height is None:
      resting, keep = on_table, on_table
    else:
      assert surface_cfg is not None, "pickup_height needs surface_cfg"
      stand_z = surface_top_z(env, surface_cfg) + MUG_HALF_HEIGHT
      height = obj.data.root_link_pos_w[:, 2] - stand_z
      resting = on_table | (height < _REST_HEIGHT)
      keep = (height < pickup_height) & (cos > math.cos(_SET_DOWN_TIP))
    standing = self.was_lifted & resting & (cos > math.cos(max_tilt))
    was_set_down = self.set_down
    self.set_down = (self.set_down | (standing & held)) & keep
    self.steps = torch.where(was_set_down & self.set_down, self.steps + 1, 0)
    return standing

  def holding_paid(
    self, env: ManagerBasedRlEnv, hold_time: float | None
  ) -> torch.Tensor:
    """False once the mug has stood set down for ``hold_time``: from then on, holding it
    there pays nothing. (B,)"""
    if hold_time is None:
      return torch.ones_like(self.set_down)
    return ~self.set_down | (self.steps < round(hold_time / env.step_dt))


class mug_lowered_upright:
  """How close to standing on the table a lifted, upright-held mug is. (B,)

  The rung between holding the mug upright and setting it down. ``mug_upright_held``
  pays the same at any height, and all of it goes the moment the mug touches the table,
  so nothing leads the mug back down: the policy holds it upright in the air. This pays
  for lowering it, ``exp(-h^2 / std^2)`` of its height ``h`` above the pose it stands
  in, and keeps paying once it touches the table, so the set-down is no cliff. Scaled by
  an uprightness gate that is 1 upright and 0 at ``max_tilt``, and paid only while held
  after a lift this episode, so a lying mug gripped on the table earns nothing. With
  ``hold_time`` it stops paying once the mug has stood set down that long (see
  ``mug_stood_upright``). With ``tcp_cfg`` (``-Overhand``) only for a hold it can set
  the mug down from (``_overhand_gate``).
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self.latch = _SetDownLatch(env)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.latch.reset(env_ids)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    object_name: str,
    grasp_sensor_names: Sequence[Sequence[str]],
    support_sensor_name: str,
    surface_cfg: SceneEntityCfg,
    std: float = 0.05,
    max_tilt: float = math.radians(45.0),
    either_end: bool = False,
    stand_tilt: float | None = None,
    pickup_height: float | None = None,
    hold_time: float | None = None,
    grasp_cfg: SceneEntityCfg | None = None,
    tcp_cfg: SceneEntityCfg | None = None,
  ) -> torch.Tensor:
    obj: Entity = env.scene[object_name]
    held = _grasped_any_bool(env, grasp_sensor_names)
    self.latch.update(
      env,
      obj,
      held,
      support_sensor_name,
      max_tilt if stand_tilt is None else stand_tilt,
      either_end,
      surface_cfg,
      pickup_height,
    )
    stand_z = surface_top_z(env, surface_cfg) + MUG_HALF_HEIGHT
    height = (obj.data.root_link_pos_w[:, 2] - stand_z).clamp_min(0.0)
    cos_limit = math.cos(max_tilt)
    upright = ((_stand_cos(obj, either_end) - cos_limit) / (1.0 - cos_limit)).clamp(
      0.0, 1.0
    )
    lowered = torch.exp(-torch.square(height / std))
    paid = held & self.latch.was_lifted & self.latch.holding_paid(env, hold_time)
    overhand = _overhand_gate(env, grasp_cfg, tcp_cfg, grasp_sensor_names)
    return paid.float() * upright * lowered * overhand


class mug_stood_upright:
  """1.0 while a mug that was lifted this episode is set down upright on the table. (B,)

  It has to have been picked up and set down (``_SetDownLatch``). Without ``released``
  the jaw has to be holding it as it stands; with ``hold_time``, only for that long after
  it was set down: holding it there for good paid ~44 a step with ``-Lower``, more than
  any release the policy found, so it held on until the arm, leaning on the mug, slid
  down it onto the table. With ``released`` it has to have been set down in the jaw and
  then let go, with every jaw ``clearance`` from it. A mug turned upright in the air and
  dropped does not count. ``pickup_height`` keeps the set-down through the release's
  rocking (``_SetDownLatch``).
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self.latch = _SetDownLatch(env)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.latch.reset(env_ids)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    object_name: str,
    grasp_sensor_names: Sequence[Sequence[str]],
    support_sensor_name: str,
    grasp_cfg: SceneEntityCfg,
    max_tilt: float,
    released: bool,
    clearance: float = 0.08,
    either_end: bool = False,
    surface_cfg: SceneEntityCfg | None = None,
    pickup_height: float | None = None,
    hold_time: float | None = None,
  ) -> torch.Tensor:
    obj: Entity = env.scene[object_name]
    held = _grasped_any_bool(env, grasp_sensor_names)
    standing = self.latch.update(
      env,
      obj,
      held,
      support_sensor_name,
      max_tilt,
      either_end,
      surface_cfg,
      pickup_height,
    )
    if released:
      ok = (
        standing
        & self.latch.set_down
        & ~held
        & _jaw_clear(env, grasp_cfg, obj, clearance)
      )
    else:
      ok = standing & held & self.latch.holding_paid(env, hold_time)
    return ok.float()


class mug_stood_releasing(mug_stood_upright):
  """How far the jaw has let go of a mug set down standing: open, then withdrawn. (B,)

  The rung between ``mug_stood_upright`` held and released, as ``mug_turned_releasing``
  is for the turn: the jaw opening wider than the mug pays ``0.5``, rising to ``1`` as
  it withdraws from ``hold`` to ``clearance``, so long as the mug still stands set down.
  """

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    object_name: str,
    grasp_sensor_names: Sequence[Sequence[str]],
    support_sensor_name: str,
    grasp_cfg: SceneEntityCfg,
    max_tilt: float,
    clearance: float = 0.08,
    either_end: bool = False,
    surface_cfg: SceneEntityCfg | None = None,
    pickup_height: float | None = None,
    hold: float = 0.03,
    open_margin: float = 0.005,
  ) -> torch.Tensor:
    obj: Entity = env.scene[object_name]
    held = _grasped_any_bool(env, grasp_sensor_names)
    standing = self.latch.update(
      env,
      obj,
      held,
      support_sensor_name,
      max_tilt,
      either_end,
      surface_cfg,
      pickup_height,
    )
    return (standing & self.latch.set_down & ~held).float() * _release_progress(
      env, grasp_cfg, obj, clearance, hold, open_margin
    )


def arm_home_once_done(
  env: ManagerBasedRlEnv, done_term: str, asset_cfg: SceneEntityCfg, std: float
) -> torch.Tensor:
  """How near the arm is to its start pose, ``exp(-|q - q0|^2 / std^2)``, while the end
  state ``done_term`` pays. (B,)

  The episode runs on after the mug is let go, and from there no rung said where the arm
  goes: ``-Overhand-Facing`` 9000 swung it about at will for the rest of the 40 s, the
  action rate and joint acceleration penalties ~0.003 an episode. The start pose is clear
  of the mug and the table, and what the real arm would go back to. ``done_term`` must
  come before this term in the reward dict, so it has paid this step.
  """
  robot: Entity = env.scene[asset_cfg.name]
  ids = asset_cfg.joint_ids
  error = robot.data.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids]
  home = torch.exp(-torch.sum(torch.square(error), dim=-1) / std**2)
  return reward_term_active(env, done_term) * home


def arm_still_once_done(
  env: ManagerBasedRlEnv, done_term: str, asset_cfg: SceneEntityCfg, std: float
) -> torch.Tensor:
  """How still the arm is, ``exp(-|dq|^2 / std^2)``, while the end state ``done_term``
  pays. (B,)

  Upright stands the mug ~4 rad off the arm's start pose: ``arm_home_once_done`` there
  had it swing back past the mug, and success fell 62% -> 52% in 400 iterations. Holding
  still where the task ended asks for no move near the mug at all. ``done_term`` must
  come before this term in the reward dict.
  """
  robot: Entity = env.scene[asset_cfg.name]
  speed = torch.sum(torch.square(robot.data.joint_vel[:, asset_cfg.joint_ids]), dim=-1)
  return reward_term_active(env, done_term) * torch.exp(-speed / std**2)


def arm_raised_once_done(
  env: ManagerBasedRlEnv,
  done_term: str,
  asset_cfg: SceneEntityCfg,
  grasp_cfg: SceneEntityCfg,
  surface_cfg: SceneEntityCfg,
  height: float,
  std: float,
) -> torch.Tensor:
  """The jaw raised over the table and held there while the end state ``done_term``
  pays: its height over the table top as a share of ``height`` (lowest jaw, clamped to
  ``[0, 1]``), half of it paid outright and half for the arm held still,
  ``exp(-|dq|^2 / std^2)``. (B,)

  ``arm_still_once_done`` alone paid a hand resting on the table as well as one held in
  the air, and the arm settled onto it: from upright 7400 to 8700, 11% -> 38% of the
  episodes that stood the mug ended on ``table_contact``, ~5 s after the release. Up is
  away from the mug for the overhand jaw, and the hand on the table pays nothing.
  ``done_term`` must come before this term in the reward dict.
  """
  robot: Entity = env.scene[asset_cfg.name]
  mid, _, _ = _jaw_frame(env.scene[grasp_cfg.name], grasp_cfg)
  h = mid[..., 2].amin(dim=-1) - surface_top_z(env, surface_cfg)
  raised = (h / height).clamp(0.0, 1.0)
  speed = torch.sum(torch.square(robot.data.joint_vel[:, asset_cfg.joint_ids]), dim=-1)
  still = torch.exp(-speed / std**2)
  return reward_term_active(env, done_term) * raised * (0.5 + 0.5 * still)


def mug_turning_held(
  env: ManagerBasedRlEnv,
  object_name: str,
  grasp_sensor_names: Sequence[Sequence[str]],
  target_yaw: float,
) -> torch.Tensor:
  """How close the handle heads to ``target_yaw`` while the mug is held upright. (B,)

  ``(1 + cos err) / 2`` times the uprightness, only while held: the turn has to be made
  with the mug in the jaw, not by pushing it round.
  """
  obj: Entity = env.scene[object_name]
  held = _grasped_any_bool(env, grasp_sensor_names).float()
  heading = (1.0 + torch.cos(handle_heading_error(obj, target_yaw))) / 2
  upright = mug_tilt_cos(obj).clamp_min(0.0)
  return held * heading * upright


class mug_turning_kept:
  """``mug_turning_held``, but the turn made in the jaw stays paid once it lets go. (B,)

  So the turn can be ratcheted -- turn, let go, turn the wrist back, grasp again -- the
  way one screws something in. Held, it pays as ``mug_turning_held`` and remembers the
  heading term; let go, it pays the smaller of that and the current one, times the
  uprightness. The mug turning further while nothing holds it -- pushed or batted round
  -- earns nothing, and turning back costs what it gives up. Zero until the first grasp.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self.kept = torch.zeros(env.num_envs, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.kept[slice(None) if env_ids is None else env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    object_name: str,
    grasp_sensor_names: Sequence[Sequence[str]],
    target_yaw: float,
  ) -> torch.Tensor:
    obj: Entity = env.scene[object_name]
    held = _grasped_any_bool(env, grasp_sensor_names)
    heading = (1.0 + torch.cos(handle_heading_error(obj, target_yaw))) / 2
    self.kept = torch.where(held, heading, self.kept)
    upright = mug_tilt_cos(obj).clamp_min(0.0)
    return torch.minimum(heading, self.kept) * upright


class mug_turned:
  """1.0 while the mug stands upright, handle within ``tolerance`` of ``target_yaw``,
  let go with every jaw ``clearance`` from it -- if it was grasped this episode. (B,)

  The grasp latches, like ``mug_stood_upright``'s lift: a mug batted round by the fingers
  does not count. With ``progress_term`` it has to have been held standing turned on the
  table instead (that term's ``placed``): grasped once and then pushed round, it counted.
  With ``hand_sensor_name`` (``-HandsOff``) no part of the hand may touch it: the jaw
  midpoint ``clearance`` from the mug centre left a finger free to rest on the rim, and
  ``-Release-Regrasp`` 3700 kept the hand on the mug, steadying it, in 83% of the
  steps this paid.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self.was_grasped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.was_grasped[slice(None) if env_ids is None else env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    object_name: str,
    grasp_sensor_names: Sequence[Sequence[str]],
    support_sensor_name: str,
    grasp_cfg: SceneEntityCfg,
    target_yaw: float,
    tolerance: float,
    max_tilt: float,
    clearance: float = 0.08,
    progress_term: str | None = None,
    hand_sensor_name: str | None = None,
  ) -> torch.Tensor:
    obj: Entity = env.scene[object_name]
    held = _grasped_any_bool(env, grasp_sensor_names)
    self.was_grasped |= held
    if progress_term is None:
      latched = self.was_grasped
    else:
      latched = env.reward_manager.get_term_cfg(progress_term).func.placed
    ok = (
      latched
      & ~held
      & _on_surface_bool(env, support_sensor_name)
      & (mug_tilt_cos(obj) > math.cos(max_tilt))
      & (handle_heading_error(obj, target_yaw) < tolerance)
      & _jaw_clear(env, grasp_cfg, obj, clearance)
      & ~_hand_touching(env, hand_sensor_name)
    )
    return ok.float()


class mug_turned_releasing:
  """How far the jaw has let go of a mug set down turned: open, then withdrawn. (B,)

  The rung between ``mug_turning_held`` and ``mug_turned``. Those two leave a gap: the
  moment the jaw opens the turning pays nothing, and the turned pays only once every jaw
  is ``clearance`` away, so the release itself -- several steps of the mug wobbling free
  -- earns nothing and the policy learns to keep holding. "Set down" latches while the
  jaw holds the mug standing on the table, handle within ``tolerance``, and clears when
  the mug leaves the table. From there this pays the jaw opening wider than the mug,
  ``0.5`` open at the mug and rising to ``1`` as the jaw withdraws from ``hold`` to
  ``clearance``, so long as the mug still stands turned. With ``hand_sensor_name`` the
  withdrawal counts only with the hand off the mug (``_release_progress``).
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self.set_down = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.set_down[slice(None) if env_ids is None else env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    object_name: str,
    grasp_sensor_names: Sequence[Sequence[str]],
    support_sensor_name: str,
    grasp_cfg: SceneEntityCfg,
    target_yaw: float,
    tolerance: float,
    max_tilt: float,
    clearance: float = 0.08,
    hold: float = 0.03,
    open_margin: float = 0.005,
    hand_sensor_name: str | None = None,
  ) -> torch.Tensor:
    obj: Entity = env.scene[object_name]
    held = _grasped_any_bool(env, grasp_sensor_names)
    on_table = _on_surface_bool(env, support_sensor_name)
    turned = (
      on_table
      & (mug_tilt_cos(obj) > math.cos(max_tilt))
      & (handle_heading_error(obj, target_yaw) < tolerance)
    )
    self.set_down = (self.set_down | (turned & held)) & on_table
    return (self.set_down & turned & ~held).float() * _release_progress(
      env, grasp_cfg, obj, clearance, hold, open_margin, hand_sensor_name
    )


# -- Progress shaping (``-Stage``) ------------------------------------------------------
# The rungs above the grasp, as one potential-based shaping term (Ng et al. 1999): each
# step pays ``gamma * phi(s') - phi(s)`` for a progress potential ``phi`` in [0, 1], so an
# episode's shaping sums to the progress made, discounted. The per-state rungs it replaces
# could be farmed: the set-down's hold timer refilled at every re-lift, and the turn paid
# the heading the mug spawned with. Going back and forth here nets nothing, and staying
# put at ``phi`` costs ``(1 - gamma) * phi`` a step. The task's end state pays on its own
# (``mug_stood_upright`` released, ``mug_turned``). ``phi`` reads only the mug, the jaw and
# latches that never clear within an episode, which ``mug_progress_latches`` shows the
# critic.


class _ProgressShaping:
  """``(gamma * phi(s') - phi(s)) / step_dt``: ``weight`` is the shaping a whole run from
  ``phi = 0`` to ``1`` pays. Zero on an episode's first step, which sets ``phi(s)``."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self.prev = torch.zeros(env.num_envs, device=env.device)
    self.fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    self.lifted = torch.zeros_like(self.fresh)
    self.placed = torch.zeros_like(self.fresh)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self.fresh[ids] = True
    self.lifted[ids] = False
    self.placed[ids] = False

  def latches(self) -> torch.Tensor:
    return torch.stack((self.lifted, self.placed), dim=-1).float()

  def _shape(self, env: ManagerBasedRlEnv, phi: torch.Tensor, gamma: float):
    # A terminal state's potential is 0: ending the episode -- tipping the mug, pushing it
    # out of view -- pays ``-phi(s)``, all that staying at ``phi`` would have cost. With
    # its own ``phi`` it would escape that cost. A time-out is bootstrapped, not terminal.
    terminated = getattr(env, "termination_manager", None)
    if terminated is not None:
      phi_next = torch.where(terminated.terminated, 0.0, phi)
    else:
      phi_next = phi
    out = torch.where(self.fresh, 0.0, gamma * phi_next - self.prev) / env.step_dt
    self.prev = phi
    self.fresh[:] = False
    return out


class mug_upright_progress(_ProgressShaping):
  """Progress standing a lying mug up and letting go of it, as shaping. (B,)

  ``phi`` is 0 until the mug has been held clear of the table (``lifted``, latched), so
  tipping it up with a push earns nothing. Then: ``0.4`` for how upright it is,
  ``((1 + cos tilt) / 2)^2``; ``0.3`` for how close it is to standing on the table,
  ``exp(-h^2 / std^2)`` of its height over its standing pose, gated by an uprightness
  that is 0 at ``max_tilt``; and ``0.3`` for the release, ``_release_progress`` while it
  stands free after the jaw set it down (``placed``, latched).
  """

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    object_name: str,
    grasp_sensor_names: Sequence[Sequence[str]],
    support_sensor_name: str,
    grasp_cfg: SceneEntityCfg,
    surface_cfg: SceneEntityCfg,
    stand_tilt: float,
    gamma: float,
    std: float = 0.05,
    max_tilt: float = math.radians(45.0),
    clearance: float = 0.08,
    hold: float = 0.03,
    open_margin: float = 0.005,
  ) -> torch.Tensor:
    obj: Entity = env.scene[object_name]
    held = _grasped_any_bool(env, grasp_sensor_names)
    on_table = _on_surface_bool(env, support_sensor_name)
    self.lifted |= held & ~on_table
    cos = mug_tilt_cos(obj)
    height = obj.data.root_link_pos_w[:, 2] - (
      surface_top_z(env, surface_cfg) + MUG_HALF_HEIGHT
    )
    standing = (on_table | (height < _REST_HEIGHT)) & (cos > math.cos(stand_tilt))
    self.placed |= self.lifted & standing & held
    cos_limit = math.cos(max_tilt)
    gate = ((cos - cos_limit) / (1.0 - cos_limit)).clamp(0.0, 1.0)
    lowered = torch.exp(-torch.square(height.clamp_min(0.0) / std))
    free = (self.placed & standing & ~held).float()
    release = free * _release_progress(
      env, grasp_cfg, obj, clearance, hold, open_margin
    )
    phi = self.lifted.float() * (
      0.4 * ((1.0 + cos) / 2).square() + 0.3 * gate * lowered + 0.3 * release
    )
    return self._shape(env, phi, gamma)


class mug_turn_progress(_ProgressShaping):
  """Progress turning the handle to ``target_yaw`` and letting go, as shaping. (B,)

  ``phi``: ``0.7`` for the heading, ``1 - err / pi``, linear so every degree pays the
  same, times an uprightness that is 0 at ``max_tilt``; and ``0.3`` for the release,
  ``_release_progress`` while the mug stands turned free after the jaw set it down so
  (``placed``, latched). The heading is not gated on a grasp: a gate would pay the spawn
  heading the step the jaw first closes. Pushing the mug round earns its shaping, which
  is bounded, but not ``mug_turned``, which needs a grasp. ``lifted`` latches the first
  grasp, for the critic.
  """

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    object_name: str,
    grasp_sensor_names: Sequence[Sequence[str]],
    support_sensor_name: str,
    grasp_cfg: SceneEntityCfg,
    target_yaw: float,
    tolerance: float,
    stand_tilt: float,
    gamma: float,
    max_tilt: float = math.radians(45.0),
    clearance: float = 0.08,
    hold: float = 0.03,
    open_margin: float = 0.005,
  ) -> torch.Tensor:
    obj: Entity = env.scene[object_name]
    held = _grasped_any_bool(env, grasp_sensor_names)
    self.lifted |= held
    err = handle_heading_error(obj, target_yaw)
    turned = (
      _on_surface_bool(env, support_sensor_name)
      & (mug_tilt_cos(obj) > math.cos(stand_tilt))
      & (err < tolerance)
    )
    self.placed |= turned & held
    free = (self.placed & turned & ~held).float()
    release = free * _release_progress(
      env, grasp_cfg, obj, clearance, hold, open_margin
    )
    heading = (1.0 - err / math.pi) * _upright_gate(obj, max_tilt)
    return self._shape(env, 0.7 * heading + 0.3 * release, gamma)


def mug_progress_latches(env: ManagerBasedRlEnv, term_name: str) -> torch.Tensor:
  """The progress shaping term's latches, ``(lifted, placed)``, for the critic. (B, 2)

  The rewards above the grasp turn on them, and the critic cannot tell them from the
  state alone: a mug standing free looks the same whether the jaw set it down or not.
  Zeros before the reward manager is built (the observation manager sizes its terms
  first).
  """
  manager = getattr(env, "reward_manager", None)
  if manager is None or term_name not in manager.active_terms:
    return torch.zeros(env.num_envs, 2, device=env.device)
  return manager.get_term_cfg(term_name).func.latches()
