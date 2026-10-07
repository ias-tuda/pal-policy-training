"""Reward terms mjlab does not provide.

mjlab already ships the shaping that gets an arm to an object and the object to a goal
(``staged_position_reward``, ``bring_object_reward``). What it has no notion of is the
*grasp* and the *release* -- its lift task succeeds by proximity alone. A stacking task is
defined by letting go at the right moment, so those terms live here.

Every term is a pure function of the scene plus the commanded goal. None reaches into
another manager's internals.
"""

from __future__ import annotations

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from pal_mjlab.tasks.manipulation.mdp.observations import (
  STILL_SPEED,
  _at_goal,
  _grasped_bool,
  _jaw_open_bool,
  _on_surface_bool,
  _pad_contact_any,
  _placed_bool,
  _release_latch,
  _was_lifted_bool,
)
from pal_mjlab.tasks.manipulation.mdp.shapes import (
  BySize,
  PerShape,
  object_half_height,
  resolve,
)
from pal_mjlab.tasks.manipulation.mdp.stack import (
  OK_DXY,
  OK_HEIGHT,
  OK_SPEED,
  OK_TILT,
  OK_YAW,
  stack_state,
)

_ROBOT = SceneEntityCfg("robot")


def grasped(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """1.0 while both pads hold the object."""
  return _grasped_bool(env, sensor_name).float()


def object_lifted(
  env: ManagerBasedRlEnv,
  grasp_sensor_name: str,
  support_sensor_name: str,
) -> torch.Tensor:
  """1.0 once the object is held *and* has left the support surface.

  Both halves matter: without the grasp term the reward pays for knocking the object off
  the table, and without the support term it pays for merely touching it.
  """
  return (
    _grasped_bool(env, grasp_sensor_name) & ~_on_surface_bool(env, support_sensor_name)
  ).float()


def reach_only(
  env: ManagerBasedRlEnv,
  object_name: str,
  reaching_std: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """The reaching half of mjlab's `staged_position_reward`, without the goal factor.

  `staged_position_reward` returns `G(d_r) * (1 + G(d_g))`, and that second factor pays for
  the object being near the goal however it got there -- so on a table-height goal it is one
  of the terms a sliding policy collects. This is the same Gaussian on tool-to-object
  distance with the factor dropped.
  """
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  reach_error = torch.sum(torch.square(ee_pos_w - obj.data.root_link_pos_w), dim=-1)
  return torch.exp(-reach_error / reaching_std**2)


def released_at_goal(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  grasp_sensor_name: str,
  threshold: float = 0.05,
  support_sensor_name: str | None = None,
  jaw_cfg: SceneEntityCfg | None = None,
  jaw_open: float | BySize | PerShape | None = None,
) -> torch.Tensor:
  """1.0 when the object is at the goal and no longer held.

  The shaping counterpart to :func:`~.terminations.object_placed`: it pays as soon as the
  jaw opens in the right place, without waiting for the object to settle, which is what
  stops the policy from converging on holding the object at the goal forever.

  `¬grasped` is also true of an object that was never grasped, so on a table-height goal
  this pays a slide. Passing `support_sensor_name` additionally requires the `was_lifted`
  latch; leaving it None is the pre-C2 behaviour. ``jaw_cfg`` / ``jaw_open`` (plan S7-4's J)
  also require the jaw open (`observations._jaw_open_bool`). ``threshold`` is a 3-D distance
  or plan S7-5's (xy, z) tolerance (`observations._at_goal`).
  """
  obj: Entity = env.scene[object_name]
  command = env.command_manager.get_term(command_name)
  at_goal = _at_goal(command.command, obj.data.root_link_pos_w, threshold)
  released = at_goal & ~_grasped_bool(env, grasp_sensor_name)
  if support_sensor_name is not None:
    released &= _was_lifted_bool(env, grasp_sensor_name, support_sensor_name)
  if jaw_cfg is not None:
    assert jaw_open is not None
    released &= _jaw_open_bool(env, jaw_cfg, jaw_open)
  return released.float()


def object_placed(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  grasp_sensor_name: str,
  threshold: float = 0.05,
  velocity_threshold: float = 0.05,
  support_sensor_name: str | None = None,
  jaw_cfg: SceneEntityCfg | None = None,
  jaw_open: float | BySize | PerShape | None = None,
) -> torch.Tensor:
  """1.0 on full success: at the goal, released, and come to rest.

  With `support_sensor_name`, additionally requires that the object was lifted at some
  point in the episode; with ``jaw_cfg`` / ``jaw_open``, the jaw open -- see `_placed_bool`.
  """
  return _placed_bool(
    env,
    command_name,
    object_name,
    grasp_sensor_name,
    threshold,
    velocity_threshold,
    support_sensor_name,
    jaw_cfg,
    jaw_open,
  ).float()


def entity_lin_vel_l2(env: ManagerBasedRlEnv, entity_name: str) -> torch.Tensor:
  """Squared planar speed of an entity, for penalising scene disturbance."""
  entity: Entity = env.scene[entity_name]
  return torch.sum(torch.square(entity.data.root_link_lin_vel_w[:, :2]), dim=-1)


def joint_pos_limits(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Amount by which the commanded joints exceed their soft position limits."""
  robot: Entity = env.scene[asset_cfg.name]
  joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
  limits = robot.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
  below = (limits[..., 0] - joint_pos).clamp_min(0.0)
  above = (joint_pos - limits[..., 1]).clamp_min(0.0)
  return torch.sum(below + above, dim=-1)


def lift_height(
  env: ManagerBasedRlEnv,
  grasp_sensor_name: str,
  object_name: str,
  surface_cfg: SceneEntityCfg,
  max_height: float = 0.10,
) -> torch.Tensor:
  """How far the object has been lifted, as a fraction of ``max_height``, while held.

  The bounded, gated counterpart to :func:`object_lifted`, which is binary. Bounded because
  an unbounded height reward pays more for throwing the object than for placing it; gated on
  the grasp because otherwise it pays for the object being anywhere high, including mid-fall
  after being knocked. The surface height is read from physics, so the term is still right
  when the table is randomised.
  """
  from pal_mjlab.tasks.manipulation.mdp.events import surface_top_z

  obj: Entity = env.scene[object_name]
  above = obj.data.root_link_pos_w[:, 2] - surface_top_z(env, surface_cfg)
  height = above.clamp(0.0, max_height)
  return (height / max_height) * _grasped_bool(env, grasp_sensor_name).float()


def bring_object_when_lifted(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  grasp_sensor_name: str,
  support_sensor_name: str,
  std: float = 0.15,
) -> torch.Tensor:
  """Distance-to-goal shaping that only pays once the object is actually off the table.

  mjlab's ``bring_object_reward`` pays for the object being near the goal however it got
  there, which on a tabletop task includes sliding it. Gating on "held and not supported"
  makes the ordering of the stages monotone: the policy cannot collect the transport reward
  without having completed the lift.
  """
  obj: Entity = env.scene[object_name]
  command = env.command_manager.get_term(command_name)
  distance = torch.norm(command.command - obj.data.root_link_pos_w, dim=-1)
  lifted = object_lifted(env, grasp_sensor_name, support_sensor_name)
  return torch.exp(-torch.square(distance / std)) * lifted


def bring_object_after_lift(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  grasp_sensor_name: str,
  support_sensor_name: str,
  std: float = 0.15,
) -> torch.Tensor:
  """Distance-to-goal shaping gated on the `was_lifted` latch (S5.2 module M2).

  `bring_object_when_lifted` pays only while the object is off the table *now*, so on a
  table-height goal its income stops the moment the object reaches the target. With a lift
  term that pays for height, hovering above the target is then the per-step optimum -- what
  every R-C-family run converged to in c1a (5-8 cm up, `s5-1-report.md` section 6). Gating on
  the latch instead keeps the slide path closed (a slid object never sets it) while paying
  through the lowering and after the release.
  """
  obj: Entity = env.scene[object_name]
  command = env.command_manager.get_term(command_name)
  distance = torch.norm(command.command - obj.data.root_link_pos_w, dim=-1)
  lifted = _was_lifted_bool(env, grasp_sensor_name, support_sensor_name).float()
  return torch.exp(-torch.square(distance / std)) * lifted


def held_object_speed_hinge(
  env: ManagerBasedRlEnv,
  object_name: str,
  grasp_sensor_name: str,
  threshold: float = 0.4,
) -> torch.Tensor:
  """Squared excess of a held object's linear speed over ``threshold`` (m/s); 0 if not held.

  Plan S7's speed bound B, the control for the slew bound A: rather than limiting how fast
  the arm *can* move, it prices how fast the cube *is* moved while in the jaw. Only while
  held, so a cube knocked over or settling after release is not the policy's speed. The
  instantaneous velocity is read, as a reward term can only see this step's state.
  """
  obj: Entity = env.scene[object_name]
  speed = torch.norm(obj.data.root_link_lin_vel_w, dim=-1)
  held = _grasped_bool(env, grasp_sensor_name)
  return torch.square(torch.clamp(speed - threshold, min=0.0)) * held.float()


def release_quality_of(
  dxy: torch.Tensor,
  height: torch.Tensor,
  dist_std: float,
  height_free: float,
  height_std: float,
  floor: float,
  tilt: torch.Tensor | None = None,
  tilt_std: float | None = None,
  speed: torch.Tensor | None = None,
  speed_std: float | None = None,
) -> torch.Tensor:
  """q = floor + (1 - floor) * G_dist_std(dxy) * G_height_std(max(height - height_free, 0)).

  Plan S7's M6 grade of one release: 1 for a set-down on the target, falling with the
  horizontal distance it was let go at and with the height above ``height_free`` it was
  dropped from, down to ``floor`` -- which keeps placing worth something even when the
  release was poor.

  Plan S7-9's PT multiplies the graded part by G_tilt_std(tilt) * G_speed_std(speed), the
  object's tilt (rad) and speed (m/s) at the let-go, when ``tilt_std`` / ``speed_std`` are set.
  Inside the ``(1 - floor)`` part, so the floor still keeps a poor release worth something.
  """
  near = torch.exp(-torch.square(dxy / dist_std))
  low = torch.exp(-torch.square((height - height_free).clamp_min(0.0) / height_std))
  graded = near * low
  if tilt_std is not None:
    assert tilt is not None
    graded = graded * torch.exp(-torch.square(tilt / tilt_std))
  if speed_std is not None:
    assert speed is not None
    graded = graded * torch.exp(-torch.square(speed / speed_std))
  return floor + (1.0 - floor) * graded


def set_down_passed(latch, set_down_steps: int) -> torch.Tensor:
  """Whether the latest release was a set-down (plan S7-2, module M7). (B,)

  At the step the grasp opened the object rested on the surface, moved slower than
  `observations.STILL_SPEED`, and had been held there, still, within
  `observations.SET_DOWN_RADIUS` of the goal since a touchdown inside that radius, for at
  least ``set_down_steps`` control steps. ``latch`` is `observations._release_latch`'s.
  """
  return (
    latch.released
    & latch.rel_on_table
    & (latch.rel_speed < STILL_SPEED)
    & (latch.rel_dwell >= set_down_steps)
  )


def release_quality(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  grasp_sensor_name: str,
  support_sensor_name: str,
  surface_cfg: SceneEntityCfg,
  dist_std: float = 0.02,
  height_free: float = 0.01,
  height_std: float = 0.01,
  floor: float = 0.25,
  set_down_steps: int = 0,
  jaw_cfg: SceneEntityCfg | None = None,
  jaw_open: float | BySize | PerShape | None = None,
  require_ok: bool = False,
  tilt_std: float | None = None,
  speed_std: float | None = None,
) -> torch.Tensor:
  """The grade of the latest release after `was_lifted` (plan S7, module M6); ``floor`` if
  there has been none this episode. ``jaw_cfg`` / ``jaw_open``: a release is the jaw opening
  (plan S7-4's J, `observations._release_latch`).

  See `release_quality_of` for the grade and `observations._release_latch` for what counts
  as the release and where it is measured.

  With ``set_down_steps`` > 0 (plan S7-2, module M7) the grade is paid only for a set-down
  (`set_down_passed`), and is ``floor`` for any other release -- let go above the surface,
  while the object still moved, too soon after putting it down, or after dragging it in.

  With ``require_ok`` (plan S9, module RG) the grade is paid only for a release on a stack goal
  at which the release condition held (`stack.StackState.ok`: within 1 cm, 1 cm up, 10 deg of
  tilt and of yaw, 5 cm/s), and is ``floor`` for any other.
  ``tilt_std`` / ``speed_std`` (plan S7-9's PT): the grade also falls with the object's tilt and
  speed at the let-go (`release_quality_of`).
  """
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
  q = release_quality_of(
    latch.dxy,
    latch.height,
    dist_std,
    height_free,
    height_std,
    floor,
    latch.tilt,
    tilt_std,
    latch.rel_speed,
    speed_std,
  )
  paid = latch.released
  if set_down_steps > 0:
    paid = set_down_passed(latch, set_down_steps)
  if require_ok:
    paid = paid & latch.rel_ok
  return torch.where(paid, q, torch.full_like(q, floor))


def released_at_goal_graded(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  grasp_sensor_name: str,
  support_sensor_name: str,
  surface_cfg: SceneEntityCfg,
  threshold: float = 0.05,
  dist_std: float = 0.02,
  height_free: float = 0.01,
  height_std: float = 0.01,
  floor: float = 0.25,
  set_down_steps: int = 0,
  jaw_cfg: SceneEntityCfg | None = None,
  jaw_open: float | BySize | PerShape | None = None,
  require_ok: bool = False,
  tilt_std: float | None = None,
  speed_std: float | None = None,
) -> torch.Tensor:
  """`released_at_goal` (gated on `was_lifted`) x `release_quality` (plan S7, module M6).

  M1 pays ``release`` and ``placed`` every step, so a cube let go short that slides onto the
  target earns the same as one set down on it, sooner -- the late release two of three S5.2
  full-fix seeds converged to. Grading both by where the cube was let go makes the set-down
  pay more for the rest of the episode. ``set_down_steps`` > 0 pays the grade for a set-down
  only (module M7, `release_quality`); ``require_ok`` only for an ``OK`` let-go (plan S9's RG).
  """
  released = released_at_goal(
    env,
    command_name,
    object_name,
    grasp_sensor_name,
    threshold,
    support_sensor_name,
    jaw_cfg,
    jaw_open,
  )
  return released * release_quality(
    env,
    command_name,
    object_name,
    grasp_sensor_name,
    support_sensor_name,
    surface_cfg,
    dist_std,
    height_free,
    height_std,
    floor,
    set_down_steps,
    jaw_cfg,
    jaw_open,
    require_ok,
    tilt_std,
    speed_std,
  )


def object_placed_graded(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  grasp_sensor_name: str,
  support_sensor_name: str,
  surface_cfg: SceneEntityCfg,
  threshold: float = 0.05,
  velocity_threshold: float = 0.05,
  dist_std: float = 0.02,
  height_free: float = 0.01,
  height_std: float = 0.01,
  floor: float = 0.25,
  set_down_steps: int = 0,
  jaw_cfg: SceneEntityCfg | None = None,
  jaw_open: float | BySize | PerShape | None = None,
  require_ok: bool = False,
  tilt_std: float | None = None,
  speed_std: float | None = None,
) -> torch.Tensor:
  """`object_placed` (gated on `was_lifted`) x `release_quality` (plan S7, module M6).

  ``set_down_steps`` > 0 pays the grade for a set-down only (module M7, `release_quality`);
  ``require_ok`` only for an ``OK`` let-go (plan S9's RG).
  """
  placed = object_placed(
    env,
    command_name,
    object_name,
    grasp_sensor_name,
    threshold,
    velocity_threshold,
    support_sensor_name,
    jaw_cfg,
    jaw_open,
  )
  return placed * release_quality(
    env,
    command_name,
    object_name,
    grasp_sensor_name,
    support_sensor_name,
    surface_cfg,
    dist_std,
    height_free,
    height_std,
    floor,
    set_down_steps,
    jaw_cfg,
    jaw_open,
    require_ok,
    tilt_std,
    speed_std,
  )


def held_object_slide_speed(
  env: ManagerBasedRlEnv,
  object_name: str,
  grasp_sensor_name: str,
  support_sensor_name: str,
) -> torch.Tensor:
  """The object's horizontal speed (m/s) while it is held *and* resting on the surface, after
  it has been lifted once this episode; 0 otherwise. Weight it negative.

  Plan S7-2's guard on M7: sliding a held cube along the table is not a set-down, and
  M7's touchdown rule already withholds the grade from a release after one. This prices the
  slide itself. Gated on `was_lifted` so that the first grasp -- the jaw closing on a cube
  that rests on the table, and nudging it -- is not charged.
  """
  obj: Entity = env.scene[object_name]
  speed_xy = torch.norm(obj.data.root_link_lin_vel_w[:, :2], dim=-1)
  held_down = (
    _grasped_bool(env, grasp_sensor_name)
    & _on_surface_bool(env, support_sensor_name)
    & _was_lifted_bool(env, grasp_sensor_name, support_sensor_name)
  )
  return speed_xy * held_down.float()


def set_down_placed(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  grasp_sensor_name: str,
  support_sensor_name: str,
  surface_cfg: SceneEntityCfg,
  jaw_cfg: SceneEntityCfg,
  jaw_open: float | BySize | PerShape,
  set_down_steps: int,
  threshold: float = 0.05,
  velocity_threshold: float = 0.05,
) -> torch.Tensor:
  """1.0 while the object is placed (plan S7-4's J) and its latest release was a set-down
  (`set_down_passed` with ``set_down_steps``); 0 otherwise. Plan S7-4's revised M7 (`S`): a
  bonus on top of M6's grade, which every release keeps -- M7 as run floored every other
  release, and was not learned."""
  placed = object_placed(
    env,
    command_name,
    object_name,
    grasp_sensor_name,
    threshold,
    velocity_threshold,
    support_sensor_name,
    jaw_cfg,
    jaw_open,
  )
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
  return placed * set_down_passed(latch, set_down_steps).float()


def _tcp_above_object_top(
  env: ManagerBasedRlEnv, object_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Height of the tool point above the object's top face, m (upright object). (B,)"""
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  tcp_z = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)[:, 2]
  half_z = object_half_height(env, object_name)
  return tcp_z - (obj.data.root_link_pos_w[:, 2] + half_z)


def retreat_height(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  grasp_sensor_name: str,
  support_sensor_name: str,
  jaw_cfg: SceneEntityCfg,
  jaw_open: float | BySize | PerShape,
  asset_cfg: SceneEntityCfg,
  clear_height: float = 0.05,
  threshold: float = 0.05,
  velocity_threshold: float = 0.05,
) -> torch.Tensor:
  """Plan S7-4's M8, the retreat: the tool point's height above the object's top as a
  fraction of ``clear_height``, capped at 1, paid only while the object is placed -- after a
  J-release: at the goal, at rest, the jaw open, lifted this episode -- and neither pad
  touches it. (B,)

  Nothing is paid before the release, so the term cannot pull the hand up while it carries;
  and `placed` keeps paying through the retreat, so a retreat that knocks the object off the
  goal loses both. A hand left resting around the placed object earns nothing here -- what
  S7-3's G20 policies did, pressing the elbow at full effort (S7-3 report section 8.4).
  """
  placed = _placed_bool(
    env,
    command_name,
    object_name,
    grasp_sensor_name,
    threshold,
    velocity_threshold,
    support_sensor_name,
    jaw_cfg,
    jaw_open,
  )
  armed = placed & ~_pad_contact_any(env, grasp_sensor_name)
  above = _tcp_above_object_top(env, object_name, asset_cfg)
  return (above / clear_height).clamp(0.0, 1.0) * armed.float()


def pad_contact_after_release(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  grasp_sensor_name: str,
  support_sensor_name: str,
  surface_cfg: SceneEntityCfg,
  jaw_cfg: SceneEntityCfg,
  jaw_open: float | BySize | PerShape,
) -> torch.Tensor:
  """1.0 while either pad touches the object after this episode's J-release; weight it
  negative (plan S7-4's M8: the nudging after the set-down in S7-2's videos). (B,)"""
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
  return (latch.released & _pad_contact_any(env, grasp_sensor_name)).float()


def open_at_goal(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  grasp_sensor_name: str,
  support_sensor_name: str,
  surface_cfg: SceneEntityCfg,
  jaw_cfg: SceneEntityCfg,
  jaw_open: float | BySize | PerShape,
  jaw_hold: float | BySize | PerShape,
  gate: tuple[float, float],
) -> torch.Tensor:
  """Plan S7-5's O: how far the jaw has opened from the hold angle towards open,
  clip((jaw_hold - q) / (jaw_hold - jaw_open), 0, 1), paid while the object has been lifted
  this episode, has not yet been J-released, and is at the goal by ``gate`` -- an (xy, z)
  tolerance (`observations._at_goal`), any tilt, table contact not required. (B,)

  Why: RCTP s2 (S7-4) parked the cube on the target and held it to the time-out, the jaw at
  ~0.30 rad. Nothing paid for *starting* to open -- J and `placed` pay only once the jaw is
  below ``jaw_open`` and the cube has settled within the tolerance -- so exploratory openings
  from a tilted, rocking hold were rarely paid while holding was. This is the gradient
  between. It stops at the first J-release (the release latch, `observations._release_latch`,
  read with the same jaw), so re-grasping cannot farm it; the grasp is not part of the gate,
  so it keeps paying between the pads' let-go and the jaw passing ``jaw_open``.
  """
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
  was_lifted = _was_lifted_bool(env, grasp_sensor_name, support_sensor_name)
  obj: Entity = env.scene[object_name]
  goal = env.command_manager.get_term(command_name).command
  near = _at_goal(goal, obj.data.root_link_pos_w, gate)
  robot: Entity = env.scene[jaw_cfg.name]
  q = robot.data.joint_pos[:, jaw_cfg.joint_ids][:, 0]
  hold, open_ = resolve(env, jaw_hold), resolve(env, jaw_open)
  opened = ((hold - q) / (hold - open_)).clamp(0.0, 1.0)
  return opened * (was_lifted & ~latch.released & near).float()


def stack_alignment(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  reference_name: str,
  grasp_sensor_name: str,
  support_sensor_name: str,
  near: float = 0.03,
) -> torch.Tensor:
  """Plan S9's SA: how close the held cube is to the release condition ``OK``, as the product of
  one Gaussian per quantity with its bound as the width, prod_i exp(-(x_i / bound_i)^2), paid
  while the cube is held (both pads) after `was_lifted` and within ``near`` of the goal
  horizontally. 1 at a perfect hold on the base; ~0.007 at the five bounds together. (B,)

  The gradient towards ``OK`` beside the lock (ST-GA): with the lock alone the policy is paid
  nothing until it happens to satisfy all five at once and can open.
  """
  state = stack_state(env, object_name, reference_name, command_name)
  score = torch.exp(
    -(
      torch.square(state.dxy / OK_DXY)
      + torch.square(state.height / OK_HEIGHT)
      + torch.square(state.tilt / OK_TILT)
      + torch.square(state.yaw / OK_YAW)
      + torch.square(state.speed / OK_SPEED)
    )
  )
  held = _grasped_bool(env, grasp_sensor_name) & _was_lifted_bool(
    env, grasp_sensor_name, support_sensor_name
  )
  return score * (held & (state.dxy < near)).float()


#: Attribute on the env caching `_joint_effort_caps`, keyed by the joints' global ids.
_EFFORT_CAPS = "_pp_effort_caps"


def _joint_effort_caps(env: ManagerBasedRlEnv, robot: Entity, ids) -> torch.Tensor:
  """The joints' ``actuatorfrcrange`` upper bound off the compiled model, cached. (n,)"""
  global_ids = tuple(robot.indexing.joint_ids[ids].tolist())
  cache = getattr(env, _EFFORT_CAPS, None)
  if cache is None:
    cache = {}
    setattr(env, _EFFORT_CAPS, cache)
  if global_ids not in cache:
    mj = env.sim.mj_model
    if not all(bool(mj.jnt_actfrclimited[j]) for j in global_ids):
      raise ValueError(
        "arm_effort_hinge needs a joint-level effort limit (actuatorfrcrange) on every "
        "joint it reads; build the robot with an arm effort scale (plan S7-4's T)"
      )
    cache[global_ids] = torch.tensor(
      [float(mj.jnt_actfrcrange[j, 1]) for j in global_ids], device=env.device
    )
  return cache[global_ids]


def arm_effort_hinge(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, threshold: float = 0.8
) -> torch.Tensor:
  """Plan S7-4's P: sum over the named joints of max(|tau| / tau_cmd - threshold, 0)^2, with
  tau the joint's actuator force (`qfrc_actuator`) and tau_cmd its effort limit as the model
  carries it -- the joint's ``actuatorfrcrange``, which plan S7-4's T lowers. Positive; weight
  it negative. Below ``threshold`` of the limit, where gravity and carrying live, it is 0. (B,)
  """
  robot: Entity = env.scene[asset_cfg.name]
  ids = asset_cfg.joint_ids
  tau_cmd = _joint_effort_caps(env, robot, ids)
  tau = robot.data.qfrc_actuator[:, ids]
  excess = (tau.abs() / tau_cmd - threshold).clamp_min(0.0)
  return torch.sum(torch.square(excess), dim=-1)


def lift_height_faded(
  env: ManagerBasedRlEnv,
  grasp_sensor_name: str,
  object_name: str,
  surface_cfg: SceneEntityCfg,
  command_name: str,
  max_height: float = 0.10,
  fade_std: float = 0.05,
) -> torch.Tensor:
  """`lift_height` × (1 − G_fade(d_g)): lift income fades out at the goal (S5.2 module M3).

  `lift_height` pays linearly for height, so lowering the object onto a table-height goal
  costs lift income all the way down. Fading it with the object's 3-D distance to the goal
  removes that cost where it matters and nowhere else. Fading on the horizontal distance alone
  would leave a dip directly over the target (plan S5.2, P2).
  """
  obj: Entity = env.scene[object_name]
  command = env.command_manager.get_term(command_name)
  distance = torch.norm(command.command - obj.data.root_link_pos_w, dim=-1)
  fade = 1.0 - torch.exp(-torch.square(distance / fade_std))
  return (
    lift_height(env, grasp_sensor_name, object_name, surface_cfg, max_height) * fade
  )


#: Corners of a unit cube, in the object's own frame. Scaled by the object's half-size to
#: give the eight keypoints of its oriented bounding box.
_UNIT_CORNERS = torch.tensor(
  [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)]
)


def _object_corners(env: ManagerBasedRlEnv, object_name: str) -> torch.Tensor:
  """The eight corners of the object's bounding box, in its own frame. (B, 8, 3)

  Read from ``geom_size`` rather than from a constant, so a task that randomises the cube's
  size gets the right keypoints for free.
  """
  obj: Entity = env.scene[object_name]
  half = env.sim.model.geom_size[:, obj.indexing.geom_ids[0], :]  # (B, 3)
  return _UNIT_CORNERS.to(half.device) * half.unsqueeze(1)


def keypoint_distance_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  kernel_a: float = 30.0,
  kernel_b: float = 2.0,
) -> torch.Tensor:
  """Allshire et al.'s keypoint reward: eight bounding-box corners, logistic kernel.

  ``K(x) = (exp(a x) + b + exp(-a x))^-1``, averaged over the eight corners of the object's
  oriented bounding box against the same corners at the goal, and normalised so a perfect
  match scores 1. From *Transferring Dexterous Manipulation from GPU Simulation to a Remote
  Real-World TriFinger* (2022), which uses ``a = 30``, ``b = 2``; that kernel's half-width is
  about 4 cm, the right scale against this task's 2.5 cm success threshold.

  Why corners rather than the centre: a keypoint set measures position and orientation as one
  quantity, in metres, with no separate rotation term to weight against the position one. v0
  does not score the placed cube's orientation -- the goal is a point, so the target corners
  are an upright cube at it -- which makes this a reward-*shape* comparison today. It is the
  term that makes scoring orientation free when v1 wants it.

  The plan sketched this variant with a Gaussian kernel. The published kernel is used
  instead: the variant exists to test the published idea against the framework's distance
  reward, and changing the kernel would test something else.
  """
  obj: Entity = env.scene[object_name]
  goal = env.command_manager.get_term(command_name).command  # (B, 3)
  corners = _object_corners(env, object_name)  # (B, 8, 3)

  from mjlab.utils.lab_api.math import quat_apply

  quat = obj.data.root_link_quat_w.unsqueeze(1).expand(-1, corners.shape[1], -1)
  current = obj.data.root_link_pos_w.unsqueeze(1) + quat_apply(quat, corners)
  target = goal.unsqueeze(1) + corners

  distance = torch.norm(current - target, dim=-1)  # (B, 8)
  kernel = 1.0 / (
    torch.exp(kernel_a * distance) + kernel_b + torch.exp(-kernel_a * distance)
  )
  return kernel.mean(dim=-1) * (2.0 + kernel_b)


def grasp_height_match(
  env: ManagerBasedRlEnv,
  object_name: str,
  grasp_sensor_name: str,
  support_sensor_name: str,
  pad_cfg: SceneEntityCfg,
  offset: float = 0.0,
  std: float = 0.005,
  near: float = 0.03,
) -> torch.Tensor:
  """Plan S8's EQ: how well the pads' centre is at the height a ball is held at.

  ``G(z_pads - z_object - offset, std)``, paid while the pads' midpoint (the two pad sites in
  ``pad_cfg``) is within ``near`` of the object horizontally and the object has not been lifted
  yet (the `was_lifted` latch), so the approach's last centimetres and the close are steered to
  the height the bench found holds best (``offset``); once it is lifted the term is 0 and cannot be
  farmed by a carry. Parallel pads touch a sphere on the line between them; off the equator the
  contact normals tilt and the ball is squeezed up or down out of the jaw. (B,)
  """
  robot: Entity = env.scene[pad_cfg.name]
  obj: Entity = env.scene[object_name]
  pads = robot.data.site_pos_w[:, pad_cfg.site_ids]
  assert pads.shape[1] == 2, f"pad_cfg must name two sites, got {pad_cfg.site_names}"
  mid = pads.mean(dim=1)
  pos = obj.data.root_link_pos_w
  dz = mid[:, 2] - pos[:, 2] - offset
  near_xy = torch.norm(mid[:, :2] - pos[:, :2], dim=-1) < near
  lifted = _was_lifted_bool(env, grasp_sensor_name, support_sensor_name)
  return torch.exp(-torch.square(dz / std)) * (near_xy & ~lifted).float()


def time_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
  """1.0 every step, for a constant per-step cost. Weight it negative."""
  return torch.ones(env.num_envs, device=env.device)


def contact_penalty(
  env: ManagerBasedRlEnv, sensor_name: str, force_threshold: float = 10.0
) -> torch.Tensor:
  """1.0 while a contact sensor reports more than ``force_threshold``. Weight it negative.

  The same predicate mjlab terminates on, used as a cost instead: leaning on the table is a
  cheap way to stabilise a grasp in simulation and not one that survives contact with a real
  table.
  """
  from mjlab.tasks.manipulation.mdp import illegal_contact

  return illegal_contact(env, sensor_name, force_threshold).float()


def keepout_box_penalty(
  env: ManagerBasedRlEnv,
  center: tuple[float, float, float],
  half_extents: tuple[float, float, float],
  margin: float,
  reference_body: str,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """How far the watched bodies have pushed into a keep-out box. (B,)

  The box is fixed in ``reference_body``'s frame. Each watched link origin scores 1 at the
  box surface, falling linearly to 0 at ``margin`` outside it and clamped at 1 inside; the
  term is their sum, so it is positive and a penalty takes a negative weight. Ported from
  Claudius's 82c4450 (S5.1 item 8).
  """
  robot: Entity = env.scene[asset_cfg.name]
  ref_id = robot.body_names.index(reference_body)
  ref_pos = robot.data.body_link_pos_w[:, ref_id]  # (B, 3)
  ref_quat = robot.data.body_link_quat_w[:, ref_id]  # (B, 4)
  pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids]  # (B, N, 3)
  n = pos_w.shape[1]
  local = quat_apply_inverse(
    ref_quat.unsqueeze(1).expand(-1, n, -1), pos_w - ref_pos.unsqueeze(1)
  )
  kwargs = {"device": local.device, "dtype": local.dtype}
  offset = torch.abs(local - torch.tensor(center, **kwargs)) - torch.tensor(
    half_extents, **kwargs
  )
  # Euclidean distance from the box surface, zero anywhere inside it.
  outside = torch.norm(offset.clamp_min(0.0), dim=-1)
  return torch.sum((1.0 - outside / margin).clamp(0.0, 1.0), dim=-1)


#: Attribute on the env caching `_collision_points`, keyed by the bodies' global ids.
_COLLISION_POINTS = "_pp_collision_points"

#: Points kept per body for `arm_table_clearance` (farthest-point sample of its collision
#: mesh's convex hull, 400-1150 vertices on Kangaroo's arm 1-6). Over 2000 random orientations
#: the sample's lowest point is <= 1.5 mm above the hull's on every link (128: <= 3.6 mm).
CLEARANCE_POINTS_PER_BODY = 256


def _farthest_points(points, n: int):
  """``n`` rows of ``points`` (N, 3), spread out greedily; all of them if N <= n."""
  import numpy as np

  if len(points) <= n:
    return np.concatenate([points, points[: n - len(points)]])  # padded by repetition
  chosen = [int(np.argmin(points[:, 2]))]  # start at the lowest vertex
  dist = np.linalg.norm(points - points[chosen[0]], axis=1)
  for _ in range(n - 1):
    i = int(np.argmax(dist))
    chosen.append(i)
    dist = np.minimum(dist, np.linalg.norm(points - points[i], axis=1))
  return points[chosen]


def _collision_points(env: ManagerBasedRlEnv, robot: Entity, body_ids) -> torch.Tensor:
  """Surface points of each body's colliding mesh geoms, in the body frame, cached. (L, n, 3)

  Read off the compiled model: a geom with contype and conaffinity both 0 is not a collider
  (`CollisionCfg` zeroes them), so it is skipped; a colliding geom that is not a mesh raises,
  since a primitive's surface is not in ``mesh_vert``."""
  import mujoco
  import numpy as np

  global_ids = tuple(robot.indexing.body_ids[body_ids].tolist())
  cache = getattr(env, _COLLISION_POINTS, None)
  if cache is None:
    cache = {}
    setattr(env, _COLLISION_POINTS, cache)
  if global_ids not in cache:
    mj = env.sim.mj_model
    per_body = []
    for body in global_ids:
      verts = []
      for g in np.flatnonzero(mj.geom_bodyid == body):
        if not (mj.geom_contype[g] or mj.geom_conaffinity[g]):
          continue
        if mj.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
          raise ValueError(
            f"geom {mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_GEOM, g)!r} collides but is "
            "not a mesh; arm_table_clearance reads mesh vertices only"
          )
        mesh = mj.geom_dataid[g]
        adr, num = mj.mesh_vertadr[mesh], mj.mesh_vertnum[mesh]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, mj.geom_quat[g])
        verts.append(
          mj.mesh_vert[adr : adr + num] @ rot.reshape(3, 3).T + mj.geom_pos[g]
        )
      if not verts:
        name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, body)
        raise ValueError(f"body {name!r} has no colliding geom")
      pts = np.concatenate(verts).astype(np.float64)
      try:
        from scipy.spatial import ConvexHull

        pts = pts[ConvexHull(pts).vertices]
      except Exception:  # noqa: BLE001 -- no scipy, or a degenerate mesh: use every vertex
        pass
      per_body.append(_farthest_points(pts, CLEARANCE_POINTS_PER_BODY))
    cache[global_ids] = torch.tensor(
      np.stack(per_body), dtype=torch.float32, device=env.device
    )
  return cache[global_ids]


def arm_table_clearance(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  surface_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Per watched body, its lowest collision point's height above the surface's top face,
  over the points above the surface's footprint; +inf for a body with none there. (B, L)

  Negative is a point inside the slab (penetration). The surface is its first geom, a box,
  read out of physics per env, so DR-scene's table shift is followed; the footprint is
  axis-aligned, which holds because DR-scene moves the table without turning it
  (`tabletop_env_cfg`, ``table_pose``: axes x, y, z)."""
  from mjlab.utils.lab_api.math import quat_apply

  robot: Entity = env.scene[asset_cfg.name]
  pts = _collision_points(env, robot, asset_cfg.body_ids)  # (L, n, 3)
  pos = robot.data.body_link_pos_w[:, asset_cfg.body_ids]  # (B, L, 3)
  quat = robot.data.body_link_quat_w[:, asset_cfg.body_ids]  # (B, L, 4)
  b, nb, n = pos.shape[0], pts.shape[0], pts.shape[1]
  world = pos.unsqueeze(2) + quat_apply(
    quat.unsqueeze(2).expand(b, nb, n, 4), pts.unsqueeze(0).expand(b, nb, n, 3)
  )  # (B, L, n, 3)
  surface: Entity = env.scene[surface_cfg.name]
  geom_id = surface.indexing.geom_ids[0]
  centre = env.sim.data.geom_xpos[:, geom_id].reshape(b, 1, 1, 3)
  half = env.sim.model.geom_size[:, geom_id].expand(b, 3).reshape(b, 1, 1, 3)
  rel = world - centre
  inside = (rel[..., 0].abs() <= half[..., 0]) & (rel[..., 1].abs() <= half[..., 1])
  height = rel[..., 2] - half[..., 2]
  inf = torch.full_like(height, float("inf"))
  return torch.where(inside, height, inf).amin(dim=-1)


def table_clearance_penalty(
  env: ManagerBasedRlEnv,
  margin: float,
  asset_cfg: SceneEntityCfg,
  surface_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Plan S7-10's AH: per watched body, 0 while its lowest point over the table is more than
  ``margin`` above the top, rising linearly to 1 at contact (and 1 below it), summed over the
  bodies. Positive; weight it negative. (B,)"""
  clearance = arm_table_clearance(env, asset_cfg, surface_cfg)
  return (1.0 - clearance / margin).clamp(0.0, 1.0).sum(dim=-1)
