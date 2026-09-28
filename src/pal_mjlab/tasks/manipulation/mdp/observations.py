"""Observation terms mjlab does not provide.

mjlab's manipulation observations are written for a fixed-base arm, where the world frame
and the robot base frame coincide. A humanoid's base is somewhere else, so anything the
policy should see needs an explicit transform into the base frame -- that is what most of
this module is.

The term *names* used by the env cfg are part of the ONNX/deployment contract; see
``tabletop_env_cfg``. Changing what these functions compute is fine; changing the shapes
they return is not.
"""

from __future__ import annotations

import dataclasses

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import euler_xyz_from_quat, quat_apply, quat_inv, quat_mul

_ROBOT = SceneEntityCfg("robot")

#: The body every observation is expressed in: the frame PAL's stack publishes the robot and
#: the tracked objects in. Not the entity's root link: with `fixed_base` the root is the mocap
#: anchor at the world origin, 0.9 m below `base_link`, and until plan S7-2 (Claudius's
#: fda9492) every position the policy saw was in that frame -- which the robot does not have.
BASE_BODY = "base_link"


def _base_pose(robot: Entity) -> tuple[torch.Tensor, torch.Tensor]:
  """World pose (pos, quat) of `BASE_BODY`. (B, 3), (B, 4)"""
  i = robot.body_names.index(BASE_BODY)
  return robot.data.body_link_pos_w[:, i], robot.data.body_link_quat_w[:, i]


def _to_base_frame(robot: Entity, pos_w: torch.Tensor) -> torch.Tensor:
  base_pos, base_quat = _base_pose(robot)
  return quat_apply(quat_inv(base_quat), pos_w - base_pos)


def ee_position_b(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Grasp-point position in the robot base frame. (B, 3)"""
  robot: Entity = env.scene[asset_cfg.name]
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  return _to_base_frame(robot, ee_pos_w)


def ee_orientation_b(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Grasp-point orientation in the robot base frame, as a (w, x, y, z) quaternion. (B, 4)

  Plan S7-2's baseline B observes it (Claudius's `ee_orientation_b`): without it the policy
  reads the hand's orientation only through the seven joint angles. The sign is fixed to
  w >= 0, so the same orientation always gives the same four numbers.
  """
  robot: Entity = env.scene[asset_cfg.name]
  ee_quat_w = robot.data.site_quat_w[:, asset_cfg.site_ids].squeeze(1)
  quat_b = quat_mul(quat_inv(_base_pose(robot)[1]), ee_quat_w)
  return torch.where(quat_b[:, :1] < 0.0, -quat_b, quat_b)


def object_position_b(
  env: ManagerBasedRlEnv, object_name: str, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Object position in the robot base frame. (B, 3)"""
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  return _to_base_frame(robot, obj.data.root_link_pos_w)


def object_yaw_b(
  env: ManagerBasedRlEnv, object_name: str, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Object yaw relative to the robot base, as ``[cos, sin]``. (B, 2)

  Encoded as a unit vector rather than an angle so the observation stays continuous across
  the +/-pi wrap, which a raw yaw does not.
  """
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  quat_b = quat_mul(quat_inv(_base_pose(robot)[1]), obj.data.root_link_quat_w)
  _, _, yaw = euler_xyz_from_quat(quat_b)
  return torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)


def goal_position_b(
  env: ManagerBasedRlEnv, command_name: str, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Commanded goal position in the robot base frame. (B, 3)"""
  robot: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_term(command_name)
  return _to_base_frame(robot, command.command)


def is_grasped(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """1.0 when *both* pads report contact with the object. (B, 1)

  This reads the contact sensor, not pad-to-object proximity. Proximity reports a grasp for
  an object merely sitting between open jaws, which makes every downstream "is it held"
  gate meaningless.
  """
  return _grasped_bool(env, sensor_name).float().unsqueeze(-1)


def _grasped_bool(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None, f"sensor {sensor_name!r} does not provide 'found'"
  # found is (B, P * num_slots), primary-major (mjlab's contact_sensor layout). A pad is in
  # contact if any of its slots is; a grasp needs every pad in contact, not any. Reading
  # the raw (B, P * num_slots) with `.all` -- as this did until 2026-09-23 -- also demanded
  # a second simultaneous contact on every pad, and together with deeper finger geoms in
  # the sensor made a real grasp all but unreachable (S5.0).
  per_pad = found.view(found.shape[0], -1, sensor.cfg.num_slots).bool().any(dim=-1)
  return per_pad.all(dim=-1)


def _pad_contact_any(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """True while *either* pad touches the object: the grasp sensor's per-pad contact, any. (B,)"""
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None, f"sensor {sensor_name!r} does not provide 'found'"
  return found.view(found.shape[0], -1, sensor.cfg.num_slots).bool().any(dim=(-1, -2))


#: Plan S7-4's module J: the jaw's measured angle below which it counts as open, rad. The jaw
#: first touches a centred 5 cm cube at ~0.21 rad and holds it at ~0.30 (S7-4 plan, from S7-1's
#: FR s2 probe), so 0.15 is open by 6 cm of pad travel and cannot be a hold.
JAW_OPEN_BELOW = 0.15


def _jaw_open_bool(
  env: ManagerBasedRlEnv, jaw_cfg: SceneEntityCfg, jaw_open: float
) -> torch.Tensor:
  """Plan S7-4's J: the jaw's measured angle is below ``jaw_open``. (B,)"""
  robot: Entity = env.scene[jaw_cfg.name]
  q = robot.data.joint_pos[:, jaw_cfg.joint_ids]
  assert q.shape[-1] == 1, f"jaw_cfg must name one joint, got {jaw_cfg.joint_names}"
  return q[:, 0] < jaw_open


def is_on_surface(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """1.0 while the object still rests on the support surface. (B, 1)"""
  return _on_surface_bool(env, sensor_name).float().unsqueeze(-1)


def _on_surface_bool(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None, f"sensor {sensor_name!r} does not provide 'found'"
  return found.bool().any(dim=-1)


#: Attribute on the env holding the `was_lifted` episode latch.
_WAS_LIFTED = "_pp_was_lifted"


def _was_lifted_bool(
  env: ManagerBasedRlEnv, grasp_sensor_name: str, support_sensor_name: str
) -> torch.Tensor:
  """Episode latch: True from the first step the object is held clear of the surface.

  The pick-and-place goal sits at the cube's resting height, so "is it off the table *now*"
  is false at the moment of placing and cannot tell a carry from a slide. A latch can: it
  remembers that the object was once lifted, which sliding never achieves. Cleared on reset
  by `events.reset_was_lifted`.

  Updated lazily, on read, rather than by a step event. mjlab evaluates terminations before
  rewards and both before the reset, so the first gated term to run each step advances the
  latch from the current physics state, and every later reader in the same step sees the
  same value -- the OR is idempotent within a step because physics does not advance between
  term evaluations.

  **The latch is deliberately not an observation.** Adding it would change the observation
  width and break comparability with C1, which is what C2 exists to measure against. The
  cost is that the policy cannot see it: from the network's point of view a cube resting on
  the goal is the same state whether it was carried or slid. The reward still separates the
  two *trajectories*, so the gradient is right; it is the value function that pays.
  """
  latch = getattr(env, _WAS_LIFTED, None)
  if latch is None or latch.shape[0] != env.num_envs:
    latch = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    setattr(env, _WAS_LIFTED, latch)
  lifted = _grasped_bool(env, grasp_sensor_name) & ~_on_surface_bool(
    env, support_sensor_name
  )
  latch |= lifted
  return latch


#: Attribute on the env holding the release latch (plan S7, module M6).
_RELEASE = "_pp_release"

#: Plan S7-2, module M7 (the set-down): a held step counts towards the on-table dwell only
#: while the object is within this horizontal distance of the goal (m)...
SET_DOWN_RADIUS = 0.015
#: ...and moving slower than this (m/s). The same speed bounds the release itself.
STILL_SPEED = 0.05
#: A touchdown counts only after the held object was lifted this far clear of its resting
#: height (m), and held since. Without it, the contact flicker of a cube dragged along the table
#: re-arms the touchdown wherever the drag happens to be (found by the drag test).
SET_DOWN_CLEARANCE = 0.005


@dataclasses.dataclass
class ReleaseLatch:
  """Where and how the object was let go the last time, per env (plan S7 M6, S7-2 M7).

  ``released`` is set once a release after `was_lifted` has been seen this episode; ``dxy`` is
  the object's horizontal distance to the goal and ``height`` its height above its resting
  height on the surface, both at the step the grasp opened. ``grasped`` is the grasp state at
  the last update: the next update compares against it to see the grasp open.

  M7's set-down state, kept alongside: ``clear`` is "held, and lifted more than
  `SET_DOWN_CLEARANCE` clear of the surface since the last touchdown"; ``touchdown_ok``
  whether the latest touchdown -- the first held-on-surface update after being clear -- was
  within `SET_DOWN_RADIUS` of the goal; ``dwell`` how many consecutive updates, up to the last,
  the object was held on the surface, still and within the radius, since such a touchdown.
  ``rel_on_table``, ``rel_speed`` and ``rel_dwell`` are the surface contact, the object's
  speed and that dwell at the latest release. ``step`` is the env step of the last update.
  """

  grasped: torch.Tensor
  released: torch.Tensor
  dxy: torch.Tensor
  height: torch.Tensor
  clear: torch.Tensor
  touchdown_ok: torch.Tensor
  dwell: torch.Tensor
  rel_on_table: torch.Tensor
  rel_speed: torch.Tensor
  rel_dwell: torch.Tensor
  step: int = -1
  # Plan S7-4's J (``jaw`` set): a release is the jaw opening, and the where / how of it is
  # the pads' let-go before it, held here until the jaw opens (``pend_*``, valid while
  # ``pend``). ``jaw_closed`` is "jaw at or above the open angle" at the last update.
  jaw: tuple[str, float] | None = None
  jaw_closed: torch.Tensor | None = None
  pend: torch.Tensor | None = None
  pend_dxy: torch.Tensor | None = None
  pend_height: torch.Tensor | None = None
  pend_on_table: torch.Tensor | None = None
  pend_speed: torch.Tensor | None = None
  pend_dwell: torch.Tensor | None = None


def _release_latch(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  grasp_sensor_name: str,
  support_sensor_name: str,
  surface_cfg: SceneEntityCfg,
  jaw_cfg: SceneEntityCfg | None = None,
  jaw_open: float | None = None,
) -> ReleaseLatch:
  """Episode latch of the latest release after `was_lifted`. Cleared on reset by
  `events.reset_release_latch`.

  With ``jaw_cfg`` and ``jaw_open`` (plan S7-4, module J) a release is the jaw's measured angle
  falling below ``jaw_open`` after `was_lifted`, not the pads losing the object: S7-2/S7-3
  policies "released" with the jaw still at the hold angle, the pads merely slipping off. What
  the release is graded on -- where the object was, how high, its contact, speed and dwell --
  is still taken when the pads let go (a drop is graded at the height it fell from), and is
  committed when the jaw then opens; a let-go the jaw never opens after is not a release.
  Every reader of one env's latch must pass the same two; mixing them raises.

  A release is the step the grasp opens -- held at the last update, not held now -- while the
  `was_lifted` latch is set, so letting go of a cube that was only ever pushed is not one.
  Each release overwrites the last: the record is of where the object was *finally* let go,
  and a second grasp and release cannot bank a better one.

  Updated lazily, on read, like `_was_lifted_bool`, and at most once per env step (``step``
  against `env.common_step_counter`): a second read in the same step returns the latch as it
  is, so the dwell counter cannot count one step twice. It sees an opening on the step it
  happens only if something reads it every step; the M6/M7 reward terms do, at a nonzero
  weight (mjlab skips a zero-weight term entirely), and `scripts/eval_pick_place.py` reads it
  after every step for the arms that have none.

  The resting height is the surface's top plus the object's half-height, read out of physics
  the same way `SurfaceOffsetCommand` places its goal, so it follows a moving table.

  M7's set-down (plan S7-2): a *touchdown* is the first held-on-surface update after the held
  object was lifted `SET_DOWN_CLEARANCE` clear. The dwell counts only after a touchdown within
  `SET_DOWN_RADIUS` of the goal, so a cube put down short and dragged along the table to the
  goal never builds one -- it has to be lifted and set down again. The dwell at a release is
  the count up to the update before it.
  """
  from pal_mjlab.tasks.manipulation.mdp.events import surface_top_z

  jaw = None if jaw_cfg is None else (jaw_cfg.name, float(jaw_open))
  latch = getattr(env, _RELEASE, None)
  if latch is None or latch.released.shape[0] != env.num_envs:
    no = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    zero = torch.zeros(env.num_envs, device=env.device)
    count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    latch = ReleaseLatch(
      grasped=no.clone(),
      released=no.clone(),
      dxy=zero.clone(),
      height=zero.clone(),
      clear=no.clone(),
      touchdown_ok=no.clone(),
      dwell=count.clone(),
      rel_on_table=no.clone(),
      rel_speed=zero.clone(),
      rel_dwell=count.clone(),
      jaw=jaw,
    )
    if jaw is not None:
      latch.jaw_closed = no.clone()
      latch.pend = no.clone()
      latch.pend_dxy, latch.pend_height = zero.clone(), zero.clone()
      latch.pend_on_table, latch.pend_speed = no.clone(), zero.clone()
      latch.pend_dwell = count.clone()
    setattr(env, _RELEASE, latch)
  if latch.jaw != jaw:
    raise ValueError(
      f"the release latch was built with jaw={latch.jaw}, read with jaw={jaw}: every "
      "reader of one env's latch must define the release the same way"
    )
  if latch.step == env.common_step_counter:
    return latch
  latch.step = env.common_step_counter
  grasped = _grasped_bool(env, grasp_sensor_name)
  was_lifted = _was_lifted_bool(env, grasp_sensor_name, support_sensor_name)
  let_go = latch.grasped & ~grasped & was_lifted
  obj: Entity = env.scene[object_name]
  goal = env.command_manager.get_term(command_name).command
  pos = obj.data.root_link_pos_w
  half_z = env.sim.model.geom_size[:, obj.indexing.geom_ids[0], 2]
  rest_z = surface_top_z(env, surface_cfg) + half_z
  dxy = torch.norm((pos - goal)[:, :2], dim=-1)
  speed = torch.norm(obj.data.root_link_lin_vel_w, dim=-1)
  on_table = _on_surface_bool(env, support_sensor_name)
  height = pos[:, 2] - rest_z
  if jaw is None:
    opened = let_go
    now = (dxy, height, on_table, speed, latch.dwell)
  else:
    assert jaw_cfg is not None and jaw_open is not None
    # The pads' let-go is held until the jaw opens; a jaw that opens with none pending (it
    # cannot, around a 5 cm cube) is graded where the object is then.
    w = lambda new, old: torch.where(let_go, new, old)  # noqa: E731
    latch.pend_dxy, latch.pend_height = (
      w(dxy, latch.pend_dxy),
      w(height, latch.pend_height),
    )
    latch.pend_on_table = w(on_table, latch.pend_on_table)
    latch.pend_speed = w(speed, latch.pend_speed)
    latch.pend_dwell = w(latch.dwell, latch.pend_dwell)
    latch.pend = latch.pend | let_go
    jaw_is_open = _jaw_open_bool(env, jaw_cfg, jaw_open)
    opened = latch.jaw_closed & jaw_is_open & was_lifted
    p = latch.pend
    now = (
      torch.where(p, latch.pend_dxy, dxy),
      torch.where(p, latch.pend_height, height),
      torch.where(p, latch.pend_on_table, on_table),
      torch.where(p, latch.pend_speed, speed),
      torch.where(p, latch.pend_dwell, latch.dwell),
    )
    latch.pend = latch.pend & ~opened
    latch.jaw_closed = ~jaw_is_open
  latch.dxy = torch.where(opened, now[0], latch.dxy)
  latch.height = torch.where(opened, now[1], latch.height)
  latch.rel_on_table = torch.where(opened, now[2], latch.rel_on_table)
  latch.rel_speed = torch.where(opened, now[3], latch.rel_speed)
  latch.rel_dwell = torch.where(opened, now[4], latch.rel_dwell)
  latch.released |= opened
  latch.grasped = grasped
  # M7's dwell, after the release has been recorded with the count up to the last update.
  held_on_table = grasped & on_table
  latch.clear = grasped & (latch.clear | (pos[:, 2] - rest_z > SET_DOWN_CLEARANCE))
  touchdown = held_on_table & latch.clear
  near = dxy < SET_DOWN_RADIUS
  latch.touchdown_ok = torch.where(touchdown, near, latch.touchdown_ok)
  latch.clear = latch.clear & ~touchdown
  still = held_on_table & latch.touchdown_ok & near & (speed < STILL_SPEED)
  latch.dwell = torch.where(still, latch.dwell + 1, torch.zeros_like(latch.dwell))
  return latch


def _placed_bool(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  grasp_sensor_name: str,
  threshold: float,
  velocity_threshold: float,
  support_sensor_name: str | None = None,
  jaw_cfg: SceneEntityCfg | None = None,
  jaw_open: float | None = None,
) -> torch.Tensor:
  """At the goal, released, and at rest. Shared by the success reward and termination.

  With ``jaw_cfg`` and ``jaw_open`` (plan S7-4, module J) "released" also needs the jaw open,
  below ``jaw_open``: an object that slipped out of a jaw still at the hold angle is not.

  `released` is `¬grasped`, which a cube that was never picked up also satisfies -- so on a
  table-height goal this counts a slide as a place. Passing `support_sensor_name` adds the
  `was_lifted` latch and closes that path; leaving it None is the pre-C2 behaviour, which
  R-A and R-C keep so their numbers stay comparable.
  """
  obj: Entity = env.scene[object_name]
  command = env.command_manager.get_term(command_name)
  at_goal = torch.norm(command.command - obj.data.root_link_pos_w, dim=-1) < threshold
  released = ~_grasped_bool(env, grasp_sensor_name)
  if jaw_cfg is not None:
    assert jaw_open is not None
    released &= _jaw_open_bool(env, jaw_cfg, jaw_open)
  settled = torch.norm(obj.data.root_link_lin_vel_w, dim=-1) < velocity_threshold
  placed = at_goal & released & settled
  if support_sensor_name is not None:
    placed &= _was_lifted_bool(env, grasp_sensor_name, support_sensor_name)
  return placed
