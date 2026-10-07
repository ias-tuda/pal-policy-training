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

from pal_mjlab.tasks.manipulation.mdp.shapes import (
  BySize,
  PerShape,
  ball_mask,
  object_half_height,
  resolve,
  shape_kind,
)
from pal_mjlab.tasks.manipulation.mdp.stack import (
  stack_command_reference,
  stack_state,
)

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


class _ObjectTracker:
  """One object's world pose as a vision tracker reports it: a few control steps stale.

  Ported from Claudius's `origin/faster-moves` (1d484c4, `mdp._ObjectTracker`) for plan S7-6,
  without the dropout model (eb05a76; occlusion is left for later). FoundationPose publishes
  a pose that is 100-150 ms old by the time the policy acts on it, against a 20 ms control
  step: 5-8 steps, not the 0-2 a marker tracker costs and every run up to S7-5 trained with.

  The lag sits on the object's *pose*, before any observation is formed from it, rather than
  on mjlab's per-term ``delay_max_lag``, for two reasons. mjlab buffers each term's finished
  output, so a term-level delay on a vector that mixes the object with the robot (Claudius's
  jaw-to-cube vector) stales the encoder side too. And each term draws its own lag, so
  ``object_pos`` and ``object_yaw`` would report the cube at two different moments. One
  tracker per object, shared by every term that reads it, has neither problem.

  The lag is drawn per env at each reset, uniformly in ``lag`` (inclusive), so the policy
  cannot tune to one value. At a reset the whole history is refilled with the new pose: an
  older sample is a cube from the previous episode, which no tracker would ever report.
  """

  def __init__(self, env: ManagerBasedRlEnv, object_name: str, lag: tuple[int, int]):
    lo, hi = int(lag[0]), int(lag[1])
    if not 0 <= lo <= hi:
      raise ValueError(f"tracker lag must be 0 <= min <= max control steps; got {lag}")
    self._name = object_name
    self.lag_range = (lo, hi)
    obj: Entity = env.scene[object_name]
    n = env.num_envs
    # Newest sample first, so the lag is an index straight into the history.
    depth = hi + 1
    self._pos = obj.data.root_link_pos_w.unsqueeze(0).repeat(depth, 1, 1)
    self._quat = obj.data.root_link_quat_w.unsqueeze(0).repeat(depth, 1, 1)
    #: The lag each env is reporting at now, control steps. Read by the tests.
    self.lag = torch.zeros(n, dtype=torch.long, device=env.device)
    self._envs = torch.arange(n, device=env.device)
    # Latched per env until the next update fills the history (see the class docstring).
    self._pending = torch.ones(n, dtype=torch.bool, device=env.device)
    self._step = -1

  def update(self, env: ManagerBasedRlEnv) -> None:
    """Append this step's pose -- once per env step, however many terms ask."""
    self._pending |= env.episode_length_buf == 0
    obj: Entity = env.scene[self._name]
    pos_w, quat_w = obj.data.root_link_pos_w, obj.data.root_link_quat_w
    if env.common_step_counter != self._step:
      self._step = env.common_step_counter
      self._pos = torch.roll(self._pos, 1, dims=0)
      self._quat = torch.roll(self._quat, 1, dims=0)
      self._pos[0] = pos_w
      self._quat[0] = quat_w
    # After the append, so a term reading mid-step never sees a half-refilled history.
    if bool(self._pending.any()):
      fresh = self._pending
      self._pos[:, fresh] = pos_w[fresh]
      self._quat[:, fresh] = quat_w[fresh]
      lo, hi = self.lag_range
      self.lag[fresh] = torch.randint(
        lo, hi + 1, (int(fresh.sum()),), device=self.lag.device
      )
      self._pending = torch.zeros_like(self._pending)

  @property
  def pose(self) -> tuple[torch.Tensor, torch.Tensor]:
    """The pose the tracker is reporting now. (B, 3), (B, 4)"""
    return self._pos[self.lag, self._envs], self._quat[self.lag, self._envs]


def object_tracker(env: ManagerBasedRlEnv, object_name: str) -> _ObjectTracker | None:
  """The tracker built for ``object_name``, or None if no term has asked for one yet."""
  return getattr(env, "_pal_object_trackers", {}).get(object_name)


def _object_pose_w(
  env: ManagerBasedRlEnv, object_name: str, lag: tuple[int, int] | None
) -> tuple[torch.Tensor, torch.Tensor]:
  """The object's world pose, as a tracker ``lag`` control steps late reports it.

  ``lag=None`` reads the simulator directly: every run up to S7-5, and the critic, which runs
  only in sim and is never exported.
  """
  if lag is None:
    obj: Entity = env.scene[object_name]
    return obj.data.root_link_pos_w, obj.data.root_link_quat_w
  trackers = getattr(env, "_pal_object_trackers", None)
  if trackers is None:
    trackers = {}
    env._pal_object_trackers = trackers  # type: ignore[attr-defined]
  tracker = trackers.get(object_name)
  if tracker is None:
    tracker = _ObjectTracker(env, object_name, lag)
    trackers[object_name] = tracker
  elif tracker.lag_range != (int(lag[0]), int(lag[1])):
    raise ValueError(
      f"object {object_name!r} is tracked at two latencies, {tracker.lag_range} and "
      f"{tuple(lag)}. One object is one tracker."
    )
  tracker.update(env)
  return tracker.pose


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
  env: ManagerBasedRlEnv,
  object_name: str,
  asset_cfg: SceneEntityCfg = _ROBOT,
  lag: tuple[int, int] | None = None,
) -> torch.Tensor:
  """Object position in the robot base frame, as a tracker ``lag`` steps late reports it
  (`_ObjectTracker`; None for the live pose). (B, 3)"""
  robot: Entity = env.scene[asset_cfg.name]
  pos_w, _ = _object_pose_w(env, object_name, lag)
  return _to_base_frame(robot, pos_w)


def object_yaw_b(
  env: ManagerBasedRlEnv,
  object_name: str,
  asset_cfg: SceneEntityCfg = _ROBOT,
  lag: tuple[int, int] | None = None,
) -> torch.Tensor:
  """Object yaw relative to the robot base, as ``[cos, sin]``. (B, 2)

  Encoded as a unit vector rather than an angle so the observation stays continuous across
  the +/-pi wrap, which a raw yaw does not. ``lag`` as for `object_position_b`; the two
  share one tracker, so they always report the cube at the same moment.

  A ball has no yaw (plan S8): its orientation is whatever it has rolled to, and the tracker
  would report noise. The observation keeps its width -- the 38-D contract -- and shows the
  constant ``[1, 0]`` for every env whose object is a ball, exactly; MIX's cubes keep their yaw.
  """
  robot: Entity = env.scene[asset_cfg.name]
  _, quat_w = _object_pose_w(env, object_name, lag)
  quat_b = quat_mul(quat_inv(_base_pose(robot)[1]), quat_w)
  _, _, yaw = euler_xyz_from_quat(quat_b)
  cos_sin = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
  if shape_kind(env, object_name) == "cube":
    return cos_sin
  constant = torch.tensor([1.0, 0.0], device=cos_sin.device).expand_as(cos_sin)
  return torch.where(ball_mask(env, object_name).unsqueeze(-1), constant, cos_sin)


def commanded_target_rel(
  env: ManagerBasedRlEnv, action_name: str = "joint_pos"
) -> torch.Tensor:
  """The target the arm and jaw were last commanded to, relative to the default pose. (B, A)

  Plan S7-8's ``CMD_OBS``, in place of `last_action`. At the stack task's speeds a saturated
  absolute action needs 5-7 s to arrive, and an integrated (delta) target is policy state; the
  raw last action shows neither. This is what the term actually sent: after the slew, the box
  and plan S7-4's bounds, before the jaw servo. The same width as `last_action`, so the actor
  stays 38-D.
  """
  term = env.action_manager.get_term(action_name)
  robot: Entity = env.scene[term.cfg.entity_name]
  return term.commanded_target - robot.data.default_joint_pos[:, term.target_ids]


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
  env: ManagerBasedRlEnv, jaw_cfg: SceneEntityCfg, jaw_open: float | BySize | PerShape
) -> torch.Tensor:
  """Plan S7-4's J: the jaw's measured angle is below ``jaw_open`` (plan S8: a `PerShape`, one
  threshold for a cube and one for a ball, per env). (B,)"""
  robot: Entity = env.scene[jaw_cfg.name]
  q = robot.data.joint_pos[:, jaw_cfg.joint_ids]
  assert q.shape[-1] == 1, f"jaw_cfg must name one joint, got {jaw_cfg.joint_names}"
  return q[:, 0] < resolve(env, jaw_open)


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

#: The grasp sensor whose latches keep the plain attributes (`_WAS_LIFTED`, `_RELEASE`): the left
#: gripper's (`GripperCfg.contact_sensor_name`), the one every single-arm task drives, and what the
#: tests and `scripts/eval_pick_place.py` read. Plan S11's JOINT has a second arm with a grasp
#: sensor -- and a cube -- of its own; its latches live under `latch_attr`'s suffixed names, so
#: the two arms' lifts and releases are never confused.
PRIMARY_GRASP_SENSOR = "left_pad_contact"
#: Attribute on the env listing every latch attribute made, for the reset events.
_LATCH_ATTRS = "_pp_latch_attrs"


def latch_attr(base: str, grasp_sensor_name: str) -> str:
  """The env attribute of latch ``base`` (`_WAS_LIFTED` or `_RELEASE`) for a grasp sensor: ``base``
  itself for `PRIMARY_GRASP_SENSOR`, ``base:<sensor>`` for any other (plan S11)."""
  if grasp_sensor_name == PRIMARY_GRASP_SENSOR:
    return base
  return f"{base}:{grasp_sensor_name}"


def _remember_latch(env: ManagerBasedRlEnv, attr: str) -> None:
  attrs = getattr(env, _LATCH_ATTRS, None)
  if attrs is None:
    attrs = set()
    setattr(env, _LATCH_ATTRS, attrs)
  attrs.add(attr)


def latch_attrs(env: ManagerBasedRlEnv, base: str) -> list[str]:
  """Every attribute of latch ``base`` this env has made so far, the plain one first if it is
  there (`events.reset_was_lifted` / `reset_release_latch` clear them all)."""
  attrs = getattr(env, _LATCH_ATTRS, set())
  keyed = sorted(a for a in attrs if a.startswith(base + ":"))
  return ([base] if hasattr(env, base) else []) + keyed


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
  attr = latch_attr(_WAS_LIFTED, grasp_sensor_name)
  latch = getattr(env, attr, None)
  if latch is None or latch.shape[0] != env.num_envs:
    latch = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    setattr(env, attr, latch)
    _remember_latch(env, attr)
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
  speed and that dwell at the latest release, ``tilt`` its tilt then (`object_tilt`, plan S7-9).
  ``step`` is the env step of the last update.

  Plan S10: ``yaw`` is the object's yaw error at the latest release, |yaw mod 90 deg| in rad
  (0 = its faces aligned with the world's -- the table's and the box's -- axes; a cube only), for
  BX's yaw grade; ``surface`` the surface the height is measured from, (entity, geom names), so
  that every reader of one env's latch measures it from the same one (plan S10's box: its rim).
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
  yaw: torch.Tensor | None = None
  surface: tuple | None = None
  step: int = -1
  # Plan S7-9's PT: the object's tilt (`object_tilt`) at the latest release, taken with the rest.
  tilt: torch.Tensor | None = None
  # Plan S7-4's J (``jaw`` set): a release is entering the released state -- jaw open and
  # the pads off the object -- and the where / how of it is the pads' let-go before it, held
  # here until then (``pend_*``, valid while ``pend``). ``jaw_closed`` is "not in the released
  # state" at the last update.
  jaw: tuple[str, float] | None = None
  jaw_closed: torch.Tensor | None = None
  pend: torch.Tensor | None = None
  pend_dxy: torch.Tensor | None = None
  pend_height: torch.Tensor | None = None
  pend_on_table: torch.Tensor | None = None
  pend_speed: torch.Tensor | None = None
  pend_dwell: torch.Tensor | None = None
  # Plan S9: on a stack goal (`EntityOffsetCommand` with ``stack_on_top``), whether the release
  # condition `stack.StackState.ok` held at the let-go -- taken when the pads let go, like the
  # other ``pend_*``, and committed with the release. Always False on a table goal.
  rel_ok: torch.Tensor | None = None
  pend_ok: torch.Tensor | None = None
  pend_tilt: torch.Tensor | None = None
  pend_yaw: torch.Tensor | None = None


def object_tilt(env: ManagerBasedRlEnv, object_name: str) -> torch.Tensor:
  """The object's tilt: the angle between the vertical and its most vertical body axis, rad.
  0 for a cube resting on a face, pi/4 on an edge. (B,)

  Plan S7-9's PT grades the release by it; the same formula as `scripts/eval_pick_place.py`'s
  ``tilt`` (S7-8's tilt-at-let-go read), so the trained quantity is the one that was measured.
  """
  w, x, y, z = env.scene[object_name].data.root_link_quat_w.unbind(-1)
  # The world-z components of the body's three axes: the last row of its rotation matrix.
  row = torch.stack(
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1
  )
  return torch.acos(row.abs().amax(-1).clamp(max=1.0))


def yaw_error_mod_90(
  quat_w: torch.Tensor, reference_quat_w: torch.Tensor | None = None
) -> torch.Tensor:
  """|yaw mod 90 deg| of a world orientation, rad in [0, pi/4]: how far a cube's faces are turned
  off the world's axes (plan S10: the box sits at yaw 0 +/- 5 deg, unobserved, so alignment is
  paid against the axes the actor does observe) -- or, given ``reference_quat_w``, off that
  orientation's axes: |wrap_90(psi - psi_ref)| (plan S10.1: against the box, which stands at any
  yaw). (B,)"""
  _, _, yaw = euler_xyz_from_quat(quat_w)
  if reference_quat_w is not None:
    _, _, ref = euler_xyz_from_quat(reference_quat_w)
    yaw = yaw - ref
  quarter = torch.pi / 2
  return torch.abs(yaw - quarter * torch.round(yaw / quarter))


def yaw_reference_quat(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor | None:
  """The world orientation M6's yaw grade measures against: the entity the goal command names in
  its cfg's ``yaw_reference`` (plan S10.1, `box.BoxGoalCommandCfg`), or None for the world's axes. (B, 4)"""
  term = env.command_manager.get_term(command_name)
  name = getattr(term.cfg, "yaw_reference", None)
  if name is None:
    return None
  return env.scene[name].data.root_link_quat_w


def _release_latch(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  grasp_sensor_name: str,
  support_sensor_name: str,
  surface_cfg: SceneEntityCfg,
  jaw_cfg: SceneEntityCfg | None = None,
  jaw_open: float | BySize | PerShape | None = None,
) -> ReleaseLatch:
  """Episode latch of the latest release after `was_lifted`. Cleared on reset by
  `events.reset_release_latch`.

  With ``jaw_cfg`` and ``jaw_open`` (plan S7-4, module J) a release is entering the released
  state -- the jaw's measured angle below ``jaw_open`` *and* the pads off the object, the
  `_placed_bool` J predicate -- after `was_lifted`, not the pads losing the object alone:
  S7-2/S7-3 policies "released" with the jaw still at the hold angle, the pads merely slipping
  off. For a cube held face to face (the pads touch at ~0.21 rad) that is the jaw opening past
  ``jaw_open``; for one held across an edge or at a yaw, which the pads touch at ~0.10 rad (7 of
  64 S7-2 W1 s1 grasps, S7-4 prep f), it is the let-go itself -- the jaw was never above
  ``jaw_open``, and "the jaw falls below it" would never fire. What
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
  the same way `SurfaceOffsetCommand` places its goal, so it follows a moving table. On a stack
  goal (plan S9: `EntityOffsetCommand` with ``stack_on_top``) it is the goal's own z -- the cube
  resting on the base -- and ``rel_ok`` records whether `stack.StackState.ok` held at the let-go.
  Plan S10 passes the box's walls as ``surface_cfg`` (`events.surface_top_z` with named geoms), so the
  height is the object's bottom above the rim; every reader must pass the same surface.

  M7's set-down (plan S7-2): a *touchdown* is the first held-on-surface update after the held
  object was lifted `SET_DOWN_CLEARANCE` clear. The dwell counts only after a touchdown within
  `SET_DOWN_RADIUS` of the goal, so a cube put down short and dragged along the table to the
  goal never builds one -- it has to be lifted and set down again. The dwell at a release is
  the count up to the update before it.
  """
  from pal_mjlab.tasks.manipulation.mdp.events import surface_top_z

  jaw = (
    None
    if jaw_cfg is None
    else (
      jaw_cfg.name,
      jaw_open if isinstance(jaw_open, (PerShape, BySize)) else float(jaw_open),
    )
  )
  surface = (
    surface_cfg.name,
    None if surface_cfg.geom_names is None else tuple(surface_cfg.geom_names),
  )
  attr = latch_attr(_RELEASE, grasp_sensor_name)
  latch = getattr(env, attr, None)
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
      yaw=zero.clone(),
      surface=surface,
      jaw=jaw,
      rel_ok=no.clone(),
      pend_ok=no.clone(),
      tilt=zero.clone(),
    )
    if jaw is not None:
      latch.jaw_closed = no.clone()
      latch.pend = no.clone()
      latch.pend_dxy, latch.pend_height = zero.clone(), zero.clone()
      latch.pend_on_table, latch.pend_speed = no.clone(), zero.clone()
      latch.pend_dwell = count.clone()
      latch.pend_tilt = zero.clone()
      latch.pend_yaw = zero.clone()
    setattr(env, attr, latch)
    _remember_latch(env, attr)
  if latch.jaw != jaw:
    raise ValueError(
      f"the release latch was built with jaw={latch.jaw}, read with jaw={jaw}: every "
      "reader of one env's latch must define the release the same way"
    )
  if latch.surface != surface:
    raise ValueError(
      f"the release latch was built with surface={latch.surface}, read with "
      f"surface={surface}: every reader of one env's latch must measure the release height "
      "from the same surface (plan S10: the box's rim)"
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
  reference = stack_command_reference(env, command_name)
  if reference is None:
    rest_z = surface_top_z(env, surface_cfg) + object_half_height(env, object_name)
    ok = torch.zeros_like(let_go)
  else:
    rest_z = goal[:, 2]
    ok = stack_state(env, object_name, reference, command_name).ok
  dxy = torch.norm((pos - goal)[:, :2], dim=-1)
  speed = torch.norm(obj.data.root_link_lin_vel_w, dim=-1)
  on_table = _on_surface_bool(env, support_sensor_name)
  height = pos[:, 2] - rest_z
  tilt = object_tilt(env, object_name)
  yaw = yaw_error_mod_90(
    obj.data.root_link_quat_w, yaw_reference_quat(env, command_name)
  )
  if jaw is None:
    opened = let_go
    now = (dxy, height, on_table, speed, latch.dwell, ok, tilt, yaw)
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
    latch.pend_ok = w(ok, latch.pend_ok)
    latch.pend_tilt = w(tilt, latch.pend_tilt)
    latch.pend_yaw = w(yaw, latch.pend_yaw)
    latch.pend = latch.pend | let_go
    released_now = _jaw_open_bool(env, jaw_cfg, jaw_open) & ~grasped
    opened = latch.jaw_closed & released_now & was_lifted
    p = latch.pend
    now = (
      torch.where(p, latch.pend_dxy, dxy),
      torch.where(p, latch.pend_height, height),
      torch.where(p, latch.pend_on_table, on_table),
      torch.where(p, latch.pend_speed, speed),
      torch.where(p, latch.pend_dwell, latch.dwell),
      torch.where(p, latch.pend_ok, ok),
      torch.where(p, latch.pend_tilt, tilt),
      torch.where(p, latch.pend_yaw, yaw),
    )
    latch.pend = latch.pend & ~opened
    latch.jaw_closed = ~released_now
  latch.dxy = torch.where(opened, now[0], latch.dxy)
  latch.height = torch.where(opened, now[1], latch.height)
  latch.rel_on_table = torch.where(opened, now[2], latch.rel_on_table)
  latch.rel_speed = torch.where(opened, now[3], latch.rel_speed)
  latch.rel_dwell = torch.where(opened, now[4], latch.rel_dwell)
  latch.rel_ok = torch.where(opened, now[5], latch.rel_ok)
  latch.tilt = torch.where(opened, now[6], latch.tilt)
  latch.yaw = torch.where(opened, now[7], latch.yaw)
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


def _at_goal(
  goal: torch.Tensor, pos: torch.Tensor, threshold: float | tuple[float, float]
) -> torch.Tensor:
  """The object is at the goal. (B,)

  A float ``threshold`` is the 3-D distance, every run up to S7-4. A pair ``(xy, z)`` is plan
  S7-5's Z: the horizontal distance below ``xy`` and the height off the goal below ``z`` in
  magnitude. The goal is the object's resting centre on the surface (`SurfaceOffsetCommand`,
  or `TARGET_Z` for the fixed goal), so ``z`` bounds the centre above its resting height: a
  5 cm cube on an edge stands 1.0 cm up, on a corner 1.8 cm.
  """
  if isinstance(threshold, (tuple, list)):
    xy, z = threshold
    d = pos - goal
    return (torch.norm(d[:, :2], dim=-1) < xy) & (d[:, 2].abs() < z)
  return torch.norm(goal - pos, dim=-1) < threshold


def _placed_bool(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  grasp_sensor_name: str,
  threshold: float | tuple[float, float],
  velocity_threshold: float,
  support_sensor_name: str | None = None,
  jaw_cfg: SceneEntityCfg | None = None,
  jaw_open: float | BySize | PerShape | None = None,
) -> torch.Tensor:
  """At the goal, released, and at rest. Shared by the success reward and termination.

  With ``jaw_cfg`` and ``jaw_open`` (plan S7-4, module J) "released" also needs the jaw open,
  below ``jaw_open``: an object that slipped out of a jaw still at the hold angle is not.
  ``threshold`` is a 3-D distance or plan S7-5's (xy, z) tolerance, `_at_goal`.

  `released` is `¬grasped`, which a cube that was never picked up also satisfies -- so on a
  table-height goal this counts a slide as a place. Passing `support_sensor_name` adds the
  `was_lifted` latch and closes that path; leaving it None is the pre-C2 behaviour, which
  R-A and R-C keep so their numbers stay comparable.
  """
  obj: Entity = env.scene[object_name]
  command = env.command_manager.get_term(command_name)
  at_goal = _at_goal(command.command, obj.data.root_link_pos_w, threshold)
  released = ~_grasped_bool(env, grasp_sensor_name)
  if jaw_cfg is not None:
    assert jaw_open is not None
    released &= _jaw_open_bool(env, jaw_cfg, jaw_open)
  settled = torch.norm(obj.data.root_link_lin_vel_w, dim=-1) < velocity_threshold
  placed = at_goal & released & settled
  if support_sensor_name is not None:
    placed &= _was_lifted_bool(env, grasp_sensor_name, support_sensor_name)
  return placed


def release_grade(env: ManagerBasedRlEnv, **params) -> torch.Tensor:
  """M6's grade of the latest release, as `rewards.release_quality` pays it (the floor before
  any). (B, 1)

  Plan S7-9's PQ, a critic-only input: M6 latches q at the let-go and multiplies every later
  ``release`` / ``placed`` step by it, but the critic sees the cube where it settled, not where it
  was let go -- two states with one observation earn x0.25 and x1.0. ``params`` are the graded
  ``placed`` term's, so the critic sees the grade the reward actually pays.
  """
  from pal_mjlab.tasks.manipulation.mdp.rewards import release_quality

  return release_quality(env, **params).unsqueeze(-1)


def release_latched(env: ManagerBasedRlEnv, **params) -> torch.Tensor:
  """1 once a release after `was_lifted` has been latched this episode, else 0 (plan S7-9's PQ,
  critic only; ``params`` as `release_grade`). (B, 1)"""
  latch = _release_latch(
    env,
    params["command_name"],
    params["object_name"],
    params["grasp_sensor_name"],
    params["support_sensor_name"],
    params["surface_cfg"],
    params.get("jaw_cfg"),
    params.get("jaw_open"),
  )
  return latch.released.float().unsqueeze(-1)
