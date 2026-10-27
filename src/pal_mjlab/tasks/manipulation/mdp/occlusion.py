"""Plan S7-7: the cube goes out of the camera's view, and what the actor is told meanwhile.

On the robot the cube is tracked by FoundationPose from the ``torso_front`` camera, and the
arm reaching for the cube, the jaw around it and the cube in the hand hide it. Up to S7-6 the
actor saw the cube on every step (late and noisy, never missing). This module adds

* a visibility model -- Claudius's line of sight (`camera_view_blocked`, `origin/faster-moves`
  90c84f5, ported unchanged) used as a test, not a penalty, plus his random tracker loss
  (eb05a76: ``(rate, min_steps, max_steps)``);
* the actor's *filler* while the cube is not visible (``OccCfg.fill``): zeros, the last
  reported pose, the pose after the reset, or O-K's rule-based estimate;
* the terms that report all of it: `object_position_occ`, `object_yaw_occ` (the actor's cube
  terms, replacing `object_position_b` / `object_yaw_b`), `object_seen` (the flag) and
  `object_unseen_steps` (the critic's).

One `CubeSensor` per object, shared by every term that reads it, updated once per control step
(as `_ObjectTracker`, which it wraps for the pose's lag). The critic reads the live cube.

Everything here is a function of what the deployer also has -- the reported cube pose and its
visibility, the TCP pose, the jaw's measured angle and its command -- so O-K's rule can be
mirrored in ``pal_policy_inferer`` and parity-tested on logged trajectories.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  euler_xyz_from_quat,
  quat_apply,
  quat_inv,
  quat_mul,
)

from .observations import _base_pose, _object_pose_w, _to_base_frame, object_tracker

_ROBOT = SceneEntityCfg("robot")

#: Cube half-size the line-of-sight test aims at (the nominal cube; `scene.get_cube_cfg`).
CUBE_HALF_SIZE = 0.025

# -- The camera and the arm as capsules (Claudius, 90c84f5) ---------------------------------
#: The ``torso_front`` RGB-D camera's colour optical frame in the torso, from the hand-eye
#: calibration of 2026-09-29 (``kangaroo_camera_calibration``, ``torso_front_2026-09-29_09-41-53``):
#: ``T_base_optical`` in ``torso_link``. The model has no ``torso_link`` body; the torso mesh
#: hangs off ``pelvis_2_link`` 6.1 mm up, unrotated, so that offset is added in z. Only the
#: position is used -- the line of sight assumes the cube is always inside the image.
TORSO_FRONT_CAMERA_BODY = "pelvis_2_link"
TORSO_FRONT_CAMERA_POS = (0.1235, 0.0267, 0.4147 + 0.0061)

Segment = tuple[
  tuple[str, tuple[float, float, float]], tuple[str, tuple[float, float, float]], float
]

_ARM_RADIUS = 0.05
_PALM_RADIUS = 0.04
#: The left arm as capsules: its link origins joined in chain order, then the wrist to the jaw's
#: knuckles 90 mm out along the flange. The fingers are left out (`LEFT_FINGER_SEGMENTS`).
LEFT_ARM_SIGHT_SEGMENTS: tuple[Segment, ...] = (
  *(
    (
      (f"arm_left_{i}_link", (0.0, 0.0, 0.0)),
      (f"arm_left_{i + 1}_link", (0.0, 0.0, 0.0)),
      _ARM_RADIUS,
    )
    for i in range(2, 7)
  ),
  (
    ("arm_left_7_link", (0.0, 0.0, 0.0)),
    ("arm_left_7_link", (0.0, 0.0, 0.09)),
    _PALM_RADIUS,
  ),
)

#: The switch behind ``OCC=hand``: the two fingers as capsules, knuckles to fingertip
#: (radius ``_FINGER_RADIUS``). On the robot the jaw around a held cube hides part of it, which
#: Claudius's penalty deliberately ignores. Set by 7-7.0's render check.
_FINGER_RADIUS = 0.015
LEFT_FINGER_SEGMENTS: tuple[Segment, ...] = tuple(
  (
    ("arm_left_7_link", (0.0, 0.0, 0.09)),
    (f"gripper_left_fingertip_{side}_link", (0.0, 0.0, 0.0)),
    _FINGER_RADIUS,
  )
  for side in ("left", "right")
)

#: Clearance from the line over which the blocked score falls from 1 to 0.
CAMERA_VIEW_MARGIN = 0.02


def segment_distance(
  p0: torch.Tensor, p1: torch.Tensor, q0: torch.Tensor, q1: torch.Tensor
) -> torch.Tensor:
  """Closest distance between segments ``p0-p1`` and ``q0-q1``. (B,)

  Claudius's `_segment_distance` (`origin/faster-moves` 90c84f5, Ericson, *Real-Time Collision
  Detection* 5.1.9), unchanged: both parameters clamped to the segments; a zero-length segment is
  a point.
  """
  eps = 1e-9
  d1, d2, r = p1 - p0, q1 - q0, p0 - q0
  a = (d1 * d1).sum(-1)
  e = (d2 * d2).sum(-1)
  b = (d1 * d2).sum(-1)
  c = (d1 * r).sum(-1)
  f = (d2 * r).sum(-1)
  denom = a * e - b * b
  s = torch.where(
    denom > eps, ((b * f - c * e) / denom.clamp_min(eps)).clamp(0.0, 1.0), 0.0
  )
  t = torch.where(e > eps, (b * s + f) / e.clamp_min(eps), 0.0)
  t_clamped = t.clamp(0.0, 1.0)
  s = torch.where(
    (t == t_clamped) & (e > eps),
    s,
    ((b * t_clamped - c) / a.clamp_min(eps)).clamp(0.0, 1.0),
  )
  closest_p = p0 + s.unsqueeze(-1) * d1
  closest_q = q0 + t_clamped.unsqueeze(-1) * d2
  return torch.norm(closest_p - closest_q, dim=-1)


def camera_view_blocked(
  env: ManagerBasedRlEnv,
  camera_body: str,
  camera_pos: tuple[float, float, float],
  segments: Sequence[Segment],
  target_w: torch.Tensor,
  stop_short: float,
  margin: float,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """How much of the robot stands between a body-mounted camera and ``target_w``. (B,)

  Claudius's `mdp.camera_view_blocked` (90c84f5), with the target's position passed in rather
  than read from the simulator. The sight line runs from ``camera_pos`` in ``camera_body``'s frame
  to the target, stopping ``stop_short`` before it. The robot is a set of capsules; a capsule
  scores 1 once it touches the line, falling linearly to 0 at ``margin`` clear of it; the term is
  the worst capsule's.
  """
  robot: Entity = env.scene[asset_cfg.name]
  pos_w = robot.data.body_link_pos_w
  quat_w = robot.data.body_link_quat_w

  def point(body: str, offset: tuple[float, float, float]) -> torch.Tensor:
    i = robot.body_names.index(body)
    local = torch.tensor(offset, device=pos_w.device, dtype=pos_w.dtype)
    return pos_w[:, i] + quat_apply(quat_w[:, i], local.expand(pos_w.shape[0], 3))

  camera = point(camera_body, camera_pos)
  ray = target_w - camera
  length = torch.norm(ray, dim=-1, keepdim=True).clamp_min(1e-6)
  end = camera + ray * ((length - stop_short).clamp_min(0.0) / length)
  cost = torch.zeros(pos_w.shape[0], device=pos_w.device, dtype=pos_w.dtype)
  for (body_a, offset_a), (body_b, offset_b), radius in segments:
    dist = segment_distance(
      camera, end, point(body_a, offset_a), point(body_b, offset_b)
    )
    cost = torch.maximum(cost, ((radius + margin - dist) / margin).clamp(0.0, 1.0))
  return cost


# -- Configuration --------------------------------------------------------------------------

FILLS = ("live", "hold", "zero", "estimate", "estimate+", "snap")


@dataclasses.dataclass(frozen=True)
class OccCfg:
  """Plan S7-7's occlusion model and filler (hashable: it rides in term params)."""

  fill: str
  """What the actor's cube terms report while the cube is not visible. ``live`` (no filler, the
  cube is never lost), ``hold`` (the last reported pose: O-H, Claudius's ``-Drop``), ``zero``
  (zeros: O-Z), ``estimate`` (O-K), ``estimate+`` (O-K+: the estimate whenever the cube is held,
  visible or not), ``snap`` (O-S: the pose after the reset, never updated)."""

  flag: bool
  """Whether the actor gets the ``object_seen`` term (39-D). The trained arms O-Z and O-K do;
  O-S does not; the eval-only fillers on a 38-D policy do not."""

  view: str = "arm"
  """The line of sight: ``arm`` (Claudius's model as is: the line stops 3 cm short of the cube,
  fingers left out), ``hand`` (the line runs to the cube's near face and the fingers are
  capsules), ``none`` (only the random loss)."""

  view_threshold: float = 1.0
  """The cube is not visible when the blocked score reaches this (1 = a capsule touching the
  line)."""

  dropout: tuple[float, int, int] | None = (0.01, 5, 50)
  """Random tracker loss ``(rate per step, min steps, max steps)`` (Claudius's ``-Drop``); a
  stand-in until the real tracker's rate is measured. None for none."""

  # O-K's "held" rule. Calibrated by 7-7.0 (`scripts/calibrate_held_rule.py`, `--trace` of RCTP s1
  # / s3 at their own lag and at `LAG=FP+JS`, 64 episodes each, against the contact sensor): the
  # F1 optimum of a 1620-point grid, with the lower band edge kept low for edge/corner grasps.
  hold_band: tuple[float, float] = (0.10, 0.43)
  """The measured jaw angle (rad) inside which a stalled, commanded-closed jaw counts as
  holding the cube. 5 cm cube: first touch ~0.21, hold 0.26-0.40 (S7-5's trace; 5-95 % of the
  contact sensor's grasped steps 0.15-0.39 here), edge/corner grasps touch at ~0.10."""
  hold_stall_vel: float = 1.5
  """|dq/dt| of the jaw below which it counts as stalled, rad/s. The jaw is a stiff PD on a
  compliant grasp: its speed while holding is 0.15 rad/s (median) but 0.9-1.3 (90th percentile),
  so a tight stall test (0.15) missed 80 % of the holds."""
  hold_margin: float = 0.10
  """The command must be this much further closed than the measured angle, rad."""
  hold_steps: int = 3
  """Consecutive control steps the three conditions must hold. The whole hold is short (median 14
  steps = 0.3 s on G20), so the rule's own trail (N - 1 steps) matters: 2-3 steps."""
  hold_release: float = 0.0
  """Not held again when the command comes back to the measured angle, rad: the jaw is being
  opened. The pads lose the cube about 8 steps before the jaw has opened below `open_below`; this
  closes ~60 % of that gap (stale "held" steps 35 % -> 14 % of the rule's held steps)."""
  open_below: float = 0.15
  """Not held again when the jaw opens below this (`observations.JAW_OPEN_BELOW`), rad."""
  envelope: float = 0.20
  """The last visible cube must be within this distance of the TCP when the hold latches, m
  (sim2sim's virtual-cube rule, ``kang_pick_place_node.cpp``). Loose on purpose: at 0.06 the rule
  lost a third of the holds (the TCP site sits 4-5 cm from the cube's centre in a grasp, and the
  last seen pose is up to 8 steps old); it only guards a jaw closing on air far from the cube."""
  reacquire: float = 0.03
  """A measured pose further than this from the estimate overrides the held belief, m."""

  noise: tuple[float, float, float, float] = (0.008, 0.002, 0.03, 0.01)
  """(position bias, position jitter, yaw bias, yaw jitter) of the reported pose as the actor
  sees it, m / rad -- the DR-obs values, applied here (not by mjlab's noise model) so that a
  filler is not noised: zeros are exactly zero, as the deployer sends them."""

  def __post_init__(self) -> None:
    if self.fill not in FILLS:
      raise ValueError(f"unknown filler {self.fill!r}; have {FILLS}")
    if self.view not in ("arm", "hand", "none"):
      raise ValueError(f"unknown view model {self.view!r}; have arm, hand, none")

  @property
  def occludes(self) -> bool:
    return self.fill != "live"


#: Occlusion models by ``OCC`` (plan 7-7.0 item 1): the visibility model alone.
OCC_VIEWS = ("none", "arm", "hand")
#: The fillers by ``OBJ_OBS``, and whether the actor gets the flag by default.
OBJ_OBS_FLAG = {
  "live": False,
  "hold": False,
  "zero": True,
  "estimate": True,
  "estimate+": False,
  "snap": False,
}


def occ_cfg(occ: str, obj_obs: str, flag: bool | None = None) -> OccCfg | None:
  """The occlusion config for the ``OCC`` / ``OBJ_OBS`` selectors; None for no occlusion
  (``OCC=none`` with ``OBJ_OBS=live``, every run up to S7-6).

  ``OCC`` names the visibility model (``none`` | ``arm`` | ``hand``), ``OBJ_OBS`` the filler
  (``live`` | ``hold`` | ``zero`` | ``estimate`` | ``estimate+`` | ``snap``); ``flag`` overrides
  whether the actor gets the visibility flag. ``OCC=none`` with a filler is the filler's machinery
  with nothing hidden -- the cube is always seen, the flag reads 1, the estimate is computed but
  reported only for ``estimate+`` -- which is how a trained policy is evaluated *without*
  occlusion, and how O-S (independent of occlusion) is trained.
  """
  if occ not in OCC_VIEWS:
    raise ValueError(f"unknown OCC={occ!r}; have {OCC_VIEWS}")
  if obj_obs not in OBJ_OBS_FLAG:
    raise ValueError(f"unknown OBJ_OBS={obj_obs!r}; have {tuple(OBJ_OBS_FLAG)}")
  if obj_obs == "live":
    if occ != "none":
      raise ValueError("OBJ_OBS=live with an occlusion model would hide nothing")
    return None
  has_flag = OBJ_OBS_FLAG[obj_obs] if flag is None else flag
  if occ == "none":
    return OccCfg(fill=obj_obs, flag=has_flag, view="none", dropout=None)
  return OccCfg(fill=obj_obs, flag=has_flag, view=occ)


# -- The sensor -----------------------------------------------------------------------------


def _history(tensor: torch.Tensor, depth: int) -> torch.Tensor:
  return tensor.unsqueeze(0).repeat(depth, *([1] * tensor.dim()))


class CubeSensor:
  """What the robot knows of the cube: its reported pose, whether it is visible, and (O-K) the
  rule-based estimate of where it is while it is not.

  Built on `_ObjectTracker` (``lag`` steps of stale pose, drawn per env per episode): the
  visibility is recorded at the live step and reported with the *same* lag, so a lost cube is
  reported lost as late as its pose would have been -- the tracker reports what it saw 100-150 ms
  ago. The TCP pose is kept in a history of the same depth, so O-K can pair the lagged cube pose
  with the TCP pose from the same step (else its stored offset is off by lag x speed).
  """

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    object_name: str,
    lag: tuple[int, int],
    occ: OccCfg,
    tcp_cfg: SceneEntityCfg,
    jaw_name: str,
  ):
    self.name = object_name
    self.lag_range = (int(lag[0]), int(lag[1]))
    self.occ = occ
    self._tcp_cfg = tcp_cfg
    self._jaw_name = jaw_name
    n, dev = env.num_envs, env.device
    depth = self.lag_range[1] + 1
    self._depth = depth
    self._n = n
    self._envs = torch.arange(n, device=dev)
    # Histories, newest first (an index by the lag is the stale sample).
    self._vis = torch.ones(depth, n, dtype=torch.bool, device=dev)
    self._tcp_pos = torch.zeros(depth, n, 3, device=dev)
    self._tcp_quat = torch.zeros(depth, n, 4, device=dev)
    self._tcp_quat[..., 0] = 1.0
    self._lost = torch.zeros(n, dtype=torch.long, device=dev)  # dropout steps left
    self._pending = torch.ones(n, dtype=torch.bool, device=dev)
    self._step = -1
    # The reported state.
    self.seen = torch.ones(n, dtype=torch.bool, device=dev)
    self.tracker_lag = torch.zeros(n, dtype=torch.long, device=dev)
    self.unseen_steps = torch.zeros(n, dtype=torch.long, device=dev)
    self.live_visible = torch.ones(n, dtype=torch.bool, device=dev)
    self.blocked_score = torch.zeros(n, device=dev)
    self.pos_w = torch.zeros(n, 3, device=dev)
    self.quat_w = torch.zeros(n, 4, device=dev)
    self.quat_w[:, 0] = 1.0
    self.last_pos = self.pos_w.clone()
    self.last_quat = self.quat_w.clone()
    # O-K.
    self.held = torch.zeros(n, dtype=torch.bool, device=dev)
    self._count = torch.zeros(n, dtype=torch.long, device=dev)
    # A belief the cube contradicted stays dropped until the jaw stops meeting the hold conditions.
    self._disarmed = torch.zeros(n, dtype=torch.bool, device=dev)
    self._has_offset = torch.zeros(n, dtype=torch.bool, device=dev)
    self._off_pos = torch.zeros(n, 3, device=dev)
    self._off_quat = self.quat_w.clone()
    self._jaw_prev = torch.zeros(n, device=dev)
    self.est_pos = self.pos_w.clone()
    self.est_quat = self.quat_w.clone()
    #: O-S / noise: per-episode draws, and the snapshot in the base frame.
    self._bias_pos = torch.zeros(n, 3, device=dev)
    self._bias_yaw = torch.zeros(n, 2, device=dev)
    self._snap_pos = torch.zeros(n, 3, device=dev)
    self._snap_yaw = torch.zeros(n, 2, device=dev)
    self._snap_pending = torch.ones(n, dtype=torch.bool, device=dev)
    self._segments = {
      "none": (),
      "arm": LEFT_ARM_SIGHT_SEGMENTS,
      "hand": (*LEFT_ARM_SIGHT_SEGMENTS, *LEFT_FINGER_SEGMENTS),
    }[occ.view]
    self._stop_short = CUBE_HALF_SIZE + 0.03 if occ.view == "arm" else CUBE_HALF_SIZE
    self._jaw_id: int | None = None
    self._jaw_action_idx: int | None = None

  # -- per-step update --------------------------------------------------------------------

  def _jaw(self, env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """The jaw's measured angle and its command. (B,), (B,)"""
    robot: Entity = env.scene["robot"]
    if self._jaw_id is None:
      self._jaw_id = robot.joint_names.index(self._jaw_name)
      term = env.action_manager.get_term("joint_pos")
      names = [n.split("/")[-1] for n in term._target_names]
      self._jaw_action_idx = names.index(self._jaw_name)
    q = robot.data.joint_pos[:, self._jaw_id]
    cmd = env.action_manager.get_term("joint_pos")._processed_actions[
      :, self._jaw_action_idx
    ]
    return q, cmd

  def _tcp(self, env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
    robot: Entity = env.scene["robot"]
    site = self._tcp_cfg.site_ids
    return (
      robot.data.site_pos_w[:, site].squeeze(1),
      robot.data.site_quat_w[:, site].squeeze(1),
    )

  def update(self, env: ManagerBasedRlEnv) -> None:
    """Advance one control step -- once per env step, however many terms ask.

    A full ``env.reset()`` does not advance the step counter: a second call at the same count
    only refreshes the envs that were reset (their history, belief and noise draws), without
    advancing anyone's clocks.
    """
    new_step = env.common_step_counter != self._step
    self._step = env.common_step_counter
    self._pending |= env.episode_length_buf == 0
    if not new_step and not bool(self._pending.any()):
      return
    n = self._n
    obj: Entity = env.scene[self.name]
    live_pos = obj.data.root_link_pos_w
    # The tracker's lagged pose (it keeps its own history and once-per-step guard).
    rep_pos, rep_quat = _object_pose_w(env, self.name, self.lag_range)
    tracker = object_tracker(env, self.name)
    lag = tracker.lag
    self.tracker_lag = lag

    # -- visibility at the live step, into the history ------------------------------------
    if self._segments:
      score = camera_view_blocked(
        env,
        TORSO_FRONT_CAMERA_BODY,
        TORSO_FRONT_CAMERA_POS,
        self._segments,
        live_pos,
        self._stop_short,
        CAMERA_VIEW_MARGIN,
      )
    else:
      score = torch.zeros(n, device=env.device)
    self.blocked_score = score
    live_vis = score < self.occ.view_threshold
    if self.occ.dropout is not None:
      rate, lo, hi = self.occ.dropout
      if new_step:
        self._lost = (self._lost - 1).clamp_min(0)
        start = (self._lost == 0) & (torch.rand(n, device=env.device) < rate)
        if bool(start.any()):
          self._lost[start] = torch.randint(
            lo, hi + 1, (int(start.sum()),), device=env.device
          )
      live_vis = live_vis & (self._lost == 0)
    self.live_visible = live_vis
    tcp_pos, tcp_quat = self._tcp(env)
    if new_step:
      self._vis = torch.roll(self._vis, 1, dims=0)
      self._tcp_pos = torch.roll(self._tcp_pos, 1, dims=0)
      self._tcp_quat = torch.roll(self._tcp_quat, 1, dims=0)
    self._vis[0] = live_vis
    self._tcp_pos[0] = tcp_pos
    self._tcp_quat[0] = tcp_quat
    if bool(self._pending.any()):
      fresh = self._pending
      self._vis[:, fresh] = live_vis[fresh]
      self._tcp_pos[:, fresh] = tcp_pos[fresh]
      self._tcp_quat[:, fresh] = tcp_quat[fresh]
      self._lost[fresh] = 0
    self.seen = self._vis[lag, self._envs]
    tcp_lag_pos = self._tcp_pos[lag, self._envs]
    tcp_lag_quat = self._tcp_quat[lag, self._envs]

    q_jaw, cmd_jaw = self._jaw(env)
    if bool(self._pending.any()):
      fresh = self._pending
      self.last_pos[fresh] = rep_pos[fresh]
      self.last_quat[fresh] = rep_quat[fresh]
      self.unseen_steps[fresh] = 0
      self.held[fresh] = False
      self._count[fresh] = 0
      self._disarmed[fresh] = False
      self._has_offset[fresh] = False
      self._jaw_prev[fresh] = q_jaw[fresh]

      def draw(*shape: int, scale: float) -> torch.Tensor:
        return (torch.rand(*shape, device=env.device) * 2 - 1) * scale

      self._bias_pos[fresh] = draw(int(fresh.sum()), 3, scale=self.occ.noise[0])
      self._bias_yaw[fresh] = draw(int(fresh.sum()), 2, scale=self.occ.noise[2])
      self._snap_pending |= fresh
      self._pending = torch.zeros_like(self._pending)
      if not new_step:
        # A reset at the same step count: report the fresh envs' cube, nothing else advances.
        self.seen[fresh] = True
        self.pos_w[fresh] = rep_pos[fresh]
        self.quat_w[fresh] = rep_quat[fresh]
        self.est_pos[fresh] = rep_pos[fresh]
        self.est_quat[fresh] = rep_quat[fresh]
        return

    # -- the pose while visible, and how long it has not been --------------------------------
    self.unseen_steps = torch.where(
      self.seen, torch.zeros_like(self.unseen_steps), self.unseen_steps + 1
    )

    # -- O-K's held rule and estimate ---------------------------------------------------------
    occ = self.occ
    dt = env.step_dt
    qdot = (q_jaw - self._jaw_prev) / dt
    self._jaw_prev = q_jaw.clone()
    lo, hi = occ.hold_band
    cand = (
      (q_jaw >= lo)
      & (q_jaw <= hi)
      & (qdot.abs() < occ.hold_stall_vel)
      & ((cmd_jaw - q_jaw) >= occ.hold_margin)
    )
    self._count = torch.where(cand, self._count + 1, torch.zeros_like(self._count))
    # The last cube seen (before this step's report), against the TCP.
    seen_now = self.seen
    last_pos = torch.where(seen_now.unsqueeze(-1), rep_pos, self.last_pos)
    last_quat = torch.where(seen_now.unsqueeze(-1), rep_quat, self.last_quat)
    in_envelope = torch.norm(last_pos - tcp_pos, dim=-1) < occ.envelope
    latch = (
      (~self.held) & ~self._disarmed & (self._count >= occ.hold_steps) & in_envelope
    )
    # Re-acquisition: the offset the belief holds, evaluated at the reported pose's moment,
    # against what was measured -- a cube that is not in the hand any more overrides the belief.
    if bool(self._has_offset.any()):
      pred_pos = tcp_lag_pos + quat_apply(tcp_lag_quat, self._off_pos)
      miss = torch.norm(rep_pos - pred_pos, dim=-1) > occ.reacquire
      drop = self.held & self._has_offset & seen_now & miss
    else:
      drop = torch.zeros_like(self.held)
    let_go = self.held & (
      (q_jaw < occ.open_below) | (q_jaw > hi) | ((cmd_jaw - q_jaw) < occ.hold_release)
    )
    stop = drop | let_go
    self._disarmed = (self._disarmed | drop) & cand
    if bool(let_go.any()):
      # The cube is roughly where it was let go: at the TCP, in the pose the estimate had.
      self.last_pos = torch.where(let_go.unsqueeze(-1), self.est_pos, self.last_pos)
      self.last_quat = torch.where(let_go.unsqueeze(-1), self.est_quat, self.last_quat)
      last_pos = torch.where(let_go.unsqueeze(-1), self.est_pos, last_pos)
      last_quat = torch.where(let_go.unsqueeze(-1), self.est_quat, last_quat)
    self.held = (self.held | latch) & ~stop
    self._has_offset = self._has_offset & ~stop
    self._count = torch.where(stop, torch.zeros_like(self._count), self._count)
    # The offset, from every visible step while held (and at the latch, if it is visible).
    upd = self.held & seen_now
    if bool(upd.any()):
      inv = quat_inv(tcp_lag_quat)
      off = quat_apply(inv, rep_pos - tcp_lag_pos)
      off[:, 0] = 0.0  # a parallel jaw centres the cube on the closing axis (TCP x)
      self._off_pos = torch.where(upd.unsqueeze(-1), off, self._off_pos)
      self._off_quat = torch.where(
        upd.unsqueeze(-1), quat_mul(inv, rep_quat), self._off_quat
      )
      self._has_offset = self._has_offset | upd
    # The estimate now: the cube rides with the TCP while held, else where it was last seen.
    riding = self.held & self._has_offset
    ride_pos = tcp_pos + quat_apply(tcp_quat, self._off_pos)
    ride_quat = quat_mul(tcp_quat, self._off_quat)
    self.est_pos = torch.where(riding.unsqueeze(-1), ride_pos, last_pos)
    self.est_quat = torch.where(riding.unsqueeze(-1), ride_quat, last_quat)
    self.last_pos = last_pos
    self.last_quat = last_quat

    # -- the report --------------------------------------------------------------------------
    fill = occ.fill
    lost = ~self.seen
    if fill in ("estimate", "estimate+"):
      use_est = lost
      if fill == "estimate+":
        use_est = lost | riding
      self.pos_w = torch.where(use_est.unsqueeze(-1), self.est_pos, rep_pos)
      self.quat_w = torch.where(use_est.unsqueeze(-1), self.est_quat, rep_quat)
    elif fill == "hold":
      self.pos_w = torch.where(lost.unsqueeze(-1), self.last_pos, rep_pos)
      self.quat_w = torch.where(lost.unsqueeze(-1), self.last_quat, rep_quat)
    else:  # zero, snap: the terms fill; the measured pose rides along
      self.pos_w, self.quat_w = rep_pos, rep_quat

  # -- what the terms read ----------------------------------------------------------------

  def output(self, env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """The actor's cube position (base frame, (B, 3)) and yaw ``[cos, sin]`` ((B, 2)), noised,
    with the filler applied."""
    robot: Entity = env.scene["robot"]
    pos_b = _to_base_frame(robot, self.pos_w)
    quat_b = quat_mul(quat_inv(_base_pose(robot)[1]), self.quat_w)
    _, _, yaw = euler_xyz_from_quat(quat_b)
    yaw_cs = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
    if env.cfg.observations["actor"].enable_corruption:
      _, pos_noise, _, yaw_noise = self.occ.noise
      pos_b = pos_b + self._bias_pos + (torch.rand_like(pos_b) * 2 - 1) * pos_noise
      yaw_cs = yaw_cs + self._bias_yaw + (torch.rand_like(yaw_cs) * 2 - 1) * yaw_noise
    fill = self.occ.fill
    if fill == "zero":
      lost = (~self.seen).unsqueeze(-1)
      pos_b = torch.where(lost, torch.zeros_like(pos_b), pos_b)
      yaw_cs = torch.where(lost, torch.zeros_like(yaw_cs), yaw_cs)
    elif fill == "snap":
      if bool(self._snap_pending.any()):
        f = self._snap_pending
        self._snap_pos[f] = pos_b[f]
        self._snap_yaw[f] = yaw_cs[f]
        self._snap_pending = torch.zeros_like(self._snap_pending)
      pos_b, yaw_cs = self._snap_pos.clone(), self._snap_yaw.clone()
    return pos_b, yaw_cs


def cube_sensor(
  env: ManagerBasedRlEnv,
  object_name: str,
  lag: tuple[int, int],
  occ: OccCfg,
  tcp_cfg: SceneEntityCfg,
  jaw_name: str,
) -> CubeSensor:
  """The sensor for ``object_name``, built on first use and updated for this step."""
  sensors = getattr(env, "_pal_cube_sensors", None)
  if sensors is None:
    sensors = {}
    env._pal_cube_sensors = sensors  # type: ignore[attr-defined]
  sensor = sensors.get(object_name)
  if sensor is None:
    sensor = CubeSensor(env, object_name, lag, occ, tcp_cfg, jaw_name)
    sensors[object_name] = sensor
  elif sensor.occ != occ:
    raise ValueError(f"object {object_name!r} is sensed under two occlusion configs")
  sensor.update(env)
  return sensor


def object_position_occ(
  env: ManagerBasedRlEnv,
  object_name: str,
  lag: tuple[int, int],
  occ: OccCfg,
  tcp_cfg: SceneEntityCfg,
  jaw_name: str,
) -> torch.Tensor:
  """The actor's cube position in the base frame under occlusion (`CubeSensor.output`). (B, 3)"""
  return cube_sensor(env, object_name, lag, occ, tcp_cfg, jaw_name).output(env)[0]


def object_yaw_occ(
  env: ManagerBasedRlEnv,
  object_name: str,
  lag: tuple[int, int],
  occ: OccCfg,
  tcp_cfg: SceneEntityCfg,
  jaw_name: str,
) -> torch.Tensor:
  """The actor's cube yaw ``[cos, sin]`` under occlusion. (B, 2)"""
  return cube_sensor(env, object_name, lag, occ, tcp_cfg, jaw_name).output(env)[1]


def object_seen(
  env: ManagerBasedRlEnv,
  object_name: str,
  lag: tuple[int, int],
  occ: OccCfg,
  tcp_cfg: SceneEntityCfg,
  jaw_name: str,
) -> torch.Tensor:
  """1.0 while the tracker reports the cube, 0.0 while it does not (reported with the pose's
  lag). (B, 1)"""
  sensor = cube_sensor(env, object_name, lag, occ, tcp_cfg, jaw_name)
  return sensor.seen.float().unsqueeze(-1)


#: Steps since the cube was last seen that the critic's term saturates at (1 s).
UNSEEN_STEPS_SCALE = 50.0


def object_unseen_steps(
  env: ManagerBasedRlEnv,
  object_name: str,
  lag: tuple[int, int],
  occ: OccCfg,
  tcp_cfg: SceneEntityCfg,
  jaw_name: str,
) -> torch.Tensor:
  """The critic's steps since the cube was last seen, / `UNSEEN_STEPS_SCALE`, clipped to 1. (B, 1)"""
  sensor = cube_sensor(env, object_name, lag, occ, tcp_cfg, jaw_name)
  return (sensor.unseen_steps.float() / UNSEEN_STEPS_SCALE).clamp(max=1.0).unsqueeze(-1)
