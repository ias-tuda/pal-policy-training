"""A scripted pick-and-place, driven through the exact action interface the policy uses.

S5.0 of the pick-and-place plan. C1 and C2 spent ~750 M steps each on a task that was not
possible through its action space: the jaw could be commanded to 22 % of its range and the
arm could move the grasp point ~12 cm. S1 had proved the *robot* could reach the cube with
IK over the full joint limits; nothing had proved the *policy* could. This is that proof,
and it is two things:

``derive``
    Solve IK for every configuration the task needs -- pre-grasp, grasp, lift, carry, place,
    retreat -- at the spawn box's corners (x, y +-2 cm, yaw +-10 deg), its centre and the
    target, and turn the joint excursions into action scales by the plan's rule:
    ``scale_j = max(current_j, 1.5 * max |q_ik,j - q_default,j|)``, rounded up to 0.05 rad
    and capped at half the joint's range. CPU MuJoCo only, no env.

``rollout``
    Build the real env, put the cube at those corners (and at random jitters), and run the
    waypoints as raw actions ``a = (q_target - offset) / scale``, clipped to [-1, 1], through
    the env's own ``JointPositionAction`` at 50 Hz. Reports, per episode, whether the grasp
    sensor fired, the cube left the table, and it ended up placed within 2.5 cm and released.

    Closed-loop where it has to be (changed 2026-09-23, S6): the arm is held at a waypoint
    until it has stopped before the jaw closes, before the place is re-planned and before
    release (`SETTLE_AFTER`), and the place is corrected from the cube's measured xy error
    with the cube lifted 2 cm clear of the table (`PLACE_CORRECTIONS`). The fixed-duration
    schedule it replaces placed 9/10 on the old jaw by timing luck -- every cube was 26-41 mm
    off at release, inside 25 mm only after post-release drift -- and 2/10 on the decision
    0002 jaw. Both jaws now place 10/10, 11-19 mm off at release.

    uv run --extra cpu python scripts/scripted_pick_place.py derive
    uv run --extra cpu python scripts/scripted_pick_place.py rollout --num-envs 10

``rollout --gains G20`` runs it on a plan-S8 gain code's plant (stiff gains, no gravity
compensation) with the same scales, clip and slew limit.

``tests/test_pick_place_scripted_feasibility.py`` runs the rollout, so a scale or pose
change that makes the task impossible again fails the suite instead of a training run.

The IK is "grasp point at p, jaw pointing straight down, pads closing across a cube face".
The last part matters: the jaw closes along the hand's -x, which sits at 25 deg of yaw in
the start pose, so a cube at yaw 0 is taken 25 deg off its faces unless the wrist turns.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
import pathlib
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from scene_model import joint_qadr, nominal_data, site_id, soft_limits  # noqa: E402

TASK = "Mjlab-PickPlace-Pal-Kangaroo-Grippers"
ARM_JOINTS = [f"arm_left_{i}_joint" for i in range(1, 8)]
JAW_JOINT = "gripper_left_finger_joint"
TCP_SITE = "robot/tcp_left"
HAND_BODY = "robot/arm_left_7_link"
TABLE_BODY = "table/table"
PAD_SITES = (
  "robot/gripper_left_fingertip_left_site",
  "robot/gripper_left_fingertip_right_site",
)

#: How far above the grasp pose the pre-grasp, lift, carry and retreat waypoints sit.
CLEAR_HEIGHT = 0.10
#: Pads above the table top at the grasp pose, as in the S1 reach map.
PAD_CLEARANCE = 0.01
#: The plan's margin on the IK excursion, and the grid scales are rounded up to.
MARGIN = 1.5
ROUND = 0.05

#: Phases of the scripted episode: (name, waypoint to move to, jaw closed?, control steps).
#: "Closed" may also be a fraction of the full close, for a jaw that holds lightly and so opens
#: within a step or two under a jaw slew (plan S7-2, M7's release while lowering).
#: Joint-space linear interpolation from the previous waypoint over that many 20 ms steps.
#: `place` and `retreat` are re-planned at the end of `steady`, from where the cube actually
#: sits in the hand -- see `replan_place`.
PHASES = (
  ("approach", "pregrasp", False, 60),
  ("descend", "grasp", False, 40),
  ("close", "grasp", True, 30),
  ("lift", "lift", True, 40),
  ("carry", "carry", True, 60),
  ("steady", "carry", True, 15),
  ("lower", "place", True, 40),
  ("hold", "place", True, 20),
  ("open", "place", False, 30),
  ("retreat", "retreat", False, 40),
  ("settle", "retreat", False, 20),
)
#: Every waypoint `waypoints` plans, in order.
WAYPOINT_NAMES = ("pregrasp", "grasp", "lift", "carry", "place", "retreat")
#: The place pose at the goal itself, for a cube set down at a release offset and then dragged
#: there while held (plan S7-2, M7's drag test). Planned from the corrected place pose when the
#: first phase that moves to it starts.
DRAG_WAYPOINT = "place_goal"
#: The place pose with the elbow's target walked `PRESS_ELBOW_DQ` (rad) further down: S7-3's
#: press after the release (report section 8.4), where the policy drove the elbow target past
#: its limit and the open hand pushed on the table at full effort (plan S7-4's M8 / P tests).
#: A joint-space offset, not a pose: the planner will not plan into the table.
PRESS_WAYPOINT = "place_press"
PRESS_ELBOW_DQ = -0.5
#: Phases after which the arm is held at its waypoint until it has actually stopped: before
#: the jaw closes (a still-swinging arm shoves the cube across the table as the pads meet
#: it), before `replan_place` measures the tracking bias (a snapshot of a swinging arm is
#: not a bias), and before release. Fixed durations were tuned to one jaw model: with the
#: S6 gripper (decision 0002) the same timings left the arm moving and placed 2/10, and
#: shortening `approach` broke the old model too (0/10). Measured 2026-09-23.
#: `approach` added by plan S7-2 (prep 2): at P2's W1 corner (0.37, 0.22) the -45 deg IK takes
#: another elbow branch, and descending before the arm had caught up with `approach` landed a
#: fingertip on the cube's top (W1 8/10 without the wait, 10/10 with it; S7 report section 6).
SETTLE_AFTER = ("approach", "descend", "steady", "hold")
#: "Stopped": every active env's arm joints under this speed (rad/s)...
SETTLE_SPEED = 0.05
#: ...or this many extra control steps, whichever comes first.
SETTLE_MAX_STEPS = 50
#: After `hold` has settled, and before the jaw opens: measure each cube's xy error to the
#: goal and move the place target by it, up to this many times. `replan_place` alone left
#: every cube 26-41 mm from the goal at release (old and S6 jaw alike, 2026-09-23), with
#: the 25 mm tolerance met only by post-release drift. Uses only the cube pose, which the
#: policy also observes.
PLACE_CORRECTIONS = 2
#: Skip the correction for an env already this close (m), and take this many control steps
#: over each corrective move.
PLACE_CORRECTION_TOL = 0.005
PLACE_CORRECTION_STEPS = 15
#: How far above the place pose the corrections are made (m).
PLACE_CORRECTION_LIFT = 0.02
#: Where the cube is released: this far above its resting height, so it drops onto the
#: table rather than being pressed into it.
RELEASE_HEIGHT = 0.003


@dataclasses.dataclass
class Kin:
  """The left arm's kinematics on a compiled single-world model."""

  model: mujoco.MjModel
  data: mujoco.MjData
  qadr: np.ndarray
  dofs: np.ndarray
  lo: np.ndarray
  hi: np.ndarray
  q_default: np.ndarray
  tcp: int
  hand: int
  is_arm: np.ndarray

  @classmethod
  def from_model(cls, model: mujoco.MjModel) -> Kin:
    data = nominal_data(model)
    _park_free_bodies(model, data)
    qadr = joint_qadr(model, ARM_JOINTS)
    dofs = np.asarray(
      [model.jnt_dofadr[list(model.jnt_qposadr).index(a)] for a in qadr]
    )
    lo, hi = soft_limits(model, ARM_JOINTS)
    is_arm = np.zeros(model.ngeom, dtype=bool)
    for gid in range(model.ngeom):
      bare = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").split(
        "/", 1
      )[-1]
      is_arm[gid] = bare.startswith(("arm_left_", "gripper_left_"))
    return cls(
      model=model,
      data=data,
      qadr=qadr,
      dofs=dofs,
      lo=lo,
      hi=hi,
      q_default=data.qpos[qadr].copy(),
      tcp=site_id(model, TCP_SITE),
      hand=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY),
      is_arm=is_arm,
    )

  def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    self.data.qpos[self.qadr] = q
    mujoco.mj_kinematics(self.model, self.data)
    mujoco.mj_comPos(self.model, self.data)
    return self.data.site_xpos[self.tcp].copy(), self.data.xmat[self.hand].reshape(
      3, 3
    ).copy()

  def lever_arms(self, q: np.ndarray) -> np.ndarray:
    """Distance from each arm joint's axis to the grasp point at pose ``q``: what plan S7-2's
    tool speed bound V multiplies each joint's step by (`mdp.ToolSpeedCappedJointPositionAction`).
    """
    if not hasattr(self, "_jids"):
      self._jids = [
        mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{j}")
        for j in ARM_JOINTS
      ]
    tcp, _ = self.fk(q)
    anchor = self.data.xanchor[self._jids]
    axis = self.data.xaxis[self._jids]
    return np.linalg.norm(np.cross(tcp - anchor, axis), axis=-1)

  def move_table(self, offset: np.ndarray) -> None:
    """Put the table at its compiled pose + ``offset`` (x, y, z) for the collision checks:
    the env's per-world table (DR-scene) is not in this single-world model."""
    bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, TABLE_BODY)
    if bid < 0:
      raise KeyError(f"no body {TABLE_BODY!r} in the model")
    if not hasattr(self, "_table_pos0"):
      self._table_pos0 = self.model.body_pos[bid].copy()
    self.model.body_pos[bid] = self._table_pos0 + np.asarray(offset, dtype=float)

  def pad_offset(self) -> float:
    """How far the pads reach past the grasp point along the approach axis (S1.3)."""
    tcp, rot = self.fk(self.q_default)
    pads = [self.data.site_xpos[site_id(self.model, s)] for s in PAD_SITES]
    return float(max(np.dot(p - tcp, rot[:, 2]) for p in pads))

  def in_collision(self, q: np.ndarray) -> list[tuple[str, str]]:
    self.fk(q)
    mujoco.mj_collision(self.model, self.data)
    hits = []
    for c in range(self.data.ncon):
      g1, g2 = int(self.data.contact[c].geom1), int(self.data.contact[c].geom2)
      if self.is_arm[g1] or self.is_arm[g2]:
        name = lambda g: mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g)  # noqa: E731
        hits.append((name(g1), name(g2)))
    return hits

  def ik(
    self, goal: np.ndarray, hand_x: np.ndarray, q_init: np.ndarray, iters: int = 300
  ) -> np.ndarray | None:
    """Damped least squares onto a full pose: grasp point at ``goal``, hand +z straight
    down, hand +x along ``hand_x``. Clamped to the soft limits every step.

    Six residuals on seven joints leave one free direction. It is spent pulling toward the
    start pose, so the solution is the one with the smallest excursion -- which is what the
    action scales are sized from."""
    z_des = np.array([0.0, 0.0, -1.0])
    x_des = hand_x / np.linalg.norm(hand_x)
    r_des = np.column_stack([x_des, np.cross(z_des, x_des), z_des])
    return self.ik_pose(goal, r_des, q_init, iters)

  def ik_pose(
    self, goal: np.ndarray, r_des: np.ndarray, q_init: np.ndarray, iters: int = 300
  ) -> np.ndarray | None:
    """`ik` onto an arbitrary hand orientation ``r_des`` (3x3, the hand's rotation matrix in
    the world): plan S7-5 §8's vertical retreat keeps whatever tilt the hand has."""
    jacp = np.zeros((3, self.model.nv))
    jacr = np.zeros((3, self.model.nv))
    q = q_init.copy()
    for _ in range(iters):
      pos, rot = self.fk(q)
      err_p = goal - pos
      err_r = 0.5 * sum(np.cross(rot[:, i], r_des[:, i]) for i in range(3))
      if np.linalg.norm(err_p) < 1e-3 and np.linalg.norm(err_r) < 1e-2:
        return q
      mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.tcp)
      jac = np.vstack([jacp[:, self.dofs], jacr[:, self.dofs]])
      err = np.concatenate([err_p, err_r])
      pinv = jac.T @ np.linalg.inv(jac @ jac.T + 1e-3 * np.eye(6))
      null = np.eye(len(q)) - pinv @ jac
      step = pinv @ err + null @ (0.2 * (self.q_default - q))
      q = np.clip(q + np.clip(step, -0.2, 0.2), self.lo, self.hi)
    return None


def _park_free_bodies(model: mujoco.MjModel, data: mujoco.MjData) -> None:
  """Move the cube out of the way; the waypoints are checked against robot and table only."""
  for jid in range(model.njnt):
    if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
      continue
    adr = model.jnt_qposadr[jid]
    data.qpos[adr : adr + 3] = (0.0, 0.0, -10.0)


def hand_x_for(kin: Kin, cube_yaw: float) -> np.ndarray:
  """The hand +x direction that closes the pads across a pair of cube faces.

  Any of the four face directions ``cube_yaw + k*90 deg`` will do for a symmetric jaw; take
  the one closest to the start pose's, so the wrist turns as little as possible.
  """
  _, rot = kin.fk(kin.q_default)
  yaw0 = math.atan2(rot[1, 0], rot[0, 0])
  cands = [cube_yaw + k * math.pi / 2 for k in range(4)]
  best = min(cands, key=lambda a: abs(math.remainder(a - yaw0, 2 * math.pi)))
  return np.array([math.cos(best), math.sin(best), 0.0])


def ball_pad_clearance(offset: float | None = None) -> float:
  """Height of the pad sites' midpoint above the table at a ball's grasp (plan S8): the ball's
  radius (its centre) plus ``offset`` -- `pick_place.rewards.EQ_OFFSET`, the height above the
  centre the gripper bench found holds best, if None."""
  from pal_mjlab.tasks.manipulation import scene
  from pal_mjlab.tasks.manipulation.pick_place import rewards

  return scene.BALL_RADIUS + (rewards.EQ_OFFSET if offset is None else offset)


def place_hand_x(hand_x: np.ndarray, cube_yaw: float, base_yaw: float) -> np.ndarray:
  """Plan S9: the hand +x for the carry and the place, turned from the grasp's ``hand_x`` by the
  base's yaw minus the cube's, folded into (-45, 45] deg, so the held cube arrives square to the
  base (the lock's yaw bound) with the least wrist turn."""
  quarter = math.pi / 2
  turn = math.remainder(base_yaw - cube_yaw, quarter)
  c, s = math.cos(turn), math.sin(turn)
  return np.array([c * hand_x[0] - s * hand_x[1], s * hand_x[0] + c * hand_x[1], 0.0])


def waypoints(
  kin: Kin,
  cube_xy: np.ndarray,
  cube_yaw: float,
  target_xy: np.ndarray,
  table_z: float | None = None,
  pad_clearance: float = PAD_CLEARANCE,
  carry_hand_x: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
  """Joint configurations for every phase, each IK seeded from the previous one so the
  sequence stays on one solution branch. Raises if any waypoint has no solution or puts
  the arm in contact. ``table_z`` is the table's top face (m); `scene.TABLE_TOP_Z` if None.
  ``pad_clearance`` is the height of the pad sites above the table at the grasp pose: 1 cm for
  the 5 cm cube; for plan S8's ball, `ball_pad_clearance` (the pad centre at the ball's equator
  plus the bench's best offset). ``carry_hand_x`` (plan S9, `place_hand_x`) is the hand +x from the
  carry on; the grasp's if None."""
  from pal_mjlab.tasks.manipulation import scene

  if table_z is None:
    table_z = scene.TABLE_TOP_Z
  z_grasp = table_z + kin.pad_offset() + pad_clearance
  z_high = z_grasp + CLEAR_HEIGHT
  hand_x = hand_x_for(kin, cube_yaw)
  goals = {
    "pregrasp": (cube_xy, z_high),
    "grasp": (cube_xy, z_grasp),
    "lift": (cube_xy, z_high),
    "carry": (target_xy, z_high),
    "place": (target_xy, z_grasp),
    "retreat": (target_xy, z_high),
  }
  out: dict[str, np.ndarray] = {}
  q = kin.q_default
  for name, (xy, z) in goals.items():
    goal = np.array([xy[0], xy[1], z])
    hx = (
      carry_hand_x
      if carry_hand_x is not None and name in ("carry", "place", "retreat")
      else hand_x
    )
    sol = _solve(kin, name, goal, hx, q)
    # Only the free-space waypoints must be contact-free; at grasp/place the pads are
    # 1 cm off the table by construction and anything they touch is the (parked) cube.
    hits = kin.in_collision(sol)
    if hits:
      raise RuntimeError(f"{name} at {goal.round(3)} collides: {hits[:3]}")
    out[name] = sol
    q = sol
  return out


#: IK restarts per waypoint: seeds around the previous waypoint first (their spread widening
#: in rad), then seeds uniform over the soft limits. Interpolated configurations checked per
#: joint-space segment.
IK_LOCAL_ATTEMPTS = 24
IK_SEED_STD = (0.3, 0.6, 1.0)
IK_GLOBAL_ATTEMPTS = 96
SEGMENT_CHECKS = 12


def segment_free(kin: Kin, a: np.ndarray, b: np.ndarray) -> bool:
  """No arm contact anywhere on the straight joint-space path from ``a`` to ``b`` -- the path
  the rollout interpolates along."""
  return not any(
    kin.in_collision(a + (b - a) * (k / SEGMENT_CHECKS))
    for k in range(1, SEGMENT_CHECKS + 1)
  )


def _solve(kin: Kin, name: str, goal: np.ndarray, hand_x: np.ndarray, q: np.ndarray):
  """A contact-free IK solution reachable from ``q`` along a contact-free straight path.

  Seeded from ``q`` first -- to stay on its solution branch -- then restarts around it, then
  (only if none of those works) seeds spread over the whole joint range, keeping the
  candidate closest to ``q``. Changed 2026-09-23 (S5.1): with the item-8 mesh colliders, the
  smallest-excursion solution for `carry` over the target folds arm 3-4 64 mm into the torso
  hull, which S1.4's boxes had let through; the elbow-out branch exists (a global search
  finds it) but the pull toward the start pose never reaches it. The segment check matters
  as much as the endpoint: an elbow-in `lift` followed by an elbow-out `carry` is two
  contact-free poses with the chest between them.
  """
  rng = np.random.default_rng(0)
  last_hits: list[tuple[str, str]] = []

  def accept(sol):
    nonlocal last_hits
    if sol is None:
      return False
    last_hits = kin.in_collision(sol)
    return not last_hits and segment_free(kin, q, sol)

  for attempt in range(IK_LOCAL_ATTEMPTS):
    std = IK_SEED_STD[attempt * len(IK_SEED_STD) // IK_LOCAL_ATTEMPTS]
    seed = (
      q if attempt == 0 else np.clip(q + rng.normal(0, std, q.shape), kin.lo, kin.hi)
    )
    sol = kin.ik(goal, hand_x, seed)
    if accept(sol):
      return sol
  best = None
  for _ in range(IK_GLOBAL_ATTEMPTS):
    sol = kin.ik(goal, hand_x, rng.uniform(kin.lo, kin.hi))
    if accept(sol):
      d = float(np.abs(sol - q).max())
      if best is None or d < best[0]:
        best = (d, sol)
  if best is not None:
    return best[1]
  if last_hits:
    raise RuntimeError(f"{name} at {goal.round(3)} collides: {last_hits[:3]}")
  raise RuntimeError(f"no contact-free IK for {name} at {goal.round(3)}")


def replan_place(
  kin: Kin,
  plan: dict[str, np.ndarray],
  hand_x: np.ndarray,
  tcp_actual: np.ndarray,
  cube_actual: np.ndarray,
  goal: np.ndarray,
) -> None:
  """Re-plan `place` and `retreat` from the cube's measured position in the hand.

  The open-loop plan assumes the cube ends up centred under the grasp point. It does not:
  it slides while the jaw closes, and the PD loop leaves the arm a centimetre or two off its
  target. Both are measured here, with the arm holding still at the carry waypoint, and the
  place target is shifted so that the *cube* lands on the goal: the grasp point goes to
  ``goal + (tcp - cube)``, commanded as that minus the current tracking bias. The policy
  observes the cube's pose too, so this uses nothing it does not have.
  """
  tcp_planned, _ = kin.fk(plan["carry"])
  bias = tcp_actual - tcp_planned
  want = goal + (tcp_actual - cube_actual) + np.array([0.0, 0.0, RELEASE_HEIGHT])
  plan["place"] = _solve(kin, "place", want - bias, hand_x, plan["carry"])
  plan["retreat"] = _solve(
    kin,
    "retreat",
    want - bias + np.array([0.0, 0.0, CLEAR_HEIGHT]),
    hand_x,
    plan["place"],
  )


def _surface_top(env, name: str) -> np.ndarray:
  """Each env's world z of the named surface entity's top face, from physics. (B,)"""
  from mjlab.managers.scene_entity_config import SceneEntityCfg
  from pal_mjlab.tasks.manipulation.mdp.events import surface_top_z

  return surface_top_z(env, SceneEntityCfg(name)).cpu().numpy()


def _surface_xy(env, name: str) -> np.ndarray:
  """Each env's world xy of the named surface entity's top geom centre, from physics. (B, 2)"""
  from mjlab.managers.scene_entity_config import SceneEntityCfg
  from pal_mjlab.tasks.manipulation.mdp.events import _surface_centre_xy

  return _surface_centre_xy(env, SceneEntityCfg(name)).cpu().numpy()


def spawn_corners(
  spawn_xy: tuple[float, float], jitter_xy: float, jitter_yaw: float
) -> list[tuple[np.ndarray, float]]:
  """The eight corners of the spawn box, and its centre."""
  out = [(np.asarray(spawn_xy, dtype=float), 0.0)]
  for dx in (-jitter_xy, jitter_xy):
    for dy in (-jitter_xy, jitter_xy):
      for yaw in (-jitter_yaw, jitter_yaw):
        out.append((np.asarray(spawn_xy) + (dx, dy), yaw))
  return out


def box_corners(box) -> list[tuple[np.ndarray, float]]:
  """A `pick_place_env_cfg.SpawnBox`'s centre at yaw 0, then its four xy corners at -/+ its
  largest yaw: `spawn_corners` for a box that need not be square."""
  centre = np.array([sum(box.x) / 2, sum(box.y) / 2])
  out = [(centre, 0.0)]
  for x in box.x:
    for y in box.y:
      for yaw in (-box.yaw, box.yaw):
        out.append((np.array([x, y]), yaw))
  return out


def base_box_corners(baseline, n: int) -> list[tuple[np.ndarray, float]]:
  """Plan S9: ``n`` base-cube poses cycling over the baseline's target box's centre and corners
  at -/+ the base's yaw range (`pick_place_env_cfg.STACK_BASE_YAW`)."""
  from pal_mjlab.tasks.manipulation.pick_place import pick_place_env_cfg as pp

  (tx, ty), h = baseline.target_xy, baseline.target_half_xy
  box = pp.SpawnBox(x=(tx - h, tx + h), y=(ty - h, ty + h), yaw=pp.STACK_BASE_YAW)
  corners = box_corners(box)
  return [corners[i % len(corners)] for i in range(n)]


#: DR-scene's table shift, for the corners check: x, y +/-2 cm and z +/-1 cm
#: (`tabletop_env_cfg._DR_SCENE_TABLE_XY` / `_DR_SCENE_TABLE_HEIGHT`).
TABLE_SHIFT_XY = 0.02
TABLE_SHIFT_Z = 0.01


def corner_cases(
  box, target_xy: tuple[float, float], target_half_xy: float
) -> list[
  tuple[tuple[np.ndarray, float], tuple[float, float], tuple[float, float, float]]
]:
  """Plan S7-2's corners check: ten (spawn, target, table shift) triples for one spawn box.

  The spawns are `box_corners` (the centre at yaw 0, the four xy corners at -/+ the largest
  yaw) and the centre at + the largest yaw. The targets and table shifts: the target box's
  centre; its four corners each with the table shifted +/-2 cm the same way, +/-1 cm up or
  down -- the targets farthest out relative to the robot; the four corners without a shift;
  and the centre with the table shifted towards the torso (-x, -y) and down.
  """
  spawns = box_corners(box) + [(np.array([sum(box.x) / 2, sum(box.y) / 2]), box.yaw)]
  tx, ty, h = target_xy[0], target_xy[1], target_half_xy
  signs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
  xy, dz = TABLE_SHIFT_XY, TABLE_SHIFT_Z
  targets = (
    [(tx, ty)] + [(tx + sx * h, ty + sy * h) for sx, sy in signs] * 2 + [(tx, ty)]
  )
  shifts = (
    [(0.0, 0.0, 0.0)]
    + [(sx * xy, sy * xy, dz if k % 2 else -dz) for k, (sx, sy) in enumerate(signs)]
    + [(0.0, 0.0, 0.0)] * 4
    + [(-xy, -xy, -dz)]
  )
  return list(zip(spawns, targets, shifts, strict=True))


# --------------------------------------------------------------------------------------
# derive
# --------------------------------------------------------------------------------------


def derive(args: argparse.Namespace) -> dict:
  from pal_mjlab.tasks.manipulation.pick_place import pick_place_env_cfg as pp
  from scene_model import compile_scene

  kin = Kin.from_model(compile_scene(TASK))
  corners = spawn_corners(pp.SPAWN_XY, pp.SPAWN_JITTER_XY, pp.SPAWN_JITTER_YAW)
  target = np.asarray(pp.TARGET_XY)

  dev = np.zeros(len(ARM_JOINTS))
  worst: list[str] = [""] * len(ARM_JOINTS)
  for xy, yaw in corners:
    wps = waypoints(kin, xy, yaw, target)
    for name, q in wps.items():
      d = np.abs(q - kin.q_default)
      for j in np.nonzero(d > dev)[0]:
        worst[j] = f"{name} @ ({xy[0]:.2f},{xy[1]:.2f},{math.degrees(yaw):+.0f}deg)"
      dev = np.maximum(dev, d)

  current = {j: s for j, s in zip(ARM_JOINTS, _current_arm_scales(), strict=True)}
  lo = [kin.model.jnt_range[_jid(kin.model, j)][0] for j in ARM_JOINTS]
  hi = [kin.model.jnt_range[_jid(kin.model, j)][1] for j in ARM_JOINTS]
  rows = []
  print(
    f"{'joint':18s} {'default':>8s} {'max|dq|':>8s} {'x1.5':>7s} {'current':>8s} "
    f"{'cap':>6s} {'scale':>6s}  worst waypoint"
  )
  for j, name in enumerate(ARM_JOINTS):
    need = MARGIN * dev[j]
    cap = 0.5 * (hi[j] - lo[j])
    scale = min(max(current[name], math.ceil(need / ROUND - 1e-9) * ROUND), cap)
    rows.append(
      dict(
        joint=name,
        default=float(kin.q_default[j]),
        max_dev=float(dev[j]),
        need=float(need),
        current=current[name],
        cap=float(cap),
        scale=round(scale, 4),
        worst=worst[j],
      )
    )
    print(
      f"{name:18s} {kin.q_default[j]:8.3f} {dev[j]:8.3f} {need:7.3f} "
      f"{current[name]:8.4f} {cap:6.2f} {scale:6.2f}  {worst[j]}"
    )
  result = dict(
    task=TASK,
    spawn_xy=list(pp.SPAWN_XY),
    target_xy=list(pp.TARGET_XY),
    corners=[[float(xy[0]), float(xy[1]), float(yaw)] for xy, yaw in corners],
    rule=f"max(current, ceil({MARGIN} * max|dq| / {ROUND}) * {ROUND}), capped at half range",
    joints=rows,
  )
  if args.json:
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2))
    print(f"wrote {args.json}")
  return result


def _jid(model: mujoco.MjModel, name: str) -> int:
  return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{name}")


def _current_arm_scales() -> list[float]:
  """The floor of the rule: the SysID training scale 0.25 * effort / K of each arm joint
  (S5.1 item 3: "use his 0.25 where it suffices") -- 0.2503 rad on arm 1-2, 0.2388 on 3-7.
  Until S5.1 the floor was the scales C1/C2 ran with."""
  from pal_mjlab.robots.pal_kangaroo import kangaroo_constants as kc

  def sysid(block: dict) -> float:
    return 0.25 * block["effort_limit"] / block["stiffness"]

  return [sysid(kc.S_PLUS)] * 2 + [sysid(kc.S_MINUS)] * 3 + [sysid(kc.XS)] * 2


# --------------------------------------------------------------------------------------
# rollout
# --------------------------------------------------------------------------------------


def _eval_module():
  spec = importlib.util.spec_from_file_location(
    "eval_pick_place", pathlib.Path(__file__).parent / "eval_pick_place.py"
  )
  assert spec and spec.loader
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def rollout(
  num_envs: int = 10,
  device: str = "cpu",
  seed: int = 0,
  reward: str = "R-A",
  corners_first: bool = True,
  verbose: bool = True,
  hook=None,
  action_slew_rate_scale: float = 1.0,
  spawns: list[tuple[tuple[float, float], float]] | None = None,
  release_offsets: list[tuple[float, float]] | None = None,
  phases: tuple[tuple[str, str, bool, int], ...] | None = None,
  robot=None,
  baseline=None,
  spawn_box=None,
  dr: str = "DR0",
  tool_speed_cap=None,
  targets: list[tuple[float, float]] | None = None,
  table_offsets: list[tuple[float, float, float]] | None = None,
  episode_length_s: float | None = None,
  effort=None,
  latency=None,
  target_area=None,
  stack_speed: bool = False,
  action_mode: str = "abs",
  shape=None,
  ball_offset: float | None = None,
  goal: str = "table",
  gate: str = "none",
  base_spawns: list[tuple[tuple[float, float], float]] | None = None,
  pregrasp=None,
) -> list[dict]:
  """One scripted episode per env; returns eval_pick_place's per-episode flags.

  Envs ``0..8`` get the spawn box's centre and eight corners when ``corners_first`` is
  set; the rest keep the reset event's own random jitter.

  Plan S7-2: ``baseline`` (`pick_place_env_cfg.BASELINES`, S71's if None), ``spawn_box``
  (`SPAWN_BOXES`, S71's if None; its corners are the forced ones), ``dr`` and
  ``tool_speed_cap`` build the env as a run would. Every env plans to its own goal and its
  own table height, read from the env, so a random target box and a moving table are
  followed. ``targets`` puts env ``i``'s goal at ``targets[i]`` (x, y in the env frame, for
  the table at its nominal place; it moves with the table) -- a `SurfaceOffsetCommand` goal
  only, i.e. a baseline with a target box or DR-scene. ``table_offsets`` moves env ``i``'s
  table by ``table_offsets[i]`` (x, y, z) -- DR-scene only, whose table is per world -- and a
  forced spawn moves with it, as the reset event would place it. ``episode_length_s``
  overrides the env's: the script is slow and closed-loop on purpose (settle waits, place
  corrections) and needs ~9 s where a trained policy needs ~1, so under baseline B's 10 s
  episode it would time out mid-place. What it tests is the action space, not the clock.

  ``hook(phase, env)``, if given, is called after every control step with the name of the
  phase that step belongs to (settle waits and place corrections count to their phase). It
  sees the env *after* the step, so a test can read reward terms and sensors at real,
  physically reached states -- a held cube cannot be teleported into the jaw.

  ``action_slew_rate_scale`` builds the env with a scaled arm slew limit (plan S7: 0.5 is
  speed bound A). The script reads the limit back from the action term, so its
  interpolation stretches to the lower limit by itself; ``over_slew_env_steps`` checks it.

  ``spawns`` puts env ``i`` at ``spawns[i]`` = ((x, y), yaw) in the env frame instead of
  the spawn box's corners (plan S7 P2's action-space reach map). A spawn whose waypoints
  have no contact-free IK is reported in ``plan_error`` and its arm stays home, rather than
  failing the whole batch; so are closed-loop re-plans that fail (``skipped_corrections``).
  ``clipped_steps`` counts the env's own clipped control steps.

  ``release_offsets`` moves env ``i``'s release point to the goal + ``release_offsets[i]``
  (x, y in m): the carry, place and retreat waypoints and the closed-loop place corrections
  all aim there, so the cube is set down that far from the target (plan S7, M6's "release
  6 cm short"). ``phases`` replaces `PHASES`; names it shares with `PHASES` keep their special
  handling (`lower` re-plans, `hold` corrects, `SETTLE_AFTER` waits).

  ``robot`` replaces `KANGAROO_GRIPPERS_LEFT_ARM`, e.g. with other action scales (plan S7 P2's
  scales for a wider spawn). The script reads scales and offsets back from the action term,
  so it drives whatever the robot cfg defines.

  ``effort`` is plan S7-4's effort bound (`pick_place_env_cfg.EFFORT_BOUNDS`; none if None);
  T's plant half comes with ``robot``. Each row's ``band_hit_steps`` / ``range_hit_steps``
  count the env's steps on which T's band / C's clip moved the arm's target, by phase.

  Plan S7-8: ``stack_speed`` (the stack task's per-joint speeds), ``action_mode`` (``delta``:
  the script's absolute waypoints are sent as the step towards them, |a| <= 1 at the stack
  speed), ``latency`` (`tabletop_env_cfg.LATENCY_MODELS`, S7-8 runs it at ``JS``) and
  ``target_area`` (`pick_place_env_cfg.TARGET_AREAS`: the goal drawn from the spawn area). The
  rows carry each episode's peak TCP and held-cube speed (``peak_tcp_speed_held``, ...).

  Plan S8: ``shape`` (`pick_place_env_cfg.OBJECT_SHAPES`; the cube if None) builds the env with a
  ball. A ball has no yaw to grasp across -- the wrist keeps its start yaw -- and is taken with the
  pad sites' midpoint ``ball_offset`` above its centre (`ball_pad_clearance`).

  Plan S9: ``goal`` / ``gate`` (`pick_place_env_cfg.GOALS` / `GATES`) build the stack task and its
  jaw lock. On a stack goal the wrist is turned for the carry and the place so the held cube
  arrives square to the base (`place_hand_x`), and ``base_spawns`` puts env ``i``'s base cube at
  ``base_spawns[i]`` = ((x, y), yaw) in the env frame (moved with the table, like a forced spawn).
  Rows carry ``gate_locked_steps`` (steps the lock refused an opening) and ``gate_armed_steps``.

  Plan S7-11: ``pregrasp`` (`pregrasp.PREGRASP_CODES` entry, None for none) builds the env with a
  ``PREGRASP`` code -- G's gate must let the script's grasp through.
  """
  import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from pal_mjlab.tasks.manipulation.pick_place import pick_place_env_cfg as pp
  from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS_LEFT_ARM

  robot_cfg = KANGAROO_GRIPPERS_LEFT_ARM if robot is None else robot
  baseline = pp.BASELINES["S71"] if baseline is None else baseline
  spawn_box = pp.SPAWN_BOXES["S71"] if spawn_box is None else spawn_box

  ev = _eval_module()
  cfg = pp.make_pick_place_env_cfg(
    robot_cfg,
    gripper_name="left",
    reward=reward,
    action_slew_rate_scale=action_slew_rate_scale,
    spawn=spawn_box,
    dr=dr,
    baseline=baseline,
    tool_speed_cap=tool_speed_cap,
    **({} if effort is None else {"effort": effort}),
    **({} if latency is None else {"latency": latency}),
    target_area=target_area,
    stack_speed=stack_speed,
    action_mode=action_mode,
    command_obs=action_mode == "delta",
    **({} if shape is None else {"shape": shape}),
    goal=goal,
    gate=gate,
    pregrasp=pregrasp,
  )
  cfg.scene.num_envs = num_envs
  cfg.seed = seed
  if episode_length_s is not None:
    cfg.episode_length_s = episode_length_s
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  grasp_sensor = robot_cfg.gripper("left").contact_sensor_name
  table = env.scene[pp.TABLE]
  # The table body itself, not the entity's first body: that is the mocap anchor
  # (`table/mocap_base`) the fixed table hangs from. DR-scene's `dr.body_pos` moves the
  # `table` body's local offset from it, and so does this -- overwriting the startup draw,
  # where writing the anchor's `body_pos` moved the table only at the next reset and on top
  # of the draw (found by the corners check's positive control, 2026-09-26).
  table_body = table.indexing.body_ids[table.body_names.index(pp.TABLE)]
  if table_offsets is not None:
    if len(table_offsets) != num_envs:
      raise ValueError(f"{len(table_offsets)} table offsets for {num_envs} envs")
    body_pos = env.sim.model.body_pos
    if body_pos.shape[0] != num_envs:
      raise ValueError(
        "table_offsets needs a per-world table: build the env at DR-scene"
      )
    local0 = torch.tensor(
      env.sim.mj_model.body_pos[int(table_body)], dtype=torch.float32, device=env.device
    )
    for i, off in enumerate(table_offsets):
      body_pos[i, table_body] = local0 + torch.tensor(off, device=env.device)
    # The reset event reads the table's geom pose out of `data`, which only a forward pass
    # brings up to date with `model`.
    env.sim.forward()

  term = env.action_manager.get_term("joint_pos")
  names = [n.split("/", 1)[-1] for n in term.target_names]
  arm_idx = [names.index(j) for j in ARM_JOINTS]
  jaw_idx = names.index(JAW_JOINT)
  scale = (
    term.scale
    if isinstance(term.scale, torch.Tensor)
    else torch.full((num_envs, len(names)), float(term.scale), device=env.device)
  )
  offset = term.offset
  jaw_closed = robot_cfg.gripper("left").closed_pos
  # The arm's slew limit, rad per control step (S5.1 item 4; inf where there is none). Every
  # interpolation below is stretched so no step asks for more than this: the action term
  # would enforce it anyway, but a script that outruns it drifts from its own schedule and
  # its settle checks start measuring lag, not arrival.
  slew = (
    term._rate[0, arm_idx].cpu().numpy()
    if hasattr(term, "_rate")
    else np.full(len(arm_idx), np.inf)
  )

  # Plan S7-2's tool speed bound V, when the action term carries one: a move is stretched so
  # its bound on the tool's travel per step, sum_i |dq_i| r_i, stays under the cap -- the slow
  # one whenever the jaw is closed (V slows a held cube only near the table or the goal, so
  # this is the conservative side), with lever arms at the move's midpoint and 20 % margin.
  # Same reason as the slew: the term enforces it anyway, and a script that outruns it measures
  # its own lag at every settle check and re-plans the place from it.
  tool_cap = getattr(term.cfg, "speed_limit", None)
  tool_cap_slow = getattr(term.cfg, "slow_speed_limit", None)

  def n_steps(a: np.ndarray, b: np.ndarray, base: int, closed: bool = False) -> int:
    need = int(np.ceil((np.abs(b - a) / slew).max() - 1e-9))
    if tool_cap is not None:
      v = tool_cap_slow if closed else tool_cap
      travel = max(
        float(np.abs(b[e] - a[e]) @ kin.lever_arms(0.5 * (a[e] + b[e])))
        for e in range(len(a))
      )
      need = max(need, int(np.ceil(1.2 * travel / (v * env.step_dt) - 1e-9)))
    return max(base, need)

  env.reset()
  obj = env.scene[pp.OBJECT]
  origins = env.scene.env_origins
  if spawns is not None and len(spawns) != num_envs:
    raise ValueError(f"{len(spawns)} spawns for {num_envs} envs")
  if release_offsets is not None and len(release_offsets) != num_envs:
    raise ValueError(f"{len(release_offsets)} release offsets for {num_envs} envs")
  # Each env's release point relative to the goal, (x, y, 0) in m.
  release_off = np.zeros((num_envs, 3))
  if release_offsets is not None:
    release_off[:, :2] = np.asarray(release_offsets, dtype=float)
  # Each env's table offset from its nominal pose, as the reset event and goal see it.
  table_xy0 = np.asarray(pp.scene.table_pos(baseline.table_top_z)[:2])
  shift = np.zeros((num_envs, 3))
  shift[:, :2] = _surface_xy(env, pp.TABLE) - origins[:, :2].cpu().numpy() - table_xy0
  shift[:, 2] = _surface_top(env, pp.TABLE) - baseline.table_top_z
  if targets is not None:
    if len(targets) != num_envs:
      raise ValueError(f"{len(targets)} targets for {num_envs} envs")
    command = env.command_manager.get_term("goal_pose")
    if not hasattr(command, "set_offsets"):
      raise ValueError(
        "targets needs a SurfaceOffsetCommand goal (target box or DR-scene)"
      )
    command.set_offsets(
      torch.arange(num_envs, device=env.device),
      torch.tensor(np.asarray(targets) - table_xy0, dtype=torch.float32),
    )
  forced = (
    spawns
    if spawns is not None
    else box_corners(spawn_box)[:num_envs]
    if corners_first
    else None
  )
  if base_spawns is not None and len(base_spawns) != num_envs:
    raise ValueError(f"{len(base_spawns)} base spawns for {num_envs} envs")
  stack = goal == "stack"
  base = env.scene[pp.BASE_OBJECT] if stack else None
  for entity, poses in ((obj, forced), (base, base_spawns)):
    if poses is None:
      continue
    if entity is None:
      raise ValueError("base_spawns needs goal='stack'")
    pose = entity.data.root_link_pose_w.clone()
    for i, (xy, yaw) in enumerate(poses):
      pose[i, 0] = origins[i, 0] + float(xy[0]) + float(shift[i, 0])
      pose[i, 1] = origins[i, 1] + float(xy[1]) + float(shift[i, 1])
      pose[i, 3:7] = torch.tensor(
        [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], device=env.device
      )
    entity.write_root_link_pose_to_sim(pose)
    entity.write_root_link_velocity_to_sim(
      torch.zeros_like(entity.data.root_link_vel_w)
    )
  if forced is not None or base_spawns is not None:
    # A teleport is not visible to the entity's data views until the sim has stepped; one
    # idle step (action 0 = hold the start pose) also lets the cube settle on the table.
    with torch.inference_mode():
      env.step(torch.zeros(env.action_space.shape, device=env.device))

  # Plan every env's waypoints from where its cube actually is, to where its goal is, over
  # its own table.
  kin = Kin.from_model(env.sim.mj_model)
  if not np.allclose(kin.q_default, offset[0, arm_idx].cpu().numpy(), atol=1e-4):
    raise RuntimeError("the IK model's default pose is not the action term's offset")
  pos = (obj.data.root_link_pos_w - origins).cpu().numpy()
  quat = obj.data.root_link_quat_w.cpu().numpy()
  goals = (env.command_manager.get_term("goal_pose").command - origins).cpu().numpy()
  table_z = _surface_top(env, pp.TABLE)
  plans = []
  hand_xs = []
  plan_error: dict[int, str] = {}
  from pal_mjlab.tasks.manipulation.mdp import ball_mask

  is_ball = ball_mask(env).cpu().numpy()
  base_quat = base.data.root_link_quat_w.cpu().numpy() if stack else None
  hand_yaw0 = math.atan2(kin.fk(kin.q_default)[1][1, 0], kin.fk(kin.q_default)[1][0, 0])
  for i in range(num_envs):
    w, x, y, z = quat[i]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    if is_ball[i]:
      yaw = hand_yaw0  # a sphere is grasped across any pair of points: no wrist turn
    carry_x = None
    if stack:
      bw, bx, by, bz = base_quat[i]
      base_yaw = math.atan2(2 * (bw * bz + bx * by), 1 - 2 * (by * by + bz * bz))
      carry_x = place_hand_x(hand_x_for(kin, yaw), yaw, base_yaw)
    kin.move_table(shift[i])
    try:
      plans.append(
        waypoints(
          kin,
          pos[i, :2],
          yaw,
          goals[i, :2] + release_off[i, :2],
          table_z=float(table_z[i]),
          pad_clearance=ball_pad_clearance(ball_offset)
          if is_ball[i]
          else PAD_CLEARANCE,
          carry_hand_x=carry_x,
        )
      )
    except RuntimeError as e:
      # Unreachable for the planner: the arm stays home and the env is scored as it ends.
      plan_error[i] = str(e)
      plans.append({name: kin.q_default.copy() for name in WAYPOINT_NAMES})
    # Every later use of hand_xs re-plans the place, so it is the carry's hand +x.
    hand_xs.append(hand_x_for(kin, yaw) if carry_x is None else carry_x)
    if verbose:
      print(
        f"env {i}: cube at ({pos[i, 0]:.3f}, {pos[i, 1]:.3f}), yaw {math.degrees(yaw):+.1f} deg"
      )

  tally = ev.Tally(num_envs, env.device)
  ended = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
  why = [""] * num_envs
  order: list[int] = []  # env id of each row of tally.done, which is in finish order
  robot = env.scene["robot"]
  tcp_idx = robot.site_names.index(robot_cfg.gripper("left").tcp_site)
  goal_w = env.command_manager.get_term("goal_pose").command
  diag: dict[str, np.ndarray] = {}
  clipped = 0
  clipped_env = np.zeros(num_envs, dtype=int)
  prev = np.stack([kin.q_default] * num_envs)
  arm_q = [robot.joint_names.index(j) for j in ARM_JOINTS]
  settle_steps: dict[str, int] = {}
  place_corrections = 0
  # Corrections the planner could not make contact-free, as (env, reason). Skipped rather than
  # forced: the env keeps its current place pose and is scored on where the cube lands.
  # First hit at slew 0.5 (plan S7 FA, 2026-09-24): a 7 mm shift toward the torso left only
  # elbow-in IK solutions for one corner, and the whole rollout used to die on it.
  skipped_corrections: list[tuple[int, str]] = []
  last_q = prev.copy()
  over_slew = (
    0  # env-steps whose commanded arm move exceeded the slew limit; should stay 0
  )
  stretched: dict[str, int] = {}
  bound_hits: dict[str, dict[str, np.ndarray]] = {"band": {}, "range": {}}
  gate_counts = {"locked": np.zeros(num_envs, int), "armed": np.zeros(num_envs, int)}

  def move_and_settle(a: np.ndarray, b: np.ndarray, closed: bool) -> None:
    """Interpolate the arm from pose ``a`` to ``b``, then hold ``b`` until it settles."""
    n = n_steps(a, b, PLACE_CORRECTION_STEPS, closed)
    for s in range(1, n + 1):
      step_to(a + (b - a) * (s / n), closed)
    for _ in range(SETTLE_MAX_STEPS):
      if bool(ended.all()):
        return
      speed = robot.data.joint_vel[:, arm_q].abs().amax(dim=-1)
      if bool((speed[~ended] < SETTLE_SPEED).all()):
        return
      step_to(b, closed)

  def step_to(q: np.ndarray, closed: bool | float) -> None:
    """One control step toward arm pose ``q`` with the jaw open or closed."""
    nonlocal clipped, clipped_env, ended, order, last_q, over_slew
    over_slew += int((np.abs(q - last_q) > slew + 1e-6).any(axis=-1).sum())
    last_q = np.array(q, copy=True)
    target_q = offset.clone()
    target_q[:, arm_idx] = torch.as_tensor(q, dtype=torch.float32, device=env.device)
    target_q[:, jaw_idx] = jaw_closed * float(closed)
    if hasattr(
      term, "_box_lo"
    ):  # plan S7-8's delta action: the step towards the waypoint
      raw = (target_q - term.commanded_target) / term._rate
    else:
      raw = (target_q - offset) / scale
    over = (raw.abs() > 1.0 + 1e-6).any(dim=-1)
    clipped += int(over.sum())
    clipped_env += over.cpu().numpy().astype(int)
    flags = ev.probe(env, grasp_sensor)
    tally.observe(**flags)
    tally.measure_speed(flags["grasped"], *ev.positions(env), env.step_dt, goal_w)
    _, _, terminated, truncated, _ = env.step(raw.clamp(-1.0, 1.0))
    for key in ("locked", "armed"):
      flag = getattr(term, key, None)
      if flag is not None:
        gate_counts[key] += (flag & ~ended).cpu().numpy().astype(int)
    for key, attr in (("band", "band_hit"), ("range", "range_hit")):
      hits = getattr(term, attr, None)
      if hits is not None:
        per_phase = bound_hits[key].setdefault(phase_now[0], np.zeros(num_envs, int))
        per_phase += (hits & ~ended).cpu().numpy().astype(int)
    tally.terminal(ev.success_flags(env) & ~ended)
    if hook is not None:
      hook(phase_now[0], env)
    done = (terminated | truncated) & ~ended
    for i in torch.nonzero(done).flatten().tolist():
      why[i] = ",".join(
        n
        for n in env.termination_manager.active_terms
        if bool(env.termination_manager.get_term(n)[i])
      )
    if done.any():
      order += torch.nonzero(done).flatten().tolist()
      tally.finish(torch.nonzero(done).flatten())
      ended |= done

  phase_list = phases if phases is not None else PHASES
  phase_now = ["start"]
  with torch.inference_mode():
    for phase, wp, closed, steps in phase_list:
      phase_now[0] = phase
      if bool(ended.all()):
        break
      if phase == "lower":
        tcp_a = (robot.data.site_pos_w[:, tcp_idx] - origins).cpu().numpy()
        cube_a = (obj.data.root_link_pos_w - origins).cpu().numpy()
        goal = (goal_w - origins).cpu().numpy()
        for i in torch.nonzero(~ended).flatten().tolist():
          if i in plan_error:
            continue
          kin.move_table(shift[i])
          try:
            replan_place(
              kin, plans[i], hand_xs[i], tcp_a[i], cube_a[i], goal[i] + release_off[i]
            )
          except RuntimeError as e:  # keep the open-loop place; scored as it lands
            skipped_corrections.append((i, f"replan: {e}"))
      if wp == DRAG_WAYPOINT and DRAG_WAYPOINT not in plans[0]:
        # The (corrected) place pose, moved along the table by -release offset: the held cube
        # is dragged from where it was set down onto the goal.
        for i, pl in enumerate(plans):
          kin.move_table(shift[i])
          try:
            pl[DRAG_WAYPOINT] = _solve(
              kin,
              "place",
              kin.fk(pl["place"])[0] - release_off[i],
              hand_xs[i],
              pl["place"],
            )
          except RuntimeError as e:
            skipped_corrections.append((i, f"drag: {e}"))
            pl[DRAG_WAYPOINT] = pl["place"]
      if wp == PRESS_WAYPOINT and PRESS_WAYPOINT not in plans[0]:
        elbow = ARM_JOINTS.index("arm_left_4_joint")
        for pl in plans:
          pl[PRESS_WAYPOINT] = pl["place"].copy()
          pl[PRESS_WAYPOINT][elbow] += PRESS_ELBOW_DQ
      nxt = np.stack([p[wp] for p in plans])
      n = n_steps(prev, nxt, steps, closed)
      if n > steps:
        stretched[phase] = n
      for s in range(1, n + 1):
        step_to(prev + (nxt - prev) * (s / n), closed)
      if phase in SETTLE_AFTER:
        waited = 0
        while waited < SETTLE_MAX_STEPS and not bool(ended.all()):
          speed = robot.data.joint_vel[:, arm_q].abs().amax(dim=-1)
          if bool((speed[~ended] < SETTLE_SPEED).all()):
            break
          step_to(nxt, closed)
          waited += 1
        settle_steps[phase] = waited
      if phase == "hold":
        # Corrections are made with the cube lifted clear of the table: resting on it, the
        # cube's friction holds the hand against a centimetre-scale move (measured: a
        # correction at release height left the error unchanged).
        lift = np.array([0.0, 0.0, PLACE_CORRECTION_LIFT])
        ups = []
        for i, (pl, hx) in enumerate(zip(plans, hand_xs, strict=True)):
          kin.move_table(shift[i])
          try:
            ups.append(
              _solve(kin, "place", kin.fk(pl["place"])[0] + lift, hx, pl["place"])
            )
          except RuntimeError as e:  # no lifted correction pose: hold where it is
            skipped_corrections.append((i, f"lift: {e}"))
            ups.append(pl["place"])
        up = np.stack(ups)
        move_and_settle(nxt, up, True)
        nxt = up
        for _ in range(PLACE_CORRECTIONS):
          err = (obj.data.root_link_pos_w - goal_w).cpu().numpy() - release_off
          err[:, 2] = 0.0
          fix = [
            i
            for i in torch.nonzero(~ended).flatten().tolist()
            if np.linalg.norm(err[i]) > PLACE_CORRECTION_TOL
          ]
          if not fix:
            break
          shifted = nxt.copy()
          for i in fix:
            kin.move_table(shift[i])
            try:
              up_i = _solve(
                kin, "place", kin.fk(nxt[i])[0] - err[i], hand_xs[i], nxt[i]
              )
              down_i = _solve(
                kin,
                "place",
                kin.fk(plans[i]["place"])[0] - err[i],
                hand_xs[i],
                plans[i]["place"],
              )
            except RuntimeError as e:
              skipped_corrections.append((i, str(e)))
              continue
            shifted[i], plans[i]["place"] = up_i, down_i
          move_and_settle(nxt, shifted, True)
          nxt = shifted
          place_corrections += 1
        down = np.stack([pl["place"] for pl in plans])
        move_and_settle(nxt, down, True)
        nxt = down
      prev = nxt
      if phase == "close":
        tcp = robot.data.site_pos_w[:, tcp_idx]
        diag["tcp_to_cube_mm"] = (
          1e3 * torch.norm(tcp - obj.data.root_link_pos_w, dim=-1).cpu().numpy()
        )
        diag["jaw_q"] = (
          robot.data.joint_pos[:, robot.joint_names.index(JAW_JOINT)].cpu().numpy()
        )
    goal = env.command_manager.get_term("goal_pose").command
    d = obj.data.root_link_pos_w - goal
    diag["final_cube_to_goal_mm"] = 1e3 * torch.norm(d, dim=-1).cpu().numpy()
    diag["final_cube_dz_mm"] = 1e3 * d[:, 2].cpu().numpy()
    diag["final_cube_speed"] = (
      torch.norm(obj.data.root_link_lin_vel_w, dim=-1).cpu().numpy()
    )
    still = torch.nonzero(~ended).flatten()
    if still.numel():
      tally.observe(**ev.probe(env, grasp_sensor))
      order += still.tolist()
      tally.finish(still)
  rows = [
    r | {"env": i}
    for i, r in sorted(zip(order, tally.done, strict=True), key=lambda t: t[0])
  ]

  env.close()
  if verbose:
    print(f"\n{clipped} env-steps needed an action outside [-1, 1] (clipped)")
    print(f"settle waits (control steps): {settle_steps}")
    print(f"place corrections applied: {place_corrections}")
    for env_i, reason in skipped_corrections:
      print(
        f"env {env_i}: place correction skipped, not plannable contact-free: {reason}"
      )
    print(f"phases stretched to respect the arm slew limit: {stretched}")
    print(f"env-steps commanding faster than the slew limit: {over_slew}")
    print(
      f"{'env':>3s} {'grasped':>8s} {'lifted':>7s} {'placed':>7s} {'pl&lift':>8s} "
      f"{'steps':>6s} {'tcp-cube':>9s} {'jaw':>6s} {'cube-goal':>10s}  ended by"
    )
    for i, r in enumerate(rows):
      print(
        f"{i:3d} {r['grasped']!s:>8s} {r['lifted']!s:>7s} {r['placed']!s:>7s} "
        f"{r['placed_and_lifted']!s:>8s} {r['length']:6d} "
        f"{diag.get('tcp_to_cube_mm', [np.nan] * num_envs)[i]:7.1f}mm "
        f"{diag.get('jaw_q', [np.nan] * num_envs)[i]:6.3f} "
        f"{diag['final_cube_to_goal_mm'][i]:8.1f}mm  {why[i] or '(script end)'}"
      )
    ok = sum(r["placed_and_lifted"] for r in rows)
    print(f"\nplaced and lifted: {ok} / {len(rows)}")
  for i, r in enumerate(rows):
    r["clipped_env_steps"] = clipped
    r["settle_steps"] = dict(settle_steps)
    r["place_corrections"] = place_corrections
    r["skipped_corrections"] = [
      reason for env_i, reason in skipped_corrections if env_i == r["env"]
    ]
    r["slew_stretched_phases"] = dict(stretched)
    r["over_slew_env_steps"] = over_slew
    r["ended_by"] = why[i]
    r["clipped_steps"] = int(clipped_env[i])
    r["plan_error"] = plan_error.get(r["env"])
    r["gate_locked_steps"] = int(gate_counts["locked"][r["env"]])
    r["gate_armed_steps"] = int(gate_counts["armed"][r["env"]])
    for key in ("band", "range"):
      r[f"{key}_hit_steps"] = {
        ph: int(v[r["env"]]) for ph, v in bound_hits[key].items() if v[r["env"]]
      }
    for key, v in diag.items():
      r[key] = float(v[i])
  return rows


def reach_map(args: argparse.Namespace) -> list[dict]:
  """Plan S7 P2: which spawn points the scripted pick-and-place reaches *through the action
  term* -- the env's own scales, clip and slew limit -- rather than by IK alone (S5.0's lesson:
  C1/C2 trained on a task the action space could not do).

  A grid over x, y and a few cube yaws, each point one env of a batched rollout to the fixed
  target. A point counts when it is placed and lifted with no clipped control step; the
  plan error, skipped re-plans and final cube-to-goal distance are kept so a failure can be
  told apart from a near-miss. Written after every batch, and resumed from what is already in
  ``--json``, so a long map survives an interruption. ``--shard``/``--shards`` split the grid
  across processes.
  """
  xs = np.arange(args.x[0], args.x[1] + 1e-9, args.step)
  ys = np.arange(args.y[0], args.y[1] + 1e-9, args.step)
  grid = [
    ((round(float(x), 4), round(float(y), 4)), float(d))
    for x in xs
    for y in ys
    for d in args.yaws
  ][args.shard :: args.shards]
  out: list[dict] = []
  if args.json.exists():
    out = json.loads(args.json.read_text())
  done = {(r["x"], r["y"], r["yaw_deg"]) for r in out}
  todo = [p for p in grid if (p[0][0], p[0][1], p[1]) not in done]
  print(f"{len(grid)} points in this shard, {len(todo)} to do")
  for b in range(0, len(todo), args.batch):
    chunk = todo[b : b + args.batch]
    rows = rollout(
      num_envs=len(chunk),
      device=args.device,
      seed=args.seed,
      reward=args.reward,
      verbose=False,
      action_slew_rate_scale=args.slew,
      spawns=[(xy, math.radians(d)) for xy, d in chunk],
    )
    for (xy, d), r in zip(chunk, rows, strict=True):
      out.append(
        {
          "x": xy[0],
          "y": xy[1],
          "yaw_deg": d,
          "reached": bool(r["placed_and_lifted"]) and r["clipped_steps"] == 0,
          "placed_and_lifted": bool(r["placed_and_lifted"]),
          "grasped": bool(r["grasped"]),
          "lifted": bool(r["lifted"]),
          "clipped_steps": r["clipped_steps"],
          "plan_error": r["plan_error"],
          "skipped_corrections": r["skipped_corrections"],
          "final_cube_to_goal_mm": r["final_cube_to_goal_mm"],
        }
      )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out, indent=1))
    ok = sum(r["reached"] for r in out)
    print(f"batch {b // args.batch + 1}: {len(out)} / {len(grid)} done, {ok} reached")
  return out


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  sub = p.add_subparsers(dest="cmd", required=True)
  d = sub.add_parser("derive", help="IK waypoints -> arm action scales")
  d.add_argument("--json", type=pathlib.Path, default=None)
  r = sub.add_parser("rollout", help="scripted episodes through the real env")
  r.add_argument("--num-envs", type=int, default=10)
  r.add_argument("--device", default="cpu")
  r.add_argument("--seed", type=int, default=0)
  r.add_argument("--reward", default="R-A")
  r.add_argument("--random-only", action="store_true", help="no forced corners")
  r.add_argument(
    "--slew", type=float, default=1.0, help="action_slew_rate_scale (S7 FA: 0.5)"
  )
  r.add_argument("--baseline", default="S71", help="BASELINES name (S7-2: B)")
  r.add_argument(
    "--spawn", default="S71", help="SPAWN_BOXES name; a W box brings P2's scales"
  )
  r.add_argument("--dr", default="DR0")
  r.add_argument("--tool-speed", default="none", help="TOOL_SPEED_CAPS name (V)")
  r.add_argument(
    "--gains",
    default=None,
    help="plan-S8 gain code the robot is built on (G0 | G15 | G20 | G20s); default: "
    "the spawn box's arm as it is, i.e. G0",
  )
  r.add_argument(
    "--stack-speed",
    action="store_true",
    help="plan S7-8: the stack task's arm and jaw speeds instead of --slew",
  )
  r.add_argument(
    "--action", default="abs", help="ACTION_MODES name (S7-8: abs | delta)"
  )
  r.add_argument("--lag", default="none", help="LATENCY_MODELS name (S7-8 runs JS)")
  r.add_argument(
    "--target",
    default="B",
    help="TARGET_AREAS name (S7-8: WC, the goal from the spawn area)",
  )
  r.add_argument("--effort", default="none", help="EFFORT_BOUNDS name (CTP)")
  r.add_argument(
    "--shape",
    default="cube",
    help="OBJECT_SHAPES name (plan S8: cube | ball | ball-dr | mix)",
  )
  r.add_argument(
    "--ball-offset",
    type=float,
    default=None,
    help="plan S8: height of the pad sites' midpoint above a ball's centre at the grasp, m "
    "(default: pick_place.rewards.EQ_OFFSET)",
  )
  r.add_argument(
    "--episode-s", type=float, default=None, help="episode length; the script is slow"
  )
  r.add_argument("--goal", default="table", help="GOALS name (plan S9: table | stack)")
  r.add_argument("--gate", default="none", help="GATES name (plan S9: none | stack)")
  r.add_argument(
    "--base-corners",
    action="store_true",
    help="plan S9: put the base cube at the target box's centre and corners (env i at "
    "box_corners[i mod 9]) instead of the reset's random draw",
  )
  r.add_argument(
    "--arm-clearance",
    action="store_true",
    help="plan S7-10: arm 1-6's lowest point over the table, per phase and link",
  )
  r.add_argument(
    "--pregrasp",
    default="none",
    help="PREGRASP_CODES name (plan S7-11: none | watch | NP | JO | G | S | OVR)",
  )
  r.add_argument("--json", type=pathlib.Path, default=None)
  m = sub.add_parser(
    "map", help="action-space reach map over a spawn grid (plan S7 P2)"
  )
  m.add_argument("--x", type=float, nargs=2, default=(0.26, 0.62), metavar=("LO", "HI"))
  m.add_argument(
    "--y", type=float, nargs=2, default=(-0.06, 0.44), metavar=("LO", "HI")
  )
  m.add_argument("--step", type=float, default=0.02, help="grid spacing, m")
  m.add_argument(
    "--yaws", type=float, nargs="+", default=(-45.0, 0.0, 45.0), help="deg"
  )
  m.add_argument("--slew", type=float, default=0.5, help="action_slew_rate_scale")
  m.add_argument("--reward", default="R-A")
  m.add_argument("--batch", type=int, default=32, help="spawn points per rollout")
  m.add_argument("--shard", type=int, default=0)
  m.add_argument("--shards", type=int, default=1)
  m.add_argument("--device", default="cpu")
  m.add_argument("--seed", type=int, default=0)
  m.add_argument("--json", type=pathlib.Path, required=True)
  return p.parse_args()


class ArmClearanceHook:
  """Plan S7-10 step 0, item 4: how close a *legal* pick-and-place brings arm 1-6 to the table.

  A `rollout` hook: per phase, per env and per link, the lowest collision point over the
  table's footprint (`mdp.arm_table_clearance`, AH's own measure). AH's margin must stay below
  the scripted motion's minimum, or the term fires on a correct pick. No sensor is needed."""

  def __init__(self) -> None:
    self.asset = None
    self.links: list[str] = []
    self.by_phase: dict = {}  # phase -> (B, L) tensor, m

  def __call__(self, phase: str, env) -> None:
    import torch
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from pal_mjlab.tasks.manipulation import mdp
    from pal_mjlab.tasks.manipulation.pick_place.arm_table import arm_body_regex
    from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS_LEFT_ARM

    if self.asset is None:
      gripper = KANGAROO_GRIPPERS_LEFT_ARM.gripper("left")
      self.asset = SceneEntityCfg("robot", body_names=(arm_body_regex(gripper),))
      self.asset.resolve(env.scene)
      robot = env.scene["robot"]
      self.links = [robot.body_names[i] for i in self.asset.body_ids]
    c = mdp.arm_table_clearance(env, self.asset, SceneEntityCfg("table"))
    prev = self.by_phase.get(phase)
    self.by_phase[phase] = c if prev is None else torch.minimum(prev, c)

  def report(self, rows: list[dict]) -> None:
    """Print the table and add each env's per-link minimum (m) to its row."""
    import torch

    if not self.by_phase:
      return
    every = torch.stack(list(self.by_phase.values())).amin(dim=0)  # (B, L)
    cm = lambda v: "      -" if not math.isfinite(v) else f"{v * 100:7.1f}"  # noqa: E731
    print(
      "\nARM 1-6 OVER THE TABLE (plan S7-10 step 0 item 4), lowest point, cm, min over envs"
    )
    print(f"{'phase':10s}" + "".join(f"{n.split('_')[2]:>7s}" for n in self.links))
    for phase, c in self.by_phase.items():
      print(f"{phase:10s}" + "".join(cm(float(v)) for v in c.amin(dim=0)))
    print(f"{'all':10s}" + "".join(cm(float(v)) for v in every.amin(dim=0)))
    for i, row in enumerate(rows):
      if i < every.shape[0]:
        row["arm_min_clearance_by_link_m"] = dict(
          zip(self.links, every[i].tolist(), strict=True)
        )


def main() -> None:
  args = parse_args()
  if args.cmd == "derive":
    derive(args)
    return
  if args.cmd == "map":
    reach_map(args)
    return
  from pal_mjlab.tasks.manipulation import tabletop_env_cfg as tt
  from pal_mjlab.tasks.manipulation.pick_place import pick_place_env_cfg as pp
  from pal_mjlab.tasks.manipulation.pick_place.kangaroo.env_cfg import SPAWN_ROBOTS
  from pal_mjlab.tasks.manipulation.pick_place.pregrasp import PREGRASP_CODES
  from pal_mjlab.tasks.manipulation.robots import kangaroo_grippers_left_arm

  arm_hook = ArmClearanceHook() if args.arm_clearance else None
  robot = SPAWN_ROBOTS[args.spawn]
  if args.gains is not None:
    effort = pp.EFFORT_BOUNDS[args.effort]
    robot = kangaroo_grippers_left_arm(
      args.gains, robot, arm_effort_scale=effort.effort_scale
    )
  rows = rollout(
    num_envs=args.num_envs,
    device=args.device,
    seed=args.seed,
    reward=args.reward,
    corners_first=not args.random_only,
    action_slew_rate_scale=args.slew,
    robot=robot,
    baseline=pp.BASELINES[args.baseline],
    spawn_box=pp.SPAWN_BOXES[args.spawn],
    dr=args.dr,
    tool_speed_cap=pp.TOOL_SPEED_CAPS[args.tool_speed],
    effort=pp.EFFORT_BOUNDS[args.effort],
    latency=tt.LATENCY_MODELS[args.lag],
    target_area=pp.TARGET_AREAS[args.target],
    stack_speed=args.stack_speed,
    action_mode=args.action,
    episode_length_s=args.episode_s,
    shape=pp.OBJECT_SHAPES[args.shape],
    ball_offset=args.ball_offset,
    goal=args.goal,
    gate=args.gate,
    base_spawns=base_box_corners(pp.BASELINES[args.baseline], args.num_envs)
    if args.base_corners
    else None,
    hook=arm_hook,
    pregrasp=PREGRASP_CODES[args.pregrasp],
  )
  if arm_hook is not None:
    arm_hook.report(rows)
  if args.json:
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(rows, indent=2))
    print(f"wrote {args.json}")


if __name__ == "__main__":
  main()
