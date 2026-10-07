"""Score a trained pick-and-place policy on the rates that name its failure mode.

A training curve is evidence about training. This is the closed-loop measurement the plan's
exit criteria are written against: run a checkpoint for N complete episodes and report what
fraction of them the policy actually finished, and *how* the rest failed.

    uv run --extra cu128 python scripts/eval_pick_place.py \
      --checkpoint logs/rsl_rl/kangaroo_pick_place/<run>/model_20000.pt \
      --episodes 200 --num-envs 64 --json artifacts/eval/c1_seed1.json

The two placed rates are the point
--------------------------------------------------------------------------------
`_placed_bool` counts `released = ¬grasped`, and the goal sits at the cube's resting
height, so **a cube that was never picked up is "placed"** and sliding it 20 cm scores a
success. C0 measured exactly that: every arm reaching success with `Episode_Reward/grasp`
at 0. So this reports both

  placed          at the goal, released, settled  -- what R-A and R-C are scored on
  placed & lifted the same, and the object left the table at some point this episode

and they are computed the same way for every arm, from the same rollout, regardless of
which gating that arm's own reward happens to use. That is what makes a C1 number and a
C2 number comparable.

The four diagnostic rates (plan, "Metrics and diagnostics")
--------------------------------------------------------------------------------
  grasped-ever    near 0 with a saturated reach term is the Isaac Lab #204 signature:
                  reach-but-never-close
  lifted-ever     high grasped, low lifted = grasps but cannot lift, or is sliding
  held-at-goal    at the goal and still gripping when the clock ran out -- the
                  Text2Reward hold-at-goal failure
  placed          the number that counts

Transport and speed, while held
--------------------------------------------------------------------------------
  transport       horizontal distance to the target while held, and the height over it
                  while held within 3 cm -- a hover, not a miss (S5.2 base selection)
  speed           per-episode peak cube and TCP speed while held, over one control step
                  -- the read of plan S7's speed bound
  release         the episode's last let-go: the cube's xy distance to the target, its
                  height above it, and its speed -- a set-down vs a cube let go short
  set-down        at that let-go: table contact, the dwell held still on the table since
                  a touchdown near the target, the cube's vertical and horizontal speed,
                  and plan S7-2's M7 grade of it (`SET_DOWN_*`, mirrored from the reward's
                  latch so the eval also runs on code without M7)

Per-episode records
--------------------------------------------------------------------------------
Every episode's spawn pose, table offset from its nominal pose, target, release point
and final cube position go into the JSON (``episodes_detail``): what plan S7-2's D5
diagnosis and the W runs' placed rate by spawn distance are read from.

Plan S7-10: the arm on the table (``ArmTally``)
--------------------------------------------------------------------------------
Per episode, the driven arm's links 1-6 against the table: whether any presses harder than
`ARM_CONTACT_N` (1 N), on how many steps, the peak force, which link, and in which phase (before
the first grasp, while held, after the let-go, other). The contact reads need the arm-table
sensor, i.e. an env built with ``ARM_TABLE`` set (``ARM_TABLE=watch`` for a policy trained without
it); without it they are left out. The clearance read needs no sensor: each link's lowest
collision point above the table top, the episode's minimum, and the share of episodes that come
within 1 / 2 / 3 cm -- the margin a real arm with tracking error needs.

Plan S7-4: the release is the jaw opening, and the retreat (``RetreatTally``)
--------------------------------------------------------------------------------
Every read above keeps its old definition (a let-go = the pads losing the cube), so a number
stays comparable with the S7-2 / S7-3 reports. Beside them, the J reads: whether the jaw really
opens -- the cube enters "jaw below `JAW_OPEN` (0.15 rad), pads off it" after the first lift; for
a cube held across an edge (pads touching at ~0.10 rad) the let-go itself -- its angle
at the pads' let-go, and `placed` / `placed & lifted` with the jaw open. With ``--handoff`` a
position controller takes over, as it will on the robot: once the jaw has opened after the lift
and the TCP has stayed `HANDOFF_CLEAR` above the cube's top for `HANDOFF_STEPS` steps, or at
`HANDOFF_TIMEOUT_S`, the env gets action 0 -- the home pose and the open jaw, reached through the
action term's own slew (and V, C, T where the run has them). Read: the lift height before the
hand-off, the time from placed to it, the cube's displacement during the lift and the return,
whether it is still placed at the end, the time to home, and S7-4's success -- placed & lifted &
J-released & lifted clear (>= `HANDOFF_CLEAR`) & still placed after the return.

Plan S7-5: the goal tolerance split into xy and z, and the tilt at the let-go
--------------------------------------------------------------------------------
Every placed rate is read under both goal tolerances, whatever the run trained on: the 3-D
distance below `GOAL_THRESHOLD` (every run to S7-4) and S7-5's Z, `GOAL_TOL_XYZ` (xy < 2 cm and
|dz| < 2 cm) -- ``*_z`` keys -- so an S7-5 number stays comparable with S7-4's. Beside them, the
cube's tilt at the pads' let-go: the angle between the vertical and the nearest face normal.

Plan S10: the box (``GOAL=box*``, `BoxTally`)
--------------------------------------------------------------------------------
When the scene has plan S10's box, beside every read above: the cube in the box at the end of
the episode -- after the hand-off's return with ``--handoff`` (the headline, with
``--handoff-lift 0.05``: the fingers go straight up out of the box before the arm turns home) --
and S10's success (J-released after the lift, handed off, in the box at the end); bounce-outs
(in the box at some step after the J-release, out at the end); the hand and the cube on a box wall
while the cube is held; the box's displacement at the end; and the let-go's xy offset, height
above the rim and yaw error (|yaw mod 90 deg|), from the reward's own release latch. Arm 1-6 on
the table is plan S7-10's `ArmTally`.

Plan S10.1 (the box on the wide area, at any yaw): the let-go's yaw error is against the box
(|wrap_90(psi_cube - psi_box)|, the latch's own), and S10's success is also binned by where the box
stood (`BOX_POSITION_BINS`), its yaw (`BOX_YAW_BINS_DEG`) and the carry from the cube's spawn to the
box (`BOX_CARRY_BINS_CM`): ``box_by_position`` / ``box_by_yaw`` / ``box_by_carry``.

Horizon
--------------------------------------------------------------------------------
Not the `--play` cfg as-is: `apply_play_overrides` sets `episode_length_s` to 1e9, so
episodes never time out and nothing ever completes. This takes the play cfg (no observation
corruption, no curriculum) and restores the *training* episode length, which is the horizon
the policy was optimised for and the only one these rates mean anything against.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
from dataclasses import asdict

import numpy as np
import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from pal_mjlab.tasks.manipulation.mdp.actions import DeltaRateLimitedJointPositionAction
from pal_mjlab.tasks.manipulation.mdp.observations import (
  _grasped_bool,
  _on_surface_bool,
  _placed_bool,
)
from pal_mjlab.tasks.manipulation.mdp.shapes import (
  PerShape,
  object_half_height,
  resolve,
)
from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import (
  GOAL_THRESHOLD,
  OBJECT,
)
from pal_mjlab.tasks.manipulation.pick_place.rewards import (
  BALL_JAW_OPEN,
  BALL_JAW_OPEN_BY_RADIUS,
)
from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS_LEFT_ARM
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import (
  COMMAND,
  JAW_SERVO_MAX_VEL,
  JAW_SERVO_TAU,
  ROBOT,
  SUPPORT_SENSOR,
  TABLE,
)

TASK = "Mjlab-PickPlace-Pal-Kangaroo-Grippers"

#: "Near the target" for the transport measurements: *horizontal* distance while the cube is
#: held. Horizontal on purpose -- `Metrics/goal_pose/position_error` is a 3-D norm, so a cube
#: carried over the target and hovering 6 cm above it reads "6 cm off" there, which is how the
#: c1a screen was first misread as "no transport" (2026-09-24, S5.2 base selection).
NEAR_XY = 0.03
#: The gripper's tool centre point, whose speed while holding is S7's speed-bound read.
TCP_SITE = KANGAROO_GRIPPERS_LEFT_ARM.gripper("left").tcp_site
#: The two pad sites, whose midpoint is the "pad centre" of plan S8's grasp-height read.
PAD_SITES = KANGAROO_GRIPPERS_LEFT_ARM.gripper("left").pad_sites

#: Plan S7-2's M7 set-down, restated from `mdp.observations` / `pick_place.rewards` rather
#: than imported, so this eval runs on checkouts that predate M7 (the D5 diagnosis reads an
#: S7-1 checkpoint on S7-1's code): a let-go is a set-down when the cube rests on the table,
#: moves slower than `SET_DOWN_STILL`, and was held there, still and within
#: `SET_DOWN_RADIUS` of the target, for `SET_DOWN_STEPS` steps since a touchdown within it.
#: `test_eval_pick_place.py` checks them against the reward's.
SET_DOWN_RADIUS = 0.015
SET_DOWN_STILL = 0.05
SET_DOWN_STEPS = 3
SET_DOWN_CLEARANCE = 0.005
#: M6's grade (`mdp.rewards.release_quality_of`), for the per-release numbers.
M6_GRADE = {"dist_std": 0.02, "height_free": 0.01, "height_std": 0.01, "floor": 0.25}

#: Plan S7-4. `JAW_OPEN`: the jaw counts as open below this measured angle (rad), module J's
#: `mdp.observations.JAW_OPEN_BELOW`. The hand-off to the position controller: the jaw open
#: after the lift and the TCP at least `HANDOFF_CLEAR` (m) above the cube's top for
#: `HANDOFF_STEPS` consecutive steps, or at `HANDOFF_TIMEOUT_S`. "Home": every driven-arm joint
#: within `HOME_TOL` (rad) of its default. Restated like `SET_DOWN_*`; the test checks them.
JAW_OPEN = 0.15
#: Plan S8: J's threshold depends on the object (L7) -- the ball's is `rewards.BALL_JAW_OPEN_BY_RADIUS`;
#: `jaw_open_for` is the number for an env's own object, per env when it can be either.
JAW_OPEN_BY_SHAPE = PerShape(cube=JAW_OPEN, ball=BALL_JAW_OPEN_BY_RADIUS)


def jaw_open_for(env: ManagerBasedRlEnv) -> float | torch.Tensor:
  """J's threshold for ``env``'s object: `JAW_OPEN` for a cube (a float, as every run before S8),
  `BALL_JAW_OPEN` for a ball, per env for plan S8's MIX."""
  return resolve(env, JAW_OPEN_BY_SHAPE)


#: ... and closed above this (a stroke is open -> closed, with hysteresis; S7-7's re-approach count).
JAW_CLOSED = 0.25
JAW_JOINT = "gripper_left_finger_joint"
HANDOFF_CLEAR = 0.03
HANDOFF_STEPS = 5
HANDOFF_TIMEOUT_S = 8.0
HOME_TOL = 0.05
#: The vertical retreat (`--handoff-lift`, S7-5 report §8 item 1): the TCP goes straight up at
#: most `LIFT_SPEED` (m/s) in `LIFT_WAYPOINT` (m) IK steps before the joint-space return; that
#: return starts once the TCP is within `LIFT_TOL` (m) of the height asked for, or
#: `LIFT_GRACE_S` after the planned path ends (the action term's slew, T's band may hold it back).
LIFT_SPEED = 0.2
LIFT_WAYPOINT = 0.01
LIFT_TOL = 0.01
LIFT_GRACE_S = 0.5
#: Plan S7-8's reads by where the cube starts and how far it goes: the old W2 spawn box about the
#: table centre (0.55, 0) -- x -0.21..-0.09, y 0.20..0.34 -- and the carry bins, m. `TARGET=WC` draws
#: 10 cm up to the area's 36 cm diagonal (the kept part `WCK`: up to 20 cm), so the plan's 10-20 / 20-30 /
#: over 30 cm bins are split at 15 cm.
W2_REL_X, W2_REL_Y = (-0.21, -0.09), (0.20, 0.34)
TABLE_CENTRE_XY = (0.55, 0.0)
CARRY_BINS = ((0.10, 0.15), (0.15, 0.20), (0.20, 0.30), (0.30, 9.99))
#: Plan S7-5's Z: "at the goal" is xy < 2 cm and |dz| < 2 cm, `pick_place.rewards.Z_GOAL_TOL`.
GOAL_TOL_XYZ = (0.02, 0.02)
#: Plan S8: a let-go this far above the resting height (m) is a drop, not a set-down.
DROP_HEIGHT = 0.03


def release_grades(dxy: float, dz: float, passed: bool) -> tuple[float, float]:
  """(M6, M7) grade of one release at ``dxy`` from the target and ``dz`` above it. M7 is M6
  without the free centimetre, paid for a set-down only (plan S7-2)."""
  g = M6_GRADE

  def q(height_free: float) -> float:
    near = math.exp(-((dxy / g["dist_std"]) ** 2))
    low = math.exp(-((max(dz - height_free, 0.0) / g["height_std"]) ** 2))
    return g["floor"] + (1.0 - g["floor"]) * near * low

  return q(g["height_free"]), (q(0.0) if passed else g["floor"])


#: Plan S8's plant reads (`--plant`): the joints a gain code changes that the policy does not
#: drive -- the parked right arm and the pelvis -- the arm it does drive, and all of them.
PARKED_JOINTS = r"(arm_right_[1-7]|pelvis_[12])_joint"
DRIVEN_ARM_JOINTS = r"arm_left_[1-7]_joint"
GAINED_JOINTS = r"(pelvis_[12]|arm_(left|right)_[1-7])_joint"
#: A joint counts as at its effort limit within this fraction of it: the limit clamps the
#: torque exactly, and the margin absorbs float32 round-off. As `hold_check.py`.
SATURATION_FRACTION = 0.999


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--checkpoint", type=pathlib.Path, required=True, help="model_*.pt")
  p.add_argument("--task", default=TASK)
  p.add_argument("--episodes", type=int, default=200, help="complete episodes to score")
  p.add_argument("--num-envs", type=int, default=64)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument(
    "--json", type=pathlib.Path, default=None, help="also write the result"
  )
  p.add_argument(
    "--plant",
    action="store_true",
    help="also read the plant (plan S8): parked-arm/pelvis hold error, the driven arm's "
    "tracking error while carrying, effort saturation, cube-table force at set-down",
  )
  p.add_argument(
    "--handoff",
    action="store_true",
    help="plan S7-4: hand the arm to a position controller (action 0: home, jaw open) once "
    "the jaw has opened after the lift and the TCP is clear of the cube, or at 8 s",
  )
  p.add_argument(
    "--handoff-timeout",
    type=float,
    default=HANDOFF_TIMEOUT_S,
    metavar="S",
    help="with --handoff: the episode time (s) at which the position controller takes over a "
    f"policy that has not lifted clear (default {HANDOFF_TIMEOUT_S}). At plan S7-8's stack speeds "
    "the return home takes up to ~8 s and the episode is 30 s: pass ~21.",
  )
  p.add_argument(
    "--handoff-lift",
    type=float,
    default=None,
    metavar="H",
    help="with --handoff: the position controller first lifts the TCP straight up to H m over "
    "the cube's top (S7-5 report §8 item 1), then goes home; default: straight home",
  )
  p.add_argument(
    "--obs-noise",
    choices=("off", "on", "on-no-object"),
    default="off",
    help="plan S7-9's precision split: the actor's observation noise. off (default, every eval "
    "before it: play mode turns corruption off); on (as trained: joint and cube-tracker bias and "
    "jitter); on-no-object (as trained but the cube's position / yaw noise off). The tracker lag "
    "is LAG's, not this flag's",
  )
  p.add_argument(
    "--solver-iterations",
    type=int,
    nargs=2,
    default=None,
    metavar=("ITER", "LS_ITER"),
    help="evaluate with the constraint solver at ITER / LS_ITER iterations instead of the training "
    "cfg's 10 / 20 (plan S10: 100 50 is converged for cube-on-box contacts, S7-9 step 0); eval only",
  )
  p.add_argument(
    "--stress-jaw",
    action="store_true",
    help="plan S7-11 stress (a): the jaw servo slower than S7-6's -- its time constant x 2 "
    "(0.22-0.38 s) and its speed limit x 0.5 (0.625-0.875 rad/s), the same absolute range for every "
    "run (also one trained on LAG=FP+JSW). Needs LAG with JS",
  )
  p.add_argument(
    "--stress-lag",
    type=int,
    nargs=2,
    default=None,
    metavar=("LO", "HI"),
    help="plan S7-11 stress (b): the cube tracker's lag, control steps, drawn per episode in "
    "[LO, HI] in place of LAG's 5-8 (e.g. 8 12 = 160-240 ms). Needs LAG with FP",
  )
  p.add_argument(
    "--stress-friction",
    type=float,
    default=None,
    metavar="MU",
    help="plan S7-11 stress (c): the cube-table sliding friction MU (e.g. 0.4): the table top's "
    "and the cube's sliding friction both set to MU. MuJoCo takes the larger of two geoms' "
    "coefficients, so the table alone would leave the cube's DR draw (0.5-1.2); the pads keep "
    "theirs (geom priority 2)",
  )
  p.add_argument(
    "--trace",
    type=pathlib.Path,
    default=None,
    help="plan S7-7.0: write every step's jaw / cube-sensor state (all envs) to this .npz, "
    "for calibrating O-K's held rule offline (needs an OCC / OBJ_OBS env)",
  )
  return p.parse_args(argv)


class Tally:
  """Per-episode outcome flags, latched across the steps of an episode.

  Latched rather than sampled because every quantity here is "did this *ever* happen in
  this episode". Placed states persist -- the cube stays where it was put -- so reading the
  state at the top of each step catches them even though the terminal state, which the
  env has already reset by the time `step` returns, is never seen.

  Except when the placement is what *ended* the episode. An arm with a `success`
  termination is reset in the same step the cube comes to rest on the goal, so the placed
  state never exists at the top of any step; those episodes have to be banked from the
  termination itself, with `terminal`, before `finish`. Until 2026-09-23 they were not,
  and every such episode was counted as a failure (found by S5.0's scripted rollout).
  """

  FIELDS = (
    "grasped",
    "lifted",
    "placed",
    "placed_and_lifted",
    "placed_and_lifted_z",
    "dropped",
    "held_at_goal",
  )

  def __init__(self, num_envs: int, device: str) -> None:
    self.num_envs = num_envs
    z = lambda: torch.zeros(num_envs, dtype=torch.bool, device=device)  # noqa: E731
    self.live = {name: z() for name in self.FIELDS}
    self.first_place = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    self.age = torch.zeros(num_envs, dtype=torch.long, device=device)
    # Transport accumulators (`measure`): steps held, horizontal distance to the target while
    # held (sum for the mean, running min), and height above the target while held within
    # NEAR_XY horizontally.
    zf = lambda v: torch.full((num_envs,), v, device=device)  # noqa: E731
    self.held_n = torch.zeros(num_envs, dtype=torch.long, device=device)
    self.dxy_sum = zf(0.0)
    self.dxy_min = zf(float("inf"))
    self.near_n = torch.zeros(num_envs, dtype=torch.long, device=device)
    self.dz_sum = zf(0.0)
    # Speed accumulators (`measure_speed`): the fastest the cube and the TCP moved over one
    # control step while the cube stayed held, per episode, and the positions they moved from.
    self.prev_obj = torch.zeros(num_envs, 3, device=device)
    self.prev_tcp = torch.zeros(num_envs, 3, device=device)
    self.prev_held = torch.zeros(num_envs, dtype=torch.bool, device=device)
    self.speed_n = torch.zeros(num_envs, dtype=torch.long, device=device)
    self.obj_v_peak = zf(0.0)
    self.tcp_v_peak = zf(0.0)
    # The last release of the episode (`measure_speed` with a goal): the cube's horizontal
    # distance to the target, its height above it, and its speed, the step the jaw let go.
    self.released = torch.zeros(num_envs, dtype=torch.bool, device=device)
    self.rel_dxy = zf(0.0)
    self.rel_dz = zf(0.0)
    self.rel_v = zf(0.0)
    # The set-down at the last let-go (`measure_set_down`), and the state it is built from.
    zb = lambda: torch.zeros(num_envs, dtype=torch.bool, device=device)  # noqa: E731
    zl = lambda: torch.zeros(num_envs, dtype=torch.long, device=device)  # noqa: E731
    self.sd_prev_held, self.sd_clear, self.sd_touch_ok = zb(), zb(), zb()
    self.sd_dwell = zl()
    self.sd_released, self.rel_on_table = zb(), zb()
    self.rel_dwell = zl()
    self.rel_vz, self.rel_vxy, self.rel_x, self.rel_y = (
      zf(0.0),
      zf(0.0),
      zf(0.0),
      zf(0.0),
    )
    # Plan S8: the pad-site midpoint's height above the object's centre at the episode's first grasp
    # (`measure_grasp_height`), m.
    self.grasp_dz = zf(0.0)
    self.grasp_seen = zb()
    # Per-episode records (`record`): the first step's spawn, table and target, the last
    # step's cube position.
    self.first = torch.zeros(num_envs, 9, device=device)
    self.last_pos = torch.zeros(num_envs, 3, device=device)
    self.done: list[dict[str, bool | int | float | None]] = []

  def observe(self, **flags: torch.Tensor) -> None:
    for name, value in flags.items():
      self.live[name] |= value
    newly = flags["placed"] & (self.first_place < 0)
    self.first_place = torch.where(newly, self.age, self.first_place)
    self.age += 1

  def measure(self, held: torch.Tensor, dxy: torch.Tensor, dz: torch.Tensor) -> None:
    """Accumulate this step's transport geometry for the episodes that hold the cube.

    ``dxy`` is the cube's horizontal distance to the target, ``dz`` its height above the
    target (the target is the cube's resting pose on the table, so ``dz`` is 0 when it
    rests there). Only held steps count; the hover height only counts held steps within
    `NEAR_XY` horizontally. Call once per step, alongside `observe`.
    """
    held = held.bool()
    near = held & (dxy < NEAR_XY)
    self.held_n += held.long()
    self.dxy_sum += torch.where(held, dxy, torch.zeros_like(dxy))
    self.dxy_min = torch.where(held, torch.minimum(self.dxy_min, dxy), self.dxy_min)
    self.near_n += near.long()
    self.dz_sum += torch.where(near, dz, torch.zeros_like(dz))

  def measure_speed(
    self,
    held: torch.Tensor,
    obj_pos: torch.Tensor,
    tcp_pos: torch.Tensor,
    step_dt: float,
    goal_pos: torch.Tensor | None = None,
  ) -> None:
    """Track the peak cube and TCP speed while the cube is held (plan S7's speed read).

    A step's speed is its displacement over one control step, ``|p_t - p_(t-1)| / step_dt``,
    counted only when the cube was held at both ends of the step: the jaw closing and the
    release are not transport, and the jump across a reset is not motion (`finish` clears
    `prev_held`). Displacement rather than the instantaneous velocity, which spikes on
    contact transients and is not what a rate limit bounds. Call once per step, alongside
    `measure`.

    With ``goal_pos``, also records the episode's last release -- the step the cube goes from
    held to not held -- as the cube's horizontal distance to the target, its height above it,
    and its speed over that step. A cube let go 5 cm short at 1 m/s and one set down on the
    target both count as placed; this is what tells them apart (S5.2 report section 6).
    """
    held = held.bool()
    both = held & self.prev_held
    if goal_pos is not None:
      let_go = self.prev_held & ~held
      dxy = torch.norm((obj_pos - goal_pos)[:, :2], dim=-1)
      v = torch.norm(obj_pos - self.prev_obj, dim=-1) / step_dt
      self.released |= let_go
      self.rel_dxy = torch.where(let_go, dxy, self.rel_dxy)
      self.rel_dz = torch.where(let_go, obj_pos[:, 2] - goal_pos[:, 2], self.rel_dz)
      self.rel_v = torch.where(let_go, v, self.rel_v)
    v_obj = torch.norm(obj_pos - self.prev_obj, dim=-1) / step_dt
    v_tcp = torch.norm(tcp_pos - self.prev_tcp, dim=-1) / step_dt
    self.obj_v_peak = torch.where(
      both, torch.maximum(self.obj_v_peak, v_obj), self.obj_v_peak
    )
    self.tcp_v_peak = torch.where(
      both, torch.maximum(self.tcp_v_peak, v_tcp), self.tcp_v_peak
    )
    self.speed_n += both.long()
    self.prev_obj = obj_pos.clone()
    self.prev_tcp = tcp_pos.clone()
    self.prev_held = held.clone()

  def measure_grasp_height(self, grasped: torch.Tensor, pad_dz: torch.Tensor) -> None:
    """Plan S8: where on the ball the jaw closes. ``pad_dz`` is the pad-site midpoint's height above
    the object's centre (m), and is kept at the step the grasp is first seen, per episode: an
    instantaneous value, so it is the open-jaw one minus the ~10 mm the sites move toward the
    fingertips as the jaw closes to the touch. Call once per step, alongside `measure`."""
    first = grasped.bool() & ~self.grasp_seen
    self.grasp_dz = torch.where(first, pad_dz, self.grasp_dz)
    self.grasp_seen |= first

  def measure_set_down(
    self,
    held: torch.Tensor,
    on_table: torch.Tensor,
    obj_pos: torch.Tensor,
    obj_vel: torch.Tensor,
    goal_pos: torch.Tensor,
  ) -> None:
    """Plan S7-2's M7 read of the episode's last let-go, the way the reward's latch builds it
    (`mdp.observations._release_latch`): a touchdown is the first step held on the table
    after the cube was held `SET_DOWN_CLEARANCE` clear of it (above the target's height, which
    is the resting height); the dwell counts steps held on the table, slower than
    `SET_DOWN_STILL` and within `SET_DOWN_RADIUS` of the target, since a touchdown within that
    radius; at a let-go the table contact, the cube's velocity (instantaneous, from physics)
    and the dwell up to the step before are recorded. Call once per step, alongside
    `measure_speed`."""
    held, on_table = held.bool(), on_table.bool()
    dxy = torch.norm((obj_pos - goal_pos)[:, :2], dim=-1)
    speed = torch.norm(obj_vel, dim=-1)
    height = obj_pos[:, 2] - goal_pos[:, 2]
    let_go = self.sd_prev_held & ~held
    self.sd_released |= let_go
    self.rel_on_table = torch.where(let_go, on_table, self.rel_on_table)
    self.rel_dwell = torch.where(let_go, self.sd_dwell, self.rel_dwell)
    self.rel_vz = torch.where(let_go, obj_vel[:, 2], self.rel_vz)
    self.rel_vxy = torch.where(let_go, torch.norm(obj_vel[:, :2], dim=-1), self.rel_vxy)
    self.rel_x = torch.where(let_go, obj_pos[:, 0], self.rel_x)
    self.rel_y = torch.where(let_go, obj_pos[:, 1], self.rel_y)
    down = held & on_table
    self.sd_clear = held & (self.sd_clear | (height > SET_DOWN_CLEARANCE))
    touchdown = down & self.sd_clear
    near = dxy < SET_DOWN_RADIUS
    self.sd_touch_ok = torch.where(touchdown, near, self.sd_touch_ok)
    self.sd_clear = self.sd_clear & ~touchdown
    still = down & self.sd_touch_ok & near & (speed < SET_DOWN_STILL)
    self.sd_dwell = torch.where(
      still, self.sd_dwell + 1, torch.zeros_like(self.sd_dwell)
    )
    self.sd_prev_held = held.clone()

  def record(
    self,
    obj_pos: torch.Tensor,
    obj_yaw: torch.Tensor,
    table_offset: torch.Tensor,
    goal_pos: torch.Tensor,
  ) -> None:
    """Keep each episode's spawn (cube x, y, yaw), table offset from its nominal pose and
    target from its first step, and the cube's position from its latest. Call once per step,
    *before* `observe` (which advances the step count)."""
    first = self.age == 0
    now = torch.cat(
      [obj_pos[:, :2], obj_yaw.unsqueeze(-1), table_offset, goal_pos], dim=-1
    )
    self.first = torch.where(first.unsqueeze(-1), now, self.first)
    self.last_pos = obj_pos.clone()

  def terminal(self, success: torch.Tensor) -> None:
    """Credit the episodes a `success` termination just ended, which `observe` cannot see.

    `success` fires only on a placement, so it sets `placed`; it counts as `placed and
    lifted` if a lift was observed earlier in the same episode -- the same test
    `_placed_bool` applies through the `was_lifted` latch. Call after `step`, before
    `finish`. Does not advance `age`: this is the same step `observe` already counted.
    """
    self.live["placed"] |= success
    self.live["placed_and_lifted"] |= success & self.live["lifted"]
    newly = success & (self.first_place < 0)
    self.first_place = torch.where(newly, self.age - 1, self.first_place)

  def finish(self, env_ids: torch.Tensor) -> None:
    """Bank the episodes that just ended and clear their latches."""
    for i in env_ids.tolist():
      row: dict[str, bool | int | float | None] = {
        n: bool(self.live[n][i]) for n in self.FIELDS
      }
      row["steps_to_place"] = int(self.first_place[i])
      row["length"] = int(self.age[i])
      held, near = int(self.held_n[i]), int(self.near_n[i])
      row["held_steps"] = held
      row["held_dxy_mean"] = float(self.dxy_sum[i]) / held if held else None
      row["held_dxy_min"] = float(self.dxy_min[i]) if held else None
      row["near_xy_while_held"] = near > 0
      row["hover_dz_mean"] = float(self.dz_sum[i]) / near if near else None
      moved = int(self.speed_n[i]) > 0
      row["peak_obj_speed_held"] = float(self.obj_v_peak[i]) if moved else None
      row["peak_tcp_speed_held"] = float(self.tcp_v_peak[i]) if moved else None
      rel = bool(self.released[i])
      row["release_dxy"] = float(self.rel_dxy[i]) if rel else None
      row["release_dz"] = float(self.rel_dz[i]) if rel else None
      row["release_speed"] = float(self.rel_v[i]) if rel else None
      sd = bool(self.sd_released[i])
      row["release_on_table"] = bool(self.rel_on_table[i]) if sd else None
      row["release_dwell"] = int(self.rel_dwell[i]) if sd else None
      row["release_vz"] = float(self.rel_vz[i]) if sd else None
      row["release_vxy"] = float(self.rel_vxy[i]) if sd else None
      row["release_x"] = float(self.rel_x[i]) if sd else None
      row["release_y"] = float(self.rel_y[i]) if sd else None
      passed = (
        sd
        and bool(self.rel_on_table[i])
        and math.hypot(float(self.rel_vz[i]), float(self.rel_vxy[i])) < SET_DOWN_STILL
        and int(self.rel_dwell[i]) >= SET_DOWN_STEPS
      )
      row["release_set_down"] = passed if sd else None
      if rel:
        q6, q7 = release_grades(float(self.rel_dxy[i]), float(self.rel_dz[i]), passed)
        row["release_q_m6"], row["release_q_m7"] = q6, q7
      else:
        row["release_q_m6"] = row["release_q_m7"] = None
      first = self.first[i].tolist()
      keys = ("spawn_x", "spawn_y", "spawn_yaw", "table_dx", "table_dy", "table_dz")
      row.update(dict(zip(keys, first[:6], strict=True)))
      row.update(target_x=first[6], target_y=first[7], target_z=first[8])
      row["carry_m"] = math.hypot(first[6] - first[0], first[7] - first[1])
      rel_x = first[0] - (TABLE_CENTRE_XY[0] + first[3])
      rel_y = first[1] - (TABLE_CENTRE_XY[1] + first[4])
      row["spawn_in_w2"] = (
        W2_REL_X[0] <= rel_x <= W2_REL_X[1] and W2_REL_Y[0] <= rel_y <= W2_REL_Y[1]
      )
      final = self.last_pos[i].tolist()
      row.update(zip(("final_x", "final_y", "final_z"), final, strict=True))
      row["grasp_pad_dz"] = (
        float(self.grasp_dz[i]) if bool(self.grasp_seen[i]) else None
      )
      self.done.append(row)
    for name in self.FIELDS:
      self.live[name][env_ids] = False
    self.first_place[env_ids] = -1
    self.age[env_ids] = 0
    self.held_n[env_ids] = 0
    self.dxy_sum[env_ids] = 0.0
    self.dxy_min[env_ids] = float("inf")
    self.near_n[env_ids] = 0
    self.dz_sum[env_ids] = 0.0
    self.prev_held[env_ids] = False
    self.speed_n[env_ids] = 0
    self.obj_v_peak[env_ids] = 0.0
    self.tcp_v_peak[env_ids] = 0.0
    self.released[env_ids] = False
    self.sd_prev_held[env_ids] = False
    self.sd_clear[env_ids] = False
    self.sd_touch_ok[env_ids] = False
    self.sd_dwell[env_ids] = 0
    self.sd_released[env_ids] = False
    self.grasp_seen[env_ids] = False
    self.grasp_dz[env_ids] = 0.0


class PlantTally:
  """Plan S8's plant reads, per episode, kept beside `Tally` and banked with it.

  * parked hold -- |q - q_default| of the parked right arm and the pelvis, which are commanded
    to their default the whole episode: the episode's max, and its mean over steps and joints;
  * tracking -- the driven arm's |q* - q| while the cube is carried (held and off the table),
    q* being its position actuators' target: the episode's mean over carrying steps and the
    seven joints, and its max over both;
  * saturation -- how many steps each pelvis/arm joint's actuator torque sits at its effort
    limit, over the whole rollout;
  * set-down force -- the peak cube-table normal force while the cube is held within
    `NEAR_XY` of the target: how hard the arm presses the cube into the table;
  * plan S7-4 -- the effort limit a joint's saturation is read against is the one the model
    carries: its ``actuatorfrcrange`` where that is set and lower (T's cap), else its actuator's
    ``forcerange``. Also the driven arm's peak |tau| / limit per joint, its saturation over the
    steps after the episode's first let-go (the press), and the share of its targets outside
    the joint range (what C prevents).

  Tensors in, so the accounting is testable without an env; `from_env` wires one to it.
  """

  def __init__(
    self,
    num_envs: int,
    device: str,
    parked_names: list[str],
    parked_default: torch.Tensor,
    gained_names: list[str],
    effort: torch.Tensor,
  ) -> None:
    self.parked_names = parked_names
    self.parked_default = parked_default
    self.gained_names = gained_names
    self.effort = effort
    zf = lambda: torch.zeros(num_envs, device=device)  # noqa: E731
    self.steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    self.park_max = zf()
    self.park_sum = zf()
    self.carry_n = torch.zeros(num_envs, dtype=torch.long, device=device)
    self.track_sum = zf()
    self.track_max = zf()
    self.setdown_n = torch.zeros(num_envs, dtype=torch.long, device=device)
    self.force_peak = zf()
    self.park_joint_max = torch.zeros(len(parked_names), device=device)
    self.sat_count = torch.zeros(len(gained_names), dtype=torch.long, device=device)
    self.sat_steps = 0
    self.driven_names: list[str] = []
    self.driven_effort: torch.Tensor | None = None
    self.driven_range: torch.Tensor | None = None
    self.peak_ratio: torch.Tensor | None = None
    self.after_sat = self.after_steps = self.out_of_range = self.target_steps = None
    self.done: list[dict[str, float | int | None]] = []

  def set_driven(
    self, names: list[str], effort: torch.Tensor, joint_range: torch.Tensor
  ) -> None:
    """Plan S7-4's driven-arm reads: its joints, effort limits (n,) and ranges (n, 2)."""
    n, device = len(names), effort.device
    self.driven_names = names
    self.driven_effort, self.driven_range = effort, joint_range
    self.peak_ratio = torch.zeros(n, device=device)
    self.after_sat = torch.zeros(n, dtype=torch.long, device=device)
    self.out_of_range = torch.zeros(n, dtype=torch.long, device=device)
    self.after_steps = 0
    self.target_steps = 0

  def measure_driven(
    self, torque: torch.Tensor, target: torch.Tensor, after: torch.Tensor
  ) -> None:
    """One step of the driven arm: torque and target (B, n), ``after`` the envs past their
    episode's first let-go (B,)."""
    assert self.driven_effort is not None and self.driven_range is not None
    ratio = torque.abs() / self.driven_effort
    self.peak_ratio = torch.maximum(self.peak_ratio, ratio.amax(dim=0))
    sat = ratio >= SATURATION_FRACTION
    after = after.bool()
    self.after_sat += (sat & after.unsqueeze(-1)).sum(dim=0)
    self.after_steps += int(after.sum())
    lo, hi = self.driven_range[:, 0], self.driven_range[:, 1]
    self.out_of_range += ((target < lo) | (target > hi)).sum(dim=0)
    self.target_steps += target.shape[0]

  @classmethod
  def from_env(cls, env: ManagerBasedRlEnv) -> PlantTally:
    import re

    import mujoco

    robot = env.scene[ROBOT]
    names = list(robot.joint_names)
    pick = lambda rx: [i for i, n in enumerate(names) if re.fullmatch(rx, n)]  # noqa: E731
    parked, driven, gained = (
      pick(PARKED_JOINTS),
      pick(DRIVEN_ARM_JOINTS),
      pick(GAINED_JOINTS),
    )
    mj = env.sim.mj_model
    actuator = {
      (mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or "").split("/")[-1]: a
      for a in range(mj.nu)
    }
    joint_ids = robot.indexing.joint_ids.tolist()

    def effort(i: int) -> float:
      # The limit that binds: the actuator's forcerange, or the joint's actuatorfrcrange
      # where it is set lower (plan S7-4's T; with G0's routed gravcomp it is the same).
      e = float(mj.actuator_forcerange[actuator[names[i]], 1])
      j = joint_ids[i]
      if mj.jnt_actfrclimited[j]:
        e = min(e, float(mj.jnt_actfrcrange[j, 1]))
      return e

    tally = cls(
      env.num_envs,
      env.device,
      [names[i] for i in parked],
      robot.data.default_joint_pos[:, parked].clone(),
      [names[i] for i in gained],
      torch.tensor([effort(i) for i in gained], device=env.device),
    )
    tally.set_driven(
      [names[i] for i in driven],
      torch.tensor([effort(i) for i in driven], device=env.device),
      torch.tensor(
        [mj.jnt_range[joint_ids[i]].tolist() for i in driven], device=env.device
      ),
    )
    tally._robot = robot
    tally._sim = env.sim
    tally._support = env.scene[SUPPORT_SENSOR]
    tally._idx = (parked, driven, gained)
    tally._driven_ctrl = torch.tensor(
      [actuator[names[i]] for i in driven], device=env.device
    )
    return tally

  def measure_env(
    self,
    carrying: torch.Tensor,
    setdown: torch.Tensor,
    after_let_go: torch.Tensor | None = None,
  ) -> None:
    """`measure` from the env `from_env` was built on, at the current state."""
    parked, driven, gained = self._idx
    q = self._robot.data.joint_pos
    force = self._support.data.force  # (B, slots, 3), contact frame: normal first
    if after_let_go is not None:
      self.measure_driven(
        self._robot.data.qfrc_actuator[:, driven],
        self._sim.data.ctrl[:, self._driven_ctrl],
        after_let_go,
      )
    self.measure(
      parked_q=q[:, parked],
      driven_q=q[:, driven],
      driven_target=self._sim.data.ctrl[:, self._driven_ctrl],
      gained_torque=self._robot.data.qfrc_actuator[:, gained],
      table_force=force[..., 0].abs().amax(dim=-1),
      carrying=carrying,
      setdown=setdown,
    )

  def measure(
    self,
    parked_q: torch.Tensor,
    driven_q: torch.Tensor,
    driven_target: torch.Tensor,
    gained_torque: torch.Tensor,
    table_force: torch.Tensor,
    carrying: torch.Tensor,
    setdown: torch.Tensor,
  ) -> None:
    """One step. Shapes (B, parked), (B, 7), (B, 7), (B, gained), (B,), (B,), (B,)."""
    carrying, setdown = carrying.bool(), setdown.bool()
    park_err = (parked_q - self.parked_default).abs()
    self.park_max = torch.maximum(self.park_max, park_err.amax(dim=-1))
    self.park_sum += park_err.mean(dim=-1)
    self.park_joint_max = torch.maximum(self.park_joint_max, park_err.amax(dim=0))
    self.steps += 1
    track = (driven_target - driven_q).abs()
    self.carry_n += carrying.long()
    self.track_sum += torch.where(
      carrying, track.mean(dim=-1), torch.zeros_like(self.track_sum)
    )
    self.track_max = torch.where(
      carrying, torch.maximum(self.track_max, track.amax(dim=-1)), self.track_max
    )
    self.setdown_n += setdown.long()
    self.force_peak = torch.where(
      setdown, torch.maximum(self.force_peak, table_force), self.force_peak
    )
    self.sat_count += (gained_torque.abs() >= SATURATION_FRACTION * self.effort).sum(
      dim=0
    )
    self.sat_steps += gained_torque.shape[0]

  def finish(self, env_ids: torch.Tensor) -> None:
    """Bank the episodes that just ended, in the order `Tally.finish` banks them."""
    for i in env_ids.tolist():
      steps, carry, setdown = (
        int(self.steps[i]),
        int(self.carry_n[i]),
        int(self.setdown_n[i]),
      )
      self.done.append(
        {
          "parked_hold_max": float(self.park_max[i]) if steps else None,
          "parked_hold_mean": float(self.park_sum[i]) / steps if steps else None,
          "carry_steps": carry,
          "track_carry_mean": float(self.track_sum[i]) / carry if carry else None,
          "track_carry_max": float(self.track_max[i]) if carry else None,
          "setdown_force_peak": float(self.force_peak[i]) if setdown else None,
        }
      )
    for acc in (
      self.park_max,
      self.park_sum,
      self.track_sum,
      self.track_max,
      self.force_peak,
    ):
      acc[env_ids] = 0.0
    for acc in (self.steps, self.carry_n, self.setdown_n):
      acc[env_ids] = 0

  def summarise(self, episodes: int) -> dict:
    rows = self.done[:episodes]
    vals = lambda k: [r[k] for r in rows if r[k] is not None]  # noqa: E731
    mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    median = lambda xs: sorted(xs)[len(xs) // 2] if xs else None  # noqa: E731
    share = (self.sat_count.float() / max(self.sat_steps, 1)).tolist()
    worst = int(torch.argmax(self.park_joint_max))
    return {
      "plant_parked_hold_max_median_rad": median(vals("parked_hold_max")),
      "plant_parked_hold_max_rad": max(vals("parked_hold_max"), default=None),
      "plant_parked_hold_mean_rad": mean(vals("parked_hold_mean")),
      "plant_parked_worst_joint": self.parked_names[worst],
      "plant_parked_max_by_joint_rad": dict(
        zip(self.parked_names, self.park_joint_max.tolist(), strict=True)
      ),
      "plant_track_carry_mean_rad": mean(vals("track_carry_mean")),
      "plant_track_carry_max_median_rad": median(vals("track_carry_max")),
      "plant_saturation_share": dict(zip(self.gained_names, share, strict=True)),
      "plant_saturation_share_max": max(share, default=None),
      "plant_setdown_force_peak_median_n": median(vals("setdown_force_peak")),
      "plant_setdown_force_peak_max_n": max(vals("setdown_force_peak"), default=None),
      **self._summarise_driven(),
    }

  def _summarise_driven(self) -> dict:
    if self.peak_ratio is None or not self.target_steps:
      return {}
    names = self.driven_names
    after = (self.after_sat.float() / max(self.after_steps, 1)).tolist()
    out = (self.out_of_range.float() / self.target_steps).tolist()
    return {
      "plant_driven_effort_limit_nm": dict(
        zip(names, self.driven_effort.tolist(), strict=True)
      ),
      "plant_driven_peak_effort_ratio": dict(
        zip(names, self.peak_ratio.tolist(), strict=True)
      ),
      "plant_driven_saturation_after_let_go": dict(zip(names, after, strict=True)),
      "plant_driven_saturation_after_let_go_max": max(after, default=None),
      "plant_driven_target_out_of_range": dict(zip(names, out, strict=True)),
      "plant_driven_target_out_of_range_max": max(out, default=None),
    }


#: Arm-table force above which a contact counts (N): plan S7-10's AC threshold.
ARM_CONTACT_N = 1.0
#: Phases an arm-table contact is filed under (plan S7-10 step 0).
ARM_PHASES = ("before_grasp", "held", "after_let_go", "other")


class ArmTally:
  """Plan S7-10's reads, per episode, kept beside `Tally` and banked with it. Tensors in, like
  `PlantTally`; `from_env` wires one to an env. ``force`` is (B, L) per link (None: no sensor),
  ``clearance`` (B, L) per link, m (+inf: not over the table)."""

  def __init__(self, num_envs: int, device: str, links: list[str]) -> None:
    self.links = links
    n = len(links)
    self.contact_steps = torch.zeros(num_envs, n, dtype=torch.long, device=device)
    self.phase_steps = torch.zeros(
      num_envs, len(ARM_PHASES), dtype=torch.long, device=device
    )
    self.peak = torch.zeros(num_envs, device=device)
    self.min_clear = torch.full((num_envs, n), float("inf"), device=device)
    self.ever_grasped = torch.zeros(num_envs, dtype=torch.bool, device=device)
    self.has_force = False
    self.done: list[dict] = []

  @classmethod
  def from_env(cls, env: ManagerBasedRlEnv) -> ArmTally:
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from pal_mjlab.tasks.manipulation.pick_place.arm_table import (
      ARM_TABLE_SENSOR,
      arm_body_regex,
    )

    gripper = KANGAROO_GRIPPERS_LEFT_ARM.gripper("left")
    asset = SceneEntityCfg(ROBOT, body_names=(arm_body_regex(gripper),))
    asset.resolve(env.scene)
    robot = env.scene[ROBOT]
    links = [robot.body_names[i] for i in asset.body_ids]
    tally = cls(env.num_envs, env.device, links)
    tally._env, tally._asset = env, asset
    tally._sensor = None
    try:
      tally._sensor = env.scene[ARM_TABLE_SENSOR]
    except KeyError:
      pass
    if tally._sensor is not None:
      names = [n.split("/")[-1] for n in tally._sensor.primary_names]
      if names != links:  # slot i must be link i for the per-link reads
        raise RuntimeError(f"arm-table sensor primaries {names} are not {links}")
      tally.has_force = True
    return tally

  def step_env(self, grasped: torch.Tensor, let_go: torch.Tensor) -> None:
    """`step` from the env `from_env` was built on, at the current state."""
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from pal_mjlab.tasks.manipulation import mdp
    from pal_mjlab.tasks.manipulation.tabletop_env_cfg import TABLE

    force = None
    if self._sensor is not None:
      hist = self._sensor.data.force_history  # (B, L, H, 3)
      force = torch.norm(hist, dim=-1).amax(dim=-1)
    clearance = mdp.arm_table_clearance(self._env, self._asset, SceneEntityCfg(TABLE))
    self.step(force, clearance, grasped, let_go)

  def step(
    self,
    force: torch.Tensor | None,
    clearance: torch.Tensor,
    grasped: torch.Tensor,
    let_go: torch.Tensor,
  ) -> None:
    grasped, let_go = grasped.bool(), let_go.bool()
    self.ever_grasped |= grasped
    self.min_clear = torch.minimum(self.min_clear, clearance)
    if force is None:
      return
    touch = force > ARM_CONTACT_N  # (B, L)
    self.contact_steps += touch.long()
    self.peak = torch.maximum(self.peak, force.amax(dim=-1))
    any_touch = touch.any(dim=-1)
    phase = torch.full_like(self.ever_grasped, 3, dtype=torch.long)  # other
    phase = torch.where(grasped, torch.ones_like(phase), phase)
    phase = torch.where(let_go & ~grasped, torch.full_like(phase, 2), phase)
    phase = torch.where(~self.ever_grasped, torch.zeros_like(phase), phase)
    self.phase_steps += torch.nn.functional.one_hot(
      phase, len(ARM_PHASES)
    ) * any_touch.unsqueeze(-1)

  def finish(self, env_ids: torch.Tensor) -> None:
    for i in env_ids.tolist():
      clear = self.min_clear[i].tolist()
      row: dict = {
        "arm_min_clearance_m": min(clear),
        "arm_min_clearance_by_link_m": dict(zip(self.links, clear, strict=True)),
      }
      if self.has_force:
        steps = self.contact_steps[i].tolist()
        row.update(
          {
            "arm_table_contact": any(steps),
            "arm_table_contact_steps": max(steps) if steps else 0,
            "arm_table_peak_force_n": float(self.peak[i]),
            "arm_table_contact_by_link": dict(zip(self.links, steps, strict=True)),
            "arm_table_contact_by_phase": dict(
              zip(ARM_PHASES, self.phase_steps[i].tolist(), strict=True)
            ),
          }
        )
      self.done.append(row)
    self.contact_steps[env_ids] = 0
    self.phase_steps[env_ids] = 0
    self.peak[env_ids] = 0.0
    self.min_clear[env_ids] = float("inf")
    self.ever_grasped[env_ids] = False

  def summarise(self, episodes: int) -> dict:
    rows = self.done[:episodes]
    if not rows:
      return {}
    n = len(rows)
    median = lambda xs: sorted(xs)[len(xs) // 2] if xs else None  # noqa: E731
    finite = [
      r["arm_min_clearance_m"] for r in rows if math.isfinite(r["arm_min_clearance_m"])
    ]
    out: dict = {
      "arm_min_clearance_median_m": median(finite),
      "arm_min_clearance_min_m": min(finite, default=None),
      "arm_min_clearance_median_by_link_m": {
        link: median(
          [
            r["arm_min_clearance_by_link_m"][link]
            for r in rows
            if math.isfinite(r["arm_min_clearance_by_link_m"][link])
          ]
        )
        for link in self.links
      },
      **{
        f"arm_within_{cm}cm_share": sum(
          r["arm_min_clearance_m"] < cm / 100 for r in rows
        )
        / n
        for cm in (1, 2, 3)
      },
    }
    if self.has_force:
      touched = [r for r in rows if r["arm_table_contact"]]
      out.update(
        {
          "arm_table_contact_share": len(touched) / n,
          "arm_table_contact_share_by_link": {
            link: sum(r["arm_table_contact_by_link"][link] > 0 for r in rows) / n
            for link in self.links
          },
          "arm_table_contact_steps_by_phase": {
            p: sum(r["arm_table_contact_by_phase"][p] for r in rows) for p in ARM_PHASES
          },
          "arm_table_peak_force_median_n": median(
            [r["arm_table_peak_force_n"] for r in touched]
          ),
          "arm_table_peak_force_max_n": max(
            (r["arm_table_peak_force_n"] for r in rows), default=None
          ),
          "arm_table_over_30n_share": sum(
            r["arm_table_peak_force_n"] > 30.0 for r in rows
          )
          / n,
        }
      )
    return out


class RetreatTally:
  """Plan S7-4's reads, per episode, kept beside `Tally` and banked with it: the jaw release
  (module J), the retreat and the hand-off to the position controller. Tensors in, like
  `PlantTally`; call `step` once per control step, at the top of the step, and apply the
  hand-off mask it returns to that step's action."""

  def __init__(
    self,
    num_envs: int,
    device: str,
    step_dt: float,
    handoff: bool = False,
    timeout_s: float = HANDOFF_TIMEOUT_S,
  ) -> None:
    self.step_dt = step_dt
    self.handoff = handoff
    self.timeout_steps = round(timeout_s / step_dt)
    zb = lambda: torch.zeros(num_envs, dtype=torch.bool, device=device)  # noqa: E731
    zl = lambda v=0: torch.full((num_envs,), v, dtype=torch.long, device=device)  # noqa: E731
    zf = lambda v=0.0: torch.full((num_envs,), v, device=device)  # noqa: E731
    self.age = zl()
    self.prev_grasped, self.prev_jaw_closed, self.lifted = zb(), zb(), zb()
    self.let_go, self.jaw_at_let_go = zb(), zf()
    self.released, self.rel_step = zb(), zl(-1)
    self.rel_xy = torch.zeros(num_envs, 2, device=device)
    self.placed_j, self.placed_j_step = zb(), zl(-1)
    self.placed_jz, self.final_placed_jz, self.tilt_at_let_go = zb(), zb(), zf()
    self.yaw_at_let_go = zf()
    self.clear_max = zf(float("-inf"))
    self.clear_run = zl()
    self.handed, self.hand_step, self.hand_by = zb(), zl(-1), zl()
    self.hand_xy = torch.zeros(num_envs, 2, device=device)
    self.hand_height = zf()
    self.home_step = zl(-1)
    self.final_placed_j, self.final_dxy = zb(), zf()
    self.final_xy = torch.zeros(num_envs, 2, device=device)
    self.done: list[dict[str, float | int | bool | str | None]] = []

  def step(
    self,
    jaw_q: torch.Tensor,
    grasped: torch.Tensor,
    lifted: torch.Tensor,
    placed_j: torch.Tensor,
    tcp_above: torch.Tensor,
    obj_xy: torch.Tensor,
    goal_dxy: torch.Tensor,
    arm_home_err: torch.Tensor,
    placed_jz: torch.Tensor | None = None,
    tilt: torch.Tensor | None = None,
    jaw_open_below: float | torch.Tensor = JAW_OPEN,
    yaw: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """One step's state -> the envs the position controller drives this step. (B,)

    ``tcp_above`` is the TCP's height above the cube's top, ``goal_dxy`` the cube's horizontal
    distance to the target, ``arm_home_err`` the driven arm's max |q - q_default|;
    ``placed_jz`` is ``placed_j`` under plan S7-5's Z tolerance, ``tilt`` the cube's tilt
    (rad), both optional; ``jaw_open_below`` J's threshold (`jaw_open_for`, per env for a mix of
    objects)."""
    grasped, lifted, placed_j = grasped.bool(), lifted.bool(), placed_j.bool()
    placed_jz = torch.zeros_like(placed_j) if placed_jz is None else placed_jz.bool()
    jaw_open = jaw_q < jaw_open_below
    self.lifted |= lifted
    let_go = self.prev_grasped & ~grasped & self.lifted
    self.jaw_at_let_go = torch.where(let_go, jaw_q, self.jaw_at_let_go)
    if tilt is not None:
      self.tilt_at_let_go = torch.where(let_go, tilt, self.tilt_at_let_go)
    if yaw is not None:
      self.yaw_at_let_go = torch.where(let_go, yaw, self.yaw_at_let_go)
    self.let_go |= let_go
    # J's release: entering "jaw open and the pads off the cube" after the lift -- for a cube
    # held across an edge (the pads touch at ~0.10 rad) that is the let-go itself.
    released_now = jaw_open & ~grasped
    # Only the policy's releases: once the position controller has taken over it opens the
    # jaw itself (a timeout hand-off of a policy still holding the cube is not a release).
    opened = (
      self.prev_jaw_closed & released_now & self.lifted & ~self.released & ~self.handed
    )
    self.released |= opened
    self.rel_step = torch.where(opened, self.age, self.rel_step)
    self.rel_xy = torch.where(opened.unsqueeze(-1), obj_xy, self.rel_xy)
    newly = placed_j & (self.placed_j_step < 0)
    self.placed_j_step = torch.where(newly, self.age, self.placed_j_step)
    self.placed_j |= placed_j
    self.placed_jz |= placed_jz
    # The lift: after the J-release and before the hand-off, how high the TCP gets.
    lifting = self.released & ~self.handed
    self.clear_max = torch.where(
      lifting, torch.maximum(self.clear_max, tcp_above), self.clear_max
    )
    clear = lifting & jaw_open & (tcp_above >= HANDOFF_CLEAR)
    self.clear_run = torch.where(
      clear, self.clear_run + 1, torch.zeros_like(self.clear_run)
    )
    if self.handoff:
      by_lift = ~self.handed & (self.clear_run >= HANDOFF_STEPS)
      by_time = ~self.handed & ~by_lift & (self.age >= self.timeout_steps)
      start = by_lift | by_time
      self.hand_by = torch.where(by_lift, 1, torch.where(by_time, 2, self.hand_by))
      self.hand_step = torch.where(start, self.age, self.hand_step)
      self.hand_xy = torch.where(start.unsqueeze(-1), obj_xy, self.hand_xy)
      self.hand_height = torch.where(start, tcp_above, self.hand_height)
      self.handed |= start
      home = self.handed & (self.home_step < 0) & (arm_home_err < HOME_TOL)
      self.home_step = torch.where(home, self.age, self.home_step)
    self.final_placed_j = placed_j
    self.final_placed_jz = placed_jz
    self.final_dxy = goal_dxy
    self.final_xy = obj_xy.clone()
    self.prev_grasped = grasped.clone()
    self.prev_jaw_closed = ~released_now
    self.age += 1
    return self.handed.clone()

  def finish(self, env_ids: torch.Tensor, placed_and_lifted: list[bool]) -> None:
    """Bank the episodes that just ended, in `Tally.finish`'s order; ``placed_and_lifted`` is
    `Tally`'s flag for each of them (the J version is this tally's own)."""
    dt = self.step_dt
    for k, i in enumerate(env_ids.tolist()):
      rel, hand = bool(self.released[i]), int(self.hand_step[i])
      handed = hand >= 0
      clear = rel and float(self.clear_max[i]) >= HANDOFF_CLEAR
      row = {
        "jaw_released": rel,
        "jaw_at_let_go": float(self.jaw_at_let_go[i]) if bool(self.let_go[i]) else None,
        "tilt_at_let_go": float(self.tilt_at_let_go[i])
        if bool(self.let_go[i])
        else None,
        "yaw_at_let_go": float(self.yaw_at_let_go[i]) if bool(self.let_go[i]) else None,
        "placed_j": bool(self.placed_j[i]),
        "placed_and_lifted_j": bool(self.placed_j[i]) and bool(self.lifted[i]),
        "lift_clear_max": float(self.clear_max[i]) if rel else None,
        "lifted_clear": clear,
        "handoff_by": ("lift", "timeout")[int(self.hand_by[i]) - 1] if handed else None,
        "handoff_height": float(self.hand_height[i]) if handed else None,
        "placed_to_handoff_s": (hand - int(self.placed_j_step[i])) * dt
        if handed and int(self.placed_j_step[i]) >= 0
        else None,
        "cube_moved_lift": float(torch.norm(self.hand_xy[i] - self.rel_xy[i]))
        if handed and rel
        else None,
        "cube_moved_return": float(torch.norm(self.final_xy[i] - self.hand_xy[i]))
        if handed
        else None,
        "final_placed_j": bool(self.final_placed_j[i]),
        "placed_and_lifted_j_z": bool(self.placed_jz[i]) and bool(self.lifted[i]),
        "final_placed_j_z": bool(self.final_placed_jz[i]),
        "final_dxy": float(self.final_dxy[i]),
        "time_to_home_s": (int(self.home_step[i]) - hand) * dt
        if handed and int(self.home_step[i]) >= 0
        else None,
      }
      row["success_retreat"] = (
        bool(placed_and_lifted[k])
        and rel
        and clear
        and bool(self.final_placed_j[i])
        and bool(self.placed_j[i])
      )
      row["success_retreat_z"] = (
        rel and clear and bool(self.final_placed_jz[i]) and bool(self.placed_jz[i])
      )
      self.done.append(row)
    for acc in (
      self.prev_grasped,
      self.prev_jaw_closed,
      self.lifted,
      self.let_go,
      self.released,
      self.placed_j,
      self.placed_jz,
      self.handed,
    ):
      acc[env_ids] = False
    for acc in (self.age, self.clear_run, self.hand_by):
      acc[env_ids] = 0
    for acc in (self.rel_step, self.placed_j_step, self.hand_step, self.home_step):
      acc[env_ids] = -1
    self.clear_max[env_ids] = float("-inf")
    self.tilt_at_let_go[env_ids] = 0.0

  def summarise(self, episodes: int) -> dict:
    rows = self.done[:episodes]
    n = max(len(rows), 1)
    vals = lambda k: [r[k] for r in rows if r[k] is not None]  # noqa: E731
    median = lambda xs: sorted(xs)[len(xs) // 2] if xs else None  # noqa: E731
    share = lambda k: sum(bool(r[k]) for r in rows) / n  # noqa: E731
    by = [r["handoff_by"] for r in rows]
    return {
      "jaw_released": share("jaw_released"),
      "jaw_at_let_go_median_rad": median(vals("jaw_at_let_go")),
      "placed_j": share("placed_j"),
      "placed_and_lifted_j": share("placed_and_lifted_j"),
      "lifted_clear": share("lifted_clear"),
      "lift_clear_max_median_m": median(vals("lift_clear_max")),
      "handoff_by_lift": by.count("lift") / n,
      "handoff_by_timeout": by.count("timeout") / n,
      "handoff_height_median_m": median(vals("handoff_height")),
      "placed_to_handoff_median_s": median(vals("placed_to_handoff_s")),
      "cube_moved_lift_median_m": median(vals("cube_moved_lift")),
      "cube_moved_lift_max_m": max(vals("cube_moved_lift"), default=None),
      "cube_moved_return_median_m": median(vals("cube_moved_return")),
      "cube_moved_return_max_m": max(vals("cube_moved_return"), default=None),
      "final_placed_j": share("final_placed_j"),
      "final_within_goal": sum(r["final_dxy"] < GOAL_THRESHOLD for r in rows) / n,
      "time_to_home_median_s": median(vals("time_to_home_s")),
      "success_retreat": share("success_retreat"),
      "tilt_at_let_go_median_rad": median(vals("tilt_at_let_go")),
      "tilt_at_let_go_over_10deg": sum(
        t > math.radians(10) for t in vals("tilt_at_let_go")
      )
      / max(len(vals("tilt_at_let_go")), 1),
      "placed_and_lifted_j_z": share("placed_and_lifted_j_z"),
      "final_placed_j_z": share("final_placed_j_z"),
      "success_retreat_z": share("success_retreat_z"),
    }


class VerticalRetreat:
  """The hand-off's first leg (S7-5 report §8 item 1): straight up before the joint-space return.

  The hand-off's home pose is reached by interpolating the joints, and at 3-7 cm over the cube
  the fingers sweep back through it (40 of 45 post-place misses at S7-5's 5 k). Here, once
  `RetreatTally` hands the arm over, the TCP first rises to ``height`` (m) over the cube's
  top on a Cartesian line -- the hand keeps its orientation, damped least squares on the env's
  own kinematics (`scripted_pick_place.Kin.ik_pose`), started from the arm's last commanded
  target -- and only then does the action go to 0 (home, jaw open) as before. Every target goes
  through the action term, so its slew, T's band and C's clip apply. An IK that fails skips the
  lift for that episode (counted in ``ik_failed``)."""

  def __init__(self, env: ManagerBasedRlEnv, height: float) -> None:
    import importlib.util
    import sys

    # Registered before it runs: its `Kin` is a dataclass, which looks its module up by name.
    sp = sys.modules.get("scripted_pick_place")
    if sp is None:
      spec = importlib.util.spec_from_file_location(
        "scripted_pick_place", pathlib.Path(__file__).parent / "scripted_pick_place.py"
      )
      assert spec and spec.loader
      sp = importlib.util.module_from_spec(spec)
      sys.modules[spec.name] = sp
      spec.loader.exec_module(sp)
    self.height = height
    self.kin = sp.Kin.from_model(env.sim.mj_model)
    self.term = env.action_manager.get_term("joint_pos")
    names = [n.split("/", 1)[-1] for n in self.term.target_names]
    self.arm_idx = [names.index(j) for j in sp.ARM_JOINTS]
    rate = getattr(self.term, "_rate", None)
    self.slew = (
      rate[0, self.arm_idx].cpu().numpy()
      if rate is not None
      else np.full(len(self.arm_idx), np.inf)
    )
    self.step_dt = env.step_dt
    self.grace = round(LIFT_GRACE_S / env.step_dt)
    n = env.num_envs
    device = env.device
    self.device = device
    self.planned_at = torch.full((n,), -2, dtype=torch.long, device=device)
    self.plan_len = torch.zeros(n, dtype=torch.long, device=device)
    self.plan = torch.zeros(n, 1, self.term.action_dim, device=device)
    self.home = torch.zeros(n, dtype=torch.bool, device=device)
    self.planned = 0
    self.ik_failed = 0
    self.home_heights: list[float] = []

  def _path(self, q0: np.ndarray, dz: float) -> np.ndarray | None:
    """Joint targets, one per control step, that lift the TCP by ``dz`` from ``q0`` with the hand
    keeping its orientation. None if the IK misses."""
    kin = self.kin
    pos0, rot0 = kin.fk(q0)
    n_way = max(1, math.ceil(dz / LIFT_WAYPOINT - 1e-9))
    # No faster than LIFT_SPEED, whatever the slew allows.
    per_way_min = math.ceil(dz / n_way / (LIFT_SPEED * self.step_dt) - 1e-9)
    q, rows = q0, []
    for i in range(1, n_way + 1):
      sol = kin.ik_pose(pos0 + np.array([0.0, 0.0, dz * i / n_way]), rot0, q, iters=100)
      if sol is None:
        return None
      need = int(np.ceil((np.abs(sol - q) / self.slew).max() - 1e-9))
      n = max(1, per_way_min, need)
      rows.extend(q + (sol - q) * k / n for k in range(1, n + 1))
      q = sol
    return np.asarray(rows)

  def _per_joint(self, x: torch.Tensor | float, e: int) -> torch.Tensor | float:
    return x[e, self.arm_idx] if isinstance(x, torch.Tensor) else x

  def act(
    self,
    handed: torch.Tensor,
    hand_step: torch.Tensor,
    age: torch.Tensor,
    tcp_above: torch.Tensor,
    actions: torch.Tensor,
  ) -> torch.Tensor:
    """This step's actions with the handed-off envs' replaced: the lift's targets, then 0.

    ``hand_step`` / ``age`` are `RetreatTally`'s, read right after its `step`; ``tcp_above``
    is the TCP's measured height over the cube's top."""
    term = self.term
    n = len(handed)
    fresh = handed & (self.planned_at != hand_step)
    for e in torch.nonzero(fresh, as_tuple=False).flatten().tolist():
      self.planned_at[e] = hand_step[e]
      self.planned += 1
      dz = self.height - float(tcp_above[e])
      bias = term._entity.data.encoder_bias[e, term.target_ids][self.arm_idx]
      # The last target the term commanded, in the plant's frame (it subtracts the bias).
      q0 = (term._processed_actions[e, self.arm_idx] - bias).cpu().numpy().astype(float)
      path = self._path(q0, dz) if dz > 1e-3 else None
      if dz > 1e-3 and path is None:
        self.ik_failed += 1
      if path is None:
        self.plan_len[e] = 0
        continue
      target = torch.as_tensor(path, dtype=torch.float32, device=self.device) + bias
      act = (target - self._per_joint(term.offset, e)) / self._per_joint(term.scale, e)
      if act.shape[0] > self.plan.shape[1]:
        grow = act.shape[0] - self.plan.shape[1]
        self.plan = torch.cat(
          [self.plan, self.plan.new_zeros(n, grow, self.plan.shape[2])], 1
        )
      self.plan[e] = 0.0
      block = self.plan[e, : act.shape[0]]  # a view, (T, A)
      block[:, self.arm_idx] = act
      self.plan_len[e] = act.shape[0]
    k = (age - 1 - hand_step).clamp(min=0)
    # Home once the path has run out and the TCP is up there (or the grace has passed); at
    # once when there is no path (already high enough, or the IK missed).
    ready = (k >= self.plan_len) & (
      (tcp_above >= self.height - LIFT_TOL) | (k >= self.plan_len + self.grace)
    )
    newly = handed & ~self.home & ((self.plan_len == 0) | ready)
    self.home_heights.extend(tcp_above[newly].tolist())
    self.home = (self.home | newly) & handed
    self.planned_at = torch.where(handed, self.planned_at, -2)
    rows = self.plan[
      torch.arange(n, device=self.device),
      torch.minimum(k, (self.plan_len - 1).clamp(min=0)),
    ]
    zero = torch.where(handed.unsqueeze(-1), torch.zeros_like(actions), actions)
    return torch.where((handed & ~self.home).unsqueeze(-1), rows, zero)

  def summarise(self) -> dict:
    h = sorted(self.home_heights)
    return {
      "handoff_lift_m": self.height,
      "handoff_lift_planned": self.planned,
      "handoff_lift_ik_failed": self.ik_failed,
      "handoff_lift_home_height_median_m": h[len(h) // 2] if h else None,
    }


class OccTally:
  """Plan S7-7's reads, per episode, from the run's `mdp.CubeSensor` (there only when the env has
  an occlusion config; ``OCC`` / ``OBJ_OBS`` name it): how much of the episode the cube is not
  reported, by phase (approach / grasp / carry / retreat), how far O-K's estimate is from the live
  cube by the steps since the cube was last seen, how the jaw-only "held" rule compares with the
  contact sensor (precision, recall, the steps it trails the real grasp by), and the re-approaches:
  grasp attempts (the pads touching the cube again) and the jaw reopening away from the goal after
  a lift. Tensors in; call `step` once per control step at the top of the step, after the sensor
  has been updated (it is, by the observation the policy is about to read)."""

  PHASES = ("approach", "grasp", "carry", "retreat")
  #: Steps-since-seen bins for the estimate's error, [lo, hi).
  BINS = ((0, 1), (1, 6), (6, 11), (11, 26), (26, 51), (51, 10**6))

  def __init__(self, num_envs: int, device: str) -> None:
    n = num_envs
    zf = lambda *shape: torch.zeros(*shape, device=device)  # noqa: E731
    zl = lambda *shape: torch.zeros(*shape, dtype=torch.long, device=device)  # noqa: E731
    zb = lambda: torch.zeros(n, dtype=torch.bool, device=device)  # noqa: E731
    self.steps = zl(n)
    self.phase_steps = zl(n, 4)
    self.lost_steps = zl(n, 4)  # reported lost (what the actor is told)
    self.blocked_steps = zl(n, 4)  # the live line of sight / random loss says hidden
    self.ever_grasped = zb()
    self.prev_grasped = zb()
    self.prev_jaw_closed = zb()
    self.was_lifted = zb()
    self.grasp_starts = zl(n)
    self.closings = zl(n)  # closing strokes of the jaw (open -> closed, hysteresis)
    self.jaw_closed = zb()
    self.reopen_away = zl(n)
    self.tp, self.fp, self.fn, self.tn = zl(n), zl(n), zl(n), zl(n)
    self.grasp_step = zl(n) - 1  # the first step the pads hold the cube
    self.held_step = zl(n) - 1  # ... the first step the rule says held
    self.err_sum = zf(len(self.BINS))
    self.err_max = zf(len(self.BINS))
    self.err_n = zl(len(self.BINS))
    self.err_ep_max = zf(n)
    self.done: list[dict] = []

  def step(
    self,
    sensor,
    grasped: torch.Tensor,
    lifted: torch.Tensor,
    obj_pos: torch.Tensor,
    jaw_q: torch.Tensor,
    dxy_goal: torch.Tensor,
  ) -> None:
    lost = ~sensor.seen
    hidden = ~sensor.live_visible
    self.ever_grasped |= grasped
    phase = torch.where(
      ~self.ever_grasped,
      0,
      torch.where(grasped & lifted, 2, torch.where(grasped, 1, 3)),
    )
    onehot = torch.nn.functional.one_hot(phase, 4)
    self.steps += 1
    self.phase_steps += onehot
    self.lost_steps += onehot * lost.unsqueeze(-1)
    self.blocked_steps += onehot * hidden.unsqueeze(-1)
    held = sensor.held
    self.tp += held & grasped
    self.fp += held & ~grasped
    self.fn += ~held & grasped
    self.tn += ~held & ~grasped
    first_g = grasped & (self.grasp_step < 0)
    self.grasp_step = torch.where(first_g, self.steps, self.grasp_step)
    first_h = held & (self.held_step < 0)
    self.held_step = torch.where(first_h, self.steps, self.held_step)
    self.grasp_starts += grasped & ~self.prev_grasped
    self.was_lifted |= lifted
    closed = jaw_q >= JAW_OPEN
    # A closing stroke: the measured jaw goes from open (< JAW_OPEN) to closed (> JAW_CLOSED).
    started = ~self.jaw_closed & (jaw_q > JAW_CLOSED)
    self.closings += started
    self.jaw_closed = (self.jaw_closed | started) & (jaw_q >= JAW_OPEN)
    self.reopen_away += (
      self.was_lifted & self.prev_jaw_closed & ~closed & (dxy_goal > 0.03)
    )
    self.prev_grasped = grasped.clone()
    self.prev_jaw_closed = closed
    err = torch.norm(sensor.est_pos - obj_pos, dim=-1)
    unseen = sensor.unseen_steps
    for b, (lo, hi) in enumerate(self.BINS):
      m = (unseen >= lo) & (unseen < hi)
      if bool(m.any()):
        self.err_sum[b] += err[m].sum()
        self.err_max[b] = torch.maximum(self.err_max[b], err[m].max())
        self.err_n[b] += int(m.sum())
    self.err_ep_max = torch.maximum(
      self.err_ep_max, torch.where(lost, err, torch.zeros_like(err))
    )

  def finish(self, env_ids: torch.Tensor) -> None:
    for i in env_ids.tolist():
      n = max(int(self.steps[i]), 1)
      row = {
        "occ_steps": int(self.steps[i]),
        "occ_lost_share": float(self.lost_steps[i].sum()) / n,
        "occ_blocked_share": float(self.blocked_steps[i].sum()) / n,
        "occ_lost_steps_by_phase": self.lost_steps[i].tolist(),
        "occ_phase_steps": self.phase_steps[i].tolist(),
        "occ_grasp_starts": int(self.grasp_starts[i]),
        "occ_closings": int(self.closings[i]),
        "occ_reopen_away": int(self.reopen_away[i]),
        "occ_held_tp_fp_fn_tn": [
          int(x[i]) for x in (self.tp, self.fp, self.fn, self.tn)
        ],
        "occ_held_lag_steps": (
          int(self.held_step[i] - self.grasp_step[i])
          if self.held_step[i] >= 0 and self.grasp_step[i] >= 0
          else None
        ),
        "occ_est_err_max_while_lost": float(self.err_ep_max[i]),
      }
      self.done.append(row)
      for acc in (
        self.steps,
        self.grasp_starts,
        self.closings,
        self.reopen_away,
        self.tp,
        self.fp,
        self.fn,
        self.tn,
      ):
        acc[i] = 0
      for acc in (self.phase_steps, self.lost_steps, self.blocked_steps):
        acc[i] = 0
      self.grasp_step[i] = -1
      self.held_step[i] = -1
      self.err_ep_max[i] = 0.0
      self.ever_grasped[i] = False
      self.prev_grasped[i] = False
      self.prev_jaw_closed[i] = False
      self.was_lifted[i] = False
      self.jaw_closed[i] = False

  def summarise(self, episodes: int) -> dict:
    rows = self.done[:episodes]
    n = max(len(rows), 1)
    ph = (
      torch.tensor([r["occ_phase_steps"] for r in rows]).sum(0)
      if rows
      else torch.zeros(4)
    )
    lost = (
      torch.tensor([r["occ_lost_steps_by_phase"] for r in rows]).sum(0)
      if rows
      else torch.zeros(4)
    )
    tp, fp, fn, tn = (sum(r["occ_held_tp_fp_fn_tn"][k] for r in rows) for k in range(4))
    lags = [
      r["occ_held_lag_steps"] for r in rows if r["occ_held_lag_steps"] is not None
    ]
    return {
      "occ_lost_share": sum(r["occ_lost_share"] for r in rows) / n,
      "occ_blocked_share": sum(r["occ_blocked_share"] for r in rows) / n,
      "occ_lost_share_by_phase": {
        k: (float(lost[i]) / float(ph[i]) if float(ph[i]) else None)
        for i, k in enumerate(self.PHASES)
      },
      "occ_phase_share": {
        k: float(ph[i]) / max(float(ph.sum()), 1.0) for i, k in enumerate(self.PHASES)
      },
      "occ_grasp_starts_mean": sum(r["occ_grasp_starts"] for r in rows) / n,
      "occ_closings_mean": sum(r["occ_closings"] for r in rows) / n,
      "occ_reapproach": sum(r["occ_closings"] > 1 for r in rows) / n,
      "occ_reopen_away": sum(r["occ_reopen_away"] > 0 for r in rows) / n,
      "occ_held_precision": tp / (tp + fp) if tp + fp else None,
      "occ_held_recall": tp / (tp + fn) if tp + fn else None,
      "occ_held_counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
      "occ_held_lag_median_steps": sorted(lags)[len(lags) // 2] if lags else None,
      "occ_est_err_by_steps_unseen": {
        f"{lo}-{hi - 1 if hi < 10**6 else 'inf'}": (
          {
            "mean_m": float(self.err_sum[b] / self.err_n[b]),
            "max_m": float(self.err_max[b]),
            "n": int(self.err_n[b]),
          }
          if int(self.err_n[b])
          else None
        )
        for b, (lo, hi) in enumerate(self.BINS)
      },
    }


class StackTally:
  """Plan S9's reads on a stack goal (`GOAL=stack`), per episode.

  At every let-go of the pads after `was_lifted`: the five quantities of the release condition
  (`mdp.stack`: xy offset to the goal on the base, height above it, tilt, yaw against the base,
  speed) and whether ``OK`` held; the last let-go's are reported, plus whether any let-go was
  ``OK`` and how many there were. The lock's steps (`GATE=stack`): armed (held, lifted, not OK)
  and refused (the policy asked to open while armed) -- the stall read. And the base cube's
  largest horizontal displacement from where it spawned.
  """

  NAMES = ("dxy", "height", "tilt", "yaw", "speed")

  def __init__(self, env: ManagerBasedRlEnv) -> None:
    from pal_mjlab.tasks.manipulation.mdp.stack import stack_command_reference

    n, device = env.num_envs, env.device
    self.reference = stack_command_reference(env, COMMAND)
    zf = lambda: torch.zeros(n, device=device)  # noqa: E731
    zl = lambda: torch.zeros(n, dtype=torch.long, device=device)  # noqa: E731
    zb = lambda: torch.zeros(n, dtype=torch.bool, device=device)  # noqa: E731
    self.prev = zb()
    self.rel = {k: zf() for k in self.NAMES}
    self.rel_ok, self.any_ok, self.n_let_go = zb(), zb(), zl()
    self.armed, self.refused = zl(), zl()
    self.fresh = torch.ones(n, dtype=torch.bool, device=device)
    self.base0 = torch.zeros(n, 2, device=device)
    self.base_moved = zf()
    self.done: list[dict] = []

  @property
  def active(self) -> bool:
    return self.reference is not None

  def step(self, env: ManagerBasedRlEnv, grasped: torch.Tensor) -> None:
    from pal_mjlab.tasks.manipulation.mdp.observations import _WAS_LIFTED
    from pal_mjlab.tasks.manipulation.mdp.stack import stack_state

    lifted = getattr(env, _WAS_LIFTED, None)
    lifted = torch.zeros_like(grasped) if lifted is None else lifted
    let_go = self.prev & ~grasped & lifted
    self.prev = grasped.clone()
    state = stack_state(env, OBJECT, self.reference, COMMAND)
    for k in self.NAMES:
      self.rel[k] = torch.where(let_go, getattr(state, k), self.rel[k])
    self.rel_ok = torch.where(let_go, state.ok, self.rel_ok)
    self.any_ok |= let_go & state.ok
    self.n_let_go += let_go.long()
    term = env.action_manager.get_term("joint_pos")
    if hasattr(term, "armed"):
      self.armed += term.armed.long()
      self.refused += term.locked.long()
    base_xy = env.scene[self.reference].data.root_link_pos_w[:, :2]
    self.base0 = torch.where(self.fresh.unsqueeze(-1), base_xy, self.base0)
    self.fresh[:] = False
    self.base_moved = torch.maximum(
      self.base_moved, torch.norm(base_xy - self.base0, dim=-1)
    )

  def finish(self, env_ids: torch.Tensor) -> None:
    for i in env_ids.tolist():
      seen = int(self.n_let_go[i]) > 0
      row = {
        f"stack_release_{k}": float(self.rel[k][i]) if seen else None
        for k in self.NAMES
      }
      row["stack_release_ok"] = bool(self.rel_ok[i]) if seen else None
      row["stack_any_ok"] = bool(self.any_ok[i])
      row["stack_let_gos"] = int(self.n_let_go[i])
      row["stack_lock_armed_steps"] = int(self.armed[i])
      row["stack_lock_refused_steps"] = int(self.refused[i])
      row["stack_base_moved_m"] = float(self.base_moved[i])
      self.done.append(row)
    for t in (self.prev, self.rel_ok, self.any_ok):
      t[env_ids] = False
    self.n_let_go[env_ids] = 0
    self.armed[env_ids] = 0
    self.refused[env_ids] = 0
    self.base_moved[env_ids] = 0.0
    self.fresh[env_ids] = True

  def summarise(self, episodes: int) -> dict:
    rows = self.done[:episodes]
    n = max(len(rows), 1)
    released = [r for r in rows if r["stack_let_gos"] > 0]
    out = {
      "stack_episodes_with_let_go": len(released) / n,
      "stack_let_go_ok": (
        sum(r["stack_release_ok"] for r in released) / len(released)
        if released
        else None
      ),
      "stack_any_ok": sum(r["stack_any_ok"] for r in rows) / n,
      "stack_lock_armed_median_s": None,
      "stack_lock_refused_episodes": sum(
        r["stack_lock_refused_steps"] > 0 for r in rows
      )
      / n,
      "stack_base_moved_median_m": sorted(r["stack_base_moved_m"] for r in rows)[
        len(rows) // 2
      ]
      if rows
      else None,
      "stack_base_moved_over_5mm": sum(r["stack_base_moved_m"] > 0.005 for r in rows)
      / n,
    }
    armed = sorted(r["stack_lock_armed_steps"] for r in rows)
    if armed and armed[-1] > 0:
      out["stack_lock_armed_median_s"] = armed[len(armed) // 2] * STEP_DT_S
    for k in self.NAMES:
      vals = sorted(r[f"stack_release_{k}"] for r in released)
      out[f"stack_release_{k}_median"] = vals[len(vals) // 2] if vals else None
      out[f"stack_release_{k}_p90"] = (
        vals[min(len(vals) - 1, round(0.9 * (len(vals) - 1)))] if vals else None
      )
    return out


#: Plan S10.1's box bins, env frame, m: near the torso, far, across the centre line (overlapping
#: filters, each with its own n), and the rest; the box's yaw folded to +/- 45 deg; the carry, cube
#: spawn to box centre (>= `box.BOX_MIN_SEPARATION`).
BOX_POSITION_BINS = {
  "near_torso": lambda x, y: x < 0.45 and y < 0.10,
  "far": lambda x, y: x > 0.55,
  "across_centre": lambda x, y: y < 0.0,
  "other": lambda x, y: not (x < 0.45 and y < 0.10) and x <= 0.55 and y >= 0.0,
}
BOX_YAW_BINS_DEG = ((0, 15), (15, 30), (30, 46))
BOX_CARRY_BINS_CM = ((0, 16), (16, 20), (20, 30), (30, 99))

#: The control step, s (`tabletop_env_cfg.CONTROL_DT`), for the stack tally's lock times.
STEP_DT_S = 0.02


class BoxTally:
  """Plan S10's reads, per episode, beside `Tally` (see the module docstring). ``None`` from
  `from_env` when the scene has no box."""

  @classmethod
  def from_env(cls, env: ManagerBasedRlEnv) -> BoxTally | None:
    from pal_mjlab.tasks.manipulation.pick_place.box import BOX

    if BOX not in env.scene.entities:
      return None
    return cls(env)

  def __init__(self, env: ManagerBasedRlEnv) -> None:
    from pal_mjlab.tasks.manipulation.pick_place import box

    self.box = box
    n, dev = env.num_envs, env.device
    zb = lambda: torch.zeros(n, dtype=torch.bool, device=dev)  # noqa: E731
    zf = lambda: torch.zeros(n, device=dev)  # noqa: E731
    self.in_box, self.in_box_after = zb(), zb()
    self.hand_wall_held, self.cube_wall_held, self.hand_wall = zb(), zb(), zb()
    self.displacement = zf()
    self.rel_dxy, self.rel_height, self.rel_yaw, self.released = zf(), zf(), zf(), zb()
    #: Plan S10.1: each episode's box (x, y, env frame; yaw, rad) and cube spawn (x, y) at its start.
    self.start = torch.zeros(n, 5, device=dev)
    self.done: list[dict] = []

  def step(
    self, env: ManagerBasedRlEnv, grasped: torch.Tensor, j_released: torch.Tensor
  ) -> None:
    """At the top of a control step, on the state the policy acts on; ``j_released`` is
    `RetreatTally.released` (the jaw opened after the lift)."""
    from pal_mjlab.tasks.manipulation import mdp

    b = self.box
    fresh = env.episode_length_buf == 0
    if bool(fresh.any()):
      origin = env.scene.env_origins[:, :2]
      box_e = env.scene[b.BOX]
      q = box_e.data.root_link_quat_w
      yaw = torch.atan2(
        2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2)
      )
      now = torch.cat(
        [
          box_e.data.root_link_pos_w[:, :2] - origin,
          yaw_mod90(yaw).unsqueeze(-1),
          env.scene[OBJECT].data.root_link_pos_w[:, :2] - origin,
        ],
        dim=-1,
      )
      self.start = torch.where(fresh.unsqueeze(-1), now, self.start)
    self.in_box = mdp.object_in_box_bool(env, OBJECT, b.BOX, b.BOX_WALLS)
    self.in_box_after |= self.in_box & j_released
    hit = lambda name: env.scene[name].data.found.bool().any(dim=-1)  # noqa: E731
    hand, cube = hit(b.BOX_WALL_HAND_SENSOR), hit(b.BOX_WALL_OBJECT_SENSOR)
    self.hand_wall |= hand
    self.hand_wall_held |= hand & grasped
    self.cube_wall_held |= cube & grasped
    self.displacement = env.command_manager.get_term(COMMAND).box_displacement()
    latch = getattr(env, "_pp_release", None)
    if latch is not None and latch.yaw is not None:
      self.released = latch.released.clone()
      self.rel_dxy, self.rel_height = latch.dxy.clone(), latch.height.clone()
      self.rel_yaw = latch.yaw.clone()

  def finish(
    self, env_ids: torch.Tensor, handed: torch.Tensor, j_released: torch.Tensor
  ) -> None:
    for i in env_ids.tolist():
      row = {
        "in_box_end": bool(self.in_box[i]),
        "box_success": bool(self.in_box[i] & handed[i] & j_released[i]),
        "bounce_out": bool(self.in_box_after[i] & ~self.in_box[i]),
        "hand_wall_held": bool(self.hand_wall_held[i]),
        "cube_wall_held": bool(self.cube_wall_held[i]),
        "hand_wall_any": bool(self.hand_wall[i]),
        "box_moved_mm": 1e3 * float(self.displacement[i]),
        "latch_released": bool(self.released[i]),
        "release_dxy_cm": 100 * float(self.rel_dxy[i]),
        "release_above_rim_cm": 100 * float(self.rel_height[i]),
        "release_yaw_deg": math.degrees(float(self.rel_yaw[i])),
      }
      bx_, by_, byaw, cx_, cy_ = self.start[i].tolist()
      row.update(
        box_x=bx_,
        box_y=by_,
        box_yaw_deg=math.degrees(byaw),
        box_carry_cm=100 * math.hypot(bx_ - cx_, by_ - cy_),
      )
      self.done.append(row)
      for t in (
        self.in_box_after,
        self.hand_wall_held,
        self.cube_wall_held,
        self.hand_wall,
      ):
        t[i] = False

  def summarise(self, episodes: int) -> dict:
    rows = self.done[:episodes]
    n = max(len(rows), 1)
    share = lambda k: sum(bool(r[k]) for r in rows) / n  # noqa: E731
    rel = [r for r in rows if r["latch_released"]]
    med = lambda k: float(np.median([r[k] for r in rel])) if rel else float("nan")  # noqa: E731
    out = {
      "box_in_box_end": share("in_box_end"),
      "box_success": share("box_success"),
      "box_bounce_out": share("bounce_out"),
      "box_hand_wall_held": share("hand_wall_held"),
      "box_cube_wall_held": share("cube_wall_held"),
      "box_moved_mm_median": float(np.median([r["box_moved_mm"] for r in rows]))
      if rows
      else float("nan"),
      "box_release_dxy_cm_median": med("release_dxy_cm"),
      "box_release_above_rim_cm_median": med("release_above_rim_cm"),
      "box_release_yaw_deg_median": med("release_yaw_deg"),
      "box_release_yaw_under_10deg": (
        sum(r["release_yaw_deg"] < 10 for r in rel) / len(rel) if rel else float("nan")
      ),
    }

    def bins(groups: dict) -> dict:
      return {
        name: {
          "n": len(g),
          "box_success": sum(r["box_success"] for r in g) / len(g) if g else None,
        }
        for name, g in groups.items()
      }

    # Plan S10.1's bins (env frame): by where the box is, by its yaw, by how far the cube is carried.
    out["box_by_position"] = bins(
      {
        name: [r for r in rows if test(r["box_x"], r["box_y"])]
        for name, test in BOX_POSITION_BINS.items()
      }
    )
    out["box_by_yaw"] = bins(
      {
        f"{lo}-{hi}deg": [r for r in rows if lo <= abs(r["box_yaw_deg"]) < hi]
        for lo, hi in BOX_YAW_BINS_DEG
      }
    )
    out["box_by_carry"] = bins(
      {
        f"{lo}-{hi}cm": [r for r in rows if lo <= r["box_carry_cm"] < hi]
        for lo, hi in BOX_CARRY_BINS_CM
      }
    )
    return out


def handoff_action(env: ManagerBasedRlEnv, zero: torch.Tensor) -> torch.Tensor:
  """The position controller's action: home pose, jaw open. ``zero`` (B, A) for an absolute
  action, whose 0 is the default pose; for plan S7-8's ``ACTION=delta`` the action moves the
  target, so it is the step home, |a| <= 1, at the stack speed."""
  term = env.action_manager.get_term("joint_pos")
  if not isinstance(term, DeltaRateLimitedJointPositionAction):
    return zero
  home = term._default_target()
  return ((home - term.commanded_target) / term._rate).clamp(-1.0, 1.0)


def retreat_state(
  env: ManagerBasedRlEnv, arm_ids: list[int], jaw_id: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """The jaw's measured angle, the TCP's height above the cube's top, and the driven arm's
  max |q - q_default|, for `RetreatTally.step`. (B,) each"""
  robot, obj = env.scene[ROBOT], env.scene[OBJECT]
  q = robot.data.joint_pos
  half_z = object_half_height(env, OBJECT)
  _, tcp = positions(env)
  above = tcp[:, 2] - (obj.data.root_link_pos_w[:, 2] + half_z)
  home_err = (q[:, arm_ids] - robot.data.default_joint_pos[:, arm_ids]).abs().amax(-1)
  return q[:, jaw_id], above, home_err


def build_env(args: argparse.Namespace) -> ManagerBasedRlEnv:
  cfg = load_env_cfg(args.task, play=True)
  # `play=True` removes observation corruption and the curriculum, both of which we want
  # gone -- but it also sets episode_length_s to 1e9. Put the training horizon back, or no
  # episode ever ends and there is nothing to count.
  cfg.episode_length_s = load_env_cfg(args.task, play=False).episode_length_s
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  noise = getattr(args, "obs_noise", "off")
  if noise != "off":
    # Plan S7-9's precision split: the noise the policy was trained with, back on -- with or
    # without the cube's own (the terms keep their tracker lag either way).
    actor = cfg.observations["actor"]
    actor.enable_corruption = True
    if noise == "on-no-object":
      actor.terms = {
        name: dataclasses.replace(term, noise=None)
        if name in ("object_pos", "object_yaw")
        else term
        for name, term in actor.terms.items()
      }
  if getattr(args, "solver_iterations", None):
    cfg.sim.mujoco.iterations, cfg.sim.mujoco.ls_iterations = args.solver_iterations
  if getattr(args, "plant", False):
    # The support sensor records contact only (`found`); the set-down read needs its force
    # too. A sensor field is read-only on the physics, and `found` -- what every other read
    # here uses -- is unchanged.
    cfg.scene.sensors = tuple(
      dataclasses.replace(s, fields=(*s.fields, "force"))
      if s.name == SUPPORT_SENSOR and "force" not in s.fields
      else s
      for s in cfg.scene.sensors
    )
  apply_stress(cfg, args)
  return ManagerBasedRlEnv(cfg=cfg, device=args.device)


#: Plan S7-11 stress (a): factors on the jaw servo's time constant and speed limit.
STRESS_JAW_TAU = 2.0
STRESS_JAW_SPEED = 0.5


def apply_stress(cfg, args: argparse.Namespace) -> None:
  """Plan S7-11's stress tests (step 0 item 4), eval only, outside S7-6's ``LAG=FP+JS`` range:
  (a) a slower jaw servo, (b) a longer cube-tracker lag, (c) a slipperier table. Each raises
  rather than silently doing nothing on an env without the model it changes."""
  from mjlab.envs.mdp import dr
  from mjlab.managers.event_manager import EventTermCfg
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  if getattr(args, "stress_jaw", False):
    act = cfg.actions["joint_pos"]
    if getattr(act, "servo_joints", None) is None:
      raise SystemExit("--stress-jaw needs the jaw servo model (LAG=JS or FP+JS)")
    # From S7-6's servo (`JAW_SERVO_TAU` / `JAW_SERVO_MAX_VEL`), not the run's own: a run trained
    # on the wide servo (``LAG=FP+JSW``) is stressed with the same absolute range as every other.
    tau, vel = JAW_SERVO_TAU, JAW_SERVO_MAX_VEL
    cfg.actions["joint_pos"] = dataclasses.replace(
      act,
      servo_tau=(tau[0] * STRESS_JAW_TAU, tau[1] * STRESS_JAW_TAU),
      servo_max_vel=(vel[0] * STRESS_JAW_SPEED, vel[1] * STRESS_JAW_SPEED),
    )
  lag = getattr(args, "stress_lag", None)
  if lag is not None:
    lo, hi = lag
    actor = cfg.observations["actor"]
    terms = {}
    for name, term in actor.terms.items():
      if name in ("object_pos", "object_yaw"):
        if term.params.get("lag") is None:
          raise SystemExit(
            "--stress-lag needs the cube tracker (LAG=FP or FP+JS, no OCC)"
          )
        term = dataclasses.replace(term, params={**term.params, "lag": (lo, hi)})
      terms[name] = term
    actor.terms = terms
  mu = getattr(args, "stress_friction", None)
  if mu is not None:
    cfg.events["stress_table_friction"] = EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg(TABLE, geom_names=("table_top",)),
        "operation": "abs",
        "distribution": "uniform",
        "axes": [0],
        "ranges": (mu, mu),
      },
    )
    cfg.events["object_friction"] = EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg(OBJECT, geom_names=(f"{OBJECT}_geom",)),
        "operation": "abs",
        "distribution": "uniform",
        "axes": [0],
        "ranges": (mu, mu),
      },
    )


def stress_of(args: argparse.Namespace) -> dict:
  """The stress conditions an eval ran under, for its JSON."""
  out: dict = {}
  if getattr(args, "stress_jaw", False):
    out["jaw"] = {"tau_x": STRESS_JAW_TAU, "speed_x": STRESS_JAW_SPEED}
  if getattr(args, "stress_lag", None) is not None:
    out["lag_steps"] = list(args.stress_lag)
  if getattr(args, "stress_friction", None) is not None:
    out["cube_table_friction"] = args.stress_friction
  return out


def load_policy(args: argparse.Namespace, env: ManagerBasedRlEnv):
  wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(load_rl_cfg(args.task)), device=args.device)
  runner.load(
    str(args.checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=args.device,
  )
  return wrapped, runner.get_inference_policy(device=args.device)


def probe(env: ManagerBasedRlEnv, grasp_sensor: str) -> dict[str, torch.Tensor]:
  """Every per-step boolean, read from physics rather than from the arm's own reward."""
  obj = env.scene[OBJECT]
  command = env.command_manager.get_term(COMMAND)
  grasped = _grasped_bool(env, grasp_sensor)
  on_surface = _on_surface_bool(env, SUPPORT_SENSOR)
  at_goal = (
    torch.norm(command.command - obj.data.root_link_pos_w, dim=-1) < GOAL_THRESHOLD
  )
  common = (env, COMMAND, OBJECT, grasp_sensor, GOAL_THRESHOLD, 0.05)
  return {
    "grasped": grasped,
    "lifted": grasped & ~on_surface,
    "placed": _placed_bool(*common),
    "placed_and_lifted": _placed_bool(*common, SUPPORT_SENSOR),
    "placed_and_lifted_z": _placed_bool(
      env, COMMAND, OBJECT, grasp_sensor, GOAL_TOL_XYZ, 0.05, SUPPORT_SENSOR
    ),
    # The object is on the floor, not the table: `object_dropped`'s own condition.
    "dropped": obj.data.root_link_pos_w[:, 2]
    < env.cfg.terminations["object_dropped"].params["min_height"],
    "held_at_goal": at_goal & grasped,
  }


def geometry(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
  """The cube's horizontal distance to the target and its height above it. (B,), (B,)"""
  pos = env.scene[OBJECT].data.root_link_pos_w
  goal = env.command_manager.get_term(COMMAND).command
  return torch.norm((pos - goal)[:, :2], dim=-1), pos[:, 2] - goal[:, 2]


def tilt(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The cube's tilt: the angle between the vertical and its nearest face normal, rad. 0 for
  a cube resting on a face, pi/4 on an edge. (B,)"""
  w, x, y, z = env.scene[OBJECT].data.root_link_quat_w.unbind(-1)
  # The world-z components of the body's three axes: the last row of its rotation matrix.
  row = torch.stack(
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1
  )
  return torch.acos(row.abs().amax(-1).clamp(max=1.0))


def reference_yaw(env: ManagerBasedRlEnv) -> torch.Tensor | float:
  """The world yaw the let-go's yaw is read against: the box's on plan S10.1's wide area (the goal
  command's ``yaw_reference``), else 0 -- the world's axes, every earlier read. (B,) or 0"""
  from pal_mjlab.tasks.manipulation.mdp import yaw_reference_quat

  q = yaw_reference_quat(env, COMMAND)
  if q is None:
    return 0.0
  return torch.atan2(
    2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2)
  )


def yaw_mod90(yaw: torch.Tensor) -> torch.Tensor:
  """A cube's yaw folded to [-pi/4, pi/4) (it looks the same every 90 deg), rad: plan S7-9's yaw
  error against a box at yaw 0 (S10's drop table). (B,)"""
  return torch.remainder(yaw + math.pi / 4, math.pi / 2) - math.pi / 4


def pad_height_above_object(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The pad-site midpoint's height above the object's centre, m (plan S8's grasp-height read). (B,)"""
  robot = env.scene[ROBOT]
  ids = [robot.site_names.index(n) for n in PAD_SITES]
  mid = robot.data.site_pos_w[:, ids].mean(dim=1)
  return mid[:, 2] - env.scene[OBJECT].data.root_link_pos_w[:, 2]


def positions(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
  """The cube's and the TCP's world positions, for `Tally.measure_speed`. (B, 3), (B, 3)"""
  robot = env.scene[ROBOT]
  tcp = robot.data.site_pos_w[:, robot.site_names.index(TCP_SITE)]
  return env.scene[OBJECT].data.root_link_pos_w, tcp


def scene_state(
  env: ManagerBasedRlEnv,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """The cube's position, yaw and linear velocity, and the table's offset from its nominal
  pose (DR-scene moves it), for `Tally.record` and `Tally.measure_set_down`."""
  obj = env.scene[OBJECT]
  q = obj.data.root_link_quat_w
  yaw = torch.atan2(
    2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2)
  )
  table = env.scene[TABLE]
  nominal = torch.tensor(
    env.cfg.scene.entities[TABLE].init_state.pos, device=env.device
  )
  offset = env.sim.data.geom_xpos[:, table.indexing.geom_ids[0]] - nominal
  return obj.data.root_link_pos_w, yaw, offset, obj.data.root_link_lin_vel_w


def success_flags(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Which envs the last `step` ended through the `success` termination.

  The termination manager keeps each term's value from the step it just computed, before
  the reset, so this is readable after `step` returns. All False for an arm without one.
  """
  tm = env.termination_manager
  if "success" not in tm.active_terms:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  return tm.get_term("success").bool()


def summarise(tally: Tally, step_dt: float, episodes: int) -> dict:
  rows = tally.done[:episodes]
  n = len(rows)
  rate = lambda k: sum(r[k] for r in rows) / n  # noqa: E731
  placed_times = [
    r["steps_to_place"] for r in rows if r["placed"] and r["steps_to_place"] >= 0
  ]
  held = [r for r in rows if r.get("held_steps")]
  near = [r["hover_dz_mean"] for r in rows if r.get("hover_dz_mean") is not None]
  mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
  median = lambda xs: sorted(xs)[len(xs) // 2] if xs else None  # noqa: E731
  transport = {
    "episodes_held": len(held),
    "held_dxy_mean_m": mean([r["held_dxy_mean"] for r in held]),
    "held_dxy_min_mean_m": mean([r["held_dxy_min"] for r in held]),
    "held_dxy_min_median_m": median([r["held_dxy_min"] for r in held]),
    "within_3cm_xy_while_held": rate("near_xy_while_held") if n else None,
    "hover_dz_mean_m": mean(near),
    "hover_dz_median_m": median(near),
  }
  speed = {}
  for key, name in (("peak_obj_speed_held", "obj"), ("peak_tcp_speed_held", "tcp")):
    peaks = [r[key] for r in rows if r.get(key) is not None]
    speed[f"peak_{name}_speed_held_mean_mps"] = mean(peaks)
    speed[f"peak_{name}_speed_held_median_mps"] = median(peaks)
    speed[f"peak_{name}_speed_held_max_mps"] = max(peaks) if peaks else None
  rel = [r for r in rows if r.get("release_dxy") is not None]
  release = {
    "episodes_released": len(rel),
    "release_dxy_median_m": median([r["release_dxy"] for r in rel]),
    "release_within_3cm_xy": sum(r["release_dxy"] < NEAR_XY for r in rel) / len(rel)
    if rel
    else None,
    "release_dz_median_m": median([r["release_dz"] for r in rel]),
    "release_speed_median_mps": median([r["release_speed"] for r in rel]),
    # Plan S8: a ball let go above the table rolls. How far it went from where it was let go to where
    # it ended (the release point is the set-down read's), and how many were let go more than
    # `DROP_HEIGHT` above their resting height -- a drop in the carry, not a set-down.
    "roll_after_let_go_median_m": median(
      [
        math.hypot(r["final_x"] - r["release_x"], r["final_y"] - r["release_y"])
        for r in rel
        if r.get("release_x") is not None and r.get("final_x") is not None
      ]
    ),
    "release_over_3cm_high": sum(r["release_dz"] > DROP_HEIGHT for r in rel) / len(rel)
    if rel
    else None,
    # Plan S8: the pad midpoint's height above the object's centre at the first grasp.
    "grasp_pad_dz_median_m": median(
      [r["grasp_pad_dz"] for r in rows if r.get("grasp_pad_dz") is not None]
    ),
  }
  sd = [r for r in rows if r.get("release_set_down") is not None]
  set_down = {
    "release_on_table": sum(r["release_on_table"] for r in sd) / len(sd)
    if sd
    else None,
    "release_set_down": sum(r["release_set_down"] for r in sd) / len(sd)
    if sd
    else None,
    "release_dwell_median": median([r["release_dwell"] for r in sd]),
    "release_vz_median_mps": median([r["release_vz"] for r in sd]),
    "release_vxy_median_mps": median([r["release_vxy"] for r in sd]),
    "release_q_m6_median": median([r["release_q_m6"] for r in rel]),
    "release_q_m7_median": median([r["release_q_m7"] for r in rel]),
  }

  def share(group: list) -> dict:
    return {
      "n": len(group),
      "placed": sum(r["placed"] for r in group) / len(group) if group else None,
      "placed_and_lifted": sum(r["placed_and_lifted"] for r in group) / len(group)
      if group
      else None,
    }

  by_carry = {
    f"{lo * 100:.0f}-{min(hi, 0.99) * 100:.0f}cm": share(
      [r for r in rows if lo <= r["carry_m"] < hi]
    )
    for lo, hi in CARRY_BINS
  }
  by_spawn = {
    "inside_w2": share([r for r in rows if r["spawn_in_w2"]]),
    "outside_w2": share([r for r in rows if not r["spawn_in_w2"]]),
  }
  return {
    "episodes": n,
    "by_carry": by_carry,
    "by_spawn": by_spawn,
    "placed": rate("placed"),
    "placed_and_lifted": rate("placed_and_lifted"),
    "placed_and_lifted_z": rate("placed_and_lifted_z"),
    "grasped_ever": rate("grasped"),
    "lifted_ever": rate("lifted"),
    "held_at_goal": rate("held_at_goal"),
    "dropped": rate("dropped"),
    "mean_episode_length_s": sum(r["length"] for r in rows) / n * step_dt,
    "time_to_place_s": (sum(placed_times) / len(placed_times) * step_dt)
    if placed_times
    else None,
    **transport,
    **speed,
    **release,
    **set_down,
  }


def report(result: dict, rewards: dict[str, float], args: argparse.Namespace) -> None:
  print(f"\ncheckpoint   {args.checkpoint}")
  print(f"task         {args.task}")
  print(
    f"episodes     {result['episodes']} over {args.num_envs} envs, seed {args.seed}\n"
  )

  print("  THE TWO PLACED RATES ---------------------------------------------")
  print(
    f"    placed                   {result['placed']:6.1%}   at goal, released, settled"
  )
  print(
    f"    placed & lifted          {result['placed_and_lifted']:6.1%}   ... and it left the table"
  )
  print(
    f"    placed & lifted, Z       {result['placed_and_lifted_z']:6.1%}   "
    f"... at the goal by S7-5's xy/z tolerance {GOAL_TOL_XYZ}"
  )
  slid = result["placed"] - result["placed_and_lifted"]
  print(f"    -> slid, never picked up {slid:6.1%}")
  by_carry = result["by_carry"]
  if sum(v["n"] > 0 for v in by_carry.values()) > 1:
    print(
      "    by carry length          "
      + "   ".join(
        f"{k} {v['placed_and_lifted']:.0%} (n={v['n']})"
        for k, v in by_carry.items()
        if v["n"]
      )
    )
    print(
      "    by spawn                 "
      + "   ".join(
        f"{k} {v['placed_and_lifted']:.0%} (n={v['n']})"
        for k, v in result["by_spawn"].items()
        if v["n"]
      )
    )

  print("\n  DIAGNOSTIC RATES -------------------------------------------------")
  print(
    f"    grasped, ever            {result['grasped_ever']:6.1%}   ~0 with high reach = Isaac Lab #204"
  )
  print(
    f"    lifted, ever             {result['lifted_ever']:6.1%}   high grasp + low lift = cannot lift"
  )
  print(
    f"    held at goal             {result['held_at_goal']:6.1%}   the hold-at-goal failure"
  )
  print(f"    dropped                  {result['dropped']:6.1%}")

  print("\n  TRANSPORT (horizontal, while held) -------------------------------")
  fmt = lambda v, u="m": "n/a" if v is None else f"{v:6.3f} {u}"  # noqa: E731
  print(f"    episodes that held it    {result['episodes_held']:6d}")
  print(f"    mean xy dist while held  {fmt(result['held_dxy_mean_m'])}")
  print(
    f"    closest xy while held    {fmt(result['held_dxy_min_median_m'])}  (median)"
  )
  w3 = result["within_3cm_xy_while_held"]
  print(f"    within 3 cm xy, held     {'n/a' if w3 is None else f'{w3:6.1%}'}")
  print(
    f"    hover height over target {fmt(result['hover_dz_mean_m'])}  (z - z_target, held within 3 cm xy)"
  )

  print("\n  SPEED WHILE HELD (per-episode peak, over one control step) -------")
  for name, label in (("obj", "cube"), ("tcp", "TCP")):
    m = result[f"peak_{name}_speed_held_mean_mps"]
    if m is None:
      print(f"    peak {label:4s} speed          n/a")
      continue
    med = result[f"peak_{name}_speed_held_median_mps"]
    top = result[f"peak_{name}_speed_held_max_mps"]
    print(
      f"    peak {label:4s} speed         {m:6.2f} m/s  (median {med:.2f}, max {top:.2f})"
    )

  print("\n  RELEASE (the episode's last let-go) ------------------------------")
  if result.get("episodes_released"):
    print(f"    episodes that let go      {result['episodes_released']:6d}")
    print(
      f"    xy to target at release  {fmt(result['release_dxy_median_m'])}  (median)"
    )
    print(f"    within 3 cm xy           {result['release_within_3cm_xy']:6.1%}")
    print(
      f"    height over target       {fmt(result['release_dz_median_m'])}  (median)"
    )
    print(
      f"    cube speed at release    {fmt(result['release_speed_median_mps'], 'm/s')}  (median)"
    )
    print(
      f"    let go > {DROP_HEIGHT * 100:.0f} cm up       {result['release_over_3cm_high']:6.1%}   "
      f"(a drop in the carry)"
    )
    print(
      f"    roll after the let-go    {fmt(result['roll_after_let_go_median_m'])}  (median, let-go point to rest)"
    )
    print(
      f"    pad midpoint above the object's centre at the first grasp  {fmt(result['grasp_pad_dz_median_m'])}  (median; the open-jaw value is ~0.010 m higher)"
    )
  else:
    print("    no release observed")

  print("\n  SET-DOWN (plan S7-2's M7, at the last let-go) ---------------------")
  if result.get("release_set_down") is not None:
    print(f"    on the table             {result['release_on_table']:6.1%}")
    print(
      f"    a set-down (M7 passes)   {result['release_set_down']:6.1%}   "
      f"on the table, < {SET_DOWN_STILL} m/s, held still >= {SET_DOWN_STEPS} steps"
    )
    print(
      f"    dwell before letting go  {result['release_dwell_median']:6d} steps (median)"
    )
    print(
      f"    cube v_z / v_xy          {fmt(result['release_vz_median_mps'], 'm/s')} / "
      f"{fmt(result['release_vxy_median_mps'], 'm/s')}  (median)"
    )
    print(
      f"    grade M6 / M7            {result['release_q_m6_median']:.2f} / "
      f"{result['release_q_m7_median']:.2f}  (median)"
    )
  else:
    print("    no release observed")

  print("\n  TIMING -----------------------------------------------------------")
  ttp = result["time_to_place_s"]
  print(f"    mean episode             {result['mean_episode_length_s']:6.2f} s")
  print(f"    time to place            {'n/a' if ttp is None else f'{ttp:6.2f} s'}")

  if "plant_parked_hold_mean_rad" in result:
    fmt3 = lambda v, u="rad": "n/a" if v is None else f"{v:7.4f} {u}"  # noqa: E731
    print("\n  PLANT (plan S8) --------------------------------------------------")
    print(
      f"    parked arm + pelvis hold {fmt3(result['plant_parked_hold_max_median_rad'])}"
      f"  (episode max, median; worst {result['plant_parked_worst_joint']})"
    )
    print(f"      max over episodes      {fmt3(result['plant_parked_hold_max_rad'])}")
    print(f"      mean                   {fmt3(result['plant_parked_hold_mean_rad'])}")
    print(
      f"    driven arm |q*-q| carry  {fmt3(result['plant_track_carry_mean_rad'])}"
      f"  (mean; episode max, median {fmt3(result['plant_track_carry_max_median_rad'])})"
    )
    sat = result["plant_saturation_share"]
    top = sorted(sat.items(), key=lambda kv: -kv[1])[:3]
    print(
      "    at effort limit          "
      + ", ".join(f"{n.removesuffix('_joint')} {v:.1%}" for n, v in top)
    )
    print(
      f"    set-down force, peak     {fmt3(result['plant_setdown_force_peak_median_n'], 'N')}"
      f"  (median; max {fmt3(result['plant_setdown_force_peak_max_n'], 'N')})"
    )

  if "plant_driven_peak_effort_ratio" in result:
    print("\n  DRIVEN ARM (plan S7-4) -------------------------------------------")
    peak = result["plant_driven_peak_effort_ratio"]
    after = result["plant_driven_saturation_after_let_go"]
    out = result["plant_driven_target_out_of_range"]
    for n, lim in result["plant_driven_effort_limit_nm"].items():
      print(
        f"    {n.removesuffix('_joint'):14s} limit {lim:5.1f} N.m  peak |tau|/limit "
        f"{peak[n]:5.2f}  at limit after let-go {after[n]:6.1%}  target out of range "
        f"{out[n]:6.1%}"
      )

  if "jaw_released" in result:
    fmt = lambda v, u="m": "n/a" if v is None else f"{v:6.3f} {u}"  # noqa: E731
    print("\n  RELEASE BY THE JAW, RETREAT, HAND-OFF (plan S7-4) ------------------")
    print(
      f"    jaw opened (< {JAW_OPEN} rad; a ball {BALL_JAW_OPEN})   "
      f"{result['jaw_released']:6.1%}   after the lift"
    )
    print(
      f"    jaw at the pads' let-go  {fmt(result['jaw_at_let_go_median_rad'], 'rad')}  (median)"
    )
    print(f"    placed, jaw open         {result['placed_j']:6.1%}")
    print(f"    placed & lifted, jaw open{result['placed_and_lifted_j']:6.1%}")
    print(
      f"    lifted clear (>= {HANDOFF_CLEAR} m) {result['lifted_clear']:6.1%}   "
      f"max TCP over cube top {fmt(result['lift_clear_max_median_m'])} (median)"
    )
    print(
      f"    hand-off by lift / time  {result['handoff_by_lift']:6.1%} / "
      f"{result['handoff_by_timeout']:6.1%}"
    )
    print(
      f"    placed -> hand-off       {fmt(result['placed_to_handoff_median_s'], 's')}  (median)"
    )
    print(
      f"    cube moved, lift         {fmt(result['cube_moved_lift_median_m'])}  "
      f"(median; max {fmt(result['cube_moved_lift_max_m'])})"
    )
    print(
      f"    cube moved, return       {fmt(result['cube_moved_return_median_m'])}  "
      f"(median; max {fmt(result['cube_moved_return_max_m'])})"
    )
    print(
      f"    still placed at the end  {result['final_placed_j']:6.1%}   within "
      f"{GOAL_THRESHOLD} m: {result['final_within_goal']:6.1%}"
    )
    print(f"    time to home             {fmt(result['time_to_home_median_s'], 's')}")
    if "handoff_lift_m" in result:
      print(
        f"    vertical lift first      to {result['handoff_lift_m']} m over the cube  "
        f"(reached {fmt(result['handoff_lift_home_height_median_m'])} median; IK missed "
        f"{result['handoff_lift_ik_failed']} of {result['handoff_lift_planned']})"
      )
    print(
      f"    SUCCESS (S7-4)           {result['success_retreat']:6.1%}   placed & lifted & "
      "jaw-released & lifted clear & still placed"
    )
    print(
      f"    ... under S7-5's Z       {result['success_retreat_z']:6.1%}   "
      f"placed & lifted, jaw open {result['placed_and_lifted_j_z']:6.1%}, still placed "
      f"{result['final_placed_j_z']:6.1%}"
    )
    tl = result["tilt_at_let_go_median_rad"]
    print(
      f"    cube tilt at let-go      "
      f"{'n/a' if tl is None else f'{math.degrees(tl):6.1f} deg'}  (median; > 10 deg "
      f"{result['tilt_at_let_go_over_10deg']:6.1%})"
    )

  if "occ_lost_share" in result:
    print("\n  OCCLUSION (plan S7-7) ---------------------------------------------")
    fmt_p = lambda v: "   n/a" if v is None else f"{v:6.1%}"  # noqa: E731
    print(
      f"    cube not reported        {result['occ_lost_share']:6.1%}   of steps (the line of "
      f"sight and the random loss hide it {result['occ_blocked_share']:6.1%})"
    )
    by = result["occ_lost_share_by_phase"]
    share = result["occ_phase_share"]
    print(
      "    lost by phase            "
      + "  ".join(f"{k} {fmt_p(by[k])} (of {share[k]:.0%})" for k in by)
    )
    print(
      f"    re-approaches            {result['occ_reapproach']:6.1%}   of episodes grasp more "
      f"than once; the jaw reopens away from the goal in {result['occ_reopen_away']:6.1%}"
    )
    print(
      f"    O-K held rule vs contact precision {fmt_p(result['occ_held_precision'])}  "
      f"recall {fmt_p(result['occ_held_recall'])}  trails the grasp by "
      f"{result['occ_held_lag_median_steps']} steps (median)"
    )
    for k, v in result["occ_est_err_by_steps_unseen"].items():
      if v:
        print(
          f"    estimate error, unseen {k:>6} steps   mean {v['mean_m'] * 100:5.2f} cm  "
          f"max {v['max_m'] * 100:5.2f} cm  (n {v['n']})"
        )

  if "stack_any_ok" in result:
    print("\n  STACK (plan S9) ---------------------------------------------------")
    print(
      f"    episodes with a let-go   {result['stack_episodes_with_let_go']:6.1%}   last let-go OK "
      f"{fmt(result['stack_let_go_ok'], '')}   any let-go OK {result['stack_any_ok']:6.1%}"
    )
    for k in StackTally.NAMES:
      print(
        f"    at the let-go {k:10s} median {fmt(result[f'stack_release_{k}_median'], '')}  "
        f"p90 {fmt(result[f'stack_release_{k}_p90'], '')}"
      )
    print(
      f"    lock refused an opening  {result['stack_lock_refused_episodes']:6.1%} of episodes; armed "
      f"{fmt(result['stack_lock_armed_median_s'], 's')} (median)"
    )
    print(
      f"    base moved               {fmt(result['stack_base_moved_median_m'], 'm')} (median); "
      f"> 5 mm in {result['stack_base_moved_over_5mm']:6.1%}"
    )
  if "arm_min_clearance_median_m" in result:
    print("\n  ARM VS TABLE, links 1-6 (plan S7-10) --------------------------------")
    cm = lambda v: "   n/a" if v is None else f"{v * 100:6.2f} cm"  # noqa: E731
    fmt3 = lambda v, u: "n/a" if v is None else f"{v:7.4f} {u}"  # noqa: E731
    print(
      f"    lowest point over table  {cm(result['arm_min_clearance_median_m'])}"
      f"  (episode min, median; min {cm(result['arm_min_clearance_min_m'])})"
    )
    for link, v in result["arm_min_clearance_median_by_link_m"].items():
      print(f"      {link:22s} {cm(v)}")
    print(
      "    within 1 / 2 / 3 cm      "
      + " / ".join(f"{result[f'arm_within_{c}cm_share']:.0%}" for c in (1, 2, 3))
    )
    if "arm_table_contact_share" in result:
      print(
        f"    episodes with contact    {result['arm_table_contact_share']:.0%}"
        f"  (> 30 N {result['arm_table_over_30n_share']:.0%}; peak median "
        f"{fmt3(result['arm_table_peak_force_median_n'], 'N')}, max "
        f"{fmt3(result['arm_table_peak_force_max_n'], 'N')})"
      )
      for link, v in result["arm_table_contact_share_by_link"].items():
        print(f"      {link:22s} {v:.0%}")
      print(
        f"    contact steps by phase   {result['arm_table_contact_steps_by_phase']}"
      )
    else:
      print("    contact: no arm-table sensor in this env (set ARM_TABLE=watch)")

  if rewards:
    print("\n  PER-TERM REWARD (mean episode sum) --------------------------------")
    for name, value in sorted(rewards.items(), key=lambda kv: -abs(kv[1])):
      print(f"    {name:24s} {value:9.4f}")
  print()


def main() -> None:
  args = parse_args()
  torch.manual_seed(args.seed)
  env = build_env(args)
  wrapped, policy = load_policy(args, env)

  gripper_cfg = env.cfg.rewards.get("grasp") or env.cfg.rewards["lift"]
  grasp_sensor = (
    gripper_cfg.params.get("sensor_name") or gripper_cfg.params["grasp_sensor_name"]
  )

  tally = Tally(args.num_envs, args.device)
  plant = PlantTally.from_env(env) if args.plant else None
  retreat = RetreatTally(
    args.num_envs, args.device, env.step_dt, args.handoff, args.handoff_timeout
  )
  if args.handoff_lift is not None and not args.handoff:
    raise SystemExit("--handoff-lift needs --handoff")
  if args.handoff_lift is not None and hasattr(
    env.action_manager.get_term("joint_pos"), "start_handoff"
  ):
    raise SystemExit(
      "--handoff-lift is the IK lift; HANDOFF_ROUTE=up-out is plan S11's route -- pick one"
    )
  if args.handoff_lift is not None and isinstance(
    env.action_manager.get_term("joint_pos"), DeltaRateLimitedJointPositionAction
  ):
    raise SystemExit("--handoff-lift is written for absolute targets, not ACTION=delta")
  lift = None if args.handoff_lift is None else VerticalRetreat(env, args.handoff_lift)
  occ = OccTally(args.num_envs, args.device)
  stack = StackTally(env)
  box = BoxTally.from_env(env)
  arm = ArmTally.from_env(env)
  trace: list[dict[str, torch.Tensor]] = []
  robot = env.scene[ROBOT]
  arm_ids = [robot.joint_names.index(n) for n in robot.joint_names]
  arm_ids = [i for i in arm_ids if robot.joint_names[i].startswith("arm_left_")]
  jaw_id = robot.joint_names.index(JAW_JOINT)
  reward_sums: dict[str, list[float]] = {}

  obs, _ = wrapped.reset()
  with torch.inference_mode():
    while len(tally.done) < args.episodes:
      goal = env.command_manager.get_term(COMMAND).command
      obj_pos, obj_yaw, table_offset, obj_vel = scene_state(env)
      tally.record(obj_pos, obj_yaw, table_offset, goal)
      flags = probe(env, grasp_sensor)
      tally.observe(**flags)
      dxy, dz = geometry(env)
      tally.measure(flags["grasped"], dxy, dz)
      tally.measure_grasp_height(flags["grasped"], pad_height_above_object(env))
      if stack.active:
        stack.step(env, flags["grasped"])
      jaw_q, tcp_above, home_err = retreat_state(env, arm_ids, jaw_id)
      jaw_thr = jaw_open_for(env)
      handed = retreat.step(
        jaw_q,
        flags["grasped"],
        flags["lifted"],
        flags["placed_and_lifted"] & (jaw_q < jaw_thr),
        tcp_above,
        obj_pos[:, :2],
        dxy,
        home_err,
        placed_jz=flags["placed_and_lifted_z"] & (jaw_q < jaw_thr),
        tilt=tilt(env),
        jaw_open_below=jaw_thr,
        yaw=yaw_mod90(obj_yaw - reference_yaw(env)),
      )
      if box is not None:
        box.step(env, flags["grasped"], retreat.released)
      if plant is not None:
        plant.measure_env(
          carrying=flags["lifted"],
          setdown=flags["grasped"] & (dxy < NEAR_XY),
          after_let_go=retreat.let_go,
        )
      arm.step_env(flags["grasped"], retreat.let_go)
      sensors = getattr(env, "_pal_cube_sensors", None)
      if sensors and OBJECT in sensors:
        occ.step(
          sensors[OBJECT], flags["grasped"], flags["lifted"], obj_pos, jaw_q, dxy
        )
        if args.trace:
          sensor = sensors[OBJECT]
          _, tcp_w = positions(env)
          term = env.action_manager.get_term("joint_pos")
          jaw_cmd = term._processed_actions[:, sensor._jaw_action_idx]
          trace.append(
            {
              "q": jaw_q,
              "cmd": jaw_cmd,
              "grasped": flags["grasped"],
              "lifted": flags["lifted"],
              "seen": sensor.seen,
              "hidden": ~sensor.live_visible,
              "held": sensor.held,
              "score": sensor.blocked_score,
              "unseen": sensor.unseen_steps,
              "tcp_cube": torch.norm(tcp_w - obj_pos, dim=-1),
              "last_tcp": torch.norm(sensor.last_pos - tcp_w, dim=-1),
              "est_err": torch.norm(sensor.est_pos - obj_pos, dim=-1),
              "dxy": dxy,
              "step": env.episode_length_buf.clone(),
            }
          )
      tally.measure_speed(flags["grasped"], *positions(env), env.step_dt, goal)
      tally.measure_set_down(
        flags["grasped"],
        _on_surface_bool(env, SUPPORT_SENSOR),
        obj_pos,
        obj_vel,
        goal,
      )
      actions = policy(obs)
      route = env.action_manager.get_term("joint_pos")
      if args.handoff and hasattr(route, "start_handoff"):
        # Plan S11's up-out-home route (`HANDOFF_ROUTE=up-out`, `mdp.handoff`): the action term
        # takes the arm over from this step on; the policy's actions pass through unread.
        route.start_handoff("left", handed)
      elif lift is not None:
        actions = lift.act(handed, retreat.hand_step, retreat.age, tcp_above, actions)
      elif args.handoff:
        # The position controller: action 0 is the home pose with the jaw open, reached
        # through the action term's slew (and its tool cap and bounds, where the run has them).
        actions = torch.where(
          handed.unsqueeze(-1), handoff_action(env, torch.zeros_like(actions)), actions
        )
      obs, _, dones, extras = wrapped.step(actions)
      tally.terminal(success_flags(env))
      finished = torch.nonzero(dones, as_tuple=False).flatten()
      if finished.numel():
        n_before = len(tally.done)
        if box is not None:  # before `retreat.finish` clears its per-episode state
          box.finish(finished, retreat.handed, retreat.released)
        tally.finish(finished)
        retreat.finish(
          finished, [r["placed_and_lifted"] for r in tally.done[n_before:]]
        )
        if occ.steps.any() or occ.done:
          occ.finish(finished)
        if stack.active:
          stack.finish(finished)
        if plant is not None:
          plant.finish(finished)
        arm.finish(finished)
        for key, value in extras.get("log", {}).items():
          if key.startswith("Episode_Reward/"):
            reward_sums.setdefault(key.removeprefix("Episode_Reward/"), []).append(
              float(value)
            )

  result = summarise(tally, env.step_dt, args.episodes)
  result.update(retreat.summarise(args.episodes))
  if occ.done:
    result.update(occ.summarise(args.episodes))
  if stack.active:
    result.update(stack.summarise(args.episodes))
  result["handoff"] = bool(args.handoff)
  if lift is not None:
    result.update(lift.summarise())
  if plant is not None:
    result.update(plant.summarise(args.episodes))
  result.update(arm.summarise(args.episodes))
  if box is not None:
    result.update(box.summarise(args.episodes))
    print("\nplan S10 (box): " + json.dumps(box.summarise(args.episodes)))
  rewards = {k: sum(v) / len(v) for k, v in reward_sums.items()}
  report(result, rewards, args)

  if args.trace and trace:
    import numpy as np

    args.trace.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
      args.trace,
      **{k: torch.stack([t[k] for t in trace]).cpu().numpy() for k in trace[0]},
      step_dt=env.step_dt,
    )
    print(f"wrote {args.trace}")

  if args.json:
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
      json.dumps(
        {
          "checkpoint": str(args.checkpoint),
          "task": args.task,
          "num_envs": args.num_envs,
          "seed": args.seed,
          "solver_iterations": args.solver_iterations,
          "stress": stress_of(args),
          **result,
          "reward_terms": rewards,
          "episodes_detail": [
            {**r, **q, **o, **st, **a, **b}
            for r, q, o, st, a, b in zip(
              tally.done[: args.episodes],
              retreat.done[: args.episodes],
              occ.done[: args.episodes] or [{}] * args.episodes,
              stack.done[: args.episodes] or [{}] * args.episodes,
              arm.done[: args.episodes],
              (box.done[: args.episodes] if box is not None else None)
              or [{}] * args.episodes,
              strict=True,
            )
          ],
        },
        indent=2,
      )
    )
    print(f"wrote {args.json}")

  env.close()


if __name__ == "__main__":
  main()
