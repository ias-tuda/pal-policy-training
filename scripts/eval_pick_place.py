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

Horizon
--------------------------------------------------------------------------------
Not the `--play` cfg as-is: `apply_play_overrides` sets `episode_length_s` to 1e9, so
episodes never time out and nothing ever completes. This takes the play cfg (no observation
corruption, no curriculum) and restores the *training* episode length, which is the horizon
the policy was optimised for and the only one these rates mean anything against.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from dataclasses import asdict

import mujoco
import numpy as np
import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from pal_mjlab.tasks.manipulation.mdp.observations import (
  _grasped_bool,
  _on_surface_bool,
  _placed_bool,
)
from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import (
  GOAL_THRESHOLD,
  OBJECT,
)
from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS_LEFT_ARM
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import (
  COMMAND,
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
#: ... and closed above this (a stroke is open -> closed, with hysteresis; S7-7's re-approach count).
JAW_CLOSED = 0.25
JAW_JOINT = "gripper_left_finger_joint"
HANDOFF_CLEAR = 0.03
HANDOFF_STEPS = 5
HANDOFF_TIMEOUT_S = 8.0
HOME_TOL = 0.05
#: The position controller's return (2026-09-30, after S7-5's 5 k read: 40 of 45 placed-but-failed
#: episodes had the cube moved by the joint-space return sweeping the fingers through it). It now
#: opens the jaw (at most `HANDOFF_OPEN_MAX_S`; a lift hand-off is already open), raises the TCP
#: straight up by `HANDOFF_RISE` at `HANDOFF_RISE_SPEED`, crosses at that height to above the home
#: TCP and comes straight down onto it at `HANDOFF_MOVE_SPEED` (`VerticalRetreat`).
#: ``--handoff-direct`` keeps the old return (home at once) for comparison with S7-4/S7-5.
HANDOFF_RISE = 0.10
HANDOFF_RISE_SPEED = 0.10
HANDOFF_MOVE_SPEED = 0.15
HANDOFF_OPEN_MAX_S = 0.5
#: Plan S7-5's Z: "at the goal" is xy < 2 cm and |dz| < 2 cm, `pick_place.rewards.Z_GOAL_TOL`.
GOAL_TOL_XYZ = (0.02, 0.02)


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
    help="plan S7-4: hand the arm to a position controller once the jaw has opened after "
    "the lift and the TCP is clear of the cube, or at 8 s; it opens the jaw, rises "
    "HANDOFF_RISE straight up, crosses to above home and comes down onto it",
  )
  p.add_argument(
    "--handoff-direct",
    action="store_true",
    help="with --handoff: the S7-4/S7-5 return -- home at once, no rise (joint-space sweep)",
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
      final = self.last_pos[i].tolist()
      row.update(zip(("final_x", "final_y", "final_z"), final, strict=True))
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


class RetreatTally:
  """Plan S7-4's reads, per episode, kept beside `Tally` and banked with it: the jaw release
  (module J), the retreat and the hand-off to the position controller. Tensors in, like
  `PlantTally`; call `step` once per control step, at the top of the step, and apply the
  hand-off mask it returns to that step's action."""

  def __init__(
    self, num_envs: int, device: str, step_dt: float, handoff: bool = False
  ) -> None:
    self.step_dt = step_dt
    self.handoff = handoff
    self.timeout_steps = round(HANDOFF_TIMEOUT_S / step_dt)
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
  ) -> torch.Tensor:
    """One step's state -> the envs the position controller drives this step. (B,)

    ``tcp_above`` is the TCP's height above the cube's top, ``goal_dxy`` the cube's horizontal
    distance to the target, ``arm_home_err`` the driven arm's max |q - q_default|;
    ``placed_jz`` is ``placed_j`` under plan S7-5's Z tolerance, ``tilt`` the cube's tilt
    (rad), both optional."""
    grasped, lifted, placed_j = grasped.bool(), lifted.bool(), placed_j.bool()
    placed_jz = torch.zeros_like(placed_j) if placed_jz is None else placed_jz.bool()
    jaw_open = jaw_q < JAW_OPEN
    self.lifted |= lifted
    let_go = self.prev_grasped & ~grasped & self.lifted
    self.jaw_at_let_go = torch.where(let_go, jaw_q, self.jaw_at_let_go)
    if tilt is not None:
      self.tilt_at_let_go = torch.where(let_go, tilt, self.tilt_at_let_go)
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


class VerticalRetreat:
  """The position controller's return with ``--handoff``: open, rise, cross over, come down home.

  Per env, from the step `RetreatTally` hands it over:

  1. *open* -- the arm held at its last commanded target, the jaw commanded open, until the
     measured jaw is below `JAW_OPEN` or `HANDOFF_OPEN_MAX_S` has passed (a lift hand-off is
     already open: zero steps);
  2. *move* -- a Cartesian path for the TCP: straight up by `HANDOFF_RISE` at
     `HANDOFF_RISE_SPEED`, the hand's orientation kept; then level, at that height, to above the
     home pose's TCP, the orientation turning to home's; then straight down onto it, both at
     `HANDOFF_MOVE_SPEED`. The last leg exists because the home TCP sits at about the cube's
     height (1-3 cm above its centre on S7-5's RCTP s3): a joint-space move home from a raised
     pose comes back down through the goal region (2026-09-30, `../results/S7-5/handoff/`);
  3. *home* -- action 0 (the default pose, jaw open), as the direct return did.

  The path is solved once, when the move starts, by damped least squares on the host model
  (`env.sim.mj_model`) with that world's joint positions and the arm at its commanded target --
  the target, not the measured pose, so the rise starts where the arm is being held and not a
  gravity sag below it; the null space is pulled toward the home pose, so the path ends on it.
  Every phase goes through the action term (slew, C, T), like the policy's actions.
  """

  OPEN, MOVE, HOME = 0, 1, 2

  def __init__(self, env: ManagerBasedRlEnv) -> None:
    self.env = env
    term = env.action_manager.get_term("joint_pos")
    self.term = term
    names = [n.split("/", 1)[-1] for n in term.target_names]
    self.arm_idx = [names.index(f"arm_left_{i}_joint") for i in range(1, 8)]
    self.jaw_idx = names.index(JAW_JOINT)
    n, dim = env.num_envs, len(names)
    scale = term.scale
    self.scale = (
      scale
      if isinstance(scale, torch.Tensor)
      else torch.full((n, dim), float(scale), device=env.device)
    )
    offset = term.offset
    self.offset = (
      offset
      if isinstance(offset, torch.Tensor)
      else torch.full((n, dim), float(offset), device=env.device)
    )
    model = env.sim.mj_model
    self.model, self.data = model, mujoco.MjData(model)
    jids = [
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot/arm_left_{i}_joint")
      for i in range(1, 8)
    ]
    # The host model names the robot's elements with its entity prefix (`robot/`).
    self.tcp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"robot/{TCP_SITE}")
    if self.tcp < 0 or min(jids) < 0:
      raise KeyError(f"no robot/{TCP_SITE} or robot/arm_left_*_joint in the host model")
    self.qadr = np.asarray([model.jnt_qposadr[j] for j in jids])
    self.dofs = np.asarray([model.jnt_dofadr[j] for j in jids])
    self.lo = model.jnt_range[jids, 0].copy()
    self.hi = model.jnt_range[jids, 1].copy()
    self.open_max = round(HANDOFF_OPEN_MAX_S / env.step_dt)
    self.phase = torch.full((n,), self.OPEN, dtype=torch.long, device=env.device)
    self.active = torch.zeros(n, dtype=torch.bool, device=env.device)
    self.k = torch.zeros(n, dtype=torch.long, device=env.device)
    self.paths: list[torch.Tensor | None] = [None] * n
    self.n_rise = [0] * n
    self.hold = torch.zeros(n, 7, device=env.device)
    #: Per env: the TCP's world z when the move started and when its rise leg ended (nan until
    #: then), and whether every waypoint's IK converged -- banked by `finish`.
    self.rise_z0 = torch.full((n,), float("nan"), device=env.device)
    self.rise_z1 = torch.full((n,), float("nan"), device=env.device)
    self.ik_ok = torch.ones(n, dtype=torch.bool, device=env.device)
    self.done: list[dict] = []

  def _fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    self.data.qpos[self.qadr] = q
    mujoco.mj_kinematics(self.model, self.data)
    return (
      self.data.site_xpos[self.tcp].copy(),
      self.data.site_xmat[self.tcp].reshape(3, 3).copy(),
    )

  def _solve(self, i: int, q_start: np.ndarray) -> tuple[np.ndarray, int, bool]:
    """World ``i``'s path: arm targets for each step (steps, 7), how many of them are the
    rise, and whether every waypoint converged."""
    m, d = self.model, self.data
    d.qpos[:] = self.env.sim.data.qpos[i].detach().cpu().numpy()
    q_home = self.offset[i, self.arm_idx].detach().cpu().numpy().astype(np.float64)
    p0, r0 = self._fk(q_start)
    ph, rh = self._fk(q_home)
    top = p0[2] + HANDOFF_RISE
    dt = self.env.step_dt
    # Waypoints (position, fraction of the turn from r0 to rh), one per control step.
    legs = [
      (p0, np.array([p0[0], p0[1], top]), HANDOFF_RISE_SPEED, (0.0, 0.0)),
      (
        np.array([p0[0], p0[1], top]),
        np.array([ph[0], ph[1], top]),
        HANDOFF_MOVE_SPEED,
        (0.0, 1.0),
      ),
      (np.array([ph[0], ph[1], top]), ph, HANDOFF_MOVE_SPEED, (1.0, 1.0)),
    ]
    way, n_rise = [], 0
    for li, (a, b, speed, (f0, f1)) in enumerate(legs):
      steps = max(1, int(np.ceil(np.linalg.norm(b - a) / (speed * dt) - 1e-6)))
      for s in range(1, steps + 1):
        u = s / steps
        way.append((a + (b - a) * u, f0 + (f1 - f0) * u))
      if li == 0:
        n_rise = len(way)
    # The turn r0 -> rh as an axis-angle, applied in fractions.
    rel = r0.T @ rh
    angle = np.arccos(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0))
    if angle > 1e-6:
      axis = np.array(
        [rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]]
      ) / (2.0 * np.sin(angle))
    else:
      axis = np.array([0.0, 0.0, 1.0])
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    q, out, ok = q_start.copy(), [], True
    for goal, frac in way:
      quat, turn = np.zeros(4), np.zeros(9)
      mujoco.mju_axisAngle2Quat(quat, axis, angle * frac)
      mujoco.mju_quat2Mat(turn, quat)
      r_des = r0 @ turn.reshape(3, 3)
      for _ in range(20):
        pos, rot = self._fk(q)
        err = np.concatenate(
          [goal - pos, 0.5 * sum(np.cross(rot[:, j], r_des[:, j]) for j in range(3))]
        )
        if np.linalg.norm(err[:3]) < 5e-4 and np.linalg.norm(err[3:]) < 1e-2:
          break
        mujoco.mj_comPos(m, d)
        mujoco.mj_jacSite(m, d, jacp, jacr, self.tcp)
        jac = np.vstack([jacp[:, self.dofs], jacr[:, self.dofs]])
        pinv = jac.T @ np.linalg.inv(jac @ jac.T + 1e-4 * np.eye(6))
        null = np.eye(7) - pinv @ jac
        step = pinv @ err + null @ (0.1 * (q_home - q))
        q = np.clip(q + np.clip(step, -0.1, 0.1), self.lo, self.hi)
      else:
        ok = False
      out.append(q.copy())
    return np.stack(out), n_rise, ok

  def action(
    self, handed: torch.Tensor, jaw_q: torch.Tensor, tcp_z: torch.Tensor
  ) -> torch.Tensor:
    """This step's raw action for the handed envs (rows of the others are unused). (B, A)"""
    # Hold the arm where it was commanded when the controller took over.
    newly = handed & ~self.active
    self.hold[newly] = self.term._last_target[newly][:, self.arm_idx]
    self.active |= newly
    in_open = handed & (self.phase == self.OPEN)
    self.k = torch.where(in_open, self.k + 1, self.k)
    go = in_open & ((jaw_q < JAW_OPEN) | (self.k > self.open_max))
    for i in torch.nonzero(go).flatten().tolist():
      q0 = self.hold[i].cpu().numpy().astype(np.float64)
      path, n_rise, ok = self._solve(i, q0)
      self.paths[i] = torch.as_tensor(
        path, dtype=torch.float32, device=self.hold.device
      )
      self.n_rise[i] = n_rise
      self.ik_ok[i] = ok
      self.rise_z0[i] = tcp_z[i]
    self.phase = torch.where(go, self.MOVE, self.phase)
    self.k = torch.where(go, torch.zeros_like(self.k), self.k)
    arm = self.hold.clone()
    for i in torch.nonzero(handed & (self.phase == self.MOVE)).flatten().tolist():
      k, path = int(self.k[i]), self.paths[i]
      if k == self.n_rise[i]:
        self.rise_z1[i] = tcp_z[i]
      if k >= len(path):
        self.phase[i] = self.HOME
        continue
      arm[i] = path[k]
      self.k[i] = k + 1
    full = self.offset.clone()
    full[:, self.arm_idx] = arm
    raw = ((full - self.offset) / self.scale).clamp(-1.0, 1.0)
    raw[:, self.jaw_idx] = 0.0  # the default jaw target: open
    home = self.phase == self.HOME
    return torch.where(home.unsqueeze(-1), torch.zeros_like(raw), raw)

  def finish(self, env_ids: torch.Tensor) -> None:
    for i in env_ids.tolist():
      z0, z1 = float(self.rise_z0[i]), float(self.rise_z1[i])
      self.done.append(
        {
          "rise": (z1 - z0) if not (math.isnan(z0) or math.isnan(z1)) else None,
          "path_ik_ok": bool(self.ik_ok[i]) if not math.isnan(z0) else None,
        }
      )
      self.paths[i] = None
    self.phase[env_ids] = self.OPEN
    self.active[env_ids] = False
    self.k[env_ids] = 0
    self.hold[env_ids] = 0.0
    self.rise_z0[env_ids] = float("nan")
    self.rise_z1[env_ids] = float("nan")
    self.ik_ok[env_ids] = True

  def summarise(self, episodes: int) -> dict:
    rows = self.done[:episodes]
    rises = sorted(r["rise"] for r in rows if r["rise"] is not None)
    oks = [r["path_ik_ok"] for r in rows if r["path_ik_ok"] is not None]
    return {
      "handoff_rise_median_m": rises[len(rises) // 2] if rises else None,
      "handoff_rise_min_m": rises[0] if rises else None,
      "handoff_path_ik_ok": sum(oks) / len(oks) if oks else None,
    }


def retreat_state(
  env: ManagerBasedRlEnv, arm_ids: list[int], jaw_id: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """The jaw's measured angle, the TCP's height above the cube's top, and the driven arm's
  max |q - q_default|, for `RetreatTally.step`. (B,) each"""
  robot, obj = env.scene[ROBOT], env.scene[OBJECT]
  q = robot.data.joint_pos
  half_z = env.sim.model.geom_size[:, obj.indexing.geom_ids[0], 2]
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
  if getattr(args, "plant", False):
    # The support sensor records contact only (`found`); the set-down read needs its force
    # too. A sensor field is read-only on the physics, and `found` -- what every other read
    # here uses -- is unchanged.
    import dataclasses

    cfg.scene.sensors = tuple(
      dataclasses.replace(s, fields=(*s.fields, "force"))
      if s.name == SUPPORT_SENSOR and "force" not in s.fields
      else s
      for s in cfg.scene.sensors
    )
  return ManagerBasedRlEnv(cfg=cfg, device=args.device)


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
  return {
    "episodes": n,
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
      f"    jaw opened (< {JAW_OPEN} rad)   {result['jaw_released']:6.1%}   after the lift"
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
  retreat = RetreatTally(args.num_envs, args.device, env.step_dt, args.handoff)
  occ = OccTally(args.num_envs, args.device)
  lift = VerticalRetreat(env) if args.handoff and not args.handoff_direct else None
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
      jaw_q, tcp_above, home_err = retreat_state(env, arm_ids, jaw_id)
      handed = retreat.step(
        jaw_q,
        flags["grasped"],
        flags["lifted"],
        flags["placed_and_lifted"] & (jaw_q < JAW_OPEN),
        tcp_above,
        obj_pos[:, :2],
        dxy,
        home_err,
        placed_jz=flags["placed_and_lifted_z"] & (jaw_q < JAW_OPEN),
        tilt=tilt(env),
      )
      if plant is not None:
        plant.measure_env(
          carrying=flags["lifted"],
          setdown=flags["grasped"] & (dxy < NEAR_XY),
          after_let_go=retreat.let_go,
        )
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
      if args.handoff:
        # The position controller, through the action term's slew (and its tool cap and
        # bounds, where the run has them): open, rise, cross, come down home (`VerticalRetreat`);
        # with --handoff-direct, action 0 at once (the home pose with the jaw open).
        ctrl = (
          torch.zeros_like(actions)
          if lift is None
          else lift.action(handed, jaw_q, positions(env)[1][:, 2])
        )
        actions = torch.where(handed.unsqueeze(-1), ctrl, actions)
      obs, _, dones, extras = wrapped.step(actions)
      tally.terminal(success_flags(env))
      finished = torch.nonzero(dones, as_tuple=False).flatten()
      if finished.numel():
        n_before = len(tally.done)
        tally.finish(finished)
        retreat.finish(
          finished, [r["placed_and_lifted"] for r in tally.done[n_before:]]
        )
        if occ.steps.any() or occ.done:
          occ.finish(finished)
        if lift is not None:
          lift.finish(finished)
        if plant is not None:
          plant.finish(finished)
        for key, value in extras.get("log", {}).items():
          if key.startswith("Episode_Reward/"):
            reward_sums.setdefault(key.removeprefix("Episode_Reward/"), []).append(
              float(value)
            )

  result = summarise(tally, env.step_dt, args.episodes)
  result.update(retreat.summarise(args.episodes))
  if occ.done:
    result.update(occ.summarise(args.episodes))
  result["handoff"] = bool(args.handoff)
  result["handoff_return"] = (
    None if not args.handoff else ("direct" if lift is None else "rise")
  )
  if lift is not None:
    result.update(lift.summarise(args.episodes))
  if plant is not None:
    result.update(plant.summarise(args.episodes))
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
          **result,
          "reward_terms": rewards,
          "episodes_detail": [
            {**r, **q, **o}
            for r, q, o in zip(
              tally.done[: args.episodes],
              retreat.done[: args.episodes],
              occ.done[: args.episodes] or [{}] * args.episodes,
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
