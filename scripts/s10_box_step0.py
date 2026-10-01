"""Plan S10 step 0, items 3 and 5: the scripted pick-and-place into the box (L1) and the finger fit.

Runs `scripted_pick_place.rollout` on `BS` + ``GOAL=box`` + ``ARM_TABLE=AC`` (G20, CTP, stack speeds,
``LAG=JS`` -- the script cannot see the cube through the tracker's lag, as in S7-8 / S8's step 0 --,
baseline B, W2, DR-scene) with the box at its target box's centre and four corners (yaw alternating
+/-5 deg) and the cube at W2's nine corner poses: 45 envs. ``--yaw-error`` turns the cube that far off
the box's axes at the place (the finger-fit check); ``--pad-clearance`` is the pads' height above the
table at the grasp. Prints the per-env rows and writes them to ``--json``.

  uv run --extra cpu python scripts/s10_box_step0.py --json ../results/S10-box/step0/corners.json
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import scripted_pick_place as scripted  # noqa: E402

RCTP = "R-C-strict+M1+M2+M3+M6+J+M8"


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--json", type=pathlib.Path, required=True)
  ap.add_argument("--goal", default="box")
  ap.add_argument("--yaw-error", type=float, default=0.0, help="deg")
  ap.add_argument(
    "--pad-clearance",
    type=float,
    default=0.05,
    help="fingertips above the table before the close, m: 5 cm holds the cube at its middle (S10 step 0)",
  )
  ap.add_argument("--spawns", type=int, default=9, help="W2 corner poses used (<= 9)")
  ap.add_argument("--boxes", type=int, default=5, help="box poses used (<= 5)")
  ap.add_argument("--dr", default="DR-scene")
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()

  from pal_mjlab.tasks.manipulation.pick_place import pick_place_env_cfg as pp
  from pal_mjlab.tasks.manipulation.pick_place.arm_table import ARM_TABLE_CODES
  from pal_mjlab.tasks.manipulation.pick_place.box import BOX_GOALS, BOX_YAW
  from pal_mjlab.tasks.manipulation.pick_place.kangaroo.env_cfg import SPAWN_ROBOTS
  from pal_mjlab.tasks.manipulation.robots import kangaroo_grippers_left_arm
  from pal_mjlab.tasks.manipulation.tabletop_env_cfg import LATENCY_MODELS

  b = pp.BASELINES["B"]
  (tx, ty), h = b.target_xy, b.target_half_xy
  boxes = [
    ((tx, ty), 0.0),
    ((tx - h, ty - h), BOX_YAW),
    ((tx - h, ty + h), -BOX_YAW),
    ((tx + h, ty - h), -BOX_YAW),
    ((tx + h, ty + h), BOX_YAW),
  ][: args.boxes]
  spawns = scripted.box_corners(pp.SPAWN_BOXES["W2"])[: args.spawns]
  cases = [(s, bx) for bx in boxes for s in spawns]
  n = len(cases)
  rows = scripted.rollout(
    num_envs=n,
    device="cpu",
    seed=args.seed,
    reward=RCTP,
    verbose=True,
    robot=kangaroo_grippers_left_arm("G20", SPAWN_ROBOTS["W2"], arm_effort_scale=0.7),
    baseline=b,
    spawn_box=pp.SPAWN_BOXES["W2"],
    dr=args.dr,
    effort=pp.EFFORT_BOUNDS["CTP"],
    latency=LATENCY_MODELS["JS"],
    stack_speed=True,
    episode_length_s=120.0,
    spawns=[c[0] for c in cases],
    box_poses=[c[1] for c in cases],
    box_goal=BOX_GOALS[args.goal],
    arm_table=ARM_TABLE_CODES["AC"],
    pad_clearance=args.pad_clearance,
    yaw_errors=[math.radians(args.yaw_error)] * n,
  )
  for r, (sp, bx) in zip(rows, cases, strict=True):
    r["spawn"] = [float(sp[0][0]), float(sp[0][1]), float(sp[1])]
    r["box"] = [float(bx[0][0]), float(bx[0][1]), float(bx[1])]
  ok = [bool(r.get("in_box")) and bool(r["placed_and_lifted"]) for r in rows]
  inb = [bool(r.get("in_box")) for r in rows]
  print(
    f"\nGOAL={args.goal} yaw_error={args.yaw_error} deg pad_clearance={args.pad_clearance} m: "
    f"in the box at the end {sum(inb)}/{n}, placed and lifted {sum(r['placed_and_lifted'] for r in rows)}/{n}, "
    f"both {sum(ok)}/{n}; hand on a wall while held in {sum(r.get('hand_wall_held_steps', 0) > 0 for r in rows)}, "
    f"cube on a wall while held in {sum(r.get('cube_wall_held_steps', 0) > 0 for r in rows)}; "
    f"plan errors {sum(r['plan_error'] is not None for r in rows)}; "
    f"box moved median {np.median([r.get('box_displacement_mm', 0) for r in rows]):.1f} mm"
  )
  args.json.parent.mkdir(parents=True, exist_ok=True)
  args.json.write_text(
    json.dumps(
      {"args": vars(args) | {"json": str(args.json)}, "rows": rows},
      indent=1,
      default=str,
    )
  )


if __name__ == "__main__":
  main()
