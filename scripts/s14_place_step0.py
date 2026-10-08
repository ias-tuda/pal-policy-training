"""Plan §14 step 0 (`pick-place-insert-plan.md` §14.6): the pin picked and placed on the wide pin area WP.

``reach`` (items 2 and 3): an IK map over x 0.32-0.68, y -0.14-0.36 (the robot frame, ``--step`` grid) on the
``GOAL=place`` scene's model at a table height (``TABLE_DZ``, mm). For each cell and start pose, the left arm's
contact-free IK (`scripted_pick_place.Kin`: the tool point at the goal, no arm geom touching the table, the torso, the
right arm or the parked pin) at the grasp, the lift (+5 cm) and the retreat (10 cm over the pin's top) -- the set-down
at a goal is the grasp's tool point again:

* standing: the tool point `insert.GRASP_ABOVE_TOP` over the 90 mm pin's top, the wrist at any yaw (a cross is
  28-32 mm across in every direction, so the jaw closes on it at any yaw);
* lying: the tool point ``--above-lying`` over the lying "x" pin's 28 mm top, straight over its centre; axis 0 / 45 /
  90 / 135 deg from x; the jaw closing across the axis (within `JAW_TOL_DEG`).

The IK (`_soft_ik`) holds the position exactly and the hand's vertical (and the jaw's direction) softly, from the
start pose and random seeds, and keeps the least-tilted contact-free solution. Per cell and pose: ``top`` (every axis
with the hand within `TOP_DOWN_DEG` of vertical through grasp, lift and retreat), ``tilt`` (within `MAX_TILT_DEG`),
``some`` (some axes not within it), ``none``. Item 3: the largest excursion of each arm joint from the start pose over every
solution, x 1.5 (S5.0's rule), against the WC arm's action scales.

  uv run --extra cu128 python scripts/s14_place_step0.py reach --table-dz 22 --json ../results/S14-pin-place/step0/reach_dz22.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

#: `BS12` + PS (plan §14.4), the selectors every subcommand builds its env or scene with.
PS_ENV = {
  "REWARD": "R-C-strict+M1+M2+M3+M6+J+M8",
  "DR": "DR-scene",
  "PPO": "PPO-stack",
  "SLEW": "stack",
  "BASELINE": "B",
  "SPAWN": "WP",
  "TOOL_SPEED": "none",
  "GAINS": "G20",
  "EFFORT": "CTP",
  "LAG": "FP+JSW",
  "ACTION": "abs",
  "CMD_OBS": "act",
  "TARGET": "WP",
  "EPISODE_S": "15",
  "CRITIC_OBS": "q",
  "ARM_TABLE": "AH10",
  "SHAPE": "pin",
  "GOAL": "place",
  "TABLE_DZ": "22",
}
#: The lying "x" pin's top over the table, m: (span + thickness) / 2 / sqrt 2 + the same again.
LYING_TOP = 2 * (0.032 + 0.008) / 2 / math.sqrt(2)
STANDING_TOP = 0.090
LIFT = 0.05
RETREAT = 0.10


def _set_env(**over) -> None:
  for k, v in {**PS_ENV, **over}.items():
    os.environ[k] = str(v)


#: A pose counts as top-down within this tilt of the hand from vertical, as reachable within `MAX_TILT_DEG` (plan
#: §14.1 *Rule*: "the hand within 35 deg of vertical"); a lying pin's jaw must close within `JAW_TOL_DEG` of across
#: its axis.
TOP_DOWN_DEG = 5.0
MAX_TILT_DEG = 35.0
JAW_TOL_DEG = 10.0
#: Seeds per IK: the start pose and draws around it (0.3-1.2 rad).
SEEDS = 12


def _soft_ik(kin, goal, axis, seed, iters=400, w=0.3):
  """Damped least squares on the tool point's position (exact) and, softly (weight ``w``), the hand's z straight down
  and -- with a lying pin's ``axis`` (horizontal unit vector) -- the hand's x (the jaw's closing direction) across it;
  the wrist yaw is otherwise free (a standing cross is gripped at any yaw). Returns (q, position error m, hand tilt
  deg, jaw error deg)."""
  import mujoco

  m = kin.model
  jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
  zdes = np.array([0.0, 0.0, -1.0])
  q = seed.copy()
  for _ in range(iters):
    p, r = kin.fk(q)
    mujoco.mj_jacSite(m, kin.data, jacp, jacr, kin.tcp)
    jr = jacr[:, kin.dofs]
    rows = [jacp[:, kin.dofs], w * jr]
    err = [goal - p, w * np.cross(r[:, 2], zdes)]
    if axis is not None:
      rows.append(w * (np.cross(r[:, 0], axis) @ jr)[None, :])
      err.append(np.array([-w * float(r[:, 0] @ axis)]))
    jac, e = np.vstack(rows), np.concatenate(err)
    pinv = jac.T @ np.linalg.inv(jac @ jac.T + 1e-3 * np.eye(len(e)))
    # What the task leaves free is spent pulling toward the start pose (`Kin.ik`'s rule): the least excursion, what
    # the action scales are sized from.
    dq = pinv @ e + (np.eye(len(q)) - pinv @ jac) @ (0.1 * (kin.q_default - q))
    q = np.clip(q + np.clip(dq, -0.2, 0.2), kin.lo, kin.hi)
  p, r = kin.fk(q)
  tilt = math.degrees(math.acos(float(np.clip(-r[2, 2], -1.0, 1.0))))
  jaw = (
    0.0
    if axis is None
    else math.degrees(math.asin(min(1.0, abs(float(r[:, 0] @ axis)))))
  )
  return q, float(np.linalg.norm(goal - p)), tilt, jaw


def _best(kin, goal, axis, seeds):
  """The contact-free solution (position within 2 mm, the jaw within `JAW_TOL_DEG`) with the hand held down hardest:
  the vertical's weight is relaxed 0.3 -> 0.1 -> 0.03 only where the stricter one cannot reach; the least-tilted
  solution at the first weight that has one, or None."""
  for w in (0.3, 0.1, 0.03):
    best = None
    for s0 in seeds:
      q, err, tilt, jaw = _soft_ik(kin, goal, axis, s0, w=w)
      if err > 2e-3 or jaw > JAW_TOL_DEG or kin.in_collision(q):
        continue
      if best is None or tilt < best[1]:
        best = (q, tilt)
        if tilt < 1.0:
          return best
    if best is not None:
      return best
  return None


def reach(args: argparse.Namespace) -> dict:
  _set_env(TABLE_DZ=args.table_dz)
  import mujoco
  import scripted_pick_place as scripted
  from pal_mjlab.tasks.manipulation.pick_place import insert as ins
  from scene_model import compile_scene

  model = compile_scene(scripted.TASK)
  kin = scripted.Kin.from_model(model)
  if args.box:
    # Item 3: the IK held inside the named arm's action box -- the start pose +/- its action scale --, so a cell
    # reached here is reached through that arm's actions.
    from pal_mjlab.tasks.manipulation.pick_place.kangaroo.env_cfg import SPAWN_ROBOTS

    sc = SPAWN_ROBOTS[args.box].action_scale
    half = np.array([sc[f"arm_left_{j}_joint"] for j in range(1, 8)])
    kin.lo = np.maximum(kin.lo, kin.q_default - half)
    kin.hi = np.minimum(kin.hi, kin.q_default + half)
  table = model.geom("table/table_top").id
  d = mujoco.MjData(model)
  mujoco.mj_forward(model, d)
  top = float(d.geom_xpos[table][2] + model.geom_size[table][2])
  rng = np.random.default_rng(args.seed)
  xs = np.round(np.arange(args.x[0], args.x[1] + 1e-9, args.step), 3)
  ys = np.round(np.arange(args.y[0], args.y[1] + 1e-9, args.step), 3)
  cells = [(float(x), float(y)) for x in xs for y in ys][args.shard :: args.shards]
  out = {
    "table_top": top,
    "above_lying": args.above_lying,
    "box": args.box,
    "cells": [],
  }
  if args.json.exists():
    out = json.loads(args.json.read_text())
  done = {(c["x"], c["y"]) for c in out["cells"]}
  t0 = time.time()
  for n, (x, y) in enumerate(cells):
    if (x, y) in done:
      continue
    row = {"x": x, "y": y}
    sols = []
    for pose in ("standing", "lying"):
      if pose == "standing":
        z = top + STANDING_TOP + ins.GRASP_ABOVE_TOP
        zs = (z, z + LIFT, top + STANDING_TOP + RETREAT)
        axes = [None]
      else:
        z = top + LYING_TOP + args.above_lying
        zs = (z, z + LIFT, top + LYING_TOP + RETREAT)
        axes = [
          np.array([math.cos(math.radians(a)), math.sin(math.radians(a)), 0.0])
          for a in (0.0, 45.0, 90.0, 135.0)
        ]
      tilts = []
      for axis in axes:
        seeds = [kin.q_default] + [
          np.clip(kin.q_default + rng.normal(0.0, 0.3 + 0.08 * k, 7), kin.lo, kin.hi)
          for k in range(SEEDS - 1)
        ]
        chain, worst = [], 0.0
        for k, zz in enumerate(zs):
          got = _best(
            kin,
            np.array([x, y, zz]),
            axis,
            seeds if k == 0 else [chain[-1]] + seeds[:3],
          )
          if got is None:
            chain = None
            break
          chain.append(got[0])
          worst = max(worst, got[1])
        tilts.append(None if chain is None else round(worst, 1))
        if chain is not None and worst <= MAX_TILT_DEG:
          sols.extend(chain)
      row[pose] = tilts
      ok = [t for t in tilts if t is not None and t <= MAX_TILT_DEG]
      row[pose + "_class"] = (
        "none"
        if not ok
        else "some"
        if len(ok) < len(tilts)
        else "top"
        if max(ok) <= TOP_DOWN_DEG
        else "tilt"
      )
    if sols:
      dev = np.abs(np.asarray(sols) - kin.q_default).max(0)
      row["max_dev"] = [round(float(v), 4) for v in dev]
    out["cells"].append(row)
    if n % 10 == 0:
      args.json.parent.mkdir(parents=True, exist_ok=True)
      args.json.write_text(json.dumps(out))
      print(f"{n + 1} / {len(cells)} cells, {time.time() - t0:.0f} s", flush=True)
  args.json.write_text(json.dumps(out))
  return out


def summarize(args: argparse.Namespace) -> None:
  """Item 2's table and item 3's scales from one or more ``reach`` JSONs (shards merged)."""
  from pal_mjlab.tasks.manipulation.robots.kangaroo import (
    KANGAROO_GRIPPERS_LEFT_ARM_WC,
  )

  cells, top = [], None
  for f in args.files:
    r = json.loads(pathlib.Path(f).read_text())
    cells += r["cells"]
    top = r["table_top"]
  wp = [c for c in cells if 0.36 <= c["x"] <= 0.64 and -0.10 <= c["y"] <= 0.32]
  print(f"table top {top:.3f} m; {len(cells)} cells, {len(wp)} on WP")
  for pose in ("standing", "lying"):
    for name, part in (("grid", cells), ("WP", wp)):
      k = {
        c: sum(1 for x in part if x[pose + "_class"] == c)
        for c in ("top", "tilt", "some", "none")
      }
      print(f"  {pose:8s} {name:4s}: " + "  ".join(f"{c} {v}" for c, v in k.items()))
    bad = sorted(
      (c["x"], c["y"], c[pose + "_class"])
      for c in wp
      if c[pose + "_class"] in ("some", "none")
    )
    if bad:
      print(
        f"    WP cells not every {pose} case reaches: {bad[:40]}{' ...' if len(bad) > 40 else ''}"
      )
  both_none = [
    (c["x"], c["y"])
    for c in wp
    if c["standing_class"] == "none" and c["lying_class"] == "none"
  ]
  print(f"  WP cells no pose reaches (the cut rule's (a)): {both_none}")
  scales = KANGAROO_GRIPPERS_LEFT_ARM_WC.action_scale
  dev = np.max([c["max_dev"] for c in wp if "max_dev" in c], axis=0)
  print("  item 3: joint, max |dq| over WP's solutions, x1.5, WC scale")
  for j, v in enumerate(dev):
    s = scales[f"arm_left_{j + 1}_joint"]
    flag = "" if 1.5 * v <= s + 1e-9 else "  > WC"
    print(f"    arm {j + 1}: {v:.3f}  {1.5 * v:.3f}  {s:.2f}{flag}")


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  sub = p.add_subparsers(dest="cmd", required=True)
  r = sub.add_parser("reach")
  r.add_argument("--table-dz", default="22")
  r.add_argument("--json", type=pathlib.Path, required=True)
  r.add_argument("--x", type=float, nargs=2, default=(0.32, 0.68))
  r.add_argument("--y", type=float, nargs=2, default=(-0.14, 0.36))
  r.add_argument("--step", type=float, default=0.02)
  r.add_argument("--above-lying", type=float, default=0.020)
  r.add_argument("--shard", type=int, default=0)
  r.add_argument("--shards", type=int, default=1)
  r.add_argument("--seed", type=int, default=0)
  r.add_argument(
    "--box",
    default=None,
    help="hold the IK inside this SPAWN's arm's action box (item 3)",
  )
  s = sub.add_parser("summarize")
  s.add_argument("files", nargs="+")
  args = p.parse_args()
  if args.cmd == "reach":
    reach(args)
  else:
    _set_env()
    summarize(args)


if __name__ == "__main__":
  main()
