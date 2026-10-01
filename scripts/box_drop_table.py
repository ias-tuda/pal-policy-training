"""Plan S10 step 0: where does a 5 cm cube end up when let go above the open box?

A standalone CPU MuJoCo measurement, no robot: one scene built from the task's own pieces
(`scene.get_table_spec`, `get_box_spec`, `get_cube_spec`) with the RL env's solver settings
(`tabletop_env_cfg.py`'s `MujocoCfg`, plus mjlab's defaults for the fields it leaves unset). The
box rests on the table centre; the cube is released (free, no spin) at a pose relative to the box
floor centre, simulated 2.0 s, and classified as inside / on_rim / outside / other.

Grid (the plan's axes): xy offset of the cube centre along +x and along the (1,1) diagonal, yaw
error, tilt about the world x axis, height of the cube's lowest point above the rim (h = -2 cm is
the "set-down" level: bottom 5 mm above the floor, already inside; poses that intersect a wall
there are marked infeasible), and downward release speed. Each cell runs N trials whose cube
sliding friction and mass are drawn once from a fixed seed and reused in every cell (common random
numbers, Latin-hypercube stratified), so cell-to-cell differences come from the pose alone.

  uv run --extra cpu python scripts/box_drop_table.py              # hand checks + full sweep
  uv run --extra cpu python scripts/box_drop_table.py --checks-only

Writes cells.csv, trials.csv and README.md to --out (default ../results/S10-box/step0/drop_table).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import platform
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np
from pal_mjlab.tasks.manipulation import scene

TABLE_TOP_Z = scene.ROBOT_CELL_TABLE_TOP_Z
BOX_XY = (0.55, 0.0)  # the table centre, scene.table_pos
HALF = 0.025  # cube half-size, get_cube_spec's default
INNER_HALF = scene.BOX_INNER / 2  # 0.04
FLOOR_LOCAL = scene.BOX_FLOOR  # floor top, box frame (origin = bottom face centre)
RIM_LOCAL = scene.BOX_FLOOR + scene.BOX_WALL_HEIGHT  # 0.030
SIM_S = 2.0
TIMESTEP = 0.002
NOMINAL_MASS = 0.05
FRICTION_RANGE = (0.5, 1.2)  # tabletop_env_cfg._DR_OBJECT_FRICTION
MASS_RANGE = (
  0.03,
  0.10,
)  # tabletop_env_cfg._DR_OBJECT_MASS_KG, drawn uniform in kg here
SEED = 20261001
SPEED_TOL = 0.02  # m/s
RIM_CONTACT_TOL = (
  0.004  # a cube-wall contact this close to the wall top (m) is "on the rim"
)

OFFSETS_CM = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
YAWS_DEG = (0, 5, 10, 20, 30, 45)
TILTS_DEG = (0, 10, 20, 30)
HS_CM = (-2.0, 0.0, 0.5, 1.0, 2.0)
V_DOWN = (0.0, 0.10)

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "results/S10-box/step0/drop_table"


# --------------------------------------------------------------------------- model


def build_model() -> mujoco.MjModel:
  world = mujoco.MjSpec()
  world.worldbody.add_geom(
    name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(3.0, 3.0, 0.1), condim=3
  )
  attach = (
    (scene.get_table_spec(TABLE_TOP_Z), scene.table_pos(TABLE_TOP_Z)),
    (scene.get_box_spec(name="box"), (BOX_XY[0], BOX_XY[1], TABLE_TOP_Z)),
    # The cube is parked on the table, 30 cm to the side, while the box settles.
    (scene.get_cube_spec(name="object"), (BOX_XY[0], 0.30, TABLE_TOP_Z + HALF)),
  )
  for child, pos in attach:
    frame = world.worldbody.add_frame(pos=pos)
    world.attach(child, frame=frame, prefix="")
  m = world.compile()
  # tabletop_env_cfg.py's MujocoCfg (timestep, iterations 10, ls_iterations 20, impratio 10,
  # elliptic) plus mjlab's MujocoCfg defaults for the rest (implicitfast, newton, tol 1e-8,
  # ls_tolerance 0.01, ccd_iterations 50, auto jacobian).
  m.opt.timestep = TIMESTEP
  m.opt.iterations = 10
  m.opt.ls_iterations = 20
  m.opt.impratio = 10
  m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
  m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  m.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
  m.opt.tolerance = 1e-8
  m.opt.ls_tolerance = 0.01
  m.opt.ccd_iterations = 50
  m.opt.jacobian = mujoco.mjtJacobian.mjJAC_AUTO
  return m


class Sim:
  """One model + data per process; the box's settled state is the start of every trial."""

  def __init__(self):
    self.m = build_model()
    self.d = mujoco.MjData(self.m)
    m = self.m
    self.cube_bid = m.body("object").id
    self.box_bid = m.body("box").id
    self.cube_gid = m.geom("object_geom").id
    self.cube_q = m.jnt_qposadr[m.joint("object_joint").id]
    self.cube_v = m.jnt_dofadr[m.joint("object_joint").id]
    self.box_q = m.jnt_qposadr[m.joint("box_joint").id]
    self.box_floor_gid = m.geom("box_floor").id
    self.wall_gids = {m.geom(n).id for n in scene.box_wall_geom_names("box")}
    self.box_gids = self.wall_gids | {self.box_floor_gid}
    self.inertia0 = m.body_inertia[self.cube_bid].copy()
    self.friction0 = m.geom_friction[self.cube_gid].copy()
    # Settle the box (and the parked cube) for 1 s.
    mujoco.mj_resetData(m, self.d)
    mujoco.mj_step(m, self.d, nstep=500)
    self.q0 = self.d.qpos.copy()
    mujoco.mj_forward(m, self.d)
    self.box_pos0 = self.d.xpos[self.box_bid].copy()
    self.box_R0 = self.d.xmat[self.box_bid].reshape(3, 3).copy()

  def set_cube_params(self, friction: float, mass: float) -> None:
    """DR-object: sliding friction, and mass with inertia scaled alike (dr.pseudo_inertia)."""
    m = self.m
    m.geom_friction[self.cube_gid] = self.friction0
    m.geom_friction[self.cube_gid, 0] = friction
    m.body_mass[self.cube_bid] = mass
    m.body_inertia[self.cube_bid] = self.inertia0 * (mass / NOMINAL_MASS)
    mujoco.mj_setConst(m, self.d)

  def place(self, pos_local, R_local, v_down: float) -> bool:
    """Reset to the settled box and put the cube at a box-frame pose. Returns feasible."""
    m, d = self.m, self.d
    mujoco.mj_resetData(m, d)
    d.qpos[:] = self.q0
    d.qvel[:] = 0.0
    p = self.box_pos0 + self.box_R0 @ np.asarray(pos_local)
    R = self.box_R0 @ R_local
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.reshape(-1))
    d.qpos[self.cube_q : self.cube_q + 3] = p
    d.qpos[self.cube_q + 3 : self.cube_q + 7] = quat
    d.qvel[self.cube_v : self.cube_v + 6] = (0.0, 0.0, -v_down, 0.0, 0.0, 0.0)
    mujoco.mj_forward(m, d)
    for c in d.contact[: d.ncon]:
      pair = {c.geom1, c.geom2}
      if self.cube_gid in pair and pair & self.box_gids and c.dist < 0.0:
        return False
    return True

  def run_and_classify(self, nstep: int = int(round(SIM_S / TIMESTEP))) -> dict:
    m, d = self.m, self.d
    mujoco.mj_step(m, d, nstep=nstep)
    box_p = d.xpos[self.box_bid]
    box_R = d.xmat[self.box_bid].reshape(3, 3)
    local = box_R.T @ (d.xpos[self.cube_bid] - box_p)
    speed = float(np.linalg.norm(d.qvel[self.cube_v : self.cube_v + 3]))
    touch_floor = False
    on_rim_contact = False
    for c in d.contact[: d.ncon]:
      if self.cube_gid not in (c.geom1, c.geom2):
        continue
      other = c.geom2 if c.geom1 == self.cube_gid else c.geom1
      if other == self.box_floor_gid:
        touch_floor = True
      elif other in self.wall_gids:
        cz = (box_R.T @ (c.pos - box_p))[2]
        if cz >= RIM_LOCAL - RIM_CONTACT_TOL:
          on_rim_contact = True
    in_footprint = abs(local[0]) < INNER_HALF and abs(local[1]) < INNER_HALF
    inside = (
      in_footprint
      and local[2] < RIM_LOCAL + 0.5 * HALF
      and (touch_floor or local[2] < FLOOR_LOCAL + HALF + 0.012)
      and speed < SPEED_TOL
    )
    if inside:
      outcome = "inside"
    elif on_rim_contact:
      outcome = "on_rim"
    elif not in_footprint:
      outcome = "outside"
    else:
      outcome = "other"  # over the inner footprint but moving, or wedged high
    tilt = math.degrees(math.acos(float(np.clip(box_R[2, 2], -1.0, 1.0))))
    disp = float(np.linalg.norm(box_p[:2] - self.box_pos0[:2]))
    return {
      "outcome": outcome,
      "cube_x": float(local[0]),
      "cube_y": float(local[1]),
      "cube_z": float(local[2]),
      "speed": speed,
      "box_disp_mm": 1000.0 * disp,
      "box_tilt_deg": tilt,
      "nan": bool(np.any(~np.isfinite(d.qpos))),
    }


def cube_rotation(yaw_deg: float, tilt_deg: float) -> np.ndarray:
  """Yaw about z first, then tilt about the (world = box) x axis."""
  y, t = math.radians(yaw_deg), math.radians(tilt_deg)
  Rz = np.array(
    [[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]]
  )
  Rx = np.array(
    [[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]]
  )
  return Rx @ Rz


def cube_pose(offset_cm, direction, yaw_deg, tilt_deg, h_cm):
  """Box-frame centre and rotation, with the cube's lowest corner h above the rim."""
  R = cube_rotation(yaw_deg, tilt_deg)
  u = np.array([1.0, 0.0]) if direction == "x" else np.array([1.0, 1.0]) / math.sqrt(2)
  xy = 0.01 * offset_cm * u
  z = RIM_LOCAL + 0.01 * h_cm + HALF * float(np.abs(R[2]).sum())
  return np.array([xy[0], xy[1], z]), R


# --------------------------------------------------------------------------- sweep

_SIM: Sim | None = None


def _init_worker():
  global _SIM
  _SIM = Sim()


def draw_params(n: int) -> np.ndarray:
  """(n, 2) friction, mass: Latin-hypercube stratified uniform draws from SEED."""
  rng = np.random.default_rng(SEED)
  cols = []
  for lo, hi in (FRICTION_RANGE, MASS_RANGE):
    u = (rng.permutation(n) + rng.uniform(size=n)) / n
    cols.append(lo + (hi - lo) * u)
  return np.stack(cols, axis=1)


def run_cell(args):
  cell, params = args
  sim = _SIM
  pos, R = cube_pose(
    cell["offset_cm"],
    cell["direction"],
    cell["yaw_deg"],
    cell["tilt_deg"],
    cell["h_cm"],
  )
  trials = []
  if not sim.place(pos, R, cell["v_down"]):
    return cell, None
  for i, (mu, mass) in enumerate(params):
    sim.set_cube_params(mu, mass)
    sim.place(pos, R, cell["v_down"])
    r = sim.run_and_classify()
    r.update(trial=i, friction=float(mu), mass=float(mass))
    trials.append(r)
  return cell, trials


def make_cells(offsets, hs):
  cells = []
  dirs = [("x", o) for o in offsets] + [("diag", o) for o in offsets if o > 0]
  for (direction, off), yaw, tilt, h, v in itertools.product(
    dirs, YAWS_DEG, TILTS_DEG, hs, V_DOWN
  ):
    cells.append(
      dict(
        offset_cm=off, direction=direction, yaw_deg=yaw, tilt_deg=tilt, h_cm=h, v_down=v
      )
    )
  return cells


def summarise(cell, trials):
  row = dict(cell)
  if trials is None:
    row.update(
      n=0,
      inside_share="",
      on_rim_share="",
      outside_share="",
      other_share="",
      infeasible=1,
      median_box_disp_mm="",
      box_tipped_share="",
      nan_trials="",
    )
    return row
  n = len(trials)
  share = lambda k: sum(t["outcome"] == k for t in trials) / n  # noqa: E731
  row.update(
    n=n,
    inside_share=round(share("inside"), 4),
    on_rim_share=round(share("on_rim"), 4),
    outside_share=round(share("outside"), 4),
    other_share=round(share("other"), 4),
    infeasible=0,
    median_box_disp_mm=round(float(np.median([t["box_disp_mm"] for t in trials])), 3),
    box_tipped_share=round(sum(t["box_tilt_deg"] > 5.0 for t in trials) / n, 4),
    nan_trials=sum(t["nan"] for t in trials),
  )
  return row


# --------------------------------------------------------------------------- hand checks


def hand_checks(params) -> list[tuple[str, str, dict]]:
  sim = Sim()
  I = np.eye(3)
  # (name, box-frame centre, expectation, steps); expectation "inside", "not inside", "outside",
  # "on_rim". (b) balances the cube's centre over the 5 mm wall, a knife edge it can fall off
  # either way, so (b2) checks the on_rim label on a cube resting on the rim (one step, no time to
  # tip) and (b3) releases it with its centre 1.25 cm beyond the wall's outer face.
  full = int(round(SIM_S / TIMESTEP))
  cases = [
    (
      "(a) centred, aligned, bottom 5 mm above floor",
      np.array([0, 0, FLOOR_LOCAL + 0.005 + HALF]),
      "inside",
      full,
    ),
    (
      "(b) centred on +x wall top (x=0.0425), 1 cm above rim",
      np.array([0.0425, 0, RIM_LOCAL + 0.01 + HALF]),
      "not inside",
      full,
    ),
    (
      "(b2) same x, resting on the rim, classified after 1 step",
      np.array([0.0425, 0, RIM_LOCAL + HALF]),
      "on_rim",
      1,
    ),
    (
      "(b3) centre x=0.0575 (beyond the wall), 1 cm above rim",
      np.array([0.0575, 0, RIM_LOCAL + 0.01 + HALF]),
      "not inside",
      full,
    ),
    (
      "(c) 15 cm from the box, on the table (5 mm up)",
      np.array([0.15, 0, 0.005 + HALF]),
      "outside",
      full,
    ),
  ]
  out = []
  sets = [("nominal mu 0.9, 50 g", 0.9, NOMINAL_MASS)] + [
    (f"draw {i}: mu {p[0]:.2f}, {1000 * p[1]:.0f} g", p[0], p[1])
    for i, p in enumerate(params)
  ]
  for name, pos, expect, nstep in cases:
    results = []
    for _label, mu, mass in sets:
      sim.set_cube_params(mu, mass)
      feasible = sim.place(pos, I, 0.0)
      r = sim.run_and_classify(nstep)
      r["feasible"] = feasible
      results.append(r)
    out.append((name, expect, results))
  return out, [s[0] for s in sets]


# --------------------------------------------------------------------------- tables


def lookup(rows, **kw):
  for r in rows:
    if all(r[k] == v for k, v in kw.items()):
      return r
  return None


def fmt_share(r):
  if r is None:
    return "--"
  if r["infeasible"]:
    return "inf."
  return f"{r['inside_share']:.2f}"


def table_offset_yaw(rows, offsets):
  lines = ["| offset (cm) | dir | " + " | ".join(f"yaw {y}" for y in YAWS_DEG) + " |"]
  lines.append("|---|---|" + "---|" * len(YAWS_DEG))
  for direction in ("x", "diag"):
    for off in offsets:
      if direction == "diag" and off == 0:
        continue
      d = "x" if off == 0 else direction
      cells = [
        fmt_share(
          lookup(
            rows,
            offset_cm=off,
            direction=d,
            yaw_deg=y,
            tilt_deg=0,
            h_cm=0.5,
            v_down=0.10,
          )
        )
        for y in YAWS_DEG
      ]
      lines.append(
        f"| {off:.1f} | {'-' if off == 0 else direction} | " + " | ".join(cells) + " |"
      )
  return "\n".join(lines)


def table_tilt_h(rows, hs):
  lines = ["| tilt (deg) | " + " | ".join(f"h {h:+.1f} cm" for h in hs) + " |"]
  lines.append("|---|" + "---|" * len(hs))
  for tilt in TILTS_DEG:
    cells = [
      fmt_share(
        lookup(
          rows,
          offset_cm=1.0,
          direction="x",
          yaw_deg=0,
          tilt_deg=tilt,
          h_cm=h,
          v_down=0.10,
        )
      )
      for h in hs
    ]
    lines.append(f"| {tilt} | " + " | ".join(cells) + " |")
  return "\n".join(lines)


def table_l13(rows, offsets):
  off = min(offsets, key=lambda o: abs(o - 2.1))
  lines = [
    f"L13 let-go (2.1 cm off, 22 deg tilt, 0.9 cm up, 0.11 m/s) -> nearest cell: offset {off} cm, "
    "tilt 20 deg, h 1.0 cm, v 0.10 m/s (yaw and offset direction not given by L13: all shown).",
    "",
    "| dir | yaw | inside | on_rim | outside | other | box disp med (mm) | tipped |",
    "|---|---|---|---|---|---|---|---|",
  ]
  for direction in ("x", "diag"):
    for y in YAWS_DEG:
      r = lookup(
        rows,
        offset_cm=off,
        direction=direction,
        yaw_deg=y,
        tilt_deg=20,
        h_cm=1.0,
        v_down=0.10,
      )
      if r is None or r["infeasible"]:
        lines.append(f"| {direction} | {y} | -- | | | | | |")
        continue
      lines.append(
        f"| {direction} | {y} | {r['inside_share']:.2f} | {r['on_rim_share']:.2f} | "
        f"{r['outside_share']:.2f} | {r['other_share']:.2f} | {r['median_box_disp_mm']:.1f} | "
        f"{r['box_tipped_share']:.2f} |"
      )
  return "\n".join(lines)


# --------------------------------------------------------------------------- main


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
  ap.add_argument("--trials", type=int, default=16)
  ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
  ap.add_argument("--budget-min", type=float, default=40.0)
  ap.add_argument("--checks-only", action="store_true")
  args = ap.parse_args()
  t_start = time.time()

  params16 = draw_params(16)
  checks, set_names = hand_checks(params16)
  check_lines = []
  for name, expect, results in checks:
    outs = [r["outcome"] for r in results]
    if expect == "not inside":
      ok = all(o != "inside" for o in outs)
    else:
      ok = all(o == expect for o in outs)
    r0 = results[0]
    check_lines.append(
      f"- {name}: expect {expect}; got {dict(Counter(outs))} over nominal + {len(outs) - 1} draws -> "
      f"{'PASS' if ok else 'FAIL'} (nominal: cube box-frame xyz "
      f"({r0['cube_x']:.4f}, {r0['cube_y']:.4f}, {r0['cube_z']:.4f}) m, speed {r0['speed']:.4f} m/s)"
    )
  print("Hand checks:")
  print("\n".join(check_lines))
  if args.checks_only:
    return

  # Time estimate on the full grid; reduce per the plan's rule if it exceeds the budget.
  sim = Sim()
  t0 = time.time()
  pos, R = cube_pose(1.0, "x", 10, 10, 0.5)
  for mu, mass in params16[:8]:
    sim.set_cube_params(mu, mass)
    sim.place(pos, R, 0.10)
    sim.run_and_classify()
  per_trial = (time.time() - t0) / 8
  offsets, hs, n_trials = OFFSETS_CM, HS_CM, args.trials
  reductions = []

  def est(offs, nt):
    return len(make_cells(offs, hs)) * nt * per_trial / args.workers / 60.0

  print(
    f"per trial {1000 * per_trial:.1f} ms, {args.workers} workers; full grid "
    f"{len(make_cells(offsets, hs))} cells x {n_trials} -> ~{est(offsets, n_trials):.1f} min"
  )
  if est(offsets, n_trials) > args.budget_min:
    n_trials = 8
    reductions.append("trials 16 -> 8")
    if est(offsets, n_trials) > args.budget_min:
      offsets = tuple(o for o in offsets if o not in (1.5, 2.5))
      reductions.append("dropped 1.5 / 2.5 cm offsets")
  params = params16 if n_trials == 16 else draw_params(n_trials)
  cells = make_cells(offsets, hs)
  print(
    f"running {len(cells)} cells x {n_trials} trials, est. {est(offsets, n_trials):.1f} min"
    + (f" (reduced: {', '.join(reductions)})" if reductions else "")
  )

  t_sweep = time.time()
  rows, trial_rows = [], []
  with Pool(args.workers, initializer=_init_worker) as pool:
    for k, (cell, trials) in enumerate(
      pool.imap(run_cell, [(c, params) for c in cells], chunksize=4)
    ):
      rows.append(summarise(cell, trials))
      for t in trials or []:
        trial_rows.append({**cell, **t})
      if (k + 1) % 200 == 0:
        print(
          f"  {k + 1}/{len(cells)} cells, {time.time() - t_sweep:.0f} s", flush=True
        )
  sweep_s = time.time() - t_sweep
  n_infeasible = sum(r["infeasible"] for r in rows)
  n_run = sum(r["n"] for r in rows)
  print(
    f"cells: expected {len(cells)}, got {len(rows)} ({n_infeasible} infeasible); "
    f"trials run {n_run}, expected {(len(cells) - n_infeasible) * n_trials}"
  )

  args.out.mkdir(parents=True, exist_ok=True)
  cols = [
    "offset_cm",
    "direction",
    "yaw_deg",
    "tilt_deg",
    "h_cm",
    "v_down",
    "n",
    "inside_share",
    "on_rim_share",
    "outside_share",
    "other_share",
    "infeasible",
    "median_box_disp_mm",
    "box_tipped_share",
    "nan_trials",
  ]
  with open(args.out / "cells.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
  tcols = [
    "offset_cm",
    "direction",
    "yaw_deg",
    "tilt_deg",
    "h_cm",
    "v_down",
    "trial",
    "friction",
    "mass",
    "outcome",
    "cube_x",
    "cube_y",
    "cube_z",
    "speed",
    "box_disp_mm",
    "box_tilt_deg",
    "nan",
  ]
  with open(args.out / "trials.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=tcols, extrasaction="ignore")
    w.writeheader()
    w.writerows(trial_rows)

  t1, t2, t3 = (
    table_offset_yaw(rows, offsets),
    table_tilt_h(rows, hs),
    table_l13(rows, offsets),
  )
  print("\n(1) inside share, offset x yaw (tilt 0, h 0.5 cm, v 0.10 m/s)\n" + t1)
  print(
    "\n(2) inside share, tilt x h (offset 1.0 cm along x, yaw 0, v 0.10 m/s)\n" + t2
  )
  print("\n(3) " + t3)
  total_s = time.time() - t_start
  feasible_rows = [r for r in rows if not r["infeasible"]]
  max_disp = max(r["median_box_disp_mm"] for r in feasible_rows)
  any_tip = sum(r["box_tipped_share"] > 0 for r in feasible_rows)
  nan_total = sum(r["nan_trials"] for r in feasible_rows)
  cmd = "cd pal_mjlab_kangaroo_talos-s10box && uv run --extra cpu python " + " ".join(
    ["scripts/box_drop_table.py"] + sys.argv[1:]
  )
  readme = f"""# S10 step 0 -- cube drop into the box, CPU MuJoCo table

Generated by `scripts/box_drop_table.py` (worktree `pal_mjlab_kangaroo_talos-s10box`, uncommitted) on
{time.strftime("%Y-%m-%d %H:%M")} ({platform.node()}, {platform.platform()}), mujoco {mujoco.__version__}
(CPU, not mujoco_warp).

Command: `{cmd}`

Runtime: {total_s / 60:.1f} min total ({sweep_s / 60:.1f} min sweep, {args.workers} worker processes,
{1000 * per_trial:.1f} ms per 2 s trial single-core).

## Scene
One MjSpec: a floor plane at z=0; `scene.get_table_spec(0.815)` attached at `scene.table_pos(0.815)`
= (0.55, 0, 0.795) (top face z=0.815, table_top friction 0.8/0.005/1e-4); `scene.get_box_spec("box")`
(free, 100 g, inner 8.0 x 8.0 cm, walls 5 mm, floor 5 mm, rim +3.0 cm, condim 3, friction 0.6) with its
origin (bottom-face centre) at (0.55, 0, 0.815), yaw 0; `scene.get_cube_spec("object")` (free 5 cm cube,
condim 4). Attached with `world.attach(child, frame=world.worldbody.add_frame(pos=...), prefix="")`.
The box (with the cube parked on the table 30 cm aside) is settled 1.0 s; every trial restarts
from that settled state with zero velocities and the cube placed in the settled box's frame.

Physics (tabletop_env_cfg.py MujocoCfg + mjlab defaults for unset fields): timestep 0.002,
iterations 10, ls_iterations 20, impratio 10, elliptic cone, implicitfast, Newton, tolerance 1e-8,
ls_tolerance 0.01, ccd_iterations 50, gravity -9.81. Contact friction is MuJoCo's element-wise max,
so a cube friction draw below the box's 0.6 acts as 0.6 against the box (0.8 against the table).

## Per trial
Cube released free, angular velocity 0, linear (0, 0, -v_down), simulated 2.0 s (1000 steps).
Cube sliding friction U{FRICTION_RANGE} and mass U{MASS_RANGE} kg (uniform in kg, as asked; the
DR-object rung draws mass log-uniform via `dr.pseudo_inertia`), inertia scaled with mass (as
`dr.pseudo_inertia`), `mj_setConst` after each change. The {n_trials} (friction, mass) pairs are
Latin-hypercube draws from `numpy.random.default_rng({SEED})` and are the same in every cell
(common random numbers).

## Classification (box frame at the end; rim z = +0.030, floor top +0.005)
- `inside`: |x|, |y| < 0.040 AND z < rim + 0.0125 AND (cube-box_floor contact OR z < floor + 0.025 +
  0.012) AND cube linear speed < 0.02 m/s.
- `on_rim`: not inside and a cube-wall contact at z >= rim - 4 mm (resting partly on a wall top).
- `outside`: neither, and the cube centre outside the inner footprint (on the table or floor).
- `other`: neither, centre over the inner footprint but still moving or held high (should be ~0).
- box displacement: |xy| of the box origin from its settled position (mm); tipped: box z axis > 5 deg
  from vertical.

## Grid
Cube centre offset from the box floor centre along +x and along (1,1)/sqrt2: {list(offsets)} cm
(offset 0 once); yaw {list(YAWS_DEG)} deg (about z, applied first); tilt about the world x axis
{list(TILTS_DEG)} deg; h = height of the cube's lowest point above the rim {list(hs)} cm, h = -2.0
being the set-down level (bottom 5 mm above the floor, already inside; a pose whose cube penetrates
any box geom at release, checked with mj_forward, is `infeasible=1`, n=0); v_down {list(V_DOWN)} m/s.
{len(cells)} cells, {n_infeasible} infeasible, {n_trials} trials per feasible cell, {n_run} trials.
{"Reductions from the plan grid: " + ", ".join(reductions) + "." if reductions else "No reduction of the requested grid was needed."}

Files: `cells.csv` (one row per cell), `trials.csv` (one row per trial: draws, outcome, final cube
box-frame position and speed, box displacement and tilt).

## Hand checks (before the sweep; nominal mu 0.9 / 50 g + the 16 draws)
{chr(10).join(check_lines)}

## (1) inside share vs offset x yaw (tilt 0, h 0.5 cm, v 0.10 m/s)
{t1}

## (2) inside share vs tilt x h (offset 1.0 cm along +x, yaw 0, v 0.10 m/s)
{t2}

("inf." = infeasible at release.)

## (3) Plan reference let-go
{t3}

## Sanity
Largest per-cell median box displacement {max_disp:.1f} mm; cells with any box tip {any_tip};
NaN trials {nan_total}. Caveat: CPU MuJoCo box-box collision, not the mujoco_warp used in training.
"""
  (args.out / "README.md").write_text(readme)
  print(
    f"\nwrote {args.out}/cells.csv, trials.csv, README.md; total {total_s / 60:.1f} min"
  )


if __name__ == "__main__":
  main()
