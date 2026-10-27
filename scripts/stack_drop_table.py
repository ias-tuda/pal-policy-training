"""Plan S7-9 step 0 item 3 (stack half): does a 5 cm cube let go above a 5 cm base cube stay on it?

A standalone CPU MuJoCo measurement, no robot and no policy, the same way as S10's box table
(`scripts/box_drop_table.py` on the S10 worktree, whose box results S7-9 reuses): one scene from the
task's own pieces (`scene.get_table_spec` at the robot cell's 0.815 m, two `scene.get_cube_spec`
cubes), the RL env's solver settings (`tabletop_env_cfg.py`'s MujocoCfg + mjlab's defaults). The base
cube rests free on the table centre (S9's base is a free cube too); the top cube is released free at
a pose relative to the base's top-face centre, simulated `SIM_S`, and classified.

Axes (the plan's: xy offset 0-3 cm, tilt 0-30 deg, height 0-2 cm, speed 0-0.15 m/s):

* offset of the top cube's centre from the base's, along +x (a face) and along the (1,1) diagonal
  (a corner);
* tilt about the horizontal axis perpendicular to the offset, ``out`` (the low edge toward the
  offset, i.e. off the base) or ``in`` (toward the base centre); at offset 0 the low edge is to +x;
* h = height of the top cube's lowest point above the base's top face (0 = touching);
* the release velocity: horizontal ``vxy`` along the offset direction (outward -- the conservative
  choice; S7-8's ABS-a let-gos move 0.06-0.13 m/s horizontally, median) and downward ``vz``
  (0 or 0.10 m/s; S7-8 medians 0.02-0.14).

Each cell runs N trials; both cubes' sliding friction U(0.5, 1.2) and mass U(0.03, 0.10) kg (DR-object's
ranges, `tabletop_env_cfg._DR_OBJECT_FRICTION` / `_DR_OBJECT_MASS_KG`) are Latin-hypercube draws from a
fixed seed reused in every cell (common random numbers), so cell differences come from the pose alone.
Contact friction is MuJoCo's element-wise max, so cube-cube friction is the larger of the two draws.

Outcome after `SIM_S` (base frame = the base's settled pose):

* ``stands``: the top cube's centre within 2.5 cm of the base's axis horizontally, its bottom within
  5 mm of the base top, its tilt < 5 deg, speed < 0.02 m/s, and the base still upright (< 5 deg);
* ``leans``: on the base (centre over it, above its top) but tilted >= 5 deg or still moving;
* ``fell``: anything else (on the table, beside the base);
* the base's displacement (mm) and whether it tipped are kept per trial.

Solver (``--solver``): ``env`` is the RL env's (iterations 10, ls_iterations 20); ``converged`` raises
them to 100 / 50. **At the env's cap a cube sliding on the base can be launched** (a flat, centred
set-down moving 0.05-0.10 m/s sideways stands in 21 of 32 trials, the others thrown up to 18 cm with
~8 rad/s of spin within 20 ms; 32 / 32 with 100 iterations, with impratio 1 or with a pyramidal cone):
the solver does not converge on the up-to-8-point box-box contact of two light bodies. ``converged`` is
the physical tolerance; ``env`` is what training's simulator would do (MuJoCo C here, not mujoco_warp).

  uv run --extra cpu python scripts/stack_drop_table.py [--solver env|converged] [--out DIR] [--trials 16]

Writes cells.csv, trials.csv and README.md.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import platform
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np
from pal_mjlab.tasks.manipulation import scene

TABLE_TOP_Z = scene.ROBOT_CELL_TABLE_TOP_Z
BASE_XY = (0.55, 0.0)  # the table centre
HALF = 0.025
SIM_S = 2.0
TIMESTEP = 0.002
NOMINAL_MASS = 0.05
FRICTION_RANGE = (0.5, 1.2)
MASS_RANGE = (0.03, 0.10)
SEED = 20261001
STILL = 0.02  # m/s
FLAT_DEG = 5.0

OFFSETS_CM = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
TILTS = (
  (0, "out"),
  (10, "out"),
  (10, "in"),
  (20, "out"),
  (20, "in"),
  (30, "out"),
  (30, "in"),
)
HS_CM = (0.0, 0.5, 1.0, 2.0)
VXY = (0.0, 0.05, 0.10, 0.15)
VZ = (0.0, 0.10)

DEFAULT_OUT = (
  Path(__file__).resolve().parents[2] / "results/S7-9/step0/stack_drop_table"
)


SOLVER_ITERATIONS = {"env": (10, 20), "converged": (100, 50)}
_SOLVER = "env"


def build_model() -> mujoco.MjModel:
  world = mujoco.MjSpec()
  world.worldbody.add_geom(
    name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(3.0, 3.0, 0.1), condim=3
  )
  attach = (
    (scene.get_table_spec(TABLE_TOP_Z), scene.table_pos(TABLE_TOP_Z)),
    (
      scene.get_cube_spec(name="base", rgba=(0.2, 0.3, 0.8, 1.0)),
      (BASE_XY[0], BASE_XY[1], TABLE_TOP_Z + HALF),
    ),
    # Parked on the table 30 cm aside while the base settles.
    (scene.get_cube_spec(name="object"), (BASE_XY[0], 0.30, TABLE_TOP_Z + HALF)),
  )
  for child, pos in attach:
    world.attach(child, frame=world.worldbody.add_frame(pos=pos), prefix="")
  m = world.compile()
  m.opt.timestep = TIMESTEP
  m.opt.iterations, m.opt.ls_iterations = SOLVER_ITERATIONS[_SOLVER]
  m.opt.impratio = 10
  m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
  m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  m.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
  m.opt.tolerance = 1e-8
  m.opt.ls_tolerance = 0.01
  m.opt.ccd_iterations = 50
  m.opt.jacobian = mujoco.mjtJacobian.mjJAC_AUTO
  return m


def rot(axis, deg: float) -> np.ndarray:
  q = np.zeros(4)
  mujoco.mju_axisAngle2Quat(q, np.asarray(axis, dtype=float), math.radians(deg))
  R = np.zeros(9)
  mujoco.mju_quat2Mat(R, q)
  return R.reshape(3, 3)


CORNERS = np.array(list(itertools.product((-HALF, HALF), repeat=3)))


class Sim:
  def __init__(self):
    self.m = build_model()
    self.d = mujoco.MjData(self.m)
    m = self.m
    self.top = m.body("object").id
    self.base = m.body("base").id
    self.top_g = m.geom("object_geom").id
    self.base_g = m.geom("base_geom").id
    self.top_q = m.jnt_qposadr[m.joint("object_joint").id]
    self.top_v = m.jnt_dofadr[m.joint("object_joint").id]
    self.inertia0 = m.body_inertia[self.top].copy()
    self.f0 = {g: m.geom_friction[g].copy() for g in (self.top_g, self.base_g)}
    mujoco.mj_resetData(m, self.d)
    mujoco.mj_step(m, self.d, nstep=500)
    self.q0 = self.d.qpos.copy()
    mujoco.mj_forward(m, self.d)
    self.base_p0 = self.d.xpos[self.base].copy()
    self.base_top = self.base_p0[2] + HALF

  def set_params(self, top_f, top_m, base_f, base_m) -> None:
    m = self.m
    for g, b, f, mass in (
      (self.top_g, self.top, top_f, top_m),
      (self.base_g, self.base, base_f, base_m),
    ):
      m.geom_friction[g] = self.f0[g]
      m.geom_friction[g, 0] = f
      m.body_mass[b] = mass
      m.body_inertia[b] = self.inertia0 * (mass / NOMINAL_MASS)
    mujoco.mj_setConst(m, self.d)

  def place(self, offset_m, direction, tilt_deg, tilt_dir, h_m, vxy, vz) -> None:
    m, d = self.m, self.d
    mujoco.mj_resetData(m, d)
    d.qpos[:] = self.q0
    d.qvel[:] = 0.0
    u = (
      np.array([1.0, 0.0, 0.0])
      if direction == "x"
      else np.array([1.0, 1.0, 0.0]) / 2**0.5
    )
    # Tilt about the horizontal axis perpendicular to u: positive rotation about (u x z) lowers the
    # +u side ("out"); "in" lowers the side toward the base centre.
    axis = np.cross(u, [0.0, 0.0, 1.0])
    R = rot(axis, tilt_deg if tilt_dir == "out" else -tilt_deg)
    lowest = float((CORNERS @ R.T)[:, 2].min())
    p = np.array(
      [
        self.base_p0[0] + offset_m * u[0],
        self.base_p0[1] + offset_m * u[1],
        self.base_top + h_m - lowest,
      ]
    )
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.reshape(-1))
    d.qpos[self.top_q : self.top_q + 3] = p
    d.qpos[self.top_q + 3 : self.top_q + 7] = quat
    d.qvel[self.top_v : self.top_v + 6] = (vxy * u[0], vxy * u[1], -vz, 0.0, 0.0, 0.0)
    mujoco.mj_forward(m, d)

  def run(self) -> dict:
    m, d = self.m, self.d
    mujoco.mj_step(m, d, nstep=int(round(SIM_S / TIMESTEP)))
    bp = d.xpos[self.base]
    bR = d.xmat[self.base].reshape(3, 3)
    tp = d.xpos[self.top]
    tR = d.xmat[self.top].reshape(3, 3)
    base_tilt = math.degrees(math.acos(min(1.0, float(np.abs(bR[2]).max()))))
    top_tilt = math.degrees(math.acos(min(1.0, float(np.abs(tR[2]).max()))))
    dxy = float(np.linalg.norm(tp[:2] - bp[:2]))
    bottom_gap = float(tp[2] - HALF - (bp[2] + HALF))
    speed = float(np.linalg.norm(d.qvel[self.top_v : self.top_v + 3]))
    over = dxy < HALF and tp[2] > bp[2] + HALF
    if (
      over
      and abs(bottom_gap) < 0.005
      and top_tilt < FLAT_DEG
      and speed < STILL
      and base_tilt < FLAT_DEG
    ):
      outcome = "stands"
    elif over:
      outcome = "leans"
    else:
      outcome = "fell"
    return {
      "outcome": outcome,
      "dxy": dxy,
      "top_tilt_deg": top_tilt,
      "speed": speed,
      "base_disp_mm": 1000.0 * float(np.linalg.norm(bp[:2] - self.base_p0[:2])),
      "base_tipped": base_tilt >= FLAT_DEG,
      "nan": bool(np.any(~np.isfinite(d.qpos))),
    }


def draws(n: int) -> np.ndarray:
  """(n, 4): top friction, top mass, base friction, base mass -- Latin hypercube, fixed seed."""
  rng = np.random.default_rng(SEED)
  u = (
    np.stack([rng.permutation(n) for _ in range(4)], axis=1) + rng.random((n, 4))
  ) / n
  lo = np.array([FRICTION_RANGE[0], MASS_RANGE[0]] * 2)
  hi = np.array([FRICTION_RANGE[1], MASS_RANGE[1]] * 2)
  return lo + u * (hi - lo)


def cells() -> list[tuple]:
  out = []
  for off in OFFSETS_CM:
    for direction in ("x",) if off == 0 else ("x", "diag"):
      for (tilt, tdir), h, vxy, vz in itertools.product(TILTS, HS_CM, VXY, VZ):
        out.append((off, direction, tilt, tdir, h, vxy, vz))
  return out


_SIM = None
_DRAWS = None


def _init(n, solver):
  global _SIM, _DRAWS, _SOLVER
  _SOLVER = solver
  _SIM, _DRAWS = Sim(), draws(n)


def _run_cell(cell):
  off, direction, tilt, tdir, h, vxy, vz = cell
  rows = []
  for k, (tf, tm, bf, bm) in enumerate(_DRAWS):
    _SIM.set_params(tf, tm, bf, bm)
    _SIM.place(off / 100, direction, tilt, tdir, h / 100, vxy, vz)
    r = _SIM.run()
    rows.append({"offset_cm": off, "direction": direction, "tilt_deg": tilt, "tilt_dir": tdir,
                 "h_cm": h, "vxy": vxy, "vz": vz, "trial": k, "top_friction": round(tf, 4),
                 "top_mass": round(tm, 4), "base_friction": round(bf, 4), "base_mass": round(bm, 4),
                 **r})  # fmt: skip
  return cell, rows


def checks() -> list[str]:
  """Hand checks with the nominal cubes: a flat set-down centred stands; one 4 cm off falls."""
  sim = Sim()
  out = []
  for label, args, want in (
    ("centred, flat, touching, still", (0.0, "x", 0, "out", 0.0, 0.0, 0.0), "stands"),
    ("4 cm off along x, flat, 1 cm up", (0.04, "x", 0, "out", 0.01, 0.0, 0.0), "fell"),
    (
      "centred, flat, 2 cm up, 0.1 m/s down",
      (0.0, "x", 0, "out", 0.02, 0.0, 0.10),
      "stands",
    ),
  ):
    sim.set_params(0.9, NOMINAL_MASS, 0.9, NOMINAL_MASS)
    sim.place(*args)
    r = sim.run()
    out.append(
      f"- {label}: expect {want}, got {r['outcome']} ({'PASS' if r['outcome'] == want else 'FAIL'}; dxy {r['dxy']:.4f} m, tilt {r['top_tilt_deg']:.1f} deg)"
    )
  return out


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--solver", choices=tuple(SOLVER_ITERATIONS), default="converged")
  p.add_argument("--out", type=Path, default=None)
  p.add_argument("--trials", type=int, default=16)
  p.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
  args = p.parse_args()
  global _SOLVER
  _SOLVER = args.solver
  args.out = args.out or DEFAULT_OUT.with_name(f"{DEFAULT_OUT.name}_{args.solver}")
  args.out.mkdir(parents=True, exist_ok=True)
  t0 = time.time()
  hand = checks()
  print("\n".join(hand))
  grid = cells()
  with Pool(args.procs, initializer=_init, initargs=(args.trials, args.solver)) as pool:
    results = pool.map(_run_cell, grid, chunksize=8)
  trials = [r for _, rows in results for r in rows]
  with open(args.out / "trials.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(trials[0]))
    w.writeheader()
    w.writerows(trials)
  with open(args.out / "cells.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["offset_cm", "direction", "tilt_deg", "tilt_dir", "h_cm", "vxy", "vz", "n",
                "stands_share", "leans_share", "fell_share", "median_base_disp_mm",
                "base_tipped_share", "nan_trials"])  # fmt: skip
    for cell, rows in results:
      c = Counter(r["outcome"] for r in rows)
      n = len(rows)
      w.writerow([*cell, n, c["stands"] / n, c["leans"] / n, c["fell"] / n,
                  float(np.median([r["base_disp_mm"] for r in rows])),
                  sum(r["base_tipped"] for r in rows) / n, sum(r["nan"] for r in rows)])  # fmt: skip
  dt = time.time() - t0
  (args.out / "README.md").write_text(
    f"# S7-9 step 0 -- stack drop table (a 5 cm cube let go above a 5 cm base cube)\n\n"
    f"Generated by `scripts/stack_drop_table.py` (pal_mjlab branch `siwei-s7-9`) on "
    f"{time.strftime('%Y-%m-%d %H:%M')} ({platform.node()}, {platform.platform()}), mujoco "
    f"{mujoco.__version__} (CPU). {len(grid)} cells x {args.trials} trials = {len(trials)} trials, "
    f"{dt / 60:.1f} min on {args.procs} processes. Solver `{args.solver}` (iterations, ls_iterations = "
    f"{SOLVER_ITERATIONS[args.solver]}).\n\n"
    f"Scene, axes, draws and outcome: the script's docstring.\n\n## Hand checks\n"
    + "\n".join(hand)
    + "\n"
  )
  print(f"wrote {args.out} ({len(trials)} trials, {dt / 60:.1f} min)")


if __name__ == "__main__":
  main()
