"""Where can the Kangaroo's left gripper actually go, at table height?

S1.3 of the pick-and-place plan: everything downstream -- where the cube spawns, where the
fixed target goes, whether the table is at a workable distance -- is a guess until this is
measured. The measurement is deliberately crude and honest: sample the left arm's seven
joints inside their *soft* limits, run forward kinematics and collision detection, and keep
the configurations whose grasp point lands in the band just above the table with nothing in
contact.

Two filters, reported separately:

* **reachable** -- the grasp point is in the band. This is the plan's criterion.
* **graspable** -- reachable *and* the jaw's approach axis is within ``--cone`` of straight
  down, so the cube can be taken from above. A point the wrist can only touch edge-on is
  not a pick-and-place target, so this is the set the spawn and the target are chosen from.

The collision model matters, and today's is incomplete: ``MANIPULATION_COLLISION``
(``kangaroo_constants.py``) enables only feet, arm 5-7 and the gripper, so the torso capsule
that ships in the XML (``pelvis_2_collision``) is switched off and the arm may pass through
the chest. ``--torso`` turns it back on; the difference between the two runs is the evidence
for the S1.4 collision decision.

Output: an ``.npz`` of the accepted samples and an occupancy grid, plus a plot. The spawn and
target proposal comes from the graspable grid: the cells furthest from any unreachable cell,
subject to a minimum separation.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from scene_model import (  # noqa: E402
  DEFAULT_TASK,
  compile_scene,
  geom_id,
  joint_qadr,
  nominal_data,
  site_id,
  soft_limits,
)

ARM_JOINTS = [f"arm_left_{i}_joint" for i in range(1, 8)]
TCP_SITE = "robot/tcp_left"
#: The body the TCP hangs off; its +z is the jaw's approach axis (``tcp_left`` sits at
#: ``0 0 0.148`` in it, i.e. straight out of the palm).
HAND_BODY = "robot/arm_left_7_link"

#: Table top, and the cube resting on it. From ``scene.py``: top face 0.76, cube half 0.025.
TABLE_TOP_Z = 0.76
CUBE_HALF = 0.025
REST_Z = TABLE_TOP_Z + CUBE_HALF


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--task", default=DEFAULT_TASK)
  p.add_argument("--samples", type=int, default=50_000)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument(
    "--band",
    type=float,
    nargs=2,
    default=(REST_Z, 0.95),
    metavar=("ZMIN", "ZMAX"),
    help="grasp-point height band to accept, in world z",
  )
  p.add_argument(
    "--cone",
    type=float,
    default=30.0,
    help="max angle (deg) between the jaw approach axis and straight down, for 'graspable'",
  )
  p.add_argument(
    "--cell", type=float, default=0.02, help="occupancy grid cell size (m)"
  )
  p.add_argument(
    "--margin",
    type=float,
    default=0.05,
    help="required clearance from the edge of the graspable region (m)",
  )
  p.add_argument(
    "--separation",
    type=float,
    default=0.15,
    help="required XY distance between the spawn centre and the target (m)",
  )
  p.add_argument(
    "--pad-clearance",
    type=float,
    default=0.01,
    help="how far the pads must clear the table top at the grasp pose (m)",
  )
  p.add_argument(
    "--pregrasp",
    type=float,
    default=0.10,
    help="height of the pre-grasp waypoint above the grasp pose (m)",
  )
  p.add_argument(
    "--torso",
    action="store_true",
    help="re-enable pelvis_2_collision (the torso capsule the variant filters out)",
  )
  p.add_argument(
    "--arm4",
    action="store_true",
    help="re-enable arm_left_4_collision, the elbow capsule MANIPULATION_COLLISION drops "
    "because it swallows arm 5-7; its contacts with those children are ignored, which is "
    "what a <contact><exclude> block would do",
  )
  p.add_argument("--out", type=pathlib.Path, default=None)
  return p.parse_args()


def _park_free_bodies(model: mujoco.MjModel, data: mujoco.MjData) -> None:
  """Move the free objects far away and stop them colliding.

  The reach map is about the arm against the robot and the table. A cube sitting wherever
  the reset event happened to put it would veto configurations for no reason.
  """
  for jid in range(model.njnt):
    if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
      continue
    adr = model.jnt_qposadr[jid]
    data.qpos[adr : adr + 3] = (0.0, 0.0, -10.0)
    body = model.jnt_bodyid[jid]
    for gid in range(
      model.body_geomadr[body], model.body_geomadr[body] + model.body_geomnum[body]
    ):
      model.geom_contype[gid] = 0
      model.geom_conaffinity[gid] = 0


ARM4_GEOM = "arm_left_4_collision"
TORSO_GEOM = "pelvis_2_collision"


def extra_geoms(args: argparse.Namespace) -> tuple[str, ...]:
  """The geom patterns the ``--torso`` / ``--arm4`` variants add to the collision set."""
  extra = []
  if args.torso:
    extra.append(TORSO_GEOM)
  if args.arm4:
    extra.append(ARM4_GEOM)
  return tuple(extra)


def elbow_geoms(model: mujoco.MjModel, args: argparse.Namespace) -> set[int]:
  return {geom_id(model, f"robot/{ARM4_GEOM}")} if args.arm4 else set()


def _skip(elbow: set[int], is_arm: np.ndarray, g1: int, g2: int) -> bool:
  """Elbow-capsule contacts with its own children, which an ``<exclude>`` would remove."""
  return bool(elbow) and ((g1 in elbow and is_arm[g2]) or (g2 in elbow and is_arm[g1]))


def left_arm_geoms(model: mujoco.MjModel) -> np.ndarray:
  """Every collidable geom on the left arm from the shoulder down, as a boolean mask.

  A configuration is rejected when one of these touches anything. Contacts that do *not*
  involve the arm are not the arm's fault: with the base welded the feet rest on the floor
  in every configuration, which is 32 contacts before the arm has moved at all.
  """
  mask = np.zeros(model.ngeom, dtype=bool)
  for gid in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
    bare = name.split("/", 1)[-1]
    if bare.startswith("arm_left_") or bare.startswith("gripper_left_"):
      mask[gid] = True
  return mask


def sample(args: argparse.Namespace) -> dict:
  model = compile_scene(args.task, extra_geoms(args))
  elbow = elbow_geoms(model, args)
  data = nominal_data(model)
  _park_free_bodies(model, data)

  qadr = joint_qadr(model, ARM_JOINTS)
  lo, hi = soft_limits(model, ARM_JOINTS)
  tcp = site_id(model, TCP_SITE)
  hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)
  is_arm = left_arm_geoms(model)

  rng = np.random.default_rng(args.seed)
  q = rng.uniform(lo, hi, size=(args.samples, len(qadr)))

  pos = np.empty((args.samples, 3))
  # cos of the angle between the hand's +z (the approach axis) and world -z.
  down = np.empty(args.samples)
  hits = np.zeros(args.samples, dtype=np.int32)
  blockers: dict[tuple[str, str], int] = {}

  def gname(gid: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"<{gid}>"

  t0 = time.time()
  for i in range(args.samples):
    data.qpos[qadr] = q[i]
    mujoco.mj_kinematics(model, data)
    mujoco.mj_collision(model, data)
    pos[i] = data.site_xpos[tcp]
    down[i] = -data.xmat[hand].reshape(3, 3)[2, 2]
    seen = set()
    for c in range(data.ncon):
      g1, g2 = int(data.contact[c].geom1), int(data.contact[c].geom2)
      if _skip(elbow, is_arm, g1, g2) or not (is_arm[g1] or is_arm[g2]):
        continue
      hits[i] += 1
      seen.add((gname(min(g1, g2)), gname(max(g1, g2))))
    for pair in seen:
      blockers[pair] = blockers.get(pair, 0) + 1
  elapsed = time.time() - t0

  free = hits == 0
  in_band = (pos[:, 2] >= args.band[0]) & (pos[:, 2] <= args.band[1])
  reachable = free & in_band
  graspable = reachable & (down >= np.cos(np.deg2rad(args.cone)))

  print(
    f"{args.samples} samples in {elapsed:.1f}s "
    f"({1e3 * elapsed / args.samples:.2f} ms/sample), torso collider "
    f"{'ON' if args.torso else 'OFF'}"
  )
  print(f"  collision-free            {free.sum():7d}  ({100 * free.mean():5.1f}%)")
  print(
    f"  in band z in [{args.band[0]:.3f},{args.band[1]:.2f}]  {in_band.sum():7d}  "
    f"({100 * in_band.mean():5.1f}%)"
  )
  print(
    f"  reachable (both)          {reachable.sum():7d}  ({100 * reachable.mean():5.1f}%)"
  )
  print(
    f"  graspable (+ within {args.cone:.0f} deg of down)  {graspable.sum():7d}  "
    f"({100 * graspable.mean():5.1f}%)"
  )
  print("  what the arm hits, by geom pair (share of all samples):")
  for pair, count in sorted(blockers.items(), key=lambda kv: -kv[1])[:12]:
    print(f"    {pair[0]:44s} {pair[1]:44s} {100 * count / args.samples:5.1f}%")
  return dict(
    model=model,
    q=q,
    pos=pos,
    down=down,
    ncon=hits,
    reachable=reachable,
    graspable=graspable,
    blockers=blockers,
  )


def _ik_top_down(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  qadr: np.ndarray,
  lo: np.ndarray,
  hi: np.ndarray,
  tcp: int,
  hand: int,
  goal: np.ndarray,
  q_init: np.ndarray,
  iters: int = 120,
  tol: float = 2e-3,
) -> np.ndarray | None:
  """Damped least squares onto "grasp point at ``goal``, jaw pointing straight down".

  Six residuals (three position, three axis alignment) on seven joints, clamped to the soft
  limits every step so a solution that only exists outside them is never reported. Returns
  the joint vector, or ``None`` if it did not converge inside ``tol``.
  """
  jacp = np.zeros((3, model.nv))
  jacr = np.zeros((3, model.nv))
  dofs = np.asarray(
    [model.jnt_dofadr[model.jnt_qposadr.tolist().index(a)] for a in qadr]
  )
  q = q_init.copy()
  down = np.array([0.0, 0.0, -1.0])
  for _ in range(iters):
    data.qpos[qadr] = q
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    err_p = goal - data.site_xpos[tcp]
    axis = data.xmat[hand].reshape(3, 3)[:, 2]
    err_r = np.cross(axis, down)
    err = np.concatenate([err_p, err_r])
    if np.linalg.norm(err_p) < tol and np.linalg.norm(err_r) < 0.05:
      return q
    mujoco.mj_jacSite(model, data, jacp, jacr, tcp)
    jac = np.vstack([jacp[:, dofs], jacr[:, dofs]])
    # Levenberg-Marquardt: lambda big enough to survive the wrist singularity.
    lam = 0.05
    dq = jac.T @ np.linalg.solve(jac @ jac.T + lam * np.eye(6), err)
    q = np.clip(q + np.clip(dq, -0.4, 0.4), lo, hi)
  return None


def pad_offset(model: mujoco.MjModel, data: mujoco.MjData) -> float:
  """How far the pads sit beyond the grasp point, along the jaw's approach axis.

  ``tcp_left`` is 148 mm out of the wrist, but the fingertip pads reach further: this is
  the distance that decides how high the grasp point must be for the pads to clear the
  table. Measured, not assumed -- at the cube's own centre height the pads are 0.6 mm
  *below* the table top and every top-down grasp is a table collision.
  """
  hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)
  axis = data.xmat[hand].reshape(3, 3)[:, 2]
  tcp = data.site_xpos[site_id(model, TCP_SITE)]
  pads = [
    data.site_xpos[site_id(model, f"robot/gripper_left_fingertip_{s}_site")]
    for s in ("left", "right")
  ]
  return float(max(np.dot(p - tcp, axis) for p in pads))


def ik_grid(args: argparse.Namespace, seeds: int = 8) -> dict:
  """Per-cell answer to "can the jaw take a cube from directly above this point?".

  Rejection sampling (``sample``) measures the workspace but concentrates its samples where
  the joint box is dense, not where the table is; the cells that decide the spawn and the
  target are exactly the sparse ones at the edge. This asks the question per cell instead,
  and asks it twice: at the grasp pose, and at a pre-grasp waypoint ``--pregrasp`` above it,
  because a grasp the arm cannot descend into is not a grasp.
  """
  model = compile_scene(args.task, extra_geoms(args))
  elbow = elbow_geoms(model, args)
  data = nominal_data(model)
  offset = pad_offset(model, data)
  _park_free_bodies(model, data)

  qadr = joint_qadr(model, ARM_JOINTS)
  lo, hi = soft_limits(model, ARM_JOINTS)
  tcp = site_id(model, TCP_SITE)
  hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)
  is_arm = left_arm_geoms(model)

  # Grasp point: high enough that the pads clear the table by `--pad-clearance`, which for
  # a 50 mm cube still puts the pads on its upper half.
  z = TABLE_TOP_Z + offset + args.pad_clearance
  print(
    f"\npad offset beyond the grasp point = {offset * 1000:.1f} mm"
    f"  ->  grasp point at z = {z:.3f} "
    f"({(z - REST_Z) * 1000:+.0f} mm relative to the cube centre)"
  )

  xs = np.arange(0.25 + args.cell / 2, 0.85, args.cell)
  ys = np.arange(-0.40 + args.cell / 2, 0.40, args.cell)
  ok = np.zeros((len(xs), len(ys)), dtype=bool)
  why = np.zeros((len(xs), len(ys)), dtype=np.int8)  # 0 ok, 1 no IK, 2 collision
  solutions: dict[tuple[int, int], np.ndarray] = {}

  def solve_free(goal, q0):
    q = _ik_top_down(model, data, qadr, lo, hi, tcp, hand, goal, q0)
    if q is None:
      return None, "ik"
    data.qpos[qadr] = q
    mujoco.mj_kinematics(model, data)
    mujoco.mj_collision(model, data)
    for c in range(data.ncon):
      g1, g2 = int(data.contact[c].geom1), int(data.contact[c].geom2)
      if _skip(elbow, is_arm, g1, g2):
        continue
      if is_arm[g1] or is_arm[g2]:
        return None, "collision"
    return q, None

  rng = np.random.default_rng(args.seed)
  t0 = time.time()
  for i, x in enumerate(xs):
    for j, y in enumerate(ys):
      reason = "ik"
      for s in range(seeds):
        q0 = 0.5 * (lo + hi) if s == 0 else rng.uniform(lo, hi)
        q, fail = solve_free(np.array([x, y, z]), q0)
        if q is None:
          reason = fail
          continue
        pre, fail = solve_free(np.array([x, y, z + args.pregrasp]), q)
        if pre is None:
          reason = fail
          continue
        ok[i, j] = True
        solutions[(i, j)] = q
        break
      if not ok[i, j]:
        why[i, j] = 1 if reason == "ik" else 2
  print(
    f"IK grid: {ok.sum()} of {ok.size} table cells graspable from above "
    f"({args.cell * 100:.0f} cm cells, {seeds} seeds, {time.time() - t0:.1f}s); "
    f"{int((why == 1).sum())} cells unreachable, {int((why == 2).sum())} blocked by a collision"
  )
  return dict(ok=ok, why=why, xs=xs, ys=ys, solutions=solutions, model=model, grasp_z=z)


def occupancy(
  pos: np.ndarray, mask: np.ndarray, cell: float
) -> tuple[np.ndarray, tuple]:
  """Boolean grid over XY: True where at least one accepted sample landed."""
  xy = pos[mask][:, :2]
  if len(xy) == 0:
    raise SystemExit("no accepted samples -- nothing to map")
  x0, y0 = np.floor(xy.min(0) / cell) * cell
  x1, y1 = np.ceil(xy.max(0) / cell) * cell
  nx = int(round((x1 - x0) / cell)) + 1
  ny = int(round((y1 - y0) / cell)) + 1
  grid = np.zeros((nx, ny), dtype=bool)
  ix = np.clip(((xy[:, 0] - x0) / cell).round().astype(int), 0, nx - 1)
  iy = np.clip(((xy[:, 1] - y0) / cell).round().astype(int), 0, ny - 1)
  grid[ix, iy] = True
  return grid, (x0, y0, cell)


def _largest_component(grid: np.ndarray) -> np.ndarray:
  """Keep only the biggest 4-connected blob: isolated cells are sampling noise."""
  from scipy import ndimage

  labels, n = ndimage.label(grid)
  if n <= 1:
    return grid
  sizes = ndimage.sum(grid, labels, range(1, n + 1))
  return labels == (1 + int(np.argmax(sizes)))


def clearance(grid: np.ndarray, cell: float) -> np.ndarray:
  """Distance from each cell to the nearest non-accepted cell, in metres.

  The grid is padded with a rejected border first: without it the transform treats the
  edge of the array as interior, and a cell on the table's near edge reports 30 cm of
  clearance because nothing outside the grid contradicts it.
  """
  from scipy import ndimage

  padded = np.pad(grid, 1, constant_values=False)
  return ndimage.distance_transform_edt(padded, sampling=cell)[1:-1, 1:-1]


def propose(grid: np.ndarray, origin: tuple, margin: float, separation: float):
  """Spawn centre and target: the two deepest interior cells that are far enough apart."""
  x0, y0, cell = origin
  dist = clearance(grid, cell)
  ix, iy = np.nonzero(dist >= margin)
  if len(ix) == 0:
    raise SystemExit(
      f"no cell has {margin * 100:.0f} cm of clearance; widen the region"
    )
  xy = np.stack([x0 + ix * cell, y0 + iy * cell], axis=1)
  depth = dist[ix, iy]
  order = np.argsort(-depth)

  # Deepest cell is the spawn (the cube must be grasped there, so give it the margin);
  # the target is the deepest cell at least `separation` away.
  spawn = xy[order[0]]
  for k in order[1:]:
    if np.linalg.norm(xy[k] - spawn) >= separation:
      return (spawn, depth[order[0]]), (xy[k], depth[k])
  raise SystemExit(
    f"no pair of interior cells is {separation * 100:.0f} cm apart; "
    "move the table or relax --separation"
  )


def widest_pair(grid: np.ndarray, origin: tuple, margin: float):
  """The furthest-apart pair of cells that both still clear ``margin``.

  The max-margin pair (``propose``) is the safe choice; this is the other end of the same
  trade, and the two together bracket what the task can be asked to transport.
  """
  x0, y0, cell = origin
  dist = clearance(grid, cell)
  ix, iy = np.nonzero(dist >= margin)
  xy = np.stack([x0 + ix * cell, y0 + iy * cell], axis=1)
  d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
  a, b = np.unravel_index(int(np.argmax(d2)), d2.shape)
  return (xy[a], dist[ix[a], iy[a]]), (xy[b], dist[ix[b], iy[b]])


def plot_ik(ik: dict, spawn, target, args, out: pathlib.Path) -> None:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from matplotlib.patches import Rectangle

  xs, ys, ok = ik["xs"], ik["ys"], ik["ok"]
  dist = clearance(ok, args.cell)
  fig, ax = plt.subplots(figsize=(7.5, 7))
  extent = (
    xs[0] - args.cell / 2,
    xs[-1] + args.cell / 2,
    ys[0] - args.cell / 2,
    ys[-1] + args.cell / 2,
  )
  ax.add_patch(
    Rectangle((0.25, -0.40), 0.60, 0.80, fc="#e8d9bf", ec="#8a6c40", lw=1.5, zorder=0)
  )
  im = ax.imshow(
    np.where(ok, dist, np.nan).T * 100,
    origin="lower",
    extent=extent,
    cmap="viridis",
    interpolation="nearest",
    zorder=1,
  )
  fig.colorbar(im, ax=ax, label="clearance from the edge of the graspable region [cm]")
  for (xy, depth), color, label in (
    (spawn, "#e53e3e", "spawn centre"),
    (target, "#ffffff", "target"),
  ):
    ax.plot(*xy, "*", color=color, ms=20, mec="k", mew=0.8, zorder=5)
    ax.annotate(
      f"{label}  ({xy[0]:.2f}, {xy[1]:.2f})\nmargin {depth * 100:.0f} cm",
      xy,
      textcoords="offset points",
      xytext=(12, -6),
      fontsize=9,
      color="k",
      zorder=6,
    )
  ax.plot(0, 0, "ks", ms=9, zorder=5)
  ax.annotate("base", (0, 0), textcoords="offset points", xytext=(6, 6))
  ax.set_xlabel("x [m]  (robot forward)")
  ax.set_ylabel("y [m]  (robot left)")
  ax.set_aspect("equal")
  ax.set_xlim(-0.12, 0.95)
  ax.set_ylim(-0.5, 0.55)
  ax.grid(alpha=0.25)
  ax.set_title(
    f"Left jaw, top-down grasp of a cube on the table (grasp point z = {ik['grasp_z']:.3f})\n"
    f"{ok.sum()} / {ok.size} cells solvable, collision-free, inside soft limits"
    f"  [torso {'ON' if args.torso else 'OFF'}, elbow {'ON' if args.arm4 else 'OFF'}]\n"
    "clearance runs to the edge of the region or of the table, whichever is nearer",
    fontsize=10,
  )
  fig.tight_layout()
  fig.savefig(out, dpi=140)
  print(f"wrote {out}")


def plot(
  result: dict, grid: np.ndarray, origin: tuple, spawn, target, args, out: pathlib.Path
):
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from matplotlib.patches import Rectangle

  x0, y0, cell = origin
  pos, reach, grasp = result["pos"], result["reachable"], result["graspable"]
  fig, axes = plt.subplots(1, 2, figsize=(13, 6))

  for ax, mask, title in (
    (axes[0], reach, f"reachable  (z in [{args.band[0]:.3f}, {args.band[1]:.2f}])"),
    (axes[1], grasp, f"graspable  (+ approach within {args.cone:.0f} deg of down)"),
  ):
    ax.add_patch(
      Rectangle((0.25, -0.40), 0.60, 0.80, fc="#e8d9bf", ec="#8a6c40", lw=1.5, zorder=0)
    )
    ax.scatter(pos[mask][:, 0], pos[mask][:, 1], s=1.5, c="#2b6cb0", alpha=0.25, lw=0)
    ax.plot(0, 0, "ks", ms=8)
    ax.annotate("base", (0, 0), textcoords="offset points", xytext=(6, 6))
    ax.set_title(f"{title}\n{mask.sum()} / {len(mask)} samples")
    ax.set_xlabel("x [m]  (robot forward)")
    ax.set_ylabel("y [m]  (robot left)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.2, 0.95)
    ax.set_ylim(-0.55, 0.75)

  ax = axes[1]
  ny, nx = grid.shape[1], grid.shape[0]
  extent = (
    x0 - cell / 2,
    x0 + (nx - 0.5) * cell,
    y0 - cell / 2,
    y0 + (ny - 0.5) * cell,
  )
  ax.imshow(
    grid.T,
    origin="lower",
    extent=extent,
    cmap="Greens",
    alpha=0.35,
    interpolation="nearest",
  )
  for (xy, depth), color, label in (
    (spawn, "#c53030", "spawn centre"),
    (target, "#6b46c1", "target"),
  ):
    ax.plot(*xy, "*", color=color, ms=18, mec="k", mew=0.6, zorder=5)
    ax.annotate(
      f"{label}\n({xy[0]:.2f}, {xy[1]:.2f})  margin {depth * 100:.0f} cm",
      xy,
      textcoords="offset points",
      xytext=(10, -18),
      fontsize=8,
    )
  fig.suptitle(
    f"Kangaroo left-gripper workspace at table height  "
    f"[torso collider {'ON' if args.torso else 'OFF'}, {args.samples} samples, seed {args.seed}]"
  )
  fig.tight_layout()
  fig.savefig(out, dpi=140)
  print(f"wrote {out}")


def main() -> None:
  args = parse_args()
  out_dir = args.out or pathlib.Path("../docs/reports/pick-place-v0")
  out_dir.mkdir(parents=True, exist_ok=True)
  tag = "torso_on" if args.torso else "torso_off"

  result = sample(args)
  grid, origin = occupancy(result["pos"], result["reachable"], args.cell)
  grid = _largest_component(grid)

  ik = ik_grid(args)
  ok = _largest_component(ik["ok"])
  ik_origin = (float(ik["xs"][0]), float(ik["ys"][0]), args.cell)
  spawn, target = propose(ok, ik_origin, args.margin, args.separation)

  print(f"\ngraspable region: {ok.sum()} cells of {args.cell * 100:.0f} cm")
  print(
    f"  spawn centre  x={spawn[0][0]:.3f}  y={spawn[0][1]:.3f}  margin {spawn[1] * 100:.1f} cm"
  )
  print(
    f"  target        x={target[0][0]:.3f}  y={target[0][1]:.3f}  margin {target[1] * 100:.1f} cm"
  )
  print(f"  separation    {np.linalg.norm(spawn[0] - target[0]) * 100:.1f} cm")
  print(f"  target z (cube resting on the table) = {REST_Z:.3f}")
  far_a, far_b = widest_pair(ok, ik_origin, args.margin)
  print(
    f"  widest alternative pair at >= {args.margin * 100:.0f} cm margin: "
    f"({far_a[0][0]:.3f}, {far_a[0][1]:.3f}) margin {far_a[1] * 100:.0f} cm  <->  "
    f"({far_b[0][0]:.3f}, {far_b[0][1]:.3f}) margin {far_b[1] * 100:.0f} cm  "
    f"[{np.linalg.norm(far_a[0] - far_b[0]) * 100:.1f} cm apart]"
  )

  npz = out_dir / f"s1_reach_map_{tag}.npz"
  np.savez_compressed(
    npz,
    q=result["q"],
    pos=result["pos"],
    down=result["down"],
    ncon=result["ncon"],
    reachable=result["reachable"],
    graspable=result["graspable"],
    grid=grid,
    origin=np.asarray(origin),
    ik_ok=ok,
    ik_xs=ik["xs"],
    ik_ys=ik["ys"],
    ik_q=np.stack([ik["solutions"][k] for k in sorted(ik["solutions"])])
    if ik["solutions"]
    else np.zeros((0, 7)),
    ik_cells=np.asarray(sorted(ik["solutions"])),
    spawn=spawn[0],
    target=target[0],
    band=np.asarray(args.band),
    cone_deg=args.cone,
    seed=args.seed,
  )
  print(f"wrote {npz}")
  plot(result, grid, origin, spawn, target, args, out_dir / f"s1_reach_map_{tag}.png")
  plot_ik(dict(ik, ok=ok), spawn, target, args, out_dir / f"s1_grasp_map_{tag}.png")


if __name__ == "__main__":
  main()
