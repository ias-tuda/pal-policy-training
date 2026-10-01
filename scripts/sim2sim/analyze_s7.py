"""One PAL sim2sim condition of an S7 (RCTP) policy against its MuJoCo reference.

Reads `<run>/epNN_policy.csv` / `epNN_joints.csv` (pal_policy_inferer_pick_place_node) or
`<run>/replay_epNN_joints.csv` (open-loop replay, `replay_reference_targets.py`) and the model
folder's `mujoco_reference.npz` / `episodes.json` / `pick_place_profile.yaml`; prints a table and
writes `<run>/summary.json`.

Policy runs, per episode: placed (PAL: the node's virtual cube at the goal, at rest, after a
lift; MuJoCo: cube within the goal threshold, on the table and at rest, after a lift), the step
of the first grasp / release, the cube's final distance to that episode's goal, the TCP against
MuJoCo's over the first second, peak TCP speed, and how often T's band moved the target.
Replay runs: the joints against MuJoCo's, in rad, over the first second and the whole episode.

  python3 analyze_s7.py --run <run dir> --model <model dir> [--replay]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def speed(p: np.ndarray, dt: float) -> np.ndarray:
  v = np.zeros(len(p))
  v[1:] = np.linalg.norm(np.diff(p, axis=0), axis=1) / dt
  return v


def mujoco_episode(r, k: int, goal: np.ndarray, thr: float, dt: float) -> dict:
  held = r["grasped"][k].astype(bool)
  on = r["on_surface"][k].astype(bool)
  cube = r["cube_b"][k]
  tcp = r["tcp_b"][k]
  v = speed(cube, dt)
  lifted = np.cumsum(held & ~on) > 0
  placed_t = (np.linalg.norm(cube - goal, axis=1) < thr) & ~held & (v < 0.05) & lifted
  return {
    "placed": bool(placed_t.any()),
    "t_first_placed": float(np.argmax(placed_t) * dt) if placed_t.any() else None,
    "t_first_grasp": float(np.argmax(held) * dt) if held.any() else None,
    "final_cube_to_goal": float(np.linalg.norm(cube[-1] - goal)),
    "peak_tcp_speed": float(speed(tcp, dt).max()),
    "band_hit_share": float(r["band_hit"][k].mean()),
    "tcp": tcp,
  }


def policy_episode(run: Path, ep: int, goal: np.ndarray, dt: float) -> dict:
  d = np.genfromtxt(run / f"ep{ep:02d}_policy.csv", delimiter=",", names=True)
  if d.size < 2:
    return None
  t = d["t"]
  held = d["cube_state"] == 1
  placed = d["placed"] > 0
  tcp = np.stack([d["tcp_x"], d["tcp_y"], d["tcp_z"]], axis=1)
  cube = np.stack([d["cube_x"], d["cube_y"], d["cube_z"]], axis=1)
  grasps = int(held[0]) + int(np.sum(~held[:-1] & held[1:]))
  return {
    "steps": int(len(t)),
    "placed": bool(placed.any()),
    "t_first_placed": float(t[np.argmax(placed)]) if placed.any() else None,
    "t_first_grasp": float(t[np.argmax(held)]) if held.any() else None,
    "grasps": grasps,
    "was_lifted": bool(d["was_lifted"].max() > 0),
    "final_cube_to_goal": float(np.linalg.norm(cube[-1] - goal)),
    "final_cube_state": int(d["cube_state"][-1]),
    "peak_tcp_speed": float(speed(tcp, dt).max()),
    "band_hit_share": float(d["band_hit"].mean()) if "band_hit" in d.dtype.names else None,
    "range_hit_share": float(d["range_hit"].mean()) if "range_hit" in d.dtype.names else None,
    "tcp": tcp,
  }


def replay_episode(run: Path, ep: int, r, k: int) -> dict:
  d = np.genfromtxt(run / f"replay_ep{ep:02d}_joints.csv", delimiter=",", names=True)
  step = d["step"].astype(int)
  q = np.stack([d[f"q{j}"] for j in range(8)], axis=1)
  n = r["q"].shape[1]
  err = []
  for s in range(n - 1):
    m = step == s
    if m.any():
      # The state after target s has been applied for one policy step is MuJoCo's q at s + 1.
      err.append(np.abs(q[m][-1] - r["q"][k, s + 1]))
  err = np.array(err)
  return {
    "max_err_first_1s": err[:50].max(0).round(4).tolist(),
    "max_err_all": err.max(0).round(4).tolist(),
    "rms_err_all": np.sqrt((err**2).mean(0)).round(4).tolist(),
  }


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--run", type=Path, required=True)
  p.add_argument("--model", type=Path, required=True)
  p.add_argument("--replay", action="store_true")
  p.add_argument("--suffix", default="", help="file suffix of per-episode sessions, e.g. _table_cube")
  args = p.parse_args()
  prof = yaml.safe_load((args.model / "pick_place_profile.yaml").read_text())
  eps = json.loads((args.model / "episodes.json").read_text())["episodes"]
  r = np.load(args.model / "mujoco_reference.npz")
  dt = float(r["dt"])
  thr = prof["goal_threshold"]
  summary = {"condition": args.run.name, "episodes": {}}
  print(f"{args.run.name}: {'open-loop replay' if args.replay else 'policy'}, {len(eps)} episodes")
  rows = []
  for k, e in enumerate(eps):
    ep = e["id"]
    goal = np.array(r["goal_b"][k][0], dtype=float)
    if args.replay:
      f = args.run / f"replay_ep{ep:02d}_joints.csv"
      if not f.exists():
        continue
      res = replay_episode(args.run, ep, r, k)
      summary["episodes"][ep] = res
      rows.append((ep, res))
      continue
    f = args.run / f"ep{ep:02d}_policy.csv"
    if not f.exists():
      print(f"  ep {ep}: no log")
      continue
    pal = policy_episode(args.run, ep, goal, dt)
    if pal is None:
      print(f"  ep {ep}: empty log")
      continue
    mj = mujoco_episode(r, k, goal, thr, dt)
    n = min(len(pal["tcp"]), len(mj["tcp"]), 50)
    # PAL's log starts at t = 0 with the first step; compare by step index.
    pal["tcp_rms_vs_mujoco_1s_cm"] = float(
      100 * np.sqrt(((pal["tcp"][:n] - mj["tcp"][:n]) ** 2).sum(1).mean())
    )
    summary["episodes"][ep] = {"pal": {k2: v for k2, v in pal.items() if k2 != "tcp"}, "mujoco": {k2: v for k2, v in mj.items() if k2 != "tcp"}}
    rows.append((ep, pal, mj))
  if args.replay:
    print("ep  max |PAL - MuJoCo| per joint (arm 1-7, jaw), first 1 s / rms over the episode")
    for ep, res in rows:
      print(f"{ep:2d}  {np.round(res['max_err_first_1s'], 3)}  rms {np.round(res['rms_err_all'], 3)}")
  else:
    print("ep | PAL placed  t_grasp t_place  grasps  cube->goal cm | MuJoCo placed t_grasp | TCP rms 1s cm | band")
    for ep, pal, mj in rows:
      fmt = lambda v: "  -  " if v is None else f"{v:5.2f}"  # noqa: E731
      print(
        f"{ep:2d} | {str(pal['placed']):5s} {fmt(pal['t_first_grasp'])} {fmt(pal['t_first_placed'])} "
        f"{pal['grasps']:3d}  {100 * pal['final_cube_to_goal']:7.1f} | {str(mj['placed']):5s} "
        f"{fmt(mj['t_first_grasp'])} | {pal['tcp_rms_vs_mujoco_1s_cm']:5.1f} | "
        f"{pal['band_hit_share'] if pal['band_hit_share'] is None else round(pal['band_hit_share'], 2)}"
      )
    n_pal = sum(1 for _, pal, _ in rows if pal["placed"])
    n_mj = sum(1 for _, _, mj in rows if mj["placed"])
    print(f"placed: PAL {n_pal}/{len(rows)}, MuJoCo {n_mj}/{len(rows)}")
    summary["placed_pal"], summary["placed_mujoco"], summary["n"] = n_pal, n_mj, len(rows)
  (args.run / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")


if __name__ == "__main__":
  main()
