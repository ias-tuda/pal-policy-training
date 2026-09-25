"""Metrics for one PAL sim2sim condition against the matched MuJoCo reference.

Reads a condition's logs (`<run_dir>/epNN_policy.csv` at the policy rate and
`<run_dir>/epNN_joints.csv` at the simulator's joint-state rate, written by
pal_policy_inferer_pick_place_node) and the model folder's `mujoco_reference.npz` (the same
policy and episode specs in the training simulator), and writes `<run_dir>/summary.json`.

Per episode and per simulator:
* placed (cube at the goal, released and at rest, after having been lifted -- the node's
  virtual cube in PAL, physics in MuJoCo), time of the first grasp and first placement,
  number of grasps;
* TCP speed, overall and while holding the cube (finite difference at the policy rate);
* joint speed and torque against PAL's URDF velocity limits and the training effort limits
  (PAL: every joint-state message; MuJoCo: the policy-rate samples only);
* tracking error, commanded (slew-limited) target minus measured position, per arm joint.

  .venv/bin/python scripts/sim2sim/analyze_sim2sim.py --run <run_dir> --model <model_dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

ARM = 7
# PAL URDF (kangaroo_description, 7dof): velocity limits, rad/s; training effort limits, N.m
# (params/env.yaml: arm 1-2 50, arm 3-7 25).
VEL_LIMIT = np.array([1.95, 1.95, 2.35, 2.35, 2.35, 2.35, 2.35])
EFFORT_LIMIT = np.array([50.0, 50.0, 25.0, 25.0, 25.0, 25.0, 25.0])


def runs(mask: np.ndarray) -> int:
  """Number of False -> True transitions (a run starting at index 0 counts)."""
  m = mask.astype(bool)
  return int(m[0]) + int(np.sum(~m[:-1] & m[1:]))


def release_point(held: np.ndarray, cube: np.ndarray):
  """Cube position at the last step of the first holding run, or None if never held."""
  h = held.astype(bool)
  if not h.any():
    return None
  start = int(np.argmax(h))
  end = start + int(np.argmax(~h[start:])) if (~h[start:]).any() else len(h)
  return cube[end - 1]


def speed(p: np.ndarray, dt: float) -> np.ndarray:
  v = np.zeros(len(p))
  v[1:] = np.linalg.norm(np.diff(p, axis=0), axis=1) / dt
  return v


def pal_episode(
  policy_csv: Path, joints_csv: Path, goal: np.ndarray, dt: float
) -> dict:
  d = np.genfromtxt(policy_csv, delimiter=",", names=True)
  t = d["t"]
  held = d["cube_state"] == 1
  placed = d["placed"] > 0
  tcp = np.stack([d["tcp_x"], d["tcp_y"], d["tcp_z"]], axis=1)
  cube = np.stack([d["cube_x"], d["cube_y"], d["cube_z"]], axis=1)
  q = np.stack([d[f"q_policy{j}"] for j in range(8)], axis=1)
  target = np.stack([d[f"target_policy{j}"] for j in range(8)], axis=1)
  err = (target - q)[:, :ARM]
  v_tcp = speed(tcp, dt)
  rel = release_point(held, cube)
  out = {
    "steps": int(len(t)),
    "tcp": tcp,
    "placed": bool(placed.any()),
    "release_point": rel.round(4).tolist() if rel is not None else None,
    "release_xy_to_goal": float(np.linalg.norm((rel - goal)[:2]))
    if rel is not None
    else None,
    "t_first_grasp": float(t[np.argmax(held)]) if held.any() else None,
    "t_first_place": float(t[np.argmax(placed)]) if placed.any() else None,
    "grasps": runs(held),
    "final_cube_goal_dist": float(np.linalg.norm(cube[-1] - goal)),
    "peak_tcp_speed": float(v_tcp.max()),
    "peak_tcp_speed_held": float(v_tcp[held].max()) if held.any() else None,
    "tcp_min_z_b": float(tcp[:, 2].min()),
    "track_err_rms": np.sqrt(np.mean(err**2, axis=0)).round(4).tolist(),
    "track_err_max": np.abs(err).max(axis=0).round(4).tolist(),
  }
  if joints_csv.exists():
    j = np.genfromtxt(joints_csv, delimiter=",", names=True)
    qd = np.abs(np.stack([j[f"qd{k}"] for k in range(ARM)], axis=1))
    tau = np.abs(np.stack([j[f"tau{k}"] for k in range(ARM)], axis=1))
    rate = float(1.0 / np.median(np.diff(j["t"])))
    # Sustained speed: |qd| averaged over 20 ms, which a contact impulse (one 0.5 ms sample at
    # the grasp) does not survive; the raw peak is kept as the impulse peak.
    n20 = max(1, int(round(0.02 * rate)))
    qd20 = np.stack(
      [np.convolve(qd[:, k], np.ones(n20) / n20, mode="valid") for k in range(ARM)],
      axis=1,
    )
    out.update(
      {
        "joint_rate_hz": rate,
        "impulse_peak_qd": qd.max(axis=0).round(3).tolist(),
        "peak_qd": qd20.max(axis=0).round(3).tolist(),
        "peak_qd_over_limit": (qd20.max(axis=0) / VEL_LIMIT).round(3).tolist(),
        "ms_over_vel_limit": (np.sum(qd20 > VEL_LIMIT, axis=0) / rate * 1000)
        .round(1)
        .tolist(),
        # As for speed: torque averaged over 20 ms is the sustained demand; the raw peak is
        # dominated by the damping term answering a one-sample velocity impulse at an impact.
        "impulse_peak_tau": tau.max(axis=0).round(2).tolist(),
        "impulse_peak_tau_over_limit": (tau.max(axis=0) / EFFORT_LIMIT)
        .round(3)
        .tolist(),
        "peak_tau": np.stack(
          [
            np.convolve(tau[:, k], np.ones(n20) / n20, mode="valid") for k in range(ARM)
          ],
          axis=1,
        )
        .max(axis=0)
        .round(2)
        .tolist(),
      }
    )
  if "peak_tau" in out:
    out["peak_tau_over_limit"] = (
      (np.array(out["peak_tau"]) / EFFORT_LIMIT).round(3).tolist()
    )
  return out


def mujoco_episode(ref: dict, k: int, goal_b: np.ndarray, dt: float) -> dict:
  """The reference, measured the way the PAL node measures (placed needs a lift first).

  An env that terminates (dropped, out of reach) is reset by the rollout, so the episode is
  cut after its first `done` step."""
  done = ref["done"][k].astype(bool)
  n = int(np.argmax(done)) + 1 if done.any() else len(done)
  g = ref["grasped"][k][:n].astype(bool)
  on = ref["on_surface"][k][:n].astype(bool)
  cube = ref["cube_b"][k][:n]
  tcp = ref["tcp_b"][k][:n]
  q, target = ref["q"][k][:n], ref["target"][k][:n]
  was_lifted = np.cumsum(g & ~on) > 0
  v_cube = speed(cube, dt)
  # cube_b is in base_link; goal_b likewise (export_for_pal.py)
  at_goal = np.linalg.norm(cube - goal_b, axis=1) < 0.025
  placed = at_goal & ~g & (v_cube < 0.05) & was_lifted
  v_tcp = speed(tcp, dt)
  err = (target - q)[:, :ARM]
  qd = np.abs(ref["qd"][k][:n, :ARM])
  t = np.arange(len(g)) * dt
  rel = release_point(g, cube)
  return {
    "steps": int(len(g)),
    "tcp": tcp,
    "release_point": rel.round(4).tolist() if rel is not None else None,
    "release_xy_to_goal": float(np.linalg.norm((rel - goal_b)[:2]))
    if rel is not None
    else None,
    "terminated_early": bool(done.any()),
    "placed": bool(placed.any()),
    "t_first_grasp": float(t[np.argmax(g)]) if g.any() else None,
    "t_first_place": float(t[np.argmax(placed)]) if placed.any() else None,
    "grasps": runs(g),
    "final_cube_goal_dist": float(np.linalg.norm(cube[-1] - goal_b)),
    "peak_tcp_speed": float(v_tcp.max()),
    "peak_tcp_speed_held": float(v_tcp[g].max()) if g.any() else None,
    "tcp_min_z_b": float(tcp[:, 2].min()),
    "track_err_rms": np.sqrt(np.mean(err**2, axis=0)).round(4).tolist(),
    "track_err_max": np.abs(err).max(axis=0).round(4).tolist(),
    "peak_qd": qd.max(axis=0).round(3).tolist(),
    "peak_qd_over_limit": (qd.max(axis=0) / VEL_LIMIT).round(3).tolist(),
    "ms_over_vel_limit": (np.sum(qd > VEL_LIMIT, axis=0) * dt * 1000).round(1).tolist(),
  }


def aggregate(rows: list[dict]) -> dict:
  def med(key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.median(vals)) if vals else None

  return {
    "episodes": len(rows),
    "placed": int(sum(r["placed"] for r in rows)),
    "median_release_xy_to_goal": med("release_xy_to_goal"),
    "median_tcp_vs_mujoco_rms_1s": med("tcp_vs_mujoco_rms_1s"),
    "median_release_vs_mujoco_xy": med("release_vs_mujoco_xy"),
    "grasped": int(sum(r["t_first_grasp"] is not None for r in rows)),
    "median_t_first_grasp": med("t_first_grasp"),
    "median_t_first_place": med("t_first_place"),
    "mean_grasps": float(np.mean([r["grasps"] for r in rows])),
    "median_peak_tcp_speed": med("peak_tcp_speed"),
    "median_peak_tcp_speed_held": med("peak_tcp_speed_held"),
    "max_peak_qd_over_limit": float(
      max(max(r["peak_qd_over_limit"]) for r in rows if "peak_qd_over_limit" in r)
    )
    if any("peak_qd_over_limit" in r for r in rows)
    else None,
    "max_impulse_peak_tau_over_limit": float(
      max(
        max(r["impulse_peak_tau_over_limit"])
        for r in rows
        if "impulse_peak_tau_over_limit" in r
      )
    )
    if any("impulse_peak_tau_over_limit" in r for r in rows)
    else None,
    "max_peak_tau_over_limit": float(
      max(max(r["peak_tau_over_limit"]) for r in rows if "peak_tau_over_limit" in r)
    )
    if any("peak_tau_over_limit" in r for r in rows)
    else None,
    "median_track_err_rms_per_joint": np.median(
      np.array([r["track_err_rms"] for r in rows]), axis=0
    )
    .round(4)
    .tolist(),
  }


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--run", type=Path, required=True)
  p.add_argument("--model", type=Path, required=True)
  args = p.parse_args()
  prof = yaml.safe_load((args.model / "pick_place_profile.yaml").read_text())
  goal = np.array(prof["goal_position_b"])
  dt = 1.0 / float(prof["policy_rate"])
  ref = dict(np.load(args.model / "mujoco_reference.npz"))
  ids = [int(i) for i in ref["episode_ids"]]

  pal, mj = [], []
  window = int(round(1.0 / dt))  # the first second: approach, grasp, carry, release
  for k, ep in enumerate(ids):
    pc = args.run / f"ep{ep:02d}_policy.csv"
    if not pc.exists():
      continue
    row = pal_episode(pc, args.run / f"ep{ep:02d}_joints.csv", goal, dt)
    row["episode"] = ep
    pal.append(row)
    mrow = mujoco_episode(ref, k, goal, dt)
    mrow["episode"] = ep
    m = min(window, len(row["tcp"]), len(mrow["tcp"]))
    d = np.linalg.norm(row["tcp"][:m] - mrow["tcp"][:m], axis=1)
    row["tcp_vs_mujoco_rms_1s"] = float(np.sqrt(np.mean(d**2)))
    row["tcp_vs_mujoco_max_1s"] = float(d.max())
    if row["release_point"] is not None and mrow["release_point"] is not None:
      row["release_vs_mujoco_xy"] = float(
        np.linalg.norm(np.subtract(row["release_point"], mrow["release_point"])[:2])
      )
    del row["tcp"], mrow["tcp"]
    mj.append(mrow)
  summary = {
    "run": str(args.run),
    "model": str(args.model),
    "source_checkpoint": prof.get("source_checkpoint"),
    "pal": {"aggregate": aggregate(pal), "episodes": pal},
    "mujoco": {"aggregate": aggregate(mj), "episodes": mj},
  }
  (args.run / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
  for name in ("mujoco", "pal"):
    a = summary[name]["aggregate"]
    print(
      f"{name:>6}: placed {a['placed']}/{a['episodes']}, grasped {a['grasped']}, "
      f"t_grasp {a['median_t_first_grasp']}, t_place {a['median_t_first_place']}, "
      f"grasps/ep {a['mean_grasps']:.2f}, peak TCP {a['median_peak_tcp_speed']}, "
      f"held {a['median_peak_tcp_speed_held']}, qd/limit {a['max_peak_qd_over_limit']}, "
      f"tau/limit {a['max_peak_tau_over_limit']}, release-to-goal {a['median_release_xy_to_goal']}, "
      f"TCP vs MuJoCo (1 s RMS) {a['median_tcp_vs_mujoco_rms_1s']}, release vs MuJoCo "
      f"{a['median_release_vs_mujoco_xy']}"
    )
  for r_mj, r_pal in zip(mj, pal, strict=True):
    print(
      f"  ep{r_pal['episode']}: MJ placed {r_mj['placed']} @ {r_mj['t_first_place']} | "
      f"PAL placed {r_pal['placed']} @ {r_pal['t_first_place']} (grasps {r_pal['grasps']}, "
      f"first grasp {r_pal['t_first_grasp']}, final dist {r_pal['final_cube_goal_dist']:.3f})"
    )


if __name__ == "__main__":
  main()
