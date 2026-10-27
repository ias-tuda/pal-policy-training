"""Figures for the PAL sim2sim report: one episode, PAL against the MuJoCo reference.

Panels: TCP position (base frame) x/y/z, jaw angle and cube state, and |joint velocity| of the
wrist joints against PAL's URDF limit (PAL at the joint-state rate, MuJoCo at the policy rate).

  .venv/bin/python scripts/sim2sim/plot_sim2sim.py --run <run_dir> --model <model_dir> \
      --episode 0 --out <png> [--compare <run_dir2> --compare-label ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

VEL_LIMIT = [1.95, 1.95, 2.35, 2.35, 2.35, 2.35, 2.35]


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--run", type=Path, required=True)
  p.add_argument("--label", default="PAL")
  p.add_argument("--compare", type=Path, default=None)
  p.add_argument("--compare-label", default="PAL (other)")
  p.add_argument("--model", type=Path, required=True)
  p.add_argument("--episode", type=int, default=0)
  p.add_argument("--seconds", type=float, default=2.0)
  p.add_argument("--out", type=Path, required=True)
  args = p.parse_args()

  prof = yaml.safe_load((args.model / "pick_place_profile.yaml").read_text())
  dt = 1.0 / float(prof["policy_rate"])
  ref = np.load(args.model / "mujoco_reference.npz")
  k = list(ref["episode_ids"]).index(args.episode)
  n = int(args.seconds / dt)
  t_mj = np.arange(n) * dt
  goal = np.array(prof["goal_position_b"])

  runs = [(args.run, args.label, "tab:red")]
  if args.compare is not None:
    runs.append((args.compare, args.compare_label, "tab:orange"))

  fig, axes = plt.subplots(5, 1, figsize=(8.5, 12), sharex=True)
  for i, axis in enumerate("xyz"):
    ax = axes[i]
    ax.plot(
      t_mj, ref["tcp_b"][k, :n, i], color="black", lw=2, label="MuJoCo (training sim)"
    )
    for run, label, color in runs:
      d = np.genfromtxt(
        run / f"ep{args.episode:02d}_policy.csv", delimiter=",", names=True
      )
      m = d["t"] <= args.seconds
      ax.plot(d["t"][m], d[f"tcp_{axis}"][m], color=color, lw=1.5, label=label)
    ax.axhline(goal[i], color="tab:green", ls=":", lw=1)
    ax.set_ylabel(f"TCP {axis} [m]")
  axes[0].legend(loc="best", fontsize=8)
  axes[2].axhline(prof["table_top_z_b"], color="saddlebrown", ls="--", lw=1)

  ax = axes[3]
  ax.plot(t_mj, ref["q"][k, :n, 7], color="black", lw=2, label="MuJoCo jaw")
  ax.fill_between(
    t_mj,
    0,
    0.8 * ref["grasped"][k, :n],
    color="black",
    alpha=0.12,
    step="post",
    label="MuJoCo grasped",
  )
  for run, label, color in runs:
    d = np.genfromtxt(
      run / f"ep{args.episode:02d}_policy.csv", delimiter=",", names=True
    )
    m = d["t"] <= args.seconds
    ax.plot(d["t"][m], d["q_policy7"][m], color=color, lw=1.5, label=f"{label} jaw")
    ax.fill_between(
      d["t"][m],
      0,
      0.8 * (d["cube_state"][m] == 1),
      color=color,
      alpha=0.15,
      step="post",
      label=f"{label} holding (virtual cube)",
    )
  ax.set_ylabel("jaw [rad]")
  ax.legend(loc="upper right", fontsize=7)

  ax = axes[4]
  for j, style in zip((4, 5, 6), ("-", "--", ":"), strict=True):
    ax.plot(
      t_mj,
      np.abs(ref["qd"][k, :n, j]),
      color="black",
      ls=style,
      lw=1.5,
      label=f"MuJoCo arm {j + 1} (50 Hz)",
    )
    jc = args.run / f"ep{args.episode:02d}_joints.csv"
    if jc.exists():
      jd = np.genfromtxt(jc, delimiter=",", names=True)
      tj = jd["t"] - jd["t"][0]
      m = tj <= args.seconds
      ax.plot(
        tj[m],
        np.abs(jd[f"qd{j}"][m]),
        color="tab:red",
        ls=style,
        lw=0.8,
        label=f"{args.label} arm {j + 1} (2 kHz)",
      )
  ax.axhline(VEL_LIMIT[4], color="gray", ls="--", lw=1, label="URDF limit 2.35 rad/s")
  ax.set_ylabel("|wrist velocity| [rad/s]")
  ax.set_xlabel("time since episode start [s]")
  ax.legend(loc="upper right", fontsize=7, ncol=2)
  fig.suptitle(
    f"{prof['source_checkpoint'].split('/')[0][20:]} / {Path(prof['source_checkpoint']).name}"
    f" -- episode {args.episode}",
    fontsize=9,
  )
  fig.tight_layout()
  args.out.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(args.out, dpi=110)
  print("wrote", args.out)


if __name__ == "__main__":
  main()
