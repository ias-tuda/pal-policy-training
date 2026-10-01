"""Summarise a `hold_probe.sh` run (pal-policy-deployment): does the arm hold its start pose under
the gains the controller runs, or does a joint ring / chatter?

  python3 analyze_hold_probe.py <run dir> <model dir> [--window 3.0]

Per joint, over the last `window` seconds of the recording: mean error against the model's
start pose (the gravity droop of a controller without feed-forward is tau_g / K), position std,
rms joint velocity, the peak-to-peak of the reported joint torque, and the share of consecutive
torque samples that change sign (1.0 = a limit cycle at the controller's Nyquist rate).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NAMES = [f"arm_left_{i}_joint" for i in range(1, 8)] + ["gripper_left_finger_joint"]


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("run", type=Path)
  p.add_argument("model", type=Path)
  p.add_argument("--window", type=float, default=3.0)
  p.add_argument("--csv", default="hold_probe.csv")
  args = p.parse_args()
  d = np.genfromtxt(args.run / args.csv, delimiter=",", names=True)
  t = d["t"] - d["t"][0]
  start = json.loads((args.model / "deployment_start_config.json").read_text())["joint_positions"]
  target = np.array([start[n] for n in NAMES])
  m = t > t[-1] - args.window
  print(f"{args.run.name}: {len(t)} samples, {t[-1]:.1f} s; last {args.window:.1f} s")
  print(f"{'joint':26s} {'err':>8s} {'q std':>8s} {'qd rms':>8s} {'tau p2p':>8s} {'tau mean':>9s} {'flips':>6s}")
  worst = 0.0
  for j, n in enumerate(NAMES):
    q, qd, tau = d[f"q{j}"][m], d[f"qd{j}"][m], d[f"tau{j}"][m]
    flips = float(np.mean(np.diff(np.sign(tau)) != 0))
    print(
      f"{n:26s} {q.mean() - target[j]:8.4f} {q.std():8.4f} {np.sqrt((qd**2).mean()):8.2f} "
      f"{np.ptp(tau):8.1f} {tau.mean():9.1f} {flips:6.2f}"
    )
    if j < 7:
      worst = max(worst, float(np.sqrt((qd**2).mean())))
  print(f"worst arm joint rms velocity: {worst:.2f} rad/s")


if __name__ == "__main__":
  main()
