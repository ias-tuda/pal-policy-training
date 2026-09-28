"""Open-loop replay: PAL's joint response to MuJoCo's commanded targets vs MuJoCo's own.

`replay_reference_targets.py` held each target of a MuJoCo reference episode for one policy
period on the PAL simulator and logged every joint state with the index of the target being
held. MuJoCo's reference logged the state *before* each step, so MuJoCo's state after target
t is `q[t + 1]`; PAL's is the last joint state logged while target t was held. Both are
compared in the POLICY joint convention.

Reports, per arm joint: RMS and max |q_PAL - q_MuJoCo| over the free-space approach (before
MuJoCo's first grasp) and over the whole replay, and the same for the joint velocity.

  .venv/bin/python scripts/sim2sim/analyze_replay.py --run <run_dir> --model <model_dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--run", type=Path, required=True)
  p.add_argument("--model", type=Path, required=True)
  args = p.parse_args()
  prof = yaml.safe_load((args.model / "pick_place_profile.yaml").read_text())
  sign = np.array(prof["joint_signs"])
  ref = np.load(args.model / "mujoco_reference.npz")
  out = {}
  for f in sorted(args.run.glob("replay_ep*_joints.csv")):
    k = int(f.stem.split("_")[1][2:])
    d = np.genfromtxt(f, delimiter=",", names=True)
    steps = d["step"].astype(int)
    q_pal = np.stack([d[f"q{j}"] for j in range(8)], axis=1) * sign
    qd_pal = np.stack([d[f"qd{j}"] for j in range(8)], axis=1) * sign
    T = ref["target"].shape[1] - 1
    rows = [np.flatnonzero(steps == t) for t in range(T)]
    have = [t for t in range(T) if rows[t].size]
    last = np.array([rows[t][-1] for t in have])
    q_p, qd_p = q_pal[last], qd_pal[last]
    q_m = ref["q"][k][np.array(have) + 1]
    qd_m = ref["qd"][k][np.array(have) + 1]
    g = ref["grasped"][k].astype(bool)
    first_grasp = int(np.argmax(g)) if g.any() else T
    free = np.array(have) + 1 < first_grasp
    e, ed = (q_p - q_m)[:, :7], (qd_p - qd_m)[:, :7]

    def stats(x, m):
      return {
        "rms": np.sqrt(np.mean(x[m] ** 2, axis=0)).round(4).tolist(),
        "max": np.abs(x[m]).max(axis=0).round(4).tolist(),
      }

    out[f"episode_{k}"] = {
      "steps_compared": int(len(have)),
      "mujoco_first_grasp_step": first_grasp,
      "q_err_free_space": stats(e, free),
      "q_err_all": stats(e, np.ones(len(have), bool)),
      "qd_err_free_space": stats(ed, free),
      "jaw_err_free_space_max": float(np.abs(q_p[free, 7] - q_m[free, 7]).max()),
    }
    print(
      f"episode {k}: {len(have)} steps, MuJoCo grasp at step {first_grasp}; "
      f"free-space |q err| max per joint {out[f'episode_{k}']['q_err_free_space']['max']}; "
      f"RMS {out[f'episode_{k}']['q_err_free_space']['rms']}"
    )
  (args.run / "replay_summary.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
  main()
