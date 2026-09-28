"""One markdown table over every analysed PAL condition (each run dir's summary.json).

.venv/bin/python scripts/sim2sim/results_table.py <runs_dir> [<runs_dir> ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def fmt(v, spec="{:.2f}"):
  return "—" if v is None else spec.format(v)


def main() -> None:
  rows = []
  for d in sys.argv[1:]:
    f = Path(d) / "summary.json"
    if not f.exists():
      continue
    s = json.loads(f.read_text())
    p, m = s["pal"]["aggregate"], s["mujoco"]["aggregate"]
    rows.append(
      "| {} | {}/{} ({}) | {}/{} ({}) | {} / {} | {} | {} / {} | {} / {} | {} | {} |".format(
        Path(d).name,
        m["placed"],
        m["episodes"],
        fmt(m["median_t_first_place"]),
        p["placed"],
        p["episodes"],
        fmt(p["median_t_first_place"]),
        fmt(m["median_t_first_grasp"]),
        fmt(p["median_t_first_grasp"]),
        fmt(p["mean_grasps"], "{:.1f}"),
        fmt(m["median_peak_tcp_speed"]),
        fmt(p["median_peak_tcp_speed"]),
        fmt(m["max_peak_qd_over_limit"]),
        fmt(p["max_peak_qd_over_limit"]),
        fmt(p["max_peak_tau_over_limit"]),
        fmt(
          None
          if p["median_tcp_vs_mujoco_rms_1s"] is None
          else 100 * p["median_tcp_vs_mujoco_rms_1s"],
          "{:.1f}",
        ),
      )
    )
  print(
    "| condition | MuJoCo placed (median t, s) | PAL placed (median t, s) | first grasp MuJoCo / PAL (s) "
    "| PAL grasps per episode | peak TCP m/s MuJoCo / PAL | max joint speed / URDF limit MuJoCo / PAL "
    "| PAL max torque / limit | PAL TCP vs MuJoCo, 1 s RMS (cm) |"
  )
  print("|---|---|---|---|---|---|---|---|---|")
  print("\n".join(rows))


if __name__ == "__main__":
  main()
