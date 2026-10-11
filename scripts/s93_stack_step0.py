"""Plan S9.3 step 0, item 2: how close to the red cube the stack's base may stand on `WC`.

Runs `scripted_pick_place.rollout` on S9.2's plant (G20, CTP, stack speeds, ``LAG=JS`` -- the script cannot
see the cube through the tracker's lag, as in every scripted step 0 --, baseline B, the WC arm, DR-scene,
``ARM_TABLE=AH10``, ``TABLE_DZ`` as set) with ``GOAL=stack`` on ``TARGET=WC``: the grasp only (approach,
descend, close, lift; S10.1's ``clearance`` with the 5 cm base cube in place of the box), the base pinned at
``--distances`` from the red cube in 8 directions, base yaw 0 / 45 deg, red cube yaw 0 / 30 / 45 deg.

A touch is the base displaced by more than 1 mm or turned by more than 1 deg from where it was put (the
hand or the red cube pushing it; the arm-table contact is counted too). `STACK_MIN_SEPARATION` = the least
distance with no touch + 2 cm.

  TABLE_DZ=22 uv run --extra cu128 python scripts/s93_stack_step0.py clearance --json ../results/S9.3-wide/step0/clearance.json
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import scripted_pick_place as scripted  # noqa: E402

RCTP = "R-C-strict+M1+M2+M3+M6+J+M8"
GRASP_PHASES = scripted.PHASES[:4]  # approach, descend, close, lift
#: The red cube's pose for the sweep: the middle of WC, env frame (x 0.40-0.60, y -0.04-0.26).
CLEARANCE_CUBE = (0.50, 0.11)
#: What counts as the base touched: displaced (m), turned (rad).
MOVED = 0.001
TURNED = math.radians(1.0)


def _common(args: argparse.Namespace) -> dict:
  from pal_mjlab.tasks.manipulation.pick_place import pick_place_env_cfg as pp
  from pal_mjlab.tasks.manipulation.pick_place.arm_table import ARM_TABLE_CODES
  from pal_mjlab.tasks.manipulation.pick_place.kangaroo.env_cfg import (
    SPAWN_ROBOTS,
    _baseline_with_table,
  )
  from pal_mjlab.tasks.manipulation.robots import kangaroo_grippers_left_arm
  from pal_mjlab.tasks.manipulation.tabletop_env_cfg import LATENCY_MODELS

  return dict(
    device=args.device,
    seed=args.seed,
    reward=RCTP,
    verbose=False,
    robot=kangaroo_grippers_left_arm(
      "G20",
      SPAWN_ROBOTS["WC"],
      arm_effort_scale=pp.EFFORT_BOUNDS["CTP"].effort_scale,
      right_home="mirror",
    ),
    baseline=_baseline_with_table(pp.BASELINES["B"]),
    spawn_box=pp.SPAWN_BOXES["WC"],
    dr=args.dr,
    effort=pp.EFFORT_BOUNDS["CTP"],
    latency=LATENCY_MODELS["JS"],
    stack_speed=True,
    episode_length_s=120.0,
    target_area=pp.TARGET_AREAS["WC"],
    goal="stack",
    arm_table=ARM_TABLE_CODES["AH10"],
    pad_clearance=args.pad_clearance,
    tracker="split",
  )


def _yaw(q):
  import torch

  w, x, y, z = q.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class _Base:
  """A rollout hook: per env and phase, the steps the base stands displaced / turned from its first
  recorded pose, and arm 1-6 on the table (AH10's contact term)."""

  def __init__(self, n: int):
    self.n = n
    self.start = None
    self.moved: dict[str, np.ndarray] = {}
    self.peak = np.zeros(n)
    self.arm = np.zeros(n, int)

  def __call__(self, phase: str, env) -> None:
    import torch

    data = env.scene["base_object"].data
    pos = data.root_link_pos_w[:, :2].clone()
    yaw = _yaw(data.root_link_quat_w)
    if self.start is None:
      self.start = (pos, yaw)
    d = torch.norm(pos - self.start[0], dim=-1)
    dy = torch.remainder(yaw - self.start[1] + math.pi / 4, math.pi / 2) - math.pi / 4
    hit = ((d > MOVED) | (dy.abs() > TURNED)).cpu().numpy()
    self.moved.setdefault(phase, np.zeros(self.n, int))
    self.moved[phase] += hit.astype(int)
    self.peak = np.maximum(self.peak, d.cpu().numpy())
    term = env.reward_manager.get_term_cfg("arm_table_contact")
    self.arm += term.func(env, **term.params).bool().cpu().numpy().astype(int)

  def row(self, i: int) -> dict:
    return {
      "base_moved_by_phase": {k: int(v[i]) for k, v in self.moved.items() if v[i]},
      "base_peak_mm": round(float(self.peak[i]) * 1000, 2),
      "arm_table_steps": int(self.arm[i]),
    }


def clearance(args: argparse.Namespace) -> dict:
  cases = []
  for d in args.distances:
    for k in range(8):
      th = k * math.pi / 4
      for base_yaw in (0.0, 45.0):
        for cube_yaw in (0.0, 30.0, 45.0):
          cases.append((d, k * 45, base_yaw, cube_yaw, th))
  n = len(cases)
  cx, cy = CLEARANCE_CUBE
  spawns = [((cx, cy), math.radians(c[3])) for c in cases]
  bases = [
    ((cx + c[0] * math.cos(c[4]), cy + c[0] * math.sin(c[4])), math.radians(c[2]))
    for c in cases
  ]
  hook = _Base(n)
  # Only the grasp is run: plan the (unused) carry and place over the cube itself (S10.1's clearance).
  full = scripted.waypoints

  def grasp_only(kin, cube_xy, cube_yaw, target_xy, *a, **kw):
    kw["carry_hand_x"] = None
    kw["place_dz"] = 0.0
    return full(kin, cube_xy, cube_yaw, np.asarray(cube_xy), *a, **kw)

  scripted.waypoints = grasp_only
  try:
    rows = scripted.rollout(
      num_envs=n,
      **_common(args),
      spawns=spawns,
      base_spawns=bases,
      phases=GRASP_PHASES,
      hook=hook,
    )
  finally:
    scripted.waypoints = full
  out = []
  for r in rows:
    i = r["env"]
    d, direction, base_yaw, cube_yaw, _ = cases[i]
    c = hook.row(i)
    out.append(
      {
        "env": i,
        "distance": d,
        "direction_deg": direction,
        "base_yaw_deg": base_yaw,
        "cube_yaw_deg": cube_yaw,
        "plan_error": r["plan_error"],
        "grasped": bool(r.get("grasped")),
        "lifted": bool(r.get("lifted")),
        **c,
        "touch": bool(c["base_moved_by_phase"]),
      }
    )
  summary = {}
  for d in args.distances:
    g = [r for r in out if r["distance"] == d]
    ok = [r for r in g if r["lifted"]]
    summary[f"{d:.2f}"] = {
      "n": len(g),
      "lifted": len(ok),
      "touch": sum(r["touch"] for r in g),
      "touch_when_lifted": sum(r["touch"] for r in ok),
      "plan_errors": sum(r["plan_error"] is not None for r in g),
      "arm_table": sum(r["arm_table_steps"] > 0 for r in g),
      "touch_cases": [
        (r["direction_deg"], r["base_yaw_deg"], r["cube_yaw_deg"], r["base_peak_mm"])
        for r in g
        if r["touch"]
      ][:16],
    }
  clear = [d for d in args.distances if summary[f"{d:.2f}"]["touch_when_lifted"] == 0]
  return {
    "summary": summary,
    "least_clear_distance": min(clear) if clear else None,
    "rows": out,
  }


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("mode", choices=("clearance",))
  ap.add_argument("--json", type=pathlib.Path, required=True)
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--dr", default="DR-scene")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument(
    "--pad-clearance",
    type=float,
    default=0.01,
    help="fingertips above the table at the close, m (1 cm: the lowest grasp, the worst case)",
  )
  ap.add_argument(
    "--distances", type=float, nargs="+", default=(0.08, 0.10, 0.12, 0.14)
  )
  args = ap.parse_args()
  res = clearance(args)
  print(json.dumps(res["summary"], indent=1))
  print(
    "least distance with no base touch on a lifted grasp:", res["least_clear_distance"]
  )
  args.json.parent.mkdir(parents=True, exist_ok=True)
  args.json.write_text(
    json.dumps(
      {"args": vars(args) | {"json": str(args.json)}, **res}, indent=1, default=str
    )
  )


if __name__ == "__main__":
  main()
