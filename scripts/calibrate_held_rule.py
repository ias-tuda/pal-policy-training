"""Plan S7-7.0 item 3: calibrate O-K's jaw-only "held" rule against the sim's contact sensor.

Reads the ``--trace`` .npz files `eval_pick_place.py` writes (one per run / lag condition) and replays
the rule on the recorded jaw angle, its command and the last visible cube's distance to the TCP, over
a grid of its parameters (`mdp.OccCfg.hold_*`, ``envelope``). The reference is the contact sensor's
``grasped`` (both pads on the cube). Per parameter set: precision and recall of "held" over all steps,
the median steps it trails the first real grasp of an episode by, and the steps it still says held
after the pads have let go (the estimate then rides with a hand that has left the cube).

    uv run python scripts/calibrate_held_rule.py ../results/S7-7/trace/*.npz --json out.json

The score ranks by F1 with the trail as a tie-break; the printed top rows are read by a person, and
the defaults in `OccCfg` are set from them.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib

import numpy as np

GRID = {
  "lo": (0.08, 0.10, 0.15),
  "hi": (0.43,),
  "stall": (0.30, 0.60, 1.0, 1.5),
  "margin": (0.02, 0.05, 0.10),
  "n": (2, 3, 4),
  "envelope": (0.06, 0.10, 1e9),
  "release": (-9.0, 0.0, 0.05, 0.10, 0.15),
}
OPEN_BELOW = 0.15


def load(paths: list[pathlib.Path]) -> list[dict]:
  runs = []
  for p in paths:
    z = np.load(p)
    runs.append({k: z[k] for k in z.files} | {"name": p.parent.name})
  return runs


def episode_starts(step: np.ndarray) -> np.ndarray:
  """(T, B) True where a new episode starts: the episode clock did not advance."""
  new = np.zeros_like(step, dtype=bool)
  new[0] = True
  new[1:] = step[1:] <= step[:-1]
  return new


def replay(run: dict, lo, hi, stall, margin, n, envelope, release=-9.0) -> np.ndarray:
  """The held rule on a trace, (T, B) bool. Mirrors `CubeSensor.update`'s latch and release."""
  q, cmd, dt = run["q"], run["cmd"], float(run["step_dt"])
  T, B = q.shape
  start = episode_starts(run["step"])
  held = np.zeros((T, B), dtype=bool)
  count = np.zeros(B, dtype=int)
  cur = np.zeros(B, dtype=bool)
  for t in range(T):
    qdot = np.zeros(B) if t == 0 else (q[t] - q[t - 1]) / dt
    qdot = np.where(start[t], 0.0, qdot)
    cand = (
      (q[t] >= lo) & (q[t] <= hi) & (np.abs(qdot) < stall) & ((cmd[t] - q[t]) >= margin)
    )
    count = np.where(cand & ~start[t], count + 1, 0)
    cur = np.where(start[t], False, cur)
    latch = ~cur & (count >= n) & (run["last_tcp"][t] < envelope)
    let_go = cur & ((q[t] < OPEN_BELOW) | (q[t] > hi) | ((cmd[t] - q[t]) < release))
    cur = (cur | latch) & ~let_go
    held[t] = cur
  return held


def score(run: dict, held: np.ndarray) -> dict:
  g = run["grasped"].astype(bool)
  tp = int((held & g).sum())
  fp = int((held & ~g).sum())
  fn = int((~held & g).sum())
  prec = tp / (tp + fp) if tp + fp else 0.0
  rec = tp / (tp + fn) if tp + fn else 0.0
  f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
  start = episode_starts(run["step"])
  # Per episode: the first grasp and the first held step.
  T, B = g.shape
  lags, stale = [], 0
  for b in range(B):
    idx = np.flatnonzero(start[:, b])
    for a, e in zip(idx, [*idx[1:], T], strict=True):
      gg, hh = g[a:e, b], held[a:e, b]
      if gg.any():
        fg = int(np.argmax(gg))
        fh = int(np.argmax(hh)) if hh.any() else None
        if fh is not None and fh >= fg - 2:
          lags.append(fh - fg)
        elif fh is None:
          lags.append(e - a)  # never latched: the whole episode
      # held after the pads let go, once they had held
      if gg.any():
        last_g = len(gg) - 1 - int(np.argmax(gg[::-1]))
        stale += int(hh[last_g + 1 :].sum())
  return {
    "precision": prec,
    "recall": rec,
    "f1": f1,
    "lag_median": float(np.median(lags)) if lags else None,
    "stale_steps": stale,
    "stale_share": stale / max(int(held.sum()), 1),
  }


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("traces", nargs="+", type=pathlib.Path)
  p.add_argument("--json", type=pathlib.Path, default=None)
  p.add_argument("--top", type=int, default=15)
  p.add_argument(
    "--current", action="store_true", help="score OccCfg's current defaults only"
  )
  args = p.parse_args()
  runs = load(args.traces)
  if args.current:
    from pal_mjlab.tasks.manipulation.mdp.occlusion import OccCfg

    c = OccCfg(fill="live", flag=False)
    grid = {
      "lo": (c.hold_band[0],),
      "hi": (c.hold_band[1],),
      "stall": (c.hold_stall_vel,),
      "margin": (c.hold_margin,),
      "n": (c.hold_steps,),
      "envelope": (c.envelope,),
      "release": (c.hold_release,),
    }
  else:
    grid = GRID
  rows = []
  for values in itertools.product(*grid.values()):
    params = dict(zip(grid, values, strict=True))
    per_run = {r["name"]: score(r, replay(r, **params)) for r in runs}
    f1 = float(np.mean([v["f1"] for v in per_run.values()]))
    lag = [v["lag_median"] for v in per_run.values() if v["lag_median"] is not None]
    rows.append(
      {
        "params": params,
        "f1": f1,
        "lag": float(np.mean(lag)) if lag else None,
        "runs": per_run,
      }
    )
  rows.sort(key=lambda r: (-r["f1"], r["lag"] if r["lag"] is not None else 1e9))
  for r in rows[: args.top]:
    pr = r["params"]
    per = "  ".join(
      f"{k}: P {v['precision']:.3f} R {v['recall']:.3f} lag {v['lag_median']} stale {v['stale_share']:.3f}"
      for k, v in r["runs"].items()
    )
    print(f"F1 {r['f1']:.3f} lag {r['lag']}  {pr}\n      {per}")
  if args.json:
    args.json.write_text(json.dumps(rows[:200], indent=1))


if __name__ == "__main__":
  main()
