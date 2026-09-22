"""Read an mjlab NaN dump and say which joint went first.

`NanGuard` (off by default; `--enable-nan-guard True`) keeps a rolling buffer of physics
states and writes them next to the compiled model when any world goes non-finite. The dump
answers the question the training log cannot: *which* degree of freedom diverged, how long
before the blow-up it started, and whether it was moving or being pushed.

    uv run --extra cpu python scripts/nan_dump_report.py artifacts/nan_dumps/nan_dump_latest.npz

Prints, per dumped world: the joints ranked by peak speed over the buffer, when each first
crossed `--vel-threshold`, and the last finite state of the worst few. A four-bar ringing
itself apart and a contact solver failing look different here -- the first shows one small
passive joint climbing over tens of steps, the second shows everything move at once.
"""

from __future__ import annotations

import argparse
import pathlib

import mujoco
import numpy as np


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("dump", type=pathlib.Path, help="nan_dump_*.npz written by NanGuard")
  p.add_argument(
    "--model",
    type=pathlib.Path,
    default=None,
    help="model_*.mjb; defaults to the name recorded in the dump's metadata",
  )
  p.add_argument("--vel-threshold", type=float, default=20.0, help="rad/s or m/s")
  p.add_argument("--top", type=int, default=8, help="joints to list per world")
  return p.parse_args()


def load(args: argparse.Namespace):
  dump = np.load(args.dump, allow_pickle=True)
  meta = dump["_metadata"].item()
  model_path = args.model or args.dump.parent / meta["model_file"]
  model = mujoco.MjModel.from_binary_path(str(model_path))
  steps = sorted(k for k in dump.files if k.startswith("states_step_"))
  return dump, meta, model, steps


def joint_names(model: mujoco.MjModel) -> list[str]:
  return [
    (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"<j{j}>").split("/", 1)[
      -1
    ]
    for j in range(model.njnt)
  ]


def main() -> None:
  args = parse_args()
  dump, meta, model, steps = load(args)
  data = mujoco.MjData(model)
  names = joint_names(model)
  dofadr = [model.jnt_dofadr[j] for j in range(model.njnt)]

  print(f"dump          {args.dump}")
  print(f"worlds        {meta['num_envs_total']} total, NaN in {meta['nan_env_ids']}")
  print(
    f"buffer        {len(steps)} states, detection at step {meta['detection_step']}"
  )

  for idx, env_id in enumerate(meta["dumped_env_ids"]):
    # (steps, nv) speed per DOF, and the same per joint for readability.
    speeds = np.full((len(steps), model.njnt), np.nan)
    for t, key in enumerate(steps):
      state = dump[key][idx]
      if not np.isfinite(state).all():
        continue
      mujoco.mj_setState(model, data, state, meta["state_spec"])
      for j in range(model.njnt):
        width = 6 if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1
        speeds[t, j] = np.abs(data.qvel[dofadr[j] : dofadr[j] + width]).max()

    finite = np.isfinite(speeds).all(axis=1)
    last_finite = int(np.nonzero(finite)[0].max()) if finite.any() else -1
    peak = np.nanmax(speeds, axis=0)
    order = np.argsort(-np.nan_to_num(peak))[: args.top]

    print(
      f"\n--- world {env_id} --- last fully finite buffered state: index {last_finite}"
    )
    print(f"{'joint':34s} {'peak |v|':>9s} {'first over':>11s} {'at last finite':>14s}")
    for j in order:
      over = np.nonzero(speeds[:, j] > args.vel_threshold)[0]
      first = int(over.min()) if over.size else None
      at_end = speeds[last_finite, j] if last_finite >= 0 else np.nan
      print(
        f"{names[j]:34s} {peak[j]:9.1f} "
        f"{('step ' + str(first)) if first is not None else '-':>11s} {at_end:14.2f}"
      )

    worst = int(order[0])
    trace = speeds[max(0, last_finite - 15) : last_finite + 1, worst]
    print(f"  {names[worst]} over the last {len(trace)} finite states:")
    print("   " + " ".join(f"{v:7.1f}" for v in trace))


if __name__ == "__main__":
  main()
