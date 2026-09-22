"""Does the robot hold the pose it is told to hold?

S4 of the pick-and-place plan. Runs the real environment -- not a kinematic model -- for
``--steps`` control steps with the *left* arm under random actions and everything else
commanded to its default, and reports, per joint, how far it ends up from that default.

The plan's acceptance is < 0.02 rad for the right arm and the pelvis and < 0.01 m for
``leg_*_length_joint``; the left arm is under random actions and is reported separately,
since its error is the policy's business rather than the reset's.

``--no-hold`` drops the ``hold_nominal_targets`` event, which is the behaviour every
manipulation task had before S4: the comparison between the two runs is the evidence for
keeping it.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pal_mjlab.tasks  # noqa: F401  # populates the task registry
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from scene_model import DEFAULT_TASK  # noqa: E402

#: Which joints each acceptance threshold applies to, and the threshold itself. The plan's
#: S4 gates exactly three groups: the pelvis and the right arm at 0.02 rad and the telescopic
#: leg drive at 0.01 m. The other two are reported because they are informative -- the left
#: arm is the one under random actions, and the leg hinges are outside the gate -- but a
#: number without a stated limit is not a pass/fail.
GROUPS = {
  "pelvis": (r"pelvis_\d_joint", 0.02, "rad"),
  "right arm": (r"arm_right_[1-7]_joint", 0.02, "rad"),
  "leg length": (r"leg_.*_length_joint", 0.01, "m"),
  "legs (other)": (r"leg_.*_[1-5]_joint", None, "rad"),
  "left arm": (r"arm_left_[1-7]_joint", None, "rad"),
}


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--task", default=DEFAULT_TASK)
  p.add_argument("--steps", type=int, default=1000)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument(
    "--agent",
    choices=("random-left", "zero"),
    default="random-left",
    help="random actions on the left arm and jaw only, or no action at all",
  )
  p.add_argument(
    "--no-hold",
    action="store_true",
    help="drop the hold_nominal_targets event (half of the pre-S4 behaviour)",
  )
  p.add_argument(
    "--no-gravcomp",
    action="store_true",
    help="rebuild the robot without gravity compensation (the other half)",
  )
  p.add_argument(
    "--out", type=pathlib.Path, default=pathlib.Path("../docs/reports/pick-place-v0")
  )
  p.add_argument("--tag", default=None)
  return p.parse_args()


def left_arm_action_mask(env) -> torch.Tensor:
  """Which entries of the action vector drive the left arm and its jaw."""
  term = env.action_manager.get_term("joint_pos")
  names = term.target_names
  return torch.tensor(
    [("_left_" in n) for n in names], dtype=torch.bool, device=env.device
  )


def run(args: argparse.Namespace) -> dict:
  torch.manual_seed(args.seed)
  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = 1
  if args.no_hold:
    env_cfg.events.pop("hold_nominal_targets", None)
  if args.no_gravcomp:
    import dataclasses

    from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import (
      get_kangaroo_grippers_tabletop_robot_cfg,
    )
    from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS

    env_cfg.scene.entities["robot"] = dataclasses.replace(
      KANGAROO_GRIPPERS,
      entity_cfg_fn=lambda: get_kangaroo_grippers_tabletop_robot_cfg(gravcomp=False),
    ).build_entity_cfg()
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")

  robot = env.scene["robot"]
  names = list(robot.joint_names)
  env.reset()
  default = robot.data.default_joint_pos.clone()
  mask = left_arm_action_mask(env)
  shape = env.action_space.shape

  history = np.empty((args.steps, len(names)))
  for step in range(args.steps):
    action = torch.zeros(shape)
    if args.agent == "random-left":
      action[:, mask] = 2 * torch.rand(int(mask.sum())) - 1
    env.step(action)
    history[step] = (robot.data.joint_pos - default).abs().numpy()[0]
  return dict(names=names, history=history, env=env)


def report(result: dict, args: argparse.Namespace) -> bool:
  import re

  names, history = result["names"], result["history"]
  print(
    f"\n{args.steps} control steps, agent={args.agent}, "
    f"hold_nominal_targets={'OFF' if args.no_hold else 'ON'}, "
    f"gravity compensation={'OFF' if args.no_gravcomp else 'ON'}"
  )
  print(f"{'group / worst joint':38s} {'max |q-q0|':>11s} {'mean':>9s}   {'limit':>7s}")
  passed = True
  for group, (pattern, limit, unit) in GROUPS.items():
    idx = [i for i, n in enumerate(names) if re.fullmatch(pattern, n)]
    if not idx:
      continue
    per_joint_max = history[:, idx].max(axis=0)
    worst = int(np.argmax(per_joint_max))
    hi, mean = per_joint_max[worst], history[:, idx].mean()
    verdict = "" if limit is None else ("  PASS" if hi < limit else "  FAIL")
    if limit is not None and hi >= limit:
      passed = False
    print(
      f"{group + ' / ' + names[idx[worst]]:38s} {hi:11.4f} {mean:9.4f}   "
      f"{('-' if limit is None else f'{limit:.3f}'):>7s} {unit}{verdict}"
    )
  return passed


def plot(result: dict, args: argparse.Namespace, out: pathlib.Path) -> None:
  import re

  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  names, history = result["names"], result["history"]
  groups = [g for g in GROUPS if any(re.fullmatch(GROUPS[g][0], n) for n in names)]
  fig, axes = plt.subplots(len(groups), 1, figsize=(10, 2.1 * len(groups)), sharex=True)
  t = np.arange(len(history)) * 0.02  # 50 Hz control
  for ax, group in zip(axes, groups, strict=True):
    pattern, limit, unit = GROUPS[group]
    for i, name in enumerate(names):
      if re.fullmatch(pattern, name):
        ax.plot(t, history[:, i], lw=1.0, label=name)
    if limit is not None:
      ax.axhline(limit, color="k", ls="--", lw=1.0, label=f"limit {limit} {unit}")
    ax.set_ylabel(f"|q - q0|\n[{unit}]")
    ax.set_title(group, fontsize=9, loc="left")
    ax.legend(fontsize=6, ncols=4, loc="upper right")
    ax.grid(alpha=0.3)
  axes[-1].set_xlabel("time [s]")
  fig.suptitle(
    f"Tracking error against the configured default  [agent={args.agent}, "
    f"hold_nominal_targets={'OFF' if args.no_hold else 'ON'}, "
    f"gravcomp={'OFF' if args.no_gravcomp else 'ON'}]"
  )
  fig.tight_layout()
  fig.savefig(out, dpi=140)
  print(f"wrote {out}")


def main() -> None:
  args = parse_args()
  args.out.mkdir(parents=True, exist_ok=True)
  tag = args.tag or (
    f"{'nohold' if args.no_hold else 'hold'}_"
    f"{'nogravcomp' if args.no_gravcomp else 'gravcomp'}_{args.agent}"
  )
  result = run(args)
  passed = report(result, args)
  plot(result, args, args.out / f"s4_hold_{tag}.png")
  np.savez_compressed(
    args.out / f"s4_hold_{tag}.npz",
    names=np.asarray(result["names"]),
    history=result["history"],
  )
  print("acceptance:", "PASS" if passed else "FAIL")


if __name__ == "__main__":
  main()
