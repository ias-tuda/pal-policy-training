"""Does the robot hold the pose it is told to hold?

S4 of the pick-and-place plan. Runs the real environment -- not a kinematic model -- for
``--steps`` control steps with the *left* arm under random actions and everything else
commanded to its default, and reports, per joint, how far it ends up from that default.

The plan's acceptance is < 0.02 rad for the right arm and the pelvis and < 0.01 m for
``leg_*_length_joint``; the left arm is under random actions and is reported separately,
since its error is the policy's business rather than the reset's. Under ``--agent zero`` the
left arm is held to 0.02 rad as well (plan S8.1 gate 1: it is the arm that grasps).

``--no-hold`` drops the ``hold_nominal_targets`` event, which is the behaviour every
manipulation task had before S4: the comparison between the two runs is the evidence for
keeping it.

``--gains`` builds the robot on a plan-S8 gain code (`kangaroo_constants.TABLETOP_GAINS`):
G0 is the plant every run before S8 trained on, G15 / G20 / G20s stiff pelvis/arm gains
without gravity compensation. For each pelvis and arm joint it then also reports the gravity
load at the start pose (``qfrc_bias`` at zero velocity), the stiffness that load needs for
0.02 rad of droop, the peak joint speed, and the share of control steps the joint's actuator
torque sits at its effort limit. The plan's S8.1 gates 1 and 2:

    uv run --extra cpu python scripts/hold_check.py --task Mjlab-PickPlace-Pal-Kangaroo-Grippers \\
        --agent zero --gains G20 --out artifacts/s8/gates
    SLEW=0.5 uv run --extra cpu python scripts/hold_check.py \\
        --task Mjlab-PickPlace-Pal-Kangaroo-Grippers --agent random-left --gains G20 \\
        --out artifacts/s8/gates
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import re
import sys

import mujoco
import numpy as np
import pal_mjlab.tasks  # noqa: F401  # populates the task registry
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import (
  DEFAULT_TABLETOP_GAINS,
  TABLETOP_GAINS,
  get_kangaroo_grippers_tabletop_robot_cfg,
)
from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS

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

#: Plan S8.1 gate 1: with zero actions the left arm has to hold its pose too, and to the
#: same 0.02 rad -- stricter than S4, since it is the arm that grasps.
ZERO_ACTION_LEFT_ARM_LIMIT = 0.02

#: The joints whose gains S8 changes, and the droop the per-joint table sizes K against.
GAINED_JOINTS = r"(pelvis_[12]|arm_(left|right)_[1-7])_joint"
DROOP_TARGET = 0.02  # rad

#: An actuator counts as saturated when its joint torque is within this fraction of the
#: effort limit (the limit clamps it exactly; the margin absorbs float32 round-off).
SATURATION_FRACTION = 0.999


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
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
    "--gains",
    choices=tuple(TABLETOP_GAINS),
    default=os.environ.get("GAINS", DEFAULT_TABLETOP_GAINS),
    help="plan-S8 gain code the robot is built on (default: $GAINS, else G0)",
  )
  p.add_argument(
    "--out", type=pathlib.Path, default=pathlib.Path("../docs/reports/pick-place-v0")
  )
  p.add_argument("--tag", default=None)
  return p.parse_args()


def gravcomp_on(args: argparse.Namespace) -> bool:
  return TABLETOP_GAINS[args.gains].gravcomp and not args.no_gravcomp


def left_arm_action_mask(env) -> torch.Tensor:
  """Which entries of the action vector drive the left arm and its jaw."""
  term = env.action_manager.get_term("joint_pos")
  names = term.target_names
  return torch.tensor(
    [("_left_" in n) for n in names], dtype=torch.bool, device=env.device
  )


def gravity_load(env) -> np.ndarray:
  """``qfrc_bias`` per robot joint at the reset pose and zero velocity: gravity alone.

  Computed on the host-side template model, which the robot's compensation does not touch:
  MuJoCo puts gravity compensation in ``qfrc_passive`` / ``qfrc_actuator``, never in
  ``qfrc_bias``, so this is the load K would have to carry without it.
  """
  robot = env.scene["robot"]
  model = env.sim.mj_model
  data = mujoco.MjData(model)
  data.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
  data.qvel[:] = 0.0
  mujoco.mj_forward(model, data)
  return data.qfrc_bias[robot.indexing.joint_v_adr.cpu().numpy()].copy()


def joint_gains(env) -> tuple[np.ndarray, np.ndarray]:
  """(K, effort limit) of the actuator driving each robot joint; nan where none does."""
  robot = env.scene["robot"]
  model = env.sim.mj_model
  qadr = robot.indexing.joint_q_adr.cpu().numpy()
  stiffness = np.full(len(qadr), np.nan)
  effort = np.full(len(qadr), np.nan)
  for a in range(model.nu):
    j = model.actuator_trnid[a, 0]
    hit = np.flatnonzero(qadr == model.jnt_qposadr[j])
    if hit.size:
      stiffness[hit[0]] = model.actuator_gainprm[a, 0]
      effort[hit[0]] = model.actuator_forcerange[a, 1]
  return stiffness, effort


def run(args: argparse.Namespace) -> dict:
  torch.manual_seed(args.seed)
  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = 1
  if args.no_hold:
    env_cfg.events.pop("hold_nominal_targets", None)
  # The robot is always rebuilt on the requested code, so the plant never depends on what
  # $GAINS was when the registry was imported. G0 without --no-gravcomp is the registered
  # tabletop robot exactly.
  env_cfg.scene.entities["robot"] = dataclasses.replace(
    KANGAROO_GRIPPERS,
    entity_cfg_fn=lambda: get_kangaroo_grippers_tabletop_robot_cfg(
      gravcomp=False if args.no_gravcomp else None, gains=args.gains
    ),
  ).build_entity_cfg()
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")

  robot = env.scene["robot"]
  names = list(robot.joint_names)
  env.reset()
  default = robot.data.default_joint_pos.clone()
  tau_g = gravity_load(env)
  stiffness, effort = joint_gains(env)
  mask = left_arm_action_mask(env)
  shape = env.action_space.shape
  guard = "non_finite_state" in env.termination_manager.active_terms

  history = np.empty((args.steps, len(names)))
  qvel = np.empty((args.steps, len(names)))
  torque = np.empty((args.steps, len(names)))
  non_finite = 0
  for step in range(args.steps):
    action = torch.zeros(shape)
    if args.agent == "random-left":
      action[:, mask] = 2 * torch.rand(int(mask.sum())) - 1
    env.step(action)
    if guard:
      non_finite += int(env.termination_manager.get_term("non_finite_state").sum())
    history[step] = (robot.data.joint_pos - default).abs().numpy()[0]
    qvel[step] = robot.data.joint_vel.abs().numpy()[0]
    torque[step] = robot.data.qfrc_actuator.numpy()[0]
  non_finite += int((~np.isfinite(history)).any(axis=1).sum())
  return dict(
    names=names,
    history=history,
    qvel=qvel,
    torque=torque,
    tau_g=tau_g,
    stiffness=stiffness,
    effort=effort,
    non_finite=non_finite,
    env=env,
  )


def limits(args: argparse.Namespace) -> dict:
  out = {g: lim for g, (_, lim, _) in GROUPS.items()}
  if args.agent == "zero":
    out["left arm"] = ZERO_ACTION_LEFT_ARM_LIMIT
  return out


def report(result: dict, args: argparse.Namespace) -> tuple[bool, dict]:
  names, history = result["names"], result["history"]
  print(
    f"\n{args.steps} control steps, agent={args.agent}, gains={args.gains}, "
    f"hold_nominal_targets={'OFF' if args.no_hold else 'ON'}, "
    f"gravity compensation={'ON' if gravcomp_on(args) else 'OFF'}, "
    f"slew={os.environ.get('SLEW', '1.0')}"
  )
  print(f"{'group / worst joint':38s} {'max |q-q0|':>11s} {'mean':>9s}   {'limit':>7s}")
  passed = True
  groups = {}
  group_limits = limits(args)
  for group, (pattern, _, unit) in GROUPS.items():
    limit = group_limits[group]
    idx = [i for i, n in enumerate(names) if re.fullmatch(pattern, n)]
    if not idx:
      continue
    per_joint_max = history[:, idx].max(axis=0)
    worst = int(np.argmax(per_joint_max))
    hi, mean = per_joint_max[worst], history[:, idx].mean()
    verdict = "" if limit is None else ("  PASS" if hi < limit else "  FAIL")
    if limit is not None and hi >= limit:
      passed = False
    groups[group] = dict(
      worst_joint=names[idx[worst]], max=float(hi), mean=float(mean), limit=limit
    )
    print(
      f"{group + ' / ' + names[idx[worst]]:38s} {hi:11.4f} {mean:9.4f}   "
      f"{('-' if limit is None else f'{limit:.3f}'):>7s} {unit}{verdict}"
    )

  # Per joint, for the joints whose gains S8 changes.
  print(
    f"\n{'joint':20s} {'tau_g Nm':>9s} {'K':>8s} {'tau_g/K':>8s} {'K@0.02':>8s} "
    f"{'max err':>8s} {'end err':>8s} {'peak dq':>8s} {'sat %':>6s}"
  )
  per_joint = {}
  for i, n in enumerate(names):
    if not re.fullmatch(GAINED_JOINTS, n):
      continue
    tau, k, eff = result["tau_g"][i], result["stiffness"][i], result["effort"][i]
    saturated = np.abs(result["torque"][:, i]) >= SATURATION_FRACTION * eff
    row = dict(
      tau_g=float(tau),
      stiffness=float(k),
      droop_tau_over_k=float(abs(tau) / k),
      k_for_droop_target=float(abs(tau) / DROOP_TARGET),
      max_err=float(history[:, i].max()),
      end_err=float(history[-1, i]),
      peak_qvel=float(result["qvel"][:, i].max()),
      saturated_share=float(saturated.mean()),
      effort=float(eff),
    )
    per_joint[n] = row
    print(
      f"{n:20s} {tau:9.2f} {k:8.1f} {row['droop_tau_over_k']:8.4f} "
      f"{row['k_for_droop_target']:8.0f} {row['max_err']:8.4f} {row['end_err']:8.4f} "
      f"{row['peak_qvel']:8.3f} {100 * row['saturated_share']:6.1f}"
    )
  if result["non_finite"]:
    passed = False
  print(f"non-finite steps: {result['non_finite']}")
  return passed, dict(groups=groups, per_joint=per_joint)


def plot(result: dict, args: argparse.Namespace, out: pathlib.Path) -> None:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  names, history = result["names"], result["history"]
  group_limits = limits(args)
  groups = [g for g in GROUPS if any(re.fullmatch(GROUPS[g][0], n) for n in names)]
  fig, axes = plt.subplots(len(groups), 1, figsize=(10, 2.1 * len(groups)), sharex=True)
  t = np.arange(len(history)) * 0.02  # 50 Hz control
  for ax, group in zip(axes, groups, strict=True):
    pattern, _, unit = GROUPS[group]
    limit = group_limits[group]
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
    f"gains={args.gains}, hold_nominal_targets={'OFF' if args.no_hold else 'ON'}, "
    f"gravcomp={'ON' if gravcomp_on(args) else 'OFF'}]"
  )
  fig.tight_layout()
  fig.savefig(out, dpi=140)
  print(f"wrote {out}")


def main() -> None:
  args = parse_args()
  args.out.mkdir(parents=True, exist_ok=True)
  tag = args.tag or (
    f"{'nohold' if args.no_hold else 'hold'}_"
    f"{'gravcomp' if gravcomp_on(args) else 'nogravcomp'}_{args.agent}"
    + ("" if args.gains == DEFAULT_TABLETOP_GAINS else f"_{args.gains}")
  )
  result = run(args)
  passed, table = report(result, args)
  plot(result, args, args.out / f"s4_hold_{tag}.png")
  np.savez_compressed(
    args.out / f"s4_hold_{tag}.npz",
    names=np.asarray(result["names"]),
    history=result["history"],
    qvel=result["qvel"],
    torque=result["torque"],
  )
  (args.out / f"s4_hold_{tag}.json").write_text(
    json.dumps(
      dict(
        task=args.task,
        agent=args.agent,
        gains=args.gains,
        gravcomp=gravcomp_on(args),
        hold_nominal=not args.no_hold,
        slew=os.environ.get("SLEW", "1.0"),
        steps=args.steps,
        seed=args.seed,
        non_finite_steps=result["non_finite"],
        passed=passed,
        **table,
      ),
      indent=1,
    )
  )
  print("acceptance:", "PASS" if passed else "FAIL")


if __name__ == "__main__":
  main()
