"""Where the policy's action box actually takes the grasp point, and how hard it shakes the jaw.

S5.0 gates 2 and 3 of the pick-and-place plan, measured locally (CPU) for both the scales
C1/C2 ran with (``--scales old``) and the S5.0 ones (``--scales new``, the default):

``commanded``
    Forward kinematics of the commanded targets ``default + scale * a`` for uniform
    ``a in [-1, 1]^7``. This is what S3 report section 5 measured (16.3 cm at the old scales).
    It ignores the table, the torso, joint limits and the PD loop.

``reached``
    The same actions *held* for ``--hold`` seconds in the real env, and the TCP read back from
    physics. Collisions, joint limits and tracking error all apply; this is where the arm
    actually gets to, and the number the 2026-09-23 per-task doc measured (12.1 cm, old
    scales).

``shake``
    Fresh uniform random actions every control step, jaw included -- the harshest excitation
    the policy can produce at initialisation -- and the peak ``|qvel|`` of the jaw's passive
    four-bar joints plus the count of ``non_finite_state`` terminations. A local proxy for the
    NaN gate; the gate itself is the C0-style job on IAS.

    uv run --extra cpu python scripts/action_envelope.py --scales new --json artifacts/s5.0/envelope_new.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

TASK = "Mjlab-PickPlace-Pal-Kangaroo-Grippers"
#: The passive joints of the left jaw's four-bar: everything on the gripper but the motor.
PASSIVE_REGEX = r"gripper_left_(?!finger_joint).*_joint"
#: C1/C2's scales, before S5.0 (KANGAROO_GRIPPERS').
OLD_SCALES = {
  r"arm_left_[1-5]_joint": 0.10,
  r"arm_left_[67]_joint": 0.05,
  r"gripper_left_finger_joint": 0.20,
}


def robot_cfg(which: str):
  from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS_LEFT_ARM

  if which == "new":
    return KANGAROO_GRIPPERS_LEFT_ARM
  return dataclasses.replace(
    KANGAROO_GRIPPERS_LEFT_ARM, action_scale=OLD_SCALES, action_clip=None
  )


def _stats(disp: np.ndarray, pts: np.ndarray, start: np.ndarray) -> dict:
  return dict(
    max_displacement_cm=float(100 * disp.max()),
    p95_displacement_cm=float(100 * np.percentile(disp, 95)),
    box_min=pts.min(0).round(3).tolist(),
    box_max=pts.max(0).round(3).tolist(),
    start=start.round(3).tolist(),
  )


def commanded(robot, samples: int, seed: int) -> dict:
  """FK of the commanded targets, as S3 section 5 did."""
  import re

  from scene_model import compile_scene
  from scripted_pick_place import ARM_JOINTS, Kin

  kin = Kin.from_model(compile_scene(TASK))
  scale = np.array(
    [
      next(v for p, v in robot.action_scale.items() if re.fullmatch(p, j))
      for j in ARM_JOINTS
    ]
  )
  rng = np.random.default_rng(seed)
  start, _ = kin.fk(kin.q_default)
  pts = np.empty((samples, 3))
  for i in range(samples):
    a = rng.uniform(-1, 1, len(ARM_JOINTS))
    pts[i], _ = kin.fk(kin.q_default + scale * a)
  return _stats(np.linalg.norm(pts - start, axis=1), pts, start)


def _env(robot, num_envs: int, seed: int, device: str):
  import pal_mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import (
    make_pick_place_env_cfg,
  )

  cfg = make_pick_place_env_cfg(robot, gripper_name="left")
  cfg.scene.num_envs = num_envs
  cfg.seed = seed
  # The cube is not what is being measured, and a success termination mid-hold would reset
  # the arm under the measurement.
  cfg.terminations.pop("success", None)
  return ManagerBasedRlEnv(cfg=cfg, device=device)


def reached(robot, num_envs: int, batches: int, hold_s: float, seed: int, device: str):
  """Hold random actions and read the TCP back from physics."""
  import torch

  env = _env(robot, num_envs, seed, device)
  r = env.scene["robot"]
  tcp = r.site_names.index(robot.gripper("left").tcp_site)
  env.reset()
  gen = torch.Generator(device=device).manual_seed(seed)
  steps = int(round(hold_s / env.step_dt))
  with torch.inference_mode():
    for _ in range(10):
      env.step(torch.zeros(env.action_space.shape, device=device))
    start = r.data.site_pos_w[0, tcp].cpu().numpy().copy()
    pts = []
    for _ in range(batches):
      a = torch.rand(env.action_space.shape, generator=gen, device=device) * 2 - 1
      a[:, -1] = 0.0  # arm only; the jaw does not move the TCP
      for _ in range(steps):
        env.step(a)
      pts.append((r.data.site_pos_w[:, tcp] - env.scene.env_origins).cpu().numpy())
      # Back to the start pose between batches so every hold starts from the same place.
      for _ in range(steps):
        env.step(torch.zeros_like(a))
  env.close()
  pts = np.concatenate(pts)
  return _stats(np.linalg.norm(pts - start, axis=1), pts, start)


def shake(robot, num_envs: int, steps: int, seed: int, device: str) -> dict:
  """Fresh random actions every step; peak passive-joint speed and divergences."""
  import re

  import torch

  env = _env(robot, num_envs, seed, device)
  r = env.scene["robot"]
  passive = [i for i, n in enumerate(r.joint_names) if re.fullmatch(PASSIVE_REGEX, n)]
  jaw = r.joint_names.index("gripper_left_finger_joint")
  env.reset()
  gen = torch.Generator(device=device).manual_seed(seed)
  peak = torch.zeros(num_envs, device=device)
  peak_jaw = torch.zeros(num_envs, device=device)
  jaw_max = torch.zeros(num_envs, device=device)
  diverged = 0
  episodes = 0
  with torch.inference_mode():
    for _ in range(steps):
      a = torch.rand(env.action_space.shape, generator=gen, device=device) * 2 - 1
      _, _, term, trunc, _ = env.step(a)
      nf = env.termination_manager.get_term("non_finite_state")
      diverged += int(nf.sum())
      episodes += int((term | trunc).sum())
      v = r.data.joint_vel
      ok = torch.isfinite(v).all(dim=-1) & ~nf
      peak = torch.where(ok, torch.maximum(peak, v[:, passive].abs().amax(-1)), peak)
      peak_jaw = torch.where(ok, torch.maximum(peak_jaw, v[:, jaw].abs()), peak_jaw)
      jaw_max = torch.maximum(jaw_max, r.data.joint_pos[:, jaw])
  env.close()
  return dict(
    env_steps=num_envs * steps,
    episodes_ended=episodes,
    non_finite_state=diverged,
    passive_peak_qvel_max=float(peak.max()),
    passive_peak_qvel_median=float(peak.median()),
    jaw_peak_qvel_max=float(peak_jaw.max()),
    jaw_q_max_reached=float(jaw_max.max()),
  )


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--scales", choices=("new", "old"), default="new")
  p.add_argument("--samples", type=int, default=20_000)
  p.add_argument("--num-envs", type=int, default=32)
  p.add_argument("--batches", type=int, default=4)
  p.add_argument("--hold", type=float, default=2.0, help="seconds each action is held")
  p.add_argument("--shake-steps", type=int, default=1000)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cpu")
  p.add_argument(
    "--skip", nargs="*", default=(), choices=("commanded", "reached", "shake")
  )
  p.add_argument("--json", type=pathlib.Path, default=None)
  return p.parse_args()


def main() -> None:
  args = parse_args()
  robot = robot_cfg(args.scales)
  out: dict = dict(scales=args.scales, action_scale=robot.action_scale)
  if "commanded" not in args.skip:
    out["commanded"] = commanded(robot, args.samples, args.seed)
    print("commanded", json.dumps(out["commanded"]))
  if "reached" not in args.skip:
    out["reached"] = reached(
      robot, args.num_envs, args.batches, args.hold, args.seed, args.device
    )
    print("reached  ", json.dumps(out["reached"]))
  if "shake" not in args.skip:
    out["shake"] = shake(robot, args.num_envs, args.shake_steps, args.seed, args.device)
    print("shake    ", json.dumps(out["shake"]))
  if args.json:
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.json}")


if __name__ == "__main__":
  main()
