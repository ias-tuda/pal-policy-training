"""Gate evaluation for the loco-manipulation lower-body policy (locomanip_docs/plan.md §P1).

Two closed-loop checks on the play config of
`Mjlab-LocoManip-Velocity-Flat-Pal-Kangaroo-Grippers`, all envs in parallel:

stand    zero twist command, scripted arms at curriculum level 1 (full amplitude, jaw
         toggling, hand payload from the startup DR). Reports the fraction of envs that
         never terminate (fell_over / illegal_contacts) and the horizontal drift of the base
         from its start position. Bar 1: >= 95 % upright over 20 s, drift < 10 cm.

walk_to  one random target pose per env, at distance `--walk-radius` max (uniform in the
         disc) and with a uniform yaw. A P-controller writes the twist command each step
         (body-frame position error x gain, clipped to the trained command range; yaw
         error x gain); inside the tolerance the command is zero. Arms hold the home pose.
         Reports final position / yaw error. Bar 2: < 5 cm and < 5 deg.

Usage:
  uv run python scripts/locomanip_eval.py --checkpoint logs/rsl_rl/kangaroo_locomanip/<run>/model_N.pt
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import pal_mjlab.tasks  # noqa: F401  (registers the task ids)
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import wrap_to_pi

TASK_ID = "Mjlab-LocoManip-Velocity-Flat-Pal-Kangaroo-Grippers"


def _make(num_envs: int, device: str, arm_hold: bool, seed: int):
  cfg = load_env_cfg(TASK_ID, play=True)
  cfg.scene.num_envs = num_envs
  cfg.seed = seed
  twist = cfg.commands["twist"]
  # The command is written by this script: no resampling, no standing/heading/world
  # envs that would overwrite it in `_update_command`.
  twist.resampling_time_range = (1e9, 1e9)
  twist.rel_standing_envs = 0.0
  twist.rel_heading_envs = 0.0
  twist.heading_command = False
  twist.ranges.heading = None
  if hasattr(twist, "rel_world_envs"):
    twist.rel_world_envs = 0.0
  twist.rel_forward_envs = 0.0
  arms = cfg.actions["arms"]
  arms.level_ramp_steps = 0  # level 1
  if arm_hold:
    arms.hold_probability = 1.0
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  return RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(TASK_ID).clip_actions)


def _policy(env, checkpoint: str, device: str):
  agent_cfg = load_rl_cfg(TASK_ID)
  runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
  return runner.get_inference_policy(device=device)


def _yaw(quat_wxyz: torch.Tensor) -> torch.Tensor:
  w, x, y, z = quat_wxyz.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def run_stand(args) -> dict:
  env = _make(args.num_envs, args.device, arm_hold=False, seed=args.seed)
  policy = _policy(env, args.checkpoint, args.device)
  raw = env.unwrapped
  robot = raw.scene["robot"]
  cmd = raw.command_manager.get_term("twist")
  obs = env.get_observations()
  start = robot.data.root_link_pos_w[:, :2].clone()
  failed = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
  steps = int(args.duration_s / raw.step_dt)
  with torch.no_grad():
    for _ in range(steps):
      cmd.vel_command_b[:] = 0.0
      obs, _, dones, _ = env.step(policy(obs))
      failed |= dones.bool()
  drift = (robot.data.root_link_pos_w[:, :2] - start).norm(dim=-1)
  ok = ~failed
  out = {
    "upright_frac": ok.float().mean().item(),
    "drift_mean_m": drift[ok].mean().item() if ok.any() else float("nan"),
    "drift_max_m": drift[ok].max().item() if ok.any() else float("nan"),
  }
  env.close()
  return out


def run_walk_to(args) -> dict:
  env = _make(args.num_envs, args.device, arm_hold=True, seed=args.seed)
  policy = _policy(env, args.checkpoint, args.device)
  raw = env.unwrapped
  robot = raw.scene["robot"]
  cmd = raw.command_manager.get_term("twist")
  rng = cmd.cfg.ranges
  obs = env.get_observations()
  n, dev = args.num_envs, args.device
  g = torch.Generator(device=dev).manual_seed(args.seed)
  r = args.walk_radius * torch.rand(n, device=dev, generator=g).sqrt()
  phi = 2 * math.pi * torch.rand(n, device=dev, generator=g)
  start = robot.data.root_link_pos_w[:, :2].clone()
  target_xy = start + torch.stack((r * phi.cos(), r * phi.sin()), dim=-1)
  target_yaw = (2 * torch.rand(n, device=dev, generator=g) - 1) * math.pi
  failed = torch.zeros(n, dtype=torch.bool, device=dev)
  steps = int(args.duration_s / raw.step_dt)
  with torch.no_grad():
    for _ in range(steps):
      pos = robot.data.root_link_pos_w[:, :2]
      yaw = _yaw(robot.data.root_link_quat_w)
      e_w = target_xy - pos
      c, s = yaw.cos(), yaw.sin()
      e_b = torch.stack(
        (c * e_w[:, 0] + s * e_w[:, 1], -s * e_w[:, 0] + c * e_w[:, 1]), -1
      )
      e_yaw = wrap_to_pi(target_yaw - yaw)
      # P-control with a speed floor: the policy does not respond to very small twist
      # commands, so a pure P-law stalls short of the target.
      d = e_b.norm(dim=-1, keepdim=True).clamp(min=1e-6)
      speed = (args.k_pos * d).clamp(min=args.v_min, max=args.v_max)
      vx, vy = (e_b / d * speed).unbind(-1)
      w = (args.k_yaw * e_yaw.abs()).clamp(min=args.w_min, max=rng.ang_vel_z[1])
      wz = torch.sign(e_yaw) * w
      # Position first, then yaw: turning on the spot is cleaner than both at once.
      far = d.squeeze(-1) >= args.tol_pos
      vx, vy = torch.where(far, vx, 0.0), torch.where(far, vy, 0.0)
      wz = torch.where(e_yaw.abs() >= args.tol_yaw, wz, 0.0)
      # A dead band at the tolerance so the robot can come to rest.
      inside = (e_w.norm(dim=-1) < args.tol_pos) & (e_yaw.abs() < args.tol_yaw)
      v = torch.stack((vx, vy, wz), -1)
      v[inside] = 0.0
      cmd.vel_command_b[:] = v
      obs, _, dones, _ = env.step(policy(obs))
      failed |= dones.bool()
  pos = robot.data.root_link_pos_w[:, :2]
  err_p = (target_xy - pos).norm(dim=-1)
  err_y = wrap_to_pi(target_yaw - _yaw(robot.data.root_link_quat_w)).abs()
  ok = ~failed
  hit = ok & (err_p < 0.05) & (err_y < math.radians(5))
  out = {
    "no_fall_frac": ok.float().mean().item(),
    "success_frac_5cm_5deg": hit.float().mean().item(),
    "pos_err_median_m": err_p[ok].median().item() if ok.any() else float("nan"),
    "pos_err_max_m": err_p[ok].max().item() if ok.any() else float("nan"),
    "yaw_err_median_deg": math.degrees(err_y[ok].median().item())
    if ok.any()
    else float("nan"),
    "yaw_err_max_deg": math.degrees(err_y[ok].max().item())
    if ok.any()
    else float("nan"),
  }
  env.close()
  return out


def main() -> None:
  global TASK_ID
  p = argparse.ArgumentParser()
  p.add_argument("--checkpoint", required=True)
  p.add_argument("--task", default=TASK_ID)
  p.add_argument("--mode", choices=("stand", "walk_to", "both"), default="both")
  p.add_argument("--num-envs", type=int, default=64)
  p.add_argument("--duration-s", type=float, default=20.0)
  p.add_argument("--walk-radius", type=float, default=2.0)
  p.add_argument("--k-pos", type=float, default=1.0)
  p.add_argument("--k-yaw", type=float, default=1.0)
  p.add_argument("--v-max", type=float, default=0.5)
  p.add_argument("--v-min", type=float, default=0.0)
  p.add_argument("--w-min", type=float, default=0.0)
  p.add_argument("--tol-pos", type=float, default=0.03)
  p.add_argument("--tol-yaw", type=float, default=math.radians(3))
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--out", default=None, help="write the results as JSON here")
  args = p.parse_args()
  TASK_ID = args.task
  res = {"checkpoint": args.checkpoint, "args": vars(args)}
  if args.mode in ("stand", "both"):
    res["stand"] = run_stand(args)
    print("stand  ", json.dumps(res["stand"]))
  if args.mode in ("walk_to", "both"):
    res["walk_to"] = run_walk_to(args)
    print("walk_to", json.dumps(res["walk_to"]))
  if args.out:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
  main()
