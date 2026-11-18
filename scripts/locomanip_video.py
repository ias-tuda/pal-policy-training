"""Demo video of the loco-manipulation lower-body policy: walk_to, then stand while the arms move.

One env of the play config. Phase 1 drives the twist command with the walk_to controller
of `locomanip_eval.py` (speed floor v_min / w_min) through a list of target poses, arms at
the home pose. Phase 2 commands zero twist and lets the scripted arms move (curriculum
level 1, jaws toggling). A caption with the phase and the remaining error is drawn on
each frame.

Usage:
  uv run python scripts/locomanip_video.py --task <task> --checkpoint <model.pt> --out <mp4>
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import mediapy
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import locomanip_eval as ev  # noqa: E402
from mjlab.utils.lab_api.math import wrap_to_pi  # noqa: E402

# Target poses relative to the start: (x m, y m, yaw deg).
WAYPOINTS = [(1.0, 0.0, 0.0), (1.0, 0.8, 90.0), (0.0, 0.8, 180.0)]


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--task", required=True)
  p.add_argument("--checkpoint", required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--v-min", type=float, default=0.2)
  p.add_argument("--w-min", type=float, default=0.3)
  p.add_argument("--v-max", type=float, default=0.5)
  p.add_argument("--leg-timeout-s", type=float, default=12.0)
  p.add_argument("--manip-s", type=float, default=10.0)
  p.add_argument("--device", default="cuda:0")
  args = p.parse_args()
  ev.TASK_ID = args.task

  cfg = ev.load_env_cfg(args.task, play=True)
  cfg.scene.num_envs = 1
  cfg.episode_length_s = 1e6
  twist = cfg.commands["twist"]
  twist.resampling_time_range = (1e9, 1e9)
  twist.rel_standing_envs = twist.rel_heading_envs = 0.0
  twist.heading_command = False
  twist.ranges.heading = None
  twist.rel_forward_envs = 0.0
  cfg.events.pop("push_robot", None)
  cfg.actions["arms"].level_ramp_steps = 0
  cfg.actions["arms"].hold_probability = 1.0
  cfg.viewer.width, cfg.viewer.height = 960, 720
  cfg.viewer.distance, cfg.viewer.elevation, cfg.viewer.azimuth = 3.0, -18.0, 135.0
  raw = ev.ManagerBasedRlEnv(cfg=cfg, device=args.device, render_mode="rgb_array")
  env = ev.RslRlVecEnvWrapper(raw, clip_actions=ev.load_rl_cfg(args.task).clip_actions)
  policy = ev._policy(env, args.checkpoint, args.device)
  robot = raw.scene["robot"]
  cmd = raw.command_manager.get_term("twist")
  arms = raw.action_manager.get_term("arms")

  obs = env.get_observations()
  every = max(1, round(1 / 30 / raw.step_dt))
  frames, step, falls = [], 0, 0

  def advance(v, caption):
    nonlocal obs, step, falls
    cmd.vel_command_b[:] = torch.tensor(v, device=args.device)
    with torch.no_grad():
      obs, _, dones, _ = env.step(policy(obs))
    falls += int(dones.any())
    if step % every == 0:
      img = Image.fromarray(raw.render())
      d = ImageDraw.Draw(img)
      d.rectangle((0, 0, img.width, 44), fill=(0, 0, 0))
      d.text((14, 10), caption, fill=(255, 255, 255), font_size=24)
      frames.append(np.asarray(img))
    step += 1

  for _ in range(int(1.5 / raw.step_dt)):
    advance((0, 0, 0), "stand")
  start = robot.data.root_link_pos_w[0, :2].clone()
  yaw0 = ev._yaw(robot.data.root_link_quat_w)[0]
  c0, s0 = yaw0.cos(), yaw0.sin()
  results = []
  for i, (x, y, yaw_deg) in enumerate(WAYPOINTS, 1):
    tgt = start + torch.stack((c0 * x - s0 * y, s0 * x + c0 * y))
    tgt_yaw = wrap_to_pi(yaw0 + math.radians(yaw_deg))
    rest = 0
    for _ in range(int(args.leg_timeout_s / raw.step_dt)):
      pos = robot.data.root_link_pos_w[0, :2]
      yaw = ev._yaw(robot.data.root_link_quat_w)[0]
      e_w = tgt - pos
      c, s = yaw.cos(), yaw.sin()
      e_b = torch.stack((c * e_w[0] + s * e_w[1], -s * e_w[0] + c * e_w[1]))
      e_yaw = wrap_to_pi(tgt_yaw - yaw)
      d = e_b.norm().clamp(min=1e-6)
      v = torch.zeros(3, device=args.device)
      if d >= 0.03:
        v[:2] = e_b / d * d.clamp(args.v_min, args.v_max)
      if e_yaw.abs() >= math.radians(3):
        v[2] = torch.sign(e_yaw) * e_yaw.abs().clamp(
          args.w_min, cmd.cfg.ranges.ang_vel_z[1]
        )
      rest = rest + 1 if not v.any() else 0
      cap = (
        f"walk_to #{i}  target ({x:+.1f}, {y:+.1f} m, {yaw_deg:+.0f} deg)   "
        f"error {100 * d.item():.0f} cm, {math.degrees(e_yaw.abs().item()):.0f} deg"
      )
      advance(v.tolist(), cap)
      if rest * raw.step_dt > 1.0:
        break
    results.append((100 * d.item(), math.degrees(e_yaw.abs().item())))
    print(f"waypoint {i}: {results[-1][0]:.1f} cm, {results[-1][1]:.1f} deg")

  arms.cfg.hold_probability = 0.2
  for _ in range(int(args.manip_s / raw.step_dt)):
    drift = 100 * (robot.data.root_link_pos_w[0, :2] - tgt).norm().item()
    advance(
      (0, 0, 0),
      f"stand, arms + grippers moving (0-1 kg in each hand)   drift {drift:.0f} cm",
    )

  mediapy.write_video(args.out, frames, fps=round(1 / (raw.step_dt * every)))
  print(
    f"wrote {args.out}: {len(frames)} frames, {step * raw.step_dt:.0f} s, terminations={falls}"
  )
  env.close()


if __name__ == "__main__":
  main()
