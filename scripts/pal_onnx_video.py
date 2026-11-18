"""Roll out a PAL-shipped ONNX walking policy in sim and record a video.

The policies come from github.com/pal-robotics/pal_policy_deployer (models.yaml,
stable/lower_body_loco). They run in `Mjlab-Velocity-Flat-Pal-Kangaroo-Lower-Body`, whose
actor observation (62 values) and joint/action order match the ONNX metadata (checked at
start). The twist command follows a fixed script, written each step.

Usage:
  uv run --with onnxruntime python scripts/pal_onnx_video.py --onnx <file> --out <mp4> \
      [--history 3] [--script omni|forward]
"""

from __future__ import annotations

import argparse

import mediapy
import numpy as np
import onnxruntime as ort
import pal_mjlab.tasks  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

TASK = "Mjlab-Velocity-Flat-Pal-Kangaroo-Lower-Body"

# (duration s, vx, vy, wz, label)
SCRIPTS = {
  "omni": [
    (2, 0.0, 0.0, 0.0, "stand"),
    (4, 0.5, 0.0, 0.0, "forward 0.5 m/s"),
    (3, -0.3, 0.0, 0.0, "backward 0.3 m/s"),
    (3, 0.0, 0.3, 0.0, "left 0.3 m/s"),
    (3, 0.0, -0.3, 0.0, "right 0.3 m/s"),
    (3, 0.0, 0.0, 0.5, "turn 0.5 rad/s"),
    (4, 0.4, 0.0, 0.3, "forward + turn"),
    (3, 0.1, 0.0, 0.0, "slow 0.1 m/s"),
    (2, 0.0, 0.0, 0.0, "stand"),
  ],
  "forward": [
    (2, 0.0, 0.0, 0.0, "stand"),
    (8, 0.25, 0.0, 0.0, "forward 0.25 m/s"),
    (4, 0.25, 0.0, 0.3, "forward + turn"),
    (3, 0.0, 0.0, 0.0, "stand"),
  ],
}


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--onnx", required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--history", type=int, default=0)
  p.add_argument("--script", choices=tuple(SCRIPTS), default="omni")
  p.add_argument("--device", default="cuda:0")
  args = p.parse_args()

  sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
  meta = sess.get_modelmeta().custom_metadata_map
  n_obs = sess.get_inputs()[0].shape[1]

  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = 1
  cfg.observations["actor"].history_length = args.history or None
  cfg.events.pop("push_robot", None)
  twist = cfg.commands["twist"]
  twist.resampling_time_range = (1e9, 1e9)
  twist.rel_standing_envs = 0.0
  twist.rel_heading_envs = 0.0
  twist.heading_command = False
  twist.ranges.heading = None
  cfg.viewer.width, cfg.viewer.height = 960, 720
  cfg.viewer.distance, cfg.viewer.elevation, cfg.viewer.azimuth = 2.6, -15.0, 135.0
  cfg.viewer.lookat = (0.0, 0.0, -0.35)
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device, render_mode="rgb_array")

  robot = env.scene["robot"]
  assert ",".join(robot.joint_names).startswith(meta["joint_names"][:200])
  assert env.observation_manager.group_obs_dim["actor"][0] == n_obs, (
    env.observation_manager.group_obs_dim["actor"],
    n_obs,
  )
  cmd = env.command_manager.get_term("twist")
  obs, _ = env.reset()
  frames, falls = [], 0
  every = max(1, round(1 / 30 / env.step_dt))  # ~30 fps
  step = 0
  for dur, vx, vy, wz, _label in SCRIPTS[args.script]:
    for _ in range(int(dur / env.step_dt)):
      cmd.vel_command_b[:] = torch.tensor([vx, vy, wz], device=args.device)
      o = obs["actor"].cpu().numpy().astype(np.float32)
      a = sess.run(None, {sess.get_inputs()[0].name: o})[0]
      obs, _, term, trunc, _ = env.step(torch.from_numpy(a).to(args.device))
      falls += int(term.any())
      if step % every == 0:
        frames.append(env.render())
      step += 1
  mediapy.write_video(args.out, frames, fps=round(1 / (env.step_dt * every)))
  print(f"wrote {args.out}: {len(frames)} frames, terminations={falls}")
  env.close()


if __name__ == "__main__":
  main()
