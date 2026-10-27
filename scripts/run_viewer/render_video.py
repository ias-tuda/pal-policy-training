"""Render headless rollout videos for one or more checkpoints of a training run.

The run viewer copies this file to the cluster and runs it inside a SLURM job, but it
works anywhere the project environment is installed:

  MUJOCO_GL=egl uv run python scripts/run_viewer/render_video.py \
    Mjlab-MugUpright-Standing-Pal-Kangaroo-Grippers \
    logs/rsl_rl/kangaroo_mug_upright/<run> --checkpoints 1400 7999

Each checkpoint becomes <run>/videos/viewer/model_<iter>.mp4. All checkpoints share
one environment and start from the same seed, so their videos are comparable.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict
from pathlib import Path


def render_run(
  task_id: str,
  run_dir: Path,
  iterations: list[int],
  *,
  episodes: int,
  seed: int,
  device: str,
  width: int,
  height: int,
) -> None:
  import mediapy as media
  import numpy as np
  import torch

  try:
    import imageio_ffmpeg

    media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
  except ImportError:
    pass

  import mjlab.tasks  # noqa: F401  (populates the task registry, incl. pal_mjlab)
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = 1
  # Play variants disable the timeout (episode_length_s=1e9); restore the training one
  # so a video holds whole episodes and the env resets between them.
  env_cfg.episode_length_s = load_env_cfg(task_id, play=False).episode_length_s
  env_cfg.viewer.width, env_cfg.viewer.height = width, height
  agent_cfg = load_rl_cfg(task_id)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device=device)

  out_dir = run_dir / "videos" / "viewer"
  out_dir.mkdir(parents=True, exist_ok=True)
  steps = episodes * env.max_episode_length
  fps = 1.0 / env.step_dt
  print(f"[render] {steps} steps per video at {fps:.0f} fps", flush=True)
  failures = []
  for iteration in iterations:
    checkpoint = run_dir / f"model_{iteration}.pt"
    output = out_dir / f"model_{iteration}.mp4"
    started = time.time()
    try:
      runner.load(
        str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
      )
      policy = runner.get_inference_policy(device=device)
      env.seed(seed)
      torch.manual_seed(seed)
      wrapped.reset()
      obs = wrapped.get_observations()
      frames = []
      with torch.no_grad():
        for _ in range(steps):
          obs, _, _, _ = wrapped.step(policy(obs))
          frame = env.render()
          if frame is None:
            continue
          frame = np.asarray(frame)
          if frame.ndim == 4:
            frame = frame[0]
          if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
          frames.append(frame)
      # Write under a temporary name so the viewer never picks up a partial file.
      partial = output.with_name(output.stem + ".partial.mp4")
      media.write_video(str(partial), frames, fps=fps)
      os.replace(partial, output)
      print(
        f"[render] {output} ({len(frames)} frames, {time.time() - started:.0f}s)",
        flush=True,
      )
    except Exception as exc:  # keep rendering the remaining checkpoints
      failures.append(iteration)
      print(f"[render] FAILED model_{iteration}: {exc!r}", flush=True)
  env.close()
  if failures:
    raise SystemExit(f"[render] failed checkpoints: {failures}")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("task", help="mjlab task id, e.g. Mjlab-Stack-Pal-Kangaroo-Grippers")
  parser.add_argument("run_dir", type=Path, help="logs/rsl_rl/<experiment>/<run>")
  parser.add_argument("--checkpoints", type=int, nargs="+", required=True)
  parser.add_argument("--episodes", type=int, default=1)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--device", default=None)
  parser.add_argument("--width", type=int, default=720)
  parser.add_argument("--height", type=int, default=540)
  args = parser.parse_args()

  import torch

  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  render_run(
    args.task,
    args.run_dir.expanduser().resolve(),
    args.checkpoints,
    episodes=args.episodes,
    seed=args.seed,
    device=device,
    width=args.width,
    height=args.height,
  )


if __name__ == "__main__":
  main()
