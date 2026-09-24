"""Record a trained pick-and-place policy to an MP4, headless.

`uv run play --video` records too, but then opens an interactive viewer; this writes the file
and exits, so it works in a batch or over SSH. Same env as `eval_pick_place.py` -- the play cfg
(no observation corruption, no curriculum) with the *training* episode length -- and the same
checkpoint loading. One env, the first `--episodes` episodes. The camera is fixed in the world and aimed at the
work area -- between the cube's spawn point and the target, from in front of the robot --
unless `--task-view` keeps the task's own torso-tracking camera, which sees the table from
behind the robot and cuts the cube off.

The reward arm, DR rung and PPO recipe come from the environment, as for training (D10): a
checkpoint only loads under the PPO recipe it was trained with.

    REWARD=R-A DR=DR0 PPO=PPO-stack uv run --extra cpu python scripts/render_policy.py \\
      --checkpoint logs/.../model_12000.pt --out artifacts/videos/r-a_s1_12k.mp4
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import pathlib

import imageio.v2 as imageio
import torch


def _eval_module():
  spec = importlib.util.spec_from_file_location(
    "eval_pick_place", pathlib.Path(__file__).parent / "eval_pick_place.py"
  )
  assert spec and spec.loader
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--checkpoint", type=pathlib.Path, required=True)
  p.add_argument("--out", type=pathlib.Path, required=True, help=".mp4 path")
  p.add_argument("--task", default="Mjlab-PickPlace-Pal-Kangaroo-Grippers")
  p.add_argument("--episodes", type=int, default=1)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cpu")
  p.add_argument("--width", type=int, default=640)
  p.add_argument("--height", type=int, default=480)
  p.add_argument("--task-view", action="store_true", help="the task's tracking camera")
  p.add_argument("--lookat", type=float, nargs=3, default=(0.42, 0.17, 0.82))
  p.add_argument("--distance", type=float, default=1.1, help="camera distance, m")
  p.add_argument("--azimuth", type=float, default=200.0, help="deg")
  p.add_argument("--elevation", type=float, default=-28.0, help="deg")
  p.add_argument(
    "--max-steps", type=int, default=None, help="stop early (framing checks)"
  )
  return p.parse_args()


def main() -> None:
  args = parse_args()
  import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  ev = _eval_module()
  cfg = load_env_cfg(args.task, play=True)
  cfg.episode_length_s = load_env_cfg(args.task, play=False).episode_length_s
  cfg.scene.num_envs = 1
  cfg.seed = args.seed
  view = {"width": args.width, "height": args.height}
  if not args.task_view:
    view.update(
      origin_type=type(cfg.viewer).OriginType.WORLD,
      lookat=tuple(args.lookat),
      distance=args.distance,
      azimuth=args.azimuth,
      elevation=args.elevation,
    )
  cfg.viewer = dataclasses.replace(cfg.viewer, **view)
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device, render_mode="rgb_array")

  eval_args = argparse.Namespace(
    task=args.task, checkpoint=args.checkpoint, device=args.device
  )
  wrapped, policy = ev.load_policy(eval_args, env)
  fps = round(1.0 / env.step_dt)

  args.out.parent.mkdir(parents=True, exist_ok=True)
  writer = imageio.get_writer(str(args.out), fps=fps, codec="libx264", quality=7)
  obs = wrapped.get_observations()
  done_episodes = steps = 0
  ended_by: list[str] = []
  with torch.inference_mode():
    writer.append_data(env.render())
    while done_episodes < args.episodes and (
      args.max_steps is None or steps < args.max_steps
    ):
      obs, _, dones, _ = wrapped.step(policy(obs))
      writer.append_data(env.render())
      steps += 1
      if bool(dones[0]):
        tm = env.termination_manager
        ended_by.append(
          ",".join(n for n in tm.active_terms if bool(tm.get_term(n)[0])) or "?"
        )
        done_episodes += 1
  writer.close()
  env.close()
  print(
    f"wrote {args.out}: {steps} steps ({steps / fps:.1f} s at {fps} fps), "
    f"episodes ended by: {ended_by}"
  )


if __name__ == "__main__":
  main()
