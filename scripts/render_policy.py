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

`--spawn`, `--target` and `--table-offset` pin the first episode's start instead of leaving it
to the reset draws: the cube's start pose, the goal, and the table's offset from its nominal
place (DR-scene's startup draw), each in the env frame for the table at its nominal place. The
moves are `scripted_pick_place.rollout`'s (plan S7-2's corners check). `--caption` writes a line
and the episode clock into the frames.

    REWARD=R-C-strict+M1+M2+M3+M6 DR=DR-scene PPO=PPO-stack SLEW=0.5 BASELINE=B SPAWN=W2 \\
      uv run --extra cpu python scripts/render_policy.py --checkpoint .../model_4999.pt \\
      --spawn 0.35 0.21 35 --target 0.46 0.08 --table-offset 0 0 0 --out w2_near.mp4
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import math
import pathlib

import imageio.v2 as imageio
import numpy as np
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
  p.add_argument(
    "--spawn",
    type=float,
    nargs=3,
    metavar=("X", "Y", "YAW_DEG"),
    help="the cube's start: env frame, table at its nominal place; it moves with the table",
  )
  p.add_argument(
    "--target",
    type=float,
    nargs=2,
    metavar=("X", "Y"),
    help="the goal: env frame, table at its nominal place (a SurfaceOffsetCommand goal)",
  )
  p.add_argument(
    "--table-offset",
    type=float,
    nargs=3,
    metavar=("DX", "DY", "DZ"),
    help="the table's offset from its nominal place, m",
  )
  p.add_argument(
    "--settle-steps",
    type=int,
    default=3,
    help="idle steps after pinning: the cube settles and the 0-2 step object lag flushes",
  )
  p.add_argument("--caption", default=None, help="a line drawn into every frame")
  return p.parse_args()


def pin_start(env, ev, args: argparse.Namespace) -> dict[str, tuple[float, ...]]:
  """Reset, put env 0's table, goal and cube where asked, and return where they ended up.

  As `scripted_pick_place.rollout`: the table by its own body's local offset (DR-scene's
  `dr.body_pos` draws it there at startup; the entity's first body is the mocap anchor it
  hangs from), the goal by `SurfaceOffsetCommand.set_offsets`, the cube by a root-pose write
  that keeps the reset event's height on the table. Then ``--settle-steps`` idle steps
  (action 0 holds the start pose) let the cube settle and the observation delay flush, so
  the policy's first observation shows the pinned scene. Positions are in the env frame.
  """
  from mjlab.managers.scene_entity_config import SceneEntityCfg
  from pal_mjlab.tasks.manipulation import scene
  from pal_mjlab.tasks.manipulation.mdp.events import _surface_centre_xy, surface_top_z
  from pal_mjlab.tasks.manipulation.pick_place import pick_place_env_cfg as pp

  env.reset()
  origin = env.scene.env_origins[0]
  table = env.scene[pp.TABLE]
  table_body = table.indexing.body_ids[table.body_names.index(pp.TABLE)]
  if args.table_offset is not None:
    local0 = torch.tensor(
      env.sim.mj_model.body_pos[int(table_body)], dtype=torch.float32, device=env.device
    )
    env.sim.model.body_pos[0, table_body] = local0 + torch.tensor(
      args.table_offset, dtype=torch.float32, device=env.device
    )
    # The reset event and the goal read the table's pose out of `data`; a forward pass
    # brings it up to date with `model`.
    env.sim.forward()
  # The table's offset from its nominal place, as the reset event and the goal see it. The
  # entity's origin sits half a slab below the top face (`scene.table_pos`).
  nominal = np.asarray(env.cfg.scene.entities[pp.TABLE].init_state.pos)
  table_xy0 = nominal[:2]
  surface = SceneEntityCfg(pp.TABLE)
  origin_np = origin.cpu().numpy()
  shift = np.zeros(3)
  shift[:2] = (
    _surface_centre_xy(env, surface)[0].cpu().numpy() - origin_np[:2] - table_xy0
  )
  shift[2] = (
    float(surface_top_z(env, surface)[0])
    - origin_np[2]
    - (nominal[2] + scene.TABLE_THICKNESS / 2)
  )
  if args.target is not None:
    command = env.command_manager.get_term(ev.COMMAND)
    if not hasattr(command, "set_offsets"):
      raise ValueError(
        "--target needs a SurfaceOffsetCommand goal (target box or DR-scene)"
      )
    command.set_offsets(
      torch.tensor([0], device=env.device),
      torch.tensor([np.asarray(args.target) - table_xy0], dtype=torch.float32),
    )
  obj = env.scene[ev.OBJECT]
  if args.spawn is not None:
    x, y, yaw_deg = args.spawn
    yaw = math.radians(yaw_deg)
    pose = obj.data.root_link_pose_w.clone()
    pose[0, 0] = origin[0] + x + float(shift[0])
    pose[0, 1] = origin[1] + y + float(shift[1])
    pose[0, 3:7] = torch.tensor(
      [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], device=env.device
    )
    obj.write_root_link_pose_to_sim(pose)
    obj.write_root_link_velocity_to_sim(torch.zeros_like(obj.data.root_link_vel_w))
  with torch.inference_mode():
    for _ in range(args.settle_steps):
      env.step(torch.zeros(env.action_space.shape, device=env.device))
  pos = (obj.data.root_link_pos_w[0] - origin).cpu().numpy()
  w, qx, qy, qz = obj.data.root_link_quat_w[0].cpu().numpy()
  goal = (env.command_manager.get_term(ev.COMMAND).command[0] - origin).cpu().numpy()
  return {
    "cube": (
      float(pos[0] - shift[0]),
      float(pos[1] - shift[1]),
      math.degrees(math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))),
    ),
    "goal": (float(goal[0] - shift[0]), float(goal[1] - shift[1])),
    "table_offset": tuple(float(v) for v in shift),
  }


def _captioner(text: str | None, step_dt: float):
  """Frame -> frame with ``text`` and the episode clock drawn in the top-left corner."""
  if text is None:
    return lambda frame, step: frame
  from PIL import Image, ImageDraw, ImageFont

  font = ImageFont.load_default(size=16)

  def draw(frame: np.ndarray, step: int) -> np.ndarray:
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    for i, line in enumerate((text, f"t = {step * step_dt:4.2f} s")):
      d.text(
        (10, 8 + 20 * i),
        line,
        fill=(255, 255, 255),
        font=font,
        stroke_width=2,
        stroke_fill=(0, 0, 0),
      )
    return np.asarray(img)

  return draw


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

  pinned = None
  if args.spawn is not None or args.target is not None or args.table_offset is not None:
    if args.episodes != 1:
      raise ValueError(
        "--spawn, --target and --table-offset pin one episode: --episodes 1"
      )
    pinned = pin_start(env, ev, args)
    print(
      "pinned start (env frame, table at its nominal place): cube ({:.3f}, {:.3f}) "
      "yaw {:+.1f} deg, goal ({:.3f}, {:.3f}), table offset ({:+.3f}, {:+.3f}, {:+.3f}) m".format(
        *pinned["cube"], *pinned["goal"], *pinned["table_offset"]
      )
    )
  caption = _captioner(args.caption, env.step_dt)
  gripper_cfg = env.cfg.rewards.get("grasp") or env.cfg.rewards["lift"]
  grasp_sensor = (
    gripper_cfg.params.get("sensor_name") or gripper_cfg.params["grasp_sensor_name"]
  )

  args.out.parent.mkdir(parents=True, exist_ok=True)
  writer = imageio.get_writer(str(args.out), fps=fps, codec="libx264", quality=7)
  obs = wrapped.get_observations()
  done_episodes = steps = 0
  ep_step = args.settle_steps if pinned else 0
  ended_by: list[str] = []
  first_placed: list[float | None] = [None]
  with torch.inference_mode():
    writer.append_data(caption(env.render(), ep_step))
    while done_episodes < args.episodes and (
      args.max_steps is None or steps < args.max_steps
    ):
      obs, _, dones, _ = wrapped.step(policy(obs))
      steps += 1
      ep_step += 1
      if bool(dones[0]):
        tm = env.termination_manager
        ended_by.append(
          ",".join(n for n in tm.active_terms if bool(tm.get_term(n)[0])) or "?"
        )
        done_episodes += 1
        # The env has reset: its frame is the next episode's start, not this one's end.
        if done_episodes >= args.episodes:
          break
        ep_step = 0
        first_placed.append(None)
      elif first_placed[-1] is None and bool(
        ev.probe(env, grasp_sensor)["placed_and_lifted"][0]
      ):
        first_placed[-1] = ep_step * env.step_dt
      writer.append_data(caption(env.render(), ep_step))
  writer.close()
  env.close()
  placed = ", ".join("never" if t is None else f"{t:.2f} s" for t in first_placed)
  print(
    f"wrote {args.out}: {steps} steps ({steps / fps:.1f} s at {fps} fps), "
    f"episodes ended by: {ended_by}; first placed and lifted at: {placed}"
  )


if __name__ == "__main__":
  main()
