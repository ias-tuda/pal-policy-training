"""Render plan S11's SEQ chain (L-T on the left, R-ST(-W) on the right through the mirror map) to MP4, headless.

The env and the loop are `eval_chain.chain`'s: the two-arm env (`ARMS=both` and BS11's selectors from the
environment, `TABLE_DZ` included), play mode, observation noise on, the left handed off on its own trigger,
the right started once the left's hand-off reaches its home leg and handed off on the same trigger or at
the right time-out. One env; each frame is rendered offscreen from a world camera that sees both arms, T and
both cubes (plus a side camera), captioned with the clock, the phases and the episode's events. Renders are
not bit-reproducible, so each episode goes to a temporary file kept only when its outcome (`eval_bimanual`'s
per-episode row) matches ``--keep``.

    REWARD=S11-BI ARMS=both HANDOFF_TRAIN=left START=scratch ... HANDOFF_ROUTE=up-out TABLE_DZ=22 MUJOCO_GL=egl \\
      python scripts/render_chain.py --lt model_9500.pt --rst model_10500.pt --out chain.mp4 --code "S11.3" --keep success
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

KEEP = {
  "any": "True",
  "success": "r['success']",
  "fail": "not r['success']",
  # A placed, B grasped, B never placed on A
  "b_miss": "r['a_placed_ever'] and r['b_grasped'] and not r['b_placed_ever']",
}


def parse_args(argv=None) -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--lt", type=pathlib.Path, required=True, help="L-T checkpoint (left)")
  p.add_argument("--rst", type=pathlib.Path, required=True, help="R-ST(-W) checkpoint (run mirrored)")
  p.add_argument("--out", type=pathlib.Path, required=True, help=".mp4 path")
  p.add_argument("--task", default="Mjlab-PickPlace-Pal-Kangaroo-Grippers")
  p.add_argument("--code", default="", help="caption: the pair, e.g. 'S11.3 L-T-H22 s2 + R-ST-W-H22 s2'")
  p.add_argument("--keep", default="success", help=f"a name ({', '.join(KEEP)}) or an expression in r")
  p.add_argument("--episodes", type=int, default=1, help="episodes to keep")
  p.add_argument("--max-tries", type=int, default=20)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--episode-s", type=float, default=30.0)
  p.add_argument("--right-timeout", type=float, default=24.0)
  p.add_argument("--obs-noise", choices=("off", "on"), default="on")
  p.add_argument("--width", type=int, default=960, help="per camera")
  p.add_argument("--height", type=int, default=600)
  p.add_argument("--lookat", type=float, nargs=3, default=(0.30, 0.0, 0.86))
  p.add_argument("--distance", type=float, default=0.92)
  p.add_argument("--azimuth", type=float, default=180.0)
  p.add_argument("--elevation", type=float, default=-40.0)
  p.add_argument(
    "--side", type=float, nargs=6, default=(0.45, 0.0, 0.86, 0.85, 90.0, -15.0),
    help="second camera: lookat x y z, distance, azimuth, elevation",
  )  # fmt: skip
  p.add_argument("--no-side", action="store_true")
  p.add_argument("--crf", type=int, default=23)
  p.add_argument("--max-steps", type=int, default=None, help="framing checks: stop early")
  p.add_argument("--still", type=pathlib.Path, default=None, help="framing: save frame 0 as PNG")
  return p.parse_args(argv)


def build_env(args):
  import pal_mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg(args.task, play=True)
  cfg.episode_length_s = args.episode_s
  cfg.scene.num_envs = 1
  cfg.seed = args.seed
  if args.obs_noise == "on":
    cfg.observations["actor"].enable_corruption = True
  if "cube_b" not in cfg.scene.entities:
    raise SystemExit("render_chain needs plan S11's two-arm env: ARMS=both and its selectors")
  cfg.viewer = dataclasses.replace(
    cfg.viewer,
    width=args.width,
    height=args.height,
    origin_type=type(cfg.viewer).OriginType.WORLD,
    lookat=tuple(args.lookat),
    distance=args.distance,
    azimuth=args.azimuth,
    elevation=args.elevation,
  )
  return ManagerBasedRlEnv(cfg=cfg, device=args.device, render_mode="rgb_array")


class Cameras:
  """The env's own (front) camera, plus an optional side camera on the same model; frames side by side."""

  def __init__(self, env, args) -> None:
    self.env = env
    self.side = None
    if not args.no_side:
      from mjlab.viewer.offscreen_renderer import OffscreenRenderer

      x, y, z, dist, az, el = args.side
      cfg = dataclasses.replace(
        env.cfg.viewer, lookat=(x, y, z), distance=dist, azimuth=az, elevation=el
      )
      self.side = OffscreenRenderer(
        model=env.sim.mj_model,
        cfg=cfg,
        scene=env.scene,
        sim_model=env.sim.model,
        expanded_fields=env.sim.expanded_fields,
      )
      self.side.initialize()
    self.width = args.width * (1 if self.side is None else 2)
    self.height = args.height

  def render(self) -> np.ndarray:
    front = self.env.render()
    if self.side is None:
      return front[..., :3]
    self.side.update(self.env.sim.data, debug_vis_callback=getattr(self.env, "update_visualizers", None))
    return np.concatenate([front[..., :3], self.side.render()[..., :3]], axis=1)

  def close(self) -> None:
    if self.side is not None:
      self.side.close()


class Writer:
  """Raw RGB frames piped to ffmpeg: H.264, yuv420p, faststart."""

  def __init__(self, path: pathlib.Path, w: int, h: int, fps: int, crf: int) -> None:
    import imageio_ffmpeg

    self.path = path
    self.proc = subprocess.Popen(
      [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(path),
      ],
      stdin=subprocess.PIPE,
    )  # fmt: skip
    self.frames = 0

  def add(self, frame: np.ndarray) -> None:
    self.proc.stdin.write(np.ascontiguousarray(frame[..., :3]).tobytes())
    self.frames += 1

  def close(self) -> None:
    self.proc.stdin.close()
    if self.proc.wait() != 0:
      raise RuntimeError(f"ffmpeg failed on {self.path}")


class Events:
  """Episode events for the caption, from `Reads`' state (env 0)."""

  def __init__(self, reads, term, dt: float) -> None:
    self.reads, self.term, self.dt = reads, term, dt
    self.k = {s: term.arm_index(s) for s in ("left", "right")}
    self.reset()

  def reset(self) -> None:
    self.log: list[tuple[float, str]] = []
    self.seen: set[str] = set()

  def _once(self, key: str, cond: bool, t: float, text: str) -> None:
    if cond and key not in self.seen:
      self.seen.add(key)
      self.log.append((t, text))

  def update(self, t_steps: int) -> None:
    r, t, dt = self.reads, t_steps * self.dt, self.dt
    self._once("a", int(r.a_step[0]) >= 0, int(r.a_step[0]) * dt, "A placed at T (left)")
    self._once("lh", int(self.term.phase[0, self.k["left"]]) > 0, t, "left hand-off (up-out-home)")
    self._once("bg", bool(r.b_grasped[0]), t, "B grasped (right, mirrored R-ST-W)")
    self._once("br", bool(r.b_released[0]), t, "B released")
    self._once("bp", int(r.b_step[0]) >= 0, int(r.b_step[0]) * dt, "B placed on A")
    self._once("rh", bool(r.right_handed[0]), t, "right hand-off")
    self._once("home", int(r.home_step[0]) >= 0, int(r.home_step[0]) * dt, "both home")

  def status(self) -> str:
    from pal_mjlab.tasks.manipulation.mdp.handoff import HOME, OUT, POLICY, UP

    name = {POLICY: "policy", UP: "hand-off up", OUT: "hand-off out", HOME: "hand-off home"}
    ph = self.term.phase[0]
    return f"left: {name.get(int(ph[self.k['left']]), '?')}   right: {name.get(int(ph[self.k['right']]), '?')}"


def captioner(header: str, h: int):
  from PIL import Image, ImageDraw, ImageFont

  size = max(14, round(h / 32))
  font = ImageFont.load_default(size=size)
  step = round(size * 1.3)

  def draw(frame: np.ndarray, t: float, status: str, events: list[tuple[float, str]]) -> np.ndarray:
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    lines = [header, f"t = {t:5.2f} s    {status}"] + [f"{te:5.2f} s  {txt}" for te, txt in events]
    for i, line in enumerate(lines):
      d.text((12, 10 + step * i), line, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    return np.asarray(img)

  return draw


def main(argv=None) -> None:
  from eval_bimanual import Reads
  from eval_chain import MirrorAdapter, OneArmActor
  from pal_mjlab.tasks.manipulation.mdp.handoff import HOME

  args = parse_args(argv)
  keep_expr = KEEP.get(args.keep, args.keep)
  env = build_env(args)
  ad = MirrorAdapter(env)
  lt, rst = OneArmActor(args.lt, args.device), OneArmActor(args.rst, args.device)
  reads = Reads(env)
  term = env.action_manager.get_term("joint_pos")
  dt = env.step_dt
  fps = round(1.0 / dt)
  rt_steps = round(args.right_timeout / dt)
  events = Events(reads, term, dt)
  table = os.environ.get("TABLE_DZ")
  table = f"table +{table} mm" if table else "table 0.815 m"
  header = (
    f"{args.code}  |  L-T {args.lt.stem} + R-ST-W {args.rst.stem} (mirrored)  |  SEQ chain, MuJoCo, {table}, "
    f"noise {args.obs_noise}"
  )
  cams = Cameras(env, args)
  draw = captioner(header, args.height)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  obs, _ = env.reset()

  kept: list[dict] = []
  tries = steps = 0
  t0 = time.time()
  with torch.inference_mode():
    while len(kept) < args.episodes and tries < args.max_tries:
      tmp = args.out.with_suffix(f".try{tries}.mp4")
      writer = Writer(tmp, cams.width, cams.height, fps, args.crf)
      events.reset()
      frame = draw(cams.render(), int(env.episode_length_buf[0]) * dt, events.status(), events.log)
      if args.still is not None and tries == 0:
        from PIL import Image

        Image.fromarray(frame).save(args.still)
      writer.add(frame)
      ep_steps = 0
      while True:
        start = reads.step(rt_steps)
        events.update(int(env.episode_length_buf[0]))
        t = ad.terms(obs, "actor")
        a_l = lt(ad.left(t))
        go = (term.phase[:, term.arm_index("left")] >= HOME).unsqueeze(-1)
        a_r = torch.where(go, rst(ad.right(t)), torch.zeros_like(a_l))
        term.start_handoff("right", start & go.squeeze(-1))
        obs, _, term_done, trunc, _ = env.step(ad.actions(a_l, a_r))
        steps += 1
        ep_steps += 1
        if bool((term_done | trunc)[0]):
          tm = env.termination_manager
          ended_by = ",".join(n for n in tm.active_terms if bool(tm.get_term(n)[0])) or "?"
          reads.finish(torch.tensor([0], device=env.device), dt)
          break
        writer.add(draw(cams.render(), int(env.episode_length_buf[0]) * dt, events.status(), events.log))
        if args.max_steps is not None and ep_steps >= args.max_steps:
          ended_by = "max-steps"
          reads.finish(torch.tensor([0], device=env.device), dt)
          break
      writer.close()
      row = reads.done[-1]
      row.update(ended_by=ended_by, frames=writer.frames, events=events.log)
      ok = bool(eval(keep_expr, {}, {"r": row}))  # noqa: S307
      brief = {k: row[k] for k in ("success", "tower_end", "a_placed_ever", "b_grasped", "b_placed_ever",
                                   "t_a_placed_s", "t_b_placed_s", "t_both_home_s", "a_moved_by_right_m",
                                   "hh_contact_steps")}  # fmt: skip
      print(f"try {tries}: keep={ok} ended_by={ended_by} frames={writer.frames} {json.dumps(brief)}", flush=True)
      if ok:
        name = args.out if args.episodes == 1 else args.out.with_name(f"{args.out.stem}_e{len(kept)}{args.out.suffix}")
        os.replace(tmp, name)
        kept.append({"file": str(name), "try": tries, **row})
      else:
        tmp.unlink()
      tries += 1
      if args.max_steps is not None:
        break
  cams.close()
  env.close()
  summary = {
    "lt": str(args.lt),
    "rst": str(args.rst),
    "table_dz": os.environ.get("TABLE_DZ"),
    "seed": args.seed,
    "keep": keep_expr,
    "tries": tries,
    "kept": kept,
    "all_rows": reads.done,
    "wall_s": round(time.time() - t0, 1),
  }
  args.out.with_suffix(".json").write_text(json.dumps(summary, indent=1, default=str))
  print(f"kept {len(kept)}/{args.episodes} after {tries} tries, {steps} steps, {summary['wall_s']} s")
  if len(kept) < args.episodes:
    sys.exit(2)


if __name__ == "__main__":
  main()
