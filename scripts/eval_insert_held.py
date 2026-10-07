"""Plan §13.1's R-I read (`pick-place-insert-plan.md` §13 *Reads and bars*): does the right (trained on the left arm)
push the pin through the block held at H by the stand-in hand, and keep it there?

One episode per row (``--episode-s``, 20 s by default), the actor's tracked observations (``--obs-noise on``), the
policy deterministic, ``S13=R-I`` / ``R-I-W`` on ``GOAL=insert`` with `BS12`'s selectors as trained (``S13=R-I`` reads
an R-I-W policy at the nominal hold):

* **through while held** -- the pin's bottom at or below the block's bottom face inside the bore
  (`mdp.InsertState.through`) with the jaw on the pin, on >= 80 % of the last second (headline), and the first time;
* the pin grasped / lifted, its bottom's offset from the hole's axis when it first reaches the entry, its depth at
  the end, the block's largest displacement from its hold, the arm on the stand-in hand (any / > 30 N), arm 1-6 on
  the table, the pin dropped;
* bins by the pin's spawn x, y and yaw, and by the hold's offset from H.

    S13=R-I GOAL=insert SHAPE=pin TARGET=WC CRITIC_OBS=q ... python scripts/eval_insert_held.py --ckpt model_5999.pt --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from eval_hold import MlpActor  # noqa: E402

TASK = "Mjlab-PickPlace-Pal-Kangaroo-Grippers"


def parse_args(argv=None) -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--ckpt", type=pathlib.Path, required=True)
  p.add_argument("--task", default=TASK)
  p.add_argument("--episodes", type=int, default=64)
  p.add_argument("--num-envs", type=int, default=64)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--episode-s", type=float, default=20.0)
  p.add_argument("--obs-noise", choices=("off", "on"), default="on")
  p.add_argument("--json", type=pathlib.Path, default=None)
  return p.parse_args(argv)


def build_env(args):
  import pal_mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg(args.task, play=False)
  if "holder" not in cfg.scene.entities:
    raise SystemExit(
      "eval_insert_held reads plan §13.1's R-I env: set S13=R-I (GOAL=insert ...)"
    )
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  cfg.episode_length_s = args.episode_s
  if args.obs_noise == "off":
    cfg.observations["actor"].enable_corruption = False
  return ManagerBasedRlEnv(cfg=cfg, device=args.device)


class Reads:
  def __init__(self, env, episode_s: float) -> None:
    from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS_LEFT_ARM

    self.env, self.episode_s = env, episode_s
    self.gripper = KANGAROO_GRIPPERS_LEFT_ARM.gripper("left")
    p = env.cfg.rewards["depth"].params
    self.geo = {
      k: p[k] for k in ("object_name", "block_name", "block_height", "hole_d")
    }
    n, dev = env.num_envs, env.device
    self.dt = env.step_dt
    self.rows: list[dict] = []
    z = lambda: torch.zeros(n, device=dev)  # noqa: E731
    b = lambda: torch.zeros(n, dtype=torch.bool, device=dev)  # noqa: E731
    nan = lambda: torch.full((n,), float("nan"), device=dev)  # noqa: E731
    self.spawn = torch.zeros(n, 3, device=dev)
    self.hold_off = z()
    self.end_through, self.end_steps, self.moved = z(), z(), z()
    self.t_through, self.entry_offset, self.end_depth = nan(), nan(), nan()
    self.grasped, self.lifted, self.hh, self.hh30, self.armt, self.dropped = (
      b(),
      b(),
      b(),
      b(),
      b(),
      b(),
    )

  def start(self, ids: torch.Tensor) -> None:
    from pal_mjlab.tasks.manipulation.pick_place.hold import HOLD_XY

    env = self.env
    o = env.scene.env_origins[ids]
    pin = env.scene["object"]
    p = pin.data.root_link_pos_w[ids] - o
    w, x, y, zq = pin.data.root_link_quat_w[ids].unbind(-1)
    yaw = torch.atan2(2 * (w * zq + x * y), 1 - 2 * (y * y + zq * zq))
    yaw90 = torch.remainder(yaw + math.pi / 4, math.pi / 2) - math.pi / 4
    self.spawn[ids] = torch.stack([p[:, 0], p[:, 1], torch.rad2deg(yaw90)], dim=-1)
    hold = env.scene["holder"].data.root_link_pos_w[ids] - o
    self.hold_off[ids] = torch.norm(
      hold[:, :2] - torch.tensor(HOLD_XY, device=env.device), dim=-1
    )
    for t in (self.end_through, self.end_steps, self.moved):
      t[ids] = 0.0
    for t in (self.t_through, self.entry_offset, self.end_depth):
      t[ids] = float("nan")
    for t in (self.grasped, self.lifted, self.hh, self.hh30, self.armt, self.dropped):
      t[ids] = False

  def step(self) -> None:
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from pal_mjlab.tasks.manipulation import mdp
    from pal_mjlab.tasks.manipulation.mdp.events import surface_top_z
    from pal_mjlab.tasks.manipulation.mdp.observations import (
      _grasped_bool,
      _was_lifted_bool,
    )
    from pal_mjlab.tasks.manipulation.tabletop_env_cfg import SUPPORT_SENSOR

    env = self.env
    tt = (env.episode_length_buf.float() - 1.0) * self.dt
    st = mdp.insert_state(env, **self.geo)
    held = _grasped_bool(env, self.gripper.contact_sensor_name)
    self.grasped |= held
    self.lifted |= _was_lifted_bool(
      env, self.gripper.contact_sensor_name, SUPPORT_SENSOR
    )
    through = st.through() & held
    first = through & torch.isnan(self.t_through)
    self.t_through[first] = tt[first]
    entry = (st.depth > 0.0) & torch.isnan(self.entry_offset)
    self.entry_offset[entry] = st.offset[entry]
    end = tt >= self.episode_s - 1.0 - self.dt
    self.end_steps += end.float()
    self.end_through += (end & through).float()
    self.end_depth = torch.where(end, st.depth, self.end_depth)
    d, _ = mdp.held_block_offset(env, "holder", "block")
    self.moved = torch.maximum(self.moved, d)
    hh = env.scene["arm_holder_contact"].data
    self.hh |= hh.found.bool().reshape(env.num_envs, -1).any(-1)
    self.hh30 |= hh.force.norm(dim=-1).reshape(env.num_envs, -1).amax(-1) > 30.0
    if "arm_table_contact" in env.scene.sensors:
      self.armt |= (
        env.scene["arm_table_contact"]
        .data.found.bool()
        .reshape(env.num_envs, -1)
        .any(-1)
      )
    top = surface_top_z(env, SceneEntityCfg("table"))
    self.dropped |= env.scene["object"].data.root_link_pos_w[:, 2] < top - 0.05

  def finish(self, ids: torch.Tensor) -> None:
    for i in ids.tolist():
      steps = float(self.end_steps[i])
      self.rows.append(
        {
          "x": float(self.spawn[i, 0]),
          "y": float(self.spawn[i, 1]),
          "yaw_deg": float(self.spawn[i, 2]),
          "hold_offset_mm": 1e3 * float(self.hold_off[i]),
          "through_held": steps > 0 and float(self.end_through[i]) >= 0.8 * steps,
          "t_first_through": float(self.t_through[i]),
          "entry_offset_mm": 1e3 * float(self.entry_offset[i]),
          "end_depth_mm": 1e3 * float(self.end_depth[i]),
          "grasped": bool(self.grasped[i]),
          "lifted": bool(self.lifted[i]),
          "block_moved_mm": 1e3 * float(self.moved[i]),
          "holder_contact": bool(self.hh[i]),
          "holder_contact_30N": bool(self.hh30[i]),
          "arm_table": bool(self.armt[i]),
          "pin_dropped": bool(self.dropped[i]),
        }
      )


def summarise(rows: list[dict]) -> dict:
  import statistics as st

  def share(k, sel=None):
    sel = rows if sel is None else sel
    return None if not sel else sum(bool(r[k]) for r in sel) / len(sel)

  fin = lambda vals: [v for v in vals if v == v]  # noqa: E731
  med = lambda v: st.median(v) if v else float("nan")  # noqa: E731
  out = {
    "episodes": len(rows),
    "through_held": share("through_held"),
    "grasped": share("grasped"),
    "lifted": share("lifted"),
    "t_first_through_median_s": med(fin([r["t_first_through"] for r in rows])),
    "entry_offset_median_mm": med(fin([r["entry_offset_mm"] for r in rows])),
    "block_moved_median_mm": med([r["block_moved_mm"] for r in rows]),
    "holder_contact": share("holder_contact"),
    "holder_contact_30N": share("holder_contact_30N"),
    "arm_table": share("arm_table"),
    "pin_dropped": share("pin_dropped"),
  }
  bins = {
    "x<0.47": lambda r: r["x"] < 0.47,
    "0.47<=x<0.53": lambda r: 0.47 <= r["x"] < 0.53,
    "x>=0.53": lambda r: r["x"] >= 0.53,
    "y<0.16": lambda r: r["y"] < 0.16,
    "0.16<=y<0.21": lambda r: 0.16 <= r["y"] < 0.21,
    "y>=0.21": lambda r: r["y"] >= 0.21,
    "|yaw|<15": lambda r: abs(r["yaw_deg"]) < 15,
    "15<=|yaw|<30": lambda r: 15 <= abs(r["yaw_deg"]) < 30,
    "|yaw|>=30": lambda r: abs(r["yaw_deg"]) >= 30,
    "hold off<1cm": lambda r: r["hold_offset_mm"] < 10,
    "hold off>=1cm": lambda r: r["hold_offset_mm"] >= 10,
  }
  out["bins"] = {
    k: {
      "n": sum(f(r) for r in rows),
      "through_held": share("through_held", [r for r in rows if f(r)]),
    }
    for k, f in bins.items()
  }
  return out


def run(args) -> dict:
  env = build_env(args)
  actor = MlpActor(args.ckpt, args.device)
  dim = env.observation_manager.group_obs_dim["actor"][0]
  if actor.obs_dim != dim or actor.act_dim != env.action_space.shape[-1]:
    raise ValueError(
      f"{args.ckpt}: actor {actor.obs_dim} -> {actor.act_dim}, env {dim} -> {env.action_space.shape}"
    )
  reads = Reads(env, args.episode_s)
  obs, _ = env.reset()
  reads.start(torch.arange(env.num_envs, device=env.device))
  with torch.inference_mode():
    while len(reads.rows) < args.episodes:
      obs, _, term, trunc, _ = env.step(actor(obs["actor"]))
      done = (term | trunc).nonzero(as_tuple=False).flatten()
      if done.numel():
        reads.finish(done)
        reads.start(done)
      reads.step()
  rows = reads.rows[: args.episodes]
  out = {
    "ckpt": str(args.ckpt),
    "seed": args.seed,
    "obs_noise": args.obs_noise,
    **summarise(rows),
  }
  for k, v in out.items():
    if k != "bins":
      print(f"{k:28s} {v}")
  for k, v in out["bins"].items():
    print(f"  bin {k:16s} n={v['n']:4d} through_held={v['through_held']}")
  env.close()
  return {**out, "episodes_detail": rows}


def main(argv=None) -> None:
  args = parse_args(argv)
  out = run(args)
  if args.json:
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.json}")


if __name__ == "__main__":
  sys.exit(main())
