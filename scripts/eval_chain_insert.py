"""Plan §13.1's chain (`pick-place-insert-plan.md` §13 *Structure*, SEQ-H): L-H on the left arm, R-I on the right arm
through the mirror map, in the two-arm env ``ARMS=both S13=CHAIN`` (`pick_place/insert_chain.py`).

The left runs L-H from the start. The right holds its home until the block is **presented** -- its tracked centre
within `PRESENT_DIST` of H and its tracked speed under `PRESENT_SPEED` for `PRESENT_S` (what FoundationPose gives on the
robot) -- then runs R-I to the end of the episode; both keep their grasps (plan D6 (a): success is counted while the
right still holds the pin). The adapter (`eval_chain.MirrorAdapter`'s maps): the right's joints, velocities and last
actions x `ARM_MIRROR_SIGNS`, positions y -> -y, its TCP frame M R F M, a 4-fold yaw (cos 4 psi, sin 4 psi) ->
(cos, -sin), the pin's tilt (x, y) -> (x, -y).

Per episode: **through with the block held** -- the pin's bottom at or below the block's bottom inside the bore, its
top above the block's top (`mdp.held_through`), the right grasping the pin and the left the block -- on >= 80 % of the last second
(headline) and the first time it holds; when the block was presented; the block's largest drift from H and tilt while
the right works; drops (either part off its hand after the presentation, or below the table); arm-arm contact (S11's
sensor: any, > 30 N) and arm 1-6 on the table per arm; bins by both parts' spawn positions.

    ARMS=both S13=CHAIN GAINS=G20 EFFORT=CTP DR=DR-scene LAG=FP+JS ARM_TABLE=AH10 TABLE_DZ=22 \\
      python scripts/eval_chain_insert.py --lh L-H/model_5999.pt --ri R-I/model_5999.pt --episodes 128 --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from eval_chain import MirrorAdapter  # noqa: E402
from eval_hold import MlpActor  # noqa: E402

TASK = "Mjlab-PickPlace-Pal-Kangaroo-Grippers"
PRESENT_DIST = 0.01
PRESENT_SPEED = 0.02
PRESENT_S = 0.5


class ChainAdapter(MirrorAdapter):
  """L-H's 40-D and R-I's mirrored 42-D observations out of the chain env's ``actor`` group; the actions back."""

  def lh(self, t: dict) -> torch.Tensor:
    return torch.cat(
      [
        t["joint_pos"][:, self.jp_left],
        t["joint_vel"][:, self.jv_left],
        t["actions"][:, self.left_cols],
        t["tcp_pos"],
        t["tcp_quat"],
        t["block_centre"],
        t["block_yaw"],
        t["hold_goal"],
        t["block_up"],
      ],
      dim=-1,
    )

  def ri(self, t: dict) -> torch.Tensor:
    dev = self.env.device
    flip = torch.tensor([1.0, -1.0, 1.0], device=dev)
    flip2 = torch.tensor([1.0, -1.0], device=dev)
    return torch.cat(
      [
        t["joint_pos"][:, self.jp_right] * self.s8,
        t["joint_vel"][:, self.jv_right] * self.s7,
        t["actions"][:, self.right_cols] * self.s8,
        t["tcp_pos_right"] * flip,
        self._mirror_quat(t["tcp_quat_right"]),
        t["pin_pos"] * flip,
        t["pin_yaw"] * flip2,
        t["block_pos"] * flip,
        t["block_yaw"] * flip2,
        t["pin_tilt"] * flip2,
      ],
      dim=-1,
    )


def parse_args(argv=None) -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument(
    "--lh", type=pathlib.Path, required=True, help="L-H / L-HP checkpoint (the left)"
  )
  p.add_argument(
    "--ri",
    type=pathlib.Path,
    required=True,
    help="R-I / R-I-W / R-I-FT checkpoint (run mirrored)",
  )
  p.add_argument("--task", default=TASK)
  p.add_argument("--episodes", type=int, default=128)
  p.add_argument("--num-envs", type=int, default=64)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--episode-s", type=float, default=30.0)
  p.add_argument("--obs-noise", choices=("off", "on"), default="on")
  p.add_argument("--json", type=pathlib.Path, default=None)
  return p.parse_args(argv)


def build_env(args):
  import pal_mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg(args.task, play=False)
  if "block_centre" not in cfg.observations["actor"].terms:
    raise SystemExit(
      "eval_chain_insert reads plan §13.1's chain env: set ARMS=both S13=CHAIN"
    )
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  cfg.episode_length_s = args.episode_s
  if args.obs_noise == "off":
    cfg.observations["actor"].enable_corruption = False
  return ManagerBasedRlEnv(cfg=cfg, device=args.device)


class Reads:
  def __init__(self, env, episode_s: float) -> None:
    from pal_mjlab.tasks.manipulation import scene
    from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS

    self.env = env
    self.episode_s = episode_s
    self.scene = scene
    self.left = KANGAROO_GRIPPERS.gripper("left")
    self.right = KANGAROO_GRIPPERS.gripper("right")
    n, dev = env.num_envs, env.device
    self.dt = env.step_dt
    self.rows: list[dict] = []
    self.block = env.scene["block"]
    self.pin = env.scene["object"]
    self.prev_centre = torch.zeros(n, 3, device=dev)
    z = lambda: torch.zeros(n, device=dev)  # noqa: E731
    b = lambda: torch.zeros(n, dtype=torch.bool, device=dev)  # noqa: E731
    nan = lambda: torch.full((n,), float("nan"), device=dev)  # noqa: E731
    self.still = z()
    self.started = b()
    self.t_present, self.t_through = nan(), nan()
    self.end_through, self.end_steps = z(), z()
    self.drift, self.tilt = z(), z()
    self.block_drop, self.pin_drop, self.lifted_block = b(), b(), b()
    self.hh, self.hh30 = b(), b()
    self.armt_l, self.armt_r, self.armt30 = b(), b(), b()
    self.spawn = torch.zeros(n, 4, device=dev)
    self.geo = {
      "object_name": "object",
      "block_name": "block",
      "block_height": scene.BLOCK_HEIGHT,
      "hole_d": scene.BLOCK_HOLE_D,
    }

  def start(self, ids: torch.Tensor) -> None:
    o = self.env.scene.env_origins[ids]
    self.spawn[ids, :2] = (self.block.data.root_link_pos_w[ids] - o)[:, :2]
    self.spawn[ids, 2:] = (self.pin.data.root_link_pos_w[ids] - o)[:, :2]
    for t in (self.still, self.end_through, self.end_steps, self.drift, self.tilt):
      t[ids] = 0.0
    for t in (self.t_present, self.t_through):
      t[ids] = float("nan")
    for t in (
      self.started,
      self.block_drop,
      self.pin_drop,
      self.lifted_block,
      self.hh,
      self.hh30,
      self.armt_l,
      self.armt_r,
      self.armt30,
    ):
      t[ids] = False

  def present(self, t: dict) -> torch.Tensor:
    """Advance the presentation rule on the tracked block; returns the envs whose right starts now."""
    centre = t["block_centre"]
    near = torch.norm(centre - t["hold_goal"], dim=-1) < PRESENT_DIST
    speed = torch.norm(centre - self.prev_centre, dim=-1) / self.dt
    self.prev_centre = centre.clone()
    calm = near & (speed < PRESENT_SPEED)
    self.still = torch.where(calm, self.still + self.dt, torch.zeros_like(self.still))
    go = ~self.started & (self.still >= PRESENT_S)
    tt = self.env.episode_length_buf.float() * self.dt
    self.t_present[go] = tt[go]
    self.started |= go
    return self.started

  def step(self) -> None:
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from pal_mjlab.tasks.manipulation import mdp
    from pal_mjlab.tasks.manipulation.mdp.events import surface_top_z
    from pal_mjlab.tasks.manipulation.mdp.observations import _grasped_bool

    env = self.env
    tt = (env.episode_length_buf.float() - 1.0) * self.dt
    hl = _grasped_bool(env, self.left.contact_sensor_name)
    hr = _grasped_bool(env, self.right.contact_sensor_name)
    through = mdp.held_through(env, **self.geo) & hl & hr
    first = through & torch.isnan(self.t_through)
    self.t_through[first] = tt[first]
    end = tt >= self.episode_s - 1.0 - self.dt
    self.end_steps += end.float()
    self.end_through += (end & through).float()
    top = surface_top_z(env, SceneEntityCfg("table"))
    bz = self.block.data.root_link_pos_w[:, 2]
    self.lifted_block |= hl & (bz > top + 0.03)
    cmd = env.command_manager.get_term("goal_pose").command
    centre = self.block.data.root_link_pos_w + (
      self.scene.BLOCK_HEIGHT / 2
    ) * mdp.object_up_w(env, "block")
    on = self.started
    self.drift = torch.where(
      on, torch.maximum(self.drift, torch.norm(centre - cmd, dim=-1)), self.drift
    )
    self.tilt = torch.where(
      on, torch.maximum(self.tilt, mdp.axis_tilt(env, "block")), self.tilt
    )
    self.block_drop |= (on & ~hl) | (bz < top - 0.05)
    self.pin_drop |= self.pin.data.root_link_pos_w[:, 2] < top - 0.05
    hh = env.scene["arm_arm_contact"].data
    f = hh.force.norm(dim=-1).reshape(env.num_envs, -1).amax(-1)
    self.hh |= hh.found.bool().reshape(env.num_envs, -1).any(-1)
    self.hh30 |= f > 30.0
    for name, flag in (
      ("arm_table_contact", self.armt_l),
      ("right_arm_table_contact", self.armt_r),
    ):
      if name in env.scene.sensors:
        d = env.scene[name].data
        flag |= d.found.bool().reshape(env.num_envs, -1).any(-1)
        if d.force is not None:
          self.armt30 |= d.force.norm(dim=-1).reshape(env.num_envs, -1).amax(-1) > 30.0

  def finish(self, ids: torch.Tensor) -> None:
    for i in ids.tolist():
      steps = float(self.end_steps[i])
      self.rows.append(
        {
          "block_xy": [round(float(v), 4) for v in self.spawn[i, :2]],
          "pin_xy": [round(float(v), 4) for v in self.spawn[i, 2:]],
          "through_held": steps > 0 and float(self.end_through[i]) >= 0.8 * steps,
          "t_presented": float(self.t_present[i]),
          "presented": bool(self.started[i]),
          "t_first_through": float(self.t_through[i]),
          "block_drift_mm": 1e3 * float(self.drift[i]),
          "block_tilt_deg": math.degrees(float(self.tilt[i])),
          "block_lifted": bool(self.lifted_block[i]),
          "block_dropped": bool(self.block_drop[i]),
          "pin_dropped": bool(self.pin_drop[i]),
          "arm_arm": bool(self.hh[i]),
          "arm_arm_30N": bool(self.hh30[i]),
          "arm_table_left": bool(self.armt_l[i]),
          "arm_table_right": bool(self.armt_r[i]),
          "arm_table_30N": bool(self.armt30[i]),
        }
      )


def summarise(rows: list[dict]) -> dict:
  import statistics as st

  def share(k, sel=None):
    sel = rows if sel is None else sel
    return None if not sel else sum(bool(r[k]) for r in sel) / len(sel)

  pres = [r for r in rows if r["presented"]]
  med = lambda v: st.median(v) if v else float("nan")  # noqa: E731
  out = {
    "episodes": len(rows),
    "through_held": share("through_held"),
    "presented": share("presented"),
    "through_given_presented": share("through_held", pres),
    "t_presented_median_s": med([r["t_presented"] for r in pres]),
    "t_first_through_median_s": med(
      [
        r["t_first_through"]
        for r in rows
        if r["t_first_through"] == r["t_first_through"]
      ]
    ),
    "block_drift_median_mm": med([r["block_drift_mm"] for r in pres]),
    "block_tilt_median_deg": med([r["block_tilt_deg"] for r in pres]),
    "block_dropped": share("block_dropped"),
    "pin_dropped": share("pin_dropped"),
    "arm_arm": share("arm_arm"),
    "arm_arm_30N": share("arm_arm_30N"),
    "arm_table_left": share("arm_table_left"),
    "arm_table_right": share("arm_table_right"),
    "arm_table_30N": share("arm_table_30N"),
  }
  bins = {
    "block x<0.475": lambda r: r["block_xy"][0] < 0.475,
    "block x>=0.475": lambda r: r["block_xy"][0] >= 0.475,
    "block y<0.17": lambda r: r["block_xy"][1] < 0.17,
    "block y>=0.17": lambda r: r["block_xy"][1] >= 0.17,
    "pin x<0.50": lambda r: r["pin_xy"][0] < 0.50,
    "pin x>=0.50": lambda r: r["pin_xy"][0] >= 0.50,
    "pin y>-0.18": lambda r: r["pin_xy"][1] > -0.18,
    "pin y<=-0.18": lambda r: r["pin_xy"][1] <= -0.18,
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
  ad = ChainAdapter(env)
  lh, ri = MlpActor(args.lh, args.device), MlpActor(args.ri, args.device)
  if (lh.obs_dim, lh.act_dim) != (40, 8) or (ri.obs_dim, ri.act_dim) != (42, 8):
    raise ValueError(
      f"L-H is 40 -> 8 and R-I 42 -> 8; got {lh.obs_dim}->{lh.act_dim}, {ri.obs_dim}->{ri.act_dim}"
    )
  reads = Reads(env, args.episode_s)
  obs, _ = env.reset()
  reads.start(torch.arange(env.num_envs, device=env.device))
  with torch.inference_mode():
    while len(reads.rows) < args.episodes:
      t = ad.terms(obs, "actor")
      go = reads.present(t)
      a_l = lh(ad.lh(t))
      a_r = torch.where(go.unsqueeze(-1), ri(ad.ri(t)), torch.zeros_like(a_l))
      obs, _, term, trunc, _ = env.step(ad.actions(a_l, a_r))
      done = (term | trunc).nonzero(as_tuple=False).flatten()
      if done.numel():
        reads.finish(done)
        reads.start(done)
      reads.step()
  rows = reads.rows[: args.episodes]
  out = {
    "lh": str(args.lh),
    "ri": str(args.ri),
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
