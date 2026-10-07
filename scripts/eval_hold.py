"""Plan §13.1's L-H read (`pick-place-insert-plan.md` §13 *Reads and bars*): does the left hold the socket block
level and still at H, and does the hold survive a push on the block's top?

One 20 s episode per row, the actor's tracked observations (``--obs-noise on``, the training corruption), the
policy deterministic. ``S13=L-H`` / ``L-HP`` and `BS12`'s selectors set as the run was trained (L-HP's own
random pushes are switched off here; the eval pushes once, the same for every policy):

* **held at H** -- `mdp.held_at_goal` with L-H's parameters (within 1 cm of H, tilt < 3 deg, slower than 2 cm/s,
  the hand horizontal and pointing forward) on >= 80 % of the steps in [14, 15) s: the bar's headline;
* **the push test** -- from 15 s, 10 N straight down on the block's top centre for 1 s (``--push-n``,
  ``--push-s``): the block's largest tilt and displacement during it, and its distance from H and its tilt
  1 s after it ends (``recovered``: within 1 cm and < 5 deg);
* side grasps (grasped with the hand within 12 deg of horizontal), lifts, time to the first held step, drops
  (the block's centre below the table top + 1 cm after a lift, or below the table), arm 1-6 on the table
  (AH10's sensor, any force / > 30 N), the hand on the table;
* bins by the block's spawn x, y and yaw.

    S13=L-H SPAWN=WC BASELINE=B TABLE_DZ=22 ... python scripts/eval_hold.py --ckpt model_5999.pt --episodes 192 --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import torch

TASK = "Mjlab-PickPlace-Pal-Kangaroo-Grippers"


class MlpActor:
  """A checkpoint's deterministic actor: ``mlp((obs - mean) / (std + 0.01))`` with ELU between the layers (rsl_rl's
  `MLPModel` with its `EmpiricalNormalization`), any input / output width."""

  EPS = 1e-2

  def __init__(self, path: pathlib.Path, device: str) -> None:
    sd = torch.load(path, map_location=device, weights_only=False)["actor_state_dict"]
    self.mean = sd["obs_normalizer._mean"]
    self.std = sd["obs_normalizer._std"]
    n = len([k for k in sd if k.startswith("mlp.") and k.endswith(".weight")])
    self.layers = [
      (sd[f"mlp.{2 * i}.weight"], sd[f"mlp.{2 * i}.bias"]) for i in range(n)
    ]
    self.obs_dim = self.layers[0][0].shape[1]
    self.act_dim = self.layers[-1][0].shape[0]

  def __call__(self, obs: torch.Tensor) -> torch.Tensor:
    x = (obs - self.mean) / (self.std + self.EPS)
    for i, (w, b) in enumerate(self.layers):
      x = x @ w.T + b
      if i < len(self.layers) - 1:
        x = torch.nn.functional.elu(x)
    return x


def parse_args(argv=None) -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--ckpt", type=pathlib.Path, required=True)
  p.add_argument("--task", default=TASK)
  p.add_argument("--episodes", type=int, default=64)
  p.add_argument("--num-envs", type=int, default=64)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--episode-s", type=float, default=20.0)
  p.add_argument("--held-window", type=float, nargs=2, default=(14.0, 15.0))
  p.add_argument("--push-at", type=float, default=15.0)
  p.add_argument("--push-s", type=float, default=1.0)
  p.add_argument("--push-n", type=float, default=10.0)
  p.add_argument("--obs-noise", choices=("off", "on"), default="on")
  p.add_argument("--json", type=pathlib.Path, default=None)
  return p.parse_args(argv)


def build_env(args):
  import pal_mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg(args.task, play=False)
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  cfg.episode_length_s = args.episode_s
  cfg.events.pop(
    "push_block", None
  )  # L-HP's random pushes: the eval pushes once, for every policy alike
  if args.obs_noise == "off":
    cfg.observations["actor"].enable_corruption = False
  return ManagerBasedRlEnv(cfg=cfg, device=args.device)


class Reads:
  """Per-env running reads of the current episode, closed into a row when it ends."""

  def __init__(self, env, args) -> None:
    from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS_LEFT_ARM

    self.env, self.args = env, args
    self.gripper = KANGAROO_GRIPPERS_LEFT_ARM.gripper("left")
    self.held_params = env.reward_manager.get_term_cfg(
      "held"
    ).params  # resolved by the manager
    n, dev = env.num_envs, env.device
    self.dt = env.step_dt
    self.rows: list[dict] = []
    self.obj = env.scene["object"]
    self.table_sensor = (
      "arm_table_contact" if "arm_table_contact" in env.scene.sensors else None
    )
    z = lambda: torch.zeros(n, device=dev)  # noqa: E731
    b = lambda: torch.zeros(n, dtype=torch.bool, device=dev)  # noqa: E731
    self.spawn = torch.zeros(n, 3, device=dev)  # x, y (env frame), yaw mod 90 (deg)
    self.side_grasp, self.lifted, self.dropped = b(), b(), b()
    self.held_win, self.win_steps = z(), z()
    self.t_first_held = torch.full((n,), float("nan"), device=dev)
    self.pre_push = torch.zeros(n, 3, device=dev)
    self.push_tilt, self.push_disp = z(), z()
    self.after_dist = torch.full((n,), float("nan"), device=dev)
    self.after_tilt = torch.full((n,), float("nan"), device=dev)
    self.arm_table, self.arm_table_hard, self.hand_table = b(), b(), b()
    self.peak_tilt_held = z()

  def start(self, ids: torch.Tensor) -> None:
    """Fresh episode in ``ids``: read the spawn, clear the reads."""
    from pal_mjlab.tasks.manipulation.mdp.hold import object_yaw4_b  # noqa: F401

    env = self.env
    pos = self.obj.data.root_link_pos_w[ids] - env.scene.env_origins[ids]
    w, x, y, zq = self.obj.data.root_link_quat_w[ids].unbind(-1)
    yaw = torch.atan2(2 * (w * zq + x * y), 1 - 2 * (y * y + zq * zq))
    yaw90 = torch.remainder(yaw + math.pi / 4, math.pi / 2) - math.pi / 4
    self.spawn[ids] = torch.stack([pos[:, 0], pos[:, 1], torch.rad2deg(yaw90)], dim=-1)
    for t in (
      self.side_grasp,
      self.lifted,
      self.dropped,
      self.arm_table,
      self.arm_table_hard,
      self.hand_table,
    ):
      t[ids] = False
    for t in (
      self.held_win,
      self.win_steps,
      self.push_tilt,
      self.push_disp,
      self.peak_tilt_held,
    ):
      t[ids] = 0.0
    for t in (self.t_first_held, self.after_dist, self.after_tilt):
      t[ids] = float("nan")

  def push(self) -> torch.Tensor:
    """Write the push for this step's envs (zero elsewhere); returns which envs are pushed."""
    t = self.env.episode_length_buf.float() * self.dt
    on = (t >= self.args.push_at) & (t < self.args.push_at + self.args.push_s)
    force = torch.zeros(self.env.num_envs, 1, 3, device=self.env.device)
    force[on, 0, 2] = -self.args.push_n
    from mjlab.utils.lab_api.math import matrix_from_quat

    up = matrix_from_quat(self.obj.data.root_link_quat_w)[..., :, 2]
    from pal_mjlab.tasks.manipulation import scene

    torque = torch.cross((scene.BLOCK_HEIGHT / 2) * up.unsqueeze(1), force, dim=-1)
    self.obj.data.write_external_wrench(force, torque, body_ids=[0])
    return on

  def step(self) -> None:
    """Update the reads from the state after a step (episodes that did not just reset)."""
    from pal_mjlab.tasks.manipulation import mdp
    from pal_mjlab.tasks.manipulation.mdp.observations import (
      _grasped_bool,
      _was_lifted_bool,
    )
    from pal_mjlab.tasks.manipulation.tabletop_env_cfg import SUPPORT_SENSOR

    env, a = self.env, self.args
    t = (
      env.episode_length_buf.float() - 1.0
    ) * self.dt  # the step just simulated ended here
    held = mdp.held_at_goal(env, **self.held_params).bool()
    grasped = _grasped_bool(env, self.gripper.contact_sensor_name)
    side = mdp.approach_elevation(env, self.held_params["side_cfg"]) < math.radians(
      12.0
    )
    self.side_grasp |= grasped & side
    self.lifted |= _was_lifted_bool(
      env, self.gripper.contact_sensor_name, SUPPORT_SENSOR
    )
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from pal_mjlab.tasks.manipulation.mdp.events import surface_top_z

    top = surface_top_z(env, SceneEntityCfg("table"))
    z = self.obj.data.root_link_pos_w[:, 2]
    self.dropped |= self.lifted & (z < top + 0.01 + 0.028) & ~grasped
    first = held & torch.isnan(self.t_first_held)
    self.t_first_held[first] = t[first]
    win = (t >= a.held_window[0]) & (t < a.held_window[1])
    self.win_steps += win.float()
    self.held_win += (win & held).float()
    tilt = mdp.axis_tilt(env, "object")
    self.peak_tilt_held = torch.where(
      held, torch.maximum(self.peak_tilt_held, tilt), self.peak_tilt_held
    )
    pos = self.obj.data.root_link_pos_w
    before = (t >= a.push_at - 0.04) & (t < a.push_at)
    self.pre_push[before] = pos[before]
    during = (t >= a.push_at) & (t < a.push_at + a.push_s)
    disp = torch.norm(pos - self.pre_push, dim=-1)
    self.push_tilt = torch.where(
      during, torch.maximum(self.push_tilt, tilt), self.push_tilt
    )
    self.push_disp = torch.where(
      during, torch.maximum(self.push_disp, disp), self.push_disp
    )
    after = (t >= a.push_at + a.push_s + 1.0) & torch.isnan(self.after_dist)
    if after.any():
      cmd = env.command_manager.get_term("goal").command
      self.after_dist[after] = torch.norm(cmd - pos, dim=-1)[after]
      self.after_tilt[after] = tilt[after]
    if self.table_sensor is not None:
      s = env.scene[self.table_sensor].data
      f = (
        s.force.norm(dim=-1).amax(dim=tuple(range(1, s.force.dim() - 1)))
        if s.force is not None
        else None
      )
      found = s.found.bool().reshape(env.num_envs, -1).any(-1)
      self.arm_table |= found
      if f is not None:
        self.arm_table_hard |= f.reshape(env.num_envs, -1).amax(-1) > 30.0
    hs = env.scene[self.gripper.surface_sensor_name].data.found
    self.hand_table |= hs.bool().reshape(env.num_envs, -1).any(-1)

  def finish(self, ids: torch.Tensor) -> None:
    for i in ids.tolist():
      steps = float(self.win_steps[i])
      self.rows.append(
        {
          "x": float(self.spawn[i, 0]),
          "y": float(self.spawn[i, 1]),
          "yaw_deg": float(self.spawn[i, 2]),
          "side_grasp": bool(self.side_grasp[i]),
          "lifted": bool(self.lifted[i]),
          "held_at_H": steps > 0 and float(self.held_win[i]) >= 0.8 * steps,
          "held_share_window": float(self.held_win[i]) / steps
          if steps
          else float("nan"),
          "t_first_held": float(self.t_first_held[i]),
          "peak_tilt_held_deg": math.degrees(float(self.peak_tilt_held[i])),
          "push_tilt_deg": math.degrees(float(self.push_tilt[i])),
          "push_disp_mm": 1e3 * float(self.push_disp[i]),
          "after_dist_mm": 1e3 * float(self.after_dist[i]),
          "after_tilt_deg": math.degrees(float(self.after_tilt[i])),
          "recovered": bool(self.after_dist[i] < 0.01)
          and bool(self.after_tilt[i] < math.radians(5.0)),
          "dropped": bool(self.dropped[i]),
          "arm_table": bool(self.arm_table[i]),
          "arm_table_30N": bool(self.arm_table_hard[i]),
          "hand_table": bool(self.hand_table[i]),
        }
      )


def _share(rows, key, where=None) -> float | None:
  sel = [r for r in rows if where is None or where(r)]
  return None if not sel else sum(bool(r[key]) for r in sel) / len(sel)


def summarise(rows: list[dict]) -> dict:
  import statistics as st

  held = [r for r in rows if r["held_at_H"]]
  out = {
    "episodes": len(rows),
    "held_at_H": _share(rows, "held_at_H"),
    "side_grasp": _share(rows, "side_grasp"),
    "lifted": _share(rows, "lifted"),
    "dropped": _share(rows, "dropped"),
    "arm_table": _share(rows, "arm_table"),
    "arm_table_30N": _share(rows, "arm_table_30N"),
    "hand_table": _share(rows, "hand_table"),
    "t_first_held_median_s": st.median(
      [r["t_first_held"] for r in rows if r["t_first_held"] == r["t_first_held"]]
      or [float("nan")]
    ),
    "push_tilt_median_deg_held": st.median(
      [r["push_tilt_deg"] for r in held] or [float("nan")]
    ),
    "push_tilt_max_deg_held": max([r["push_tilt_deg"] for r in held] or [float("nan")]),
    "push_disp_median_mm_held": st.median(
      [r["push_disp_mm"] for r in held] or [float("nan")]
    ),
    "recovered_given_held": _share(held, "recovered"),
    "push_tilt_lt5_given_held": (
      sum(r["push_tilt_deg"] < 5.0 for r in held) / len(held)
    )
    if held
    else None,
  }
  bins = {
    "x<0.45": lambda r: r["x"] < 0.45,
    "0.45<=x<0.50": lambda r: 0.45 <= r["x"] < 0.50,
    "x>=0.50": lambda r: r["x"] >= 0.50,
    "y<0.14": lambda r: r["y"] < 0.14,
    "0.14<=y<0.20": lambda r: 0.14 <= r["y"] < 0.20,
    "y>=0.20": lambda r: r["y"] >= 0.20,
    "|yaw|<15": lambda r: abs(r["yaw_deg"]) < 15,
    "15<=|yaw|<30": lambda r: 15 <= abs(r["yaw_deg"]) < 30,
    "|yaw|>=30": lambda r: abs(r["yaw_deg"]) >= 30,
  }
  out["bins"] = {
    k: {"n": sum(f(r) for r in rows), "held_at_H": _share(rows, "held_at_H", f)}
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
  reads = Reads(env, args)
  obs, _ = env.reset()
  reads.start(torch.arange(env.num_envs, device=env.device))
  started = env.num_envs
  with torch.inference_mode():
    while len(reads.rows) < args.episodes:
      reads.push()
      obs, _, term, trunc, _ = env.step(actor(obs["actor"]))
      done = (term | trunc).nonzero(as_tuple=False).flatten()
      # A finished env is already reset (its state is the next episode's): close its row on the reads up to
      # the step before, then start the new one, then read this step for every env.
      if done.numel():
        reads.finish(done)
        reads.start(done)
        started += done.numel()
      reads.step()
  rows = reads.rows[: args.episodes]
  out = {
    "ckpt": str(args.ckpt),
    "seed": args.seed,
    "obs_noise": args.obs_noise,
    "push_n": args.push_n,
    **summarise(rows),
  }
  for k, v in out.items():
    if k != "bins":
      print(f"{k:28s} {v}")
  for k, v in out["bins"].items():
    print(f"  bin {k:16s} n={v['n']:4d} held={v['held_at_H']}")
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
