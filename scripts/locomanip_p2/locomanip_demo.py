"""P2b step 3: walk_to the table, then grasp -- a lower-body and an upper-body policy side by side.

Two slots run every step:
- lower body: P1.2 (14 leg/pelvis targets), always on. Its twist command is the only input
  a skill writes: walk_to writes the P-controller twist (speed floor v_min / w_min, as
  scripts/locomanip_eval.py), every other phase writes zero.
- upper body: a swappable controller for the left arm + jaw (8 actions). "home" holds the
  default pose (action 0); "grasp" is the vyq9vrpy actor. A place policy would be one more
  entry in UPPER.

The env is stand_grasp.build_cfg (grasp play cfg on the walking plant), with the robot
spawned at --start (x, y, yaw deg) instead of in front of the table, the table raised by
--table-dz, and `out_of_reach` removed (the object is out of reach while walking). Per env:
walk_to(0, 0, 0) [the grasp task's stance pose] -> rest --rest-s -> grasp until success,
termination or --grasp-s. A termination is final for that env (no retry).

Usage (code of e61bd8e, as stand_grasp.py):
  locomanip_demo.py <grasp ckpt> <p1.2 ckpt> <num_envs> [--video out.mp4] [--start -1.2 0.4 20]
"""

import argparse
import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_from_euler_xyz, wrap_to_pi

sys.path.insert(0, str(Path(__file__).parent))
import mujoco  # noqa: E402
from stand_grasp import P12_DEFAULT_OVERRIDE, Actor, build_cfg  # noqa: E402

WALK, REST, GRASP, DONE = 0, 1, 2, 3
PHASE_NAME = {WALK: "walk_to", REST: "settle", GRASP: "grasp", DONE: "done"}


def yaw_of(q: torch.Tensor) -> torch.Tensor:
  w, x, y, z = q.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class WalkTo:
  """walk_to(x, y, yaw) as a twist law: position first, then yaw, with a speed floor."""

  def __init__(self, a):
    self.a = a

  def __call__(self, pos, yaw, tgt, tgt_yaw):
    a = self.a
    e_w = tgt - pos
    c, s = yaw.cos(), yaw.sin()
    e_b = torch.stack(
      (c * e_w[:, 0] + s * e_w[:, 1], -s * e_w[:, 0] + c * e_w[:, 1]), -1
    )
    e_yaw = wrap_to_pi(tgt_yaw - yaw)
    d = e_b.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    v_xy = e_b / d * d.clamp(a.v_min, a.v_max)
    wz = torch.sign(e_yaw) * e_yaw.abs().clamp(a.w_min, a.w_max)
    far = d.squeeze(-1) >= a.tol_pos
    v_xy = torch.where(far[:, None], v_xy, 0.0)
    wz = torch.where(e_yaw.abs() >= a.tol_yaw, wz, 0.0)
    return torch.cat((v_xy, wz[:, None]), -1), d.squeeze(-1), e_yaw


def main():
  p = argparse.ArgumentParser()
  p.add_argument("grasp_ckpt")
  p.add_argument("p12_ckpt")
  p.add_argument("num_envs", type=int)
  p.add_argument("--start", type=float, nargs=3, default=(-1.2, 0.4, 20.0))
  p.add_argument("--table-dz", type=float, default=0.06)
  p.add_argument("--walk-s", type=float, default=15.0)
  p.add_argument("--rest-s", type=float, default=1.0)
  p.add_argument("--grasp-s", type=float, default=12.0)
  p.add_argument("--v-min", type=float, default=0.2)
  p.add_argument("--v-max", type=float, default=0.5)
  p.add_argument("--w-min", type=float, default=0.3)
  p.add_argument("--w-max", type=float, default=0.5)
  p.add_argument("--tol-pos", type=float, default=0.03)
  p.add_argument("--tol-yaw", type=float, default=math.radians(3))
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--video", default=None)
  args = p.parse_args()
  dev = "cuda:0"

  cfg = build_cfg(args.num_envs, "p12")
  cfg.seed = args.seed
  cfg.terminations.pop("out_of_reach")
  # build_cfg marks every env "standing" (the command term then zeroes the twist each step);
  # here the twist is written by walk_to.
  cfg.commands["twist"].rel_standing_envs = 0.0
  rob = cfg.scene.entities["robot"]
  sx, sy, syaw = args.start
  q = quat_from_euler_xyz(
    torch.tensor(0.0), torch.tensor(0.0), torch.tensor(math.radians(syaw))
  )
  rob.init_state = dataclasses.replace(
    rob.init_state, pos=(sx, sy, rob.init_state.pos[2]), rot=tuple(q.tolist())
  )
  tb = cfg.scene.entities["table"]
  x, y, z = tb.init_state.pos
  tb.init_state = dataclasses.replace(tb.init_state, pos=(x, y, z + args.table_dz))
  if args.video:
    cfg.viewer.width, cfg.viewer.height = 960, 720
    cfg.viewer.distance, cfg.viewer.elevation, cfg.viewer.azimuth = 3.2, -20.0, 140.0
  env = ManagerBasedRlEnv(
    cfg=cfg, device=dev, render_mode="rgb_array" if args.video else None
  )
  robot = env.scene["robot"]
  n = args.num_envs

  # Left-arm gravcomp only (see stand_grasp.py: the free joint's share is clamped away).
  m = env.sim.mj_model
  keep = torch.tensor(
    [
      b.startswith(("robot/arm_left", "robot/gripper_left"))
      for b in (
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)
      )
    ],
    device=dev,
  )
  env.sim.model.body_gravcomp[0] = (env.sim.model.body_gravcomp[0] * keep).clone()

  # Lower slot: P1.2, with its jaw observation relative to its own default (0).
  loco = Actor(args.p12_ckpt, dev)
  jn = robot.joint_names
  pos_off = torch.zeros(len(jn), device=dev)
  for name, v in P12_DEFAULT_OVERRIDE.items():
    i = jn.index(name)
    pos_off[i] = robot.data.default_joint_pos[0, i] - v
  jp = slice(3, 3 + len(jn))
  # Upper slot: one controller per phase.
  grasp = Actor(args.grasp_ckpt, dev)
  UPPER = {
    "home": lambda o: torch.zeros(o.shape[0], 8, device=dev),
    "grasp": grasp,
  }
  upper_of_phase = {WALK: "home", REST: "home", GRASP: "grasp", DONE: "home"}
  walk_to = WalkTo(args)

  twist = env.command_manager.get_term("twist")
  goal = env.command_manager.get_term("goal_pose")
  obs, _ = env.reset()
  # walk_to target: the grasp task's stance, (0, 0, yaw 0) in each env's frame.
  tgt = env.scene.env_origins[:, :2].clone()
  tgt_yaw = torch.zeros(n, device=dev)

  phase = torch.full((n,), WALK, dtype=torch.long, device=dev)
  t_phase = torch.zeros(n, device=dev)
  t_walk = torch.full((n,), float("nan"), device=dev)
  walk_err = torch.full((n, 2), float("nan"), device=dev)
  t_grasp = torch.full((n,), float("nan"), device=dev)
  succ = torch.zeros(n, dtype=torch.bool, device=dev)
  fail = {}
  frames, every = [], max(1, round(1 / 30 / env.step_dt))
  dt = env.step_dt
  T = int((args.walk_s + args.rest_s + args.grasp_s) / dt) + 10
  for step in range(T):
    pos = robot.data.root_link_pos_w[:, :2]
    yaw = yaw_of(robot.data.root_link_quat_w)
    v, d, e_yaw = walk_to(pos, yaw, tgt, tgt_yaw)
    v[phase != WALK] = 0.0
    twist.vel_command_b[:] = v

    # Phase transitions (before acting, so the slot switch takes effect this step).
    t_phase += dt
    arrived = (phase == WALK) & (d < args.tol_pos) & (e_yaw.abs() < args.tol_yaw)
    timeout = (phase == WALK) & (t_phase > args.walk_s)
    go_rest = arrived | timeout
    t_walk[go_rest] = step * dt
    walk_err[go_rest] = torch.stack((100 * d, torch.rad2deg(e_yaw.abs())), -1)[go_rest]
    phase[go_rest], t_phase[go_rest] = REST, 0.0
    go_grasp = (phase == REST) & (t_phase > args.rest_s)
    phase[go_grasp], t_phase[go_grasp] = GRASP, 0.0
    end = (phase == GRASP) & (t_phase > args.grasp_s)
    phase[end] = DONE

    lo = obs["loco"].clone()
    lo[:, jp] += pos_off
    a_legs = loco(lo)
    a_up = torch.zeros(n, 8, device=dev)
    for ph, name in upper_of_phase.items():
      sel = phase == ph
      if sel.any():
        a_up[sel] = UPPER[name](obs["actor"][sel])
    obs, _, term, trunc, _ = env.step(torch.cat((a_up, a_legs), -1))

    live = phase != DONE
    s = (goal.episode_success > 0) & (phase == GRASP) & ~succ
    t_grasp[s] = t_phase[s]
    succ |= s
    phase[s] = DONE
    ended = (term | trunc) & live & ~s
    for k in env.termination_manager.active_terms:
      hit = env.termination_manager.get_term(k) & ended
      if hit.any():
        for ph in (WALK, REST, GRASP):
          c = int((hit & (phase == ph)).sum())
          if c:
            fail[f"{k} during {PHASE_NAME[ph]}"] = (
              fail.get(f"{k} during {PHASE_NAME[ph]}", 0) + c
            )
    phase[ended] = DONE

    if args.video and step % every == 0:
      from PIL import Image, ImageDraw

      ph = int(phase[0])
      if ph == WALK:
        cap = f"walk_to(0, 0, 0)  [lower: P1.2 walking | upper: home]   error {100 * d[0]:.0f} cm, {math.degrees(abs(e_yaw[0])):.0f} deg"
      elif ph == REST:
        cap = "arrived, settling  [lower: P1.2 standing | upper: home]"
      elif ph == GRASP:
        cap = (
          f"grasp  [lower: P1.2 standing | upper: grasp policy]   {t_phase[0]:.1f} s"
        )
      else:
        cap = "grasp: success" if succ[0] else "ended"
      img = Image.fromarray(env.render())
      dr = ImageDraw.Draw(img)
      dr.rectangle((0, 0, img.width, 40), fill=(0, 0, 0))
      dr.text((12, 8), cap, fill=(255, 255, 255), font_size=20)
      frames.append(np.asarray(img))
    if not (phase != DONE).any():
      if args.video:  # hold the last frame for a second
        frames += [frames[-1]] * 30
      break

  reached = (walk_err[:, 0] < 5) & (walk_err[:, 1] < 5)
  print(
    f"start {tuple(args.start)}, table +{100 * args.table_dz:.0f} cm, {n} envs, seed {args.seed}"
  )
  print(
    f"  walk_to within 5 cm / 5 deg: {int(reached.sum())}/{n}, "
    f"time median {t_walk.nanmedian():.1f} s, error median {walk_err[:, 0].nanmedian():.1f} cm / {walk_err[:, 1].nanmedian():.1f} deg"
  )
  print(
    f"  grasp success: {int(succ.sum())}/{n} ({succ.float().mean():.3f}), "
    f"time median {t_grasp.nanmedian():.1f} s after the grasp starts"
  )
  print(f"  terminations: {fail}")
  if args.video:
    import mediapy

    mediapy.write_video(args.video, frames, fps=round(1 / (dt * every)))
    print("wrote", args.video, len(frames), "frames")


if __name__ == "__main__":
  main()
