"""Plan S11.2's snapshot bank for ``START=hold`` (`pick-place-s11-bimanual-plan.md`, *S11.2*, training changes).

Runs the SEQ chain (`eval_chain.py`: L-T on the left, R-ST / R-ST-W mirrored on the right) in JOINT's env, as
training sees it (DR-scene, the actor's observation noise on), and every `EVERY` control steps saves the envs
that are in the hold state the plan asks for:

* B in the right hand (its grasp sensor), lifted, over A (xy < `OVER`) with its bottom 2-6 cm above A's top;
* A placed at T (the left's gate, `mdp.base_placed_bool`) and within `A_AT_T` of T;
* the left handed off (its phase >= OUT: leaving or going home);
* at most `MAX_STEP` steps into the episode, so that a 15 s training episode has time left.

Each snapshot keeps what `bimanual.hold_start` restores: every robot joint's position and velocity, A's and
B's poses (env frame) and B's velocity, the action term's last targets and jaw servos, the episode step and
T's offset from the env origin. It also keeps the table's top (env frame), to report how far DR-scene moved
it. The file is a torch dict; it is data (``results/``), not code.

  REWARD=S11-BI ARMS=both ... python scripts/s112_hold_bank.py --lt L-T.pt --rst R-ST-W.pt --episodes 1024 \\
    --out hold_bank.pt
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

EVERY = 10
OVER = 0.03
GAP = (0.02, 0.06)
A_AT_T = 0.01
MAX_STEP = 600


def parse_args(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--lt", type=pathlib.Path, required=True)
  p.add_argument("--rst", type=pathlib.Path, required=True)
  p.add_argument("--task", default="Mjlab-PickPlace-Pal-Kangaroo-Grippers")
  p.add_argument("--episodes", type=int, default=1024)
  p.add_argument("--num-envs", type=int, default=256)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--episode-s", type=float, default=30.0)
  p.add_argument("--right-timeout", type=float, default=24.0)
  p.add_argument("--obs-noise", choices=("off", "on"), default="on")
  p.add_argument("--out", type=pathlib.Path, required=True)
  return p.parse_args(argv)


def main(argv=None) -> None:
  args = parse_args(argv)
  from eval_bimanual import Reads, build_env
  from eval_chain import MirrorAdapter, OneArmActor
  from pal_mjlab.tasks.manipulation.mdp.events import surface_top_z
  from pal_mjlab.tasks.manipulation.mdp.handoff import HOME, OUT
  from pal_mjlab.tasks.manipulation.mdp.observations import _grasped_bool
  from pal_mjlab.tasks.manipulation.pick_place import bimanual as bi

  from mjlab.managers.scene_entity_config import SceneEntityCfg

  env = build_env(args)
  ad = MirrorAdapter(env)
  lt, rst = OneArmActor(args.lt, args.device), OneArmActor(args.rst, args.device)
  reads = Reads(env)
  term = env.action_manager.get_term("joint_pos")
  robot = env.scene["robot"]
  a_ent, b_ent = env.scene["object"], env.scene[bi.CUBE_B]
  goal = env.command_manager.get_term("goal_pose")
  table = SceneEntityCfg("table")
  table.resolve(env.scene)
  origins = env.scene.env_origins
  right_timeout = round(args.right_timeout / env.step_dt)
  k_left = term.arm_index("left")
  rows: dict[str, list] = {
    k: [] for k in "q qd a_pos a_quat b_pos b_quat b_vel target servo step goal_offset table_z".split()
  }
  finished = 0
  obs, _ = env.reset()
  with torch.inference_mode():
    while finished < args.episodes:
      start = reads.step(right_timeout)
      t = ad.terms(obs, "actor")
      a_l = lt(ad.left(t))
      go = (term.phase[:, k_left] >= HOME).unsqueeze(-1)
      a_r = torch.where(go, rst(ad.right(t)), torch.zeros_like(a_l))
      term.start_handoff("right", start & go.squeeze(-1))
      step = env.episode_length_buf
      if True:
        a = a_ent.data.root_link_pos_w
        b = b_ent.data.root_link_pos_w
        gap = (b[:, 2] - 0.025) - (a[:, 2] + 0.025)
        pick = (
          _grasped_bool(env, "right_pad_contact")
          & (torch.norm((b - a)[:, :2], dim=-1) < OVER)
          & (gap >= GAP[0])
          & (gap <= GAP[1])
          & reads._last["a_placed"]
          & (torch.norm(a[:, :2] - goal.command[:, :2], dim=-1) < A_AT_T)
          & (term.phase[:, k_left] >= OUT)
          & (step <= MAX_STEP)
          & (step % EVERY == 0)
        )
        ids = torch.nonzero(pick, as_tuple=False).flatten()
        if ids.numel():
          rows["q"].append(robot.data.joint_pos[ids].clone())
          rows["qd"].append(robot.data.joint_vel[ids].clone())
          rows["a_pos"].append((a - origins)[ids].clone())
          rows["a_quat"].append(a_ent.data.root_link_quat_w[ids].clone())
          rows["b_pos"].append((b - origins)[ids].clone())
          rows["b_quat"].append(b_ent.data.root_link_quat_w[ids].clone())
          rows["b_vel"].append(b_ent.data.root_link_vel_w[ids].clone())
          rows["target"].append(term._last_target[ids].clone())
          servo = term._servo if term._servo_idx is not None else torch.zeros(env.num_envs, 0, device=env.device)
          rows["servo"].append(servo[ids].clone())
          rows["step"].append(step[ids].clone())
          rows["goal_offset"].append((goal.command[:, :2] - origins[:, :2])[ids].clone())
          rows["table_z"].append((surface_top_z(env, table) - origins[:, 2])[ids].clone())
      obs, _, term_done, trunc, _ = env.step(ad.actions(a_l, a_r))
      done = torch.nonzero(term_done | trunc, as_tuple=False).flatten()
      if done.numel():
        reads.finish(done, env.step_dt)
        finished += done.numel()
  bank = {k: torch.cat(v).cpu() if v else torch.zeros(0) for k, v in rows.items()}
  bank["joint_names"] = list(robot.joint_names)
  bank["target_names"] = [n.split("/")[-1] for n in term.target_names]
  bank["source"] = {"lt": str(args.lt), "rst": str(args.rst), "episodes": finished, "seed": args.seed, "every": EVERY}
  n = len(bank["step"])
  gaps = (bank["b_pos"][:, 2] - bank["a_pos"][:, 2] - 0.05) if n else torch.zeros(0)
  summary = {
    "snapshots": n,
    "episodes": finished,
    "step_min_median_max": [int(bank["step"].min()), int(bank["step"].median()), int(bank["step"].max())] if n else None,
    "gap_cm_median": float(gaps.median() * 100) if n else None,
    "table_z_spread_mm": float((bank["table_z"].max() - bank["table_z"].min()) * 1000) if n else None,
  }
  bank["summary"] = summary
  args.out.parent.mkdir(parents=True, exist_ok=True)
  torch.save(bank, args.out)
  (args.out.with_suffix(".json")).write_text(json.dumps({**summary, "source": bank["source"]}, indent=1))
  print(json.dumps(summary, indent=1))
  env.close()


if __name__ == "__main__":
  sys.exit(main())
