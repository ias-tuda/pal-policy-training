"""What stops the arm at the grasp in MuJoCo? Replays the export's episode specs and logs,
per step, hand-table contact (and force), cube-table contact, grasp, TCP speed, and which
robot geoms touch the table or the cube.

  .venv/bin/python scripts/sim2sim/contact_probe.py --checkpoint <model_*.pt> --steps 40
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

RUN_RE = re.compile(
  r"_pp_[^_]+_(?P<reward>R-[^_]+)_(?P<dr>DR[^_]*)_(?P<ppo>PPO-[^_]+)_n\d+_seed(?P<seed>\d+)$"
)


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--checkpoint", type=Path, required=True)
  p.add_argument("--steps", type=int, default=40)
  p.add_argument(
    "--episodes", type=Path, default=Path(__file__).with_name("episodes.json")
  )
  p.add_argument("--out", type=Path, default=None)
  args = p.parse_args()
  m = RUN_RE.search(args.checkpoint.resolve().parent.name)
  os.environ["REWARD"], os.environ["DR"], os.environ["PPO"] = (
    m["reward"],
    m["dr"],
    m["ppo"],
  )

  from dataclasses import asdict

  import mujoco
  import pal_mjlab.tasks  # noqa: F401
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.utils.lab_api.math import quat_from_euler_xyz
  from pal_mjlab.tasks.manipulation import scene
  from pal_mjlab.tasks.manipulation.mdp.observations import _grasped_bool
  from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import SPAWN_XY

  task = "Mjlab-PickPlace-Pal-Kangaroo-Grippers"
  episodes = json.loads(args.episodes.read_text())["episodes"]
  cfg = load_env_cfg(task, play=True)
  cfg.episode_length_s = load_env_cfg(task, play=False).episode_length_s
  cfg.scene.num_envs = len(episodes)
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
  runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(
    wrapped, asdict(load_rl_cfg(task)), device="cpu"
  )
  runner.load(
    str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location="cpu"
  )
  policy = runner.get_inference_policy(device="cpu")
  robot, obj = env.scene["robot"], env.scene["object"]
  n = len(episodes)
  wrapped.reset()
  q0 = robot.data.default_joint_pos.clone()
  robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
  pose = torch.zeros(n, 7)
  for k, e in enumerate(episodes):
    pose[k, :3] = torch.tensor(
      [SPAWN_XY[0] + e["dx"], SPAWN_XY[1] + e["dy"], scene.TABLE_TOP_Z + 0.025]
    )
  z = torch.zeros(n)
  pose[:, 3:] = quat_from_euler_xyz(z, z, torch.tensor([e["dyaw"] for e in episodes]))
  obj.write_root_link_pose_to_sim(pose)
  obj.write_root_link_velocity_to_sim(torch.zeros(n, 6))
  env.sim.forward()
  obs = wrapped.get_observations()

  mjm = env.sim.mj_model
  hand = env.scene["left_hand_surface_contact"]
  site = robot.site_names.index("tcp_left")
  table_body = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "table/table")
  cube_body = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "object/object")
  rows = []
  prev_tcp = robot.data.site_pos_w[:, site].clone()
  for t in range(args.steps):
    with torch.no_grad():
      a = policy(obs)
    obs, _, _, _ = wrapped.step(a)
    tcp = robot.data.site_pos_w[:, site].clone()
    speed = torch.norm(tcp - prev_tcp, dim=-1) / env.step_dt
    prev_tcp = tcp
    found = hand.data.found.view(n, -1).bool().any(-1)
    force = hand.data.force.view(n, -1, 3).norm(dim=-1).sum(-1)
    grasped = _grasped_bool(env, "left_pad_contact")
    # Which robot geoms touch the table / the cube, world 0 only (named, for the record).
    d = env.sim.data
    ncon = int(d.nacon.numpy()[0]) if hasattr(d, "nacon") else 0
    touching = set()
    if ncon:
      geom = d.contact.geom.numpy()[:ncon]
      world = d.contact.worldid.numpy()[:ncon]
      for (g1, g2), w in zip(geom, world, strict=True):
        if w != 0:
          continue
        b1, b2 = mjm.geom_bodyid[g1], mjm.geom_bodyid[g2]
        for g, b, other in ((g1, b1, b2), (g2, b2, b1)):
          if other in (table_body, cube_body) and b not in (table_body, cube_body):
            nm = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
            touching.add(f"{nm}->{'table' if other == table_body else 'cube'}")
    rows.append(
      {
        "step": t,
        "hand_table": found.int().tolist(),
        "hand_table_force": [round(float(f), 1) for f in force],
        "grasped": grasped.int().tolist(),
        "tcp_speed": [round(float(s), 3) for s in speed],
        "ep0_touching": sorted(touching),
      }
    )
    print(
      f"step {t:2d} grasped {grasped.int().tolist()} hand-table {found.int().tolist()} "
      f"F0 {float(force[0]):6.1f} v_tcp0 {float(speed[0]):.2f} {sorted(touching)}"
    )
  if args.out:
    args.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
  main()
