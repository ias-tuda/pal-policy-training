"""Export an S7-8 (stack-speed) pick-and-place checkpoint for the PAL sim2sim stack, and write down
everything a deployment needs to know about the plant it was trained on.

`export_for_pal_s7.py` is RCTP's (S7-4 / S7-5): SLEW 0.5, the raw last action in the actor, 6 s of
reference. S7-8 changes the speed bound (`SLEW=stack`: arm 1-3 at 1/16 of the URDF speed, arm 4-7 at 1/8,
jaw 0.03 rad/step), the cube / goal area (`SPAWN=WC`, `TARGET=WC`), and for the `-c` codes what the
actor's `actions` term shows (`CMD_OBS=cmd`: the commanded target relative to the default pose, not the
raw last action). The rest is S7's: baseline B, G20, CTP, LAG=FP+JS, 38-D actor. Selectors by environment:

  common   REWARD R-C-strict+M1+M2+M3+M6+J+M8   DR DR-scene   PPO PPO-stack   SLEW stack   BASELINE B
           GAINS G20   EFFORT CTP   TOOL_SPEED none   LAG FP+JS   ACTION abs   EPISODE_S 55
  ABS-a    SPAWN W2   TARGET B                                  (actions = raw last action)
  WC-ABS-c SPAWN WC   TARGET WC   CMD_OBS cmd

Written next to the ONNX: `pick_place_profile.yaml` (what `pal_policy_inferer_pick_place_node` reads;
`actions_obs: act | cmd` is new), `deployment_info.yaml`, `deployment_start_config.json`,
`start_observation_distribution.json`, `params/`, `mujoco_reference.npz`, `grasp_calibration.json`,
`episodes.json` (the spec the reference and the PAL runs share).

  uv run --extra cu128 python scripts/sim2sim/export_for_pal_s78.py \\
    --checkpoint ~/s78/ckpt/WC-ABS-c_s1_model_4999.pt --out <models>/pp_s78_WC-ABS-c_s1_it4999 \\
    --episodes scripts/sim2sim/episodes_s78_wc.json --ref-seconds 20 --params-dir <the run's params/>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

#: The S7-8 recipe shared by every code (`submit_s78.sh`); SPAWN / TARGET / CMD_OBS come from the code.
S78 = {
  "REWARD": "R-C-strict+M1+M2+M3+M6+J+M8",
  "DR": "DR-scene",
  "PPO": "PPO-stack",
  "SLEW": "stack",
  "BASELINE": "B",
  "SPAWN": "W2",
  "TARGET": "B",
  "GAINS": "G20",
  "EFFORT": "CTP",
  "TOOL_SPEED": "none",
  "LAG": "FP+JS",
  "ACTION": "abs",
  "EPISODE_S": "55",
}
for _k, _v in S78.items():
  os.environ.setdefault(_k, _v)

BASE_BODY = "base_link"
TCP_SITE = "tcp_left"
GRASP_SENSOR = "left_pad_contact"
SUPPORT_SENSOR = "object_table_contact"
ARM = [f"arm_left_{i}_joint" for i in range(1, 8)]
JAW = "gripper_left_finger_joint"


def sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--checkpoint", type=Path, required=True)
  p.add_argument("--out", type=Path, required=True)
  p.add_argument("--episodes", type=Path, required=True)
  p.add_argument("--ref-seconds", type=float, default=20.0)
  p.add_argument("--dist-envs", type=int, default=256)
  p.add_argument("--device", default="cpu")
  p.add_argument(
    "--params-dir",
    type=Path,
    default=None,
    help="the training run's params/ (env.yaml, agent.yaml as train.py dumped them); "
    "without it they are regenerated from the selectors",
  )
  p.add_argument("--container-model-root", default="/catkin_ws/src/models")
  return p.parse_args()


def _git(*args: str) -> str:
  """git in the checkout pal_mjlab is imported from (the code the policy is exported with)."""
  import pal_mjlab

  try:
    return subprocess.check_output(
      ["git", *args],
      cwd=Path(pal_mjlab.__file__).resolve().parent,
      text=True,
      stderr=subprocess.DEVNULL,
    ).strip()
  except Exception:  # noqa: BLE001
    return "unknown"


def main() -> None:
  args = parse_args()
  ckpt = args.checkpoint.resolve()
  out = args.out.resolve()
  (out / "params").mkdir(parents=True, exist_ok=True)
  iteration = int(re.search(r"model_(\d+)\.pt$", ckpt.name).group(1))
  spec = json.loads(args.episodes.read_text())
  episodes = spec["episodes"]
  shutil.copy2(args.episodes, out / "episodes.json")

  import mujoco
  import numpy as np
  import pal_mjlab.tasks  # noqa: F401  (registers the task ids)
  import torch
  from dataclasses import asdict

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
  from mjlab.tasks.registry import load_runner_cls
  from mjlab.utils.lab_api.math import (
    euler_xyz_from_quat,
    quat_apply,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
  )
  from mjlab.utils.os import dump_yaml
  from pal_mjlab.tasks.manipulation import scene
  from pal_mjlab.tasks.manipulation.mdp.observations import (
    _grasped_bool,
    _on_surface_bool,
  )
  from pal_mjlab.tasks.manipulation.pick_place.kangaroo.env_cfg import (
    kangaroo_pick_place_env_cfg,
    kangaroo_pick_place_rl_cfg,
  )
  from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import (
    BASELINES,
    GOAL_THRESHOLD,
    SPAWN_BOXES,
    TARGET_AREAS,
  )
  from pal_mjlab.tasks.manipulation.tabletop_env_cfg import COMMAND, CONTROL_DT, OBJECT

  selectors = {k: os.environ[k] for k in S78} | {
    k: os.environ[k] for k in ("CMD_OBS", "OCC", "OBJ_OBS", "OCC_FLAG") if k in os.environ
  }
  cmd_obs = selectors.get("CMD_OBS", "act") == "cmd"
  assert selectors["ACTION"] == "abs", "the node implements the absolute action only (DEL needs an integrator)"
  baseline = BASELINES[selectors["BASELINE"]]
  spawn_box = SPAWN_BOXES[selectors["SPAWN"]]
  target_area = TARGET_AREAS[selectors["TARGET"]]
  episode_s = float(selectors["EPISODE_S"])
  device = args.device

  def cfg_for(dr: str, play: bool, num_envs: int):
    """The task cfg with DR set for this call: the registry builds its cfg once at import, the
    factory reads the selectors every time it is called."""
    keep = os.environ["DR"]
    os.environ["DR"] = dr
    try:
      cfg = kangaroo_pick_place_env_cfg(play=play)
    finally:
      os.environ["DR"] = keep
    cfg.scene.num_envs = num_envs
    cfg.seed = 0
    return cfg

  # ------------------------------------------------ training-config dump (as train.py does)
  train_cfg = cfg_for(selectors["DR"], play=False, num_envs=1)
  agent_cfg = kangaroo_pick_place_rl_cfg()
  if args.params_dir is not None:
    for f in ("env.yaml", "agent.yaml"):
      shutil.copy2(args.params_dir / f, out / "params" / f)
    # The selectors above name the recipe by hand: regenerate the cfg from them and diff it with the
    # run's own dump (num_envs and the seed differ by design), so a wrong selector shows here.
    dump_yaml(out / "params" / "env_regenerated.yaml", asdict(train_cfg))
    _diff_dumps(out / "params" / "env.yaml", out / "params" / "env_regenerated.yaml")
  else:
    dump_yaml(out / "params" / "env.yaml", train_cfg)
    dump_yaml(out / "params" / "agent.yaml", agent_cfg)

  # ---------------------------------------------------------- env for the reference rollout
  # DR0 + play: the nominal scene (no table shift) and no observation noise; the checkpoint is
  # the same whatever it trained under. Observation latency is a delay, not noise, and stays.
  env_cfg = cfg_for("DR0", play=True, num_envs=len(episodes))
  env_cfg.episode_length_s = 1.0e9
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
  runner_cls = load_runner_cls("Mjlab-PickPlace-Pal-Kangaroo-Grippers")
  from mjlab.rl import MjlabOnPolicyRunner

  runner_cls = runner_cls or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  robot = env.scene["robot"]
  obj = env.scene[OBJECT]
  term = env.action_manager.get_term("joint_pos")
  action_joints = [n.split("/")[-1] for n in term.target_names]
  assert action_joints == ARM + [JAW], action_joints
  jidx = [robot.joint_names.index(n) for n in action_joints]
  default_q = robot.data.default_joint_pos[0, jidx].tolist()
  scale = term.scale[0].tolist() if torch.is_tensor(term.scale) else [term.scale] * 8
  offset = (
    term.offset[0].tolist() if torch.is_tensor(term.offset) else [term.offset] * 8
  )
  assert np.allclose(offset, default_q), (offset, default_q)
  big = 1.0e9

  def finite(v: float) -> float:
    return float(v) if math.isfinite(v) else (big if v > 0 else -big)

  clip = term._clip[0].tolist() if getattr(term, "_clip", None) is not None else None
  clip_lo = [finite(c[0]) for c in clip] if clip else [-big] * 8
  clip_hi = [finite(c[1]) for c in clip] if clip else [big] * 8
  rate = [finite(x) for x in term._rate[0].tolist()]
  # C: the arm's joint range. T: rows of (tau_cmd / K, D / K, K) on the arm.
  assert term._range_idx is not None and term._band_idx is not None, "not a CTP policy"
  assert term._range_idx.tolist() == list(range(7)) and term._band_idx.tolist() == list(range(7))
  range_lo = [float(v) for v in term._range[0, :, 0]]
  range_hi = [float(v) for v in term._range[0, :, 1]]
  band = term._band.cpu().numpy()
  band_half, band_dk, band_k = band[:, 0].tolist(), band[:, 1].tolist(), band[:, 2].tolist()
  obs_names = list(env.observation_manager.active_terms["actor"])
  obs_dims = [
    int(np.prod(d)) for d in env.observation_manager._group_obs_term_dim["actor"]
  ]
  assert obs_names == [
    "joint_pos", "joint_vel", "actions", "tcp_pos", "tcp_quat",
    "object_pos", "object_yaw", "target_pos",
  ], obs_names  # fmt: skip
  assert sum(obs_dims) == 38, obs_dims

  # ---------------------------------------------------------------------------- ONNX
  onnx_path = out / f"model_{iteration}.onnx"
  runner.export_policy_to_onnx(str(out), onnx_path.name)
  metadata = get_base_metadata(env, ckpt.parent.name)
  metadata["action_joint_names"] = action_joints
  metadata["selectors"] = json.dumps(selectors)
  attach_metadata_to_onnx(str(onnx_path), metadata)
  obs_td = wrapped.get_observations()
  x = obs_td["actor"].float()
  noisy = x.repeat(64 // x.shape[0] + 1, 1)[:64] + 0.3 * torch.randn(64, x.shape[1])
  xs = torch.cat([x, noisy], dim=0)
  with torch.no_grad():
    ref_a = torch.cat([policy(row) for row in _rows(obs_td, xs)], dim=0)
  np.savez(out / "onnx_check.npz", obs=xs.cpu().numpy(), actions=ref_a.cpu().numpy())

  # ----------------------------------------------------------------- geometry, base_link
  bi = robot.body_names.index(BASE_BODY)
  base_pos = robot.data.body_link_pos_w[0, bi].clone()
  base_quat = robot.data.body_link_quat_w[0, bi].clone()

  def to_b(p_w):
    p = torch.as_tensor(p_w, dtype=torch.float32, device=base_pos.device)
    return quat_apply(quat_inv(base_quat), p - base_pos).tolist()

  top_z = baseline.table_top_z
  half = 0.025
  rest_z = top_z + half
  spawn_c = ((spawn_box.x[0] + spawn_box.x[1]) / 2, (spawn_box.y[0] + spawn_box.y[1]) / 2)
  target_c = (
    baseline.target_xy
    if target_area is None
    else (sum(target_area.x) / 2, sum(target_area.y) / 2)
  )
  spawn_b = to_b([spawn_c[0], spawn_c[1], rest_z])
  goal_b = to_b([target_c[0], target_c[1], top_z + half])
  table_pos = scene.table_pos(top_z)
  table_top_b = top_z - float(base_pos[2])
  tx = (
    table_pos[0] - scene.TABLE_HALF_X - float(base_pos[0]),
    table_pos[0] + scene.TABLE_HALF_X - float(base_pos[0]),
  )
  ty = (
    table_pos[1] - scene.TABLE_HALF_Y - float(base_pos[1]),
    table_pos[1] + scene.TABLE_HALF_Y - float(base_pos[1]),
  )
  base_yaw = float(euler_xyz_from_quat(base_quat.unsqueeze(0))[2][0])

  # TCP site against the arm_left_7_link frame the node's KDL chain ends in.
  mj = env.sim.mj_model
  site_id = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, f"robot/{TCP_SITE}")
  site_body = mj.site_bodyid[site_id]
  site_body_name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, site_body)
  tcp_offset = mj.site_pos[site_id].tolist()
  tcp_quat = mj.site_quat[site_id].tolist()  # wxyz, relative to the site's body
  print("tcp site", site_body_name, tcp_offset, tcp_quat)
  assert site_body_name.split("/")[-1] == "arm_left_7_link", site_body_name

  # ------------------------------------------------------------- pad gap against jaw angle
  wrapped.reset()
  gripper = _gripper_cfg(env)
  jaw_gap = _probe_pad_gap(env, robot, term, jidx, gripper, TCP_SITE, torch, np, quat_apply, quat_inv)

  # --------------------------------------------------------------- MuJoCo reference run
  n = len(episodes)
  steps = int(round(args.ref_seconds / env.step_dt))
  wrapped.reset()
  q0 = robot.data.default_joint_pos.clone()
  robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
  # The spawn poses are given in the env frame (the training's own); the goal by its offset
  # from the table centre, as SurfaceOffsetCommand takes it.
  cube_pose = torch.zeros(n, 7, device=q0.device)
  yaws = torch.tensor([e["yaw"] for e in episodes], device=q0.device)
  for k, e in enumerate(episodes):
    cube_pose[k, 0], cube_pose[k, 1], cube_pose[k, 2] = e["x"], e["y"], rest_z
  zeros = torch.zeros(n, device=q0.device)
  cube_pose[:, 3:] = quat_from_euler_xyz(zeros, zeros, yaws)
  obj.write_root_link_pose_to_sim(cube_pose)
  obj.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=q0.device))
  goal_off = torch.tensor(
    [
      [target_c[0] + e["goal_dx"] - table_pos[0], target_c[1] + e["goal_dy"] - table_pos[1]]
      for e in episodes
    ],
    device=q0.device,
    dtype=torch.float32,
  )
  env.command_manager.get_term(COMMAND).set_offsets(
    torch.arange(n, device=q0.device), goal_off
  )
  env.sim.forward()
  # Empty the delay buffers, so the first steps do not read the pre-overwrite cube.
  env.observation_manager.reset(torch.arange(n, device=q0.device))
  obs_td = wrapped.get_observations()

  site = robot.site_names.index(TCP_SITE)
  keys = (
    "q qd action target tcp_b tcp_quat_b cube_b cube_yaw_b cube_quat_b goal_b grasped "
    "on_surface done obs band_hit range_hit"
  ).split()
  logs: dict[str, list] = {k: [] for k in keys}
  finished = torch.zeros(n, dtype=torch.bool, device=q0.device)
  command = env.command_manager.get_term(COMMAND)
  for _t in range(steps):
    bq = robot.data.body_link_quat_w[:, bi]
    bp = robot.data.body_link_pos_w[:, bi]
    tcp_w = robot.data.site_pos_w[:, site]
    tcp_q_w = robot.data.site_quat_w[:, site]
    cube_w = obj.data.root_link_pos_w
    cube_q_b = quat_mul(quat_inv(bq), obj.data.root_link_quat_w)
    logs["q"].append(robot.data.joint_pos[:, jidx].clone())
    logs["qd"].append(robot.data.joint_vel[:, jidx].clone())
    logs["tcp_b"].append(quat_apply(quat_inv(bq), tcp_w - bp))
    logs["tcp_quat_b"].append(quat_mul(quat_inv(bq), tcp_q_w))
    logs["cube_b"].append(quat_apply(quat_inv(bq), cube_w - bp))
    logs["cube_quat_b"].append(cube_q_b)
    logs["cube_yaw_b"].append(euler_xyz_from_quat(cube_q_b)[2])
    logs["goal_b"].append(quat_apply(quat_inv(bq), command.command - bp))
    logs["grasped"].append(_grasped_bool(env, GRASP_SENSOR).clone())
    logs["on_surface"].append(_on_surface_bool(env, SUPPORT_SENSOR).clone())
    logs["obs"].append(obs_td["actor"].clone())
    with torch.no_grad():
      a = policy(obs_td)
    obs_td, _, dones, _ = wrapped.step(a)
    logs["action"].append(a.clone())
    logs["target"].append(term._processed_actions.clone())
    logs["band_hit"].append(term.band_hit.clone())
    logs["range_hit"].append(term.range_hit.clone())
    finished |= dones.bool()
    logs["done"].append(finished.clone())

  ref = {k: torch.stack(v, dim=1).cpu().numpy() for k, v in logs.items()}
  ref["dt"] = np.array(env.step_dt)
  ref["episode_ids"] = np.array([e["id"] for e in episodes])
  np.savez_compressed(out / "mujoco_reference.npz", **ref)
  calib = _grasp_calibration(ref, quat_apply, quat_inv, torch)
  calib["source"] = (
    f"{ckpt.name} sha256 {sha256(ckpt)[:16]}, {n} episodes x {steps} steps, MuJoCo play cfg "
    "(no obs noise), DR0 scene, robot at exact defaults"
  )
  (out / "grasp_calibration.json").write_text(json.dumps(calib, indent=2) + "\n")

  # ----------------------------------------------------------- the plant, every actuator
  info_plant = _plant_tables(env, robot, mj, mujoco, np)
  all_names = list(robot.joint_names)
  all_q = robot.data.default_joint_pos[0].tolist()
  start = {
    "schema_version": 1,
    "description": "Resolved robot default joint state (training reset pose). Revolute joints "
    "use radians; slide joints use meters.",
    "note": "Training reset randomization may perturb these reference positions.",
    "joint_convention": "ROBOT (PAL URDF) = policy convention: this model has arm_*_5/7 about "
    "+z (pal_mjlab 5e09c30 ported fda9492), so joint_signs are all +1.",
    "joint_names": all_names,
    "joint_positions": {nme: round(float(v), 8) for nme, v in zip(all_names, all_q, strict=True)},
    "joint_velocities": {nme: 0.0 for nme in all_names},
  }
  (out / "deployment_start_config.json").write_text(json.dumps(start, indent=2) + "\n")

  # -------------------------------------------- start_observation_distribution (training env)
  env.close()
  dist_env = ManagerBasedRlEnv(
    cfg=cfg_for(selectors["DR"], play=False, num_envs=args.dist_envs), device=device
  )
  _write_start_observation_distribution(
    dist_env, out / "start_observation_distribution.json", torch
  )
  obs_model = _obs_model(train_cfg)
  dist_env.close()

  # ----------------------------------------------------------------------- ROS profile
  model_in_container = f"{args.container_model_root}/{out.name}/{onnx_path.name}"
  prof = {
    "model_path": model_in_container,
    "source_checkpoint": ckpt.name,
    "source_checkpoint_sha256": sha256(ckpt),
    "selectors": selectors,
    "action_joint_names": action_joints,
    "joint_signs": [1.0] * 8,
    "default_joint_pos": default_q,
    "action_scale": scale,
    "action_clip_lower": clip_lo,
    "action_clip_upper": clip_hi,
    "action_rate_limit": rate,
    # S7-4 C and T, applied after the slew, T first (RateLimitedJointPositionAction._bound).
    "effort_band_half": band_half,  # tau_cmd / K, rad, arm 1-7
    "effort_band_dk": band_dk,  # D / K, s
    "effort_tau_cmd": [h * k for h, k in zip(band_half, band_k, strict=True)],  # N.m
    "range_clip_lower": range_lo,
    "range_clip_upper": range_hi,
    "observation_terms": obs_names,
    "observation_dims": obs_dims,
    # What the `actions` term shows: `act` = the raw last policy output; `cmd` = the target last
    # commanded (after slew, T and C) minus the default pose (CMD_OBS=cmd, S7-8's `-c` codes).
    "actions_obs": "cmd" if cmd_obs else "act",
    "base_link": BASE_BODY,
    "tip_link": "arm_left_7_link",
    "tcp_offset": tcp_offset,
    "tcp_offset_quat_wxyz": tcp_quat,
    "policy_rate": 1.0 / env.step_dt,
    "goal_position_b": goal_b,
    "goal_threshold": GOAL_THRESHOLD,
    "cube_half_size": half,
    "cube_spawn_center_b": spawn_b,
    "table_top_z_b": table_top_b,
    "table_x_b": list(tx),
    "table_y_b": list(ty),
    "obs_frame": "base_link (pal_mjlab 5e09c30 = fda9492 ported): identity",
    "base_in_obs_frame_pos": [0.0, 0.0, 0.0],
    "base_in_obs_frame_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
    "training_base_pose_w": {
      "pos": base_pos.tolist(),
      "quat_wxyz": base_quat.tolist(),
      "yaw": base_yaw,
    },
    "grasp": _virtual_grasp(calib),
    "jaw_gap": jaw_gap,
  }
  _write_yaml(out / "pick_place_profile.yaml", prof)

  # ------------------------------------------------------------------ deployment info
  info = {
    "policy": {
      "name": out.name,
      "checkpoint": ckpt.name,
      "checkpoint_sha256": sha256(ckpt),
      "onnx": onnx_path.name,
      "onnx_sha256": sha256(onnx_path),
      "task": "Mjlab-PickPlace-Pal-Kangaroo-Grippers",
      "selectors": selectors,
      "code": {
        "pal_mjlab_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "pal_mjlab_commit": _git("rev-parse", "HEAD"),
      },
      "note": "S7-8: RCTP (R-C-strict+M1+M2+M3+M6+J+M8 on G20 / CTP / baseline B) at the stack task's "
      f"speeds (SLEW=stack), spawn {selectors['SPAWN']}, target {selectors['TARGET']}, "
      f"actions obs {'commanded target' if cmd_obs else 'raw last action'}, LAG=FP+JS, "
      f"{episode_s:g} s episodes, 5000 iterations, DR-scene, 2048 envs, PPO-stack.",
    },
    "timing": {
      "physics_dt_s": float(env.physics_dt),
      "decimation": int(round(env.step_dt / env.physics_dt)),
      "policy_rate_hz": 1.0 / float(env.step_dt),
      "control_dt_s": CONTROL_DT,
      "episode_length_s": episode_s,
    },
    "joint_convention": {
      "joint_signs": dict(zip(action_joints, [1.0] * 8, strict=True)),
      "note": "identity: arm_left_5/7 already about +z as on PAL's URDF; observations are in "
      "base_link. (Policies from S5.2 / S7-1 are different: see sim2sim-pal-report.)",
    },
    "action": {
      "type": "absolute joint position targets, 8 = arm_left_1..7 + gripper_left_finger",
      "pipeline": [
        "raw = ONNX(obs_38)",
        "target = default + scale * raw; jaw clipped to [0, 0.78] (action_clip)",
        "slew: target = last_target + clamp(target - last_target, +-rate)  (stack speeds: arm 1-3 at 1/16, arm 4-7 at 1/8 of the URDF speed x 20 ms; jaw 0.03 rad/step = 1.5 rad/s)",
        "T (arm only): target = clamp(target, q + (D/K) qdot -+ tau_cmd/K), q, qdot the measured joint state",
        "C (arm only): target = clamp(target, joint_range_lower, joint_range_upper)",
        "last_target = target; send target with qdot_des = 0 and NO gravity feed-forward",
      ],
      "default_joint_pos": dict(zip(action_joints, default_q, strict=True)),
      "scale": dict(zip(action_joints, scale, strict=True)),
      "slew_rate_rad_per_step": dict(zip(action_joints, rate, strict=True)),
      "T_effort_band": {
        j: {"tau_cmd_over_K_rad": h, "D_over_K_s": d, "K": k, "tau_cmd_Nm": h * k}
        for j, h, d, k in zip(ARM, band_half, band_dk, band_k, strict=True)
      },
      "C_range_clip": {j: [lo, hi] for j, lo, hi in zip(ARM, range_lo, range_hi, strict=True)},
      "jaw_clip": [clip_lo[7], clip_hi[7]],
    },
    "observation": {
      "dim": 38,
      "terms": [
        {"name": nme, "dim": d} for nme, d in zip(obs_names, obs_dims, strict=True)
      ],
      "frame": "base_link (PAL's base_link; positions in m, quaternion wxyz with w >= 0)",
      "definitions": {
        "joint_pos": "q - default for arm_left_1..7 and the jaw (8)",
        "joint_vel": "qdot of arm_left_1..7 (7; the jaw's velocity is NOT observed)",
        "actions": (
          "the previous commanded target minus the default pose, rad (8): after the slew, T and C, "
          "as sent; 0 at the start of an episode (CMD_OBS=cmd)"
          if cmd_obs
          else "the previous raw policy output (8)"
        ),
        "tcp_pos": f"site {TCP_SITE} = arm_left_7_link + {tcp_offset} (rot {tcp_quat}), in base_link",
        "tcp_quat": "orientation of that site in base_link, (w,x,y,z), sign fixed to w >= 0",
        "object_pos": "cube centre in base_link (from the tracker in deployment)",
        "object_yaw": "[cos, sin] of the cube's yaw in base_link",
        "target_pos": "goal centre in base_link (cube resting height: table top + 0.025)",
      },
      "training_corruption": obs_model,
    },
    "scene": {
      "table_top_z_env_m": top_z,
      "table_top_z_base_link_m": table_top_b,
      "table_centre_env_xy": list(table_pos[:2]),
      "table_half_xy": [scene.TABLE_HALF_X, scene.TABLE_HALF_Y],
      "table_thickness_m": scene.TABLE_THICKNESS,
      "spawn_box_env": {"x": list(spawn_box.x), "y": list(spawn_box.y), "yaw_half_range_rad": spawn_box.yaw},
      "spawn_centre_base_link": spawn_b,
      "goal_centre_env_xy": list(target_c),
      "goal_half_box_xy_m": baseline.target_half_xy,
      "goal_centre_base_link": goal_b,
      "goal_threshold_m": GOAL_THRESHOLD,
      "cube_edge_m": 2 * half,
      "base_link_pose_env": {"pos": base_pos.tolist(), "quat_wxyz": base_quat.tolist(), "yaw_rad": base_yaw},
      "note": "env frame = MuJoCo world of the training scene; base_link is 0.9 m above the fixed-base mocap anchor.",
    },
    "gripper": jaw_gap,
    "plant": info_plant,
    "start_pose": {
      "left_arm_and_jaw": dict(zip(action_joints, default_q, strict=True)),
      "all_joints": start["joint_positions"],
    },
    "pal_side": {
      "controller": "subscriber_controller / direct_impedance_control: tau = K (q_des - q) + D (qd_des - qd) + tau_ff",
      "gains_to_load": "params/env.yaml actuator stiffness / damping (spawn_grasp_controller.py); G20 = impedance_control_v37 for arm 1-4, arm 5-7 need files",
      "gravity_compensation": "OFF (G20 was trained without it): no pal_gravity_compensation, tau_ff = 0, qd_des = 0",
      "jaw": "DirectPositionControl on gripper_left_finger_joint (training kp 8 / kd 0.5 is the sim actuator, not what PAL runs); PAL's jaw settles +0.0387 rad from its command in the simulator",
    },
  }
  (out / "deployment_info.yaml").write_text(
    "# Generated by scripts/sim2sim/export_for_pal_s78.py; see its docstring.\n"
    + _dump(info)
  )
  print(_dump({k: info[k] for k in ("policy", "timing", "action")})[:6000])
  print("grasp calibration:", json.dumps(calib, indent=1)[:2500])
  print("wrote", out)


def _diff_dumps(run_dump: Path, regenerated: Path) -> None:
  import difflib

  a = [ln for ln in run_dump.read_text().splitlines() if "num_envs" not in ln and "seed" not in ln]
  b = [ln for ln in regenerated.read_text().splitlines() if "num_envs" not in ln and "seed" not in ln]
  diff = [
    ln
    for ln in difflib.unified_diff(a, b, "run env.yaml", "regenerated", lineterm="", n=0)
    if not ln.startswith(("---", "+++", "@@"))
  ]
  print(f"env.yaml (run) vs regenerated from the selectors: {len(diff)} differing lines")
  for ln in diff[:60]:
    print("   ", ln)


def _dump(data) -> str:
  import yaml

  return yaml.safe_dump(data, sort_keys=False, default_flow_style=None, width=110)


def _obs_model(cfg) -> dict:
  """The actor's training-time corruption per term: noise, bias, latency (steps of 20 ms)."""
  res = {}
  group = cfg.observations["actor"]
  for name, t in group.terms.items():
    entry = {}
    noise = getattr(t, "noise", None)
    if noise is not None:
      entry["noise"] = repr(noise)
    entry["delay_lag_steps"] = [
      getattr(t, "delay_min_lag", 0),
      getattr(t, "delay_max_lag", 0),
    ]
    res[name] = entry
  res["enable_corruption"] = bool(group.enable_corruption)
  return res


def _plant_tables(env, robot, mj, mujoco, np) -> dict:
  """Every actuator's gains and limits, every joint's default and range, as compiled."""
  acts = {}
  for i in range(mj.nu):
    name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"actuator{i}"
    jid = int(mj.actuator_trnid[i, 0])
    jname = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"joint{jid}"
    dof = int(mj.jnt_dofadr[jid])
    acts[jname.split("/")[-1]] = {
      "actuator": name.split("/")[-1],
      "kp": float(mj.actuator_gainprm[i, 0]),
      "kd": float(-mj.actuator_biasprm[i, 2]),
      "bias_kp_check": float(-mj.actuator_biasprm[i, 1]),
      "force_range_Nm": [float(x) for x in mj.actuator_forcerange[i]] if mj.actuator_forcelimited[i] else None,
      "joint_actfrc_range_Nm": [float(x) for x in mj.jnt_actfrcrange[jid]] if mj.jnt_actfrclimited[jid] else None,
      "armature": float(mj.dof_armature[dof]),
      "joint_damping": float(mj.dof_damping[dof]),
      "frictionloss": float(mj.dof_frictionloss[dof]),
      "joint_range": [float(x) for x in mj.jnt_range[jid]] if mj.jnt_limited[jid] else None,
    }
  joints = {}
  dq = robot.data.default_joint_pos[0].tolist()
  for nme, q in zip(robot.joint_names, dq, strict=True):
    joints[nme] = {"default": round(float(q), 8)}
  gravcomp_bodies = {
    mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, b).split("/")[-1]: float(mj.body_gravcomp[b])
    for b in range(mj.nbody)
    if mj.body_gravcomp[b] != 0
  }
  return {
    "actuators": acts,
    "joints_default": joints,
    "bodies_with_gravity_compensation": gravcomp_bodies or "none (G20: gravity compensation off)",
    "note": "kp/kd are the position actuator's gain / -bias[2]; the action joints' effort limit used "
    "by T is joint_actfrc_range (0.7 x effort on arm_left_1-7). Legs, pelvis and the right arm are "
    "held at their default pose by their own actuators in training (fixed base); on PAL the "
    "simulator's / the robot's own controllers hold them.",
  }


def _rows(obs_td, xs):
  for i in range(xs.shape[0]):
    td = obs_td[:1].clone()
    td["actor"] = xs[i : i + 1]
    if "critic" in td.keys():
      td["critic"] = td["critic"].new_zeros(td["critic"].shape)
    yield td


def _gripper_cfg(env):
  """The left gripper's pad sites, from the task's robot description."""
  from pal_mjlab.tasks.manipulation.robots.kangaroo import KANGAROO_GRIPPERS_LEFT_ARM

  return KANGAROO_GRIPPERS_LEFT_ARM.gripper("left")


def _probe_pad_gap(env, robot, term, jidx, gripper, tcp_site, torch, np, quat_apply, quat_inv) -> dict:
  """The pads' distance against the jaw angle, measured on the compiled model.

  The jaw is stepped through its travel with the arm held at its default pose; at each angle
  the physics runs 0.16 s and the fingertip pad sites' distance is read. The site distance
  is not the gap between the pad *surfaces*: it is anchored to the 79.3 mm the shipped
  meshes have at q = 0 (`surface_gap = site_distance - site_distance(0) + 0.0793`), which
  is what the virtual cube's width test needs. `closing_axis_tcp` is the pad-to-pad
  direction in the TCP frame.
  """
  pads = [robot.site_names.index(sname) for sname in gripper.pad_sites]
  tcp = robot.site_names.index(tcp_site)
  default = robot.data.default_joint_pos.clone()
  target = default[:, jidx].clone()
  qs, dists, meas = [], [], []
  axis = None
  for q_jaw in np.linspace(0.0, 0.78, 40):
    target[:, 7] = float(q_jaw)
    for _ in range(80):
      robot.set_joint_position_target(target, joint_ids=term._target_ids)
      env.scene.write_data_to_sim()
      env.sim.step()
      env.scene.update(dt=env.physics_dt)
    p = robot.data.site_pos_w
    d = (p[:, pads[0]] - p[:, pads[1]]).norm(dim=-1).mean().item()
    if axis is None:
      v = quat_apply(quat_inv(robot.data.site_quat_w[:, tcp]), p[:, pads[0]] - p[:, pads[1]])
      axis = torch.nn.functional.normalize(v.mean(dim=0), dim=0).tolist()
    qs.append(float(q_jaw))
    meas.append(robot.data.joint_pos[:, jidx[7]].mean().item())
    dists.append(d)
  surface = [d - dists[0] + 0.0793 for d in dists]
  return {
    "q_target": qs,
    "q_measured": meas,
    "pad_site_distance": dists,
    "surface_gap": surface,
    "closing_axis_tcp": axis,
    "open_gap_m": 0.0793,
    "margin_rad": 0.015,
    "release_margin_rad": 0.03,
    "note": "surface_gap = pad site distance - its value at q=0 + 0.0793 m (pads 79.3 mm apart open, "
    "measured on the shipped meshes); the arm is held at its default pose, 0.16 s per angle.",
  }


def _virtual_grasp(calib: dict) -> dict:
  onset = calib["cube_in_tcp_at_onset"]
  zs = [o[2] for o in onset]
  ys = [abs(o[1]) for o in onset]
  return {
    "jaw_contact": calib["jaw_contact"],
    "jaw_hold_median": calib["jaw_hold_median"],
    "jaw_release": calib["jaw_contact"] - 0.005,
    "jaw_hold_command": calib["jaw_hold_median"],
    "x_tol": 0.0145,
    "y_tol": max(ys) + 0.01,
    "z_lo": min(zs) - 0.01,
    "z_hi": max(zs) + 0.01,
    "cube_in_tcp_median": calib["cube_in_tcp_median"],
  }


def _grasp_calibration(ref, quat_apply, quat_inv, torch) -> dict:
  import numpy as np

  g = ref["grasped"].astype(bool)
  jaw = ref["q"][..., 7]
  tcp = torch.as_tensor(ref["tcp_b"])
  tq = torch.as_tensor(ref["tcp_quat_b"])
  cube = torch.as_tensor(ref["cube_b"])
  rel = quat_apply(quat_inv(tq.reshape(-1, 4)), (cube - tcp).reshape(-1, 3)).reshape(
    cube.shape
  )
  rel = rel.numpy()
  held_rel = rel[g]
  onset, release, held_jaw, onset_rel = [], [], jaw[g], []
  for k in range(g.shape[0]):
    idx = np.flatnonzero(g[k])
    if idx.size == 0:
      continue
    onset.append(jaw[k, idx[0]])
    onset_rel.append(rel[k, idx[0]])
    after = np.flatnonzero(~g[k, idx[0] :])
    if after.size:
      release.append(jaw[k, idx[0] + after[0]])
  onset_rel = np.array(onset_rel) if onset_rel else np.zeros((0, 3))
  pad = 0.006
  return {
    "episodes_grasped": int(len(onset)),
    "held_steps": int(g.sum()),
    "jaw_contact": float(np.min(onset)) if onset else None,
    "jaw_onset_values": [float(v) for v in onset],
    "jaw_hold_median": float(np.median(held_jaw)) if held_jaw.size else None,
    "jaw_hold_p05_p95": [float(np.percentile(held_jaw, 5)), float(np.percentile(held_jaw, 95))]
    if held_jaw.size
    else None,
    "jaw_release": float(np.max(release)) if release else None,
    "jaw_release_values": [float(v) for v in release],
    "cube_in_tcp_median": np.median(held_rel, axis=0).tolist() if held_rel.size else None,
    "cube_in_tcp_min": held_rel.min(axis=0).tolist() if held_rel.size else None,
    "cube_in_tcp_max": held_rel.max(axis=0).tolist() if held_rel.size else None,
    "cube_in_tcp_at_onset": onset_rel.tolist(),
    "x_tol": float(np.abs(onset_rel[:, 0]).max() + pad) if len(onset_rel) else 0.015,
    "y_tol": float(np.abs(onset_rel[:, 1]).max() + pad) if len(onset_rel) else 0.02,
    "z_lo": float(onset_rel[:, 2].min() - pad) if len(onset_rel) else -0.01,
    "z_hi": float(onset_rel[:, 2].max() + pad) if len(onset_rel) else 0.04,
  }


def _write_start_observation_distribution(env, path: Path, torch) -> None:
  manager = env.observation_manager
  observations, _ = env.reset()
  actor_obs = observations["actor"]
  terms = {}
  offset = 0
  names = manager._group_obs_term_names["actor"]
  dims = manager._group_obs_term_dim["actor"]
  for name, shape in zip(names, dims, strict=True):
    width = int(torch.tensor(shape).prod().item())
    value = actor_obs[:, offset : offset + width]
    offset += width
    flat = value.detach().reshape(env.num_envs, -1).float()
    summary = torch.stack(
      (
        flat.min(dim=0).values,
        torch.quantile(flat, 0.01, dim=0),
        flat.mean(dim=0),
        flat.std(dim=0, unbiased=False),
        torch.quantile(flat, 0.99, dim=0),
        flat.max(dim=0).values,
      ),
      dim=0,
    ).cpu()
    terms[name] = {
      k: summary[i].tolist() for i, k in enumerate(("min", "p01", "mean", "std", "p99", "max"))
    }
  payload = {
    "schema_version": 1,
    "description": "Per-feature actor observation statistics over the initially reset "
    "training environments (policy convention = robot convention here).",
    "num_initial_envs": env.num_envs,
    "terms": terms,
  }
  path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_yaml(path: Path, data: dict) -> None:
  path.write_text(
    "# Generated by scripts/sim2sim/export_for_pal_s78.py. Policy convention = robot convention\n"
    "# (joint_signs all +1); everything in base_link.\n" + _dump(data)
  )


if __name__ == "__main__":
  main()
