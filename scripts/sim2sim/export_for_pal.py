"""Export one S5.2 pick-and-place checkpoint for the PAL deployment stack, plus a matched
MuJoCo reference rollout.

The model folder follows Claudius's layout in `pal-policy-deployment` (ONNX beside
`deployment_start_config.json`, `start_observation_distribution.json` and `params/`), so his
`spawn_grasp_controller.py` (gains from `params/env.yaml`) and `move_to_start_config.py`
work on it unchanged. Two things are added for the joint-position pick-place policy:

* `pick_place_profile.yaml` -- the parameters `pal_policy_inferer_pick_place_node` needs:
  action contract, observation contract, the virtual cube and the goal, all in the base
  frame.
* `mujoco_reference.npz` / `grasp_calibration.json` -- the same policy rolled out in the
  training simulator from the same episode specs, and the grasp geometry measured there,
  which parameterises the virtual cube.

**Joint convention.** The checkpoints were trained on the pre-`fda9492` MJCF, in which
`arm_*_5_joint` and `arm_*_7_joint` turn about -z. PAL's URDF (and the robot) turn them
about +z; everything else in base_link -> arm_left_7_link is identical (FK over 500 random
configurations: 0.001 mm / 0.0001 deg with the two signs flipped, 148 mm / 180 deg
without). So q_policy = s * q_robot with s = -1 on those joints. Files the robot side reads
(`deployment_start_config.json`) are written in the ROBOT convention; everything the policy
sees (observation statistics, defaults, scales) is in the POLICY convention, and the node
applies `joint_signs` at the boundary.

  REWARD/DR/PPO are read off the run directory name, as training set them.

  .venv/bin/python scripts/sim2sim/export_for_pal.py \
    --checkpoint artifacts/sim2sim/ckpt/<run>/model_8000.pt --out artifacts/sim2sim/models/<name>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path

TASK = "Mjlab-PickPlace-Pal-Kangaroo-Grippers"
RUN_RE = re.compile(
  r"_pp_[^_]+_(?P<reward>R-[^_]+)_(?P<dr>DR[^_]*)_(?P<ppo>PPO-[^_]+)_n\d+_seed(?P<seed>\d+)$"
)
#: pal_mjlab fda9492 ("fix kang arm kinematics!"): these axes are -z in the training MJCF.
FLIPPED_JOINTS = (
  "arm_left_5_joint",
  "arm_left_7_joint",
  "arm_right_5_joint",
  "arm_right_7_joint",
)
TCP_SITE = "tcp_left"
BASE_BODY = "base_link"
GRASP_SENSOR = "left_pad_contact"
SUPPORT_SENSOR = "object_table_contact"


def sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--checkpoint", type=Path, required=True)
  p.add_argument("--out", type=Path, required=True)
  p.add_argument(
    "--episodes", type=Path, default=Path(__file__).with_name("episodes.json")
  )
  p.add_argument("--ref-seconds", type=float, default=8.0)
  p.add_argument("--dist-envs", type=int, default=256)
  p.add_argument(
    "--container-model-root",
    default="/catkin_ws/src/models",
    help="where the model folder is mounted inside the PAL container",
  )
  return p.parse_args()


def main() -> None:
  args = parse_args()
  ckpt = args.checkpoint.resolve()
  run_dir = ckpt.parent
  m = RUN_RE.search(run_dir.name)
  if m is None:
    raise SystemExit(f"cannot parse REWARD/DR/PPO from {run_dir.name}")
  os.environ["REWARD"], os.environ["DR"], os.environ["PPO"] = (
    m["reward"],
    m["dr"],
    m["ppo"],
  )
  iteration = int(re.search(r"model_(\d+)\.pt$", ckpt.name).group(1))

  # Imports after the environment is set: the task registers its cfg at import (D10).
  from dataclasses import asdict

  import numpy as np
  import pal_mjlab.tasks  # noqa: F401
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.utils.lab_api.math import (
    euler_xyz_from_quat,
    quat_apply,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
  )
  from pal_mjlab.tasks.manipulation import scene
  from pal_mjlab.tasks.manipulation.mdp.observations import (
    _grasped_bool,
    _on_surface_bool,
  )
  from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import (
    GOAL_THRESHOLD,
    SPAWN_XY,
    TARGET_XY,
    TARGET_Z,
  )

  out = args.out.resolve()
  (out / "params").mkdir(parents=True, exist_ok=True)
  spec = json.loads(args.episodes.read_text())
  episodes = spec["episodes"]
  assert tuple(spec["spawn_center_env_xy"]) == tuple(SPAWN_XY), "episodes.json centre"

  # --------------------------------------------------------------------- env + policy
  env_cfg = load_env_cfg(TASK, play=True)
  env_cfg.episode_length_s = load_env_cfg(TASK, play=False).episode_length_s
  env_cfg.scene.num_envs = len(episodes)
  env_cfg.seed = 0
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
  wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
  agent_cfg = load_rl_cfg(TASK)
  runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device="cpu")
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location="cpu")
  policy = runner.get_inference_policy(device="cpu")

  robot = env.scene["robot"]
  obj = env.scene["object"]
  term = env.action_manager.get_term("joint_pos")
  action_joints = list(term.target_names)
  assert action_joints == [f"arm_left_{i}_joint" for i in range(1, 8)] + [
    "gripper_left_finger_joint"
  ], action_joints
  jidx = [robot.joint_names.index(n) for n in action_joints]
  signs = [(-1.0 if n in FLIPPED_JOINTS else 1.0) for n in action_joints]
  default_q = robot.data.default_joint_pos[0, jidx].tolist()
  scale = term.scale[0].tolist() if torch.is_tensor(term.scale) else [term.scale] * 8
  offset = (
    term.offset[0].tolist() if torch.is_tensor(term.offset) else [term.offset] * 8
  )
  assert np.allclose(offset, default_q), (offset, default_q)
  clip = term._clip[0].tolist() if getattr(term, "_clip", None) is not None else None
  rate = term._rate[0].tolist()
  obs_names = env.observation_manager.active_terms["actor"]
  obs_dims = [
    int(np.prod(d)) for d in env.observation_manager._group_obs_term_dim["actor"]
  ]

  # ----------------------------------------------------------------------- ONNX export
  onnx_path = out / f"model_{iteration}.onnx"
  runner.export_policy_to_onnx(str(out), onnx_path.name)
  metadata = get_base_metadata(env, run_dir.name)
  metadata["action_joint_names"] = action_joints
  metadata["joint_convention"] = (
    "training MJCF (pre-fda9492): arm_*_5/7 are negated relative to PAL's URDF"
  )
  attach_metadata_to_onnx(str(onnx_path), metadata)

  # ONNX == torch is checked outside this venv (no onnxruntime here): save the inputs
  # and the torch outputs; `check_onnx.py` runs them through onnxruntime.
  obs_td = wrapped.get_observations()
  x = obs_td["actor"].float()
  noisy = x.repeat(64 // x.shape[0] + 1, 1)[:64] + 0.3 * torch.randn(64, x.shape[1])
  xs = torch.cat([x, noisy], dim=0)
  with torch.no_grad():
    ref_a = torch.cat([policy(row) for row in _rows(obs_td, xs)], dim=0)
  np.savez(out / "onnx_check.npz", obs=xs.numpy(), actions=ref_a.numpy())

  for f in ("env.yaml", "agent.yaml"):
    shutil.copy2(run_dir / "params" / f, out / "params" / f)

  # ------------------------------------------------------------- geometry, base frame
  bi = robot.body_names.index(BASE_BODY)
  base_pos = robot.data.body_link_pos_w[0, bi].clone()
  base_quat = robot.data.body_link_quat_w[0, bi].clone()
  # The policy's observations are NOT in base_link: `mdp.observations._to_base_frame` uses
  # `root_link_pos_w/quat_w`, and with `fixed_base` the entity root is the mocap anchor,
  # 0.9 m below base_link (pal_mjlab fda9492 switched to base_link; these checkpoints
  # predate it). Record base_link's pose in that frame; the node applies it.
  root_pos = robot.data.root_link_pos_w[0].clone()
  root_quat = robot.data.root_link_quat_w[0].clone()
  base_in_obs_pos = quat_apply(quat_inv(root_quat), base_pos - root_pos).tolist()
  base_in_obs_quat = quat_mul(quat_inv(root_quat), base_quat).tolist()
  print("obs frame = root link; base_link in it:", base_in_obs_pos, base_in_obs_quat)

  def to_b(p_w):
    p = torch.as_tensor(p_w, dtype=torch.float32)
    return quat_apply(quat_inv(base_quat), p - base_pos).tolist()

  half = 0.025
  rest_z = scene.TABLE_TOP_Z + half
  spawn_b = to_b([SPAWN_XY[0], SPAWN_XY[1], rest_z])
  goal_b = to_b([TARGET_XY[0], TARGET_XY[1], TARGET_Z])
  table_top_b = scene.TABLE_TOP_Z - float(base_pos[2])
  tx = (
    scene.TABLE_POS[0] - scene.TABLE_HALF_X - float(base_pos[0]),
    scene.TABLE_POS[0] + scene.TABLE_HALF_X - float(base_pos[0]),
  )
  ty = (
    scene.TABLE_POS[1] - scene.TABLE_HALF_Y - float(base_pos[1]),
    scene.TABLE_POS[1] + scene.TABLE_HALF_Y - float(base_pos[1]),
  )
  base_yaw_err = float(euler_xyz_from_quat(base_quat.unsqueeze(0))[2][0])

  # ------------------------------------------------------------ MuJoCo reference run
  n = len(episodes)
  steps = int(round(args.ref_seconds / env.step_dt))
  wrapped.reset()
  q0 = robot.data.default_joint_pos.clone()
  robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
  cube_pose = torch.zeros(n, 7)
  for k, e in enumerate(episodes):
    cube_pose[k, 0] = SPAWN_XY[0] + e["dx"]
    cube_pose[k, 1] = SPAWN_XY[1] + e["dy"]
    cube_pose[k, 2] = rest_z
  zeros = torch.zeros(n)
  cube_pose[:, 3:] = quat_from_euler_xyz(
    zeros, zeros, torch.tensor([e["dyaw"] for e in episodes])
  )
  obj.write_root_link_pose_to_sim(cube_pose)
  obj.write_root_link_velocity_to_sim(torch.zeros(n, 6))
  env.sim.forward()
  obs_td = wrapped.get_observations()

  site = robot.site_names.index(TCP_SITE)
  logs: dict[str, list] = {
    k: []
    for k in (
      "q",
      "qd",
      "action",
      "target",
      "tcp_b",
      "tcp_quat_b",
      "cube_b",
      "cube_yaw_b",
      "cube_quat_b",
      "grasped",
      "on_surface",
      "done",
      "obs",
    )
  }
  finished = torch.zeros(n, dtype=torch.bool)
  for _t in range(steps):
    bq = robot.data.body_link_quat_w[:, bi]
    bp = robot.data.body_link_pos_w[:, bi]
    tcp_w = robot.data.site_pos_w[:, site]
    tcp_q_w = robot.data.site_quat_w[:, site]
    cube_w = obj.data.root_link_pos_w
    cube_q_w = obj.data.root_link_quat_w
    cube_q_b = quat_mul(quat_inv(bq), cube_q_w)
    logs["q"].append(robot.data.joint_pos[:, jidx].clone())
    logs["qd"].append(robot.data.joint_vel[:, jidx].clone())
    logs["tcp_b"].append(quat_apply(quat_inv(bq), tcp_w - bp))
    logs["tcp_quat_b"].append(quat_mul(quat_inv(bq), tcp_q_w))
    logs["cube_b"].append(quat_apply(quat_inv(bq), cube_w - bp))
    logs["cube_quat_b"].append(cube_q_b)
    logs["cube_yaw_b"].append(euler_xyz_from_quat(cube_q_b)[2])
    logs["grasped"].append(_grasped_bool(env, GRASP_SENSOR).clone())
    logs["on_surface"].append(_on_surface_bool(env, SUPPORT_SENSOR).clone())
    logs["obs"].append(obs_td["actor"].clone())
    with torch.no_grad():
      a = policy(obs_td)
    obs_td, _, dones, _ = wrapped.step(a)
    logs["action"].append(a.clone())
    logs["target"].append(term._processed_actions.clone())
    finished |= dones.bool()
    logs["done"].append(finished.clone())

  ref = {k: torch.stack(v, dim=1).numpy() for k, v in logs.items()}
  ref["dt"] = np.array(env.step_dt)
  ref["episode_ids"] = np.array([e["id"] for e in episodes])
  np.savez_compressed(out / "mujoco_reference.npz", **ref)

  calib = _grasp_calibration(ref, quat_apply, quat_inv, torch)
  calib.update(
    {
      "source": f"{run_dir.name}/model_{iteration}.pt, {n} episodes x {steps} steps, "
      "MuJoCo play cfg (no obs noise), robot at exact defaults",
    }
  )
  (out / "grasp_calibration.json").write_text(json.dumps(calib, indent=2) + "\n")

  # ------------------------------------------------ deployment_start_config (ROBOT conv.)
  all_names = list(robot.joint_names)
  all_q = robot.data.default_joint_pos[0].tolist()
  all_qd = robot.data.default_joint_vel[0].tolist()
  start = {
    "schema_version": 1,
    "description": "Resolved robot default joint state. Revolute joints use radians; "
    "slide joints use meters.",
    "note": "Training reset randomization may perturb these reference positions.",
    "joint_convention": "ROBOT (PAL URDF). arm_*_5_joint and arm_*_7_joint are negated "
    "relative to the training MJCF (pal_mjlab fda9492); the policy node applies "
    "joint_signs.",
    "joint_names": all_names,
    "joint_positions": {
      nme: round((-1.0 if nme in FLIPPED_JOINTS else 1.0) * float(v), 8)
      for nme, v in zip(all_names, all_q, strict=True)
    },
    "joint_velocities": {
      nme: round(float(v), 8) for nme, v in zip(all_names, all_qd, strict=True)
    },
  }
  (out / "deployment_start_config.json").write_text(json.dumps(start, indent=2) + "\n")

  # ------------------------------------- start_observation_distribution (POLICY conv.)
  env.close()
  dist_cfg = load_env_cfg(TASK, play=False)
  dist_cfg.scene.num_envs = args.dist_envs
  dist_env = ManagerBasedRlEnv(cfg=dist_cfg, device="cpu")
  _write_start_observation_distribution(
    dist_env, out / "start_observation_distribution.json", torch
  )
  dist_env.close()

  # -------------------------------------------------------------- ROS profile (YAML)
  big = 1.0e9
  lo = [c[0] if clip else -big for c in (clip or [None] * 8)]
  hi = [c[1] if clip else big for c in (clip or [None] * 8)]
  lo = [(-big if not math.isfinite(v) else v) for v in lo]
  hi = [(big if not math.isfinite(v) else v) for v in hi]
  rate = [(big if not math.isfinite(v) else v) for v in rate]
  model_in_container = f"{args.container_model_root}/{out.name}/{onnx_path.name}"
  prof = {
    "model_path": model_in_container,
    "source_checkpoint": f"{run_dir.name}/model_{iteration}.pt",
    "source_checkpoint_sha256": sha256(ckpt),
    "reward": m["reward"],
    "seed": int(m["seed"]),
    "action_joint_names": action_joints,
    "joint_signs": signs,
    "default_joint_pos": default_q,
    "action_scale": scale,
    "action_clip_lower": lo,
    "action_clip_upper": hi,
    "action_rate_limit": rate,
    "observation_terms": list(obs_names),
    "observation_dims": obs_dims,
    "base_link": BASE_BODY,
    "tip_link": "arm_left_7_link",
    "tcp_offset": [0.0, 0.0, 0.148],
    "policy_rate": 1.0 / env.step_dt,
    "goal_position_b": goal_b,
    "goal_threshold": GOAL_THRESHOLD,
    "cube_half_size": half,
    "cube_spawn_center_b": spawn_b,
    "table_top_z_b": table_top_b,
    "table_x_b": list(tx),
    "table_y_b": list(ty),
    "obs_frame": "root_link (fixed-base mocap anchor), not base_link",
    "base_in_obs_frame_pos": base_in_obs_pos,
    "base_in_obs_frame_quat_wxyz": base_in_obs_quat,
    "training_base_pose_w": {
      "pos": base_pos.tolist(),
      "quat_wxyz": base_quat.tolist(),
      "yaw": base_yaw_err,
    },
    "grasp": _virtual_grasp(calib),
  }
  _write_yaml(out / "pick_place_profile.yaml", prof)
  print(json.dumps({k: prof[k] for k in prof if k != "grasp"}, indent=1)[:4000])
  print("grasp calibration:", json.dumps(calib, indent=1))
  print("wrote", out)


def _rows(obs_td, xs):
  """One TensorDict per row of `xs`, with every group replaced by that row."""
  for i in range(xs.shape[0]):
    td = obs_td[:1].clone()
    td["actor"] = xs[i : i + 1]
    if "critic" in td.keys():
      td["critic"] = xs[i : i + 1]
    yield td


def _virtual_grasp(calib: dict) -> dict:
  """Virtual-cube grasp rule for the PAL node, from the MuJoCo calibration.

  * Onset: the jaw closes past the smallest angle at which MuJoCo registered a grasp, with
    the cube inside the capture envelope (TCP frame).
  * Envelope: x (closing axis) is the physical capture range, (79.3 mm open gap - 50 mm
    cube) / 2 = 14.5 mm, not the 1-3 mm MuJoCo onsets show -- those are after the closing
    pads have already centred the cube. y and z are the onsets seen in MuJoCo widened by 1 cm.
  * Release: the pads leave the cube once the jaw opens past the angle at which they first
    touched it, so the measured jaw falling 5 mrad below `jaw_contact` ends the grasp. (A
    first version used the holding range minus 0.01 rad, which sits *above* the contact angle
    on seeds 2-3; PAL's jaw, unopposed by a real cube, rests at its command and the policy
    holds it near 0.23 rad, so the virtual cube was dropped 1-3 steps after the grasp.)
  * While held the jaw command is capped at `jaw_hold_command` so the jaw stops where a
    cube would stop it; the PAL gripper's steady-state offset is calibrated in PAL, not here.
  """
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
  """Where the cube sits in the TCP frame while held, and the jaw angles that start and end
  a grasp, measured in MuJoCo. Parameterises the virtual cube on the PAL side."""
  import numpy as np

  g = ref["grasped"].astype(bool)  # (n, T)
  jaw = ref["q"][..., 7]
  tcp = torch.as_tensor(ref["tcp_b"])
  tq = torch.as_tensor(ref["tcp_quat_b"])
  cube = torch.as_tensor(ref["cube_b"])
  rel = quat_apply(quat_inv(tq.reshape(-1, 4)), (cube - tcp).reshape(-1, 3)).reshape(
    cube.shape
  )
  rel = rel.numpy()
  held_rel = rel[g]
  onset, release, held_jaw = [], [], jaw[g]
  onset_rel = []
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
    "jaw_hold_p05_p95": [
      float(np.percentile(held_jaw, 5)),
      float(np.percentile(held_jaw, 95)),
    ]
    if held_jaw.size
    else None,
    "jaw_release": float(np.max(release)) if release else None,
    "jaw_release_values": [float(v) for v in release],
    "cube_in_tcp_median": np.median(held_rel, axis=0).tolist()
    if held_rel.size
    else None,
    "cube_in_tcp_min": held_rel.min(axis=0).tolist() if held_rel.size else None,
    "cube_in_tcp_max": held_rel.max(axis=0).tolist() if held_rel.size else None,
    "cube_in_tcp_at_onset": onset_rel.tolist(),
    # Envelope for starting a virtual grasp: the onset positions seen in MuJoCo, widened
    # by `pad`.
    "x_tol": float(np.abs(onset_rel[:, 0]).max() + pad) if len(onset_rel) else 0.015,
    "y_tol": float(np.abs(onset_rel[:, 1]).max() + pad) if len(onset_rel) else 0.02,
    "z_lo": float(onset_rel[:, 2].min() - pad) if len(onset_rel) else -0.01,
    "z_hi": float(onset_rel[:, 2].max() + pad) if len(onset_rel) else 0.04,
  }


def _write_start_observation_distribution(env, path: Path, torch) -> None:
  """Claudius's `write_start_observation_distribution` (pal_mjlab origin/faster-moves,
  src/pal_mjlab/tasks/manipulation/deployment.py), which this branch does not carry."""
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
      k: summary[i].tolist()
      for i, k in enumerate(("min", "p01", "mean", "std", "p99", "max"))
    }
  payload = {
    "schema_version": 1,
    "description": "Per-feature actor observation statistics over the initially reset "
    "training environments (POLICY joint convention).",
    "num_initial_envs": env.num_envs,
    "terms": terms,
  }
  path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_yaml(path: Path, data: dict) -> None:
  import yaml

  path.write_text(
    "# Generated by scripts/sim2sim/export_for_pal.py. POLICY joint convention\n"
    "# unless noted; joint_signs maps robot -> policy (q_policy = s * q_robot).\n"
    + yaml.safe_dump(data, sort_keys=False, default_flow_style=None, width=100)
  )


if __name__ == "__main__":
  main()
