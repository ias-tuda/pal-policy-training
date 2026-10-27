"""Export plan §13.1's SEQ-H pair (L-H on the left, R-I on the right through the mirror map) for the PAL sim2sim stack,
and roll the chain out in MuJoCo on fixed episodes as the reference (`export_s11_seq.py`'s recipe, for §13).

Everything comes from the chain env `scripts/eval_chain_insert.py` scores (``ARMS=both S13=CHAIN``, BS12's plant), here in
play mode, DR0, no observation noise:

* `left/model.onnx` (40 -> 8), `right/model.onnx` (42 -> 8): each actor (`eval_hold.MlpActor`: the checkpoint's normaliser
  and MLP), torch's legacy exporter, checked against the torch actor on the reference's own observations;
* `insert_profile.yaml`: what `pal_policy_inferer_insert_chain_node` reads -- per arm the action pipeline in the *policy*
  (left-arm) convention (default, scale, clip, slew, S7-4's T band and C clip) read off the two-arm action term (the left at
  L-H's side-grasp scales, the right at R-I's WC scales, mapped through the mirror signs); the TCP site; the observation
  layout; the virtual grasp per arm, calibrated on the reference's own grasps (the block's centre and the pin's centre in the
  TCP frame at the first grasp, the jaw angles); the parts (block 56 mm with its hole and chamfer, the pin's cross), the
  table, H and the presentation rule; 30 s episodes;
* `episodes.json`: the reference's own spawns (the block on the left's area, the pin on the right's), H, in base_link;
* `mujoco_reference.npz` + `chain_reference.json`: per step both arms' joints, targets, observations and actions, the block
  and the pin (position, quaternion), both TCPs and grasps, the right's start; per episode `eval_chain_insert.Reads`' row;
* `deployment_start_config.json` (every joint's default; the right at the mirrored home), `gains.json`, `export_info.json`.

  python scripts/sim2sim/export_s13_chain.py --lh L-H/model_5999.pt --ri R-I/model_5999.pt --out <models>/pp_s13_chain_... \\
    --episodes 24 --jaw-gap-from <a DEP1 pick_place_profile.yaml>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import sys

#: The chain env, DR0 for the reference (BS12's plant otherwise; `eval_chain_insert.py`'s).
CHAIN = {
  "ARMS": "both",
  "S13": "CHAIN",
  "GAINS": "G20",
  "EFFORT": "CTP",
  "DR": "DR0",
  "LAG": "FP+JS",
  "ARM_TABLE": "AH10",
  "TABLE_DZ": "22",
}
for _k, _v in CHAIN.items():
  os.environ[_k] = _v

TASK = "Mjlab-PickPlace-Pal-Kangaroo-Grippers"
LH_TERMS = (
  "joint_pos",
  "joint_vel",
  "actions",
  "tcp_pos",
  "tcp_quat",
  "object_pos",
  "object_yaw",
  "target_pos",
  "object_up",
)
LH_DIMS = (8, 7, 8, 3, 4, 3, 2, 3, 2)
RI_TERMS = (
  "joint_pos",
  "joint_vel",
  "actions",
  "tcp_pos",
  "tcp_quat",
  "object_pos",
  "object_yaw",
  "target_pos",
  "box_yaw",
  "pin_tilt",
)
RI_DIMS = (8, 7, 8, 3, 4, 3, 2, 3, 2, 2)


def sha256(path: pathlib.Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args():
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--lh", type=pathlib.Path, required=True)
  p.add_argument("--ri", type=pathlib.Path, required=True)
  p.add_argument("--out", type=pathlib.Path, required=True)
  p.add_argument("--episodes", type=int, default=24)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--episode-s", type=float, default=30.0)
  p.add_argument(
    "--jaw-gap-from",
    type=pathlib.Path,
    required=True,
    help="a DEP1 pick_place_profile.yaml (its jaw_gap block)",
  )
  p.add_argument("--container-model-root", default="/catkin_ws/src/models")
  return p.parse_args()


def main() -> None:
  args = parse_args()
  repo = pathlib.Path(__file__).resolve().parents[2]
  sys.path.insert(0, str(repo / "scripts"))
  import mujoco
  import numpy as np
  import pal_mjlab.tasks  # noqa: F401
  import torch
  import yaml
  from eval_chain_insert import (
    PRESENT_DIST,
    PRESENT_S,
    PRESENT_SPEED,
    ChainAdapter,
    Reads,
    summarise,
  )
  from eval_hold import MlpActor
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg
  from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul
  from pal_mjlab.tasks.manipulation import scene
  from pal_mjlab.tasks.manipulation.mdp.observations import _grasped_bool
  from pal_mjlab.tasks.manipulation.pick_place import hold as H
  from pal_mjlab.tasks.manipulation.pick_place.kangaroo.env_cfg import (
    _baseline_with_table,
  )
  from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import BASELINES
  from pal_mjlab.tasks.manipulation.robots import ARM_MIRROR_SIGNS

  out = args.out.resolve()
  (out / "left").mkdir(parents=True, exist_ok=True)
  (out / "right").mkdir(parents=True, exist_ok=True)
  dev = args.device
  n = args.episodes

  cfg = load_env_cfg(TASK, play=True)
  cfg.episode_length_s = args.episode_s
  cfg.scene.num_envs = n
  cfg.seed = args.seed
  assert not cfg.observations["actor"].enable_corruption
  env = ManagerBasedRlEnv(cfg=cfg, device=dev)
  ad = ChainAdapter(env)
  lh, ri = MlpActor(args.lh, dev), MlpActor(args.ri, dev)
  assert (lh.obs_dim, ri.obs_dim) == (40, 42)
  reads = Reads(env, args.episode_s)
  term = env.action_manager.get_term("joint_pos")
  robot = env.scene["robot"]
  block, pin = env.scene["block"], env.scene["object"]
  goal_term = env.command_manager.get_term("goal_pose")
  dt = float(env.step_dt)
  steps = round(args.episode_s / dt)
  bi_idx = robot.body_names.index("base_link")
  site = {s: robot.site_names.index(f"tcp_{s}") for s in ("left", "right")}

  def base():
    return robot.data.body_link_pos_w[:, bi_idx], robot.data.body_link_quat_w[:, bi_idx]

  def to_b(p_w):
    bp, bq = base()
    return quat_apply(quat_inv(bq), p_w - bp)

  def q_b(q_w):
    _, bq = base()
    return quat_mul(quat_inv(bq), q_w)

  obs, _ = env.reset()
  reads.start(torch.arange(n, device=dev))

  # ------------------------------------------------------------------ the episodes
  half = scene.BLOCK_HEIGHT / 2

  def yaw_of(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

  bb, pb = to_b(block.data.root_link_pos_w), to_b(pin.data.root_link_pos_w)
  by, py = (
    yaw_of(q_b(block.data.root_link_quat_w)),
    yaw_of(q_b(pin.data.root_link_quat_w)),
  )
  g_b = to_b(goal_term.command)
  episodes = [
    {
      "id": k,
      "block_x": float(bb[k, 0]),
      "block_y": float(bb[k, 1]),
      "block_z": float(bb[k, 2]),
      "block_yaw": float(by[k]),
      "pin_x": float(pb[k, 0]),
      "pin_y": float(pb[k, 1]),
      "pin_z": float(pb[k, 2]),
      "pin_yaw": float(py[k]),
      "goal_x": float(g_b[k, 0]),
      "goal_y": float(g_b[k, 1]),
      "goal_z": float(g_b[k, 2]),
    }  # fmt: skip
    for k in range(n)
  ]
  (out / "episodes.json").write_text(
    json.dumps(
      {
        "description": "§13.1 chain episodes: the chain env's own resets (play, DR0, seed "
        f"{args.seed}); the block (origin at its bottom face's centre) on the left's area, the pin (origin at its centre) on "
        "the right's, H (the block's centre at the hold); base_link frame, m / rad",
        "episodes": episodes,
      },
      indent=1,
    )
    + "\n"
  )

  # ------------------------------------------------------------------ the reference rollout
  names = robot.joint_names
  arm_ids = {s: [names.index(f"arm_{s}_{i}_joint") for i in range(1, 8)] + [names.index(f"gripper_{s}_finger_joint")] for s in ("left", "right")}  # fmt: skip
  keys = "q_left q_right qd_left qd_right target obs_left obs_right act_left act_right block_b block_q pin_b pin_q tcp_left tcp_right tcpq_left tcpq_right grasp_left grasp_right go"  # noqa: E501
  log = {k: [] for k in keys.split()}
  done_at = torch.full((n,), -1, dtype=torch.long, device=dev)
  rows: dict[int, dict] = {}
  grasp_sensor = {"left": "left_pad_contact", "right": "right_pad_contact"}
  with torch.inference_mode():
    for t_step in range(steps + 5):
      t = ad.terms(obs, "actor")
      go = reads.present(t)
      o_l, o_r = ad.lh(t), ad.ri(t)
      a_l = lh(o_l)
      a_r = torch.where(go.unsqueeze(-1), ri(o_r), torch.zeros_like(a_l))
      for s in ("left", "right"):
        log[f"q_{s}"].append(robot.data.joint_pos[:, arm_ids[s]].clone())
        log[f"qd_{s}"].append(robot.data.joint_vel[:, arm_ids[s]].clone())
        log[f"tcp_{s}"].append(to_b(robot.data.site_pos_w[:, site[s]]))
        log[f"tcpq_{s}"].append(q_b(robot.data.site_quat_w[:, site[s]]))
        log[f"grasp_{s}"].append(_grasped_bool(env, grasp_sensor[s]).clone())
      log["obs_left"].append(o_l.clone())
      log["obs_right"].append(o_r.clone())
      log["act_left"].append(a_l.clone())
      log["act_right"].append(a_r.clone())
      log["go"].append(go.clone())
      log["block_b"].append(to_b(block.data.root_link_pos_w))
      log["block_q"].append(q_b(block.data.root_link_quat_w))
      log["pin_b"].append(to_b(pin.data.root_link_pos_w))
      log["pin_q"].append(q_b(pin.data.root_link_quat_w))
      obs, _, term_done, trunc, _ = env.step(ad.actions(a_l, a_r))
      log["target"].append(term._processed_actions.clone())
      done = (term_done | trunc) & (done_at < 0)
      finished = torch.nonzero(done, as_tuple=False).flatten()
      if finished.numel():
        before = len(reads.rows)
        reads.finish(finished)
        for j, i in enumerate(finished.tolist()):
          rows[i] = reads.rows[before + j]
        done_at[finished] = t_step
      reads.step()
      if bool((done_at >= 0).all()):
        break
  ref = {k: torch.stack(v, dim=1).float().cpu().numpy() for k, v in log.items()}
  ref["done_at"] = done_at.cpu().numpy()
  ref["dt"] = np.array(dt)
  ref["action_joint_names"] = np.array([nm.split("/")[-1] for nm in term.target_names])
  np.savez_compressed(out / "mujoco_reference.npz", **ref)
  ordered = [rows[k] | {"id": k, "done_step": int(done_at[k])} for k in range(n)]
  summary = summarise(ordered)
  (out / "chain_reference.json").write_text(
    json.dumps(
      {"lh": str(args.lh), "ri": str(args.ri), **summary, "episodes_detail": ordered},
      indent=1,
    )
  )
  for k, v in summary.items():
    if k != "bins":
      print(f"  {k:30s} {v}")
  print("per episode through_held:", [int(r["through_held"]) for r in ordered])

  # ------------------------------------------------------------------ action pipeline per arm
  tnames = [nm.split("/")[-1] for nm in term.target_names]
  s7 = list(ARM_MIRROR_SIGNS)
  s8 = s7 + [1.0]
  big = 1.0e9

  def fin(v):
    v = float(v)
    return v if math.isfinite(v) else (big if v > 0 else -big)

  def col(x, i):
    return float(x[0, i]) if torch.is_tensor(x) else float(x)

  def arm_pipeline(side: str) -> dict:
    cols = [tnames.index(f"arm_{side}_{i}_joint") for i in range(1, 8)] + [
      tnames.index(f"gripper_{side}_finger_joint")
    ]
    jids = [robot.joint_names.index(tnames[c]) for c in cols]
    default = robot.data.default_joint_pos[0, jids].tolist()
    scale = [col(term.scale, c) for c in cols]
    offset = [col(term.offset, c) for c in cols]
    assert np.allclose(offset, default), (side, offset, default)
    clip = term._clip[0].tolist()
    ridx, bidx = term._range_idx.tolist(), term._band_idx.tolist()
    rng = [term._range[0, ridx.index(c)].tolist() for c in cols[:7]]
    band = term._band.cpu().numpy()
    bnd = [band[bidx.index(c)].tolist() for c in cols[:7]]
    return {
      "joints": [tnames[c] for c in cols],
      "default": default, "scale": scale,
      "clip_lo": [fin(clip[c][0]) for c in cols], "clip_hi": [fin(clip[c][1]) for c in cols],
      "rate": [fin(term._rate[0, c]) for c in cols],
      "range_lo": [r[0] for r in rng], "range_hi": [r[1] for r in rng],
      "band_half": [b[0] for b in bnd], "band_dk": [b[1] for b in bnd], "band_k": [b[2] for b in bnd],
    }  # fmt: skip

  pl, pr = arm_pipeline("left"), arm_pipeline("right")
  # The right in the policy convention: q_p = s q_robot (R-I was trained on the left arm).
  pr_p = dict(pr)
  pr_p["default"] = [s * v for s, v in zip(s8, pr["default"], strict=True)]
  pr_p["clip_lo"] = [
    min(s * a, s * b) for s, a, b in zip(s8, pr["clip_lo"], pr["clip_hi"], strict=True)
  ]
  pr_p["clip_hi"] = [
    max(s * a, s * b) for s, a, b in zip(s8, pr["clip_lo"], pr["clip_hi"], strict=True)
  ]
  pr_p["range_lo"] = [
    min(s * a, s * b)
    for s, a, b in zip(s7, pr["range_lo"], pr["range_hi"], strict=True)
  ]
  pr_p["range_hi"] = [
    max(s * a, s * b)
    for s, a, b in zip(s7, pr["range_lo"], pr["range_hi"], strict=True)
  ]

  # ------------------------------------------------------------------ TCP sites, pads
  mj = env.sim.mj_model

  def tcp_site(side):
    sid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, f"robot/tcp_{side}")
    body = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, mj.site_bodyid[sid])
    assert body.split("/")[-1] == f"arm_{side}_7_link", body
    return mj.site_pos[sid].tolist(), mj.site_quat[sid].tolist()

  def pad_axis(side):
    pads = [
      robot.site_names.index(f"gripper_{side}_fingertip_{lr}_site")
      for lr in ("left", "right")
    ]
    p = robot.data.site_pos_w
    v = quat_apply(
      quat_inv(robot.data.site_quat_w[:, site[side]]), p[:, pads[0]] - p[:, pads[1]]
    )
    return torch.nn.functional.normalize(v.mean(dim=0), dim=0).tolist()

  with torch.inference_mode():
    env.reset()
    axes = {s: pad_axis(s) for s in ("left", "right")}

  # ------------------------------------------------------------------ grasp calibration per arm
  def calib(
    side: str, pos_key: str, quat_key: str, local: tuple[float, float, float]
  ) -> dict:
    """The part's reference point (its pose's origin + ``local`` in its frame) in the TCP frame at the first grasp step
    of each episode, the jaw there and while held."""
    g = ref[f"grasp_{side}"].astype(bool)
    jaw = ref[f"q_{side}"][..., 7]
    tcp = torch.as_tensor(ref[f"tcp_{side}"])
    tq = torch.as_tensor(ref[f"tcpq_{side}"])
    p = torch.as_tensor(ref[pos_key])
    pq = torch.as_tensor(ref[quat_key])
    loc = torch.tensor(local, dtype=p.dtype).expand(p.shape)
    c = p + quat_apply(pq.reshape(-1, 4), loc.reshape(-1, 3)).reshape(p.shape)
    rel = (
      quat_apply(quat_inv(tq.reshape(-1, 4)), (c - tcp).reshape(-1, 3))
      .reshape(c.shape)
      .numpy()
    )
    onset, onset_rel = [], []
    for k in range(g.shape[0]):
      idx = np.flatnonzero(g[k, : int(done_at[k]) + 1])
      if idx.size == 0:
        continue
      onset.append(float(jaw[k, idx[0]]))
      onset_rel.append(rel[k, idx[0]].tolist())
    if not onset:
      return {"episodes_grasped": 0}
    held = jaw[g]
    o = np.array(onset_rel)
    return {
      "episodes_grasped": len(onset),
      "jaw_onset_values": onset,
      "part_in_tcp_at_onset": onset_rel,
      "virtual": {
        "jaw_contact": min(onset),
        "jaw_hold_median": float(np.median(held)),
        "jaw_release": min(onset) - 0.005,
        "jaw_hold_command": float(np.median(held)),
        "x_tol": float(max(np.abs(o[:, 0]).max() + 0.005, 0.01)),
        "y_tol": float(np.abs(o[:, 1]).max() + 0.01),
        "z_lo": float(o[:, 2].min() - 0.01),
        "z_hi": float(o[:, 2].max() + 0.01),
      },
    }

  cal = {
    "left": calib("left", "block_b", "block_q", (0.0, 0.0, half)),
    "right": calib("right", "pin_b", "pin_q", (0.0, 0.0, 0.0)),
  }
  (out / "grasp_calibration.json").write_text(json.dumps(cal, indent=1) + "\n")
  print("grasp calibration (virtual):", {s: c.get("virtual") for s, c in cal.items()})

  # ------------------------------------------------------------------ ONNX
  import onnx
  from onnx.reference import ReferenceEvaluator

  class Wrap(torch.nn.Module):
    def __init__(self, actor: MlpActor):
      super().__init__()
      self.register_buffer("mean", actor.mean.detach().cpu().float().clone())
      self.register_buffer("std", actor.std.detach().cpu().float().clone())
      self.lin = torch.nn.ModuleList()
      for w, b in actor.layers:
        m = torch.nn.Linear(w.shape[1], w.shape[0])
        m.weight.data.copy_(w.cpu())
        m.bias.data.copy_(b.cpu())
        self.lin.append(m)

    def forward(self, x):
      x = (x - self.mean) / (self.std + MlpActor.EPS)
      for i, m in enumerate(self.lin):
        x = m(x)
        if i < len(self.lin) - 1:
          x = torch.nn.functional.elu(x)
      return x

  onnx_info = {}
  for side, actor, ckpt, terms in (
    ("left", lh, args.lh, LH_TERMS),
    ("right", ri, args.ri, RI_TERMS),
  ):
    w = Wrap(actor).eval()
    path = out / side / "model.onnx"
    torch.onnx.export(
      w, torch.zeros(1, actor.obs_dim), str(path), export_params=True, opset_version=18,
      input_names=["obs"], output_names=["actions"], dynamic_axes={}, dynamo=False,
    )  # fmt: skip
    m = onnx.load(str(path))
    for k, v in {
      "checkpoint": str(ckpt),
      "checkpoint_sha256": sha256(ckpt),
      "observation_terms": ",".join(terms),
    }.items():
      e = m.metadata_props.add()
      e.key, e.value = k, v
    onnx.save(m, str(path))
    sess = ReferenceEvaluator(str(path))
    xs = ref[f"obs_{side}"].reshape(-1, actor.obs_dim)
    pick = xs[np.linspace(0, len(xs) - 1, 400).astype(int)]
    with torch.no_grad():
      want = actor(torch.as_tensor(pick, device=dev)).cpu().numpy()
    got = np.concatenate(
      [sess.run(None, {"obs": row[None].astype(np.float32)})[0] for row in pick]
    )
    err = float(np.abs(got - want).max())
    print(
      f"{side}: onnx vs torch max |diff| over 400 reference observations: {err:.2e}"
    )
    assert err < 1e-4, err
    onnx_info[side] = {"onnx_sha256": sha256(path), "check_max_abs_diff": err, "checkpoint": str(ckpt), "checkpoint_sha256": sha256(ckpt)}  # fmt: skip

  # ------------------------------------------------------------------ scene and profile
  baseline = _baseline_with_table(BASELINES["B"])
  top_z = baseline.table_top_z
  tp = scene.table_pos(top_z)
  bp, _ = base()
  bp = (bp[0] - env.scene.env_origins[0]).tolist()
  jaw_gap = yaml.safe_load(args.jaw_gap_from.read_text())["jaw_gap"]

  def arm_profile(side: str, pipe: dict, signs: list[float], terms, dims) -> dict:
    pos, quat = tcp_site(side)
    c = cal[side].get("virtual", {})
    return {
      "model_path": f"{args.container_model_root}/{out.name}/{side}/model.onnx",
      "action_joint_names": pipe["joints"],
      "joint_signs": signs,
      "mirror_observations": side == "right",
      "default_joint_pos": pipe["default"],
      "action_scale": pipe["scale"],
      "action_clip_lower": pipe["clip_lo"],
      "action_clip_upper": pipe["clip_hi"],
      "action_rate_limit": pipe["rate"],
      "effort_band_half": pipe["band_half"],
      "effort_band_dk": pipe["band_dk"],
      "effort_tau_cmd": [
        h * k for h, k in zip(pipe["band_half"], pipe["band_k"], strict=True)
      ],
      "range_clip_lower": pipe["range_lo"],
      "range_clip_upper": pipe["range_hi"],
      "observation_terms": list(terms),
      "observation_dims": list(dims),
      "actions_obs": "act",
      "tip_link": f"arm_{side}_7_link",
      "tcp_offset": pos,
      "tcp_offset_quat_wxyz": quat,
      "side": 1.0 if side == "left" else -1.0,
      "grasp": {
        k: c[k]
        for k in (
          "jaw_contact",
          "jaw_hold_median",
          "jaw_release",
          "jaw_hold_command",
          "x_tol",
          "y_tol",
          "z_lo",
          "z_hi",
        )
        if k in c
      },  # noqa: E501
      "jaw_gap": {**jaw_gap, "closing_axis_tcp": axes[side]},
    }

  prof = {
    "description": "§13.1 chain: L-H on the left (side grasp of the socket block, hold at H), R-I on the right through the "
    "mirror map (the pin through the held block); export_s13_chain.py",
    "selectors": CHAIN | {"DR": "DR-scene (training); the reference is DR0"},
    "base_link": "base_link",
    "policy_rate": 1.0 / dt,
    "episode_s": args.episode_s,
    "present": {"dist": PRESENT_DIST, "speed": PRESENT_SPEED, "hold_s": PRESENT_S},
    "table_top_z_b": top_z - bp[2],
    "table_x_b": [
      tp[0] - scene.TABLE_HALF_X - bp[0],
      tp[0] + scene.TABLE_HALF_X - bp[0],
    ],
    "table_y_b": [
      tp[1] - scene.TABLE_HALF_Y - bp[1],
      tp[1] + scene.TABLE_HALF_Y - bp[1],
    ],
    "hold_xy_b": [H.HOLD_XY[0] - bp[0], H.HOLD_XY[1] - bp[1]],
    "hold_dz": H.HOLD_DZ,
    "block": {
      "size": scene.BLOCK_SIZE,
      "height": scene.BLOCK_HEIGHT,
      "hole_d": scene.BLOCK_HOLE_D,
      "chamfer": list(scene.BLOCK_CHAMFER),
    },  # noqa: E501
    "pin": {
      "span": scene.PIN_SPAN,
      "thickness": scene.PIN_THICKNESS,
      "length": scene.PIN_LENGTH,
    },
    "left": arm_profile("left", pl, [1.0] * 8, LH_TERMS, LH_DIMS),
    "right": arm_profile("right", pr_p, s8, RI_TERMS, RI_DIMS),
  }
  prof["right"]["action_joint_names"] = pr["joints"]
  (out / "insert_profile.yaml").write_text(
    "# Generated by export_s13_chain.py. Per arm: policy convention = the left arm's; the right maps\n"
    "# q_policy = joint_signs * q_robot and mirrors positions / frames (eval_chain_insert.ChainAdapter).\n"
    + yaml.safe_dump(prof, sort_keys=False, default_flow_style=None, width=110)
  )

  # ------------------------------------------------------------------ start pose, gains, info
  all_q = robot.data.default_joint_pos[0].tolist()
  (out / "deployment_start_config.json").write_text(
    json.dumps(
      {
        "schema_version": 1,
        "description": "§13 chain env default joint state: left home, right = mirror(left home)",
        "joint_names": list(robot.joint_names),
        "joint_positions": {
          k: round(float(v), 8) for k, v in zip(robot.joint_names, all_q, strict=True)
        },
        "joint_velocities": {k: 0.0 for k in robot.joint_names},
      },
      indent=2,
    )
    + "\n"
  )
  gains = {}
  for i in range(mj.nu):
    jid = int(mj.actuator_trnid[i, 0])
    jn = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, jid).split("/")[-1]
    if jn.startswith(("arm_", "gripper_")):
      gains[jn] = [float(mj.actuator_gainprm[i, 0]), float(-mj.actuator_biasprm[i, 2])]
  (out / "gains.json").write_text(json.dumps(gains, indent=1) + "\n")
  info = {
    "onnx": onnx_info,
    "pad_axes_tcp": axes,
    "base_link_in_env": bp,
    "reference_summary": {k: v for k, v in summary.items() if k != "bins"},
    "pal_mjlab_commit": os.popen(f"git -C {repo} rev-parse HEAD").read().strip(),
    "pal_mjlab_dirty": os.popen(f"git -C {repo} status --short").read().strip(),
  }
  (out / "export_info.json").write_text(json.dumps(info, indent=1) + "\n")
  env.close()
  print("wrote", out)


if __name__ == "__main__":
  main()
