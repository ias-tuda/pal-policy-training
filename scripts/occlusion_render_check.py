"""Plan S7-7.0 item 2: does the line-of-sight model agree with what a camera at the ``torso_front``
mount would actually see?

Rolls a checkpoint out and, on every ``--stride``-th control step, renders a MuJoCo *segmentation*
image from the calibrated camera position (`mdp.occlusion.TORSO_FRONT_CAMERA_*`, aimed at the cube:
the model's own assumption is that the cube is always inside the image, and the camera's
orientation is not in code) and counts the cube's pixels, against the pixels the cube would show
with nothing in front of it (a second render with only the cube's geom group on). The share is
what the cube's silhouette keeps. A step is *seen* in the render when the share is at least
``--seen`` (0.5 by default: half the silhouette).

Against it the line-of-sight model is scored in its four variants -- the line stopping 3 cm short
of the cube (Claudius's) or running to the near face, the fingers left out or in as capsules --
each blocked at score >= 1 (a capsule touching the line) and at >= 0.5. The confusion matrix is
reported per phase (approach / grasp / carry / retreat), which is where the two disagree.

    REWARD=... DR=DR-scene PPO=PPO-stack SLEW=0.5 BASELINE=B SPAWN=W2 GAINS=G20 EFFORT=CTP \\
      uv run --extra cu128 python scripts/occlusion_render_check.py \\
      --checkpoint .../model_4999.pt --episodes 20 --json ../results/S7-7/render_check.json

The env is the checkpoint's own (``OCC`` unset: nothing is hidden from the policy); the model is
scored beside it, not in the loop. ``MUJOCO_GL=egl`` on a headless host.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pathlib

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from pal_mjlab.tasks.manipulation.mdp import occlusion as occ

PHASES = ("approach", "grasp", "carry", "retreat")
#: name -> (stop_short, fingers)
VARIANTS = {
  "arm": (occ.CUBE_HALF_SIZE + 0.03, False),
  "arm+fingers": (occ.CUBE_HALF_SIZE + 0.03, True),
  "face": (occ.CUBE_HALF_SIZE, False),
  "hand": (occ.CUBE_HALF_SIZE, True),
}


def _eval_module():
  spec = importlib.util.spec_from_file_location(
    "eval_pick_place", pathlib.Path(__file__).parent / "eval_pick_place.py"
  )
  assert spec and spec.loader
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--checkpoint", type=pathlib.Path, required=True)
  p.add_argument("--task", default="Mjlab-PickPlace-Pal-Kangaroo-Grippers")
  p.add_argument("--episodes", type=int, default=20)
  p.add_argument("--stride", type=int, default=2, help="render every n-th control step")
  p.add_argument("--seen", type=float, default=0.5, help="silhouette share for 'seen'")
  p.add_argument("--fovy", type=float, default=42.0, help="vertical field of view, deg")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cpu")
  p.add_argument("--json", type=pathlib.Path, default=None)
  p.add_argument(
    "--frames", type=pathlib.Path, default=None, help="save a few PNGs here"
  )
  p.add_argument(
    "--no-policy", action="store_true", help="zero actions (geometry only)"
  )
  return p.parse_args()


def main() -> None:
  args = parse_args()
  ev = _eval_module()
  torch.manual_seed(args.seed)
  cfg = load_env_cfg(args.task, play=True)
  cfg.episode_length_s = load_env_cfg(args.task, play=False).episode_length_s
  cfg.scene.num_envs = 1
  cfg.seed = args.seed
  cfg.viewer.width, cfg.viewer.height = 320, 180
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device, render_mode="rgb_array")
  if args.no_policy:
    wrapped, policy = None, None
  else:
    ns = argparse.Namespace(
      task=args.task, checkpoint=args.checkpoint, device=args.device
    )
    wrapped, policy = ev.load_policy(ns, env)
  obj = env.scene[ev.OBJECT]
  gripper_cfg = env.cfg.rewards.get("grasp") or env.cfg.rewards["lift"]
  grasp_sensor = (
    gripper_cfg.params.get("sensor_name") or gripper_cfg.params["grasp_sensor_name"]
  )

  rend = env._offline_renderer
  rend.update(env.sim.data)  # builds the copy's data and syncs env 0
  r = rend.renderer
  model, data = rend._model, rend._data
  model.vis.map.znear = 0.002
  model.vis.global_.fovy = args.fovy
  cube_geom = int(obj.indexing.geom_ids[0])
  cam_body = next(
    i
    for i in range(model.nbody)
    if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "").split("/")[-1]
    == occ.TORSO_FRONT_CAMERA_BODY
  )
  cube_body = int(model.geom_bodyid[cube_geom])
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  opt_all = mujoco.MjvOption()
  opt_cube = mujoco.MjvOption()
  opt_cube.geomgroup[:] = 0
  opt_cube.geomgroup[0] = 1  # the cube's group (0) only

  def seg_count(opt: mujoco.MjvOption) -> int:
    r.update_scene(data, camera=cam, scene_option=opt)
    seg = r.render()
    return int(
      (
        (seg[..., 0] == cube_geom) & (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
      ).sum()
    )

  r.enable_segmentation_rendering()
  rows = []
  frames_saved = 0
  obs, _ = wrapped.reset() if wrapped else env.reset()
  ep, step, ever_grasped = 0, 0, False
  rest_z = None
  with torch.inference_mode():
    while ep < args.episodes:
      if step % args.stride == 0:
        rend.update(env.sim.data)
        cam_pos = data.xpos[cam_body] + data.xmat[cam_body].reshape(3, 3) @ np.array(
          occ.TORSO_FRONT_CAMERA_POS
        )
        cube_pos = data.xpos[cube_body].copy()
        v = cube_pos - cam_pos
        dist = float(np.linalg.norm(v))
        cam.lookat[:] = cube_pos
        cam.distance = dist
        cam.azimuth = math.degrees(math.atan2(v[1], v[0]))
        cam.elevation = math.degrees(math.asin(v[2] / dist))
        n_ref = seg_count(opt_cube)
        n_full = seg_count(opt_all)
        share = n_full / n_ref if n_ref else float("nan")
        grasped = bool(ev._grasped_bool(env, grasp_sensor)[0])
        ever_grasped |= grasped
        z = float(obj.data.root_link_pos_w[0, 2])
        rest_z = z if rest_z is None else min(rest_z, z)
        lifted = z > rest_z + 0.02
        phase = (
          "approach"
          if not ever_grasped
          else ("carry" if grasped and lifted else "grasp" if grasped else "retreat")
        )
        row = {"ep": ep, "step": step, "phase": phase, "share": share, "dist": dist}
        target_w = obj.data.root_link_pos_w
        for name, (stop, fingers) in VARIANTS.items():
          segs = (
            *occ.LEFT_ARM_SIGHT_SEGMENTS,
            *(occ.LEFT_FINGER_SEGMENTS if fingers else ()),
          )
          score = occ.camera_view_blocked(
            env,
            occ.TORSO_FRONT_CAMERA_BODY,
            occ.TORSO_FRONT_CAMERA_POS,
            segs,
            target_w,
            stop,
            occ.CAMERA_VIEW_MARGIN,
          )
          row[f"score_{name}"] = float(score[0])
        rows.append(row)
        if (
          args.frames
          and frames_saved < 12
          and phase in ("grasp", "carry")
          and step % 40 == 0
        ):
          import imageio.v2 as imageio

          r.disable_segmentation_rendering()
          r.update_scene(data, camera=cam, scene_option=opt_all)
          args.frames.mkdir(parents=True, exist_ok=True)
          imageio.imwrite(
            args.frames / f"ep{ep}_s{step}_{phase}_share{share:.2f}.png", r.render()
          )
          r.enable_segmentation_rendering()
          frames_saved += 1
      actions = (
        policy(obs)
        if policy
        else torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
      )
      if wrapped:
        obs, _, dones, _ = wrapped.step(actions)
      else:
        obs, _, term, trunc, _ = env.step(actions)
        dones = term | trunc
      step += 1
      if bool(dones[0]):
        ep += 1
        step, ever_grasped, rest_z = 0, False, None
  r.disable_segmentation_rendering()

  result = summarise(rows, args.seen)
  print(json.dumps(result, indent=1))
  if args.json:
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
      json.dumps(
        {"args": {k: str(v) for k, v in vars(args).items()}, **result, "rows": rows},
        indent=1,
      )
    )
    print(f"wrote {args.json}")
  env.close()


def summarise(rows: list[dict], seen: float) -> dict:
  out: dict = {"steps": len(rows), "seen_threshold": seen, "phases": {}, "models": {}}
  for phase in (*PHASES, "all"):
    sel = [x for x in rows if phase == "all" or x["phase"] == phase]
    if not sel:
      continue
    render_hidden = np.array([x["share"] < seen for x in sel])
    ph = {
      "n": len(sel),
      "render_hidden": float(render_hidden.mean()),
      "share_median": float(np.median([x["share"] for x in sel])),
    }
    for name in VARIANTS:
      for thr in (1.0, 0.5):
        pred = np.array([x[f"score_{name}"] >= thr - 1e-6 for x in sel])
        tp = int((pred & render_hidden).sum())
        fp = int((pred & ~render_hidden).sum())
        fn = int((~pred & render_hidden).sum())
        tn = int((~pred & ~render_hidden).sum())
        ph[f"{name}@{thr}"] = {
          "model_hidden": float(pred.mean()),
          "tp": tp,
          "fp": fp,
          "fn": fn,
          "tn": tn,
          "agree": (tp + tn) / len(sel),
        }
    out["phases"][phase] = ph
  return out


if __name__ == "__main__":
  main()
