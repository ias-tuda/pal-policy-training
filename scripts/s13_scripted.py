"""Plan §13 step 0 (`pick-place-insert-plan.md`, items 4-5, L1): the §13.1 skills scripted through the exact action
interface a policy uses, in the training env.

``hold`` (``S13=L-H``): the left's side grasp of the socket block and its hold at H, then the push test. Per env: the
block's pose and the table read from the env after its reset (or forced: ``--forced`` puts envs 0.. at the spawn
area's corners and centre at yaw 0 and 45 deg), IK on a CPU copy of the env's scene (`s13_kin`) for above ->
pregrasp -> grasp -> lift -> H, the hand horizontal and pointing at the face nearest the robot (as L-H's reward
asks), then: the arm driven along straight joint-space lines at <= 0.03 rad per step (under the stack speeds, so the
env's rate limit never bends the path IK checked), the jaw closed at the grasp, 2 s held at H, 10 N down on the
block's top for 1 s, 1 s more. Reported per env: grasped, lifted, held at H (`mdp.held_at_goal`, L-H's parameters)
over the 2 s, the push's peak tilt and displacement, the distance from H and the tilt 1 s after, arm 1-6 / hand on the
table, control steps that needed |a| > 1.

    S13=L-H SPAWN=WC BASELINE=B TABLE_DZ=22 GAINS=G20 EFFORT=CTP SLEW=stack LAG=FP+JS ... \\
      python scripts/s13_scripted.py hold --num-envs 16 --forced --json hold.json
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import s13_kin as K  # noqa: E402

TASK = "Mjlab-PickPlace-Pal-Kangaroo-Grippers"
#: Per-step bound on the scripted joint move, rad (the stack speeds are 0.039-0.047 rad per 20 ms step).
STEP_RAD = 0.03
SETTLE_SPEED = 0.05
SETTLE_MAX_S = 1.5
JAW_CLOSE_S = 0.8


def build_env(args):
  import pal_mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg(args.task, play=False)
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  cfg.episode_length_s = 60.0
  cfg.events.pop("push_block", None)
  cfg.terminations.pop("out_of_reach", None)
  return ManagerBasedRlEnv(cfg=cfg, device=args.device), cfg


class Driver:
  """Steps the env with joint-space waypoints turned into the action term's raw actions."""

  def __init__(self, env, side: str = "left") -> None:
    self.env = env
    self.term = env.action_manager.get_term("joint_pos")
    names = [n.split("/")[-1] for n in self.term.target_names]
    self.arm_cols = [names.index(f"arm_{side}_{i}_joint") for i in range(1, 8)]
    self.jaw_col = names.index(f"gripper_{side}_finger_joint")
    self.offset = self.term._offset.clone()
    self.scale = self.term._scale.clone()
    robot = env.scene["robot"]
    self.arm_q = [robot.joint_names.index(f"arm_{side}_{i}_joint") for i in range(1, 8)]
    self.robot = robot
    self.clipped = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self.resets = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self.hooks: list = []
    self.jaw = torch.zeros(env.num_envs, device=env.device)

  def q(self) -> np.ndarray:
    return self.robot.data.joint_pos[:, self.arm_q].cpu().numpy()

  def step(self, q_target: np.ndarray) -> None:
    target = self.offset.clone()
    target[:, self.arm_cols] = torch.as_tensor(
      q_target, dtype=torch.float32, device=self.env.device
    )
    target[:, self.jaw_col] = self.jaw * 0.78
    raw = (target - self.offset) / self.scale
    self.clipped += (raw.abs() > 1.0 + 1e-6).any(dim=-1).long()
    before = self.env.episode_length_buf.clone()
    self.env.step(raw.clamp(-1.0, 1.0))
    self.resets += (self.env.episode_length_buf <= before).long()
    for h in self.hooks:
      h()

  def move(self, a: np.ndarray, b: np.ndarray) -> None:
    n = max(1, int(math.ceil(float(np.abs(b - a).max()) / STEP_RAD)))
    for s in range(1, n + 1):
      self.step(a + (b - a) * (s / n))
    for _ in range(round(SETTLE_MAX_S / self.env.step_dt)):
      speed = self.robot.data.joint_vel[:, self.arm_q].abs().amax(dim=-1)
      if bool((speed < SETTLE_SPEED).all()):
        return
      self.step(b)

  def hold(self, q: np.ndarray, seconds: float) -> None:
    for _ in range(round(seconds / self.env.step_dt)):
      self.step(q)


class Recorder:
  """Every ``every`` control steps, the listed envs' states copied into a CPU copy of the scene and rendered
  (front-left view on the work area), tiled side by side; written as an MP4."""

  def __init__(
    self,
    env,
    model,
    envs: list[int],
    every: int = 2,
    size=(360, 480),
    close: bool = False,
  ) -> None:
    self.env, self.m, self.envs, self.every, self.close = env, model, envs, every, close
    self.lookat = (
      (env.scene["object"].data.root_link_pos_w - env.scene.env_origins).cpu().numpy()
    )
    self.d = mujoco.MjData(model)
    self.r = mujoco.Renderer(model, size[0], size[1])
    self.frames: list[np.ndarray] = []
    self.k = 0
    self.table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table/table")

  def __call__(self) -> None:
    self.k += 1
    if self.k % self.every:
      return
    sim = self.env.sim
    qpos = sim.data.qpos.cpu().numpy()
    mpos = sim.data.mocap_pos.cpu().numpy()
    mquat = sim.data.mocap_quat.cpu().numpy()
    bpos = sim.model.body_pos
    bpos = bpos.cpu().numpy() if bpos.ndim == 3 else None
    origins = self.env.scene.env_origins.cpu().numpy()
    tiles = []
    for i in self.envs:
      self.d.qpos[:] = qpos[i]
      self.d.mocap_pos[:] = mpos[i] - origins[i]
      self.d.mocap_quat[:] = mquat[i]
      for j in range(self.m.njnt):
        if self.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
          a = self.m.jnt_qposadr[j]
          self.d.qpos[a : a + 3] -= origins[i]
      if bpos is not None:
        self.m.body_pos[self.table] = bpos[i, self.table]
      mujoco.mj_forward(self.m, self.d)
      cam = mujoco.MjvCamera()
      if self.close:
        cam.lookat[:] = self.lookat[i] + np.array([0.0, 0.0, 0.03])
        cam.distance, cam.azimuth, cam.elevation = 0.32, 135.0, -12.0
      else:
        cam.lookat[:] = (0.47, 0.10, 0.90)
        cam.distance, cam.azimuth, cam.elevation = 0.95, 210.0, -28.0
      self.r.update_scene(self.d, cam)
      tiles.append(self.r.render().copy())
    self.frames.append(np.concatenate(tiles, axis=1))

  def write(self, path: pathlib.Path, fps: float) -> None:
    import imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(path), self.frames, fps=fps, codec="libx264", quality=7)
    print(f"wrote {path} ({len(self.frames)} frames)")


def _forced_poses(n: int) -> list[tuple[float, float, float]]:
  from pal_mjlab.tasks.manipulation.pick_place.hold import HOLD_SPAWN_X, HOLD_SPAWN_Y

  (x0, x1), (y0, y1) = HOLD_SPAWN_X, HOLD_SPAWN_Y
  pts = [((x0 + x1) / 2, (y0 + y1) / 2), (x0, y0), (x0, y1), (x1, y0), (x1, y1)]
  poses = [(x, y, yaw) for yaw in (0.0, math.pi / 4) for x, y in pts]
  return poses[:n]


def hold_rollout(args) -> dict:
  from mjlab.managers.scene_entity_config import SceneEntityCfg
  from pal_mjlab.tasks.manipulation import mdp, scene
  from pal_mjlab.tasks.manipulation.mdp.events import surface_top_z
  from pal_mjlab.tasks.manipulation.mdp.observations import (
    _grasped_bool,
    _was_lifted_bool,
  )
  from pal_mjlab.tasks.manipulation.pick_place import hold as H
  from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS_LEFT_ARM
  from pal_mjlab.tasks.manipulation.tabletop_env_cfg import SUPPORT_SENSOR
  from scene_model import nominal_data

  env, cfg = build_env(args)
  env.reset()
  dev = env.device
  obj = env.scene["object"]
  origins = env.scene.env_origins
  half = scene.BLOCK_HEIGHT / 2
  if args.forced:
    forced = _forced_poses(env.num_envs)
    ids = torch.arange(len(forced), device=dev)
    top = surface_top_z(env, SceneEntityCfg("table"))[ids]
    pose = torch.zeros(len(forced), 7, device=dev)
    for k, (x, y, yaw) in enumerate(forced):
      pose[k, :3] = torch.tensor([x, y, 0.0], device=dev) + origins[k]
      pose[k, 2] = top[k] + half
      pose[k, 3:] = torch.tensor(
        [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)], device=dev
      )
    obj.write_root_link_pose_to_sim(pose, env_ids=ids)
    obj.write_root_link_velocity_to_sim(
      torch.zeros(len(forced), 6, device=dev), env_ids=ids
    )
  drv = Driver(env, "left")
  drv.hold(drv.q(), 0.3)  # settle the block and let the observations catch up

  model = K.compile_env_scene(cfg)
  data = nominal_data(model)
  arm = K.Arm(model, data, "left")
  pad_off = arm.pad_offset()
  block_geoms = {
    g
    for g in range(model.ngeom)
    if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(
      "object/"
    )
  }
  table_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table/table")
  table0 = model.body_pos[table_bid].copy()
  qfree = model.jnt_qposadr[
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object/object_joint")
  ]
  rng = np.random.default_rng(args.seed)

  top_env = surface_top_z(env, SceneEntityCfg("table")).cpu().numpy()
  from pal_mjlab.tasks.manipulation.mdp.events import _surface_centre_xy

  txy = (
    _surface_centre_xy(env, SceneEntityCfg("table")) - origins[:, :2]
  ).cpu().numpy() - np.asarray(scene.TABLE_POS[:2])
  pos = (obj.data.root_link_pos_w - origins).cpu().numpy()
  quat = obj.data.root_link_quat_w.cpu().numpy()
  plans, errors = [], []
  for i in range(env.num_envs):
    w, x, y, z = quat[i]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    mujoco.mj_resetData(model, data)
    data.qpos[:] = nominal_data(model).qpos
    model.body_pos[table_bid] = table0
    model_top = None
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table/table_top")
    mujoco.mj_forward(model, data)
    if gid >= 0:
      model_top = float(data.geom_xpos[gid][2] + model.geom_size[gid][2])
    if model_top is None:
      raise RuntimeError("no table/table_top geom in the CPU model")
    # DR-scene moves each env's table +/- 2 cm in xy and +/- 1 cm in z: the CPU model's table follows this env's.
    shift = np.array(
      [float(txy[i, 0]), float(txy[i, 1]), float(top_env[i]) - model_top]
    )
    model.body_pos[table_bid] = table0 + shift
    data.qpos[qfree : qfree + 3] = pos[i]
    data.qpos[qfree + 3 : qfree + 7] = quat[i]
    mujoco.mj_forward(model, data)
    bottom = pos[i][2] - half
    # Every face whose outward normal is within L-H's reach cone of the robot's direction, best aligned first.
    normals = [
      np.array([math.cos(yaw + k * math.pi / 2), math.sin(yaw + k * math.pi / 2)])
      for k in range(4)
    ]
    faces = sorted(
      (n for n in normals if -n[0] >= math.cos(H.REACH_FACE_CONE)), key=lambda n: n[0]
    )
    plan, err = None, ""
    cases = (
      (hp, n, sg)
      for hp in (H.PAD_HEIGHT, 0.045, 0.050)
      for n in faces
      for sg in (1.0, -1.0)
    )
    for hp, n, sign in cases:
      a = np.array([-n[0], -n[1], 0.0])
      pads = np.array([pos[i][0], pos[i][1], bottom + hp])
      tcp = pads - pad_off * a
      r = K.hand_frame(a, sign * np.array([-a[1], a[0], 0.0]))
      r_h = K.hand_frame(np.array(H.HOLD_APPROACH), sign * np.array([0.0, 1.0, 0.0]))
      h_pads = np.array(
        [H.HOLD_XY[0], H.HOLD_XY[1], float(top_env[i]) + H.HOLD_DZ + hp]
      )
      tcp_h = h_pads - pad_off * np.array(H.HOLD_APPROACH)

      def near(q):
        return arm.contacts(q)

      def free(q):
        return arm.contacts(q, ignore=block_geoms)

      q_g = K.solve_endpoint(arm, tcp, r, near, rng, attempts=48)
      if q_g is None:
        err = "grasp"
        continue
      q_p = None
      for back, up in ((0.08, 0.02), (0.0, 0.10), (0.04, 0.06)):
        q_p = K.solve(
          arm,
          tcp - back * a + np.array([0, 0, up]),
          r,
          q_g,
          near,
          rng,
          local=8,
          glob=16,
        )
        if q_p is not None:
          break
      if q_p is None:
        err = "pregrasp"
        continue
      q_up = K.solve(
        arm,
        tcp - 0.04 * a + np.array([0, 0, 0.14]),
        r,
        q_p,
        free,
        rng,
        local=8,
        glob=16,
      )
      q_l = K.solve(arm, tcp + np.array([0, 0, 0.10]), r, q_g, free, rng)
      q_hh = None if q_l is None else K.solve(arm, tcp_h, r_h, q_l, free, rng)
      if q_up is None or q_l is None or q_hh is None:
        err = "up" if q_up is None else ("lift" if q_l is None else "hold_H")
        continue
      plan = {"above": q_up, "pregrasp": q_p, "grasp": q_g, "lift": q_l, "H": q_hh}
      break
    if plan is None:
      q0 = arm.q_default
      plan = dict.fromkeys(("above", "pregrasp", "grasp", "lift", "H"), q0)
    plans.append(plan)
    errors.append(err)
    print(
      f"env {i}: block ({pos[i][0]:.3f}, {pos[i][1]:.3f}) yaw {math.degrees(yaw):+.1f} -> {'plan' if not err else 'NO PLAN: ' + err}",
      flush=True,
    )

  gripper = KANGAROO_GRIPPERS_LEFT_ARM.gripper("left")
  hp = env.reward_manager.get_term_cfg("held").params  # resolved by the manager
  n_env = env.num_envs
  reads = {
    k: torch.zeros(n_env, device=dev)
    for k in (
      "held_steps",
      "hold_steps",
      "push_tilt",
      "push_disp",
      "after_dist",
      "after_tilt",
    )
  }
  flags = {
    k: torch.zeros(n_env, dtype=torch.bool, device=dev)
    for k in ("grasped", "lifted", "arm_table", "hand_table", "held_after")
  }
  phase = {"name": ""}
  pre = torch.zeros(n_env, 3, device=dev)

  def hook():
    sensor = (
      env.scene.sensors.get("arm_table_contact")
      if hasattr(env.scene.sensors, "get")
      else None
    )
    if sensor is not None:
      flags["arm_table"] |= sensor.data.found.bool().reshape(n_env, -1).any(-1)
    flags["hand_table"] |= (
      env.scene[gripper.surface_sensor_name]
      .data.found.bool()
      .reshape(n_env, -1)
      .any(-1)
    )
    flags["lifted"] |= _was_lifted_bool(
      env, gripper.contact_sensor_name, SUPPORT_SENSOR
    )
    if phase["name"] == "hold":
      reads["hold_steps"] += 1
      reads["held_steps"] += mdp.held_at_goal(env, **hp)
    if phase["name"] == "push":
      tilt = mdp.axis_tilt(env, "object")
      reads["push_tilt"] = torch.maximum(reads["push_tilt"], tilt)
      reads["push_disp"] = torch.maximum(
        reads["push_disp"], torch.norm(obj.data.root_link_pos_w - pre, dim=-1)
      )

  drv.hooks.append(hook)
  rec = None
  if args.video is not None:
    rec = Recorder(env, model, args.video_envs, every=2, close=args.video_close)
    drv.hooks.append(rec)
  stack = lambda key: np.stack([p[key] for p in plans])  # noqa: E731
  q = drv.q()
  phase["name"] = "approach"
  drv.move(q, stack("above"))
  drv.move(stack("above"), stack("pregrasp"))
  drv.move(stack("pregrasp"), stack("grasp"))
  phase["name"] = "close"
  drv.jaw[:] = 1.0
  drv.hold(stack("grasp"), JAW_CLOSE_S)
  flags["grasped"] = _grasped_bool(env, gripper.contact_sensor_name)
  phase["name"] = "lift"
  drv.move(stack("grasp"), stack("lift"))
  drv.move(stack("lift"), stack("H"))
  phase["name"] = "hold"
  drv.hold(stack("H"), 2.0)
  pre[:] = obj.data.root_link_pos_w
  phase["name"] = "push"
  from mjlab.utils.lab_api.math import matrix_from_quat

  for _ in range(round(1.0 / env.step_dt)):
    up = matrix_from_quat(obj.data.root_link_quat_w)[..., :, 2]
    force = torch.zeros(n_env, 1, 3, device=dev)
    force[:, 0, 2] = -args.push_n
    torque = torch.cross(half * up.unsqueeze(1), force, dim=-1)
    obj.data.write_external_wrench(force, torque, body_ids=[0])
    drv.step(stack("H"))
  obj.data.write_external_wrench(
    torch.zeros(n_env, 1, 3, device=dev),
    torch.zeros(n_env, 1, 3, device=dev),
    body_ids=[0],
  )
  phase["name"] = "after"
  drv.hold(stack("H"), 1.0)
  cmd = env.command_manager.get_term("goal_pose").command
  reads["after_dist"] = torch.norm(cmd - obj.data.root_link_pos_w, dim=-1)
  reads["after_tilt"] = mdp.axis_tilt(env, "object")
  flags["held_after"] = mdp.held_at_goal(env, **hp).bool()

  if rec is not None:
    rec.write(args.video, fps=1.0 / (2 * env.step_dt))
  rows = []
  for i in range(n_env):
    rows.append(
      {
        "env": i,
        "plan_error": errors[i],
        "block_xy": pos[i][:2].round(4).tolist(),
        "grasped": bool(flags["grasped"][i]),
        "lifted": bool(flags["lifted"][i]),
        "held_share_2s": float(
          reads["held_steps"][i] / reads["hold_steps"][i].clamp(min=1)
        ),
        "push_tilt_deg": math.degrees(float(reads["push_tilt"][i])),
        "push_disp_mm": 1e3 * float(reads["push_disp"][i]),
        "after_dist_mm": 1e3 * float(reads["after_dist"][i]),
        "after_tilt_deg": math.degrees(float(reads["after_tilt"][i])),
        "held_after_push": bool(flags["held_after"][i]),
        "arm_table": bool(flags["arm_table"][i]),
        "hand_table": bool(flags["hand_table"][i]),
        "clipped_steps": int(drv.clipped[i]),
        "resets": int(drv.resets[i]),
      }
    )
    print(json.dumps(rows[-1]), flush=True)
  ok = [r for r in rows if not r["plan_error"] and r["held_share_2s"] >= 0.8]
  summary = {
    "envs": n_env,
    "planned": sum(not r["plan_error"] for r in rows),
    "held_at_H": len(ok),
    "held_after_push": sum(r["held_after_push"] for r in rows),
    "push_tilt_max_deg": max((r["push_tilt_deg"] for r in ok), default=float("nan")),
    "push_disp_max_mm": max((r["push_disp_mm"] for r in ok), default=float("nan")),
    "arm_table": sum(r["arm_table"] for r in rows),
    "clipped_envs": sum(r["clipped_steps"] > 0 for r in rows),
    "pad_offset": pad_off,
  }
  print(json.dumps(summary, indent=1))
  env.close()
  return {"summary": summary, "rows": rows}


def main(argv=None) -> None:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("mode", choices=("hold",))
  p.add_argument("--task", default=TASK)
  p.add_argument("--num-envs", type=int, default=16)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--forced", action="store_true")
  p.add_argument("--push-n", type=float, default=10.0)
  p.add_argument("--json", type=pathlib.Path, default=None)
  p.add_argument("--video", type=pathlib.Path, default=None)
  p.add_argument("--video-envs", type=int, nargs="+", default=[0, 1, 2, 3])
  p.add_argument(
    "--video-close", action="store_true", help="a close side view on each env's block"
  )
  args = p.parse_args(argv)
  out = hold_rollout(args)
  if args.json:
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
  sys.exit(main())
