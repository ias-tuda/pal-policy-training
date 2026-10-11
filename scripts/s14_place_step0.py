"""Plan §14 step 0 (`pick-place-insert-plan.md` §14.6): the pin picked and placed on the wide pin area WP.

``reach`` (items 2 and 3): an IK map over x 0.32-0.68, y -0.14-0.36 (the robot frame, ``--step`` grid) on the
``GOAL=place`` scene's model at a table height (``TABLE_DZ``, mm). For each cell and start pose, the left arm's
contact-free IK (`scripted_pick_place.Kin`: the tool point at the goal, no arm geom touching the table, the torso, the
right arm or the parked pin) at the grasp, the lift (+5 cm) and the retreat (10 cm over the pin's top) -- the set-down
at a goal is the grasp's tool point again:

* standing: the tool point `insert.GRASP_ABOVE_TOP` over the 90 mm pin's top, the wrist at any yaw (a cross is
  28-32 mm across in every direction, so the jaw closes on it at any yaw);
* lying: the tool point ``--above-lying`` over the lying "x" pin's 28 mm top, straight over its centre; axis 0 / 45 /
  90 / 135 deg from x; the jaw closing across the axis (within `JAW_TOL_DEG`).

The IK (`_soft_ik`) holds the position exactly and the hand's vertical (and the jaw's direction) softly, from the
start pose and random seeds, and keeps the least-tilted contact-free solution. Per cell and pose: ``top`` (every axis
with the hand within `TOP_DOWN_DEG` of vertical through grasp, lift and retreat), ``tilt`` (within `MAX_TILT_DEG`),
``some`` (some axes not within it), ``none``. Item 3: the largest excursion of each arm joint from the start pose over every
solution, x 1.5 (S5.0's rule), against the WC arm's action scales.

  uv run --extra cu128 python scripts/s14_place_step0.py reach --table-dz 22 --json ../results/S14-pin-place/step0/reach_dz22.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

#: `BS12` + PS (plan §14.4), the selectors every subcommand builds its env or scene with.
PS_ENV = {
  "REWARD": "R-C-strict+M1+M2+M3+M6+J+M8",
  "DR": "DR-scene",
  "PPO": "PPO-stack",
  "SLEW": "stack",
  "BASELINE": "B",
  "SPAWN": "WP",
  "TOOL_SPEED": "none",
  "GAINS": "G20",
  "EFFORT": "CTP",
  "LAG": "FP+JSW",
  "ACTION": "abs",
  "CMD_OBS": "act",
  "TARGET": "WP",
  "EPISODE_S": "15",
  "CRITIC_OBS": "q",
  "ARM_TABLE": "AH10",
  "SHAPE": "pin",
  "GOAL": "place",
  "TABLE_DZ": "22",
}
#: The lying "x" pin's top over the table, m: (span + thickness) / 2 / sqrt 2 + the same again.
LYING_TOP = 2 * (0.032 + 0.008) / 2 / math.sqrt(2)
STANDING_TOP = 0.090
LIFT = 0.05
RETREAT = 0.10


def _set_env(**over) -> None:
  for k, v in {**PS_ENV, **over}.items():
    os.environ[k] = str(v)


#: A pose counts as top-down within this tilt of the hand from vertical, as reachable within `MAX_TILT_DEG` (plan
#: §14.1 *Rule*: "the hand within 35 deg of vertical"); a lying pin's jaw must close within `JAW_TOL_DEG` of across
#: its axis.
TOP_DOWN_DEG = 5.0
MAX_TILT_DEG = 35.0
JAW_TOL_DEG = 10.0
#: Seeds per IK: the start pose and draws around it (0.3-1.2 rad).
SEEDS = 12


def _soft_ik(kin, goal, axis, seed, iters=400, w=0.3):
  """Damped least squares on the tool point's position (exact) and, softly (weight ``w``), the hand's z straight down
  and -- with a lying pin's ``axis`` (horizontal unit vector) -- the hand's x (the jaw's closing direction) across it;
  the wrist yaw is otherwise free (a standing cross is gripped at any yaw). Returns (q, position error m, hand tilt
  deg, jaw error deg)."""
  import mujoco

  m = kin.model
  jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
  zdes = np.array([0.0, 0.0, -1.0])
  q = seed.copy()
  for _ in range(iters):
    p, r = kin.fk(q)
    mujoco.mj_jacSite(m, kin.data, jacp, jacr, kin.tcp)
    jr = jacr[:, kin.dofs]
    rows = [jacp[:, kin.dofs], w * jr]
    err = [goal - p, w * np.cross(r[:, 2], zdes)]
    if axis is not None:
      rows.append(w * (np.cross(r[:, 0], axis) @ jr)[None, :])
      err.append(np.array([-w * float(r[:, 0] @ axis)]))
    jac, e = np.vstack(rows), np.concatenate(err)
    pinv = jac.T @ np.linalg.inv(jac @ jac.T + 1e-3 * np.eye(len(e)))
    # What the task leaves free is spent pulling toward the start pose (`Kin.ik`'s rule): the least excursion, what
    # the action scales are sized from.
    dq = pinv @ e + (np.eye(len(q)) - pinv @ jac) @ (0.1 * (kin.q_default - q))
    q = np.clip(q + np.clip(dq, -0.2, 0.2), kin.lo, kin.hi)
  p, r = kin.fk(q)
  tilt = math.degrees(math.acos(float(np.clip(-r[2, 2], -1.0, 1.0))))
  jaw = (
    0.0
    if axis is None
    else math.degrees(math.asin(min(1.0, abs(float(r[:, 0] @ axis)))))
  )
  return q, float(np.linalg.norm(goal - p)), tilt, jaw


def _best(kin, goal, axis, seeds):
  """The contact-free solution (position within 2 mm, the jaw within `JAW_TOL_DEG`) with the hand held down hardest:
  the vertical's weight is relaxed 0.3 -> 0.1 -> 0.03 only where the stricter one cannot reach; the least-tilted
  solution at the first weight that has one, or None."""
  for w in (0.3, 0.1, 0.03):
    best = None
    for s0 in seeds:
      q, err, tilt, jaw = _soft_ik(kin, goal, axis, s0, w=w)
      if err > 2e-3 or jaw > JAW_TOL_DEG or kin.in_collision(q):
        continue
      if best is None or tilt < best[1]:
        best = (q, tilt)
        if tilt < 1.0:
          return best
    if best is not None:
      return best
  return None


def reach(args: argparse.Namespace) -> dict:
  _set_env(TABLE_DZ=args.table_dz)
  import mujoco
  import scripted_pick_place as scripted
  from pal_mjlab.tasks.manipulation.pick_place import insert as ins
  from scene_model import compile_scene

  model = compile_scene(scripted.TASK)
  kin = scripted.Kin.from_model(model)
  if args.box:
    # Item 3: the IK held inside the named arm's action box -- the start pose +/- its action scale --, so a cell
    # reached here is reached through that arm's actions.
    from pal_mjlab.tasks.manipulation.pick_place.kangaroo.env_cfg import SPAWN_ROBOTS

    sc = SPAWN_ROBOTS[args.box].action_scale
    half = np.array([sc[f"arm_left_{j}_joint"] for j in range(1, 8)])
    kin.lo = np.maximum(kin.lo, kin.q_default - half)
    kin.hi = np.minimum(kin.hi, kin.q_default + half)
  table = model.geom("table/table_top").id
  d = mujoco.MjData(model)
  mujoco.mj_forward(model, d)
  top = float(d.geom_xpos[table][2] + model.geom_size[table][2])
  rng = np.random.default_rng(args.seed)
  xs = np.round(np.arange(args.x[0], args.x[1] + 1e-9, args.step), 3)
  ys = np.round(np.arange(args.y[0], args.y[1] + 1e-9, args.step), 3)
  cells = [(float(x), float(y)) for x in xs for y in ys][args.shard :: args.shards]
  out = {
    "table_top": top,
    "above_lying": args.above_lying,
    "box": args.box,
    "cells": [],
  }
  if args.json.exists():
    out = json.loads(args.json.read_text())
  done = {(c["x"], c["y"]) for c in out["cells"]}
  t0 = time.time()
  for n, (x, y) in enumerate(cells):
    if (x, y) in done:
      continue
    row = {"x": x, "y": y}
    sols = []
    for pose in ("standing", "lying"):
      if pose == "standing":
        z = top + STANDING_TOP + ins.GRASP_ABOVE_TOP
        zs = (z, z + LIFT, top + STANDING_TOP + RETREAT)
        axes = [None]
      else:
        z = top + LYING_TOP + args.above_lying
        zs = (z, z + LIFT, top + LYING_TOP + RETREAT)
        axes = [
          np.array([math.cos(math.radians(a)), math.sin(math.radians(a)), 0.0])
          for a in (0.0, 45.0, 90.0, 135.0)
        ]
      tilts = []
      for axis in axes:
        seeds = [kin.q_default] + [
          np.clip(kin.q_default + rng.normal(0.0, 0.3 + 0.08 * k, 7), kin.lo, kin.hi)
          for k in range(SEEDS - 1)
        ]
        chain, worst = [], 0.0
        for k, zz in enumerate(zs):
          got = _best(
            kin,
            np.array([x, y, zz]),
            axis,
            seeds if k == 0 else [chain[-1]] + seeds[:3],
          )
          if got is None:
            chain = None
            break
          chain.append(got[0])
          worst = max(worst, got[1])
        tilts.append(None if chain is None else round(worst, 1))
        if chain is not None and worst <= MAX_TILT_DEG:
          sols.extend(chain)
      row[pose] = tilts
      ok = [t for t in tilts if t is not None and t <= MAX_TILT_DEG]
      row[pose + "_class"] = (
        "none"
        if not ok
        else "some"
        if len(ok) < len(tilts)
        else "top"
        if max(ok) <= TOP_DOWN_DEG
        else "tilt"
      )
    if sols:
      dev = np.abs(np.asarray(sols) - kin.q_default).max(0)
      row["max_dev"] = [round(float(v), 4) for v in dev]
    out["cells"].append(row)
    if n % 10 == 0:
      args.json.parent.mkdir(parents=True, exist_ok=True)
      args.json.write_text(json.dumps(out))
      print(f"{n + 1} / {len(cells)} cells, {time.time() - t0:.0f} s", flush=True)
  args.json.write_text(json.dumps(out))
  return out


def summarize(args: argparse.Namespace) -> None:
  """Item 2's table and item 3's scales from one or more ``reach`` JSONs (shards merged)."""
  from pal_mjlab.tasks.manipulation.robots.kangaroo import (
    KANGAROO_GRIPPERS_LEFT_ARM_WC,
  )

  cells, top = [], None
  for f in args.files:
    r = json.loads(pathlib.Path(f).read_text())
    cells += r["cells"]
    top = r["table_top"]
  wp = [c for c in cells if 0.36 <= c["x"] <= 0.64 and -0.10 <= c["y"] <= 0.32]
  print(f"table top {top:.3f} m; {len(cells)} cells, {len(wp)} on WP")
  for pose in ("standing", "lying"):
    for name, part in (("grid", cells), ("WP", wp)):
      k = {
        c: sum(1 for x in part if x[pose + "_class"] == c)
        for c in ("top", "tilt", "some", "none")
      }
      print(f"  {pose:8s} {name:4s}: " + "  ".join(f"{c} {v}" for c, v in k.items()))
    bad = sorted(
      (c["x"], c["y"], c[pose + "_class"])
      for c in wp
      if c[pose + "_class"] in ("some", "none")
    )
    if bad:
      print(
        f"    WP cells not every {pose} case reaches: {bad[:40]}{' ...' if len(bad) > 40 else ''}"
      )
  both_none = [
    (c["x"], c["y"])
    for c in wp
    if c["standing_class"] == "none" and c["lying_class"] == "none"
  ]
  print(f"  WP cells no pose reaches (the cut rule's (a)): {both_none}")
  scales = KANGAROO_GRIPPERS_LEFT_ARM_WC.action_scale
  dev = np.max([c["max_dev"] for c in wp if "max_dev" in c], axis=0)
  print("  item 3: joint, max |dq| over WP's solutions, x1.5, WC scale")
  for j, v in enumerate(dev):
    s = scales[f"arm_left_{j + 1}_joint"]
    flag = "" if 1.5 * v <= s + 1e-9 else "  > WC"
    print(f"    arm {j + 1}: {v:.3f}  {1.5 * v:.3f}  {s:.2f}{flag}")


# -- The env driver (items 5-7) --------------------------------------------------------------------------


def _make_env(n: int, device: str, seed: int = 0, **over):
  """The ``GOAL=place`` env (`PS_ENV` + ``over``) with ``n`` worlds and a long episode."""
  _set_env(**over)
  from mjlab.envs import ManagerBasedRlEnv
  from pal_mjlab.tasks.manipulation.pick_place.kangaroo.env_cfg import (
    kangaroo_pick_place_env_cfg,
  )

  cfg = kangaroo_pick_place_env_cfg()
  cfg.scene.num_envs = n
  cfg.seed = seed
  cfg.episode_length_s = 120.0
  return ManagerBasedRlEnv(cfg=cfg, device=device)


class Driver:
  """Joint targets for the left arm and its jaw through the env's own action term (``ACTION=abs``: a raw action is
  ``(q - offset) / scale``, clipped to [-1, 1]), as `scripted_pick_place.rollout` sends them; the arm's slew limit
  is respected by spreading every move over enough steps."""

  def __init__(self, env):
    import scripted_pick_place as scripted
    import torch
    from pal_mjlab.tasks.manipulation.robots.kangaroo import (
      KANGAROO_GRIPPERS_LEFT_ARM,
    )

    self.env, self.torch = env, torch
    term = env.action_manager.get_term("joint_pos")
    names = [n.split("/", 1)[-1] for n in term.target_names]
    self.arm = [names.index(j) for j in scripted.ARM_JOINTS]
    self.jaw = names.index(scripted.JAW_JOINT)
    n = env.num_envs
    self.scale = (
      term.scale
      if isinstance(term.scale, torch.Tensor)
      else torch.full((n, len(names)), float(term.scale), device=env.device)
    )
    self.offset = term.offset.clone()
    self.target = self.offset.clone()
    self.jaw_closed = KANGAROO_GRIPPERS_LEFT_ARM.gripper("left").closed_pos
    self.slew = (
      term._rate[0, self.arm].cpu().numpy() if hasattr(term, "_rate") else None
    )
    self.kin = scripted.Kin.from_model(env.sim.mj_model)
    # The IK inside the action box (the start pose +/- the arm's scales), so that no planned target is clipped.
    lo = (self.offset[0, self.arm] - self.scale[0, self.arm]).cpu().numpy()
    hi = (self.offset[0, self.arm] + self.scale[0, self.arm]).cpu().numpy()
    self.kin.lo = np.maximum(self.kin.lo, lo + 1e-3)
    self.kin.hi = np.minimum(self.kin.hi, hi - 1e-3)
    self.clipped = 0
    self.hook = None

  def q_arm(self):
    return self.target[:, self.arm].cpu().numpy()

  def step(self, q=None, jaw=None):
    torch = self.torch
    if q is not None:
      self.target[:, self.arm] = torch.as_tensor(
        q, dtype=torch.float32, device=self.env.device
      )
    if jaw is not None:
      self.target[:, self.jaw] = torch.as_tensor(
        jaw, dtype=torch.float32, device=self.env.device
      )
    raw = (self.target - self.offset) / self.scale
    self.clipped += int((raw.abs() > 1.0 + 1e-6).any(dim=-1).sum())
    with torch.inference_mode():
      self.env.step(raw.clamp(-1.0, 1.0))
    if self.hook is not None:
      self.hook(self)

  def move(self, q, jaw=None, steps: int = 25):
    """Interpolate the arm's targets to ``q`` (n, 7) over at least ``steps`` steps (more if the slew needs)."""
    a = self.q_arm()
    q = np.asarray(q, dtype=float)
    need = steps
    if self.slew is not None:
      need = max(steps, int(np.ceil((np.abs(q - a) / self.slew).max() - 1e-9)))
    for k in range(1, need + 1):
      self.step(a + (q - a) * (k / need), jaw)

  def hold(self, steps: int, jaw=None):
    for _ in range(steps):
      self.step(None, jaw)


def _table_tops(env):
  import torch
  from mjlab.managers.scene_entity_config import SceneEntityCfg
  from pal_mjlab.tasks.manipulation.mdp.events import surface_top_z

  with torch.inference_mode():
    return surface_top_z(env, SceneEntityCfg("table")).cpu().numpy()


def _put_pin(env, pos_env: np.ndarray, quat: np.ndarray, vel: np.ndarray | None = None):
  """Teleport each world's pin to ``pos_env`` (n, 3; the env frame) with orientation ``quat`` (n, 4) and, optionally,
  linear velocity ``vel`` (n, 3)."""
  import torch

  pin = env.scene["object"]
  dev = env.device
  pos = (
    torch.as_tensor(pos_env, dtype=torch.float32, device=dev) + env.scene.env_origins
  )
  q = torch.as_tensor(quat, dtype=torch.float32, device=dev)
  v = torch.zeros(env.num_envs, 6, device=dev)
  if vel is not None:
    v[:, :3] = torch.as_tensor(vel, dtype=torch.float32, device=dev)
  pin.write_root_link_pose_to_sim(torch.cat([pos, q], dim=-1))
  pin.write_root_link_velocity_to_sim(v)


def _quat_np(fn, *args) -> np.ndarray:
  import torch

  return fn(*[torch.as_tensor(np.asarray(a, dtype=np.float32)) for a in args]).numpy()


def _yaw_q(yaw: np.ndarray) -> np.ndarray:
  from mjlab.utils.lab_api.math import quat_from_euler_xyz

  z = np.zeros_like(yaw)
  return _quat_np(quat_from_euler_xyz, z, z, yaw)


def _lying_q(axis_yaw: np.ndarray, roll: np.ndarray) -> np.ndarray:
  """A lying pin, its axis at ``axis_yaw``, turned ``roll`` about its axis (pi/4: the "x" pose, 0: the "+")."""
  from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

  z = np.zeros_like(axis_yaw)
  r = _quat_np(quat_from_euler_xyz, z, z, roll)
  t = _quat_np(quat_from_euler_xyz, z, np.full_like(z, math.pi / 2), z)
  y = _quat_np(quat_from_euler_xyz, z, z, axis_yaw)
  return _quat_np(quat_mul, y, _quat_np(quat_mul, t, r))


def _pin_state(env):
  """Per world: the pin's centre (env frame), its lowest point over the table, standing or not, speed. numpy."""
  import torch
  from pal_mjlab.tasks.manipulation import mdp

  with torch.inference_mode():
    pin = env.scene["object"]
    below, above = mdp.pin_extent_z(pin.data.root_link_quat_w)
    pos = (pin.data.root_link_pos_w - env.scene.env_origins).cpu().numpy()
    tops = _table_tops(env)
    return {
      "pos": pos,
      "z_min": pos[:, 2]
      + env.scene.env_origins[:, 2].cpu().numpy()
      - below.cpu().numpy()
      - tops,
      "top_over": above.cpu().numpy(),
      "standing": mdp.pin_standing(pin.data.root_link_quat_w).cpu().numpy(),
      "speed": torch.norm(pin.data.root_link_lin_vel_w, dim=-1).cpu().numpy(),
    }


def _grasp_flags(env):
  """(grasped by both pads, the gripper touching the table) per world. numpy."""
  import torch
  from pal_mjlab.tasks.manipulation.mdp.observations import _grasped_bool
  from pal_mjlab.tasks.manipulation.robots.kangaroo import KANGAROO_GRIPPERS_LEFT_ARM

  g = KANGAROO_GRIPPERS_LEFT_ARM.gripper("left")
  with torch.inference_mode():
    grasped = _grasped_bool(env, g.contact_sensor_name).cpu().numpy()
    found = env.scene[g.surface_sensor_name].data.found
    table = found.bool().reshape(env.num_envs, -1).any(-1).cpu().numpy()
  return grasped, table


def _jaw_q(env):
  import scripted_pick_place as scripted
  import torch

  robot = env.scene["robot"]
  j = robot.joint_names.index(scripted.JAW_JOINT)
  with torch.inference_mode():
    return robot.data.joint_pos[:, j].cpu().numpy()


def _tcp_env(env):
  import torch

  robot = env.scene["robot"]
  sid = robot.site_names.index("tcp_left")
  with torch.inference_mode():
    return (robot.data.site_pos_w[:, sid] - env.scene.env_origins).cpu().numpy()


def bench(args: argparse.Namespace) -> dict:
  """Item 5: the lying grasp on mujoco_warp with the real gripper (G20, DR0, the 0.837 m table). One world per case:
  pose ("x" / "+"), the pin's axis 0 / 45 / 90 deg off the jaw's closing normal, the tool point 10 / 15 / 20 / 25
  mm over the pin's top; two standing controls (yaw 0 / 45 deg, 20 mm over the top, §12.1's bench). The hand comes
  down open over the pin at (0.50, 0.18), the jaw target is ramped open -> closed over 2 s, held 0.5 s, the pin
  lifted 5 cm, set back and the jaw ramped open over 1 s. Read: the jaw angle at the first step both pads touch the
  pin (touch), at the end of the hold (hold), the fingers on the table, the pin lifted and how far it slipped, the
  angle at which the pads let go on opening."""
  cases = []
  for pose, roll in (("x", math.pi / 4), ("+", 0.0)):
    for off in (0.0, 45.0, 90.0):
      for h in (0.010, 0.015, 0.020, 0.025):
        cases.append({"pose": pose, "roll": roll, "off_deg": off, "above": h})
  for yaw in (0.0, 45.0):
    cases.append({"pose": "standing", "yaw_deg": yaw, "above": 0.020})
  n = len(cases)
  env = _make_env(n, args.device, DR="DR0", LAG="JS")
  import torch

  with torch.inference_mode():
    env.reset()
  drv = Driver(env)
  kin = drv.kin
  top = float(_table_tops(env)[0])
  xy = np.array([0.50, 0.18])
  # The jaw's closing direction for a lying pin: across its axis; the pin's axis set off that by `off`.
  axis_yaw = np.radians(90.0)
  quats, pos = [], []
  for c in cases:
    if c["pose"] == "standing":
      quats.append(_yaw_q(np.array([math.radians(c["yaw_deg"])]))[0])
      pos.append([xy[0], xy[1], top + 0.045])
    else:
      yaw = axis_yaw + math.radians(c["off_deg"])
      quats.append(_lying_q(np.array([yaw]), np.array([c["roll"]]))[0])
      below = (0.032 + 0.008) / 2 / math.sqrt(2) if c["pose"] == "x" else 0.016
      pos.append([xy[0], xy[1], top + below])
  _put_pin(env, np.array(pos), np.array(quats))
  drv.hold(5)
  st = _pin_state(env)
  rng = np.random.default_rng(0)
  plans = []
  for i, c in enumerate(cases):
    top_z = st["pos"][i, 2] + st["top_over"][i]
    goal = np.array([xy[0], xy[1], top_z + c["above"]])
    jaw_axis = None
    if c["pose"] != "standing":
      jaw_axis = np.array([math.cos(axis_yaw), math.sin(axis_yaw), 0.0])
    seeds = [kin.q_default] + [
      np.clip(kin.q_default + rng.normal(0, 0.3, 7), kin.lo, kin.hi) for _ in range(5)
    ]
    g = _best(kin, goal, jaw_axis, seeds)
    pre = _best(kin, goal + np.array([0, 0, 0.08]), jaw_axis, [g[0]] if g else seeds)
    up = _best(kin, goal + np.array([0, 0, 0.05]), jaw_axis, [g[0]] if g else seeds)
    if g is None or pre is None or up is None:
      raise RuntimeError(f"no IK for bench case {c}")
    plans.append((pre[0], g[0], up[0]))
  pre, grasp, up = (np.array([p[k] for p in plans]) for k in range(3))
  drv.move(pre, jaw=0.0, steps=100)
  drv.hold(25, jaw=0.0)
  # The arm sags under its load at G20 (~2 cm): re-plan the grasp and the lift for the tool point's measured offset
  # from its planned place at the pre-grasp, as `scripted_pick_place.replan_place` does.
  bias = _tcp_env(env) - np.array([kin.fk(q)[0] for q in pre])
  for i, c in enumerate(cases):
    jaw_axis = (
      None
      if c["pose"] == "standing"
      else np.array([math.cos(axis_yaw), math.sin(axis_yaw), 0.0])
    )
    goal = kin.fk(grasp[i])[0] - bias[i]
    g = _best(kin, goal, jaw_axis, [grasp[i]])
    u = _best(kin, goal + np.array([0, 0, 0.05]), jaw_axis, [up[i]])
    if g is not None and u is not None:
      grasp[i], up[i] = g[0], u[0]
  drv.move(grasp, jaw=0.0, steps=50)
  drv.hold(25, jaw=0.0)
  tcp0 = _tcp_env(env)
  pin0 = _pin_state(env)["pos"]
  ramp = 100
  q_tr, g_tr, t_tr = [], [], []
  for k in range(1, ramp + 1):
    drv.step(None, jaw=drv.jaw_closed * k / ramp)
    gr, tb = _grasp_flags(env)
    q_tr.append(_jaw_q(env))
    g_tr.append(gr)
    t_tr.append(tb)
  drv.hold(25, jaw=drv.jaw_closed)
  q_hold = _jaw_q(env)
  g_hold, t_hold = _grasp_flags(env)
  pin_hold = _pin_state(env)["pos"]
  drv.move(up, jaw=drv.jaw_closed, steps=50)
  drv.hold(25, jaw=drv.jaw_closed)
  tcp1, st1 = _tcp_env(env), _pin_state(env)
  g_up, _ = _grasp_flags(env)
  drv.move(grasp, jaw=drv.jaw_closed, steps=50)
  drv.hold(10, jaw=drv.jaw_closed)
  q_tr2, g_tr2 = [], []
  for k in range(ramp // 2, -1, -1):
    drv.step(None, jaw=drv.jaw_closed * k / (ramp // 2))
    gr, _ = _grasp_flags(env)
    q_tr2.append(_jaw_q(env))
    g_tr2.append(gr)
  q_tr, g_tr, t_tr = np.array(q_tr), np.array(g_tr), np.array(t_tr)
  q_tr2, g_tr2 = np.array(q_tr2), np.array(g_tr2)
  rows = []
  for i, c in enumerate(cases):
    k = np.nonzero(g_tr[:, i])[0]
    k2 = np.nonzero(~g_tr2[:, i])[0]
    rows.append(
      {
        **{kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in c.items()},
        "touch_q": None if not len(k) else round(float(q_tr[k[0], i]), 3),
        "hold_q": round(float(q_hold[i]), 3),
        "held": bool(g_hold[i]),
        "fingers_on_table_close": bool(t_tr[:, i].any()),
        "fingers_on_table_hold": bool(t_hold[i]),
        "pin_moved_at_close_mm": round(
          1e3 * float(np.linalg.norm(pin_hold[i, :2] - pin0[i, :2])), 1
        ),
        "tcp_over_table_mm": round(1e3 * float(tcp0[i, 2] - top), 1),
        "lifted": bool(g_up[i]) and float(st1["z_min"][i]) > 0.03,
        "slip_mm": round(
          1e3 * float((tcp1[i, 2] - tcp0[i, 2]) - (st1["pos"][i, 2] - pin_hold[i, 2])),
          1,
        ),
        "release_q": None if not len(k2) else round(float(q_tr2[k2[0], i]), 3),
      }
    )
  out = {
    "table_top": top,
    "jaw_closed": drv.jaw_closed,
    "clipped_steps": drv.clipped,
    "cases": rows,
  }
  args.json.parent.mkdir(parents=True, exist_ok=True)
  args.json.write_text(json.dumps(out, indent=1))
  hdr = (
    "pose  off  above  touch  hold  held table(close/hold) moved  lifted slip  release"
  )
  print(hdr)
  for r in rows:
    print(
      f"{r['pose']:8s} {r.get('off_deg', r.get('yaw_deg', 0)):4.0f} {1e3 * r['above']:5.0f}  {r['touch_q']!s:6s} "
      f"{r['hold_q']:.3f} {r['held']!s:5s} {r['fingers_on_table_close']!s:5s}/{r['fingers_on_table_hold']!s:5s} "
      f"{r['pin_moved_at_close_mm']:5.1f} {r['lifted']!s:5s} {r['slip_mm']:5.1f} {r['release_q']!s}"
    )
  return out


def rest(args: argparse.Namespace) -> dict:
  """Item 6: the resting table on mujoco_warp (the training physics; the plan names CPU MuJoCo first). One world per
  release: a standing pin let go with its foot 0 / 2 / 5 / 10 mm over the table, tilted 0 / 3 / 5 / 8 / 12 deg (leaning
  the way it moves), moving 0 / 2 / 5 / 10 / 20 cm/s horizontally; a lying pin ("x") let go 0 / 5 / 10 / 20 / 30 mm up
  at the same speeds; the table's friction 0.8 (the scene's) or 0.4. The arm at home, the pin at (0.56, 0.02) +/-. After
  1.5 s: the centre's distance from the release point, the end pose, still within 2 cm."""
  cases = []
  for mu in (0.8, 0.4):
    for h in (0.0, 0.002, 0.005, 0.010):
      for tilt in (0.0, 3.0, 5.0, 8.0, 12.0):
        for v in (0.0, 0.02, 0.05, 0.10, 0.20):
          cases.append({"pose": "standing", "mu": mu, "h": h, "tilt_deg": tilt, "v": v})
    for h in (0.0, 0.005, 0.010, 0.020, 0.030):
      for v in (0.0, 0.02, 0.05, 0.10, 0.20):
        cases.append({"pose": "lying", "mu": mu, "h": h, "tilt_deg": 0.0, "v": v})
  n = len(cases)
  env = _make_env(n, args.device, DR="DR-scene", LAG="JS")
  import torch
  from mjlab.utils.lab_api.math import quat_from_euler_xyz

  with torch.inference_mode():
    env.reset()
  drv = Driver(env)
  model = env.sim.model
  table_gid = [
    i
    for i in range(env.sim.mj_model.ngeom)
    if env.sim.mj_model.geom(i).name == "table/table_top"
  ][0]
  with torch.inference_mode():
    fr = model.geom_friction
    if fr.shape[0] != n:
      raise RuntimeError(
        "geom_friction is not per world; build with a DR rung that expands it"
      )
    fr[:, table_gid, 0] = torch.tensor([c["mu"] for c in cases], device=env.device)
  drv.hold(2)
  tops = _table_tops(env)
  # The pin at (0.56, 0.02) about the shifted table, from the env origin.
  table = env.scene["table"]
  tb = table.indexing.body_ids[table.body_names.index("table")]
  with torch.inference_mode():
    shift = (env.sim.data.xpos[:, tb, :2] - env.scene.env_origins[:, :2]).cpu().numpy()
  from pal_mjlab.tasks.manipulation import scene

  shift = shift - np.asarray(scene.table_pos(0.837)[:2])
  pos, quats, vel = [], [], []
  for i, c in enumerate(cases):
    x, y = 0.56 + shift[i, 0], 0.02 + shift[i, 1]
    if c["pose"] == "standing":
      t = math.radians(c["tilt_deg"])
      # Leaning +x (the way it moves): rotate about +y by t; the foot's lowest corner `h` over the table.
      q = _quat_np(quat_from_euler_xyz, np.zeros(1), np.full(1, t), np.zeros(1))[0]
      below = 0.045 * math.cos(t) + math.hypot(0.016, 0.004) * math.sin(t)
    else:
      q = _lying_q(np.array([math.radians(90.0)]), np.array([math.pi / 4]))[0]
      below = (0.032 + 0.008) / 2 / math.sqrt(2)
    pos.append([x, y, tops[i] - env.scene.env_origins[i, 2].item() + below + c["h"]])
    quats.append(q)
    vel.append([c["v"], 0.0, 0.0])
  pos = np.array(pos)
  _put_pin(env, pos, np.array(quats), np.array(vel))
  drv.hold(75)
  st = _pin_state(env)
  rows = []
  for i, c in enumerate(cases):
    d = float(np.linalg.norm(st["pos"][i, :2] - pos[i, :2]))
    rows.append(
      {
        **c,
        "moved_mm": round(1e3 * d, 1),
        "end": "standing" if st["standing"][i] else "lying",
        "rested": bool(abs(st["z_min"][i]) < 0.003 and st["speed"][i] < 0.02),
        "within_2cm": bool(d < 0.02),
      }
    )
  out = {"cases": rows}
  args.json.parent.mkdir(parents=True, exist_ok=True)
  args.json.write_text(json.dumps(out, indent=1))
  for mu in (0.8, 0.4):
    print(
      f"== table friction {mu}: standing, share within 2 cm (end standing) by foot height x tilt; columns speed 0/2/5/10/20 cm/s"
    )
    for h in (0.0, 0.002, 0.005, 0.010):
      for tilt in (0.0, 3.0, 5.0, 8.0, 12.0):
        r = [
          x
          for x in rows
          if x["pose"] == "standing"
          and x["mu"] == mu
          and x["h"] == h
          and x["tilt_deg"] == tilt
        ]
        cells = " ".join(
          f"{'Y' if x['within_2cm'] else '.'}{'s' if x['end'] == 'standing' else 'l'}{x['moved_mm']:5.1f}"
          for x in r
        )
        print(f"  h {1e3 * h:4.0f} mm tilt {tilt:4.0f}:  {cells}")
    print(
      f"== table friction {mu}: lying, by drop height; columns speed 0/2/5/10/20 cm/s"
    )
    for h in (0.0, 0.005, 0.010, 0.020, 0.030):
      r = [x for x in rows if x["pose"] == "lying" and x["mu"] == mu and x["h"] == h]
      cells = " ".join(
        f"{'Y' if x['within_2cm'] else '.'}{'s' if x['end'] == 'standing' else 'l'}{x['moved_mm']:5.1f}"
        for x in r
      )
      print(f"  h {1e3 * h:4.0f} mm:  {cells}")
  return out


def scripted(args: argparse.Namespace) -> dict:
  """Item 7 (L1, recorded, not gated): the scripted pick-and-place at the stack speed, ``LAG=JS``, DR-scene, 0.837 m,
  the WP arm, both start poses. Worlds 0-15: the pin at WP's four corners and four edge midpoints, standing then
  lying (axis 0 / 45 / 90 / 135 deg in turn), the goal the reset's; the rest the reset's own draws. Per world, IK
  inside the action box (`Driver`): over the pin's grasp point (a standing pin 20 mm over its top, any wrist yaw; a
  lying pin ``--above-lying`` over its top, the jaw across its axis), down, the jaw closed over 1 s, up 10 cm, over the
  goal, down until the pin's lowest point is 2 mm over the table with its centre over the goal (closed loop on the
  pin's measured pose), the jaw opened, up 10 cm, home; then 1.5 s. Read: plan errors, grasped, lifted, rested
  within 2 cm of the goal at the end (SUCC without the hand-off), the end pose, topples, clipped steps, the time."""
  import torch
  from pal_mjlab.tasks.manipulation import mdp
  from pal_mjlab.tasks.manipulation.pick_place import place_pin as pp

  n = args.num_envs
  env = _make_env(n, args.device, seed=args.seed, LAG="JS")
  with torch.inference_mode():
    env.reset()
  drv = Driver(env)
  kin = drv.kin
  origins = env.scene.env_origins.cpu().numpy()
  table = env.scene["table"]
  tb = table.indexing.body_ids[table.body_names.index("table")]
  from pal_mjlab.tasks.manipulation import scene

  with torch.inference_mode():
    shift = (env.sim.data.xpos[:, tb, :2] - env.scene.env_origins[:, :2]).cpu().numpy()
  shift = shift - np.asarray(scene.table_pos(0.837)[:2])
  tops = _table_tops(env)
  forced = []
  for x, y in (
    (0.36, -0.10),
    (0.36, 0.32),
    (0.64, -0.10),
    (0.64, 0.32),
    (0.36, 0.11),
    (0.64, 0.11),
    (0.50, -0.10),
    (0.50, 0.32),
  ):
    forced.append((x, y))
  st0 = _pin_state(env)
  pos = st0["pos"].copy()
  quats = env.scene["object"].data.root_link_quat_w.cpu().numpy().copy()
  kinds = ["reset"] * n
  for k in range(min(16, n)):
    x, y = forced[k % 8]
    x, y = x + shift[k, 0], y + shift[k, 1]
    if k < 8:
      quats[k] = _yaw_q(np.array([0.3]))[0]
      pos[k] = [x, y, tops[k] - origins[k, 2] + 0.045]
      kinds[k] = "corner_standing"
    else:
      quats[k] = _lying_q(
        np.array([math.radians(45.0 * (k % 4))]), np.array([math.pi / 4])
      )[0]
      pos[k] = [x, y, tops[k] - origins[k, 2] + (0.032 + 0.008) / 2 / math.sqrt(2)]
      kinds[k] = "corner_lying"
  _put_pin(env, pos, quats)
  drv.hold(5)
  # Re-draw the goals of the forced worlds clear of their pins (the command drew them for the reset's pins).
  cmd = env.command_manager.get_term("goal_pose")
  with torch.inference_mode():
    ids = torch.arange(min(16, n), device=env.device)
    if len(ids):
      cmd.offsets[ids] = cmd._draw_clear_of_object(ids)
  goal = cmd.goal_surface.cpu().numpy() - origins
  st = _pin_state(env)
  quat = env.scene["object"].data.root_link_quat_w
  with torch.inference_mode():
    axis_w = mdp.pin_axis_canonical_w(env, quat).cpu().numpy()
  standing0 = st["standing"].copy()
  rng = np.random.default_rng(args.seed)
  plan_err = [None] * n
  tcp_g = np.zeros((n, 3))
  jaw_axes = [None] * n
  q_pre, q_grasp, q_up = (np.tile(kin.q_default, (n, 1)) for _ in range(3))

  def seeds_from(q):
    return [q] + [np.clip(q + rng.normal(0, 0.3, 7), kin.lo, kin.hi) for _ in range(5)]

  for i in range(n):
    top_z = st["pos"][i, 2] + st["top_over"][i]
    if standing0[i]:
      g = st["pos"][i] + np.array([0, 0, 0.045 + 0.020])
      ax = None
    else:
      g = np.array([st["pos"][i, 0], st["pos"][i, 1], top_z + args.above_lying])
      ax = axis_w[i] / max(np.linalg.norm(axis_w[i][:2]), 1e-9)
      ax = np.array([ax[0], ax[1], 0.0])
    jaw_axes[i] = ax
    tcp_g[i] = g
    a = _best(kin, g + np.array([0, 0, 0.10]), ax, seeds_from(kin.q_default))
    b = _best(kin, g, ax, [a[0]] + seeds_from(a[0])[1:]) if a else None
    if a is None or b is None:
      plan_err[i] = "grasp"
      continue
    q_pre[i], q_grasp[i], q_up[i] = a[0], b[0], a[0]
  ok = np.array([e is None for e in plan_err])
  hold = lambda: np.where(ok[:, None], 0, 0)  # noqa: E731
  del hold
  drv.move(np.where(ok[:, None], q_pre, kin.q_default), jaw=0.0, steps=100)
  drv.hold(25, jaw=0.0)
  # The arm's sag (~2 cm at G20), measured at the pre-grasp: the grasp re-planned for it.
  bias = _tcp_env(env) - np.array([kin.fk(q)[0] for q in q_pre])
  for i in range(n):
    if ok[i]:
      r = _best(kin, tcp_g[i] - bias[i], jaw_axes[i], [q_grasp[i]])
      if r is not None:
        q_grasp[i] = r[0]
  drv.move(np.where(ok[:, None], q_grasp, q_pre), jaw=0.0, steps=50)
  drv.hold(10, jaw=0.0)
  for k in range(1, 51):
    drv.step(None, jaw=np.where(ok, drv.jaw_closed * k / 50, 0.0))
  st_close = _pin_state(env)
  drv.hold(15, jaw=np.where(ok, drv.jaw_closed, 0.0))
  grasped, _ = _grasp_flags(env)
  drv.move(
    np.where(ok[:, None], q_up, drv.q_arm()),
    jaw=np.where(ok, drv.jaw_closed, 0.0),
    steps=50,
  )
  drv.hold(10, jaw=np.where(ok, drv.jaw_closed, 0.0))
  st_l = _pin_state(env)
  lifted = _grasp_flags(env)[0] & (st_l["z_min"] > 0.03)
  # Carry: the tool point over the goal at the lift height, the hand's yaw kept (the jaw across a lying axis).
  tcp = _tcp_env(env)
  q_carry = drv.q_arm().copy()
  for i in range(n):
    if not ok[i]:
      continue
    off = tcp[i, :2] - st_l["pos"][i, :2]
    c = np.array([goal[i, 0] + off[0], goal[i, 1] + off[1], tcp[i, 2]])
    r = _best(kin, c, jaw_axes[i], seeds_from(q_up[i]))
    if r is None:
      plan_err[i] = "carry"
      ok[i] = False
      continue
    q_carry[i] = r[0]
  jaw_c = np.where(ok, drv.jaw_closed, 0.0)
  drv.move(q_carry, jaw=jaw_c, steps=100)
  drv.hold(25, jaw=jaw_c)
  # Set down, closed loop on the pin's measured pose: its lowest point to 2 mm, its centre over the goal.
  q_place = drv.q_arm().copy()
  for _ in range(3):
    st_c, tcp = _pin_state(env), _tcp_env(env)
    for i in range(n):
      if not ok[i]:
        continue
      tgt = tcp[i] + np.array(
        [
          goal[i, 0] - st_c["pos"][i, 0],
          goal[i, 1] - st_c["pos"][i, 1],
          0.002 - st_c["z_min"][i],
        ]
      )
      # Planned where the arm, sagging as it does now, puts the tool point at ``tgt``.
      sag = tcp[i] - kin.fk(q_place[i])[0]
      r = _best(kin, tgt - sag, jaw_axes[i], seeds_from(q_place[i]))
      if r is None:
        plan_err[i] = plan_err[i] or "place"
        continue
      q_place[i] = r[0]
    drv.move(q_place, jaw=jaw_c, steps=40)
    drv.hold(15, jaw=jaw_c)
  st_p = _pin_state(env)
  for k in range(1, 21):
    drv.step(None, jaw=jaw_c * (1 - k / 20))
  drv.hold(15, jaw=0.0)
  tcp = _tcp_env(env)
  q_ret = drv.q_arm().copy()
  for i in range(n):
    if not ok[i]:
      continue
    r = _best(kin, tcp[i] + np.array([0, 0, 0.10]), None, seeds_from(q_place[i]))
    if r is not None:
      q_ret[i] = r[0]
  drv.move(q_ret, jaw=0.0, steps=50)
  drv.move(np.tile(kin.q_default, (n, 1)), jaw=0.0, steps=100)
  drv.hold(75, jaw=0.0)
  st_e = _pin_state(env)
  grasped_e, _ = _grasp_flags(env)
  with torch.inference_mode():
    rs = mdp.pin_rest_state(env, "object", "goal_pose")
    rested = (
      rs.rested(
        torch.as_tensor(grasped_e, device=env.device),
        pp.PLACE_RADIUS,
        pp.PLACED_TOL[1],
        pp.REST_SPEED,
        pp.REST_ANGULAR_SPEED,
      )
      .cpu()
      .numpy()
    )
    dxy = rs.goal_dxy.cpu().numpy()
  rows = []
  for i in range(n):
    rows.append(
      {
        "kind": kinds[i],
        "start": "standing" if standing0[i] else "lying",
        "pin_xy": [round(float(v), 3) for v in st["pos"][i, :2] - shift[i]],
        "goal_xy": [round(float(v), 3) for v in goal[i, :2] - shift[i]],
        "plan_error": plan_err[i],
        "grasped": bool(grasped[i]),
        "lifted": bool(lifted[i]),
        "zmin_before_open_mm": round(1e3 * float(st_p["z_min"][i]), 1),
        "pushed_at_close_mm": round(
          1e3 * float(np.linalg.norm(st_close["pos"][i, :2] - st["pos"][i, :2])), 1
        ),
        "end": "standing" if st_e["standing"][i] else "lying",
        "toppled": bool(standing0[i] and not st_e["standing"][i]),
        "centre_error_cm": round(100 * float(dxy[i]), 2),
        "success": bool(rested[i] and dxy[i] < pp.PLACE_RADIUS),
      }
    )
  out = {"clipped_steps": drv.clipped, "rows": rows}
  args.json.parent.mkdir(parents=True, exist_ok=True)
  args.json.write_text(json.dumps(out, indent=1))
  for kind in ("corner_standing", "corner_lying", "reset"):
    g = [r for r in rows if r["kind"] == kind]
    if not g:
      continue
    planned = [r for r in g if r["plan_error"] is None]
    print(
      f"{kind:16s}: {len(g)} worlds, planned {len(planned)}, grasped {sum(r['grasped'] for r in planned)}, "
      f"lifted {sum(r['lifted'] for r in planned)}, SUCC {sum(r['success'] for r in g)} "
      f"(of planned {sum(r['success'] for r in planned)}); toppled {sum(r['toppled'] for r in g)}; "
      f"plan errors {sorted(set(r['plan_error'] for r in g if r['plan_error']))}"
    )
    for pose in ("standing", "lying"):
      gp = [r for r in g if r["start"] == pose]
      if gp and kind == "reset":
        print(
          f"   {pose}: {len(gp)}, SUCC {sum(r['success'] for r in gp)}, planned {sum(r['plan_error'] is None for r in gp)}"
        )
  print("clipped steps", drv.clipped)
  return out


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  sub = p.add_subparsers(dest="cmd", required=True)
  r = sub.add_parser("reach")
  r.add_argument("--table-dz", default="22")
  r.add_argument("--json", type=pathlib.Path, required=True)
  r.add_argument("--x", type=float, nargs=2, default=(0.32, 0.68))
  r.add_argument("--y", type=float, nargs=2, default=(-0.14, 0.36))
  r.add_argument("--step", type=float, default=0.02)
  r.add_argument("--above-lying", type=float, default=0.020)
  r.add_argument("--shard", type=int, default=0)
  r.add_argument("--shards", type=int, default=1)
  r.add_argument("--seed", type=int, default=0)
  r.add_argument(
    "--box",
    default=None,
    help="hold the IK inside this SPAWN's arm's action box (item 3)",
  )
  s = sub.add_parser("summarize")
  s.add_argument("files", nargs="+")
  for name in ("bench", "rest", "scripted"):
    b = sub.add_parser(name)
    b.add_argument("--json", type=pathlib.Path, required=True)
    b.add_argument("--device", default="cuda:0")
    b.add_argument("--num-envs", type=int, default=80)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--above-lying", type=float, default=0.020)
  args = p.parse_args()
  if args.cmd == "reach":
    reach(args)
  elif args.cmd == "bench":
    bench(args)
  elif args.cmd == "rest":
    rest(args)
  elif args.cmd == "scripted":
    scripted(args)
  else:
    _set_env()
    summarize(args)


if __name__ == "__main__":
  main()
