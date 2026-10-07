"""Plan §13's kinematics helpers (`pick-place-insert-plan.md` §13 step 0): either arm's IK onto a full hand pose, on a
compiled single-world copy of an env's scene (CPU MuJoCo, no warp), with contact checks. `scripted_pick_place.Kin`
generalised to the right arm and to an arbitrary hand orientation; used by `s13_scripted.py` and the step-0 maps
(`oppsie-data/results/S13-insert/step0/s13_step0.py`, the same code before it moved here).

The hand frame: +z is the approach axis (the TCP site sits 0.148 m out along the `arm_*_7_link`'s z, the site's own
frame being the link's), +x the direction the pads close along. A parallel jaw cannot tell +x from -x, so every
solver tries both."""

from __future__ import annotations

import math

import mujoco
import numpy as np


def compile_env_scene(env_cfg) -> mujoco.MjModel:
  """The env cfg's scene as a single-world CPU model with the env's solver options."""
  import dataclasses

  from mjlab.scene import Scene

  scene_cfg = dataclasses.replace(env_cfg.scene, num_envs=1)
  model = Scene(scene_cfg, "cpu").compile()
  env_cfg.sim.mujoco.apply(model)
  return model


class Arm:
  """One arm's kinematics on a single-world model."""

  def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, side: str) -> None:
    from scene_model import joint_qadr, site_id, soft_limits

    self.m, self.d, self.side = model, data, side
    self.joints = [f"arm_{side}_{i}_joint" for i in range(1, 8)]
    self.qadr = joint_qadr(model, self.joints)
    jids = [
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{j}")
      for j in self.joints
    ]
    self.dofs = np.asarray([model.jnt_dofadr[j] for j in jids])
    self.lo, self.hi = soft_limits(model, self.joints)
    self.tcp = site_id(model, f"robot/tcp_{side}")
    self.hand = mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_BODY, f"robot/arm_{side}_7_link"
    )
    self.q_default = data.qpos[self.qadr].copy()
    self.pads = [
      site_id(model, f"robot/gripper_{side}_fingertip_{s}_site")
      for s in ("left", "right")
    ]
    names = [
      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
      for g in range(model.ngeom)
    ]
    bare = [n.split("/", 1)[-1] for n in names]
    self.geoms = np.asarray(
      [
        g
        for g in range(model.ngeom)
        if bare[g].startswith((f"arm_{side}_", f"gripper_{side}_"))
      ]
    )
    self.names = names

  def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    self.d.qpos[self.qadr] = q
    mujoco.mj_kinematics(self.m, self.d)
    mujoco.mj_comPos(self.m, self.d)
    return self.d.site_xpos[self.tcp].copy(), self.d.xmat[self.hand].reshape(
      3, 3
    ).copy()

  def pad_offset(self) -> float:
    """How far the pads reach past the TCP along the approach axis, jaw open (25.6 mm, S1.3)."""
    tcp, rot = self.fk(self.q_default)
    return float(max(np.dot(self.d.site_xpos[p] - tcp, rot[:, 2]) for p in self.pads))

  def ik_pose(self, goal, r_des, q_init, iters: int = 300):
    """Damped least squares onto TCP at ``goal`` and hand rotation ``r_des``, soft limits, a pull towards home in the
    null space (scripted_pick_place.Kin.ik_pose)."""
    jacp = np.zeros((3, self.m.nv))
    jacr = np.zeros((3, self.m.nv))
    q = q_init.copy()
    for _ in range(iters):
      pos, rot = self.fk(q)
      err_p = goal - pos
      err_r = 0.5 * sum(np.cross(rot[:, i], r_des[:, i]) for i in range(3))
      if np.linalg.norm(err_p) < 1e-3 and np.linalg.norm(err_r) < 1e-2:
        return q
      mujoco.mj_jacSite(self.m, self.d, jacp, jacr, self.tcp)
      jac = np.vstack([jacp[:, self.dofs], jacr[:, self.dofs]])
      err = np.concatenate([err_p, err_r])
      pinv = jac.T @ np.linalg.inv(jac @ jac.T + 1e-3 * np.eye(6))
      null = np.eye(len(q)) - pinv @ jac
      step = pinv @ err + null @ (0.2 * (self.q_default - q))
      q = np.clip(q + np.clip(step, -0.2, 0.2), self.lo, self.hi)
    return None

  def contacts(
    self, q: np.ndarray, ignore: set[int] = frozenset()
  ) -> list[tuple[str, str]]:
    """This arm's contacts at ``q``, skipping the geoms in ``ignore`` (e.g. the object it is about to grasp)."""
    self.fk(q)
    mujoco.mj_collision(self.m, self.d)
    mine = set(int(g) for g in self.geoms)
    hits = []
    for c in range(self.d.ncon):
      g1, g2 = int(self.d.contact[c].geom1), int(self.d.contact[c].geom2)
      if not (g1 in mine or g2 in mine):
        continue
      other = g2 if g1 in mine else g1
      if other in ignore:
        continue
      hits.append((self.names[g1], self.names[g2]))
    return hits


def hand_frame(approach: np.ndarray, close: np.ndarray) -> np.ndarray:
  """Rotation with hand +z = ``approach`` and +x along ``close`` (made orthogonal)."""
  z = approach / np.linalg.norm(approach)
  x = close - np.dot(close, z) * z
  x = x / np.linalg.norm(x)
  return np.column_stack([x, np.cross(z, x), z])


def solve(
  arm: Arm,
  goal,
  r_des,
  q,
  check,
  rng,
  local: int = 16,
  glob: int = 48,
  segments: int = 12,
):
  """A solution with no contacts (``check(q)`` empty) reachable from ``q`` along a contact-free straight joint path;
  seeded from ``q``, then around it, then (closest to ``q``) from anywhere in the soft limits. None if none."""

  def ok(sol):
    if sol is None or check(sol):
      return False
    return not any(check(q + (sol - q) * (k / segments)) for k in range(1, segments))

  for attempt in range(local):
    std = (0.3, 0.6, 1.0)[attempt * 3 // local]
    seed = (
      q if attempt == 0 else np.clip(q + rng.normal(0, std, q.shape), arm.lo, arm.hi)
    )
    sol = arm.ik_pose(goal, r_des, seed)
    if ok(sol):
      return sol
  best = None
  for _ in range(glob):
    sol = arm.ik_pose(goal, r_des, rng.uniform(arm.lo, arm.hi))
    if ok(sol):
      dq = float(np.abs(sol - q).max())
      if best is None or dq < best[0]:
        best = (dq, sol)
  return None if best is None else best[1]


def solve_endpoint(arm: Arm, goal, r_des, check, rng, attempts: int = 64):
  """A contact-free solution, no path requirement; the one closest to home."""
  best = None
  for t in range(attempts):
    seed = arm.q_default if t == 0 else rng.uniform(arm.lo, arm.hi)
    sol = arm.ik_pose(goal, r_des, seed)
    if sol is None or check(sol):
      continue
    d = float(np.abs(sol - arm.q_default).max())
    if best is None or d < best[0]:
      best = (d, sol)
  return None if best is None else best[1]


def face_normal_against(yaw: float, direction_xy) -> np.ndarray:
  """The outward normal (xy) of a 4-fold block's side face at ``yaw`` that points most against ``direction_xy``."""
  d = np.asarray(direction_xy, float)
  normals = [
    np.array([math.cos(yaw + k * math.pi / 2), math.sin(yaw + k * math.pi / 2)])
    for k in range(4)
  ]
  return min(normals, key=lambda n: float(np.dot(n, d)))
