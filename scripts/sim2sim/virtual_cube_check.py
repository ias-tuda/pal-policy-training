"""How faithful is the PAL node's virtual cube? Drive it with MuJoCo's own gripper motion.

A Python port of pal_policy_inferer_pick_place_node's virtual cube (same state machine, same
profile parameters) is fed the TCP pose and jaw angle MuJoCo recorded in the reference
rollout, open loop, and its cube is compared with MuJoCo's physical one: when it is grasped,
where it ends up, and whether it counts as placed. `--release straight` is the node's first
version (the cube drops where it is let go), `momentum` the current one (it keeps its velocity,
lands and slides to rest at mu*g).

  .venv/bin/python scripts/sim2sim/virtual_cube_check.py --model <model_dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

G = 9.81
REST, HELD, FALLING, DROPPED, SLIDING = 0, 1, 2, 3, 4


def quat_to_R(w, x, y, z):
  return np.array(
    [
      [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
  )


def rotz(a):
  c, s = np.cos(a), np.sin(a)
  return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def yaw_of(R):
  return np.arctan2(R[1, 0], R[0, 0])


class VirtualCube:
  def __init__(self, prof: dict, start: np.ndarray, yaw: float, release: str):
    g = prof["grasp"]
    self.p, self.R, self.v = start.astype(float), rotz(yaw), np.zeros(3)
    self.state, self.was_lifted, self.release = REST, False, release
    self.T_rel = None
    self.dt = 1.0 / prof["policy_rate"]
    self.rest_z = prof["table_top_z_b"] + prof["cube_half_size"]
    self.tx, self.ty = prof["table_x_b"], prof["table_y_b"]
    self.half = prof["cube_half_size"]
    self.env = (g["x_tol"], g["y_tol"], g["z_lo"], g["z_hi"])
    self.jaw_contact, self.jaw_release = g["jaw_contact"], g["jaw_release"]
    self.mu = 0.9
    self.floor = -0.9

  def over_table(self, p):
    return self.tx[0] <= p[0] <= self.tx[1] and self.ty[0] <= p[1] <= self.ty[1]

  def step(self, R_tcp, p_tcp, jaw):
    dt = self.dt
    if self.state == REST:
      rel = R_tcp.T @ (self.p - p_tcp)
      xt, yt, zl, zh = self.env
      if (
        abs(rel[0]) <= xt
        and abs(rel[1]) <= yt
        and zl <= rel[2] <= zh
        and jaw >= self.jaw_contact
      ):
        self.T_rel = (R_tcp.T @ self.R, R_tcp.T @ (self.p - p_tcp))
        self.state = HELD
    elif self.state == HELD:
      Rr, pr = self.T_rel
      p_new = R_tcp @ pr + p_tcp
      if self.over_table(p_new) and p_new[2] < self.rest_z:
        p_new[2] = self.rest_z
        self.T_rel = (Rr, R_tcp.T @ (p_new - p_tcp))
      self.v = (p_new - self.p) / dt
      self.p = p_new
      self.R = rotz(yaw_of(R_tcp @ Rr))
      if not (self.over_table(self.p) and self.p[2] <= self.rest_z + 0.002):
        self.was_lifted = True
      if jaw < self.jaw_release:
        self.state = FALLING
        if self.release == "straight":
          self.v = np.zeros(3)
    elif self.state == FALLING:
      self.v[2] -= G * dt
      self.p = self.p + self.v * dt
      if self.over_table(self.p) and self.p[2] <= self.rest_z:
        self.p[2], self.v[2] = self.rest_z, 0.0
        self.state = SLIDING if np.linalg.norm(self.v[:2]) > 1e-3 else REST
      elif self.p[2] <= self.floor + self.half:
        self.state = DROPPED
    elif self.state == SLIDING:
      sp = np.linalg.norm(self.v[:2])
      dv = self.mu * G * dt
      if sp <= dv:
        self.v[:] = 0.0
        self.state = REST
      else:
        self.v[:2] *= (sp - dv) / sp
        self.p[:2] += self.v[:2] * dt
        if not self.over_table(self.p):
          self.state = FALLING


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--model", type=Path, required=True)
  args = p.parse_args()
  prof = yaml.safe_load((args.model / "pick_place_profile.yaml").read_text())
  ref = np.load(args.model / "mujoco_reference.npz")
  goal = np.array(prof["goal_position_b"])
  thr = prof["goal_threshold"]
  n_ep, T = ref["q"].shape[:2]
  for release in ("straight", "momentum"):
    rows = []
    for k in range(n_ep):
      done = ref["done"][k].astype(bool)
      n = int(np.argmax(done)) + 1 if done.any() else T
      cube0 = ref["cube_b"][k, 0]
      yaw0 = float(ref["cube_yaw_b"][k, 0])
      vc = VirtualCube(prof, cube0, yaw0, release)
      grasp_t = place_t = None
      for t in range(n):
        R = quat_to_R(*ref["tcp_quat_b"][k, t])
        vc.step(R, ref["tcp_b"][k, t], ref["q"][k, t, 7])
        if vc.state == HELD and grasp_t is None:
          grasp_t = t
        placed = (
          np.linalg.norm(vc.p - goal) < thr and vc.state == REST and vc.was_lifted
        )
        if placed and place_t is None:
          place_t = t
      g = ref["grasped"][k, :n].astype(bool)
      mj_final = ref["cube_b"][k, n - 1]
      rows.append(
        (
          k,
          int(np.argmax(g)) if g.any() else None,
          grasp_t,
          place_t is not None,
          float(np.linalg.norm(vc.p - mj_final)),
          float(np.linalg.norm(vc.p - goal)),
          float(np.linalg.norm(mj_final - goal)),
        )
      )
    placed = sum(r[3] for r in rows)
    print(
      f"[{release}] virtual cube placed {placed}/{n_ep} on MuJoCo's own gripper motion"
    )
    for k, g_mj, g_v, pl, d_mj, d_goal_v, d_goal_mj in rows:
      print(
        f"  ep{k}: grasp step MuJoCo {g_mj} / virtual {g_v}; placed {pl}; final virtual-vs-"
        f"MuJoCo cube {100 * d_mj:.1f} cm; to goal: virtual {100 * d_goal_v:.1f} cm, "
        f"MuJoCo {100 * d_goal_mj:.1f} cm"
      )


if __name__ == "__main__":
  main()
