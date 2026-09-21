"""Find a tabletop-ready arm pose for the Kangaroo, and print it as an init-state block.

S1.2 of the pick-and-place plan. ``INIT_STATE`` in ``kangaroo_constants.py`` is the
*locomotion* pose: it names arm 1-4 only, leaves arm 5-7 and the jaw at zero, and puts the
left grasp point at (0.178, 0.326, 0.558) -- 20 cm below the table top and 7 cm behind its
near edge, with the arm hanging beside the body. A manipulation task that starts there
spends its first second of every episode getting the hand over the table.

This solves for a pose that instead holds the open jaw above the working area, pointing
down, comfortably inside the soft limits and clear of the table and of the robot itself,
and prints it in the form ``EntityCfg.InitialStateCfg`` takes. It changes nothing: the
output is meant to be pasted into a new ``MANIPULATION_INIT_STATE`` block, leaving the
locomotion pose alone.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from reach_map import (  # noqa: E402
  ARM4_GEOM,
  ARM_JOINTS,
  HAND_BODY,
  TABLE_TOP_Z,
  TCP_SITE,
  TORSO_GEOM,
  _park_free_bodies,
  _skip,
  left_arm_geoms,
  pad_offset,
)
from scene_model import (  # noqa: E402
  DEFAULT_TASK,
  compile_scene,
  geom_id,
  joint_qadr,
  nominal_data,
  site_id,
  soft_limits,
)


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--task", default=DEFAULT_TASK)
  p.add_argument(
    "--tcp",
    type=float,
    nargs=3,
    default=(0.41, 0.15, 0.95),
    metavar=("X", "Y", "Z"),
    help="where the grasp point should sit in the nominal pose (world frame)",
  )
  p.add_argument("--restarts", type=int, default=400)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument(
    "--no-torso",
    action="store_true",
    help="do not enable the torso capsule while checking the pose (not recommended: the "
    "pose has to stay collision-free once S1.4 turns it on)",
  )
  p.add_argument(
    "--arm4",
    action="store_true",
    help="also check against arm_left_4_collision, the oversized elbow capsule, ignoring "
    "its contacts with its own children",
  )
  return p.parse_args()


def solve(args: argparse.Namespace) -> dict:
  from reach_map import _ik_top_down

  extra = () if args.no_torso else (TORSO_GEOM,)
  if args.arm4:
    extra += (ARM4_GEOM,)
  model = compile_scene(args.task, extra)
  elbow = {geom_id(model, f"robot/{ARM4_GEOM}")} if args.arm4 else set()
  data = nominal_data(model)
  offset = pad_offset(model, data)
  _park_free_bodies(model, data)

  qadr = joint_qadr(model, ARM_JOINTS)
  lo, hi = soft_limits(model, ARM_JOINTS)
  mid = 0.5 * (lo + hi)
  span = 0.5 * (hi - lo)
  tcp = site_id(model, TCP_SITE)
  hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)
  is_arm = left_arm_geoms(model)
  goal = np.asarray(args.tcp, dtype=float)

  rng = np.random.default_rng(args.seed)
  best, best_cost = None, np.inf
  solved = free = 0
  for s in range(args.restarts):
    q0 = mid if s == 0 else rng.uniform(lo, hi)
    q = _ik_top_down(model, data, qadr, lo, hi, tcp, hand, goal, q0)
    if q is None:
      continue
    solved += 1
    data.qpos[qadr] = q
    mujoco.mj_kinematics(model, data)
    mujoco.mj_collision(model, data)
    if any(
      not _skip(elbow, is_arm, int(data.contact[c].geom1), int(data.contact[c].geom2))
      and (is_arm[int(data.contact[c].geom1)] or is_arm[int(data.contact[c].geom2)])
      for c in range(data.ncon)
    ):
      continue
    free += 1
    # Comfort: how close the pose sits to the middle of each joint's soft range. A nominal
    # pose at a limit is one the `joint_pos_limits` penalty fights from step zero.
    cost = float(np.max(np.abs((q - mid) / span)))
    if cost < best_cost:
      best, best_cost = q.copy(), cost
  print(
    f"{args.restarts} restarts: {solved} IK solutions, {free} of them collision-free "
    f"(torso collider {'OFF' if args.no_torso else 'ON'})"
  )
  if best is None:
    raise SystemExit(f"no collision-free pose puts the grasp point at {goal}")
  return dict(
    model=model,
    data=data,
    q=best,
    cost=best_cost,
    qadr=qadr,
    lo=lo,
    hi=hi,
    tcp=tcp,
    hand=hand,
    offset=offset,
    goal=goal,
  )


def report(res: dict) -> None:
  model, data, q = res["model"], res["data"], res["q"]
  data.qpos[res["qadr"]] = q
  mujoco.mj_forward(model, data)

  print("\nnominal left-arm pose (rad), and where each joint sits in its soft range:")
  for name, value, low, high in zip(ARM_JOINTS, q, res["lo"], res["hi"], strict=True):
    frac = (value - 0.5 * (low + high)) / (0.5 * (high - low))
    print(
      f"  {name:22s} {value:+7.3f}   soft [{low:+6.3f},{high:+6.3f}]  at {frac:+.2f} of range"
    )
  print(f"  worst-case fraction of soft range used: {res['cost']:.2f}")

  tcp_pos = data.site_xpos[res["tcp"]]
  axis = data.xmat[res["hand"]].reshape(3, 3)[:, 2]
  pads = [
    data.site_xpos[site_id(model, f"robot/gripper_left_fingertip_{s}_site")]
    for s in ("left", "right")
  ]
  print(f"\n  grasp point      {tcp_pos.round(4)}  (asked for {res['goal']})")
  print(
    f"  approach axis    {axis.round(3)}  ({np.degrees(np.arccos(-axis[2])):.1f} deg off straight down)"
  )
  print(f"  jaw opening      {np.linalg.norm(pads[0] - pads[1]) * 1000:.1f} mm")
  print(
    f"  lowest pad       z = {min(p[2] for p in pads):.4f}  ({(min(p[2] for p in pads) - TABLE_TOP_Z) * 100:+.1f} cm vs the table top)"
  )

  # Clearance of the parts of the arm that have no collider today but will once S1.4 lands.
  print(
    "\n  clearance above the table top (z of the lowest point of each link's collider):"
  )
  for body in (
    "arm_left_4_link",
    "arm_left_5_link",
    "arm_left_6_link",
    "arm_left_7_link",
  ):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"robot/{body}")
    lowest = None
    for gid in range(
      model.body_geomadr[bid], model.body_geomadr[bid] + model.body_geomnum[bid]
    ):
      name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
      if not name.endswith("_collision"):
        continue
      z = data.geom_xpos[gid][2] - model.geom_rbound[gid]
      lowest = z if lowest is None else min(lowest, z)
    if lowest is not None:
      print(f"    {body:18s} >= {lowest:.3f}  ({(lowest - TABLE_TOP_Z) * 100:+.1f} cm)")

  print("\npaste-ready init-state entries:\n")
  for name, value in zip(ARM_JOINTS, q, strict=True):
    print(f'    "{name}": {value:.3f},')
  print('    "gripper_left_finger_joint": 0.0,   # jaw open')


def main() -> None:
  args = parse_args()
  report(solve(args))


if __name__ == "__main__":
  main()
