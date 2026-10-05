"""Plan S10.1 step 0, items 2-4: the box on the wide area `WC`, at any yaw, with the scripted planner.

Runs `scripted_pick_place.rollout` on `BS2`'s plant (G20, CTP, stack speeds, ``LAG=JS`` -- the script
cannot see the cube through the tracker's lag, as in every scripted step 0 --, baseline B, the WC arm,
DR-scene, ``ARM_TABLE=AH10``) with ``GOAL=box-rim`` on ``TARGET=WC``.

``clearance`` (item 2): the grasp only (approach, descend, close, lift) with the box pinned at
``--distances`` from the cube in 8 directions, box yaw 0 / 45 deg, cube yaw 0 / 30 / 45 deg. Counts,
per phase, the steps on which the hand (the `arm_left_7_link` subtree: palm, fingers, pads) or the
cube touches a box wall. `BOX_MIN_SEPARATION` = the least distance with no touch + 2 cm.

``place`` (item 3's gate and item 4, by IK alone): for each forced box pose x cube yaw, the whole plan from a
spawn the script grasps from; the waypoint without contact-free IK, and |action| per joint on the WC arm.

``reach`` (items 3 and 4, simulated): the whole pick-into-box -- 12 forced box poses (the area's corners, edge
midpoints, middle, two at y < 0, one near the torso) at yaw 0 and 45 deg, the cube at the area corner
farthest from the box, plus ``--random`` episodes of the reset's own draws. Per env: plan errors (the
gate: IK must reach the place pose at every forced pose), clipped action steps (item 4: 0), in the box,
placed, the hand / cube on a wall while held, arm 1-6 on the table (AH10's contact term). A record by
box bin, not a gate (the script planned 8 / 64 of the table task's WC, RL reached 95 %).

  uv run --extra cu128 python scripts/s101_box_step0.py clearance --json ../results/S10-1/step0/clearance.json
  uv run --extra cu128 python scripts/s101_box_step0.py reach --json ../results/S10-1/step0/reach.json
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import scripted_pick_place as scripted  # noqa: E402

RCTP = "R-C-strict+M1+M2+M3+M6+J+M8"
GRASP_PHASES = scripted.PHASES[:4]  # approach, descend, close, lift
#: The cube's pose for the clearance sweep: the middle of WC, env frame (x 0.40-0.60, y -0.04-0.26).
CLEARANCE_CUBE = (0.50, 0.11)


def _common(args: argparse.Namespace) -> dict:
  from pal_mjlab.tasks.manipulation.pick_place import pick_place_env_cfg as pp
  from pal_mjlab.tasks.manipulation.pick_place.arm_table import ARM_TABLE_CODES
  from pal_mjlab.tasks.manipulation.pick_place.box import BOX_GOALS
  from pal_mjlab.tasks.manipulation.pick_place.kangaroo.env_cfg import SPAWN_ROBOTS
  from pal_mjlab.tasks.manipulation.robots import kangaroo_grippers_left_arm
  from pal_mjlab.tasks.manipulation.tabletop_env_cfg import LATENCY_MODELS

  return dict(
    device=args.device,
    seed=args.seed,
    reward=RCTP,
    verbose=False,
    robot=kangaroo_grippers_left_arm("G20", SPAWN_ROBOTS["WC"], arm_effort_scale=0.7),
    baseline=pp.BASELINES["B"],
    spawn_box=pp.SPAWN_BOXES["WC"],
    dr=args.dr,
    effort=pp.EFFORT_BOUNDS["CTP"],
    latency=LATENCY_MODELS["JS"],
    stack_speed=True,
    episode_length_s=120.0,
    target_area=pp.TARGET_AREAS["WC"],
    box_goal=BOX_GOALS["box-rim"],
    arm_table=ARM_TABLE_CODES["AH10"],
    pad_clearance=args.pad_clearance,
  )


class _Contacts:
  """A rollout hook: per env and phase, the steps the hand / the cube touch a box wall, and arm
  1-6 on the table (AH10's contact term)."""

  def __init__(self, n: int):
    self.n = n
    self.hand: dict[str, np.ndarray] = {}
    self.cube: dict[str, np.ndarray] = {}
    self.arm = np.zeros(n, int)

  def __call__(self, phase: str, env) -> None:
    from pal_mjlab.tasks.manipulation.pick_place import box as bx

    def hit(name: str) -> np.ndarray:
      return env.scene[name].data.found.bool().any(dim=-1).cpu().numpy()

    for store, name in (
      (self.hand, bx.BOX_WALL_HAND_SENSOR),
      (self.cube, bx.BOX_WALL_OBJECT_SENSOR),
    ):
      store.setdefault(phase, np.zeros(self.n, int))
      store[phase] += hit(name).astype(int)
    term = env.reward_manager.get_term_cfg("arm_table_contact")
    self.arm += term.func(env, **term.params).bool().cpu().numpy().astype(int)

  def row(self, i: int) -> dict:
    return {
      "hand_wall_by_phase": {k: int(v[i]) for k, v in self.hand.items() if v[i]},
      "cube_wall_by_phase": {k: int(v[i]) for k, v in self.cube.items() if v[i]},
      "arm_table_steps": int(self.arm[i]),
    }


def clearance(args: argparse.Namespace) -> dict:
  cases = []
  for d in args.distances:
    for k in range(8):
      th = k * math.pi / 4
      for box_yaw in (0.0, 45.0):
        for cube_yaw in (0.0, 30.0, 45.0):
          cases.append((d, k * 45, box_yaw, cube_yaw, th))
  n = len(cases)
  cx, cy = CLEARANCE_CUBE
  spawns = [((cx, cy), math.radians(c[3])) for c in cases]
  boxes = [
    ((cx + c[0] * math.cos(c[4]), cy + c[0] * math.sin(c[4])), math.radians(c[2]))
    for c in cases
  ]
  hook = _Contacts(n)
  # Only the grasp is run: plan the (unused) carry and place over the cube itself, so that a box
  # pinned where the arm cannot place (x > 0.6 m, behind the torso) does not leave the arm at home --
  # with the full plan 29-38 of 48 cases per distance failed on the carry / place (2026-10-05).
  full = scripted.waypoints

  def grasp_only(kin, cube_xy, cube_yaw, target_xy, *a, **kw):
    kw["carry_hand_x"] = None
    kw["place_dz"] = 0.0
    return full(kin, cube_xy, cube_yaw, np.asarray(cube_xy), *a, **kw)

  scripted.waypoints = grasp_only
  try:
    rows = scripted.rollout(
      num_envs=n,
      **_common(args),
      spawns=spawns,
      box_poses=boxes,
      phases=GRASP_PHASES,
      hook=hook,
    )
  finally:
    scripted.waypoints = full
  out = []
  for r in rows:
    i = r["env"]
    d, direction, box_yaw, cube_yaw, _ = cases[i]
    c = hook.row(i)
    out.append(
      {
        "env": i,
        "distance": d,
        "direction_deg": direction,
        "box_yaw_deg": box_yaw,
        "cube_yaw_deg": cube_yaw,
        "plan_error": r["plan_error"],
        "grasped": bool(r.get("grasped")),
        "lifted": bool(r.get("lifted")),
        **c,
        "touch": bool(c["hand_wall_by_phase"] or c["cube_wall_by_phase"]),
      }
    )
  summary = {}
  for d in args.distances:
    g = [r for r in out if r["distance"] == d]
    summary[f"{d:.2f}"] = {
      "n": len(g),
      "touch": sum(r["touch"] for r in g),
      "hand_touch": sum(bool(r["hand_wall_by_phase"]) for r in g),
      "cube_touch": sum(bool(r["cube_wall_by_phase"]) for r in g),
      "plan_errors": sum(r["plan_error"] is not None for r in g),
      "lifted": sum(r["lifted"] for r in g),
      "touch_cases": [
        (r["direction_deg"], r["box_yaw_deg"], r["cube_yaw_deg"], r["hand_wall_by_phase"])
        for r in g
        if r["touch"]
      ][:16],
    }
  clear = [d for d in args.distances if summary[f"{d:.2f}"]["touch"] == 0]
  return {"summary": summary, "least_clear_distance": min(clear) if clear else None, "rows": out}


#: The 12 forced box centres, env frame (WC: x 0.40-0.60, y -0.04-0.26).
FORCED_BOXES = (
  (0.40, -0.04),
  (0.40, 0.26),
  (0.60, -0.04),
  (0.60, 0.26),
  (0.40, 0.11),
  (0.60, 0.11),
  (0.50, -0.04),
  (0.50, 0.26),
  (0.50, 0.11),
  (0.45, -0.02),
  (0.55, -0.02),
  (0.43, 0.05),
)


def _bin(x: float, y: float) -> str:
  if x < 0.45 and y < 0.10:
    return "near_torso"
  if x > 0.55:
    return "far"
  if y < 0.0:
    return "across_centre"
  return "other"


def reach(args: argparse.Namespace) -> dict:
  from pal_mjlab.tasks.manipulation.pick_place import pick_place_env_cfg as pp

  wc = pp.SPAWN_BOXES["WC"]
  corners = [(x, y) for x in wc.x for y in wc.y]
  forced = []
  for k, (bxy, yaw) in enumerate((b, y) for y in (0.0, 45.0) for b in FORCED_BOXES):
    far = max(corners, key=lambda c: math.hypot(c[0] - bxy[0], c[1] - bxy[1]))
    # Pulled 2 cm into the area so the cube is not on its very edge; yaw 0 / 30 / 45 in turn.
    cube = (far[0] + (0.02 if far[0] < 0.5 else -0.02), far[1] + (0.02 if far[1] < 0.1 else -0.02))
    forced.append((bxy, yaw, cube, (0.0, 30.0, 45.0)[k % 3]))
  results = {}
  hook = _Contacts(len(forced))
  rows = scripted.rollout(
    num_envs=len(forced),
    **_common(args),
    spawns=[(c[2], math.radians(c[3])) for c in forced],
    box_poses=[(c[0], math.radians(c[1])) for c in forced],
    hook=hook,
  )
  results["forced"] = [
    {
      **_row(r),
      **hook.row(r["env"]),
      "box": [*forced[r["env"]][0], forced[r["env"]][1]],
      "cube": [*forced[r["env"]][2], forced[r["env"]][3]],
      "bin": _bin(*forced[r["env"]][0]),
    }
    for r in rows
  ]
  if args.random:
    hook = _Contacts(args.random)
    rows = scripted.rollout(
      num_envs=args.random, corners_first=False, **_common(args), hook=hook
    )
    results["random"] = [{**_row(r), **hook.row(r["env"])} for r in rows]
  return results


#: Spawns the place check plans from: inside the part of WC the script grasps from (W2-like, S7-8 step 0),
#: tried in order until one is >= `box.BOX_MIN_SEPARATION` from the box and its grasp plans.
PLACE_SPAWNS = ((0.44, 0.22), (0.40, 0.26), (0.52, 0.24), (0.46, 0.14), (0.42, 0.16), (0.56, 0.18))
GRASP_WAYPOINTS = ("pregrasp", "grasp", "lift")


def place(args: argparse.Namespace) -> dict:
  """Items 3 (the gate) and 4 by IK alone, no simulation: for each forced box pose x cube yaw 0 / 30 / 45
  deg, plan the whole pick-into-box from a spawn the script can grasp from (`PLACE_SPAWNS`) and record
  which waypoint has no contact-free IK; for every full plan, |action| per arm joint at each waypoint on
  the WC arm (> 1 is clipped)."""
  import torch
  from mjlab.envs import ManagerBasedRlEnv

  from pal_mjlab.tasks.manipulation import scene
  from pal_mjlab.tasks.manipulation.pick_place import pick_place_env_cfg as pp
  from pal_mjlab.tasks.manipulation.pick_place.box import BOX_MIN_SEPARATION

  c = _common(args)
  cfg = pp.make_pick_place_env_cfg(
    c["robot"],
    gripper_name="left",
    reward=c["reward"],
    spawn=c["spawn_box"],
    dr="DR0",
    baseline=c["baseline"],
    effort=c["effort"],
    latency=c["latency"],
    target_area=c["target_area"],
    stack_speed=True,
    box=c["box_goal"],
    arm_table=c["arm_table"],
  )
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  kin = scripted.Kin.from_model(env.sim.mj_model)
  term = env.action_manager.get_term("joint_pos")
  names = [n.split("/", 1)[-1] for n in term.target_names]
  arm = [names.index(j) for j in scripted.ARM_JOINTS]
  offset = term.offset[0, arm].cpu().numpy()
  scale = (
    term.scale[0, arm].cpu().numpy()
    if isinstance(term.scale, torch.Tensor)
    else np.full(len(arm), float(term.scale))
  )
  env.close()
  table_z = c["baseline"].table_top_z
  rows = []
  for bxy, byaw in ((b, y) for y in (0.0, 45.0) for b in FORCED_BOXES):
    for cyaw in (0.0, 30.0, 45.0):
      carry_x = scripted.box_place_hand_x(kin, math.radians(cyaw), math.radians(byaw))
      tried, row = [], None
      for sp in PLACE_SPAWNS:
        if math.hypot(sp[0] - bxy[0], sp[1] - bxy[1]) < BOX_MIN_SEPARATION:
          continue
        try:
          plan = scripted.waypoints(
            kin,
            np.asarray(sp),
            math.radians(cyaw),
            np.asarray(bxy),
            table_z=table_z,
            pad_clearance=args.pad_clearance,
            carry_hand_x=carry_x,
            place_dz=scene.BOX_FLOOR,
          )
        except RuntimeError as e:
          msg = str(e)
          wp = next((w for w in scripted.WAYPOINT_NAMES if f" {w} at" in f" {msg}"), "?")
          tried.append((sp, wp, msg[:90]))
          if wp in GRASP_WAYPOINTS:
            continue  # this spawn's grasp; try the next spawn
          row = {"ok": False, "fails_at": wp, "error": msg[:120], "spawn": sp}
          break
        excursion = {
          w: (np.abs(plan[w] - offset) / scale).round(3).tolist()
          for w in scripted.WAYPOINT_NAMES
        }
        worst = np.max(np.stack([np.asarray(v) for v in excursion.values()]), axis=0)
        row = {
          "ok": True,
          "spawn": sp,
          "max_abs_action_per_joint": worst.round(3).tolist(),
          "clipped_joints": [scripted.ARM_JOINTS[j] for j in np.nonzero(worst > 1.0)[0]],
        }
        break
      if row is None:
        row = {"ok": None, "fails_at": "no spawn grasps", "tried": tried}
      rows.append({"box": [*bxy, byaw], "cube_yaw": cyaw, "bin": _bin(*bxy), **row})
  return {"place": rows}


def _row(r: dict) -> dict:
  keep = (
    "env",
    "plan_error",
    "clipped_steps",
    "in_box",
    "placed",
    "placed_and_lifted",
    "hand_wall_held_steps",
    "cube_wall_held_steps",
    "box_displacement_mm",
    "spawn_x",
    "spawn_y",
    "target_x",
    "target_y",
  )
  return {k: r.get(k) for k in keep}


def _print_reach(res: dict) -> None:
  for key, rows in res.items():
    n = len(rows)
    ok = [r for r in rows if r["plan_error"] is None]
    print(
      f"{key}: {n} envs, planned {len(ok)}, clipped steps (sum) "
      f"{sum(int(r['clipped_steps'] or 0) for r in rows)}, in the box {sum(bool(r['in_box']) for r in rows)}, "
      f"placed+lifted {sum(bool(r['placed_and_lifted']) for r in rows)}, hand on a wall while held "
      f"{sum((r['hand_wall_held_steps'] or 0) > 0 for r in rows)}, arm on the table "
      f"{sum(r['arm_table_steps'] > 0 for r in rows)}"
    )
    if key == "forced":
      for r in rows:
        print(
          f"  box {r['box']} cube {r['cube']} [{r['bin']}]: plan {'ok' if r['plan_error'] is None else r['plan_error'][:60]}"
          f" clipped {r['clipped_steps']} in_box {r['in_box']} placed {r['placed_and_lifted']}"
        )


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("mode", choices=("clearance", "reach", "place"))
  ap.add_argument("--json", type=pathlib.Path, required=True)
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--dr", default="DR-scene")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument(
    "--pad-clearance",
    type=float,
    default=0.01,
    help="fingertips above the table at the close, m (1 cm: the lowest grasp, the worst case)",
  )
  ap.add_argument(
    "--distances", type=float, nargs="+", default=(0.12, 0.14, 0.16, 0.18)
  )
  ap.add_argument("--random", type=int, default=64)
  args = ap.parse_args()
  res = {"clearance": clearance, "reach": reach, "place": place}[args.mode](args)
  if args.mode == "clearance":
    print(json.dumps(res["summary"], indent=1))
    print("least distance with no wall touch:", res["least_clear_distance"])
  elif args.mode == "reach":
    _print_reach(res)
  else:
    for r in res["place"]:
      print(
        f"box {r['box']} [{r['bin']}] cube yaw {r['cube_yaw']:4.0f}: "
        + (
          f"ok from {r['spawn']}, max |a| {r['max_abs_action_per_joint']} clipped {r['clipped_joints']}"
          if r["ok"]
          else f"FAILS at {r['fails_at']}: {r.get('error', '')[:70]}"
        )
      )
    ok = [r for r in res["place"] if r["ok"]]
    print(
      f"place reachable {len(ok)} / {len(res['place'])}; clipped in "
      f"{sum(bool(r['clipped_joints']) for r in ok)}; worst |a| per joint "
      f"{np.max([r['max_abs_action_per_joint'] for r in ok], axis=0).round(3).tolist() if ok else None}"
    )
  args.json.parent.mkdir(parents=True, exist_ok=True)
  args.json.write_text(
    json.dumps({"args": vars(args) | {"json": str(args.json)}, **res}, indent=1, default=str)
  )


if __name__ == "__main__":
  main()
