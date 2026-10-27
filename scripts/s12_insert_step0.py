"""Plan §12 step 0 (`pick-place-insert-plan.md`), items 4 and 5: the scripted pick-and-insert on `BS12`.

Runs `scripted_pick_place.rollout` with ``insert`` (`INSERT_PHASES`) on `BS12`'s plant -- G20, CTP, the stack
task's speeds, ``LAG=JS`` (the script cannot see through the tracker's lag, as in every scripted step 0), baseline
B on the 0.837 m table (``TABLE_DZ=22``), the WC arm, DR-scene, AH10 -- with the pin and an insert ``GOAL``.

``scripted`` (item 5, L1): the forced cases -- the block at WC's four corners and its middle, at yaw 0 and 45 deg,
the pin at the area corner farthest from the block (yaw 0 / 30 deg alternately) -- then ``--random`` episodes of the
reset's own draws. Per env: plan errors, clipped action steps, ``seated`` at the end (the gate: >= 9 / 10 forced),
the pin's height / offset / tilt, the block's displacement, the hand on the block, arm 1-6 on the table (AH10's
contact term, must be 0).

``separation`` (item 4): the grasp only, the block pinned ``--distances`` from the pin in 8 directions, block yaw
0 / 45 deg, pin yaw 0 / 45 deg; the steps the hand touches the block, per phase. `PIN_MIN_SEPARATION` = the least
distance with no touch + 2 cm (S10.1's rule).

  uv run --extra cu128 python scripts/s12_insert_step0.py scripted --json ../results/S12-insert/step0/scripted.json
  uv run --extra cu128 python scripts/s12_insert_step0.py separation --json ../results/S12-insert/step0/separation.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import scripted_pick_place as scripted  # noqa: E402

RCTP = "R-C-strict+M1+M2+M3+M6+J+M8"
#: The robot cell's table (plan §0.3): baseline B's 0.815 m + 22 mm.
TABLE_TOP_Z = 0.837
GRASP_PHASES = scripted.INSERT_PHASES[:4]  # approach, descend, close, lift


def _common(args: argparse.Namespace) -> dict:
  from pal_mjlab.tasks.manipulation.pick_place import pick_place_env_cfg as pp
  from pal_mjlab.tasks.manipulation.pick_place.arm_table import ARM_TABLE_CODES
  from pal_mjlab.tasks.manipulation.pick_place.insert import INSERT_GOALS
  from pal_mjlab.tasks.manipulation.pick_place.kangaroo.env_cfg import SPAWN_ROBOTS
  from pal_mjlab.tasks.manipulation.robots import kangaroo_grippers_left_arm
  from pal_mjlab.tasks.manipulation.tabletop_env_cfg import LATENCY_MODELS

  return dict(
    device=args.device,
    seed=args.seed,
    reward=RCTP,
    verbose=False,
    robot=kangaroo_grippers_left_arm("G20", SPAWN_ROBOTS["WC"], arm_effort_scale=0.7),
    baseline=dataclasses.replace(pp.BASELINES["B"], table_top_z=TABLE_TOP_Z),
    spawn_box=pp.SPAWN_BOXES["WC"],
    dr=args.dr,
    effort=pp.EFFORT_BOUNDS["CTP"],
    latency=LATENCY_MODELS["JS"],
    stack_speed=True,
    episode_length_s=120.0,
    target_area=pp.TARGET_AREAS["WC"],
    shape=pp.OBJECT_SHAPES["pin"],
    insert=INSERT_GOALS[args.goal],
    arm_table=ARM_TABLE_CODES["AH10"],
    pad_clearance=_pad_clearance(args.above_top),
    clear_height=args.clear_height,
  )


def _pad_clearance(above_top: float | None) -> float:
  """The pads' height over the table at the grasp, as `scripted.waypoints` takes it (the open jaw's pad sites,
  `Kin.pad_offset` = 25.6 mm past the TCP): the TCP ``above_top`` over the pin's top (`insert.GRASP_ABOVE_TOP`)."""
  from pal_mjlab.tasks.manipulation import scene
  from pal_mjlab.tasks.manipulation.pick_place.insert import GRASP_ABOVE_TOP

  above = GRASP_ABOVE_TOP if above_top is None else above_top
  return scene.PIN_LENGTH + above - PAD_OFFSET


#: `scripted.Kin.pad_offset` on the WC arm at its start pose (m): measured 25.6 mm (finger_geometry.json).
PAD_OFFSET = 0.0256


class _ArmTable:
  """A rollout hook: per env, the steps AH10's contact term fires (arm 1-6 on the table); the phase in which the
  pin first tilts past 10 deg; the TCP's xy offset from the pin's axis at the top of its last ``close`` step."""

  def __init__(self, n: int):
    self.arm = np.zeros(n, int)
    self.tipped = [""] * n
    self.close_offset = np.full(n, np.nan)
    self.close_dz = np.full(n, np.nan)

  def __call__(self, phase: str, env) -> None:
    from pal_mjlab.tasks.manipulation import mdp

    term = env.reward_manager.get_term_cfg("arm_table_contact")
    self.arm += term.func(env, **term.params).bool().cpu().numpy().astype(int)
    tilt = mdp.observations.object_tilt(env, "object").cpu().numpy()
    for i in np.nonzero(tilt > np.radians(10.0))[0]:
      if not self.tipped[i]:
        self.tipped[i] = phase
    if phase == "close":
      robot, pin = env.scene["robot"], env.scene["object"]
      tcp = robot.data.site_pos_w[:, robot.site_names.index("tcp_left")]
      top = pin.data.root_link_pos_w[:, 2] + mdp.object_half_height(env, "object")
      self.close_offset = 1e3 * torch_norm_xy(tcp - pin.data.root_link_pos_w)
      self.close_dz = 1e3 * (tcp[:, 2] - top).cpu().numpy()


def torch_norm_xy(v):
  return v[:, :2].norm(dim=-1).cpu().numpy()


def _forced_cases() -> list[
  tuple[tuple[float, float], float, tuple[float, float], float]
]:
  """(pin xy, pin yaw, block xy, block yaw) for the gate: the block at WC's corners and middle, yaw 0 / 45 deg,
  the pin at the WC corner farthest from it. Env frame, the table at its nominal place."""
  from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import SPAWN_BOXES

  wc = SPAWN_BOXES["WC"]
  inset = 0.03  # the block's whole footprint on the area's corner, the pin's too
  xs = (wc.x[0] + inset, wc.x[1] - inset)
  ys = (wc.y[0] + inset, wc.y[1] - inset)
  corners = [(x, y) for x in xs for y in ys]
  mid = (sum(wc.x) / 2, sum(wc.y) / 2)
  cases = []
  for k, block in enumerate([*corners, mid]):
    pin = max(corners, key=lambda c: math.dist(c, block))
    for b_yaw in (0.0, 45.0):
      p_yaw = 0.0 if (k + int(b_yaw)) % 2 == 0 else 30.0
      cases.append((pin, math.radians(p_yaw), block, math.radians(b_yaw)))
  return cases


def scripted_run(args: argparse.Namespace) -> dict:
  forced = _forced_cases()
  n = len(forced) + args.random
  hook = _ArmTable(n)
  # The random episodes keep the reset's draws: force them where they already are (box_poses needs one per env).
  rows = _rollout_with(args, forced, n, hook, scripted.INSERT_PHASES)
  for r in rows:
    r["arm_table_steps"] = int(hook.arm[r["env"]])
    r["tipped_in"] = hook.tipped[r["env"]]
    r["close_tcp_offset_mm"] = float(hook.close_offset[r["env"]])
    r["close_tcp_over_top_mm"] = float(hook.close_dz[r["env"]])
  gate = [r for r in rows if r["env"] < len(forced)]
  out = {
    "forced_cases": [
      dict(
        pin_xy=c[0],
        pin_yaw_deg=round(math.degrees(c[1])),
        block_xy=c[2],
        block_yaw_deg=round(math.degrees(c[3])),
      )
      for c in forced
    ],
    "forced_seated": sum(bool(r.get("seated")) for r in gate),
    "forced_n": len(gate),
    "random_seated": sum(
      bool(r.get("seated")) for r in rows if r["env"] >= len(forced)
    ),
    "random_n": args.random,
    "plan_errors": sum(r["plan_error"] is not None for r in rows),
    "clipped_env_steps": rows[0]["clipped_env_steps"] if rows else 0,
    "arm_table_envs": sum(r["arm_table_steps"] > 0 for r in rows),
    "rows": rows,
  }
  print(
    f"forced seated {out['forced_seated']}/{out['forced_n']}, random {out['random_seated']}/{args.random}, "
    f"plan errors {out['plan_errors']}, clipped env-steps {out['clipped_env_steps']}, arm on table "
    f"{out['arm_table_envs']} envs"
  )
  for r in rows:
    print(
      f"env {r['env']:3d} seated {bool(r.get('seated'))!s:5s} bottom {r.get('pin_bottom_over_block_mm', float('nan')):6.1f} mm "
      f"offset {r.get('pin_offset_mm', float('nan')):5.1f} mm tilt {r.get('pin_tilt_deg', float('nan')):5.1f} deg "
      f"block moved {r.get('block_displacement_mm', float('nan')):5.1f} mm hand-block {int(r.get('hand_block_steps', 0)):4d} "
      f"grasped {r['grasped']!s:5s} tipped {r['tipped_in'] or '-':8s} close tcp off {r['close_tcp_offset_mm']:5.1f} "
      f"over top {r['close_tcp_over_top_mm']:5.1f} mm ended {r['ended_by'] or '-'} "
      f"{('ERR ' + r['plan_error'][:50]) if r['plan_error'] else ''}"
    )
  return out


def _rollout_with(args, forced, n, hook, phases):
  """`scripted.rollout` with env i's pin and block at ``forced[i]`` (the rest: the reset's own draws)."""
  common = _common(args)
  spawns = [((c[0][0], c[0][1]), c[1]) for c in forced]
  blocks = [((c[2][0], c[2][1]), c[3]) for c in forced]
  if n > len(forced):
    # Draw the rest as the reset would: a throwaway env reads its own reset (same seed, same DR draws).
    extra = _reset_draws(args, n)
    spawns += extra["spawns"][len(forced) :]
    blocks += extra["blocks"][len(forced) :]
  return scripted.rollout(
    num_envs=n,
    **common,
    spawns=spawns,
    box_poses=blocks,
    phases=phases,
    hook=hook,
  )


def _reset_draws(args, n: int) -> dict:
  """The pin's and the block's reset poses for ``n`` envs (env frame, the table at its nominal place)."""
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from pal_mjlab.tasks.manipulation.pick_place import pick_place_env_cfg as pp
  from pal_mjlab.tasks.manipulation.pick_place.insert import BLOCK

  c = _common(args)
  cfg = pp.make_pick_place_env_cfg(
    c["robot"],
    gripper_name="left",
    reward=RCTP,
    spawn=c["spawn_box"],
    dr=c["dr"],
    baseline=c["baseline"],
    effort=c["effort"],
    latency=c["latency"],
    target_area=c["target_area"],
    stack_speed=True,
    shape=c["shape"],
    arm_table=c["arm_table"],
    insert=c["insert"],
  )
  cfg.scene.num_envs = n
  cfg.seed = args.seed + 1
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  env.reset()
  table = env.scene[pp.TABLE]
  tb = table.indexing.body_ids[table.body_names.index(pp.TABLE)]
  table_xy0 = torch.tensor(pp.scene.table_pos(TABLE_TOP_Z)[:2], device=env.device)
  shift = env.sim.data.xpos[:, tb, :2] - env.scene.env_origins[:, :2] - table_xy0
  out = {}
  for key, name in (("spawns", pp.OBJECT), ("blocks", BLOCK)):
    ent = env.scene[name]
    xy = (
      (ent.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2] - shift)
      .cpu()
      .numpy()
    )
    w, x, y, z = ent.data.root_link_quat_w.cpu().numpy().T
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    out[key] = [
      ((float(a), float(b)), float(c)) for (a, b), c in zip(xy, yaw, strict=True)
    ]
  env.close()
  return out


class _Touch:
  """A rollout hook: per env and phase, the steps the hand touches the block."""

  def __init__(self, n: int):
    self.n = n
    self.hand: dict[str, np.ndarray] = {}

  def __call__(self, phase: str, env) -> None:
    from pal_mjlab.tasks.manipulation.pick_place import insert as ins

    hit = env.scene[ins.BLOCK_HAND_SENSOR].data.found.bool().any(dim=-1).cpu().numpy()
    self.hand.setdefault(phase, np.zeros(self.n, int))
    self.hand[phase] += hit.astype(int)


def separation(args: argparse.Namespace) -> dict:
  cx, cy = 0.50, 0.11
  cases = []
  for d in args.distances:
    for k in range(8):
      th = k * math.pi / 4
      for b_yaw in (0.0, 45.0):
        for p_yaw in (0.0, 45.0):
          cases.append(
            (
              (cx, cy),
              math.radians(p_yaw),
              (cx + d * math.cos(th), cy + d * math.sin(th)),
              math.radians(b_yaw),
              d,
              k * 45,
            )
          )
  n = len(cases)
  hook = _Touch(n)
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
      spawns=[(c[0], c[1]) for c in cases],
      box_poses=[(c[2], c[3]) for c in cases],
      phases=GRASP_PHASES,
      hook=hook,
    )
  finally:
    scripted.waypoints = full
  by_d: dict[float, dict] = {}
  for i, c in enumerate(cases):
    touched = sum(int(v[i]) for v in hook.hand.values())
    e = by_d.setdefault(c[4], {"cases": 0, "touched": 0, "lifted": 0, "worst": []})
    e["cases"] += 1
    e["touched"] += int(touched > 0)
    e["lifted"] += int(rows[i]["lifted"])
    if touched:
      e["worst"].append(
        dict(
          dir_deg=c[5],
          block_yaw=round(math.degrees(c[3])),
          pin_yaw=round(math.degrees(c[1])),
          steps=touched,
        )
      )
  for d, e in sorted(by_d.items()):
    print(
      f"distance {d:.3f} m: hand touched the block in {e['touched']}/{e['cases']}, lifted {e['lifted']}/{e['cases']}"
    )
  return {"by_distance": {f"{d:.3f}": e for d, e in by_d.items()}}


def main() -> None:
  p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  p.add_argument("mode", choices=("scripted", "separation"))
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--dr", default="DR-scene")
  p.add_argument("--goal", default="insert")
  p.add_argument("--random", type=int, default=0)
  p.add_argument("--above-top", type=float, default=None)
  p.add_argument(
    "--clear-height",
    type=float,
    default=0.07,
    help="pre-grasp / lift / carry / retreat over the grasp and place (m); 7 cm: the pin's bottom 14 mm over the "
    "block's top in the carry",
  )
  p.add_argument("--distances", type=float, nargs="+", default=[0.08, 0.10, 0.12])
  p.add_argument("--json", type=pathlib.Path, default=None)
  args = p.parse_args()
  out = scripted_run(args) if args.mode == "scripted" else separation(args)
  out["args"] = {k: str(v) for k, v in vars(args).items()}
  if args.json is not None:
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out, indent=1, default=str))
    print(f"wrote {args.json}")


if __name__ == "__main__":
  main()
