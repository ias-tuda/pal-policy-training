"""Plan S11.2's step 0 (`pick-place-s11-bimanual-plan.md`, *S11.2*, items 1 and 2), no training.

``ledger`` -- item 1, the reward ledger. The SEQ chain (`eval_chain.py`: L-T on the left, R-ST / R-ST-W
mirrored on the right, the right handed off on the eval's trigger) runs in JOINT's env under S11-BI, and
every reward term is logged per env and step (`RewardManager._step_reward` x dt: what PPO receives). Per
episode it marks five moments of the right's stack and reads the per-step income there:

* ``hover``: B held, over A (xy < 2 cm), its bottom 3-5 cm above A's top;
* ``touch``: B held, over A, its bottom within 4 mm of A's top;
* ``release``: the right's release latch fires (J-release after the lift);
* ``up5``: after the release, the right TCP 5 cm over B's top;
* ``handed``: 0.5 s after the right's hand-off started.

and whether the gate ("A placed", `mdp.base_placed_bool`) drops within +/-20 steps of ``touch``, and
which of its conditions dropped it: A at T, the left released (no grasp, jaw open), A still, was_lifted,
no left pad on A.

``joint`` -- item 2. A BI-J / BI-JM checkpoint: each action dimension's std (the policy's ``std`` or
``log_std``), and over a 25 s `eval_bimanual` rollout where the arm-arm clearance drops under 5 cm --
which capsule pair (forearm / wrist / hand of each arm) and in which phase of the task.

  REWARD=S11-BI ARMS=both ... python scripts/s112_step0.py ledger --lt L-T.pt --rst R-ST-W.pt --out d/
  REWARD=S11-BI ARMS=both ... python scripts/s112_step0.py joint --checkpoint BI-J/model_4999.pt --out d/
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

#: The moments of the ledger and the gate window around ``touch``, steps.
MOMENTS = ("hover", "touch", "release", "up5", "handed")
GATE_WINDOW = 20
#: The clearance the HH margin starts at, m (`bimanual.HH_MARGIN`).
CLOSE = 0.05


def parse_args(argv=None):
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("mode", choices=("ledger", "joint"))
  p.add_argument("--lt", type=pathlib.Path)
  p.add_argument("--rst", type=pathlib.Path)
  p.add_argument("--checkpoint", type=pathlib.Path)
  p.add_argument("--task", default="Mjlab-PickPlace-Pal-Kangaroo-Grippers")
  p.add_argument("--episodes", type=int, default=64)
  p.add_argument("--num-envs", type=int, default=64)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--episode-s", type=float, default=None)
  p.add_argument("--right-timeout", type=float, default=None)
  p.add_argument("--obs-noise", choices=("off", "on"), default="on")
  p.add_argument("--out", type=pathlib.Path, required=True)
  return p.parse_args(argv)


class GateParts:
  """The gate's conditions, apart (`mdp.base_placed_bool` = all of them)."""

  def __init__(self, env) -> None:
    p = env.cfg.rewards["placed"].params
    self.env, self.p = env, p

  def __call__(self) -> dict[str, torch.Tensor]:
    from pal_mjlab.tasks.manipulation.mdp.observations import (
      _at_goal,
      _grasped_bool,
      _jaw_open_bool,
      _pad_contact_any,
      _was_lifted_bool,
    )

    env, p = self.env, self.p
    obj = env.scene[p["object_name"]]
    cmd = env.command_manager.get_term(p["command_name"]).command
    parts = {
      "at_t": _at_goal(cmd, obj.data.root_link_pos_w, p["threshold"]),
      "left_released": ~_grasped_bool(env, p["grasp_sensor_name"])
      & _jaw_open_bool(env, p["jaw_cfg"], p["jaw_open"]),
      "a_still": torch.norm(obj.data.root_link_lin_vel_w, dim=-1)
      < p.get("velocity_threshold", 0.05),
      "was_lifted": _was_lifted_bool(env, p["grasp_sensor_name"], p["support_sensor_name"]),
      "no_left_pad": ~_pad_contact_any(env, p["grasp_sensor_name"]),
    }
    gate = torch.ones_like(parts["at_t"])
    for v in parts.values():
      gate = gate & v
    parts["gate"] = gate
    return parts


def _term_groups(names: list[str]) -> dict[str, list[int]]:
  """Reward terms by owner: the right half (``right_*``), HH (``arm_arm_*``), the rest is the left half
  and the regularisers kept once over both arms (`bimanual.SHARED`)."""
  from pal_mjlab.tasks.manipulation.pick_place.bimanual import SHARED

  g = {"right": [], "left": [], "shared": [], "hh": []}
  for i, n in enumerate(names):
    if n.startswith("right_"):
      g["right"].append(i)
    elif n.startswith("arm_arm_"):
      g["hh"].append(i)
    elif n in SHARED:
      g["shared"].append(i)
    else:
      g["left"].append(i)
  return g


def ledger(args) -> dict:
  from eval_bimanual import Reads, build_env, summarise
  from eval_chain import MirrorAdapter, OneArmActor
  from pal_mjlab.tasks.manipulation.mdp.handoff import HOME
  from pal_mjlab.tasks.manipulation.mdp.observations import (
    _grasped_bool,
    _release_latch,
  )
  from pal_mjlab.tasks.manipulation.pick_place import bimanual as bi

  args.episode_s = args.episode_s or 30.0
  args.right_timeout = args.right_timeout or 24.0
  env = build_env(args)
  ad = MirrorAdapter(env)
  lt, rst = OneArmActor(args.lt, args.device), OneArmActor(args.rst, args.device)
  reads = Reads(env)
  term = env.action_manager.get_term("joint_pos")
  rm = env.reward_manager
  names = list(rm._term_names)
  scale = env.step_dt if getattr(rm, "_scale_by_dt", True) else 1.0
  gate_parts = GateParts(env)
  robot = env.scene["robot"]
  a_ent, b_ent = env.scene["object"], env.scene[bi.CUBE_B]
  site_r = robot.site_names.index("tcp_right")
  right_timeout = round(args.right_timeout / env.step_dt)
  steps = round(args.episode_s / env.step_dt)
  n = env.num_envs
  keys = "rew gate parts held_b dxy gap tcp_up released_r phase_r go".split()
  log = {k: [] for k in keys}
  done_at = torch.full((n,), -1, dtype=torch.long, device=env.device)
  rows = {}
  obs, _ = env.reset()
  rl = reads.right_latch
  with torch.inference_mode():
    for t_step in range(steps + 5):
      start = reads.step(right_timeout)
      t = ad.terms(obs, "actor")
      a_l = lt(ad.left(t))
      go = (term.phase[:, term.arm_index("left")] >= HOME).unsqueeze(-1)
      a_r = torch.where(go, rst(ad.right(t)), torch.zeros_like(a_l))
      term.start_handoff("right", start & go.squeeze(-1))
      obs, _, term_done, trunc, _ = env.step(ad.actions(a_l, a_r))
      # The state the rewards of this step were computed on (before any reset of a done env).
      log["rew"].append((rm._step_reward * scale).clone())
      gp = gate_parts()
      log["gate"].append(gp.pop("gate"))
      log["parts"].append(torch.stack(list(gp.values()), -1))
      a = a_ent.data.root_link_pos_w
      b = b_ent.data.root_link_pos_w
      log["held_b"].append(_grasped_bool(env, "right_pad_contact").clone())
      log["dxy"].append(torch.norm((b - a)[:, :2], dim=-1))
      log["gap"].append((b[:, 2] - 0.025) - (a[:, 2] + 0.025))
      log["tcp_up"].append(robot.data.site_pos_w[:, site_r, 2] - (b[:, 2] + 0.025))
      log["released_r"].append(_release_latch(env, *rl).released.clone())
      log["phase_r"].append(term.phase[:, term.arm_index("right")].clone())
      log["go"].append(go.squeeze(-1).clone())
      done = (term_done | trunc) & (done_at < 0)
      fin = torch.nonzero(done, as_tuple=False).flatten()
      if fin.numel():
        before = len(reads.done)
        reads.finish(fin, env.step_dt)
        for j, i in enumerate(fin.tolist()):
          rows[i] = reads.done[before + j]
        done_at[fin] = t_step
      if bool((done_at >= 0).all()):
        break
  L = {k: torch.stack(v, 1).cpu().numpy() for k, v in log.items()}
  part_names = ["at_t", "left_released", "a_still", "was_lifted", "no_left_pad"]
  np.savez_compressed(
    args.out / "ledger.npz", **L, done_at=done_at.cpu().numpy(), term_names=np.array(names),
    part_names=np.array(part_names), dt=np.array(env.step_dt),
  )  # fmt: skip
  groups = _term_groups(names)

  per_ep = []
  for k in range(n):
    end = int(done_at[k]) + 1
    held, dxy, gap = L["held_b"][k, :end], L["dxy"][k, :end], L["gap"][k, :end]
    over = dxy < 0.02
    mom = {}
    idx = np.flatnonzero(held & over & (gap >= 0.03) & (gap <= 0.05))
    mom["hover"] = int(idx[0]) if idx.size else None
    idx = np.flatnonzero(held & over & (gap < 0.004))
    mom["touch"] = int(idx[0]) if idx.size else None
    idx = np.flatnonzero(L["released_r"][k, :end])
    mom["release"] = int(idx[0]) if idx.size else None
    if mom["release"] is not None:
      idx = np.flatnonzero(L["tcp_up"][k, mom["release"] : end] >= 0.05)
      mom["up5"] = mom["release"] + int(idx[0]) if idx.size else None
    else:
      mom["up5"] = None
    idx = np.flatnonzero(L["phase_r"][k, :end] > 0)
    mom["handed"] = int(idx[0]) + 25 if idx.size and int(idx[0]) + 25 < end else None
    income = {}
    for m, s in mom.items():
      if s is None:
        continue
      r = L["rew"][k, s]
      income[m] = {g: float(r[ids].sum()) for g, ids in groups.items()} | {
        "total": float(r.sum()),
        "terms": {names[i]: float(r[i]) for i in range(len(names)) if abs(r[i]) > 1e-9},
      }
    flick = None
    if mom["touch"] is not None:
      lo, hi = max(mom["touch"] - GATE_WINDOW, 0), min(mom["touch"] + GATE_WINDOW + 1, end)
      g = L["gate"][k, lo:hi]
      drops = np.flatnonzero(g[:-1] & ~g[1:])
      causes = []
      for d in drops:
        parts = L["parts"][k, lo + d + 1]
        causes.append([part_names[j] for j in range(len(part_names)) if not parts[j]])
      flick = {
        "gate_on_at_touch": bool(L["gate"][k, mom["touch"]]),
        "drops": len(drops),
        "steps_off": int((~g).sum()),
        "causes": causes,
      }
    per_ep.append({"id": k, "moments": mom, "income": income, "gate_at_touch": flick, "success": rows[k]["success"]})

  def med(m, g):
    v = [e["income"][m][g] for e in per_ep if m in e["income"]]
    return (statistics.median(v), len(v)) if v else (None, 0)

  table = {m: {g: med(m, g) for g in ("right", "left", "shared", "hh", "total")} for m in MOMENTS}
  # The right's terms at each moment, median over the episodes that reach it.
  term_table = {}
  for m in MOMENTS:
    vals = {}
    for e in per_ep:
      for nme, v in e["income"].get(m, {}).get("terms", {}).items():
        vals.setdefault(nme, []).append(v)
    term_table[m] = {nme: statistics.median(v) for nme, v in sorted(vals.items())}
  touched = [e for e in per_ep if e["gate_at_touch"] is not None]
  flicker = {
    "episodes_touching": len(touched),
    "gate_on_at_touch": sum(e["gate_at_touch"]["gate_on_at_touch"] for e in touched),
    "episodes_with_a_drop": sum(e["gate_at_touch"]["drops"] > 0 for e in touched),
    "drop_causes": {},
  }
  for e in touched:
    for c in e["gate_at_touch"]["causes"]:
      key = "+".join(c) or "none"
      flicker["drop_causes"][key] = flicker["drop_causes"].get(key, 0) + 1
  out = {
    "mode": "ledger",
    "lt": str(args.lt),
    "rst": str(args.rst),
    "per_step_income_median": table,
    "terms_median": term_table,
    "gate_around_touch": flicker,
    "chain": summarise([rows[k] for k in range(n)]),
    "episodes": per_ep,
  }
  env.close()
  return out


def joint(args) -> dict:
  from eval_bimanual import Reads, build_env, load_policy, summarise
  from pal_mjlab.tasks.manipulation.mdp import bimanual as mb
  from pal_mjlab.tasks.manipulation.mdp.observations import _grasped_bool
  from pal_mjlab.tasks.manipulation.pick_place import bimanual as bi

  args.episode_s = args.episode_s or 25.0
  args.right_timeout = args.right_timeout or 19.0
  sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["actor_state_dict"]
  std_key = next(k for k in sd if k.startswith("distribution.") and "std" in k)
  std = sd[std_key].float()
  if "log_std" in std_key:
    std = std.exp()
  env = build_env(args)
  wrapped, policy = load_policy(args, env)
  term = env.action_manager.get_term("joint_pos")
  tnames = [x.split("/")[-1] for x in term.target_names]
  reads = Reads(env)
  robot = env.scene["robot"]
  a_ent, b_ent = env.scene["object"], env.scene[bi.CUBE_B]
  right_timeout = round(args.right_timeout / env.step_dt)
  n = env.num_envs
  cap = [c[0] for c in mb.HH_CAPSULES]
  rows: dict[int, dict] = {}
  done_at = torch.full((n,), -1, dtype=torch.long, device=env.device)
  close_log = []  # (env, step, clearance, pair, phase label)
  obs, _ = wrapped.reset()
  with torch.inference_mode():
    for t_step in range(round(args.episode_s / env.step_dt) + 5):
      start = reads.step(right_timeout)
      term.start_handoff("right", start)
      # Clearance per capsule pair, before the step.
      pl, ql, rl = mb._capsules(env, "left")
      pr, qr, rr = mb._capsules(env, "right")
      m = pl.shape[1]
      d = mb.segment_distance(
        pl.unsqueeze(2).expand(-1, m, m, 3), ql.unsqueeze(2).expand(-1, m, m, 3),
        pr.unsqueeze(1).expand(-1, m, m, 3), qr.unsqueeze(1).expand(-1, m, m, 3),
      ) - rl.view(1, m, 1) - rr.view(1, 1, m)  # fmt: skip
      flat = d.flatten(1)
      cmin, arg = flat.min(-1)
      held_b = _grasped_bool(env, "right_pad_contact")
      a, b = a_ent.data.root_link_pos_w, b_ent.data.root_link_pos_w
      over = torch.norm((b - a)[:, :2], dim=-1) < 0.03
      ph_l = term.phase[:, term.arm_index("left")]
      a_placed = reads._last["a_placed"]
      for i in torch.nonzero((cmin < CLOSE) & (done_at < 0), as_tuple=False).flatten().tolist():
        if ph_l[i] == 0 and not bool(a_placed[i]):
          label = "left placing A"
        elif ph_l[i] > 0 and ph_l[i] < 3:
          label = "left handing off (up/out)"
        elif bool(held_b[i]) and bool(over[i]):
          label = "right holding B over A"
        elif bool(held_b[i]):
          label = "right carrying B"
        else:
          label = "other"
        k = int(arg[i])
        close_log.append((i, t_step, float(cmin[i]), f"L-{cap[k // m]} / R-{cap[k % m]}", label))
      actions = policy(obs)
      obs, _, dones, _ = wrapped.step(actions)
      done = dones.bool() & (done_at < 0)
      fin = torch.nonzero(done, as_tuple=False).flatten()
      if fin.numel():
        before = len(reads.done)
        reads.finish(fin, env.step_dt)
        for j, i in enumerate(fin.tolist()):
          rows[i] = reads.done[before + j]
        done_at[fin] = t_step
      if bool((done_at >= 0).all()):
        break
  by = {}
  for _, _, c, pair, label in close_log:
    by.setdefault(label, {}).setdefault(pair, []).append(c)
  eps_close = {}
  for i, _, _, _, label in close_log:
    eps_close.setdefault(label, set()).add(i)
  out = {
    "mode": "joint",
    "checkpoint": str(args.checkpoint),
    "std_key": std_key,
    "action_std": dict(zip(tnames, [float(v) for v in std], strict=True)),
    "close_steps_by_phase_and_pair": {
      lab: {pair: {"steps": len(v), "min_clearance_m": min(v)} for pair, v in d.items()}
      for lab, d in by.items()
    },
    "episodes_close_by_phase": {lab: len(s) for lab, s in eps_close.items()},
    "summary": summarise([rows[k] for k in range(n)]),
  }
  env.close()
  return out


def main(argv=None) -> None:
  args = parse_args(argv)
  args.num_envs = args.episodes
  args.out.mkdir(parents=True, exist_ok=True)
  out = ledger(args) if args.mode == "ledger" else joint(args)
  name = "ledger.json" if args.mode == "ledger" else f"joint_{args.checkpoint.parent.name[-6:]}_{args.checkpoint.stem}.json"
  (args.out / name).write_text(json.dumps(out, indent=1, default=lambda o: None if isinstance(o, float) and math.isnan(o) else str(o)))
  slim = {k: v for k, v in out.items() if k not in ("episodes",)}
  print(json.dumps(slim, indent=1, default=str)[:12000])


if __name__ == "__main__":
  sys.exit(main())
