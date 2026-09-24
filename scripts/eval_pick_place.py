"""Score a trained pick-and-place policy on the rates that name its failure mode.

A training curve is evidence about training. This is the closed-loop measurement the plan's
exit criteria are written against: run a checkpoint for N complete episodes and report what
fraction of them the policy actually finished, and *how* the rest failed.

    uv run --extra cu128 python scripts/eval_pick_place.py \
      --checkpoint logs/rsl_rl/kangaroo_pick_place/<run>/model_20000.pt \
      --episodes 200 --num-envs 64 --json artifacts/eval/c1_seed1.json

The two placed rates are the point
--------------------------------------------------------------------------------
`_placed_bool` counts `released = ¬grasped`, and the goal sits at the cube's resting
height, so **a cube that was never picked up is "placed"** and sliding it 20 cm scores a
success. C0 measured exactly that: every arm reaching success with `Episode_Reward/grasp`
at 0. So this reports both

  placed          at the goal, released, settled  -- what R-A and R-C are scored on
  placed & lifted the same, and the object left the table at some point this episode

and they are computed the same way for every arm, from the same rollout, regardless of
which gating that arm's own reward happens to use. That is what makes a C1 number and a
C2 number comparable.

The four diagnostic rates (plan, "Metrics and diagnostics")
--------------------------------------------------------------------------------
  grasped-ever    near 0 with a saturated reach term is the Isaac Lab #204 signature:
                  reach-but-never-close
  lifted-ever     high grasped, low lifted = grasps but cannot lift, or is sliding
  held-at-goal    at the goal and still gripping when the clock ran out -- the
                  Text2Reward hold-at-goal failure
  placed          the number that counts

Horizon
--------------------------------------------------------------------------------
Not the `--play` cfg as-is: `apply_play_overrides` sets `episode_length_s` to 1e9, so
episodes never time out and nothing ever completes. This takes the play cfg (no observation
corruption, no curriculum) and restores the *training* episode length, which is the horizon
the policy was optimised for and the only one these rates mean anything against.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import asdict

import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from pal_mjlab.tasks.manipulation.mdp.observations import (
  _grasped_bool,
  _on_surface_bool,
  _placed_bool,
)
from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import (
  GOAL_THRESHOLD,
  OBJECT,
)
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import COMMAND, SUPPORT_SENSOR

TASK = "Mjlab-PickPlace-Pal-Kangaroo-Grippers"

#: "Near the target" for the transport measurements: *horizontal* distance while the cube is
#: held. Horizontal on purpose -- `Metrics/goal_pose/position_error` is a 3-D norm, so a cube
#: carried over the target and hovering 6 cm above it reads "6 cm off" there, which is how the
#: c1a screen was first misread as "no transport" (2026-09-24, S5.2 base selection).
NEAR_XY = 0.03


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--checkpoint", type=pathlib.Path, required=True, help="model_*.pt")
  p.add_argument("--task", default=TASK)
  p.add_argument("--episodes", type=int, default=200, help="complete episodes to score")
  p.add_argument("--num-envs", type=int, default=64)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument(
    "--json", type=pathlib.Path, default=None, help="also write the result"
  )
  return p.parse_args()


class Tally:
  """Per-episode outcome flags, latched across the steps of an episode.

  Latched rather than sampled because every quantity here is "did this *ever* happen in
  this episode". Placed states persist -- the cube stays where it was put -- so reading the
  state at the top of each step catches them even though the terminal state, which the
  env has already reset by the time `step` returns, is never seen.

  Except when the placement is what *ended* the episode. An arm with a `success`
  termination is reset in the same step the cube comes to rest on the goal, so the placed
  state never exists at the top of any step; those episodes have to be banked from the
  termination itself, with `terminal`, before `finish`. Until 2026-09-23 they were not,
  and every such episode was counted as a failure (found by S5.0's scripted rollout).
  """

  FIELDS = (
    "grasped",
    "lifted",
    "placed",
    "placed_and_lifted",
    "dropped",
    "held_at_goal",
  )

  def __init__(self, num_envs: int, device: str) -> None:
    self.num_envs = num_envs
    z = lambda: torch.zeros(num_envs, dtype=torch.bool, device=device)  # noqa: E731
    self.live = {name: z() for name in self.FIELDS}
    self.first_place = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    self.age = torch.zeros(num_envs, dtype=torch.long, device=device)
    # Transport accumulators (`measure`): steps held, horizontal distance to the target while
    # held (sum for the mean, running min), and height above the target while held within
    # NEAR_XY horizontally.
    zf = lambda v: torch.full((num_envs,), v, device=device)  # noqa: E731
    self.held_n = torch.zeros(num_envs, dtype=torch.long, device=device)
    self.dxy_sum = zf(0.0)
    self.dxy_min = zf(float("inf"))
    self.near_n = torch.zeros(num_envs, dtype=torch.long, device=device)
    self.dz_sum = zf(0.0)
    self.done: list[dict[str, bool | int | float | None]] = []

  def observe(self, **flags: torch.Tensor) -> None:
    for name, value in flags.items():
      self.live[name] |= value
    newly = flags["placed"] & (self.first_place < 0)
    self.first_place = torch.where(newly, self.age, self.first_place)
    self.age += 1

  def measure(self, held: torch.Tensor, dxy: torch.Tensor, dz: torch.Tensor) -> None:
    """Accumulate this step's transport geometry for the episodes that hold the cube.

    ``dxy`` is the cube's horizontal distance to the target, ``dz`` its height above the
    target (the target is the cube's resting pose on the table, so ``dz`` is 0 when it
    rests there). Only held steps count; the hover height only counts held steps within
    `NEAR_XY` horizontally. Call once per step, alongside `observe`.
    """
    held = held.bool()
    near = held & (dxy < NEAR_XY)
    self.held_n += held.long()
    self.dxy_sum += torch.where(held, dxy, torch.zeros_like(dxy))
    self.dxy_min = torch.where(held, torch.minimum(self.dxy_min, dxy), self.dxy_min)
    self.near_n += near.long()
    self.dz_sum += torch.where(near, dz, torch.zeros_like(dz))

  def terminal(self, success: torch.Tensor) -> None:
    """Credit the episodes a `success` termination just ended, which `observe` cannot see.

    `success` fires only on a placement, so it sets `placed`; it counts as `placed and
    lifted` if a lift was observed earlier in the same episode -- the same test
    `_placed_bool` applies through the `was_lifted` latch. Call after `step`, before
    `finish`. Does not advance `age`: this is the same step `observe` already counted.
    """
    self.live["placed"] |= success
    self.live["placed_and_lifted"] |= success & self.live["lifted"]
    newly = success & (self.first_place < 0)
    self.first_place = torch.where(newly, self.age - 1, self.first_place)

  def finish(self, env_ids: torch.Tensor) -> None:
    """Bank the episodes that just ended and clear their latches."""
    for i in env_ids.tolist():
      row: dict[str, bool | int | float | None] = {
        n: bool(self.live[n][i]) for n in self.FIELDS
      }
      row["steps_to_place"] = int(self.first_place[i])
      row["length"] = int(self.age[i])
      held, near = int(self.held_n[i]), int(self.near_n[i])
      row["held_steps"] = held
      row["held_dxy_mean"] = float(self.dxy_sum[i]) / held if held else None
      row["held_dxy_min"] = float(self.dxy_min[i]) if held else None
      row["near_xy_while_held"] = near > 0
      row["hover_dz_mean"] = float(self.dz_sum[i]) / near if near else None
      self.done.append(row)
    for name in self.FIELDS:
      self.live[name][env_ids] = False
    self.first_place[env_ids] = -1
    self.age[env_ids] = 0
    self.held_n[env_ids] = 0
    self.dxy_sum[env_ids] = 0.0
    self.dxy_min[env_ids] = float("inf")
    self.near_n[env_ids] = 0
    self.dz_sum[env_ids] = 0.0


def build_env(args: argparse.Namespace) -> ManagerBasedRlEnv:
  cfg = load_env_cfg(args.task, play=True)
  # `play=True` removes observation corruption and the curriculum, both of which we want
  # gone -- but it also sets episode_length_s to 1e9. Put the training horizon back, or no
  # episode ever ends and there is nothing to count.
  cfg.episode_length_s = load_env_cfg(args.task, play=False).episode_length_s
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  return ManagerBasedRlEnv(cfg=cfg, device=args.device)


def load_policy(args: argparse.Namespace, env: ManagerBasedRlEnv):
  wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(load_rl_cfg(args.task)), device=args.device)
  runner.load(
    str(args.checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=args.device,
  )
  return wrapped, runner.get_inference_policy(device=args.device)


def probe(env: ManagerBasedRlEnv, grasp_sensor: str) -> dict[str, torch.Tensor]:
  """Every per-step boolean, read from physics rather than from the arm's own reward."""
  obj = env.scene[OBJECT]
  command = env.command_manager.get_term(COMMAND)
  grasped = _grasped_bool(env, grasp_sensor)
  on_surface = _on_surface_bool(env, SUPPORT_SENSOR)
  at_goal = (
    torch.norm(command.command - obj.data.root_link_pos_w, dim=-1) < GOAL_THRESHOLD
  )
  common = (env, COMMAND, OBJECT, grasp_sensor, GOAL_THRESHOLD, 0.05)
  return {
    "grasped": grasped,
    "lifted": grasped & ~on_surface,
    "placed": _placed_bool(*common),
    "placed_and_lifted": _placed_bool(*common, SUPPORT_SENSOR),
    # The object is on the floor, not the table: `object_dropped`'s own condition.
    "dropped": obj.data.root_link_pos_w[:, 2]
    < env.cfg.terminations["object_dropped"].params["min_height"],
    "held_at_goal": at_goal & grasped,
  }


def geometry(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
  """The cube's horizontal distance to the target and its height above it. (B,), (B,)"""
  pos = env.scene[OBJECT].data.root_link_pos_w
  goal = env.command_manager.get_term(COMMAND).command
  return torch.norm((pos - goal)[:, :2], dim=-1), pos[:, 2] - goal[:, 2]


def success_flags(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Which envs the last `step` ended through the `success` termination.

  The termination manager keeps each term's value from the step it just computed, before
  the reset, so this is readable after `step` returns. All False for an arm without one.
  """
  tm = env.termination_manager
  if "success" not in tm.active_terms:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  return tm.get_term("success").bool()


def summarise(tally: Tally, step_dt: float, episodes: int) -> dict:
  rows = tally.done[:episodes]
  n = len(rows)
  rate = lambda k: sum(r[k] for r in rows) / n  # noqa: E731
  placed_times = [
    r["steps_to_place"] for r in rows if r["placed"] and r["steps_to_place"] >= 0
  ]
  held = [r for r in rows if r.get("held_steps")]
  near = [r["hover_dz_mean"] for r in rows if r.get("hover_dz_mean") is not None]
  mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
  median = lambda xs: sorted(xs)[len(xs) // 2] if xs else None  # noqa: E731
  transport = {
    "episodes_held": len(held),
    "held_dxy_mean_m": mean([r["held_dxy_mean"] for r in held]),
    "held_dxy_min_mean_m": mean([r["held_dxy_min"] for r in held]),
    "held_dxy_min_median_m": median([r["held_dxy_min"] for r in held]),
    "within_3cm_xy_while_held": rate("near_xy_while_held") if n else None,
    "hover_dz_mean_m": mean(near),
    "hover_dz_median_m": median(near),
  }
  return {
    "episodes": n,
    "placed": rate("placed"),
    "placed_and_lifted": rate("placed_and_lifted"),
    "grasped_ever": rate("grasped"),
    "lifted_ever": rate("lifted"),
    "held_at_goal": rate("held_at_goal"),
    "dropped": rate("dropped"),
    "mean_episode_length_s": sum(r["length"] for r in rows) / n * step_dt,
    "time_to_place_s": (sum(placed_times) / len(placed_times) * step_dt)
    if placed_times
    else None,
    **transport,
  }


def report(result: dict, rewards: dict[str, float], args: argparse.Namespace) -> None:
  print(f"\ncheckpoint   {args.checkpoint}")
  print(f"task         {args.task}")
  print(
    f"episodes     {result['episodes']} over {args.num_envs} envs, seed {args.seed}\n"
  )

  print("  THE TWO PLACED RATES ---------------------------------------------")
  print(
    f"    placed                   {result['placed']:6.1%}   at goal, released, settled"
  )
  print(
    f"    placed & lifted          {result['placed_and_lifted']:6.1%}   ... and it left the table"
  )
  slid = result["placed"] - result["placed_and_lifted"]
  print(f"    -> slid, never picked up {slid:6.1%}")

  print("\n  DIAGNOSTIC RATES -------------------------------------------------")
  print(
    f"    grasped, ever            {result['grasped_ever']:6.1%}   ~0 with high reach = Isaac Lab #204"
  )
  print(
    f"    lifted, ever             {result['lifted_ever']:6.1%}   high grasp + low lift = cannot lift"
  )
  print(
    f"    held at goal             {result['held_at_goal']:6.1%}   the hold-at-goal failure"
  )
  print(f"    dropped                  {result['dropped']:6.1%}")

  print("\n  TRANSPORT (horizontal, while held) -------------------------------")
  fmt = lambda v, u="m": "n/a" if v is None else f"{v:6.3f} {u}"  # noqa: E731
  print(f"    episodes that held it    {result['episodes_held']:6d}")
  print(f"    mean xy dist while held  {fmt(result['held_dxy_mean_m'])}")
  print(
    f"    closest xy while held    {fmt(result['held_dxy_min_median_m'])}  (median)"
  )
  w3 = result["within_3cm_xy_while_held"]
  print(f"    within 3 cm xy, held     {'n/a' if w3 is None else f'{w3:6.1%}'}")
  print(
    f"    hover height over target {fmt(result['hover_dz_mean_m'])}  (z - z_target, held within 3 cm xy)"
  )

  print("\n  TIMING -----------------------------------------------------------")
  ttp = result["time_to_place_s"]
  print(f"    mean episode             {result['mean_episode_length_s']:6.2f} s")
  print(f"    time to place            {'n/a' if ttp is None else f'{ttp:6.2f} s'}")

  if rewards:
    print("\n  PER-TERM REWARD (mean episode sum) --------------------------------")
    for name, value in sorted(rewards.items(), key=lambda kv: -abs(kv[1])):
      print(f"    {name:24s} {value:9.4f}")
  print()


def main() -> None:
  args = parse_args()
  torch.manual_seed(args.seed)
  env = build_env(args)
  wrapped, policy = load_policy(args, env)

  gripper_cfg = env.cfg.rewards.get("grasp") or env.cfg.rewards["lift"]
  grasp_sensor = (
    gripper_cfg.params.get("sensor_name") or gripper_cfg.params["grasp_sensor_name"]
  )

  tally = Tally(args.num_envs, args.device)
  reward_sums: dict[str, list[float]] = {}

  obs, _ = wrapped.reset()
  with torch.inference_mode():
    while len(tally.done) < args.episodes:
      flags = probe(env, grasp_sensor)
      tally.observe(**flags)
      tally.measure(flags["grasped"], *geometry(env))
      obs, _, dones, extras = wrapped.step(policy(obs))
      tally.terminal(success_flags(env))
      finished = torch.nonzero(dones, as_tuple=False).flatten()
      if finished.numel():
        tally.finish(finished)
        for key, value in extras.get("log", {}).items():
          if key.startswith("Episode_Reward/"):
            reward_sums.setdefault(key.removeprefix("Episode_Reward/"), []).append(
              float(value)
            )

  result = summarise(tally, env.step_dt, args.episodes)
  rewards = {k: sum(v) / len(v) for k, v in reward_sums.items()}
  report(result, rewards, args)

  if args.json:
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
      json.dumps(
        {
          "checkpoint": str(args.checkpoint),
          "task": args.task,
          "num_envs": args.num_envs,
          "seed": args.seed,
          **result,
          "reward_terms": rewards,
        },
        indent=2,
      )
    )
    print(f"wrote {args.json}")

  env.close()


if __name__ == "__main__":
  main()
