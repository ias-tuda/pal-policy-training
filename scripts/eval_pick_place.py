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
    self.done: list[dict[str, bool | int]] = []

  def observe(self, **flags: torch.Tensor) -> None:
    for name, value in flags.items():
      self.live[name] |= value
    newly = flags["placed"] & (self.first_place < 0)
    self.first_place = torch.where(newly, self.age, self.first_place)
    self.age += 1

  def finish(self, env_ids: torch.Tensor) -> None:
    """Bank the episodes that just ended and clear their latches."""
    for i in env_ids.tolist():
      row: dict[str, bool | int] = {n: bool(self.live[n][i]) for n in self.FIELDS}
      row["steps_to_place"] = int(self.first_place[i])
      row["length"] = int(self.age[i])
      self.done.append(row)
    for name in self.FIELDS:
      self.live[name][env_ids] = False
    self.first_place[env_ids] = -1
    self.age[env_ids] = 0


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


def summarise(tally: Tally, step_dt: float, episodes: int) -> dict:
  rows = tally.done[:episodes]
  n = len(rows)
  rate = lambda k: sum(r[k] for r in rows) / n  # noqa: E731
  placed_times = [
    r["steps_to_place"] for r in rows if r["placed"] and r["steps_to_place"] >= 0
  ]
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
      tally.observe(**probe(env, grasp_sensor))
      obs, _, dones, extras = wrapped.step(policy(obs))
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
