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

Transport and speed, while held
--------------------------------------------------------------------------------
  transport       horizontal distance to the target while held, and the height over it
                  while held within 3 cm -- a hover, not a miss (S5.2 base selection)
  speed           per-episode peak cube and TCP speed while held, over one control step
                  -- the read of plan S7's speed bound
  release         the episode's last let-go: the cube's xy distance to the target, its
                  height above it, and its speed -- a set-down vs a cube let go short

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
from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS_LEFT_ARM
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import COMMAND, ROBOT, SUPPORT_SENSOR

TASK = "Mjlab-PickPlace-Pal-Kangaroo-Grippers"

#: "Near the target" for the transport measurements: *horizontal* distance while the cube is
#: held. Horizontal on purpose -- `Metrics/goal_pose/position_error` is a 3-D norm, so a cube
#: carried over the target and hovering 6 cm above it reads "6 cm off" there, which is how the
#: c1a screen was first misread as "no transport" (2026-09-24, S5.2 base selection).
NEAR_XY = 0.03
#: The gripper's tool centre point, whose speed while holding is S7's speed-bound read.
TCP_SITE = KANGAROO_GRIPPERS_LEFT_ARM.gripper("left").tcp_site

#: Plan S8's plant reads (`--plant`): the joints a gain code changes that the policy does not
#: drive -- the parked right arm and the pelvis -- the arm it does drive, and all of them.
PARKED_JOINTS = r"(arm_right_[1-7]|pelvis_[12])_joint"
DRIVEN_ARM_JOINTS = r"arm_left_[1-7]_joint"
GAINED_JOINTS = r"(pelvis_[12]|arm_(left|right)_[1-7])_joint"
#: A joint counts as at its effort limit within this fraction of it: the limit clamps the
#: torque exactly, and the margin absorbs float32 round-off. As `hold_check.py`.
SATURATION_FRACTION = 0.999


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
  p.add_argument(
    "--plant",
    action="store_true",
    help="also read the plant (plan S8): parked-arm/pelvis hold error, the driven arm's "
    "tracking error while carrying, effort saturation, cube-table force at set-down",
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
    # Speed accumulators (`measure_speed`): the fastest the cube and the TCP moved over one
    # control step while the cube stayed held, per episode, and the positions they moved from.
    self.prev_obj = torch.zeros(num_envs, 3, device=device)
    self.prev_tcp = torch.zeros(num_envs, 3, device=device)
    self.prev_held = torch.zeros(num_envs, dtype=torch.bool, device=device)
    self.speed_n = torch.zeros(num_envs, dtype=torch.long, device=device)
    self.obj_v_peak = zf(0.0)
    self.tcp_v_peak = zf(0.0)
    # The last release of the episode (`measure_speed` with a goal): the cube's horizontal
    # distance to the target, its height above it, and its speed, the step the jaw let go.
    self.released = torch.zeros(num_envs, dtype=torch.bool, device=device)
    self.rel_dxy = zf(0.0)
    self.rel_dz = zf(0.0)
    self.rel_v = zf(0.0)
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

  def measure_speed(
    self,
    held: torch.Tensor,
    obj_pos: torch.Tensor,
    tcp_pos: torch.Tensor,
    step_dt: float,
    goal_pos: torch.Tensor | None = None,
  ) -> None:
    """Track the peak cube and TCP speed while the cube is held (plan S7's speed read).

    A step's speed is its displacement over one control step, ``|p_t - p_(t-1)| / step_dt``,
    counted only when the cube was held at both ends of the step: the jaw closing and the
    release are not transport, and the jump across a reset is not motion (`finish` clears
    `prev_held`). Displacement rather than the instantaneous velocity, which spikes on
    contact transients and is not what a rate limit bounds. Call once per step, alongside
    `measure`.

    With ``goal_pos``, also records the episode's last release -- the step the cube goes from
    held to not held -- as the cube's horizontal distance to the target, its height above it,
    and its speed over that step. A cube let go 5 cm short at 1 m/s and one set down on the
    target both count as placed; this is what tells them apart (S5.2 report section 6).
    """
    held = held.bool()
    both = held & self.prev_held
    if goal_pos is not None:
      let_go = self.prev_held & ~held
      dxy = torch.norm((obj_pos - goal_pos)[:, :2], dim=-1)
      v = torch.norm(obj_pos - self.prev_obj, dim=-1) / step_dt
      self.released |= let_go
      self.rel_dxy = torch.where(let_go, dxy, self.rel_dxy)
      self.rel_dz = torch.where(let_go, obj_pos[:, 2] - goal_pos[:, 2], self.rel_dz)
      self.rel_v = torch.where(let_go, v, self.rel_v)
    v_obj = torch.norm(obj_pos - self.prev_obj, dim=-1) / step_dt
    v_tcp = torch.norm(tcp_pos - self.prev_tcp, dim=-1) / step_dt
    self.obj_v_peak = torch.where(
      both, torch.maximum(self.obj_v_peak, v_obj), self.obj_v_peak
    )
    self.tcp_v_peak = torch.where(
      both, torch.maximum(self.tcp_v_peak, v_tcp), self.tcp_v_peak
    )
    self.speed_n += both.long()
    self.prev_obj = obj_pos.clone()
    self.prev_tcp = tcp_pos.clone()
    self.prev_held = held.clone()

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
      moved = int(self.speed_n[i]) > 0
      row["peak_obj_speed_held"] = float(self.obj_v_peak[i]) if moved else None
      row["peak_tcp_speed_held"] = float(self.tcp_v_peak[i]) if moved else None
      rel = bool(self.released[i])
      row["release_dxy"] = float(self.rel_dxy[i]) if rel else None
      row["release_dz"] = float(self.rel_dz[i]) if rel else None
      row["release_speed"] = float(self.rel_v[i]) if rel else None
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
    self.prev_held[env_ids] = False
    self.speed_n[env_ids] = 0
    self.obj_v_peak[env_ids] = 0.0
    self.tcp_v_peak[env_ids] = 0.0
    self.released[env_ids] = False


class PlantTally:
  """Plan S8's plant reads, per episode, kept beside `Tally` and banked with it.

  * parked hold -- |q - q_default| of the parked right arm and the pelvis, which are commanded
    to their default the whole episode: the episode's max, and its mean over steps and joints;
  * tracking -- the driven arm's |q* - q| while the cube is carried (held and off the table),
    q* being its position actuators' target: the episode's mean over carrying steps and the
    seven joints, and its max over both;
  * saturation -- how many steps each pelvis/arm joint's actuator torque sits at its effort
    limit, over the whole rollout;
  * set-down force -- the peak cube-table normal force while the cube is held within
    `NEAR_XY` of the target: how hard the arm presses the cube into the table.

  Tensors in, so the accounting is testable without an env; `from_env` wires one to it.
  """

  def __init__(
    self,
    num_envs: int,
    device: str,
    parked_names: list[str],
    parked_default: torch.Tensor,
    gained_names: list[str],
    effort: torch.Tensor,
  ) -> None:
    self.parked_names = parked_names
    self.parked_default = parked_default
    self.gained_names = gained_names
    self.effort = effort
    zf = lambda: torch.zeros(num_envs, device=device)  # noqa: E731
    self.steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    self.park_max = zf()
    self.park_sum = zf()
    self.carry_n = torch.zeros(num_envs, dtype=torch.long, device=device)
    self.track_sum = zf()
    self.track_max = zf()
    self.setdown_n = torch.zeros(num_envs, dtype=torch.long, device=device)
    self.force_peak = zf()
    self.park_joint_max = torch.zeros(len(parked_names), device=device)
    self.sat_count = torch.zeros(len(gained_names), dtype=torch.long, device=device)
    self.sat_steps = 0
    self.done: list[dict[str, float | int | None]] = []

  @classmethod
  def from_env(cls, env: ManagerBasedRlEnv) -> PlantTally:
    import re

    import mujoco

    robot = env.scene[ROBOT]
    names = list(robot.joint_names)
    pick = lambda rx: [i for i, n in enumerate(names) if re.fullmatch(rx, n)]  # noqa: E731
    parked, driven, gained = (
      pick(PARKED_JOINTS),
      pick(DRIVEN_ARM_JOINTS),
      pick(GAINED_JOINTS),
    )
    mj = env.sim.mj_model
    actuator = {
      (mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or "").split("/")[-1]: a
      for a in range(mj.nu)
    }
    tally = cls(
      env.num_envs,
      env.device,
      [names[i] for i in parked],
      robot.data.default_joint_pos[:, parked].clone(),
      [names[i] for i in gained],
      torch.tensor(
        [float(mj.actuator_forcerange[actuator[names[i]], 1]) for i in gained],
        device=env.device,
      ),
    )
    tally._robot = robot
    tally._sim = env.sim
    tally._support = env.scene[SUPPORT_SENSOR]
    tally._idx = (parked, driven, gained)
    tally._driven_ctrl = torch.tensor(
      [actuator[names[i]] for i in driven], device=env.device
    )
    return tally

  def measure_env(self, carrying: torch.Tensor, setdown: torch.Tensor) -> None:
    """`measure` from the env `from_env` was built on, at the current state."""
    parked, driven, gained = self._idx
    q = self._robot.data.joint_pos
    force = self._support.data.force  # (B, slots, 3), contact frame: normal first
    self.measure(
      parked_q=q[:, parked],
      driven_q=q[:, driven],
      driven_target=self._sim.data.ctrl[:, self._driven_ctrl],
      gained_torque=self._robot.data.qfrc_actuator[:, gained],
      table_force=force[..., 0].abs().amax(dim=-1),
      carrying=carrying,
      setdown=setdown,
    )

  def measure(
    self,
    parked_q: torch.Tensor,
    driven_q: torch.Tensor,
    driven_target: torch.Tensor,
    gained_torque: torch.Tensor,
    table_force: torch.Tensor,
    carrying: torch.Tensor,
    setdown: torch.Tensor,
  ) -> None:
    """One step. Shapes (B, parked), (B, 7), (B, 7), (B, gained), (B,), (B,), (B,)."""
    carrying, setdown = carrying.bool(), setdown.bool()
    park_err = (parked_q - self.parked_default).abs()
    self.park_max = torch.maximum(self.park_max, park_err.amax(dim=-1))
    self.park_sum += park_err.mean(dim=-1)
    self.park_joint_max = torch.maximum(self.park_joint_max, park_err.amax(dim=0))
    self.steps += 1
    track = (driven_target - driven_q).abs()
    self.carry_n += carrying.long()
    self.track_sum += torch.where(
      carrying, track.mean(dim=-1), torch.zeros_like(self.track_sum)
    )
    self.track_max = torch.where(
      carrying, torch.maximum(self.track_max, track.amax(dim=-1)), self.track_max
    )
    self.setdown_n += setdown.long()
    self.force_peak = torch.where(
      setdown, torch.maximum(self.force_peak, table_force), self.force_peak
    )
    self.sat_count += (gained_torque.abs() >= SATURATION_FRACTION * self.effort).sum(
      dim=0
    )
    self.sat_steps += gained_torque.shape[0]

  def finish(self, env_ids: torch.Tensor) -> None:
    """Bank the episodes that just ended, in the order `Tally.finish` banks them."""
    for i in env_ids.tolist():
      steps, carry, setdown = (
        int(self.steps[i]),
        int(self.carry_n[i]),
        int(self.setdown_n[i]),
      )
      self.done.append(
        {
          "parked_hold_max": float(self.park_max[i]) if steps else None,
          "parked_hold_mean": float(self.park_sum[i]) / steps if steps else None,
          "carry_steps": carry,
          "track_carry_mean": float(self.track_sum[i]) / carry if carry else None,
          "track_carry_max": float(self.track_max[i]) if carry else None,
          "setdown_force_peak": float(self.force_peak[i]) if setdown else None,
        }
      )
    for acc in (
      self.park_max,
      self.park_sum,
      self.track_sum,
      self.track_max,
      self.force_peak,
    ):
      acc[env_ids] = 0.0
    for acc in (self.steps, self.carry_n, self.setdown_n):
      acc[env_ids] = 0

  def summarise(self, episodes: int) -> dict:
    rows = self.done[:episodes]
    vals = lambda k: [r[k] for r in rows if r[k] is not None]  # noqa: E731
    mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    median = lambda xs: sorted(xs)[len(xs) // 2] if xs else None  # noqa: E731
    share = (self.sat_count.float() / max(self.sat_steps, 1)).tolist()
    worst = int(torch.argmax(self.park_joint_max))
    return {
      "plant_parked_hold_max_median_rad": median(vals("parked_hold_max")),
      "plant_parked_hold_max_rad": max(vals("parked_hold_max"), default=None),
      "plant_parked_hold_mean_rad": mean(vals("parked_hold_mean")),
      "plant_parked_worst_joint": self.parked_names[worst],
      "plant_parked_max_by_joint_rad": dict(
        zip(self.parked_names, self.park_joint_max.tolist(), strict=True)
      ),
      "plant_track_carry_mean_rad": mean(vals("track_carry_mean")),
      "plant_track_carry_max_median_rad": median(vals("track_carry_max")),
      "plant_saturation_share": dict(zip(self.gained_names, share, strict=True)),
      "plant_saturation_share_max": max(share, default=None),
      "plant_setdown_force_peak_median_n": median(vals("setdown_force_peak")),
      "plant_setdown_force_peak_max_n": max(vals("setdown_force_peak"), default=None),
    }


def build_env(args: argparse.Namespace) -> ManagerBasedRlEnv:
  cfg = load_env_cfg(args.task, play=True)
  # `play=True` removes observation corruption and the curriculum, both of which we want
  # gone -- but it also sets episode_length_s to 1e9. Put the training horizon back, or no
  # episode ever ends and there is nothing to count.
  cfg.episode_length_s = load_env_cfg(args.task, play=False).episode_length_s
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  if getattr(args, "plant", False):
    # The support sensor records contact only (`found`); the set-down read needs its force
    # too. A sensor field is read-only on the physics, and `found` -- what every other read
    # here uses -- is unchanged.
    import dataclasses

    cfg.scene.sensors = tuple(
      dataclasses.replace(s, fields=(*s.fields, "force"))
      if s.name == SUPPORT_SENSOR and "force" not in s.fields
      else s
      for s in cfg.scene.sensors
    )
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


def positions(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
  """The cube's and the TCP's world positions, for `Tally.measure_speed`. (B, 3), (B, 3)"""
  robot = env.scene[ROBOT]
  tcp = robot.data.site_pos_w[:, robot.site_names.index(TCP_SITE)]
  return env.scene[OBJECT].data.root_link_pos_w, tcp


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
  speed = {}
  for key, name in (("peak_obj_speed_held", "obj"), ("peak_tcp_speed_held", "tcp")):
    peaks = [r[key] for r in rows if r.get(key) is not None]
    speed[f"peak_{name}_speed_held_mean_mps"] = mean(peaks)
    speed[f"peak_{name}_speed_held_median_mps"] = median(peaks)
    speed[f"peak_{name}_speed_held_max_mps"] = max(peaks) if peaks else None
  rel = [r for r in rows if r.get("release_dxy") is not None]
  release = {
    "episodes_released": len(rel),
    "release_dxy_median_m": median([r["release_dxy"] for r in rel]),
    "release_within_3cm_xy": sum(r["release_dxy"] < NEAR_XY for r in rel) / len(rel)
    if rel
    else None,
    "release_dz_median_m": median([r["release_dz"] for r in rel]),
    "release_speed_median_mps": median([r["release_speed"] for r in rel]),
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
    **speed,
    **release,
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

  print("\n  SPEED WHILE HELD (per-episode peak, over one control step) -------")
  for name, label in (("obj", "cube"), ("tcp", "TCP")):
    m = result[f"peak_{name}_speed_held_mean_mps"]
    if m is None:
      print(f"    peak {label:4s} speed          n/a")
      continue
    med = result[f"peak_{name}_speed_held_median_mps"]
    top = result[f"peak_{name}_speed_held_max_mps"]
    print(
      f"    peak {label:4s} speed         {m:6.2f} m/s  (median {med:.2f}, max {top:.2f})"
    )

  print("\n  RELEASE (the episode's last let-go) ------------------------------")
  if result.get("episodes_released"):
    print(f"    episodes that let go      {result['episodes_released']:6d}")
    print(
      f"    xy to target at release  {fmt(result['release_dxy_median_m'])}  (median)"
    )
    print(f"    within 3 cm xy           {result['release_within_3cm_xy']:6.1%}")
    print(
      f"    height over target       {fmt(result['release_dz_median_m'])}  (median)"
    )
    print(
      f"    cube speed at release    {fmt(result['release_speed_median_mps'], 'm/s')}  (median)"
    )
  else:
    print("    no release observed")

  print("\n  TIMING -----------------------------------------------------------")
  ttp = result["time_to_place_s"]
  print(f"    mean episode             {result['mean_episode_length_s']:6.2f} s")
  print(f"    time to place            {'n/a' if ttp is None else f'{ttp:6.2f} s'}")

  if "plant_parked_hold_mean_rad" in result:
    fmt3 = lambda v, u="rad": "n/a" if v is None else f"{v:7.4f} {u}"  # noqa: E731
    print("\n  PLANT (plan S8) --------------------------------------------------")
    print(
      f"    parked arm + pelvis hold {fmt3(result['plant_parked_hold_max_median_rad'])}"
      f"  (episode max, median; worst {result['plant_parked_worst_joint']})"
    )
    print(f"      max over episodes      {fmt3(result['plant_parked_hold_max_rad'])}")
    print(f"      mean                   {fmt3(result['plant_parked_hold_mean_rad'])}")
    print(
      f"    driven arm |q*-q| carry  {fmt3(result['plant_track_carry_mean_rad'])}"
      f"  (mean; episode max, median {fmt3(result['plant_track_carry_max_median_rad'])})"
    )
    sat = result["plant_saturation_share"]
    top = sorted(sat.items(), key=lambda kv: -kv[1])[:3]
    print(
      "    at effort limit          "
      + ", ".join(f"{n.removesuffix('_joint')} {v:.1%}" for n, v in top)
    )
    print(
      f"    set-down force, peak     {fmt3(result['plant_setdown_force_peak_median_n'], 'N')}"
      f"  (median; max {fmt3(result['plant_setdown_force_peak_max_n'], 'N')})"
    )

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
  plant = PlantTally.from_env(env) if args.plant else None
  reward_sums: dict[str, list[float]] = {}

  obs, _ = wrapped.reset()
  with torch.inference_mode():
    while len(tally.done) < args.episodes:
      flags = probe(env, grasp_sensor)
      tally.observe(**flags)
      dxy, dz = geometry(env)
      tally.measure(flags["grasped"], dxy, dz)
      if plant is not None:
        plant.measure_env(
          carrying=flags["lifted"], setdown=flags["grasped"] & (dxy < NEAR_XY)
        )
      tally.measure_speed(
        flags["grasped"],
        *positions(env),
        env.step_dt,
        env.command_manager.get_term(COMMAND).command,
      )
      obs, _, dones, extras = wrapped.step(policy(obs))
      tally.terminal(success_flags(env))
      finished = torch.nonzero(dones, as_tuple=False).flatten()
      if finished.numel():
        tally.finish(finished)
        if plant is not None:
          plant.finish(finished)
        for key, value in extras.get("log", {}).items():
          if key.startswith("Episode_Reward/"):
            reward_sums.setdefault(key.removeprefix("Episode_Reward/"), []).append(
              float(value)
            )

  result = summarise(tally, env.step_dt, args.episodes)
  if plant is not None:
    result.update(plant.summarise(args.episodes))
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
