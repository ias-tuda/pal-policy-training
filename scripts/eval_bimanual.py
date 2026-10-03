"""Plan S11's JOINT read (`ARMS=both`): one 16-D policy drives both arms; score it per episode.

The env is the run's own (its selectors in the environment, as for `eval_pick_place.py`), in play
mode with the training horizon plus the hand-offs' time (``--episode-s``). The left arm hands itself
off on its own trigger, as in training (`mdp.handoff`); the right is handed off here, on the same
trigger -- its jaw opened after the lift (J) and the TCP `HANDOFF_CLEAR` over B's top for
`HANDOFF_CLEAR_STEPS` steps -- or at ``--right-timeout``. Both go up, out and home.

Per episode (plan S11 *Reads*):

* the headline, **tower standing after both returns**: at the episode's last step B is placed on A
  (the right's J-placed predicate on its stack goal, no right pad on B), A is placed at T (the left's,
  no left pad on A), and both arms are handed off and home;
* per phase: A placed at T (first step), A's final offset and |yaw mod 90| at T, B grasped, B's
  let-go (S9's five ``OK`` quantities: dxy, height, tilt, yaw against A, speed), B placed;
* the base: how far the right moved A after it was placed (max horizontal displacement);
* contacts: HH -- the arm-arm sensor over 1 N (steps, peak force) and the least capsule clearance;
  arm 1-6 on the table per arm (steps);
* time: to A placed, to B placed, to both home; the share of steps both arms move (overlap).

    ARMS=both ... python scripts/eval_bimanual.py --checkpoint model_4999.pt --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys
from dataclasses import asdict

import torch

TASK = "Mjlab-PickPlace-Pal-Kangaroo-Grippers"
#: Joint speed above which an arm counts as moving (rad/s, any joint), for the overlap read.
MOVING = 0.05
#: An arm is home within this of its home pose (rad, every joint), as `eval_pick_place.HOME_TOL`.
HOME_TOL = 0.05


def parse_args(argv=None) -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--checkpoint", type=pathlib.Path, required=True)
  p.add_argument("--task", default=TASK)
  p.add_argument("--episodes", type=int, default=128)
  p.add_argument("--num-envs", type=int, default=64)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument(
    "--episode-s",
    type=float,
    default=25.0,
    help="episode length: training's 15 s + 10 s for the two hand-offs (plan S11's eval)",
  )
  p.add_argument(
    "--right-timeout",
    type=float,
    default=19.0,
    help="episode time (s) at which the right arm is handed off whatever it is doing",
  )
  p.add_argument("--obs-noise", choices=("off", "on"), default="on")
  p.add_argument("--json", type=pathlib.Path, default=None)
  return p.parse_args(argv)


def build_env(args):
  import pal_mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg(args.task, play=True)
  cfg.episode_length_s = args.episode_s
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  if args.obs_noise == "on":
    cfg.observations["actor"].enable_corruption = True
  if "cube_b" not in cfg.scene.entities:
    raise SystemExit(
      "eval_bimanual reads plan S11's JOINT env: set ARMS=both and its selectors"
    )
  return ManagerBasedRlEnv(cfg=cfg, device=args.device)


def load_policy(args, env):
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

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


class Reads:
  """Per-env episode state; ``finish`` turns the envs that ended into rows."""

  def __init__(self, env) -> None:
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from pal_mjlab.tasks.manipulation.pick_place import bimanual as bi

    self.env, self.bi = env, bi
    n, dev = env.num_envs, env.device
    self.robot = env.scene["robot"]
    names = self.robot.joint_names
    self.arm = {
      s: [names.index(f"arm_{s}_{i}_joint") for i in range(1, 8)]
      for s in ("left", "right")
    }
    self.term = env.action_manager.get_term("joint_pos")
    p = env.cfg.rewards["placed"].params
    self.left_gate = {
      "command_name": p["command_name"],
      "object_name": p["object_name"],
      "grasp_sensor_name": p["grasp_sensor_name"],
      "support_sensor_name": p["support_sensor_name"],
      "jaw_cfg": p["jaw_cfg"],
      "jaw_open": p["jaw_open"],
      "threshold": p["threshold"],
      "velocity_threshold": 0.05,
    }
    jaw_r = SceneEntityCfg("robot", joint_names=("gripper_right_finger_joint",))
    jaw_r.resolve(env.scene)
    self.right_gate = {
      "command_name": bi.COMMAND_B,
      "object_name": bi.CUBE_B,
      "grasp_sensor_name": "right_pad_contact",
      "support_sensor_name": bi.SUPPORT_SENSOR_B,
      "jaw_cfg": jaw_r,
      "jaw_open": p["jaw_open"],
      "threshold": p["threshold"],
      "velocity_threshold": 0.05,
    }
    rp = env.cfg.rewards["right_placed"].params
    self.right_latch = (
      rp["command_name"],
      rp["object_name"],
      rp["grasp_sensor_name"],
      rp["support_sensor_name"],
      rp["surface_cfg"],
      rp["jaw_cfg"],
      rp["jaw_open"],
    )
    zb = lambda: torch.zeros(n, dtype=torch.bool, device=dev)  # noqa: E731
    zl = lambda v=-1: torch.full((n,), v, dtype=torch.long, device=dev)  # noqa: E731
    zf = lambda v=0.0: torch.full((n,), v, device=dev)  # noqa: E731
    self.a_step, self.b_step, self.home_step = zl(), zl(), zl()
    self.b_grasped, self.b_released = zb(), zb()
    self.a_ref = torch.zeros(n, 2, device=dev)
    self.a_moved = zf()
    self.hh_steps, self.hh_peak, self.hh_clear = zl(0), zf(), zf(float("inf"))
    self.at_left, self.at_right = zl(0), zl(0)
    self.both_moving, self.steps = zl(0), zl(0)
    self.rel = {k: zf(float("nan")) for k in ("dxy", "height", "tilt", "yaw", "speed")}
    self.rel_ok = zb()
    self.prev_right_grasp = zb()
    self.clear_run = zl(0)
    self.right_handed = zb()
    self.done: list[dict] = []

  def _home(self, side: str) -> torch.Tensor:
    q = self.robot.data.joint_pos[:, self.arm[side]]
    d = self.robot.data.default_joint_pos[:, self.arm[side]]
    return (q - d).abs().amax(-1) < HOME_TOL

  def step(self, right_timeout_steps: int) -> torch.Tensor:
    """Read the state the policy acts on; returns the envs whose right arm to hand off now."""
    from pal_mjlab.tasks.manipulation import mdp
    from pal_mjlab.tasks.manipulation.mdp.handoff import (
      HANDOFF_CLEAR,
      HANDOFF_CLEAR_STEPS,
    )
    from pal_mjlab.tasks.manipulation.mdp.observations import (
      _grasped_bool,
      _release_latch,
      yaw_error_mod_90,
    )
    from pal_mjlab.tasks.manipulation.mdp.shapes import object_half_height
    from pal_mjlab.tasks.manipulation.mdp.stack import stack_state

    env, bi = self.env, self.bi
    t = env.episode_length_buf
    a = env.scene["object"].data.root_link_pos_w
    b = env.scene[bi.CUBE_B]
    a_placed = mdp.base_placed_bool(env, **self.left_gate)
    fresh_a = a_placed & (self.a_step < 0)
    self.a_step = torch.where(fresh_a, t, self.a_step)
    self.a_ref = torch.where(fresh_a.unsqueeze(-1), a[:, :2], self.a_ref)
    moved = torch.norm(a[:, :2] - self.a_ref, dim=-1)
    self.a_moved = torch.where(
      self.a_step >= 0, torch.maximum(self.a_moved, moved), self.a_moved
    )
    b_placed = mdp.base_placed_bool(env, **self.right_gate) & a_placed
    self.b_step = torch.where(b_placed & (self.b_step < 0), t, self.b_step)
    grasp_r = _grasped_bool(env, "right_pad_contact")
    self.b_grasped |= grasp_r
    latch = _release_latch(env, *self.right_latch)
    let_go = self.prev_right_grasp & ~grasp_r & self.b_grasped
    self.prev_right_grasp = grasp_r
    st = stack_state(env, bi.CUBE_B, "object", bi.COMMAND_B)
    for k in self.rel:
      self.rel[k] = torch.where(let_go, getattr(st, k), self.rel[k])
    self.rel_ok = torch.where(let_go, st.ok, self.rel_ok)
    self.b_released |= latch.released
    # HH and arm-table, over the whole control step (the sensors' force history).
    peak = _peak_force(env, bi.ARM_ARM_SENSOR)
    self.hh_steps += (peak > bi.HH_CONTACT_FORCE).long()
    self.hh_peak = torch.maximum(self.hh_peak, peak)
    self.hh_clear = torch.minimum(self.hh_clear, mdp.arm_arm_clearance(env))
    for name, acc in (
      ("arm_table_contact", "at_left"),
      ("right_arm_table_contact", "at_right"),
    ):
      if name in env.scene.sensors:
        hit = _peak_force(env, name) > 1.0
        setattr(self, acc, getattr(self, acc) + hit.long())
    qd = self.robot.data.joint_vel
    moving = [qd[:, self.arm[s]].abs().amax(-1) > MOVING for s in ("left", "right")]
    self.both_moving += (moving[0] & moving[1]).long()
    self.steps += 1
    # Both home (after both hand-offs).
    phase = self.term.phase
    home = (
      (phase[:, 0] > 0) & (phase[:, 1] > 0) & self._home("left") & self._home("right")
    )
    self.home_step = torch.where(home & (self.home_step < 0), t, self.home_step)
    # The right's hand-off trigger, as the left's (`mdp.handoff`).
    site = self.robot.site_names.index("tcp_right")
    top = b.data.root_link_pos_w[:, 2] + object_half_height(env, bi.CUBE_B)
    above = self.robot.data.site_pos_w[:, site, 2] - top
    clear = latch.released & (above >= HANDOFF_CLEAR)
    self.clear_run = torch.where(
      clear, self.clear_run + 1, torch.zeros_like(self.clear_run)
    )
    start = (self.clear_run >= HANDOFF_CLEAR_STEPS) | (t >= right_timeout_steps)
    start &= ~self.right_handed
    self.right_handed |= start
    # The state this step began in: an env that ends on this step is reset inside it, so its
    # episode is scored on these.
    origins = env.scene.env_origins
    self._last = {
      "a_placed": a_placed,
      "b_placed": b_placed,
      "home": home,
      "a_xy": (a - origins)[:, :2].clone(),
      "goal_xy": (env.command_manager.get_term("goal_pose").command - origins)[
        :, :2
      ].clone(),
      "a_yaw": yaw_error_mod_90(env.scene["object"].data.root_link_quat_w),
    }
    return start

  def finish(self, env_ids: torch.Tensor, dt: float) -> None:
    from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import T11_XY

    last = self._last
    a_placed, b_placed, home = last["a_placed"], last["b_placed"], last["home"]
    a, goal, yaw = last["a_xy"], last["goal_xy"], last["a_yaw"]
    sec = lambda s: None if int(s) < 0 else int(s) * dt  # noqa: E731
    for i in env_ids.tolist():
      row = {
        "success": bool(a_placed[i] & b_placed[i] & home[i]),
        "tower_end": bool(a_placed[i] & b_placed[i]),
        "both_home": bool(home[i]),
        "a_placed_ever": int(self.a_step[i]) >= 0,
        "b_placed_ever": int(self.b_step[i]) >= 0,
        "b_grasped": bool(self.b_grasped[i]),
        "b_released": bool(self.b_released[i]),
        "t_a_placed_s": sec(self.a_step[i]),
        "t_b_placed_s": sec(self.b_step[i]),
        "t_both_home_s": sec(self.home_step[i]),
        "a_final_dxy_m": float(torch.norm(a[i] - goal[i])),
        "a_final_from_t_nominal_m": float(
          math.hypot(float(a[i, 0]) - T11_XY[0], float(a[i, 1]) - T11_XY[1])
        ),
        "a_final_yaw_deg": math.degrees(float(yaw[i])),
        "a_moved_by_right_m": float(self.a_moved[i]),
        "b_let_go": {k: float(v[i]) for k, v in self.rel.items()},
        "b_let_go_ok": bool(self.rel_ok[i]),
        "hh_contact_steps": int(self.hh_steps[i]),
        "hh_peak_force_n": float(self.hh_peak[i]),
        "hh_min_clearance_m": float(self.hh_clear[i]),
        "arm_table_left_steps": int(self.at_left[i]),
        "arm_table_right_steps": int(self.at_right[i]),
        "overlap_share": float(self.both_moving[i]) / max(int(self.steps[i]), 1),
      }
      self.done.append(row)
    for name in ("a_step", "b_step", "home_step"):
      getattr(self, name)[env_ids] = -1
    for name in (
      "b_grasped",
      "b_released",
      "rel_ok",
      "prev_right_grasp",
      "right_handed",
    ):
      getattr(self, name)[env_ids] = False
    for name in (
      "hh_steps",
      "at_left",
      "at_right",
      "both_moving",
      "steps",
      "clear_run",
    ):
      getattr(self, name)[env_ids] = 0
    self.a_moved[env_ids] = 0.0
    self.hh_peak[env_ids] = 0.0
    self.hh_clear[env_ids] = float("inf")
    for v in self.rel.values():
      v[env_ids] = float("nan")


def _peak_force(env, name: str) -> torch.Tensor:
  """The strongest contact force a sensor saw over the last control step (its history), N. (B,)"""
  data = env.scene[name].data
  force = data.force_history if data.force_history is not None else data.force
  return torch.norm(force.reshape(env.num_envs, -1, 3), dim=-1).amax(-1)


def summarise(rows: list[dict]) -> dict:
  n = max(len(rows), 1)

  def share(k):
    return sum(bool(r[k]) for r in rows) / n

  def med(vals):
    vals = [
      v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))
    ]
    return statistics.median(vals) if vals else None

  placed = [r for r in rows if r["a_placed_ever"]]
  let = [r for r in rows if r["b_released"]]
  return {
    "episodes": len(rows),
    "success_tower_after_both_returns": share("success"),
    "tower_at_end": share("tower_end"),
    "both_home": share("both_home"),
    "a_placed": share("a_placed_ever"),
    "b_grasped": share("b_grasped"),
    "b_placed": share("b_placed_ever"),
    "a_within_1cm_of_t": sum(r["a_final_dxy_m"] < 0.01 for r in placed) / n,
    "a_yaw_within_10deg": sum(r["a_final_yaw_deg"] < 10 for r in placed) / n,
    "a_final_dxy_median_m": med([r["a_final_dxy_m"] for r in placed]),
    "a_final_yaw_median_deg": med([r["a_final_yaw_deg"] for r in placed]),
    "a_moved_by_right_median_m": med([r["a_moved_by_right_m"] for r in placed]),
    "a_moved_by_right_over_5mm": sum(r["a_moved_by_right_m"] > 0.005 for r in placed)
    / n,
    **{
      f"b_let_go_{k}_median": med([r["b_let_go"][k] for r in let])
      for k in ("dxy", "height", "tilt", "yaw", "speed")
    },
    "b_let_go_ok": sum(r["b_let_go_ok"] for r in let) / max(len(let), 1),
    "hh_contact_episodes": sum(r["hh_contact_steps"] > 0 for r in rows) / n,
    "hh_over_30n_episodes": sum(r["hh_peak_force_n"] > 30 for r in rows) / n,
    "hh_min_clearance_median_m": med([r["hh_min_clearance_m"] for r in rows]),
    "arm_table_left_episodes": sum(r["arm_table_left_steps"] > 0 for r in rows) / n,
    "arm_table_right_episodes": sum(r["arm_table_right_steps"] > 0 for r in rows) / n,
    "t_a_placed_median_s": med([r["t_a_placed_s"] for r in rows]),
    "t_b_placed_median_s": med([r["t_b_placed_s"] for r in rows]),
    "t_both_home_median_s": med([r["t_both_home_s"] for r in rows]),
    "overlap_share_median": med([r["overlap_share"] for r in rows]),
  }


def main(argv=None) -> None:
  args = parse_args(argv)
  env = build_env(args)
  wrapped, policy = load_policy(args, env)
  reads = Reads(env)
  term = env.action_manager.get_term("joint_pos")
  right_timeout = round(args.right_timeout / env.step_dt)
  obs, _ = wrapped.reset()
  with torch.inference_mode():
    while len(reads.done) < args.episodes:
      start = reads.step(right_timeout)
      term.start_handoff("right", start)
      actions = policy(obs)
      obs, _, dones, _ = wrapped.step(actions)
      finished = torch.nonzero(dones, as_tuple=False).flatten()
      if finished.numel():
        reads.finish(finished, env.step_dt)
  rows = reads.done[: args.episodes]
  result = {"checkpoint": str(args.checkpoint), "seed": args.seed, **summarise(rows)}
  for k, v in result.items():
    print(f"{k:36s} {v}")
  if args.json:
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({**result, "episodes_detail": rows}, indent=1))
    print(f"wrote {args.json}")
  env.close()


if __name__ == "__main__":
  sys.exit(main())
