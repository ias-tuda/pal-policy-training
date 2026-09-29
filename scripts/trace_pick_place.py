"""Per-step trace of a pick-and-place policy: what the cube, the jaw and the arm do at the goal.

Plan S7-5, prep 3 (the rocking diagnostic, no GPU training). RCTP s2 @ 5 k (S7-4) carries the
cube to the target and holds it there to the time-out; the video shows it tilted on one edge,
rocking. The likely cause the plan names: the arm's target sits below the table, the capped
elbow presses, P's hinge pushes back, and the pinched cube pivots about the pad axis on its
edge. This records, per env and control step, everything that tests it:

  cube      height of its centre over its resting height (dz), tilt (deg, 0 = on a face),
            horizontal distance to the target, speed, table contact, cube-table normal force
  jaw       measured angle, grasp (both pads), any pad contact
  arm       the TCP's height over the table top, measured and *commanded* -- forward
            kinematics of the action term's joint targets (after the slew and C / T) -- and
            per driven joint the target minus the measured angle and |tau| / its limit
  P         the effort hinge's value (unweighted), when the run has it

    EFFORT=CTP GAINS=G20 SPAWN=W2 BASELINE=B DR=DR-scene SLEW=0.5 \\
    REWARD=R-C-strict+M1+M2+M3+M6+J+M8 uv run --extra cpu python scripts/trace_pick_place.py \\
      --checkpoint ../results/S7-4/ckpt/RCTP_s2/model_4999.pt --num-envs 8 --device cpu \\
      --out ../results/S7-5/trace/RCTP_s2

Writes ``trace.csv`` (one row per env and step), ``summary.json`` (per episode: the share of
steps held at the goal with the commanded TCP below the measured one / below the table, the
tilt and force while held there, ...) and ``trace_env<i>.png``. One episode per env, the
training horizon, the play cfg (no observation noise), as `eval_pick_place.py`.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import pathlib

import numpy as np
import torch


def _eval_module():
  spec = importlib.util.spec_from_file_location(
    "eval_pick_place", pathlib.Path(__file__).parent / "eval_pick_place.py"
  )
  assert spec and spec.loader
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--checkpoint", type=pathlib.Path, required=True)
  p.add_argument("--out", type=pathlib.Path, required=True, help="output directory")
  p.add_argument("--task", default="Mjlab-PickPlace-Pal-Kangaroo-Grippers")
  p.add_argument("--num-envs", type=int, default=8)
  p.add_argument("--device", default="cpu")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--no-plot", action="store_true")
  return p.parse_args()


#: "Held at the goal" for the summary: grasped, and within this horizontal distance (m).
AT_GOAL_XY = 0.03


class CommandedTcp:
  """Forward kinematics of the arm's joint targets on a CPU copy of the model: where the TCP
  would be if every driven joint were at its target, the rest of the robot where it is."""

  def __init__(self, env, term, tcp_site: str) -> None:
    import mujoco

    self._mujoco = mujoco
    self.model = env.sim.mj_model
    self.data = mujoco.MjData(self.model)
    robot = env.scene["robot"]
    self.qadr = robot.indexing.joint_q_adr[term._target_ids].cpu().numpy()
    names = [
      mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i)
      for i in range(self.model.nsite)
    ]
    self.site = next(i for i, n in enumerate(names) if n and n.endswith(tcp_site))

  def __call__(self, qpos: np.ndarray, targets: np.ndarray) -> np.ndarray:
    out = np.zeros((qpos.shape[0], 3))
    for i in range(qpos.shape[0]):
      self.data.qpos[:] = qpos[i]
      self.data.qpos[self.qadr] = targets[i]
      self._mujoco.mj_kinematics(self.model, self.data)
      out[i] = self.data.site_xpos[self.site]
    return out


def main() -> None:
  args = parse_args()
  import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
  from mjlab.managers.scene_entity_config import SceneEntityCfg
  from pal_mjlab.tasks.manipulation.mdp.events import surface_top_z
  from pal_mjlab.tasks.manipulation.mdp.observations import (
    _grasped_bool,
    _on_surface_bool,
    _pad_contact_any,
  )

  ev = _eval_module()
  torch.manual_seed(args.seed)
  env = ev.build_env(
    argparse.Namespace(
      task=args.task,
      num_envs=args.num_envs,
      seed=args.seed,
      device=args.device,
      plant=True,
    )
  )
  wrapped, policy = ev.load_policy(
    argparse.Namespace(task=args.task, checkpoint=args.checkpoint, device=args.device),
    env,
  )
  gripper_cfg = env.cfg.rewards.get("grasp") or env.cfg.rewards["lift"]
  grasp = (
    gripper_cfg.params.get("sensor_name") or gripper_cfg.params["grasp_sensor_name"]
  )
  robot, obj = env.scene[ev.ROBOT], env.scene[ev.OBJECT]
  term = env.action_manager.get_term("joint_pos")
  arm_cols = [
    k
    for k, j in enumerate(term._target_ids.tolist())
    if robot.joint_names[j].startswith("arm_left_")
  ]
  arm_joints = [robot.joint_names[term._target_ids[k]] for k in arm_cols]
  arm_ids = [robot.joint_names.index(n) for n in arm_joints]
  jaw_id = robot.joint_names.index(ev.JAW_JOINT)
  plant = ev.PlantTally.from_env(env)
  # The binding effort limit per driven joint: T's lowered cap where the run has it.
  caps = plant.driven_effort[[plant.driven_names.index(n) for n in arm_joints]]
  hinge_cfg = None
  if "arm_effort" in env.reward_manager.active_terms:
    hinge_cfg = env.reward_manager.get_term_cfg("arm_effort")
  fk = CommandedTcp(env, term, ev.TCP_SITE)
  table = SceneEntityCfg(ev.TABLE)
  table.resolve(env.scene)
  support = env.scene[ev.SUPPORT_SENSOR]
  steps = round(env.cfg.episode_length_s / env.step_dt)

  rows: list[dict] = []
  obs, _ = wrapped.reset()
  with torch.inference_mode():
    for t in range(steps):
      goal = env.command_manager.get_term(ev.COMMAND).command
      pos = obj.data.root_link_pos_w
      top = surface_top_z(env, table)
      tcp = robot.data.site_pos_w[:, robot.site_names.index(ev.TCP_SITE)]
      qpos = env.sim.data.qpos.cpu().numpy()
      targets = term._processed_actions.cpu().numpy()
      tcp_cmd = fk(qpos, targets)
      q = robot.data.joint_pos
      tau = robot.data.qfrc_actuator[:, arm_ids]
      err = term._processed_actions[:, arm_cols] - q[:, arm_ids]
      force = support.data.force[..., 0].abs().amax(dim=-1)
      hinge = (
        hinge_cfg.func(env, **hinge_cfg.params)
        if hinge_cfg is not None
        else torch.zeros(env.num_envs, device=env.device)
      )
      cols = {
        "dz": (pos - goal)[:, 2],
        "tilt_deg": torch.rad2deg(ev.tilt(env)),
        "dxy": torch.norm((pos - goal)[:, :2], dim=-1),
        "speed": torch.norm(obj.data.root_link_lin_vel_w, dim=-1),
        "on_table": _on_surface_bool(env, ev.SUPPORT_SENSOR).float(),
        "table_force": force,
        "jaw": q[:, jaw_id],
        "grasped": _grasped_bool(env, grasp).float(),
        "pad_any": _pad_contact_any(env, grasp).float(),
        "tcp_over_table": tcp[:, 2] - top,
        "tcp_cmd_over_table": torch.as_tensor(tcp_cmd[:, 2], device=env.device) - top,
        "hinge": hinge,
      }
      for k, n in enumerate(arm_joints):
        short = n.removesuffix("_joint").removeprefix("arm_left_")
        cols[f"err_{short}"] = err[:, k]
        cols[f"tau_{short}"] = tau[:, k].abs() / caps[k]
      host = {k: v.float().cpu().numpy() for k, v in cols.items()}
      for i in range(env.num_envs):
        rows.append({"env": i, "t": t, **{k: float(v[i]) for k, v in host.items()}})
      obs, _, _, _ = wrapped.step(policy(obs))
  env.close()

  args.out.mkdir(parents=True, exist_ok=True)
  with open(args.out / "trace.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
  summary = summarise(rows, args.num_envs, env.step_dt, arm_joints)
  summary["checkpoint"] = str(args.checkpoint)
  (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
  print(json.dumps(summary["aggregate"], indent=2))
  if not args.no_plot:
    plot(rows, args.num_envs, env.step_dt, args.out)
  print(f"wrote {args.out}")


def summarise(rows: list[dict], n: int, dt: float, arm_joints: list[str]) -> dict:
  per_env = []
  shorts = [j.removesuffix("_joint").removeprefix("arm_left_") for j in arm_joints]
  for i in range(n):
    r = [x for x in rows if x["env"] == i]
    held = [x for x in r if x["grasped"] and x["dxy"] < AT_GOAL_XY]
    med = lambda xs: float(np.median(xs)) if xs else None  # noqa: E731
    share = lambda xs, f: sum(f(x) for x in xs) / len(xs) if xs else None  # noqa: E731
    tilt_sign = [x["tilt_deg"] for x in held]
    per_env.append(
      {
        "env": i,
        "held_at_goal_s": len(held) * dt,
        "held_at_goal_share": len(held) / len(r),
        "held": {
          "tcp_cmd_below_tcp_m_median": med(
            [x["tcp_over_table"] - x["tcp_cmd_over_table"] for x in held]
          ),
          "tcp_cmd_below_table_share": share(
            held, lambda x: x["tcp_cmd_over_table"] < 0
          ),
          "tcp_over_table_m_median": med([x["tcp_over_table"] for x in held]),
          "tilt_deg_median": med(tilt_sign),
          "tilt_over_5deg_share": share(held, lambda x: x["tilt_deg"] > 5),
          "dz_m_median": med([x["dz"] for x in held]),
          "on_table_share": share(held, lambda x: x["on_table"] > 0),
          "table_force_n_median": med([x["table_force"] for x in held]),
          "table_force_n_max": max((x["table_force"] for x in held), default=None),
          "speed_mps_median": med([x["speed"] for x in held]),
          "jaw_rad_median": med([x["jaw"] for x in held]),
          "hinge_median": med([x["hinge"] for x in held]),
          "hinge_nonzero_share": share(held, lambda x: x["hinge"] > 0),
          "tau_over_cap_median": {
            s: med([x[f"tau_{s}"] for x in held]) for s in shorts
          },
          "tau_at_cap_share": {
            s: share(held, lambda x, s=s: x[f"tau_{s}"] > 0.999) for s in shorts
          },
          "target_minus_q_rad_median": {
            s: med([x[f"err_{s}"] for x in held]) for s in shorts
          },
        },
      }
    )
  agg_keys = ("tcp_cmd_below_table_share", "tilt_over_5deg_share", "on_table_share")
  held_envs = [e for e in per_env if e["held_at_goal_s"] > 0]
  return {
    "at_goal_xy_m": AT_GOAL_XY,
    "aggregate": {
      "envs": n,
      "envs_holding_at_goal": len(held_envs),
      "held_at_goal_s_median": float(np.median([e["held_at_goal_s"] for e in per_env])),
      **{
        f"{k}_median": float(np.median([e["held"][k] for e in held_envs]))
        if held_envs
        else None
        for k in agg_keys
      },
      "table_force_n_median": float(
        np.median([e["held"]["table_force_n_median"] for e in held_envs])
      )
      if held_envs
      else None,
    },
    "per_env": per_env,
  }


def plot(rows: list[dict], n: int, dt: float, out: pathlib.Path) -> None:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  for i in range(n):
    r = [x for x in rows if x["env"] == i]
    t = np.array([x["t"] for x in r]) * dt
    col = lambda k, r=r: np.array([x[k] for x in r])  # noqa: E731
    fig, ax = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    ax[0].plot(t, col("dz") * 100, label="cube dz (cm)")
    ax[0].plot(t, col("dxy") * 100, label="cube dxy (cm)")
    ax[0].plot(t, col("tilt_deg") / 10, label="tilt (10 deg)")
    ax[0].legend(loc="upper right")
    ax[1].plot(t, col("tcp_over_table") * 100, label="TCP over table (cm)")
    ax[1].plot(t, col("tcp_cmd_over_table") * 100, label="commanded TCP (cm)")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].legend(loc="upper right")
    ax[2].plot(t, col("jaw"), label="jaw (rad)")
    ax[2].plot(t, col("grasped") * 0.4, label="grasped", alpha=0.6)
    ax[2].plot(t, col("on_table") * 0.35, label="cube on table", alpha=0.6)
    ax[2].axhline(0.15, color="k", lw=0.5, ls="--")
    ax[2].legend(loc="upper right")
    for k in [c for c in r[0] if c.startswith("tau_")]:
      ax[3].plot(t, col(k), label=k, lw=0.8)
    ax[3].axhline(0.8, color="k", lw=0.5, ls="--")
    ax[3].set_ylabel("|tau| / cap")
    ax[3].legend(loc="upper right", ncol=4, fontsize=7)
    ax[4].plot(t, col("table_force"), label="cube-table force (N)")
    ax[4].plot(t, col("hinge") * 100, label="P hinge x 100")
    ax[4].legend(loc="upper right")
    ax[4].set_xlabel("t (s)")
    fig.tight_layout()
    fig.savefig(out / f"trace_env{i}.png", dpi=80)
    plt.close(fig)


if __name__ == "__main__":
  main()
