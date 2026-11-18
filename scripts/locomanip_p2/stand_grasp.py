"""P2b step 2: the vyq9vrpy grasp policy on a robot balanced by the P1.2 walking policy.

Runs on the code of e61bd8e (the grasp task), with PYTHONPATH=<worktree of e61bd8e>/src.
The env is the grasp task's play cfg with three changes, so both policies see their own
training plant and observation:
- leg actuators back to the walking plant (the grasp robot stiffens them x30 to stand on PD),
  pelvis at the G20 gains P1.2 was trained on; arms, jaw, gravcomp stay as the grasp plant;
- an action term "legs" (P1.2's joint_pos action: 14 lower-body targets) after the grasp terms;
- an observation group "loco": P1.2's 114-dim actor observation, with a twist command held at 0.
Both actors are run from their checkpoints (MLP + empirical normaliser), not through a runner.

Usage: stand_grasp.py <grasp ckpt> <p1.2 ckpt> <num_envs> [--plant p12|grasp] [--video out.mp4]
"""

import argparse
import dataclasses

import mujoco
import pal_mjlab.tasks  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.observations import last_action
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from pal_mjlab.robots import (
  KANGAROO_LOWER_BODY_ACTION_SCALE,
  KANGAROO_LOWER_BODY_ACTUATOR_NAMES,
)
from pal_mjlab.tasks.manipulation.robots.kangaroo import _STANDING_LEG_STIFFNESS_SCALE
from pal_mjlab.tasks.velocity.kangaroo.env_cfgs import pal_kangaroo_flat_env_cfg

GRASP = "Mjlab-Grasp-LeftArm-Standing-Gravcomp-ArmV36-Slow50-NoTable-Smooth-Wide-Patient-LeanObs-Pal-Kangaroo-Grippers"
# P1.2's pelvis actuator (G20_S_PLUS on skema-locomanip), read from its compiled model.
P12_PELVIS_KP_KD = (998.79, 63.585)
# P1.2's arm actuators (G20), for the right arm, which the grasp policy does not drive.
# Without gravcomp the grasp plant's v36 gains let it sag 7 cm onto the table in 0.2 s.
P12_ARM_KP_KD = {1: (998.79, 63.585), 2: (998.79, 63.585)} | {
  k: (523.54, 33.329) for k in range(3, 8)
}
# P1.2's default jaw angle; the grasp robot's is 0.047 (only joints whose defaults differ).
P12_DEFAULT_OVERRIDE = {
  "gripper_left_finger_joint": 0.0,
  "gripper_right_finger_joint": 0.0,
}


def grasp_last_action(env) -> torch.Tensor:
  am = env.action_manager
  return torch.cat([am.get_term(k).raw_action for k in ("joint_pos", "gripper")], -1)


def build_cfg(num_envs: int, plant: str):
  cfg = load_env_cfg(GRASP, play=True)
  cfg.scene.num_envs = num_envs
  cfg.seed = 0
  robot = cfg.scene.entities["robot"]
  if plant == "p12":
    acts = []
    for a in robot.articulation.actuators:
      names = a.target_names_expr
      if names[0].startswith("leg_"):
        a = dataclasses.replace(
          a,
          stiffness=a.stiffness / _STANDING_LEG_STIFFNESS_SCALE,
          damping=a.damping / _STANDING_LEG_STIFFNESS_SCALE**0.5,
        )
      elif names[0].startswith("pelvis_"):
        a = dataclasses.replace(
          a, stiffness=P12_PELVIS_KP_KD[0], damping=P12_PELVIS_KP_KD[1]
        )
      elif names[0].startswith("arm_.*_"):
        k = int(names[0].split("_")[2])
        kp, kd = P12_ARM_KP_KD[k]
        acts.append(dataclasses.replace(a, target_names_expr=(f"arm_left_{k}_joint",)))
        a = dataclasses.replace(
          a, target_names_expr=(f"arm_right_{k}_joint",), stiffness=kp, damping=kd
        )
      acts.append(a)
    base_spec_fn = robot.spec_fn

    def spec_fn():
      # Body gravcomp reaches every DOF of the bodies' chain, and MuJoCo applies the share
      # of a joint without actgravcomp as a passive force: on the free joint that is an
      # external lift of the floating base (the arm's weight), which a real robot never
      # gets. Route the free joint through the actuator path and clamp it to ~0 there.
      spec = base_spec_fn()
      for j in spec.joints:
        if j.type == mujoco.mjtJoint.mjJNT_FREE:
          j.actgravcomp = True
          j.actfrclimited = mujoco.mjtLimited.mjLIMITED_TRUE
          j.actfrcrange = [-1e-9, 1e-9]
      return spec

    robot = dataclasses.replace(robot, spec_fn=spec_fn)
    # The settled stance settles once with the legs held on PD; on the walking plant's
    # soft legs that is a collapse. The walking policy balances from the default pose.
    cfg.events.pop("settled_stance")
    cfg.scene.entities["robot"] = dataclasses.replace(
      robot, articulation=dataclasses.replace(robot.articulation, actuators=tuple(acts))
    )

  vel = pal_kangaroo_flat_env_cfg(play=True)
  legs = vel.actions["joint_pos"]
  assert isinstance(legs, JointPositionActionCfg)
  legs.scale = KANGAROO_LOWER_BODY_ACTION_SCALE
  legs.actuator_names = KANGAROO_LOWER_BODY_ACTUATOR_NAMES
  cfg.actions["legs"] = legs

  twist = vel.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.resampling_time_range = (1e9, 1e9)
  twist.rel_standing_envs = 1.0
  twist.rel_heading_envs = 0.0
  twist.heading_command = False
  twist.ranges.heading = None
  twist.debug_vis = False
  cfg.commands["twist"] = twist

  terms = {k: t for k, t in vel.observations["actor"].terms.items() if t is not None}
  terms["actions"] = ObservationTermCfg(
    func=terms["actions"].func, params={"action_name": "legs"}
  )
  # The grasp actor's last-action term must keep seeing only the grasp terms (8 dims).
  for name, t in cfg.observations["actor"].terms.items():
    if t is not None and t.func is last_action and not t.params.get("action_name"):
      t.func = grasp_last_action
      print("actor obs", name, "-> grasp terms only")
  cfg.observations["loco"] = ObservationGroupCfg(
    terms=terms, concatenate_terms=True, enable_corruption=False
  )
  return cfg


class Actor(torch.nn.Module):
  def __init__(self, path: str, device: str):
    super().__init__()
    sd = torch.load(path, map_location=device, weights_only=False)["actor_state_dict"]
    w = [sd[f"mlp.{i}.weight"] for i in (0, 2, 4, 6)]
    self.layers = [(sd[f"mlp.{i}.weight"], sd[f"mlp.{i}.bias"]) for i in (0, 2, 4, 6)]
    self.mean, self.std = sd["obs_normalizer._mean"], sd["obs_normalizer._std"]
    self.n_in, self.n_out = w[0].shape[1], w[-1].shape[0]

  @torch.no_grad()
  def forward(self, x):
    x = (x - self.mean) / (self.std + 1e-2)
    for i, (w, b) in enumerate(self.layers):
      x = x @ w.T + b
      if i < 3:
        x = torch.nn.functional.elu(x)
    return x


def main():
  p = argparse.ArgumentParser()
  p.add_argument("grasp_ckpt")
  p.add_argument("p12_ckpt")
  p.add_argument("num_envs", type=int)
  p.add_argument("--plant", choices=("p12", "grasp"), default="p12")
  p.add_argument("--legs", choices=("policy", "hold"), default="policy")
  p.add_argument("--no-dr", action="store_true")
  p.add_argument("--no-gravcomp", action="store_true")
  p.add_argument("--gc-left-only", action="store_true")
  p.add_argument("--upper", choices=("grasp", "hold"), default="grasp")
  p.add_argument("--keep-terms", action="store_true")
  p.add_argument("--table-dz", type=float, default=0.0)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--video", default=None)
  p.add_argument("--seconds", type=float, default=None)
  args = p.parse_args()
  dev = "cuda:0"
  cfg = build_cfg(args.num_envs, args.plant)
  cfg.seed = args.seed
  if args.upper == "hold" and not args.keep_terms:  # a stand test: only a fall ends it
    for k in list(cfg.terminations):
      if k not in ("time_out", "base_tilt", "non_finite"):
        cfg.terminations.pop(k)
  if args.table_dz:
    tb = cfg.scene.entities["table"]
    x, y, z = tb.init_state.pos
    tb.init_state = dataclasses.replace(tb.init_state, pos=(x, y, z + args.table_dz))
  if args.no_dr:
    for k in ("pd_gains", "joint_damping", "joint_friction"):
      cfg.events.pop(k, None)
  if args.video:
    cfg.viewer.width, cfg.viewer.height = 960, 720
  env = ManagerBasedRlEnv(
    cfg=cfg, device=dev, render_mode="rgb_array" if args.video else None
  )
  robot = env.scene["robot"]
  grasp, loco = Actor(args.grasp_ckpt, dev), Actor(args.p12_ckpt, dev)
  am = env.action_manager
  dims = {k: am.get_term(k).action_dim for k in am.active_terms}
  print("action terms", dims, "obs", env.observation_manager.group_obs_dim)
  assert list(dims) == ["joint_pos", "gripper", "legs"] and dims["legs"] == 14
  assert env.observation_manager.group_obs_dim["actor"][0] == grasp.n_in
  assert env.observation_manager.group_obs_dim["loco"][0] == loco.n_in == 114
  m = env.sim.mj_model
  if args.gc_left_only or args.plant == "p12":
    keep = torch.tensor(
      [
        b.startswith(("robot/arm_left", "robot/gripper_left"))
        for b in (
          mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)
        )
      ],
      device=dev,
    )
    env.sim.model.body_gravcomp[0] = (env.sim.model.body_gravcomp[0] * keep).clone()
    print("gravcomp kept on", int((env.sim.model.body_gravcomp[0] > 0).sum()), "bodies")
    fj = [
      j
      for j in range(m.njnt)
      if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and "robot" in m.joint(j).name
    ]
    print(
      "free joint actgravcomp",
      m.jnt_actgravcomp[fj].tolist(),
      "actfrcrange",
      m.jnt_actfrcrange[fj].tolist(),
    )
  if args.no_gravcomp:
    env.sim.model.body_gravcomp[:] = 0.0
    print("gravcomp off:", float(env.sim.model.body_gravcomp.abs().sum()))
  for n in (
    "leg_left_1_joint",
    "leg_left_length_joint",
    "pelvis_1_joint",
    "arm_left_1_joint",
    "arm_left_5_joint",
    "arm_right_1_joint",
    "arm_right_5_joint",
  ):
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"robot/{n}")
    print(
      f"  plant {n}: kp {m.actuator_gainprm[i, 0]:.1f} kd {-m.actuator_biasprm[i, 2]:.2f}"
    )

  # The jaw dims of P1.2's joint_pos observation are relative to its default of 0.
  jn = robot.joint_names
  pos_off = torch.zeros(len(jn), device=dev)
  for n, v in P12_DEFAULT_OVERRIDE.items():
    i = jn.index(n)
    pos_off[i] = robot.data.default_joint_pos[0, i] - v
  # joint_pos sits after base_ang_vel (3) in the loco group.
  jp = slice(3, 3 + len(jn))

  cmd = env.command_manager.get_term("goal_pose")
  twist = env.command_manager.get_term("twist")
  n = args.num_envs
  T = int((args.seconds or cfg.episode_length_s) / env.step_dt) - 1
  obs, _ = env.reset()
  start = robot.data.root_link_pos_w[:, :2].clone()
  succ = torch.zeros(n, device=dev)
  done = torch.zeros(n, dtype=torch.bool, device=dev)
  t_succ = torch.full((n,), float("nan"), device=dev)
  peak = torch.zeros(n, device=dev)
  reasons, frames = {}, []
  every = max(1, round(1 / 30 / env.step_dt))
  for i in range(T):
    twist.vel_command_b[:] = 0.0
    lo = obs["loco"].clone()
    lo[:, jp] += pos_off
    a_legs = loco(lo) if args.legs == "policy" else torch.zeros(n, 14, device=dev)
    a_up = (
      grasp(obs["actor"]) if args.upper == "grasp" else torch.zeros(n, 8, device=dev)
    )
    a = torch.cat((a_up, a_legs), dim=-1)
    obs, _, term, trunc, _ = env.step(a)
    live = ~done
    if args.upper == "hold" and i % 10 == 0 and i < 120:
      f = env.scene.sensors["arm_table_contact"].data.found > 0
      bn = robot.body_names
      gl, gr = bn.index("arm_left_7_link"), bn.index("arm_right_7_link")
      bp = robot.data.body_link_pos_w
      print(
        f"  t={i * env.step_dt:.1f} contact envs [L,R] {f.sum(0).tolist()}  hand z L {bp[:, gl, 2].mean():.3f} R {bp[:, gr, 2].mean():.3f}"
        f"  x L {bp[:, gl, 0].mean():.3f} R {bp[:, gr, 0].mean():.3f}"
      )
    s = cmd.episode_success.clone()
    newly = live & (s > 0) & (succ == 0)
    t_succ[newly] = i * env.step_dt
    succ = torch.where(live, torch.maximum(succ, s), succ)
    d = (robot.data.root_link_pos_w[:, :2] - start).norm(dim=-1)
    peak = torch.where(live, torch.maximum(peak, d), peak)
    for k in env.termination_manager.active_terms:
      v = env.termination_manager.get_term(k) & live
      if v.any() and k != "time_out":
        reasons[k] = reasons.get(k, 0) + int(v.sum())
        if k == "table_contact":
          f = env.scene.sensors["arm_table_contact"].data.found[v]
          print(
            f"  table_contact t={i * env.step_dt:.2f}s envs {int(v.sum())} [left, right] {(f > 0).sum(0).tolist()}"
          )
    done |= term | trunc
    if i in (0, 50, 100, 250):
      ob, tb = (
        env.scene["object"].data.root_link_pos_w,
        env.scene["table"].data.body_link_pos_w,
      )
      print(f"  object z {ob[:, 2].mean():.3f}  table bodies z {tb[0, :, 2].tolist()}")
      g = robot.data.projected_gravity_b
      print(
        f"  t={i * env.step_dt:.1f}s base z {robot.data.root_link_pos_w[:, 2].mean():.4f} "
        f"pitch {torch.rad2deg(torch.asin(-g[:, 0])).mean():+.2f} deg "
        f"roll {torch.rad2deg(torch.asin(g[:, 1])).mean():+.2f} deg"
      )
    if args.video and i % every == 0:
      frames.append(env.render())
  print(
    f"[{args.plant} plant, legs={args.legs}, upper={args.upper}, no_dr={args.no_dr}, no_gc={args.no_gravcomp}] success {succ.mean().item():.3f} "
    f"({int(succ.sum())}/{n}), time to success median {t_succ.nanmedian().item():.1f} s, "
    f"early terminations {reasons}, base drift peak mean/max "
    f"{100 * peak.mean().item():.1f}/{100 * peak.max().item():.1f} cm"
  )
  if args.video:
    import mediapy

    mediapy.write_video(args.video, frames, fps=round(1 / (env.step_dt * every)))
    print("wrote", args.video, len(frames), "frames")


if __name__ == "__main__":
  main()
