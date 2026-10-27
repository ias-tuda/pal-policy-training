"""Plan S11's SEQ chain (`MIRROR=right`): L-T on the left arm, then R-ST on the right arm through the
mirror map, in the two-arm env; and the mirror adapter's exactness test.

SEQ trains both skills on the left arm (plan S11, *Two structures*): L-T places A at T, R-ST stacks a
cube on a base at T. On the robot the right arm runs R-ST through a fixed sign map -- exact, since the
right arm is the left's mirror image (`robots.ARM_MIRROR_SIGNS`, measured) and the scene is symmetric
about y = 0 with the right home mirrored. The adapter, in the robot (base) frame:

* observations: joint position (relative to home), velocity and last action x s on arm 1-7, the jaw
  unchanged; positions y -> -y; the TCP frame R -> M R F M (M = diag(1, -1, 1), F = diag(-1, -1, 1)
  -- the right hand's frame is the mirrored left frame turned 180 deg about the approach axis); a yaw
  (cos, sin) -> (cos, -sin); R-ST's goal is A's tracked position plus the two half-heights;
* actions: x s on arm 1-7, the jaw unchanged (the right's action origin is mirror(left home)).

The chain: L-T drives the left arm and the right holds its home; once the left's hand-off (`mdp.handoff`,
on its own trigger as in JOINT's training) reaches its home leg, R-ST drives the right arm; the right is
handed off on the same trigger (`eval_bimanual.Reads`). Scored per episode as JOINT is
(`eval_bimanual.summarise`).

``--mirror-test`` runs one actor on both arms from mirrored states (DR0, no observation noise, live
poses) and reports how far the right TCP's path is from the left's mirror image -- plan S11 step 0 item 1
asks for 1e-4 m.

    ARMS=both ... python scripts/eval_chain.py --lt model_8999.pt --rst model_7999.pt --json out.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

TASK = "Mjlab-PickPlace-Pal-Kangaroo-Grippers"
#: The one-arm actor's observation (BS11's 38-D, `tabletop_env_cfg._deployable_observations`).
ONE_ARM_TERMS = (
  "joint_pos",
  "joint_vel",
  "actions",
  "tcp_pos",
  "tcp_quat",
  "object_pos",
  "object_yaw",
  "target_pos",
)


class OneArmActor:
  """A one-arm checkpoint's deterministic actor: ``mlp((obs - mean) / (std + 0.01))`` (rsl_rl's
  `MLPModel` with its `EmpiricalNormalization`), ELU between the layers."""

  EPS = 1e-2

  def __init__(self, path: pathlib.Path, device: str) -> None:
    sd = torch.load(path, map_location=device, weights_only=False)["actor_state_dict"]
    self.mean = sd["obs_normalizer._mean"]
    self.std = sd["obs_normalizer._std"]
    n = len([k for k in sd if k.startswith("mlp.") and k.endswith(".weight")])
    self.layers = [
      (sd[f"mlp.{2 * i}.weight"], sd[f"mlp.{2 * i}.bias"]) for i in range(n)
    ]
    self.obs_dim = self.layers[0][0].shape[1]
    self.act_dim = self.layers[-1][0].shape[0]
    if (self.obs_dim, self.act_dim) != (38, 8):
      raise ValueError(
        f"{path}: a one-arm actor is 38 -> 8, this is {self.obs_dim} -> {self.act_dim}"
      )

  def __call__(self, obs: torch.Tensor) -> torch.Tensor:
    x = (obs - self.mean) / (self.std + self.EPS)
    for i, (w, b) in enumerate(self.layers):
      x = x @ w.T + b
      if i < len(self.layers) - 1:
        x = torch.nn.functional.elu(x)
    return x


class MirrorAdapter:
  """The two arms' one-arm observations out of the two-arm env's groups, and the right's actions
  back (see the module docstring)."""

  def __init__(self, env) -> None:
    from pal_mjlab.tasks.manipulation.robots import ARM_MIRROR_SIGNS

    self.env = env
    dev = env.device
    s = torch.tensor(ARM_MIRROR_SIGNS, device=dev)
    self.s8 = torch.cat([s, torch.ones(1, device=dev)])  # arm 1-7 and the jaw
    self.s7 = s
    self.m = torch.diag(torch.tensor([1.0, -1.0, 1.0], device=dev))
    self.f = torch.diag(torch.tensor([-1.0, -1.0, 1.0], device=dev))
    term = env.action_manager.get_term("joint_pos")
    names = [n.split("/")[-1] for n in term.target_names]
    self.left_cols = [names.index(f"arm_left_{i}_joint") for i in range(1, 8)]
    self.left_cols.append(names.index("gripper_left_finger_joint"))
    self.right_cols = [names.index(f"arm_right_{i}_joint") for i in range(1, 8)]
    self.right_cols.append(names.index("gripper_right_finger_joint"))
    jp_names = self._term_joints(env, "joint_pos")
    self.jp_left = [jp_names.index(names[i]) for i in self.left_cols]
    self.jp_right = [jp_names.index(names[i]) for i in self.right_cols]
    jv_names = self._term_joints(env, "joint_vel")
    self.jv_left = [jv_names.index(f"arm_left_{i}_joint") for i in range(1, 8)]
    self.jv_right = [jv_names.index(f"arm_right_{i}_joint") for i in range(1, 8)]

  @staticmethod
  def _term_joints(env, term: str) -> list[str]:
    """The joints an actor joint term reads, in its order (its entity cfg resolved here)."""
    import dataclasses

    cfg = dataclasses.replace(
      env.cfg.observations["actor"].terms[term].params["asset_cfg"]
    )
    cfg.resolve(env.scene)
    robot = env.scene["robot"]
    ids = cfg.joint_ids
    if isinstance(ids, slice):
      ids = range(len(robot.joint_names))[ids]
    return [robot.joint_names[int(i)] for i in ids]

  def terms(self, obs: dict, group: str) -> dict[str, torch.Tensor]:
    """The group's terms, by name, sliced out of the env's concatenated vector for it -- what
    ``env.step`` / ``env.reset`` returned, not a second computation (that would advance the terms'
    delay buffers twice in a step)."""
    mgr = self.env.observation_manager
    obs = obs[group]
    out, i = {}, 0
    for name, shape in zip(
      mgr.active_terms[group], mgr.group_obs_term_dim[group], strict=True
    ):
      w = shape[0]
      out[name] = obs[:, i : i + w]
      i += w
    return out

  def _mirror_quat(self, quat: torch.Tensor) -> torch.Tensor:
    from mjlab.utils.lab_api.math import matrix_from_quat, quat_from_matrix

    rot = self.m @ matrix_from_quat(quat) @ self.f @ self.m
    q = quat_from_matrix(rot)
    return torch.where(q[:, :1] < 0.0, -q, q)

  def left(self, t: dict) -> torch.Tensor:
    """L-T's 38-D observation: the left arm, A and T."""
    return torch.cat(
      [
        t["joint_pos"][:, self.jp_left],
        t["joint_vel"][:, self.jv_left],
        t["actions"][:, self.left_cols],
        t["tcp_pos"],
        t["tcp_quat"],
        t["object_pos"],
        t["object_yaw"],
        t["target_pos"],
      ],
      dim=-1,
    )

  def right(
    self, t: dict, base_term: str = "object_pos", object_term: str = "object_b_pos"
  ) -> torch.Tensor:
    """R-ST's 38-D observation, mirrored: the right arm, B, and the goal on A (A's tracked position
    plus both half-heights, `mdp.stack_goal_position_tracked_b`'s)."""
    flip = torch.tensor([1.0, -1.0, 1.0], device=self.env.device)
    yaw = t[object_term.replace("_pos", "_yaw")] * torch.tensor(
      [1.0, -1.0], device=self.env.device
    )
    goal = t[base_term].clone()
    goal[:, 2] += 0.05
    return torch.cat(
      [
        t["joint_pos"][:, self.jp_right] * self.s8,
        t["joint_vel"][:, self.jv_right] * self.s7,
        t["actions"][:, self.right_cols] * self.s8,
        t["tcp_pos_right"] * flip,
        self._mirror_quat(t["tcp_quat_right"]),
        t[object_term] * flip,
        yaw,
        goal * flip,
      ],
      dim=-1,
    )

  def actions(
    self, left: torch.Tensor | None, right: torch.Tensor | None
  ) -> torch.Tensor:
    n, dev = self.env.num_envs, self.env.device
    a = torch.zeros(n, len(self.left_cols) + len(self.right_cols), device=dev)
    if left is not None:
      a[:, self.left_cols] = left
    if right is not None:
      a[:, self.right_cols] = right * self.s8
    return a


def parse_args(argv=None) -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--lt", type=pathlib.Path, help="L-T checkpoint (the left's placer)")
  p.add_argument(
    "--rst", type=pathlib.Path, help="R-ST / R-ST-W checkpoint (run mirrored)"
  )
  p.add_argument("--task", default=TASK)
  p.add_argument("--episodes", type=int, default=128)
  p.add_argument("--num-envs", type=int, default=64)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--episode-s", type=float, default=30.0)
  p.add_argument("--right-timeout", type=float, default=24.0)
  p.add_argument("--obs-noise", choices=("off", "on"), default="on")
  p.add_argument("--json", type=pathlib.Path, default=None)
  p.add_argument(
    "--mirror-test",
    action="store_true",
    help="step 0 item 1: one actor (--lt) on both arms from mirrored states; TCP paths compared",
  )
  p.add_argument("--steps", type=int, default=50, help="--mirror-test: control steps")
  return p.parse_args(argv)


def mirror_test(args) -> dict:
  """Both arms driven by the same actor -- the left plainly, the right through the adapter -- from
  mirrored states, DR0, no noise, live poses (the critic group); the right TCP's path against the
  left's mirror image."""
  import pal_mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg
  from pal_mjlab.tasks.manipulation.pick_place import bimanual as bi

  cfg = load_env_cfg(args.task, play=True)
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  cfg.episode_length_s = 60.0
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  obs, _ = env.reset()
  ad = MirrorAdapter(env)
  actor = OneArmActor(args.lt, args.device)
  # Mirror the scene: B at A's mirror image (pose and yaw), the robot at its (mirrored) home.
  a = env.scene["object"]
  b = env.scene[bi.CUBE_B]
  pose = a.data.root_link_pose_w.clone()
  origins = env.scene.env_origins
  rel = pose[:, :3] - origins
  rel[:, 1] = -rel[:, 1]
  w, x, y, z = pose[:, 3:7].unbind(-1)
  b_pose = torch.cat([rel + origins, torch.stack([w, -x, y, -z], dim=-1)], dim=-1)
  b.write_root_link_pose_to_sim(b_pose)
  b.write_root_link_velocity_to_sim(torch.zeros_like(b.data.root_link_vel_w))
  # One idle step so that every observation sees the mirrored scene.
  obs, *_ = env.step(torch.zeros(env.action_space.shape, device=env.device))
  robot = env.scene["robot"]
  site_l = robot.site_names.index("tcp_left")
  site_r = robot.site_names.index("tcp_right")
  err = []
  with torch.inference_mode():
    for _ in range(args.steps):
      t = ad.terms(obs, "critic")
      # The right's goal is T's mirror image (T is on the centre line), not A's top.
      t_r = dict(t)
      t_r["object_pos_for_goal"] = t["target_pos"] * torch.tensor(
        [1.0, -1.0, 1.0], device=env.device
      )
      t_r["object_pos_for_goal"][:, 2] -= 0.05
      obs_l = ad.left(t)
      obs_r = ad.right(t_r, base_term="object_pos_for_goal")
      last_l, last_r = actor(obs_l), actor(obs_r)
      obs, *_ = env.step(ad.actions(last_l, last_r))
      pl = robot.data.site_pos_w[:, site_l] - origins
      pr = robot.data.site_pos_w[:, site_r] - origins
      err.append(
        torch.norm(pr - pl * torch.tensor([1.0, -1.0, 1.0], device=env.device), dim=-1)
      )
  e = torch.stack(err)  # (T, B)
  out = {
    "steps": args.steps,
    "tcp_mirror_error_max_m": float(e.max()),
    "tcp_mirror_error_median_m": float(e.median()),
    "per_step_max_m": [float(v) for v in e.amax(dim=1)],
  }
  print(json.dumps({k: v for k, v in out.items() if k != "per_step_max_m"}, indent=1))
  env.close()
  return out


def chain(args) -> dict:
  from eval_bimanual import Reads, build_env, summarise
  from pal_mjlab.tasks.manipulation.mdp.handoff import HOME

  env = build_env(args)
  ad = MirrorAdapter(env)
  lt, rst = OneArmActor(args.lt, args.device), OneArmActor(args.rst, args.device)
  reads = Reads(env)
  term = env.action_manager.get_term("joint_pos")
  right_timeout = round(args.right_timeout / env.step_dt)
  obs, _ = env.reset()
  with torch.inference_mode():
    while len(reads.done) < args.episodes:
      start = reads.step(right_timeout)
      t = ad.terms(obs, "actor")
      a_l = lt(ad.left(t))
      go = (term.phase[:, term.arm_index("left")] >= HOME).unsqueeze(-1)
      a_r = torch.where(go, rst(ad.right(t)), torch.zeros_like(a_l))
      term.start_handoff("right", start & go.squeeze(-1))
      obs, _, term_done, trunc, _ = env.step(ad.actions(a_l, a_r))
      done = term_done | trunc
      finished = torch.nonzero(done, as_tuple=False).flatten()
      if finished.numel():
        reads.finish(finished, env.step_dt)
  rows = reads.done[: args.episodes]
  result = {
    "lt": str(args.lt),
    "rst": str(args.rst),
    "seed": args.seed,
    **summarise(rows, args.right_timeout),
  }
  for k, v in result.items():
    print(f"{k:36s} {v}")
  env.close()
  return {**result, "episodes_detail": rows}


def main(argv=None) -> None:
  args = parse_args(argv)
  out = mirror_test(args) if args.mirror_test else chain(args)
  if args.json:
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.json}")


if __name__ == "__main__":
  sys.exit(main())
