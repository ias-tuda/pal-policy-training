"""Plan S10.1 step 0 item 1: a W2 box checkpoint loaded into its WX-*-FT env computes what it computed.

Builds the resume's env (``BS2``-style selectors from the environment, ``SPAWN=WC TARGET=WC BOX_OBS=pose``,
``CKPT_SPAWN=W2``), loads ``--checkpoint`` through `ManipulationRunner.load` (widening the actor by the box
yaw, the critic by the box yaw and -- from a checkpoint without PQ -- PQ's inputs, and converting the actor
to the WC arm's action scales), and compares on a batch of the env's own observations: the commanded
target ``scale * mean action`` and the critic's value, against the checkpoint's networks evaluated directly
on the same observations in the old layout (the old inputs only, the raw last action in the old units).

  SPAWN=WC TARGET=WC GOAL=box-rim BOX_OBS=pose CKPT_SPAWN=W2 CMD_OBS=act ... \
    uv run python scripts/s101_resume_check.py --checkpoint BX-RIM_s1_model_4999.pt
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib

import torch


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--checkpoint", type=pathlib.Path, required=True)
  ap.add_argument("--num-envs", type=int, default=64)
  ap.add_argument(
    "--steps", type=int, default=40, help="random-action steps before the batch"
  )
  ap.add_argument("--device", default="cuda:0")
  args = ap.parse_args()

  import pal_mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from pal_mjlab.tasks.manipulation.pick_place.kangaroo.env_cfg import (
    kangaroo_pick_place_env_cfg,
    kangaroo_pick_place_rl_cfg,
  )
  from pal_mjlab.tasks.manipulation.runner import ManipulationRunner
  from rsl_rl.models import MLPModel
  from tensordict import TensorDict

  cfg = kangaroo_pick_place_env_cfg()
  cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
  rl = kangaroo_pick_place_rl_cfg()
  runner = ManipulationRunner(wrapped, dataclasses.asdict(rl), device=args.device)
  runner.load(str(args.checkpoint), map_location=args.device)
  ratio, cols = runner._action_rescale(rl.rescale_actions_on_load or {})
  ratio = ratio.to(args.device)

  obs = wrapped.get_observations()
  with torch.inference_mode():
    for _ in range(args.steps):
      obs, *_ = wrapped.step(
        0.3 * torch.randn(wrapped.action_space.shape, device=args.device)
      )
  xa, xc = obs["actor"].clone(), obs["critic"].clone()

  saved = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
  a_sd, c_sd = saved["actor_state_dict"], saved["critic_state_dict"]
  na, nc = a_sd["mlp.0.weight"].shape[1], c_sd["mlp.0.weight"].shape[1]
  xa_old, xc_old = xa[:, :na].clone(), xc[:, :nc].clone()
  if cols is not None:
    xa_old[:, cols] = xa[:, cols] / ratio
    xc_old[:, cols] = xc[:, cols] / ratio

  def net(sd, key, width, out):
    m = MLPModel(
      TensorDict({key: torch.zeros(2, width)}, batch_size=[2]),
      {key: [key]},
      key,
      out,
      (512, 256, 128),
      "elu",
      True,
    ).to(args.device)
    m.load_state_dict(
      {k: v for k, v in sd.items() if not k.startswith("distribution.")}
    )
    return m.eval()

  old_actor = net(a_sd, "actor", na, ratio.numel())
  old_critic = net(c_sd, "critic", nc, 1)
  term = env.action_manager.get_term("joint_pos")
  scale_new = (
    term.scale[0] if torch.is_tensor(term.scale) else torch.full_like(ratio, term.scale)
  )
  scale_old = scale_new * ratio
  with torch.no_grad():
    t_old = scale_old * old_actor(TensorDict({"actor": xa_old}, batch_size=[len(xa)]))
    mean_new = runner.alg._raw_actor(TensorDict({"actor": xa}, batch_size=[len(xa)]))
    t_new = scale_new * mean_new
    v_old = old_critic(TensorDict({"critic": xc_old}, batch_size=[len(xc)]))
    v_new = runner.alg._raw_critic(TensorDict({"critic": xc}, batch_size=[len(xc)]))
  print(
    f"actor in {na} -> {xa.shape[1]}, critic in {nc} -> {xc.shape[1]}, ratio {ratio.tolist()}, "
    f"last-action cols {cols}"
  )
  print(
    f"max |target_new - target_old| = {float((t_new - t_old).abs().max()):.3e} rad "
    f"(targets up to {float(t_old.abs().max()):.2f})"
  )
  print(
    f"max |value_new - value_old|  = {float((v_new - v_old).abs().max()):.3e} "
    f"(values up to {float(v_old.abs().max()):.2f})"
  )
  std = runner.alg._raw_actor.distribution
  log_std = getattr(std, "log_std_param", None)
  if log_std is not None:
    print(
      "noise in target space, new / old:",
      [
        round(x, 5)
        for x in (
          scale_new
          * log_std.exp()
          / (scale_old * a_sd["distribution.log_std_param"].exp())
        ).tolist()
      ],
    )


if __name__ == "__main__":
  main()
