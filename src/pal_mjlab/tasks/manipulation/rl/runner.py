"""Manipulation runner with a deployment friendly startup pose artifact."""

from __future__ import annotations

from pathlib import Path

import torch
import wandb
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.rl.runner import MjlabOnPolicyRunner
from pal_mjlab.tasks.manipulation.deployment import (
  write_deployment_start_config,
  write_start_observation_distribution,
)
from pal_mjlab.tasks.manipulation.mdp import BinaryJawAction
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import GRIPPER_ACTION
from rsl_rl.modules.distribution import GaussianDistribution


class ManipulationOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
    super().__init__(env, train_cfg, log_dir, device)
    update = self.alg.update

    def update_and_clamp_std():
      losses = update()
      self._clamp_std_param()
      return losses

    self.alg.update = update_and_clamp_std
    run_dir = self.logger.log_dir
    if run_dir is None or (self.is_distributed and self.gpu_global_rank != 0):
      return

    output_dir = Path(run_dir)
    deployment_env = self.env.unwrapped
    write_deployment_start_config(
      deployment_env, output_dir / "deployment_start_config.json"
    )
    write_start_observation_distribution(
      deployment_env, output_dir / "start_observation_distribution.json"
    )

  @torch.no_grad()
  def _clamp_std_param(self) -> None:
    """Keep the actor's std parameter inside its ``std_range``.

    rsl_rl clamps the parameter only where it reads it, and a clamp passes no gradient
    outside its range: a parameter past the cap stays there for good. The entropy bonus
    drove the mug policies' std onto the 0.5 cap, and fine-tuned under a 0.4 cap, 5 to 8
    of the 8 action dims sat at 0.4 for the whole run. Clamped in place, the parameter
    sits on the bound, where the gradient still reaches it.
    """
    dist = getattr(self.alg.actor, "distribution", None)
    if not isinstance(dist, GaussianDistribution):
      return
    if dist.std_type == "log":
      dist.log_std_param.clamp_(*dist.log_std_range)
    else:
      dist.std_param.clamp_(*dist.std_range)

  def load(self, path: str, *args, **kwargs):
    infos = super().load(path, *args, **kwargs)
    self._clamp_std_param()
    return infos

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_dir, filename, onnx_path = self._get_export_paths(path)
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)
      run_name: str = (
        wandb.run.name
        if self.logger.logger_type in ("wandb", "WandbLogWriter") and wandb.run
        else "local"
      )  # type: ignore[assignment]
      env = self.env.unwrapped
      metadata = get_base_metadata(env, run_name)
      # A delta policy's action moves the target (``target += action_scale * clip(a)``)
      # instead of setting it; the deployer has to know which contract it is running.
      metadata["action_type"] = type(env.action_manager.get_term("joint_pos")).__name__
      # A binary jaw term sits after ``joint_pos`` in the action vector, one entry per
      # jaw: ``> 0`` ramps its target to closed, ``<= 0`` to open, at ``rate`` per step.
      jaw = (
        env.action_manager.get_term(GRIPPER_ACTION)
        if GRIPPER_ACTION in env.action_manager.active_terms
        else None
      )
      if isinstance(jaw, BinaryJawAction):
        metadata["gripper_action_type"] = type(jaw).__name__
        metadata["gripper_joint_names"] = list(jaw.target_names)
        metadata["gripper_open_pos"] = jaw.bounds[0, :, 0].cpu().tolist()
        metadata["gripper_closed_pos"] = jaw.bounds[0, :, 1].cpu().tolist()
        metadata["gripper_rate"] = jaw.rate[0].cpu().tolist()
      attach_metadata_to_onnx(str(onnx_path), metadata)

      start_config_path = Path(onnx_path).with_name("deployment_start_config.json")
      write_deployment_start_config(env, start_config_path)
    except Exception as e:
      print(f"[WARN] ONNX export failed (training continues): {e}")
