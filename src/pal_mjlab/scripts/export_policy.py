"""Export an RSL-RL policy to ONNX, optionally loading a local checkpoint.

Example:
  uv run export-policy Mjlab-Stack-Pal-Kangaroo-Grippers \
    logs/rsl_rl/kangaroo_stack/my-run/model_2000.pt
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import asdict
from pathlib import Path


def export_policy(
  task_id: str,
  checkpoint: Path | None = None,
  output: Path | None = None,
  *,
  device: str = "cpu",
  motion_file: Path | None = None,
) -> Path:
  """Create ``task_id``'s runner, optionally load ``checkpoint``, and export its actor."""
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from pal_mjlab.tasks.manipulation.deployment import write_deployment_start_config

  if checkpoint is not None:
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
      raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

  output = (
    output.expanduser().resolve()
    if output is not None
    else checkpoint.with_suffix(".onnx")
    if checkpoint is not None
    else Path("policy.onnx").resolve()
  )
  output.parent.mkdir(parents=True, exist_ok=True)

  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = 1
  if motion_file is not None:
    if "motion" not in env_cfg.commands:
      raise ValueError(f"{task_id!r} is not a motion-tracking task")
    env_cfg.commands["motion"].motion_file = str(motion_file.expanduser().resolve())

  agent_cfg = load_rl_cfg(task_id)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  try:
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped_env, asdict(agent_cfg), device=device)
    if checkpoint is not None:
      runner.load(
        str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
      )
    runner.export_policy_to_onnx(str(output.parent), output.name)

    try:
      run_name = checkpoint.parent.name if checkpoint is not None else "untrained"
      metadata = get_base_metadata(env, run_name)
      if "motion" in env.command_manager.active_terms:
        motion_cfg = env.command_manager.get_term("motion").cfg
        if hasattr(motion_cfg, "anchor_body_name"):
          metadata["anchor_body_name"] = motion_cfg.anchor_body_name
        if hasattr(motion_cfg, "body_names"):
          metadata["body_names"] = list(motion_cfg.body_names)
      attach_metadata_to_onnx(str(output), metadata)
      write_deployment_start_config(
        env, output.with_name("deployment_start_config.json")
      )
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
      # Some custom tasks split joint control across several action terms. Their actor is
      # still exportable, but mjlab's metadata helper currently assumes one "joint_pos"
      # term. Do not discard an otherwise valid ONNX file for optional metadata.
      warnings.warn(f"ONNX exported without deployment metadata: {exc}", stacklevel=2)
  finally:
    env.close()

  return output


def main() -> None:
  # Importing the task packages populates mjlab's entry-point-backed registry.
  import mjlab.tasks  # noqa: F401

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("task", help="Registered mjlab task ID used for training")
  parser.add_argument(
    "checkpoint",
    type=Path,
    nargs="?",
    help="Optional local model_*.pt checkpoint",
  )
  parser.add_argument("-o", "--output", type=Path, help="Output .onnx path")
  parser.add_argument("--device", default="cpu", help="Load device (default: cpu)")
  parser.add_argument(
    "--motion-file",
    type=Path,
    help="Local .npz motion required by some tracking-task configurations",
  )
  args = parser.parse_args()

  output = export_policy(
    args.task,
    args.checkpoint,
    args.output,
    device=args.device,
    motion_file=args.motion_file,
  )
  print(f"Exported ONNX policy to {output}")


if __name__ == "__main__":
  main()
