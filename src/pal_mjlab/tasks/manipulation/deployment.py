"""Write the resolved robot startup pose beside a deployable policy."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv


def write_deployment_start_config(env: ManagerBasedRlEnv, output_path: Path) -> Path:
  """Write ordered, resolved robot defaults rather than regex-based config entries."""
  robot = env.scene["robot"]
  positions = robot.data.default_joint_pos[0].detach().cpu().tolist()
  velocities = robot.data.default_joint_vel[0].detach().cpu().tolist()
  payload = {
    "schema_version": 1,
    "description": "Resolved robot default joint state. Revolute joints use radians; slide joints use meters.",
    "note": "Training reset randomization may perturb these reference positions.",
    "joint_names": list(robot.joint_names),
    "joint_positions": {
      name: round(float(value), 8)
      for name, value in zip(robot.joint_names, positions, strict=True)
    },
    "joint_velocities": {
      name: round(float(value), 8)
      for name, value in zip(robot.joint_names, velocities, strict=True)
    },
  }
  output_path = output_path.expanduser().resolve()
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(json.dumps(payload, indent=2) + "\n")
  return output_path


def write_start_observation_distribution(
  env: ManagerBasedRlEnv, output_path: Path
) -> Path:
  """Write per-feature startup ranges from the initial parallel environments.

  The percentiles provide a lightweight reference for a deployment startup check. They
  describe reset observations, not every state encountered during training.
  """
  manager = env.observation_manager
  # Reset first: the command manager only resamples (and mjlab only adds the per-env grid
  # offset to the goal) during reset, so observations read before it have a bogus goal.
  observations, _ = env.reset()
  actor_obs = observations["actor"]
  if not isinstance(actor_obs, torch.Tensor):
    raise TypeError("Expected concatenated actor observations")

  terms: dict[str, dict[str, list[float]]] = {}
  offset = 0
  names = manager._group_obs_term_names["actor"]
  dims = manager._group_obs_term_dim["actor"]
  for name, shape in zip(names, dims, strict=True):
    width = int(torch.tensor(shape).prod().item())
    value = actor_obs[:, offset : offset + width]
    offset += width
    flat = value.detach().reshape(env.num_envs, -1).float()
    summary = torch.stack(
      (
        flat.min(dim=0).values,
        torch.quantile(flat, 0.01, dim=0),
        flat.mean(dim=0),
        flat.std(dim=0, unbiased=False),
        torch.quantile(flat, 0.99, dim=0),
        flat.max(dim=0).values,
      ),
      dim=0,
    ).cpu()
    terms[name] = {
      "min": summary[0].tolist(),
      "p01": summary[1].tolist(),
      "mean": summary[2].tolist(),
      "std": summary[3].tolist(),
      "p99": summary[4].tolist(),
      "max": summary[5].tolist(),
    }

  payload = {
    "schema_version": 1,
    "description": (
      "Per-feature actor observation statistics over the initially reset training "
      "environments. Use matching observation order, frames, scaling, and preprocessing "
      "when checking a deployment observation."
    ),
    "num_initial_envs": env.num_envs,
    "terms": terms,
  }
  output_path = output_path.expanduser().resolve()
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(json.dumps(payload, indent=2) + "\n")
  return output_path
