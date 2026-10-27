"""Metric terms for the manipulation tasks.

The goal command's ``episode_success`` reads only the object's position against the
goal: the mug held 2 cm over the coaster counts, a mug stood upright and let go does not.
These read the task's own end state instead, from the reward term that pays it.
"""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.metrics_manager import MetricsTermCfg


def reward_term_active(
  env: ManagerBasedRlEnv, term_name: str, threshold: float = 0.5
) -> torch.Tensor:
  """1.0 where reward term ``term_name`` paid more than ``threshold`` of its weight this
  step, as of the last reward computation. (B,)

  For a binary end-state term (``stacked``, ``stood_released``, ``turned``): 1.0 while the
  task stands done. Metrics are computed after the rewards, so this is the current step.
  """
  rewards = env.reward_manager
  idx = rewards.active_terms.index(term_name)
  weight = rewards.get_term_cfg(term_name).weight
  return (rewards._step_reward[:, idx] / weight > threshold).float()


def success_metrics(term_name: str) -> dict[str, MetricsTermCfg]:
  """``episode_success``: the share of episodes that reached the end state ``term_name``
  pays at all; ``final_success``: that stood in it on their last step."""
  params = {"term_name": term_name}
  return {
    "episode_success": MetricsTermCfg(
      func=reward_term_active, params=params, reduce="max"
    ),
    "final_success": MetricsTermCfg(
      func=reward_term_active, params=params, reduce="last"
    ),
  }
