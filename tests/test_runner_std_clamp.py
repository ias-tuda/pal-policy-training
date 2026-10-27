"""The runner keeps the actor's std parameter inside its range, where it can learn."""

import math

import torch
from pal_mjlab.tasks.manipulation.rl.runner import ManipulationOnPolicyRunner
from rsl_rl.modules.distribution import GaussianDistribution


def test_a_std_parameter_past_the_cap_is_pulled_onto_it_and_learns_again():
  dist = GaussianDistribution(3, init_std=0.5, std_range=(0.1, 0.4), std_type="log")

  def std_grad():
    dist.log_std_param.grad = None
    dist.update(torch.zeros(1, 3))
    dist._distribution.entropy().sum().backward()  # type: ignore[union-attr]
    return dist.log_std_param.grad

  # Loaded from a run capped at 0.5, the parameter is dead under the 0.4 cap.
  assert torch.all(std_grad() == 0)

  class _Runner:
    class alg:
      class actor:
        distribution = dist

  ManipulationOnPolicyRunner._clamp_std_param(_Runner())  # type: ignore[arg-type]
  assert torch.allclose(dist.log_std_param.exp(), torch.full((3,), 0.4))
  assert torch.all(std_grad() != 0)
  assert math.isclose(dist.log_std_param.min().exp().item(), 0.4, rel_tol=1e-6)
