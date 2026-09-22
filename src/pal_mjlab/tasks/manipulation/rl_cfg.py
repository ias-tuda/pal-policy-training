"""PPO configuration shared by the manipulation tasks."""

from __future__ import annotations

from mjlab.rl import RslRlModelCfg, RslRlPpoAlgorithmCfg

from pal_mjlab.tasks.manipulation.runner import ManipulationRunnerCfg


def manipulation_ppo_runner_cfg(
  experiment_name: str,
  max_iterations: int = 20_000,
) -> ManipulationRunnerCfg:
  """PPO for the manipulation tasks.

  Kept identical to mjlab's own lift-cube recipe apart from the horizon: the point of the
  first tasks is to validate the framework, so any divergence in learning behaviour should
  be attributable to the environment, not to a re-tuned optimiser.
  """
  return ManipulationRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name=experiment_name,
    # Every 1000 iterations, not mjlab's 50 or the 100 this used before: a 20k-iteration run
    # wrote 201 checkpoints, and nothing reads the intermediate ones. 21 is still dense
    # enough that a crash costs at most ~40 min at C1's 2.4 s/iteration.
    save_interval=1000,
    num_steps_per_env=24,
    max_iterations=max_iterations,
  )
