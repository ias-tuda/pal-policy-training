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


def stack_ppo_runner_cfg(
  experiment_name: str,
  max_iterations: int = 20_000,
) -> ManipulationRunnerCfg:
  """`manipulation_ppo_runner_cfg` with Claudius's exploration settings.

  Ported from `stack_ppo_runner_cfg` on `origin/fix-stack-exploration-collapse` (9fc0dba,
  8498955), where it trains his one-arm stack and grasp tasks; adopted for C1r/C2r by S5.1
  item 9 of the pick-and-place plan. Four changes, everything else as the base recipe:

  * log-space std, initial 0.35, clamped to [1e-3, 0.5] -- on that branch a direct-space
    std either collapsed before the first grasp (low entropy) or rose past 2.7 and ruined
    placement (entropy 0.005); the upper clamp guards against the second;
  * entropy 0.005 -> 0.001;
  * learning rate 1e-3 -> 3e-4 (still adaptive, KL 0.01);
  * 24 -> 32 steps per env per iteration.
  """
  cfg = manipulation_ppo_runner_cfg(experiment_name, max_iterations)
  cfg.actor.distribution_cfg = {
    "class_name": "GaussianDistribution",
    "init_std": 0.35,
    "std_type": "log",
    "std_range": (1.0e-3, 0.5),
  }
  cfg.algorithm.entropy_coef = 0.001
  cfg.algorithm.learning_rate = 3.0e-4
  cfg.num_steps_per_env = 32
  return cfg


#: Named PPO recipes, for tasks that select one by name (pick-and-place, D10's pattern).
#: ``PPO-A`` is the plan's name for `manipulation_ppo_runner_cfg`, the recipe C0-C2 ran.
PPO_RECIPES = {
  "PPO-A": manipulation_ppo_runner_cfg,
  "PPO-stack": stack_ppo_runner_cfg,
}
