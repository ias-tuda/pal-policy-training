"""RL configuration for the KANGAROO loco-manipulation lower-body policy (P1)."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from pal_mjlab.tasks.velocity.kangaroo.rl_cfg import pal_kangaroo_ppo_runner_cfg


def pal_kangaroo_locomanip_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = pal_kangaroo_ppo_runner_cfg()
  cfg.experiment_name = "kangaroo_locomanip"
  cfg.max_iterations = 10_000
  return cfg
