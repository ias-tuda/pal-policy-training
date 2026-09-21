"""Kangaroo block-stacking task registration."""

from mjlab.tasks.manipulation.rl.runner import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from pal_mjlab.tasks.manipulation.rl_cfg import manipulation_ppo_runner_cfg

from .env_cfg import kangaroo_stack_env_cfg

register_mjlab_task(
  task_id="Mjlab-Stack-Pal-Kangaroo-Grippers",
  env_cfg=kangaroo_stack_env_cfg(),
  play_env_cfg=kangaroo_stack_env_cfg(play=True),
  rl_cfg=manipulation_ppo_runner_cfg("kangaroo_stack"),
  runner_cls=ManipulationOnPolicyRunner,
)
