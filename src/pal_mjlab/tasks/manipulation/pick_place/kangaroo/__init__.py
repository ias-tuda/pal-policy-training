"""Kangaroo pick-and-place task registration."""

from mjlab.tasks.registry import register_mjlab_task

from pal_mjlab.tasks.manipulation.runner import ManipulationRunner

from .env_cfg import kangaroo_pick_place_env_cfg, kangaroo_pick_place_rl_cfg

register_mjlab_task(
  task_id="Mjlab-PickPlace-Pal-Kangaroo-Grippers",
  env_cfg=kangaroo_pick_place_env_cfg(),
  play_env_cfg=kangaroo_pick_place_env_cfg(play=True),
  rl_cfg=kangaroo_pick_place_rl_cfg(),
  # ManipulationRunner subclasses mjlab's ManipulationOnPolicyRunner -- the manipulation
  # runner, not the velocity one, because it is the one that exports ONNX with the metadata
  # pal_policy_inferer reads -- and adds `save_only_last`.
  runner_cls=ManipulationRunner,
)
