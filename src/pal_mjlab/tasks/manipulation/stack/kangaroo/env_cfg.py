"""Kangaroo block-stacking configuration."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS
from pal_mjlab.tasks.manipulation.stack.stack_env_cfg import make_stack_env_cfg


def kangaroo_stack_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return make_stack_env_cfg(KANGAROO_GRIPPERS, gripper_name="left", play=play)
