"""Kangaroo lift configuration."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from pal_mjlab.tasks.manipulation.lift.lift_env_cfg import make_lift_env_cfg
from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS


def kangaroo_lift_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return make_lift_env_cfg(KANGAROO_GRIPPERS, gripper_name="left", play=play)
