"""Kangaroo pick-and-place configuration."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import (
  make_pick_place_env_cfg,
)
from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS_LEFT_ARM


def kangaroo_pick_place_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v0: the left arm only, a fixed target, and no domain randomisation.

  `KANGAROO_GRIPPERS_LEFT_ARM`, not `KANGAROO_GRIPPERS`: the right arm is held at its
  nominal pose by its own actuators and is neither driven nor observed, which is what makes
  the action space 8 rather than 16.
  """
  return make_pick_place_env_cfg(
    KANGAROO_GRIPPERS_LEFT_ARM, gripper_name="left", dr="DR0", play=play
  )
