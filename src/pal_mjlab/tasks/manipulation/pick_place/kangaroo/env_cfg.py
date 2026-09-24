"""Kangaroo pick-and-place configuration.

**D10 (decided 2026-09-22): one task id, and the reward arm and DR rung are selected by
environment variable.** The alternative was a task id per (arm, rung), which multiplies D7's
single id by thirty and makes every recorded result depend on which id was spelled. The
registry builds its cfg at import, so a tyro `--env.*` flag cannot reach a *constructor*
argument that decides which reward terms exist; an environment variable read here can.

    REWARD=R-C-strict DR=DR0 uv run train Mjlab-PickPlace-Pal-Kangaroo-Grippers ...

`scripts/pick_place.sbatch` passes both through and puts them in the run name and the wandb
tags, so a curve always names its arm. The resolved cfg -- the actual reward terms, not the
name -- is dumped to `params/env.yaml` in the run directory, so a result's provenance does
not depend on the variable still being set when someone reads it later.

Unknown names raise here rather than falling back to a default: a typo that silently trained
R-A under a run named `R-C-strict` is the failure this decision exists to prevent.

The PPO recipe is selected the same way, by `PPO` (S5.1 item 9). Unlike the other two its
default is not C1's: C1r/C2r run `PPO-stack`, and `PPO=PPO-A` gives back what C0-C2 ran.

So is the arm's speed bound, by `SLEW` (plan S7, speed bound A): a factor in (0, 1] on the
URDF slew limit, `action_slew_rate_scale`. `SLEW=0.5` is FA; unset is 1.0, every run before
S7. A value that is not a number in that range raises, for the same reason as a bad name.
"""

from __future__ import annotations

import os

from mjlab.envs import ManagerBasedRlEnvCfg

from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import (
  make_pick_place_env_cfg,
)
from pal_mjlab.tasks.manipulation.pick_place.rewards import REWARD_VARIANTS
from pal_mjlab.tasks.manipulation.rl_cfg import PPO_RECIPES
from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS_LEFT_ARM
from pal_mjlab.tasks.manipulation.runner import ManipulationRunnerCfg
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import DR_LEVELS

#: Environment variables read at registration. Named as the plan names them.
REWARD_ENV_VAR = "REWARD"
DR_ENV_VAR = "DR"
PPO_ENV_VAR = "PPO"
SLEW_ENV_VAR = "SLEW"

#: What a run with no variable set gets: C1's reward and rung, and C1r/C2r's PPO recipe.
DEFAULT_REWARD = "R-A"
DEFAULT_DR = "DR0"
DEFAULT_PPO = "PPO-stack"
DEFAULT_SLEW = 1.0


def _select(var: str, default: str, valid) -> str:
  name = os.environ.get(var, default)
  if name not in valid:
    raise ValueError(
      f"{var}={name!r} is not a known name; have {tuple(sorted(valid))}. "
      f"Set {var} to one of those, or unset it for {default!r}."
    )
  return name


def _select_slew() -> float:
  raw = os.environ.get(SLEW_ENV_VAR)
  if raw is None:
    return DEFAULT_SLEW
  try:
    value = float(raw)
  except ValueError:
    value = float("nan")
  if not 0.0 < value <= 1.0:  # also catches nan
    raise ValueError(
      f"{SLEW_ENV_VAR}={raw!r} is not a slew-rate factor in (0, 1]. Set it to e.g. 0.5 "
      f"(plan S7, speed bound A), or unset it for {DEFAULT_SLEW}."
    )
  return value


def kangaroo_pick_place_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v0: the left arm only, a fixed target, and the arm/rung named by the environment.

  `KANGAROO_GRIPPERS_LEFT_ARM`, not `KANGAROO_GRIPPERS`: the right arm is held at its
  nominal pose by its own actuators and is neither driven nor observed, which is what makes
  the action space 8 rather than 16.
  """
  return make_pick_place_env_cfg(
    KANGAROO_GRIPPERS_LEFT_ARM,
    gripper_name="left",
    reward=_select(REWARD_ENV_VAR, DEFAULT_REWARD, REWARD_VARIANTS),
    dr=_select(DR_ENV_VAR, DEFAULT_DR, DR_LEVELS),
    play=play,
    action_slew_rate_scale=_select_slew(),
  )


def kangaroo_pick_place_rl_cfg() -> ManipulationRunnerCfg:
  """The PPO recipe named by the environment. See `rl_cfg.PPO_RECIPES`."""
  recipe = _select(PPO_ENV_VAR, DEFAULT_PPO, PPO_RECIPES)
  return PPO_RECIPES[recipe]("kangaroo_pick_place")
