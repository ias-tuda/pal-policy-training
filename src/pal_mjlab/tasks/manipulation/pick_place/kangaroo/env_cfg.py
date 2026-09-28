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

Plan S7-2 adds three more, by name, each defaulting to what every run up to S7-1 had:
`BASELINE` (`S71` | `B`, `pick_place_env_cfg.BASELINES`: table height, episode length, target
box, observations, jaw slew), `SPAWN` (`S71` | `W1` | `W2`, `SPAWN_BOXES`; a W box brings P2's
wider action scales, `KANGAROO_GRIPPERS_LEFT_ARM_WIDE`) and `TOOL_SPEED` (`none` | `V`,
`TOOL_SPEED_CAPS`: module V, the tool speed bound in the action term).

And the plant, by `GAINS` (plan S8): a gain code from `kangaroo_constants.TABLETOP_GAINS`.
`G0`, the default, is the robot every run before S8 trained on (SysID gains, gravity
compensation on); `G15`, `G20` and `G20s` are stiff pelvis/arm gains with no gravity
compensation anywhere. Each code fixes both, so there is no invalid pairing to spell.

Plan S7-4 adds `EFFORT` (`none` | `C` | `CTP`, `pick_place_env_cfg.EFFORT_BOUNDS`): the bound on
the driven arm's effort. `C` clips the arm's targets to the joint range; `CTP` adds T -- the arm's
effort limit lowered to 0.7 x effort on the plant, mirrored as a target band in the action term
-- and P, the effort hinge. The plant half of T is built into the robot here, from the same code.
`TOOL_SPEED=V0.5` is S7-4's V at 0.5 m/s. The release-by-jaw (J) and the retreat (M8) are reward
modules, named in `REWARD`.
"""

from __future__ import annotations

import os

from mjlab.envs import ManagerBasedRlEnvCfg

from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import (
  DEFAULT_TABLETOP_GAINS,
  TABLETOP_GAINS,
)
from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import (
  BASELINES,
  EFFORT_BOUNDS,
  SPAWN_BOXES,
  TOOL_SPEED_CAPS,
  make_pick_place_env_cfg,
)
from pal_mjlab.tasks.manipulation.pick_place.rewards import REWARD_VARIANTS
from pal_mjlab.tasks.manipulation.rl_cfg import PPO_RECIPES
from pal_mjlab.tasks.manipulation.robots import (
  KANGAROO_GRIPPERS_LEFT_ARM,
  KANGAROO_GRIPPERS_LEFT_ARM_WIDE,
  kangaroo_grippers_left_arm,
)
from pal_mjlab.tasks.manipulation.runner import ManipulationRunnerCfg
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import DR_LEVELS

#: Environment variables read at registration. Named as the plan names them.
REWARD_ENV_VAR = "REWARD"
DR_ENV_VAR = "DR"
PPO_ENV_VAR = "PPO"
SLEW_ENV_VAR = "SLEW"
BASELINE_ENV_VAR = "BASELINE"
SPAWN_ENV_VAR = "SPAWN"
TOOL_SPEED_ENV_VAR = "TOOL_SPEED"
GAINS_ENV_VAR = "GAINS"
EFFORT_ENV_VAR = "EFFORT"

#: What a run with no variable set gets: C1's reward and rung, and C1r/C2r's PPO recipe.
DEFAULT_REWARD = "R-A"
DEFAULT_DR = "DR0"
DEFAULT_PPO = "PPO-stack"
DEFAULT_SLEW = 1.0
DEFAULT_BASELINE = "S71"
DEFAULT_SPAWN = "S71"
DEFAULT_TOOL_SPEED = "none"
DEFAULT_GAINS = DEFAULT_TABLETOP_GAINS
DEFAULT_EFFORT = "none"

#: The arm each spawn box trains with: the W boxes need P2's wider action scales (plan S7 P2).
SPAWN_ROBOTS = {
  "S71": KANGAROO_GRIPPERS_LEFT_ARM,
  "W1": KANGAROO_GRIPPERS_LEFT_ARM_WIDE,
  "W2": KANGAROO_GRIPPERS_LEFT_ARM_WIDE,
}


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
  """v0: the left arm only, and the arm/rung/baseline/spawn named by the environment.

  `KANGAROO_GRIPPERS_LEFT_ARM` (or its wide-scale twin for a W spawn box), not
  `KANGAROO_GRIPPERS`: the right arm is held at its nominal pose by its own actuators and is
  neither driven nor observed, which is what makes the action space 8 rather than 16. Its
  plant is the `GAINS` code's, with the `EFFORT` code's arm effort limit; G0 with no effort
  bound is exactly the spawn box's arm.
  """
  spawn = _select(SPAWN_ENV_VAR, DEFAULT_SPAWN, SPAWN_BOXES)
  baseline = _select(BASELINE_ENV_VAR, DEFAULT_BASELINE, BASELINES)
  tool_speed = _select(TOOL_SPEED_ENV_VAR, DEFAULT_TOOL_SPEED, TOOL_SPEED_CAPS)
  effort = EFFORT_BOUNDS[_select(EFFORT_ENV_VAR, DEFAULT_EFFORT, EFFORT_BOUNDS)]
  return make_pick_place_env_cfg(
    kangaroo_grippers_left_arm(
      _select(GAINS_ENV_VAR, DEFAULT_GAINS, TABLETOP_GAINS),
      SPAWN_ROBOTS[spawn],
      arm_effort_scale=effort.effort_scale,
    ),
    gripper_name="left",
    spawn=SPAWN_BOXES[spawn],
    reward=_select(REWARD_ENV_VAR, DEFAULT_REWARD, REWARD_VARIANTS),
    dr=_select(DR_ENV_VAR, DEFAULT_DR, DR_LEVELS),
    play=play,
    action_slew_rate_scale=_select_slew(),
    baseline=BASELINES[baseline],
    tool_speed_cap=TOOL_SPEED_CAPS[tool_speed],
    effort=effort,
  )


def kangaroo_pick_place_rl_cfg() -> ManipulationRunnerCfg:
  """The PPO recipe named by the environment. See `rl_cfg.PPO_RECIPES`."""
  recipe = _select(PPO_ENV_VAR, DEFAULT_PPO, PPO_RECIPES)
  return PPO_RECIPES[recipe]("kangaroo_pick_place")
