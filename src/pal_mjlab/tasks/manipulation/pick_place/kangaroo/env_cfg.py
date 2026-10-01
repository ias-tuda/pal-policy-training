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
Plan S7-6 adds `LAG` (`none` | `FP` | `JS` | `FP+JS`, `tabletop_env_cfg.LATENCY_MODELS`): the
real robot's latencies -- `FP` the FoundationPose cube tracker's 5-8 steps on the cube pose,
`JS` the jaw servo's lag behind its command. Unset is `none`, every run up to S7-5.
`TOOL_SPEED=V0.5` is S7-4's V at 0.5 m/s. The release-by-jaw (J) and the retreat (M8) are reward
modules, named in `REWARD`.
Plan S7-8 adds `SLEW=stack` (the stack task's speeds, per joint group, in place of a factor),
`ACTION` (`abs` | `delta`: the action sets or moves the commanded target), `CMD_OBS` (`act` | `cmd`:
what the actor's ``actions`` term shows -- the raw last action, or the commanded target; `delta`
always shows the target), `SPAWN=WC` / `WCK` and `TARGET` (`B` | `WC` | `WCK`: the cube and the target
drawn from one wide area -- Claudius's, or the part of it that plans -- >= 10 cm apart) and `EPISODE_S` (the episode length, s). Unset is S7-7's cfg exactly.
Plan S8 adds `SHAPE` (`cube` | `ball` | `ball-dr` | `mix`, `pick_place_env_cfg.OBJECT_SHAPES`): what is
picked up -- the 5 cm cube of every run up to S7-8, a 6 cm ball, the ball with its size and rolling
friction drawn, or a cube or a ball at random each episode. Unset is the cube.
Plan S7-9 adds `CRITIC_OBS` (`base` | `q`: the critic also observes M6's latched grade and the release
latch, PQ; the actor is unchanged, and a resume widens the loaded critic for the two inputs). Its other
codes are a reward module (`REWARD=...+PXY` / `+PT`) and a tool cap (`TOOL_SPEED=PV`). Unset is `base`.
"""

from __future__ import annotations

import math
import os

from mjlab.envs import ManagerBasedRlEnvCfg

from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import (
  DEFAULT_TABLETOP_GAINS,
  TABLETOP_GAINS,
)
from pal_mjlab.tasks.manipulation import mdp
from pal_mjlab.tasks.manipulation.mdp.occlusion import OBJ_OBS_FLAG, OCC_VIEWS, occ_cfg
from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import (
  BASELINES,
  EFFORT_BOUNDS,
  OBJECT_SHAPES,
  SPAWN_BOXES,
  TARGET_AREAS,
  TOOL_SPEED_CAPS,
  make_pick_place_env_cfg,
)
from pal_mjlab.tasks.manipulation.pick_place.rewards import REWARD_VARIANTS
from pal_mjlab.tasks.manipulation.rl_cfg import PPO_RECIPES
from pal_mjlab.tasks.manipulation.robots import (
  KANGAROO_GRIPPERS_LEFT_ARM,
  KANGAROO_GRIPPERS_LEFT_ARM_WC,
  KANGAROO_GRIPPERS_LEFT_ARM_WCK,
  KANGAROO_GRIPPERS_LEFT_ARM_WIDE,
  kangaroo_grippers_left_arm,
)
from pal_mjlab.tasks.manipulation.runner import ManipulationRunnerCfg
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import (
  ACTION_MODES,
  DR_LEVELS,
  LATENCY_MODELS,
)

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
LAG_ENV_VAR = "LAG"
OCC_ENV_VAR = "OCC"
OBJ_OBS_ENV_VAR = "OBJ_OBS"
OCC_FLAG_ENV_VAR = "OCC_FLAG"
ACTION_ENV_VAR = "ACTION"
CMD_OBS_ENV_VAR = "CMD_OBS"
TARGET_ENV_VAR = "TARGET"
EPISODE_ENV_VAR = "EPISODE_S"
SHAPE_ENV_VAR = "SHAPE"
CRITIC_OBS_ENV_VAR = "CRITIC_OBS"

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
DEFAULT_LAG = "none"
DEFAULT_OCC = "none"
DEFAULT_OBJ_OBS = "live"
DEFAULT_ACTION = "abs"
DEFAULT_TARGET = "B"
DEFAULT_SHAPE = "cube"
DEFAULT_CRITIC_OBS = "base"
#: What `CRITIC_OBS` may be (plan S7-9): the critic as before, or with M6's grade and the release
#: latch (PQ). Whether the critic gets the two extra inputs.
CRITIC_OBS_NAMES = {"base": False, "q": True}
#: What `SLEW` reads for the stack task's speeds (plan S7-8), in place of a factor.
SLEW_STACK = "stack"
#: What `CMD_OBS` may be: the raw last action (today's) or the commanded target.
CMD_OBS_NAMES = ("act", "cmd")

#: The arm each spawn box trains with: the W boxes need P2's wider action scales (plan S7 P2).
SPAWN_ROBOTS = {
  "S71": KANGAROO_GRIPPERS_LEFT_ARM,
  "W1": KANGAROO_GRIPPERS_LEFT_ARM_WIDE,
  "W2": KANGAROO_GRIPPERS_LEFT_ARM_WIDE,
  "WC": KANGAROO_GRIPPERS_LEFT_ARM_WC,
  "WCK": KANGAROO_GRIPPERS_LEFT_ARM_WCK,
}


def _select(var: str, default: str, valid) -> str:
  name = os.environ.get(var, default)
  if name not in valid:
    raise ValueError(
      f"{var}={name!r} is not a known name; have {tuple(sorted(valid))}. "
      f"Set {var} to one of those, or unset it for {default!r}."
    )
  return name


def _select_slew() -> tuple[float, bool]:
  """(factor on the URDF slew, whether it is the stack task's speeds): `SLEW` is a number in
  (0, 1] or ``stack`` (plan S7-8, `tabletop_env_cfg.STACK_ARM_SPEED_FRACTION`)."""
  raw = os.environ.get(SLEW_ENV_VAR)
  if raw is None:
    return DEFAULT_SLEW, False
  if raw == SLEW_STACK:
    return DEFAULT_SLEW, True
  try:
    value = float(raw)
  except ValueError:
    value = float("nan")
  if not 0.0 < value <= 1.0:  # also catches nan
    raise ValueError(
      f"{SLEW_ENV_VAR}={raw!r} is not a slew-rate factor in (0, 1] or {SLEW_STACK!r}. Set it "
      f"to e.g. 0.5 (plan S7, speed bound A), or unset it for {DEFAULT_SLEW}."
    )
  return value, False


def _select_command_obs(action: str) -> bool:
  """`CMD_OBS`: whether the actor's ``actions`` term shows the commanded target. Unset is the raw
  last action for ``abs`` and the target for ``delta``, whose integrator is state the policy must
  see; ``CMD_OBS=act`` with ``delta`` raises (plan S7-8)."""
  name = os.environ.get(CMD_OBS_ENV_VAR)
  if name is None:
    return action == "delta"
  if name not in CMD_OBS_NAMES:
    raise ValueError(
      f"{CMD_OBS_ENV_VAR}={name!r} is not a known name; have {CMD_OBS_NAMES}. Set it to one "
      "of those, or unset it."
    )
  if name == "act" and action == "delta":
    raise ValueError(
      f"{CMD_OBS_ENV_VAR}=act with {ACTION_ENV_VAR}=delta: a delta action integrates, so the "
      "policy has to see the target (plan S7-8). Use cmd, or unset it."
    )
  return name == "cmd"


def _select_episode() -> float | None:
  """`EPISODE_S`: the episode length in seconds, or None for the baseline's (plan S7-8)."""
  raw = os.environ.get(EPISODE_ENV_VAR)
  if raw is None:
    return None
  try:
    value = float(raw)
  except ValueError:
    value = float("nan")
  if not value > 0.0 or math.isinf(value):  # also catches nan
    raise ValueError(f"{EPISODE_ENV_VAR}={raw!r} is not a positive number of seconds")
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
  slew, stack_speed = _select_slew()
  action = _select(ACTION_ENV_VAR, DEFAULT_ACTION, ACTION_MODES)
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
    action_slew_rate_scale=slew,
    baseline=BASELINES[baseline],
    tool_speed_cap=TOOL_SPEED_CAPS[tool_speed],
    effort=effort,
    latency=LATENCY_MODELS[_select(LAG_ENV_VAR, DEFAULT_LAG, LATENCY_MODELS)],
    occlusion=_select_occlusion(),
    target_area=TARGET_AREAS[_select(TARGET_ENV_VAR, DEFAULT_TARGET, TARGET_AREAS)],
    episode_length_s=_select_episode(),
    stack_speed=stack_speed,
    action_mode=action,
    command_obs=_select_command_obs(action),
    shape=OBJECT_SHAPES[_select(SHAPE_ENV_VAR, DEFAULT_SHAPE, OBJECT_SHAPES)],
    critic_release=_critic_release(),
  )


def _critic_release() -> bool:
  """Plan S7-9's ``CRITIC_OBS``: whether the critic observes M6's grade and the release latch."""
  return CRITIC_OBS_NAMES[
    _select(CRITIC_OBS_ENV_VAR, DEFAULT_CRITIC_OBS, CRITIC_OBS_NAMES)
  ]


def _select_occlusion() -> mdp.OccCfg | None:
  """Plan S7-7's ``OCC`` (visibility model) and ``OBJ_OBS`` (what the actor sees while the cube
  is lost); ``OCC_FLAG`` = 0 / 1 overrides whether the actor gets the visibility flag (an
  eval-only filler on a policy trained without it). Unset is no occlusion, every run to S7-6."""
  occ = _select(OCC_ENV_VAR, DEFAULT_OCC, OCC_VIEWS)
  obj_obs = _select(OBJ_OBS_ENV_VAR, DEFAULT_OBJ_OBS, OBJ_OBS_FLAG)
  flag = os.environ.get(OCC_FLAG_ENV_VAR)
  if flag not in (None, "0", "1"):
    raise ValueError(f"{OCC_FLAG_ENV_VAR}={flag!r} is not 0 or 1")
  return occ_cfg(occ, obj_obs, None if flag is None else flag == "1")


def kangaroo_pick_place_rl_cfg() -> ManipulationRunnerCfg:
  """The PPO recipe named by the environment. See `rl_cfg.PPO_RECIPES`."""
  recipe = _select(PPO_ENV_VAR, DEFAULT_PPO, PPO_RECIPES)
  cfg = PPO_RECIPES[recipe]("kangaroo_pick_place")
  # PQ resumes from checkpoints whose critic had two inputs fewer (plan S7-9).
  cfg.widen_critic_on_resume = _critic_release()
  return cfg
