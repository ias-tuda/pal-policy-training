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
Plan S9 adds `GOAL` (`table` | `stack`, `pick_place_env_cfg.GOALS`: the cube on the table, or on top of a
blue base cube spawned in baseline B's target box) and `GATE` (`none` | `stack`: the jaw lock that keeps
the jaw shut while the held cube is not aligned, low and slow over the base). Unset is S8's cfg exactly.
Plan S7-9 adds `CRITIC_OBS` (`base` | `q`: the critic also observes M6's latched grade and the release
latch, PQ; the actor is unchanged, and a resume widens the loaded critic for the two inputs). Its other
codes are a reward module (`REWARD=...+PXY` / `+PT`) and a tool cap (`TOOL_SPEED=PV`). Unset is `base`.
Plan S7-10 adds `ARM_TABLE` (`none` | `watch` | `AC` | `AH` | `AT`, `arm_table.ARM_TABLE_CODES`):
the driven arm's links 1-6 against the table -- a sensor only (`watch`, for the eval), a contact
penalty (AC), + a clearance penalty (AH), or + a termination on a hard contact (AT). Unset is
`none`, every run up to S7-9.
Plan S10 adds the box goals to `GOAL` (`box` | `box-ny` | `box-rim` | `box-dr`, `box.BOX_GOALS`): the
cube goes into an open box standing in baseline B's target box -- BX, without the yaw grade, with a rim
penalty, with the box's size and mass drawn. One variable for S9's and S10's goals; unset is `table`.
Plan S10.1 puts the box on an area (`TARGET=WC` with a box `GOAL`: the box drawn from the area at any yaw,
clear of the cube) and adds `BOX_OBS` (`goal` | `pose`, `box.BOX_OBS_NAMES`): the actor sees the goal point
computed from the box (S10's, unset) or the box's own position and yaw; a resume widens the loaded actor
and critic for the appended yaw. `CKPT_SPAWN` names the spawn whose arm a checkpoint was trained with when it
is not this run's (`W2` for a W2 box checkpoint on `SPAWN=WC`, whose arm 1 / arm 4 action scales differ): every
load -- a resume's, an eval's -- then converts the actor to this env's scales (`runner.rescale_actions`).
Plan S10.2 adds `TABLE_DZ` (mm): the table's top face raised (or lowered) by that much from the
baseline's height. The real robot cell's table stands about 22 mm above baseline B's 0.815 m (Siwei,
2026-10-06, on the robot), so `TABLE_DZ=22` trains on 0.837 m. Everything else that touches the table
reads its live surface (spawns, the target, the box, the rewards, DR-scene's +/-1 cm on top), so the
height is this one number. Unset or 0 is the baseline's table exactly; |TABLE_DZ| above 50 mm raises.
Plan §12 (`pick-place-insert-plan.md`) adds the insert codes to `GOAL` (`insert` | `insert-t` | `insert-na` |
`insert-lad`, `insert.INSERT_GOALS`) and the pin to `SHAPE` (`pin`): push the cross pin through a socket block
drawn from `TARGET=WC`, on RCTP; the two only go together.
Plan S11 (two arms) adds `ARM_HOME` (`default` | `mirror`: the right arm's home, Claudius's or the
left home's mirror image), `TARGET=T11` | `T11W` (the meeting point T in the robot frame: L-T's goal,
or R-ST's / R-ST-W's base with `GOAL=stack`), `TRACKER` (`B` | `split`: the 6 + 4 mm bias split, the
base through its own tracker), `HANDOFF_ROUTE` (`none` | `up-out`: the up-out-home controller in the
action term, for the eval to start) and `ARMS` (`left` | `both`): `both` is JOINT, one policy for
both arms (`pick_place/bimanual.py`), with `HANDOFF_TRAIN` (`none` | `left`) and `START` (`scratch` |
`mixed`). Unset is every run before S11.
Plan S9.1 adds `BASE_OBS` (`goal` | `pose`, `pick_place_env_cfg.BASE_OBS_NAMES`): on a stack goal the actor
sees the goal on the base (every run before S9.1) or also the base's yaw, with M6's yaw grade taken
against the base. And `CONTACT` (`box` | `ccd`, `CONTACT_FLAGS`), the one selector whose unset value
is not the old one: ``box`` sends box-box pairs to mujoco_warp's ``box_box`` collider (decision 0004,
`tabletop_env_cfg.SIM_DISABLE_FLAGS`); ``ccd`` gives back the GJK/EPA path every run before S9.1
trained on, to re-read those runs on their own physics.
Plan S11.3 applies `TABLE_DZ` to the two-arm env too (`ARMS=both`, which `eval_chain.py` and `eval_bimanual.py`
build), so a chain is read on the same table as its one-arm skills.
"""

from __future__ import annotations

import dataclasses
import math
import os

from mjlab.envs import ManagerBasedRlEnvCfg

from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import (
  DEFAULT_TABLETOP_GAINS,
  TABLETOP_GAINS,
)
from pal_mjlab.tasks.manipulation import mdp
from pal_mjlab.tasks.manipulation.mdp.occlusion import OBJ_OBS_FLAG, OCC_VIEWS, occ_cfg
from pal_mjlab.tasks.manipulation.pick_place.arm_table import ARM_TABLE_CODES
from pal_mjlab.tasks.manipulation.pick_place.bimanual import (
  EPISODE_S as BIMANUAL_EPISODE_S,
)
from pal_mjlab.tasks.manipulation.pick_place.bimanual import (
  STARTS as BIMANUAL_STARTS,
)
from pal_mjlab.tasks.manipulation.pick_place.bimanual import (
  handoff_arm,
  make_bimanual_env_cfg,
)
from pal_mjlab.tasks.manipulation.pick_place.box import (
  BOX_GOALS,
  BOX_OBS_NAMES,
  BoxGoal,
)
from pal_mjlab.tasks.manipulation.pick_place.hold import HOLD_CODES, apply_hold
from pal_mjlab.tasks.manipulation.pick_place.insert import INSERT_GOALS, InsertGoal
from pal_mjlab.tasks.manipulation.pick_place.insert_held import (
  HOLD_CODES as HELD_INSERT_CODES,
)
from pal_mjlab.tasks.manipulation.pick_place.insert_held import apply_held_insert
from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import (
  BASE_OBS_NAMES,
  BASELINES,
  EFFORT_BOUNDS,
  GATES,
  GOALS,
  OBJECT_SHAPES,
  ROBOT_TARGETS,
  SPAWN_BOXES,
  TARGET_AREAS,
  TOOL_SPEED_CAPS,
  TRACKERS,
  make_pick_place_env_cfg,
)
from pal_mjlab.tasks.manipulation.pick_place.rewards import REWARD_VARIANTS
from pal_mjlab.tasks.manipulation.rl_cfg import PPO_RECIPES
from pal_mjlab.tasks.manipulation.robots import (
  ARM_HOMES,
  KANGAROO_GRIPPERS_LEFT_ARM,
  KANGAROO_GRIPPERS_LEFT_ARM_SIDE,
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
GOAL_ENV_VAR = "GOAL"
GATE_ENV_VAR = "GATE"
CRITIC_OBS_ENV_VAR = "CRITIC_OBS"
ARM_TABLE_ENV_VAR = "ARM_TABLE"
BOX_FEET_ENV_VAR = "BOX_FEET"
BOX_OBS_ENV_VAR = "BOX_OBS"
CKPT_SPAWN_ENV_VAR = "CKPT_SPAWN"
TABLE_DZ_ENV_VAR = "TABLE_DZ"
ARM_HOME_ENV_VAR = "ARM_HOME"
TRACKER_ENV_VAR = "TRACKER"
HANDOFF_ROUTE_ENV_VAR = "HANDOFF_ROUTE"
ARMS_ENV_VAR = "ARMS"
HANDOFF_TRAIN_ENV_VAR = "HANDOFF_TRAIN"
START_ENV_VAR = "START"
BASE_OBS_ENV_VAR = "BASE_OBS"
CONTACT_ENV_VAR = "CONTACT"
S13_ENV_VAR = "S13"

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
DEFAULT_GOAL = "table"
DEFAULT_GATE = "none"
DEFAULT_CRITIC_OBS = "base"
#: What `CRITIC_OBS` may be (plan S7-9): the critic as before, or with M6's grade and the release
#: latch (PQ). Whether the critic gets the two extra inputs.
CRITIC_OBS_NAMES = {"base": False, "q": True}
DEFAULT_ARM_TABLE = "none"
DEFAULT_BOX_OBS = "goal"
#: The largest table offset `TABLE_DZ` may ask for, mm (plan S10.2): a guard against a value in metres
#: or centimetres, not a reach limit.
TABLE_DZ_MAX_MM = 50.0
DEFAULT_ARM_HOME = "default"
DEFAULT_TRACKER = "B"
DEFAULT_BASE_OBS = "goal"
#: What `CONTACT` may be, and the disable flags each keeps out of `SIM_DISABLE_FLAGS` (decision 0004):
#: ``box`` the tabletop default, ``ccd`` mujoco_warp's own default (nativeccd on) -- every run before
#: plan S9.1.
CONTACT_FLAGS = {"box": (), "ccd": ("nativeccd",)}
DEFAULT_CONTACT = "box"
#: Plan §13.1's one-arm skills (`S13`): ``L-H`` / ``L-HP``, the left's side grasp and hold of the socket
#: block at H (`pick_place/hold.py`, on the `SPAWN=WC` cfg with the side-grasp arm); ``R-I`` / ``R-I-W``, the
#: right's insertion into the block held at H by a stand-in hand (`pick_place/insert_held.py`, on ``GOAL=insert``,
#: trained on the left arm and run mirrored). Unset is none.
S13_CODES = ("none", *HOLD_CODES, *HELD_INSERT_CODES)
#: Plan S11's selectors: the hand-off controller in the action term (eval), and JOINT's.
HANDOFF_ROUTES = ("none", "up-out")
ARMS_NAMES = ("left", "both")
HANDOFF_TRAINS = ("none", "left")
#: The reward name JOINT's runs carry (`REWARD`): its two halves are fixed, L-T's and R-ST's
#: (`bimanual.LEFT_REWARD` / `RIGHT_REWARD`), so the name only labels the run.
BIMANUAL_REWARD = "S11-BI"
#: What JOINT is built on (BS11, plan S11); every other selector must be unset or this.
BIMANUAL_FIXED = {
  "SLEW": "stack",
  "BASELINE": "B",
  "SPAWN": "W2",
  "ACTION": "abs",
  "CMD_OBS": "act",
  "TOOL_SPEED": "none",
  "OCC": "none",
  "SHAPE": "cube",
  "GOAL": "table",
  "GATE": "none",
  "TARGET": "T11",
  "TRACKER": "split",
  "ARM_HOME": "mirror",
  "CRITIC_OBS": "q",
  "HANDOFF_ROUTE": "up-out",
}
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
  if _select(ARMS_ENV_VAR, "left", ARMS_NAMES) == "both":
    return _apply_contact(_bimanual_env_cfg(play))
  spawn = _select(SPAWN_ENV_VAR, DEFAULT_SPAWN, SPAWN_BOXES)
  baseline = _select(BASELINE_ENV_VAR, DEFAULT_BASELINE, BASELINES)
  tool_speed = _select(TOOL_SPEED_ENV_VAR, DEFAULT_TOOL_SPEED, TOOL_SPEED_CAPS)
  effort = EFFORT_BOUNDS[_select(EFFORT_ENV_VAR, DEFAULT_EFFORT, EFFORT_BOUNDS)]
  slew, stack_speed = _select_slew()
  action = _select(ACTION_ENV_VAR, DEFAULT_ACTION, ACTION_MODES)
  goal, box, insert = _select_goal()
  target = _select(TARGET_ENV_VAR, DEFAULT_TARGET, (*TARGET_AREAS, *ROBOT_TARGETS))
  s13 = _select(S13_ENV_VAR, "none", S13_CODES)
  arm = SPAWN_ROBOTS[spawn]
  if s13 in HOLD_CODES:
    if (
      spawn != "WC"
      or target != DEFAULT_TARGET
      or goal != DEFAULT_GOAL
      or box is not None
      or insert is not None
    ):
      raise ValueError(
        f"{S13_ENV_VAR}={s13} is built on SPAWN=WC with TARGET and GOAL unset (H is its goal); got "
        f"SPAWN={spawn}, TARGET={target}, GOAL={os.environ.get(GOAL_ENV_VAR)}"
      )
    arm = KANGAROO_GRIPPERS_LEFT_ARM_SIDE
  if s13 in HELD_INSERT_CODES and (insert is None or insert.ladder):
    raise ValueError(
      f"{S13_ENV_VAR}={s13} is built on GOAL=insert (IN's terms; plan §13.1); got "
      f"GOAL={os.environ.get(GOAL_ENV_VAR)}"
    )
  for name in (HANDOFF_TRAIN_ENV_VAR, START_ENV_VAR):
    if os.environ.get(name) is not None:
      raise ValueError(f"{name} is JOINT's (plan S11); it needs {ARMS_ENV_VAR}=both")
  cfg = make_pick_place_env_cfg(
    kangaroo_grippers_left_arm(
      _select(GAINS_ENV_VAR, DEFAULT_GAINS, TABLETOP_GAINS),
      arm,
      arm_effort_scale=effort.effort_scale,
      right_home=_select(ARM_HOME_ENV_VAR, DEFAULT_ARM_HOME, ARM_HOMES),
    ),
    gripper_name="left",
    spawn=SPAWN_BOXES[spawn],
    reward=_select(REWARD_ENV_VAR, DEFAULT_REWARD, REWARD_VARIANTS),
    dr=_select(DR_ENV_VAR, DEFAULT_DR, DR_LEVELS),
    play=play,
    action_slew_rate_scale=slew,
    baseline=_baseline_with_table(BASELINES[baseline]),
    tool_speed_cap=TOOL_SPEED_CAPS[tool_speed],
    effort=effort,
    latency=LATENCY_MODELS[_select(LAG_ENV_VAR, DEFAULT_LAG, LATENCY_MODELS)],
    occlusion=_select_occlusion(),
    target_area=TARGET_AREAS.get(target),
    robot_target=ROBOT_TARGETS.get(target),
    tracker=_select(TRACKER_ENV_VAR, DEFAULT_TRACKER, TRACKERS),
    episode_length_s=_select_episode(),
    stack_speed=stack_speed,
    action_mode=action,
    command_obs=_select_command_obs(action),
    shape=OBJECT_SHAPES[_select(SHAPE_ENV_VAR, DEFAULT_SHAPE, OBJECT_SHAPES)],
    goal=goal,
    gate=_select(GATE_ENV_VAR, DEFAULT_GATE, GATES),
    critic_release=_critic_release(),
    arm_table=ARM_TABLE_CODES[
      _select(ARM_TABLE_ENV_VAR, DEFAULT_ARM_TABLE, ARM_TABLE_CODES)
    ],
    box=box,
    box_obs=_select_box_obs(),
    base_obs=_select(BASE_OBS_ENV_VAR, DEFAULT_BASE_OBS, BASE_OBS_NAMES),
    insert=insert,
  )
  _apply_contact(cfg)
  if s13 in HOLD_CODES:
    apply_hold(cfg, KANGAROO_GRIPPERS_LEFT_ARM.gripper("left"), HOLD_CODES[s13])
  if s13 in HELD_INSERT_CODES:
    apply_held_insert(
      cfg, KANGAROO_GRIPPERS_LEFT_ARM.gripper("left"), HELD_INSERT_CODES[s13]
    )
  if _select(HANDOFF_ROUTE_ENV_VAR, "none", HANDOFF_ROUTES) == "up-out":
    # Plan S11's hand-off route, for the eval to start (`mdp.handoff`); training is unchanged by it.
    left = KANGAROO_GRIPPERS_LEFT_ARM.gripper("left")
    cfg.actions["joint_pos"] = mdp.with_handoff(
      cfg.actions["joint_pos"], arms=(handoff_arm(left, +1.0, None),)
    )
  return cfg


def _apply_contact(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """`CONTACT`: drop the disable flags the named contact model keeps on (decision 0004)."""
  keep_on = CONTACT_FLAGS[_select(CONTACT_ENV_VAR, DEFAULT_CONTACT, CONTACT_FLAGS)]
  cfg.sim.mujoco.disableflags = tuple(
    f for f in cfg.sim.mujoco.disableflags if f not in keep_on
  )
  return cfg


def _bimanual_env_cfg(play: bool) -> ManagerBasedRlEnvCfg:
  """Plan S11's JOINT (`ARMS=both`, `pick_place/bimanual.py`). Its plant, DR, latency, arm-table
  code and episode come from `GAINS`, `EFFORT`, `DR`, `LAG`, `ARM_TABLE` and `EPISODE_S` (15 s if
  unset); everything else is fixed by BS11 (`BIMANUAL_FIXED`), and a variable set to anything else
  raises rather than train something the run's name does not say."""
  reward = os.environ.get(REWARD_ENV_VAR)
  if reward not in (None, BIMANUAL_REWARD):
    raise ValueError(
      f"{REWARD_ENV_VAR}={reward!r}: JOINT's reward is fixed (L-T's and R-ST's halves); set "
      f"{BIMANUAL_REWARD!r} or unset it"
    )
  for name, value in BIMANUAL_FIXED.items():
    got = os.environ.get(name)
    if got is not None and got != value:
      raise ValueError(f"{ARMS_ENV_VAR}=both is built on {name}={value}; got {got!r}")
  return make_bimanual_env_cfg(
    gains=_select(GAINS_ENV_VAR, DEFAULT_GAINS, TABLETOP_GAINS),
    effort=EFFORT_BOUNDS[_select(EFFORT_ENV_VAR, DEFAULT_EFFORT, EFFORT_BOUNDS)],
    baseline=_baseline_with_table(BASELINES[BIMANUAL_FIXED["BASELINE"]]),
    dr_level=_select(DR_ENV_VAR, DEFAULT_DR, DR_LEVELS),
    latency=LATENCY_MODELS[_select(LAG_ENV_VAR, DEFAULT_LAG, LATENCY_MODELS)],
    episode_length_s=_select_episode() or BIMANUAL_EPISODE_S,
    arm_table=ARM_TABLE_CODES[
      _select(ARM_TABLE_ENV_VAR, DEFAULT_ARM_TABLE, ARM_TABLE_CODES)
    ],
    handoff_train=_select(HANDOFF_TRAIN_ENV_VAR, "left", HANDOFF_TRAINS) == "left",
    start=_select(START_ENV_VAR, "scratch", BIMANUAL_STARTS),
    play=play,
  )


def _select_table_dz() -> float:
  """Plan S10.2's ``TABLE_DZ``: the table's offset from the baseline's height, in metres (the variable is
  in mm). Unset is 0."""
  raw = os.environ.get(TABLE_DZ_ENV_VAR)
  if raw is None:
    return 0.0
  try:
    value = float(raw)
  except ValueError:
    value = float("nan")
  if not abs(value) <= TABLE_DZ_MAX_MM:  # also catches nan
    raise ValueError(
      f"{TABLE_DZ_ENV_VAR}={raw!r} is not a table offset in mm within +/-{TABLE_DZ_MAX_MM:g}. Set it "
      "to e.g. 22 (the robot cell's table, plan S10.2), or unset it for the baseline's table."
    )
  return value / 1000.0


def _baseline_with_table(baseline):
  """The baseline with its table moved by ``TABLE_DZ``; the same object when the offset is 0, so an
  unset variable resolves to the baseline's cfg exactly."""
  dz = _select_table_dz()
  if dz == 0.0:
    return baseline
  return dataclasses.replace(baseline, table_top_z=round(baseline.table_top_z + dz, 6))


def _select_box_obs() -> str:
  """Plan S10.1's ``BOX_OBS``: what the actor observes of the box (`box.BOX_OBS_NAMES`)."""
  return _select(BOX_OBS_ENV_VAR, DEFAULT_BOX_OBS, BOX_OBS_NAMES)


def _select_goal() -> tuple[str, BoxGoal | None, InsertGoal | None]:
  """`GOAL`: plan S9's goal (`GOALS`: ``table`` | ``stack``), plan S10's box (`box.BOX_GOALS`) and plan §12's
  insert (`insert.INSERT_GOALS`), one variable for all. A box or insert name is the table goal plus that box /
  block. ``BOX_FEET=0`` builds the box with the flat floor every run of `0ba3f60` trained on (to re-read those
  runs on their own box); unset or 1 the corner feet (S10 report §8)."""
  name = _select(GOAL_ENV_VAR, DEFAULT_GOAL, (*GOALS, *BOX_GOALS, *INSERT_GOALS))
  feet = os.environ.get(BOX_FEET_ENV_VAR, "1")
  if feet not in ("0", "1"):
    raise ValueError(f"{BOX_FEET_ENV_VAR}={feet!r} is not 0 or 1")
  if name in GOALS:
    return name, None, None
  if name in INSERT_GOALS:
    return DEFAULT_GOAL, None, INSERT_GOALS[name]
  box = BOX_GOALS[name]
  if box is None or feet == "1":
    return DEFAULT_GOAL, box, None
  return DEFAULT_GOAL, dataclasses.replace(box, feet=False), None


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
  # PQ resumes from checkpoints whose critic had two inputs fewer (plan S7-9); plan S10.1's box pose
  # from W2 box checkpoints without the box's yaw (actor and critic) and, from BX-RIM, without PQ.
  # JOINT trains from scratch (plan S11).
  # Plan §12's insert from a box checkpoint (IN-FT): the actor +2 (the pin's tilt), the critic +3 (and the
  # box's yaw, from a checkpoint without it).
  both = _select(ARMS_ENV_VAR, "left", ARMS_NAMES) == "both"
  insert = not both and _select_goal()[2] is not None
  box_pose = not both and (_select_box_obs() == "pose" or insert)
  cfg.widen_critic_on_resume = False if both else (_critic_release() or box_pose)
  cfg.widen_actor_on_resume = box_pose
  cfg.rescale_actions_on_load = _ckpt_action_ratio()
  return cfg


def _ckpt_action_ratio() -> dict[str, float] | None:
  """Plan S10.1's ``CKPT_SPAWN``: joint regex -> (the checkpoint's arm's action scale / this run's), for
  the joints whose scales differ; None when unset or the same arm."""
  ckpt = os.environ.get(CKPT_SPAWN_ENV_VAR)
  if ckpt is None:
    return None
  ckpt = _select(CKPT_SPAWN_ENV_VAR, DEFAULT_SPAWN, SPAWN_BOXES)
  spawn = _select(SPAWN_ENV_VAR, DEFAULT_SPAWN, SPAWN_BOXES)
  old, new = SPAWN_ROBOTS[ckpt].action_scale, SPAWN_ROBOTS[spawn].action_scale
  if set(old) != set(new):
    raise ValueError(f"CKPT_SPAWN={ckpt} and SPAWN={spawn} scale different joints")
  ratio = {k: old[k] / new[k] for k in new if old[k] != new[k]}
  return ratio or None
