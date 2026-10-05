"""Plan S7-11: open the jaw before the grasp (`PREGRASP`).

Siwei, 2026-10-05: the deployed WC-c s1 @ 5500 misses on the robot when the jaw is not fully
open on the way in and a finger pushes the cube away. The codes, each on top of the run's
reward variant (`mdp.pregrasp` has the terms):

* ``none`` -- unset; the cfg is exactly what it was before S7-11;
* ``watch`` -- only the per-episode state's reset event, so that the eval reads the push and
  the grasp onset the same way for every policy. Rewards and action unchanged;
* ``NP`` -- no push: -2 x clip((d - 8 mm) / 2 cm, 0, 1) per step to the episode's end, d the
  cube's largest horizontal displacement from its spawn before the first grasp;
* ``JO`` -- jaw open on the approach: -1 x the jaw's angle per step while before the grasp and
  the cube is outside the grasp window (`mdp.pregrasp.GRASP_WINDOW`);
* ``G`` -- the gate: before the grasp, while the cube as observed is outside the window, the
  jaw's commanded target is at most 0.03 rad. A deployer applies the same gate;
* ``S`` -- NP + JO + YG (once, at the first two-pad contact, exp(-(rel / 15 deg)^2) of the
  hand's yaw against the cube's, mod 90 deg), for the from-scratch run;
* ``OVR`` -- eval only: the deployer's patch, the jaw's target forced to 0 under G's
  condition, on a policy trained without it.

NP, JO, YG and ``watch`` change no observation and no action, and G / OVR keep the action's
width, so WC-c s1 @ 5500 resumes under each (plan S7-11's WJ-*).
"""

from __future__ import annotations

import dataclasses
import math
import re

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from pal_mjlab.tasks.manipulation.mdp import pregrasp as pg
from pal_mjlab.tasks.manipulation.robot_cfg import GripperCfg
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import OBJECT, ROBOT, SUPPORT_SENSOR

#: NP (plan S7-11 section 3): the dead band covers the cube being squared up by a closing jaw
#: (step 0 item 3 measures it; 8 mm is the plan's assumption), full penalty 2 cm past it.
NP_WEIGHT = -2.0
NP_DEADBAND = 0.008
NP_SPAN = 0.02
#: JO: per step, x the jaw angle in rad (plan: w = 1, to be checked against the term sizes at
#: +500 iterations, L15).
JO_WEIGHT = -1.0
#: G: the largest commanded jaw angle while armed, rad. OVR forces it to 0.
G_CAP = 0.03
#: YG: a one-step bonus, so its weight is in units of one step's reward. 25 x dt (0.02 s) =
#: 0.5, half a second of the `grasped` term's income -- the plan leaves the weight open ("the
#: size of M6's orientation term"); this is a judgment call, written down.
YG_WEIGHT = 25.0
YG_YAW_STD = math.radians(15.0)


@dataclasses.dataclass(frozen=True)
class PregraspCfg:
  push_weight: float = 0.0
  open_weight: float = 0.0
  align_weight: float = 0.0
  gate_cap: float | None = None
  """The jaw cap of G / OVR, rad; None for no gate."""


PREGRASP_CODES: dict[str, PregraspCfg | None] = {
  "none": None,
  "watch": PregraspCfg(),
  "NP": PregraspCfg(push_weight=NP_WEIGHT),
  "JO": PregraspCfg(open_weight=JO_WEIGHT),
  "G": PregraspCfg(gate_cap=G_CAP),
  "S": PregraspCfg(
    push_weight=NP_WEIGHT, open_weight=JO_WEIGHT, align_weight=YG_WEIGHT
  ),
  "OVR": PregraspCfg(gate_cap=0.0),
}


def apply_pregrasp(
  cfg: ManagerBasedRlEnvCfg, gripper: GripperCfg, code: PregraspCfg | None
) -> None:
  """Add a code's event, rewards and gate to a built pick-and-place cfg, in place."""
  if code is None:
    return
  grasp = gripper.contact_sensor_name
  tcp = SceneEntityCfg(ROBOT, site_names=(gripper.tcp_site,))
  # After every other reset event: it only marks the env, the spawn is read at the next step.
  cfg.events["reset_pregrasp"] = EventTermCfg(func=pg.reset_pregrasp, mode="reset")
  if code.push_weight:
    cfg.rewards["pregrasp_push"] = RewardTermCfg(
      func=pg.pregrasp_push_penalty,
      weight=code.push_weight,
      params={
        "object_name": OBJECT,
        "grasp_sensor_name": grasp,
        "deadband": NP_DEADBAND,
        "span": NP_SPAN,
      },
    )
  if code.open_weight:
    cfg.rewards["pregrasp_jaw"] = RewardTermCfg(
      func=pg.pregrasp_jaw_closed,
      weight=code.open_weight,
      params={
        "object_name": OBJECT,
        "grasp_sensor_name": grasp,
        "support_sensor_name": SUPPORT_SENSOR,
        "jaw_cfg": SceneEntityCfg(
          ROBOT, joint_names=(re.escape(gripper.finger_joint),)
        ),
        "tcp_cfg": tcp,
      },
    )
  if code.align_weight:
    cfg.rewards["pregrasp_align"] = RewardTermCfg(
      func=pg.pregrasp_grasp_alignment,
      weight=code.align_weight,
      params={
        "object_name": OBJECT,
        "grasp_sensor_name": grasp,
        "tcp_cfg": tcp,
        "yaw_std": YG_YAW_STD,
      },
    )
  if code.gate_cap is not None:
    cfg.actions["joint_pos"] = pg.gate_cfg_from(
      cfg.actions["joint_pos"],
      gate_joints=(re.escape(gripper.finger_joint),),
      cap=code.gate_cap,
      tcp_site=gripper.tcp_site,
      object_name=OBJECT,
      grasp_sensor_name=grasp,
      support_sensor_name=SUPPORT_SENSOR,
    )
