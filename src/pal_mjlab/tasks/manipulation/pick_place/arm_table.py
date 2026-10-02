"""Plan S7-10: keep the driven arm off the table (`ARM_TABLE`).

Siwei, 2026-10-01: on the real robot the **forearm** touched the table. In every run up to
S7-9 nothing priced that. `TABLETOP_COLLISION` makes arm 1-7 collide with the table, so the
arm cannot pass through it, but the only table penalty, `table_contact`, watches the subtree
of `arm_*_7_link` (wrist and gripper). Arm 1-6 resting on the table cost nothing, and no
contact with the table ended an episode.

The codes (plan S7-10), each on top of the reward variant named in `REWARD`:

* ``none`` -- unset; the cfg is exactly what it was before S7-10 (no sensor, no term);
* ``watch`` -- only the sensor below, so that `scripts/eval_pick_place.py` can count arm-table
  contacts of a policy trained without it. Rewards and terminations are unchanged;
* ``AC`` -- + `arm_table_contact`: -1.0 per step while any of arm 1-6 presses on the table
  harder than 1 N;
* ``AH`` -- AC + `arm_table_clearance`: per link of arm 1-6, 0 at `CLEARANCE_MARGIN` above the
  table top, rising linearly to 1 at contact, summed over the links, weight -0.5. The forearm
  is warned before it touches -- the margin a real arm with tracking error needs;
* ``AT`` -- AC + the termination `arm_table_force`: an arm-table force above 30 N ends the
  episode as a failure.
* ``AH10`` -- AH with both weights x 10 (contact -10, clearance -5). Siwei, 2026-10-02, after
  the first read: at AH's weights the two terms were 0.3-0.7 % of the `goal` reward and the
  robot's policy (WC-ABS-c s1) kept its forearm ~1 cm above the table after +1 k iterations.

None of them changes an observation or the action, so a policy trained without them resumes
under any of them (plan S7-10 resumes S7-8's ABS-c).
"""

from __future__ import annotations

import dataclasses
import re

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.manipulation.mdp import illegal_contact

from pal_mjlab.tasks.manipulation import mdp
from pal_mjlab.tasks.manipulation.robot_cfg import GripperCfg
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import DECIMATION, ROBOT, TABLE

#: The sensor's name: the driven arm's links 1-6 against the table.
ARM_TABLE_SENSOR = "arm_table_contact"

#: Links watched: 1-6. Link 7 and the gripper are the hand, which `table_contact` already
#: watches and which has to come close to the table to pick a cube off it.
ARM_LINKS = "[1-6]"

#: Height above the table top at which AH's clearance term starts, m. The scripted
#: pick-and-place's lowest forearm point (plan S7-10 step 0, item 3) must stay above it, or
#: the term fires on a correct pick; `tests/test_pick_place_s710.py` checks it at the start pose.
CLEARANCE_MARGIN = 0.03


@dataclasses.dataclass(frozen=True)
class ArmTableCfg:
  """What an `ARM_TABLE` code adds. Weights are negative (they are costs)."""

  sensor: bool = True
  contact_weight: float = 0.0
  contact_force: float = 1.0
  """Force above which an arm-table contact counts, N. Sensor noise and a grazing contact
  that MuJoCo resolves at ~0 N stay below it."""
  clearance_weight: float = 0.0
  clearance_margin: float = CLEARANCE_MARGIN
  terminate_force: float | None = None
  """Arm-table force that ends the episode (a failure), N; None: no termination."""


ARM_TABLE_CODES: dict[str, ArmTableCfg | None] = {
  "none": None,
  "watch": ArmTableCfg(),
  "AC": ArmTableCfg(contact_weight=-1.0),
  "AH": ArmTableCfg(contact_weight=-1.0, clearance_weight=-0.5),
  "AT": ArmTableCfg(contact_weight=-1.0, terminate_force=30.0),
  "AH10": ArmTableCfg(contact_weight=-10.0, clearance_weight=-5.0),
}


def arm_body_regex(gripper: GripperCfg) -> str:
  """``arm_<side>_[1-6]_link`` for the gripper's arm, from its hand body ``arm_<side>_7_link``."""
  match = re.fullmatch(r"(.+)_7_link", gripper.hand_body)
  if match is None:
    raise ValueError(
      f"hand body {gripper.hand_body!r} is not an arm's link 7; plan S7-10 assumes a "
      "7-link arm (the grippers variant)"
    )
  return f"{match.group(1)}_{ARM_LINKS}_link"


def make_arm_table_sensor(gripper: GripperCfg) -> ContactSensorCfg:
  """Arm 1-6 vs the table, one slot per link (its strongest contact), with a force history so
  that `illegal_contact`'s force threshold applies -- without a history it returns ``found``
  and ignores the threshold (mjlab ``terminations.py``).

  The history is the whole control step (`DECIMATION` substeps), not the hand sensor's 4: with
  4 of 10 substeps, a hard hit that pushes the arm clear in the step's first 6 substeps is not
  seen by AC, AT or the eval (found by the positive control, which read 0 N on a forearm driven
  2 cm into the table in 3 of 16 envs, and 600-15 000 N in the others)."""
  return ContactSensorCfg(
    name=ARM_TABLE_SENSOR,
    primary=ContactMatch(mode="body", pattern=arm_body_regex(gripper), entity=ROBOT),
    secondary=ContactMatch(mode="body", pattern=TABLE, entity=TABLE),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=DECIMATION,
  )


def apply_arm_table(
  cfg: ManagerBasedRlEnvCfg, gripper: GripperCfg, code: ArmTableCfg | None
) -> None:
  """Add a code's sensor, rewards and termination to a built pick-and-place cfg, in place."""
  if code is None:
    return
  if code.sensor or code.contact_weight or code.terminate_force is not None:
    cfg.scene.sensors = (*cfg.scene.sensors, make_arm_table_sensor(gripper))
  if code.contact_weight:
    cfg.rewards["arm_table_contact"] = RewardTermCfg(
      func=mdp.contact_penalty,
      weight=code.contact_weight,
      params={"sensor_name": ARM_TABLE_SENSOR, "force_threshold": code.contact_force},
    )
  if code.clearance_weight:
    cfg.rewards["arm_table_clearance"] = RewardTermCfg(
      func=mdp.table_clearance_penalty,
      weight=code.clearance_weight,
      params={
        "margin": code.clearance_margin,
        "asset_cfg": SceneEntityCfg(ROBOT, body_names=(arm_body_regex(gripper),)),
        "surface_cfg": SceneEntityCfg(TABLE),
      },
    )
  if code.terminate_force is not None:
    cfg.terminations["arm_table_force"] = TerminationTermCfg(
      func=illegal_contact,
      params={"sensor_name": ARM_TABLE_SENSOR, "force_threshold": code.terminate_force},
    )
