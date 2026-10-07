"""Plan §13.1's right skill (`pick-place-insert-plan.md`): R-I / R-I-W -- push the cross pin through the socket block
held at H by a stand-in for the left hand; trained on the left arm, run on the right through the mirror map.

``S13=R-I`` / ``R-I-W`` on top of plan §12's ``GOAL=insert`` cfg (`pick_place/insert.py`, ``SHAPE=pin``, ``TARGET=WC``).
Against that cfg:

* **the block** is not drawn on the table: `mdp.reset_held_block` puts it at H -- its bottom 10 cm over the table at
  (0.45, 0.00) in the robot frame, the same point under the mirror -- with the hold's noise (`HoldNoise`), and a
  soft weld (`mdp.held_block_weld`) ties it to a mocap **holder** that carries the left gripper's boxes where the
  left hand sits around the block at L-H's grasp (from the front, pads 48 mm up, 10 mm short of the axis:
  `DUMMY_HAND`, measured on the real meshes, step 0). The holder collides with the arm and the pin, not the block;
* **the pin** is drawn from `PIN_AREA`, the part of `WC` at least ~10 cm from H's xy, any yaw;
* **the reward** is IN's (RCTP + A + D + U, block moved) with the release ending removed: ``release`` / ``placed``
  stay at weight 0 (PQ's critic inputs), ``retreat`` and the pad-contact term go, and `through_held` pays per step
  (`THROUGH_WEIGHT`, M1's placed weight) while the pin is held through the block (`mdp.held_through`: its bottom at
  or below the block's bottom face, inside the bore, its top above the block's top face; plan §13: "flush or below
  while the right still holds it"); IN's approach (goal, goal_near), A and D pay only while the pin's top is above
  the block's top face (`mdp.over_block_gated`): their "below the entry" was written for a block on the table and,
  with the block in the air, paid a pin held under it (`GATED_OVER_BLOCK`); arm-holder contact costs `HH_WEIGHT` per
  step above 1 N (S11's HH);
* **the observations** are IN's 42-D actor and its critic, + ``pin_through`` appended to the critic.
"""

from __future__ import annotations

import dataclasses
import math

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from pal_mjlab.tasks.manipulation import mdp, scene
from pal_mjlab.tasks.manipulation.pick_place.hold import HOLD_DZ, HOLD_XY
from pal_mjlab.tasks.manipulation.pick_place.insert import BLOCK
from pal_mjlab.tasks.manipulation.robot_cfg import GripperCfg
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import DECIMATION, ROBOT, TABLE

#: IN's terms that read "below the hole's entry" and are paid only while the pin's top is above the held block's top
#: face (`mdp.over_block_gated`). Found by the early R-I-FT check (2026-10-08, `results/S13-insert/early/`): a pin held
#: under the block on the hole's axis read as through, and the approach's offset, A's window and D's axis gate all read
#: it as at the hole. U's gate (the bottom within 5 cm of the entry, in 3-D) and the faded lift need none.
GATED_OVER_BLOCK = ("goal", "goal_near", "align", "depth")

#: The holder's entity name.
HOLDER = "holder"
#: The left gripper (and arm 7) at L-H's grasp, each collision geom's axis-aligned box in the block's frame (origin
#: at its bottom face's centre, +x the left hand's pointing direction = the robot's forward), m: (name, centre,
#: half-size). `results/S13-insert/step0/dummy_hand_48_d10.json` (pads 48 mm over the bottom, 10 mm short of the
#: axis, the jaw settled on the block at 0.307 rad). Mirror-symmetric in y, so the same under the right's mirror.
DUMMY_HAND = (
  ("arm7", (-0.1789, 0.0, 0.0474), (0.0184, 0.0375, 0.0375)),
  ("palm", (-0.1112, 0.0, 0.0474), (0.0554, 0.0375, 0.0375)),
  ("inner_l", (-0.0665, -0.0257, 0.0474), (0.0272, 0.0179, 0.0190)),
  ("inner_r", (-0.0665, 0.0257, 0.0474), (0.0272, 0.0179, 0.0190)),
  ("outer_l", (-0.0720, -0.0324, 0.0474), (0.0278, 0.0269, 0.0100)),
  ("outer_r", (-0.0720, 0.0324, 0.0474), (0.0278, 0.0269, 0.0100)),
  ("tip_l", (-0.0256, -0.0428, 0.0474), (0.0289, 0.0153, 0.0170)),
  ("tip_r", (-0.0256, 0.0428, 0.0474), (0.0289, 0.0153, 0.0170)),
)
#: Where the pin is drawn: `WC` (x 0.40-0.60, y -0.04-0.26) from y 0.10 on, which keeps it >= 10 cm from H's xy
#: (plan §13.1: not under the held block). Env frame for the table at its nominal place, any yaw.
PIN_AREA_X = (0.40, 0.60)
PIN_AREA_Y = (0.10, 0.26)
#: The per-step income of a held pin through the block (M1's ``placed`` weight) and the arm-holder contact cost.
THROUGH_WEIGHT = 8.0
HH_WEIGHT = -10.0
HH_FORCE = 1.0
HH_SENSOR = "arm_holder_contact"


@dataclasses.dataclass(frozen=True)
class HoldNoise:
  """The hold's draw about H and the weld's softness (plan §13.1: R-I nominal, R-I-W wide)."""

  half_xy: float
  half_z: float
  yaw: float
  tilt: float
  solref: tuple[float, float]
  """The weld's (timeconst, dampratio) -- or (-stiffness, -damping) -- as MuJoCo reads it."""

  torquescale: float
  """The weld's angular residual's weight (`mdp.held_block_weld`)."""


#: R-I: xy +/- 1.5 cm, z +/- 1 cm, yaw +/- 10 deg, tilt +/- 3 deg; R-I-W: xy +/- 3 cm, tilt +/- 6 deg, the weld 2x
#: softer. The weld's stiffness from the scripted L-H hold (step 0 item 4: a 10 N push on the block's top moved it
#: ~10-18 mm, 1-4 deg, on the G20 arm): (-500, -45) gives 11 mm under the same push and ~1 mm of sag at rest; the
#: angular residual weighted 10x (at 1 the 100 g block's inertia let it flip over under the push, at 10 it turns
#: 0.1 deg) -- `scripts/s13_scripted.py weld`, `results/S13-insert/scripted/weld*.json`. R-I-W: (-250, -32), 22 mm,
#: torquescale 5.
HOLD_SOLREF = (-500.0, -45.0)
HOLD_TORQUESCALE = 10.0
HOLD_CODES = {
  "R-I": HoldNoise(
    half_xy=0.015,
    half_z=0.01,
    yaw=math.radians(10.0),
    tilt=math.radians(3.0),
    solref=HOLD_SOLREF,
    torquescale=HOLD_TORQUESCALE,
  ),
  "R-I-W": HoldNoise(
    half_xy=0.03,
    half_z=0.01,
    yaw=math.radians(10.0),
    tilt=math.radians(6.0),
    solref=(-250.0, -32.0),
    torquescale=5.0,
  ),
}


def holder_cfg() -> EntityCfg:
  return EntityCfg(spec_fn=lambda: mdp.holder_spec(HOLDER, DUMMY_HAND))


def _hh_sensor(gripper: GripperCfg) -> ContactSensorCfg:
  """The driven arm from its forearm (link 4) to the fingertips against the holder, the whole control step, the
  strongest contact (S11's HH sensor against the stand-in hand)."""
  forearm = gripper.hand_body.replace("_7_link", "_4_link")
  return ContactSensorCfg(
    name=HH_SENSOR,
    primary=ContactMatch(mode="subtree", pattern=forearm, entity=ROBOT),
    secondary=ContactMatch(mode="body", pattern=HOLDER, entity=HOLDER),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=DECIMATION,
  )


def apply_held_insert(
  cfg: ManagerBasedRlEnvCfg, gripper: GripperCfg, code: HoldNoise
) -> ManagerBasedRlEnvCfg:
  """Turn a built ``GOAL=insert`` cfg into R-I / R-I-W (the module docstring). Raises on any other cfg."""
  if BLOCK not in cfg.scene.entities or "reset_block" not in cfg.events:
    raise ValueError("R-I is built on plan §12's GOAL=insert (the block and its reset)")
  if "ladder" in cfg.rewards:
    raise ValueError("R-I is built on IN's terms, not IN-LAD's ladder")

  # -- Scene: the holder, the weld. ---------------------------------------------------------------
  cfg.scene.entities = {**cfg.scene.entities, HOLDER: holder_cfg()}
  cfg.scene.spec_fn = mdp.chain_spec_fns(
    cfg.scene.spec_fn,
    mdp.held_block_weld(
      f"{HOLDER}/{HOLDER}", f"{BLOCK}/{BLOCK}", code.solref, code.torquescale
    ),
  )
  cfg.scene.sensors = (*cfg.scene.sensors, _hh_sensor(gripper))

  # -- Events: the hold instead of the block's draw on the table; the pin away from H. --------------
  cfg.events["reset_block"] = EventTermCfg(
    func=mdp.reset_held_block,
    mode="reset",
    params={
      "holder_cfg": SceneEntityCfg(HOLDER),
      "block_cfg": SceneEntityCfg(BLOCK),
      "surface_cfg": SceneEntityCfg(TABLE),
      "xy": HOLD_XY,
      "dz": HOLD_DZ,
      "half_xy": code.half_xy,
      "half_z": code.half_z,
      "yaw": code.yaw,
      "tilt": code.tilt,
    },
  )
  cx, cy = scene.TABLE_POS[0], scene.TABLE_POS[1]
  reset = cfg.events["reset_objects"]
  reset.params["pose_range"] = {
    **reset.params["pose_range"],
    "x": (PIN_AREA_X[0] - cx, PIN_AREA_X[1] - cx),
    "y": (PIN_AREA_Y[0] - cy, PIN_AREA_Y[1] - cy),
  }
  reset.params.pop("goal_area", None)

  # -- Reward: no release ending; through while held; HH. ---------------------------------------
  rw = cfg.rewards
  for name in ("release", "placed"):
    rw[name] = dataclasses.replace(rw[name], weight=0.0)
  for name in ("retreat", "pad_contact"):
    rw.pop(name, None)
  depth = rw["depth"].params
  pin = {k: depth[k] for k in ("object_name", "block_name", "block_height", "hole_d")}
  for name in GATED_OVER_BLOCK:
    term = rw[name]
    rw[name] = dataclasses.replace(
      term,
      func=mdp.over_block_gated,
      params={"inner": term.func, **term.params},
    )
  rw["through"] = RewardTermCfg(
    func=mdp.pin_through_held,
    weight=THROUGH_WEIGHT,
    params={**pin, "grasp_sensor_name": gripper.contact_sensor_name},
  )
  rw["arm_holder_contact"] = RewardTermCfg(
    func=mdp.contact_penalty,
    weight=HH_WEIGHT,
    params={"sensor_name": HH_SENSOR, "force_threshold": HH_FORCE},
  )
  # The block cannot leave the hold (and must not end the episode for it).
  cfg.terminations.pop("block_dropped", None)

  # -- Critic: + through. --------------------------------------------------------------------------
  critic = cfg.observations["critic"]
  critic.terms = {
    **critic.terms,
    "pin_through": ObservationTermCfg(
      func=mdp.pin_through_obs,
      params={**pin, "grasp_sensor_name": gripper.contact_sensor_name},
    ),
  }
  return cfg
