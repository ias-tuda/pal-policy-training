"""Pal Robotics KANGAROO constants."""

import dataclasses
import math
import re
from pathlib import Path

import mujoco
import torch
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg
from pal_mjlab import PAL_MJLAB_SRC_PATH

# There are multiple variants of the KANGAROO robot. For simplicity, we only implemented the following:
# - kangaroo: simplified model with 4 DoF per arm and a fake forearm
# - kangaroo_hands: simplified model with 5 DoF per arm and a Seed Robotics hand
# - kangaroo_gripper: simplified model with 7 DoF per arm and a gripper
# - kangaroo_full: full model with 4 DoF per arm and a fake forearm

REGEX_LEG_LENGTH_JOINTS_ONLY = r"leg_.*_length_joint"
# Excludes the femur/knee joints (driven by the leg four-bar equalities) and the gripper
# finger joints (driven by the jaw four-bar). The grippers variant would otherwise pull
# its 12 finger joints into the locomotion posture reward, whose per-joint std tables
# only cover legs, pelvis and arms.
REGEX_ALL_ACTUATED_JOINTS = (
  r"^(?!leg_.*_femur_joint$|leg_.*_knee_joint$|gripper_.*_joint$).*$"
)
REGEX_FEMUR_AND_KNEE_LINKS = (
  r"^(leg_left_femur_link|leg_right_femur_link|leg_left_knee_link|leg_right_knee_link)$"
)

KANGAROO_PATH = PAL_MJLAB_SRC_PATH / "robots" / "pal_kangaroo" / "xmls"
KANGAROO_XML = KANGAROO_PATH / "kangaroo.xml"
KANGAROO_LOWER_BODY_XML = KANGAROO_PATH / "kangaroo_lower_body.xml"
KANGAROO_HANDS_XML = KANGAROO_PATH / "kangaroo_hands.xml"
KANGAROO_GRIPPERS_XML = KANGAROO_PATH / "kangaroo_grippers.xml"

for p in [
  KANGAROO_PATH,
  KANGAROO_XML,
  KANGAROO_LOWER_BODY_XML,
  KANGAROO_HANDS_XML,
  KANGAROO_GRIPPERS_XML,
]:
  assert p.exists(), f"Missing: {p}"

##
# Actuator Parameters (BeyondMimic methodology)
##

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0
FACTOR = 0.05

HIP_XY_CONVEX_HULL_POINTS = torch.tensor(
  [
    [-0.742, 0.035],
    [-0.742, -0.094],
    [-0.742, -0.167],
    [-0.707, -0.243],
    [-0.655, -0.349],
    [-0.61, -0.411],
    [-0.344, -0.413],
    [-0.061, -0.41],
    [0.307, -0.404],
    [0.486, -0.4],
    [0.55, -0.354],
    [0.638, -0.282],
    [0.709, -0.186],
    [0.72, -0.081],
    [0.722, 0.054],
    [0.708, 0.18],
    [0.641, 0.301],
    [0.531, 0.389],
    [0.448, 0.45],
    [0.171, 0.453],
    [-0.164, 0.455],
    [-0.434, 0.461],
    [-0.605, 0.467],
    [-0.659, 0.404],
    [-0.713, 0.309],
    [-0.742, 0.222],
    [-0.742, 0.133],
  ]
)

ANKLE_XY_CONVEX_HULL_POINTS = torch.tensor(
  [
    [0.707, 0.005],
    [0.648, 0.112],
    [0.576, 0.23],
    [0.484, 0.38],
    [0.443, 0.439],
    [0.266, 0.443],
    [0.008, 0.441],
    [-0.293, 0.45],
    [-0.46, 0.448],
    [-0.505, 0.379],
    [-0.594, 0.244],
    [-0.686, 0.098],
    [-0.744, 0.001],
    [-0.688, -0.099],
    [-0.604, -0.231],
    [-0.499, -0.394],
    [-0.445, -0.472],
    [-0.254, -0.469],
    [0.005, -0.462],
    [0.232, -0.456],
    [0.429, -0.46],
    [0.475, -0.382],
    [0.583, -0.207],
    [0.665, -0.071],
  ]
)


def _calc_actuator_params(
  gear_ratio: float, motor_inertia: float, effort: float
) -> dict:
  """Calculate armature, stiffness, and damping for an actuator."""
  armature = FACTOR * motor_inertia * gear_ratio**2
  stiffness = round(armature * NATURAL_FREQ**2, 3)
  damping = round(2.0 * DAMPING_RATIO * armature * NATURAL_FREQ, 3)
  return {
    "armature": armature,
    "stiffness": stiffness,
    "damping": damping,
    "effort_limit": effort,
  }


def _calc_leg_params(
  stiffness: float,
  effort: float,
  armature: float,
  frictionloss: float,
  viscous_damping: float,
) -> dict:
  """Calculate leg actuator parameters."""
  damping = round(2.0 * DAMPING_RATIO * stiffness / NATURAL_FREQ, 3)
  return {
    "armature": armature,
    "stiffness": stiffness,
    "damping": damping,
    "effort_limit": effort,
    "frictionloss": frictionloss,
    "viscous_damping": viscous_damping,
  }


def _pal_impedance_params(
  stiffness: float, damping: float, effort: float, armature: float
) -> dict:
  """Actuator params for the impedance-driven joints: K, D, effort and armature, explicit.

  History: these were PAL's deployed impedance-controller gains
  (``kangaroo_controller_configuration/.../impedance_control/*.yaml``: arm K 40/20, D 1.0,
  torque_clamp 50), which is what ``pal_local_joint_control`` runs on hardware. Since S5.1
  (2026-09-23) they are the SysID training gains -- see the blocks below. Those equal what
  ``_calc_actuator_params`` derives from the drivetrain (K 49.9/26.2, D 3.18/1.67), which
  PAL's values had been measured against. Whether the deployed controller is retuned to
  match is a sim-to-real question this file does not answer.

  Armature stays from ``_calc_actuator_params`` -- it models reflected rotor inertia, which
  is a property of the drivetrain and independent of the controller gains.
  """
  return {
    "armature": armature,
    "stiffness": stiffness,
    "damping": damping,
    "effort_limit": effort,
  }


# Reflected rotor inertia, from (gear_ratio, motor_inertia).
_ARMATURE_S_PLUS = FACTOR * 1.728e-5 * 121**2  # arm 1-2, pelvis
_ARMATURE_S_MINUS = FACTOR * 1.3e-5 * 101**2  # arm 3-7

# The SysID training gains (S5.1 item 1, 2026-09-23; Claudius's
# `origin/fix-stack-exploration-collapse` 42aa79f, "fix gains from Nico"): K/D from the
# drivetrain model -- the `_calc_actuator_params` values quoted above -- and effort 50 on the
# S+ drives, 25 on the S- drives. They replace PAL's impedance-config 40/20 K, 1.0/2.0 D and
# uniform 50 N.m clamp. Both pelvis joints get the S+ values. Global: every Kangaroo variant
# and task uses these blocks. A knock-on in the velocity/reaching tasks, whose non-arm
# action scales are 0.25 * effort / K: pelvis_2's scale goes 0.625 -> 0.250 rad.
S_PLUS = _pal_impedance_params(49.940, 3.179, 50, _ARMATURE_S_PLUS)  # arm_*_1, arm_*_2
S_MINUS = _pal_impedance_params(26.177, 1.666, 25, _ARMATURE_S_MINUS)  # arm_*_3,4,5
XS = _pal_impedance_params(26.177, 1.666, 25, _ARMATURE_S_MINUS)  # arm_*_6,7
PELVIS_1 = _pal_impedance_params(49.940, 3.179, 50, _ARMATURE_S_PLUS)
PELVIS_2 = _pal_impedance_params(49.940, 3.179, 50, _ARMATURE_S_PLUS)

# The gripper is driven by gripper_controller (a JointTrajectoryController), not by the
# impedance path, so it has no impedance_control entry. PAL's nominal is pids.yaml
# {p: 500, d: 0.5, i: 2, torque_clamp: 50} -- but p=500 on a ~5e-6 kg.m^2 finger inside an
# equality-constrained four-bar is not numerically viable at dt=5ms (a servo sweep drives the
# jaw cleanly at kp=8). The value below is chosen for a stable, firm pinch and is
# UNCALIBRATED against the real grip force.
GRIPPER = _pal_impedance_params(20.0, 0.5, 20, 1e-4)

# The tabletop tasks' jaw (decision 0002, S6). At kp 20 the jaw closes on a centred 5 cm cube
# at ~117 N per pad, and that load buckles the four-bar -- the equalities stretch by ~31 mm
# and the cube is thrown out. kp 8 grips at ~36 N per pad, still ~70x what a 50 g cube needs,
# and with the passive armature below it keeps every passive joint under 10 rad/s
# (`scripts/gripper_bench.py`). Equally UNCALIBRATED against the real grip force. Tabletop
# only: the velocity/reaching grippers tasks keep GRIPPER.
GRIPPER_TABLETOP = _pal_impedance_params(8.0, 0.5, 20, 1e-4)

##
# MJCF & Assets
##


def _load_spec(xml_path: Path) -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(xml_path))
  return spec


def get_kangaroo_spec() -> mujoco.MjSpec:
  return _load_spec(KANGAROO_XML)


def get_kangaroo_lower_body_spec() -> mujoco.MjSpec:
  return _load_spec(KANGAROO_LOWER_BODY_XML)


def get_kangaroo_hands_spec() -> mujoco.MjSpec:
  return _load_spec(KANGAROO_HANDS_XML)


def get_kangaroo_grippers_spec() -> mujoco.MjSpec:
  return _load_spec(KANGAROO_GRIPPERS_XML)


# Collision proxies for the links a tabletop task needs and the shipped MJCF does not give
# usable ones for: the torso, and the upper arm / forearm / elbow of both arms. Each is the
# link-frame bounding box of that link's visual meshes -- an oriented box, since MuJoCo
# boxes rotate with their body. Fitted with `scripts/fit_link_colliders.py` against
# kangaroo_grippers.xml, 2026-09-22; re-run it if PAL ships new meshes. Left and right come
# out identical in their own link frames (the meshes are shared and the mirroring lives in
# the body frames), so one entry covers both.
#
# Boxes rather than capsules, and the difference is not cosmetic. A capsule that *contains*
# a link has to put a full hemisphere beyond each end of the mesh; on the upper arm that is
# 74 mm deep and points straight into the chest, and the shoulder then reports contact with
# the torso in 46% of sampled arm configurations. Measured over the same configurations, the
# arm's real meshes come within 1 mm of the torso -- these are genuine pairs, so they cannot
# simply be excluded -- but the capsule fires roughly 10 cm early. Boxes have no overhang and
# are 2.4x smaller by volume.
#
# `arm_*_4_collision` is a *replacement*, not an addition. PAL's is a hand-authored capsule
# running from x=-0.30 to x=0 of radius 0.07, but the arm-4 mesh only spans x in
# [-0.155, 0.043]: the extra 0.145 m reaches down the forearm and swallows arm 5-7, which is
# why MANIPULATION_COLLISION drops it rather than fixing it.
_TABLETOP_BOXES = {
  "pelvis_2_collision": (
    "pelvis_2_link",
    (0.0000, 0.0000, 0.2328),
    (0.1344, 0.1609, 0.2406),
  ),
  "arm_{side}_2_collision": (
    "arm_{side}_2_link",
    (0.0655, -0.0100, -0.0432),
    (0.1090, 0.0535, 0.0497),
  ),
  "arm_{side}_3_collision": (
    "arm_{side}_3_link",
    (-0.0061, -0.0100, -0.1144),
    (0.0497, 0.0535, 0.0993),
  ),
  "arm_{side}_4_collision": (
    "arm_{side}_4_link",
    (-0.0558, 0.0100, -0.0447),
    (0.0992, 0.0535, 0.0511),
  ),
}

#: Link pairs whose proxies overlap without the links being anywhere near each other. Both
#: are a link and its grandparent across one joint, where neighbouring boxes necessarily
#: interpenetrate at the joint; MuJoCo already excludes parent/child pairs itself, and these
#: are the next ones out.
#:
#: Measured, not assumed. Over 4000 random arm configurations inside the soft limits, the
#: closest the real meshes ever come is 82.8 mm for arm 2 vs arm 4 and 71.7 mm for arm 4 vs
#: arm 6 -- they cannot touch. The same test keeps other suspicious-looking pairs *in*: arm 5
#: vs arm 7 closes to 9.2 mm and the torso to within 1 mm of every arm link, so those
#: contacts are real and stay detected.
_TABLETOP_CONTACT_EXCLUDES: tuple[tuple[str, str], ...] = tuple(
  (f"arm_{side}_{a}_link", f"arm_{side}_{b}_link")
  for side in ("left", "right")
  for a, b in ((2, 4), (4, 6))
)


#: Bodies whose weight the simulated plant carries for the actuator, rather than the
#: actuator carrying it: the torso, both arms and both grippers. Everything from `pelvis_1`
#: outwards, and nothing below it -- the legs keep their real weight, so foot contact and
#: the leg four-bar behave exactly as they did.
#:
#: PAL's impedance gains (`_pal_impedance_params`) are the ones their deployed controller
#: runs, and against raw gravity they are nowhere near enough to hold an arm: at the tabletop
#: start pose `arm_left_1_joint` needs 16.19 Nm and has a stiffness of 40, which is 0.405 rad
#: of droop, and `arm_left_4_joint` needs 10.30 Nm at a stiffness of 20, which is 0.515 rad.
#: A real 7-DoF arm running those gains does not sag half a radian, so the plant those gains
#: were tuned against is a gravity-compensated one. **That inference is not verified against
#: PAL's controller** -- nothing in this repo documents a feedforward gravity term -- and it
#: is the thing that would overturn this choice; see
#: ../docs/decisions/0001-gravity-compensation-for-tabletop-manipulation.md.
_TABLETOP_GRAVCOMP_BODIES = (
  "pelvis_1_link",
  "pelvis_2_link",
  "arm_left_",
  "arm_right_",
  "gripper_left_",
  "gripper_right_",
)


#: The jaw four-bar's passive joints: everything in a gripper but the actuated
#: `gripper_*_finger_joint`. They follow it only through the MJCF's <equality> block.
_GRIPPER_PASSIVE_JOINT_REGEX = (
  r"gripper_(left|right)_(inner_finger_(left|right)|fingertip_(left|right)"
  r"|outer_finger_right)_joint"
)

#: Armature on those joints for tabletop work (decision 0002, S6). The XML's 1e-4 is "the
#: smallest value that holds" at rest, but under a full-range jaw the fingertip joints ring
#: to ~2000 rad/s and the world diverges -- they are the joints every NaN dump names first.
#: 5e-3 slows that mode enough that no passive joint exceeds ~8 rad/s under random jaw
#: actions, a full-stroke step or a shaken grasp, with no divergence in 2.56 M CPU steps,
#: while the open gap (79.27 mm) and the pad-gap curve (within 0.3 mm) are unchanged.
#: Numerical, like the 1e-4 it replaces -- not a claim about the real linkage. Softer
#: equality constraints, the other obvious lever, make every one of those numbers worse.
_TABLETOP_GRIPPER_PASSIVE_ARMATURE = 5e-3


def get_kangaroo_grippers_tabletop_spec(gravcomp: bool = True) -> mujoco.MjSpec:
  """The grippers model with a collision set a tabletop task can trust.

  The shipped MJCF gives the grippers variant collision geometry for the feet, the legs,
  arm 5-7 and the gripper, and `MANIPULATION_COLLISION` narrows that to the feet, arm 5-7
  and the gripper. Nothing above the elbow collides, so the upper arm and the torso sweep
  through the table. This adds them, in code, leaving PAL's XML untouched (see CLAUDE.md,
  *External Dependencies*): three boxes per arm and one for the torso, all sized from the
  link meshes, plus the four contact excludes they make necessary.

  It raises the armature on the jaw four-bar's passive joints to
  `_TABLETOP_GRIPPER_PASSIVE_ARMATURE` (decision 0002).

  It also switches on gravity compensation for the torso and the arms -- see
  `_TABLETOP_GRAVCOMP_BODIES`, which is a decision, not a detail. `gravcomp=False` is there
  to measure against, not to train with: without it the arm droops half a radian.
  """
  spec = _load_spec(KANGAROO_GRIPPERS_XML)
  collision_class = spec.find_default("collision")

  for side in ("left", "right"):
    for template, (body, pos, half) in _TABLETOP_BOXES.items():
      geom_name = template.format(side=side)
      existing = next((g for g in spec.geoms if g.name == geom_name), None)
      if existing is not None:
        geom = existing
      else:
        # The `collision` default class carries group 3, the orange debug material and
        # contype/conaffinity 1, exactly as the geoms already in the XML get them.
        geom = spec.body(body.format(side=side)).add_geom(
          name=geom_name, default=collision_class
        )
      geom.type = mujoco.mjtGeom.mjGEOM_BOX
      geom.pos = pos
      geom.size = half
      geom.quat = (1.0, 0.0, 0.0, 0.0)
      # `arm_*_4_collision` ships as a fromto capsule, and MuJoCo rejects a geom that has
      # both a fromto and a pos. NaN is how a spec marks fromto unset.
      geom.fromto = (math.nan,) * 6
      # The links carry explicit <inertial> elements, so geom density never reaches the
      # body mass; set it to zero anyway so that stays true if an inertial is ever dropped.
      geom.density = 0.0

  for joint in spec.joints:
    if re.fullmatch(_GRIPPER_PASSIVE_JOINT_REGEX, joint.name):
      joint.armature = _TABLETOP_GRIPPER_PASSIVE_ARMATURE

  if gravcomp:
    for body in spec.bodies:
      if body.name.startswith(_TABLETOP_GRAVCOMP_BODIES):
        body.gravcomp = 1.0

  for body_a, body_b in _TABLETOP_CONTACT_EXCLUDES:
    spec.add_exclude(name=f"{body_a}__{body_b}", bodyname1=body_a, bodyname2=body_b)
  return spec


##
# Actuator Configs
##

# Legs
KANGAROO_LEG_ACTUATORS = (
  BuiltinPositionActuatorCfg(
    target_names_expr=("leg_.*_1_joint",),
    **_calc_leg_params(100.0, 80.0, 0.01, None, None),
  ),
  BuiltinPositionActuatorCfg(
    target_names_expr=("leg_.*_2_joint",),
    **_calc_leg_params(100.0, 230.0, 0.01, None, None),
  ),
  BuiltinPositionActuatorCfg(
    target_names_expr=("leg_.*_3_joint",),
    **_calc_leg_params(100.0, 139.0, 0.01, None, None),
  ),
  BuiltinPositionActuatorCfg(
    target_names_expr=("leg_.*_4_joint",),
    **_calc_leg_params(30.0, 140.0, 0.01, None, None),
  ),
  BuiltinPositionActuatorCfg(
    target_names_expr=("leg_.*_5_joint",),
    **_calc_leg_params(30.0, 82.0, 0.01, None, None),
  ),
  BuiltinPositionActuatorCfg(
    target_names_expr=("leg_.*_length_joint",),
    **_calc_leg_params(1600.0, 1100.0, 0.01, None, None),
  ),
)

# Arms & Torso
# Kept apart from S_PLUS so the pelvis can be tuned on its own; under the SysID gains both
# pelvis joints currently carry the S_PLUS values.
KANGAROO_PELVIS_1_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("pelvis_1_joint",),
  **PELVIS_1,
)
KANGAROO_PELVIS_2_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("pelvis_2_joint",),
  **PELVIS_2,
)
KANGAROO_S_PLUS_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "arm_.*_1_joint",
    "arm_.*_2_joint",
  ),
  **S_PLUS,
)
KANGAROO_S_MINUS_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(r"arm_.*_(?![1267]_joint)\d+_joint",),
  **S_MINUS,
)
KANGAROO_XS_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(r"arm_.*_(?![12345]_joint)\d+_joint",),
  **XS,
)

# Drives the jaw. Only the outer-left finger joint is actuated on the real gripper; the other
# five finger joints mimic it 1:1 (fingertips at -1), which the MJCF approximates with its
# <equality><connect>/<joint polycoef> block.
KANGAROO_GRIPPER_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(r"gripper_.*_finger_joint",),
  **GRIPPER,
)

KANGAROO_GRIPPER_TABLETOP_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(r"gripper_.*_finger_joint",),
  **GRIPPER_TABLETOP,
)

COMMON_ACTUATORS = KANGAROO_LEG_ACTUATORS + (
  KANGAROO_S_PLUS_ACTUATOR_CFG,
  KANGAROO_PELVIS_1_ACTUATOR_CFG,
  KANGAROO_PELVIS_2_ACTUATOR_CFG,
  KANGAROO_S_MINUS_ACTUATOR_CFG,
)

KANGAROO_LOWER_BODY_ACTUATORS = KANGAROO_LEG_ACTUATORS + (
  KANGAROO_PELVIS_1_ACTUATOR_CFG,
  KANGAROO_PELVIS_2_ACTUATOR_CFG,
)

##
# Initial State
##

#: Left arm 1-7 for the tabletop pose. `arm_left_4_joint` is pinned at pi/2 -- elbow square
#: -- and the remaining six solve for "open jaw directly above the working area, axis
#: straight down"; see MANIPULATION_INIT_STATE.
_MANIPULATION_LEFT_ARM = (-0.319, 1.468, 1.855, 1.571, 0.014, 1.894, -0.891)

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.9),
  joint_pos={
    "leg_left_1_joint": -0.012,
    "leg_right_1_joint": 0.012,
    "leg_.*_2_joint": 0.054,
    "leg_left_3_joint": 0.04,
    "leg_right_3_joint": -0.04,
    "leg_.*_length_joint": 0.6,
    "leg_.*_4_joint": -0.053,
    "leg_.*_5_joint": 0.0,
    "leg_.*_femur_joint": 0.9,
    "leg_.*_knee_joint": 1.8,
    "arm_left_1_joint": 0.24,
    "arm_right_1_joint": -0.24,
    "arm_.*_2_joint": 1.32,
    "arm_left_3_joint": 1.57,
    "arm_right_3_joint": -1.57,
    "arm_.*_4_joint": 0.8,
    "pelvis_1_joint": 0.0,
    "pelvis_2_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

# Tabletop starting pose. INIT_STATE is the locomotion pose: it names arm 1-4 only, leaves
# arm 5-7 and the jaw at zero, and puts the left grasp point at (0.178, 0.326, 0.558) --
# 20 cm below the table top and behind its near edge, arm hanging by the side. A
# manipulation episode that starts there spends its first second getting the hand over the
# table.
#
# This holds the *left* arm with the elbow square and the open jaw directly above the
# working area, pointing straight down, so the policy starts where the work is. Legs,
# pelvis and the right arm keep their locomotion values; the arm entries are written per
# side rather than as `arm_.*_N_joint` patterns so that nothing here overlaps a pattern in
# INIT_STATE. Solved by `scripts/nominal_pose.py --elbow 1.5708`; see
# ../docs/reports/pick-place-v0/.
MANIPULATION_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=INIT_STATE.pos,
  joint_pos={
    **{
      k: v
      for k, v in INIT_STATE.joint_pos.items()
      if k.startswith("leg_") or k.startswith("pelvis_")
    },
    "arm_right_1_joint": -0.24,
    "arm_right_2_joint": 1.32,
    "arm_right_3_joint": -1.57,
    "arm_right_4_joint": 0.8,
    "arm_left_1_joint": _MANIPULATION_LEFT_ARM[0],
    "arm_left_2_joint": _MANIPULATION_LEFT_ARM[1],
    "arm_left_3_joint": _MANIPULATION_LEFT_ARM[2],
    "arm_left_4_joint": _MANIPULATION_LEFT_ARM[3],
    "arm_left_5_joint": _MANIPULATION_LEFT_ARM[4],
    "arm_left_6_joint": _MANIPULATION_LEFT_ARM[5],
    "arm_left_7_joint": _MANIPULATION_LEFT_ARM[6],
  },
  joint_vel={".*": 0.0},
)

##
# Collision Configs
##

_FOOT_REGEX = ".*_foot.*_collision"

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_FOOT_REGEX,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=1,
  conaffinity=1,
  condim={_FOOT_REGEX: 3, ".*_collision": 1},
  priority={_FOOT_REGEX: 1, ".*_collision": 0},
  friction={_FOOT_REGEX: (0.6,)},
)

# Manipulation: feet plus the hand and forearm, so the gripper can actually hold something and
# the arm cannot sweep through a table. The gripper/arm colliders are PAL's own collision
# meshes -- its URDF gives <collision> to the 4 gripper link types and to arm 5/6/7, each
# referencing the same STL as its <visual>; the MJCF conversion had dropped them.
_PAD_REGEX = r"gripper_.*_(fingertip|inner_finger)_(left|right)_collision"
_HAND_REGEX = r"(gripper_.*_collision|arm_.*_[567]_collision)"

# arm_*_4_collision is deliberately NOT included: it is a hand-authored 0.3 m capsule of
# radius 0.07 that encloses arm 5/6/7, so with those links now colliding it penetrates its own
# children by ~65 mm. It remains in FULL_COLLISION for the 4-DoF `kangaroo` variant, where it
# is the only arm collider and has no children to swallow.
MANIPULATION_COLLISION = CollisionCfg(
  geom_names_expr=(_FOOT_REGEX, _HAND_REGEX),
  contype=1,
  conaffinity=1,
  # condim=4 on the pads is what decides whether a smooth cylinder rolls in the jaw: with
  # condim=3 there is no torsional friction and the mug spins about the closing axis no
  # matter how high the sliding coefficient is.
  condim={_PAD_REGEX: 4, _FOOT_REGEX: 3, ".*": 3},
  # Pads outrank the object so their friction wins the pairwise combination.
  priority={_PAD_REGEX: 2, _FOOT_REGEX: 1, ".*": 0},
  # 0.9 is PAL's own mu1/mu2 for every finger link (gripper.urdf.xacro <gazebo> block).
  # Torsional/rolling terms are sim tuning, set by the grasp-hold test.
  friction={_PAD_REGEX: (0.9, 0.02, 0.0001), _FOOT_REGEX: (0.6,)},
  # A stiff pad contact (time constant 4 ms rather than the 10 ms default). Measured with
  # check_grasp_hold: a 0.25 kg mug slides 2.8 mm through the jaw over 6 s at 0.01, against
  # 0.84 mm at 0.004. The slide is gravity-driven and real -- it drops to 0.3 mm with gravity
  # off -- so this is load-bearing for a task whose insertion clearance is 8 mm.
  solref={_PAD_REGEX: (0.004, 1.0)},
  solimp={_PAD_REGEX: (0.95, 0.99, 0.001, 0.5, 2.0)},
)

# The manipulation set plus the links `get_kangaroo_grippers_tabletop_spec` gives usable
# colliders: the torso, and the upper arm, forearm and elbow of both arms. Everything else --
# pad condim, priority, friction, the stiff pad contact -- is inherited unchanged, so the
# only physics difference against MANIPULATION_COLLISION is which links exist to be hit.
# Active robot colliders go from 36 to 50.
#
# It costs workspace, and the cost is measured, not estimated: of the 1200 2 cm cells of the
# table top, the left jaw can take a cube from above on 445 under MANIPULATION_COLLISION and
# on 353 under this one, a 21% loss (../docs/reports/pick-place-v0/s1-report.md). Most of it
# is the torso, and it is not an artifact -- the arm's real meshes come within 1 mm of the
# torso, so it was passing straight through the chest before.
TABLETOP_COLLISION = dataclasses.replace(
  MANIPULATION_COLLISION,
  geom_names_expr=(
    _FOOT_REGEX,
    _HAND_REGEX,
    r"arm_.*_[234]_collision",
    "pelvis_2_collision",
  ),
)

##
# Articulation Configs
##

KANGAROO_ARTICULATION = EntityArticulationInfoCfg(
  actuators=COMMON_ACTUATORS, soft_joint_pos_limit_factor=0.9
)
KANGAROO_LOWER_BODY_ARTICULATION = EntityArticulationInfoCfg(
  actuators=KANGAROO_LOWER_BODY_ACTUATORS,
  soft_joint_pos_limit_factor=0.9,
)
KANGAROO_HANDS_ARTICULATION = EntityArticulationInfoCfg(
  actuators=COMMON_ACTUATORS, soft_joint_pos_limit_factor=0.9
)
KANGAROO_GRIPPERS_ARTICULATION = EntityArticulationInfoCfg(
  actuators=COMMON_ACTUATORS
  + (KANGAROO_XS_ACTUATOR_CFG, KANGAROO_GRIPPER_ACTUATOR_CFG),
  soft_joint_pos_limit_factor=0.9,
)
# Same as KANGAROO_GRIPPERS_ARTICULATION but for the jaw gain (decision 0002).
KANGAROO_GRIPPERS_TABLETOP_ARTICULATION = EntityArticulationInfoCfg(
  actuators=COMMON_ACTUATORS
  + (KANGAROO_XS_ACTUATOR_CFG, KANGAROO_GRIPPER_TABLETOP_ACTUATOR_CFG),
  soft_joint_pos_limit_factor=0.9,
)


_ROBOT_CONFIGS = {
  "kangaroo": (get_kangaroo_spec, KANGAROO_ARTICULATION, FULL_COLLISION),
  "lower_body": (
    get_kangaroo_lower_body_spec,
    KANGAROO_LOWER_BODY_ARTICULATION,
    FULL_COLLISION,
  ),
  "hands": (
    get_kangaroo_hands_spec,
    KANGAROO_HANDS_ARTICULATION,
    FULL_COLLISION,
  ),
  "grippers": (
    get_kangaroo_grippers_spec,
    KANGAROO_GRIPPERS_ARTICULATION,
    MANIPULATION_COLLISION,
  ),
  "grippers_tabletop": (
    get_kangaroo_grippers_tabletop_spec,
    KANGAROO_GRIPPERS_TABLETOP_ARTICULATION,
    TABLETOP_COLLISION,
  ),
}


def _make_robot_cfg(
  variant: str, init_state: EntityCfg.InitialStateCfg = INIT_STATE
) -> EntityCfg:
  spec_fn, articulation, collision = _ROBOT_CONFIGS[variant]
  return EntityCfg(
    init_state=init_state,
    collisions=(collision,),
    spec_fn=spec_fn,
    articulation=articulation,
  )


def get_kangaroo_robot_cfg() -> EntityCfg:
  return _make_robot_cfg("kangaroo")


def get_kangaroo_lower_body_robot_cfg() -> EntityCfg:
  lower_body_cfg = _make_robot_cfg("lower_body")
  # This is needed for the new feet
  lower_body_cfg.init_state.pos = (0.0, 0.0, 0.945)
  return lower_body_cfg


def get_kangaroo_hands_robot_cfg() -> EntityCfg:
  return _make_robot_cfg("hands")


def get_kangaroo_grippers_robot_cfg() -> EntityCfg:
  return _make_robot_cfg("grippers")


def get_kangaroo_grippers_tabletop_robot_cfg(gravcomp: bool = True) -> EntityCfg:
  """The grippers variant for tabletop manipulation.

  Full arm and torso colliders, an arm-up start pose, and a gravity-compensated torso and
  arms. Separate from `get_kangaroo_grippers_robot_cfg` on purpose: that one is what
  `Mjlab-Velocity-*-Pal-Kangaroo-Grippers` and `Mjlab-Reaching-Flat-Pal-Kangaroo-Grippers`
  instantiate, and none of the three changes belongs under those tasks.

  `gravcomp=False` reproduces the plant before S4, for measuring against.
  """
  cfg = _make_robot_cfg("grippers_tabletop", MANIPULATION_INIT_STATE)
  if not gravcomp:
    cfg = dataclasses.replace(
      cfg, spec_fn=lambda: get_kangaroo_grippers_tabletop_spec(gravcomp=False)
    )
  return cfg


_EXCLUDED_JOINTS = {
  "leg_left_knee_joint",
  "leg_right_knee_joint",
  "leg_left_femur_joint",
  "leg_right_femur_joint",
}


# The derived 0.25 * effort / stiffness lands at 0.31 rad for the arms under PAL's gains,
# which is far too coarse for manipulation. These are set from the reach a single control
# step should cover, not from the actuator sizing.
_ACTION_SCALE_OVERRIDES: dict[str, float] = {
  "arm_.*_1_joint": 0.10,
  "arm_.*_2_joint": 0.10,
  r"arm_.*_(?![1267]_joint)\d+_joint": 0.10,  # S_MINUS: arm 3, 4, 5
  r"arm_.*_(?![12345]_joint)\d+_joint": 0.05,  # XS: wrist 6, 7
  r"gripper_.*_finger_joint": 0.50,  # jaw: coarse on purpose, it is used as a latch
}


def _build_action_scales(
  articulation: EntityArticulationInfoCfg, exclude: set = frozenset()
) -> tuple[dict, tuple]:
  """Build action scale dict and actuator names from articulation config."""
  scales, names = {}, []
  for a in articulation.actuators:
    e = (
      a.effort_limit
      if isinstance(a.effort_limit, dict)
      else {n: a.effort_limit for n in a.target_names_expr}
    )
    s = (
      a.stiffness
      if isinstance(a.stiffness, dict)
      else {n: a.stiffness for n in a.target_names_expr}
    )
    for n in a.target_names_expr:
      if n in e and n in s and s[n] and n not in exclude:
        scales[n] = _ACTION_SCALE_OVERRIDES.get(n, 0.25 * e[n] / s[n])
        names.append(n)
  return scales, tuple(names)


KANGAROO_ACTION_SCALE, KANGAROO_ACTUATOR_NAMES = _build_action_scales(
  KANGAROO_ARTICULATION, _EXCLUDED_JOINTS
)
KANGAROO_LOWER_BODY_ACTION_SCALE, KANGAROO_LOWER_BODY_ACTUATOR_NAMES = (
  _build_action_scales(KANGAROO_LOWER_BODY_ARTICULATION, _EXCLUDED_JOINTS)
)
KANGAROO_HANDS_ACTION_SCALE, KANGAROO_HANDS_ACTUATOR_NAMES = _build_action_scales(
  KANGAROO_HANDS_ARTICULATION
)
KANGAROO_GRIPPERS_ACTION_SCALE, KANGAROO_GRIPPERS_ACTUATOR_NAMES = _build_action_scales(
  KANGAROO_GRIPPERS_ARTICULATION
)

if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(get_kangaroo_robot_cfg())
  viewer.launch(robot.spec.compile())
