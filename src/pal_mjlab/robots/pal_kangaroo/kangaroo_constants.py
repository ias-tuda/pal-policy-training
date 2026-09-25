"""Pal Robotics KANGAROO constants."""

import dataclasses
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


#: Effort limit of every joint whose gravity compensation is routed through its actuator,
#: by regex -- the same numbers the actuator cfgs below carry (S_PLUS, S_MINUS, XS,
#: PELVIS_*), restated because the spec is edited before the actuators exist.
_TABLETOP_ACTGRAVCOMP_EFFORT = {
  r"pelvis_[12]_joint": 50.0,
  r"arm_(left|right)_[12]_joint": 50.0,
  r"arm_(left|right)_[3-7]_joint": 25.0,
}


def _route_gravcomp_through_actuators(spec: mujoco.MjSpec) -> None:
  """Make gravity compensation count against the pelvis/arm effort limits (S5.1 item 6).

  With ``actgravcomp`` a joint's gravity compensation is added to ``qfrc_actuator``
  instead of ``qfrc_passive``, like the feedforward a real torque-limited servo would carry
  (Claudius's cee261a). On its own that does *not* bound it: MuJoCo clamps the sum only by
  the joint's ``actuatorfrcrange``, and mjlab's ``BuiltinPositionActuatorCfg`` puts the
  effort limit on the actuator's ``forcerange`` instead (`mjlab/utils/spec.py`), which
  clamps the PD term alone. So the joint range is set here too: the PD term stays within
  its own limit, and PD + gravity together within the same limit.
  """
  matched = 0
  for joint in spec.joints:
    effort = next(
      (
        e
        for rx, e in _TABLETOP_ACTGRAVCOMP_EFFORT.items()
        if re.fullmatch(rx, joint.name)
      ),
      None,
    )
    if effort is None:
      continue
    joint.actgravcomp = True
    joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_TRUE
    joint.actfrcrange = (-effort, effort)
    matched += 1
  if matched != 16:
    raise ValueError(
      f"expected 16 pelvis/arm joints for actgravcomp, matched {matched}"
    )


def get_kangaroo_grippers_tabletop_spec(gravcomp: bool = True) -> mujoco.MjSpec:
  """The grippers model with a collision set a tabletop task can trust.

  `MANIPULATION_COLLISION` narrows the grippers model's colliders to the feet, arm 5-7 and
  the gripper, so nothing above the elbow collides and the upper arm and torso sweep
  through the table. `TABLETOP_COLLISION` switches the rest on. Their geometry lives in
  ``kangaroo_grippers.xml`` since S5.1 item 8 (Claudius's 602cd9d / 57af0d3): PAL's collision
  hulls for the base, pelvis_1, pelvis_2 and torso, and the visual meshes as colliders for
  arm 1-4 -- an edit to PAL's XML, recorded as an exception to D4. (Before, this function
  built oriented boxes for pelvis_2 and arm 2-4 in code, fitted by
  `scripts/fit_link_colliders.py`; the meshes replace them.) It adds the four arm contact
  excludes below.

  It raises the armature on the jaw four-bar's passive joints to
  `_TABLETOP_GRIPPER_PASSIVE_ARMATURE` (decision 0002).

  It also switches on gravity compensation for the torso, the arms and the grippers -- see
  `_TABLETOP_GRAVCOMP_BODIES`, which is a decision, not a detail -- and routes it through
  the pelvis/arm actuators so it counts against their effort limits
  (`_route_gravcomp_through_actuators`). `gravcomp=False` is there
  to measure against, not to train with: without it the arm droops half a radian.
  """
  spec = _load_spec(KANGAROO_GRIPPERS_XML)

  for joint in spec.joints:
    if re.fullmatch(_GRIPPER_PASSIVE_JOINT_REGEX, joint.name):
      joint.armature = _TABLETOP_GRIPPER_PASSIVE_ARMATURE

  if gravcomp:
    for body in spec.bodies:
      if body.name.startswith(_TABLETOP_GRAVCOMP_BODIES):
        body.gravcomp = 1.0
    _route_gravcomp_through_actuators(spec)

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

#: The tabletop arm pose: both hands in front of the torso just above the table's near edge,
#: elbows down, grippers looking forward and slightly down at the work area. Claudius's
#: `_ARM_HOME_POSE` (`origin/fix-stack-exploration-collapse` 980526a, kept through 82a6d5d),
#: "solved by damped least-squares IK and picked for manipulability and joint-limit margin";
#: no script for it was committed. Adopted by S5.1 item 2 (2026-09-23), replacing S1's
#: left-arm-only pose (elbow at pi/2, jaw straight down, right arm left at its locomotion
#: pose). It is also the action origin (`use_default_offset=True`).
#:
#: Arm 5 and 7 are signed for their axes about +z, PAL's URDF convention (plan S7-2 prep 1,
#: Claudius's fda9492 "fix kang arm kinematics!"): `kangaroo_grippers.xml` had them about -z,
#: so a policy's joint 5/7 commands were sign-flipped against PAL's controllers. The pose is
#: physically the same one; only the sign of those four angles changed with the axes.
_MANIPULATION_ARM_HOME_POSE = {
  "arm_left_1_joint": 0.102,
  "arm_left_2_joint": 1.452,
  "arm_left_3_joint": 1.777,
  "arm_left_4_joint": 1.670,
  "arm_left_5_joint": 0.458,
  "arm_left_6_joint": 0.404,
  "arm_left_7_joint": -0.009,
  "arm_right_1_joint": -0.139,
  "arm_right_2_joint": 1.480,
  "arm_right_3_joint": -1.717,
  "arm_right_4_joint": 1.691,
  "arm_right_5_joint": -0.382,
  "arm_right_6_joint": 0.368,
  "arm_right_7_joint": 0.005,
}

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
# arm 5-7 and the jaw at zero, and hangs the arms by the robot's sides, below the table top.
# This keeps its legs and pelvis and puts both arms at `_MANIPULATION_ARM_HOME_POSE`. The arm
# entries are exact names, not `arm_.*_N_joint` patterns, so nothing here overlaps a pattern
# in INIT_STATE (which would win over a later exact name).
MANIPULATION_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=INIT_STATE.pos,
  joint_pos={
    **{
      k: v
      for k, v in INIT_STATE.joint_pos.items()
      if k.startswith("leg_") or k.startswith("pelvis_")
    },
    **_MANIPULATION_ARM_HOME_POSE,
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

# The manipulation set plus the trunk and upper-arm colliders `kangaroo_grippers.xml` carries
# since S5.1 item 8: PAL's hulls for the base, pelvis_1, pelvis_2 and torso, and arm 1-4's
# visual meshes. Everything else -- pad condim, priority, friction, the stiff pad contact -- is
# inherited unchanged, so the only physics difference against MANIPULATION_COLLISION is which
# links exist to be hit. Active robot colliders go from 36 to 49.
#
# These names have to be listed here to collide at all: `CollisionCfg` zeroes contype and
# conaffinity on every geom it does not match. On Claudius's branch, where the meshes come
# from, they are *not* listed (his robot keeps MANIPULATION_COLLISION), so there they are
# inert and only the keep-out penalty keeps the arms off the torso. Here they are live:
# over 2000 random arm poses nothing collides by construction (the most frequent pair, arm 4
# vs torso, in 3.5 % of poses), and the start pose has no robot-robot contact.
#
# It costs workspace. Under S1.4's boxes, of the 1200 2 cm cells of the table top the left
# jaw could take a cube from above on 445 under MANIPULATION_COLLISION and 353 under the boxes
# (../docs/reports/pick-place-v0/s1-report.md); the meshes have not been re-mapped.
TABLETOP_COLLISION = dataclasses.replace(
  MANIPULATION_COLLISION,
  geom_names_expr=(
    _FOOT_REGEX,
    _HAND_REGEX,
    r"arm_.*_[1234]_collision",
    r"pelvis_[12]_collision",
    "torso_collision",
    "base_link_collision",
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
