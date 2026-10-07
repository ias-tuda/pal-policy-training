"""Kangaroo, described for the manipulation framework.

This is the *only* file in the manipulation package that names a Kangaroo body part.
Everything else -- tasks, rewards, observations -- goes through
:class:`ManipulationRobotCfg`.
"""

from __future__ import annotations

import dataclasses
import functools

from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import (
  DEFAULT_TABLETOP_GAINS,
  MANIPULATION_INIT_STATE,
  TABLETOP_GAINS,
  get_kangaroo_grippers_tabletop_robot_cfg,
)
from pal_mjlab.tasks.manipulation.robot_cfg import (
  GripperCfg,
  KeepoutBoxCfg,
  ManipulationRobotCfg,
  ReachCfg,
)

#: Jaw travel. The pads are 79.3 mm apart at q=0 (measured on the shipped PAL meshes).
#: "Closed" is the commanded close, not the joint's 0.95 rad upper limit: the fingertip
#: joints' own -0.803 rad limits stop the linkage at ~0.82 rad, so any target above that
#: leaves the actuator pushing the four-bar against its limits for as long as it is held.
#: 0.78 keeps the target inside what the linkage can reach (decision 0002, S6). It is also
#: the action scale and the clip's upper bound below, so action +1 is exactly this target.
_JAW_OPEN = 0.0
_JAW_CLOSED = 0.78


def _gripper(side: str) -> GripperCfg:
  return GripperCfg(
    name=side,
    tcp_site=f"tcp_{side}",
    pad_sites=(
      f"gripper_{side}_fingertip_left_site",
      f"gripper_{side}_fingertip_right_site",
    ),
    pad_geom_regex=rf"gripper_{side}_(fingertip|inner_finger)_(left|right)_collision",
    finger_joint=f"gripper_{side}_finger_joint",
    open_pos=_JAW_OPEN,
    closed_pos=_JAW_CLOSED,
    hand_body=f"arm_{side}_7_link",
    # The fingertip pads alone: the inner-finger geoms sit ~2 cm deeper and a 5 cm cube
    # held at the pads never reaches them (measured 2026-09-23, S5.0).
    grasp_geom_regex=rf"gripper_{side}_fingertip_(left|right)_collision",
  )


#: The action space: both arms' 7 DoF plus both jaws. The legs are deliberately excluded --
#: on hardware they are held by PAL's own balance controller, and `pal_policy_inferer`
#: builds its action vector from the ONNX `joint_names`, so anything listed here must be
#: something the deployer is actually allowed to drive.
_ACTION_JOINTS = (
  r"arm_.*_[1-7]_joint",
  r"gripper_.*_finger_joint",
)

# `get_kangaroo_grippers_tabletop_robot_cfg`, not `get_kangaroo_grippers_robot_cfg`: the
# tabletop variant is the one whose torso, upper arm and elbow have collision geometry and
# whose start pose holds the left jaw over the table. The locomotion/reaching tasks keep the
# other one -- see kangaroo_constants.py.
KANGAROO_GRIPPERS = ManipulationRobotCfg(
  entity_cfg_fn=get_kangaroo_grippers_tabletop_robot_cfg,
  grippers=(_gripper("left"), _gripper("right")),
  action_joint_regex=_ACTION_JOINTS,
  # Arm and wrist follow kangaroo_constants._ACTION_SCALE_OVERRIDES. The jaw is
  # deliberately lower than the 0.50 used there: driving the finger joint that hard excites
  # the gripper four-bar, and halving the excitation cut peak |qvel| from 129 to 51 rad/s
  # in a random-action rollout. 0.2 rad per step still covers the 0.95 rad jaw range in
  # five steps, which is plenty for a latch.
  action_scale={
    r"arm_.*_[1-5]_joint": 0.10,
    r"arm_.*_[67]_joint": 0.05,
    r"gripper_.*_finger_joint": 0.20,
  },
  viewer_body="pelvis_2_link",
  # Tabletop manipulation only: balance is PAL's controller's job, and a Kangaroo that
  # topples over drowns the manipulation reward signal in fall recovery.
  fixed_base=True,
  # S5.1 item 8 (Claudius's 82c4450). The box is `torso_collision`'s bounding box in
  # `pelvis_2_link`'s frame, measured on the compiled model (his centre is 3.7 mm higher).
  # `arm_.*_1_link` is left out: its origin is the shoulder, inside the box by construction.
  # The 0.05 m margin must stay below the closest any watched link origin sits to the box at
  # the start pose; `tests/test_trunk_keepout_and_reach.py` checks that it does.
  trunk_keepout=KeepoutBoxCfg(
    reference_body="pelvis_2_link",
    body_name_regex=(r"arm_.*_[2-7]_link",),
    center=(0.0, 0.0, 0.2368),
    half_extents=(0.1344, 0.1609, 0.2367),
    margin=0.05,
  ),
  # Horizontal distance from `base_link` past which the cube is out of reach and the episode
  # fails. His 0.75 m: the cube spawns ~0.4-0.6 m out, and at 0.75 m the arm is near full
  # extension -- past it the episode is not worth running.
  reach=ReachCfg(reference_body="base_link", max_distance=0.75),
  # S5.1 item 4 (Claudius's 602cd9d): PAL's URDF joint speeds (`arm/4dof`, `arm/7dof`
  # xacro), 1.95 rad/s on the shoulder pair and 2.35 from the humeral roll out, i.e.
  # 0.039 / 0.047 rad per 20 ms control step. The jaw is not limited (decision 0002 was
  # chosen over his jaw slew, S5.1 item 5).
  joint_velocity_limits={
    r"arm_.*_[12]_joint": 1.95,
    r"arm_.*_[3-7]_joint": 2.35,
  },
)


# One-arm variant. Dropping the right arm from the action regex drops it from the observation
# vector too -- `make_tabletop_env_cfg` builds `joint_pos`/`joint_vel` from the same
# `action_joint_regex` -- so a single-arm task neither drives nor sees the arm it is not
# using. Action dim 8: seven joints and the jaw.
#
# What holds the right arm once the policy stops commanding it is
# `mdp.hold_nominal_targets` plus the gravity compensation from S4; before those, an arm
# taken out of the action space was an arm commanded to zero.
#
# Action scales (S5.0, 2026-09-23). The action is an *absolute* target, `default + scale *
# action`, so a scale is the half-width of the box the policy can command -- not a step
# size. KANGAROO_GRIPPERS' scales cap the jaw at 22 % of its range (it never closes on the
# cube) and the arm at ~12 cm of grasp-point travel, which is what C1/C2 died on. These are
# sized from the task instead: `scripts/scripted_pick_place.py derive` solves IK for
# pre-grasp, grasp, lift, carry, place and retreat at the spawn box's corners and the target,
# and each scale is 1.5x the largest excursion from the start pose, rounded up to 0.05 rad.
# The jaw scale spans open to `_JAW_CLOSED`, so action +1 is fully closed; the clip stops
# action < 0 from commanding a target below the open limit, which the simulator does not clamp
# (mjlab position actuators are `ctrllimited=False`) and which would press the four-bar
# into its stop at up to 19 N.m.
#
# Re-derived by S5.1 item 3 (2026-09-23) for the item-2 home pose, item-8 colliders and
# item-1 gains, with the floor raised to the SysID training scale 0.25 * effort / K (~0.25
# rad): it suffices on no joint. The wrist (arm 6) needs 2.35 because the home pose has the
# gripper looking forward and every top-down grasp turns it ~1.5 rad; arm 1-5 and 7 need
# 0.60-0.75. S5.0's scales were 0.20/0.55/0.40/0.55/0.55/0.50/1.05 from the S1 pose.
# `artifacts/s5.1-chain/derive.json`.
#
# Only this variant: Lift and Stack keep KANGAROO_GRIPPERS' scales.
KANGAROO_GRIPPERS_LEFT_ARM = dataclasses.replace(
  KANGAROO_GRIPPERS,
  action_joint_regex=(r"arm_left_[1-7]_joint", r"gripper_left_finger_joint"),
  action_scale={
    r"arm_left_1_joint": 0.75,
    r"arm_left_2_joint": 0.60,
    r"arm_left_3_joint": 0.65,
    r"arm_left_4_joint": 0.70,
    r"arm_left_5_joint": 0.75,
    r"arm_left_6_joint": 2.35,
    r"arm_left_7_joint": 0.65,
    r"gripper_left_finger_joint": _JAW_CLOSED - _JAW_OPEN,
  },
  action_clip={r"gripper_left_finger_joint": (_JAW_OPEN, _JAW_CLOSED)},
)


# The same arm with the action scales plan S7's P2 derived for the wider spawn boxes W1 and
# W2 at every cube yaw (S5.0 rule over both boxes' corners and centres at -45/0/+45 deg, never
# below today's; S7 report section 6, `artifacts/s7/p2/`). Approved by Siwei 2026-09-25 17:49
# for S7-2's W runs. What +/-45 deg needs is mostly the forearm and wrist rolls, arm 5 and 7.
# Derived on the model before fda9492; the scales are half-widths about the home pose, so the
# sign flip of arm 5/7 leaves them valid, and the corners test re-checks them on the new model.
KANGAROO_GRIPPERS_LEFT_ARM_WIDE = dataclasses.replace(
  KANGAROO_GRIPPERS_LEFT_ARM,
  action_scale={
    r"arm_left_1_joint": 0.90,
    r"arm_left_2_joint": 0.85,
    r"arm_left_3_joint": 0.65,
    r"arm_left_4_joint": 0.80,
    r"arm_left_5_joint": 1.55,
    r"arm_left_6_joint": 2.40,
    r"arm_left_7_joint": 1.20,
    r"gripper_left_finger_joint": _JAW_CLOSED - _JAW_OPEN,
  },
)


# The same arm with the action scales plan S7-8 Part B (2026-09-30) derived for the wide areas, by the
# S5.0 rule (1.5 x the largest excursion of any waypoint, rounded up to 0.05 rad, never below the W2
# scales, capped at half the joint's range) over the contact-free plans of the area's spawn x target
# pairs at cube yaw -45 / 0 / +45 deg, on the 0.815 m table with the 5 cm carry
# (`results/S7-8/wc/`).
#
# `WC` is Claudius's whole `-Wide` area (x 0.40-0.60, y -0.04-0.26). NOT TRAINED: only 35 % of its
# (spawn, target >= 10 cm apart, yaw) triples have a plan against the plan's 98 %, and the scales
# below are driven by elbow-flipped solutions at edge triples (arm 1 and arm 4 at their caps).
KANGAROO_GRIPPERS_LEFT_ARM_WC = dataclasses.replace(
  KANGAROO_GRIPPERS_LEFT_ARM_WIDE,
  action_scale={
    r"arm_left_1_joint": 2.35,
    r"arm_left_2_joint": 0.85,
    r"arm_left_3_joint": 0.65,
    r"arm_left_4_joint": 1.79,
    r"arm_left_5_joint": 1.55,
    r"arm_left_6_joint": 2.40,
    r"arm_left_7_joint": 1.20,
    r"gripper_left_finger_joint": _JAW_CLOSED - _JAW_OPEN,
  },
)
# `WCK` is the part of it that plans robustly: x 0.40-0.52, y 0.16-0.26 (12 x 10 cm), where the whole 4 cm
# pair sweep plans on the first pass. Its scales are the largest of the rectangles y 0.12-0.26 and
# y 0.16-0.26 (both 100 % plannable), so the boundary rows are inside them: arm 1 and arm 4 widen to
# 1.10, the rest are W2's.
KANGAROO_GRIPPERS_LEFT_ARM_WCK = dataclasses.replace(
  KANGAROO_GRIPPERS_LEFT_ARM_WIDE,
  action_scale={
    r"arm_left_1_joint": 1.10,
    r"arm_left_2_joint": 0.85,
    r"arm_left_3_joint": 0.65,
    r"arm_left_4_joint": 1.10,
    r"arm_left_5_joint": 1.55,
    r"arm_left_6_joint": 2.40,
    r"arm_left_7_joint": 1.20,
    r"gripper_left_finger_joint": _JAW_CLOSED - _JAW_OPEN,
  },
)


# Plan §13's L-H (`pick-place-insert-plan.md` §13.1, step 0 item 1, 2026-10-07): the side grasp of the 56 mm socket
# block -- the hand horizontal, pointing forward (+/- 60 deg), the pads 40-50 mm over the 0.837 m table -- and its hold at
# H (0.45, 0.00, table + 0.10). The S5.0 rule (1.5 x the largest excursion from home over the contact-free plans of the
# block on x 0.40-0.55, y 0.08-0.26 at yaw 0/15/30/45 deg -- pregrasp, grasp, lift, H -- rounded up to 0.05 rad, never
# below WC's, capped at half the joint's range): arm 2, 3, 5 and 7 widen (the horizontal hand turns the upper arm out and
# the forearm over), arm 3 / 5 / 6 hit the cap (`results/S13-insert/step0/reach_x3.json`).
KANGAROO_GRIPPERS_LEFT_ARM_SIDE = dataclasses.replace(
  KANGAROO_GRIPPERS_LEFT_ARM_WIDE,
  action_scale={
    r"arm_left_1_joint": 2.35,
    r"arm_left_2_joint": 1.25,
    r"arm_left_3_joint": 2.60,
    r"arm_left_4_joint": 1.79,
    r"arm_left_5_joint": 2.60,
    r"arm_left_6_joint": 2.40,
    r"arm_left_7_joint": 2.50,
    r"gripper_left_finger_joint": _JAW_CLOSED - _JAW_OPEN,
  },
)

# -- Plan S11: two arms -----------------------------------------------------------------------
#
# The right arm is the left arm's mirror image through the robot's sagittal plane (y -> -y): with
# q_right = ARM_MIRROR_SIGNS x q_left on arm 1-7 the right TCP is the left TCP with y negated, and
# the right hand's frame the mirrored left frame turned 180 deg about the approach axis, which a
# parallel jaw cannot tell apart (plan S11 *Measured today*, 2026-10-02: <= 2.4e-5 m over 200 random
# poses; `tests/test_pick_place_s11.py` re-measures it on the tabletop spec). The jaw keeps its sign.
ARM_MIRROR_SIGNS = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0)

#: The right arm's home by name (`ARM_HOME`): ``default`` is Claudius's pose, every run before S11;
#: ``mirror`` is mirror(left home) -- up to 0.076 rad (arm 5) off it -- which makes the whole scene
#: symmetric about y = 0 but for the torso camera (plan S11, Open item 4, as proposed).
ARM_HOMES = ("default", "mirror")


def mirrored_right_home() -> dict[str, float]:
  """The right arm's joint positions that mirror the left arm's home (`ARM_MIRROR_SIGNS`)."""
  left = {
    k: v
    for k, v in MANIPULATION_INIT_STATE.joint_pos.items()
    if k.startswith("arm_left_")
  }
  return {
    f"arm_right_{i}_joint": ARM_MIRROR_SIGNS[i - 1] * left[f"arm_left_{i}_joint"]
    for i in range(1, 8)
  }


def _with_right_home(entity_cfg_fn, home: str):
  """``entity_cfg_fn`` with the right arm's start pose -- the action origin and the parked arm's
  hold -- set to ``home`` (`ARM_HOMES`)."""
  if home not in ARM_HOMES:
    raise ValueError(f"unknown right-arm home {home!r}; have {ARM_HOMES}")
  if home == "default":
    return entity_cfg_fn

  def build():
    cfg = entity_cfg_fn()
    joint_pos = {**cfg.init_state.joint_pos, **mirrored_right_home()}
    return dataclasses.replace(
      cfg, init_state=dataclasses.replace(cfg.init_state, joint_pos=joint_pos)
    )

  return build


#: Plan S11's JOINT: both arms and both jaws in the action space, 16-D, each arm at the W2 action
#: scales (the right's are the left's: a scale is a half-width about the home, the same for the
#: mirror image), both jaws clipped open-to-closed.
KANGAROO_GRIPPERS_BOTH_ARMS_WIDE = dataclasses.replace(
  KANGAROO_GRIPPERS,
  action_joint_regex=(
    r"arm_left_[1-7]_joint",
    r"gripper_left_finger_joint",
    r"arm_right_[1-7]_joint",
    r"gripper_right_finger_joint",
  ),
  action_scale={
    **KANGAROO_GRIPPERS_LEFT_ARM_WIDE.action_scale,
    **{
      k.replace("left", "right"): v
      for k, v in KANGAROO_GRIPPERS_LEFT_ARM_WIDE.action_scale.items()
    },
  },
  action_clip={
    r"gripper_left_finger_joint": (_JAW_OPEN, _JAW_CLOSED),
    r"gripper_right_finger_joint": (_JAW_OPEN, _JAW_CLOSED),
  },
)


def kangaroo_grippers_both_arms(
  gains: str = DEFAULT_TABLETOP_GAINS,
  arm_effort_scale: float | None = None,
  right_home: str = "mirror",
) -> ManipulationRobotCfg:
  """`KANGAROO_GRIPPERS_BOTH_ARMS_WIDE` on gain code ``gains``, with plan S7-4's T on both arms
  (``arm_effort_scale``) and the right arm's home ``right_home`` (plan S11: ``mirror``)."""
  if gains not in TABLETOP_GAINS:
    raise ValueError(
      f"unknown tabletop gain code {gains!r}; have {tuple(TABLETOP_GAINS)}"
    )
  return dataclasses.replace(
    KANGAROO_GRIPPERS_BOTH_ARMS_WIDE,
    entity_cfg_fn=_with_right_home(
      functools.partial(
        get_kangaroo_grippers_tabletop_robot_cfg,
        gains=gains,
        left_arm_effort_scale=arm_effort_scale,
        right_arm_effort_scale=arm_effort_scale,
      ),
      right_home,
    ),
  )


def kangaroo_grippers_left_arm(
  gains: str = DEFAULT_TABLETOP_GAINS,
  base: ManipulationRobotCfg = KANGAROO_GRIPPERS_LEFT_ARM,
  arm_effort_scale: float | None = None,
  right_home: str = "default",
) -> ManipulationRobotCfg:
  """``base`` on the plant of plan-S8 gain code ``gains``.

  ``base`` is `KANGAROO_GRIPPERS_LEFT_ARM` or its wide-scale twin for plan S7-2's W boxes
  (plan S7-3 runs both on G20). Only the entity changes: the pelvis and arm K/D, and whether
  gravity is compensated (`kangaroo_constants.TABLETOP_GAINS`). The action space, its scales,
  clip and slew limit are ``base``'s for every code, so two runs that differ in the code differ
  in the plant alone. G0 builds exactly ``base``'s robot.

  ``arm_effort_scale`` lowers the driven (left) arm's effort limit to that fraction of its
  effort -- plan S7-4's module T, 0.7: 35 N.m on arm 1-2, 17.5 on arm 3-7. None keeps it.

  ``right_home`` is where the parked right arm is held (`ARM_HOMES`; plan S11's ``ARM_HOME``):
  ``default`` is every run before S11, ``mirror`` the left home's mirror image.
  """
  if gains not in TABLETOP_GAINS:
    raise ValueError(
      f"unknown tabletop gain code {gains!r}; have {tuple(TABLETOP_GAINS)}"
    )
  # Every gain code rebuilds the tabletop articulation; a base on any other entity would have
  # its robot silently swapped for it.
  if base.entity_cfg_fn is not get_kangaroo_grippers_tabletop_robot_cfg:
    raise ValueError(
      "gain codes apply to the tabletop robot only; base is built by "
      f"{getattr(base.entity_cfg_fn, '__name__', base.entity_cfg_fn)!r}"
    )
  return dataclasses.replace(
    base,
    entity_cfg_fn=_with_right_home(
      functools.partial(
        get_kangaroo_grippers_tabletop_robot_cfg,
        gains=gains,
        left_arm_effort_scale=arm_effort_scale,
      ),
      right_home,
    ),
  )
