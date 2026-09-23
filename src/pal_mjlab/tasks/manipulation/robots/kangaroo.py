"""Kangaroo, described for the manipulation framework.

This is the *only* file in the manipulation package that names a Kangaroo body part.
Everything else -- tasks, rewards, observations -- goes through
:class:`ManipulationRobotCfg`.
"""

from __future__ import annotations

import dataclasses

from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import (
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
# Only this variant: Lift and Stack keep KANGAROO_GRIPPERS' scales.
KANGAROO_GRIPPERS_LEFT_ARM = dataclasses.replace(
  KANGAROO_GRIPPERS,
  action_joint_regex=(r"arm_left_[1-7]_joint", r"gripper_left_finger_joint"),
  action_scale={
    r"arm_left_1_joint": 0.20,
    r"arm_left_2_joint": 0.55,
    r"arm_left_3_joint": 0.40,
    r"arm_left_4_joint": 0.55,
    r"arm_left_5_joint": 0.55,
    r"arm_left_6_joint": 0.50,
    r"arm_left_7_joint": 1.05,
    r"gripper_left_finger_joint": _JAW_CLOSED - _JAW_OPEN,
  },
  action_clip={r"gripper_left_finger_joint": (_JAW_OPEN, _JAW_CLOSED)},
)
