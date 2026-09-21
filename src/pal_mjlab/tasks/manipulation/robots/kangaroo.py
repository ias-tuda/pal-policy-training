"""Kangaroo, described for the manipulation framework.

This is the *only* file in the manipulation package that names a Kangaroo body part.
Everything else -- tasks, rewards, observations -- goes through
:class:`ManipulationRobotCfg`.
"""

from __future__ import annotations

import dataclasses

from mjlab.entity import EntityCfg

from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import (
  get_kangaroo_grippers_robot_cfg,
)
from pal_mjlab.tasks.manipulation.robot_cfg import GripperCfg, ManipulationRobotCfg

#: Pelvis joints, which no manipulation policy commands.
_PELVIS_JOINTS = ("pelvis_1_joint", "pelvis_2_joint")

#: Stiffness/damping the pelvis actuators get here, in place of PAL's 50/20 N.m/rad and
#: 2.0 N.m.s/rad. See ``_rigid_torso_robot_cfg``. Damping is roughly critical for the
#: torso+arms inertia the pelvis carries (~2 kg.m^2): 2*sqrt(2000*2) ~ 126.
_PELVIS_STIFFNESS = 2000.0
_PELVIS_DAMPING = 120.0


def _rigid_torso_robot_cfg() -> EntityCfg:
  """The gripper Kangaroo with a torso that holds still.

  PAL's own impedance gains (K 50 / 20) are what the arms will meet on hardware, so the
  arm actuators keep them -- but the pelvis is a different case. It is not in the action
  space (see ``_ACTION_JOINTS``): on hardware the torso is held by PAL's balance
  controller, which this task does not model, and in sim it is left to a position actuator
  commanded at the default pose. At K=50 that is soft enough to matter: with both arms
  driven to saturation, ``pelvis_1_joint`` settles 0.35 rad (20 deg) off its default, so
  the whole workspace leans with the arm load and the policy learns against a torso that
  moves in a way the real one, held by the balance controller, will not.

  At K=2000 the same rollout holds it to 0.001 rad (0.06 deg), with no overshoot. The
  holding torque needed is ~17 N.m, well inside the 50 N.m clamp, so the joint is stiff
  rather than saturated and the effort limit is untouched.

  Local to manipulation: ``kangaroo_constants`` is shared with the velocity and reaching
  tasks, where the pelvis *is* policy-driven and PAL's real gains are the point.
  """
  cfg = get_kangaroo_grippers_robot_cfg()
  articulation = cfg.articulation
  assert articulation is not None
  actuators = tuple(
    dataclasses.replace(a, stiffness=_PELVIS_STIFFNESS, damping=_PELVIS_DAMPING)
    if set(a.target_names_expr) & set(_PELVIS_JOINTS)
    else a
    for a in articulation.actuators
  )
  if actuators == articulation.actuators:
    raise ValueError(f"no actuator drives {_PELVIS_JOINTS}")
  return dataclasses.replace(
    cfg, articulation=dataclasses.replace(articulation, actuators=actuators)
  )


#: Jaw travel, measured on the shipped PAL meshes: the pads are 79.3 mm apart at q=0 and
#: closed at the joint's upper limit.
_JAW_OPEN = 0.0
_JAW_CLOSED = 0.95


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
  )


#: The action space: both arms' 7 DoF plus both jaws. The legs are deliberately excluded --
#: on hardware they are held by PAL's own balance controller, and `pal_policy_inferer`
#: builds its action vector from the ONNX `joint_names`, so anything listed here must be
#: something the deployer is actually allowed to drive.
_ACTION_JOINTS = (
  r"arm_.*_[1-7]_joint",
  r"gripper_.*_finger_joint",
)

KANGAROO_GRIPPERS = ManipulationRobotCfg(
  entity_cfg_fn=_rigid_torso_robot_cfg,
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
  # The arms have no frictionloss in the model at all (see ManipulationRobotCfg), so this
  # adds stiction they currently lack rather than perturbing an existing value. 1.0 N.m is
  # 2% of the 50 N.m arm effort limit. The jaw gets three orders of magnitude less: its
  # armature is 1e-4 and the four-bar is stiff, so anything larger stalls the latch.
  # The arms have no joint-level viscous damping in the model (it all lives in the actuator
  # kd), so this adds a modest real term rather than perturbing one: 0.05 is ~5% of the
  # arms' kd of 1.0. The jaw does have a default near 0.03, so its range brackets that.
  joint_damping_range={
    r"arm_.*_[1-7]_joint": (0.0, 0.05),
    r"gripper_.*_finger_joint": (0.02, 0.04),
  },
  joint_friction_range={
    r"arm_.*_[1-7]_joint": (0.0, 1.0),
    r"gripper_.*_finger_joint": (0.0, 0.02),
  },
  viewer_body="pelvis_2_link",
  # Tabletop manipulation only: balance is PAL's controller's job, and a Kangaroo that
  # topples over drowns the manipulation reward signal in fall recovery.
  fixed_base=True,
)
