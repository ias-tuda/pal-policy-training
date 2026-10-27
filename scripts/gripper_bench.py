"""Bench the gripper four-bar on its own: does it ring, and does it still grip?

S6 of the pick-and-place plan. Each Kangaroo jaw is a four-bar: one position-driven joint
(`gripper_*_finger_joint`) and five passive joints that follow it only through the
`<equality>` block of `kangaroo_grippers.xml`. The passive links weigh 16-34 g with
~5e-6 kg.m^2 of inertia, and when the jaw is driven hard they ring until the solver
diverges. This script measures that in isolation, on CPU MuJoCo, against the real
pick-and-place scene (same robot overlay, same solver settings, base welded), so a model
change can be screened in seconds rather than in a 2048-env training job.

Stimuli, all on the *left* jaw (the one the one-arm task drives); everything else is held
at its nominal target, as `hold_nominal_targets` does in the task:

* ``step``    open -> closed -> open with the extreme actions (+1 / -1), no object. Gives
  the stroke time.
* ``random``  a new uniform jaw action in [-1, 1] every 20 ms control step, many worlds.
* ``shake``   a 50 g cube clamped in the jaw, then random left-arm actions at 50 Hz.
  Gives grip force, slip and whether the cube is dropped.
* ``kinematics``  a slow open -> closed ramp; pad gap against jaw angle, the curve every
  model change has to reproduce.

Plan S8 (the 6 cm ball) adds two more, for whichever object ``$SHAPE`` / ``--shape`` picks
(``cube`` | ``ball``); neither is part of ``--stimulus all``:

* ``touch``   the object supported between the open pads (pad-site midpoint at its centre
  height) while the jaw target ramps slowly from open to closed: the jaw angle at which each
  pad first reports a force, the angle at which both do ("touch angle"), then the settled hold
  angle and per-pad normal forces once the support is released.
* ``ballheight``   the object clamped with the pad-site midpoint ``--offsets`` mm off its
  centre along the hand's approach axis (positive = pads toward the wrist, i.e. above a
  top-down grasp's centre), then the ``shake`` -- arm-target slew-limited to the stack task's
  speeds with ``--shake-slew stack``. Slip, drops and pad force against grasp height.

The jaw target is formed exactly as mjlab's `JointPositionAction` forms it:
``default + scale * action`` with no clip, and the jaw actuator is not ``ctrllimited``, so
at ``--jaw-scale 0.95`` an action of -1 commands -0.95 rad and drives the jaw into its open
stop. That is what a policy does; the bench does not soften it.

Divergence is counted the way CPU MuJoCo reports it -- an ``mjWARN_BADQACC`` warning, after
which the engine resets that world -- plus any non-finite state. The world is then
re-initialised and the stimulus continues, so a divergence costs one event, not the run.
mujoco_warp does *not* reset (memory §5.0); rates measured here are a screening number, and
the training-scale NaN rate still has to be measured on GPU.

Levers (S6 L1-L5) are applied to the `MjSpec` before compile, i.e. in the same place a
permanent change would live (`get_kangaroo_grippers_tabletop_spec`), so the constants
MuJoCo derives at compile time (``dof_invweight0`` and friends, which scale constraint
softness) are the real ones:

    uv run --extra cpu python scripts/gripper_bench.py --stimulus all --name baseline
    uv run --extra cpu python scripts/gripper_bench.py --stimulus all --name L4-solref0.02 \\
        --solref 0.02
    uv run --extra cpu python scripts/gripper_bench.py --report artifacts/s6
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

TASK = "Mjlab-PickPlace-Pal-Kangaroo-Grippers"
SIDE = "left"

#: The five joints of one jaw that nothing drives: they follow the actuated
#: `gripper_*_finger_joint` through two `<connect>` constraints and one `<joint>` equality.
PASSIVE_JOINT_REGEX = (
  r"gripper_(left|right)_(inner_finger_(left|right)|fingertip_(left|right)"
  r"|outer_finger_right)_joint"
)
PAD_GEOM_REGEX = rf"gripper_{SIDE}_(fingertip|inner_finger)_(left|right)_collision"

CONTROL_DT = 0.02  # 50 Hz, the task's decimation 10 x 2 ms
PAD_CLEARANCE = 0.01  # m, pad sites over the table at the scripted pick's grasp pose
JAW_CLOSED = 0.95  # the joint's upper limit, `robots/kangaroo.py` _JAW_CLOSED
MIN_PAD_FORCE = 0.5  # N per side; the plan's upgraded grasp test (ManiSkill min_force)
DROP_SLIP = 0.02  # m of cube motion in the hand frame that counts as a drop
STIMULI = ("step", "random", "shake", "kinematics")
#: Plan S8's object stimuli. Not in ``--stimulus all``, which stays the S6 set.
S8_STIMULI = ("touch", "ballheight")
TOUCH_FORCE = 0.05  # N per pad; "the pad is in contact" for `touch`
#: Seconds the arm settles under gravity before an S8 stimulus places the object.
SETTLE_S = 1.0
#: ``$BENCH_POSE`` (set by ``--pose``): ``nominal`` is the robot's start pose, the hand
#: pointing forward and ~22 deg down, as S6 ran every stimulus; ``topdown`` is the arm at the
#: pose the scripted pick grasps from -- the hand's +z straight down, so the approach axis is
#: vertical and gravity acts along it, as it does in the task (plan S8).
POSE_ENV_VAR = "BENCH_POSE"
#: The selectors the task reads from the environment (`kangaroo/env_cfg.py`); the record
#: keeps them, since a result is only reproducible with the same plant and object.
ENV_SELECTORS = ("GAINS", "SHAPE", "BASELINE", "SPAWN", "DR", "EFFORT", POSE_ENV_VAR)


@dataclasses.dataclass(frozen=True)
class Levers:
  """One S6 configuration. ``None`` leaves the shipped value in place."""

  armature: float | None = None  # L1, the 5 passive joints of each jaw
  inertia_scale: float | None = None  # L2, uniform on the passive bodies' diaginertia
  damping: float | None = None  # L3, the 5 passive joints
  solref: float | None = None  # L4, time constant of the jaw equalities (s)
  solimp_width: float | None = None  # L4, solimp[2] of the jaw equalities
  kp: float | None = None  # L5, jaw actuator stiffness
  kv: float | None = None  # L5, jaw actuator damping
  jaw_armature: float | None = None  # armature of the actuated finger joint itself
  rate_limit: float | None = None  # L6, max jaw-target change per control step (rad)
  # Not in the plan's list: clamp the jaw target to [0, jaw_clip]. The actuator is not
  # ctrllimited and mjlab does not clip, so a full-range action commands -0.95 rad into
  # the open stop and 0.95 rad past the 0.80 rad the passive joints allow.
  jaw_clip: float | None = None


def _slew_arg(text: str) -> float | str:
  return text if text == "stack" else float(text)


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  p.add_argument("--stimulus", choices=(*STIMULI, *S8_STIMULI, "all"), default="all")
  p.add_argument("--name", default="baseline", help="label for this configuration")
  p.add_argument("--jaw-scale", type=float, default=0.95, help="S5.0 full-range jaw")
  p.add_argument("--worlds", type=int, default=64, help="random/shake worlds")
  p.add_argument("--seconds", type=float, default=20.0, help="random/shake duration")
  p.add_argument("--shake-amp", type=float, default=0.3, help="rad, arm target noise")
  p.add_argument(
    "--shake-slew",
    type=_slew_arg,
    default=None,
    help="rate-limit the shake's arm target like the task does: a factor x the URDF "
    "slew (plan S7 FA: 0.5), or 'stack' for the stack task's per-joint speeds (plan "
    "S7-8; arm 1-3 at 1/16, 4-7 at 1/8 of the URDF limit). Default: unlimited, as S6 "
    "ran it",
  )
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--workers", type=int, default=8)
  p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("artifacts/s6"))
  p.add_argument(
    "--pose",
    choices=("nominal", "topdown"),
    default=None,
    help="arm pose the stimuli run in: the robot's start pose (S6's) or the scripted "
    "pick's top-down grasp pose, where the approach axis is vertical. Sets "
    "$BENCH_POSE. Default: as it is, else nominal",
  )
  p.add_argument(
    "--shape",
    default=None,
    help="plan-S8 object (cube | ball); sets $SHAPE for the registry. Default: $SHAPE as "
    "it is, else cube",
  )
  p.add_argument(
    "--offsets",
    type=float,
    nargs="+",
    default=[-20, -15, -10, -5, 0, 5, 10, 15, 20],
    help="ballheight: height (mm) of the pad-site midpoint above the object's centre, "
    "along the hand's approach axis",
  )
  p.add_argument(
    "--touch-offsets",
    type=float,
    nargs="+",
    default=[0.0],
    help="touch: heights (mm) of the pad-site midpoint above the object's centre; the "
    "first is the headline",
  )
  p.add_argument(
    "--touch-yaw",
    type=float,
    default=0.0,
    help="touch: turn the object about the approach axis by this many degrees "
    "(0 = a cube's faces square to the pads)",
  )
  p.add_argument(
    "--touch-seconds",
    type=float,
    default=5.0,
    help="touch: duration of the jaw-target ramp from open to --jaw-scale",
  )
  p.add_argument(
    "--gains",
    default=None,
    help="plan-S8 gain code (G0 | G15 | G20 | G20s) the task's robot is built on; sets "
    "$GAINS for the registry. Default: $GAINS as it is, else G0",
  )
  for f in dataclasses.fields(Levers):
    p.add_argument(f"--{f.name.replace('_', '-')}", type=float, default=None)
  p.add_argument(
    "--report",
    type=pathlib.Path,
    default=None,
    help="print the comparison table for every result JSON in this directory and exit",
  )
  return p.parse_args()


##
# Model.
##


def _passive(name: str) -> bool:
  return re.fullmatch(PASSIVE_JOINT_REGEX, name) is not None


def apply_spec_levers(spec: mujoco.MjSpec, lv: Levers) -> mujoco.MjSpec:
  """L1-L4 on the robot's spec, before compile. Both jaws, so the model stays symmetric."""
  for joint in spec.joints:
    if not _passive(joint.name):
      continue
    if lv.armature is not None:
      joint.armature = lv.armature
    if lv.damping is not None:
      # A spec joint's damping is a 3-vector in MuJoCo 3.11 (linear term first).
      damping = np.zeros_like(np.asarray(joint.damping))
      damping[0] = lv.damping
      joint.damping = damping
    if lv.inertia_scale is not None:
      body = joint.parent
      body.inertia = np.asarray(body.inertia) * lv.inertia_scale
  for eq in spec.equalities:
    # The connects name bodies and the joint equality names joints; all start `gripper_`.
    # The two leg closed chains do not, and are left alone.
    if not eq.name1.startswith("gripper_"):
      continue
    if lv.solref is not None:
      eq.solref = (lv.solref, 1.0)
    if lv.solimp_width is not None:
      solimp = list(eq.solimp)
      solimp[2] = lv.solimp_width
      eq.solimp = solimp
  return spec


def build_model(lv: Levers) -> mujoco.MjModel:
  """The pick-place scene with ``lv`` applied, solver settings as the task runs them."""
  import pal_mjlab.tasks  # noqa: F401  # populates the task registry
  from mjlab.scene import Scene
  from mjlab.tasks.registry import load_env_cfg

  env_cfg = load_env_cfg(TASK, play=True)
  env_cfg.scene.num_envs = 1
  robot_cfg = env_cfg.scene.entities["robot"]
  base_spec_fn = robot_cfg.spec_fn
  env_cfg.scene.entities["robot"] = dataclasses.replace(
    robot_cfg, spec_fn=lambda: apply_spec_levers(base_spec_fn(), lv)
  )
  model = Scene(env_cfg.scene, "cpu").compile()
  env_cfg.sim.mujoco.apply(model)
  # L5 on the compiled model: a builtin position actuator is force = kp*(ctrl - q) - kv*dq,
  # stored as gainprm[0] = kp, biasprm[1:3] = (-kp, -kv). No compile-time constant
  # depends on it.
  for a in range(model.nu):
    if re.search(
      r"gripper_.*_finger_joint", _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
    ):
      if lv.kp is not None:
        model.actuator_gainprm[a, 0] = lv.kp
        model.actuator_biasprm[a, 1] = -lv.kp
      if lv.kv is not None:
        model.actuator_biasprm[a, 2] = -lv.kv
      if lv.jaw_armature is not None:
        # The actuator cfg writes this joint's armature after spec_fn runs, so it cannot
        # go in apply_spec_levers; set it here and recompute the compile-time constants
        # (dof_invweight0 etc.) that depend on it.
        model.dof_armature[model.jnt_dofadr[model.actuator_trnid[a, 0]]] = (
          lv.jaw_armature
        )
  if lv.jaw_armature is not None:
    mujoco.mj_setConst(model, mujoco.MjData(model))
  return model


def _name(model, obj, i) -> str:
  return (mujoco.mj_id2name(model, obj, i) or "").split("/", 1)[-1]


@dataclasses.dataclass
class Ids:
  """Addresses the bench reads every step, resolved once per model."""

  passive_dof: np.ndarray
  passive_names: list[str]
  jaw_dof: int
  jaw_qpos: int
  jaw_act: int
  arm_act: np.ndarray
  default_ctrl: np.ndarray
  connects: list[tuple[int, int, np.ndarray, np.ndarray]]
  joint_eq: list[tuple[int, int]]
  pad_sites: tuple[int, int]
  tcp_site: int
  hand_body: int
  cube_body: int
  cube_qpos: int
  cube_dof: int
  cube_geoms: set
  pad_geoms_left: set
  pad_geoms_right: set
  table_geoms: list[int]


def resolve(model: mujoco.MjModel) -> Ids:
  J, B, A, S, G, E = (
    mujoco.mjtObj.mjOBJ_JOINT,
    mujoco.mjtObj.mjOBJ_BODY,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    mujoco.mjtObj.mjOBJ_SITE,
    mujoco.mjtObj.mjOBJ_GEOM,
    mujoco.mjtObj.mjOBJ_EQUALITY,
  )
  joints = [_name(model, J, j) for j in range(model.njnt)]
  # `startswith`, not `"_left_" in n`: gripper_right_inner_finger_left_joint contains it.
  passive = [
    j for j, n in enumerate(joints) if _passive(n) and n.startswith(f"gripper_{SIDE}_")
  ]
  jaw = joints.index(f"gripper_{SIDE}_finger_joint")
  acts = [_name(model, A, a) for a in range(model.nu)]
  jaw_act = acts.index(f"gripper_{SIDE}_finger_joint")
  arm_act = np.array(
    [acts.index(f"arm_{SIDE}_{k}_joint") for k in range(1, 8)], dtype=int
  )

  def body(n):
    return mujoco.mj_name2id(model, B, n)

  connects, joint_eq = [], []
  for e in range(model.neq):
    if model.eq_type[e] == mujoco.mjtEq.mjEQ_CONNECT:
      if not _name(model, E, e).startswith(f"gripper_{SIDE}_"):
        continue
      connects.append(
        (
          model.eq_obj1id[e],
          model.eq_obj2id[e],
          model.eq_data[e, 0:3].copy(),
          model.eq_data[e, 3:6].copy(),
        )
      )
    elif model.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT:
      j1, j2 = model.eq_obj1id[e], model.eq_obj2id[e]
      if f"gripper_{SIDE}_" in joints[j1]:
        joint_eq.append((model.jnt_qposadr[j1], model.jnt_qposadr[j2]))

  cube = body("object/object")
  cube_jnt = next(j for j in range(model.njnt) if model.jnt_bodyid[j] == cube)
  geoms = [_name(model, G, g) for g in range(model.ngeom)]
  pads = [g for g, n in enumerate(geoms) if re.fullmatch(PAD_GEOM_REGEX, n)]
  table = [
    g for g in range(model.ngeom) if _name(model, B, model.geom_bodyid[g]) == "table"
  ]
  if not table:
    table = [g for g, n in enumerate(geoms) if n.startswith("table")]

  # Hold-nominal targets: every actuator at its joint's configured start value, as
  # `mdp.hold_nominal_targets` writes them after each reset.
  data = _nominal(model)
  default_ctrl = np.array(
    [data.qpos[model.jnt_qposadr[model.actuator_trnid[a, 0]]] for a in range(model.nu)]
  )
  return Ids(
    passive_dof=np.array([model.jnt_dofadr[j] for j in passive], dtype=int),
    passive_names=[joints[j] for j in passive],
    jaw_dof=int(model.jnt_dofadr[jaw]),
    jaw_qpos=int(model.jnt_qposadr[jaw]),
    jaw_act=jaw_act,
    arm_act=arm_act,
    default_ctrl=default_ctrl,
    connects=connects,
    joint_eq=joint_eq,
    pad_sites=(
      mujoco.mj_name2id(model, S, f"robot/gripper_{SIDE}_fingertip_left_site"),
      mujoco.mj_name2id(model, S, f"robot/gripper_{SIDE}_fingertip_right_site"),
    ),
    tcp_site=mujoco.mj_name2id(model, S, f"robot/tcp_{SIDE}"),
    hand_body=body(f"robot/arm_{SIDE}_7_link"),
    cube_body=cube,
    cube_qpos=int(model.jnt_qposadr[cube_jnt]),
    cube_dof=int(model.jnt_dofadr[cube_jnt]),
    cube_geoms={g for g in range(model.ngeom) if model.geom_bodyid[g] == cube},
    pad_geoms_left={g for g in pads if geoms[g].endswith("_left_collision")},
    pad_geoms_right={g for g in pads if geoms[g].endswith("_right_collision")},
    table_geoms=table,
  )


_TOPDOWN_CACHE: dict[int, np.ndarray] = {}


def _set_topdown_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
  """Write the arm pose the scripted pick grasps the spawn point from into ``data``.

  Damped least squares on the left arm for the grasp point (``tcp_left``) above `SPAWN_XY`,
  the hand's +z straight down and its +x along whichever world axis the start pose's is
  nearest -- the IK of `scripted_pick_place.Kin` (soft joint limits, a pull toward the start
  pose), copied so the bench does not import a script that changes under it. The height is a
  grasp at the table: the pad sites `PAD_CLEARANCE` above it.
  """
  from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import SPAWN_XY
  from pal_mjlab.tasks.manipulation.scene import TABLE_TOP_Z
  from scene_model import joint_qadr, site_id, soft_limits

  names = [f"arm_{SIDE}_{k}_joint" for k in range(1, 8)]
  qadr = joint_qadr(model, names)
  if id(model) not in _TOPDOWN_CACHE:
    work = mujoco.MjData(model)
    work.qpos[:] = data.qpos
    lo, hi = soft_limits(model, names)
    tcp = site_id(model, f"robot/tcp_{SIDE}")
    hand = mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_BODY, f"robot/arm_{SIDE}_7_link"
    )
    pads = [
      site_id(model, f"robot/gripper_{SIDE}_fingertip_{s}_site")
      for s in ("left", "right")
    ]
    dofs = np.array(
      [
        model.jnt_dofadr[
          mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{n}")
        ]
        for n in names
      ]
    )

    def fk(q):
      work.qpos[qadr] = q
      mujoco.mj_kinematics(model, work)
      mujoco.mj_comPos(model, work)
      return work.site_xpos[tcp].copy(), work.xmat[hand].reshape(3, 3).copy()

    q0 = data.qpos[qadr].copy()
    pos0, rot0 = fk(q0)
    yaw0 = round(math.atan2(rot0[1, 0], rot0[0, 0]) / (math.pi / 2)) * (math.pi / 2)
    x_des = np.array([math.cos(yaw0), math.sin(yaw0), 0.0])
    z_des = np.array([0.0, 0.0, -1.0])
    r_des = np.column_stack([x_des, np.cross(z_des, x_des), z_des])
    reach = max(float((work.site_xpos[p] - pos0) @ rot0[:, 2]) for p in pads)
    goal = np.array([*SPAWN_XY, TABLE_TOP_Z + reach + PAD_CLEARANCE])
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    q = q0.copy()
    for _ in range(600):
      pos, rot = fk(q)
      err = np.concatenate(
        [goal - pos, 0.5 * sum(np.cross(rot[:, i], r_des[:, i]) for i in range(3))]
      )
      if np.linalg.norm(err[:3]) < 1e-3 and np.linalg.norm(err[3:]) < 1e-2:
        break
      mujoco.mj_jacSite(model, work, jacp, jacr, tcp)
      jac = np.vstack([jacp[:, dofs], jacr[:, dofs]])
      pinv = jac.T @ np.linalg.inv(jac @ jac.T + 1e-3 * np.eye(6))
      step = pinv @ err + (np.eye(7) - pinv @ jac) @ (0.2 * (q0 - q))
      q = np.clip(q + np.clip(step, -0.2, 0.2), lo, hi)
    else:
      raise RuntimeError(
        f"the top-down grasp pose has no IK solution: goal {goal}, residual {err}"
      )
    _TOPDOWN_CACHE[id(model)] = q
  data.qpos[qadr] = _TOPDOWN_CACHE[id(model)]
  mujoco.mj_forward(model, data)


def _nominal(model: mujoco.MjModel) -> mujoco.MjData:
  from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import SPAWN_XY
  from pal_mjlab.tasks.manipulation.scene import TABLE_TOP_Z
  from scene_model import nominal_data

  data = nominal_data(model)
  if os.environ.get(POSE_ENV_VAR, "nominal") == "topdown":
    _set_topdown_pose(model, data)
  # Park the cube at the task's spawn point so it sits where it would in an episode and
  # stays out of the jaw's way for the object-free stimuli.
  cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object/object")
  jnt = next(j for j in range(model.njnt) if model.jnt_bodyid[j] == cube)
  adr = model.jnt_qposadr[jnt]
  # Its centre's resting height: a box's half-size, a sphere's radius (plan S8's ball is
  # 3 cm, not 2.5). Every geom of the body has its size in geom_size[:, 0] (sphere) or
  # geom_size[:, :3] (box), and the largest number is the half-height of either.
  rest = max(
    float(model.geom_size[g].max())
    for g in range(model.ngeom)
    if model.geom_bodyid[g] == cube
  )
  data.qpos[adr : adr + 7] = (*SPAWN_XY, TABLE_TOP_Z + rest, 1.0, 0.0, 0.0, 0.0)
  mujoco.mj_forward(model, data)
  return data


##
# Per-step measurement.
##


class World:
  """One simulated world plus the running statistics the bench reports."""

  BADQACC = int(mujoco.mjtWarning.mjWARN_BADQACC)

  def __init__(self, model: mujoco.MjModel, ids: Ids, lv: Levers):
    self.m, self.ids, self.lv = model, ids, lv
    self.d = _nominal(model)
    self.d.ctrl[:] = ids.default_ctrl
    self.samples: list[
      np.ndarray
    ] = []  # |qvel| of the passive joints, per physics step
    self.jaw_speed = 0.0
    self.max_connect = 0.0  # m
    self.max_joint_eq = 0.0  # rad
    self.divergences = 0
    self.steps = 0
    self._warn = 0
    self._jaw_target = ids.default_ctrl[ids.jaw_act]
    self._jaw_command = self._jaw_target

  def snapshot(self):
    return (
      self.d.qpos.copy(),
      self.d.qvel.copy(),
      self.d.act.copy(),
      self.d.ctrl.copy(),
    )

  def restore(self, snap):
    mujoco.mj_resetData(self.m, self.d)
    self.d.qpos[:], self.d.qvel[:], self.d.act[:], self.d.ctrl[:] = snap
    mujoco.mj_forward(self.m, self.d)
    self._warn = self.d.warning[self.BADQACC].number
    self._jaw_target = self.d.ctrl[self.ids.jaw_act]
    self._jaw_command = self._jaw_target

  def set_jaw(self, target: float):
    """Command the jaw, as a policy action held until the next call would.

    The command is clipped here; the rate limit is applied per control step in
    `_advance_jaw`, so a command held over several steps ramps to its target the way
    `RateLimitedJointPositionAction` does. (Until 2026-09-23 the limit was applied here,
    once per call, so a stimulus that set the jaw once and held -- ``step``, ``shake`` --
    left a rate-limited jaw parked one slew step from open. Every "rate limit fails the
    grasp" result before then is that artefact.)
    """
    if self.lv.jaw_clip is not None:
      target = float(np.clip(target, 0.0, self.lv.jaw_clip))
    self._jaw_command = target

  def _advance_jaw(self):
    target = self._jaw_command
    if self.lv.rate_limit is not None:
      step = np.clip(target - self._jaw_target, -self.lv.rate_limit, self.lv.rate_limit)
      target = self._jaw_target + step
    self._jaw_target = target
    self.d.ctrl[self.ids.jaw_act] = target

  def control_step(self, record: bool = True, on_physics=None) -> bool:
    """Ten physics steps at the held ctrl. False if the world diverged.

    A control step that diverges contributes the event and nothing else: its samples and
    violations are rolled back, since the steps just before a blow-up are unbounded by
    definition and would otherwise *be* the peak statistic.
    """
    self._advance_jaw()
    mark = (len(self.samples), self.steps, self.jaw_speed, self.max_connect)
    mark += (self.max_joint_eq,)
    for _ in range(10):
      mujoco.mj_step(self.m, self.d)
      n = self.d.warning[self.BADQACC].number
      if n != self._warn or not np.isfinite(self.d.qvel).all():
        self._warn = n
        if record:
          self.divergences += 1
          del self.samples[mark[0] :]
          _, self.steps, self.jaw_speed, self.max_connect, self.max_joint_eq = mark
        return False
      if record:
        self.steps += 1
        self.samples.append(
          np.abs(self.d.qvel[self.ids.passive_dof]).astype(np.float32)
        )
        self.jaw_speed = max(self.jaw_speed, abs(self.d.qvel[self.ids.jaw_dof]))
        self._violation()
      if on_physics is not None:
        on_physics()
    return True

  def _violation(self):
    xp, xm = self.d.xpos, self.d.xmat
    for b1, b2, a1, a2 in self.ids.connects:
      p1 = xp[b1] + xm[b1].reshape(3, 3) @ a1
      p2 = xp[b2] + xm[b2].reshape(3, 3) @ a2
      self.max_connect = max(self.max_connect, float(np.linalg.norm(p1 - p2)))
    for q1, q2 in self.ids.joint_eq:
      err = (self.d.qpos[q1] - self.m.qpos0[q1]) - (self.d.qpos[q2] - self.m.qpos0[q2])
      self.max_joint_eq = max(self.max_joint_eq, abs(float(err)))

  def pad_gap(self) -> float:
    s1, s2 = self.ids.pad_sites
    return float(np.linalg.norm(self.d.site_xpos[s1] - self.d.site_xpos[s2]))

  def pad_mid_in_hand(self) -> np.ndarray:
    """The midpoint of the two pad sites in the hand frame (x closing, z approach), m."""
    s1, s2 = self.ids.pad_sites
    mid = 0.5 * (self.d.site_xpos[s1] + self.d.site_xpos[s2])
    hb = self.ids.hand_body
    return self.d.xmat[hb].reshape(3, 3).T @ (mid - self.d.xpos[hb])

  def pad_forces(self) -> tuple[float, float]:
    """Normal force (N) of the cube on the left-side and right-side pads."""
    left = right = 0.0
    f = np.zeros(6)
    ids = self.ids
    for i in range(self.d.ncon):
      c = self.d.contact[i]
      pair = {c.geom1, c.geom2}
      if not pair & ids.cube_geoms:
        continue
      mujoco.mj_contactForce(self.m, self.d, i, f)
      if pair & ids.pad_geoms_left:
        left += abs(f[0])
      elif pair & ids.pad_geoms_right:
        right += abs(f[0])
    return left, right

  def object_contacts(self) -> list[dict]:
    """Every contact the object is in: which geom, how deep, the normal force (N), and where
    it is and which way it pushes, in the hand frame (x closing, z approach). The normal points
    from the other geom toward the object."""
    ids, m, d = self.ids, self.m, self.d
    hb = ids.hand_body
    rot, origin = d.xmat[hb].reshape(3, 3), d.xpos[hb]
    f = np.zeros(6)
    out = []
    for i in range(d.ncon):
      c = d.contact[i]
      pair = (int(c.geom1), int(c.geom2))
      mine = [g in ids.cube_geoms for g in pair]
      if not any(mine):
        continue
      other = pair[1] if mine[0] else pair[0]
      mujoco.mj_contactForce(m, d, i, f)
      normal = np.array(c.frame[:3])
      if mine[0]:
        normal = -normal  # frame[:3] points geom1 -> geom2
      side = (
        "left"
        if other in ids.pad_geoms_left
        else "right"
        if other in ids.pad_geoms_right
        else None
      )
      out.append(
        {
          "geom": _name(m, mujoco.mjtObj.mjOBJ_GEOM, other),
          "side": side,
          "dist": float(c.dist),
          "force_N": float(abs(f[0])),
          "pos_hand": (rot.T @ (np.array(c.pos) - origin)).tolist(),
          "normal_hand": (rot.T @ normal).tolist(),
        }
      )
    return out

  def cube_in_hand(self) -> np.ndarray:
    hb = self.ids.hand_body
    rel = self.d.xpos[self.ids.cube_body] - self.d.xpos[hb]
    return self.d.xmat[hb].reshape(3, 3).T @ rel


def settle(w: World, seconds: float = 0.4):
  for _ in range(round(seconds / CONTROL_DT)):
    w.control_step(record=False)


def summarise(worlds: list[World]) -> dict:
  return _unpack([_pack(worlds)])


##
# Stimuli. Each worker builds its own model; a world never crosses a process.
##


def run_step(lv: Levers, args) -> dict:
  model = build_model(lv)
  w = World(model, resolve(model), lv)
  settle(w)
  q = w.ids.jaw_qpos
  strokes = {"close": [], "open": []}
  for _cycle in range(3):
    for phase, action in (("close", 1.0), ("open", -1.0)):
      w.set_jaw(w.ids.default_ctrl[w.ids.jaw_act] + args.jaw_scale * action)
      trace = [float(w.d.qpos[q])]
      for _ in range(round(1.0 / CONTROL_DT)):
        if not w.control_step():
          break
        trace.append(float(w.d.qpos[q]))
      strokes[phase].append(_t95(trace))
  out = summarise([w])
  out["stroke_close_s"] = _worst(strokes["close"])
  out["stroke_open_s"] = _worst(strokes["open"])
  return out


def _t95(trace: list[float]) -> float | None:
  """Time to cover 95 % of the way to where the jaw settles, not to where it was told.

  The jaw does not reach a closed command of 0.95 rad: the passive joints' own limits
  (0.803 rad) stop the linkage first, so "95 % of 0.95" would never fire.
  """
  start, end = trace[0], trace[-1]
  if abs(end - start) < 1e-3:
    return None
  for k, x in enumerate(trace):
    if abs(x - start) >= 0.95 * abs(end - start):
      return k * CONTROL_DT
  return None


def _worst(ts):
  return None if any(t is None for t in ts) else max(ts)


def _random_worker(lv: Levers, args, world_ids: list[int]) -> dict:
  model = build_model(lv)
  ids = resolve(model)
  worlds = []
  n_ctrl = round(args.seconds / CONTROL_DT)
  for wid in world_ids:
    rng = np.random.default_rng([args.seed, wid])
    w = World(model, ids, lv)
    settle(w)
    snap = w.snapshot()
    for _ in range(n_ctrl):
      w.set_jaw(ids.default_ctrl[ids.jaw_act] + args.jaw_scale * rng.uniform(-1, 1))
      if not w.control_step():
        w.restore(snap)
    worlds.append(w)
  return _pack(worlds)


def _shake_worker(lv: Levers, args, world_ids: list[int]) -> dict:
  model = build_model(lv)
  ids = resolve(model)
  # Take the table out of the way: the start pose holds the jaw over it and a random arm
  # would otherwise push the clamped cube into it, which is a different experiment.
  for g in ids.table_geoms:
    model.geom_contype[g] = 0
    model.geom_conaffinity[g] = 0
  cube_weight = model.body_subtreemass[ids.cube_body] * -model.opt.gravity[2]
  n_ctrl = round(args.seconds / CONTROL_DT)
  slew = None if args.shake_slew is None else _arm_slew_per_step(args.shake_slew)
  worlds, grips, slips, drops, forces, tcp_speed = [], [], [], [], [], []
  for wid in world_ids:
    rng = np.random.default_rng([args.seed, 10_000 + wid])
    w = World(model, ids, lv)
    settle(w)
    _place_cube_in_jaw(w)
    # Hold the cube up against gravity while the jaw closes on it, then let go.
    w.d.xfrc_applied[ids.cube_body, 2] = cube_weight
    w.set_jaw(ids.default_ctrl[ids.jaw_act] + args.jaw_scale)
    ok = all(w.control_step() for _ in range(round(0.5 / CONTROL_DT)))
    w.d.xfrc_applied[ids.cube_body, 2] = 0.0
    ok = ok and all(w.control_step() for _ in range(round(0.5 / CONTROL_DT)))
    fl, fr = w.pad_forces()
    grips.append(ok and min(fl, fr) >= MIN_PAD_FORCE)
    forces.append(min(fl, fr))
    ref = w.cube_in_hand()
    max_slip = 0.0
    tcp = _PeakSpeed(w.d, ids.tcp_site, model.opt.timestep)

    arm_default = ids.default_ctrl[ids.arm_act]
    arm_target = arm_default.copy()
    for _ in range(n_ctrl if ok else 0):
      proposal = arm_default + rng.uniform(-args.shake_amp, args.shake_amp, size=7)
      if slew is None:
        arm_target = proposal
      else:
        arm_target = arm_target + np.clip(proposal - arm_target, -slew, slew)
      w.d.ctrl[ids.arm_act] = arm_target
      if not w.control_step(on_physics=tcp):
        break
      max_slip = max(max_slip, float(np.linalg.norm(w.cube_in_hand() - ref)))
    slips.append(max_slip)
    drops.append((not ok) or max_slip > DROP_SLIP)
    tcp_speed.append(tcp.peak)
    worlds.append(w)
  out = _pack(worlds)
  out["shake"] = {
    "grasped": grips,
    "min_pad_force_N": forces,
    "slip_m": slips,
    "dropped": drops,
    "tcp_speed": tcp_speed,
  }
  return out


def _arm_slew_per_step(factor: float | str) -> np.ndarray:
  """The task's arm slew limit in rad per control step, in `Ids.arm_act` order.

  The robot's URDF joint speeds (`KANGAROO_GRIPPERS.joint_velocity_limits`) x the control
  period x ``factor`` -- what `mdp.RateLimitedJointPositionAction` allows a target to move per
  step. Without it a shake jumps the target by up to 2 x ``--shake-amp`` per step, which a
  stiff arm (plan S8) follows at several m/s at the tool.

  ``"stack"`` is plan S7-8's per-joint speeds instead of one factor: arm joints 1-3 at 1/16
  and 4-7 at 1/8 of their URDF limit (`tabletop_env_cfg.stack_arm_rate_limit`, 2.4-5.9 mrad
  per step).
  """
  from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS

  limits = KANGAROO_GRIPPERS.joint_velocity_limits
  if factor == "stack":
    from pal_mjlab.tasks.manipulation.tabletop_env_cfg import stack_arm_rate_limit

    per_step = stack_arm_rate_limit(limits)
    return np.array([per_step[f"arm_.*_{k}_joint"] for k in range(1, 8)])
  return np.array(
    [
      next(v for rx, v in limits.items() if re.fullmatch(rx, f"arm_{SIDE}_{k}_joint"))
      * CONTROL_DT
      * factor
      for k in range(1, 8)
    ]
  )


class _PeakSpeed:
  """Peak speed of a site, by finite difference over physics steps."""

  def __init__(self, data: mujoco.MjData, site: int, dt: float):
    self.data, self.site, self.dt = data, site, dt
    self.prev = data.site_xpos[site].copy()
    self.peak = 0.0

  def __call__(self):
    p = self.data.site_xpos[self.site]
    self.peak = max(self.peak, float(np.linalg.norm(p - self.prev)) / self.dt)
    self.prev = p.copy()


def _place_cube_in_jaw(w: World, height: float = 0.0, yaw: float = 0.0):
  """Put the object between the open pads, faces square to them, at rest.

  ``height`` (m) is how far the pad-site midpoint is *above the object's centre* along the
  hand's approach axis (z, from the wrist toward the fingertips; it points down when the hand
  is top-down, as in a pick). So the object's centre is placed at ``mid + height * z``:
  ``height = -0.015`` puts the pads 15 mm below the centre, deeper toward the wrist for the
  object, which is where the scripted pick puts them for the cube. ``yaw`` (rad) turns the
  object about the approach axis (a cube's faces are square to the pads at 0). A sphere has
  no faces; its orientation does not matter.
  """
  ids, d = w.ids, w.d
  s1, s2 = ids.pad_sites
  mid = 0.5 * (d.site_xpos[s1] + d.site_xpos[s2])
  x = d.site_xpos[s2] - d.site_xpos[s1]
  x /= np.linalg.norm(x)
  z = d.xmat[ids.hand_body].reshape(3, 3)[:, 2]
  z = z - x * (z @ x)
  z /= np.linalg.norm(z)
  rot = np.column_stack([x, np.cross(z, x), z])
  c, sn = math.cos(yaw), math.sin(yaw)
  rot = rot @ np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
  quat = np.zeros(4)
  mujoco.mju_mat2Quat(quat, rot.ravel())
  d.qpos[ids.cube_qpos : ids.cube_qpos + 7] = (*(mid + height * z), *quat)
  d.qvel[ids.cube_dof : ids.cube_dof + 6] = 0.0
  mujoco.mj_forward(w.m, d)


def _pack(worlds: list[World]) -> dict:
  """What a worker sends back: raw samples, so the parent computes one set of stats."""
  return {
    "samples": [np.stack(w.samples) if w.samples else np.zeros((0, 5)) for w in worlds],
    "names": worlds[0].ids.passive_names,
    "steps": [w.steps for w in worlds],
    "div": [w.divergences for w in worlds],
    "jaw": [w.jaw_speed for w in worlds],
    "conn": [w.max_connect for w in worlds],
    "jeq": [w.max_joint_eq for w in worlds],
  }


def _unpack(parts: list[dict]) -> dict:
  steps = sum(sum(p["steps"]) for p in parts)
  samples = np.concatenate([s for p in parts for s in p["samples"] if len(s)])
  names = parts[0]["names"]
  per_joint = {
    n: {
      "peak": float(samples[:, k].max()),
      "p99": float(np.percentile(samples[:, k], 99)),
    }
    for k, n in enumerate(names)
  }
  div = sum(sum(p["div"]) for p in parts)
  return {
    "physics_steps": steps,
    "divergences": div,
    "divergences_per_Msteps": div / max(steps, 1) * 1e6,
    "passive_qvel": per_joint,
    "peak_passive_qvel": max(v["peak"] for v in per_joint.values()),
    "p99_passive_qvel": max(v["p99"] for v in per_joint.values()),
    "peak_jaw_qvel": max(max(p["jaw"]) for p in parts),
    "max_connect_violation_mm": 1e3 * max(max(p["conn"]) for p in parts),
    "max_joint_eq_violation_mrad": 1e3 * max(max(p["jeq"]) for p in parts),
  }


def _parallel(worker, lv: Levers, args) -> tuple[dict, list[dict]]:
  chunks = [c.tolist() for c in np.array_split(np.arange(args.worlds), args.workers)]
  chunks = [c for c in chunks if c]
  with concurrent.futures.ProcessPoolExecutor(len(chunks)) as pool:
    parts = list(pool.map(worker, [lv] * len(chunks), [args] * len(chunks), chunks))
  return _unpack(parts), parts


def run_random(lv: Levers, args) -> dict:
  out, _ = _parallel(_random_worker, lv, args)
  return out


def run_shake(lv: Levers, args) -> dict:
  out, parts = _parallel(_shake_worker, lv, args)
  sh = {k: [v for p in parts for v in p["shake"][k]] for k in parts[0]["shake"]}
  out.update(
    grasp_rate=float(np.mean(sh["grasped"])),
    drop_rate=float(np.mean(sh["dropped"])),
    min_pad_force_N=float(np.min(sh["min_pad_force_N"])),
    median_pad_force_N=float(np.median(sh["min_pad_force_N"])),
    max_slip_mm=1e3 * float(np.max(sh["slip_m"])),
    median_slip_mm=1e3 * float(np.median(sh["slip_m"])),
    peak_tcp_speed=float(np.max(sh["tcp_speed"])),
  )
  return out


def run_kinematics(lv: Levers, args) -> dict:
  """Pad gap against jaw angle on a slow ramp, sampled at every control step."""
  model = build_model(lv)
  w = World(model, resolve(model), lv)
  settle(w, 1.0)
  q0, gap0 = float(w.d.qpos[w.ids.jaw_qpos]), w.pad_gap()
  qs, gaps = [], []
  n = round(5.0 / CONTROL_DT)
  for k in range(n + round(1.0 / CONTROL_DT)):
    w.set_jaw(JAW_CLOSED * min(1.0, (k + 1) / n))
    w.control_step()
    qs.append(float(w.d.qpos[w.ids.jaw_qpos]))
    gaps.append(w.pad_gap())
  out = summarise([w])
  out.update(
    open_q=q0,
    open_gap_mm=1e3 * gap0,
    closed_q=qs[-1],
    closed_gap_mm=1e3 * gaps[-1],
    curve_q=qs,
    curve_gap_mm=[1e3 * g for g in gaps],
  )
  return out


##
# Plan S8: the object in the jaw (touch angle, hold angle, grasp height).
##


def _free_table(model: mujoco.MjModel, ids: Ids) -> None:
  """Take the table out of the way, as `_shake_worker` does: a jaw over the start pose would
  otherwise push a clamped object into it."""
  for g in ids.table_geoms:
    model.geom_contype[g] = 0
    model.geom_conaffinity[g] = 0


def _first(trace: list[float], q: list[float], thr: float) -> float | None:
  """Jaw angle at the first sample where ``trace`` reaches ``thr``; None if it never does."""
  return next((qk for qk, x in zip(q, trace, strict=True) if x >= thr), None)


def _touch_once(model: mujoco.MjModel, ids: Ids, lv: Levers, args, offset_mm: float):
  """One `touch` ramp with the pad-site midpoint ``offset_mm`` above the object's centre."""
  weight = model.body_subtreemass[ids.cube_body] * -model.opt.gravity[2]
  w = World(model, ids, lv)
  settle(w, SETTLE_S)
  _place_cube_in_jaw(w, 1e-3 * offset_mm, math.radians(args.touch_yaw))
  start_overlap = max(
    [-c["dist"] for c in w.object_contacts() if c["side"] is None], default=0.0
  )
  w.d.xfrc_applied[ids.cube_body, 2] = weight
  base = ids.default_ctrl[ids.jaw_act]
  n = round(args.touch_seconds / CONTROL_DT)
  q, tgt, gap, left, right, any_l, any_r, mid_z = [], [], [], [], [], [], [], []
  ok = True
  for k in range(n):
    w.set_jaw(base + args.jaw_scale * (k + 1) / n)
    ok = w.control_step() and ok
    fl, fr = w.pad_forces()
    sides = {c["side"] for c in w.object_contacts()}
    q.append(float(w.d.qpos[ids.jaw_qpos]))
    tgt.append(float(w._jaw_target))
    gap.append(1e3 * w.pad_gap())
    mid_z.append(1e3 * float(w.pad_mid_in_hand()[2]))
    left.append(fl)
    right.append(fr)
    any_l.append(float("left" in sides))
    any_r.append(float("right" in sides))
  both = [min(a, b) for a, b in zip(left, right, strict=True)]
  any_both = [min(a, b) for a, b in zip(any_l, any_r, strict=True)]
  q_ramp, (fl_ramp, fr_ramp) = q[-1], w.pad_forces()
  ref = w.cube_in_hand()
  ok = all(w.control_step() for _ in range(round(0.5 / CONTROL_DT))) and ok
  q_supported, forces_supported = float(w.d.qpos[ids.jaw_qpos]), w.pad_forces()
  w.d.xfrc_applied[ids.cube_body, 2] = 0.0
  ok = all(w.control_step() for _ in range(round(1.0 / CONTROL_DT))) and ok
  fl, fr = w.pad_forces()
  touch_q = _first(both, q, TOUCH_FORCE)
  k_touch = q.index(touch_q) if touch_q is not None else None
  out = summarise([w])
  out.update(
    offset_mm=offset_mm,
    start_palm_overlap_mm=1e3 * start_overlap,
    # Where the pad-site midpoint has moved to along the approach axis by the touch, from
    # where it was with the jaw open (the pads' z shifts toward the fingertips as they close):
    # the requested offset is the open jaw's, the offset at the touch is this much lower.
    mid_z_shift_at_touch_mm=None if k_touch is None else mid_z[k_touch] - mid_z[0],
    mid_z_shift_at_hold_mm=1e3 * float(w.pad_mid_in_hand()[2]) - mid_z[0],
    h_hold_mm=1e3 * float(w.cube_in_hand()[2] - w.pad_mid_in_hand()[2]),
    yaw_deg=args.touch_yaw,
    ok=ok,
    first_force_q={
      "left": _first(left, q, TOUCH_FORCE),
      "right": _first(right, q, TOUCH_FORCE),
    },
    first_contact_q={"left": _first(any_l, q, 1.0), "right": _first(any_r, q, 1.0)},
    touch_q=touch_q,
    touch_contact_q=_first(any_both, q, 1.0),
    touch_force_N=TOUCH_FORCE,
    ramp_end_q=q_ramp,
    ramp_end_forces_N=[fl_ramp, fr_ramp],
    supported_hold_q=q_supported,
    supported_forces_N=list(forces_supported),
    hold_q=float(w.d.qpos[ids.jaw_qpos]),
    hold_forces_N=[fl, fr],
    hold_gap_mm=1e3 * w.pad_gap(),
    slip_after_release_mm=1e3 * float(np.linalg.norm(w.cube_in_hand() - ref)),
    contacts=w.object_contacts(),
    object_in_hand_mm=(1e3 * w.cube_in_hand()).tolist(),
    curve_q=q,
    curve_target=tgt,
    curve_gap_mm=gap,
    curve_mid_z_mm=mid_z,
    curve_left_N=left,
    curve_right_N=right,
  )
  return out


def run_touch(lv: Levers, args) -> dict:
  """The object supported between the open pads, the jaw target ramped open -> closed.

  Its weight is carried by an applied upward force while the jaw closes (a free object would
  otherwise sit on whatever it touches first), which is released after the ramp and a
  settle. Reads the jaw angle at which each pad first reports a normal force >= ``TOUCH_FORCE``
  on the object (``first_force_q``), the angle at which both do at the same step
  (``touch_q``), the same for a contact of any force, and -- after the release -- the hold
  angle and the per-pad normal forces.

  The top level is the first of ``--touch-offsets`` (default 0: the pad-site midpoint at the
  object's centre height); ``by_offset`` has every one, without the curves.
  """
  model = build_model(lv)
  ids = resolve(model)
  _free_table(model, ids)
  runs = [_touch_once(model, ids, lv, args, off) for off in args.touch_offsets]
  out = dict(runs[0])
  out["by_offset"] = [
    {k: v for k, v in r.items() if not k.startswith("curve")} for r in runs
  ]
  return out


def _ballheight_worker(lv: Levers, args, jobs: list[tuple[float, int]]) -> list[dict]:
  """`_shake_worker` for one object height per job: ``(offset_mm, world id)``.

  The shake seed depends on the world id only, so every offset sees the same 'random'
  arm motions and the offsets are compared on equal terms.
  """
  model = build_model(lv)
  ids = resolve(model)
  _free_table(model, ids)
  weight = model.body_subtreemass[ids.cube_body] * -model.opt.gravity[2]
  n_ctrl = round(args.seconds / CONTROL_DT)
  slew = None if args.shake_slew is None else _arm_slew_per_step(args.shake_slew)
  arm_default = ids.default_ctrl[ids.arm_act]
  out = []
  for off_mm, wid in jobs:
    rng = np.random.default_rng([args.seed, 20_000 + wid])
    w = World(model, ids, lv)
    settle(w, SETTLE_S)
    _place_cube_in_jaw(w, 1e-3 * off_mm)
    start = w.object_contacts()
    mid_z_open = float(w.pad_mid_in_hand()[2])
    w.d.xfrc_applied[ids.cube_body, 2] = weight
    w.set_jaw(ids.default_ctrl[ids.jaw_act] + args.jaw_scale)
    ok = all(w.control_step() for _ in range(round(0.5 / CONTROL_DT)))
    w.d.xfrc_applied[ids.cube_body, 2] = 0.0
    ok = ok and all(w.control_step() for _ in range(round(0.5 / CONTROL_DT)))
    fl, fr = w.pad_forces()
    contacts = w.object_contacts()
    pads = [c for c in contacts if c["side"] is not None]
    hold_q = float(w.d.qpos[ids.jaw_qpos])
    ref = w.cube_in_hand()
    h_hold = 1e3 * float(ref[2] - w.pad_mid_in_hand()[2])
    max_slip, min_force, n_loss = 0.0, min(fl, fr), 0
    # The object in the hand frame, mm from `ref`, once a second.
    track = [[0.0, 0.0, 0.0]]
    arm_target = arm_default.copy()
    for k in range(n_ctrl if ok else 0):
      proposal = arm_default + rng.uniform(-args.shake_amp, args.shake_amp, size=7)
      if slew is None:
        arm_target = proposal
      else:
        arm_target = arm_target + np.clip(proposal - arm_target, -slew, slew)
      w.d.ctrl[ids.arm_act] = arm_target
      if not w.control_step():
        break
      max_slip = max(max_slip, float(np.linalg.norm(w.cube_in_hand() - ref)))
      f = min(w.pad_forces())
      min_force = min(min_force, f)
      n_loss += f < TOUCH_FORCE
      if (k + 1) % round(1.0 / CONTROL_DT) == 0:
        track.append((1e3 * (w.cube_in_hand() - ref)).tolist())
    end = w.cube_in_hand() - ref
    other = [c for c in contacts if c["side"] is None]
    out.append(
      {
        "offset_mm": off_mm,
        "world": wid,
        "ok": bool(ok),
        "diverged": w.divergences > 0,
        "grasped": bool(ok and min(fl, fr) >= MIN_PAD_FORCE),
        "dropped": bool((not ok) or w.divergences > 0 or max_slip > DROP_SLIP),
        "force_left_N": fl,
        "force_right_N": fr,
        "min_pad_force_N": min(fl, fr),
        "hold_q": hold_q,
        "h_hold_mm": h_hold,
        "mid_z_shift_hold_mm": 1e3 * float(w.pad_mid_in_hand()[2] - mid_z_open),
        "start_palm_overlap_mm": 1e3
        * max([-c["dist"] for c in start if c["side"] is None], default=0.0),
        "end_q": float(w.d.qpos[ids.jaw_qpos]),
        "slip_m": max_slip,
        "end_slip_mm": (1e3 * end).tolist(),
        "track_mm": track,
        "shake_min_force_N": min_force,
        "shake_steps_below_0.05N": int(n_loss),
        "n_pad_contacts": len(pads),
        "pad_normal_z": [c["normal_hand"][2] for c in pads],
        "pad_contact_z_mm": [1e3 * c["pos_hand"][2] for c in pads],
        "other_contacts": [(c["geom"], c["force_N"]) for c in other],
        "max_connect_mm": 1e3 * w.max_connect,
      }
    )
  return out


def _wilson_upper(k: int, n: int, z: float = 1.96) -> float:
  """Upper end of the 95 % Wilson interval for k events in n."""
  p = k / n
  centre = p + z * z / (2 * n)
  half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
  return (centre + half) / (1 + z * z / n)


def _height_row(offset_mm: float, rs: list[dict]) -> dict:
  n = len(rs)
  drops = sum(r["dropped"] for r in rs)
  forces = [r["min_pad_force_N"] for r in rs]
  return {
    "offset_mm": offset_mm,
    "n": n,
    "grasp_rate": float(np.mean([r["grasped"] for r in rs])),
    "drop_rate": drops / n,
    "drop_upper95": _wilson_upper(drops, n),
    "diverged": int(sum(r["diverged"] for r in rs)),
    "median_slip_mm": 1e3 * float(np.median([r["slip_m"] for r in rs])),
    "max_slip_mm": 1e3 * float(np.max([r["slip_m"] for r in rs])),
    "hold_q_median": float(np.median([r["hold_q"] for r in rs])),
    "h_hold_mm_median": float(np.median([r["h_hold_mm"] for r in rs])),
    "palm_overlap_mm": float(max(r["start_palm_overlap_mm"] for r in rs)),
    "end_q_median": float(np.median([r["end_q"] for r in rs])),
    "min_pad_force_N": float(np.min(forces)),
    "median_pad_force_N": float(np.median(forces)),
    "shake_min_force_median_N": float(np.median([r["shake_min_force_N"] for r in rs])),
    "worlds_losing_contact": int(sum(r["shake_steps_below_0.05N"] > 0 for r in rs)),
    "median_normal_z": float(
      np.median([abs(z) for r in rs for z in r["pad_normal_z"]] or [math.nan])
    ),
  }


def run_ballheight(lv: Levers, args) -> dict:
  """Slip and drops against the height of the grasp on the object (plan S8 step 0.3)."""
  jobs = [(off, wid) for off in args.offsets for wid in range(args.worlds)]
  chunks = [jobs[i :: args.workers] for i in range(args.workers)]
  chunks = [c for c in chunks if c]
  with concurrent.futures.ProcessPoolExecutor(len(chunks)) as pool:
    parts = list(
      pool.map(_ballheight_worker, [lv] * len(chunks), [args] * len(chunks), chunks)
    )
  per_world = sorted(
    (r for part in parts for r in part),
    key=lambda r: (args.offsets.index(r["offset_mm"]), r["world"]),
  )
  rows = [
    _height_row(off, [r for r in per_world if r["offset_mm"] == off])
    for off in args.offsets
  ]
  return {
    "rows": rows,
    "per_world": per_world,
    "shape": os.environ.get("SHAPE", "cube"),
  }


RUNNERS = {
  "step": run_step,
  "random": run_random,
  "shake": run_shake,
  "kinematics": run_kinematics,
  "touch": run_touch,
  "ballheight": run_ballheight,
}


##
# Reporting.
##


def provenance() -> dict:
  def git(*a):
    try:
      return subprocess.check_output(["git", *a], text=True).strip()
    except Exception:
      return None

  return {
    "commit": git("rev-parse", "HEAD"),
    "dirty": bool(git("status", "--porcelain", "--", "src", "scripts")),
    "mujoco": mujoco.__version__,
    "host": subprocess.check_output(["hostname"], text=True).strip(),
    "command": " ".join(sys.argv),
  }


def curve_error_mm(res: dict, ref: dict) -> float:
  """Max |pad gap - reference pad gap| at matching jaw angle, over the shared range."""
  q_ref, g_ref = np.asarray(ref["curve_q"]), np.asarray(ref["curve_gap_mm"])
  q, g = np.asarray(res["curve_q"]), np.asarray(res["curve_gap_mm"])
  order = np.argsort(q_ref)
  lo, hi = max(q.min(), q_ref.min()), min(q.max(), q_ref.max())
  keep = (q >= lo) & (q <= hi)
  return float(np.max(np.abs(g[keep] - np.interp(q[keep], q_ref[order], g_ref[order]))))


def report(directory: pathlib.Path) -> None:
  rows = [json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))]
  base = next((r for r in rows if r["name"] == "baseline"), None)
  hdr = (
    "| config | levers | random peak / p99 rad/s | random div/Msteps | step peak | "
    "shake peak | stroke close / open s | grasp / drop | min pad N | max slip mm | "
    "max connect mm | open gap mm | Δcurve mm |"
  )
  print(hdr)
  print("|" + "---|" * (hdr.count("|") - 1))
  for r in rows:
    res = r["results"]
    stp, rnd, sh, kin = (
      res.get(k, {}) for k in ("step", "random", "shake", "kinematics")
    )
    lev = (
      ", ".join(f"{k}={v:g}" for k, v in r["levers"].items() if v is not None) or "—"
    )
    dcurve = (
      f"{curve_error_mm(kin, base['results']['kinematics']):.2f}"
      if kin and base and "kinematics" in base["results"]
      else "—"
    )
    conn = max(
      (x.get("max_connect_violation_mm", 0) for x in (rnd, stp, sh, kin) if x),
      default=math.nan,
    )
    print(
      f"| {r['name']} | {lev} "
      f"| {_f(rnd, 'peak_passive_qvel')} / {_f(rnd, 'p99_passive_qvel')} "
      f"| {_f(rnd, 'divergences_per_Msteps', '.2f')} ({rnd.get('divergences', '—')}) "
      f"| {_f(stp, 'peak_passive_qvel')} | {_f(sh, 'peak_passive_qvel')} "
      f"| {_f(stp, 'stroke_close_s', '.2f')} / {_f(stp, 'stroke_open_s', '.2f')} "
      f"| {_f(sh, 'grasp_rate', '.2f')} / {_f(sh, 'drop_rate', '.2f')} "
      f"| {_f(sh, 'min_pad_force_N', '.2f')} | {_f(sh, 'max_slip_mm', '.2f')} "
      f"| {conn:.3f} | {_f(kin, 'open_gap_mm', '.2f')} | {dcurve} |"
    )


def height_table(rows: list[dict]) -> str:
  """The ballheight sweep as a markdown table, one row per offset."""
  hdr = (
    "| offset mm | h at hold mm | palm overlap mm | n | grasp | drop (95 % upper) | "
    "median / max slip mm | hold q rad | min / median pad N | "
    "shake min force (median) N | contact lost | normal z |"
  )
  lines = [hdr, "|" + "---|" * (hdr.count("|") - 1)]
  for r in rows:
    lines.append(
      f"| {r['offset_mm']:+g} | {r['h_hold_mm_median']:+.1f} | {r['palm_overlap_mm']:.1f} "
      f"| {r['n']} | {r['grasp_rate']:.3f} "
      f"| {r['drop_rate']:.3f} ({r['drop_upper95']:.3f}) "
      f"| {r['median_slip_mm']:.2f} / {r['max_slip_mm']:.1f} | {r['hold_q_median']:.3f} "
      f"| {r['min_pad_force_N']:.1f} / {r['median_pad_force_N']:.1f} "
      f"| {r['shake_min_force_median_N']:.1f} | {r['worlds_losing_contact']}/{r['n']} "
      f"| {r['median_normal_z']:.3f} |"
    )
  return "\n".join(lines)


def _f(d: dict, key: str, fmt: str = ".1f") -> str:
  v = d.get(key) if d else None
  return "—" if v is None else format(v, fmt)


def main() -> None:
  args = parse_args()
  if args.report is not None:
    report(args.report)
    return
  lv = Levers(**{f.name: getattr(args, f.name) for f in dataclasses.fields(Levers)})
  if args.gains is not None:
    # `build_model` imports the registry lazily, here and in every worker process, and the
    # pick-place cfg reads $GAINS at import (plan S8) -- so setting it now selects the plant.
    os.environ["GAINS"] = args.gains
  if args.pose is not None:
    os.environ[POSE_ENV_VAR] = args.pose
  if args.shape is not None:
    # Same for the object: the task cfg reads $SHAPE when `build_model` builds it.
    os.environ["SHAPE"] = args.shape
  stimuli = STIMULI if args.stimulus == "all" else (args.stimulus,)
  args.out.mkdir(parents=True, exist_ok=True)
  path = args.out / f"{args.name}.json"
  record = json.loads(path.read_text()) if path.exists() else {"results": {}}
  record.update(
    name=args.name,
    levers=dataclasses.asdict(lv),
    jaw_scale=args.jaw_scale,
    worlds=args.worlds,
    seconds=args.seconds,
    shake_amp=args.shake_amp,
    shake_slew=args.shake_slew,
    seed=args.seed,
    gains=os.environ.get("GAINS", "G0"),
    shape=os.environ.get("SHAPE", "cube"),
    env={k: os.environ.get(k) for k in ENV_SELECTORS},
    provenance=provenance(),
  )
  for s in stimuli:
    t0 = time.time()
    res = RUNNERS[s](lv, args)
    res["wall_s"] = time.time() - t0
    record["results"][s] = res
    brief = {
      k: v
      for k, v in res.items()
      if not isinstance(v, (list, dict)) and not k.startswith("curve")
    }
    print(f"[{args.name}] {s}: {json.dumps(brief, default=float)}", flush=True)
    if s == "ballheight":
      print(height_table(res["rows"]), flush=True)
  path.write_text(json.dumps(record, indent=1, default=float))
  print(f"wrote {path}")


if __name__ == "__main__":
  main()
