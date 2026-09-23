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
JAW_CLOSED = 0.95  # the joint's upper limit, `robots/kangaroo.py` _JAW_CLOSED
MIN_PAD_FORCE = 0.5  # N per side; the plan's upgraded grasp test (ManiSkill min_force)
DROP_SLIP = 0.02  # m of cube motion in the hand frame that counts as a drop
STIMULI = ("step", "random", "shake", "kinematics")


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


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  p.add_argument("--stimulus", choices=(*STIMULI, "all"), default="all")
  p.add_argument("--name", default="baseline", help="label for this configuration")
  p.add_argument("--jaw-scale", type=float, default=0.95, help="S5.0 full-range jaw")
  p.add_argument("--worlds", type=int, default=64, help="random/shake worlds")
  p.add_argument("--seconds", type=float, default=20.0, help="random/shake duration")
  p.add_argument("--shake-amp", type=float, default=0.3, help="rad, arm target noise")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--workers", type=int, default=8)
  p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("artifacts/s6"))
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


def _nominal(model: mujoco.MjModel) -> mujoco.MjData:
  from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import SPAWN_XY
  from pal_mjlab.tasks.manipulation.scene import TABLE_TOP_Z
  from scene_model import nominal_data

  data = nominal_data(model)
  # Park the cube at the task's spawn point so it sits where it would in an episode and
  # stays out of the jaw's way for the object-free stimuli.
  cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object/object")
  jnt = next(j for j in range(model.njnt) if model.jnt_bodyid[j] == cube)
  adr = model.jnt_qposadr[jnt]
  data.qpos[adr : adr + 7] = (*SPAWN_XY, TABLE_TOP_Z + 0.025, 1.0, 0.0, 0.0, 0.0)
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
    for _ in range(n_ctrl if ok else 0):
      w.d.ctrl[ids.arm_act] = arm_default + rng.uniform(
        -args.shake_amp, args.shake_amp, size=7
      )
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


def _place_cube_in_jaw(w: World):
  """Put the cube between the open pads, faces square to them, at rest."""
  ids, d = w.ids, w.d
  s1, s2 = ids.pad_sites
  mid = 0.5 * (d.site_xpos[s1] + d.site_xpos[s2])
  x = d.site_xpos[s2] - d.site_xpos[s1]
  x /= np.linalg.norm(x)
  z = d.xmat[ids.hand_body].reshape(3, 3)[:, 2]
  z = z - x * (z @ x)
  z /= np.linalg.norm(z)
  rot = np.column_stack([x, np.cross(z, x), z])
  quat = np.zeros(4)
  mujoco.mju_mat2Quat(quat, rot.ravel())
  d.qpos[ids.cube_qpos : ids.cube_qpos + 7] = (*mid, *quat)
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


RUNNERS = {
  "step": run_step,
  "random": run_random,
  "shake": run_shake,
  "kinematics": run_kinematics,
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


def _f(d: dict, key: str, fmt: str = ".1f") -> str:
  v = d.get(key) if d else None
  return "—" if v is None else format(v, fmt)


def main() -> None:
  args = parse_args()
  if args.report is not None:
    report(args.report)
    return
  lv = Levers(**{f.name: getattr(args, f.name) for f in dataclasses.fields(Levers)})
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
    seed=args.seed,
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
  path.write_text(json.dumps(record, indent=1, default=float))
  print(f"wrote {path}")


if __name__ == "__main__":
  main()
