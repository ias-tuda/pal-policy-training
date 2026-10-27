"""Compile a registered task's scene to a plain ``MjModel``, without warp or torch.

Everything in S1 of the pick-and-place plan -- the reach map, the nominal-pose search, the
screenshots -- is kinematics and collision on the *compiled scene*, not simulation. Going
through ``ManagerBasedRlEnv`` for that costs a warp kernel build (~60 s cold) and gives
nothing back, so these tools build the scene directly instead.

The model this returns is the same one the env runs: same entities, same mocap weld for
``fixed_base``, same collision filtering from ``MANIPULATION_COLLISION``.
"""

from __future__ import annotations

import dataclasses

import mujoco
import numpy as np
import pal_mjlab.tasks  # noqa: F401  # populates the task registry
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg

#: The task whose scene S1 works on. Pick-and-place does not exist yet; lift shares the
#: robot, the table and the cube with it (``make_tabletop_env_cfg``), so it is the scene.
DEFAULT_TASK = "Mjlab-Lift-Pal-Kangaroo-Grippers"


def compile_scene(
  task_id: str = DEFAULT_TASK, baseline: bool = False
) -> mujoco.MjModel:
  """The compiled single-env scene for ``task_id``.

  ``baseline`` swaps the robot back to ``get_kangaroo_grippers_robot_cfg`` -- the collision
  set and start pose the manipulation tasks had before S1.4 -- so any "what did the new
  colliders cost" question is answered against the real previous model rather than against
  a remembered number.

  Note that collision geometry can only be changed *before* ``Scene.compile()``.
  ``CollisionCfg.edit_spec`` zeroes ``contype``/``conaffinity`` on every geom it does not
  match, and MuJoCo's compiler then leaves those geoms out of the bounding-volume
  hierarchy: a geom re-enabled on the compiled model reports ``contype=1`` and still never
  reaches narrow phase.
  """
  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = 1
  if baseline:
    from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import (
      get_kangaroo_grippers_robot_cfg,
    )
    from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS

    env_cfg.scene.entities["robot"] = dataclasses.replace(
      KANGAROO_GRIPPERS, entity_cfg_fn=get_kangaroo_grippers_robot_cfg
    ).build_entity_cfg()
  model = Scene(env_cfg.scene, "cpu").compile()
  # The env applies its solver settings (2 ms, elliptic cone, impratio 10, iteration
  # counts) to the compiled model inside Simulation; do the same here so anything stepped
  # or rendered from this model is the same physics the task runs.
  env_cfg.sim.mujoco.apply(model)
  return model


def nominal_data(model: mujoco.MjModel, init_state=None) -> mujoco.MjData:
  """``MjData`` at the robot's configured initial joint positions.

  The env applies ``init_state.joint_pos`` through its reset event; here we write the same
  values straight into ``qpos`` so a pose can be inspected without stepping. The mocap body
  that welds the base is already at its spawn pose in ``model``, so ``mj_forward`` alone
  puts the robot where the env puts it.

  Defaults to ``MANIPULATION_INIT_STATE``, the pose the tabletop tasks start from.
  """
  from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import MANIPULATION_INIT_STATE

  if init_state is None:
    init_state = MANIPULATION_INIT_STATE
  data = mujoco.MjData(model)
  set_joint_pos(model, data, init_state.joint_pos)
  mujoco.mj_forward(model, data)
  return data


def set_joint_pos(
  model: mujoco.MjModel, data: mujoco.MjData, joint_pos: dict[str, float]
) -> None:
  """Write a ``{joint-name-regex: value}`` map into ``qpos``, as mjlab's reset does."""
  import re

  names = joint_names(model)
  for pattern, value in joint_pos.items():
    matched = [n for n in names if re.fullmatch(pattern, _bare(n))]
    if not matched:
      raise KeyError(f"no joint matches {pattern!r}")
    for name in matched:
      jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
      data.qpos[model.jnt_qposadr[jid]] = value


def joint_names(model: mujoco.MjModel) -> list[str]:
  return [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
    for j in range(model.njnt)
  ]


def _bare(name: str) -> str:
  """``robot/arm_left_1_joint`` -> ``arm_left_1_joint``."""
  return name.split("/", 1)[-1]


def joint_qadr(model: mujoco.MjModel, names: list[str]) -> np.ndarray:
  """qpos addresses of ``names`` (entity-qualified or bare)."""
  out = []
  for name in names:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
      jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{name}")
    if jid < 0:
      raise KeyError(f"no joint {name!r}")
    out.append(model.jnt_qposadr[jid])
  return np.asarray(out, dtype=int)


def soft_limits(
  model: mujoco.MjModel, names: list[str], factor: float = 0.9
) -> tuple[np.ndarray, np.ndarray]:
  """Joint ranges shrunk about their midpoint by ``factor``.

  ``EntityArticulationInfoCfg.soft_joint_pos_limit_factor`` is 0.9 for every Kangaroo
  variant (``kangaroo_constants.py``), and the ``joint_pos_limits`` reward penalises the
  policy outside this band -- so it is the band a reachable target must lie in.
  """
  lo, hi = [], []
  for name in names:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
      jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{name}")
    low, high = model.jnt_range[jid]
    mid, half = 0.5 * (low + high), 0.5 * (high - low)
    lo.append(mid - factor * half)
    hi.append(mid + factor * half)
  return np.asarray(lo), np.asarray(hi)


def geom_id(model: mujoco.MjModel, name: str) -> int:
  gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
  if gid < 0:
    raise KeyError(f"no geom {name!r}")
  return gid


def site_id(model: mujoco.MjModel, name: str) -> int:
  sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
  if sid < 0:
    raise KeyError(f"no site {name!r}")
  return sid
