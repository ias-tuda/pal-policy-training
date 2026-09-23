"""S5.1 item 6: the tabletop robot's gravity compensation counts against its effort limits.

`actgravcomp` moves a joint's gravity compensation into ``qfrc_actuator``; what bounds it
there is the joint's ``actuatorfrcrange``, which mjlab's position actuators do not set (they
limit the actuator's ``forcerange``, i.e. the PD term alone). These pin both halves, and
that nothing outside the tabletop overlay changed.
"""

import re

import mujoco
import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import pytest
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg

EFFORT = {
  r"pelvis_[12]_joint": 50.0,
  r"arm_(left|right)_[12]_joint": 50.0,
  r"arm_(left|right)_[3-7]_joint": 25.0,
}


def _compile(task: str) -> mujoco.MjModel:
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = 1
  model = Scene(cfg.scene, "cpu").compile()
  cfg.sim.mujoco.apply(model)
  return model


def _joints(model: mujoco.MjModel) -> dict[str, int]:
  return {
    (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "").split("/")[-1]: j
    for j in range(model.njnt)
  }


@pytest.fixture(scope="module")
def tabletop() -> mujoco.MjModel:
  return _compile("Mjlab-PickPlace-Pal-Kangaroo-Grippers")


def test_exactly_the_pelvis_and_arm_joints_route_gravcomp(tabletop):
  routed = {n for n, j in _joints(tabletop).items() if tabletop.jnt_actgravcomp[j]}
  expected = {n for n in _joints(tabletop) if any(re.fullmatch(rx, n) for rx in EFFORT)}
  assert len(expected) == 16
  assert routed == expected


def test_the_joint_range_is_the_effort_limit(tabletop):
  for name, j in _joints(tabletop).items():
    effort = next((e for rx, e in EFFORT.items() if re.fullmatch(rx, name)), None)
    if effort is None:
      continue
    assert tabletop.jnt_actfrclimited[j], name
    assert tuple(tabletop.jnt_actfrcrange[j]) == (-effort, effort), name


def test_the_joint_range_is_what_bounds_gravity(tabletop):
  """Positive control: shrink one joint's range and the total actuator torque follows it."""
  model = tabletop
  data = mujoco.MjData(model)
  j = _joints(model)["arm_left_4_joint"]
  dof = model.jnt_dofadr[j]
  # Hold every position actuator at the current (default) pose, so the PD term is zero and
  # what is left in qfrc_actuator is the routed gravity compensation.
  mujoco.mj_forward(model, data)
  for a in range(model.nu):
    if model.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE:
      data.ctrl[a] = data.qpos[model.jnt_qposadr[model.actuator_trnid[a, 0]]]
  mujoco.mj_forward(model, data)
  load = abs(float(data.qfrc_actuator[dof]))
  assert load > 0.5, "no gravity on the elbow; nothing is being tested"

  saved = model.jnt_actfrcrange[j].copy()
  model.jnt_actfrcrange[j] = (-0.1, 0.1)
  mujoco.mj_forward(model, data)
  clamped = abs(float(data.qfrc_actuator[dof]))
  model.jnt_actfrcrange[j] = saved
  assert clamped == pytest.approx(0.1)


def test_the_locomotion_grippers_model_is_untouched():
  model = _compile("Mjlab-Velocity-Flat-Pal-Kangaroo-Grippers")
  assert not model.jnt_actgravcomp.any()
