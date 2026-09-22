"""The Kangaroo grippers collision models: the tabletop one, and the one it must not touch.

`get_kangaroo_grippers_tabletop_robot_cfg` adds torso, upper-arm and elbow colliders and a
different start pose for tabletop manipulation. `get_kangaroo_grippers_robot_cfg` is what
`Mjlab-Velocity-*-Pal-Kangaroo-Grippers` and `Mjlab-Reaching-Flat-Pal-Kangaroo-Grippers`
instantiate and has to keep behaving exactly as it did; the pair of tests below is the
guard on that boundary.

Neither entity compiles on its own -- `kangaroo_grippers.xml` excludes the feet against a
body named `terrain`, which only exists once a scene supplies it -- so both go through a
minimal `Scene`.
"""

import mujoco
import pytest
from mjlab.scene import Scene, SceneCfg
from mjlab.terrains import TerrainEntityCfg
from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import (
  INIT_STATE,
  MANIPULATION_INIT_STATE,
  get_kangaroo_grippers_robot_cfg,
  get_kangaroo_grippers_tabletop_robot_cfg,
)

#: What `MANIPULATION_COLLISION` selects: feet, arm 5-7 and the gripper, nothing above the
#: elbow. Recorded here so that a change to it fails a test rather than a training run.
_BASELINE_ROBOT_COLLIDERS = 36
_TABLETOP_EXTRA_COLLIDERS = ("pelvis_2_collision",) + tuple(
  f"arm_{side}_{i}_collision" for side in ("left", "right") for i in (2, 3, 4)
)


def _compile(entity_cfg) -> mujoco.MjModel:
  scene_cfg = SceneCfg(
    num_envs=1,
    env_spacing=2.5,
    entities={"robot": entity_cfg},
    terrain=TerrainEntityCfg(terrain_type="plane"),
  )
  return Scene(scene_cfg, "cpu").compile()


def _active_collision_geoms(model: mujoco.MjModel) -> set[str]:
  return {
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    for g in range(model.ngeom)
    if model.geom_contype[g] or model.geom_conaffinity[g]
  }


def _robot_geoms(names: set[str]) -> set[str]:
  return {n for n in names if n.startswith("robot/")}


@pytest.fixture(scope="module")
def baseline() -> mujoco.MjModel:
  return _compile(get_kangaroo_grippers_robot_cfg())


@pytest.fixture(scope="module")
def tabletop() -> mujoco.MjModel:
  return _compile(get_kangaroo_grippers_tabletop_robot_cfg())


def test_baseline_grippers_collision_set_is_unchanged(baseline):
  """The locomotion/reaching variant still collides on feet, arm 5-7 and the gripper only."""
  geoms = _robot_geoms(_active_collision_geoms(baseline))
  assert len(geoms) == _BASELINE_ROBOT_COLLIDERS
  assert not [g for g in geoms if "pelvis" in g]
  assert not [g for g in geoms if "arm_left_2" in g or "arm_left_3" in g]


def test_tabletop_adds_the_torso_and_upper_arm(baseline, tabletop):
  geoms = _robot_geoms(_active_collision_geoms(tabletop))
  for name in _TABLETOP_EXTRA_COLLIDERS:
    assert f"robot/{name}" in geoms, name
  assert geoms == _robot_geoms(_active_collision_geoms(baseline)) | {
    f"robot/{n}" for n in _TABLETOP_EXTRA_COLLIDERS
  }


def test_tabletop_colliders_are_primitives(tabletop):
  """The links this adds get boxes and capsules, not meshes: they are proxies, not shapes."""
  for name in _TABLETOP_EXTRA_COLLIDERS:
    gid = mujoco.mj_name2id(tabletop, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{name}")
    assert tabletop.geom_type[gid] in (
      mujoco.mjtGeom.mjGEOM_BOX,
      mujoco.mjtGeom.mjGEOM_CAPSULE,
    ), name


def test_tabletop_colliders_add_no_mass(baseline, tabletop):
  """Every link carries an explicit <inertial>, so the proxies must not change the robot."""
  assert tabletop.body_mass.sum() == pytest.approx(baseline.body_mass.sum())


def test_no_self_collision_at_the_start_pose(tabletop):
  """S1.4's exit criterion: nothing on the robot touches anything else on it at rest.

  Contacts against the floor are expected -- the robot is standing on it.
  """
  data = mujoco.MjData(tabletop)
  for pattern, value in MANIPULATION_INIT_STATE.joint_pos.items():
    for jid in range(tabletop.njnt):
      name = mujoco.mj_id2name(tabletop, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
      if _matches(pattern, name):
        data.qpos[tabletop.jnt_qposadr[jid]] = value
  mujoco.mj_forward(tabletop, data)

  offenders = []
  for c in range(data.ncon):
    names = [
      mujoco.mj_id2name(tabletop, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or ""
      for g in (data.contact[c].geom1, data.contact[c].geom2)
    ]
    if all(n.startswith("robot/") for n in names):
      offenders.append(tuple(names))
  assert not offenders, (
    f"robot-robot contacts at the start pose: {sorted(set(offenders))}"
  )


def test_start_pose_holds_the_left_elbow_square(tabletop):
  """The pose is a decision, not an accident: elbow at pi/2, jaw open and over the table."""
  assert MANIPULATION_INIT_STATE.joint_pos["arm_left_4_joint"] == pytest.approx(
    1.571, abs=1e-3
  )
  assert "gripper_left_finger_joint" not in MANIPULATION_INIT_STATE.joint_pos
  # Legs and pelvis are inherited verbatim; only the arms differ from the locomotion pose.
  for key, value in INIT_STATE.joint_pos.items():
    if key.startswith(("leg_", "pelvis_")):
      assert MANIPULATION_INIT_STATE.joint_pos[key] == value


def _matches(pattern: str, name: str) -> bool:
  import re

  return re.fullmatch(pattern, name.split("/", 1)[-1]) is not None
