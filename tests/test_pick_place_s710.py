"""Plan S7-10 (Siwei, 2026-10-01: the real robot's forearm touched the table): ``ARM_TABLE``.

* unset / ``none`` builds exactly the cfg every run before S7-10 built;
* each code adds what it says and nothing else -- no observation, action, command or event
  changes, which is what lets S7-8's ABS-c resume under it;
* the sensor watches the driven arm's links 1-6, one slot each, in link order;
* at the start pose nothing of arm 1-6 is within the clearance margin of the table, so AH costs
  nothing before the policy moves;
* the clearance's point sample is within 1 cm of the mesh's true lowest point over the table;
* positive control: a table moved up into the forearm is seen by the sensor, the contact term,
  the clearance term and AT's termination.
"""

import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.manipulation.mdp import illegal_contact
from pal_mjlab.tasks.manipulation import mdp
from pal_mjlab.tasks.manipulation.pick_place import arm_table as at
from pal_mjlab.tasks.manipulation.pick_place.kangaroo import env_cfg as kenv
from pal_mjlab.tasks.manipulation.pick_place.rewards import S52_BASE
from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS_LEFT_ARM
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import DECIMATION

#: S7-8's ABS-c, as `pick_place.sbatch` exported it (run log, jobs 160566-160569).
ABS_C_ENV = {
  "REWARD": f"{S52_BASE}+M1+M2+M3+M6+J+M8",
  "DR": "DR-scene",
  "BASELINE": "B",
  "GAINS": "G20",
  "EFFORT": "CTP",
  "LAG": "FP+JS",
  "SLEW": "stack",
  "EPISODE_S": "55",
  "SPAWN": "W2",
  "TARGET": "B",
  "ACTION": "abs",
  "CMD_OBS": "cmd",
}
LINKS = [f"arm_left_{i}_link" for i in range(1, 7)]
GRIPPER = KANGAROO_GRIPPERS_LEFT_ARM.gripper("left")
ARM = SceneEntityCfg("robot", body_names=(at.arm_body_regex(GRIPPER),))
TABLE = SceneEntityCfg("table")


def _cfg(monkeypatch, code: str | None, num_envs: int | None = None):
  for var, value in {**ABS_C_ENV, "ARM_TABLE": code}.items():
    if value is None:
      monkeypatch.delenv(var, raising=False)
    else:
      monkeypatch.setenv(var, value)
  cfg = kenv.kangaroo_pick_place_env_cfg()
  if num_envs is not None:
    cfg.scene.num_envs = num_envs
    # An env test draws DR-scene's table shift from a fixed seed, not mjlab's random one.
    cfg.seed = 0
  return cfg


# -- Configuration --------------------------------------------------------------------------


def test_unset_and_none_build_the_s78_cfg(monkeypatch):
  base = _cfg(monkeypatch, None)
  none = _cfg(monkeypatch, "none")
  assert [s.name for s in base.scene.sensors] == [s.name for s in none.scene.sensors]
  assert at.ARM_TABLE_SENSOR not in [s.name for s in base.scene.sensors]
  for field in (
    "actions",
    "observations",
    "commands",
    "events",
    "terminations",
    "rewards",
  ):
    assert getattr(base, field) == getattr(none, field), field
  assert not {"arm_table_contact", "arm_table_clearance"} & set(base.rewards)
  assert "arm_table_force" not in base.terminations


@pytest.mark.parametrize("code", ["watch", "AC", "AH", "AT", "AH10"])
def test_each_code_adds_only_its_terms(monkeypatch, code):
  base = _cfg(monkeypatch, None)
  cfg = _cfg(monkeypatch, code)
  spec = at.ARM_TABLE_CODES[code]
  # Nothing the policy sees or does changes: a checkpoint from ABS-c resumes.
  for field in ("actions", "observations", "commands", "events", "curriculum"):
    assert getattr(cfg, field) == getattr(base, field), field
  assert [s.name for s in cfg.scene.sensors] == [
    *(s.name for s in base.scene.sensors),
    at.ARM_TABLE_SENSOR,
  ]
  added = set(cfg.rewards) - set(base.rewards)
  want = set()
  if spec.contact_weight:
    want.add("arm_table_contact")
    term = cfg.rewards["arm_table_contact"]
    assert term.weight == spec.contact_weight and term.params["force_threshold"] == 1.0
  if spec.clearance_weight:
    want.add("arm_table_clearance")
    term = cfg.rewards["arm_table_clearance"]
    assert term.weight == spec.clearance_weight
    assert term.params["margin"] == at.CLEARANCE_MARGIN
  assert added == want
  for name in base.rewards:
    assert cfg.rewards[name] == base.rewards[name], name
  extra = set(cfg.terminations) - set(base.terminations)
  assert extra == ({"arm_table_force"} if spec.terminate_force else set())
  if spec.terminate_force:
    term = cfg.terminations["arm_table_force"]
    assert term.func is illegal_contact and term.params["force_threshold"] == 30.0
    assert not term.time_out  # a failure, not a truncation


def test_the_codes_weights():
  """The plan's weights, and AH10 = AH x 10 (Siwei, 2026-10-02)."""
  codes = at.ARM_TABLE_CODES
  assert (codes["AC"].contact_weight, codes["AC"].clearance_weight) == (-1.0, 0.0)
  assert (codes["AH"].contact_weight, codes["AH"].clearance_weight) == (-1.0, -0.5)
  assert (codes["AT"].contact_weight, codes["AT"].terminate_force) == (-1.0, 30.0)
  ah, ah10 = codes["AH"], codes["AH10"]
  assert ah10.contact_weight == 10 * ah.contact_weight
  assert ah10.clearance_weight == 10 * ah.clearance_weight
  assert (ah10.contact_force, ah10.clearance_margin, ah10.terminate_force) == (
    ah.contact_force,
    ah.clearance_margin,
    None,
  )


def test_a_bad_code_raises(monkeypatch):
  with pytest.raises(ValueError, match="ARM_TABLE"):
    _cfg(monkeypatch, "AX")


def test_the_sensor_keeps_a_force_history():
  """Without one, `illegal_contact` returns ``found`` and ignores its force threshold."""
  sensor = at.make_arm_table_sensor(GRIPPER)
  assert sensor.history_length > 0 and "force" in sensor.fields
  # Every substep of a control step: a hit early in the step must not be missed.
  assert sensor.history_length == DECIMATION
  assert sensor.primary.pattern == r"arm_left_[1-6]_link"


# -- In an env ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ah_env():
  mp = pytest.MonkeyPatch()
  env = ManagerBasedRlEnv(cfg=_cfg(mp, "AH", num_envs=2), device=get_test_device())
  env.reset()
  yield env
  env.close()
  mp.undo()


def test_the_sensor_watches_arm_1_to_6_in_order(ah_env):
  sensor = ah_env.scene[at.ARM_TABLE_SENSOR]
  assert [n.split("/")[-1] for n in sensor.primary_names] == LINKS
  ARM.resolve(ah_env.scene)
  robot = ah_env.scene["robot"]
  assert [robot.body_names[i] for i in ARM.body_ids] == LINKS


def test_the_start_pose_is_clear_of_the_margin(ah_env):
  ARM.resolve(ah_env.scene)
  clearance = mdp.arm_table_clearance(ah_env, ARM, TABLE)
  assert (clearance > at.CLEARANCE_MARGIN).all(), clearance
  # The forearm is over the table: the term is measuring something, not +inf everywhere.
  assert torch.isfinite(clearance).any(dim=-1).all()
  assert (
    mdp.table_clearance_penalty(ah_env, at.CLEARANCE_MARGIN, ARM, TABLE) == 0
  ).all()
  force = torch.norm(ah_env.scene[at.ARM_TABLE_SENSOR].data.force_history, dim=-1)
  assert (force <= 1.0).all()


def test_the_point_sample_is_within_a_cm_of_the_mesh(ah_env):
  """Against every vertex of each link's colliding meshes over the table's footprint, in world,
  at the start pose."""
  import mujoco
  import numpy as np
  from mjlab.utils.lab_api.math import quat_apply

  ARM.resolve(ah_env.scene)
  robot = ah_env.scene["robot"]
  mj = ah_env.sim.mj_model
  top = mdp.surface_top_z(ah_env, TABLE)
  sampled = mdp.arm_table_clearance(ah_env, ARM, TABLE)
  geom = ah_env.scene["table"].indexing.geom_ids[0]
  centre = ah_env.sim.data.geom_xpos[:, geom]
  half = ah_env.sim.model.geom_size[:, geom].expand(ah_env.num_envs, 3)
  checked = 0
  for k, local in enumerate(ARM.body_ids):
    body = int(robot.indexing.body_ids[local])
    verts = []
    for g in np.flatnonzero(mj.geom_bodyid == body):
      if not (mj.geom_contype[g] or mj.geom_conaffinity[g]):
        continue
      m = mj.geom_dataid[g]
      rot = np.zeros(9)
      mujoco.mju_quat2Mat(rot, mj.geom_quat[g])
      v = mj.mesh_vert[mj.mesh_vertadr[m] : mj.mesh_vertadr[m] + mj.mesh_vertnum[m]]
      verts.append(v @ rot.reshape(3, 3).T + mj.geom_pos[g])
    pts = torch.tensor(np.concatenate(verts), dtype=torch.float32, device=top.device)
    for e in range(ah_env.num_envs):
      if not torch.isfinite(sampled[e, k]):
        continue
      pos = robot.data.body_link_pos_w[e, local]
      quat = robot.data.body_link_quat_w[e, local].expand(len(pts), 4)
      world = pos + quat_apply(quat, pts)
      # The term's footprint: only points over the table's top face count.
      rel = world - centre[e]
      over = (rel[:, 0].abs() <= half[e, 0]) & (rel[:, 1].abs() <= half[e, 1])
      assert over.any()
      true_min = float(world[over, 2].min() - top[e])
      assert sampled[e, k] >= true_min - 1e-4  # a sample cannot go below the mesh
      assert sampled[e, k] - true_min < 0.01, (LINKS[k], float(sampled[e, k]), true_min)
      checked += 1
  assert checked


def test_positive_control_a_table_moved_into_the_forearm(monkeypatch):
  """Slide the table back so that its far edge sits between the forearm and the gripper, and
  raise its top 2 cm above the forearm's lowest point: arm 1-6 now presses on the table and
  the gripper is off it."""
  cfg = _cfg(monkeypatch, "AT", num_envs=2)
  # Keep the terminal state: an auto-reset would clear the sensor that AT just fired on.
  cfg.auto_reset = False
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  try:
    env.reset()
    ARM.resolve(env.scene)
    robot = env.scene["robot"]
    table = env.scene["table"]
    tcfg = SceneEntityCfg("table", body_names=("table",))
    tcfg.resolve(env.scene)
    body = int(table.indexing.body_ids[tcfg.body_ids[0]])
    geom = table.indexing.geom_ids[0]
    forearm = LINKS.index("arm_left_5_link")
    # The forearm's lowest point over the whole mesh: DR-scene's table shift can leave it off
    # the footprint at the start (+inf in the term), which the move below puts right.
    from mjlab.utils.lab_api.math import quat_apply
    from pal_mjlab.tasks.manipulation.mdp.rewards import _collision_points

    pts = _collision_points(env, robot, ARM.body_ids)[forearm]
    i5 = ARM.body_ids[forearm]
    low = torch.stack(
      [
        (
          robot.data.body_link_pos_w[e, i5]
          + quat_apply(robot.data.body_link_quat_w[e, i5].expand(len(pts), 4), pts)
        )[:, 2].min()
        for e in range(env.num_envs)
      ]
    )
    # Far edge 3 cm past link 6's origin, short of link 7's; top 2 cm into the forearm.
    i6, i7 = (
      robot.body_names.index("arm_left_6_link"),
      robot.body_names.index("arm_left_7_link"),
    )
    x6 = robot.data.body_link_pos_w[:, i6, 0]
    x7 = robot.data.body_link_pos_w[:, i7, 0]
    assert (x7 - x6 > 0.05).all()
    half = env.sim.model.geom_size[:, geom].expand(env.num_envs, 3)
    centre = env.sim.data.geom_xpos[:, geom]
    dx = (x6 + 0.03) - (centre[:, 0] + half[:, 0])
    dz = low - mdp.surface_top_z(env, TABLE) + 0.02
    env.sim.model.body_pos[:, body, 0] += dx
    env.sim.model.body_pos[:, body, 2] += dz
    act = torch.zeros(
      env.num_envs, env.action_manager.total_action_dim, device=env.device
    )
    _, _, terminated, _, _ = env.step(act)
    peak = torch.norm(env.scene[at.ARM_TABLE_SENSOR].data.force_history, dim=-1).amax(
      -1
    )
    # Moved back and raised, the table meets the elbow (link 4) first: it hangs lower than
    # the forearm. Any of arm 1-6 is what the codes price.
    assert (peak.amax(dim=-1) > 1.0).all(), peak
    assert (mdp.contact_penalty(env, at.ARM_TABLE_SENSOR, 1.0) == 1.0).all()
    # The contact pushes the arm back up within the step (to ~0.6 cm in one env at seed 0),
    # so the term is inside the margin, not necessarily at its contact value of 1.
    assert (
      mdp.table_clearance_penalty(env, at.CLEARANCE_MARGIN, ARM, TABLE) > 0.0
    ).all()
    ended = illegal_contact(env, at.ARM_TABLE_SENSOR, 30.0)
    assert torch.equal(ended, (peak > 30.0).any(dim=-1))
    assert torch.equal(env.termination_manager.get_term("arm_table_force"), ended), (
      "AT's term is not what the env ended on"
    )
    assert (terminated >= ended).all()
    # The hand is off the table: what fired is the arm, not the jaw scraping.
    hand = env.scene[GRIPPER.surface_sensor_name].data.found
    assert not hand.any()
  finally:
    env.close()
