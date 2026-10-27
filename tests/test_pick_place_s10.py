"""Plan S10 (`pick-place-v1-objects-plan.md`): the cube into a box -- step 0's model checks.

The env is `BS` (S7-8's plant, ABS-a, 10 s, W2 + B) with ``ARM_TABLE=AC`` and ``GOAL=box*``, built
on CPU. What is checked: the box is the CAD's; ``GOAL`` unset is the table task exactly; the box
lands in baseline B's target box at yaw 0 +/- 5 deg and the goal sits on its floor; ``in_box`` and
``placed`` on cubes dropped into, onto the rim of and beside the box; the release latch measures
from the rim, carries the yaw, and refuses a reader that measures from elsewhere; the yaw grade;
pushing the box costs; the tracker bias split; BX-DR's per-world geometry; AC's sensor fires on
an arm pressed by the table (the positive control plan S10 step 0 item 2 asks for).
"""

from __future__ import annotations

import collections
import importlib.util
import math
import pathlib
import sys

import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  euler_xyz_from_quat,
  quat_apply,
  quat_from_euler_xyz,
)
from pal_mjlab.tasks.manipulation import mdp, scene
from pal_mjlab.tasks.manipulation.mdp.observations import (
  _WAS_LIFTED,
  _placed_bool,
  _release_latch,
)
from pal_mjlab.tasks.manipulation.pick_place import box as bx
from pal_mjlab.tasks.manipulation.pick_place.kangaroo.env_cfg import (
  kangaroo_pick_place_env_cfg,
)
from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import (
  BASELINES,
  GOAL_THRESHOLD,
)

RCTP = "R-C-strict+M1+M2+M3+M6+J+M8"
#: Plan S10's base: `BS` (ABS-a = ACTION=abs, CMD_OBS=act; 10 s) + S7-10's AC.
BS_ENV = {
  "REWARD": RCTP,
  "DR": "DR-scene",
  "PPO": "PPO-stack",
  "SLEW": "stack",
  "BASELINE": "B",
  "SPAWN": "W2",
  "TOOL_SPEED": "none",
  "GAINS": "G20",
  "EFFORT": "CTP",
  "LAG": "FP+JS",
  "ACTION": "abs",
  "CMD_OBS": "act",
  "TARGET": "B",
  "EPISODE_S": "10",
  "ARM_TABLE": "AC",
}


def _cfg(monkeypatch, goal: str | None, num_envs: int = 4, **changes):
  for var, value in {**BS_ENV, **changes, "GOAL": goal}.items():
    if value is None:
      monkeypatch.delenv(var, raising=False)
    else:
      monkeypatch.setenv(var, value)
  for var in ("SHAPE", "OCC", "OBJ_OBS"):
    if var not in changes:
      monkeypatch.delenv(var, raising=False)
  cfg = kangaroo_pick_place_env_cfg()
  cfg.scene.num_envs = num_envs
  cfg.seed = 0  # the draws (DR-scene's table, the box, the cube) fixed: a control must not depend on them
  return cfg


def _env(monkeypatch, goal: str | None = "box", num_envs: int = 4, **changes):
  env = ManagerBasedRlEnv(
    cfg=_cfg(monkeypatch, goal, num_envs, **changes), device=get_test_device()
  )
  env.reset()
  return env


def _hold(env, steps: int) -> None:
  """``steps`` control steps holding the start pose (action 0, jaw open)."""
  with torch.inference_mode():
    for _ in range(steps):
      env.step(torch.zeros(env.action_space.shape, device=env.device))


def _put_cube(env, offset_xy, height_above_floor: float, yaw: float = 0.0) -> None:
  """Teleport every env's cube into the box's frame: ``offset_xy`` from the floor centre (box
  axes), its bottom ``height_above_floor`` above the floor top, turned ``yaw`` off the box, at
  rest."""
  box = env.scene[bx.BOX]
  obj = env.scene["object"]
  n = env.num_envs
  floor = mdp.box.box_floor_top_w(env, bx.BOX, scene.BOX_FLOOR)
  bq = box.data.root_link_quat_w
  off = torch.zeros(n, 3, device=env.device)
  off[:, 0], off[:, 1] = offset_xy
  pose = obj.data.root_link_pose_w.clone()
  pose[:, :3] = floor + quat_apply(bq, off)
  pose[:, 2] = floor[:, 2] + 0.025 + height_above_floor
  _, _, byaw = euler_xyz_from_quat(bq)
  zero = torch.zeros(n, device=env.device)
  pose[:, 3:7] = quat_from_euler_xyz(zero, zero, byaw + yaw)
  obj.write_root_link_pose_to_sim(pose)
  obj.write_root_link_velocity_to_sim(torch.zeros_like(obj.data.root_link_vel_w))


def _placed(env) -> torch.Tensor:
  setattr(
    env, _WAS_LIFTED, torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
  )
  jaw = SceneEntityCfg("robot", joint_names=("gripper_left_finger_joint",))
  jaw.resolve(env.scene)
  return _placed_bool(
    env,
    "goal_pose",
    "object",
    "left_pad_contact",
    GOAL_THRESHOLD,
    0.05,
    "object_table_contact",
    jaw,
    0.15,
  )


# -- The model --------------------------------------------------------------------------


def test_the_box_is_the_cad():
  """`cad/s10_box/make_box.py`: inner 80, walls 5, floor 5, 25 mm of wall above the floor."""
  import mujoco

  m = scene.get_box_spec().compile()
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  half = m.geom_size.copy()
  sphere = m.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE
  half[sphere] = m.geom_size[
    sphere, :1
  ]  # a sphere's extent is its radius on every axis
  lo = (d.geom_xpos - half).min(axis=0)
  hi = (d.geom_xpos + half).max(axis=0)
  assert hi[:2] == pytest.approx((0.045, 0.045), abs=1e-9)
  assert lo[:2] == pytest.approx((-0.045, -0.045), abs=1e-9)
  assert (lo[2], hi[2]) == pytest.approx((0.0, 0.030), abs=1e-9)
  assert m.body_mass[1] == pytest.approx(scene.BOX_MASS)
  walls = [
    mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, w)
    for w in scene.box_wall_geom_names()
  ]
  inner = d.geom_xpos[walls[0], 0] - m.geom_size[walls[0], 0]
  assert inner == pytest.approx(0.04, abs=1e-9)


def test_goal_unset_is_the_table_task(monkeypatch):
  """``GOAL`` unset and ``GOAL=table`` build the same cfg, with no box anywhere in it."""
  a = _cfg(monkeypatch, None)
  b = _cfg(monkeypatch, "table")
  assert set(a.scene.entities) == set(b.scene.entities) == {"robot", "table", "object"}
  assert sorted(a.rewards) == sorted(b.rewards)
  assert "box_moved" not in a.rewards
  assert a.observations["actor"].terms["target_pos"].func is mdp.goal_position_b
  assert type(a.commands["goal_pose"]) is type(b.commands["goal_pose"])
  assert not isinstance(a.commands["goal_pose"], mdp.BoxGoalCommandCfg)
  for name, term in a.rewards.items():
    assert "yaw_std" not in term.params, name


@pytest.mark.parametrize("goal", ["box", "box-ny", "box-rim", "box-dr"])
def test_every_box_code_measures_the_release_from_the_rim(monkeypatch, goal):
  cfg = _cfg(monkeypatch, goal)
  readers = [
    n
    for n, t in cfg.rewards.items()
    if "surface_cfg" in t.params and t.func is not mdp.lift_height_faded
  ]
  assert {"release", "placed", "pad_contact"} <= set(readers)
  for name in readers:
    surface = cfg.rewards[name].params["surface_cfg"]
    assert (surface.name, tuple(surface.geom_names)) == (bx.BOX, bx.BOX_WALLS), name
  # The lift is still measured from the table.
  assert cfg.rewards["lift"].params["surface_cfg"].name == "table"
  yaw = [cfg.rewards[n].params.get("yaw_std") for n in ("release", "placed")]
  assert yaw == ([None, None] if goal == "box-ny" else [math.radians(10)] * 2)
  assert ("rim_contact" in cfg.rewards) == (goal == "box-rim")
  assert cfg.rewards["arm_table_contact"].weight == -1.0


def test_the_box_is_put_down_in_the_target_box(monkeypatch):
  env = _env(monkeypatch, num_envs=16)
  _hold(env, 5)
  b = BASELINES["B"]
  box = env.scene[bx.BOX]
  table = SceneEntityCfg("table")
  top = mdp.surface_top_z(env, table)
  centre = mdp.events._surface_centre_xy(env, table)
  rel = box.data.root_link_pos_w[:, :2] - centre
  tx, ty = b.target_xy[0] - scene.TABLE_POS[0], b.target_xy[1] - scene.TABLE_POS[1]
  assert ((rel[:, 0] - tx).abs() <= b.target_half_xy + 1e-3).all()
  assert ((rel[:, 1] - ty).abs() <= b.target_half_xy + 1e-3).all()
  assert (box.data.root_link_pos_w[:, 2] - top).abs().max() < 1e-3
  _, _, yaw = euler_xyz_from_quat(box.data.root_link_quat_w)
  yaw = torch.remainder(yaw + math.pi, 2 * math.pi) - math.pi
  assert yaw.abs().max() <= bx.BOX_YAW + 1e-3
  goal = env.command_manager.get_term("goal_pose").command
  assert (goal[:, 2] - top - 0.030).abs().max() < 1e-3
  assert torch.norm(goal[:, :2] - box.data.root_link_pos_w[:, :2], dim=-1).max() < 1e-3
  rim = mdp.box_rim_z(env, bx.BOX, bx.BOX_WALLS)
  assert (rim - top - 0.030).abs().max() < 1e-3
  assert not mdp.object_in_box_bool(env, "object", bx.BOX, bx.BOX_WALLS).any()
  assert env.observation_manager.compute()["actor"].shape[-1] == 38
  env.close()


def test_a_cube_dropped_into_the_box_is_in_it_and_placed(monkeypatch):
  """Plan S10 step 0 item 6: a centred, aligned drop from 1 cm over the rim is ``placed``."""
  env = _env(monkeypatch)
  _hold(env, 3)
  _put_cube(env, (0.0, 0.0), scene.BOX_WALL_HEIGHT + 0.01)
  _hold(env, 60)
  assert mdp.object_in_box_bool(env, "object", bx.BOX, bx.BOX_WALLS).all()
  assert _placed(env).all()
  env.close()


def test_a_cube_on_the_rim_is_neither_in_the_box_nor_placed(monkeypatch):
  """... and one let go across the rim (its centre over the wall) is not."""
  env = _env(monkeypatch)
  _hold(env, 3)
  _put_cube(
    env, (scene.BOX_INNER / 2 + scene.BOX_WALL / 2, 0.0), scene.BOX_WALL_HEIGHT + 0.005
  )
  _hold(env, 60)
  assert not mdp.object_in_box_bool(env, "object", bx.BOX, bx.BOX_WALLS).any()
  assert not _placed(env).any()
  env.close()


def test_the_goal_follows_a_pushed_box_and_pushing_costs(monkeypatch):
  env = _env(monkeypatch)
  _hold(env, 3)
  box = env.scene[bx.BOX]
  pose = box.data.root_link_pose_w.clone()
  pose[:, 0] += 0.03
  box.write_root_link_pose_to_sim(pose)
  _hold(env, 3)
  command = env.command_manager.get_term("goal_pose")
  assert (command.box_displacement() - 0.03).abs().max() < 2e-3
  assert (
    torch.norm(command.command[:, :2] - box.data.root_link_pos_w[:, :2], dim=-1).max()
    < 1e-3
  )
  cost = mdp.box_displacement_penalty(env, "goal_pose", 0.02)
  assert (cost == 1.0).all()
  assert env.reward_manager.get_term_cfg("box_moved").weight == -2.0
  env.close()


# -- The release latch -------------------------------------------------------------------


def test_the_release_latch_refuses_a_second_surface(monkeypatch):
  env = _env(monkeypatch, num_envs=2)
  _hold(env, 2)
  jaw = SceneEntityCfg("robot", joint_names=("gripper_left_finger_joint",))
  jaw.resolve(env.scene)
  rim = bx.rim_surface()
  rim.resolve(env.scene)
  args = ("goal_pose", "object", "left_pad_contact", "object_table_contact")
  _release_latch(env, *args, rim, jaw, 0.15)
  table = SceneEntityCfg("table")
  table.resolve(env.scene)
  env.common_step_counter += 1
  with pytest.raises(ValueError, match="surface"):
    _release_latch(env, *args, table, jaw, 0.15)
  env.close()


def test_the_yaw_grade():
  """BX's M6 x G(yaw mod 90, 10 deg): aligned is M6's own grade, 10 deg off is e^-1 of it."""
  dxy, h = torch.zeros(3), torch.zeros(3)
  yaw = torch.tensor([0.0, math.radians(10), math.radians(45)])
  q = mdp.rewards.release_quality_of(
    dxy, h, 0.02, 0.01, 0.01, 0.25, yaw, math.radians(10)
  )
  assert q[0] == pytest.approx(1.0)
  assert q[1] == pytest.approx(0.25 + 0.75 * math.exp(-1.0), abs=1e-6)
  assert q[2] == pytest.approx(0.25, abs=1e-6)
  plain = mdp.rewards.release_quality_of(dxy, h, 0.02, 0.01, 0.01, 0.25, yaw, None)
  assert torch.allclose(plain, torch.ones(3))


def test_yaw_error_mod_90():
  def quat(deg):
    a = math.radians(deg) / 2
    return torch.tensor([[math.cos(a), 0.0, 0.0, math.sin(a)]])

  for deg, want in (
    (0, 0),
    (10, 10),
    (-10, 10),
    (80, 10),
    (95, 5),
    (45, 45),
    (-135, 45),
    (180, 0),
  ):
    got = math.degrees(float(mdp.yaw_error_mod_90(quat(deg))[0]))
    assert got == pytest.approx(want, abs=1e-4), deg


# -- The tracker -------------------------------------------------------------------------


def test_the_tracker_bias_is_shared_plus_own(monkeypatch):
  env = _env(monkeypatch, num_envs=64)
  bias = mdp.box._tracker_bias(env)
  assert set(bias) == {"shared", "object", bx.BOX}
  assert bias["shared"].abs().max() <= bx.TRACKER_SHARED_BIAS
  for key in ("object", bx.BOX):
    assert bias[key].abs().max() <= bx.TRACKER_OWN_BIAS
  diff = mdp.tracker_bias(env, "object") - mdp.tracker_bias(env, bx.BOX)
  assert torch.allclose(diff, bias["object"] - bias[bx.BOX])
  # Not the same draw twice: the per-axis spread of the shared part is that of U(-6, 6) mm.
  assert bias["shared"].std() > 0.002
  # The actor's goal is the box's tracked floor + the bias (+ 2 mm jitter); the critic's is live.
  _hold(env, 12)  # past the tracker's 5-8 steps of lag, the box at rest
  obs = env.observation_manager.compute()
  actor_goal, critic_goal = obs["actor"][:, -3:], obs["critic"][:, -4:-1]
  err = actor_goal - critic_goal - mdp.tracker_bias(env, bx.BOX)
  assert err.abs().max() < 0.0025
  env.close()


# -- BX-DR -------------------------------------------------------------------------------


def test_box_dr_draws_the_box_per_world(monkeypatch):
  env = _env(monkeypatch, "box-dr", num_envs=32)
  half = mdp.box_inner_half(env, bx.BOX, bx.BOX_WALLS)
  table = mdp.surface_top_z(env, SceneEntityCfg("table"))
  _hold(env, 3)
  rim = mdp.box_rim_z(env, bx.BOX, bx.BOX_WALLS) - mdp.surface_top_z(
    env, SceneEntityCfg("table")
  )
  assert (
    half.min() >= bx.BOX_DR_INNER[0] / 2 - 1e-6
    and half.max() <= bx.BOX_DR_INNER[1] / 2 + 1e-6
  )
  assert half.std() > 1e-4
  h = rim - scene.BOX_FLOOR
  assert (
    h.min() >= bx.BOX_DR_WALL_HEIGHT[0] - 1e-3
    and h.max() <= bx.BOX_DR_WALL_HEIGHT[1] + 1e-3
  )
  assert h.std() > 1e-4
  mass = env.sim.model.body_mass[:, env.scene[bx.BOX].indexing.body_ids[0]]
  assert (
    mass.min() >= bx.BOX_DR_MASS_KG[0] - 1e-3
    and mass.max() <= bx.BOX_DR_MASS_KG[1] + 1e-3
  )
  del table
  env.close()


# -- AC's sensor: the positive control ---------------------------------------------------


def test_arm_table_contact_fires_on_an_arm_pressed_by_the_table(monkeypatch):
  """Plan S10 step 0 item 2: AC must register a real arm-table contact. As S7-10's own control
  (`test_pick_place_s710.py`): env 0's table (DR-scene's per-world pose) is slid back until its
  far edge is 3 cm past arm 6's origin -- between the forearm and the gripper -- and raised 2 cm
  into the forearm's lowest collision point; env 1 keeps its table. One step: the sensor's force
  over that step exceeds AC's 1 N in env 0 only, and AC prices env 0 only."""
  from mjlab.utils.lab_api.math import quat_apply
  from pal_mjlab.tasks.manipulation.mdp.rewards import _collision_points
  from pal_mjlab.tasks.manipulation.pick_place.arm_table import ARM_TABLE_SENSOR

  env = _env(monkeypatch, num_envs=2)
  robot = env.scene["robot"]
  table = env.scene["table"]
  body = int(table.indexing.body_ids[table.body_names.index("table")])
  geom = table.indexing.geom_ids[0]
  arm5 = SceneEntityCfg("robot", body_names=("arm_left_5_link",))
  arm5.resolve(env.scene)
  i5 = arm5.body_ids[0]
  pts = _collision_points(env, robot, arm5.body_ids)[0]
  low = (
    robot.data.body_link_pos_w[0, i5]
    + quat_apply(robot.data.body_link_quat_w[0, i5].expand(len(pts), 4), pts)
  )[:, 2].min()
  x6 = robot.data.body_link_pos_w[0, robot.body_names.index("arm_left_6_link"), 0]
  half_x = env.sim.model.geom_size[:, geom].expand(env.num_envs, 3)[0, 0]
  centre_x = env.sim.data.geom_xpos[0, geom, 0]
  top = mdp.surface_top_z(env, SceneEntityCfg("table"))[0]
  env.sim.model.body_pos[0, body, 0] += (x6 + 0.03) - (centre_x + half_x)
  env.sim.model.body_pos[0, body, 2] += low - top + 0.02
  env.step(torch.zeros(env.action_space.shape, device=env.device))
  peak = (
    torch.norm(env.scene[ARM_TABLE_SENSOR].data.force_history, dim=-1)
    .flatten(1)
    .amax(-1)
  )
  term = env.reward_manager.get_term_cfg("arm_table_contact")
  cost = term.func(env, **term.params)
  assert float(peak[0]) > 1.0 and float(peak[1]) == 0.0, peak
  assert float(cost[0]) == 1.0 and float(cost[1]) == 0.0, cost
  env.close()


# -- L5: the per-step reward rises along the task, into the box ---------------------------------

_SCRIPT = pathlib.Path(__file__).parent.parent / "scripts/scripted_pick_place.py"
_SPEC = importlib.util.spec_from_file_location("scripted_pick_place_s10", _SCRIPT)
assert _SPEC and _SPEC.loader
scripted = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = scripted
_SPEC.loader.exec_module(scripted)

ORDER = ("reach", "carry", "over_target", "held_on_target", "placed")
#: c1a's hover height, as in the S5.2 / S8 ordering tests.
CARRY_HEIGHT = 0.07
VERTICAL_LOWERING = tuple(
  step
  for phase in scripted.PHASES
  for step in (
    (("lower", "retreat", True, 30), ("lower_down", "place", True, 40))
    if phase[0] == "lower"
    else (phase,)
  )
)
#: The fingertip sites' height above the table before the jaw closes (`scripted_pick_place.PAD_CLEARANCE`
#: is 1 cm). Closing drops them ~1 cm (the four-bar, S8's L16), so at 5 cm the cube is held at its
#: middle and the tips stay above the rim with the cube on the box floor. At 1-3 cm they close at the
#: table and end 1.4 cm below the rim in the box: the opening jaw presses them on the walls and the
#: friction pins the hand -- the jaw stays at 0.35 rad and the arm cannot lift (S10 report, step 0).
HIGH_GRASP = 0.05
_TERMS = ("reach", "grasp", "lift", "goal", "goal_near", "release", "placed", "retreat")


@pytest.fixture(scope="module")
def box_episode():
  """One scripted episode into the box under BX (`BS` + AC + ``GOAL=box``: G20, CTP, stack
  speeds, B, W2), every reward term of the task recorded at every step for env 0, weighted: the
  states are reached physically."""
  from pal_mjlab.tasks.manipulation.pick_place import pick_place_env_cfg as pp
  from pal_mjlab.tasks.manipulation.pick_place.arm_table import ARM_TABLE_CODES
  from pal_mjlab.tasks.manipulation.pick_place.kangaroo.env_cfg import SPAWN_ROBOTS
  from pal_mjlab.tasks.manipulation.robots import kangaroo_grippers_left_arm
  from pal_mjlab.tasks.manipulation.tabletop_env_cfg import LATENCY_MODELS

  rec = collections.defaultdict(list)
  terms: dict = {}

  def hook(phase, env):
    if not terms:
      for name in env.reward_manager.active_terms:
        terms[name] = env.reward_manager.get_term_cfg(name)
    rec[phase].append(
      {
        name: float(cfg.weight * cfg.func(env, **cfg.params)[0])
        for name, cfg in terms.items()
        if name in _TERMS
      }
    )

  saved = scripted.CLEAR_HEIGHT
  scripted.CLEAR_HEIGHT = CARRY_HEIGHT
  try:
    rows = scripted.rollout(
      num_envs=1,
      device=get_test_device(),
      reward=RCTP,
      verbose=False,
      hook=hook,
      phases=VERTICAL_LOWERING,
      robot=kangaroo_grippers_left_arm("G20", SPAWN_ROBOTS["W2"], arm_effort_scale=0.7),
      baseline=pp.BASELINES["B"],
      spawn_box=pp.SPAWN_BOXES["W2"],
      dr="DR0",
      effort=pp.EFFORT_BOUNDS["CTP"],
      latency=LATENCY_MODELS["JS"],
      stack_speed=True,
      episode_length_s=120.0,
      box_goal=bx.BOX_GOALS["box"],
      arm_table=ARM_TABLE_CODES["AC"],
      pad_clearance=HIGH_GRASP,
    )
  finally:
    scripted.CLEAR_HEIGHT = saved
  assert rows[0]["placed_and_lifted"] and rows[0]["in_box"], rows[0]
  return rec


def test_the_reward_rises_strictly_along_the_task_into_the_box(box_episode):
  total = lambda step: sum(step.values())  # noqa: E731
  # At the stack task's jaw speed (0.03 rad per step) and the jaw servo's lag the jaw passes J's
  # 0.15 rad at the end of the 30-step `open` (on the table too): the first placed step after it.
  after = box_episode["open"] + box_episode["retreat"]
  placed = [t for t in after if t["placed"] > 0][:1]
  assert placed, "the cube never counted as placed after the jaw opened"
  totals = {
    "reach": total(box_episode["descend"][-1]),
    "carry": total(box_episode["lift"][-1]),
    "over_target": total(box_episode["steady"][-1]),
    "held_on_target": total(box_episode["hold"][-1]),
    "placed": total(placed[-1]),
  }
  values = [totals[name] for name in ORDER]
  assert all(a < b for a, b in zip(values, values[1:], strict=False)), totals


def test_placed_keeps_paying_through_the_retreat_from_the_box(box_episode):
  after = box_episode["open"] + box_episode["retreat"] + box_episode["settle"]
  first = next(i for i, t in enumerate(after) if t["placed"] > 0)
  paid = [t["placed"] > 0 for t in after[first:]]
  assert sum(paid) / len(paid) > 0.9


# -- The box on the table: no bobbing (S10 report §8) ---------------------------------------


@pytest.mark.parametrize("feet", ["1", "0"])
def test_a_rocked_box_settles_on_its_feet(monkeypatch, feet):
  """A box with the cube resting in it, rocked once (1 rad/s about x, 0.5 about y, 5 cm/s down),
  must settle on the table. On its corner feet it does (sinks < 1 mm, still within 1 s); the flat
  floor every `0ba3f60` run trained on keeps bobbing 3-5 mm into the table -- kept here as the
  positive control of the measurement."""
  env = _env(monkeypatch, num_envs=16, BOX_FEET=feet)
  _hold(env, 5)
  _put_cube(env, (0.0, 0.0), 0.002)
  _hold(env, 50)
  box = env.scene[bx.BOX]
  v = box.data.root_link_vel_w.clone()
  v[:, 2], v[:, 3], v[:, 4] = -0.05, 1.0, 0.5
  box.write_root_link_velocity_to_sim(v)
  table = SceneEntityCfg("table")
  heights, speeds = [], []
  with torch.inference_mode():
    for k in range(150):
      env.step(torch.zeros(env.action_space.shape, device=env.device))
      if k >= 50:
        top = mdp.surface_top_z(env, table)
        heights.append(box.data.root_link_pos_w[:, 2] - top)
        speeds.append(box.data.root_link_lin_vel_w[:, 2].abs())
  sink = -torch.stack(heights).min()
  vz = torch.stack(speeds).max()
  if feet == "1":
    assert sink < 0.001 and vz < 0.02, (float(sink), float(vz))
  else:
    assert sink > 0.002 or vz > 0.05, (float(sink), float(vz))
  env.close()
