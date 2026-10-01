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

import math

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
  lo = (d.geom_xpos - m.geom_size).min(axis=0)
  hi = (d.geom_xpos + m.geom_size).max(axis=0)
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
  """Plan S10 step 0 item 2: AC must register a real arm-table contact. At the start pose arm
  1-6 is behind the table's near edge (arm 5 at x ~0.06 m, the table from 0.25 m), so env 0's
  table (DR-scene's per-world table pose) is moved 15 cm towards the robot and up until its top
  is 1 cm above arm 5's link origin less 3 cm; env 1 keeps its table. Only env 0 is priced."""
  from pal_mjlab.tasks.manipulation.pick_place.arm_table import ARM_TABLE_SENSOR

  env = _env(monkeypatch, num_envs=2)
  robot = env.scene["robot"]
  table = env.scene["table"]
  tid = table.indexing.body_ids[table.body_names.index("table")]
  top = float(mdp.surface_top_z(env, SceneEntityCfg("table"))[0])
  forearm = robot.body_names.index("arm_left_5_link")
  low = float(robot.data.body_link_pos_w[0, forearm, 2]) - 0.03
  env.sim.model.body_pos[0, tid, 0] -= 0.15
  env.sim.model.body_pos[0, tid, 2] += low + 0.01 - top
  env.sim.forward()
  _hold(env, 10)
  found = env.scene[ARM_TABLE_SENSOR].data.found.bool().any(dim=-1)
  term = env.reward_manager.get_term_cfg("arm_table_contact")
  cost = term.func(env, **term.params)
  assert bool(found[0]) and not bool(found[1]), found
  assert float(cost[0]) == 1.0 and float(cost[1]) == 0.0
  env.close()
