"""Plan S8 (Siwei, 2026-09-30 22:30-23:00): pick and place a 6 cm ball instead of the 5 cm cube.

What is checked here (``pick-place-v1-objects-plan.md``, *S8*), on the baseline the plan names -- S7-8's
plant (G20, baseline B, CTP, DR-scene, LAG=FP+JS, the stack task's speeds):

* ``SHAPE`` builds a cube (unset: S7-8's cfg exactly), a 6 cm ball, the ball with its size and rolling
  friction drawn, or one body that is a cube or a ball at random each episode; bad names and the
  combinations that are not calibrated (occlusion, O) raise;
* the object's half-height is its radius for a ball (a sphere's ``geom_size`` is ``(r, 0, 0)``, so
  the cube's ``geom_size[..., 2]`` read would have put the goal in the table): the object rests on
  the table and the goal is at the ball's resting height, at every radius the DR draws;
* the actor's yaw is the constant (1, 0) for a ball, the real yaw for a cube; J's threshold is per
  shape; the grasp sensor fires on a ball wedged between the pads (L2);
* the ball let go at 0.05 m/s on the table stops within 1 cm (the plan's rule for the rolling
  friction) -- and the table's own 0.0001 would roll it ~10 cm;
* EQ (the pads' closing height) pays at the right height before the lift and only then.
"""

import dataclasses

import mujoco
import numpy as np
import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import Scene
from pal_mjlab.tasks.manipulation import mdp, scene
from pal_mjlab.tasks.manipulation import tabletop_env_cfg as tt
from pal_mjlab.tasks.manipulation.mdp.events import surface_top_z
from pal_mjlab.tasks.manipulation.mdp.observations import (
  _RELEASE,
  _WAS_LIFTED,
  _grasped_bool,
  _jaw_open_bool,
)
from pal_mjlab.tasks.manipulation.mdp.shapes import PerShape
from pal_mjlab.tasks.manipulation.pick_place import pick_place_env_cfg as pp
from pal_mjlab.tasks.manipulation.pick_place import rewards as rw
from pal_mjlab.tasks.manipulation.pick_place.kangaroo import env_cfg as kenv
from pal_mjlab.tasks.manipulation.pick_place.rewards import S52_BASE
from pal_mjlab.tasks.manipulation.robots import (
  KANGAROO_GRIPPERS_LEFT_ARM_WIDE,
  kangaroo_grippers_left_arm,
)

RCTP = f"{S52_BASE}+M1+M2+M3+M6+J+M8"
RCTP_EQ = f"{RCTP}+EQ"
CTP = pp.EFFORT_BOUNDS["CTP"]
#: S7-8's ABS-c (the baseline S8 names), as `pick_place.sbatch` exports it.
BS_ENV = {
  "REWARD": RCTP,
  "DR": "DR-scene",
  "SLEW": "stack",
  "BASELINE": "B",
  "SPAWN": "W2",
  "GAINS": "G20",
  "EFFORT": "CTP",
  "LAG": "FP+JS",
  "ACTION": "abs",
  "CMD_OBS": "cmd",
  "EPISODE_S": "55",
}


def _env_vars(monkeypatch, **changes):
  for var, value in {**BS_ENV, **changes}.items():
    if value is None:
      monkeypatch.delenv(var, raising=False)
    else:
      monkeypatch.setenv(var, value)
  if "SHAPE" not in changes:
    monkeypatch.delenv("SHAPE", raising=False)


def _cfg(shape: str, num_envs: int = 4, reward: str = RCTP):
  cfg = pp.make_pick_place_env_cfg(
    kangaroo_grippers_left_arm(
      "G20", KANGAROO_GRIPPERS_LEFT_ARM_WIDE, arm_effort_scale=CTP.effort_scale
    ),
    gripper_name="left",
    reward=reward,
    action_slew_rate_scale=1.0,
    baseline=pp.BASELINES["B"],
    spawn=pp.SPAWN_BOXES["W2"],
    dr="DR-scene",
    effort=CTP,
    latency=tt.LATENCY_MODELS["FP+JS"],
    stack_speed=True,
    command_obs=True,
    shape=pp.OBJECT_SHAPES[shape],
  )
  cfg.scene.num_envs = num_envs
  return cfg


# -- Specs and configuration ------------------------------------------------------------------


def test_the_ball_spec_is_the_cubes_with_a_sphere_for_the_box():
  cube, ball = scene.get_cube_spec().compile(), scene.get_ball_spec().compile()
  assert [ball.body(i).name for i in range(ball.nbody)] == [
    cube.body(i).name for i in range(cube.nbody)
  ]
  assert ball.geom(0).name == cube.geom(0).name == "object_geom"
  assert ball.geom_type[0] == mujoco.mjtGeom.mjGEOM_SPHERE
  # A sphere keeps its radius in the first size slot and zeros after it.
  assert ball.geom_size[0].tolist() == [scene.BALL_RADIUS, 0.0, 0.0]
  assert ball.body_mass[1] == pytest.approx(scene.BALL_MASS)
  assert ball.geom_condim[0] == 6  # rolling friction exists only from condim 6
  assert ball.geom_friction[0, 2] == pytest.approx(scene.BALL_ROLLING_FRICTION)


def test_the_mixed_spec_has_the_cube_first_and_a_massless_ball():
  m = scene.get_mixed_spec().compile()
  assert m.ngeom == 2 and m.nbody == 2
  assert [m.geom(i).name for i in range(2)] == ["object_geom", "object_geom_ball"]
  assert list(m.geom_type) == [mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_SPHERE]
  # The body is the cube's (mass and inertia); the ball geom adds none.
  cube = scene.get_cube_spec().compile()
  assert m.body_mass[1] == pytest.approx(cube.body_mass[1])
  assert np.allclose(m.body_inertia[1], cube.body_inertia[1])


def test_the_shapes_are_named_and_unset_is_the_cube(monkeypatch):
  assert tuple(pp.OBJECT_SHAPES) == ("cube", "ball", "ball-dr", "mix")
  cube = pp.OBJECT_SHAPES["cube"]
  assert cube.objects is None and cube.jaw_open == rw.JAW_OPEN == 0.15
  assert cube.ball_probability is None and cube.rolling_scale is None
  _env_vars(monkeypatch)
  unset = kenv.kangaroo_pick_place_env_cfg()
  cube_env = pp.make_pick_place_env_cfg(
    kangaroo_grippers_left_arm(
      "G20", KANGAROO_GRIPPERS_LEFT_ARM_WIDE, arm_effort_scale=CTP.effort_scale
    ),
    gripper_name="left",
    reward=RCTP,
    baseline=pp.BASELINES["B"],
    spawn=pp.SPAWN_BOXES["W2"],
    dr="DR-scene",
    effort=CTP,
    latency=tt.LATENCY_MODELS["FP+JS"],
    stack_speed=True,
    command_obs=True,
    episode_length_s=55.0,
  )
  assert list(unset.events) == list(cube_env.events)
  assert list(unset.rewards) == list(cube_env.rewards)
  assert {k: v.params for k, v in unset.rewards.items()} == {
    k: v.params for k, v in cube_env.rewards.items()
  }
  # the cube's J threshold is the float every run up to S7-8 used, not a per-shape object
  assert unset.rewards["release"].params["jaw_open"] == 0.15


def test_a_bad_shape_raises(monkeypatch):
  _env_vars(monkeypatch, SHAPE="sphere")
  with pytest.raises(ValueError, match="SHAPE='sphere'"):
    kenv.kangaroo_pick_place_env_cfg()


@pytest.mark.parametrize("shape", ["ball", "ball-dr", "mix"])
def test_the_shape_selector_builds_the_shape(monkeypatch, shape):
  _env_vars(monkeypatch, SHAPE=shape)
  cfg = kenv.kangaroo_pick_place_env_cfg()
  assert "object" in cfg.scene.entities
  assert cfg.rewards["release"].params["jaw_open"] == pp.OBJECT_SHAPES[shape].jaw_open


def test_occlusion_with_a_ball_is_refused(monkeypatch):
  _env_vars(monkeypatch, SHAPE="ball", OCC="arm", OBJ_OBS="zero")
  with pytest.raises(ValueError, match="line of sight"):
    kenv.kangaroo_pick_place_env_cfg()


def test_the_cube_hold_angle_of_O_is_refused_for_a_ball(monkeypatch):
  _env_vars(monkeypatch, SHAPE="ball", REWARD=f"{RCTP}+O+Z")
  with pytest.raises(ValueError, match="not calibrated"):
    kenv.kangaroo_pick_place_env_cfg()


def test_the_reward_context_default_is_the_cubes_jaw_open():
  ctx = rw.RewardContext.__dataclass_fields__["jaw_open"]
  assert ctx.default == rw.JAW_OPEN


def test_ball_dr_and_mix_draw_their_shape_before_the_object_is_placed():
  for shape in ("ball-dr", "mix"):
    events = list(_cfg(shape).events)
    assert events.index("reset_object_shape") < events.index("reset_objects"), events
  assert "reset_object_shape" not in _cfg("cube").events
  assert "reset_object_shape" not in _cfg("ball").events
  ev = _cfg("mix").events["reset_object_shape"].params
  assert ev["ball_probability"] == 0.5 and ev["ball_radius"] is None
  ev = _cfg("ball-dr").events["reset_object_shape"].params
  assert ev["ball_probability"] == 1.0 and ev["ball_radius"] == pp.BALL_DR_RADIUS
  roll = _cfg("ball-dr").events["ball_rolling_friction"]
  assert roll.params["axes"] == [2] and roll.params["operation"] == "scale"
  assert roll.params["ranges"] == pp.BALL_DR_ROLLING_SCALE


def test_the_friction_dr_of_object_covers_both_geoms_of_mix():
  assert _cfg("mix").events["object_friction"].params["asset_cfg"].geom_names == (
    "object_geom",
    "object_geom_ball",
  )
  assert _cfg("ball").events["object_friction"].params["asset_cfg"].geom_names == (
    "object_geom",
  )


def test_the_S8_arm_is_a_jaw_release_arm_with_a_success_free_episode():
  assert RCTP_EQ in rw.REWARD_VARIANTS and RCTP_EQ in rw.JAW_RELEASE
  assert RCTP_EQ in rw.NO_SUCCESS_TERMINATION
  rewards = _cfg("ball", reward=RCTP_EQ).rewards
  assert set(rewards) - set(_cfg("ball").rewards) == {"pad_height"}
  assert rewards["pad_height"].weight == rw.EQ_WEIGHT
  with pytest.raises(ValueError, match="needs M1"):
    rw._s52_arm(("EQ",))


# -- The object in a live env ---------------------------------------------------------------


def _step_zero(env):
  return env.step(torch.zeros(env.action_space.shape, device=env.device))


@pytest.fixture(scope="module")
def mix_env():
  env = ManagerBasedRlEnv(cfg=_cfg("mix", num_envs=16), device=get_test_device())
  env.reset()
  yield env
  env.close()


def _resting_ok(env):
  hh = mdp.object_half_height(env)
  top = surface_top_z(env, SceneEntityCfg("table"))
  pos = env.scene["object"].data.root_link_pos_w
  goal = env.command_manager.get_term("goal_pose").command
  return hh, pos[:, 2] - top, goal[:, 2] - top


def test_mix_draws_both_shapes_and_the_object_rests_and_the_goal_sits_at_its_half_height(
  mix_env,
):
  env = mix_env
  balls = []
  for _ in range(6):
    env.reset()
    mask = mdp.ball_mask(env)
    balls.append(mask)
    hh, rest, goal = _resting_ok(env)
    assert torch.allclose(hh[mask], torch.full_like(hh[mask], scene.BALL_RADIUS))
    assert torch.allclose(hh[~mask], torch.full_like(hh[~mask], 0.025))
    assert torch.allclose(rest, hh, atol=1e-4), (rest, hh)
    assert torch.allclose(goal, hh, atol=1e-4), (goal, hh)
  share = torch.cat(balls).float().mean().item()
  assert 0.3 < share < 0.7, share  # 96 draws at p = 0.5
  assert mdp.shape_kind(env) == "mixed"


def test_the_shape_that_is_off_is_a_speck_inside_the_one_that_is_on(mix_env):
  env = mix_env
  env.reset()
  mask = mdp.ball_mask(env)
  size = env.sim.model.geom_size
  cube_g, ball_g = (
    mdp.shapes._object_geoms(env, "object")[0][0],
    mdp.shapes._object_geoms(env, "object")[1][0],
  )
  off = scene.INACTIVE_SIZE
  assert torch.allclose(size[mask, cube_g], torch.full_like(size[mask, cube_g], off))
  assert torch.allclose(
    size[~mask, ball_g, 0], torch.full_like(size[~mask, ball_g, 0], off)
  )
  # The bounds the broadphase reads follow the sizes.
  rbound = env.sim.model.geom_rbound
  assert torch.allclose(rbound[mask, ball_g], size[mask, ball_g, 0])
  assert torch.allclose(
    rbound[~mask, cube_g], size[~mask, cube_g].norm(dim=-1), atol=1e-6
  )


def test_the_actor_sees_a_constant_yaw_for_a_ball_and_the_real_one_for_a_cube(mix_env):
  env = mix_env
  env.reset()
  mask = mdp.ball_mask(env)
  yaw = mdp.object_yaw_b(env, "object")
  assert torch.equal(
    yaw[mask], torch.tensor([1.0, 0.0], device=yaw.device).expand(int(mask.sum()), 2)
  )
  # A cube's unit vector is its yaw: W2 draws it over +/-45 degrees, so not all are (1, 0).
  assert torch.allclose(yaw.norm(dim=-1), torch.ones_like(yaw[:, 0]), atol=1e-5)
  assert bool((yaw[~mask, 1].abs() > 1e-3).any())


def test_J_threshold_is_per_shape_per_env(mix_env):
  env = mix_env
  env.reset()
  mask = mdp.ball_mask(env)
  thr = mdp.resolve(env, PerShape(cube=rw.JAW_OPEN, ball=rw.BALL_JAW_OPEN))
  assert torch.allclose(thr[mask], torch.full_like(thr[mask], rw.BALL_JAW_OPEN))
  assert torch.allclose(thr[~mask], torch.full_like(thr[~mask], rw.JAW_OPEN))
  # A float stays a float; a cube's number is exactly what it was.
  assert mdp.resolve(env, 0.15) == 0.15
  # `_jaw_open_bool` compares each env's jaw with its own threshold: put the jaw between them.
  robot = env.scene["robot"]
  jaw = SceneEntityCfg("robot", joint_names=("gripper_left_finger_joint",))
  jaw.resolve(env.scene)
  between = 0.5 * (rw.BALL_JAW_OPEN + rw.JAW_OPEN)
  q = robot.data.joint_pos.clone()
  q[:, jaw.joint_ids[0]] = between
  robot.write_joint_position_to_sim(q)
  opened = _jaw_open_bool(env, jaw, PerShape(cube=rw.JAW_OPEN, ball=rw.BALL_JAW_OPEN))
  assert bool(opened[~mask].all()) and not bool(opened[mask].any())


@pytest.fixture(scope="module")
def ball_dr_env():
  env = ManagerBasedRlEnv(cfg=_cfg("ball-dr", num_envs=24), device=get_test_device())
  env.reset()
  yield env
  env.close()


def test_ball_dr_draws_the_radius_per_episode_and_the_goal_follows_it(ball_dr_env):
  env = ball_dr_env
  radii = []
  for _ in range(5):
    env.reset()
    hh, rest, goal = _resting_ok(env)
    radii.append(hh.clone())  # `object_half_height` is a view of the model's geom_size
    assert torch.allclose(rest, hh, atol=1e-4)
    assert torch.allclose(goal, hh, atol=1e-4)
  r = torch.cat(radii)
  lo, hi = pp.BALL_DR_RADIUS
  assert r.min() >= lo - 1e-6 and r.max() <= hi + 1e-6
  # 120 uniform draws cover the range to within 10 % at each end, and differ across episodes.
  assert r.min() < lo + 0.1 * (hi - lo) and r.max() > hi - 0.1 * (hi - lo)
  assert not torch.allclose(radii[0], radii[1])
  # The rolling friction is a per-world factor on the nominal value, inside the range.
  ball_g = mdp.shapes._object_geoms(env, "object")[0][0]
  f = env.sim.model.geom_friction[:, ball_g, 2] / scene.BALL_ROLLING_FRICTION
  lo_s, hi_s = pp.BALL_DR_ROLLING_SCALE
  assert f.min() >= lo_s - 1e-4 and f.max() <= hi_s + 1e-4 and f.std() > 0.1


def test_the_grasp_sensor_fires_on_a_ball_between_the_pads_and_not_before(ball_dr_env):
  """L2 (plan S8): the sensor watches pad geoms against the object's body, so it does for a sphere.
  The ball is put between the open pads, its weight supported as the gripper bench does, and the jaw
  is closed through the action term: no grasp while the pads are open, a grasp once they close."""
  env = ball_dr_env
  env.reset()
  robot, obj = env.scene["robot"], env.scene["object"]
  sites = [
    robot.site_names.index(f"gripper_left_fingertip_{side}_site")
    for side in ("left", "right")
  ]
  term = env.action_manager.get_term("joint_pos")
  jaw = [n.split("/", 1)[-1] for n in term.target_names].index(
    "gripper_left_finger_joint"
  )
  for _ in range(
    30
  ):  # 0.6 s: the arm settles from the reset's jitter before the ball is placed
    _step_zero(env)
  mid = robot.data.site_pos_w[:, sites].mean(dim=1)
  pose = obj.data.root_link_pose_w.clone()
  pose[:, :3] = mid
  pose[:, 2] += (
    0.008  # the pad sites' midpoint 8 mm below the ball's centre, inside the pads
  )
  pose[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
  obj.write_root_link_pose_to_sim(pose)
  obj.write_root_link_velocity_to_sim(torch.zeros_like(obj.data.root_link_vel_w))
  weight = torch.zeros(env.num_envs, 1, 3, device=env.device)
  # Each world's own ball: DR-object scales its mass (30-100 g).
  weight[:, 0, 2] = env.sim.model.body_mass[:, obj.indexing.body_ids[0]] * 9.81
  no_torque = torch.zeros_like(weight)
  sensor = robot_cfg_sensor(env)

  def step(jaw_action: float):
    a = torch.zeros(env.action_space.shape, device=env.device)
    a[:, jaw] = jaw_action
    obj.write_external_wrench_to_sim(weight, no_torque)
    env.step(a)

  step(0.0)
  # The ball (<= 3.25 cm radius) in the 79 mm open gap: no pad touches it yet.
  assert not bool(_grasped_bool(env, sensor).any())
  for _ in range(
    30
  ):  # 0.03 rad per step and the jaw servo's lag: past the ball's touch angle
    step(1.0)
  held = _grasped_bool(env, sensor)
  obj.write_external_wrench_to_sim(torch.zeros_like(weight), no_torque)
  assert bool(held.all()), (
    f"a ball closed on by the jaw is not grasped: {held.tolist()}"
  )


def robot_cfg_sensor(env) -> str:
  from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS_LEFT_ARM

  return KANGAROO_GRIPPERS_LEFT_ARM.gripper("left").contact_sensor_name


# -- A slid ball, a thrown ball ---------------------------------------------------------------

#: The terms a slide must not pay: everything but the approach.
GATED_TERMS = (
  "grasp",
  "lift",
  "goal",
  "goal_near",
  "release",
  "placed",
  "retreat",
  "pad_contact",
)


def _term(env, name: str) -> torch.Tensor:
  cfg = env.reward_manager.get_term_cfg(name)
  return cfg.func(env, **cfg.params)


@pytest.fixture(scope="module")
def slid_ball_env():
  """L4: the ball pushed onto the target along the table, never lifted: at rest on the goal, not
  held, on the table. Stepped after the teleport, since contact sensors are empty until a step."""
  env = ManagerBasedRlEnv(cfg=_cfg("ball", num_envs=2), device=get_test_device())
  env.reset()
  zero = torch.zeros(env.action_space.shape, device=env.device)
  for _ in range(3):
    env.step(zero)
  obj = env.scene["object"]
  pose = obj.data.root_link_pose_w.clone()
  pose[:, :3] = env.command_manager.get_term("goal_pose").command
  obj.write_root_link_pose_to_sim(pose)
  obj.write_root_link_velocity_to_sim(torch.zeros_like(obj.data.root_link_vel_w))
  env.step(zero)
  yield env
  env.close()


def test_a_slid_ball_earns_nothing_past_reach(slid_ball_env):
  env = slid_ball_env
  assert not bool(getattr(env, _WAS_LIFTED).any()), "the ball was never lifted"
  assert torch.all(_term(env, "reach") > 0.0)
  for name in GATED_TERMS:
    value = _term(env, name)
    assert torch.all(value == 0.0), f"{name} paid a slide: {value}"


def test_the_same_ball_pays_once_it_has_been_lifted_and_let_go(slid_ball_env):
  """Positive control: a term that returns zero because it is broken also 'passes' the slide
  test. With the latches set as a set-down 2 mm off and 1 mm up would set them, `release` and
  `placed` pay."""
  env = slid_ball_env
  was_lifted, latch = getattr(env, _WAS_LIFTED), getattr(env, _RELEASE)
  saved = (was_lifted.clone(), latch.released.clone(), latch.dxy, latch.height)
  was_lifted[:] = True
  latch.released[:] = True
  latch.dxy = torch.full_like(latch.dxy, 0.002)
  latch.height = torch.full_like(latch.height, 0.001)
  try:
    for name in ("release", "placed"):
      value = _term(env, name)
      assert torch.all(value > 0.9), (
        name,
        value,
      )  # M6's grade of a set-down, 0.25 at the floor
  finally:
    was_lifted[:] = saved[0]
    latch.released[:] = saved[1]
    latch.dxy, latch.height = saved[2], saved[3]


def test_a_thrown_ball_earns_less_than_one_set_down_and_rolls_away():
  """Plan S8, test 6: a ball released 5 cm up at 0.3 m/s earns less than one set down on the goal.
  On a table a released ball keeps its horizontal speed: this one rolls ~30 cm, so the throw ends
  outside `placed`'s tolerance for good, where a set-down is paid every step."""
  env = ManagerBasedRlEnv(cfg=_cfg("ball", num_envs=2), device=get_test_device())
  try:
    env.reset()
    zero = torch.zeros(env.action_space.shape, device=env.device)
    for _ in range(3):
      env.step(zero)
    obj = env.scene["object"]
    goal = env.command_manager.get_term("goal_pose").command
    pose = obj.data.root_link_pose_w.clone()
    pose[:, :3] = goal
    pose[1, 2] += 0.05
    obj.write_root_link_pose_to_sim(pose)
    vel = torch.zeros_like(obj.data.root_link_vel_w)
    vel[1, 0] = 0.3
    obj.write_root_link_velocity_to_sim(vel)
    # As a J-release would leave the latches: lifted, let go at the goal (env 0 on the table, env 1
    # 5 cm up).
    getattr(env, _WAS_LIFTED)[:] = True
    env.step(zero)
    latch = getattr(env, _RELEASE)
    latch.released[:] = True
    latch.dxy = torch.zeros_like(latch.dxy)
    latch.height = torch.tensor([0.001, 0.05], device=env.device)
    earned = torch.zeros(2, device=env.device)
    for _ in range(150):  # 3 s
      env.step(zero)
      earned += env.reward_manager.get_term_cfg("placed").weight * _term(env, "placed")
    final = (obj.data.root_link_pos_w - goal)[:, :2].norm(dim=-1)
    assert final[0] < 0.01 and final[1] > 0.10, final.tolist()
    assert earned[1] < 0.1 * earned[0], earned.tolist()
    assert earned[0] > 100.0, earned.tolist()
  finally:
    env.close()


# -- Rolling -----------------------------------------------------------------------------------


def _roll_model(rolling: float | None):
  """The task's own scene and solver settings, with the ball's rolling friction set."""
  cfg = _cfg("ball", num_envs=1)
  if rolling is not None:
    cfg.scene.entities["object"] = dataclasses.replace(
      cfg.scene.entities["object"],
      spec_fn=lambda: scene.get_ball_spec(rolling=rolling),
    )
  model = Scene(cfg.scene, "cpu").compile()
  cfg.sim.mujoco.apply(model)
  return model


def _roll(model, v0: float, spinning: bool, seconds: float = 4.0) -> float:
  """How far (m) the ball goes along +x when it is let go at ``v0`` on the table."""
  import sys
  from pathlib import Path

  sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
  from scene_model import nominal_data

  data = nominal_data(model)
  body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object/object")
  joint = next(j for j in range(model.njnt) if model.jnt_bodyid[j] == body)
  qa, da = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
  top_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table/table_top")
  top = data.geom_xpos[top_geom][2] + model.geom_size[top_geom][2]
  x0 = np.array([0.50, 0.15, top + scene.BALL_RADIUS])
  data.qpos[qa : qa + 7] = (*x0, 1, 0, 0, 0)
  data.qvel[da : da + 6] = 0
  data.qvel[da] = v0
  if spinning:  # rolling without slipping: omega_y = v / r
    data.qvel[da + 4] = v0 / scene.BALL_RADIUS
  mujoco.mj_forward(model, data)
  for _ in range(int(seconds / model.opt.timestep)):
    mujoco.mj_step(model, data)
  return float(data.qpos[qa] - x0[0])


@pytest.mark.parametrize("spinning", [False, True])
def test_the_ball_let_go_at_5_cm_s_stops_in_1_cm(spinning):
  """The plan's rule for the ball's rolling friction, checked on the task's scene and solver:
  let go with no spin (a carried ball) and already rolling (the stricter start)."""
  assert _roll(_roll_model(None), 0.05, spinning) < 0.01


def test_the_tables_own_rolling_friction_would_roll_it_ten_centimetres():
  """Why the ball carries its own: the contact takes the element-wise max of the two geoms'
  friction, so a ball with less than the table's 0.0001 is rolling on the table's."""
  assert _roll(_roll_model(1e-4), 0.05, True) > 0.10
