"""The mug model, and the geometry the mug tasks' rewards read off it."""

import math

import mujoco
import pytest
import torch
from pal_mjlab.tasks.manipulation import scene
from pal_mjlab.tasks.manipulation.mdp.events import _mug_lying_roll
from pal_mjlab.tasks.manipulation.mdp.mug_rewards import handle_heading_error


def test_mug_matches_the_isaac_collider():
  # Mug018_solid_collision.usd under the Isaac env's (1.3, 1.3, 2.0) scale.
  assert scene.MUG_WIDTH == pytest.approx(0.065)
  assert 2 * scene.MUG_HALF_HEIGHT == pytest.approx(0.123)
  assert scene.MUG_HANDLE_REACH == pytest.approx(0.065)
  # The jaw opens to 79.3 mm between the pads.
  assert scene.MUG_WIDTH < 0.0793


def test_mug_mass_and_first_geom():
  model = scene.get_mug_spec().compile()
  assert model.body_mass[1] == pytest.approx(scene.MUG_MASS)
  # The placement code reads the first geom as the body.
  assert model.geom_type[0] == mujoco.mjtGeom.mjGEOM_CYLINDER


def test_lying_mug_rests_on_body_and_handle():
  assert _mug_lying_roll() == pytest.approx(math.radians(60.0))


class _Obj:
  class data:
    root_link_quat_w: torch.Tensor


def _mug_at_yaw(yaw: float) -> _Obj:
  obj = _Obj()
  obj.data = _Obj.data()
  obj.data.root_link_quat_w = torch.tensor(
    [[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]]
  )
  return obj


@pytest.mark.parametrize(
  ("yaw", "error"),
  [(math.pi, 0.0), (-math.pi, 0.0), (0.0, math.pi), (math.pi / 2, math.pi / 2)],
)
def test_handle_heading_error_wraps(yaw, error):
  got = handle_heading_error(_mug_at_yaw(yaw), math.pi)
  assert got.item() == pytest.approx(error, abs=1e-5)


def test_turning_kept_pays_only_the_turn_made_in_the_jaw(monkeypatch):
  from pal_mjlab.tasks.manipulation.mdp import mug_rewards

  held = torch.tensor([False])
  monkeypatch.setattr(mug_rewards, "_grasped_any_bool", lambda env, names: held)

  class _Env:
    num_envs = 1
    device = "cpu"
    scene: dict

  env = _Env()
  term = mug_rewards.mug_turning_kept(None, env)

  def pay(yaw: float, grasped: bool) -> float:
    held[0] = grasped
    env.scene = {"mug": _mug_at_yaw(yaw)}
    return term(env, "mug", (), math.pi).item()

  def heading(yaw: float) -> float:
    return (1 + math.cos(math.pi - yaw)) / 2

  assert pay(math.pi / 2, grasped=False) == 0.0  # nothing before the first grasp
  assert pay(math.pi / 2, grasped=True) == pytest.approx(heading(math.pi / 2))
  assert pay(2.5, grasped=True) == pytest.approx(heading(2.5))
  # Let go: the turn made in the jaw stays paid, pushing it further does not pay.
  assert pay(2.5, grasped=False) == pytest.approx(heading(2.5))
  assert pay(math.pi, grasped=False) == pytest.approx(heading(2.5))
  # Turning back costs what it gives up.
  assert pay(2.0, grasped=False) == pytest.approx(heading(2.0))
  term.reset()
  assert pay(math.pi, grasped=False) == 0.0


def test_turn_spawn_is_30_to_120_deg_off_either_side():
  from pal_mjlab.tasks.manipulation.mug.mug_env_cfg import (
    TURN_TARGET_YAW,
    mug_turn_env_cfg,
  )

  params = mug_turn_env_cfg().events["reset_objects"].params
  low, high = params["pose_range"]["yaw"]
  mirror = params["yaw_mirror"]
  for yaw in (low, high, 2 * mirror - low, 2 * mirror - high):
    off = math.degrees(abs(yaw - TURN_TARGET_YAW))
    assert 30.0 - 1e-6 <= off <= 120.0 + 1e-6
  assert mirror == TURN_TARGET_YAW


def test_set_down_survives_the_release_rocking_and_holding_stops_paying(monkeypatch):
  from pal_mjlab.tasks.manipulation.mdp import mug_rewards

  held, on_table, clear = (torch.tensor([False]) for _ in range(3))
  monkeypatch.setattr(mug_rewards, "_grasped_any_bool", lambda env, names: held)
  monkeypatch.setattr(mug_rewards, "_on_surface_bool", lambda env, name: on_table)
  monkeypatch.setattr(mug_rewards, "_jaw_clear", lambda env, cfg, obj, c: clear)
  monkeypatch.setattr(mug_rewards, "surface_top_z", lambda env, cfg: torch.zeros(1))

  class _Env:
    num_envs = 1
    device = "cpu"
    step_dt = 0.5
    scene: dict

  env = _Env()
  mug = _mug_at_yaw(0.0)
  mug.data.root_link_pos_w = torch.zeros(1, 3)
  env.scene = {"mug": mug}
  robust = {"surface_cfg": "table", "pickup_height": 0.02}

  def make():
    return (
      mug_rewards.mug_stood_upright(None, env),
      mug_rewards.mug_stood_upright(None, env),
    )

  def step(terms, height, grasped, touching, jaw_clear=False, **kw) -> tuple:
    held[0], on_table[0], clear[0] = grasped, touching, jaw_clear
    mug.data.root_link_pos_w[0, 2] = scene.MUG_HALF_HEIGHT + height
    stood, released = terms
    args = (env, "mug", (), "table", None, math.radians(10.0))
    return (
      stood(*args, released=False, hold_time=1.0, **kw).item(),
      released(*args, released=True, **kw).item(),
    )

  for kw in ({}, robust):
    terms = make()
    step(terms, 0.10, grasped=True, touching=False, **kw)  # lifted
    # Set down in the jaw: holding it standing pays for hold_time (two steps), then not.
    assert [step(terms, 0.0, True, True, **kw)[0] for _ in range(4)] == [1, 1, 0, 0]
    # The jaw opens and the mug rocks 3 mm off the table, then settles; jaw clear.
    step(terms, 0.003, grasped=False, touching=False, **kw)
    paid = step(terms, 0.0, grasped=False, touching=True, jaw_clear=True, **kw)[1]
    # Only the robust latch keeps the set-down through the rocking.
    assert paid == (1.0 if kw else 0.0)

  # Picked up again: the set-down clears, so standing it free pays nothing.
  terms = make()
  step(terms, 0.10, grasped=True, touching=False, **robust)
  step(terms, 0.0, grasped=True, touching=True, **robust)
  step(terms, 0.05, grasped=True, touching=False, **robust)
  assert step(terms, 0.0, False, True, jaw_clear=True, **robust)[1] == 0.0


def test_upright_release_rewards_are_ordered():
  from pal_mjlab.tasks.manipulation.mug import mug_env_cfg as m

  cfg = m.mug_upright_env_cfg(lower=True, release=True)
  r = cfg.rewards
  assert r["stood"].params["hold_time"] == m.STOOD_HOLD_TIME
  assert r["lowered"].params["hold_time"] == m.STOOD_HOLD_TIME
  # Open (half the rung) beats the grasp ladder kept by holding on; clear beats all.
  assert 0.5 * r["stood_releasing"].weight > r["grasp"].weight + r["secure"].weight
  assert r["stood_released"].weight > r["stood"].weight + r["lowered"].weight
  for name in ("stood", "stood_released", "stood_releasing"):
    assert r[name].params["pickup_height"] == m.STOOD_PICKUP_HEIGHT
  plain = m.mug_upright_env_cfg(lower=True).rewards
  assert "stood_releasing" not in plain
  assert "pickup_height" not in plain["stood_released"].params


def _shaping_env(monkeypatch, held, on_table):
  from pal_mjlab.tasks.manipulation.mdp import mug_rewards

  monkeypatch.setattr(mug_rewards, "_grasped_any_bool", lambda env, names: held)
  monkeypatch.setattr(mug_rewards, "_on_surface_bool", lambda env, name: on_table)
  monkeypatch.setattr(mug_rewards, "surface_top_z", lambda env, cfg: torch.zeros(1))
  # The jaw: fully withdrawn (1) when nothing holds the mug, else on it (0).
  monkeypatch.setattr(
    mug_rewards,
    "_release_progress",
    lambda env, cfg, obj, *a: (~held).float(),
  )

  class _Env:
    num_envs = 1
    device = "cpu"
    step_dt = 0.02
    scene: dict

  env = _Env()
  mug = _mug_at_yaw(0.0)
  mug.data.root_link_pos_w = torch.zeros(1, 3)
  env.scene = {"mug": mug}
  return mug_rewards, env, mug


def _tilted(roll: float, yaw: float = 0.0) -> torch.Tensor:
  """Quaternion of a yaw about z, then a roll about the mug's x."""
  cy, sy, cr, sr = (
    math.cos(yaw / 2),
    math.sin(yaw / 2),
    math.cos(roll / 2),
    math.sin(roll / 2),
  )
  return torch.tensor([[cy * cr, cy * sr, sy * sr, sy * cr]])


def test_upright_progress_cannot_be_farmed_by_lifting_again(monkeypatch):
  held, on_table = torch.tensor([False]), torch.tensor([True])
  mug_rewards, env, mug = _shaping_env(monkeypatch, held, on_table)
  term = mug_rewards.mug_upright_progress(None, env)
  args = (env, "mug", (), "support", None, "table", math.radians(10.0))

  def step(height, roll, grasped, touching, gamma=1.0):
    held[0], on_table[0] = grasped, touching
    mug.data.root_link_pos_w[0, 2] = scene.MUG_HALF_HEIGHT + height
    mug.data.root_link_quat_w = _tilted(roll)
    return term(*args, gamma=gamma).item() * env.step_dt

  lying = math.pi / 2
  assert step(0.0, lying, False, True) == 0.0  # first step only sets phi
  assert step(0.0, lying, True, True) == 0.0  # gripped on the table: not lifted yet
  total = step(0.10, lying, True, False)  # lifted: the uprightness term starts
  assert total == pytest.approx(0.4 * 0.25)
  total += step(0.10, 0.0, True, False)  # stood up in the air
  total += step(0.0, 0.0, True, True)  # set down standing
  assert total == pytest.approx(0.7)
  # Lifting it again and setting it back down nets nothing, however often.
  cycle = sum(step(h, 0.0, True, h > 0) for _ in range(3) for h in (0.05, 0.0))
  assert cycle == pytest.approx(0.0, abs=1e-6)
  total += step(0.0, 0.0, False, True)  # let go, jaw clear
  assert total == pytest.approx(1.0)
  # Discounted, staying put at phi costs (1 - gamma) * phi a step.
  assert step(0.0, 0.0, False, True, gamma=0.99) == pytest.approx(-0.01)
  assert term.latches().tolist() == [[1.0, 1.0]]
  term.reset()
  assert term.latches().tolist() == [[0.0, 0.0]]


def test_upright_progress_ignores_a_mug_pushed_upright(monkeypatch):
  held, on_table = torch.tensor([False]), torch.tensor([True])
  mug_rewards, env, mug = _shaping_env(monkeypatch, held, on_table)
  term = mug_rewards.mug_upright_progress(None, env)
  mug.data.root_link_pos_w[0, 2] = scene.MUG_HALF_HEIGHT
  for roll in (math.pi / 2, 0.0):
    mug.data.root_link_quat_w = _tilted(roll)
    paid = term(env, "mug", (), "support", None, "table", 0.2, gamma=1.0).item()
    assert paid == 0.0


def test_turn_progress_pays_the_turn_not_the_spawn_heading(monkeypatch):
  held, on_table = torch.tensor([False]), torch.tensor([True])
  mug_rewards, env, mug = _shaping_env(monkeypatch, held, on_table)
  term = mug_rewards.mug_turn_progress(None, env)
  tol = math.radians(25.0)

  def step(yaw, grasped):
    held[0] = grasped
    mug.data.root_link_quat_w = _tilted(0.0, yaw)
    paid = term(env, "mug", (), "support", None, math.pi, tol, 0.26, gamma=1.0)
    return paid.item() * env.step_dt

  spawn = math.pi - math.radians(30.0)  # 30 deg off: the old turning paid 0.93 here
  assert step(spawn, False) == 0.0
  assert step(spawn, True) == pytest.approx(0.0)  # the first grasp pays nothing
  # Turned 30 deg, back 20 and forward again: only the net turn pays.
  turned = sum(step(math.pi - math.radians(d), True) for d in (20, 10, 0, 20, 0))
  assert turned == pytest.approx(0.7 * 30 / 180)
  assert step(math.pi, False) == pytest.approx(0.3)  # let go, jaw clear
  assert term.latches().tolist() == [[1.0, 1.0]]


def test_stage_variants_keep_only_the_end_state_per_step():
  from pal_mjlab.tasks.manipulation.mug import mug_env_cfg as m

  up = m.mug_upright_env_cfg(stage=True)
  assert {"lift", "upright_held", "stood", "lowered"}.isdisjoint(up.rewards)
  assert up.rewards["stood_released"].params["pickup_height"] == m.STOOD_PICKUP_HEIGHT
  turn = m.mug_turn_env_cfg(stage=True)
  assert {"turning", "releasing"}.isdisjoint(turn.rewards)
  assert turn.rewards["turned"].params["progress_term"] == m.PROGRESS_TERM
  assert "progress_term" not in turn.rewards[m.PROGRESS_TERM].params
  for cfg in (up, turn):
    assert cfg.rewards[m.PROGRESS_TERM].params["gamma"] == m.PATIENT_GAMMA
    assert "progress_latches" in cfg.observations["critic"].terms
    assert "progress_latches" not in cfg.observations["actor"].terms


def test_progress_ending_the_episode_pays_back_the_potential(monkeypatch):
  held, on_table = torch.tensor([True]), torch.tensor([True])
  mug_rewards, env, mug = _shaping_env(monkeypatch, held, on_table)
  term = mug_rewards.mug_turn_progress(None, env)

  class _Terminations:
    terminated = torch.tensor([False])

  env.termination_manager = _Terminations()

  def step(yaw, gamma=0.99):
    mug.data.root_link_quat_w = _tilted(0.0, yaw)
    paid = term(env, "mug", (), "support", None, math.pi, 0.4, 0.26, gamma=gamma)
    return paid.item() * env.step_dt

  step(math.pi / 2)
  phi = 0.7 * 0.5
  assert step(math.pi / 2) == pytest.approx(
    -0.01 * phi, rel=1e-4
  )  # staying costs a little
  _Terminations.terminated = torch.tensor([True])
  assert step(math.pi / 2) == pytest.approx(-phi, rel=1e-4)  # quitting costs it all


@pytest.mark.parametrize(
  ("held", "on_table", "roll", "ends"),
  [
    (False, True, math.radians(90.0), True),  # knocked over on the table
    (False, True, math.radians(5.0), False),  # standing on the table
    (False, False, math.radians(90.0), False),  # tilted between the two jaws
    (True, True, math.radians(90.0), False),  # held tilted against the table
    (True, False, math.radians(90.0), False),  # carried tilted
  ],
)
def test_tip_ends_the_episode_only_for_a_free_mug_on_the_table(
  monkeypatch, held, on_table, roll, ends
):
  from pal_mjlab.tasks.manipulation.mdp import terminations
  from pal_mjlab.tasks.manipulation.mug.mug_env_cfg import TURN_TIP_LIMIT

  monkeypatch.setattr(
    terminations, "_grasped_any_bool", lambda env, names: torch.tensor([held])
  )
  monkeypatch.setattr(
    terminations, "_on_surface_bool", lambda env, name: torch.tensor([on_table])
  )
  mug = _Obj()
  mug.data = _Obj.data()
  # Gravity in the body frame of a mug rolled by ``roll`` about its x.
  mug.data.projected_gravity_b = torch.tensor([[0.0, -math.sin(roll), -math.cos(roll)]])

  class _Env:
    scene = {"mug": mug}

  got = terminations.object_tipped_free(
    _Env(), "mug", [["s"]], "table", limit_angle=TURN_TIP_LIMIT
  )
  assert got.tolist() == [ends]


@pytest.mark.parametrize(
  ("faces", "on_table", "vz", "ends"),
  [
    ((False, False, False, False), False, -0.8, True),  # let go in the air
    ((False, False, True, False), False, -0.8, False),  # slipping, one face holds
    ((False, False, False, False), True, -0.8, False),  # landing on the table
    ((False, False, False, False), False, -0.1, False),  # not falling
  ],
)
def test_drop_ends_only_a_mug_no_jaw_face_touches(
  monkeypatch, faces, on_table, vz, ends
):
  from pal_mjlab.tasks.manipulation.mdp import terminations

  names = [["la", "lb"], ["ra", "rb"]]
  touching = dict(zip(["la", "lb", "ra", "rb"], faces, strict=True))
  monkeypatch.setattr(
    terminations, "_face_touching", lambda env, n: torch.tensor([touching[n]])
  )
  monkeypatch.setattr(
    terminations, "_on_surface_bool", lambda env, name: torch.tensor([on_table])
  )
  mug = _Obj()
  mug.data = _Obj.data()
  mug.data.root_link_lin_vel_w = torch.tensor([[0.0, 0.0, vz]])

  class _Env:
    scene = {"mug": mug}

  got = terminations.object_dropped_free(_Env(), "mug", names, "table")
  assert got.tolist() == [ends]


@pytest.mark.parametrize(
  ("approach_z", "grasped", "gate"),
  [
    (-1.0, True, 1.0),  # fingers straight down
    (-0.2, True, 1.0),
    (0.0, True, 0.5),  # level
    (0.2, True, 0.0),
    (1.0, True, 0.0),  # fingers up, the hand under the mug
    (-1.0, False, 0.0),  # not holding
  ],
)
def test_overhand_gate_pays_only_a_hold_that_can_set_the_mug_down(
  monkeypatch, approach_z, grasped, gate
):
  from mjlab.managers.scene_entity_config import SceneEntityCfg
  from pal_mjlab.tasks.manipulation.mdp import mug_rewards

  side = math.sqrt(1.0 - approach_z**2)
  mid = torch.tensor([[[side, 0.0, approach_z]]])  # (B, G, 3), the TCP at the origin
  monkeypatch.setattr(mug_rewards, "_jaw_frame", lambda robot, cfg: (mid, None, None))
  monkeypatch.setattr(
    mug_rewards, "_grasped_bool", lambda env, pair: torch.tensor([grasped])
  )

  class _Robot:
    class data:
      site_pos_w = torch.zeros(1, 1, 3)

  class _Env:
    scene = {"robot": _Robot()}

  tcp = SceneEntityCfg("robot", site_names=("tcp",))
  tcp.site_ids = [0]
  got = mug_rewards._overhand_gate(_Env(), SceneEntityCfg("robot"), tcp, [["a", "b"]])
  assert got.item() == pytest.approx(gate)


@pytest.mark.parametrize(("hand_on_mug", "paid"), [(False, 1.0), (True, 0.0)])
def test_hands_off_turned_pays_only_with_the_hand_off_the_mug(
  monkeypatch, hand_on_mug, paid
):
  from pal_mjlab.tasks.manipulation.mdp import mug_rewards

  held = torch.tensor([True])
  monkeypatch.setattr(mug_rewards, "_grasped_any_bool", lambda env, names: held)
  monkeypatch.setattr(
    mug_rewards,
    "_on_surface_bool",
    lambda env, name: torch.tensor([hand_on_mug if name == "hand" else True]),
  )
  # Every jaw midpoint ``clearance`` from the mug, as with a finger on the rim.
  monkeypatch.setattr(
    mug_rewards, "_jaw_clear", lambda env, cfg, obj, c: torch.tensor([True])
  )

  class _Env:
    num_envs = 1
    device = "cpu"
    scene = {"mug": _mug_at_yaw(math.pi)}

  env = _Env()
  term = mug_rewards.mug_turned(None, env)
  args = (env, "mug", (), "support", None, math.pi, 0.4, 0.26)
  assert term(*args, hand_sensor_name="hand").item() == 0.0  # grasped: latches
  held[0] = False
  assert term(*args, hand_sensor_name="hand").item() == paid
  assert term(*args).item() == 1.0  # without the sensor, as before


def test_turn_hands_off_and_near_variants():
  from pal_mjlab.tasks.manipulation.mug import mug_env_cfg as m

  cfg = m.mug_turn_env_cfg(release=True, regrasp=True, hands_off=True, near=True)
  assert m.HAND_OBJECT_SENSOR in [s.name for s in cfg.scene.sensors]
  for name in ("turned", "releasing"):
    assert cfg.rewards[name].params["hand_sensor_name"] == m.HAND_OBJECT_SENSOR
  params = cfg.events["reset_objects"].params
  low, high = params["pose_range"]["yaw"]
  assert math.degrees(m.TURN_TARGET_YAW - high) == pytest.approx(30.0)
  assert math.degrees(m.TURN_TARGET_YAW - low) == pytest.approx(70.0)
  plain = m.mug_turn_env_cfg(release=True, regrasp=True)
  assert m.HAND_OBJECT_SENSOR not in [s.name for s in plain.scene.sensors]
  assert "hand_sensor_name" not in plain.rewards["turned"].params


def test_upright_overhand_gates_the_held_and_lowered_rungs():
  from pal_mjlab.tasks.manipulation.mug import mug_env_cfg as m

  cfg = m.mug_upright_env_cfg(lower=True, release=True, overhand=True)
  for name in ("upright_held", "lowered"):
    assert cfg.rewards[name].params["tcp_cfg"].site_names
  plain = m.mug_upright_env_cfg(lower=True, release=True)
  assert "tcp_cfg" not in plain.rewards["upright_held"].params


def test_upright_facing_spawns_the_top_toward_the_robot_and_gates_the_lift():
  from pal_mjlab.tasks.manipulation.mug import mug_env_cfg as m

  cfg = m.mug_upright_env_cfg(lower=True, release=True, overhand=True, facing=True)
  yaw = cfg.events["reset_objects"].params["pose_range"]["yaw"]
  assert yaw == m.UPRIGHT_FACING_YAW
  # Every heading in it is at least 90 deg off pointing away from the robot.
  assert math.cos(yaw[0]) <= 1e-9 and math.cos(yaw[1]) <= 1e-9
  assert cfg.rewards["lift"].func.__name__ == "mug_lifted_overhand"
  assert cfg.rewards["lift"].params["tcp_cfg"].site_names
  plain = m.mug_upright_env_cfg(lower=True, release=True, overhand=True)
  assert plain.rewards["lift"].func.__name__ == "object_lifted"
  with pytest.raises(AssertionError):
    m.mug_upright_env_cfg(lower=True, facing=True)


def test_upright_away_spawns_the_top_away_from_the_robot():
  from pal_mjlab.tasks.manipulation.mug import mug_env_cfg as m

  cfg = m.mug_upright_env_cfg(lower=True, release=True, overhand=True, away=True)
  yaw = cfg.events["reset_objects"].params["pose_range"]["yaw"]
  assert yaw == m.UPRIGHT_AWAY_YAW
  # The facing range turned half round.
  assert all(
    math.isclose(math.cos(a - math.pi), math.cos(f), abs_tol=1e-9)
    for a, f in zip(yaw, m.UPRIGHT_FACING_YAW, strict=True)
  )
  assert cfg.rewards["lift"].func.__name__ == "mug_lifted_overhand"
  with pytest.raises(AssertionError):
    m.mug_upright_env_cfg(
      lower=True, release=True, overhand=True, facing=True, away=True
    )


def test_upright_view_prices_the_arm_on_the_camera_line():
  from mjlab.tasks.registry import load_env_cfg
  from pal_mjlab.tasks.manipulation.mug import mug_env_cfg as m

  task = (
    "Mjlab-MugUpright-Standing-Lower-Release-Overhand-Facing-View-Pal-Kangaroo-Grippers"
  )
  term = load_env_cfg(task).rewards["camera_view_blocked"]
  assert term.weight == -m.UPRIGHT_VIEW_PENALTY
  assert term.params["object_name"] == "object"
  plain = (
    "Mjlab-MugUpright-Standing-Lower-Release-Overhand-Facing-Pal-Kangaroo-Grippers"
  )
  assert "camera_view_blocked" not in load_env_cfg(plain).rewards
  from mjlab.tasks.registry import load_rl_cfg
  from pal_mjlab.tasks.manipulation.mug.kangaroo import STAGE_MIN_STD

  scratch = task.replace("-View-", "-View-Scratch-")
  assert set(load_env_cfg(scratch).rewards) == set(load_env_cfg(task).rewards)
  assert load_rl_cfg(scratch).actor.distribution_cfg["std_range"][0] == STAGE_MIN_STD


def test_upright_blind_stops_the_actors_mug_pose_at_the_grasp():
  from mjlab.tasks.registry import load_env_cfg

  plain = load_env_cfg(
    "Mjlab-MugUpright-Standing-Lower-Release-Overhand-Facing-Pal-Kangaroo-Grippers"
  )
  blind = load_env_cfg(
    "Mjlab-MugUpright-Standing-Lower-Release-Overhand-Facing-Still-Blind-Pal-Kangaroo-Grippers"
  )
  sensors = tuple(tuple(s) for s in blind.rewards["grasp"].params["sensor_names"])
  for key in ("jaw_to_object", "object_rot"):
    assert blind.observations["actor"].terms[key].params["blind_after_grasp"] == sensors
    assert "blind_after_grasp" not in plain.observations["actor"].terms[key].params
    assert "blind_after_grasp" not in blind.observations["critic"].terms[key].params
  # Same observations, so the -Facing checkpoints load; the arm held still once done.
  assert list(blind.observations["actor"].terms) == list(
    plain.observations["actor"].terms
  )
  assert "still" in blind.rewards and "raised" not in blind.rewards
  assert set(blind.rewards) - {"still"} == set(plain.rewards) - {"raised"}


def test_arm_goes_home_only_once_the_task_is_done():
  from types import SimpleNamespace

  from mjlab.tasks.registry import load_env_cfg
  from pal_mjlab.tasks.manipulation import mdp as m

  for task, term, name in (
    ("Mjlab-MugUpright-Standing-Lower-Release-Overhand-Facing-Pal-Kangaroo-Grippers",
     "stood_released", "raised"),
    ("Mjlab-MugTurn-Standing-Release-Regrasp-Near-HandsOff-Pal-Kangaroo-Grippers",
     "turned", "home"),
  ):  # fmt: skip
    rewards = load_env_cfg(task).rewards
    assert rewards[name].params["done_term"] == term
    assert list(rewards).index(term) < list(rewards).index(name)

  q0 = torch.zeros(3, 2)
  data = SimpleNamespace(default_joint_pos=q0, joint_pos=torch.tensor(
    [[0.0, 0.0], [0.0, 0.0], [0.5, 0.0]]
  ))  # fmt: skip
  manager = SimpleNamespace(
    active_terms=["done"],
    _step_reward=torch.tensor([[60.0], [0.0], [60.0]]),
    get_term_cfg=lambda name: SimpleNamespace(weight=60.0),
  )
  env = SimpleNamespace(
    scene={"robot": SimpleNamespace(data=data)}, reward_manager=manager
  )
  cfg = SimpleNamespace(name="robot", joint_ids=slice(None))
  home = m.arm_home_once_done(env, "done", cfg, std=0.5)
  # Done at the start pose: all of it; not done: nothing; done, 0.5 rad off: e^-1.
  assert torch.allclose(home, torch.tensor([1.0, 0.0, math.exp(-1.0)]))

  data.joint_vel = data.joint_pos
  still = m.arm_still_once_done(env, "done", cfg, std=0.5)
  # Done and still: all of it; not done: nothing; done, 0.5 rad/s: e^-1.
  assert torch.allclose(still, torch.tensor([1.0, 0.0, math.exp(-1.0)]))

  # Raised: jaw 16 cm up and still, 8 cm up and still, at 16 cm moving 0.5 rad/s.
  import pal_mjlab.tasks.manipulation.mdp.mug_rewards as mr

  data.joint_vel = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.5, 0.0]])
  manager._step_reward = torch.tensor([[60.0], [60.0], [60.0]])
  mid = torch.tensor([0.16, 0.08, 0.16]).view(3, 1, 1).expand(3, 1, 3)
  orig_jaw, orig_top = mr._jaw_frame, mr.surface_top_z
  mr._jaw_frame = lambda robot, cfg: (mid, None, None)
  mr.surface_top_z = lambda env, cfg: torch.zeros(3)
  try:
    raised = m.arm_raised_once_done(env, "done", cfg, cfg, cfg, height=0.16, std=0.5)
  finally:
    mr._jaw_frame, mr.surface_top_z = orig_jaw, orig_top
  assert torch.allclose(raised, torch.tensor([1.0, 0.5, 0.5 + 0.5 * math.exp(-1.0)]))
