"""Plan §14 (`pick-place-insert-plan.md`): pick the cross pin up on the wide pin area WP, standing or lying, and put
it down at a goal point in any pose (``GOAL=place``) -- step 0 item 1.

The env is `BS12` (plan §0.3) with ``LAG=FP+JSW``, ``SPAWN=WP TARGET=WP``, ``SHAPE=pin``, ``GOAL=place``, built on CPU.
What is checked: the pin's geometry (`mdp.pin_extent_z`) at the three resting poses; the cfg's widths, terms and
refusals; the reset puts the pin and the goal on WP about the shifted table, >= 10 cm apart, standing pins and goals
outside the hands' keepouts, lying pins at rest by the first step, the start share; the actor's observations are the
truth with the noise off, a standing pin's are plan §12's, the axis's sign is fixed; the pose-following goal; ``rested``
for a pin set down standing or lying on the goal (positive controls) and not 3 cm off or held; ``placed`` needs the
lift (L4); the jaw-yaw term is for lying pins only and prefers a jaw across the axis; GC's gate; and the per-step
reward rises along the task with the let-go >= 3 per step over the hold (L5).
"""

from __future__ import annotations

import math

import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul
from pal_mjlab.tasks.manipulation import mdp, scene
from pal_mjlab.tasks.manipulation.mdp.events import in_keepout, surface_top_z
from pal_mjlab.tasks.manipulation.mdp.observations import _placed_bool
from pal_mjlab.tasks.manipulation.pick_place import insert as ins
from pal_mjlab.tasks.manipulation.pick_place import place_pin as pp
from pal_mjlab.tasks.manipulation.pick_place import rewards as rw_mod
from pal_mjlab.tasks.manipulation.pick_place.kangaroo.env_cfg import (
  kangaroo_pick_place_env_cfg,
  kangaroo_pick_place_rl_cfg,
)
from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import (
  SPAWN_BOXES,
  TARGET_MIN_SEPARATION,
)

RCTP = "R-C-strict+M1+M2+M3+M6+J+M8"
#: `BS12` + PS (plan §14.4's batch, every code but its start).
PS_ENV = {
  "REWARD": RCTP,
  "DR": "DR-scene",
  "PPO": "PPO-stack",
  "SLEW": "stack",
  "BASELINE": "B",
  "SPAWN": "WP",
  "TOOL_SPEED": "none",
  "GAINS": "G20",
  "EFFORT": "CTP",
  "LAG": "FP+JSW",
  "ACTION": "abs",
  "CMD_OBS": "act",
  "TARGET": "WP",
  "EPISODE_S": "15",
  "CRITIC_OBS": "q",
  "ARM_TABLE": "AH10",
  "SHAPE": "pin",
  "GOAL": "place",
  "TABLE_DZ": "22",
}
WP = SPAWN_BOXES["WP"]
HALF = scene.PIN_LENGTH / 2
X_REST = (scene.PIN_SPAN + scene.PIN_THICKNESS) / 2 / math.sqrt(2)
_UNSET = (
  "OCC",
  "OBJ_OBS",
  "GATE",
  "BOX_FEET",
  "BOX_OBS",
  "CKPT_SPAWN",
  "TRACKER",
  "PIN_START",
  "PIN_LIE_P",
  "INSERT_HOLE_D",
  "INSERT_FRICTION",
)


def _set_env(monkeypatch, **changes) -> None:
  for var, value in {**PS_ENV, **changes}.items():
    if value is None:
      monkeypatch.delenv(var, raising=False)
    else:
      monkeypatch.setenv(var, value)
  for var in _UNSET:
    if var not in changes:
      monkeypatch.delenv(var, raising=False)


def _cfg(monkeypatch, num_envs: int = 4, **changes):
  _set_env(monkeypatch, **changes)
  cfg = kangaroo_pick_place_env_cfg()
  cfg.scene.num_envs = num_envs
  cfg.seed = 0
  return cfg


@pytest.fixture(scope="module")
def env():
  """PS (the mix start) on CPU, 16 envs, the actor's noise off."""
  mp = pytest.MonkeyPatch()
  cfg = _cfg(mp, num_envs=16)
  cfg.observations["actor"].enable_corruption = False
  e = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  e.reset()
  yield e
  e.close()
  mp.undo()


def _hold(env, steps: int) -> None:
  with torch.inference_mode():
    for _ in range(steps):
      env.step(torch.zeros(env.action_space.shape, device=env.device))


def _reset(env) -> None:
  with torch.inference_mode():
    env.reset()


def _place(entity, pos: torch.Tensor, quat: torch.Tensor) -> None:
  entity.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1))
  entity.write_root_link_velocity_to_sim(torch.zeros_like(entity.data.root_link_vel_w))


def _yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
  z = torch.zeros_like(yaw)
  return quat_from_euler_xyz(z, z, yaw)


def _cut(env, group: str = "actor") -> dict[str, torch.Tensor]:
  with torch.inference_mode():
    obs = env.observation_manager.compute_group(group)
  terms = env.observation_manager.active_terms[group]
  dims = env.observation_manager.group_obs_term_dim[group]
  cut, i = {}, 0
  for n, dd in zip(terms, dims, strict=True):
    cut[n] = obs[:, i : i + dd[0]]
    i += dd[0]
  return cut


# -- The pin's geometry -----------------------------------------------------------------------------------


def test_the_pins_extent_in_its_three_resting_poses():
  one = torch.tensor([0.3])
  standing = _yaw_quat(one)
  lying_x = mdp.lying_quat(one)
  zero = torch.zeros(1)
  lying_plus = quat_mul(
    _yaw_quat(one),
    quat_from_euler_xyz(zero, torch.full_like(zero, math.pi / 2), zero),
  )
  for q, want in (
    (standing, HALF),
    (lying_x, X_REST),
    (lying_plus, scene.PIN_SPAN / 2),
  ):
    below, above = mdp.pin_extent_z(q)
    assert float(below) == pytest.approx(want, abs=1e-6)
    assert float(above) == pytest.approx(want, abs=1e-6)
  assert bool(mdp.pin_standing(standing)) and not bool(mdp.pin_standing(lying_x))
  # The lying start's axis is horizontal at the asked yaw.
  axis = mdp.place_pin.pin_axis_w(lying_x)
  assert float(axis[0, 2]) == pytest.approx(0.0, abs=1e-6)
  assert math.atan2(float(axis[0, 1]), float(axis[0, 0])) == pytest.approx(
    0.3, abs=1e-5
  )


# -- The cfg -------------------------------------------------------------------------------------------


def test_the_place_cfg(monkeypatch):
  cfg = _cfg(monkeypatch)
  actor = list(cfg.observations["actor"].terms)
  assert actor[-5:] == [
    "object_pos",
    "object_yaw",
    "target_pos",
    "goal_yaw",
    "pin_axis",
  ]
  critic = list(cfg.observations["critic"].terms)
  assert critic[-6:] == [
    "grasped",
    "release_grade",
    "release_latched",
    "goal_yaw",
    "pin_axis",
    "pin_rested",
  ]
  assert ins.BLOCK not in cfg.scene.entities
  assert "nativeccd" in cfg.sim.mujoco.disableflags
  rw = cfg.rewards
  assert rw["reach"].func is mdp.pin_reach
  assert rw["lift"].func is mdp.pin_lift_faded
  assert rw["placed"].func is mdp.pin_placed_graded
  assert rw["retreat"].func is mdp.pin_retreat_height
  assert rw["jaw_yaw"].weight == pp.JAW_YAW_WEIGHT
  assert rw["release"].params["threshold"] == pp.RELEASE_TOL
  assert rw["placed"].params["threshold"] == pp.PLACED_TOL
  assert rw["placed"].params["speed_std"] == pp.M6_SPEED_STD
  assert rw["placed"].params["height_free"] == 0.0
  assert (
    "tilt_std" not in rw["placed"].params or rw["placed"].params["tilt_std"] is None
  )
  assert "grasp_closure" not in rw
  assert not {"depth", "align", "upright", "block_moved"} & set(rw)
  assert "block_dropped" not in cfg.terminations
  assert isinstance(cfg.commands["goal_pose"], mdp.PinPlaceCommandCfg)
  assert cfg.commands["goal_pose"].min_object_distance == TARGET_MIN_SEPARATION
  assert cfg.events["reset_objects"].params["keepout"] == pp.START_KEEPOUT
  ev = list(cfg.events)
  assert ev.index("reset_pin_start") == ev.index("reset_objects") + 1
  assert cfg.events["reset_pin_start"].params["lie_probability"] == pp.PIN_LIE_P
  rl = kangaroo_pick_place_rl_cfg()
  assert rl.widen_actor_on_resume and rl.widen_critic_on_resume


@pytest.mark.parametrize("start, share", [("stand", 0.0), ("lie", 1.0), ("mix", 0.5)])
def test_pin_start_names_the_lying_share(monkeypatch, start, share):
  cfg = _cfg(monkeypatch, PIN_START=start)
  assert cfg.events["reset_pin_start"].params["lie_probability"] == share
  cfg = _cfg(monkeypatch, PIN_START="mix", PIN_LIE_P="0.3")
  assert cfg.events["reset_pin_start"].params["lie_probability"] == 0.3


def test_wp_trains_the_wp_arm_and_a_wc_checkpoint_is_rescaled(monkeypatch):
  """SPAWN=WP is WC's arm with arm 7 at its half-range cap (step 0 item 3); ``CKPT_SPAWN=WC`` (§12's IN, S10.2's
  WX-a-H22) converts arm 7's actions exactly (decision 0003), nothing else."""
  from pal_mjlab.tasks.manipulation.robots.kangaroo import (
    KANGAROO_GRIPPERS_LEFT_ARM_WC,
    KANGAROO_GRIPPERS_LEFT_ARM_WP,
  )

  wc, wp = (
    a.action_scale
    for a in (KANGAROO_GRIPPERS_LEFT_ARM_WC, KANGAROO_GRIPPERS_LEFT_ARM_WP)
  )
  assert {k for k in wp if wp[k] != wc[k]} == {r"arm_left_7_joint"}
  assert wp[r"arm_left_7_joint"] == 2.35
  from pal_mjlab.tasks.manipulation.pick_place.kangaroo.env_cfg import SPAWN_ROBOTS

  assert SPAWN_ROBOTS["WP"] is SPAWN_ROBOTS["WCP"] is KANGAROO_GRIPPERS_LEFT_ARM_WP
  _set_env(monkeypatch, CKPT_SPAWN="WC")
  rl = kangaroo_pick_place_rl_cfg()
  assert rl.rescale_actions_on_load == {r"arm_left_7_joint": pytest.approx(1.20 / 2.35)}


def test_gc_is_ps_sgs_module(monkeypatch):
  cfg = _cfg(monkeypatch, REWARD=pp.PLACE_REWARD_GC)
  gc = cfg.rewards["grasp_closure"]
  assert gc.func is mdp.grasp_closure and gc.weight == rw_mod.GC_WEIGHT
  assert gc.params["q_touch"] == pp.GC_Q_TOUCH
  assert gc.params["above_top"] == ins.GRASP_ABOVE_TOP
  assert gc.params["above_top_lying"] == pp.GRASP_ABOVE_TOP_LYING
  other = _cfg(monkeypatch)
  assert set(cfg.rewards) - set(other.rewards) == {"grasp_closure"}


def test_the_place_refusals(monkeypatch):
  with pytest.raises(ValueError, match="SHAPE=pin"):
    _cfg(monkeypatch, SHAPE="cube")
  with pytest.raises(ValueError, match="TARGET"):
    _cfg(monkeypatch, TARGET="B")
  with pytest.raises(ValueError):
    _cfg(monkeypatch, REWARD="R-C-strict+M1+M2+M3+M6+J+M8+S")
  with pytest.raises(ValueError, match="BOX_OBS"):
    _cfg(monkeypatch, BOX_OBS="pose")
  with pytest.raises(ValueError, match="needs GOAL=place"):
    _cfg(monkeypatch, GOAL="insert", PIN_START="mix")
  with pytest.raises(ValueError, match="GC"):
    _cfg(
      monkeypatch,
      GOAL="table",
      SHAPE="cube",
      SPAWN="WC",
      TARGET="WC",
      REWARD=pp.PLACE_REWARD_GC,
    )
  with pytest.raises(ValueError, match="PIN_LIE_P"):
    _cfg(monkeypatch, PIN_START="stand", PIN_LIE_P="0.2")
  with pytest.raises(ValueError, match="PIN_LIE_P"):
    _cfg(monkeypatch, PIN_LIE_P="2")


# -- The env ------------------------------------------------------------------------------------------


def _table_shift(env) -> torch.Tensor:
  table = env.scene["table"]
  tb = table.indexing.body_ids[table.body_names.index("table")]
  shift = env.sim.data.xpos[:, tb, :2] - env.scene.env_origins[:, :2]
  return shift - torch.tensor(scene.table_pos(0.837)[:2], device=env.device)


def test_the_reset_draws_pin_and_goal_on_wp(monkeypatch):
  """3 resets x 64 envs: the pin's centre and the goal on WP about the shifted table, >= 10 cm apart; standing pins
  outside both hands' keepouts, no pin in WP's cut corners, goals outside the right hand's keepout and the cut (robot
  frame); about half the pins lying; the goal on the table's surface."""
  cfg = _cfg(monkeypatch, num_envs=64)
  e = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  try:
    lying = []
    for _ in range(3):
      e.reset()
      pin = e.scene["object"]
      cmd = e.command_manager.get_term("goal_pose")
      shift = _table_shift(e)
      origin = e.scene.env_origins[:, :2]
      goal = cmd.goal_surface
      for xy in (pin.data.root_link_pos_w[:, :2], goal[:, :2]):
        rel = xy - origin - shift
        assert bool(
          ((rel[:, 0] >= WP.x[0] - 1e-6) & (rel[:, 0] <= WP.x[1] + 1e-6)).all()
        )
        assert bool(
          ((rel[:, 1] >= WP.y[0] - 1e-6) & (rel[:, 1] <= WP.y[1] + 1e-6)).all()
        )
      d = torch.norm(pin.data.root_link_pos_w[:, :2] - goal[:, :2], dim=-1)
      assert bool((d >= TARGET_MIN_SEPARATION - 1e-6).all())
      standing = mdp.pin_standing(pin.data.root_link_quat_w)
      robot_xy = pin.data.root_link_pos_w[:, :2] - origin
      assert not bool((in_keepout(robot_xy, pp.START_KEEPOUT) & standing).any())
      assert not bool(in_keepout(robot_xy, pp.CUT).any())
      assert not bool(in_keepout(goal[:, :2] - origin, pp.GOAL_KEEPOUT).any())
      top = surface_top_z(e, SceneEntityCfg("table"))
      assert torch.allclose(goal[:, 2], top, atol=1e-6)
      lying.append((~standing).float().mean().item())
    assert 0.35 < sum(lying) / 3 < 0.65, lying
  finally:
    e.close()


def test_the_pins_are_at_rest_by_the_first_step(env):
  _reset(env)
  _hold(env, 1)
  st = mdp.pin_rest_state(env, "object", "goal_pose")
  assert float(st.speed.max()) < 0.01
  assert float(st.z_min.abs().max()) < 0.001
  assert bool((~st.standing).any()) and bool(st.standing.any())


def test_the_observations_are_the_truth_with_the_noise_off(env):
  from pal_mjlab.tasks.manipulation.mdp.observations import _to_base_frame

  _reset(env)
  _hold(env, 10)  # past the tracker's lag: nothing has moved
  robot, pin = env.scene["robot"], env.scene["object"]
  cut = _cut(env)
  cmd = env.command_manager.get_term("goal_pose")
  assert torch.allclose(
    cut["object_pos"], _to_base_frame(robot, pin.data.root_link_pos_w), atol=3e-4
  )  # the tracker's lag behind a lying pin's last 0.1 mm of settling
  assert torch.allclose(
    cut["target_pos"], _to_base_frame(robot, cmd.goal_surface), atol=1e-5
  )
  assert torch.equal(
    cut["goal_yaw"], torch.tensor([[1.0, 0.0]], device=env.device).expand(16, 2)
  )
  standing = mdp.pin_standing(pin.data.root_link_quat_w)
  axis = cut["pin_axis"]
  assert torch.allclose(axis[standing], torch.zeros_like(axis[standing]), atol=5e-3)
  lying_axis = axis[~standing]
  assert torch.allclose(lying_axis.norm(dim=-1), torch.ones(len(lying_axis)), atol=1e-3)
  assert bool((lying_axis[:, 0] >= 0.0).all())
  # A lying "x" pin: 4 psi = 180 deg.
  yaw = cut["object_yaw"][~standing]
  assert torch.allclose(yaw[:, 0], -torch.ones(len(yaw)), atol=1e-3)


def test_a_standing_pin_is_observed_as_in_s12(env):
  """For standing pins -- upright, tilted, a quarter turn on, end for end -- the yaw and tilt slots are plan §12's
  (`box.box_yaw_b`, `insert.pin_tilt_b`), so §12's checkpoints see what they were trained on."""
  pin = env.scene["object"]
  pos = pin.data.root_link_pos_w.clone()
  n = pos.shape[0]
  yaw = torch.linspace(-0.7, 0.7, n)
  tilt = quat_from_euler_xyz(
    torch.full((n,), 0.2), torch.full((n,), -0.1), torch.zeros(n)
  )
  flip = quat_from_euler_xyz(torch.full((n,), math.pi), torch.zeros(n), torch.zeros(n))
  for q in (
    _yaw_quat(yaw),
    quat_mul(tilt, _yaw_quat(yaw)),
    quat_mul(quat_mul(tilt, _yaw_quat(yaw)), flip),
  ):
    pos[:, 2] = surface_top_z(env, SceneEntityCfg("table")) + 0.2
    _place(pin, pos, q)
    env.sim.forward()
    assert bool(mdp.pin_standing(q).all())
    assert torch.allclose(
      mdp.pin_axis_b(env, "object"), mdp.pin_tilt_b(env, "object"), atol=1e-6
    )
    assert torch.allclose(
      mdp.pin_psi_b(env, "object"), mdp.box_yaw_b(env, "object"), atol=1e-5
    )


def test_a_lying_pins_axis_sign_is_fixed(env):
  """End for end and a quarter turn about its own axis, a lying pin reads the same axis and the same psi slot."""
  pin = env.scene["object"]
  pos = pin.data.root_link_pos_w.clone()
  n = pos.shape[0]
  q = mdp.lying_quat(torch.linspace(0.05, math.pi - 0.05, n))
  zero = torch.zeros(n)
  flip = quat_from_euler_xyz(torch.full((n,), math.pi), zero, zero)
  quarter = _yaw_quat(torch.full((n,), math.pi / 2))
  out = []
  for qq in (q, quat_mul(q, flip), quat_mul(q, quarter)):
    pos[:, 2] = surface_top_z(env, SceneEntityCfg("table")) + 0.2
    _place(pin, pos, qq)
    env.sim.forward()
    out.append((mdp.pin_axis_b(env, "object"), mdp.pin_psi_b(env, "object")))
  for axis, psi in out[1:]:
    assert torch.allclose(axis, out[0][0], atol=1e-5)
    assert torch.allclose(psi, out[0][1], atol=1e-4)


def _set_on_goal(env, pose: str, dxy=(0.0, 0.0)) -> None:
  """The pin resting on the table at the goal point (plus ``dxy``), standing or lying."""
  pin = env.scene["object"]
  cmd = env.command_manager.get_term("goal_pose")
  n = env.num_envs
  q = (
    _yaw_quat(torch.full((n,), 0.3))
    if pose == "stand"
    else mdp.lying_quat(torch.full((n,), 0.7))
  )
  below, _ = mdp.pin_extent_z(q)
  pos = cmd.goal_surface.clone()
  pos[:, 0] += dxy[0]
  pos[:, 1] += dxy[1]
  pos[:, 2] += below
  _place(pin, pos, q)
  env.sim.forward()


@pytest.mark.parametrize("pose", ["stand", "lie"])
def test_the_goal_follows_the_pins_pose(env, pose):
  _reset(env)
  _set_on_goal(env, pose)
  cmd = env.command_manager.get_term("goal_pose")
  pin = env.scene["object"]
  assert torch.allclose(cmd.command, pin.data.root_link_pos_w, atol=1e-6)
  top = surface_top_z(env, SceneEntityCfg("table"))
  want = HALF if pose == "stand" else X_REST
  assert torch.allclose(cmd.command[:, 2] - top, torch.full_like(top, want), atol=1e-6)


@pytest.mark.parametrize("pose", ["stand", "lie"])
def test_rested_holds_for_a_pin_set_down_on_the_goal(env, pose):
  """Positive controls: a pin put down on the goal, standing or lying, and left (the jaw open, the hand at home) is
  ``rested`` after 0.5 s and stays within 2 mm of where it was put; 3 cm off the goal it is not."""
  _reset(env)
  _set_on_goal(env, pose)
  p0 = env.scene["object"].data.root_link_pos_w.clone()
  _hold(env, 25)
  args = dict(radius=pp.PLACE_RADIUS, height=pp.PLACED_TOL[1], speed=pp.REST_SPEED)
  st = mdp.pin_rest_state(env, "object", "goal_pose")
  grasped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  rested = st.rested(grasped, angular_speed=pp.REST_ANGULAR_SPEED, **args)
  moved = torch.norm(env.scene["object"].data.root_link_pos_w - p0, dim=-1)
  # Every pin not hit by the parked arm (none is on WP's goal area, which avoids the hands).
  assert bool(rested.all()), (st.z_min, st.goal_dxy, st.speed)
  assert float(moved.max()) < 0.002
  assert not bool(
    st.rested(~grasped, angular_speed=pp.REST_ANGULAR_SPEED, **args).any()
  )
  _set_on_goal(env, pose, dxy=(0.03, 0.0))
  _hold(env, 5)
  st = mdp.pin_rest_state(env, "object", "goal_pose")
  assert not bool(st.rested(grasped, angular_speed=pp.REST_ANGULAR_SPEED, **args).any())


def test_placed_needs_the_lift(env):
  """A pin put on the goal without ever being lifted is not placed (L4), though it rests there."""
  _reset(env)
  _set_on_goal(env, "lie")
  _hold(env, 5)
  term = env.reward_manager.get_term_cfg("placed")
  keys = (
    "command_name",
    "object_name",
    "grasp_sensor_name",
    "threshold",
    "support_sensor_name",
    "jaw_cfg",
    "jaw_open",
  )
  placed = _placed_bool(
    env, velocity_threshold=pp.REST_SPEED, **{k: term.params[k] for k in keys}
  )
  assert not bool(placed.any())
  st = mdp.pin_rest_state(env, "object", "goal_pose")
  assert bool((st.goal_dxy < pp.PLACE_RADIUS).all())


def test_the_jaw_yaw_term_is_for_lying_pins_and_prefers_the_jaw_across(env):
  _reset(env)
  term = env.reward_manager.get_term_cfg("jaw_yaw")
  pads = term.params["pads_cfg"]
  pin = env.scene["object"]
  pos = pin.data.root_link_pos_w.clone()
  n = env.num_envs
  _place(pin, pos, _yaw_quat(torch.zeros(n)))
  env.sim.forward()
  assert torch.equal(mdp.pin_jaw_yaw(env, **term.params), torch.zeros(n))
  # Lying pins with their axes at 0..180 deg: the angle to the jaw's closing direction covers 0..90 deg, and the
  # pin whose axis is across the closing direction gets the smaller angle.
  _place(pin, pos, mdp.lying_quat(torch.linspace(0.0, math.pi, n)))
  env.sim.forward()
  a = mdp.jaw_axis_angle(env, "object", pads)
  assert float(a.min()) < math.radians(10) and float(a.max()) > math.radians(80)
  robot = env.scene["robot"]
  close = (
    robot.data.site_pos_w[:, pads.site_ids][:, 1, :2]
    - robot.data.site_pos_w[:, pads.site_ids][:, 0, :2]
  )
  best = torch.atan2(close[:, 1], close[:, 0]) + math.pi / 2  # the axis across the jaw
  i = int(a.argmin())
  got = math.atan2(
    float(mdp.place_pin.pin_axis_w(pin.data.root_link_quat_w)[i, 1]),
    float(mdp.place_pin.pin_axis_w(pin.data.root_link_quat_w)[i, 0]),
  )
  diff = (got - float(best[i]) + math.pi / 2) % math.pi - math.pi / 2
  assert abs(diff) < math.radians(10)


def test_gc_pays_only_at_the_grasp_point(monkeypatch):
  cfg = _cfg(monkeypatch, num_envs=2, REWARD=pp.PLACE_REWARD_GC)
  e = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  try:
    e.reset()
    term = e.reward_manager.get_term_cfg("grasp_closure")
    # The hand at home is far from every grasp point.
    assert torch.equal(mdp.grasp_closure(e, **term.params), torch.zeros(2))
    # Moving the pin's grasp point onto the tool point opens the gate: the jaw at home is open, so ~0 is paid;
    # the gate itself is what is checked.
    pin, robot = e.scene["object"], e.scene["robot"]
    tcp = term.params["asset_cfg"]
    t = robot.data.site_pos_w[:, tcp.site_ids].squeeze(1)
    pos = t.clone()
    pos[:, 2] = t[:, 2] - HALF - ins.GRASP_ABOVE_TOP
    _place(pin, pos, _yaw_quat(torch.zeros(2)))
    e.sim.forward()
    g = mdp.pin_grasp_point_w(
      e, "object", ins.GRASP_ABOVE_TOP, pp.GRASP_ABOVE_TOP_LYING
    )
    assert torch.allclose(g, t, atol=1e-5)
    q = robot.data.joint_pos[:, term.params["jaw_cfg"].joint_ids][:, 0]
    want = (q / pp.GC_Q_TOUCH).clamp(0, 1)
    assert torch.allclose(mdp.grasp_closure(e, **term.params), want, atol=1e-6)
  finally:
    e.close()


# -- L5: the per-step reward rises along the task -------------------------------------------------------


def _g(x, s):
  return math.exp(-((x / s) ** 2))


def _ps_income(stage: str, speed: float = 0.0, height: float = 0.0) -> float:
  """PS's task terms at plan §14's L5 stages, from the term definitions (`mdp.place_pin`) and the weights (RCTP's,
  `place_pin`'s): reach 1 while the hand is at the grasp point, the grasp 1; d the distance from (c_x, c_y, z_min)
  to the goal point."""
  w_goal, w_near = rw_mod.M2_GOAL_WEIGHT, rw_mod.M2_NEAR_WEIGHT

  def goal(d):
    return w_goal * _g(d, rw_mod.BRINGING_STD) + w_near * _g(d, rw_mod.M2_NEAR_STD)

  def lift(h, d):
    return 2.0 * min(h, 0.10) / 0.10 * (1 - _g(d, pp.LIFT_FADE_STD))

  if stage == "reach":
    return 1.0
  if (
    stage == "grasped"
  ):  # on the table, 20 cm from the goal (M2 not yet: was_lifted unset)
    return 1 + 1
  if stage == "carry":  # 10 cm up, 20 cm from the goal
    return 1 + 1 + lift(0.10, math.hypot(0.2, 0.1)) + goal(math.hypot(0.2, 0.1))
  if stage == "over":  # 3 cm over the goal
    return 1 + 1 + lift(0.03, 0.03) + goal(0.03)
  if stage == "held_at_goal":  # touching the table at the goal, still held
    return 1 + 1 + goal(0.0)
  floor = rw_mod.M6_FLOOR
  q = floor + (1 - floor) * _g(speed, pp.M6_SPEED_STD) * _g(height, pp.M6_HEIGHT_STD)
  if stage == "rested":  # let go, resting; the hand still around it (reach ~1)
    return 1 + goal(0.0) + 8 * q
  if stage == "retreated":  # the hand 10 cm over the pin's top
    return _g(0.10, 0.2) + goal(0.0) + 8 * q + 2.0
  raise ValueError(stage)


def test_l5_the_ps_reward_rises_along_the_task():
  stages = ["reach", "grasped", "carry", "over", "held_at_goal", "rested", "retreated"]
  inc = [_ps_income(s) for s in stages]
  assert all(b > a for a, b in zip(inc, inc[1:], strict=False)), dict(
    zip(stages, inc, strict=True)
  )
  # The let-go out-earns the hold by >= 3 per step for a set-down at <= 2 cm/s and <= 2 mm over the table.
  assert _ps_income("rested", 0.02, 0.002) >= _ps_income("held_at_goal") + 3.0
  assert _ps_income("retreated", 0.02, 0.002) >= _ps_income("held_at_goal") + 3.0
