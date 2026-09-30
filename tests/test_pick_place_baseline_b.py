"""Plan S7-2's baseline B, switch by switch, and S7-1's settings (``S71``) unchanged.

B (decided by Siwei 2026-09-25 18:41; target box 2026-09-26) is FR on the fda9492 model with:
the deployment cell's table height (0.815 m), 10 s episodes, a target drawn per episode from a
table-relative box, the 38-D deployable actor (jaw velocity out, TCP quaternion in, `grasped`
critic-only, a per-episode joint bias and 0-1 steps of joint lag, the tracker error model at
every rung) and the jaw at PAL's 2 rad/s. Each is a field of `pick_place_env_cfg.Baseline`, so B
can be taken apart against S7-1's settings. The spawn boxes (``SPAWN``) and module V
(``TOOL_SPEED``) are selected the same way.
"""

import dataclasses
import itertools
import math

import numpy as np
import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.manipulation.mdp import LiftingCommandCfg
from pal_mjlab.tasks.manipulation import mdp, scene
from pal_mjlab.tasks.manipulation.mdp.observations import BASE_BODY
from pal_mjlab.tasks.manipulation.pick_place import pick_place_env_cfg as pp
from pal_mjlab.tasks.manipulation.pick_place.kangaroo import env_cfg as kangaroo
from pal_mjlab.tasks.manipulation.robots import (
  KANGAROO_GRIPPERS_LEFT_ARM,
  KANGAROO_GRIPPERS_LEFT_ARM_WIDE,
)
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import (
  _DR_OBS_OBJECT_POS_BIAS,
  _DR_OBS_OBJECT_POS_NOISE,
  OBSERVATION_SETS,
)

FR = "R-C-strict+M1+M2+M3+M6"
B = pp.BASELINES["B"]
S71 = pp.BASELINES["S71"]
B_ACTOR = (
  ("joint_pos", 8),
  ("joint_vel", 7),
  ("actions", 8),
  ("tcp_pos", 3),
  ("tcp_quat", 4),
  ("object_pos", 3),
  ("object_yaw", 2),
  ("target_pos", 3),
)


def _cfg(baseline=B, dr="DR0", spawn="S71", num_envs=None, robot=None, **kw):
  cfg = pp.make_pick_place_env_cfg(
    robot or KANGAROO_GRIPPERS_LEFT_ARM,
    gripper_name="left",
    reward=FR,
    action_slew_rate_scale=0.5,
    baseline=baseline,
    dr=dr,
    spawn=pp.SPAWN_BOXES[spawn],
    **kw,
  )
  if num_envs is not None:
    cfg.scene.num_envs = num_envs
  return cfg


# -- S71 is every run up to S7-1 ----------------------------------------------------------


def test_s71_builds_what_the_default_built():
  """The disabled path: ``S71`` and no V is exactly what the factory built before S7-2."""
  s71 = _cfg(S71)
  assert s71.episode_length_s == pp.DEFAULT_EPISODE_LENGTH_S == 20.0
  command = s71.commands["goal_pose"]
  assert isinstance(command, LiftingCommandCfg)
  assert command.target_position_range.x == (0.42, 0.42)
  assert command.target_position_range.y == (0.07, 0.07)
  assert command.target_position_range.z == (pp.TARGET_Z, pp.TARGET_Z)
  assert s71.scene.entities["table"].init_state.pos == scene.TABLE_POS
  assert s71.terminations["object_dropped"].params["min_height"] == pytest.approx(
    scene.TABLE_TOP_Z - 0.2
  )
  actor, critic = (s71.observations[g].terms for g in ("actor", "critic"))
  assert list(actor) == list(critic)
  assert "grasped" in actor and "tcp_quat" not in actor
  action = s71.actions["joint_pos"]
  assert type(action) is mdp.RateLimitedJointPositionActionCfg
  assert not any("gripper" in k for k in action.rate_limit)
  pose_range = s71.events["reset_objects"].params["pose_range"]
  assert pose_range["x"] == pytest.approx((0.42 - 0.02 - 0.55, 0.42 + 0.02 - 0.55))
  assert pose_range["yaw"] == pytest.approx((-pp.SPAWN_JITTER_YAW, pp.SPAWN_JITTER_YAW))


def test_b_changes_its_five_settings_and_nothing_in_the_reward():
  b, s71 = _cfg(B), _cfg(S71)
  assert b.scene.entities["table"].init_state.pos == pytest.approx(
    (0.55, 0.0, 0.815 - scene.TABLE_THICKNESS / 2)
  )
  assert b.terminations["object_dropped"].params["min_height"] == pytest.approx(0.615)
  assert b.episode_length_s == 10.0
  command = b.commands["goal_pose"]
  assert isinstance(command, mdp.SurfaceOffsetCommandCfg)
  assert command.offset == pytest.approx((0.46 - 0.55, 0.08))
  assert command.offset_range == (0.02, 0.02)
  assert command.object_pose_range is None
  assert b.actions["joint_pos"].rate_limit["gripper_left_finger_joint"] == 0.04
  # Rewards and terminations are the arm's, whatever the baseline.
  assert b.rewards.keys() == s71.rewards.keys()
  for name, term in b.rewards.items():
    assert (
      term.func is s71.rewards[name].func and term.weight == s71.rewards[name].weight
    )
  assert b.terminations.keys() == s71.terminations.keys()


@pytest.mark.parametrize("field", [f.name for f in dataclasses.fields(pp.Baseline)])
def test_each_switch_builds_on_its_own(field):
  """B minus any one switch is a valid env: the settings can be taken apart."""
  one = dataclasses.replace(B, **{field: getattr(S71, field)})
  cfg = _cfg(one)
  assert cfg.episode_length_s == (one.episode_length_s or pp.DEFAULT_EPISODE_LENGTH_S)


def test_unknown_observation_set_raises():
  with pytest.raises(ValueError, match="unknown observation set"):
    _cfg(dataclasses.replace(B, observation_set="S72"))
  assert OBSERVATION_SETS == ("S71", "B")


# -- The deployable actor -----------------------------------------------------------------


def test_b_actor_is_38_d_without_grasped_and_the_critic_has_it():
  cfg = _cfg(B)
  actor, critic = cfg.observations["actor"], cfg.observations["critic"]
  assert tuple(actor.terms) == tuple(name for name, _ in B_ACTOR)
  assert tuple(critic.terms) == tuple(actor.terms) + ("grasped",)
  assert actor.enable_corruption and not critic.enable_corruption
  joint_pos = actor.terms["joint_pos"]
  assert joint_pos.noise.bias_noise_cfg.n_max == 0.01
  assert joint_pos.noise.bias_noise_cfg.operation == "abs"  # else it random-walks
  assert (joint_pos.delay_min_lag, joint_pos.delay_max_lag) == (0, 1)
  assert actor.terms["joint_vel"].params["asset_cfg"].joint_names == (
    r"arm_left_[1-7]_joint",
  )
  for name in ("object_pos", "object_yaw"):
    term = actor.terms[name]
    assert term.noise.bias_noise_cfg.operation == "abs"
    assert (term.delay_min_lag, term.delay_max_lag) == (0, 2)
  assert actor.terms["object_pos"].noise.bias_noise_cfg.n_max == _DR_OBS_OBJECT_POS_BIAS
  assert actor.terms["object_pos"].noise.noise_cfg.n_max == _DR_OBS_OBJECT_POS_NOISE
  for term in critic.terms.values():
    assert term.delay_max_lag == 0, "the critic reads the true, current state"


@pytest.fixture(scope="module")
def b_env():
  env = ManagerBasedRlEnv(cfg=_cfg(B, num_envs=8), device=get_test_device())
  env.reset()
  yield env
  env.close()


def _slice(env, obs, group: str, name: str) -> torch.Tensor:
  manager = env.observation_manager
  start = 0
  for term, dim in zip(
    manager.active_terms[group], manager.group_obs_term_dim[group], strict=True
  ):
    width = int(np.prod(dim))
    if term == name:
      return obs[group][:, start : start + width]
    start += width
  raise KeyError(name)


def test_b_observation_widths_on_the_built_env(b_env):
  obs = b_env.observation_manager.compute()
  assert obs["actor"].shape[-1] == 38
  assert obs["critic"].shape[-1] == 39
  for name, width in B_ACTOR:
    assert _slice(b_env, obs, "actor", name).shape[-1] == width, name


def test_the_cube_sits_where_the_robot_measured_it(b_env):
  """The point of 0.815: the resting cube at z = -0.060 in `base_link` (Claudius, f44b49d)."""
  zero = torch.zeros(b_env.num_envs, b_env.action_manager.total_action_dim)
  obs = b_env.step(zero)[0]
  z = _slice(b_env, obs, "critic", "object_pos")[:, 2]
  assert torch.allclose(z, torch.full_like(z, -0.060), atol=1e-3), z
  robot = b_env.scene["robot"]
  base_z = robot.data.body_link_pos_w[:, robot.body_names.index(BASE_BODY), 2]
  assert torch.allclose(base_z, torch.full_like(base_z, 0.900), atol=1e-3)


def test_the_tcp_quaternion_is_a_unit_quaternion_with_w_up(b_env):
  obs = b_env.observation_manager.compute()
  quat = _slice(b_env, obs, "critic", "tcp_quat")
  assert torch.allclose(quat.norm(dim=-1), torch.ones(b_env.num_envs), atol=1e-4)
  assert bool((quat[:, 0] >= 0.0).all())


def test_the_joint_bias_is_per_episode_and_bounded():
  """The actor's joint positions differ from the critic's by a per-env constant within
  +/-(0.01 + 0.002) rad -- a bias, redrawn at reset and not accumulated."""
  env = ManagerBasedRlEnv(cfg=_cfg(B, num_envs=16), device=get_test_device())
  zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
  for _ in range(20):
    env.reset()
  errors = []
  for _ in range(40):
    obs = env.step(zero)[0]
    errors.append(
      _slice(env, obs, "actor", "joint_pos") - _slice(env, obs, "critic", "joint_pos")
    )
  env.close()
  # (T, B, 8), once the arm and the jaw -- reset at its 0.0475 rad soft limit, commanded to
  # 0 -- have come to rest: before that the 0-1 step lag shows up as a difference too.
  err = torch.stack(errors[-10:]).cpu().numpy()
  assert np.abs(err).max() <= 0.01 + 0.002 + 1e-4
  per_env = err.mean(axis=0)
  assert per_env.std(axis=0).max() > 0.004, "no per-env bias"
  # Within the per-step noise's full width of the env's mean: a bias that moved would not be.
  assert np.abs(err - per_env).max() <= 2 * 0.002 + 1e-4, "not a constant per episode"


# -- The target box -----------------------------------------------------------------------


def test_the_target_is_drawn_per_episode_from_the_box(b_env):
  command = b_env.command_manager.get_term("goal_pose")
  centre = torch.tensor(B.target_xy, device=b_env.device) - torch.tensor(
    scene.TABLE_POS[:2], device=b_env.device
  )
  seen = []
  for _ in range(6):
    b_env.reset()
    seen.append(command.offsets.clone())
    goal = command.command
    top = mdp.surface_top_z(b_env, SceneEntityCfg("table"))
    assert torch.allclose(goal[:, 2], top + 0.025, atol=1e-5)
  offsets = torch.cat(seen)
  assert bool(((offsets - centre).abs() <= B.target_half_xy + 1e-6).all())
  assert float((offsets - centre).std(dim=0).min()) > 0.005, "not random"


def test_at_dr_scene_the_target_moves_with_the_table():
  env = ManagerBasedRlEnv(
    cfg=_cfg(B, dr="DR-scene", num_envs=8), device=get_test_device()
  )
  env.reset()
  command = env.command_manager.get_term("goal_pose")
  table = SceneEntityCfg("table")
  from pal_mjlab.tasks.manipulation.mdp.events import _surface_centre_xy

  rel = command.command[:, :2] - _surface_centre_xy(env, table)
  shift = _surface_centre_xy(env, table) - torch.tensor(
    scene.TABLE_POS[:2], device=env.device
  )
  env.close()
  assert torch.allclose(rel, command.offsets, atol=1e-5)
  assert float(shift.abs().max()) > 0.005, "the table did not move"


def test_the_spawn_boxes_clear_the_target_box_by_10_cm():
  """The plan's >= 10 cm spawn-target rule, against the target box's far corners. Both boxes
  are table-relative, so DR-scene's table shift does not change it."""
  h = B.target_half_xy
  tx = (B.target_xy[0] - h, B.target_xy[0] + h)
  ty = (B.target_xy[1] - h, B.target_xy[1] + h)
  for name, box in pp.SPAWN_BOXES.items():
    if name.startswith("WC"):
      # Plan S7-8: the target is not this box but a draw from the cube's own area, redrawn until it
      # is >= 10 cm from the cube (`TARGET=WC` / `WCK`; tests/test_pick_place_s78.py).
      continue
    dx = max(0.0, box.x[0] - tx[1], tx[0] - box.x[1])
    dy = max(0.0, box.y[0] - ty[1], ty[0] - box.y[1])
    assert math.hypot(dx, dy) >= 0.10 - 1e-9, name


# -- The jaw slew and the table clearance -------------------------------------------------


def test_the_jaw_closes_at_0_04_rad_a_step(b_env):
  b_env.reset()
  term = b_env.action_manager.get_term("joint_pos")
  names = [n.split("/", 1)[-1] for n in term.target_names]
  jaw = names.index("gripper_left_finger_joint")
  action = torch.zeros(b_env.num_envs, term.action_dim)
  action[:, jaw] = 1.0
  targets = []
  for _ in range(25):
    b_env.step(action)
    targets.append(float(term._processed_actions[0, jaw]))
  steps = np.diff([0.0, *targets])
  assert steps.max() <= 0.04 + 1e-6
  assert targets[0] == pytest.approx(0.04, abs=1e-6)
  assert targets[-1] == pytest.approx(0.78, abs=1e-6)  # closed, clipped at 0.78


def test_the_start_pose_clears_the_raised_table(b_env):
  b_env.reset()
  zero = torch.zeros(b_env.num_envs, b_env.action_manager.total_action_dim)
  for _ in range(5):
    b_env.step(zero)
  found = b_env.scene["left_hand_surface_contact"].data.found
  assert not bool(found.any()), "the hand touches the 0.815 m table at the start pose"
  robot = b_env.scene["robot"]
  tcp_z = robot.data.site_pos_w[:, robot.site_names.index("tcp_left"), 2]
  assert bool((tcp_z > 0.815 + 0.1).all()), tcp_z


# -- Spawn boxes and the selectors --------------------------------------------------------


@pytest.mark.parametrize(
  ("name", "x", "y", "yaw"),
  [
    ("W1", (0.37, 0.47), (0.22, 0.32), math.pi / 4),
    ("W2", (0.34, 0.46), (0.20, 0.34), math.pi / 4),
  ],
)
def test_a_w_box_is_the_spawn_even_at_dr_scene(name, x, y, yaw):
  for dr in ("DR0", "DR-scene"):
    pose_range = _cfg(B, dr=dr, spawn=name).events["reset_objects"].params["pose_range"]
    assert pose_range["x"] == pytest.approx((x[0] - 0.55, x[1] - 0.55))
    assert pose_range["y"] == pytest.approx(y)
    assert pose_range["yaw"] == pytest.approx((-yaw, yaw))
  s71 = _cfg(B, dr="DR-scene").events["reset_objects"].params["pose_range"]
  assert s71["x"] == pytest.approx((0.40 - 0.03 - 0.55, 0.44 + 0.03 - 0.55))


def test_the_wide_robot_is_p2s_scales():
  scales = KANGAROO_GRIPPERS_LEFT_ARM_WIDE.action_scale
  want = (0.90, 0.85, 0.65, 0.80, 1.55, 2.40, 1.20)
  assert tuple(scales[f"arm_left_{i}_joint"] for i in range(1, 8)) == want
  assert scales["gripper_left_finger_joint"] == pytest.approx(0.78)
  assert KANGAROO_GRIPPERS_LEFT_ARM_WIDE.action_clip == (
    KANGAROO_GRIPPERS_LEFT_ARM.action_clip
  )


@pytest.mark.parametrize(
  ("baseline", "spawn", "tool"),
  list(itertools.product(("S71", "B"), ("S71", "W1", "W2"), ("none", "V"))),
)
def test_the_selectors_build_what_they_name(monkeypatch, baseline, spawn, tool):
  monkeypatch.setenv("BASELINE", baseline)
  monkeypatch.setenv("SPAWN", spawn)
  monkeypatch.setenv("TOOL_SPEED", tool)
  monkeypatch.setenv("REWARD", FR)
  monkeypatch.setenv("SLEW", "0.5")
  cfg = kangaroo.kangaroo_pick_place_env_cfg()
  assert cfg.episode_length_s == (10.0 if baseline == "B" else 20.0)
  scales = cfg.actions["joint_pos"].scale
  assert scales["arm_left_5_joint"] == (0.75 if spawn == "S71" else 1.55)
  capped = type(cfg.actions["joint_pos"]) is mdp.ToolSpeedCappedJointPositionActionCfg
  assert capped == (tool == "V")


@pytest.mark.parametrize("var", ["BASELINE", "SPAWN", "TOOL_SPEED"])
def test_an_unknown_name_raises(monkeypatch, var):
  monkeypatch.setenv(var, "nope")
  with pytest.raises(ValueError, match=f"{var}='nope' is not a known name"):
    kangaroo.kangaroo_pick_place_env_cfg()
