"""Plan S10.1 (`pick-place-v1-objects-plan.md`): the box on the wide area, at any yaw -- step 0 item 1.

The env is `BS2` (BS + PQ + AH10, ``CMD_OBS=cmd``) with ``GOAL=box-rim``, ``SPAWN=WC TARGET=WC``,
``EPISODE_S=15`` and ``BOX_OBS=pose``, built on CPU. What is checked: the box is drawn inside the area
about the (shifted) table, clear of the cube, at a yaw in +/- 45 deg; the goal follows a pushed and
turned box; the actor and the critic observe the box's own position and its yaw as (cos 4 psi,
sin 4 psi), equal to the true pose with the noise off and the same for psi and psi + 90 deg; the
yaw grade is taken against the box; ``in_box`` and ``placed`` hold for a cube resting in a box at
45 deg; ``TARGET=B`` with the S10 observation is the S10 / DEP1 cfg; the combinations that are not
built are refused; and a resume widens the actor and the critic exactly.
"""

from __future__ import annotations

import math

import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import euler_xyz_from_quat, quat_from_euler_xyz
from pal_mjlab.tasks.manipulation import mdp, scene
from pal_mjlab.tasks.manipulation.mdp import observations as obs_mod
from pal_mjlab.tasks.manipulation.mdp.observations import (
  _WAS_LIFTED,
  _placed_bool,
  _release_latch,
)
from pal_mjlab.tasks.manipulation.pick_place import box as bx
from pal_mjlab.tasks.manipulation.pick_place.kangaroo.env_cfg import (
  kangaroo_pick_place_env_cfg,
  kangaroo_pick_place_rl_cfg,
)
from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import (
  GOAL_THRESHOLD,
  SPAWN_BOXES,
)
from pal_mjlab.tasks.manipulation.runner import rescale_actions, widen_input

RCTP = "R-C-strict+M1+M2+M3+M6+J+M8"
#: Plan S10.1's codes: `BS2` + box-rim on the wide area, the box pose observed (WX-c).
WX_ENV = {
  "REWARD": RCTP,
  "DR": "DR-scene",
  "PPO": "PPO-stack",
  "SLEW": "stack",
  "BASELINE": "B",
  "SPAWN": "WC",
  "TOOL_SPEED": "none",
  "GAINS": "G20",
  "EFFORT": "CTP",
  "LAG": "FP+JS",
  "ACTION": "abs",
  "CMD_OBS": "cmd",
  "TARGET": "WC",
  "EPISODE_S": "15",
  "CRITIC_OBS": "q",
  "ARM_TABLE": "AH10",
  "GOAL": "box-rim",
  "BOX_OBS": "pose",
}
WC = SPAWN_BOXES["WC"]


def _set_env(monkeypatch, **changes) -> None:
  for var, value in {**WX_ENV, **changes}.items():
    if value is None:
      monkeypatch.delenv(var, raising=False)
    else:
      monkeypatch.setenv(var, value)
  for var in ("SHAPE", "OCC", "OBJ_OBS", "GATE", "BOX_FEET", "CKPT_SPAWN"):
    if var not in changes:
      monkeypatch.delenv(var, raising=False)


def _cfg(monkeypatch, num_envs: int = 4, **changes):
  _set_env(monkeypatch, **changes)
  cfg = kangaroo_pick_place_env_cfg()
  cfg.scene.num_envs = num_envs
  return cfg


def _env(monkeypatch, num_envs: int = 4, corrupt: bool = True, **changes):
  cfg = _cfg(monkeypatch, num_envs, **changes)
  cfg.observations["actor"].enable_corruption = corrupt
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  env.reset()
  return env


def _hold(env, steps: int) -> None:
  with torch.inference_mode():
    for _ in range(steps):
      env.step(torch.zeros(env.action_space.shape, device=env.device))


def _yaw(quat: torch.Tensor) -> torch.Tensor:
  _, _, yaw = euler_xyz_from_quat(quat)
  return torch.remainder(yaw + math.pi, 2 * math.pi) - math.pi


def _put_box(env, yaw: float, dxy=(0.0, 0.0)) -> None:
  """Turn every env's box to ``yaw`` (world) and move it by ``dxy``, at rest."""
  box = env.scene[bx.BOX]
  pose = box.data.root_link_pose_w.clone()
  pose[:, 0] += dxy[0]
  pose[:, 1] += dxy[1]
  zero = torch.zeros(env.num_envs, device=env.device)
  pose[:, 3:7] = quat_from_euler_xyz(zero, zero, zero + yaw)
  box.write_root_link_pose_to_sim(pose)
  box.write_root_link_velocity_to_sim(torch.zeros_like(box.data.root_link_vel_w))


def _cube_in_box(env, yaw_off: float, above_floor: float) -> None:
  """Every env's cube centred over the box floor, turned ``yaw_off`` off the box, its bottom
  ``above_floor`` over the floor top, at rest."""
  box = env.scene[bx.BOX]
  obj = env.scene["object"]
  floor = mdp.box.box_floor_top_w(env, bx.BOX, scene.BOX_FLOOR)
  pose = obj.data.root_link_pose_w.clone()
  pose[:, :2] = floor[:, :2]
  pose[:, 2] = floor[:, 2] + 0.025 + above_floor
  zero = torch.zeros(env.num_envs, device=env.device)
  pose[:, 3:7] = quat_from_euler_xyz(
    zero, zero, _yaw(box.data.root_link_quat_w) + yaw_off
  )
  obj.write_root_link_pose_to_sim(pose)
  obj.write_root_link_velocity_to_sim(torch.zeros_like(obj.data.root_link_vel_w))


# -- The cfg -----------------------------------------------------------------------------


def test_the_wide_box_cfg(monkeypatch):
  cfg = _cfg(monkeypatch)
  actor = cfg.observations["actor"].terms
  critic = cfg.observations["critic"].terms
  assert list(actor)[-2:] == ["target_pos", "box_yaw"]
  assert actor["target_pos"].func is mdp.object_position_biased_b
  assert actor["target_pos"].params["object_name"] == bx.BOX
  assert actor["target_pos"].params["lag"] == actor["object_pos"].params["lag"]
  assert actor["box_yaw"].params["lag"] == actor["object_pos"].params["lag"]
  assert list(critic)[-4:] == [
    "grasped",
    "release_grade",
    "release_latched",
    "box_yaw",
  ]
  assert critic["target_pos"].func is mdp.object_position_b
  assert critic["target_pos"].params == {"object_name": bx.BOX}
  reset = cfg.events["reset_box"]
  assert reset.func is mdp.reset_box_in_area
  assert reset.params["min_distance"] == bx.BOX_MIN_SEPARATION
  assert reset.params["yaw"] == pytest.approx(math.pi / 4)
  events = list(cfg.events)
  assert events.index("reset_objects") < events.index("reset_box")
  goal_area = cfg.events["reset_objects"].params["goal_area"]
  assert goal_area["distance"] == bx.BOX_MIN_SEPARATION
  assert cfg.commands["goal_pose"].yaw_reference == bx.BOX
  assert "reset_tracker_yaw_bias" in cfg.events
  rl = kangaroo_pick_place_rl_cfg()
  assert rl.widen_actor_on_resume and rl.widen_critic_on_resume


def test_the_box_area_without_the_pose_is_the_zero_shot_cfg(monkeypatch):
  """Step 0 item 5: W2 box checkpoints read on the wide box with the S10 observation."""
  cfg = _cfg(monkeypatch, BOX_OBS=None)
  actor = cfg.observations["actor"].terms
  assert "box_yaw" not in actor and "box_yaw" not in cfg.observations["critic"].terms
  assert actor["target_pos"].func is mdp.box_goal_position_tracked_b
  assert cfg.events["reset_box"].func is mdp.reset_box_in_area
  assert cfg.commands["goal_pose"].yaw_reference == bx.BOX
  assert not kangaroo_pick_place_rl_cfg().widen_actor_on_resume


def test_target_b_with_the_s10_observation_is_bx_c(monkeypatch):
  """``TARGET=B`` and ``BOX_OBS`` unset build DEP1's BX-c: the box in B's target box at 0 +/- 5 deg,
  the goal point observed, the yaw graded against the world's axes, nothing widened in the actor.
  (The full resolved cfg was diffed against 0f02c9c's when S10.1 was built: S10.1 report.)"""
  cfg = _cfg(monkeypatch, SPAWN="W2", TARGET="B", EPISODE_S="10", BOX_OBS=None)
  actor = cfg.observations["actor"].terms
  assert list(actor)[-1] == "target_pos"
  assert actor["target_pos"].func is mdp.box_goal_position_tracked_b
  assert list(cfg.observations["critic"].terms)[-1] == "release_latched"
  reset = cfg.events["reset_box"]
  assert reset.func is mdp.reset_box_on_surface and reset.params["yaw"] == bx.BOX_YAW
  assert cfg.commands["goal_pose"].yaw_reference is None
  assert "reset_tracker_yaw_bias" not in cfg.events
  assert cfg.events["reset_objects"].params.get("goal_area") is None
  rl = kangaroo_pick_place_rl_cfg()
  assert not rl.widen_actor_on_resume and rl.widen_critic_on_resume


def test_the_wide_area_table_task_is_unchanged(monkeypatch):
  """``TARGET=WC`` with the table goal (DEP1's WC-c) builds S7-8's goal command, no box."""
  cfg = _cfg(monkeypatch, GOAL=None, BOX_OBS=None)
  assert "box" not in cfg.scene.entities
  assert cfg.commands["goal_pose"].min_object_distance == pytest.approx(0.10)
  assert cfg.events["reset_objects"].params["goal_area"]["distance"] == pytest.approx(
    0.10
  )
  assert list(cfg.observations["actor"].terms)[-1] == "target_pos"


def test_the_combinations_not_built_are_refused(monkeypatch):
  with pytest.raises(ValueError, match="cannot keep a box"):
    _cfg(monkeypatch, SPAWN="WCK", TARGET="WCK")
  with pytest.raises(ValueError, match="needs a box GOAL"):
    _cfg(monkeypatch, GOAL="table")
  with pytest.raises(ValueError, match="BOX_OBS"):
    _cfg(monkeypatch, BOX_OBS="goalpoint")
  assert bx.area_holds_the_box(WC.x, WC.y)
  assert not bx.area_holds_the_box(SPAWN_BOXES["WCK"].x, SPAWN_BOXES["WCK"].y)


# -- The box on the area -----------------------------------------------------------------


def test_every_box_is_in_the_area_clear_of_the_cube_at_any_yaw(monkeypatch):
  env = _env(monkeypatch, num_envs=64)
  table = SceneEntityCfg("table")
  box, cube = env.scene[bx.BOX], env.scene["object"]
  cx, cy = scene.TABLE_POS[0], scene.TABLE_POS[1]
  yaws = []
  for episode in range(3):
    if episode:
      with torch.inference_mode():
        env.reset()
    _hold(env, 2)
    centre = mdp.events._surface_centre_xy(env, table)
    rel = box.data.root_link_pos_w[:, :2] - centre
    assert (rel[:, 0] >= WC.x[0] - cx - 1e-3).all()
    assert (rel[:, 0] <= WC.x[1] - cx + 1e-3).all()
    assert (rel[:, 1] >= WC.y[0] - cy - 1e-3).all()
    assert (rel[:, 1] <= WC.y[1] - cy + 1e-3).all()
    top = mdp.surface_top_z(env, table)
    assert (box.data.root_link_pos_w[:, 2] - top).abs().max() < 2e-3
    d = torch.norm(
      box.data.root_link_pos_w[:, :2] - cube.data.root_link_pos_w[:, :2], dim=-1
    )
    # Not overlapping: the box's half-diagonal and the cube's are 10 cm together.
    assert d.min() >= bx.BOX_MIN_SEPARATION - 2e-3
    assert not mdp.object_in_box_bool(env, "object", bx.BOX, bx.BOX_WALLS).any()
    y = _yaw(box.data.root_link_quat_w)
    assert y.abs().max() <= math.pi / 4 + 1e-3
    yaws.append(y)
  yaws = torch.cat(yaws)
  # Uniform in +/- 45 deg: both halves and the far ends are drawn.
  assert (yaws > math.radians(30)).any() and (yaws < -math.radians(30)).any()
  assert env.observation_manager.compute()["actor"].shape[-1] == 40
  assert env.observation_manager.compute()["critic"].shape[-1] == 43
  env.close()


def test_the_goal_follows_a_pushed_and_turned_box(monkeypatch):
  env = _env(monkeypatch)
  _hold(env, 3)
  _put_box(env, math.radians(30), (0.02, -0.01))
  _hold(env, 3)
  command = env.command_manager.get_term("goal_pose")
  floor = mdp.box.box_floor_top_w(env, bx.BOX, scene.BOX_FLOOR)
  assert torch.allclose(command.command[:, :2], floor[:, :2], atol=1e-3)
  assert (command.command[:, 2] - floor[:, 2] - 0.025).abs().max() < 1e-3
  assert (command.box_displacement() - math.hypot(0.02, 0.01)).abs().max() < 2e-3
  env.close()


# -- The observation ---------------------------------------------------------------------


def test_the_observed_box_is_the_true_box_with_the_noise_off(monkeypatch):
  env = _env(monkeypatch, corrupt=False)
  _put_box(env, math.radians(-35))
  # Past the tracker's 8-step lag, and the box settled on its feet after the teleport (it bobs
  # ~0.1 mm in the first 10 steps, which the lagged reading then trails).
  _hold(env, 40)
  obs = env.observation_manager.compute()
  box = env.scene[bx.BOX]
  true_pos = mdp.object_position_b(env, bx.BOX)
  psi = mdp.yaw_relative_to_base(env, box.data.root_link_quat_w)
  true_yaw = torch.stack([torch.cos(4 * psi), torch.sin(4 * psi)], dim=-1)
  assert torch.allclose(obs["actor"][:, 35:38], true_pos, atol=1e-5)
  assert torch.allclose(obs["actor"][:, 38:40], true_yaw, atol=1e-5)
  assert torch.allclose(obs["critic"][:, 35:38], true_pos, atol=1e-5)
  assert torch.allclose(obs["critic"][:, 41:43], true_yaw, atol=1e-5)
  # The base is not turned: psi is the world yaw of the box.
  assert torch.allclose(psi, torch.full_like(psi, math.radians(-35)), atol=1e-3)
  env.close()


def test_the_box_yaw_is_the_same_every_90_deg(monkeypatch):
  env = _env(monkeypatch, corrupt=False)
  got = []
  for yaw in (math.radians(20), math.radians(20) - math.pi / 2, math.radians(-44)):
    _put_box(env, yaw)
    _hold(env, 10)
    got.append(env.observation_manager.compute()["actor"][:, 38:40].clone())
  assert torch.allclose(got[0], got[1], atol=1e-4)
  # Continuous across +/- 45 deg: -44 deg and +44 deg are 2 deg apart, 8 deg on the 4-fold circle.
  assert torch.allclose(
    got[2],
    torch.tensor(
      [math.cos(math.radians(-176)), math.sin(math.radians(-176))]
    ).expand_as(got[2]),
    atol=1e-3,
  )
  env.close()


def test_the_corrupted_box_yaw_carries_a_bias_and_a_jitter(monkeypatch):
  env = _env(monkeypatch, num_envs=32)
  _put_box(env, 0.0)
  _hold(env, 10)
  box = env.scene[bx.BOX]
  psi = mdp.yaw_relative_to_base(env, box.data.root_link_quat_w)
  enc = mdp.box_yaw_b(env, bx.BOX, bias_key=bx.BOX, noise=bx.BOX_YAW_NOISE)
  seen = torch.atan2(enc[:, 1], enc[:, 0]) / 4
  err = seen - psi
  assert err.abs().max() <= bx.BOX_YAW_BIAS + bx.BOX_YAW_NOISE + 1e-4
  assert err.abs().max() > 1e-3
  bias = mdp.tracker_yaw_bias(env, bx.BOX)
  assert bias.abs().max() <= bx.BOX_YAW_BIAS and bias.std() > 0
  env.close()


# -- The yaw grade and the box at 45 deg -------------------------------------------------


def test_the_yaw_grade_is_taken_against_the_box(monkeypatch):
  """A cube let go square to a box at 30 deg (turned 90 deg further) is graded aligned; one let go
  square to the world's axes is 30 deg off. The release latch, through a monkeypatched let-go."""
  env = _env(monkeypatch, num_envs=2)
  _hold(env, 2)
  _put_box(env, math.radians(30))
  _hold(env, 2)
  obj = env.scene["object"]
  pose = obj.data.root_link_pose_w.clone()
  zero = torch.zeros(2, device=env.device)
  pose[:, 3:7] = quat_from_euler_xyz(
    zero, zero, torch.tensor([math.radians(120), 0.0], device=env.device)
  )
  obj.write_root_link_pose_to_sim(pose)
  _hold(env, 1)
  held = {"v": True}
  monkeypatch.setattr(
    obs_mod,
    "_grasped_bool",
    lambda e, s: torch.full(
      (e.num_envs,), held["v"], dtype=torch.bool, device=e.device
    ),
  )
  monkeypatch.setattr(
    obs_mod,
    "_jaw_open_bool",
    lambda e, c, o: torch.ones(e.num_envs, dtype=torch.bool, device=e.device),
  )
  setattr(env, _WAS_LIFTED, torch.ones(2, dtype=torch.bool, device=env.device))
  jaw = SceneEntityCfg("robot", joint_names=("gripper_left_finger_joint",))
  jaw.resolve(env.scene)
  rim = bx.rim_surface()
  rim.resolve(env.scene)
  args = (
    "goal_pose",
    "object",
    "left_pad_contact",
    "object_table_contact",
    rim,
    jaw,
    0.15,
  )
  env.common_step_counter += 1
  _release_latch(env, *args)
  held["v"] = False
  env.common_step_counter += 1
  latch = _release_latch(env, *args)
  assert latch.released.all()
  assert math.degrees(float(latch.yaw[0])) == pytest.approx(0.0, abs=0.5)
  assert math.degrees(float(latch.yaw[1])) == pytest.approx(30.0, abs=0.5)
  # M6 x G(yaw, 10 deg): 1 for the aligned let-go.
  q = mdp.rewards.release_quality_of(
    torch.zeros(2),
    torch.zeros(2),
    0.02,
    0.01,
    0.01,
    0.25,
    latch.yaw.cpu(),
    math.radians(10),
  )
  assert q[0] == pytest.approx(1.0, abs=1e-3)
  assert q[1] < 0.3
  env.close()


def test_a_cube_in_a_box_at_45_deg_is_in_it_and_placed(monkeypatch):
  env = _env(monkeypatch)
  _hold(env, 3)
  _put_box(env, math.pi / 4)
  _hold(env, 3)
  _cube_in_box(env, 0.0, scene.BOX_WALL_HEIGHT + 0.01)
  _hold(env, 60)
  assert mdp.object_in_box_bool(env, "object", bx.BOX, bx.BOX_WALLS).all()
  setattr(
    env, _WAS_LIFTED, torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
  )
  jaw = SceneEntityCfg("robot", joint_names=("gripper_left_finger_joint",))
  jaw.resolve(env.scene)
  placed = _placed_bool(
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
  assert placed.all()
  # A cube across the rim at 45 deg -- its centre over the +x wall, in the box's frame -- is not in.
  _hold(env, 2)
  box = env.scene[bx.BOX]
  obj = env.scene["object"]
  pose = obj.data.root_link_pose_w.clone()
  r = scene.BOX_INNER / 2 + scene.BOX_WALL / 2
  pose[:, 0] = box.data.root_link_pos_w[:, 0] + r * math.cos(math.pi / 4)
  pose[:, 1] = box.data.root_link_pos_w[:, 1] + r * math.sin(math.pi / 4)
  pose[:, 2] = mdp.box_rim_z(env, bx.BOX, bx.BOX_WALLS) + 0.03
  obj.write_root_link_pose_to_sim(pose)
  obj.write_root_link_velocity_to_sim(torch.zeros_like(obj.data.root_link_vel_w))
  _hold(env, 1)
  assert not mdp.object_in_box_bool(env, "object", bx.BOX, bx.BOX_WALLS).any()
  env.close()


# -- The resume --------------------------------------------------------------------------


def _model(width: int, key: str):
  from rsl_rl.models import MLPModel
  from tensordict import TensorDict

  obs = TensorDict({key: torch.zeros(4, width)}, batch_size=[4])
  return MLPModel(obs, {key: [key]}, key, 3, (16, 8), "elu", True)


def test_the_actor_and_the_critic_widen_exactly_with_one_optimiser():
  """WX-a-FT from BX-RIM: the actor widens by 2 (the box yaw), the critic by 4 (PQ + the box yaw),
  through one optimiser -- each model's own moments only."""
  from tensordict import TensorDict

  torch.manual_seed(0)
  old_a, old_c = _model(38, "policy"), _model(39, "critic")
  new_a, new_c = _model(40, "policy"), _model(43, "critic")
  for m, w in ((old_a, 38), (old_c, 39)):
    m.train()
    m.obs_normalizer.update(1.0 + 2.0 * torch.randn(256, w))
  opt = torch.optim.Adam([*old_a.parameters(), *old_c.parameters()], lr=1e-3)
  xa, xc = torch.randn(32, 38), torch.randn(32, 39)
  loss = old_a(TensorDict({"policy": xa}, batch_size=[32])).pow(2).mean()
  loss = loss + old_c(TensorDict({"critic": xc}, batch_size=[32])).pow(2).mean()
  loss.backward()
  opt.step()
  loaded = {
    "actor_state_dict": {k: v.clone() for k, v in old_a.state_dict().items()},
    "critic_state_dict": {k: v.clone() for k, v in old_c.state_dict().items()},
    "optimizer_state_dict": opt.state_dict(),
  }
  params = [*new_a.parameters(), *new_c.parameters()]
  new_opt = torch.optim.Adam(params, lr=1e-3)
  assert widen_input(loaded, "actor_state_dict", new_a, params) == 2
  assert widen_input(loaded, "critic_state_dict", new_c, params) == 4
  new_a.load_state_dict(loaded["actor_state_dict"])
  new_c.load_state_dict(loaded["critic_state_dict"])
  new_opt.load_state_dict(loaded["optimizer_state_dict"])
  for m in (old_a, old_c, new_a, new_c):
    m.eval()
  with torch.no_grad():
    a0 = old_a(TensorDict({"policy": xa}, batch_size=[32]))
    a1 = new_a(
      TensorDict({"policy": torch.cat([xa, torch.rand(32, 2)], 1)}, batch_size=[32])
    )
    c0 = old_c(TensorDict({"critic": xc}, batch_size=[32]))
    c1 = new_c(
      TensorDict({"critic": torch.cat([xc, torch.rand(32, 4)], 1)}, batch_size=[32])
    )
  assert torch.allclose(a0, a1, atol=1e-6)
  assert torch.allclose(c0, c1, atol=1e-6)
  assert new_opt.state[new_a.mlp[0].weight]["exp_avg"].shape == (16, 40)
  assert new_opt.state[new_c.mlp[0].weight]["exp_avg"].shape == (16, 43)
  assert torch.equal(
    new_opt.state[new_c.mlp[0].weight]["exp_avg"][:, :39],
    opt.state[old_c.mlp[0].weight]["exp_avg"],
  )


def test_a_w2_actor_rescaled_to_the_wc_arm_commands_the_same_targets():
  """``CKPT_SPAWN=W2`` on ``SPAWN=WC``: arm 1 and arm 4 scales 0.90 -> 2.35 and 0.80 -> 1.79. The
  converted actor's mean action times the new scales is the old one's times the old scales, its
  noise in target space is the same, and under ``act`` its last-action inputs are read in the new
  units."""
  from tensordict import TensorDict

  torch.manual_seed(1)
  old_scale = torch.tensor([0.90, 0.85, 0.65, 0.80, 1.55, 2.40, 1.20, 0.5])
  new_scale = torch.tensor([2.35, 0.85, 0.65, 1.79, 1.55, 2.40, 1.20, 0.5])
  ratio = old_scale / new_scale
  cols = list(
    range(15, 23)
  )  # the actor's `actions` term (after joint_pos 8, joint_vel 7)
  old_a, new_a, old_c = _model(38, "policy"), _model(38, "policy"), _model(39, "critic")
  # An 8-action actor: `_model` makes 3 outputs, so rebuild the last layer.
  from rsl_rl.models import MLPModel

  obs = TensorDict({"policy": torch.zeros(4, 38)}, batch_size=[4])
  old_a = MLPModel(obs, {"policy": ["policy"]}, "policy", 8, (16, 8), "elu", True)
  new_a = MLPModel(obs, {"policy": ["policy"]}, "policy", 8, (16, 8), "elu", True)
  old_a.train()
  old_a.obs_normalizer.update(0.3 * torch.randn(256, 38))
  old_c.train()
  old_c.obs_normalizer.update(0.02 + 0.3 * torch.randn(256, 39))
  sd = {k: v.clone() for k, v in old_a.state_dict().items()}
  sd["distribution.log_std_param"] = torch.full((8,), -1.5)
  loaded = {
    "actor_state_dict": sd,
    "critic_state_dict": {k: v.clone() for k, v in old_c.state_dict().items()},
    "optimizer_state_dict": {"state": {}},
  }
  rescale_actions(
    loaded, ratio, cols, {"actor_state_dict": new_a}, list(new_a.parameters())
  )
  log_std = loaded["actor_state_dict"].pop("distribution.log_std_param")
  new_a.load_state_dict(loaded["actor_state_dict"])
  old_a.eval()
  new_a.eval()
  x = torch.randn(32, 38)
  x_new = x.clone()
  x_new[:, cols] = x[:, cols] * ratio
  with torch.no_grad():
    a_old = old_a(TensorDict({"policy": x}, batch_size=[32]))
    a_new = new_a(TensorDict({"policy": x_new}, batch_size=[32]))
  assert torch.allclose(a_new * new_scale, a_old * old_scale, atol=1e-5)
  assert torch.allclose(
    torch.exp(log_std) * new_scale, torch.exp(torch.full((8,), -1.5)) * old_scale
  )
  crit = loaded["critic_state_dict"]["obs_normalizer._mean"]
  assert torch.allclose(
    crit[:, cols], old_c.state_dict()["obs_normalizer._mean"][:, cols] * ratio
  )
  # The critic computes the same value on the new-unit inputs.
  new_c = _model(39, "critic")
  new_c.load_state_dict(loaded["critic_state_dict"])
  old_c.eval()
  new_c.eval()
  xc = torch.randn(32, 39)
  xc_new = xc.clone()
  xc_new[:, cols] = xc[:, cols] * ratio
  with torch.no_grad():
    assert torch.allclose(
      new_c(TensorDict({"critic": xc_new}, batch_size=[32])),
      old_c(TensorDict({"critic": xc}, batch_size=[32])),
      atol=1e-5,
    )


def test_ckpt_spawn_names_the_ratios(monkeypatch):
  _set_env(monkeypatch, CKPT_SPAWN="W2")
  ratio = kangaroo_pick_place_rl_cfg().rescale_actions_on_load
  assert ratio == {
    "arm_left_1_joint": pytest.approx(0.90 / 2.35),
    "arm_left_4_joint": pytest.approx(0.80 / 1.79),
  }
  _set_env(monkeypatch)
  assert kangaroo_pick_place_rl_cfg().rescale_actions_on_load is None
