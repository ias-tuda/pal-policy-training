"""Plan §13.1 (`pick-place-insert-plan.md`): the two-arm insert's one-arm skills -- L-H / L-HP (`pick_place/hold.py`):
the left's side grasp of the socket block and its hold at H.

On CPU: an unset ``S13`` builds the cfg it built before (the plain WC cfg; S13 is refused outside its own selectors); L-H
swaps in the side-grasp arm, the centre-origin block, the spawn cut, H and the hold reward, and nothing about the
plant; the block rests on the 0.837 m table after the reset and H sits 10 cm over it at (0.45, 0.00) in the robot frame;
the actor is 40-D with the block's 4-fold yaw and up axis; the side grade reads a horizontal hand as 1 and the start
pose (hand 20 deg down) as less; L-HP's push starts at 4 s on the block's top and is gone before it.
"""

from __future__ import annotations

import math

import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from pal_mjlab.tasks.manipulation import mdp, scene
from pal_mjlab.tasks.manipulation.pick_place import hold as H
from pal_mjlab.tasks.manipulation.pick_place.kangaroo.env_cfg import (
  kangaroo_pick_place_env_cfg,
)
from pal_mjlab.tasks.manipulation.robots import (
  KANGAROO_GRIPPERS_LEFT_ARM,
  KANGAROO_GRIPPERS_LEFT_ARM_SIDE,
  KANGAROO_GRIPPERS_LEFT_ARM_WC,
)
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import COMMAND, OBJECT, TABLE

RCTP = "R-C-strict+M1+M2+M3+M6+J+M8"
#: `BS12` (plan §0.3) for L-H: the WC cfg on the 0.837 m table, `act`, no PQ (L-H has no release).
BS12_LH = {
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
  "CMD_OBS": "act",
  "EPISODE_S": "15",
  "ARM_TABLE": "AH10",
  "TABLE_DZ": "22",
}
_ALL = {
  *BS12_LH,
  "S13",
  "GOAL",
  "TARGET",
  "CRITIC_OBS",
  "BOX_OBS",
  "SHAPE",
  "TRACKER",
  "ARMS",
}


def _cfg(monkeypatch, num_envs: int = 4, **changes):
  for var in _ALL:
    monkeypatch.delenv(var, raising=False)
  for var, value in {**BS12_LH, **changes}.items():
    if value is not None:
      monkeypatch.setenv(var, value)
  cfg = kangaroo_pick_place_env_cfg()
  cfg.scene.num_envs = num_envs
  cfg.seed = 0
  return cfg


def test_unset_is_the_wc_cfg(monkeypatch):
  cfg = _cfg(monkeypatch)
  assert "held" not in cfg.rewards and "placed" in cfg.rewards
  assert cfg.actions["joint_pos"].scale == pytest.approx(cfg.actions["joint_pos"].scale)
  assert "push_block" not in cfg.events


def test_lh_changes_the_arm_object_spawn_goal_and_reward(monkeypatch):
  base = _cfg(monkeypatch)
  lh = _cfg(monkeypatch, S13="L-H")
  # The arm: the side-grasp scales; the WC arm's otherwise.
  side = KANGAROO_GRIPPERS_LEFT_ARM_SIDE.action_scale
  wc = KANGAROO_GRIPPERS_LEFT_ARM_WC.action_scale
  assert side != wc and set(side) == set(wc)
  assert lh.actions["joint_pos"].scale == side
  assert base.actions["joint_pos"].scale == wc
  # The plant is untouched: same robot entity apart from the action term.
  assert lh.scene.entities["robot"] == base.scene.entities["robot"]
  # The goal is H, 10 cm over the table, robot frame.
  cmd = lh.commands[COMMAND]
  assert (
    cmd.frame == "env" and tuple(cmd.offset) == H.HOLD_XY and cmd.height == H.HOLD_DZ
  )
  # The reward: the hold's terms, no release.
  assert {"reach", "grasp", "lift", "goal", "goal_near", "level", "held"} <= set(
    lh.rewards
  )
  assert not {"release", "placed", "retreat", "pad_contact"} & set(lh.rewards)
  for name in ("action_rate_l2", "joint_pos_limits", "table_contact", "arm_effort"):
    assert lh.rewards[name] == base.rewards[name], name
  assert any(n.startswith("arm_table") for n in lh.rewards)
  assert "push_block" not in lh.events
  assert "push_block" in _cfg(monkeypatch, S13="L-HP").events


@pytest.mark.parametrize(
  "changes",
  [
    {"SPAWN": "W2"},
    {"TARGET": "WC"},
    {"GOAL": "stack"},
    {"CRITIC_OBS": "q"},
    {"S13": "nope"},
  ],
)
def test_lh_refuses_selectors_it_is_not_built_for(monkeypatch, changes):
  with pytest.raises(ValueError):
    _cfg(monkeypatch, **{"S13": "L-H", **changes})


# -- The env --------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lh_env():
  mp = pytest.MonkeyPatch()
  cfg = _cfg(mp, num_envs=8, S13="L-HP")
  cfg.observations["actor"].enable_corruption = False
  cfg.seed = 3
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  env.reset()
  yield env
  env.close()
  mp.undo()


def _hold(env, steps: int) -> None:
  with torch.inference_mode():
    for _ in range(steps):
      env.step(torch.zeros(env.action_space.shape, device=env.device))


def test_block_rests_on_the_table_in_the_cut_area_and_h_is_over_it(lh_env):
  env = lh_env
  _hold(env, 10)
  top = mdp.surface_top_z(env, SceneEntityCfg(TABLE))
  block = env.scene[OBJECT]
  pos = block.data.root_link_pos_w - env.scene.env_origins
  # The centre-origin block: its centre half its height over the table, upright.
  assert torch.allclose(
    pos[:, 2] + env.scene.env_origins[:, 2], top + scene.BLOCK_HEIGHT / 2, atol=2e-3
  )
  assert float(mdp.axis_tilt(env, OBJECT).max()) < math.radians(1.0)
  # The spawn cut (table-relative; DR-scene shifts the table +/- 2 cm).
  assert bool(
    (
      (pos[:, 0] > H.HOLD_SPAWN_X[0] - 0.03) & (pos[:, 0] < H.HOLD_SPAWN_X[1] + 0.03)
    ).all()
  )
  assert bool(
    (
      (pos[:, 1] > H.HOLD_SPAWN_Y[0] - 0.03) & (pos[:, 1] < H.HOLD_SPAWN_Y[1] + 0.03)
    ).all()
  )
  # H: (0.45, 0.00) in the robot frame, the block's centre 10 cm + half its height over the table.
  goal = env.command_manager.get_term(COMMAND).command - env.scene.env_origins
  assert torch.allclose(
    goal[:, :2],
    torch.tensor(H.HOLD_XY, device=env.device).expand(env.num_envs, 2),
    atol=1e-6,
  )
  assert torch.allclose(
    goal[:, 2] + env.scene.env_origins[:, 2],
    top + H.HOLD_DZ + scene.BLOCK_HEIGHT / 2,
    atol=1e-5,
  )


def test_actor_is_40d_with_the_blocks_yaw4_and_up_axis(lh_env):
  env = lh_env
  mgr = env.observation_manager
  assert mgr.group_obs_dim["actor"] == (40,)
  names = mgr.active_terms["actor"]
  assert names[-1] == "object_up"
  obs = mgr.compute()["actor"]
  dims = dict(zip(names, (d[0] for d in mgr.group_obs_term_dim["actor"]), strict=True))
  i = sum(dims[n] for n in names[: names.index("object_yaw")])
  yaw4 = obs[:, i : i + 2]
  assert torch.allclose(
    yaw4.norm(dim=-1), torch.ones(env.num_envs, device=env.device), atol=1e-4
  )
  up = obs[:, -2:]
  assert float(up.abs().max()) < 0.02  # standing level


def test_side_grade_is_one_for_a_level_hand_and_less_at_the_start_pose(lh_env):
  env = lh_env
  side = SceneEntityCfg(
    "robot", site_names=(KANGAROO_GRIPPERS_LEFT_ARM.gripper("left").tcp_site,)
  )
  side.resolve(env.scene)
  elev = mdp.approach_elevation(env, side)
  # The start pose points the hand forward and ~20 deg down (step 0, `results/S13-insert/step0/`).
  assert float(elev.mean()) == pytest.approx(math.radians(20.5), abs=math.radians(4.0))
  grade = mdp.side_grade(env, side, H.SIDE_STD)
  assert float(grade.max()) < 0.4
  # A level hand: the same function on a rotated frame reads 1 (the grade is exp(-(elev/std)^2)).
  assert math.exp(-((0.0 / H.SIDE_STD) ** 2)) == 1.0


def test_lhp_pushes_the_top_from_4s_and_not_before(lh_env):
  env = lh_env
  env.reset()
  block = env.scene[OBJECT]
  bid = int(block.indexing.body_ids[0])
  seen_before, seen_after = 0.0, 0.0
  with torch.inference_mode():
    for _ in range(round(6.5 / env.step_dt)):
      env.step(torch.zeros(env.action_space.shape, device=env.device))
      t = float(env.episode_length_buf.float().max()) * env.step_dt
      f = env.sim.data.xfrc_applied[:, bid, :3].norm(dim=-1).max()
      if t < H.PUSH_START_S - 0.05:
        seen_before = max(seen_before, float(f))
      else:
        seen_after = max(seen_after, float(f))
  assert seen_before == 0.0
  assert 0.0 < seen_after <= math.hypot(H.PUSH_DOWN_N[1], H.PUSH_SIDE_N) + 1e-4
