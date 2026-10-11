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


def _flat(obj, prefix=""):
  """Every leaf of a cfg as path -> repr, closures' addresses dropped (`test_pick_place_s113`)."""
  import dataclasses
  import re

  out = {}
  if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
    for f in dataclasses.fields(obj):
      out.update(_flat(getattr(obj, f.name), f"{prefix}.{f.name}"))
  elif isinstance(obj, dict):
    for k, v in obj.items():
      out.update(_flat(v, f"{prefix}[{k!r}]"))
  elif isinstance(obj, (list, tuple)):
    for i, v in enumerate(obj):
      out.update(_flat(v, f"{prefix}[{i}]"))
  else:
    out[prefix] = re.sub(r" at 0x[0-9a-f]+", "", repr(obj))
  return out


def _diff(a, b):
  fa, fb = _flat(a), _flat(b)
  return {
    k: (fa.get(k), fb.get(k)) for k in fa.keys() | fb.keys() if fa.get(k) != fb.get(k)
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
  # The plant is untouched: the same robot entity (the action term carries the scales).
  assert _diff(lh.scene.entities["robot"], base.scene.entities["robot"]) == {}
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


def test_lh2_adds_roll_level_and_mid_goal_and_changes_nothing_else(monkeypatch):
  lh = _cfg(monkeypatch, S13="L-H")
  lh2 = _cfg(monkeypatch, S13="L-H2")
  # L-H is as trained (side_graded, no new terms); L-H2 adds goal_mid and level_lifted.
  assert lh.rewards["grasp"].func is mdp.side_graded
  assert not {"goal_mid", "level_lifted"} & set(lh.rewards)
  assert set(lh2.rewards) - set(lh.rewards) == {"goal_mid", "level_lifted"}
  for name in ("grasp", "lift", "goal", "goal_near", "goal_mid"):
    term = lh2.rewards[name]
    assert term.func is mdp.side_roll_graded, name
    assert term.params["roll_std"] == H.ROLL_STD, name
  # goal_mid is M2's own 3 cm term; goal_near stays at 1 cm.
  assert lh2.rewards["goal_mid"].params["std"] == pytest.approx(0.03)
  assert lh2.rewards["goal_near"].params["std"] == H.NEAR_STD
  lv = lh2.rewards["level_lifted"]
  assert lv.weight == H.LEVEL_LIFTED_WEIGHT and lv.params["std"] == H.LEVEL_LIFTED_STD
  # held (the eval's bar), the observations' widths, the scene and the events are L-H's.
  assert _diff(lh2.rewards["held"], lh.rewards["held"]) == {}
  assert list(lh2.observations["actor"].terms) == list(lh.observations["actor"].terms)
  assert list(lh2.observations["critic"].terms) == list(lh.observations["critic"].terms)
  assert lh2.observations["critic"].terms["side_grade"].func is mdp.side_roll_grade_obs
  assert _diff(lh2.scene, lh.scene) == {} and _diff(lh2.events, lh.events) == {}


def test_lh3_grades_the_goal_terms_by_the_block_not_the_hand(monkeypatch):
  lh2 = _cfg(monkeypatch, S13="L-H2")
  lh3 = _cfg(monkeypatch, S13="L-H3")
  assert set(lh3.rewards) == set(lh2.rewards)
  for name in H.LEVEL_GOAL_TERMS:
    t2, t3 = lh2.rewards[name], lh3.rewards[name]
    assert t3.func is mdp.level_block_graded, name
    assert t3.params["inner"] is t2.params["inner"], name
    assert t3.weight == t2.weight, name
    assert t3.params["tilt_std"] == H.LEVEL_GOAL_TILT_STD
    assert t3.params["elevation_std"] == H.LEVEL_GOAL_ELEVATION_STD
    # inner's own parameters (std, sensor, command ...) pass through unchanged.
    for k, v in t2.params.items():
      if k not in ("inner", "side_cfg", "side_std", "roll_std"):
        assert t3.params[k] == v, (name, k)
  for name in set(lh2.rewards) - set(H.LEVEL_GOAL_TERMS):
    assert _diff(lh3.rewards[name], lh2.rewards[name]) == {}, name
  assert _diff(lh3.observations, lh2.observations) == {}


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


def test_roll_grade_and_level_lifted_on_the_env(lh_env):
  env = lh_env
  side = SceneEntityCfg(
    "robot", site_names=(KANGAROO_GRIPPERS_LEFT_ARM.gripper("left").tcp_site,)
  )
  side.resolve(env.scene)
  roll = mdp.hand_roll(env, side)
  assert bool(((roll >= 0.0) & (roll <= math.pi / 2)).all())
  # The roll grade only ever lowers the side grade.
  g = mdp.side_grade(env, side, H.SIDE_STD)
  gr = mdp.side_roll_grade(env, side, H.SIDE_STD, H.ROLL_STD)
  assert bool((gr <= g + 1e-6).all())
  want = g * torch.exp(-torch.square(roll / H.ROLL_STD))
  assert torch.allclose(gr, want, atol=1e-6)
  # The block on the table, not grasped: level_lifted pays nothing.
  sensor = KANGAROO_GRIPPERS_LEFT_ARM.gripper("left").contact_sensor_name
  table = SceneEntityCfg(TABLE)
  table.resolve(env.scene)
  lv = mdp.level_lifted(
    env,
    OBJECT,
    sensor,
    table,
    scene.BLOCK_HEIGHT / 2,
    H.LEVEL_LIFTED_CLEAR,
    H.LEVEL_LIFTED_STD,
  )
  assert float(lv.abs().max()) == 0.0


def test_level_block_grade_on_the_env(lh_env):
  env = lh_env
  side = SceneEntityCfg(
    "robot", site_names=(KANGAROO_GRIPPERS_LEFT_ARM.gripper("left").tcp_site,)
  )
  side.resolve(env.scene)
  ones = lambda env, **_: torch.ones(env.num_envs, device=env.device)  # noqa: E731
  g = mdp.level_block_graded(
    env,
    ones,
    OBJECT,
    side,
    H.LEVEL_GOAL_TILT_STD,
    H.ROLL_STD,
    H.LEVEL_GOAL_ELEVATION_STD,
  )
  want = (
    torch.exp(-torch.square(mdp.axis_tilt(env, OBJECT) / H.LEVEL_GOAL_TILT_STD))
    * torch.exp(-torch.square(mdp.hand_roll(env, side) / H.ROLL_STD))
    * torch.exp(
      -torch.square(mdp.approach_elevation(env, side) / H.LEVEL_GOAL_ELEVATION_STD)
    )
  )
  assert torch.allclose(g, want)
  # The start pose's hand points ~20 deg down: the loose elevation grade still pays most of it.
  elev = mdp.approach_elevation(env, side)
  assert float(
    torch.exp(-torch.square(elev / H.LEVEL_GOAL_ELEVATION_STD)).mean()
  ) > float(torch.exp(-torch.square(elev / H.SIDE_STD)).mean())


def test_lhp_pushes_the_top_from_4s_and_not_before(lh_env):
  env = lh_env
  with torch.inference_mode():
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


# -- R-I: the insertion into the held block (`pick_place/insert_held.py`) ----------------------------

RI = {
  **{k: v for k, v in BS12_LH.items()},
  "GOAL": "insert",
  "SHAPE": "pin",
  "TARGET": "WC",
  "CRITIC_OBS": "q",
}


def test_ri_cfg_holds_the_block_and_keeps_ins_actor(monkeypatch):
  from pal_mjlab.tasks.manipulation.pick_place import insert_held as IH

  base = _cfg(monkeypatch, **RI)
  ri = _cfg(monkeypatch, **RI, S13="R-I")
  assert IH.HOLDER in ri.scene.entities and IH.HOLDER not in base.scene.entities
  assert list(ri.observations["actor"].terms) == list(base.observations["actor"].terms)
  assert list(ri.observations["critic"].terms) == [
    *base.observations["critic"].terms,
    "pin_through",
  ]
  assert ri.rewards["release"].weight == 0.0 and ri.rewards["placed"].weight == 0.0
  assert "retreat" not in ri.rewards and "through" in ri.rewards
  assert ri.events["reset_block"].func is mdp.reset_held_block
  assert (
    "keepout" in ri.events["reset_objects"].params
  )  # §12's HOME_HAND_KEEPOUT kept for the pin
  wide = _cfg(monkeypatch, **RI, S13="R-I-W")
  assert (
    wide.events["reset_block"].params["half_xy"]
    > ri.events["reset_block"].params["half_xy"]
  )


def test_ri2_grades_through_and_depth_by_grasp_height_and_ends_on_hard_contact(
  monkeypatch,
):
  from pal_mjlab.tasks.manipulation.pick_place import insert_held as IH

  ri = _cfg(monkeypatch, **RI, S13="R-I")
  ri2 = _cfg(monkeypatch, **RI, S13="R-I2")
  # R-I is as trained: no grasp-height gate, no hard-contact termination.
  assert ri.rewards["through"].func is mdp.pin_through_held
  assert ri.rewards["depth"].func is mdp.over_block_gated
  assert "arm_holder_force" not in ri.terminations
  # R-I2: through and D wrapped (D keeps its over-block gate inside), the episode ends above 30 N.
  t = ri2.rewards["through"]
  assert (
    t.func is mdp.grasp_high_gated and t.params["high_inner"] is mdp.pin_through_held
  )
  d = ri2.rewards["depth"]
  assert (
    d.func is mdp.grasp_high_gated and d.params["high_inner"] is mdp.over_block_gated
  )
  assert d.params["inner"] is mdp.insert_depth
  for term in (t, d):
    assert term.params["high_low"] == IH.GRASP_HIGH_LOW
    assert term.params["high_full"] == IH.GRASP_HIGH_FULL
  term = ri2.terminations["arm_holder_force"]
  assert term.params == {
    "sensor_name": IH.HH_SENSOR,
    "force_threshold": IH.HH_TERMINATE_N,
  }
  assert not term.time_out
  # Nothing else moves: the other rewards, the observations, the scene and the events are R-I's.
  for name in set(ri.rewards) - set(IH.GRASP_HIGH_GATED):
    assert _diff(ri2.rewards[name], ri.rewards[name]) == {}, name
  assert list(ri2.observations["actor"].terms) == list(ri.observations["actor"].terms)
  assert list(ri2.observations["critic"].terms) == list(ri.observations["critic"].terms)
  assert _diff(ri2.events, ri.events) == {}


@pytest.mark.parametrize(
  "changes", [{"GOAL": "insert-lad"}, {"GOAL": None, "SHAPE": None, "TARGET": None}]
)
def test_ri_refuses_anything_but_ins_insert(monkeypatch, changes):
  with pytest.raises(ValueError):
    _cfg(monkeypatch, **{**RI, "S13": "R-I", **changes})


@pytest.fixture(scope="module")
def ri_env():
  mp = pytest.MonkeyPatch()
  cfg = _cfg(mp, num_envs=8, **RI, S13="R-I")
  cfg.observations["actor"].enable_corruption = False
  cfg.seed = 5
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  with torch.inference_mode():
    env.reset()
  yield env
  env.close()
  mp.undo()


def test_ri_block_hangs_at_h_and_the_pin_stands_clear(ri_env):
  from pal_mjlab.tasks.manipulation.pick_place import insert_held as IH
  from pal_mjlab.tasks.manipulation.pick_place.insert import HOME_HAND_KEEPOUT

  env = ri_env
  _hold(env, 50)
  top = mdp.surface_top_z(env, SceneEntityCfg(TABLE))
  block = env.scene["block"]
  pos = block.data.root_link_pos_w - env.scene.env_origins
  # H: the block's bottom 10 cm (+/- 1 cm) over the table at (0.45, 0.00) +/- 1.5 cm, sagging ~1 mm on the weld.
  assert bool((torch.abs(pos[:, 0] - H.HOLD_XY[0]) < 0.016).all())
  assert bool((torch.abs(pos[:, 1] - H.HOLD_XY[1]) < 0.016).all())
  dz = pos[:, 2] + env.scene.env_origins[:, 2] - top
  assert bool(((dz > H.HOLD_DZ - 0.013) & (dz < H.HOLD_DZ + 0.011)).all()), dz
  d, ang = mdp.held_block_offset(env, IH.HOLDER, "block")
  assert float(d.max()) < 0.003 and float(ang.max()) < math.radians(1.0)
  # The pin stands on the table in its area, out of the start pose's finger keepout.
  pin = env.scene[OBJECT].data.root_link_pos_w - env.scene.env_origins
  assert bool((pin[:, 1] > IH.PIN_AREA_Y[0] - 0.03).all())
  x0, x1, y0, y1 = HOME_HAND_KEEPOUT
  inside = (pin[:, 0] > x0) & (pin[:, 0] < x1) & (pin[:, 1] > y0) & (pin[:, 1] < y1)
  assert not bool(inside.any())
  assert float(env.reward_manager.get_term_cfg("through").weight) == IH.THROUGH_WEIGHT


def test_through_reads_a_pin_pushed_through_the_held_block(ri_env):
  env = ri_env
  geo = {
    k: env.cfg.rewards["depth"].params[k]
    for k in ("object_name", "block_name", "block_height", "hole_d")
  }
  block = env.scene["block"]
  pin = env.scene[OBJECT]
  with torch.inference_mode():
    bp, bq = block.data.root_link_pos_w.clone(), block.data.root_link_quat_w.clone()
    half = scene.PIN_LENGTH / 2
    from mjlab.utils.lab_api.math import quat_apply

    up = torch.zeros_like(bp)
    up[:, 2] = 1.0
    axis = quat_apply(bq, up)
    # (bottom under the block's bottom face (negative: above it), through, the pin's top above the block's top):
    # pushed through; not yet out of the bottom; and the whole pin *under* the block on the hole's axis, which
    # `InsertState.through` alone reads as through (the early R-I-FT check, 2026-10-08) and `held_through` does not.
    for below, want, over in (
      (0.002, True, True),
      (-0.010, False, True),
      (0.095, False, False),
    ):
      centre = bp + axis * (half - below)
      pin.write_root_link_pose_to_sim(torch.cat([centre, bq], dim=-1))
      pin.write_root_link_velocity_to_sim(
        torch.zeros(env.num_envs, 6, device=env.device)
      )
      env.sim.forward()
      st = mdp.insert_state(env, **geo)
      assert bool((st.through() == (below >= 0.0)).all()), (below, st.height)
      assert bool((mdp.held_through(env, **geo) == want).all()), (below, st.height)
      assert bool((mdp.pin_over_block(env, **geo) == over).all()), below


def test_grasp_height_and_its_grade_on_the_env(ri_env):
  from pal_mjlab.tasks.manipulation.pick_place import insert_held as IH

  env = ri_env
  geo = {
    k: env.cfg.rewards["depth"].params[k]
    for k in ("object_name", "block_name", "block_height", "hole_d")
  }
  tcp = env.reward_manager.get_term_cfg("reach").params["asset_cfg"]
  with torch.inference_mode():
    h = mdp.pin_grasp_height(env, **geo, tcp_cfg=tcp)
    g = mdp.grasp_high_grade(
      env, **geo, tcp_cfg=tcp, low=IH.GRASP_HIGH_LOW, full=IH.GRASP_HIGH_FULL
    )
    # By hand, from the pin's own pose (an earlier test may have left it tilted in the block): its axis turned up,
    # its top half a length along it, the tool point projected on it.
    from mjlab.utils.lab_api.math import quat_apply

    pin = env.scene[OBJECT]
    tool = env.scene["robot"].data.site_pos_w[:, tcp.site_ids].squeeze(1)
    ez = torch.zeros_like(tool)
    ez[:, 2] = 1.0
    axis = quat_apply(pin.data.root_link_quat_w, ez)
    axis = torch.where(axis[:, 2:3] < 0.0, -axis, axis)
    top = pin.data.root_link_pos_w + axis * scene.PIN_LENGTH / 2
    by_hand = ((tool - top) * axis).sum(-1)
  assert torch.allclose(h, by_hand, atol=1e-4), (h, by_hand)
  want = ((h - IH.GRASP_HIGH_LOW) / (IH.GRASP_HIGH_FULL - IH.GRASP_HIGH_LOW)).clamp(
    0, 1
  )
  assert torch.allclose(g, want)
  assert bool(((g >= 0.0) & (g <= 1.0)).all())


def test_ri_pays_the_approach_terms_only_over_the_block(ri_env):
  from pal_mjlab.tasks.manipulation.pick_place import insert_held as IH

  env = ri_env
  for name in IH.GATED_OVER_BLOCK:
    term = env.reward_manager.get_term_cfg(name)
    assert term.func is mdp.over_block_gated, name
    assert term.params["inner"] in (
      mdp.insert_approach,
      mdp.insert_align,
      mdp.insert_depth,
    ), name
  assert env.reward_manager.get_term_cfg("through").func is mdp.pin_through_held
