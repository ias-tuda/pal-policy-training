"""Plan §15 (`pick-place-insert-plan.md` §15): a Ø 50 mm ball dropped into the chamfer of §12's block, ``GOAL=ball-seat``
-- step 0 item 1.

The env is `BS12` (plan §0.3) with ``SHAPE=ball50`` and ``GOAL=ball-seat``, built on CPU. What is checked: the cfg's widths
(§12's 42 / 46), terms, codes, refusals and the resume widening; PB-DR's radius draw; the reset puts the ball and the
block on WC, clear of each other, the block out of the start-hand keepout, and neither moves when nothing touches them;
the actor's ball and block observations are the truth with the noise off and the yaw / tilt slots constant; a ball
dropped on the axis comes to rest in the seat at `ball_seat.SEAT`'s height (positive control) and one dropped 30 mm off
or set on the land does not; the goal terms pay nothing before the lift (L4) and nothing to a held ball dragged into the
block's side (the clearance gate); pushing the block costs; and the per-step reward rises along the task with the
let-go >= 3 over the hold (L5).
"""

from __future__ import annotations

import math

import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply
from pal_mjlab.tasks.manipulation import mdp, scene
from pal_mjlab.tasks.manipulation.mdp.events import in_keepout, surface_top_z
from pal_mjlab.tasks.manipulation.mdp.observations import _was_lifted_bool
from pal_mjlab.tasks.manipulation.pick_place import ball_seat as bs
from pal_mjlab.tasks.manipulation.pick_place.insert import BLOCK
from pal_mjlab.tasks.manipulation.pick_place.kangaroo.env_cfg import (
  kangaroo_pick_place_env_cfg,
  kangaroo_pick_place_rl_cfg,
)
from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import SPAWN_BOXES

RCTP = "R-C-strict+M1+M2+M3+M6+J+M8"
#: `BS12` + PB-BX (plan §15.4, every code but its start).
PB_ENV = {
  "REWARD": RCTP,
  "DR": "DR-scene",
  "PPO": "PPO-stack",
  "SLEW": "stack",
  "BASELINE": "B",
  "SPAWN": "WC",
  "TOOL_SPEED": "none",
  "GAINS": "G20",
  "EFFORT": "CTP",
  "LAG": "FP+JSW",
  "ACTION": "abs",
  "CMD_OBS": "act",
  "TARGET": "WC",
  "EPISODE_S": "15",
  "CRITIC_OBS": "q",
  "ARM_TABLE": "AH10",
  "SHAPE": "ball50",
  "GOAL": "ball-seat",
  "TABLE_DZ": "22",
}
WC = SPAWN_BOXES["WC"]
H = scene.BLOCK_HEIGHT
GRASP = "left_pad_contact"


def _set_env(monkeypatch, **changes) -> None:
  for var, value in {**PB_ENV, **changes}.items():
    if value is None:
      monkeypatch.delenv(var, raising=False)
    else:
      monkeypatch.setenv(var, value)
  for var in ("OCC", "OBJ_OBS", "GATE", "BOX_FEET", "BOX_OBS", "CKPT_SPAWN", "TRACKER"):
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
  """PB-BX's env on CPU, 8 envs, the actor's noise off."""
  mp = pytest.MonkeyPatch()
  cfg = _cfg(mp, num_envs=8)
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


def _place(entity, pos: torch.Tensor) -> None:
  quat = torch.zeros(pos.shape[0], 4, device=pos.device)
  quat[:, 0] = 1.0
  entity.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1))
  entity.write_root_link_velocity_to_sim(
    torch.zeros(pos.shape[0], 6, device=pos.device)
  )


def _over_block(env, xy_off: float, z: torch.Tensor) -> torch.Tensor:
  """World points ``xy_off`` along the block's x from its axis at heights ``z`` over its bottom face. (B, 3)"""
  block = env.scene[BLOCK]
  off = torch.zeros(env.num_envs, 3, device=env.device)
  off[:, 0] = xy_off
  off[:, 2] = z
  return block.data.root_link_pos_w + quat_apply(block.data.root_link_quat_w, off)


def _state(env):
  return mdp.seat_state(env, "object", BLOCK, H, bs.SEAT)


# -- The cfg -------------------------------------------------------------------------------------------


def test_the_ball_seat_cfg(monkeypatch):
  cfg = _cfg(monkeypatch)
  actor = list(cfg.observations["actor"].terms)
  critic = list(cfg.observations["critic"].terms)
  # §12's layout: the yaw / target / block-yaw / tilt slots in §12's order, the critic's three appended after PQ's.
  assert actor[-5:] == ["object_pos", "object_yaw", "target_pos", "box_yaw", "pin_tilt"]
  assert critic[-5:] == [
    "release_grade",
    "release_latched",
    "box_yaw",
    "pin_tilt",
    "pin_seated",
  ]
  assert cfg.observations["actor"].terms["object_yaw"].func is mdp.no_yaw
  assert cfg.observations["actor"].terms["pin_tilt"].func is mdp.no_tilt
  rw = cfg.rewards
  assert rw["release"].func is mdp.seat_release and rw["release"].weight == 4.0
  assert rw["placed"].weight == 8.0 and rw["placed"].params["threshold"] == bs.SEAT_TOL
  assert rw["placed"].params["velocity_threshold"] == bs.SEAT_SPEED
  assert (
    rw["goal"].func is mdp.seat_approach and rw["goal"].weight == bs.APPROACH_WEIGHT
  )
  assert rw["goal_near"].weight == bs.NEAR_WEIGHT
  assert rw["over_seat"].weight == bs.OVER_WEIGHT
  assert rw["retreat"].params["clear_height"] == bs.RETREAT_CLEAR
  assert rw["block_moved"].weight == -2.0
  assert rw["release"].params["jaw_open"].values == (0.361, 0.337, 0.313)
  assert "block_dropped" in cfg.terminations
  assert cfg.episode_length_s == 15.0
  assert BLOCK in cfg.scene.entities
  rl = kangaroo_pick_place_rl_cfg()
  assert rl.widen_actor_on_resume and rl.widen_critic_on_resume


def test_the_ball_seat_refusals(monkeypatch):
  with pytest.raises(ValueError, match="needs GOAL=ball-seat"):
    _cfg(monkeypatch, GOAL="table")
  with pytest.raises(ValueError, match="ball seat is built for"):
    _cfg(monkeypatch, SHAPE="ball")
  with pytest.raises(ValueError, match="ball seat is built for"):
    _cfg(monkeypatch, TARGET="B")
  with pytest.raises(ValueError, match="ball seat is built for"):
    _cfg(monkeypatch, REWARD=f"{RCTP}+RO")


def test_unset_selectors_resolve_as_before(monkeypatch):
  """Without GOAL=ball-seat nothing of §15 is in the cfg: §12's IN builds as on §12's branch."""
  cfg = _cfg(monkeypatch, SHAPE="pin", GOAL="insert", LAG="FP+JS")
  assert "over_seat" not in cfg.rewards
  assert cfg.rewards["release"].func is not mdp.seat_release
  assert not isinstance(cfg.commands["goal_pose"], mdp.BallSeatCommandCfg)


# -- The reset ----------------------------------------------------------------------------------------


def test_the_reset_puts_the_ball_and_the_block_on_wc_clear_of_each_other(env):
  ball, block = env.scene["object"], env.scene[BLOCK]
  top = surface_top_z(env, SceneEntityCfg("table"))
  table = env.scene["table"]
  tb = table.indexing.body_ids[table.body_names.index("table")]
  shift = env.sim.data.xpos[:, tb, :2] - env.scene.env_origins[:, :2]
  shift = shift - torch.tensor(scene.table_pos(0.837)[:2], device=env.device)
  for ent, half in ((ball, bs.BALL_SEAT_RADIUS), (block, 0.0)):
    xy = ent.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2] - shift
    assert bool(((xy[:, 0] >= WC.x[0] - 1e-6) & (xy[:, 0] <= WC.x[1] + 1e-6)).all())
    assert bool(((xy[:, 1] >= WC.y[0] - 1e-6) & (xy[:, 1] <= WC.y[1] + 1e-6)).all())
    assert torch.allclose(
      ent.data.root_link_pos_w[:, 2] - top, torch.full_like(top, half), atol=2e-4
    )
  d = torch.norm(
    ball.data.root_link_pos_w[:, :2] - block.data.root_link_pos_w[:, :2], dim=-1
  )
  assert bool((d >= bs.BALL_MIN_SEPARATION - 1e-6).all())
  block_xy = block.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
  assert not bool(in_keepout(block_xy, bs.BLOCK_KEEPOUT).any())


def test_nothing_moves_when_nothing_touches_it(env):
  ball, block = env.scene["object"], env.scene[BLOCK]
  p0, b0 = ball.data.root_link_pos_w.clone(), block.data.root_link_pos_w.clone()
  _hold(env, 50)
  assert torch.norm(ball.data.root_link_pos_w - p0, dim=-1).max() < 1e-3
  assert torch.norm(block.data.root_link_pos_w - b0, dim=-1).max() < 5e-4


def test_pb_dr_draws_the_radius(monkeypatch):
  cfg = _cfg(monkeypatch, num_envs=32, SHAPE="ball50-dr")
  e = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  try:
    e.reset()
    r = mdp.object_half_height(e, "object")
    lo, hi = bs.BALL_SEAT_RADIUS_RANGE
    assert bool(((r >= lo - 1e-6) & (r <= hi + 1e-6)).all())
    assert float(r.max() - r.min()) > 0.001
    # The ball rests on the table at its own radius, and its seat follows it.
    top = surface_top_z(e, SceneEntityCfg("table"))
    z = e.scene["object"].data.root_link_pos_w[:, 2] - top
    assert torch.allclose(z, r, atol=2e-4)
    st = mdp.seat_state(e, "object", BLOCK, H, bs.SEAT)
    assert torch.allclose(st.seat_z, bs.SEAT.at(r))
  finally:
    e.close()


# -- Observations ---------------------------------------------------------------------------------------


def test_the_observed_ball_and_block_are_the_truth_with_the_noise_off(env):
  from pal_mjlab.tasks.manipulation.mdp.observations import _to_base_frame

  robot, ball, block = env.scene["robot"], env.scene["object"], env.scene[BLOCK]
  _hold(env, 10)  # past the tracker's lag: nothing has moved
  obs = env.observation_manager.compute_group("actor")
  terms = env.observation_manager.active_terms["actor"]
  dims = env.observation_manager.group_obs_term_dim["actor"]
  start, got = 0, {}
  for name, dim in zip(terms, dims, strict=True):
    got[name] = obs[:, start : start + dim[0]]
    start += dim[0]
  assert start == 42
  assert torch.allclose(
    got["object_pos"], _to_base_frame(robot, ball.data.root_link_pos_w), atol=1e-5
  )
  assert torch.allclose(
    got["target_pos"], _to_base_frame(robot, block.data.root_link_pos_w), atol=1e-5
  )
  assert torch.allclose(got["object_yaw"], torch.tensor([1.0, 0.0]).expand(8, 2))
  assert torch.allclose(got["pin_tilt"], torch.zeros(8, 2))
  assert torch.allclose(
    got["box_yaw"], mdp.box_yaw_b(env, BLOCK), atol=1e-5
  )  # §12's block slot


# -- The seat -------------------------------------------------------------------------------------------


def test_a_ball_dropped_on_the_axis_comes_to_rest_in_the_seat(env):
  """Positive control: let go 2 cm over the seat, 3 mm off the axis (any real let-go is off it: an exact-axis drop rests
  1.5 mm lower, `results/S15-ball-seat/step0/`), at rest after 1.5 s at `SEAT`'s height; `seated_b` and the critic's
  slot say so."""
  ball = env.scene["object"]
  st = _state(env)
  _place(ball, _over_block(env, 0.003, st.seat_z + 0.02))
  _hold(env, 75)
  st = _state(env)
  assert bool(st.seated().all()), (1e3 * st.offset, 1e3 * st.dz)
  assert float(st.dz.abs().max()) < 0.002
  ok = mdp.seated_b(env, "object", BLOCK, H, bs.SEAT, GRASP)
  assert bool(ok.all())
  critic = env.observation_manager.compute_group("critic")
  assert torch.allclose(critic[:, -1], torch.ones(8))


def test_a_ball_dropped_30mm_off_or_on_the_land_is_not_seated(env):
  ball = env.scene["object"]
  st = _state(env)
  _place(ball, _over_block(env, 0.030, st.seat_z + 0.02))
  _hold(env, 75)
  assert not bool(_state(env).seated().any())
  # On the land: the ball's bottom on the block's top, its centre over a corner (the land is widest there).
  st = _state(env)
  corner = 0.5 * (24.0 + scene.BLOCK_SIZE * 1e3 / 2) * 1e-3 * math.sqrt(2)
  block = env.scene[BLOCK]
  off = torch.zeros(env.num_envs, 3, device=env.device)
  off[:, 0] = off[:, 1] = corner / math.sqrt(2)
  off[:, 2] = H + st.radius + 1e-3
  _place(
    ball, block.data.root_link_pos_w + quat_apply(block.data.root_link_quat_w, off)
  )
  env.sim.forward()
  assert not bool(_state(env).seated().any())


def test_the_goal_terms_pay_nothing_before_the_lift_and_nothing_to_a_drag_into_the_side(
  env,
):
  ball = env.scene["object"]
  goal = env.reward_manager.get_term_cfg("goal")
  p = goal.params
  lifted = _was_lifted_bool(env, GRASP, p["support_sensor_name"])
  lifted[:] = False
  st = _state(env)
  # L4: at the drop point, never lifted -> 0.
  _place(ball, _over_block(env, 0.0, st.seat_z + bs.DROP_HEIGHT))
  env.sim.forward()
  assert float(mdp.seat_approach(env, **p).abs().max()) == 0.0
  lifted[:] = True
  assert torch.allclose(mdp.seat_approach(env, **p), torch.ones(8), atol=1e-4)
  # In the seat (below the drop point, on the axis): still the full income.
  _place(ball, _over_block(env, 0.0, st.seat_z))
  env.sim.forward()
  assert torch.allclose(mdp.seat_approach(env, **p), torch.ones(8), atol=1e-4)
  # Beside the block (4 cm off the axis, its bottom level with the block's top): gated to 0.
  _place(ball, _over_block(env, 0.04, H + st.radius))
  env.sim.forward()
  assert float(mdp.seat_approach(env, **p).abs().max()) == 0.0
  # ... and 1 cm higher, clear of the top: paid again.
  _place(ball, _over_block(env, 0.04, H + st.radius + 0.01))
  env.sim.forward()
  assert float(mdp.seat_approach(env, **p).min()) > 0.0
  lifted[:] = False


def test_pushing_the_block_costs(env):
  block = env.scene[BLOCK]
  cmd = env.command_manager.get_term("goal_pose")
  pos = block.data.root_link_pos_w.clone()
  pos[:, :2] = cmd.box_start_xy + torch.tensor([0.02, 0.0], device=env.device)
  quat = block.data.root_link_quat_w.clone()
  block.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1))
  env.sim.forward()
  pen = mdp.box_displacement_penalty(env, "goal_pose", 0.02)
  assert torch.allclose(pen, torch.ones_like(pen), atol=1e-3)
  goal = cmd._goal()
  assert torch.allclose(goal[:, :2], pos[:, :2], atol=1e-4)
  r = mdp.object_half_height(env, "object")
  assert torch.allclose(goal[:, 2] - pos[:, 2], bs.SEAT.at(r), atol=1e-4)
  assert torch.allclose(cmd.rest_centre_z(), goal[:, 2])


# -- L5: the per-step reward rises along the task -----------------------------------------------------


def _g(x, s):
  return math.exp(-((x / s) ** 2))


def _pb_income(stage: str, letgo_mm: float = 0.0, letgo_cm_s: float = 0.0) -> float:
  """PB's task terms at plan §15.2's L5 stages, from the term definitions (`mdp.ball_seat`) and weights (`ball_seat`):
  reach 1 while the hand is at the ball; the ball held still unless said."""
  r = bs.BALL_SEAT_RADIUS
  seat_over_table = bs.SEAT.values[1]

  def goal(dist):
    return bs.APPROACH_WEIGHT * _g(dist, bs.APPROACH_STD) + bs.NEAR_WEIGHT * _g(
      dist, bs.NEAR_STD
    )

  def lift(centre_over_table, to_seat):
    return 2.0 * min(centre_over_table, 0.10) / 0.10 * (1 - _g(to_seat, 0.05))

  if stage == "reach":
    return 1.0
  if stage == "grasped":
    return 2.0 + lift(r, 0.2)
  if stage == "carry":  # 10 cm up, 20 cm from the drop point
    return 2.0 + lift(0.10, 0.2) + goal(0.2)
  if stage == "near":  # 3 cm from the drop point, over the block
    return 2.0 + lift(seat_over_table + 0.03, 0.03) + goal(0.03)
  if stage == "over":  # held still at the drop point
    d = bs.DROP_HEIGHT
    return 2.0 + lift(seat_over_table + d, d) + goal(0.0) + bs.OVER_WEIGHT
  q = 0.25 + 0.75 * _g(letgo_mm * 1e-3, bs.M6_DIST_STD) * _g(
    letgo_cm_s * 1e-2, bs.M6_SPEED_STD
  )
  if (
    stage == "seated"
  ):  # let go, seated, the hand still around the drop point (reach ~1)
    return 1.0 + goal(0.0) + 4 * q + 8 * q
  if stage == "retreated":  # the hand 10 cm over the ball's top
    return _g(0.10 + r, 0.2) + goal(0.0) + 4 * q + 8 * q + 2.0
  raise ValueError(stage)


def test_l5_the_pb_reward_rises_along_the_task():
  stages = ["reach", "grasped", "carry", "near", "over", "seated", "retreated"]
  inc = [_pb_income(s) for s in stages]
  assert all(b > a for a, b in zip(inc, inc[1:], strict=False)), dict(
    zip(stages, inc, strict=True)
  )
  # The let-go out-earns the hold over the seat by >= 3 per step for a let-go 5 mm off at 2 cm/s (§12.1's lesson).
  assert _pb_income("seated", 5.0, 2.0) >= _pb_income("over") + 3.0
  assert _pb_income("retreated", 5.0, 2.0) >= _pb_income("over") + 3.0
