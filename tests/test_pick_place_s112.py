"""Plan S11.2 (`pick-place-s11-bimanual-plan.md`, *S11.2*, step 0 item 4): the S12-BI ladders, the A latch,
the right's training hand-off, hold starts and BI-F's factored SEQ-initialised policy."""

from __future__ import annotations

import pathlib
import sys

import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from pal_mjlab.tasks.manipulation import bimanual_policy as bp
from pal_mjlab.tasks.manipulation import mdp
from pal_mjlab.tasks.manipulation.mdp import handoff as ho
from pal_mjlab.tasks.manipulation.mdp import ladder
from pal_mjlab.tasks.manipulation.mdp.observations import _WAS_LIFTED, latch_attr
from pal_mjlab.tasks.manipulation.pick_place import bimanual as bi
from pal_mjlab.tasks.manipulation.pick_place.kangaroo.env_cfg import (
  kangaroo_pick_place_env_cfg,
  kangaroo_pick_place_rl_cfg,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

BS11_JOINT = {
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
  "EPISODE_S": "15",
  "CRITIC_OBS": "q",
  "ARM_TABLE": "AH10",
  "ARM_HOME": "mirror",
  "TRACKER": "split",
  "TARGET": "T11",
  "ARMS": "both",
  "REWARD": "S11-BI",
  "HANDOFF_TRAIN": "left",
  "START": "scratch",
}
CODES = {
  "BI-J": {},
  "BI-S": {"REWARD": "S12-BI", "HANDOFF_TRAIN": "both"},
}
_ALL = {
  *BS11_JOINT,
  "GOAL",
  "GATE",
  "HANDOFF_ROUTE",
  "SHAPE",
  "OCC",
  "OBJ_OBS",
  "CMD_OBS",
  "HOLD_BANK",
  "INIT",
  "INIT_LT",
  "INIT_RST",
}


def _setenv(monkeypatch, code: str, **changes) -> None:
  for var in _ALL:
    monkeypatch.delenv(var, raising=False)
  for var, value in {**BS11_JOINT, **CODES[code], **changes}.items():
    if value is not None:
      monkeypatch.setenv(var, value)


def _cfg(monkeypatch, code: str, num_envs: int = 4, **changes):
  _setenv(monkeypatch, code, **changes)
  cfg = kangaroo_pick_place_env_cfg()
  cfg.scene.num_envs = num_envs
  cfg.seed = 0
  return cfg


def _env(monkeypatch, code: str, num_envs: int = 4, cfg=None, **changes):
  if cfg is None:
    cfg = _cfg(monkeypatch, code, num_envs, **changes)
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  env.reset()
  return env


def _step(env, steps: int) -> None:
  with torch.inference_mode():
    for _ in range(steps):
      env.step(torch.zeros(env.action_space.shape, device=env.device))


def _place_a(env, offset=(0.0, 0.0)) -> None:
  """A resting at T (+ ``offset``), the left's lift latched (the release latches on the next steps)."""
  a = env.scene["object"]
  goal = env.command_manager.get_term("goal_pose").command
  pose = a.data.root_link_pose_w.clone()
  pose[:, :3] = goal
  pose[:, 0] += offset[0]
  pose[:, 1] += offset[1]
  pose[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
  a.write_root_link_pose_to_sim(pose)
  a.write_root_link_velocity_to_sim(torch.zeros_like(a.data.root_link_vel_w))
  getattr(env, latch_attr(_WAS_LIFTED, "left_pad_contact"))[:] = True


def _gate(env) -> dict:
  return env.cfg.rewards["a_displaced"].params["gate"]


# -- L5 over the ladders --------------------------------------------------------------------


def test_every_level_of_each_ladder_beats_every_lower_one():
  """The plan's ordering, fetch < wait < carry < stacked < released < success (and the left's): the most a
  level's shaping can add stays below the next floor."""
  right_max_shaping = (0.9, 0.5, ladder.SHAPE, ladder.SHAPE, ladder.SHAPE, 0.0)
  for k in range(5):
    assert ladder.RIGHT_FLOORS[k] + right_max_shaping[k] < ladder.RIGHT_FLOORS[k + 1]
  left_max_shaping = (0.9, ladder.SHAPE, ladder.SHAPE, ladder.SHAPE)
  for k in range(3):
    assert ladder.LEFT_FLOORS[k] + left_max_shaping[k] < ladder.LEFT_FLOORS[k + 1]
  # The plan's hover ceiling 2.45 < stacked floor 4 < released floor 6 < success 8.
  assert ladder.RIGHT_FLOORS[2] + ladder.SHAPE == pytest.approx(2.45)
  assert ladder.RIGHT_FLOORS[3:] == (4.0, 6.0, 8.0)
  # Decision 1: each stacking gap (2 floors x the scale) >= 2 x the ledger's p99 right-side penalty (5.6); w_A ~ one gap.
  assert 2.0 * ladder.LADDER_SCALE >= 2 * 5.6
  assert -ladder.A_MOVED_WEIGHT == pytest.approx(2.0 * ladder.LADDER_SCALE)


def test_s12_bi_replaces_the_task_terms_and_keeps_the_rest(monkeypatch):
  s11 = _cfg(monkeypatch, "BI-J")
  s12 = _cfg(monkeypatch, "BI-S")
  dropped = set(s11.rewards) - set(s12.rewards)
  assert dropped == {
    *bi.LADDER_REPLACES,
    *(f"right_{n}" for n in bi.LADDER_REPLACES),
    "right_stack_align",
  }
  assert set(s12.rewards) - set(s11.rewards) == {
    "left_ladder",
    "right_ladder",
    "a_displaced",
  }
  assert (
    s12.rewards["arm_arm_clearance"].func is mdp.arm_arm_clearance_before_left_handoff
  )
  assert s12.rewards["left_ladder"].weight == ladder.LADDER_SCALE
  assert list(s12.events)[0] == "reset_ladder_state"
  assert "a_latched" in s12.observations["critic"].terms
  # The right hands off on its own trigger in training, or at decision 5's 14 s safety net.
  arms = {a.name: a for a in s12.actions["joint_pos"].arms}
  assert arms["right"].trigger is not None
  assert arms["right"].trigger.timeout_s == bi.RIGHT_HANDOFF_TIMEOUT_S == 14.0
  assert arms["right"].trigger.object_name == bi.CUBE_B


def test_unset_s112_selectors_leave_bi_j_and_its_ppo_alone(monkeypatch):
  cfg = _cfg(monkeypatch, "BI-J")
  assert "left_ladder" not in cfg.rewards and "placed" in cfg.rewards
  assert "a_latched" not in cfg.observations["critic"].terms
  assert "right_target" not in cfg.observations["actor"].terms
  assert not cfg.metrics
  rl = kangaroo_pick_place_rl_cfg()
  assert rl.algorithm.class_name == "PPO" and rl.actor.class_name == "MLPModel"
  _setenv(monkeypatch, "BI-S")
  assert kangaroo_pick_place_rl_cfg().algorithm.class_name == bp.PPO_CLASS


def test_s112_selectors_need_arms_both_and_their_partners(monkeypatch):
  _setenv(monkeypatch, "BI-J", START="hold")
  with pytest.raises(ValueError, match="HOLD_BANK"):
    kangaroo_pick_place_env_cfg()
  _setenv(monkeypatch, "BI-J", INIT="seq")
  with pytest.raises(ValueError, match="INIT_LT"):
    kangaroo_pick_place_rl_cfg()
  _setenv(monkeypatch, "BI-J", ARMS="left", INIT="seq")
  with pytest.raises(ValueError, match="JOINT"):
    kangaroo_pick_place_env_cfg()


# -- The A latch ------------------------------------------------------------------------------


def test_the_a_latch_survives_a_5_mm_nudge_and_a_push_costs_once(monkeypatch):
  env = _env(monkeypatch, "BI-S", num_envs=2)
  _place_a(env)
  _step(env, 4)
  latched, _ = ladder.update_a_latch(env, _gate(env))
  assert bool(latched.all())
  a_ref = getattr(env, ladder._A_REF).clone()
  # A 5 mm nudge: still latched, no cost.
  _place_a(env, offset=(0.005, 0.0))
  costs = []
  for _ in range(4):
    _step(env, 1)
    costs.append(ladder.a_displaced(env, _gate(env)).clone())
  assert bool(getattr(env, ladder._A_LATCHED).all())
  assert float(torch.stack(costs).sum()) == 0.0
  # Pushed 1.5 cm off its reference: exactly one event per env, and the latch stays.
  pose = env.scene["object"].data.root_link_pose_w.clone()
  pose[:, 0] = a_ref[:, 0] + 0.015
  env.scene["object"].write_root_link_pose_to_sim(pose)
  idx = list(env.reward_manager.active_terms).index("a_displaced")
  hits = torch.zeros(env.num_envs, device=env.device)
  for _ in range(6):
    _step(env, 1)
    hits += (env.reward_manager._step_reward[:, idx] != 0).float()
  assert hits.tolist() == [1.0, 1.0]
  assert bool(getattr(env, ladder._A_LATCHED).all())


def test_a_reset_clears_the_a_latch_for_bi_f(monkeypatch):
  """BI-F (S11-BI + INIT=seq) reads the latch through its gated right target; a reset must clear it, as
  S12-BI's does (batch 2's BI-F trained with it set from an env's first latch on)."""
  assert "reset_ladder_state" not in _cfg(monkeypatch, "BI-J").events
  cfg = _cfg(monkeypatch, "BI-J", num_envs=2, INIT="seq", INIT_LT="lt.pt", INIT_RST="rst.pt")
  assert "right_target" in cfg.observations["actor"].terms
  assert list(cfg.events)[0] == "reset_ladder_state"
  env = _env(monkeypatch, "BI-J", cfg=cfg)
  _place_a(env)
  _step(env, 4)
  assert bool(getattr(env, ladder._A_LATCHED).all())
  with torch.inference_mode():
    env.reset()
  assert not bool(getattr(env, ladder._A_LATCHED).any())


def test_the_left_ladder_freezes_at_its_hand_off(monkeypatch):
  env = _env(monkeypatch, "BI-S", num_envs=2, HANDOFF_TRAIN="none")
  _place_a(env)
  _step(env, 4)
  term = env.action_manager.get_term("joint_pos")
  idx = list(env.reward_manager.active_terms).index("left_ladder")
  term.start_handoff(
    "left", torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
  )
  _step(env, 1)
  frozen = env.reward_manager._step_reward[:, idx].clone()
  # A knocked off T: the left's income does not move.
  _place_a(env, offset=(0.06, 0.0))
  _step(env, 3)
  assert torch.allclose(env.reward_manager._step_reward[:, idx], frozen)
  assert bool(getattr(env, ladder._LEFT_FROZEN).all())


def test_hh_clearance_stops_at_the_left_hand_off(monkeypatch):
  env = _env(monkeypatch, "BI-S", num_envs=2, HANDOFF_TRAIN="none")
  term = env.action_manager.get_term("joint_pos")
  margin = 10.0  # every clearance is inside it: the penalty is > 0 while the left is the policy's
  assert bool((mdp.arm_arm_clearance_before_left_handoff(env, margin) > 0).all())
  term.start_handoff(
    "left", torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
  )
  assert bool((mdp.arm_arm_clearance_before_left_handoff(env, margin) == 0).all())


# -- Hold starts ------------------------------------------------------------------------------


def _bank_from(env, steps: int) -> dict:
  """A snapshot bank made of this env's own state after ``steps`` random steps."""
  with torch.inference_mode():
    for _ in range(steps):
      env.step(0.3 * torch.randn(env.action_space.shape, device=env.device))
  robot = env.scene["robot"]
  term = env.action_manager.get_term("joint_pos")
  a, b = env.scene["object"], env.scene[bi.CUBE_B]
  o = env.scene.env_origins
  goal = env.command_manager.get_term("goal_pose").command
  servo = term._servo if term._servo_idx is not None else torch.zeros(env.num_envs, 0)
  return {
    "q": robot.data.joint_pos.clone().cpu(),
    "qd": robot.data.joint_vel.clone().cpu(),
    "a_pos": (a.data.root_link_pos_w - o).clone().cpu(),
    "a_quat": a.data.root_link_quat_w.clone().cpu(),
    "b_pos": (b.data.root_link_pos_w - o).clone().cpu(),
    "b_quat": b.data.root_link_quat_w.clone().cpu(),
    "b_vel": b.data.root_link_vel_w.clone().cpu(),
    "target": term._last_target.clone().cpu(),
    "servo": servo.clone().cpu(),
    "step": torch.full((env.num_envs,), 123, dtype=torch.long),
    "goal_offset": (goal[:, :2] - o[:, :2]).clone().cpu(),
    "table_z": torch.zeros(env.num_envs),
    "joint_names": list(robot.joint_names),
    "target_names": [n.split("/")[-1] for n in term.target_names],
  }


def test_a_hold_start_reproduces_the_snapshot(monkeypatch, tmp_path):
  src = _env(monkeypatch, "BI-S", num_envs=1)
  bank = _bank_from(src, 20)
  src.close()
  path = tmp_path / "bank.pt"
  torch.save(bank, path)
  cfg = _cfg(monkeypatch, "BI-S", num_envs=2, START="hold", HOLD_BANK=str(path))
  cfg.events["start_hold"].params["hold_share"] = 1.0
  assert list(cfg.events)[-1] == "start_hold"
  env = _env(monkeypatch, "BI-S", cfg=cfg)
  robot = env.scene["robot"]
  term = env.action_manager.get_term("joint_pos")
  dev = env.device
  q = bank["q"].to(dev).expand(2, -1)
  assert float((robot.data.joint_pos - q).abs().max()) < 1e-6
  assert float((term._last_target - bank["target"].to(dev)).abs().max()) < 1e-6
  o = env.scene.env_origins
  b = env.scene[bi.CUBE_B].data.root_link_pos_w - o
  assert float((b - bank["b_pos"].to(dev)).abs().max()) < 1e-6
  a = env.scene["object"].data.root_link_pos_w - o
  assert float((a[:, :2] - bank["a_pos"][:, :2].to(dev)).abs().max()) < 1e-6
  assert bool((term.phase[:, term.arm_index("left")] == ho.HOME).all())
  # The clock and T come back on the first step.
  _step(env, 1)
  assert env.episode_length_buf.tolist() == [124, 124]
  goal = env.command_manager.get_term("goal_pose").command[:, :2] - o[:, :2]
  assert float((goal - bank["goal_offset"].to(dev)).abs().max()) < 1e-5
  assert bool((env._pp_start_kind == bi.START_KINDS["hold"]).all())


# -- BI-F -------------------------------------------------------------------------------------


def _fake_one_arm_checkpoint(path: pathlib.Path, seed: int) -> None:
  g = torch.Generator().manual_seed(seed)
  dims = [38, 512, 256, 128, 8]
  sd = {
    "obs_normalizer._mean": torch.randn(1, 38, generator=g) * 0.1,
    "obs_normalizer._var": torch.rand(1, 38, generator=g) + 0.5,
    "obs_normalizer.count": torch.tensor(12345),
    "distribution.log_std_param": torch.log(torch.rand(8, generator=g) * 0.3 + 0.05),
  }
  sd["obs_normalizer._std"] = sd["obs_normalizer._var"].sqrt()
  for i in range(4):
    sd[f"mlp.{2 * i}.weight"] = torch.randn(dims[i + 1], dims[i], generator=g) * 0.05
    sd[f"mlp.{2 * i}.bias"] = torch.randn(dims[i + 1], generator=g) * 0.05
  torch.save({"actor_state_dict": sd}, path)


def test_bi_f_at_iteration_0_is_the_seq_pair(monkeypatch, tmp_path):
  """The factored actor computes L-T on the left and R-ST mirrored on the right (`eval_chain`'s adapter, with
  the gated goal) to 1e-5, the other hand's TCP not mattering yet."""
  from eval_chain import MirrorAdapter, OneArmActor

  lt, rst = tmp_path / "lt.pt", tmp_path / "rst.pt"
  _fake_one_arm_checkpoint(lt, 1)
  _fake_one_arm_checkpoint(rst, 2)
  env = _env(
    monkeypatch, "BI-S", num_envs=4, INIT="seq", INIT_LT=str(lt), INIT_RST=str(rst)
  )
  _place_a(env)
  _step(env, 5)
  rl = kangaroo_pick_place_rl_cfg()
  assert (
    rl.actor.class_name == bp.MODEL_CLASS and rl.algorithm.kl_coef == bp.SEQ_KL_COEF
  )
  layout = bp.layout_of(env)
  obs = env.observation_manager.compute()
  model = bp.FactoredArmModel(
    obs,
    {"actor": ["actor"], "critic": ["critic"]},
    "actor",
    16,
    hidden_dims=(512, 256, 128),
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 0.35,
      "std_type": "log",
      "std_range": (1e-3, 0.5),
    },
    init_lt=str(lt),
    init_rst=str(rst),
    layout=layout,
  )
  ad = MirrorAdapter(env)
  t = ad.terms(obs, "actor")
  # eval_chain's right goal is A's tracked top; give it the gated one the factored actor reads.
  t2 = dict(t)
  t2["goal_base"] = t["right_target"].clone()
  t2["goal_base"][:, 2] -= 0.05
  # Both sides on the CPU in fp32: a GPU may run the matmuls in TF32 (~1e-4 here).
  with torch.no_grad():
    want_l = OneArmActor(lt, "cpu")(ad.left(t).cpu())
    want_r = (
      OneArmActor(rst, "cpu")(ad.right(t2, base_term="goal_base").cpu()) * ad.s8.cpu()
    )
    got = model.mean_of(obs["actor"].cpu())
  assert float((got[:, layout["left_cols"]] - want_l).abs().max()) < 1e-5
  assert float((got[:, layout["right_cols"]] - want_r).abs().max()) < 1e-5
  # The std starts at the sources'.
  src = torch.load(rst, weights_only=False)["actor_state_dict"][
    "distribution.log_std_param"
  ]
  assert torch.allclose(
    model.distribution.log_std_param[layout["right_cols"]].cpu(), src
  )


def test_the_mirror_quaternion_is_the_frame_map():
  """`bimanual_policy.mirror_quat` = R -> M R F M (`eval_chain.MirrorAdapter._mirror_quat`), sign w >= 0."""
  from mjlab.utils.lab_api.math import matrix_from_quat, quat_from_matrix

  q = torch.nn.functional.normalize(torch.randn(64, 4), dim=-1)
  m = torch.diag(torch.tensor([1.0, -1.0, 1.0]))
  f = torch.diag(torch.tensor([-1.0, -1.0, 1.0]))
  want = quat_from_matrix(m @ matrix_from_quat(q) @ f @ m)
  want = torch.where(want[:, :1] < 0, -want, want)
  assert torch.allclose(bp.mirror_quat(q), want, atol=1e-5)


def test_a_timed_out_right_hand_off_freezes_the_right_ladder(monkeypatch):
  """Decision 5: the right's 14 s time-out is a safety net -- the jaw it opens pays no release, leave or success."""
  env = _env(monkeypatch, "BI-S", num_envs=2)
  term = env.action_manager.get_term("joint_pos")
  k = term.arm_index("right")
  idx = list(env.reward_manager.active_terms).index("right_ladder")
  with (
    torch.inference_mode()
  ):  # the env's buffers are inference tensors once it has stepped
    env.episode_length_buf[:] = round(bi.RIGHT_HANDOFF_TIMEOUT_S / env.step_dt)
    _step(env, 1)
    assert bool(term.by_timeout[:, k].all()) and bool((term.phase[:, k] > 0).all())
    frozen = env.reward_manager._step_reward[:, idx].clone()
    # Whatever happens to B now, the right's income stays where the time-out found it.
    b = env.scene[bi.CUBE_B]
    pose = b.data.root_link_pose_w.clone()
    pose[:, :3] = env.scene["object"].data.root_link_pos_w + torch.tensor(
      [0.0, 0.0, 0.05], device=env.device
    )
    b.write_root_link_pose_to_sim(pose)
    _step(env, 3)
    assert torch.allclose(env.reward_manager._step_reward[:, idx], frozen)
    # A reset clears the record.
    env.reset()
  assert not bool(term.by_timeout.any())


def test_the_kl_holds_then_anneals_and_the_right_waits_for_the_latch():
  """Decision 3: weight 1 to update 250, linear to 0 at 1000; the right's KL only where A is latched and the left
  handed off."""
  ppo = bp.JointPPO.__new__(bp.JointPPO)
  ppo.teacher = object()
  ppo.kl_coef, ppo.kl_hold_iterations, ppo.kl_anneal_iterations = 1.0, 250, 1000
  for u, w in ((0, 1.0), (249, 1.0), (250, 1.0), (625, 0.5), (1000, 0.0), (2000, 0.0)):
    ppo.updates_done = u
    assert ppo.kl_weight() == pytest.approx(w)
  ppo.kl_layout = {
    "left_cols": [0],
    "right_cols": [1],
    "a_latched": [0, 1],
    "left_handed_off": [1, 2],
  }
  mu_t = torch.zeros(4, 2)
  std = torch.ones(4, 2)
  mu_s = torch.ones(4, 2)  # each dimension's KL = 0.5
  critic = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
  kl = ppo.kl_to_teacher(critic, mu_t, std, mu_s, std)
  assert kl.tolist() == [0.5, 0.5, 0.5, 1.0]


def test_every_s112_code_logs_the_tower(monkeypatch):
  for code, changes in (("BI-S", {}), ("BI-J", {"HANDOFF_TRAIN": "both"})):
    cfg = _cfg(monkeypatch, code, **changes)
    assert cfg.metrics["tower_standing"].reduce == "last"
