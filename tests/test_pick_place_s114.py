"""Plan S11.4 (`pick-place-s11-bimanual-plan.md`, *S11.4*, step 0 item 2): JOINT on the high table -- HOLD_MIX,
KL_FLOOR, HH_TERM and RIGHT_GATE=clear, each default-off."""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from pal_mjlab.tasks.manipulation import bimanual_policy as bp
from pal_mjlab.tasks.manipulation import mdp
from pal_mjlab.tasks.manipulation.mdp import handoff as ho
from pal_mjlab.tasks.manipulation.mdp import ladder
from pal_mjlab.tasks.manipulation.pick_place import bimanual as bi
from pal_mjlab.tasks.manipulation.pick_place.kangaroo.env_cfg import (
  kangaroo_pick_place_env_cfg,
  kangaroo_pick_place_rl_cfg,
)
from pal_mjlab.tasks.manipulation.robots import ARM_MIRROR_SIGNS
from test_pick_place_s112 import _bank_from, _place_a

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

#: BS114: S11.2's JOINT baseline (BS11, S11-BI, the left's training hand-off) on the 0.837 m table.
BS114 = {
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
  "TABLE_DZ": "22",
}
SEQ = {"INIT": "seq", "INIT_LT": "lt.pt", "INIT_RST": "rst.pt"}
_ALL = {
  *BS114,
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
  "HOLD_MIX",
  "HH_TERM",
  "RIGHT_GATE",
  "KL_FLOOR",
}


def _setenv(monkeypatch, **changes) -> None:
  for var in _ALL:
    monkeypatch.delenv(var, raising=False)
  for var, value in {**BS114, **changes}.items():
    if value is not None:
      monkeypatch.setenv(var, value)


def _cfg(monkeypatch, num_envs: int = 2, **changes):
  _setenv(monkeypatch, **changes)
  cfg = kangaroo_pick_place_env_cfg()
  cfg.scene.num_envs = num_envs
  cfg.seed = 0
  return cfg


def _r(cfg) -> str:
  """A cfg's repr without object addresses (each build makes new spec lambdas)."""
  return re.sub(r" at 0x[0-9a-f]+", "", repr(cfg))


def _env(cfg) -> ManagerBasedRlEnv:
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  env.reset()
  return env


def _step(env, steps: int, action=None) -> None:
  with torch.inference_mode():
    for _ in range(steps):
      env.step(
        torch.zeros(env.action_space.shape, device=env.device)
        if action is None
        else action
      )


@pytest.fixture
def bank(monkeypatch, tmp_path) -> str:
  """A tiny hold bank made of a JOINT env's own state at 0.837 m."""
  src = _env(_cfg(monkeypatch, num_envs=2))
  b = _bank_from(src, 10)
  src.close()
  path = tmp_path / "bank.pt"
  torch.save(b, path)
  return str(path)


# -- Unset is S11.2 -------------------------------------------------------------------------------


def test_unset_s114_selectors_leave_the_s112_cfgs(monkeypatch, bank):
  # BI-H: the hold mix spelled as today's default is the same cfg as no HOLD_MIX at all.
  plain = _cfg(monkeypatch, START="hold", HOLD_BANK=bank)
  same = _cfg(monkeypatch, START="hold", HOLD_BANK=bank, HOLD_MIX="30/20/50")
  assert _r(plain) == _r(same)
  p = plain.events["start_hold"].params
  assert (p["hold_share"], p["placed_share"]) == (bi.HOLD_SHARE, bi.PLACED_SHARE)
  assert bi.ARM_ARM_TERM not in plain.terminations
  # BI-F: no left_home anywhere, the right's reward gated on A alone, the KL to 0.
  cfg = _cfg(monkeypatch, **SEQ)
  assert "left_home" not in cfg.observations["critic"].terms
  assert "left_home" not in cfg.observations["actor"].terms["right_target"].params
  assert "gate_left_home" not in cfg.rewards["right_goal"].params
  assert _r(cfg) == _r(_cfg(monkeypatch, RIGHT_GATE="latch", **SEQ))
  rl = kangaroo_pick_place_rl_cfg()
  assert rl.algorithm.kl_floor == 0.0 and rl.algorithm.kl_coef == bp.SEQ_KL_COEF


def test_s114_selectors_need_their_partners(monkeypatch, bank):
  for changes, match in (
    ({"HOLD_MIX": "10/20/70"}, "HOLD_MIX"),  # START=scratch
    ({"START": "hold", "HOLD_BANK": bank, "HOLD_MIX": "10/20/60"}, "HOLD_MIX"),
    ({"START": "hold", "HOLD_BANK": bank, "HOLD_MIX": "0.1/0.2/0.7"}, "HOLD_MIX"),
    ({"RIGHT_GATE": "clear"}, "INIT=seq"),
    ({"RIGHT_GATE": "open", **SEQ}, "RIGHT_GATE"),
    ({"HH_TERM": "0"}, "HH_TERM"),
    ({"HH_TERM": "thirty"}, "HH_TERM"),
  ):
    _setenv(monkeypatch, **changes)
    with pytest.raises(ValueError, match=match):
      kangaroo_pick_place_env_cfg()
  for changes in ({"KL_FLOOR": "0.2"}, {"KL_FLOOR": "1.5", **SEQ}):
    _setenv(monkeypatch, **changes)
    with pytest.raises(ValueError, match="KL_FLOOR"):
      kangaroo_pick_place_rl_cfg()
  # The one-arm env refuses them.
  _setenv(monkeypatch, ARMS="left", HH_TERM="30")
  with pytest.raises(ValueError, match="JOINT"):
    kangaroo_pick_place_env_cfg()


# -- HOLD_MIX ---------------------------------------------------------------------------------------


def test_hold_mix_shares_hold_within_2_percent_over_10k_resets(monkeypatch, bank):
  cfg = _cfg(
    monkeypatch, num_envs=50, START="hold", HOLD_BANK=bank, HOLD_MIX="10/20/70"
  )
  p = cfg.events["start_hold"].params
  assert (p["hold_share"], p["placed_share"]) == (0.1, 0.2)
  env = _env(cfg)
  ids = torch.arange(env.num_envs, device=env.device)
  counts = torch.zeros(3)
  with torch.inference_mode():
    for _ in range(200):  # 10 000 draws
      bi.hold_start(env, ids, **env.cfg.events["start_hold"].params)
      counts += torch.bincount(env._pp_start_kind.cpu(), minlength=3).float()
  share = counts / counts.sum()
  k = bi.START_KINDS
  assert abs(float(share[k["hold"]]) - 0.1) < 0.02, share
  assert abs(float(share[k["placed"]]) - 0.2) < 0.02, share
  assert abs(float(share[k["scratch"]]) - 0.7) < 0.02, share


# -- KL_FLOOR ---------------------------------------------------------------------------------------


def test_the_kl_falls_to_its_floor_and_never_below(monkeypatch):
  """BI-F2: weight 1 until update 250, linear to 0.2 at 1000, then 0.2 for good."""
  ppo = bp.JointPPO.__new__(bp.JointPPO)
  ppo.teacher = object()
  ppo.kl_coef, ppo.kl_hold_iterations, ppo.kl_anneal_iterations = 1.0, 250, 1000
  ppo.kl_floor = 0.2
  for u, w in ((0, 1.0), (249, 1.0), (250, 1.0), (625, 0.6), (1000, 0.2), (4999, 0.2)):
    ppo.updates_done = u
    assert ppo.kl_weight() == pytest.approx(w)
  ws = []
  for u in range(0, 5000, 7):
    ppo.updates_done = u
    ws.append(ppo.kl_weight())
  assert min(ws) == pytest.approx(0.2) and all(
    a >= b for a, b in zip(ws, ws[1:], strict=False)
  )
  _setenv(monkeypatch, KL_FLOOR="0.2", **SEQ)
  rl = kangaroo_pick_place_rl_cfg()
  assert rl.algorithm.kl_floor == 0.2 and rl.algorithm.class_name == bp.PPO_CLASS


# -- HH_TERM ----------------------------------------------------------------------------------------


def _mirrored_arm_targets(env, alpha: float) -> torch.Tensor:
  """Both arms' joint positions a share ``alpha`` of the way from home towards a mirrored pose whose hands
  overlap at the midline (the arms' capsules meet near ``alpha`` = 0.7). (B, n_joints)"""
  robot = env.scene["robot"]
  names = robot.joint_names
  left = [names.index(f"arm_left_{i}_joint") for i in range(1, 8)]
  right = [names.index(f"arm_right_{i}_joint") for i in range(1, 8)]
  q = robot.data.default_joint_pos.clone()
  far = torch.tensor([-0.408, 2.357, 0.57, 0.902], device=env.device)
  ql = q[:, left].clone()
  ql[:, :4] = (1.0 - alpha) * ql[:, :4] + alpha * far
  q[:, left] = ql
  q[:, right] = torch.tensor(ARM_MIRROR_SIGNS, device=env.device) * ql
  return q


def _action_for(env, q: torch.Tensor) -> torch.Tensor:
  """The absolute action whose targets are ``q``'s arm joints (`ACTION=abs`: target = offset + scale x a)."""
  term = env.action_manager.get_term("joint_pos")
  names = env.scene["robot"].joint_names
  cols = [names.index(n.split("/")[-1]) for n in term.target_names]
  return (q[:, cols] - term._offset) / term._scale


def test_hh_term_positive_control_two_arms_driven_together(monkeypatch):
  """Two arms driven into each other end the episode on ``arm_arm_force`` -- a termination, not a time-out --
  and the contact the sensor saw is over the threshold. Without HH_TERM the same drive does not end it."""
  for hh_term in ("30", None):
    cfg = _cfg(monkeypatch, num_envs=1, HH_TERM=hh_term)
    cfg.auto_reset = False  # keep the terminal state
    assert (bi.ARM_ARM_TERM in cfg.terminations) == (hh_term is not None)
    env = _env(cfg)
    try:
      robot = env.scene["robot"]
      with torch.inference_mode():
        q = _mirrored_arm_targets(env, 0.55)
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        env.sim.forward()
        assert bool((mdp.arm_arm_clearance(env) > 0.03).all())
        act = _action_for(env, _mirrored_arm_targets(env, 1.0))
        ended_at, peak = -1, 0.0
        for t in range(60):
          _, _, terminated, truncated, _ = env.step(act)
          f = torch.norm(env.scene[bi.ARM_ARM_SENSOR].data.force_history, dim=-1)
          peak = max(peak, float(f.max()))
          assert not bool(truncated.any())
          if bool(terminated.any()):
            ended_at = t
            break
      assert peak > 30.0, peak
      if hh_term is None:
        assert ended_at < 0, ended_at
      else:
        assert ended_at >= 0
        assert bool(env.termination_manager.get_term(bi.ARM_ARM_TERM).all())
        assert not bool(env.termination_manager.time_outs.any())
    finally:
      env.close()


def test_the_eval_drops_hh_term(monkeypatch):
  import eval_bimanual
  from mjlab.tasks import registry

  cfg = _cfg(monkeypatch, HH_TERM="30")
  assert bi.ARM_ARM_TERM in cfg.terminations
  monkeypatch.setattr(registry, "load_env_cfg", lambda task, play: cfg)
  args = argparse.Namespace(
    task="Mjlab-PickPlace-Pal-Kangaroo-Grippers",
    episode_s=25.0,
    num_envs=2,
    seed=0,
    obs_noise="on",
    device=get_test_device(),
  )
  env = eval_bimanual.build_env(args)
  try:
    assert bi.ARM_ARM_TERM not in env.cfg.terminations
    assert bi.ARM_ARM_TERM not in env.termination_manager.active_terms
  finally:
    env.close()


# -- RIGHT_GATE=clear -------------------------------------------------------------------------------


def test_right_gate_clear_waits_for_the_left_home_leg_and_never_opens_before_a(
  monkeypatch,
):
  cfg = _cfg(monkeypatch, RIGHT_GATE="clear", **SEQ)
  assert cfg.observations["actor"].terms["right_target"].params["left_home"] is True
  assert "left_home" in cfg.observations["critic"].terms
  for name in bi.GATED_RIGHT:
    assert cfg.rewards[f"right_{name}"].params["gate_left_home"] is True
  env = _env(cfg)
  term = env.action_manager.get_term("joint_pos")
  k = term.arm_index("left")
  p = cfg.observations["actor"].terms["right_target"].params
  plain = {k2: v for k2, v in p.items() if k2 != "left_home"}

  def target(left_home: bool) -> torch.Tensor:
    return ladder.right_target_gated(env, **plain, left_home=left_home)

  def wait() -> torch.Tensor:
    # A not latched: both forms give the wait point.
    return target(False)

  with torch.inference_mode():
    # Before A: the left home or not, the wait point.
    term.phase[:, k] = ho.HOME
    assert not bool(
      getattr(env, ladder._A_LATCHED, torch.zeros(1, dtype=torch.bool)).any()
    )
    w = wait()
    assert torch.allclose(target(True), w)
    term.phase[:, k] = ho.POLICY
  _place_a(env)
  _step(env, 4)
  with torch.inference_mode():
    assert bool(getattr(env, ladder._A_LATCHED).all())
    stack = target(False)
    assert float((stack - w).norm(dim=-1).min()) > 0.01  # S11.2's gate has opened
    for phase in (ho.POLICY, ho.UP, ho.OUT):
      term.phase[:, k] = phase
      assert torch.allclose(target(True), w, atol=1e-5), phase
      assert float(mdp.handoff_home(env, "left").max()) == 0.0
    term.phase[:, k] = ho.HOME
    assert torch.allclose(target(True), stack, atol=1e-5)
    assert float(mdp.handoff_home(env, "left").min()) == 1.0


def test_the_right_kl_mask_follows_the_clear_gate():
  ppo = bp.JointPPO.__new__(bp.JointPPO)
  ppo.kl_layout = {
    "left_cols": [0],
    "right_cols": [1],
    "a_latched": [0, 1],
    "left_handed_off": [1, 2],
    "left_home": [2, 3],
  }
  mu_t, std, mu_s = (
    torch.zeros(4, 2),
    torch.ones(4, 2),
    torch.ones(4, 2),
  )  # KL 0.5 a dimension
  # latched, handed off, home: only the last row opens the right's KL.
  critic = torch.tensor(
    [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
  )
  assert ppo.kl_to_teacher(critic, mu_t, std, mu_s, std).tolist() == [
    0.5,
    0.5,
    0.5,
    1.0,
  ]
