"""Plan S7-11 (Siwei, 2026-10-05: the deployed WC-c s1 @ 5500 pushes the cube with a half-open jaw):
``PREGRASP``.

* unset / ``none`` builds exactly the cfg every run before S7-11 built;
* each code adds what it says and nothing else -- no observation or command changes, and the
  action keeps its width, which is what lets WC-c s1 @ 5500 resume under it;
* the grasp window is in the TCP frame, and holds the cube at WC-c's own grasps;
* NP: a cube slid 3 cm before any grasp costs the full penalty, a reset clears it;
* JO: a closed jaw away from the cube costs its angle;
* G / OVR: the jaw command is capped while the observed cube is outside the window, free inside.

The scripted check that G lets a real grasp through is `scripts/scripted_pick_place.py rollout
--pregrasp G` (plan S7-11 step 0, `../results/S7-11/step0/`).
"""

import math

import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply
from pal_mjlab.tasks.manipulation.mdp import pregrasp as pg
from pal_mjlab.tasks.manipulation.pick_place import pregrasp as pre
from pal_mjlab.tasks.manipulation.pick_place.kangaroo import env_cfg as kenv
from pal_mjlab.tasks.manipulation.pick_place.rewards import S52_BASE
from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS_LEFT_ARM

#: WC-c (DEP1, jobs 161293-4), as `submit_dep1.sh` exported it.
WC_C_ENV = {
  "REWARD": f"{S52_BASE}+M1+M2+M3+M6+J+M8",
  "DR": "DR-scene",
  "PPO": "PPO-stack",
  "SLEW": "stack",
  "BASELINE": "B",
  "TOOL_SPEED": "none",
  "GAINS": "G20",
  "EFFORT": "CTP",
  "LAG": "FP+JS",
  "ACTION": "abs",
  "GATE": "none",
  "CRITIC_OBS": "q",
  "ARM_TABLE": "AH10",
  "CMD_OBS": "cmd",
  "SPAWN": "WC",
  "TARGET": "WC",
  "EPISODE_S": "15",
}
TCP = SceneEntityCfg("robot", site_names=("tcp_left",))
GRASP_SENSOR = KANGAROO_GRIPPERS_LEFT_ARM.gripper("left").contact_sensor_name
#: Cube in the TCP frame at the first two-pad contact, cm, in WC-c s1 @ 5500's 12 MuJoCo reference
#: episodes (`results/sim2sim/dep1/models/pp_dep1_WC-c_s1_it5500/mujoco_reference.npz`).
WC_C_ONSETS_CM = [
  (-0.2, 0.5, 4.4),
  (-0.0, -0.6, 6.6),
  (0.5, -0.7, 4.5),
  (-0.3, 0.7, 5.0),
  (0.5, -0.4, 5.8),
  (0.5, 1.3, 3.9),
  (-0.7, 0.6, 6.0),
  (-0.4, -0.1, 6.0),
  (-0.2, 0.3, 4.4),
  (-0.6, 1.3, 4.9),
  (0.6, 0.2, 4.9),
  (0.1, 0.6, 4.4),
]


def _cfg(monkeypatch, code: str | None, num_envs: int | None = None):
  for var, value in {**WC_C_ENV, "PREGRASP": code}.items():
    if value is None:
      monkeypatch.delenv(var, raising=False)
    else:
      monkeypatch.setenv(var, value)
  cfg = kenv.kangaroo_pick_place_env_cfg()
  if num_envs is not None:
    cfg.scene.num_envs = num_envs
    cfg.seed = 0
  return cfg


# -- Configuration --------------------------------------------------------------------------


def test_unset_and_none_build_the_wc_c_cfg(monkeypatch):
  base = _cfg(monkeypatch, None)
  none = _cfg(monkeypatch, "none")
  for field in (
    "actions",
    "observations",
    "commands",
    "events",
    "terminations",
    "rewards",
  ):
    assert getattr(base, field) == getattr(none, field), field
  assert "reset_pregrasp" not in base.events
  assert not [n for n in base.rewards if n.startswith("pregrasp")]


@pytest.mark.parametrize("code", ["watch", "NP", "JO", "G", "S", "OVR"])
def test_each_code_adds_only_its_terms(monkeypatch, code):
  base = _cfg(monkeypatch, None)
  cfg = _cfg(monkeypatch, code)
  spec = pre.PREGRASP_CODES[code]
  for field in ("observations", "commands", "terminations", "curriculum"):
    assert getattr(cfg, field) == getattr(base, field), field
  assert set(cfg.events) - set(base.events) == {"reset_pregrasp"}
  for name in base.events:
    assert cfg.events[name] == base.events[name], name
  want = set()
  if spec.push_weight:
    want.add("pregrasp_push")
  if spec.open_weight:
    want.add("pregrasp_jaw")
  if spec.align_weight:
    want.add("pregrasp_align")
  assert set(cfg.rewards) - set(base.rewards) == want
  for name in base.rewards:
    assert cfg.rewards[name] == base.rewards[name], name
  old, new = base.actions["joint_pos"], cfg.actions["joint_pos"]
  if spec.gate_cap is None:
    assert new == old
  else:
    assert isinstance(new, pg.PregraspGateJointPositionActionCfg)
    assert new.cap == spec.gate_cap
    # Everything the plain action had, unchanged: the ONNX contract and the resume hold.
    for f in ("scale", "offset", "clip", "rate_limit", "servo_joints", "servo_tau"):
      assert getattr(new, f) == getattr(old, f), f


def test_the_codes_weights():
  c = pre.PREGRASP_CODES
  assert (c["NP"].push_weight, pre.NP_DEADBAND, pre.NP_SPAN) == (-2.0, 0.008, 0.02)
  assert c["JO"].open_weight == -1.0
  assert c["G"].gate_cap == 0.03 and c["OVR"].gate_cap == 0.0
  s = c["S"]
  assert (s.push_weight, s.open_weight, s.gate_cap) == (-2.0, -1.0, None)
  assert s.align_weight > 0 and pre.YG_YAW_STD == pytest.approx(math.radians(15))


def test_a_bad_code_raises(monkeypatch):
  with pytest.raises(ValueError, match="PREGRASP"):
    _cfg(monkeypatch, "NX")


def test_the_gate_and_the_stack_lock_do_not_mix(monkeypatch):
  stack = {
    "TARGET": "B",
    "SPAWN": "W2",
    "EPISODE_S": "10",
    "GOAL": "stack",
    "GATE": "stack",
  }
  for var, value in {**WC_C_ENV, **stack, "PREGRASP": "G"}.items():
    monkeypatch.setenv(var, value)
  with pytest.raises(ValueError, match="GATE=stack"):
    kenv.kangaroo_pick_place_env_cfg()


def test_fold_of_the_relative_yaw():
  """`grasp_relative_yaw` folds mod 90 deg into [0, 45]."""
  from pal_mjlab.tasks.manipulation.mdp.stack import fold_quarter

  deg = torch.tensor([0.0, 10.0, 44.0, 46.0, 90.0, 100.0, -80.0, 180.0])
  got = torch.rad2deg(fold_quarter(torch.deg2rad(deg)))
  assert torch.allclose(
    got, torch.tensor([0.0, 10.0, 44.0, 44.0, 0.0, 10.0, 10.0, 0.0]), atol=1e-4
  )


# -- In an env ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def s_env():
  mp = pytest.MonkeyPatch()
  env = ManagerBasedRlEnv(cfg=_cfg(mp, "S", num_envs=2), device=get_test_device())
  env.reset()
  TCP.resolve(env.scene)
  yield env
  env.close()
  mp.undo()


def _tcp_point(env, offset):
  robot = env.scene["robot"]
  pos = robot.data.site_pos_w[:, TCP.site_ids].squeeze(1)
  quat = robot.data.site_quat_w[:, TCP.site_ids].squeeze(1)
  off = torch.tensor(offset, device=env.device).expand_as(pos)
  return pos + quat_apply(quat, off)


def test_the_window_holds_wc_c_grasps(s_env):
  for x, y, z in WC_C_ONSETS_CM:
    p = _tcp_point(s_env, (x / 100, y / 100, z / 100))
    assert pg.in_grasp_window(s_env, p, TCP).all(), (x, y, z)
    assert torch.allclose(
      pg.cube_in_tcp(s_env, p, TCP), torch.tensor([x, y, z]) / 100, atol=1e-5
    )
  # The scripted grasp's (step 0): the cube at z ~0.5 cm.
  assert pg.in_grasp_window(s_env, _tcp_point(s_env, (0.0, 0.0, 0.005)), TCP).all()
  for off in (
    (0.0, 0.0, 0.10),
    (0.0, 0.0, -0.01),
    (0.03, 0.0, 0.05),
    (0.0, 0.035, 0.05),
  ):
    assert not pg.in_grasp_window(s_env, _tcp_point(s_env, off), TCP).any(), off


def _zero(env):
  return torch.zeros(
    env.num_envs, env.action_manager.total_action_dim, device=env.device
  )


def _move_cube(env, dxy):
  """Slide the cube on the table by ``dxy`` (world pose in, as the reset events write it)."""
  obj = env.scene["object"]
  pose = torch.cat(
    [obj.data.root_link_pos_w, obj.data.root_link_quat_w], dim=-1
  ).clone()
  pose[:, :2] += torch.tensor(dxy, device=env.device)
  obj.write_root_link_pose_to_sim(pose)
  obj.write_root_link_velocity_to_sim(torch.zeros(env.num_envs, 6, device=env.device))


def test_np_prices_a_slid_cube_and_a_reset_clears_it(s_env):
  env = s_env
  env.reset()
  for _ in range(3):
    env.step(_zero(env))
  st = pg.pregrasp_state(env, "object", GRASP_SENSOR)
  assert (st.push < 0.002).all(), st.push
  push = env.reward_manager.get_term_cfg("pregrasp_push")
  assert (push.func(env, **push.params) == 0).all()
  _move_cube(env, (0.03, 0.0))
  env.sim.forward()
  env.step(_zero(env))
  got = push.func(env, **push.params)
  assert (st.push > 0.025).all(), st.push
  assert (got == 1.0).all(), got
  env.reset()
  env.step(_zero(env))
  assert (push.func(env, **push.params) == 0).all()


def test_jo_prices_a_closed_jaw_away_from_the_cube(s_env):
  env = s_env
  env.reset()
  jaw = env.reward_manager.get_term_cfg("pregrasp_jaw")
  a = _zero(env)
  a[:, -1] = 1.0  # close
  for _ in range(40):
    env.step(a)
  val = jaw.func(env, **jaw.params)
  q = env.scene["robot"].data.joint_pos[
    :, env.scene["robot"].joint_names.index("gripper_left_finger_joint")
  ]
  assert (q > 0.3).all()
  assert torch.allclose(val, q)


@pytest.mark.parametrize("code,cap", [("G", 0.03), ("OVR", 0.0)])
def test_the_gate_caps_outside_the_window_and_not_inside(monkeypatch, code, cap):
  env = ManagerBasedRlEnv(
    cfg=_cfg(monkeypatch, code, num_envs=2), device=get_test_device()
  )
  try:
    env.reset()
    TCP.resolve(env.scene)
    term = env.action_manager.get_term("joint_pos")
    a = _zero(env)
    a[:, -1] = 1.0
    for _ in range(5):
      env.step(a)
    assert term.armed.all() and term.capped.all()
    assert torch.allclose(term.commanded_target[:, -1], torch.full((2,), cap))
    # The observed cube put into the window: the jaw is free to close.
    monkeypatch.setattr(
      pg, "observed_object_pos_w", lambda e, name: _tcp_point(e, (0.0, 0.0, 0.05))
    )
    for _ in range(5):
      env.step(a)
    assert not term.armed.any() and not term.capped.any()
    assert (term.commanded_target[:, -1] > 0.1).all()
  finally:
    env.close()
