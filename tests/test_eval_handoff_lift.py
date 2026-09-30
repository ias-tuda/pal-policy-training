"""`eval_pick_place.py --handoff-lift` (S7-5 report §8 item 1): the hand-off's position controller
first lifts the TCP straight up over the cube, then goes home.

* the flag parses, off by default, and needs `--handoff`;
* `Kin.ik_pose` onto the current orientation moves the TCP up and keeps the hand's rotation;
* through the real action term (S7-4's plant: G20, CTP, slew 0.5) the lift's targets raise the TCP
  by the height asked for, with the TCP's horizontal drift a fraction of the rise, and then the
  action goes to 0 (home, jaw open) -- the old hand-off;
* an env already high enough goes home at once, and one whose IK misses is counted and goes home.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from pal_mjlab.tasks.manipulation.pick_place import pick_place_env_cfg as pp
from pal_mjlab.tasks.manipulation.pick_place.rewards import S52_BASE
from pal_mjlab.tasks.manipulation.robots import (
  KANGAROO_GRIPPERS_LEFT_ARM_WIDE,
  kangaroo_grippers_left_arm,
)

_SCRIPTS = pathlib.Path(__file__).parent.parent / "scripts"


def _load(name: str):
  spec = importlib.util.spec_from_file_location(f"{name}_lift", _SCRIPTS / f"{name}.py")
  assert spec and spec.loader
  mod = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = mod
  spec.loader.exec_module(mod)
  return mod


ev = _load("eval_pick_place")
scripted = _load("scripted_pick_place")

R = f"{S52_BASE}+M1+M2+M3+M6+J+M8"
CTP = pp.EFFORT_BOUNDS["CTP"]


def test_eval_parses_the_lift_flag():
  assert ev.parse_args(["--checkpoint", "x.pt", "--handoff"]).handoff_lift is None
  args = ev.parse_args(["--checkpoint", "x.pt", "--handoff", "--handoff-lift", "0.1"])
  assert args.handoff_lift == pytest.approx(0.1)


@pytest.fixture(scope="module")
def env():
  robot = kangaroo_grippers_left_arm(
    "G20", KANGAROO_GRIPPERS_LEFT_ARM_WIDE, arm_effort_scale=CTP.effort_scale
  )
  cfg = pp.make_pick_place_env_cfg(
    robot,
    gripper_name="left",
    reward=R,
    action_slew_rate_scale=0.5,
    baseline=pp.BASELINES["B"],
    spawn=pp.SPAWN_BOXES["W2"],
    dr="DR-scene",
    effort=CTP,
  )
  cfg.scene.num_envs = 2
  e = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  yield e
  e.close()


def _tcp(env) -> torch.Tensor:
  return ev.positions(env)[1].clone()


def _lift_from_a_pose(env, lift, height, start_above):
  """Hand the arm off ``start_above`` m over an imagined cube top and step until the lift is done.
  Returns the TCP at the hand-off, at each step, and each step's action."""
  env.reset()
  term = env.action_manager.get_term("joint_pos")
  # Reach a pose by the term itself: the policy's last command, as at a real hand-off.
  hold = torch.zeros(env.num_envs, term.action_dim, device=env.device)
  for _ in range(10):
    env.step(hold)
  top = _tcp(env)[:, 2] - start_above
  handed = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
  hand_step = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
  tcp0 = _tcp(env)
  path, acts = [], []
  for age in range(1, 400):
    above = _tcp(env)[:, 2] - top
    a = lift.act(handed, hand_step, torch.full_like(hand_step, age), above, hold + 1.0)
    acts.append(a.clone())
    env.step(a)
    path.append(_tcp(env))
    if bool(lift.home.all()):
      break
  return tcp0, torch.stack(path), acts


def test_the_lift_goes_straight_up_then_home(env):
  lift = ev.VerticalRetreat(env, 0.08)
  tcp0, path, acts = _lift_from_a_pose(env, lift, 0.08, start_above=0.03)
  assert bool(lift.home.all()), "the lift never handed over to home"
  assert lift.ik_failed == 0 and lift.planned == env.num_envs
  # Up by ~5 cm at the top (peak: the envs go home at their own pace), and no more than a few
  # mm off the vertical on the way up.
  peak = path[..., 2].max(0)
  rise = peak.values - tcp0[:, 2]
  assert rise.min() > 0.035 and rise.max() < 0.07, rise
  for e in range(env.num_envs):
    up = path[: int(peak.indices[e]) + 1, e, :2] - tcp0[e, :2]
    assert up.norm(dim=-1).max() < 0.01, (
      f"env {e} left the vertical by {up.norm(dim=-1).max()} m"
    )
  # While lifting the action is the lift's (not the policy's +1 fill, not 0); at the end it is 0.
  assert not torch.allclose(acts[0], torch.zeros_like(acts[0]))
  assert not torch.allclose(acts[0], torch.ones_like(acts[0]))
  assert torch.allclose(acts[-1], torch.zeros_like(acts[-1]))
  # Each lifting action moves the arm's target by no more than the term's slew (the step to
  # home afterwards is the term's to slew: it is one big jump in action space).
  term = env.action_manager.get_term("joint_pos")
  slew = term._rate[0, lift.arm_idx].cpu()
  scale = term.scale[0, lift.arm_idx].cpu()
  up = [a[0, lift.arm_idx].cpu() * scale for a in acts if a[0].abs().max() > 0]
  assert len(up) > 5
  targets = torch.stack([torch.zeros_like(slew), *up])  # from the default pose it held
  assert ((targets[1:] - targets[:-1]).abs() <= slew + 1e-4).all()


def test_a_cube_already_cleared_goes_straight_home(env):
  lift = ev.VerticalRetreat(env, 0.05)
  _, _, acts = _lift_from_a_pose(env, lift, 0.05, start_above=0.06)
  assert lift.planned == env.num_envs and lift.ik_failed == 0
  assert torch.allclose(acts[0], torch.zeros_like(acts[0]))


def test_an_unreachable_height_is_counted_and_goes_home(env):
  lift = ev.VerticalRetreat(env, 3.0)
  _, _, acts = _lift_from_a_pose(env, lift, 3.0, start_above=0.03)
  assert lift.ik_failed == env.num_envs
  assert torch.allclose(acts[0], torch.zeros_like(acts[0]))


def _kin():
  from scene_model import compile_scene  # on the path once `scripted` has loaded

  return scripted.Kin.from_model(compile_scene(scripted.TASK))


def test_ik_pose_keeps_the_orientation_it_is_given():
  kin = _kin()
  q0 = kin.q_default
  pos, rot = kin.fk(q0)
  q1 = kin.ik_pose(pos + np.array([0.0, 0.0, 0.03]), rot, q0)
  assert q1 is not None
  pos1, rot1 = kin.fk(q1)
  assert pos1[2] - pos[2] == pytest.approx(0.03, abs=2e-3)
  assert np.linalg.norm(pos1[:2] - pos[:2]) < 2e-3
  assert np.abs(rot1 - rot).max() < 0.02


def test_ik_is_ik_pose_with_the_straight_down_hand():
  kin = _kin()
  q0 = kin.q_default
  pos, _ = kin.fk(q0)
  goal, hand_x = pos + np.array([0.02, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
  z = np.array([0.0, 0.0, -1.0])
  r_des = np.column_stack([hand_x, np.cross(z, hand_x), z])
  a, b = kin.ik(goal, hand_x, q0), kin.ik_pose(goal, r_des, q0)
  assert (a is None) == (b is None)
  if a is not None:
    assert np.allclose(a, b)
