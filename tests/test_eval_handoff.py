"""The eval's position controller after the hand-off (`eval_pick_place.RiseReturn`).

S7-5's 5 k read (2026-09-29): of 45 placed-but-failed episodes, 40 had the cube moved by the
return to home -- action 0 at once, a joint-space move that sweeps the fingers back down through
the goal region, because the home TCP sits at about the cube's height. The controller now
opens, rises straight up, crosses at that height to above home and comes straight down onto it.

Checked here, on S7-5's RCTP plant (G20, W2, B, CTP, DR-scene), without a policy:

* the path: the rise is vertical with the hand's orientation kept, the crossing is level, the
  path never dips below the lower of its two ends, and it ends on the home pose;
* no step of the path asks the arm for more than the action term's slew;
* closed loop: an env handed over with the jaw open rises by about `HANDOFF_RISE`, never goes
  back below its start height near the cube, and gets home; a closed jaw is opened first;
* ``--handoff-direct`` keeps the old return (action 0 at once).
"""

import importlib.util
import pathlib

import mujoco
import numpy as np
import pytest
import torch
from conftest import get_test_device

_SPEC = importlib.util.spec_from_file_location(
  "eval_pick_place", pathlib.Path(__file__).parent.parent / "scripts/eval_pick_place.py"
)
assert _SPEC and _SPEC.loader
ev = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ev)

RCTP_ENV = {
  "REWARD": "R-C-strict+M1+M2+M3+M6+J+M8",
  "DR": "DR-scene",
  "PPO": "PPO-stack",
  "SLEW": "0.5",
  "BASELINE": "B",
  "SPAWN": "W2",
  "GAINS": "G20",
  "EFFORT": "CTP",
  "TOOL_SPEED": "none",
}


@pytest.fixture(scope="module")
def env():
  mp = pytest.MonkeyPatch()
  for k, v in RCTP_ENV.items():
    mp.setenv(k, v)
  args = ev.parse_args(
    ["--checkpoint", "x", "--num-envs", "2", "--device", get_test_device(), "--handoff"]
  )
  e = ev.build_env(args)
  e.reset()
  yield e
  mp.undo()


def _start_pose(vr, i: int) -> np.ndarray:
  """An arm pose with the TCP over the table ~12 cm from home and ~4 cm lower -- where the
  policy lets go -- found by the controller's own IK from the home pose."""
  vr.data.qpos[:] = vr.env.sim.data.qpos[i].detach().cpu().numpy()
  q_home = vr.offset[i, vr.arm_idx].cpu().numpy().astype(np.float64)
  ph, rh = vr._fk(q_home)
  goal = ph + np.array([0.04, -0.11, -0.04])
  jacp, jacr = np.zeros((3, vr.model.nv)), np.zeros((3, vr.model.nv))
  q = q_home.copy()
  for _ in range(200):
    pos, _ = vr._fk(q)
    mujoco.mj_comPos(vr.model, vr.data)
    mujoco.mj_jacSite(vr.model, vr.data, jacp, jacr, vr.tcp)
    j = jacp[:, vr.dofs]
    q = np.clip(
      q + j.T @ np.linalg.solve(j @ j.T + 1e-4 * np.eye(3), goal - pos), vr.lo, vr.hi
    )
  assert np.linalg.norm(vr._fk(q)[0] - goal) < 1e-3
  return q


def test_the_path_rises_straight_crosses_level_and_ends_home(env):
  vr = ev.RiseReturn(env)
  q0 = _start_pose(vr, 0)
  path, n_rise, ok = vr._solve(0, q0)
  assert ok
  poses = [vr._fk(q) for q in path]
  pos = np.stack([p for p, _ in poses])
  p0, r0 = vr._fk(q0)
  ph, _ = vr._fk(vr.offset[0, vr.arm_idx].cpu().numpy().astype(np.float64))
  rise = pos[:n_rise]
  # Vertical, by HANDOFF_RISE, the hand's orientation kept.
  assert np.abs(rise[:, :2] - p0[:2]).max() < 2e-3
  assert rise[-1, 2] - p0[2] == pytest.approx(ev.HANDOFF_RISE, abs=2e-3)
  for _, rot in poses[:n_rise]:
    assert np.linalg.norm(rot - r0) < 0.03
  # Level while crossing, then down onto home.
  top = p0[2] + ev.HANDOFF_RISE
  cross = pos[n_rise:][np.linalg.norm(pos[n_rise:, :2] - ph[:2], axis=1) > 2e-3]
  assert np.abs(cross[:, 2] - top).max() < 3e-3
  assert pos[:, 2].min() > min(p0[2], ph[2]) - 2e-3
  assert np.linalg.norm(pos[-1] - ph) < 2e-3
  # The path's rise leg takes HANDOFF_RISE / HANDOFF_RISE_SPEED.
  assert n_rise == round(ev.HANDOFF_RISE / (ev.HANDOFF_RISE_SPEED * env.step_dt))


def test_no_step_of_the_path_outruns_the_slew(env):
  vr = ev.RiseReturn(env)
  q0 = _start_pose(vr, 0)
  path, _, _ = vr._solve(0, q0)
  rate = vr.term._rate[0, vr.arm_idx].cpu().numpy()
  steps = np.abs(np.diff(np.vstack([q0, path]), axis=0))
  assert (steps <= rate + 1e-6).all()


def _run(env, vr, handed, steps):
  robot = env.scene[ev.ROBOT]
  jaw_id = robot.joint_names.index(ev.JAW_JOINT)
  tcp_z, phases = [], []
  for _ in range(steps):
    _, tcp = ev.positions(env)
    jaw_q = robot.data.joint_pos[:, jaw_id]
    act = vr.action(handed, jaw_q, tcp[:, 2])
    env.step(act)
    tcp_z.append(ev.positions(env)[1][:, 2].clone())
    phases.append(vr.phase.clone())
  return torch.stack(tcp_z), torch.stack(phases)


def test_closed_loop_it_rises_and_gets_home(env):
  env.reset()
  vr = ev.RiseReturn(env)
  # Put the arm's command (and, by stepping to it, the arm) at the let-go pose, jaw open.
  q0 = torch.as_tensor(_start_pose(vr, 0), dtype=torch.float32)
  full = vr.offset.clone()
  full[:, vr.arm_idx] = q0
  raw = ((full - vr.offset) / vr.scale).clamp(-1, 1)
  raw[:, vr.jaw_idx] = 0.0
  for _ in range(150):
    env.step(raw)
  z0 = ev.positions(env)[1][:, 2].clone()
  handed = torch.tensor([True, False], device=env.device)
  z, phases = _run(env, vr, handed, 400)
  assert int(phases[0, 0]) == vr.MOVE  # the jaw is already open: no wait
  assert float(z[:, 0].max() - z0[0]) > 0.8 * ev.HANDOFF_RISE
  assert float(z[:, 0].min() - z0[0]) > -0.01
  assert int(phases[-1, 0]) == vr.HOME
  robot = env.scene[ev.ROBOT]
  arm = [robot.joint_names.index(f"arm_left_{i}_joint") for i in range(1, 8)]
  err = (robot.data.joint_pos[0, arm] - robot.data.default_joint_pos[0, arm]).abs()
  assert float(err.max()) < ev.HOME_TOL
  # The env that was not handed over is untouched by the controller's bookkeeping.
  assert int(vr.phase[1]) == vr.OPEN and not bool(vr.active[1])
  summary = vr.summarise(0)
  assert summary["handoff_rise_median_m"] is None  # nothing banked yet
  vr.finish(torch.tensor([0], device=env.device))
  row = vr.done[-1]
  assert row["rise"] == pytest.approx(ev.HANDOFF_RISE, abs=0.03)
  assert row["path_ik_ok"] is True
  assert int(vr.phase[0]) == vr.OPEN and not bool(vr.active[0])


def test_a_closed_jaw_is_opened_before_the_arm_moves(env):
  env.reset()
  vr = ev.RiseReturn(env)
  robot = env.scene[ev.ROBOT]
  jaw_id = robot.joint_names.index(ev.JAW_JOINT)
  closed = vr.offset.clone()
  raw = ((closed - vr.offset) / vr.scale).clamp(-1, 1)
  raw[:, vr.jaw_idx] = 1.0  # close
  for _ in range(40):
    env.step(raw)
  assert float(robot.data.joint_pos[0, jaw_id]) > ev.JAW_OPEN
  handed = torch.tensor([True, True], device=env.device)
  jaw = robot.data.joint_pos[:, jaw_id]
  act = vr.action(handed, jaw, ev.positions(env)[1][:, 2])
  assert int(vr.phase[0]) == vr.OPEN
  assert float(act[0, vr.jaw_idx]) == 0.0  # the jaw commanded open
  # The arm is held at its last command meanwhile.
  target = vr.term._last_target[0, vr.arm_idx]
  held = vr.offset[0, vr.arm_idx] + act[0, vr.arm_idx] * vr.scale[0, vr.arm_idx]
  assert torch.allclose(held, target, atol=1e-5)


def test_direct_keeps_the_old_return():
  args = ev.parse_args(["--checkpoint", "x", "--handoff", "--handoff-direct"])
  assert args.handoff and args.handoff_direct
  args = ev.parse_args(["--checkpoint", "x", "--handoff"])
  assert not args.handoff_direct
