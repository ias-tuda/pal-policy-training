"""Plan S7-6 (Siwei, 2026-09-29 18:12): S7-5's setup plus the real robot's two latencies,
ported from Claudius's `origin/faster-moves` -- the FoundationPose tracker's 5-8 control steps
on the cube pose (1d484c4) and the jaw servo's lag behind its command (71e1147). Occlusion
(his dropout model, eb05a76) is not in it.

What is checked here, on S7-5's plant (G20, W2's arm, baseline B, CTP, DR-scene):

* ``LAG`` is read and validated, and unset it builds exactly S7-5's cfg;
* ``FP`` moves the actor's cube terms onto one tracker at 5-8 steps and the critic onto the
  live pose, and the actor then reports the cube exactly its drawn lag late -- position and
  yaw from the same moment -- and never reports a previous episode's cube;
* ``JS`` puts the servo between the jaw command and the joint: the written target moves no
  faster than the drawn speed limit, trails the command during a stroke, the joint follows
  the servo rather than the command, and the command itself is unchanged.
"""

import dataclasses

import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from pal_mjlab.tasks.manipulation import mdp
from pal_mjlab.tasks.manipulation import tabletop_env_cfg as tt
from pal_mjlab.tasks.manipulation.mdp.observations import object_tracker
from pal_mjlab.tasks.manipulation.pick_place import pick_place_env_cfg as pp
from pal_mjlab.tasks.manipulation.pick_place.kangaroo import env_cfg as kenv
from pal_mjlab.tasks.manipulation.pick_place.rewards import S52_BASE
from pal_mjlab.tasks.manipulation.robots import (
  KANGAROO_GRIPPERS_LEFT_ARM_WIDE,
  kangaroo_grippers_left_arm,
)

OZ = f"{S52_BASE}+M1+M2+M3+M6+J+M8+O+Z"
CTP = pp.EFFORT_BOUNDS["CTP"]
S75_ENV = {
  "REWARD": OZ,
  "DR": "DR-scene",
  "SLEW": "0.5",
  "BASELINE": "B",
  "SPAWN": "W2",
  "GAINS": "G20",
  "EFFORT": "CTP",
}


def _cfg(lag: str, num_envs: int | None = None):
  cfg = pp.make_pick_place_env_cfg(
    kangaroo_grippers_left_arm(
      "G20", KANGAROO_GRIPPERS_LEFT_ARM_WIDE, arm_effort_scale=CTP.effort_scale
    ),
    gripper_name="left",
    reward=OZ,
    action_slew_rate_scale=0.5,
    baseline=pp.BASELINES["B"],
    spawn=pp.SPAWN_BOXES["W2"],
    dr="DR-scene",
    effort=CTP,
    latency=tt.LATENCY_MODELS[lag],
  )
  if num_envs is not None:
    cfg.scene.num_envs = num_envs
  return cfg


# -- Configuration --------------------------------------------------------------------------


def test_the_measured_numbers():
  assert tt.VISION_TRACKER_LAG == (5, 8)
  assert tt.CONTROL_DT == pytest.approx(0.02)
  # 5-8 steps is 100-160 ms: the 100-150 ms Claudius measured, rounded up at the top.
  assert tt.VISION_TRACKER_LAG[0] * tt.CONTROL_DT == pytest.approx(0.10)
  assert tt.JAW_SERVO_TAU == (0.11, 0.19)
  assert tt.JAW_SERVO_MAX_VEL == (1.25, 1.75)


def test_lag_is_read_and_validated(monkeypatch):
  for var, value in S75_ENV.items():
    monkeypatch.setenv(var, value)
  monkeypatch.delenv("LAG", raising=False)
  unset = kenv.kangaroo_pick_place_env_cfg()
  monkeypatch.setenv("LAG", "none")
  none = kenv.kangaroo_pick_place_env_cfg()
  assert unset.observations == none.observations
  assert unset.actions == none.actions
  monkeypatch.setenv("LAG", "FP+JS")
  both = kenv.kangaroo_pick_place_env_cfg()
  assert both.observations["actor"].terms["object_pos"].params["lag"] == (5, 8)
  assert both.actions["joint_pos"].servo_joints is not None
  monkeypatch.setenv("LAG", "FP+GS")
  with pytest.raises(ValueError, match="LAG"):
    kenv.kangaroo_pick_place_env_cfg()


def test_none_is_s75():
  cfg = _cfg("none")
  for name in ("object_pos", "object_yaw"):
    actor = cfg.observations["actor"].terms[name]
    assert "lag" not in actor.params
    assert (actor.delay_min_lag, actor.delay_max_lag) == (0, 2), "S7-5's DR-obs lag"
    critic = cfg.observations["critic"].terms[name]
    assert (critic.delay_min_lag, critic.delay_max_lag) == (0, 0)
  action = cfg.actions["joint_pos"]
  assert (action.servo_joints, action.servo_tau, action.servo_max_vel) == (None,) * 3


@pytest.mark.parametrize("lag", ["FP", "FP+JS"])
def test_fp_moves_the_cube_terms_onto_the_tracker(lag):
  cfg, base = _cfg(lag), _cfg("none")
  for name in ("object_pos", "object_yaw"):
    actor = cfg.observations["actor"].terms[name]
    assert actor.params["lag"] == (5, 8)
    assert (actor.delay_min_lag, actor.delay_max_lag) == (0, 0), (
      "the tracker's lag only"
    )
    assert actor.noise == base.observations["actor"].terms[name].noise, "noise kept"
    critic = cfg.observations["critic"].terms[name]
    assert critic.params["lag"] is None
    assert (critic.delay_min_lag, critic.delay_max_lag) == (0, 0)
  # Nothing else in the observations changes: same terms, same order, same other settings.
  for group in ("actor", "critic"):
    got, want = cfg.observations[group].terms, base.observations[group].terms
    assert list(got) == list(want)
    for name in want:
      if name not in ("object_pos", "object_yaw"):
        assert got[name] == want[name], (group, name)


@pytest.mark.parametrize("lag", ["JS", "FP+JS"])
def test_js_puts_the_servo_on_the_jaw_only(lag):
  action = _cfg(lag).actions["joint_pos"]
  assert action.servo_joints == (r"gripper_left_finger_joint",)
  assert action.servo_tau == tt.JAW_SERVO_TAU
  assert action.servo_max_vel == tt.JAW_SERVO_MAX_VEL
  base = _cfg("none").actions["joint_pos"]
  assert (
    dataclasses.replace(action, servo_joints=None, servo_tau=None, servo_max_vel=None)
    == base
  )


def test_servo_cfg_is_one_model():
  action = _cfg("none").actions["joint_pos"]
  with pytest.raises(ValueError, match="together"):
    dataclasses.replace(action, servo_joints=("gripper_left_finger_joint",))
  with pytest.raises(ValueError, match="servo_tau"):
    dataclasses.replace(
      action,
      servo_joints=("gripper_left_finger_joint",),
      servo_tau=(0.0, 0.1),
      servo_max_vel=(1.0, 2.0),
    )


# -- The tracker ----------------------------------------------------------------------------

N_ENVS = 8
STEPS = 24


def test_the_actor_sees_the_cube_its_lag_late_and_never_a_previous_episodes():
  cfg = _cfg("FP", num_envs=N_ENVS)
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  try:
    env.reset()
    obj = env.scene[pp.OBJECT]
    zero = torch.zeros(env.action_space.shape, device=env.device)
    start = obj.data.root_link_pose_w.clone()
    live_pos, live_yaw, seen_pos, seen_yaw = [], [], [], []
    for t in range(STEPS):
      # A cube that moves every step, 10 cm above the table so the arm never meets it.
      pose = start.clone()
      pose[:, 0] += 0.004 * t
      pose[:, 2] += 0.10
      obj.write_root_link_pose_to_sim(pose)
      env.step(zero)
      tracker = object_tracker(env, pp.OBJECT)
      assert tracker is not None, "the actor's terms read through the tracker"
      assert tracker.lag_range == (5, 8)
      live_pos.append(mdp.object_position_b(env, pp.OBJECT))
      live_yaw.append(mdp.object_yaw_b(env, pp.OBJECT))
      seen_pos.append(mdp.object_position_b(env, pp.OBJECT, lag=(5, 8)))
      seen_yaw.append(mdp.object_yaw_b(env, pp.OBJECT, lag=(5, 8)))
    assert not bool(env.reset_buf.any()), "no env may reset inside the window"
    lag = tracker.lag
    assert bool(((lag >= 5) & (lag <= 8)).all()), lag
    for t in range(8, STEPS):
      for e in range(N_ENVS):
        k = t - int(lag[e])
        assert torch.allclose(seen_pos[t][e], live_pos[k][e], atol=1e-6), (t, e)
        # Yaw from the same moment as the position: one tracker, not two term buffers.
        assert torch.allclose(seen_yaw[t][e], live_yaw[k][e], atol=1e-6), (t, e)
    # The cube moved over the window, so a lagged reading is not trivially the live one.
    assert (live_pos[-1][:, 0] - seen_pos[-1][:, 0]).abs().min() > 0.01

    # A reset refills the history: after it the tracker reports the new episode's cube (as it
    # was at the reset, lag >= 5 steps ago), never the last episode's hovering one.
    old = mdp.object_position_b(env, pp.OBJECT)
    env.reset()
    at_reset = mdp.object_position_b(env, pp.OBJECT)
    for _ in range(3):
      env.step(zero)
      seen = mdp.object_position_b(env, pp.OBJECT, lag=(5, 8))
      assert torch.allclose(seen, at_reset, atol=1e-6)
      assert bool(((seen - old).norm(dim=-1) > 0.01).all())
  finally:
    env.close()


# -- The jaw servo --------------------------------------------------------------------------


def _jaw_column(term) -> int:
  names = [n.split("/")[-1] for n in term._target_names]
  return names.index("gripper_left_finger_joint")


#: The stroke: the jaw commanded one way for `CLOSE` steps, reversed for `OPEN`, then held.
CLOSE, OPEN, HOLD = 25, 25, 50


def test_the_jaw_follows_the_servo_not_the_command():
  env = ManagerBasedRlEnv(cfg=_cfg("JS", num_envs=4), device=get_test_device())
  try:
    env.reset()
    term = env.action_manager.get_term("joint_pos")
    col = _jaw_column(term)
    tau, vmax = term._servo_tau[:, 0], term._servo_max_vel[:, 0]
    assert bool(((tau >= 0.11) & (tau <= 0.19)).all()), tau
    assert bool(((vmax >= 1.25) & (vmax <= 1.75)).all()), vmax
    robot = env.scene["robot"]
    jaw_id = term._target_ids[col]
    # The jaw resets at its open stop, so +1 is the direction that moves it.
    close = torch.zeros(env.action_space.shape, device=env.device)
    close[:, col] = 1.0
    cmd, servo, q = [], [], []
    for t in range(CLOSE + OPEN + HOLD):
      env.step(close if t < CLOSE else -close)
      cmd.append(term._processed_actions[:, col].clone())
      servo.append(term.servo_target()[:, col].clone())
      q.append(robot.data.joint_pos[:, jaw_id].clone())
    cmd, servo, q = torch.stack(cmd), torch.stack(servo), torch.stack(q)
    # The command ramps at B's jaw slew (0.04 rad/step), as in S7-5: the servo does not touch it.
    assert (cmd[1:] - cmd[:-1]).abs().max() <= 0.04 + 1e-6
    stroke = cmd[CLOSE - 1] - cmd[0]
    assert bool((stroke > 0.2).all()), stroke
    # The servo never outruns its drawn speed limit over a control step.
    assert bool(((servo[1:] - servo[:-1]).abs() <= vmax * tt.CONTROL_DT + 1e-5).all())
    # It trails the command during the closing stroke...
    gap = cmd[5:CLOSE] - servo[5:CLOSE]
    assert bool((gap > 0.02).all()), gap
    # ...keeps closing for a moment after the command reverses, as the real jaw does...
    assert bool((servo[CLOSE] > servo[CLOSE - 1]).all())
    # ...never gets as far as the command did...
    assert bool((servo.max(dim=0).values < cmd.max(dim=0).values - 0.02).all())
    # ...and settles on the command once it stops.
    assert torch.allclose(servo[-1], cmd[-1], atol=5e-3)
    # The joint tracks the servo, not the command, through the stroke.
    mid = slice(5, CLOSE + 10)
    assert (
      (q[mid] - servo[mid]).abs().mean() < (q[mid] - cmd[mid]).abs().mean()
    ).item()
    # A reset restarts the servo at the command's reset value.
    env.reset()
    assert torch.allclose(term._servo[:, 0], term._last_target[:, col])
  finally:
    env.close()


def test_without_js_the_written_target_is_the_command():
  env = ManagerBasedRlEnv(cfg=_cfg("none", num_envs=2), device=get_test_device())
  try:
    env.reset()
    term = env.action_manager.get_term("joint_pos")
    env.step(torch.ones(env.action_space.shape, device=env.device))
    assert term.servo_target() is term._processed_actions
  finally:
    env.close()
