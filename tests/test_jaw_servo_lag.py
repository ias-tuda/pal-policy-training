"""The jaw servo model, against the strokes measured on hardware.

Recording ``/pal_policy/desired_state`` against the measured jaw gave a peak speed of
1.3-1.7 rad/s and a steady 0.13-0.17 s of lag behind the command. Without the model the
sim jaw is a massless finger behind a stiff PD and reaches its command inside one control
step, so reversing the jaw costs a policy nothing -- which is how a deployed run came to
reverse it four times in 1.1 s without ever completing a stroke.
"""

import pal_mjlab.tasks  # noqa: F401
import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from pal_mjlab.tasks.manipulation.mdp import BinaryJawAction
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import (
  CONTROL_DT,
  GRIPPER_ACTION,
  JAW_SERVO_MAX_VEL,
  JAW_SERVO_TAU,
)

TASK = "Mjlab-Grasp-UpperBody-Standing-ArmV36-Pal-Kangaroo-Grippers"

#: The model is sampled per env; pinning it makes one episode's response the assertion.
TAU = 0.15
MAX_VEL = 1.5

#: Long enough for the lagged jaw to settle against the stop the empty fingers meet at.
STROKE_STEPS = 60


def _closing_stroke(lag: bool) -> tuple[torch.Tensor, torch.Tensor]:
  """Jaw command and measured jaw position per control step, holding "close". (T,) each."""
  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = 1
  jaw_cfg = cfg.actions[GRIPPER_ACTION]
  jaw_cfg.servo_tau = (TAU, TAU) if lag else None
  jaw_cfg.servo_max_vel = (MAX_VEL, MAX_VEL) if lag else None
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    env.reset()
    jaw: BinaryJawAction = env.action_manager.get_term(GRIPPER_ACTION)
    robot = env.scene[jaw.cfg.entity_name]
    # Zero on the arms holds their integrated targets where the reset left them, so the
    # only thing moving is the jaw.
    action = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
    action[:, -jaw.action_dim :] = 1.0
    command, measured = [], []
    for _ in range(STROKE_STEPS):
      env.step(action)
      command.append(jaw.target[0, 0].item())
      measured.append(robot.data.joint_pos[0, jaw.target_ids[0]].item())
    return torch.tensor(command), torch.tensor(measured)
  finally:
    env.close()


def _time_to(measured: torch.Tensor, level: float) -> float:
  """When the jaw first passes ``level``, seconds from the start of the stroke."""
  reached = (measured > level).nonzero()
  assert reached.numel() > 0, f"the jaw never reached {level}"
  return int(reached[0]) * CONTROL_DT


@pytest.fixture(scope="module")
def lagged() -> tuple[torch.Tensor, torch.Tensor]:
  return _closing_stroke(lag=True)


@pytest.fixture(scope="module")
def unlagged() -> tuple[torch.Tensor, torch.Tensor]:
  return _closing_stroke(lag=False)


def test_jaw_never_exceeds_the_measured_hardware_speed(lagged):
  _, measured = lagged
  speed = (measured.diff() / CONTROL_DT).abs()
  # The pinned ceiling, with room for the PD's own overshoot on the servo target.
  assert speed.max() < 1.1 * MAX_VEL
  # ...and it does get up to speed: a jaw that never moved would pass the bound above.
  assert speed.max() > 0.8 * MAX_VEL


def test_jaw_lags_its_command_by_the_measured_time_constant(lagged):
  command, measured = lagged
  # Only while the command is still moving: once it stops, the lag decays instead.
  moving = command.diff() > 1e-6
  # A first-order lag under a ramp settles ``tau`` behind its input, so the position
  # error divided by the ramp speed is the time constant.
  lag = ((command[:-1] - measured[:-1])[moving] / MAX_VEL).abs()
  assert lag.numel() > 0
  assert TAU * 0.5 < lag.max() < TAU * 1.5


def test_the_lag_is_what_makes_a_stroke_cost_most_of_a_second(lagged, unlagged):
  _, with_lag = lagged
  _, without = unlagged
  # The empty fingers meet well short of the commanded stop, so the stroke is measured
  # against where the jaw actually ends up, not against what was asked for.
  level = without[0] + 0.9 * (without[-1] - without[0])
  slow, fast = _time_to(with_lag, level), _time_to(without, level)
  # Without the model a full stroke is over in about half a second...
  assert fast < 0.55
  # ...and the model costs it roughly a time constant more, which is the whole point:
  # a jaw reversed mid-stroke now never gets anywhere.
  assert TAU * 0.5 < slow - fast < TAU * 3.0


def test_the_model_defaults_to_the_measured_ranges():
  jaw_cfg = load_env_cfg(TASK).actions[GRIPPER_ACTION]
  assert jaw_cfg.servo_tau == JAW_SERVO_TAU
  assert jaw_cfg.servo_max_vel == JAW_SERVO_MAX_VEL


def test_the_model_is_all_or_nothing():
  jaw_cfg = load_env_cfg(TASK).actions[GRIPPER_ACTION]
  with pytest.raises(ValueError, match="must be given together"):
    type(jaw_cfg)(**{**vars(jaw_cfg), "servo_max_vel": None})
