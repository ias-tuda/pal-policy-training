"""S5.1 item 4: the tabletop arms' commanded targets move no faster than the hardware.

`mdp.RateLimitedJointPositionAction` (Claudius's 8068e64) clamps each control step's change
in the commanded target to ``joint_velocity_limits * CONTROL_DT``: 0.039 rad on arm 1-2,
0.047 on arm 3-7. These check that a saturated action ramps at exactly that rate, that the
jaw is not limited (decision 0002 was chosen over his jaw slew), and that a reset re-seeds
the reference -- otherwise the first step of an episode is measured against the last step
of the previous one.
"""

import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import (
  make_pick_place_env_cfg,
)
from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS_LEFT_ARM
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import CONTROL_DT


@pytest.fixture(scope="module")
def env():
  cfg = make_pick_place_env_cfg(KANGAROO_GRIPPERS_LEFT_ARM, gripper_name="left")
  cfg.scene.num_envs = 1
  e = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  e.reset()
  yield e
  e.close()


def _names(term) -> list[str]:
  return [n.split("/", 1)[-1] for n in term.target_names]


def test_the_limits_are_the_urdf_speeds_times_the_control_period(env):
  term = env.action_manager.get_term("joint_pos")
  rate = dict(zip(_names(term), term._rate[0].tolist(), strict=True))
  assert CONTROL_DT == pytest.approx(0.02)
  for i in (1, 2):
    assert rate[f"arm_left_{i}_joint"] == pytest.approx(1.95 * 0.02)
  for i in range(3, 8):
    assert rate[f"arm_left_{i}_joint"] == pytest.approx(2.35 * 0.02)
  assert rate["gripper_left_finger_joint"] == float("inf")


def test_a_saturated_action_ramps_at_the_limit_and_resets(env):
  term = env.action_manager.get_term("joint_pos")
  names = _names(term)
  default = term._default_target()[0].clone()
  ones = torch.ones(1, len(names), device=env.device)
  targets = []
  for _ in range(3):
    env.step(ones)
    targets.append(term._processed_actions[0].clone())
  steps = torch.diff(torch.stack([default, *targets]), dim=0)
  rate = term._rate[0]
  arm = torch.isfinite(rate)
  # Every arm joint moves by exactly its limit (the scaled action asks for far more).
  assert torch.allclose(steps[:, arm], rate[arm].expand(3, -1), atol=1e-6)
  # The jaw jumps straight to its clipped target on the first step.
  jaw = names.index("gripper_left_finger_joint")
  assert targets[0][jaw] == pytest.approx(float(term._clip[0, jaw, 1]))

  env.reset()
  assert torch.allclose(term._last_target[0], default)
