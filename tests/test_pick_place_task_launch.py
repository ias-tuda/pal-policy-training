"""Smoke tests launching train and play for the Kangaroo pick-and-place task.

Mirrors ``test_kangaroo_velocity_task_launch.py``. It also pins the numbers the task is
built on -- the action space, the goal, the spawn box -- because those came out of a
measurement (``../docs/reports/pick-place-v0/s1-report.md``) and a silent change to any of
them makes every result recorded against this task id incomparable.
"""

import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import pytest
import torch
from conftest import get_test_device
from mjlab.scripts.play import PlayConfig, run_play
from mjlab.scripts.train import TrainConfig, run_train
from mjlab.tasks.registry import list_tasks, load_env_cfg
from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import (
  GOAL_THRESHOLD,
  SPAWN_JITTER_XY,
  SPAWN_XY,
  TARGET_XY,
  TARGET_Z,
)

TASK = "Mjlab-PickPlace-Pal-Kangaroo-Grippers"

NUM_ENVS = 2
NUM_STEPS = 4


class StepOnlyViewer:
  """Viewer stub that steps the policy instead of opening a window."""

  def __init__(self, env, policy, **kwargs):
    del kwargs
    self.env = env
    self.policy = policy

  def run(self) -> None:
    obs = self.env.get_observations()
    for _ in range(NUM_STEPS):
      with torch.inference_mode():
        actions = self.policy(obs)
      obs = self.env.step(actions)[0]


def test_task_is_registered():
  assert TASK in list_tasks()


def test_train_launches(tmp_path, monkeypatch):
  device = get_test_device()
  monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0" if device == "cuda" else "")

  cfg = TrainConfig.from_task(TASK)
  cfg.env.scene.num_envs = NUM_ENVS
  cfg.agent.logger = "tensorboard"
  cfg.agent.num_steps_per_env = NUM_STEPS
  cfg.agent.algorithm.num_mini_batches = 1
  cfg.agent.max_iterations = 1

  log_dir = tmp_path / "train"
  run_train(TASK, cfg, log_dir)

  assert (log_dir / "model_0.pt").exists()


def test_play_launches(monkeypatch):
  monkeypatch.setattr("mjlab.scripts.play.NativeMujocoViewer", StepOnlyViewer)

  run_play(
    TASK,
    PlayConfig(
      agent="zero",
      num_envs=NUM_ENVS,
      device=get_test_device(),
      viewer="native",
    ),
  )


def test_one_arm_action_space():
  """Eight actions: the left arm's seven joints and its jaw. The right arm is held, not driven."""
  cfg = load_env_cfg(TASK, play=False)
  action = cfg.actions["joint_pos"]
  assert action.actuator_names == (
    r"arm_left_[1-7]_joint",
    r"gripper_left_finger_joint",
  )


def test_goal_is_the_measured_point_on_the_table():
  cfg = load_env_cfg(TASK, play=False)
  command = cfg.commands["goal_pose"]
  assert command.difficulty == "dynamic", (
    "difficulty='fixed' hardcodes (0.4, 0, 0.3), which is below this table"
  )
  for axis, value in zip(("x", "y"), TARGET_XY, strict=True):
    assert getattr(command.target_position_range, axis) == (value, value)
  assert command.target_position_range.z == (TARGET_Z, TARGET_Z)
  assert command.success_threshold == GOAL_THRESHOLD
  # Placement belongs to the reset event, which knows where the table is.
  assert command.object_pose_range is None


def test_spawn_box_is_relative_to_the_table_not_the_env():
  """`reset_objects_on_surface` adds the table's own XY, so the range has to be an offset."""
  from pal_mjlab.tasks.manipulation import scene

  cfg = load_env_cfg(TASK, play=False)
  pose_range = cfg.events["reset_objects"].params["pose_range"]
  centre_x = sum(pose_range["x"]) / 2 + scene.TABLE_POS[0]
  centre_y = sum(pose_range["y"]) / 2 + scene.TABLE_POS[1]
  assert (centre_x, centre_y) == pytest.approx(SPAWN_XY)
  assert pose_range["x"][1] - pose_range["x"][0] == pytest.approx(2 * SPAWN_JITTER_XY)


def test_dr0_removes_the_randomisation_the_base_cfg_turns_on():
  """v0 trains at DR0; test_manipulation_dr_ladder covers the rungs above it."""
  cfg = load_env_cfg(TASK, play=False)
  assert "pad_friction_slide" not in cfg.events
  assert "pad_friction_spin" not in cfg.events
  assert cfg.events["reset_robot_joints"].params["position_range"] == (0.0, 0.0)


def test_release_and_success_are_present():
  """Without them the policy carries the cube to the goal and never lets go."""
  cfg = load_env_cfg(TASK, play=False)
  assert "release" in cfg.rewards
  assert "success" in cfg.rewards
  assert "success" in cfg.terminations
