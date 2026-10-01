"""The two files a deployment checks the robot's start against (Claudius's fda9492 / a82fdf1).

`ManipulationRunner` writes `deployment_start_config.json` (the resolved default joint state) and
`start_observation_distribution.json` (per-feature actor observation statistics over the
initially reset training envs) beside a run's checkpoints, once, when the run starts.
`scripts/write_start_distribution.py` writes the second one for a run that predates this.

* end to end through mjlab's `run_train`, both files appear in the run directory, with the
  actor's terms in order and each as wide as the observation manager says;
* a runner without a log dir (play, eval, the exporters) writes nothing, and under multi-GPU only
  rank 0 does;
* a failure writing them is a warning: the run goes on;
* the statistics are the ones named, over the rows of the observation, read after a reset.
"""

import json
import math
import types

import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import pal_mjlab.tasks.manipulation.runner as runner_module
import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scripts.train import TrainConfig, run_train
from mjlab.tasks.registry import load_env_cfg
from pal_mjlab.tasks.manipulation.deployment import (
  write_start_observation_distribution,
)
from pal_mjlab.tasks.manipulation.runner import ManipulationRunner

TASK = "Mjlab-PickPlace-Pal-Kangaroo-Grippers"
NUM_ENVS = 4

#: mjlab's `ManipulationOnPolicyRunner`, whose `__init__` is stubbed so no env or policy is built.
_PARENT = ManipulationRunner.__mro__[1]


def _width(shape) -> int:
  return math.prod(shape)


# -- End to end -----------------------------------------------------------------------------


def test_train_writes_both_files_beside_the_checkpoints(tmp_path, monkeypatch):
  device = get_test_device()
  monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0" if device == "cuda" else "")
  cfg = TrainConfig.from_task(TASK)
  cfg.env.scene.num_envs = NUM_ENVS
  cfg.agent.logger = "tensorboard"
  cfg.agent.num_steps_per_env = 4
  cfg.agent.algorithm.num_mini_batches = 1
  cfg.agent.max_iterations = 1
  cfg.agent.save_interval = 1
  run = tmp_path / "logs/rsl_rl/kangaroo_pick_place/run"
  run_train(TASK, cfg, run)

  env_cfg = load_env_cfg(TASK)
  env_cfg.scene.num_envs = NUM_ENVS
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  try:
    manager = env.observation_manager
    names = manager.active_terms["actor"]
    widths = [_width(s) for s in manager.group_obs_term_dim["actor"]]
    joint_names = list(env.scene["robot"].joint_names)
  finally:
    env.close()

  dist = json.loads((run / "start_observation_distribution.json").read_text())
  assert dist["num_initial_envs"] == NUM_ENVS
  assert list(dist["terms"]) == names
  for name, width in zip(names, widths, strict=True):
    term = dist["terms"][name]
    assert set(term) == {"min", "p01", "mean", "std", "p99", "max"}
    assert all(len(v) == width and all(map(math.isfinite, v)) for v in term.values())

  start = json.loads((run / "deployment_start_config.json").read_text())
  assert start["joint_names"] == joint_names
  assert list(start["joint_positions"]) == joint_names


# -- When it is written ---------------------------------------------------------------------


def _construct(log_dir, monkeypatch, rank=0, world_size=1):
  """Build a `ManipulationRunner` over a stubbed parent; return what the two writers got."""
  written = []

  def parent_init(self, env, train_cfg, log_dir=None, device="cpu"):
    self.env = types.SimpleNamespace(unwrapped="the env")
    self.logger = types.SimpleNamespace(log_dir=log_dir)
    self.is_distributed = world_size > 1
    self.gpu_global_rank = rank

  monkeypatch.setattr(_PARENT, "__init__", parent_init)
  for name in ("write_deployment_start_config", "write_start_observation_distribution"):
    monkeypatch.setattr(
      runner_module,
      name,
      lambda env, path, name=name: written.append((name, env, path.name)),
    )
  ManipulationRunner(None, {}, log_dir)
  return written


def test_a_runner_with_a_log_dir_writes_both(tmp_path, monkeypatch):
  assert _construct(str(tmp_path), monkeypatch) == [
    ("write_deployment_start_config", "the env", "deployment_start_config.json"),
    (
      "write_start_observation_distribution",
      "the env",
      "start_observation_distribution.json",
    ),
  ]


def test_a_runner_without_a_log_dir_writes_nothing(monkeypatch):
  """Play, eval and the exporters build the runner with `log_dir=None`."""
  assert _construct(None, monkeypatch) == []


def test_only_rank_zero_writes(tmp_path, monkeypatch):
  assert _construct(str(tmp_path), monkeypatch, rank=1, world_size=2) == []
  assert len(_construct(str(tmp_path), monkeypatch, rank=0, world_size=2)) == 2


def test_a_failed_write_is_a_warning(tmp_path, monkeypatch, capsys):
  def parent_init(self, env, train_cfg, log_dir=None, device="cpu"):
    self.env = types.SimpleNamespace(unwrapped=None)
    self.logger = types.SimpleNamespace(log_dir=log_dir)
    self.is_distributed = False

  def broken(env, path):
    raise TypeError("Expected concatenated actor observations")

  monkeypatch.setattr(_PARENT, "__init__", parent_init)
  monkeypatch.setattr(runner_module, "write_deployment_start_config", broken)
  ManipulationRunner(None, {}, str(tmp_path))
  assert "[WARN] deployment start files not written" in capsys.readouterr().out


# -- The statistics -------------------------------------------------------------------------


class _Manager:
  _group_obs_term_names = {"actor": ["a", "b"]}
  _group_obs_term_dim = {"actor": [(2,), (1,)]}


class _Env:
  """What `write_start_observation_distribution` touches. Its observation is a bogus one
  until `reset` has been called: the goal is only drawn there."""

  num_envs = 5
  observation_manager = _Manager()

  def __init__(self):
    self.was_reset = False

  def reset(self):
    self.was_reset = True
    column = torch.arange(5, dtype=torch.float32)
    return {"actor": torch.stack((column, 10 * column, -column), dim=1)}, {}


def test_the_statistics_are_per_feature_over_the_envs_after_a_reset(tmp_path):
  env = _Env()
  path = write_start_observation_distribution(env, tmp_path / "d.json")
  assert env.was_reset
  dist = json.loads(path.read_text())
  assert dist["num_initial_envs"] == 5
  a, b = dist["terms"]["a"], dist["terms"]["b"]
  assert a["min"] == [0.0, 0.0] and a["max"] == [4.0, 40.0]
  assert a["mean"] == [2.0, 20.0]
  assert a["std"] == pytest.approx([2**0.5, 10 * 2**0.5])  # population std, not sample
  assert a["p01"] == pytest.approx([0.04, 0.4]) and a["p99"] == pytest.approx(
    [3.96, 39.6]
  )
  assert b["min"] == [-4.0] and b["max"] == [0.0]


def test_a_dict_observation_is_refused(tmp_path):
  env = _Env()
  env.reset = lambda: ({"actor": {"x": torch.zeros(5, 3)}}, {})
  with pytest.raises(TypeError, match="concatenated actor observations"):
    write_start_observation_distribution(env, tmp_path / "d.json")
