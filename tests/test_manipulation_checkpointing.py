"""`save_only_last` collapses the checkpoint series to one rolling file.

rsl_rl saves `model_<iter>.pt` every `save_interval` iterations and keeps them all. That is
the right default when a run's intermediate policies matter, and the wrong one for a sweep
whose deliverable is the final policy. The flag is off by default, so this checks both
branches -- a flag that silently does nothing looks exactly like a flag that works.
"""

import os

import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import pytest
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from pal_mjlab.tasks.manipulation.rl_cfg import manipulation_ppo_runner_cfg
from pal_mjlab.tasks.manipulation.runner import (
  LAST_CHECKPOINT,
  ManipulationRunner,
  ManipulationRunnerCfg,
)

MANIPULATION_TASKS = (
  "Mjlab-PickPlace-Pal-Kangaroo-Grippers",
  "Mjlab-Lift-Pal-Kangaroo-Grippers",
  "Mjlab-Stack-Pal-Kangaroo-Grippers",
)


#: mjlab's `ManipulationOnPolicyRunner` -- the class whose `save` actually writes the file,
#: and the one stubbed out so these tests need no policy, no env and no disk.
_PARENT = ManipulationRunner.__mro__[1]


class _Recorder(ManipulationRunner):
  """A runner that records the path it would write to, without writing one."""

  def __init__(self, cfg: dict) -> None:
    self.cfg = cfg
    self.written: str | None = None


def _save(cfg: dict, path: str, monkeypatch) -> str | None:
  monkeypatch.setattr(
    _PARENT, "save", lambda self, p, infos=None: setattr(self, "written", p)
  )
  runner = _Recorder(cfg)
  runner.save(path)
  return runner.written


def test_the_default_keeps_the_numbered_series(monkeypatch):
  assert (
    _save({"save_only_last": False}, "/logs/run/model_7000.pt", monkeypatch)
    == "/logs/run/model_7000.pt"
  )


def test_the_flag_overwrites_one_file(monkeypatch):
  written = _save({"save_only_last": True}, "/logs/run/model_7000.pt", monkeypatch)
  assert written == os.path.join("/logs/run", LAST_CHECKPOINT)


def test_the_final_save_lands_on_the_same_file(monkeypatch):
  """rsl_rl saves once more after the loop, under a different iteration number.

  Overriding `save` rather than `learn` is what makes both call sites collapse to one file;
  if they did not, a finished run would leave two checkpoints and the flag would be a lie.
  """
  periodic = _save({"save_only_last": True}, "/logs/run/model_20000.pt", monkeypatch)
  final = _save({"save_only_last": True}, "/logs/run/model_19999.pt", monkeypatch)
  assert periodic == final


def test_a_cfg_without_the_field_still_saves(monkeypatch):
  """`self.cfg` is a plain dict; a cfg predating the flag must not raise."""
  assert _save({}, "/logs/run/model_100.pt", monkeypatch) == "/logs/run/model_100.pt"


def test_the_default_interval_is_a_thousand():
  cfg = manipulation_ppo_runner_cfg("test")
  assert cfg.save_interval == 1000
  assert cfg.save_only_last is False, "the flag must be opt-in"


@pytest.mark.parametrize("task_id", MANIPULATION_TASKS)
def test_every_manipulation_task_can_honour_the_flag(task_id):
  """A task left on the base runner would accept the flag and ignore it."""
  runner_cls = load_runner_cls(task_id)
  rl_cfg = load_rl_cfg(task_id)
  assert runner_cls is not None and issubclass(runner_cls, ManipulationRunner), task_id
  assert isinstance(rl_cfg, ManipulationRunnerCfg), task_id
  assert rl_cfg.save_interval == 1000, task_id
