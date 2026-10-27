"""Resuming a pick-and-place run from a checkpoint (plan S7: "5 k plus a resume if still rising").

mjlab's `train` can resume -- `--agent.resume True --agent.load-run <regex>
--agent.load-checkpoint <regex>` -- and rsl_rl's `load` brings back the policy, the optimiser,
the learning rate and the iteration count. Three things stood between that and extending a
running S7 run faithfully, and each is pinned here:

* the run names carry `+` (`R-C-strict+M1+M2+M3`), which as a regex is a quantifier, so the
  plain name matches nothing; `scripts/pick_place_resume.sh` escapes and anchors it;
* mjlab looks for the run only under the checkout's own log root, and the run's checkout must
  not be touched while it has jobs; the script links the run directory in;
* the env-step clock the reward curriculum reads starts again at 0 in a new process, which
  would put the joint-velocity hinge back to its first stage for ~750 iterations;
  `ManipulationRunner.learn` restarts it where the checkpoint left off.
"""

import os
import pathlib
import subprocess
import types

import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scripts.train import TrainConfig, run_train
from mjlab.utils.os import get_checkpoint_path
from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import (
  make_pick_place_env_cfg,
)
from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS_LEFT_ARM
from pal_mjlab.tasks.manipulation.runner import ManipulationRunner

_SCRIPT = pathlib.Path(__file__).parent.parent / "scripts/pick_place_resume.sh"
_SBATCH = pathlib.Path(__file__).parent.parent / "scripts/pick_place.sbatch"
TASK = "Mjlab-PickPlace-Pal-Kangaroo-Grippers"
EXPERIMENT = "kangaroo_pick_place"
RUN = (
  "2026-09-25_09-12-56_pp_s71-FA_R-C-strict+M1+M2+M3_DR0_PPO-stack_slew0.5_n2048_seed2"
)

#: mjlab's `ManipulationOnPolicyRunner`, whose `learn` is stubbed so no env or policy is built.
_PARENT = ManipulationRunner.__mro__[1]


# -- Finding the checkpoint -----------------------------------------------------------------


@pytest.fixture
def runs(tmp_path):
  """The run in one checkout (`first`), and the checkout the resume starts from (`repo`), which
  already holds a newer run whose name extends the first's -- what an unanchored or unescaped
  pattern would pick instead."""
  first = tmp_path / "first/logs/rsl_rl" / EXPERIMENT / RUN
  first.mkdir(parents=True)
  for it in (500, 900, 1000, 4999):
    (first / f"model_{it}.pt").write_bytes(b"x")
  repo = tmp_path / "repo"
  decoy = repo / "logs/rsl_rl" / EXPERIMENT / f"{RUN}x"
  decoy.mkdir(parents=True)
  (decoy / "model_4999.pt").write_bytes(b"y")
  return first, repo


def _flags(checkpoint: pathlib.Path, repo: pathlib.Path) -> dict[str, str]:
  out = subprocess.run(
    ["bash", str(_SCRIPT)],
    env={
      "PATH": os.environ["PATH"],
      "RESUME": str(checkpoint),
      "REPO": str(repo),
      "EXPERIMENT": EXPERIMENT,
    },
    capture_output=True,
    text=True,
    check=True,
  ).stdout.split()
  return dict(zip(out[0::2], out[1::2], strict=True))


@pytest.mark.parametrize("it", (900, 4999))
def test_the_flags_resolve_to_exactly_that_checkpoint(runs, it):
  """Through mjlab's own resolver, from the other checkout's log root."""
  first, repo = runs
  flags = _flags(first / f"model_{it}.pt", repo)
  assert flags["--agent.resume"] == "True"
  found = get_checkpoint_path(
    repo / "logs/rsl_rl" / EXPERIMENT,
    flags["--agent.load-run"],
    flags["--agent.load-checkpoint"],
  )
  assert found.resolve() == (first / f"model_{it}.pt").resolve()
  assert (repo / "logs/rsl_rl" / EXPERIMENT / RUN).is_symlink()


def test_the_plain_run_name_matches_nothing(runs):
  """Positive control for the escaping: `+` in the name is a regex quantifier."""
  first, _ = runs
  with pytest.raises(ValueError, match="No run directories"):
    get_checkpoint_path(first.parent, f"^{RUN}$", r"^model_900\.pt$")


def test_running_it_twice_reuses_the_link(runs):
  first, repo = runs
  assert _flags(first / "model_900.pt", repo) == _flags(first / "model_900.pt", repo)


def test_a_missing_checkpoint_refuses(runs):
  first, repo = runs
  with pytest.raises(subprocess.CalledProcessError):
    _flags(first / "model_123.pt", repo)


def test_the_sbatch_passes_the_flags_through():
  """RESUME reaches `train`, and says so in the job's provenance and wandb tags."""
  text = _SBATCH.read_text()
  assert 'RESUME="${RESUME:-}"' in text
  assert 'source "$REPO/scripts/pick_place_resume.sh"' in text
  assert '${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}' in text
  assert 'echo "resume      ${RESUME:-(none)}"' in text
  assert "resumed" in text


def test_train_resumes_from_the_named_checkpoint(tmp_path, monkeypatch):
  """End to end through mjlab's `run_train`, the path `pick_place.sbatch` execs: two
  iterations in one checkout, then two more resumed from `model_1.pt` in another. The
  iteration count continues (a fresh start would save `model_0.pt`), into a new directory."""
  device = get_test_device()
  monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0" if device == "cuda" else "")

  def cfg() -> TrainConfig:
    c = TrainConfig.from_task(TASK)
    c.env.scene.num_envs = 2
    c.agent.logger = "tensorboard"
    c.agent.num_steps_per_env = 4
    c.agent.algorithm.num_mini_batches = 1
    c.agent.max_iterations = 2
    c.agent.save_interval = 1
    return c

  first = tmp_path / "first/logs/rsl_rl" / EXPERIMENT / RUN
  run_train(TASK, cfg(), first)
  assert sorted(p.name for p in first.glob("model_*.pt")) == [
    "model_0.pt",
    "model_1.pt",
  ]

  repo = tmp_path / "repo"
  flags = _flags(first / "model_1.pt", repo)
  resumed = cfg()
  resumed.agent.resume = True
  resumed.agent.load_run = flags["--agent.load-run"]
  resumed.agent.load_checkpoint = flags["--agent.load-checkpoint"]
  second = repo / "logs/rsl_rl" / EXPERIMENT / f"2026-09-25_12-00-00_{RUN}"
  run_train(TASK, resumed, second)
  assert sorted(p.name for p in second.glob("model_*.pt")) == [
    "model_1.pt",
    "model_2.pt",
  ]
  assert torch.load(second / "model_2.pt", weights_only=False)["iter"] == 2
  # The first run is untouched: resuming reads it through the link and writes elsewhere.
  assert sorted(p.name for p in first.glob("model_*.pt")) == [
    "model_0.pt",
    "model_1.pt",
  ]


# -- The env-step clock ---------------------------------------------------------------------


def _counter_at_learn(start_iteration: int, monkeypatch) -> int:
  seen = {}
  monkeypatch.setattr(
    _PARENT,
    "learn",
    lambda self, n, init_at_random_ep_len=False: seen.update(
      counter=self.env.unwrapped.common_step_counter
    ),
  )
  runner = ManipulationRunner.__new__(ManipulationRunner)
  runner.cfg = {"num_steps_per_env": 32}
  runner.env = types.SimpleNamespace(
    unwrapped=types.SimpleNamespace(common_step_counter=0)
  )
  runner.current_learning_iteration = start_iteration
  runner.learn(3001)
  return seen["counter"]


def test_a_resumed_run_continues_the_env_step_clock(monkeypatch):
  assert _counter_at_learn(4999, monkeypatch) == 5000 * 32


def test_a_fresh_run_starts_it_at_zero(monkeypatch):
  assert _counter_at_learn(0, monkeypatch) == 0


def test_the_curriculum_reads_that_clock():
  """Why it matters: the joint-velocity hinge's weight is a function of the clock alone."""
  cfg = make_pick_place_env_cfg(
    KANGAROO_GRIPPERS_LEFT_ARM, gripper_name="left", reward="R-C-strict+M1+M2+M3"
  )
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  try:
    env.reset()
    hinge = env.reward_manager.get_term_cfg("joint_vel_hinge")
    ids = torch.arange(2, device=env.device)
    weights = []
    for counter in (0, 5000 * 32):
      env.common_step_counter = counter
      env.curriculum_manager.compute(env_ids=ids)
      weights.append(hinge.weight)
    assert weights == [-0.01, -1.0]
  finally:
    env.close()
