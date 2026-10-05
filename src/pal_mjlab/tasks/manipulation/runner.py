"""Checkpointing policy for the manipulation tasks.

rsl_rl writes one iteration-numbered file per save (`model_1000.pt`, `model_2000.pt`, ...)
and never removes them, which is the right default: a run that dies keeps every snapshot,
and at 139 KB a policy is cheap. It is the wrong default when the checkpoints are not the
deliverable -- a sweep whose point is the final policy accumulates hundreds of files per run
that nothing will ever read.

`save_only_last` collapses them to a single rolling `model_last.pt`. The iteration is not
lost with the filename: `OnPolicyRunner.save` stores it in the checkpoint as `iter`, and
`load` reads it back from there, so resuming works the same either way.

Resuming (`--agent.resume True`, or `RESUME=` in `scripts/pick_place.sbatch`) also restarts
the env-step clock the reward curriculum reads where the checkpoint left it, which a new
process would otherwise start at 0 (plan S7, "5 k plus a resume").

`widen_critic_on_resume` (plan S7-9's PQ) lets a resume load a checkpoint whose critic had fewer
inputs: the critic's new inputs are appended last, and the loaded critic gets zero weights (and an
identity normaliser) for them, so the value function it resumes with is the checkpoint's.

These pieces live here rather than as edits to mjlab or rsl_rl, which are read-mostly
dependencies (CLAUDE.md, *External Dependencies*): a cfg subclass that adds the field and a
runner subclass that acts on it, in the one place the manipulation tasks already share.
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
from pathlib import Path

import torch
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.manipulation.rl.runner import ManipulationOnPolicyRunner

from pal_mjlab.tasks.manipulation.deployment import (
  write_deployment_start_config,
  write_start_observation_distribution,
)

#: The single file `save_only_last` writes. Matches rsl_rl's default `load_checkpoint`
#: regex `model_.*.pt`, so resuming needs no other change -- and matches it *uniquely*,
#: which the numbered files do not: rsl_rl picks the alphabetically last match, and
#: `model_900.pt` sorts after `model_1000.pt`.
LAST_CHECKPOINT = "model_last.pt"


@dataclasses.dataclass
class ManipulationRunnerCfg(RslRlOnPolicyRunnerCfg):
  """`RslRlOnPolicyRunnerCfg` plus a checkpoint-retention flag."""

  save_only_last: bool = False
  """Overwrite one `model_last.pt` instead of keeping every `model_<iter>.pt`.

  Off by default, so existing runs are unchanged. On, a killed run leaves exactly the last
  save -- at `save_interval=1000` that is up to 1000 iterations of lost progress, which is
  the trade being made.
  """

  widen_critic_on_resume: bool = False
  """Let `load` widen a checkpoint's critic for inputs appended since it was saved (plan S7-9's PQ,
  `widen_critic_input`). Off by default: an unexpected shape mismatch should still fail."""


def widen_critic_input(loaded: dict, critic: torch.nn.Module, params: list) -> int:
  """Widen ``loaded``'s critic, in place, to ``critic``'s input width; returns the inputs added.

  The new inputs are the last ones (`pick_place_env_cfg._add_critic_release` appends them). The
  first linear layer gets zero columns for them and the observation normaliser mean 0 and std 1,
  so the widened critic computes exactly what the saved one did. The optimiser's Adam moments of
  that layer are padded with zeros the same way. ``params`` is the optimiser's parameter list in
  its order (`state` is keyed by the index into it). Any other shape difference raises.
  """
  saved = loaded["critic_state_dict"]
  want = critic.state_dict()
  added = None
  for key, value in want.items():
    old = saved[key]
    if old.shape == value.shape:
      continue
    if (
      old.dim() != 2 or old.shape[0] != value.shape[0] or old.shape[1] > value.shape[1]
    ):
      raise ValueError(
        f"critic {key}: {tuple(old.shape)} cannot widen to {tuple(value.shape)}"
      )
    extra = value.shape[1] - old.shape[1]
    if added not in (None, extra):
      raise ValueError(f"critic {key}: widened by {extra}, another tensor by {added}")
    added = extra
    fill = 1.0 if key.endswith(("_var", "_std")) else 0.0
    pad = torch.full((old.shape[0], extra), fill, dtype=old.dtype, device=old.device)
    saved[key] = torch.cat([old, pad], dim=1)
  if not added:
    return 0
  state = loaded["optimizer_state_dict"]["state"]
  for i, p in enumerate(params):
    for name, moment in state.get(i, {}).items():
      if torch.is_tensor(moment) and moment.dim() == 2 and moment.shape != p.shape:
        assert moment.shape[0] == p.shape[0] and moment.shape[1] + added == p.shape[1]
        pad = torch.zeros(
          moment.shape[0], added, dtype=moment.dtype, device=moment.device
        )
        # A new dict: `Optimizer.state_dict` shares the inner ones with the live optimiser.
        state[i] = {**state[i], name: torch.cat([moment, pad], dim=1)}
  return added


class ManipulationRunner(ManipulationOnPolicyRunner):
  """`ManipulationOnPolicyRunner` that honours `save_only_last`, resumes a run's reward
  curriculum along with its policy, and writes the files a deployment checks the robot's
  start against.

  Overriding `save` rather than `learn` covers both of rsl_rl's call sites -- the periodic
  save inside the loop and the final one after it -- without reimplementing the loop.
  """

  def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
    # Plan S11.2's JOINT recipe reads its actor layout and action names off the env before rsl_rl builds
    # the models (`bimanual_policy.prepare_train_cfg`; a no-op for every other recipe).
    from pal_mjlab.tasks.manipulation.bimanual_policy import prepare_train_cfg

    prepare_train_cfg(env.unwrapped, train_cfg)
    super().__init__(env, train_cfg, log_dir, device)
    # `deployment_start_config.json` and `start_observation_distribution.json` beside the
    # checkpoints, once per run (Claudius's fda9492 / a82fdf1, `origin/faster-moves`, which
    # writes them from its own runner). Not on a play / eval runner (no log dir) and, under
    # multi-GPU, only on rank 0. The distribution resets the env, so it is written before the
    # first iteration. A failure here must not cost the run, so it is a warning; the
    # `scripts/write_start_distribution.py` backfill makes the file from the run's selectors.
    run_dir = self.logger.log_dir
    if run_dir is None or (self.is_distributed and self.gpu_global_rank != 0):
      return
    try:
      env = self.env.unwrapped
      write_deployment_start_config(env, Path(run_dir) / "deployment_start_config.json")
      write_start_observation_distribution(
        env, Path(run_dir) / "start_observation_distribution.json"
      )
    except Exception as e:
      print(f"[WARN] deployment start files not written (training continues): {e}")

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    # A resumed run (`load` has set the iteration) must continue the env-step clock too.
    # mjlab's curricula read `common_step_counter` (`reward_curriculum`,
    # `mjlab/envs/mdp/curriculums.py`), which a new process starts at 0: resuming from
    # iteration 4999 would otherwise put the joint-velocity hinge back to its first stage
    # (-0.01 instead of -1) for the next ~750 iterations. Each iteration is
    # `num_steps_per_env` env steps; the checkpoint at iteration i was saved after i + 1.
    if self.current_learning_iteration > 0:
      env = self.env.unwrapped
      steps = (self.current_learning_iteration + 1) * self.cfg["num_steps_per_env"]
      env.common_step_counter = max(env.common_step_counter, steps)
    super().learn(num_learning_iterations, init_at_random_ep_len)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    if not self.cfg.get("widen_critic_on_resume", False):
      return super().load(path, load_cfg, strict, map_location)
    loaded = torch.load(path, map_location=map_location, weights_only=False)
    params = [p for g in self.alg.optimizer.param_groups for p in g["params"]]
    added = widen_critic_input(loaded, self.alg._raw_critic, params)
    if not added:
      return super().load(path, load_cfg, strict, map_location)
    print(
      f"[INFO]: widened the critic of {path} by {added} inputs (zero weights; plan S7-9 PQ)"
    )
    with tempfile.TemporaryDirectory() as tmp:
      widened = os.path.join(tmp, os.path.basename(path))
      torch.save(loaded, widened)
      return super().load(widened, load_cfg, strict, map_location)

  def save(self, path: str, infos: dict | None = None) -> None:
    # `self.cfg` is the agent cfg as a plain dict (`asdict`, in mjlab's train script), so a
    # cfg that predates this field simply does not carry it.
    if self.cfg.get("save_only_last", False):
      path = os.path.join(os.path.dirname(path), LAST_CHECKPOINT)
    super().save(path, infos)
