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

Both pieces live here rather than as edits to mjlab or rsl_rl, which are read-mostly
dependencies (CLAUDE.md, *External Dependencies*): a cfg subclass that adds the field and a
runner subclass that acts on it, in the one place the manipulation tasks already share.
"""

from __future__ import annotations

import dataclasses
import os

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.manipulation.rl.runner import ManipulationOnPolicyRunner

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


class ManipulationRunner(ManipulationOnPolicyRunner):
  """`ManipulationOnPolicyRunner` that honours `save_only_last`, and resumes a run's reward
  curriculum along with its policy.

  Overriding `save` rather than `learn` covers both of rsl_rl's call sites -- the periodic
  save inside the loop and the final one after it -- without reimplementing the loop.
  """

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

  def save(self, path: str, infos: dict | None = None) -> None:
    # `self.cfg` is the agent cfg as a plain dict (`asdict`, in mjlab's train script), so a
    # cfg that predates this field simply does not carry it.
    if self.cfg.get("save_only_last", False):
      path = os.path.join(os.path.dirname(path), LAST_CHECKPOINT)
    super().save(path, infos)
