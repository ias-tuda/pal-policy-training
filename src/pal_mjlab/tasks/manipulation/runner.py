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
`widen_actor_on_resume` (plan S10.1's box yaw) does the same for the actor, so the policy it
resumes with computes exactly the checkpoint's actions.

`rescale_actions_on_load` (plan S10.1: W2 box checkpoints on the WC arm, whose arm 1 and arm 4
action scales are 2.6x and 2.2x W2's) converts a loaded actor to the env's action scales: the action
is an absolute target ``default + scale * a``, so the output layer, the action noise and -- where
the actor observes its raw last action -- that input's normaliser are multiplied by old / new scale
per joint, and the loaded policy commands exactly the targets it commanded before
(`rescale_actions`). Applied on every load, the resume's and an eval's alike.

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

  widen_actor_on_resume: bool = False
  """The same for the actor (plan S10.1's box yaw, `widen_input`). Off by default."""

  rescale_actions_on_load: dict[str, float] | None = None
  """Joint regex -> (the checkpoint's action scale / this env's) for the joints whose scales
  differ (plan S10.1's ``CKPT_SPAWN``, `rescale_actions`); None loads the actor as saved."""


def widen_input(
  loaded: dict, key: str, model: torch.nn.Module, params: list
) -> int:
  """Widen ``loaded[key]`` (``"critic_state_dict"`` or ``"actor_state_dict"``), in place, to
  ``model``'s input width; returns the inputs added.

  The new inputs are the last ones (`pick_place_env_cfg._add_critic_release`, plan S10.1's
  `box.append_critic_box_yaw` / ``box_yaw`` append them). The first linear layer gets zero columns
  for them and the observation normaliser mean 0 and std 1, so the widened model computes exactly
  what the saved one did. The optimiser's Adam moments of that layer are padded with zeros the same
  way -- only ``model``'s own: the actor and the critic share one optimiser and may widen by
  different amounts. ``params`` is the optimiser's parameter list in its order (`state` is keyed
  by the index into it). Any other shape difference raises.
  """
  what = key.removesuffix("_state_dict")
  saved = loaded[key]
  want = model.state_dict()
  added = None
  for name, value in want.items():
    old = saved[name]
    if old.shape == value.shape:
      continue
    if (
      old.dim() != 2 or old.shape[0] != value.shape[0] or old.shape[1] > value.shape[1]
    ):
      raise ValueError(
        f"{what} {name}: {tuple(old.shape)} cannot widen to {tuple(value.shape)}"
      )
    extra = value.shape[1] - old.shape[1]
    if added not in (None, extra):
      raise ValueError(f"{what} {name}: widened by {extra}, another tensor by {added}")
    added = extra
    fill = 1.0 if name.endswith(("_var", "_std")) else 0.0
    pad = torch.full((old.shape[0], extra), fill, dtype=old.dtype, device=old.device)
    saved[name] = torch.cat([old, pad], dim=1)
  if not added:
    return 0
  own = {id(p) for p in model.parameters()}
  state = loaded["optimizer_state_dict"]["state"]
  for i, p in enumerate(params):
    if id(p) not in own:
      continue
    for name, moment in state.get(i, {}).items():
      if torch.is_tensor(moment) and moment.dim() == 2 and moment.shape != p.shape:
        assert moment.shape[0] == p.shape[0] and moment.shape[1] + added == p.shape[1]
        pad = torch.zeros(
          moment.shape[0], added, dtype=moment.dtype, device=moment.device
        )
        # A new dict: `Optimizer.state_dict` shares the inner ones with the live optimiser.
        state[i] = {**state[i], name: torch.cat([moment, pad], dim=1)}
  return added


def _scale_moments(state: dict, i: int, factor: torch.Tensor) -> None:
  """Scale optimiser state ``i``'s Adam moments as their parameter was: first by ``factor``,
  second by ``factor ** 2``."""
  if i not in state:
    return
  state[i] = {
    **state[i],
    **{
      k: v * (factor if k == "exp_avg" else factor * factor)
      for k, v in state[i].items()
      if k in ("exp_avg", "exp_avg_sq") and torch.is_tensor(v)
    },
  }


def rescale_actions(
  loaded: dict,
  ratio: torch.Tensor,
  action_cols: list[int] | None,
  models: dict[str, torch.nn.Module],
  params: list,
  eps: float = 1e-2,
) -> None:
  """Convert ``loaded``'s actor, in place, to new action scales: ``ratio`` (A,) is old scale /
  new scale per action. An absolute action ``a`` on the old scales is the target
  ``default + old * a`` = ``default + new * (ratio * a)``, so the actor's last linear layer (its
  rows) and the Gaussian's std are multiplied by ``ratio`` (a log-std shifted by log ratio): the
  mean action, the target it commands and the exploration noise in target space are unchanged.

  ``action_cols`` are the inputs that observe the raw last action (``CMD_OBS=act``; None for
  ``cmd``, which observes the commanded target in rad), at the same columns in the actor and the
  critic: their values are now ``ratio`` times the old ones, so their normaliser statistics are
  carried into the new units (mean and std x ratio, var x ratio**2) and, since rsl_rl normalises by
  ``std + eps``, the first layer's columns by (ratio s + eps) / (ratio (s + eps)) -- the first
  layer then sees exactly what it saw. ``models`` maps ``"actor_state_dict"`` /
  ``"critic_state_dict"`` to the live models (for the optimiser's moments, scaled with their
  parameters); ``params`` as for `widen_input`; ``eps`` the normaliser's."""
  sd = loaded["actor_state_dict"]
  layers = sorted(
    (int(k.split(".")[1]), k)
    for k in sd
    if k.startswith("mlp.") and k.endswith(".weight")
  )
  w_key = layers[-1][1]
  b_key = w_key.removesuffix("weight") + "bias"
  if sd[w_key].shape[0] != ratio.numel():
    raise ValueError(
      f"actor output {tuple(sd[w_key].shape)} does not match {ratio.numel()} action scales"
    )
  r = ratio.to(sd[w_key])
  # (state-dict key, parameter name, factor) of every parameter scaled, for the moments.
  scaled = [
    ("actor_state_dict", w_key, r.unsqueeze(-1)),
    ("actor_state_dict", b_key, r),
  ]
  sd[w_key] = sd[w_key] * r.unsqueeze(-1)
  sd[b_key] = sd[b_key] * r
  if "distribution.log_std_param" in sd:
    sd["distribution.log_std_param"] = sd["distribution.log_std_param"] + torch.log(r)
  elif "distribution.std_param" in sd:
    sd["distribution.std_param"] = sd["distribution.std_param"] * r
    scaled.append(("actor_state_dict", "distribution.std_param", r))
  else:
    raise ValueError("the actor has no Gaussian std parameter to rescale")
  if action_cols is not None:
    cols = torch.as_tensor(action_cols, device=r.device)
    for key in ("actor_state_dict", "critic_state_dict"):
      net = loaded[key]
      first = sorted(
        (int(k.split(".")[1]), k)
        for k in net
        if k.startswith("mlp.") and k.endswith(".weight")
      )[0][1]
      s_old = net["obs_normalizer._std"][0, cols]
      col_factor = torch.ones(net[first].shape[1], dtype=r.dtype, device=r.device)
      col_factor[cols] = (r * s_old + eps) / (r * (s_old + eps))
      net[first] = net[first] * col_factor
      scaled.append((key, first, col_factor))
      for name, power in (
        ("obs_normalizer._mean", 1),
        ("obs_normalizer._std", 1),
        ("obs_normalizer._var", 2),
      ):
        t = net[name].clone()
        t[:, cols] = t[:, cols] * r.to(t) ** power
        net[name] = t
  by_id = {
    id(p): (key, name)
    for key, model in models.items()
    for name, p in model.named_parameters()
  }
  state = loaded["optimizer_state_dict"]["state"]
  for i, p in enumerate(params):
    where = by_id.get(id(p))
    for key, name, factor in scaled:
      if where == (key, name):
        _scale_moments(state, i, factor)


def widen_critic_input(loaded: dict, critic: torch.nn.Module, params: list) -> int:
  """`widen_input` on the critic (plan S7-9's PQ)."""
  return widen_input(loaded, "critic_state_dict", critic, params)


class ManipulationRunner(ManipulationOnPolicyRunner):
  """`ManipulationOnPolicyRunner` that honours `save_only_last`, resumes a run's reward
  curriculum along with its policy, and writes the files a deployment checks the robot's
  start against.

  Overriding `save` rather than `learn` covers both of rsl_rl's call sites -- the periodic
  save inside the loop and the final one after it -- without reimplementing the loop.
  """

  def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
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
    widen = [
      (key, model)
      for flag, key, model in (
        ("widen_actor_on_resume", "actor_state_dict", self.alg._raw_actor),
        ("widen_critic_on_resume", "critic_state_dict", self.alg._raw_critic),
      )
      if self.cfg.get(flag, False)
    ]
    rescale = self.cfg.get("rescale_actions_on_load")
    if not widen and not rescale:
      return super().load(path, load_cfg, strict, map_location)
    loaded = torch.load(path, map_location=map_location, weights_only=False)
    params = [p for g in self.alg.optimizer.param_groups for p in g["params"]]
    added = {key: widen_input(loaded, key, model, params) for key, model in widen}
    for key, n in added.items():
      if n:
        print(
          f"[INFO]: widened the {key.removesuffix('_state_dict')} of {path} by {n} inputs "
          "(zero weights; plan S7-9 PQ / S10.1 box pose)"
        )
    if rescale:
      ratio, cols = self._action_rescale(rescale)
      models = {
        "actor_state_dict": self.alg._raw_actor,
        "critic_state_dict": self.alg._raw_critic,
      }
      eps = float(getattr(self.alg._raw_actor.obs_normalizer, "eps", 1e-2))
      rescale_actions(loaded, ratio, cols, models, params, eps)
      print(
        f"[INFO]: rescaled the actor of {path} to this env's action scales, x "
        f"{[round(float(x), 4) for x in ratio]} (plan S10.1 CKPT_SPAWN); last-action inputs "
        f"{'rescaled' if cols is not None else 'not observed'}"
      )
    elif not any(added.values()):
      return super().load(path, load_cfg, strict, map_location)
    with tempfile.TemporaryDirectory() as tmp:
      widened = os.path.join(tmp, os.path.basename(path))
      torch.save(loaded, widened)
      return super().load(widened, load_cfg, strict, map_location)

  def _action_rescale(self, regex_ratio: dict) -> tuple[torch.Tensor, list[int] | None]:
    """(old / new scale per action, in the action term's order; the actor's columns of the raw
    last action, or None when it observes the commanded target) for `rescale_actions`."""
    import re

    from pal_mjlab.tasks.manipulation import mdp

    env = self.env.unwrapped
    term = env.action_manager.get_term("joint_pos")
    names = [n.split("/", 1)[-1] for n in term.target_names]
    ratio = torch.ones(len(names))
    for i, n in enumerate(names):
      for pattern, r in regex_ratio.items():
        if re.fullmatch(pattern, n):
          ratio[i] = float(r)
    om = env.observation_manager
    cols, offset = None, 0
    for name, cfg, dim in zip(
      om._group_obs_term_names["actor"],
      om._group_obs_term_cfgs["actor"],
      om._group_obs_term_dim["actor"],
      strict=True,
    ):
      width = int(torch.tensor(dim).prod())
      if name == "actions" and cfg.func is mdp.last_action:
        cols = list(range(offset, offset + width))
      offset += width
    return ratio, cols

  def save(self, path: str, infos: dict | None = None) -> None:
    # `self.cfg` is the agent cfg as a plain dict (`asdict`, in mjlab's train script), so a
    # cfg that predates this field simply does not carry it.
    if self.cfg.get("save_only_last", False):
      path = os.path.join(os.path.dirname(path), LAST_CHECKPOINT)
    super().save(path, infos)
