"""Plan S11.2's training side for JOINT (`pick-place-s11-bimanual-plan.md`, *S11.2*): BI-F's factored policy,
its critic warm-up and KL to the SEQ pair, and a per-dimension std log for every S11.2 code.

* `FactoredArmModel` -- the actor of ``INIT=seq``: two per-arm MLPs, the left's copied from L-T and the
  right's from R-ST / R-ST-W, each on its one-arm 38-D observation plus the other hand's TCP (41-D; the three
  new input columns zero-weighted, normaliser mean 0 / std 1, so iteration 0 computes exactly what the SEQ
  actors do). The right sees its observation through the mirror map of `scripts/eval_chain.MirrorAdapter`
  (joints x s, positions y -> -y, the TCP frame R -> M R F M, yaw -> -yaw) and its goal through the
  JOINT env's gated ``right_target`` (`mdp.right_target_gated`: the wait point before A_latched, A's tracked
  top after); its actions go back x s. The observation normalisers keep the sources' statistics and count,
  and the action std starts at the sources' (not PPO-stack's 0.35).
* `JointPPO` -- rsl_rl 5.4.2's PPO with: a critic warm-up (the first ``critic_warmup_iterations`` updates
  train the value function only, the actor and the adaptive learning rate frozen); a KL term
  ``lambda_k KL(pi_SEQ || pi)`` on the batch, ``lambda_k`` falling linearly from ``kl_coef`` to 0 over
  ``kl_anneal_iterations`` (Kickstarting), the teacher being the factored actor as initialised (= SEQ); and
  every action dimension's std in the loss log (``std/<joint>``).
* `prepare_train_cfg` -- called by `runner.ManipulationRunner` before rsl_rl builds the models: reads the
  actor observation's layout and the action columns off the env and puts them into the cfg.

`JointPPO.update` is a copy of rsl_rl 5.4.2's `PPO.update` (`rsl_rl_lib==5.4.2`, the version `uv.lock`
pins) with those three additions; RND, symmetry and recurrent models are refused rather than half-supported.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

import torch
import torch.nn as nn
from mjlab.rl import RslRlModelCfg, RslRlPpoAlgorithmCfg
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP, EmpiricalNormalization
from rsl_rl.utils import resolve_callable

#: The actor terms each half reads (the one-arm 38-D order, `eval_chain.ONE_ARM_TERMS`), and its extra term.
LEFT_TERMS = (
  "joint_pos",
  "joint_vel",
  "actions",
  "tcp_pos",
  "tcp_quat",
  "object_pos",
  "object_yaw",
  "target_pos",
)
RIGHT_TERMS = (
  "joint_pos",
  "joint_vel",
  "actions",
  "tcp_pos_right",
  "tcp_quat_right",
  "object_b_pos",
  "object_b_yaw",
  "right_target",
)
ONE_ARM_DIM = 38
OTHER_TCP = 3

MODEL_CLASS = "pal_mjlab.tasks.manipulation.bimanual_policy:FactoredArmModel"
PPO_CLASS = "pal_mjlab.tasks.manipulation.bimanual_policy:JointPPO"


@dataclasses.dataclass
class FactoredModelCfg(RslRlModelCfg):
  """`RslRlModelCfg` for `FactoredArmModel`."""

  class_name: str = MODEL_CLASS
  init_lt: str = ""
  """L-T's checkpoint (the left half)."""
  init_rst: str = ""
  """R-ST / R-ST-W's checkpoint (the right half, mirrored)."""
  layout: dict | None = None
  """Filled by `prepare_train_cfg` from the env: term slices, joint and action indices, mirror signs."""


@dataclasses.dataclass
class JointPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """`RslRlPpoAlgorithmCfg` for `JointPPO`."""

  class_name: str = PPO_CLASS
  kl_coef: float = 0.0
  """lambda_0 of the KL to the SEQ teacher; 0: none (every code but BI-F / BI-FS)."""
  kl_anneal_iterations: int = 0
  critic_warmup_iterations: int = 0
  action_names: list | None = None
  """Filled by `prepare_train_cfg`: the action term's target names, for the std log."""


#: BI-F's schedule (plan S11.2): 50 iterations of critic only, the KL from 1 to 0 over 1000.
SEQ_KL_COEF = 1.0
SEQ_KL_ANNEAL = 1000
SEQ_CRITIC_WARMUP = 50


def apply_joint_rl_cfg(cfg, init: str, init_lt: str | None, init_rst: str | None):
  """The PPO recipe ``cfg`` for a plan S11.2 JOINT run: `JointPPO` always (the std log); with ``init="seq"`` the
  factored actor and BI-F's warm-up and KL schedule."""
  alg = dataclasses.asdict(cfg.algorithm)
  alg.pop("class_name", None)
  cfg.algorithm = JointPpoAlgorithmCfg(**alg)
  if init == "seq":
    actor = dataclasses.asdict(cfg.actor)
    actor.pop("class_name", None)
    cfg.actor = FactoredModelCfg(**actor, init_lt=init_lt, init_rst=init_rst)
    cfg.algorithm.kl_coef = SEQ_KL_COEF
    cfg.algorithm.kl_anneal_iterations = SEQ_KL_ANNEAL
    cfg.algorithm.critic_warmup_iterations = SEQ_CRITIC_WARMUP
  return cfg


def _term_joints(env, term: str) -> list[str]:
  cfg = dataclasses.replace(
    env.cfg.observations["actor"].terms[term].params["asset_cfg"]
  )
  cfg.resolve(env.scene)
  robot = env.scene["robot"]
  ids = cfg.joint_ids
  if isinstance(ids, slice):
    ids = range(len(robot.joint_names))[ids]
  return [robot.joint_names[int(i)] for i in ids]


def layout_of(env) -> dict:
  """The actor observation's layout and the action columns of a JOINT env, as `FactoredArmModel` needs them."""
  from pal_mjlab.tasks.manipulation.robots import ARM_MIRROR_SIGNS

  mgr = env.observation_manager
  slices, i = {}, 0
  for name, shape in zip(
    mgr.active_terms["actor"], mgr.group_obs_term_dim["actor"], strict=True
  ):
    w = int(torch.tensor(shape).prod())
    slices[name] = [i, i + w]
    i += w
  term = env.action_manager.get_term("joint_pos")
  targets = [n.split("/")[-1] for n in term.target_names]

  def cols(side):
    return [targets.index(f"arm_{side}_{k}_joint") for k in range(1, 8)] + [
      targets.index(f"gripper_{side}_finger_joint")
    ]

  jp = _term_joints(env, "joint_pos")
  jv = _term_joints(env, "joint_vel")
  out = {
    "slices": slices,
    "obs_dim": i,
    "action_names": targets,
    "left_cols": cols("left"),
    "right_cols": cols("right"),
    "jp_left": [jp.index(targets[c]) for c in cols("left")],
    "jp_right": [jp.index(targets[c]) for c in cols("right")],
    "jv_left": [jv.index(f"arm_left_{k}_joint") for k in range(1, 8)],
    "jv_right": [jv.index(f"arm_right_{k}_joint") for k in range(1, 8)],
    "signs": list(ARM_MIRROR_SIGNS),
  }
  missing = [t for t in (*LEFT_TERMS, *RIGHT_TERMS) if t not in slices]
  if missing:
    raise ValueError(
      f"the factored actor needs actor terms {missing} (INIT=seq builds right_target)"
    )
  return out


def prepare_train_cfg(env, train_cfg: dict) -> None:
  """Before rsl_rl builds the models: the env's layout into a factored actor's cfg, the action names into
  `JointPPO`'s. A no-op for every other recipe."""
  alg = train_cfg.get("algorithm", {})
  if alg.get("class_name") != PPO_CLASS:
    return
  layout = layout_of(env)
  alg["action_names"] = layout["action_names"]
  if train_cfg.get("actor", {}).get("class_name") == MODEL_CLASS:
    train_cfg["actor"]["layout"] = layout


def mirror_quat(q: torch.Tensor) -> torch.Tensor:
  """The TCP frame's mirror map R -> M R F M on a (w, x, y, z) quaternion, w >= 0: M R M is (w, -x, y, -z), and
  F (pi about z) on the right gives (z, y, x, w) of the original. (..., 4)"""
  out = torch.stack([q[..., 3], q[..., 2], q[..., 1], q[..., 0]], dim=-1)
  return torch.where(out[..., :1] < 0.0, -out, out)


class _Halves(nn.Module):
  """The two per-arm inputs out of the JOINT actor vector, the left plain and the right mirrored. Pure
  tensor ops, so it exports to ONNX with the model."""

  def __init__(self, layout: dict) -> None:
    super().__init__()
    self.sl = {k: tuple(v) for k, v in layout["slices"].items()}
    s7 = torch.tensor(layout["signs"], dtype=torch.float32)
    self.register_buffer("s7", s7)
    self.register_buffer("s8", torch.cat([s7, torch.ones(1)]))
    self.register_buffer("flip", torch.tensor([1.0, -1.0, 1.0]))
    self.register_buffer("yaw_flip", torch.tensor([1.0, -1.0]))
    for k in ("left_cols", "right_cols", "jp_left", "jp_right", "jv_left", "jv_right"):
      self.register_buffer(k, torch.tensor(layout[k], dtype=torch.long))

  def _t(self, x: torch.Tensor, name: str) -> torch.Tensor:
    a, b = self.sl[name]
    return x[:, a:b]

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    t = self._t
    left = torch.cat(
      [
        t(x, "joint_pos")[:, self.jp_left],
        t(x, "joint_vel")[:, self.jv_left],
        t(x, "actions")[:, self.left_cols],
        t(x, "tcp_pos"),
        t(x, "tcp_quat"),
        t(x, "object_pos"),
        t(x, "object_yaw"),
        t(x, "target_pos"),
        t(x, "tcp_pos_right"),
      ],
      dim=-1,
    )
    right = torch.cat(
      [
        t(x, "joint_pos")[:, self.jp_right] * self.s8,
        t(x, "joint_vel")[:, self.jv_right] * self.s7,
        t(x, "actions")[:, self.right_cols] * self.s8,
        t(x, "tcp_pos_right") * self.flip,
        mirror_quat(t(x, "tcp_quat_right")),
        t(x, "object_b_pos") * self.flip,
        t(x, "object_b_yaw") * self.yaw_flip,
        t(x, "right_target") * self.flip,
        t(x, "tcp_pos") * self.flip,
      ],
      dim=-1,
    )
    return left, right

  def join(self, mean_l: torch.Tensor, mean_r: torch.Tensor) -> torch.Tensor:
    out = mean_l.new_zeros(mean_l.shape[0], 16)
    out[:, self.left_cols] = mean_l
    out[:, self.right_cols] = mean_r * self.s8
    return out


class FactoredArmModel(MLPModel):
  """BI-F's actor (module docstring). rsl_rl builds it by name from `FactoredModelCfg`; it is an `MLPModel`
  for everything PPO calls (distribution, log-probs, KL, normalisation), with two MLPs in place of one."""

  def __init__(
    self,
    obs,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims=(512, 256, 128),
    activation: str = "elu",
    obs_normalization: bool = True,
    distribution_cfg: dict | None = None,
    init_lt: str = "",
    init_rst: str = "",
    layout: dict | None = None,
  ) -> None:
    nn.Module.__init__(self)
    if layout is None:
      raise ValueError(
        "FactoredArmModel needs the env's layout (runner.ManipulationRunner fills it)"
      )
    if output_dim != 16 or distribution_cfg is None or not obs_normalization:
      raise ValueError(
        "FactoredArmModel is JOINT's 16-D stochastic actor with normalisation"
      )
    self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
    if self.obs_dim != layout["obs_dim"]:
      raise ValueError(
        f"actor observation {self.obs_dim}-D, layout {layout['obs_dim']}-D"
      )
    self.obs_normalization = True
    self.obs_normalizer = nn.Identity()  # each half normalises its own input
    self.halves = _Halves(layout)
    n_in = ONE_ARM_DIM + OTHER_TCP
    self.norm_left = EmpiricalNormalization(n_in)
    self.norm_right = EmpiricalNormalization(n_in)
    self.mlp_left = MLP(n_in, 8, hidden_dims, activation)
    self.mlp_right = MLP(n_in, 8, hidden_dims, activation)
    dist_class = resolve_callable(distribution_cfg.pop("class_name"))
    self.distribution = dist_class(output_dim, **distribution_cfg)
    self.mlp = nn.Identity()
    self._load_halves(init_lt, init_rst)

  def _load_halves(self, init_lt: str, init_rst: str) -> None:
    """Copy each source actor into its half: the first layer widened by the zero-weighted other-TCP columns,
    the normaliser's statistics (new columns mean 0 / var 1) and count, the std into the half's action
    columns."""
    log_std = (
      self.distribution.log_std_param
      if hasattr(self.distribution, "log_std_param")
      else None
    )
    if log_std is None:
      raise ValueError("FactoredArmModel expects PPO-stack's log-space std")
    for path, mlp, norm, cols in (
      (init_lt, self.mlp_left, self.norm_left, self.halves.left_cols),
      (init_rst, self.mlp_right, self.norm_right, self.halves.right_cols),
    ):
      sd = torch.load(path, map_location="cpu", weights_only=False)["actor_state_dict"]
      w0 = sd["mlp.0.weight"]
      if w0.shape[1] != ONE_ARM_DIM or sd["mlp.6.weight"].shape[0] != 8:
        raise ValueError(f"{path}: not a one-arm 38 -> 8 actor")
      new = mlp.state_dict()
      for k, v in new.items():
        src = sd[f"mlp.{k}"]
        if k == "0.weight":
          src = torch.cat([src, torch.zeros(src.shape[0], OTHER_TCP)], dim=1)
        if src.shape != v.shape:
          raise ValueError(f"{path}: mlp.{k} {tuple(src.shape)} != {tuple(v.shape)}")
        new[k] = src
      mlp.load_state_dict(new)
      for buf, fill in (("_mean", 0.0), ("_var", 1.0), ("_std", 1.0)):
        src = sd[f"obs_normalizer.{buf}"]
        pad = torch.full((1, OTHER_TCP), fill)
        getattr(norm, buf).copy_(torch.cat([src, pad], dim=1))
      norm.count.copy_(sd["obs_normalizer.count"])
      with torch.no_grad():
        log_std[cols] = sd["distribution.log_std_param"].to(log_std)

  def get_latent(self, obs, masks=None, hidden_state=None) -> torch.Tensor:
    return torch.cat([obs[g] for g in self.obs_groups], dim=-1)

  def forward(
    self, obs, masks=None, hidden_state=None, stochastic_output: bool = False
  ) -> torch.Tensor:
    x = self.get_latent(obs, masks, hidden_state)
    mean = self.mean_of(x)
    if stochastic_output:
      self.distribution.update(mean)
      return self.distribution.sample()
    return self.distribution.deterministic_output(mean)

  def mean_of(self, x: torch.Tensor) -> torch.Tensor:
    """The deterministic action for a (B, obs_dim) actor vector. (B, 16)"""
    left, right = self.halves(x)
    return self.halves.join(
      self.mlp_left(self.norm_left(left)), self.mlp_right(self.norm_right(right))
    )

  def update_normalization(self, obs) -> None:
    left, right = self.halves(self.get_latent(obs))
    self.norm_left.update(left)
    self.norm_right.update(right)

  def as_onnx(self, verbose: bool) -> nn.Module:
    return _OnnxFactored(self)

  def as_jit(self) -> nn.Module:
    return _OnnxFactored(self)


class _OnnxFactored(nn.Module):
  is_recurrent: bool = False

  def __init__(self, model: FactoredArmModel) -> None:
    super().__init__()
    self.halves = copy.deepcopy(model.halves)
    self.norm_left = copy.deepcopy(model.norm_left)
    self.norm_right = copy.deepcopy(model.norm_right)
    self.mlp_left = copy.deepcopy(model.mlp_left)
    self.mlp_right = copy.deepcopy(model.mlp_right)
    self.input_size = model.obs_dim

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    left, right = self.halves(x)
    return self.halves.join(
      self.mlp_left(self.norm_left(left)), self.mlp_right(self.norm_right(right))
    )

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]


def _gaussian_kl(mu_t, std_t, mu_s, std_s) -> torch.Tensor:
  """KL(N(mu_t, std_t) || N(mu_s, std_s)) summed over the action dimensions. (B,)"""
  return (
    torch.log(std_s / std_t)
    + (std_t.square() + (mu_t - mu_s).square()) / (2.0 * std_s.square())
    - 0.5
  ).sum(-1)


class JointPPO(PPO):
  """rsl_rl 5.4.2's PPO with plan S11.2's critic warm-up, KL to the SEQ teacher and per-dimension std log."""

  def __init__(
    self,
    actor,
    critic,
    storage,
    kl_coef: float = 0.0,
    kl_anneal_iterations: int = 0,
    critic_warmup_iterations: int = 0,
    action_names: list | None = None,
    **kwargs: Any,
  ) -> None:
    if kwargs.get("rnd_cfg") or kwargs.get("symmetry_cfg"):
      raise ValueError("JointPPO supports neither RND nor symmetry")
    super().__init__(actor, critic, storage, **kwargs)
    if actor.is_recurrent or critic.is_recurrent:
      raise ValueError("JointPPO is for feed-forward models")
    self.kl_coef = float(kl_coef)
    self.kl_anneal_iterations = int(kl_anneal_iterations)
    self.critic_warmup_iterations = int(critic_warmup_iterations)
    self.action_names = list(action_names or [])
    self.updates_done = 0
    self.teacher = None
    if self.kl_coef > 0.0:
      # The actor as initialised: for BI-F that is the SEQ pair exactly.
      self.teacher = copy.deepcopy(self._raw_actor).eval()
      for p in self.teacher.parameters():
        p.requires_grad_(False)

  def kl_weight(self) -> float:
    if self.teacher is None:
      return 0.0
    if self.kl_anneal_iterations <= 0:
      return self.kl_coef
    return self.kl_coef * max(0.0, 1.0 - self.updates_done / self.kl_anneal_iterations)

  def update(self) -> dict[str, float]:
    warm = self.updates_done < self.critic_warmup_iterations
    lam_kl = self.kl_weight()
    mean_value_loss = mean_surrogate_loss = mean_entropy = mean_kl_seq = 0.0
    generator = self.storage.mini_batch_generator(
      self.num_mini_batches, self.num_learning_epochs
    )
    for batch in generator:
      original_batch_size = batch.observations.batch_size[0]
      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          batch.advantages = (batch.advantages - batch.advantages.mean()) / (
            batch.advantages.std() + 1e-8
          )
      self.actor(
        batch.observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[0],
        stochastic_output=True,
      )
      actions_log_prob = self.actor.get_output_log_prob(batch.actions)
      values = self.critic(
        batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1]
      )
      distribution_params = tuple(
        p[:original_batch_size] for p in self.actor.output_distribution_params
      )
      entropy = self.actor.output_entropy[:original_batch_size]

      # The adaptive learning rate reads the actor's KL; while the actor is frozen it would only grow.
      if self.desired_kl is not None and self.schedule == "adaptive" and not warm:
        with torch.inference_mode():
          kl = self.actor.get_kl_divergence(
            batch.old_distribution_params, distribution_params
          )
          kl_mean = torch.mean(kl)
          if kl_mean > self.desired_kl * 2.0:
            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
          elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
          for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

      ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
      surrogate = -torch.squeeze(batch.advantages) * ratio
      surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if self.use_clipped_value_loss:
        value_clipped = batch.values + (values - batch.values).clamp(
          -self.clip_param, self.clip_param
        )
        value_losses = (values - batch.returns).pow(2)
        value_losses_clipped = (value_clipped - batch.returns).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (batch.returns - values).pow(2).mean()

      kl_seq = torch.zeros((), device=self.device)
      if self.teacher is not None:
        with torch.no_grad():
          self.teacher(batch.observations, stochastic_output=True)
          mu_t, std_t = self.teacher.output_mean, self.teacher.output_std
        kl_seq = _gaussian_kl(
          mu_t, std_t, self.actor.output_mean, self.actor.output_std
        ).mean()

      loss = self.value_loss_coef * value_loss
      if not warm:
        loss = (
          loss + surrogate_loss - self.entropy_coef * entropy.mean() + lam_kl * kl_seq
        )

      self.optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy.mean().item()
      mean_kl_seq += float(kl_seq)

    num_updates = self.num_learning_epochs * self.num_mini_batches
    loss_dict = {
      "value": mean_value_loss / num_updates,
      "surrogate": mean_surrogate_loss / num_updates,
      "entropy": mean_entropy / num_updates,
    }
    if self.teacher is not None:
      loss_dict["kl_seq"] = mean_kl_seq / num_updates
      loss_dict["kl_seq_weight"] = lam_kl
      loss_dict["critic_warmup"] = float(warm)
    with torch.no_grad():
      std = self.actor.output_std
      std = std.mean(0) if std.dim() > 1 else std
    for i, v in enumerate(std.tolist()):
      name = self.action_names[i] if i < len(self.action_names) else str(i)
      loss_dict[f"std/{name}"] = v
    self.storage.clear()
    self.updates_done += 1
    return loss_dict
