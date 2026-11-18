"""Scripted upper-body targets for decoupled loco-manipulation training.

The lower-body policy (legs + pelvis) is trained while the arms and jaws are driven by
something it does not control -- here a random smooth target generator, later a
manipulation policy (WC-c). This is the HOMIE-style split: the balance policy has to have
seen arm motion and hand payloads during training, which is what the earlier mobile-grasp
attempt (motionplanning-sampling `mjlab_kangaroo/RESULTS.md`, arms only ever in a carry
pose) did not give it.

`ScriptedArmJointPositionAction` is an action term with ``action_dim == 0``: it takes no
part of the policy's action vector and writes joint position targets for its joints from
its own generator. Each target trajectory is a chain of minimum-jerk segments between
random poses. A pose is drawn uniformly inside a box around the default (home) pose whose
half-width is ``amplitude * level`` per joint, clipped to the soft joint limits; ``level``
ramps linearly from ``level_start`` to 1 over ``level_ramp_steps`` env steps, which is
the curriculum. A fraction of segments are holds (goal == start).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class ScriptedArmJointPositionActionCfg(JointPositionActionCfg):
  """Joint targets for ``actuator_names`` from a random smooth generator, not the policy.

  ``scale`` and ``offset`` of the parent are unused (``action_dim`` is 0); the default
  joint pose is the centre of the sampling box (``use_default_offset`` must stay True).
  """

  amplitude: dict[str, float] = field(default_factory=lambda: {".*": 0.5})
  """Half-width (rad) of the sampling box around the default pose, per actuator regex.
  Every controlled actuator must match exactly one key (checked at build)."""

  segment_duration_s: tuple[float, float] = (1.0, 3.0)
  """Duration range of one minimum-jerk segment."""

  hold_probability: float = 0.3
  """Probability that a segment keeps the current pose (goal == start)."""

  level_start: float = 0.0
  """Curriculum level at env step 0; the box half-width is ``amplitude * level``."""

  level_ramp_steps: int = 0
  """Env steps (``env.common_step_counter``) from ``level_start`` to 1. 0: always 1."""

  binary: dict[str, tuple[float, float]] = field(default_factory=dict)
  """Actuators that switch between two values instead of moving inside a box (the jaw:
  open/closed). Regex -> (value_a, value_b); each segment picks one of the two."""

  def build(self, env: ManagerBasedRlEnv) -> ScriptedArmJointPositionAction:
    return ScriptedArmJointPositionAction(self, env)


class ScriptedArmJointPositionAction(JointPositionAction):
  cfg: ScriptedArmJointPositionActionCfg

  def __init__(self, cfg: ScriptedArmJointPositionActionCfg, env: ManagerBasedRlEnv):
    assert cfg.use_default_offset, "the default pose is the sampling centre"
    super().__init__(cfg=cfg, env=env)
    n = self._num_targets
    # The parent sized its buffers from `action_dim`, which is 0 here; the processed
    # targets are what `apply_actions` writes, so they need the full joint width.
    self._processed_actions = torch.zeros(self.num_envs, n, device=self.device)
    self._center = self._entity.data.default_joint_pos[:, self._target_ids].clone()
    limits = self._entity.data.soft_joint_pos_limits[:, self._target_ids]
    self._lo, self._hi = limits[..., 0].clone(), limits[..., 1].clone()

    self._amplitude = torch.zeros(1, n, device=self.device)
    self._is_binary = torch.zeros(n, dtype=torch.bool, device=self.device)
    self._binary_a = torch.zeros(1, n, device=self.device)
    self._binary_b = torch.zeros(1, n, device=self.device)
    if cfg.binary:
      idx, _, vals = resolve_matching_names_values(cfg.binary, self._target_names)
      self._is_binary[idx] = True
      self._binary_a[0, idx] = torch.tensor([v[0] for v in vals], device=self.device)
      self._binary_b[0, idx] = torch.tensor([v[1] for v in vals], device=self.device)
    box_names = [
      nm for i, nm in enumerate(self._target_names) if not self._is_binary[i]
    ]
    idx_box, _, amps = resolve_matching_names_values(cfg.amplitude, box_names)
    if len(idx_box) != len(box_names):
      raise ValueError(f"amplitude does not cover all of {box_names}")
    box_ids = (~self._is_binary).nonzero().squeeze(-1)
    self._amplitude[0, box_ids[idx_box]] = torch.tensor(amps, device=self.device)

    self._start = self._center.clone()
    self._goal = self._center.clone()
    self._t = torch.zeros(self.num_envs, device=self.device)
    self._duration = torch.ones(self.num_envs, device=self.device)
    self._target = self._center.clone()

  # The policy's action vector carries nothing for this term.

  @property
  def action_dim(self) -> int:
    return 0

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions[:, :0]

  @property
  def level(self) -> float:
    c = self.cfg
    if c.level_ramp_steps <= 0:
      return 1.0
    frac = min(1.0, self._env.common_step_counter / c.level_ramp_steps)
    return c.level_start + (1.0 - c.level_start) * frac

  @property
  def target(self) -> torch.Tensor:
    """Current joint targets, (num_envs, num_targets)."""
    return self._target

  def _new_segment(self, env_ids: torch.Tensor) -> None:
    k = len(env_ids)
    if k == 0:
      return
    c = self.cfg
    self._start[env_ids] = self._target[env_ids]
    u = torch.rand(k, self._num_targets, device=self.device) * 2.0 - 1.0
    goal = self._center[env_ids] + u * self._amplitude * self.level
    pick_a = torch.rand(k, self._num_targets, device=self.device) < 0.5
    binary_goal = torch.where(pick_a, self._binary_a, self._binary_b)
    goal = torch.where(self._is_binary, binary_goal, goal)
    goal = torch.clamp(goal, self._lo[env_ids], self._hi[env_ids])
    hold = torch.rand(k, device=self.device) < c.hold_probability
    goal[hold] = self._start[env_ids][hold]
    self._goal[env_ids] = goal
    lo, hi = c.segment_duration_s
    self._duration[env_ids] = lo + (hi - lo) * torch.rand(k, device=self.device)
    self._t[env_ids] = 0.0

  def process_actions(self, actions: torch.Tensor) -> None:
    del actions  # empty
    self._t += self._env.step_dt
    done = (self._t >= self._duration).nonzero().squeeze(-1)
    if len(done) > 0:
      self._target[done] = self._goal[done]
      self._new_segment(done)
    s = torch.clamp(self._t / self._duration, 0.0, 1.0).unsqueeze(-1)
    s = s**3 * (10.0 - 15.0 * s + 6.0 * s**2)  # minimum jerk
    blend = self._start + (self._goal - self._start) * s
    # Binary joints step at the segment start rather than interpolate.
    self._target = torch.where(self._is_binary, self._goal, blend)
    self._processed_actions = self._target

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None or isinstance(env_ids, slice):
      env_ids = torch.arange(self.num_envs, device=self.device)
    self._target[env_ids] = self._center[env_ids]
    self._processed_actions[env_ids] = self._center[env_ids]
    self._new_segment(env_ids)
