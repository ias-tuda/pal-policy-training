"""Action terms mjlab does not provide.

Ported verbatim from Claudius's `origin/fix-stack-exploration-collapse` (8068e64) by S5.1
item 4 of the pick-and-place plan, where it carries the arms' hardware speed limits.

mjlab's ``JointPositionAction`` turns an action into an absolute joint target in one step:
``default + scale * action``, clipped. For an arm that is fine -- the target moves as far as
the policy asks and the servo follows. For the jaw it is not, because the jaw's whole travel
is 0.95 rad and the policy is free to command either end of it on consecutive control steps.
Nothing on the robot can do that, and nothing in the solver enjoys being asked.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.utils.lab_api.string import resolve_matching_names_values


class RateLimitedJointPositionAction(JointPositionAction):
  """Joint position targets with a per-step slew limit on named joints.

  The target still comes from ``default + scale * action`` and is still absolute, so the
  deployment contract is unchanged: the ONNX emits the same vector and the deployer applies
  it the same way. What changes is that the *commanded* target may only move so far per
  control step, so a saturated action ramps to the stop over several steps instead of
  arriving there in one.
  """

  cfg: RateLimitedJointPositionActionCfg

  def __init__(self, cfg: RateLimitedJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    # inf on every joint the cfg does not name: unnamed joints keep mjlab's behaviour.
    self._rate = torch.full(
      (self.num_envs, self.action_dim), float("inf"), device=self.device
    )
    if cfg.rate_limit:
      index_list, _, value_list = resolve_matching_names_values(
        cfg.rate_limit, self._target_names
      )
      if not index_list:
        raise ValueError(
          f"rate_limit matched no actuator; patterns={list(cfg.rate_limit)}, "
          f"targets={self._target_names}"
        )
      self._rate[:, index_list] = torch.tensor(value_list, device=self.device)
    # The last target actually commanded, which the next one is measured against. Seeded
    # with the default pose, which is what `_offset` holds under `use_default_offset`.
    self._last_target = self._default_target()

  def _default_target(self) -> torch.Tensor:
    if isinstance(self._offset, torch.Tensor):
      return self._offset.clone()
    return torch.full(
      (self.num_envs, self.action_dim), float(self._offset), device=self.device
    )

  def process_actions(self, actions: torch.Tensor) -> None:
    super().process_actions(actions)  # scale, offset, then clip
    delta = (self._processed_actions - self._last_target).clamp(-self._rate, self._rate)
    self._processed_actions = self._last_target + delta
    self._last_target = self._processed_actions.clone()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    # The episode restarts from the default pose, so the slew reference must too --
    # otherwise the first step of a new episode is measured against the last step of the
    # old one and the limit is silently skipped exactly where the state jumps most.
    if env_ids is None:
      env_ids = slice(None)
    self._last_target[env_ids] = self._default_target()[env_ids]


@dataclass(kw_only=True)
class RateLimitedJointPositionActionCfg(JointPositionActionCfg):
  rate_limit: dict[str, float] | None = None
  """Max change in commanded target per control step, in rad, keyed by actuator regex.

  Joints not matched are unlimited. This is a property of what the hardware can execute, not
  a tuning knob: a target the real servo cannot reach within one control period is a target
  that only ever existed in the solver.
  """

  def build(self, env: ManagerBasedRlEnv) -> RateLimitedJointPositionAction:
    return RateLimitedJointPositionAction(self, env)
