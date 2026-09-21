"""Goal commands that mjlab does not provide.

mjlab's :class:`LiftingCommand` samples a goal in a fixed box. That covers "move the object
somewhere", but not the much more common tabletop shape: "put the object *on*, *in* or
*through* that other thing". Stacking, rack-hanging and insertion all need a goal defined
relative to another entity, and one that keeps tracking it if the robot nudges it.

:class:`EntityOffsetCommand` is that goal, and it is deliberately the only command this
package adds. It subclasses mjlab's ``LiftingCommand`` so that every mjlab manipulation term
(``staged_position_reward``, ``bring_object_reward``, ``object_to_goal_distance``,
``target_position``, ...) accepts it -- those terms ``isinstance``-check against
``LiftingCommand``, so subclassing is what keeps the native library usable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from mjlab.entity import Entity
from mjlab.tasks.manipulation.mdp.commands import LiftingCommand, LiftingCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class EntityOffsetCommand(LiftingCommand):
  """A goal pinned to another entity's pose, plus an offset.

  The offset's z component can be resolved from geometry rather than hardcoded: with
  ``stack_on_top``, the goal is the height at which the commanded object rests on the
  reference object, computed from both half-heights. That keeps one command correct across
  cube sizes and across randomisation.

  Object placement is *not* done here -- construct with ``object_pose_range=None`` and let
  a reset event own it (see ``mdp.events.reset_objects_on_surface``). A command that also
  teleports objects is a command that every task has to subclass.
  """

  cfg: EntityOffsetCommandCfg

  def __init__(self, cfg: EntityOffsetCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.reference: Entity = env.scene[cfg.reference_name]
    self._offset = torch.tensor(cfg.offset, device=self.device, dtype=torch.float32)

  @property
  def _stacking_height(self) -> torch.Tensor:
    """Half-height of the reference plus half-height of the object, per env."""
    geom_size = self._env.sim.model.geom_size
    ref_half_z = geom_size[:, self.reference.indexing.geom_ids[0], 2]
    obj_half_z = geom_size[:, self.object.indexing.geom_ids[0], 2]
    return ref_half_z + obj_half_z

  def _goal_from_reference(self) -> torch.Tensor:
    goal = self.reference.data.root_link_pos_w + self._offset
    if self.cfg.stack_on_top:
      goal = goal.clone()
      goal[:, 2] = self.reference.data.root_link_pos_w[:, 2] + self._stacking_height
    return goal

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    self.target_pos[env_ids] = self._goal_from_reference()[env_ids]

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    super()._update_command(env_ids)
    # Recomputed every step, so the goal follows the reference if it gets knocked.
    self.target_pos = self._goal_from_reference()


@dataclass(kw_only=True)
class EntityOffsetCommandCfg(LiftingCommandCfg):
  reference_name: str
  """Scene entity the goal is defined relative to."""

  offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
  """Offset from the reference's origin, in world axes."""

  stack_on_top: bool = False
  """Replace the offset's z with the resting height of the object on the reference."""

  # Placement belongs to a reset event, not to the command.
  object_pose_range: LiftingCommandCfg.ObjectPoseRangeCfg | None = None
  difficulty: Literal["fixed", "dynamic"] = "dynamic"

  def build(self, env: ManagerBasedRlEnv) -> EntityOffsetCommand:
    return EntityOffsetCommand(self, env)
