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
from mjlab.managers.scene_entity_config import SceneEntityCfg
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


class SurfaceOffsetCommand(LiftingCommand):
  """A goal at a point on a surface, with the surface read out of physics every step.

  :class:`EntityOffsetCommand` above cannot do this. It reads the reference's
  ``root_link_pos_w``, and for a fixed-base entity -- which a table is -- that is the pose
  the entity was *spawned* at, not where its body currently is. Randomise the table height
  with ``dr.body_pos`` and ``geom_xpos`` moves while ``root_link_pos_w`` stays at 0.74, so
  the goal silently stops being on the table: a 2 cm offset against a 2.5 cm success
  threshold leaves nothing for the placement error itself.

  This reads the same two quantities ``mdp.events.reset_objects_on_surface`` does -- the
  surface geom's world XY and the height of its top face -- so the goal and the cube's
  spawn agree about where the table is by construction.

  With a nonzero ``offset_range`` the offset is drawn per env from a box around ``offset``
  each time the command resamples -- once per episode for a pick-and-place goal -- so the goal
  is a random point on the surface that stays fixed relative to it (plan S7-2 baseline B).
  """

  cfg: SurfaceOffsetCommandCfg

  def __init__(self, cfg: SurfaceOffsetCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._surface_cfg = SceneEntityCfg(cfg.surface_name)
    self._offset = torch.tensor(cfg.offset, device=self.device, dtype=torch.float32)
    self._half_range = torch.tensor(
      cfg.offset_range, device=self.device, dtype=torch.float32
    )
    #: This episode's offset from the surface centre, per env. (B, 2)
    self.offsets = self._offset.repeat(self.num_envs, 1)

  def _goal_on_surface(self) -> torch.Tensor:
    from pal_mjlab.tasks.manipulation.mdp.events import (
      _surface_centre_xy,
      surface_top_z,
    )

    centre_xy = _surface_centre_xy(self._env, self._surface_cfg)
    top_z = surface_top_z(self._env, self._surface_cfg)
    half_z = self._env.sim.model.geom_size[:, self.object.indexing.geom_ids[0], 2]
    return torch.cat([centre_xy + self.offsets, (top_z + half_z).unsqueeze(-1)], dim=-1)

  def set_offsets(self, env_ids: torch.Tensor, offsets: torch.Tensor) -> None:
    """Put the goal of ``env_ids`` at ``offsets`` (N, 2) from the surface centre, for the rest
    of their episode. For scripted tests that need a goal at a chosen point of the box."""
    self.offsets[env_ids] = offsets.to(self.offsets)
    self.target_pos[env_ids] = self._goal_on_surface()[env_ids]

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    noise = (2.0 * torch.rand(len(env_ids), 2, device=self.device) - 1.0) * (
      self._half_range
    )
    self.offsets[env_ids] = self._offset + noise
    self.target_pos[env_ids] = self._goal_on_surface()[env_ids]

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    super()._update_command(env_ids)
    self.target_pos = self._goal_on_surface()


@dataclass(kw_only=True)
class SurfaceOffsetCommandCfg(LiftingCommandCfg):
  surface_name: str
  """Scene entity whose first geom is the surface. Its top face sets the goal height."""

  offset: tuple[float, float] = (0.0, 0.0)
  """XY offset from the surface geom's centre. The height is not an input -- it is the
  surface plus the object's own half-height, which is what "resting on it" means."""

  offset_range: tuple[float, float] = (0.0, 0.0)
  """Half-widths in x and y of the box the offset is drawn from at each resample, centred on
  ``offset``. Zero, the default, is a fixed point."""

  # Placement belongs to a reset event, not to the command -- as for `EntityOffsetCommandCfg`.
  # Without this the inherited default is mjlab's own spawn range (x 0.30-0.35, y +/-0.10,
  # z 0.02-0.05 in the env frame), and `LiftingCommand._resample_command` re-placed the object
  # there *after* the reset event had put it on the table: at DR-scene, the only pick-and-place
  # rung that uses this command, every episode started with the cube on the floor. Found
  # 2026-09-24 (plan S7, D5 alignment); DR-scene had never been trained.
  object_pose_range: LiftingCommandCfg.ObjectPoseRangeCfg | None = None

  def build(self, env: ManagerBasedRlEnv) -> SurfaceOffsetCommand:
    return SurfaceOffsetCommand(self, env)


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
