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

from pal_mjlab.tasks.manipulation.mdp.shapes import object_half_height

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
    #: The reference's world xy when the episode's goal was drawn (plan S11's `base_moved`). (B, 2)
    self.reference_start_xy = torch.zeros(self.num_envs, 2, device=self.device)

  def _reference_xy(self) -> torch.Tensor:
    """The reference's world xy from its free joint's ``qpos`` -- what a reset event wrote, valid
    before the next ``forward()``, when the derived pose is still the last episode's. (B, 2)"""
    adr = self.reference.data.indexing.free_joint_q_adr[:2]
    return self.reference.data.data.qpos[:, adr]

  def reference_displacement(self) -> torch.Tensor:
    """Horizontal distance the reference has moved since the goal was drawn, m. (B,)"""
    return torch.norm(self._reference_xy() - self.reference_start_xy, dim=-1)

  @property
  def _stacking_height(self) -> torch.Tensor:
    """Half-height of the reference plus half-height of the object, per env."""
    ref_half_z = object_half_height(self._env, self.cfg.reference_name)
    obj_half_z = object_half_height(self._env, self.cfg.entity_name)
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
    self.reference_start_xy[env_ids] = self._reference_xy()[env_ids]

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
  spawn agree about where the table is by construction. The goal's height is the surface plus
  the object's own half-height (a ball's radius, plan S8: `shapes.object_half_height`).

  With a nonzero ``offset_range`` the offset is drawn per env from a box around ``offset``
  each time the command resamples -- once per episode for a pick-and-place goal -- so the goal
  is a random point on the surface that stays fixed relative to it (plan S7-2 baseline B).

  With ``min_object_distance`` > 0 (plan S7-8's ``TARGET=WC``) the box is the area the cube was
  also drawn from, and the goal is redrawn until it is at least that far from where the cube
  now sits: ``reset_objects_on_surface`` runs before the command manager's reset, and the cube's
  pose is read from its free joint's ``qpos``, which the event wrote (the derived ``xpos`` is
  stale until the next ``forward()``). The distance is logged as ``carry_distance``.
  """

  cfg: SurfaceOffsetCommandCfg

  #: Rounds of redrawing for ``min_object_distance``; each fails with probability <= 0.53 for
  #: plan S7-8's area and 10 cm, so 32 rounds leave ~1e-9 of the envs to the corner fallback.
  MAX_REDRAWS = 32

  def __init__(self, cfg: SurfaceOffsetCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    if cfg.frame not in ("surface", "env"):
      raise ValueError(f"unknown frame {cfg.frame!r}; have ('surface', 'env')")
    if cfg.frame == "env" and cfg.min_object_distance > 0.0:
      raise ValueError(
        "min_object_distance is drawn about the surface centre; frame='env'"
      )
    self._surface_cfg = SceneEntityCfg(cfg.surface_name)
    self._offset = torch.tensor(cfg.offset, device=self.device, dtype=torch.float32)
    self._half_range = torch.tensor(
      cfg.offset_range, device=self.device, dtype=torch.float32
    )
    #: This episode's offset from the surface centre, per env. (B, 2)
    self.offsets = self._offset.repeat(self.num_envs, 1)
    if cfg.min_object_distance > 0.0:
      #: Plan S7-8: the goal's distance from the cube's spawn this episode, m, set at the resample.
      self.metrics["carry_distance"] = torch.zeros(self.num_envs, device=self.device)

  def _goal_on_surface(self) -> torch.Tensor:
    from pal_mjlab.tasks.manipulation.mdp.events import (
      _surface_centre_xy,
      surface_top_z,
    )

    centre_xy = (
      _surface_centre_xy(self._env, self._surface_cfg)
      if self.cfg.frame == "surface"
      else self._env.scene.env_origins[:, :2]
    )
    top_z = surface_top_z(self._env, self._surface_cfg)
    half_z = object_half_height(self._env, self.cfg.entity_name)
    return torch.cat([centre_xy + self.offsets, (top_z + half_z).unsqueeze(-1)], dim=-1)

  def set_offsets(self, env_ids: torch.Tensor, offsets: torch.Tensor) -> None:
    """Put the goal of ``env_ids`` at ``offsets`` (N, 2) from the surface centre, for the rest
    of their episode. For scripted tests that need a goal at a chosen point of the box."""
    self.offsets[env_ids] = offsets.to(self.offsets)
    self.target_pos[env_ids] = self._goal_on_surface()[env_ids]

  def _draw(self, n: int) -> torch.Tensor:
    """``n`` offsets, uniform in the box about ``offset``. (n, 2)"""
    noise = (2.0 * torch.rand(n, 2, device=self.device) - 1.0) * self._half_range
    return self._offset + noise

  def _object_xy(self, env_ids: torch.Tensor) -> torch.Tensor:
    """The cube's world xy, from the free joint's ``qpos`` -- what a reset event wrote, valid
    before the next ``forward()``. (N, 2)"""
    adr = self.object.data.indexing.free_joint_q_adr[:2]
    return self.object.data.data.qpos[:, adr][env_ids]

  def _draw_clear_of_object(self, env_ids: torch.Tensor) -> torch.Tensor:
    """Offsets for ``env_ids`` at least ``min_object_distance`` from the cube, by vectorised
    rejection (each round redraws only the envs still too close); the few left after
    ``MAX_REDRAWS`` rounds take the box's corner farthest from the cube, which is at least
    ``min_object_distance`` away when the cube was drawn where one exists (``goal_area`` of
    `reset_objects_on_surface`). Always terminates, never loops per env. (N, 2)"""
    from pal_mjlab.tasks.manipulation.mdp.events import _surface_centre_xy

    n = len(env_ids)
    centre = _surface_centre_xy(self._env, self._surface_cfg)[env_ids]
    cube = self._object_xy(env_ids)
    dist = self.cfg.min_object_distance
    offsets = self._draw(n)
    for _ in range(self.MAX_REDRAWS):
      too_close = torch.norm(centre + offsets - cube, dim=-1) < dist
      if not bool(too_close.any()):
        return offsets
      offsets = torch.where(too_close.unsqueeze(-1), self._draw(n), offsets)
    too_close = torch.norm(centre + offsets - cube, dim=-1) < dist
    signs = torch.tensor(
      [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]], device=self.device
    )
    corners = self._offset + signs * self._half_range  # (4, 2)
    far = torch.norm(centre.unsqueeze(1) + corners - cube.unsqueeze(1), dim=-1).argmax(
      -1
    )
    return torch.where(too_close.unsqueeze(-1), corners[far], offsets)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    if self.cfg.min_object_distance > 0.0:
      offsets = self._draw_clear_of_object(env_ids)
    else:
      offsets = self._draw(len(env_ids))
    self.offsets[env_ids] = offsets
    self.target_pos[env_ids] = self._goal_on_surface()[env_ids]
    if self.cfg.min_object_distance > 0.0:
      self.metrics["carry_distance"][env_ids] = torch.norm(
        self.target_pos[env_ids, :2] - self._object_xy(env_ids), dim=-1
      )

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

  min_object_distance: float = 0.0
  """Redraw the offset until the goal is at least this far (xy, m) from the cube's current
  position -- plan S7-8's ``TARGET=WC``, where the goal is drawn from the area the cube spawned
  in. Zero, the default, draws once. Needs the cube already placed when the command resamples,
  which the reset events guarantee."""

  frame: str = "surface"
  """What ``offset`` is from: the surface geom's centre (``surface``, every task before plan S11),
  or the env origin (``env``): plan S11's meeting point T, fixed in the robot frame while
  DR-scene shifts the table under it. The height is the surface's either way."""

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

  yaw_reference: str | None = None
  """Plan S9.1's SC-Y: the entity M6's yaw grade measures the object's yaw against (the base, which
  stands at +/- 45 deg there), read by the release latch (`observations._release_latch`) as
  S10.1's `BoxGoalCommandCfg.yaw_reference`; None measures it against the world's axes."""

  # Placement belongs to a reset event, not to the command.
  object_pose_range: LiftingCommandCfg.ObjectPoseRangeCfg | None = None
  difficulty: Literal["fixed", "dynamic"] = "dynamic"

  def build(self, env: ManagerBasedRlEnv) -> EntityOffsetCommand:
    return EntityOffsetCommand(self, env)
