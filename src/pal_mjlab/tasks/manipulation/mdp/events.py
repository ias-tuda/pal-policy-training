"""Reset events for tabletop scenes.

Object placement lives here rather than inside a command term. mjlab's ``LiftingCommand``
can teleport its object, but only into an axis-aligned box in world coordinates -- it knows
nothing about a table, and nothing about a second object it must not spawn inside. Since
every task beyond the simplest needs both, placement is an event, which is the manager that
owns reset-time state anyway.

The second event here answers the opposite question -- what happens to the joints the policy
does *not* drive -- and is just as reset-time a concern.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, sample_uniform

from pal_mjlab.tasks.manipulation.mdp.observations import (
  _RELEASE,
  _WAS_LIFTED,
  latch_attrs,
)
from pal_mjlab.tasks.manipulation.mdp.shapes import object_half_height

#: Same module-level singleton the observation terms use: a SceneEntityCfg built in an
#: argument default is evaluated once at import and shared, which ruff flags as B008.
_ROBOT = SceneEntityCfg("robot")


def surface_top_z(env: ManagerBasedRlEnv, surface_cfg: SceneEntityCfg) -> torch.Tensor:
  """World-frame z of a surface entity's top face, per env.

  Read out of physics data rather than from a config constant, so that height
  randomisation, mocap wrapping and env-origin offsets are all picked up for free.

  The surface is the entity's first geom, unless ``surface_cfg`` names geoms: then it is the
  highest top among them -- plan S10's box rim, its four walls (`scene.box_wall_geom_names`),
  which BX-DR makes per world. Upright box geoms only: the half-size along z is read as the
  half-height (the box is yawed, never tilted, at a reset).
  """
  surface: Entity = env.scene[surface_cfg.name]
  if surface_cfg.geom_names is not None:
    ids = surface.indexing.geom_ids[surface_cfg.geom_ids]
    tops = env.sim.data.geom_xpos[:, ids, 2] + env.sim.model.geom_size[:, ids, 2]
    return tops.amax(dim=-1)
  geom_id = surface.indexing.geom_ids[0]
  centre_z = env.sim.data.geom_xpos[:, geom_id, 2]
  half_z = env.sim.model.geom_size[:, geom_id, 2]
  return centre_z + half_z


def _surface_centre_xy(
  env: ManagerBasedRlEnv, surface_cfg: SceneEntityCfg
) -> torch.Tensor:
  surface: Entity = env.scene[surface_cfg.name]
  geom_id = surface.indexing.geom_ids[0]
  return env.sim.data.geom_xpos[:, geom_id, 0:2]


def hold_nominal_targets(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> None:
  """Command every joint to its configured default, not to zero.

  ``Entity.clear_state`` zeroes ``joint_pos_target`` on every reset
  (``mjlab/entity/data.py:238``), and only the joints named by an action term get a target
  written back: everything else is commanded to 0 for the whole episode. On a humanoid with
  the base welded that is not a small error. Measured on the Kangaroo grippers variant,
  without this event: ``leg_*_length_joint`` runs from its nominal 0.600 m to 0.128 m -- its
  mechanical minimum -- in 0.2 s, the feet leave the floor after the first control step, and
  ``pelvis_1_joint`` sits on its hard limit for the rest of the episode.

  Runs after ``Entity.clear_state`` because the event manager runs after ``scene.reset``
  (``mjlab/envs/manager_based_rl_env.py:580-590``). Writes every joint, including the ones
  the policy drives: the action term overwrites those on the first step anyway, and naming
  them here would duplicate the action regex in a second place.
  """
  asset: Entity = env.scene[asset_cfg.name]
  asset.set_joint_position_target(
    asset.data.default_joint_pos[env_ids], env_ids=env_ids
  )


#: What a spawn range is relative to (`reset_objects_on_surface`'s ``frame``): the surface's centre
#: (every task before plan S11) or the env origin, the robot frame (plan S11's T).
SPAWN_FRAMES = ("surface", "env")


def reset_objects_on_surface(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  asset_cfgs: Sequence[SceneEntityCfg],
  surface_cfg: SceneEntityCfg,
  pose_range: dict[str, tuple[float, float]],
  min_separation: float = 0.0,
  max_attempts: int = 10,
  goal_area: dict | None = None,
  frame: str = "surface",
  keepout: tuple[float, float, float, float] | None = None,
) -> None:
  """Place free objects upright on a surface, clear of one another.

  Each object is dropped exactly onto the surface (its own half-height above the measured
  top face -- a box's half-z, a ball's radius, `shapes.object_half_height`), at an XY offset sampled from ``pose_range`` relative to the surface centre,
  with a yaw from ``pose_range["yaw"]``.

  ``min_separation`` is enforced by vectorised rejection: up to ``max_attempts`` rounds,
  each resampling only the envs that are still too close to an already-placed object. Any
  env still conflicting afterwards is pushed apart deterministically, so the function always
  terminates with a valid layout and never loops per-env in Python.

  ``goal_area`` (plan S7-8's ``TARGET=WC`` / ``WCK``) is ``{"x": (lo, hi), "y": (lo, hi),
  "distance": d}``, the area the goal will be drawn from, relative to the surface centre like
  ``pose_range``, and the distance it must keep from the object. The object is drawn only where
  the goal area has a point at least ``d`` away -- the farthest of its four corners is -- by the same
  vectorised rejection, and a straggler goes to a random corner of the goal area. Without it a cube
  near the centre of a small area has no valid goal at all (12 x 10 cm: every point is within
  7.8 cm of the centre), and the command would have to break the distance it promises.

  ``frame="env"`` (plan S11's T, `pick_place_env_cfg.ROBOT_TARGETS`) reads ``pose_range``'s x and y
  about the env origin -- the robot frame, ``base_link`` at the origin -- instead of the surface's
  centre: the object lands at the same place relative to the robot wherever DR-scene has put the
  table. Its height is still the surface's top.

  ``keepout`` (plan §12) is a box (x lo, x hi, y lo, y hi) about the env origin -- the robot frame -- the
  objects' centres must stay out of, whatever DR-scene's table shift: under the start pose's fingers, which
  hang below a standing pin's top. Redrawn by the same rejection (with the goal area's test, if any); a
  straggler after 64 rounds stays where it is.
  """
  if not asset_cfgs:
    return
  if frame not in SPAWN_FRAMES:
    raise ValueError(f"unknown frame {frame!r}; have {SPAWN_FRAMES}")

  n = len(env_ids)
  device = env.device
  top_z = surface_top_z(env, surface_cfg)[env_ids]
  centre_xy = (
    _surface_centre_xy(env, surface_cfg)[env_ids]
    if frame == "surface"
    else env.scene.env_origins[env_ids, :2]
  )

  x_range = pose_range.get("x", (0.0, 0.0))
  y_range = pose_range.get("y", (0.0, 0.0))
  yaw_range = pose_range.get("yaw", (0.0, 0.0))
  lower = torch.tensor([x_range[0], y_range[0]], device=device)
  upper = torch.tensor([x_range[1], y_range[1]], device=device)

  placed: list[torch.Tensor] = []  # XY of objects already positioned, each (n, 2).

  for asset_cfg in asset_cfgs:
    asset: Entity = env.scene[asset_cfg.name]

    xy = sample_uniform(lower, upper, (n, 2), device=device) + centre_xy
    if placed and min_separation > 0.0:
      for _ in range(max_attempts):
        too_close = torch.zeros(n, dtype=torch.bool, device=device)
        for other in placed:
          too_close |= torch.norm(xy - other, dim=-1) < min_separation
        if not bool(too_close.any()):
          break
        resampled = sample_uniform(lower, upper, (n, 2), device=device) + centre_xy
        xy = torch.where(too_close.unsqueeze(-1), resampled, xy)
      else:
        # Deterministic fallback: push the stragglers directly away from the nearest
        # already-placed object until they clear it.
        for other in placed:
          delta = xy - other
          dist = torch.norm(delta, dim=-1, keepdim=True).clamp_min(1e-6)
          direction = torch.where(
            dist > 1e-4, delta / dist, torch.tensor([[1.0, 0.0]], device=device)
          )
          needs_push = (dist.squeeze(-1) < min_separation).unsqueeze(-1)
          xy = torch.where(needs_push, other + direction * min_separation, xy)

    if goal_area is not None:
      xy = _clear_of_goal_area(
        xy, centre_xy, goal_area, lower, upper, n, device, max_attempts
      )
    if keepout is not None:
      origin = env.scene.env_origins[env_ids, :2]
      for _ in range(64):
        bad = in_keepout(xy - origin, keepout)
        if goal_area is not None:
          bad |= _goal_area_reach(xy, centre_xy, goal_area, device) < float(
            goal_area["distance"]
          )
        if not bool(bad.any()):
          break
        redraw = sample_uniform(lower, upper, (n, 2), device=device) + centre_xy
        xy = torch.where(bad.unsqueeze(-1), redraw, xy)
    placed.append(xy)

    half_z = object_half_height(env, asset_cfg.name, env_ids)
    pos = torch.cat([xy, (top_z + half_z).unsqueeze(-1)], dim=-1)

    zeros = torch.zeros(n, device=device)
    yaw = sample_uniform(yaw_range[0], yaw_range[1], (n,), device=device)
    quat = quat_from_euler_xyz(zeros, zeros, yaw)

    asset.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    asset.write_root_link_velocity_to_sim(
      torch.zeros(n, 6, device=device), env_ids=env_ids
    )


def in_keepout(
  xy: torch.Tensor, keepout: tuple[float, float, float, float]
) -> torch.Tensor:
  """Whether each point (n, 2) is inside the box (x lo, x hi, y lo, y hi), same frame. (n,)"""
  x0, x1, y0, y1 = keepout
  return (xy[:, 0] > x0) & (xy[:, 0] < x1) & (xy[:, 1] > y0) & (xy[:, 1] < y1)


def _goal_area_reach(
  xy: torch.Tensor, centre_xy: torch.Tensor, goal_area: dict, device
) -> torch.Tensor:
  """Distance from each point (n, 2, world) to the goal area's farthest corner. (n,)"""
  (x0, x1), (y0, y1) = goal_area["x"], goal_area["y"]
  corners = torch.tensor([[x0, y0], [x0, y1], [x1, y0], [x1, y1]], device=device)
  return torch.norm((xy - centre_xy).unsqueeze(1) - corners, dim=-1).amax(dim=-1)


def _clear_of_goal_area(
  xy: torch.Tensor,
  centre_xy: torch.Tensor,
  goal_area: dict,
  lower: torch.Tensor,
  upper: torch.Tensor,
  n: int,
  device,
  max_attempts: int,
) -> torch.Tensor:
  """``xy`` (n, 2, world) redrawn, where it has no goal-area point at least
  ``goal_area["distance"]`` away, from the same box; see `reset_objects_on_surface`."""
  (x0, x1), (y0, y1) = goal_area["x"], goal_area["y"]
  corners = torch.tensor([[x0, y0], [x0, y1], [x1, y0], [x1, y1]], device=device)
  distance = float(goal_area["distance"])

  def reach(p: torch.Tensor) -> torch.Tensor:  # distance to the farthest corner (n,)
    return torch.norm((p - centre_xy).unsqueeze(1) - corners, dim=-1).amax(dim=-1)

  # A big enough margin over the worst-case draw: each round fails for at most the share of the
  # box that lies within `distance` of all four corners, well under one half for the areas used.
  for _ in range(max(max_attempts, 32)):
    bad = reach(xy) < distance
    if not bool(bad.any()):
      return xy
    redraw = sample_uniform(lower, upper, (n, 2), device=device) + centre_xy
    xy = torch.where(bad.unsqueeze(-1), redraw, xy)
  bad = reach(xy) < distance
  pick = torch.randint(0, 4, (n,), device=device)
  return torch.where(bad.unsqueeze(-1), corners[pick] + centre_xy, xy)


def reset_was_lifted(
  env: ManagerBasedRlEnv, env_ids: Sequence[int] | torch.Tensor
) -> None:
  """Clear the `was_lifted` latch for the environments being reset.

  Without this the latch is set once and never falls, so every episode after the first in a
  given world would count a slide as a pick.
  """
  for attr in latch_attrs(env, _WAS_LIFTED):
    getattr(env, attr)[env_ids] = False


def reset_release_latch(
  env: ManagerBasedRlEnv, env_ids: Sequence[int] | torch.Tensor
) -> None:
  """Clear the release latch (plan S7 M6, S7-2 M7) for the environments being reset.

  Without this the last episode's release would grade the next one's. The remembered grasp
  state is cleared too: an episode that ended with the object in the jaw would otherwise see
  the grasp "open" on the next one's first step. So is M7's set-down state, or a dwell built
  at the end of one episode would count towards the first release of the next.
  """
  for attr in latch_attrs(env, _RELEASE):
    _clear_release_latch(getattr(env, attr), env_ids)


def _clear_release_latch(latch, env_ids: Sequence[int] | torch.Tensor) -> None:
  if latch is not None:
    latch.grasped[env_ids] = False
    latch.released[env_ids] = False
    latch.dxy[env_ids] = 0.0
    latch.height[env_ids] = 0.0
    latch.clear[env_ids] = False
    latch.touchdown_ok[env_ids] = False
    latch.dwell[env_ids] = 0
    latch.rel_on_table[env_ids] = False
    latch.rel_speed[env_ids] = 0.0
    latch.rel_dwell[env_ids] = 0
    latch.rel_ok[env_ids] = False
    latch.pend_ok[env_ids] = False
    if latch.tilt is not None:
      latch.tilt[env_ids] = 0.0
    if latch.yaw is not None:
      latch.yaw[env_ids] = 0.0
    # Plan S7-4's J state, when the latch defines a release by the jaw: a jaw closed at the
    # end of one episode must not "open" on the next one's first step, nor a let-go pending.
    if latch.jaw is not None:
      latch.jaw_closed[env_ids] = False
      latch.pend[env_ids] = False
      if latch.pend_yaw is not None:
        latch.pend_yaw[env_ids] = 0.0
