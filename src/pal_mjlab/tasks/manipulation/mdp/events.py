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

from pal_mjlab.tasks.manipulation.mdp.observations import _RELEASE, _WAS_LIFTED

#: Same module-level singleton the observation terms use: a SceneEntityCfg built in an
#: argument default is evaluated once at import and shared, which ruff flags as B008.
_ROBOT = SceneEntityCfg("robot")


def surface_top_z(env: ManagerBasedRlEnv, surface_cfg: SceneEntityCfg) -> torch.Tensor:
  """World-frame z of a surface entity's top face, per env.

  Read out of physics data rather than from a config constant, so that height
  randomisation, mocap wrapping and env-origin offsets are all picked up for free.
  """
  surface: Entity = env.scene[surface_cfg.name]
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


def reset_objects_on_surface(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  asset_cfgs: Sequence[SceneEntityCfg],
  surface_cfg: SceneEntityCfg,
  pose_range: dict[str, tuple[float, float]],
  min_separation: float = 0.0,
  max_attempts: int = 10,
) -> None:
  """Place free objects upright on a surface, clear of one another.

  Each object is dropped exactly onto the surface (its own half-height above the measured
  top face), at an XY offset sampled from ``pose_range`` relative to the surface centre,
  with a yaw from ``pose_range["yaw"]``.

  ``min_separation`` is enforced by vectorised rejection: up to ``max_attempts`` rounds,
  each resampling only the envs that are still too close to an already-placed object. Any
  env still conflicting afterwards is pushed apart deterministically, so the function always
  terminates with a valid layout and never loops per-env in Python.
  """
  if not asset_cfgs:
    return

  n = len(env_ids)
  device = env.device
  top_z = surface_top_z(env, surface_cfg)[env_ids]
  centre_xy = _surface_centre_xy(env, surface_cfg)[env_ids]

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

    placed.append(xy)

    half_z = env.sim.model.geom_size[env_ids, asset.indexing.geom_ids[0], 2]
    pos = torch.cat([xy, (top_z + half_z).unsqueeze(-1)], dim=-1)

    zeros = torch.zeros(n, device=device)
    yaw = sample_uniform(yaw_range[0], yaw_range[1], (n,), device=device)
    quat = quat_from_euler_xyz(zeros, zeros, yaw)

    asset.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    asset.write_root_link_velocity_to_sim(
      torch.zeros(n, 6, device=device), env_ids=env_ids
    )


def reset_was_lifted(
  env: ManagerBasedRlEnv, env_ids: Sequence[int] | torch.Tensor
) -> None:
  """Clear the `was_lifted` latch for the environments being reset.

  Without this the latch is set once and never falls, so every episode after the first in a
  given world would count a slide as a pick.
  """
  latch = getattr(env, _WAS_LIFTED, None)
  if latch is not None:
    latch[env_ids] = False


def reset_release_latch(
  env: ManagerBasedRlEnv, env_ids: Sequence[int] | torch.Tensor
) -> None:
  """Clear the release latch (plan S7, module M6) for the environments being reset.

  Without this the last episode's release would grade the next one's. The remembered grasp
  state is cleared too: an episode that ended with the object in the jaw would otherwise see
  the grasp "open" on the next one's first step.
  """
  latch = getattr(env, _RELEASE, None)
  if latch is not None:
    latch.grasped[env_ids] = False
    latch.released[env_ids] = False
    latch.dxy[env_ids] = 0.0
    latch.height[env_ids] = 0.0
