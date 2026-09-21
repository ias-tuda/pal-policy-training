"""Reset events for tabletop scenes.

Object placement lives here rather than inside a command term. mjlab's ``LiftingCommand``
can teleport its object, but only into an axis-aligned box in world coordinates -- it knows
nothing about a table, and nothing about a second object it must not spawn inside. Since
every task beyond the simplest needs both, placement is an event, which is the manager that
owns reset-time state anyway.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, sample_uniform


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


def hold_joints_at_default(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | Sequence[int] | None,
  asset_cfg: SceneEntityCfg,
) -> None:
  """Point every position actuator at the default pose.

  ``EntityData.clear_state`` zeroes ``joint_pos_target`` on every reset, and the action
  manager only ever writes the joints the policy owns. On a robot whose action space is
  deliberately narrower than ``.*`` -- a humanoid whose legs belong to a separate balance
  controller -- that leaves the remaining actuators commanded to 0 rather than held at
  their default. Kangaroo's telescopic legs duly retracted from 0.60 m to their 0.132 m
  mechanical limit within a second, leaving the fixed base at 0.9 m with the feet dangling
  0.47 m above the floor.

  Targeting *all* joints is deliberate: the action term overwrites its own joints on the
  first step of the episode, so this is simply "everything holds the default pose until
  something asks otherwise".
  """
  asset: Entity = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = slice(None)  # type: ignore[assignment]
  asset.set_joint_position_target(
    asset.data.default_joint_pos[env_ids], env_ids=env_ids
  )
