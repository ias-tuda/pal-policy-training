"""What shape the manipulated object is (plan S8: a 6 cm ball beside the 5 cm cube).

Everything that used to read ``geom_size[:, geom_ids[0], 2]`` as "the object's half-height" assumed
a box: a sphere stores its radius in ``geom_size[..., 0]`` and zeros after it, so the same read
puts a ball's goal and resting height at the table's surface. `object_half_height` is the one
place that knows the difference.

An object is one body with one geom per shape it can take: a cube's box, a ball's sphere, or both
(plan S8's MIX, `scene.get_mixed_spec`). ``geom_type`` is one per model in mujoco_warp, not per
world, so a world switches a shape *off* by shrinking it to `scene.INACTIVE_SIZE`, at the body's
centre and inside the shape that is on, where nothing can touch it. The half-height of the object
is then the largest half-height over its geoms, and "is it a ball" is whether the sphere is the
large one (`ball_mask`). Nothing else needs to be told which shape a world has.

`PerShape` is a number that depends on the shape -- plan S7-4's J threshold, which has to be
recalibrated for an object that the jaw touches at another angle (plan S8, L7).
"""

from __future__ import annotations

import dataclasses

import mujoco
import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.dr.geom import _recompute_geom_bounds
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

from pal_mjlab.tasks.manipulation.scene import INACTIVE_SIZE

_SPHERE = int(mujoco.mjtGeom.mjGEOM_SPHERE)
_BOX = int(mujoco.mjtGeom.mjGEOM_BOX)

#: Attribute on the env caching each object's (geom id, geom type) pairs.
_GEOMS = "_pal_shape_geoms"


@dataclasses.dataclass(frozen=True)
class PerShape:
  """A value that depends on the object's shape, per env (plan S8's MIX)."""

  cube: float
  ball: float


def _object_geoms(env: ManagerBasedRlEnv, object_name: str) -> list[tuple[int, int]]:
  """``(global geom id, mujoco geom type)`` of every geom of the object, in model order, cached."""
  cache = getattr(env, _GEOMS, None)
  if cache is None:
    cache = {}
    setattr(env, _GEOMS, cache)
  if object_name not in cache:
    obj: Entity = env.scene[object_name]
    types = env.sim.mj_model.geom_type
    geoms = [(int(g), int(types[int(g)])) for g in obj.indexing.geom_ids.tolist()]
    bad = [(g, t) for g, t in geoms if t not in (_SPHERE, _BOX)]
    if bad:
      raise ValueError(
        f"{object_name!r} has geoms that are neither a box nor a sphere: {bad}"
      )
    cache[object_name] = geoms
  return cache[object_name]


def object_half_height(
  env: ManagerBasedRlEnv,
  object_name: str = "object",
  env_ids: torch.Tensor | None = None,
) -> torch.Tensor:
  """Height of the object's centre above a surface it rests on, per env (m): a box's half-z, a
  sphere's radius, the larger of the two when the object carries both (the other is at
  `INACTIVE_SIZE`). (B,), or (len(env_ids),)"""
  size = env.sim.model.geom_size
  half = None
  for gid, gtype in _object_geoms(env, object_name):
    col = 0 if gtype == _SPHERE else 2
    h = size[:, gid, col] if env_ids is None else size[env_ids, gid, col]
    half = h if half is None else torch.maximum(half, h)
  assert half is not None, f"{object_name!r} has no geoms"
  return half


def shape_kind(env: ManagerBasedRlEnv, object_name: str = "object") -> str:
  """``cube`` (only boxes), ``ball`` (only spheres) or ``mixed`` (both): a property of the model,
  not of a world."""
  types = {t for _, t in _object_geoms(env, object_name)}
  if types == {_BOX}:
    return "cube"
  if types == {_SPHERE}:
    return "ball"
  return "mixed"


def ball_mask(env: ManagerBasedRlEnv, object_name: str = "object") -> torch.Tensor:
  """True for the envs whose object is a ball at the moment. (B,)"""
  kind = shape_kind(env, object_name)
  if kind != "mixed":
    return torch.full(
      (env.num_envs,), kind == "ball", dtype=torch.bool, device=env.device
    )
  size = env.sim.model.geom_size
  (sphere,) = [g for g, t in _object_geoms(env, object_name) if t == _SPHERE]
  return size[:, sphere, 0] > 2.0 * INACTIVE_SIZE


def resolve(
  env: ManagerBasedRlEnv, value: float | PerShape, object_name: str = "object"
) -> float | torch.Tensor:
  """``value`` for this object: a float stays one; a `PerShape` is the model's shape's number,
  per env when the object can be either (B,)."""
  if not isinstance(value, PerShape):
    return value
  kind = shape_kind(env, object_name)
  if kind == "cube":
    return value.cube
  if kind == "ball":
    return value.ball
  return torch.where(
    ball_mask(env, object_name),
    torch.full((env.num_envs,), value.ball, device=env.device),
    torch.full((env.num_envs,), value.cube, device=env.device),
  )


@requires_model_fields("geom_size", "geom_rbound", "geom_aabb")
def set_object_shape(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  asset_cfg: SceneEntityCfg,
  ball_probability: float = 1.0,
  ball_radius: tuple[float, float] | None = None,
  cube_half: float = 0.025,
) -> None:
  """Reset event: draw the object's shape and size for ``env_ids`` (plan S8's MIX and BALL-DR).

  Each env's object is a ball with probability ``ball_probability`` -- 1 for a pure ball, 0.5
  for MIX -- and a cube otherwise; a shape the object's body does not carry is ignored. A ball's
  radius is ``ball_radius`` (lo, hi) drawn per episode, or the spec's own if None, and the cube's
  half-size is ``cube_half``. The shape that is off is shrunk to `INACTIVE_SIZE`. Runs *before*
  ``reset_objects_on_surface`` (and the goal command's resample), which read the half-height this
  sets -- `set_object_shape` is inserted ahead of it in the event dict.

  Writes ``geom_size`` and, through mjlab's `dr.geom_size` helper, the bounding sphere and box the
  broadphase reads, for exactly the envs reset. The mass and inertia are not touched: the DR of
  the object's mass (`DR-object`) already scales both together.
  """
  geoms = _object_geoms(env, asset_cfg.name)
  n = len(env_ids)
  device = env.device
  is_ball = torch.rand(n, device=device) < ball_probability
  size = env.sim.model.geom_size
  for gid, gtype in geoms:
    if gtype == _SPHERE:
      if ball_radius is None:
        on = size[env_ids, gid, 0].new_full(
          (n,), float(env.sim.mj_model.geom_size[gid, 0])
        )
      else:
        lo, hi = ball_radius
        on = lo + (hi - lo) * torch.rand(n, device=device)
      off = torch.full_like(on, INACTIVE_SIZE)
      size[env_ids, gid, 0] = torch.where(is_ball, on, off)
    else:
      on = torch.full((n,), cube_half, device=device)
      off = torch.full_like(on, INACTIVE_SIZE)
      half = torch.where(is_ball, off, on)
      size[env_ids, gid, 0] = half
      size[env_ids, gid, 1] = half
      size[env_ids, gid, 2] = half
  _recompute_geom_bounds(env, env_ids.to(dtype=torch.int), asset_cfg)
