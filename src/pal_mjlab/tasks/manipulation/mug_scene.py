"""The mug, the coaster and the kugel: the free objects of the mug tasks (``mug/``,
``coaster/``, ``kugel/``), beside the cube and ball of ``scene``.

Same naming rule as ``scene``: an object's MuJoCo body name is its scene entity key.
"""

from __future__ import annotations

import math

import mujoco
from mjlab.entity import EntityCfg

# -- Mug ----------------------------------------------------------------------------------
# ``Mug018`` as the Isaac Lab talos env builds it (motionplanning-sampling,
# ``envs/talos/constants.py``): the solid-collision variant of the asset -- one cylinder for
# the body and three boxes for the handle -- scaled (1.3, 1.3, 2.0), so the body is 65 mm
# across and 123 mm tall and the handle's outer bar stands 58 mm tall off it. Local frame
# as in the USD: origin at the body centre, z along the body axis, the handle along +x.
# The primitives are the collider there too, not an approximation made here.
_MUG_SCALE = (1.3, 1.3, 2.0)
MUG_RADIUS = 0.025 * _MUG_SCALE[0]
MUG_HALF_HEIGHT = 0.03075 * _MUG_SCALE[2]
#: Body diameter: what the jaw has to admit across the body (79.3 mm open).
MUG_WIDTH = 2 * MUG_RADIUS
#: The handle bars as (half extents, centre), in the mug frame.
MUG_HANDLE_BOXES = (
  (
    (0.0125 * _MUG_SCALE[0], 0.005 * _MUG_SCALE[1], 0.0025 * _MUG_SCALE[2]),
    (0.0375 * _MUG_SCALE[0], 0.0, 0.012 * _MUG_SCALE[2]),
  ),
  (
    (0.0125 * _MUG_SCALE[0], 0.005 * _MUG_SCALE[1], 0.0025 * _MUG_SCALE[2]),
    (0.0375 * _MUG_SCALE[0], 0.0, -0.012 * _MUG_SCALE[2]),
  ),
  (
    (0.0025 * _MUG_SCALE[0], 0.005 * _MUG_SCALE[1], 0.0145 * _MUG_SCALE[2]),
    (0.0475 * _MUG_SCALE[0], 0.0, 0.0),
  ),
)
#: How far the handle's outer face stands off the body axis.
MUG_HANDLE_REACH = 0.050 * _MUG_SCALE[0]
#: A 3D-printed mug: 50-80 g.
MUG_MASS = 0.065


def get_mug_spec(
  name: str = "object",
  mass: float = MUG_MASS,
  rgba: tuple[float, float, float, float] = (0.85, 0.85, 0.80, 1.0),
) -> mujoco.MjSpec:
  """A free mug, body cylinder first: the placement code reads the first geom's size.

  The mass is split by volume, so the handle carries its ~4% of it.
  """
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name=name)
  body.add_freejoint(name=f"{name}_joint")
  body_vol = math.pi * MUG_RADIUS**2 * 2 * MUG_HALF_HEIGHT
  bar_vols = [8 * hx * hy * hz for (hx, hy, hz), _ in MUG_HANDLE_BOXES]
  total = body_vol + sum(bar_vols)
  contact = dict(
    condim=4,
    friction=(0.9, 0.02, 0.0001),
    solref=(0.02, 1.0),
    solimp=(0.99, 0.995, 0.001, 0.5, 2),
  )
  body.add_geom(
    name=f"{name}_body_geom",
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    size=(MUG_RADIUS, MUG_HALF_HEIGHT, 0.0),
    mass=mass * body_vol / total,
    rgba=rgba,
    **contact,
  )
  for i, ((size, pos), vol) in enumerate(zip(MUG_HANDLE_BOXES, bar_vols, strict=True)):
    body.add_geom(
      name=f"{name}_handle{i}_geom",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=size,
      pos=pos,
      mass=mass * vol / total,
      rgba=rgba,
      **contact,
    )
  return spec


def get_mug_cfg(name: str = "object", **kwargs) -> EntityCfg:
  return EntityCfg(spec_fn=lambda: get_mug_spec(name=name, **kwargs))


# -- Coaster ------------------------------------------------------------------------------
#: A square coaster for the mug to be set down on: 10 cm a side, so a mug (65 mm) centred
#: within 17 mm of it stands wholly on it, and 8 mm thick, under the 30 mm the handle's
#: lower bar hangs above the table. Square, not round: mujoco_warp's cylinder-cylinder
#: contact rests the mug 4.5 mm high and the contact sensor never sees it, where
#: cylinder-box is the pair the mug already stands on the table with.
COASTER_HALF_SIZE = 0.05
COASTER_HALF_HEIGHT = 0.004
#: Heavier than a cork coaster, as a cork-backed or rubber one is: light enough to be
#: shoved by a mug dragged onto it, which is a failure worth seeing.
COASTER_MASS = 0.1


def get_coaster_spec(
  name: str = "coaster",
  mass: float = COASTER_MASS,
  rgba: tuple[float, float, float, float] = (0.25, 0.25, 0.30, 1.0),
) -> mujoco.MjSpec:
  """A free, flat coaster: one box, so the first geom gives its half height."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name=name)
  body.add_freejoint(name=f"{name}_joint")
  body.add_geom(
    name=f"{name}_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(COASTER_HALF_SIZE, COASTER_HALF_SIZE, COASTER_HALF_HEIGHT),
    mass=mass,
    rgba=rgba,
    condim=4,
    friction=(1.0, 0.02, 0.0001),
    solref=(0.02, 1.0),
    solimp=(0.99, 0.995, 0.001, 0.5, 2),
  )
  return spec


def get_coaster_cfg(name: str = "coaster", **kwargs) -> EntityCfg:
  return EntityCfg(spec_fn=lambda: get_coaster_spec(name=name, **kwargs))


# -- Hollow mug and Kugel -----------------------------------------------------------------
# The mug above is solid, as the Isaac Lab collider was. For something to go *in* it, the
# body is rebuilt from ``Mug018.obj``'s inside, measured off the mesh and scaled as above:
# the wall is 1.7 mm thick (inner radius 23.3 of 25.0 mm) and the floor 2.0 mm, so at
# (1.3, 1.3, 2.0) the opening is 60.6 mm across and 118.9 mm deep.
MUG_INNER_RADIUS = 0.0233 * _MUG_SCALE[0]
MUG_FLOOR_THICKNESS = 0.0020 * _MUG_SCALE[2]
#: How many flat staves stand in for the round wall. Their inner faces sit on
#: ``MUG_INNER_RADIUS``; at 16 the corners between them reach in 0.6 mm less than that.
MUG_WALL_STAVES = 16

#: A wooden ball for the jaw to drop into the hollow mug. 40 mm: narrower than the 50 mm
#: cube the jaw already holds, with 10.3 mm to spare each side of the 60.6 mm opening, and
#: resting on the floor its top is 79 mm below the rim.
KUGEL_RADIUS = 0.020
KUGEL_WIDTH = 2 * KUGEL_RADIUS
#: Beech, 0.7 g/cm^3.
KUGEL_MASS = 0.023
#: Rolling friction, m. Without it a sphere nudged by a finger rolls off the table: a real
#: wooden ball on a wooden top stops within a few centimetres.
KUGEL_ROLLING_FRICTION = 0.002


def get_hollow_mug_spec(
  name: str = "mug",
  mass: float = MUG_MASS,
  rgba: tuple[float, float, float, float] = (0.85, 0.85, 0.80, 1.0),
) -> mujoco.MjSpec:
  """``get_mug_spec`` with the body open at the top: a floor slab and a ring of staves.

  The first geom is still the whole body cylinder, which the placement and goal code read
  for the mug's half height, but it collides with nothing and weighs nothing. The floor is
  a square inside the ring: a 40 mm ball resting on it touches it within 10 mm of the axis.
  """
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name=name)
  body.add_freejoint(name=f"{name}_joint")
  body.add_geom(
    name=f"{name}_body_geom",
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    size=(MUG_RADIUS, MUG_HALF_HEIGHT, 0.0),
    mass=0.0,
    contype=0,
    conaffinity=0,
    rgba=(*rgba[:3], 0.0),
  )
  wall = MUG_RADIUS - MUG_INNER_RADIUS
  stave_half_w = MUG_RADIUS * math.tan(math.pi / MUG_WALL_STAVES)
  floor_half = MUG_INNER_RADIUS / math.sqrt(2.0)
  boxes = [
    (
      (floor_half, floor_half, MUG_FLOOR_THICKNESS / 2),
      (0.0, 0.0, -MUG_HALF_HEIGHT + MUG_FLOOR_THICKNESS / 2),
      0.0,
    ),
    *(
      (
        (wall / 2, stave_half_w, MUG_HALF_HEIGHT),
        (
          (MUG_INNER_RADIUS + wall / 2) * math.cos(a),
          (MUG_INNER_RADIUS + wall / 2) * math.sin(a),
          0.0,
        ),
        a,
      )
      for a in (2 * math.pi * i / MUG_WALL_STAVES for i in range(MUG_WALL_STAVES))
    ),
    *((size, pos, 0.0) for size, pos in MUG_HANDLE_BOXES),
  ]
  vols = [8 * hx * hy * hz for (hx, hy, hz), _, _ in boxes]
  contact = dict(
    condim=4,
    friction=(0.9, 0.02, 0.0001),
    solref=(0.02, 1.0),
    solimp=(0.99, 0.995, 0.001, 0.5, 2),
  )
  for i, ((size, pos, yaw), vol) in enumerate(zip(boxes, vols, strict=True)):
    body.add_geom(
      name=f"{name}_part{i}_geom",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=size,
      pos=pos,
      quat=(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)),
      mass=mass * vol / sum(vols),
      rgba=rgba,
      **contact,
    )
  return spec


def get_hollow_mug_cfg(name: str = "mug", **kwargs) -> EntityCfg:
  return EntityCfg(spec_fn=lambda: get_hollow_mug_spec(name=name, **kwargs))


def get_kugel_spec(
  name: str = "object",
  radius: float = KUGEL_RADIUS,
  mass: float = KUGEL_MASS,
  rgba: tuple[float, float, float, float] = (0.9, 0.6, 0.1, 1.0),
) -> mujoco.MjSpec:
  """A free wooden ball. ``condim=6`` for the rolling friction."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name=name)
  body.add_freejoint(name=f"{name}_joint")
  body.add_geom(
    name=f"{name}_geom",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=(radius, 0.0, 0.0),
    mass=mass,
    rgba=rgba,
    condim=6,
    friction=(0.9, 0.02, KUGEL_ROLLING_FRICTION),
    solref=(0.02, 1.0),
    solimp=(0.99, 0.995, 0.001, 0.5, 2),
  )
  return spec


def get_kugel_cfg(name: str = "object", **kwargs) -> EntityCfg:
  return EntityCfg(spec_fn=lambda: get_kugel_spec(name=name, **kwargs))
