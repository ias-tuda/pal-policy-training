"""Shared scene pieces for tabletop manipulation tasks.

Every task in this package puts a robot in front of a table and manipulates one or more
free objects on it. The table and the object primitives live here so a new task only has to
describe its own goal, not re-derive a workspace.

The table is an :class:`EntityCfg`, not a scene-level ``spec_fn``: the placement event reads
the table geom's world-frame pose out of physics data every reset (so that table-height
randomisation and env-origin offsets are picked up automatically), and that lookup needs an
addressable entity.

Naming rule, relied upon by the contact sensors in ``tabletop_env_cfg``: **an object's MuJoCo
body name is its scene entity key.** ``get_cube_cfg(name=...)`` sets both.
"""

from __future__ import annotations

import mujoco
from mjlab.entity import EntityCfg

#: Table top height. A real table height, and inside Kangaroo's vertical reach.
TABLE_TOP_Z = 0.76
TABLE_HALF_X = 0.30
TABLE_HALF_Y = 0.40
TABLE_THICKNESS = 0.04
#: Table centre in the robot's frame, in front of the base.
TABLE_POS = (0.55, 0.0, TABLE_TOP_Z - TABLE_THICKNESS / 2)


def get_table_spec() -> mujoco.MjSpec:
  """A static table: one top slab plus four legs."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="table")
  body.add_geom(
    name="table_top",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(TABLE_HALF_X, TABLE_HALF_Y, TABLE_THICKNESS / 2),
    rgba=(0.55, 0.42, 0.28, 1.0),
    condim=3,
    friction=(0.8, 0.005, 0.0001),
    solref=(0.02, 1.0),
    solimp=(0.95, 0.99, 0.001, 0.5, 2),
  )
  leg_half_z = (TABLE_TOP_Z - TABLE_THICKNESS) / 2
  for sx in (-1, 1):
    for sy in (-1, 1):
      body.add_geom(
        name=f"table_leg_{'p' if sx > 0 else 'm'}{'p' if sy > 0 else 'm'}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(
          sx * (TABLE_HALF_X - 0.04),
          sy * (TABLE_HALF_Y - 0.04),
          -TABLE_THICKNESS / 2 - leg_half_z,
        ),
        size=(0.03, 0.03, leg_half_z),
        rgba=(0.45, 0.34, 0.22, 1.0),
        condim=3,
      )
  return spec


def get_table_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=get_table_spec,
    init_state=EntityCfg.InitialStateCfg(pos=TABLE_POS),
  )


def get_cube_spec(
  name: str = "object",
  half_size: float = 0.025,
  mass: float = 0.05,
  rgba: tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1.0),
) -> mujoco.MjSpec:
  """A free cube, sized to fit the Kangaroo jaw (79 mm open, so 50 mm leaves 29 mm).

  ``name`` must match the scene entity key this cube is registered under; the body name is
  what the contact sensors match on.
  """
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name=name)
  body.add_freejoint(name=f"{name}_joint")
  body.add_geom(
    name=f"{name}_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(half_size,) * 3,
    mass=mass,
    rgba=rgba,
    # condim=4 so the pads' torsional friction can actually hold the object; see
    # MANIPULATION_COLLISION in kangaroo_constants.py.
    condim=4,
    friction=(0.9, 0.02, 0.0001),
    solref=(0.02, 1.0),
    solimp=(0.99, 0.995, 0.001, 0.5, 2),
  )
  return spec


def get_cube_cfg(name: str = "object", **kwargs) -> EntityCfg:
  return EntityCfg(spec_fn=lambda: get_cube_spec(name=name, **kwargs))
