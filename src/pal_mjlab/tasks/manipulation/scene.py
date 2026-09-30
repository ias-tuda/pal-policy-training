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

#: Table top height. A real table height, and inside Kangaroo's vertical reach. Every run up
#: to S7-1 trained here.
TABLE_TOP_Z = 0.76
#: The table top of the deployment cell, relative to the robot (plan S7-2 baseline B; Claudius's
#: f44b49d). What has to match the cell is the cube's height in `base_link`, where the policy
#: sees it: `base_link` stands at 0.900 m in this scene and the robot measured the resting cube
#: at z_b = -0.060, so 0.900 - 0.060 - 0.025 = 0.815. At 0.76 the sim's cube sat 5.5 cm lower
#: (z_b = -0.115) than the real one.
ROBOT_CELL_TABLE_TOP_Z = 0.815
TABLE_HALF_X = 0.30
TABLE_HALF_Y = 0.40
TABLE_THICKNESS = 0.04


def table_pos(top_z: float = TABLE_TOP_Z) -> tuple[float, float, float]:
  """Table centre in the robot's frame, in front of the base, for a top face at ``top_z``."""
  return (0.55, 0.0, top_z - TABLE_THICKNESS / 2)


#: Table centre in the robot's frame, in front of the base.
TABLE_POS = table_pos()


def get_table_spec(top_z: float = TABLE_TOP_Z) -> mujoco.MjSpec:
  """A static table: one top slab plus four legs, its top face at ``top_z``."""
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
  leg_half_z = (top_z - TABLE_THICKNESS) / 2
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


def get_table_cfg(top_z: float = TABLE_TOP_Z) -> EntityCfg:
  return EntityCfg(
    spec_fn=lambda: get_table_spec(top_z),
    init_state=EntityCfg.InitialStateCfg(pos=table_pos(top_z)),
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


#: Plan S8 (the ball): a 6 cm sphere in place of the 5 cm cube. Its centre rests 3 cm above the
#: table (the cube's 2.5 cm), and it leaves 9.7 mm per side in the 79.3 mm open jaw (the cube
#: 14.7 mm).
BALL_RADIUS = 0.03
#: The real ball's mass is not known yet (plan S8, open item 1): the cube's 50 g until it is.
BALL_MASS = 0.05
#: Rolling friction of the ball (MuJoCo's third friction coefficient, a length, m). The plan's
#: rule until the real ball is measured: a ball let go at 0.05 m/s stops within 1 cm. Calibrated
#: against the task's own scene and solver settings (`tests/test_pick_place_s8.py`, step 0 of the
#: S8 report): released at 0.05 m/s with no spin it stops after 0.53 cm, rolling already (the
#: stricter start) after 0.95 cm; at 5e-4 the rolling start took 1.16 cm. The table's 1e-4, which
#: the element-wise max would leave in force, rolls the same ball 9-13 cm.
BALL_ROLLING_FRICTION = 6e-4
#: A shape that is switched off in one env (plan S8's MIX, and the DR of the ball's size) is not
#: removed -- mujoco_warp's ``geom_type`` is one per model, not per world -- but shrunk to this
#: half-size / radius (m) at the body's centre, inside the shape that is on.
INACTIVE_SIZE = 1e-3


def get_ball_spec(
  name: str = "object",
  radius: float = BALL_RADIUS,
  mass: float = BALL_MASS,
  rolling: float = BALL_ROLLING_FRICTION,
  rgba: tuple[float, float, float, float] = (0.2, 0.4, 0.9, 1.0),
) -> mujoco.MjSpec:
  """A free ball, for the same jaw (79 mm open, so a 60 mm ball leaves 19 mm).

  The body is named ``name`` and its geom ``{name}_geom`` exactly as the cube's, so every sensor
  and term that matches the cube's body (and the DR of its friction and mass) matches the ball.
  ``condim=6`` so that rolling friction exists: the pads' own contact keeps condim 4 (priority 2
  picks theirs), and the ball's rolling friction only acts against the table.
  """
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
    friction=(0.9, 0.02, rolling),
    solref=(0.02, 1.0),
    solimp=(0.99, 0.995, 0.001, 0.5, 2),
  )
  return spec


def get_ball_cfg(name: str = "object", **kwargs) -> EntityCfg:
  return EntityCfg(spec_fn=lambda: get_ball_spec(name=name, **kwargs))


def mixed_ball_geom_name(name: str = "object") -> str:
  """The ball geom of `get_mixed_spec`'s body (its cube geom is ``{name}_geom``, as a cube's)."""
  return f"{name}_geom_ball"


def get_mixed_spec(
  name: str = "object",
  half_size: float = 0.025,
  radius: float = BALL_RADIUS,
  mass: float = 0.05,
  rolling: float = BALL_ROLLING_FRICTION,
) -> mujoco.MjSpec:
  """One body that is a cube in some worlds and a ball in others (plan S8's MIX).

  Both shapes are geoms of the one body -- the cube first, so ``geom_ids[0]`` is the cube's as
  for every other object -- and ``mdp.set_object_shape`` switches one of them off per env and
  episode by shrinking it to `INACTIVE_SIZE` (`mdp.shapes`). The body's mass and inertia are
  the cube's (2.1e-5 kg m^2); a 50 g ball's would be 1.8e-5, which is not worth a per-world
  inertia.
  """
  spec = get_cube_spec(name=name, half_size=half_size, mass=mass)
  body = spec.body(name)
  body.add_geom(
    name=mixed_ball_geom_name(name),
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=(radius, 0.0, 0.0),
    mass=0.0,
    rgba=(0.2, 0.4, 0.9, 1.0),
    condim=6,
    friction=(0.9, 0.02, rolling),
    solref=(0.02, 1.0),
    solimp=(0.99, 0.995, 0.001, 0.5, 2),
  )
  return spec


def get_mixed_cfg(name: str = "object", **kwargs) -> EntityCfg:
  return EntityCfg(spec_fn=lambda: get_mixed_spec(name=name, **kwargs))
