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


#: Plan S10 (`pick-place-v1-objects-plan.md`, *The box*; built to these numbers, Siwei 2026-10-01 22:02):
#: an open box for the 5 cm cube, `oppsie-data/cad/s10_box/make_box.py` geom for geom. Inner 8.0 x 8.0 cm
#: (the cube + 3 cm), walls 5 mm thick and 2.5 cm above the floor (half the cube), floor 5 mm: outer
#: 9.0 x 9.0 x 3.0 cm, so the rim is at the table + 3.0 cm and a cube resting inside has its centre there too.
BOX_INNER = 0.08
BOX_WALL = 0.005
BOX_FLOOR = 0.005
BOX_WALL_HEIGHT = 0.025
#: The real box is not weighed yet (plan S10 step 0, item 1): ~ the solid volume (83 cm^3) in PLA until it is.
BOX_MASS = 0.10
#: Box-table sliding friction is the table's 0.8 (element-wise max, L10); this is the box's own, which
#: decides nothing against the cube's 0.9 either -- kept for the record of the plan's "inside: 0.6".
BOX_FRICTION = 0.6


def box_wall_geom_names(name: str = "box") -> tuple[str, ...]:
  """The four wall geoms of `get_box_spec`'s body, +x, -x, +y, -y."""
  return tuple(f"{name}_wall_{side}" for side in ("px", "mx", "py", "my"))


def get_box_spec(
  name: str = "box",
  inner: float = BOX_INNER,
  wall: float = BOX_WALL,
  floor: float = BOX_FLOOR,
  wall_height: float = BOX_WALL_HEIGHT,
  mass: float = BOX_MASS,
  rgba: tuple[float, float, float, float] = (0.25, 0.55, 0.35, 1.0),
) -> mujoco.MjSpec:
  """A free open-top box (plan S10): a floor and four walls, the body's origin at the centre of
  its bottom face (as `make_box.py`'s mesh), so it rests on a surface at the surface's height.

  Five box geoms, condim 3, the mass spread over them by volume (one density). The two y walls
  span the outer width and the two x walls the inner, so no two geoms overlap. A site
  ``{name}_floor`` marks the floor's top centre -- the goal's xy, and the cube's resting height
  minus its half-height. The walls are named (`box_wall_geom_names`) so that their top -- the rim --
  and their inner faces can be read per world, which plan S10's BX-DR draws."""
  half_in = inner / 2
  half_out = half_in + wall
  wall_z = floor + wall_height / 2
  geoms = [
    (f"{name}_floor", (0.0, 0.0, floor / 2), (half_out, half_out, floor / 2)),
    (
      f"{name}_wall_px",
      (half_in + wall / 2, 0.0, wall_z),
      (wall / 2, half_in, wall_height / 2),
    ),
    (
      f"{name}_wall_mx",
      (-half_in - wall / 2, 0.0, wall_z),
      (wall / 2, half_in, wall_height / 2),
    ),
    (
      f"{name}_wall_py",
      (0.0, half_in + wall / 2, wall_z),
      (half_out, wall / 2, wall_height / 2),
    ),
    (
      f"{name}_wall_my",
      (0.0, -half_in - wall / 2, wall_z),
      (half_out, wall / 2, wall_height / 2),
    ),
  ]
  volume = sum(8.0 * s[0] * s[1] * s[2] for _, _, s in geoms)
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name=name)
  body.add_freejoint(name=f"{name}_joint")
  for geom_name, pos, size in geoms:
    body.add_geom(
      name=geom_name,
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=pos,
      size=size,
      density=mass / volume,
      rgba=rgba,
      condim=3,
      friction=(BOX_FRICTION, 0.005, 0.0001),
      solref=(0.02, 1.0),
      solimp=(0.99, 0.995, 0.001, 0.5, 2),
    )
  body.add_site(name=f"{name}_floor", pos=(0.0, 0.0, floor), size=(0.004, 0.0, 0.0))
  return spec


def get_box_cfg(name: str = "box", **kwargs) -> EntityCfg:
  return EntityCfg(spec_fn=lambda: get_box_spec(name=name, **kwargs))
