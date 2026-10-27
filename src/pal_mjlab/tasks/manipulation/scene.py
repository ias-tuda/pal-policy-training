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

import math

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


def get_table_spec(top_z: float = TABLE_TOP_Z, contype: int = 1) -> mujoco.MjSpec:
  """A static table: one top slab plus four legs, its top face at ``top_z``. ``contype`` is the
  geoms' contact type (conaffinity stays 1): plan §12's `INSERT_TABLE_CONTYPE` keeps the socket
  block's inner geoms off the table; 1, every task before, collides with everything."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="table")
  body.add_geom(
    name="table_top",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(TABLE_HALF_X, TABLE_HALF_Y, TABLE_THICKNESS / 2),
    rgba=(0.55, 0.42, 0.28, 1.0),
    condim=3,
    contype=contype,
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
        contype=contype,
      )
  return spec


def get_table_cfg(top_z: float = TABLE_TOP_Z, contype: int = 1) -> EntityCfg:
  return EntityCfg(
    spec_fn=lambda: get_table_spec(top_z, contype),
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


#: The box stands on four small spheres at the corners of its bottom face (radius, inset from the
#: outer edge, m), and its floor slab is raised `BOX_FOOT_GAP` off the surface. Found 2026-10-02 (S10
#: report §8): a 5 mm floor slab lying flat on the table slab, once rocked -- a cube dropped in, a
#: finger on a wall -- switches between corner contact sets (4 <-> 8 contacts) and keeps bobbing,
#: 3-5 mm into the table every ~0.2 s, at 10 and at 100 solver iterations alike; on the feet it
#: settles within 0.4 mm. A flat-bottomed box on a table bears at its rim anyway.
BOX_FOOT_RADIUS = 0.003
BOX_FOOT_INSET = 0.005
BOX_FOOT_GAP = 0.001


def box_foot_geom_names(name: str = "box") -> tuple[str, ...]:
  """The four foot spheres of `get_box_spec`'s body (with ``feet``), +x+y, +x-y, -x+y, -x-y."""
  return tuple(f"{name}_foot_{c}" for c in ("pp", "pm", "mp", "mm"))


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
  feet: bool = True,
) -> mujoco.MjSpec:
  """A free open-top box (plan S10): a floor and four walls, the body's origin at the centre of
  its bottom face (as `make_box.py`'s mesh), so it rests on a surface at the surface's height.

  Five box geoms, condim 3, the mass spread over them by volume (one density). The two y walls
  span the outer width and the two x walls the inner, so no two geoms overlap. A site
  ``{name}_floor`` marks the floor's top centre -- the goal's xy, and the cube's resting height
  minus its half-height. The walls are named (`box_wall_geom_names`) so that their top -- the rim --
  and their inner faces can be read per world, which plan S10's BX-DR draws.

  With ``feet`` (the default since 2026-10-02) the box stands on four spheres of radius
  `BOX_FOOT_RADIUS` at the corners, their bottoms at the box's bottom face, and the floor slab
  starts `BOX_FOOT_GAP` above it: the floor's top, the rim, the outer size and the mass are
  unchanged. Without, the floor slab rests on the surface itself, which bobs once rocked (see
  `BOX_FOOT_RADIUS`) -- every S10 run of `0ba3f60` trained on that."""
  half_in = inner / 2
  half_out = half_in + wall
  wall_z = floor + wall_height / 2
  gap = BOX_FOOT_GAP if feet else 0.0
  geoms = [
    (
      f"{name}_floor",
      (0.0, 0.0, (floor + gap) / 2),
      (half_out, half_out, (floor - gap) / 2),
    ),
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
  r, c = BOX_FOOT_RADIUS, half_out - BOX_FOOT_INSET
  foot_pos = [(c, c), (c, -c), (-c, c), (-c, -c)]
  volume = sum(8.0 * s[0] * s[1] * s[2] for _, _, s in geoms)
  if feet:
    volume += 4 * (4.0 / 3.0) * 3.141592653589793 * r**3
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
  if feet:
    for foot, (x, y) in zip(box_foot_geom_names(name), foot_pos, strict=True):
      body.add_geom(
        name=foot,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=(x, y, r),
        size=(r, 0.0, 0.0),
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


# -- Plan §12 / §13 (`pick-place-insert-plan.md` §0.2): the cross pin and the socket block -----------
#
# Built to `oppsie-data/cad/s12_insert/insert_geoms.py` box for box (the v2 parts, Ø 38 hole since
# 2026-10-07 14:51); every pin-block pair is box-box, so decision 0004's `box_box` collider handles it.
# The sharp boxes differ from the rounded print only at the rounds (plan §0.2 *Rounds*).

#: The pin: two plates (span x thickness) crossing at its axis, `PIN_LENGTH` long (§0.2).
PIN_SPAN = 0.032
PIN_THICKNESS = 0.008
PIN_LENGTH = 0.090
#: The printed pin's model mass (solid PLA 49 g; plan §0.2's 40 g until it is weighed).
PIN_MASS = 0.04
#: Pin-block sliding friction (PLA on PLA, to be measured): the pin's and the block's own, so their
#: pair is this (the element-wise max, L10); the pads keep theirs (priority 2), the table its 0.8.
PIN_BLOCK_FRICTION = 0.4

#: The socket block: a `BLOCK_SIZE` cube with a round through-hole of `BLOCK_HOLE_D` on its vertical
#: axis and an entry chamfer `BLOCK_CHAMFER` (radial width, depth) at the top (§0.2, D1 / D3).
BLOCK_SIZE = 0.056
BLOCK_HEIGHT = 0.056
BLOCK_HOLE_D = 0.038
BLOCK_CHAMFER = (0.005, 0.010)
#: Plan §12's IN-T: the earlier, tight hole (1.5 mm per side) with the chamfer chosen for it (§0.4).
BLOCK_HOLE_D_TIGHT = 0.035
BLOCK_CHAMFER_TIGHT = (0.007, 0.012)
#: The block's model mass (solid PLA 132 g; plan §0.2's 100 g until it is weighed).
BLOCK_MASS = 0.10
#: `insert_geoms.py`'s layout constants (mm there): the hole and the chamfer are 16-gons, the outer
#: square ring's inner half-size is at least this, the top collar this thick.
_BLOCK_SIDES = 16
_BLOCK_RING_IN = 0.023
_BLOCK_COLLAR = 0.004
_BLOCK_CHAMFER_SLAB = 0.004
#: Collision bits of the block's inner geoms (staves, chamfer, fillers, collar, caps): they collide
#: with the pin, the hand and any object (contype 1 / conaffinity 1 there) through their conaffinity
#: bit 1, and not with a table built with `INSERT_TABLE_CONTYPE` -- only the outer ring carries the
#: block on the table, which keeps 20 box-box pairs (~80 contacts) per world out of the solver. On a
#: default table (contype 1) they collide with it too: the same support, more contacts.
BLOCK_INNER_CONTYPE = 2
BLOCK_INNER_CONAFFINITY = 1
#: The table's contype for the insert tasks: collides with every contype-1 geom (robot, objects, the
#: block's ring) through its conaffinity 1, and not with the block's inner geoms.
INSERT_TABLE_CONTYPE = 4


def _yaw_pitch_quat(
  yaw: float, pitch: float = 0.0
) -> tuple[float, float, float, float]:
  """(w, x, y, z) of R = Rz(yaw) Ry(pitch) (`insert_geoms._q`)."""
  cz, sz = math.cos(yaw / 2), math.sin(yaw / 2)
  cy, sy = math.cos(pitch / 2), math.sin(pitch / 2)
  return (cz * cy, -sz * sy, cz * sy, sz * cy)


def socket_block_boxes(
  size: float = BLOCK_SIZE,
  height: float = BLOCK_HEIGHT,
  hole_d: float = BLOCK_HOLE_D,
  chamfer: tuple[float, float] = BLOCK_CHAMFER,
) -> list[tuple[str, tuple[float, ...], tuple[float, ...], tuple[float, ...]]]:
  """The block's boxes, (name, centre, half-size, quat (w, x, y, z)), m, its origin at the centre of
  the bottom face on the hole axis: a square ring, `_BLOCK_SIDES` hole staves whose inner faces form a
  16-gon inscribed in the hole, as many chamfer staves, 4 diagonal fillers, a 16-box top collar and 4
  corner caps; no floor. `insert_geoms.block_boxes` in metres (`tests/data/s12_block_boxes_*.json`)."""
  mm = 1e3
  r, top, s = hole_d / 2 * mm, height * mm, size / 2 * mm
  cr, ch = chamfer[0] * mm, chamfer[1] * mm
  n = _BLOCK_SIDES
  ri = max(_BLOCK_RING_IN * mm, r + cr + 0.5)
  out: list[tuple[str, tuple, tuple, tuple]] = [
    (
      "ring_xp",
      ((ri + s) / 2, 0.0, top / 2),
      ((s - ri) / 2, s, top / 2),
      (1.0, 0.0, 0.0, 0.0),
    ),
    (
      "ring_xn",
      (-(ri + s) / 2, 0.0, top / 2),
      ((s - ri) / 2, s, top / 2),
      (1.0, 0.0, 0.0, 0.0),
    ),
    (
      "ring_yp",
      (0.0, (ri + s) / 2, top / 2),
      (s, (s - ri) / 2, top / 2),
      (1.0, 0.0, 0.0, 0.0),
    ),
    (
      "ring_yn",
      (0.0, -(ri + s) / 2, top / 2),
      (s, (s - ri) / 2, top / 2),
      (1.0, 0.0, 0.0, 0.0),
    ),
  ]
  wall_top = top - ch
  r_out = ri + 1.0
  half_w = r_out * math.tan(math.pi / n)
  for k in range(n):
    th = 2 * math.pi * k / n
    rc = (r + r_out) / 2
    out.append(
      (
        f"stave{k}",
        (rc * math.cos(th), rc * math.sin(th), wall_top / 2),
        ((r_out - r) / 2, half_w, wall_top / 2),
        _yaw_pitch_quat(th),
      )
    )
  t = _BLOCK_CHAMFER_SLAB * mm
  beta = math.atan2(cr, ch)  # the slope's angle from the vertical
  length = math.hypot(cr, ch)
  cx = r + cr / 2 + math.cos(beta) * t / 2  # into the material: out and down
  cz = wall_top + ch / 2 - math.sin(beta) * t / 2
  for k in range(n):
    th = 2 * math.pi * k / n
    out.append(
      (
        f"cham{k}",
        (cx * math.cos(th), cx * math.sin(th), cz),
        (t / 2, (r + cr + 1) * math.tan(math.pi / n), length / 2),
        _yaw_pitch_quat(th, beta),
      )
    )
  r0, r1 = r + cr, ri * math.sqrt(2)
  hw = s * math.sqrt(2) - r1  # stays inside the outer square at r1
  for k in range(4):
    th = math.pi / 4 + k * math.pi / 2
    rc = (r0 + r1) / 2
    out.append(
      (
        f"diag{k}",
        (rc * math.cos(th), rc * math.sin(th), top / 2),
        ((r1 - r0) / 2, min(hw, 8.0), top / 2),
        _yaw_pitch_quat(th),
      )
    )
  collar = _BLOCK_COLLAR * mm
  for k in range(n):
    th = 2 * math.pi * k / n + math.pi / n
    # Each collar box reaches as far out as its two outer corners stay inside the outline (0.1 mm).
    c1 = r0 + 0.5
    steps = int((s * math.sqrt(2) - (r0 + 0.5)) / 0.05) + 1
    for i in range(steps):
      rr = r0 + 0.5 + 0.05 * i
      w = rr * math.tan(math.pi / n)
      corners = [
        (
          rr * math.cos(th) - sg * w * math.sin(th),
          rr * math.sin(th) + sg * w * math.cos(th),
        )
        for sg in (-1, 1)
      ]
      if max(max(abs(cx_), abs(cy_)) for cx_, cy_ in corners) > s - 0.1:
        break
      c1 = rr
    rc = (r0 + c1) / 2
    out.append(
      (
        f"collar{k}",
        (rc * math.cos(th), rc * math.sin(th), top - collar / 2),
        ((c1 - r0) / 2, c1 * math.tan(math.pi / n), collar / 2),
        _yaw_pitch_quat(th),
      )
    )
  a = r0 / math.sqrt(2) + 0.2
  for k, (sx, sy) in enumerate(((1, 1), (-1, 1), (-1, -1), (1, -1))):
    out.append(
      (
        f"cap{k}",
        (sx * (a + s) / 2, sy * (a + s) / 2, top - collar / 2),
        ((s - a) / 2, (s - a) / 2, collar / 2),
        (1.0, 0.0, 0.0, 0.0),
      )
    )
  return [
    (name, tuple(v / mm for v in c), tuple(v / mm for v in h), q)
    for name, c, h, q in out
  ]


def socket_block_ring_geom_names(name: str = "block") -> tuple[str, ...]:
  """The outer ring's four geoms of `get_socket_block_spec`'s body: the ones that stand on the table."""
  return tuple(f"{name}_ring_{side}" for side in ("xp", "xn", "yp", "yn"))


def get_socket_block_spec(
  name: str = "block",
  size: float = BLOCK_SIZE,
  height: float = BLOCK_HEIGHT,
  hole_d: float = BLOCK_HOLE_D,
  chamfer: tuple[float, float] = BLOCK_CHAMFER,
  mass: float = BLOCK_MASS,
  friction: float = PIN_BLOCK_FRICTION,
  rgba: tuple[float, float, float, float] = (0.55, 0.6, 0.72, 1.0),
) -> mujoco.MjSpec:
  """A free socket block (plan §0.2): `socket_block_boxes` as box geoms ``{name}_<box>``, the body's
  origin at the centre of its bottom face on the hole axis (+z up), so it rests on a surface at the
  surface's height -- the tracked pose, as S10.1's box. One density (``mass`` over the boxes' volume),
  condim 3. The outer ring (`socket_block_ring_geom_names`) collides as every object does; the inner
  geoms carry `BLOCK_INNER_CONTYPE` / `BLOCK_INNER_CONAFFINITY` (see there). Nothing here assumes the
  block stays on the table: plan §13 picks it up. A site ``{name}_top`` marks the hole's entry (the
  top face's centre)."""
  boxes = socket_block_boxes(size, height, hole_d, chamfer)
  volume = sum(8.0 * h[0] * h[1] * h[2] for _, _, h, _ in boxes)
  ring = set(socket_block_ring_geom_names(name))
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name=name)
  body.add_freejoint(name=f"{name}_joint")
  for box_name, centre, half, quat in boxes:
    geom = f"{name}_{box_name}"
    inner = geom not in ring
    body.add_geom(
      name=geom,
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=centre,
      size=half,
      quat=quat,
      density=mass / volume,
      rgba=rgba,
      condim=3,
      contype=BLOCK_INNER_CONTYPE if inner else 1,
      conaffinity=BLOCK_INNER_CONAFFINITY if inner else 1,
      friction=(friction, 0.005, 0.0001),
      solref=(0.02, 1.0),
      solimp=(0.99, 0.995, 0.001, 0.5, 2),
    )
  body.add_site(name=f"{name}_top", pos=(0.0, 0.0, height), size=(0.004, 0.0, 0.0))
  return spec


def get_socket_block_cfg(name: str = "block", **kwargs) -> EntityCfg:
  return EntityCfg(spec_fn=lambda: get_socket_block_spec(name=name, **kwargs))


def pin_geom_names(name: str = "object") -> tuple[str, ...]:
  """`get_pin_spec`'s two plates: ``{name}_geom`` (along x, first, so ``geom_ids[0]`` is a plate as
  for every object) and ``{name}_geom_y``."""
  return (f"{name}_geom", f"{name}_geom_y")


def get_pin_spec(
  name: str = "object",
  span: float = PIN_SPAN,
  thickness: float = PIN_THICKNESS,
  length: float = PIN_LENGTH,
  mass: float = PIN_MASS,
  friction: float = PIN_BLOCK_FRICTION,
  rgba: tuple[float, float, float, float] = (0.85, 0.3, 0.2, 1.0),
) -> mujoco.MjSpec:
  """The free cross pin (plan §0.2): two box plates, ``span`` x ``thickness`` in section, crossing on
  its axis, ``length`` long. **The body's origin is the pin's centre** (half-way along its axis), not
  the CAD's bottom face: every reader of an object's pose here takes the origin for its centre
  (`shapes.object_half_height` is then the half-length, and a resting pin's centre is that far above
  the surface, as a cube's). A deployer tracking the CAD mesh (origin at the bottom face) adds half the
  length along the pin's axis. Named as the cube (body ``name``, first geom ``{name}_geom``), so the
  grasp and support sensors match it; condim 4 as the cube (the pads' torsional friction)."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name=name)
  body.add_freejoint(name=f"{name}_joint")
  halves = (
    (span / 2, thickness / 2, length / 2),
    (thickness / 2, span / 2, length / 2),
  )
  for geom, half in zip(pin_geom_names(name), halves, strict=True):
    body.add_geom(
      name=geom,
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=half,
      mass=mass / 2,
      rgba=rgba,
      condim=4,
      friction=(friction, 0.02, 0.0001),
      solref=(0.02, 1.0),
      solimp=(0.99, 0.995, 0.001, 0.5, 2),
    )
  return spec


def get_pin_cfg(name: str = "object", **kwargs) -> EntityCfg:
  return EntityCfg(spec_fn=lambda: get_pin_spec(name=name, **kwargs))
