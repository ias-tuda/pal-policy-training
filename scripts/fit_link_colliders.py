"""Fit a primitive collision proxy to each link's visual meshes, in the link's own frame.

The source of the numbers in `_TABLETOP_ARM_BOXES` and `_TABLETOP_TORSO_BOX`
(`kangaroo_constants.py`). Re-run it if PAL ships new meshes.

Boxes, not capsules. A capsule that *contains* a link has to put a full hemisphere beyond
each end of the mesh, and on the upper arm that hemisphere is 74 mm deep and points straight
at the chest: with capsules the shoulder reports contact with the torso in 46% of sampled arm
configurations, against a real mesh-to-mesh distance that is usually 10 cm or more. The
link-frame bounding box has no such overhang and is 2.4x smaller by volume.
"""

from __future__ import annotations

import pathlib
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from scene_model import compile_scene, nominal_data  # noqa: E402

LINKS = (
  "pelvis_2_link",
  "arm_left_2_link",
  "arm_left_3_link",
  "arm_left_4_link",
  "arm_right_2_link",
  "arm_right_3_link",
  "arm_right_4_link",
)


def link_mesh_vertices(
  model: mujoco.MjModel, data: mujoco.MjData, body_name: str
) -> np.ndarray:
  """Every visual-mesh vertex of a body, expressed in that body's own frame.

  Goes through the world pose rather than through ``mesh_pos``/``mesh_quat``: MuJoCo
  recentres mesh vertices at compile time, and the world transform is the one place the
  compiled result is unambiguous.
  """
  bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
  rot, pos = data.xmat[bid].reshape(3, 3), data.xpos[bid]
  out = []
  for gid in range(
    model.body_geomadr[bid], model.body_geomadr[bid] + model.body_geomnum[bid]
  ):
    if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
      continue
    mesh = model.geom_dataid[gid]
    adr, num = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    vert = model.mesh_vert[adr : adr + num].reshape(-1, 3)
    world = data.geom_xpos[gid] + vert @ data.geom_xmat[gid].reshape(3, 3).T
    out.append((world - pos) @ rot)
  return np.concatenate(out)


def main() -> None:
  model = compile_scene()
  data = nominal_data(model)
  print(f"{'link':20s} {'centre (m)':>26s} {'half-size (m)':>26s}   volume")
  for link in LINKS:
    vert = link_mesh_vertices(model, data, f"robot/{link}")
    lo, hi = vert.min(0), vert.max(0)
    centre, half = (hi + lo) / 2, (hi - lo) / 2
    print(
      f"{link:20s} "
      f"({centre[0]:7.4f}, {centre[1]:7.4f}, {centre[2]:7.4f}) "
      f"({half[0]:7.4f}, {half[1]:7.4f}, {half[2]:7.4f})   "
      f"{8 * half.prod() * 1e3:6.2f} L   from {len(vert)} vertices"
    )


if __name__ == "__main__":
  main()
