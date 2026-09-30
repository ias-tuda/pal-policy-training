"""SDF worlds for the PAL physics simulator: the training table, optionally a static cube.

The PAL simulator steps one articulated body (the robot); world objects are static collision
shapes. So the cube the policy picks up is simulated by the policy node (a virtual cube), and
these worlds only add what is static in the training scene:

* `table`: the training table's top slab (scene.TABLE_*), 0.14 m below base_link.
* `table_cube_epNN`: the table plus a *static* 5 cm box at episode NN's spawn pose. In MuJoCo
  the fingertips reach the resting cube at ~1.4 m/s and it is the cube, pinned on the table,
  that stops the arm (contact probe: no hand-table contact). A static box reproduces that
  arrival stop; it stays behind once the virtual cube has been picked up.

Poses are placed relative to the robot's standing base in the PAL simulator, measured from
/simulator/floating_base_state after it settles (the base is not fixed there).

  python3 make_worlds.py --out <dir> [--base x y z qx qy qz qw] [--profile pick_place_profile.yaml]

`--profile` (S7 models, `export_for_pal_s7.py`) takes the table height, table extent and spawn
centre from the model's profile instead of the S5.2 constants above, and the episodes'
offsets are from that spawn centre. The table's top is then 0.085 m below base_link
(baseline B, table at 0.815 m), the spawn centre (0.40, 0.27).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# pal_mjlab tasks/manipulation/scene.py and pick_place_env_cfg.py, in the training base frame
# (base_link at the origin; export_for_pal.py: table top -0.14, spawn centre (0.42, 0.27)).
TABLE_TOP_Z_B = -0.14
TABLE_CENTER_B = (0.55, 0.0)
TABLE_HALF = (0.30, 0.40)
TABLE_THICKNESS = 0.04
CUBE_HALF = 0.025
SPAWN_B = (0.42, 0.27)

GROUND = """    <model name='ground_plane'>
      <static>1</static>
      <link name='link'>
        <collision name='collision'>
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
          <surface><friction><ode><mu>100</mu><mu2>50</mu2></ode></friction></surface>
        </collision>
      </link>
    </model>
"""


def quat_to_R(qx, qy, qz, qw):
  return [
    [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
    [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
    [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
  ]


def to_world(base, p_b):
  origin, R = base[:3], quat_to_R(*base[3:])
  return [origin[i] + sum(R[i][k] * p_b[k] for k in range(3)) for i in range(3)]


def base_yaw(base):
  _, _, _, qx, qy, qz, qw = base
  return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def box_model(name, pose, size):
  return f"""    <model name='{name}'>
      <static>1</static>
      <pose>{" ".join(f"{v:.5f}" for v in pose)}</pose>
      <link name='link'>
        <collision name='collision'>
          <geometry><box><size>{" ".join(f"{v:.4f}" for v in size)}</size></box></geometry>
        </collision>
        <visual name='visual'>
          <geometry><box><size>{" ".join(f"{v:.4f}" for v in size)}</size></box></geometry>
        </visual>
      </link>
    </model>
"""


def world(models: str) -> str:
  return (
    "<?xml version='1.0'?>\n<sdf version='1.4'>\n  <world name='default'>\n"
    + GROUND
    + models
    + "  </world>\n</sdf>\n"
  )


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--out", type=Path, required=True)
  p.add_argument(
    "--base",
    type=float,
    nargs=7,
    # Standing base_link in the PAL simulator with the arms spawned at the training home
    # pose (kang_pick_place_simulator.launch), /simulator/floating_base_state, 2026-09-25.
    # With the locomotion arm pose it stands 0.9 mm further back and pitched 0.2 deg more.
    default=(
      0.0203826,
      0.0006047,
      0.9045220,
      0.00022349,
      -0.00050708,
      0.00139254,
      0.99999888,
    ),
  )
  p.add_argument(
    "--episodes", type=Path, default=Path(__file__).with_name("episodes.json")
  )
  p.add_argument("--profile", type=Path, default=None)
  args = p.parse_args()
  table_top_z, table_center, table_half, spawn = (
    TABLE_TOP_Z_B,
    TABLE_CENTER_B,
    TABLE_HALF,
    SPAWN_B,
  )
  if args.profile is not None:
    import yaml

    prof = yaml.safe_load(args.profile.read_text())
    tx, ty = prof["table_x_b"], prof["table_y_b"]
    table_top_z = prof["table_top_z_b"]
    table_center = ((tx[0] + tx[1]) / 2, (ty[0] + ty[1]) / 2)
    table_half = ((tx[1] - tx[0]) / 2, (ty[1] - ty[0]) / 2)
    spawn = tuple(prof["cube_spawn_center_b"][:2])
  args.out.mkdir(parents=True, exist_ok=True)
  base = tuple(args.base)
  yaw0 = base_yaw(base)

  top_c = to_world(
    base, (table_center[0], table_center[1], table_top_z - TABLE_THICKNESS / 2)
  )
  table = box_model(
    "table",
    (*top_c, 0.0, 0.0, yaw0),
    (2 * table_half[0], 2 * table_half[1], TABLE_THICKNESS),
  )
  (args.out / "table.world").write_text(world(table))
  meta = {
    "base_pose_xyz_qxyzw": base,
    "table_top_center_w": top_c,
    "worlds": ["table.world"],
  }
  for e in json.loads(args.episodes.read_text())["episodes"]:
    c = to_world(
      base, (spawn[0] + e["dx"], spawn[1] + e["dy"], table_top_z + CUBE_HALF)
    )
    cube = box_model(
      "static_cube", (*c, 0.0, 0.0, yaw0 + e["dyaw"]), (2 * CUBE_HALF,) * 3
    )
    name = f"table_cube_ep{e['id']:02d}.world"
    (args.out / name).write_text(world(table + cube))
    meta["worlds"].append(name)
  (args.out / "worlds.json").write_text(json.dumps(meta, indent=1) + "\n")
  print(json.dumps(meta, indent=1))


if __name__ == "__main__":
  main()
