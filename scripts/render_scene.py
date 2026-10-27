"""Render the tabletop scene offscreen, so a pose can be looked at rather than described.

S1.1 and S1.6 of the pick-and-place plan. Three poses are worth looking at:

``--pose start``
    ``qpos`` = the robot cfg's own initial joint positions, no dynamics. What the
    configuration *says* the robot starts as.
``--pose settled``
    the same, then stepped with the control the env actually applies: the action term writes
    its default target to the 16 arm/jaw actuators, and every other actuator is left at
    ``ctrl = 0``. What the robot *becomes*.
``--pose locomotion``
    ``INIT_STATE`` instead, which is where the tabletop tasks started before S1.2.

``--baseline`` renders the pre-S1.4 robot, and ``--colliders`` swaps the visual meshes for
the collision geometry, which is the only way to see what a collision decision actually did.

Views: the task's own viewer angle (``ViewerConfig`` in ``tabletop_env_cfg.py``: azimuth
140, elevation -15), a top-down plan view, and a close-up of the working area. The target
and the object spawn centre are drawn as overlay markers -- overlays, not model sites, so
nothing about the scene being rendered is changed to draw them.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from scene_model import (  # noqa: E402
  DEFAULT_TASK,
  compile_scene,
  geom_id,
  nominal_data,
)

#: 20 s of the task's own control rate: 50 Hz control, 2 ms physics, decimation 10.
SETTLE_CONTROL_STEPS = 1000
DECIMATION = 10

#: The 16 actuators the policy drives (``ManipulationRobotCfg.action_joint_regex``); every
#: other actuator is left at ctrl = 0, which is what the env does today.
ACTION_ACTUATORS = (r"arm_.*_[1-7]_joint", r"gripper_.*_finger_joint")


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--task", default=DEFAULT_TASK)
  p.add_argument("--pose", choices=("start", "settled", "locomotion"), default="start")
  p.add_argument(
    "--arm-pose",
    type=float,
    nargs=7,
    default=None,
    metavar="Q",
    help="left arm 1-7 joint targets for --pose manipulation",
  )
  p.add_argument(
    "--hold-nominal",
    action="store_true",
    help="with --pose settled, also write the nominal target to the actuators the action "
    "term does not touch -- i.e. what a hold_nominal_targets event would do",
  )
  p.add_argument(
    "--spawn", type=float, nargs=2, default=(0.42, 0.23), metavar=("X", "Y")
  )
  p.add_argument(
    "--target", type=float, nargs=2, default=(0.40, 0.07), metavar=("X", "Y")
  )
  p.add_argument(
    "--baseline",
    action="store_true",
    help="render the pre-S1.4 robot: no torso, upper-arm or elbow collider, locomotion pose",
  )
  p.add_argument(
    "--colliders",
    action="store_true",
    help="draw the collision geometry instead of the visual meshes",
  )
  p.add_argument("--width", type=int, default=1100)
  p.add_argument("--height", type=int, default=800)
  p.add_argument("--tag", default=None, help="file name suffix; defaults to --pose")
  p.add_argument(
    "--out", type=pathlib.Path, default=pathlib.Path("../docs/reports/pick-place-v0")
  )
  return p.parse_args()


def action_actuator_ids(model: mujoco.MjModel) -> np.ndarray:
  import re

  ids = []
  for a in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
    bare = name.split("/", 1)[-1]
    if any(re.fullmatch(pattern, bare) for pattern in ACTION_ACTUATORS):
      ids.append(a)
  return np.asarray(ids, dtype=int)


def actuator_default_ctrl(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
  """The position target each actuator would hold to keep its joint where it is now."""
  ctrl = np.zeros(model.nu)
  for a in range(model.nu):
    jid = int(model.actuator_trnid[a, 0])
    ctrl[a] = data.qpos[model.jnt_qposadr[jid]]
  return ctrl


def table_top_z(model: mujoco.MjModel, data: mujoco.MjData) -> float:
  """World z of the table's top face, read out of data like ``mdp.events.surface_top_z``."""
  top = geom_id(model, "table/table_top")
  return float(data.geom_xpos[top][2] + model.geom_size[top][2])


def place_object(model: mujoco.MjModel, data: mujoco.MjData, xy) -> None:
  """Drop the manipulated cube onto the table at ``xy``, as the reset event does.

  Requires ``mj_forward`` to have run: the table's height lives in the compiled world pose,
  not in ``body_pos``, which is the same reason ``reset_objects_on_surface`` reads it from
  ``geom_xpos`` rather than from the ``TABLE_POS`` constant.
  """
  jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object/object_joint")
  if jid < 0:
    return
  adr = model.jnt_qposadr[jid]
  half = model.geom_size[geom_id(model, "object/object_geom")][2]
  data.qpos[adr : adr + 3] = (xy[0], xy[1], table_top_z(model, data) + half)
  data.qpos[adr + 3 : adr + 7] = (1.0, 0.0, 0.0, 0.0)


def build(args: argparse.Namespace) -> tuple[mujoco.MjModel, mujoco.MjData]:
  from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import INIT_STATE

  model = compile_scene(args.task, baseline=args.baseline)
  init = INIT_STATE if (args.pose == "locomotion" or args.baseline) else None
  data = nominal_data(model, init)

  mujoco.mj_forward(model, data)
  place_object(model, data, args.spawn)
  mujoco.mj_forward(model, data)

  if args.pose == "settled":
    hold = actuator_default_ctrl(model, data)
    data.ctrl[:] = 0.0
    driven = action_actuator_ids(model)
    for _ in range(SETTLE_CONTROL_STEPS):
      # The action term writes target = default + scale * action every control step; with
      # zero actions that is just the default. Nothing writes the other actuators.
      data.ctrl[driven] = hold[driven]
      if args.hold_nominal:
        data.ctrl[:] = hold
      for _ in range(DECIMATION):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
  return model, data


def marker(
  scene: mujoco.MjvScene, pos, size, rgba, kind=mujoco.mjtGeom.mjGEOM_SPHERE
) -> None:
  if scene.ngeom >= scene.maxgeom:
    return
  g = scene.geoms[scene.ngeom]
  mujoco.mjv_initGeom(
    g,
    kind,
    np.asarray(size, float),
    np.asarray(pos, float),
    np.eye(3).flatten(),
    np.asarray(rgba, np.float32),
  )
  scene.ngeom += 1


def overlays(scene: mujoco.MjvScene, args: argparse.Namespace, rest_z: float) -> None:
  # Target: a translucent cube the size of the object, at its resting pose on the table.
  marker(
    scene,
    (args.target[0], args.target[1], rest_z),
    (0.025, 0.025, 0.025),
    (0.42, 0.27, 0.76, 0.45),
    mujoco.mjtGeom.mjGEOM_BOX,
  )
  # Spawn centre, and the +/- 2 cm box the cube is sampled in.
  marker(
    scene,
    (args.spawn[0], args.spawn[1], rest_z + 0.06),
    (0.012,) * 3,
    (0.9, 0.1, 0.1, 0.9),
  )
  marker(
    scene,
    (args.spawn[0], args.spawn[1], rest_z - 0.024),
    (0.02, 0.02, 0.001),
    (0.9, 0.1, 0.1, 0.5),
    mujoco.mjtGeom.mjGEOM_BOX,
  )


def views(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, mujoco.MjvCamera]:
  pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/pelvis_2_link")
  table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table/table")
  out = {}

  # The task's own viewer: ViewerConfig(distance=1.8, elevation=-15, azimuth=140).
  cam = mujoco.MjvCamera()
  cam.lookat[:] = data.xpos[pelvis]
  cam.distance, cam.elevation, cam.azimuth = 1.8, -15.0, 140.0
  out["viewer"] = cam

  # From in front of the robot, over the table: the view that shows whether the hand is
  # anywhere near the working area.
  cam = mujoco.MjvCamera()
  cam.lookat[:] = (0.34, 0.02, 0.86)
  cam.distance, cam.elevation, cam.azimuth = 2.6, -20.0, 48.0
  out["front"] = cam

  # Plan view. Azimuth 180 puts +x down the image and +y to the right.
  cam = mujoco.MjvCamera()
  cam.lookat[:] = (0.35, 0.0, 0.76)
  cam.distance, cam.elevation, cam.azimuth = 1.9, -89.0, 180.0
  out["top"] = cam

  # Close-up of the working area, from the robot's front left.
  cam = mujoco.MjvCamera()
  cam.lookat[:] = data.xpos[table] + np.array([-0.14, 0.12, 0.08])
  cam.distance, cam.elevation, cam.azimuth = 1.7, -30.0, 52.0
  out["work"] = cam
  return out


def main() -> None:
  args = parse_args()
  args.out.mkdir(parents=True, exist_ok=True)
  tag = args.tag or args.pose

  model, data = build(args)
  rest_z = (
    table_top_z(model, data) + model.geom_size[geom_id(model, "object/object_geom")][2]
  )

  option = mujoco.MjvOption()
  option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
  if args.colliders:
    # Group 2 is the visual meshes, group 3 the colliders (see the `visual` / `collision`
    # default classes in kangaroo_grippers.xml). Group 3 also holds the colliders the
    # variant switched *off* -- the legs, for one -- and drawing those would make the
    # picture claim a collision model the robot does not have, so move them out of sight.
    option.geomgroup[2] = 0
    option.geomgroup[3] = 1
    for gid in range(model.ngeom):
      if model.geom_group[gid] == 3 and not (
        model.geom_contype[gid] or model.geom_conaffinity[gid]
      ):
        model.geom_group[gid] = 4
  with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
    from PIL import Image

    for name, cam in views(model, data).items():
      renderer.update_scene(data, camera=cam, scene_option=option)
      overlays(renderer.scene, args, rest_z)
      path = args.out / f"s1_scene_{tag}_{name}.png"
      Image.fromarray(renderer.render()).save(path)
      print(f"wrote {path}")

  # A one-line summary of what is being looked at, so the picture is never orphaned.
  tcp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "robot/tcp_left")
  print(
    f"pose={args.pose}  grasp point={data.site_xpos[tcp].round(3)}  "
    f"ncon={data.ncon}  object z={data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'object/object_joint')] + 2]:.3f}"
  )


if __name__ == "__main__":
  main()
