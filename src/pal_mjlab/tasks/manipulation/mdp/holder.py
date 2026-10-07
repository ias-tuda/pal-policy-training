"""Plan §13's held block (`pick-place-insert-plan.md` §13 *Structure*, R-I): the socket block held at H by a stand-in
for the left hand -- a mocap "holder" carrying boxes where the left gripper sits around the block, and a soft weld from
the holder to the block -- so the right's insertion can be trained on one arm without a left policy.

The holder is a fixed-base entity (mjlab wraps it in a mocap body, `utils.spec.auto_wrap_fixed_base_mocap`); its
body frame is the block's at the hold (origin at the block's bottom-face centre, +x the left hand's pointing
direction). `held_block_weld` adds the equality and the holder-block contact exclusion to the scene's spec; the
weld's relative pose is the identity, given explicitly (not MuJoCo's default "from the initial configuration": plan
§0.4's v1 table read a press off exactly that default). `reset_held_block` draws the hold about H per reset and puts
the block on it, so the weld starts satisfied.
"""

from __future__ import annotations

import mujoco
import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

from pal_mjlab.tasks.manipulation.mdp.events import surface_top_z
from pal_mjlab.tasks.manipulation.mdp.insert import insert_state
from pal_mjlab.tasks.manipulation.mdp.observations import _grasped_bool


def holder_spec(
  name: str,
  boxes: tuple[tuple[str, tuple[float, float, float], tuple[float, float, float]], ...],
  rgba: tuple[float, float, float, float] = (0.3, 0.45, 0.85, 0.5),
) -> mujoco.MjSpec:
  """A body ``name`` with no joint (so a fixed-base, mocap-wrapped entity) carrying ``boxes`` -- (geom suffix,
  centre, half-size), m, in the holder's frame -- as colliding geoms ``{name}_{suffix}`` (contype / conaffinity 1:
  the robot and the pin touch them)."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name=name)
  for suffix, centre, half in boxes:
    body.add_geom(
      name=f"{name}_{suffix}",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=centre,
      size=half,
      rgba=rgba,
      contype=1,
      conaffinity=1,
      condim=3,
    )
  return spec


def held_block_weld(
  holder_body: str,
  block_body: str,
  solref: tuple[float, float],
  torquescale: float = 1.0,
):
  """A scene ``spec_fn``: weld ``block_body`` to ``holder_body`` at the identity relative pose with ``solref`` (the
  hold's softness) and ``torquescale`` (MuJoCo's weight of the weld's angular residual against its linear one: the
  block's tiny rotational inertia otherwise makes a weld soft enough to give a few mm turn it tens of degrees), and
  exclude their contacts (the holder's fingers sit on the block's faces)."""

  def edit(spec: mujoco.MjSpec) -> None:
    spec.add_equality(
      type=mujoco.mjtEq.mjEQ_WELD,
      objtype=mujoco.mjtObj.mjOBJ_BODY,
      name1=holder_body,
      name2=block_body,
      data=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, float(torquescale)],
      solref=list(solref),
      name=f"{block_body.split('/')[0]}_hold",
    )
    spec.add_exclude(bodyname1=holder_body, bodyname2=block_body)

  return edit


def chain_spec_fns(*fns):
  """Compose scene ``spec_fn``s (None skipped), in order."""
  fns = tuple(f for f in fns if f is not None)

  def edit(spec: mujoco.MjSpec) -> None:
    for f in fns:
      f(spec)

  return edit


def reset_held_block(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  holder_cfg: SceneEntityCfg,
  block_cfg: SceneEntityCfg,
  surface_cfg: SceneEntityCfg,
  xy: tuple[float, float],
  dz: float,
  half_xy: float,
  half_z: float,
  yaw: float,
  tilt: float,
) -> None:
  """Reset event: the hold drawn about H -- (``xy``, the surface top + ``dz``) in the env frame (the robot frame),
  +/- ``half_xy`` / ``half_z`` m, yaw +/- ``yaw``, tilt (roll and pitch each) +/- ``tilt`` rad -- written to the
  holder's mocap and to the block's free joint, at rest."""
  n = len(env_ids)
  dev = env.device
  holder: Entity = env.scene[holder_cfg.name]
  block: Entity = env.scene[block_cfg.name]
  top = surface_top_z(env, surface_cfg)[env_ids]
  pos = env.scene.env_origins[env_ids].clone()
  pos[:, 0] += xy[0] + sample_uniform(-half_xy, half_xy, (n,), device=dev)
  pos[:, 1] += xy[1] + sample_uniform(-half_xy, half_xy, (n,), device=dev)
  pos[:, 2] = top + dz + sample_uniform(-half_z, half_z, (n,), device=dev)
  roll = sample_uniform(-tilt, tilt, (n,), device=dev)
  pitch = sample_uniform(-tilt, tilt, (n,), device=dev)
  psi = sample_uniform(-yaw, yaw, (n,), device=dev)
  zeros = torch.zeros(n, device=dev)
  quat = quat_mul(
    quat_from_euler_xyz(zeros, zeros, psi), quat_from_euler_xyz(roll, pitch, zeros)
  )
  pose = torch.cat([pos, quat], dim=-1)
  holder.write_mocap_pose_to_sim(pose, env_ids=env_ids)
  block.write_root_link_pose_to_sim(pose, env_ids=env_ids)
  block.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)


def held_block_offset(
  env: ManagerBasedRlEnv, holder_name: str, block_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
  """How far the block sits from its hold: position (m) and angle (rad). (B,), (B,)"""
  holder: Entity = env.scene[holder_name]
  block: Entity = env.scene[block_name]
  hp, hq = holder.data.root_link_pos_w, holder.data.root_link_quat_w
  bp, bq = block.data.root_link_pos_w, block.data.root_link_quat_w
  dot = (hq * bq).sum(-1).abs().clamp(max=1.0)
  return torch.norm(bp - hp, dim=-1), 2.0 * torch.acos(dot)


def held_block_moved_penalty(
  env: ManagerBasedRlEnv, holder_name: str, block_name: str, scale: float
) -> torch.Tensor:
  """S10's ``box_moved`` for a held block: its displacement from the hold as a fraction of ``scale`` (m), capped at
  1. Weight it negative. (B,)"""
  d, _ = held_block_offset(env, holder_name, block_name)
  return (d / scale).clamp(0.0, 1.0)


def pin_through_held(
  env: ManagerBasedRlEnv,
  object_name: str,
  block_name: str,
  block_height: float,
  hole_d: float,
  grasp_sensor_name: str,
) -> torch.Tensor:
  """1 while the pin is held with its bottom at or below the block's bottom face, inside the bore (plan §13's
  "through", `mdp.InsertState.through`). (B,)"""
  st = insert_state(env, object_name, block_name, block_height, hole_d)
  return (st.through() & _grasped_bool(env, grasp_sensor_name)).float()


def pin_through_obs(env: ManagerBasedRlEnv, **params) -> torch.Tensor:
  """`pin_through_held`, for the critic. (B, 1)"""
  return pin_through_held(env, **params).unsqueeze(-1)
