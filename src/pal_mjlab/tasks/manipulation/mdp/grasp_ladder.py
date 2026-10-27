"""The grasp ladder of the mug tasks (``mug/``, ``coaster/``, ``kugel/``), on this package's
controller.

Claudius's standing recipe (``faster-moves``) scored the jaw on the midpoint of its pads, not on
the TCP site, and paid the grasp up a ladder -- reach, an open jaw straddling the object
(``jaw_aligned_at_object``), the jaw closing on it (``jaw_closing_on_object``), the grasp, the
lift to the goal. The mug, coaster and kugel rungs (``mug_rewards``, ``coaster_rewards``,
``kugel_rewards``) sit on top of it. This module carries that ladder over for the one left arm of
``robots.kangaroo_grippers_left_arm``: the per-episode arm assignment is gone (one gripper,
``G = 1``), and the tracker is ``observations._object_pose_w``, lag only.

A grasp here is the two jaw *faces* each touching the object by any of their geoms, read off
two sensors per gripper (``tabletop_env_cfg.make_grasp_sensor`` watches the fingertip pads
alone, which a mug held across the body by the inner fingers never reaches). The sensors come
in pairs, ``((face_a, face_b), ...)``, one pair per gripper.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
  euler_xyz_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
)

from pal_mjlab.tasks.manipulation.mdp.commands import (
  EntityOffsetCommand,
  EntityOffsetCommandCfg,
)
from pal_mjlab.tasks.manipulation.mdp.events import _surface_centre_xy, surface_top_z
from pal_mjlab.tasks.manipulation.mdp.observations import (
  _base_pose,
  _object_pose_w,
  _on_surface_bool,
  _to_base_frame,
)
from pal_mjlab.tasks.manipulation.mdp.occlusion import camera_view_blocked

_ROBOT = SceneEntityCfg("robot")


# -- Grasp state --------------------------------------------------------------------------


def _face_touching(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """True where this jaw face touches the object anywhere. (B,)"""
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None, f"sensor {sensor_name!r} does not provide 'found'"
  # (B, n_face_geoms * num_slots): any flag means the face is in contact.
  return found.bool().any(dim=-1)


def _grasped_pair_bool(
  env: ManagerBasedRlEnv, sensor_names: Sequence[str]
) -> torch.Tensor:
  """True where one gripper holds the object between both of its jaw faces. (B,)"""
  a, b = sensor_names
  return _face_touching(env, a) & _face_touching(env, b)


def _grasped_any_bool(
  env: ManagerBasedRlEnv, sensor_names: Sequence[Sequence[str]]
) -> torch.Tensor:
  """True where any gripper holds the object. (B,)"""
  held = _grasped_pair_bool(env, sensor_names[0])
  for pair in sensor_names[1:]:
    held = held | _grasped_pair_bool(env, pair)
  return held


def grasped_pair(
  env: ManagerBasedRlEnv, sensor_names: Sequence[Sequence[str]]
) -> torch.Tensor:
  """1.0 while a gripper holds the object between both faces. (B,)"""
  return _grasped_any_bool(env, sensor_names).float()


def object_lifted_pair(
  env: ManagerBasedRlEnv,
  grasp_sensor_names: Sequence[Sequence[str]],
  support_sensor_name: str,
) -> torch.Tensor:
  """1.0 once the object is held *and* off its support surface. (B,)"""
  return (
    _grasped_any_bool(env, grasp_sensor_names)
    & ~_on_surface_bool(env, support_sensor_name)
  ).float()


def object_lifted_to_goal_pair(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  grasp_sensor_names: Sequence[Sequence[str]],
  support_sensor_name: str,
  std: float,
) -> torch.Tensor:
  """``object_lifted_pair``, scaled by ``exp(-d^2 / std^2)`` of the object's distance to the
  commanded goal. (B,)"""
  obj: Entity = env.scene[object_name]
  command = env.command_manager.get_term(command_name)
  error = torch.sum(torch.square(command.command - obj.data.root_link_pos_w), dim=-1)
  lifted = object_lifted_pair(env, grasp_sensor_names, support_sensor_name)
  return lifted * torch.exp(-error / std**2)


def bring_object_while_held_pair(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  grasp_sensor_names: Sequence[Sequence[str]],
  std: float,
) -> torch.Tensor:
  """``bring_object_reward``, only while a gripper holds the object: ungated it pays an
  object the policy never touches. (B,)"""
  obj: Entity = env.scene[object_name]
  command = env.command_manager.get_term(command_name)
  error = torch.sum(torch.square(command.command - obj.data.root_link_pos_w), dim=-1)
  held = _grasped_any_bool(env, grasp_sensor_names).float()
  return torch.exp(-error / std**2) * held


# -- The jaw frame ------------------------------------------------------------------------


def _jaw_frame(
  robot: Entity, grasp_cfg: SceneEntityCfg
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """The grasp centre, closing axis and aperture of every gripper.

  ``grasp_cfg`` names the two pad sites of each gripper, in pairs (``preserve_order=True``).
  The pad midpoint, not the TCP site, is where a parallel jaw grasps: on Kangaroo ``tcp_left``
  sits 27 mm behind it with the jaw open and 43 mm closed, all along the approach axis.

  Returns ``(mid (B, G, 3), axis (B, G, 3), aperture (B, G))``.
  """
  pads = robot.data.site_pos_w[:, grasp_cfg.site_ids]  # (B, 2G, 3)
  if pads.shape[1] % 2 != 0:
    raise ValueError(
      f"grasp_cfg must name two pad sites per gripper, got {pads.shape[1]}: "
      f"{grasp_cfg.site_names}"
    )
  pads = pads.reshape(pads.shape[0], -1, 2, 3)
  face_a, face_b = pads[:, :, 0], pads[:, :, 1]
  delta = face_b - face_a
  aperture = torch.norm(delta, dim=-1)
  axis = delta / aperture.clamp_min(1e-6).unsqueeze(-1)
  return 0.5 * (face_a + face_b), axis, aperture


def _jaw_alignment(
  mid: torch.Tensor,
  axis: torch.Tensor,
  obj_pos_w: torch.Tensor,
  lateral_std: float,
  approach_std: float,
) -> torch.Tensor:
  """How well each jaw straddles the object, in ``[0, 1]``. (B, G)

  Anisotropic: off along the closing axis the pads miss the object, off along the approach
  the hand is only not deep enough yet.
  """
  offset = obj_pos_w.unsqueeze(1) - mid  # (B, G, 3)
  lateral = torch.sum(offset * axis, dim=-1)
  approach = torch.norm(offset - lateral.unsqueeze(-1) * axis, dim=-1)
  return torch.exp(-((lateral / lateral_std) ** 2)) * torch.exp(
    -((approach / approach_std) ** 2)
  )


def _on_base(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  stack_contact_sensor_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Whether the object touches its base, and its xy and z error to the goal. (B,) x3"""
  obj: Entity = env.scene[object_name]
  command = env.command_manager.get_term(command_name)
  error = command.command - obj.data.root_link_pos_w
  contact = env.scene[stack_contact_sensor_name].data.found
  assert contact is not None
  return contact.bool().any(dim=-1), torch.norm(error[:, :2], dim=-1), error[:, 2].abs()


# -- The ladder ---------------------------------------------------------------------------


def staged_position_reward_jaw(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  reaching_std: float,
  bringing_std: float,
  grasp_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """mjlab's ``staged_position_reward``, ``reaching * (1 + bringing)``, with the reach
  measured from the pad midpoint (best gripper) instead of the TCP site. (B,)"""
  robot: Entity = env.scene[grasp_cfg.name]
  obj_pos_w = env.scene[object_name].data.root_link_pos_w
  command = env.command_manager.get_term(command_name)
  mid, _, _ = _jaw_frame(robot, grasp_cfg)
  reach_error = torch.sum(torch.square(mid - obj_pos_w.unsqueeze(1)), dim=-1)
  reaching = torch.exp(-reach_error / reaching_std**2).amax(dim=-1)
  position_error = torch.sum(torch.square(command.command - obj_pos_w), dim=-1)
  return reaching * (1.0 + torch.exp(-position_error / bringing_std**2))


def jaw_aligned_at_object(
  env: ManagerBasedRlEnv,
  object_name: str,
  grasp_cfg: SceneEntityCfg,
  object_width: float,
  lateral_std: float = 0.02,
  approach_std: float = 0.03,
  admit_margin: float = 0.005,
  admit_slack: float = 0.010,
) -> torch.Tensor:
  """An *open* jaw around the object: alignment times a sigmoid on the aperture, centred
  ``admit_slack`` below the object width so closing onto it is not a loss. (B,)"""
  robot: Entity = env.scene[grasp_cfg.name]
  obj: Entity = env.scene[object_name]
  mid, axis, aperture = _jaw_frame(robot, grasp_cfg)
  align = _jaw_alignment(mid, axis, obj.data.root_link_pos_w, lateral_std, approach_std)
  admits = torch.sigmoid((aperture - (object_width - admit_slack)) / admit_margin)
  return (align * admits).amax(dim=-1)


def jaw_closing_on_object(
  env: ManagerBasedRlEnv,
  object_name: str,
  grasp_cfg: SceneEntityCfg,
  object_width: float,
  lateral_std: float = 0.015,
  approach_std: float = 0.02,
  gap_std: float = 0.015,
) -> torch.Tensor:
  """The jaw closing *onto* a well-placed object: alignment (tighter than the open rung)
  times ``exp(-gap^2 / gap_std^2)`` of the aperture's surplus over the object width. (B,)"""
  robot: Entity = env.scene[grasp_cfg.name]
  obj: Entity = env.scene[object_name]
  mid, axis, aperture = _jaw_frame(robot, grasp_cfg)
  align = _jaw_alignment(mid, axis, obj.data.root_link_pos_w, lateral_std, approach_std)
  gap = (aperture - object_width).clamp_min(0.0)
  return (align * torch.exp(-((gap / gap_std) ** 2))).amax(dim=-1)


# -- The place ladder (the coaster) -------------------------------------------------------


def object_held_on_base(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  grasp_sensor_names: Sequence[Sequence[str]],
  stack_contact_sensor_name: str,
  xy_std: float = 0.01,
) -> torch.Tensor:
  """The held object resting on its base, scored on how well it is centred. (B,)"""
  touching, xy, _ = _on_base(env, command_name, object_name, stack_contact_sensor_name)
  held = _grasped_any_bool(env, grasp_sensor_names)
  return (held & touching).float() * torch.exp(-torch.square(xy / xy_std))


def release_on_base(
  env: ManagerBasedRlEnv,
  object_name: str,
  command_name: str,
  stack_contact_sensor_name: str,
  grasp_cfg: SceneEntityCfg,
  object_width: float,
  xy_std: float = 0.015,
  open_margin: float = 0.01,
) -> torch.Tensor:
  """An open jaw (``open_margin`` wider than the object) while the object rests on its
  base, centred. (B,)"""
  touching, xy, _ = _on_base(env, command_name, object_name, stack_contact_sensor_name)
  _, _, aperture = _jaw_frame(env.scene[grasp_cfg.name], grasp_cfg)
  opening = torch.sigmoid((aperture[:, 0] - object_width - open_margin) / 0.005)
  return touching.float() * torch.exp(-torch.square(xy / xy_std)) * opening


class object_stacked_set_down:
  """The object let go on its base after being set down on it in the jaw. (B,)

  "Set down" latches while the jaw holds the object on its base within ``xy_threshold`` of
  the goal and clears when the object leaves the base, so a dropped object does not count.
  With ``clearance`` the jaw also has to be that far from the object.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self.set_down = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.set_down[slice(None) if env_ids is None else env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    object_name: str,
    grasp_sensor_names: Sequence[Sequence[str]],
    stack_contact_sensor_name: str,
    grasp_cfg: SceneEntityCfg,
    clearance: float | None = None,
    xy_threshold: float = 0.02,
    z_threshold: float = 0.02,
  ) -> torch.Tensor:
    touching, xy, z = _on_base(
      env, command_name, object_name, stack_contact_sensor_name
    )
    held = _grasped_any_bool(env, grasp_sensor_names)
    at_goal = (xy < xy_threshold) & (z < z_threshold)
    self.set_down = (self.set_down | (held & touching & at_goal)) & touching
    ok = self.set_down & touching & at_goal & ~held
    if clearance is not None:
      obj: Entity = env.scene[object_name]
      mid, _, _ = _jaw_frame(env.scene[grasp_cfg.name], grasp_cfg)
      ok &= torch.norm(mid[:, 0] - obj.data.root_link_pos_w, dim=-1) > clearance
    return ok.float()


# -- Contacts and the camera --------------------------------------------------------------


def contact_force_excess(
  env: ManagerBasedRlEnv, sensor_name: str, threshold: float, cap: float
) -> torch.Tensor:
  """Newtons by which a contact pushes harder than ``threshold``, at most ``cap``, peak over
  the sensor's history, summed over primaries. (B,)"""
  history = env.scene[sensor_name].data.force_history
  assert history is not None, f"sensor {sensor_name!r} needs 'force' and a history"
  peak = history.norm(dim=-1).amax(dim=-1)  # (B, P)
  return (peak - threshold).clamp(0.0, cap).sum(dim=-1)


def self_collision(env: ManagerBasedRlEnv, sensor_names: Sequence[str]) -> torch.Tensor:
  """How many watched contact slots are touching right now. (B,)"""
  total = torch.zeros(env.num_envs, device=env.device)
  for name in sensor_names:
    found = env.scene[name].data.found
    assert found is not None, f"sensor {name!r} does not provide 'found'"
    total += (found > 0).float().flatten(start_dim=1).sum(dim=-1)
  return total


def object_view_blocked(
  env: ManagerBasedRlEnv,
  camera_body: str,
  camera_pos: tuple[float, float, float],
  segments: Sequence,
  object_name: str,
  stop_short: float,
  margin: float,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """``occlusion.camera_view_blocked`` on the object's true centre. (B,)"""
  return camera_view_blocked(
    env,
    camera_body,
    camera_pos,
    segments,
    env.scene[object_name].data.root_link_pos_w,
    stop_short,
    margin,
    asset_cfg,
  )


# -- Observations -------------------------------------------------------------------------


def _jaw_frame_b(
  robot: Entity, grasp_cfg: SceneEntityCfg
) -> tuple[torch.Tensor, torch.Tensor]:
  """Pad midpoint and closing axis of every gripper, in the base frame. ((B, G, 3) x2)"""
  base_pos, base_quat = _base_pose(robot)
  pads = robot.data.site_pos_w[:, grasp_cfg.site_ids]  # (B, 2G, 3)
  b, n = pads.shape[0], pads.shape[1] // 2
  quat = base_quat.unsqueeze(1).expand(b, 2 * n, 4)
  pads_b = quat_apply_inverse(
    quat.reshape(-1, 4), (pads - base_pos.unsqueeze(1)).reshape(-1, 3)
  ).reshape(b, n, 2, 3)
  delta = pads_b[:, :, 1] - pads_b[:, :, 0]
  axis = delta / torch.norm(delta, dim=-1, keepdim=True).clamp_min(1e-6)
  return pads_b.mean(dim=2), axis


def jaw_to_object_b(
  env: ManagerBasedRlEnv,
  object_name: str,
  grasp_cfg: SceneEntityCfg,
  asset_cfg: SceneEntityCfg = _ROBOT,
  lag: tuple[int, int] | None = None,
) -> torch.Tensor:
  """Vector from each jaw's pad midpoint to the object, in the base frame. (B, 3 * G)

  The jaw side is read live from the encoders; only the object carries the tracker ``lag``.
  """
  robot: Entity = env.scene[asset_cfg.name]
  pos_w, _ = _object_pose_w(env, object_name, lag)
  mid, _ = _jaw_frame_b(robot, grasp_cfg)
  rel = _to_base_frame(robot, pos_w).unsqueeze(1) - mid
  return rel.reshape(rel.shape[0], -1)


def object_yaw_to_jaw(
  env: ManagerBasedRlEnv,
  object_name: str,
  grasp_cfg: SceneEntityCfg,
  asset_cfg: SceneEntityCfg = _ROBOT,
  lag: tuple[int, int] | None = None,
) -> torch.Tensor:
  """Object yaw relative to each jaw's closing axis, ``[cos 4d, sin 4d]``: a cube's faces
  repeat every 90 deg. (B, 2 * G)"""
  robot: Entity = env.scene[asset_cfg.name]
  _, quat_w = _object_pose_w(env, object_name, lag)
  _, axis = _jaw_frame_b(robot, grasp_cfg)
  axis_yaw = torch.atan2(axis[..., 1], axis[..., 0])  # (B, G)
  quat_b = quat_mul(quat_inv(_base_pose(robot)[1]), quat_w)
  _, _, obj_yaw = euler_xyz_from_quat(quat_b)
  d = 4.0 * (obj_yaw.unsqueeze(1) - axis_yaw)
  return torch.stack([torch.cos(d), torch.sin(d)], dim=-1).reshape(d.shape[0], -1)


def object_to_goal_b(
  env: ManagerBasedRlEnv,
  object_name: str,
  command_name: str,
  asset_cfg: SceneEntityCfg = _ROBOT,
  lag: tuple[int, int] | None = None,
) -> torch.Tensor:
  """Vector from the object to the commanded goal, in the base frame. (B, 3)"""
  robot: Entity = env.scene[asset_cfg.name]
  pos_w, _ = _object_pose_w(env, object_name, lag)
  goal_w = env.command_manager.get_term(command_name).command
  return _to_base_frame(robot, goal_w) - _to_base_frame(robot, pos_w)


def object_rot6d_b(
  env: ManagerBasedRlEnv,
  object_name: str,
  asset_cfg: SceneEntityCfg = _ROBOT,
  lag: tuple[int, int] | None = None,
) -> torch.Tensor:
  """Object orientation in the base frame, as the tracker reports it: its x and z axes,
  continuous over SO(3). A mug's x axis is its handle, its z axis its body. (B, 6)"""
  robot: Entity = env.scene[asset_cfg.name]
  _, quat_w = _object_pose_w(env, object_name, lag)
  quat_b = quat_mul(quat_inv(_base_pose(robot)[1]), quat_w)
  x = quat_apply(
    quat_b, torch.tensor([1.0, 0.0, 0.0], device=env.device).expand_as(quat_b[:, :3])
  )
  z = quat_apply(
    quat_b, torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(quat_b[:, :3])
  )
  return torch.cat([x, z], dim=-1)


# -- Terminations -------------------------------------------------------------------------


def entity_outside_surface_box(
  env: ManagerBasedRlEnv,
  entity_name: str,
  surface_cfg: SceneEntityCfg,
  box: dict[str, tuple[float, float]],
) -> torch.Tensor:
  """Failure: the entity's centre left an ``x``/``y`` box relative to the surface centre --
  the part of the table the real camera sees."""
  entity: Entity = env.scene[entity_name]
  xy = entity.data.root_link_pos_w[:, :2] - _surface_centre_xy(env, surface_cfg)
  lo = torch.tensor([box["x"][0], box["y"][0]], device=env.device)
  hi = torch.tensor([box["x"][1], box["y"][1]], device=env.device)
  return ((xy < lo) | (xy > hi)).any(dim=-1)


def object_dropped_free(
  env: ManagerBasedRlEnv,
  object_name: str,
  grasp_sensor_names: Sequence[Sequence[str]],
  support_sensor_name: str,
  fall_speed: float = 0.5,
) -> torch.Tensor:
  """Failure: the object falls -- touching no jaw face, off the table, moving down faster
  than ``fall_speed``. Any face counts: slipping in the jaw, the mug loses one face for a
  few steps while the other still carries it."""
  obj: Entity = env.scene[object_name]
  on_table = _on_surface_bool(env, support_sensor_name)
  touched = torch.zeros_like(on_table)
  for pair in grasp_sensor_names:
    for face in pair:
      touched = touched | _face_touching(env, face)
  falling = obj.data.root_link_lin_vel_w[:, 2] < -fall_speed
  return ~touched & ~on_table & falling


def object_tipped_free(
  env: ManagerBasedRlEnv,
  object_name: str,
  grasp_sensor_names: Sequence[Sequence[str]],
  support_sensor_name: str,
  limit_angle: float,
) -> torch.Tensor:
  """Failure: the object lies on the table, in no jaw, its up axis past ``limit_angle``.
  Held, its tilt is the hand's to correct."""
  obj: Entity = env.scene[object_name]
  cos = (-obj.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0)
  held = _grasped_any_bool(env, grasp_sensor_names)
  on_table = _on_surface_bool(env, support_sensor_name)
  return ~held & on_table & (torch.acos(cos) > limit_angle)


# -- Events -------------------------------------------------------------------------------


def _mug_lying_roll() -> float:
  """Roll about its own axis at which a mug lying on its side rests on body and handle:
  the handle's outer face meets the table where ``reach * cos(roll) = radius``."""
  from pal_mjlab.tasks.manipulation.mug_scene import MUG_HANDLE_REACH, MUG_RADIUS

  return float(torch.arccos(torch.tensor(MUG_RADIUS / MUG_HANDLE_REACH)))


def reset_mug_on_surface(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  asset_cfg: SceneEntityCfg,
  surface_cfg: SceneEntityCfg,
  pose_range: dict[str, tuple[float, float]],
  lying: bool = False,
  yaw_mirror: float | None = None,
) -> None:
  """Place a mug (``mug_scene.get_mug_spec``) on a surface, upright or lying on its side.

  ``pose_range`` gives x/y relative to the surface centre and ``yaw``: upright, the direction
  the handle points in world axes; lying, the direction of the mug's axis. With
  ``yaw_mirror`` half the mugs have that yaw reflected about it. A lying mug rests on body
  and handle, rolled to one side or the other at random, dropped 2 mm above that pose.
  """
  from pal_mjlab.tasks.manipulation.mug_scene import MUG_HALF_HEIGHT, MUG_RADIUS

  asset: Entity = env.scene[asset_cfg.name]
  n = len(env_ids)
  device = env.device
  x_range = pose_range.get("x", (0.0, 0.0))
  y_range = pose_range.get("y", (0.0, 0.0))
  yaw_range = pose_range.get("yaw", (0.0, 0.0))
  xy = (
    sample_uniform(
      torch.tensor([x_range[0], y_range[0]], device=device),
      torch.tensor([x_range[1], y_range[1]], device=device),
      (n, 2),
      device=device,
    )
    + _surface_centre_xy(env, surface_cfg)[env_ids]
  )
  yaw = sample_uniform(yaw_range[0], yaw_range[1], (n,), device=device)
  if yaw_mirror is not None:
    yaw = torch.where(torch.rand(n, device=device) < 0.5, 2 * yaw_mirror - yaw, yaw)
  zeros = torch.zeros(n, device=device)
  top_z = surface_top_z(env, surface_cfg)[env_ids]
  if lying:
    side = torch.where(torch.rand(n, device=device) < 0.5, -1.0, 1.0)
    # Axis along world x first (pitch 90 deg puts local z on +x and the handle, local +x,
    # straight down), rolled about the axis, then turned to ``yaw``.
    roll = quat_from_euler_xyz(zeros, zeros, side * _mug_lying_roll())
    pitch = quat_from_euler_xyz(zeros, torch.full_like(zeros, torch.pi / 2), zeros)
    turn = quat_from_euler_xyz(zeros, zeros, yaw)
    quat = quat_mul(turn, quat_mul(pitch, roll))
    z = top_z + MUG_RADIUS + 0.002
  else:
    quat = quat_from_euler_xyz(zeros, zeros, yaw)
    z = top_z + MUG_HALF_HEIGHT + 0.001
  pos = torch.cat([xy, z.unsqueeze(-1)], dim=-1)
  asset.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
  asset.write_root_link_velocity_to_sim(
    torch.zeros(n, 6, device=device), env_ids=env_ids
  )
  env.sim.forward()


# -- Commands -----------------------------------------------------------------------------


class PlacementCommand(EntityOffsetCommand):
  """``EntityOffsetCommand`` that also logs a released-and-settled placement:
  ``placement_success`` this step and ``episode_placement_success`` (latched)."""

  cfg: PlacementCommandCfg

  def __init__(self, cfg: PlacementCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.episode_placement_success = torch.zeros(self.num_envs, device=self.device)
    self.metrics["placement_success"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["episode_placement_success"] = torch.zeros(
      self.num_envs, device=self.device
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    self.episode_placement_success[env_ids] = 0.0

  def _update_metrics(self) -> None:
    super()._update_metrics()
    error = torch.norm(self.target_pos - self.object.data.root_link_pos_w, dim=-1)
    released = ~_grasped_any_bool(self._env, self.cfg.placement_grasp_sensor_names)
    settled = (
      torch.norm(self.object.data.root_link_lin_vel_w, dim=-1)
      < self.cfg.placement_velocity_threshold
    )
    placed = ((error < self.cfg.success_threshold) & released & settled).float()
    self.episode_placement_success = torch.maximum(
      self.episode_placement_success, placed
    )
    self.metrics["placement_success"] = placed
    self.metrics["episode_placement_success"] = self.episode_placement_success


@dataclass(kw_only=True)
class PlacementCommandCfg(EntityOffsetCommandCfg):
  placement_grasp_sensor_names: tuple[tuple[str, str], ...]
  """Grasp sensor pairs: placed means held by none of them."""

  placement_velocity_threshold: float = 0.05
  """Maximum object speed counted as a settled placement."""

  def build(self, env: ManagerBasedRlEnv) -> PlacementCommand:
    return PlacementCommand(self, env)
