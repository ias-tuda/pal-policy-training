"""Plan S11 (`pick-place-v1-objects-plan.md`, *S11 -- two arms*): the left arm places cube A at a
meeting point T, the right arm stacks cube B on it.

The terms here serve both structures the plan trains. SEQ trains two one-arm policies on the left
arm -- L-T, the table placer to T, and R-ST, the stacker on a base at T that runs on the right arm
through the mirror map -- and needs the stack goal's base priced (`reference_displacement_penalty`)
and seen through its tracker (`stack_goal_position_tracked_b`). JOINT drives both arms with one
policy; its own terms are in `pick_place/bimanual.py` and the rest of this module.
"""

from __future__ import annotations

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from pal_mjlab.tasks.manipulation.mdp.box import _corrupted, tracker_bias
from pal_mjlab.tasks.manipulation.mdp.observations import (
  _object_pose_w,
  _to_base_frame,
)
from pal_mjlab.tasks.manipulation.mdp.shapes import object_half_height

_ROBOT = SceneEntityCfg("robot")


def stack_goal_position_tracked_b(
  env: ManagerBasedRlEnv,
  base_name: str,
  object_name: str,
  bias_key: str,
  asset_cfg: SceneEntityCfg = _ROBOT,
  lag: tuple[int, int] | None = None,
) -> torch.Tensor:
  """A stack goal as the actor sees it on the robot (plan S11's R-ST): the base cube's pose as its
  tracker reports it (``lag`` steps late, like the carried cube's; None for the live pose), plus the
  two half-heights -- where the carried cube rests on it (`EntityOffsetCommand`'s ``stack_on_top``)
  -- plus the base's tracker bias (`box.tracker_bias`: the shared part and the base's own) while
  the actor is corrupted, in the base frame. (B, 3)"""
  robot: Entity = env.scene[asset_cfg.name]
  pos_w, _ = _object_pose_w(env, base_name, lag)
  goal_w = pos_w.clone()
  goal_w[:, 2] += object_half_height(env, base_name) + object_half_height(
    env, object_name
  )
  goal_b = _to_base_frame(robot, goal_w)
  if _corrupted(env):
    goal_b = goal_b + tracker_bias(env, bias_key)
  return goal_b


def reference_displacement_penalty(
  env: ManagerBasedRlEnv, command_name: str, scale: float
) -> torch.Tensor:
  """How far a stack goal's base has been pushed since the reset, as a fraction of ``scale`` (m),
  capped at 1 -- plan S11's `base_moved`, S10's `box_moved` on the base cube. Weight it negative.
  (B,)"""
  command = env.command_manager.get_term(command_name)
  return (command.reference_displacement() / scale).clamp(0.0, 1.0)


# -- JOINT: the right's stack terms wait for A ---------------------------------------------------


def base_placed_bool(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  grasp_sensor_name: str,
  support_sensor_name: str,
  jaw_cfg: SceneEntityCfg,
  jaw_open: float,
  threshold: float,
  velocity_threshold: float,
) -> torch.Tensor:
  """Plan S11's "A is placed": the left arm let go of it after the lift with its jaw open (plan
  S7-4's J, `observations._placed_bool`), A at T within ``threshold`` and still, and no pad --
  of the arm that placed it -- touching it. (B,)"""
  from pal_mjlab.tasks.manipulation.mdp.observations import (
    _pad_contact_any,
    _placed_bool,
  )

  placed = _placed_bool(
    env,
    command_name,
    object_name,
    grasp_sensor_name,
    threshold,
    velocity_threshold,
    support_sensor_name,
    jaw_cfg,
    jaw_open,
  )
  return placed & ~_pad_contact_any(env, grasp_sensor_name)


def when_base_placed(
  env: ManagerBasedRlEnv,
  inner,
  gate_command_name: str,
  gate_object_name: str,
  gate_grasp_sensor_name: str,
  gate_support_sensor_name: str,
  gate_jaw_cfg: SceneEntityCfg,
  gate_jaw_open: float,
  gate_threshold: float,
  gate_velocity_threshold: float,
  **params,
) -> torch.Tensor:
  """``inner(env, **params)`` while the base is placed (`base_placed_bool` on the ``gate_*``
  parameters), else 0 -- plan S11's JOINT coupling 1: the right's ``goal``, ``goal_near``,
  ``release``, ``placed`` (and M6's grade in them), SA and M8's retreat pay only while A is placed,
  so placing A switches terms on and can never lower the total (L5 by construction), and knocking
  A off T stops the right's stack income. Every parameter is a top-level one, so the manager
  resolves both sets' entity cfgs."""
  gate = base_placed_bool(
    env,
    gate_command_name,
    gate_object_name,
    gate_grasp_sensor_name,
    gate_support_sensor_name,
    gate_jaw_cfg,
    gate_jaw_open,
    gate_threshold,
    gate_velocity_threshold,
  )
  return inner(env, **params) * gate.float()


def base_placed_obs(env: ManagerBasedRlEnv, **gate) -> torch.Tensor:
  """`base_placed_bool` as a critic input (``gate`` as its parameters). (B, 1)"""
  return base_placed_bool(env, **gate).float().unsqueeze(-1)


def handoff_active(env: ManagerBasedRlEnv, arm: str) -> torch.Tensor:
  """1 while ``arm`` is handed to plan S11's hand-off controller (`mdp.handoff`), else 0; a critic
  input -- the policy's columns for that arm do nothing then. (B, 1)"""
  term = env.action_manager.get_term("joint_pos")
  return (term.phase[:, term.arm_index(arm)] > 0).float().unsqueeze(-1)


# -- JOINT: the arms keep apart (HH) -------------------------------------------------------------
#
# Each arm is three capsules: the forearm (arm 4's origin to arm 6's), the wrist (arm 6's to arm
# 7's) and the hand (arm 7's origin to the TCP), each with the radius that holds every collision-hull
# point of its links -- arm 4-5, arm 6, arm 7 and the gripper's links -- as they sit at the start pose
# (computed once, from env 0). The clearance is the least surface distance over the 3 x 3 capsule
# pairs, negative when they overlap. A capsule hull is a bound, so the clearance reads low by up to a
# centimetre or two near the joints; the contact term (the sensor) is the exact one.
HH_CAPSULES = (
  ("forearm", ("{s}_4_link", "{s}_6_link"), ("{s}_4_link", "{s}_5_link")),
  ("wrist", ("{s}_6_link", "{s}_7_link"), ("{s}_6_link",)),
  ("hand", ("{s}_7_link", "tcp_{side}"), ("{s}_7_link", "gripper_{side}_.*")),
)
_HH_CACHE = "_pp_hh_capsules"


def _capsule_spec(
  env: ManagerBasedRlEnv, side: str
) -> list[tuple[int, int, bool, float]]:
  """Per capsule of arm ``side``: (start body, end body or site, end is a site, radius). Cached."""
  import re

  from mjlab.utils.lab_api.math import quat_apply

  from pal_mjlab.tasks.manipulation.mdp.rewards import _collision_points

  cache = getattr(env, _HH_CACHE, None)
  if cache is None:
    cache = {}
    setattr(env, _HH_CACHE, cache)
  if side in cache:
    return cache[side]
  robot: Entity = env.scene["robot"]
  names = list(robot.body_names)
  s = f"arm_{side}"
  out = []
  for _, (a, b), bodies in HH_CAPSULES:
    a_id = names.index(a.format(s=s, side=side))
    end = b.format(s=s, side=side)
    end_is_site = end not in names
    end_id = robot.site_names.index(end) if end_is_site else names.index(end)
    pats = [p.format(s=s, side=side) for p in bodies]
    ids = [i for i, n in enumerate(names) if any(re.fullmatch(p, n) for p in pats)]
    pts = _collision_points(env, robot, ids)  # (L, n, 3), body frames
    pos = robot.data.body_link_pos_w[0, ids]
    quat = robot.data.body_link_quat_w[0, ids]
    world = pos.unsqueeze(1) + quat_apply(
      quat.unsqueeze(1).expand(-1, pts.shape[1], 4), pts
    )
    p0 = robot.data.body_link_pos_w[0, a_id]
    p1 = (
      robot.data.site_pos_w[0, end_id]
      if end_is_site
      else robot.data.body_link_pos_w[0, end_id]
    )
    d = _point_segment_distance(world.reshape(-1, 3), p0, p1)
    out.append((a_id, end_id, end_is_site, float(d.max())))
  cache[side] = out
  return out


def _point_segment_distance(
  pts: torch.Tensor, p0: torch.Tensor, p1: torch.Tensor
) -> torch.Tensor:
  seg = p1 - p0
  t = ((pts - p0) @ seg / seg.dot(seg).clamp_min(1e-12)).clamp(0.0, 1.0)
  return torch.norm(pts - (p0 + t.unsqueeze(-1) * seg), dim=-1)


def segment_distance(
  p1: torch.Tensor, q1: torch.Tensor, p2: torch.Tensor, q2: torch.Tensor
) -> torch.Tensor:
  """Least distance between segments p1-q1 and p2-q2, batched over the leading dims (Ericson,
  *Real-Time Collision Detection*, 5.1.9; both segments of nonzero length). (...,)"""
  d1, d2, r = q1 - p1, q2 - p2, p1 - p2
  a = (d1 * d1).sum(-1)
  e = (d2 * d2).sum(-1)
  f = (d2 * r).sum(-1)
  c = (d1 * r).sum(-1)
  b = (d1 * d2).sum(-1)
  denom = a * e - b * b
  s = torch.where(
    denom > 1e-12, ((b * f - c * e) / denom.clamp_min(1e-12)).clamp(0.0, 1.0), 0.0
  )
  t = (b * s + f) / e
  s = torch.where(
    t < 0.0,
    (-c / a).clamp(0.0, 1.0),
    torch.where(t > 1.0, ((b - c) / a).clamp(0.0, 1.0), s),
  )
  t = t.clamp(0.0, 1.0)
  closest1 = p1 + d1 * s.unsqueeze(-1)
  closest2 = p2 + d2 * t.unsqueeze(-1)
  return torch.norm(closest1 - closest2, dim=-1)


def _capsules(env: ManagerBasedRlEnv, side: str):
  robot: Entity = env.scene["robot"]
  spec = _capsule_spec(env, side)
  starts, ends, radii = [], [], []
  for a_id, end_id, end_is_site, radius in spec:
    starts.append(robot.data.body_link_pos_w[:, a_id])
    ends.append(
      robot.data.site_pos_w[:, end_id]
      if end_is_site
      else robot.data.body_link_pos_w[:, end_id]
    )
    radii.append(radius)
  return (
    torch.stack(starts, 1),
    torch.stack(ends, 1),
    torch.tensor(radii, device=env.device),
  )


def arm_arm_clearance(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The least surface distance between the left and the right arm's capsules (HH_CAPSULES), m;
  negative when two overlap. (B,)"""
  pl, ql, rl = _capsules(env, "left")
  pr, qr, rr = _capsules(env, "right")
  n = pl.shape[1]
  d = segment_distance(
    pl.unsqueeze(2).expand(-1, n, n, 3),
    ql.unsqueeze(2).expand(-1, n, n, 3),
    pr.unsqueeze(1).expand(-1, n, n, 3),
    qr.unsqueeze(1).expand(-1, n, n, 3),
  )
  return (d - rl.view(1, n, 1) - rr.view(1, 1, n)).flatten(1).amin(-1)


def arm_arm_clearance_penalty(env: ManagerBasedRlEnv, margin: float) -> torch.Tensor:
  """Plan S11's HH clearance: 0 while the arms are more than ``margin`` apart, rising linearly to 1
  at contact (and 1 past it). Weight it negative. (B,)"""
  return (1.0 - arm_arm_clearance(env) / margin).clamp(0.0, 1.0)
