"""Plan S9 (stack one cube on another; Siwei 2026-10-01 22:04): the release condition ``OK``.

"Let go only when aligned and slow" (`pick-place-v1-objects-plan.md`, *S9 revised again*). Five
quantities of the carried cube against the goal on top of the base cube, from the true sim state:

=========  ==================================================================  =======
quantity   what                                                                 bound
=========  ==================================================================  =======
dxy        horizontal distance, cube centre to the goal (the base's top)       1.0 cm
height     cube centre above its resting height on the base (goal z), >= 0      1.0 cm
tilt       angle between world z and the cube's most vertical body axis          10 deg
yaw        the cube's yaw against the base's, folded into [0, 45 deg]            10 deg
speed      the cube's linear speed                                              5 cm/s
=========  ==================================================================  =======

``OK`` is all five inside their bounds. It is read by the jaw lock (`GATE=stack`,
`actions.StackGateJointPositionAction`), recorded at each let-go by the release latch
(`observations._release_latch`, ``rel_ok``) for RG, and graded by SA (`stack_alignment`).

A cube looks the same every 90 deg about any axis, so its yaw is that of a horizontal body axis
-- the first one that is not the most vertical -- folded mod 90 deg, and the tilt is that of
whichever axis is nearest to vertical: a cube lying on any face is upright.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import matrix_from_quat

#: The plan's bounds on the five quantities (Siwei, 2026-10-01 22:10: "the strict thresholds").
OK_DXY = 0.010  # m
OK_HEIGHT = 0.010  # m
OK_TILT = math.radians(10.0)
OK_YAW = math.radians(10.0)
OK_SPEED = 0.05  # m/s, `observations.STILL_SPEED`


class StackState(NamedTuple):
  """The carried cube against the stack goal, per env. Each (B,)."""

  dxy: torch.Tensor
  height: torch.Tensor
  tilt: torch.Tensor
  yaw: torch.Tensor
  speed: torch.Tensor

  @property
  def ok(self) -> torch.Tensor:
    """All five inside the plan's bounds. (B,)"""
    return (
      (self.dxy <= OK_DXY)
      & (self.height <= OK_HEIGHT)
      & (self.tilt <= OK_TILT)
      & (self.yaw <= OK_YAW)
      & (self.speed <= OK_SPEED)
    )


def _tilt_and_yaw(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """(tilt, yaw) of a cube from its world quaternion (w, x, y, z). (B,), (B,)

  tilt: the angle of the most vertical body axis to world z, in [0, 54.7 deg]. yaw: the
  heading of a horizontal body axis (the next one after the most vertical, cyclically), in
  (-pi, pi]; only meaningful mod 90 deg.
  """
  rot = matrix_from_quat(quat)  # (B, 3, 3), columns are the body axes in the world
  up = rot[:, 2, :].abs()  # z component of each body axis
  k = up.argmax(dim=-1)
  tilt = torch.acos(up.gather(-1, k.unsqueeze(-1)).squeeze(-1).clamp(max=1.0))
  j = (k + 1) % 3
  axis = rot.gather(-1, j.view(-1, 1, 1).expand(-1, 3, 1)).squeeze(-1)  # (B, 3)
  yaw = torch.atan2(axis[:, 1], axis[:, 0])
  return tilt, yaw


def fold_quarter(angle: torch.Tensor) -> torch.Tensor:
  """|angle| folded mod 90 deg into [0, 45 deg]: how far apart two cube yaws are."""
  quarter = math.pi / 2
  return torch.remainder(angle + quarter / 2, quarter).sub(quarter / 2).abs()


def stack_state(
  env: ManagerBasedRlEnv,
  object_name: str,
  reference_name: str,
  command_name: str,
) -> StackState:
  """The five ``OK`` quantities of ``object_name`` against the command's goal, which sits on
  ``reference_name`` (`EntityOffsetCommand` with ``stack_on_top``). Height is the cube centre
  above the goal's z, clamped at 0 (a cube pressed into the base is at its resting height)."""
  obj: Entity = env.scene[object_name]
  ref: Entity = env.scene[reference_name]
  goal = env.command_manager.get_term(command_name).command
  pos = obj.data.root_link_pos_w
  d = pos - goal
  tilt, yaw = _tilt_and_yaw(obj.data.root_link_quat_w)
  _, ref_yaw = _tilt_and_yaw(ref.data.root_link_quat_w)
  return StackState(
    dxy=torch.norm(d[:, :2], dim=-1),
    height=d[:, 2].clamp_min(0.0),
    tilt=tilt,
    yaw=fold_quarter(yaw - ref_yaw),
    speed=torch.norm(obj.data.root_link_lin_vel_w, dim=-1),
  )


def stack_command_reference(env: ManagerBasedRlEnv, command_name: str) -> str | None:
  """The base's entity name when the goal is a stack goal (`EntityOffsetCommand` with
  ``stack_on_top``), else None: every table goal."""
  cfg = env.command_manager.get_term(command_name).cfg
  if getattr(cfg, "stack_on_top", False):
    return cfg.reference_name
  return None
