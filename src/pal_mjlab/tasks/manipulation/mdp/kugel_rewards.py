"""Reward terms of MugKugel: grasp the Kugel, carry it over the mug, drop it in, leave.

The Kugel is in the mug when its centre lies inside the hollow body (``mug_scene.
get_hollow_mug_spec``) with no jaw face on it and the mug standing. Released over the
opening it falls in by itself: dropped 5 cm over the rim it went in up to 30 mm off the
axis, the rim deflecting it, and bounced out from 35 mm. So the place is the opening, not
a pose to set it down in, and the release is the one action that finishes the task.

The coaster's lesson carries over: a hold at the goal that pays for good is a hold that
never lets go. ``paid_within_budget`` stops ``lift`` once it has paid a few seconds, and
``kugel_drop_progress`` pulls the Kugel to the opening as shaping, its potential staying
full through the drop and in the mug, so letting go costs nothing to take.

No term here ends the episode: a Kugel dropped beside the mug can be picked up again, and
a tipped mug costs ``mug_tipped`` every second it lies.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from pal_mjlab.tasks.manipulation import mug_scene as scene
from pal_mjlab.tasks.manipulation.mdp.grasp_ladder import _grasped_any_bool, _jaw_frame
from pal_mjlab.tasks.manipulation.mdp.mug_rewards import (
  _hand_touching,
  _ProgressShaping,
  mug_tilt_cos,
)

#: The mug counts as standing up to this cosine of tilt (~30 deg): a Kugel in a mug tipped
#: further is on its way out.
_STANDING_COS = 0.85
#: How far over the rim a free Kugel still counts as falling in.
_FALL_HEIGHT = 0.15


def _kugel_in_mug_frame(env: ManagerBasedRlEnv, object_name: str, mug_name: str):
  """The Kugel centre in the mug's frame: radial distance from the axis and height over
  the mug centre. ``((B,), (B,))``"""
  obj: Entity = env.scene[object_name]
  mug: Entity = env.scene[mug_name]
  rel = quat_apply_inverse(
    mug.data.root_link_quat_w, obj.data.root_link_pos_w - mug.data.root_link_pos_w
  )
  return torch.norm(rel[:, :2], dim=-1), rel[:, 2]


def _in_mug(env, object_name, mug_name, grasp_sensor_names) -> torch.Tensor:
  radial, z = _kugel_in_mug_frame(env, object_name, mug_name)
  mug: Entity = env.scene[mug_name]
  inside = (radial < scene.MUG_INNER_RADIUS) & (z < scene.MUG_HALF_HEIGHT)
  standing = mug_tilt_cos(mug) > _STANDING_COS
  return inside & standing & ~_grasped_any_bool(env, grasp_sensor_names)


def kugel_in_mug(
  env: ManagerBasedRlEnv,
  object_name: str,
  mug_name: str,
  grasp_sensor_names: Sequence[Sequence[str]],
  grasp_cfg: SceneEntityCfg | None = None,
  hand_sensor_name: str | None = None,
  clearance: float | None = None,
) -> torch.Tensor:
  """1.0 while the Kugel lies in the standing mug, let go; with ``clearance``, also the
  jaw midpoint that far from it and no part of the hand on the mug. (B,)"""
  ok = _in_mug(env, object_name, mug_name, grasp_sensor_names)
  if clearance is not None:
    assert grasp_cfg is not None, "clearance needs grasp_cfg"
    mid, _, _ = _jaw_frame(env.scene[grasp_cfg.name], grasp_cfg)
    obj: Entity = env.scene[object_name]
    dist = torch.norm(mid - obj.data.root_link_pos_w.unsqueeze(1), dim=-1).amin(dim=-1)
    ok &= (dist > clearance) & ~_hand_touching(env, hand_sensor_name)
  return ok.float()


def kugel_releasing_over_mug(
  env: ManagerBasedRlEnv,
  object_name: str,
  mug_name: str,
  grasp_cfg: SceneEntityCfg,
  object_width: float = scene.KUGEL_WIDTH,
  over_xy: float = 0.025,
  open_margin: float = 0.005,
) -> torch.Tensor:
  """How far the jaw has opened past the Kugel while the Kugel is over the mug's
  opening: within ``over_xy`` of the axis and above the rim. (B,)

  The gradient into opening: the drop itself pays only once the Kugel lands, and the
  grasp rungs below stop paying the moment the jaw lets go.
  """
  radial, z = _kugel_in_mug_frame(env, object_name, mug_name)
  over = (radial < over_xy) & (z > scene.MUG_HALF_HEIGHT)
  _, _, aperture = _jaw_frame(env.scene[grasp_cfg.name], grasp_cfg)
  opening = torch.sigmoid((aperture.amin(dim=-1) - object_width - open_margin) / 0.003)
  return over.float() * opening


class kugel_drop_progress(_ProgressShaping):
  """Carrying the Kugel to the mug's opening and dropping it in, as shaping. (B,)

  ``phi = exp(-d / std)`` while the jaw holds the Kugel, ``d`` its distance to the goal
  over the opening, or with ``xy_only`` its distance from the mug's axis alone; 1 once it
  is free and falling into the standing mug or lying in it -- within the opening, no
  more than ``_FALL_HEIGHT`` over the rim -- so the drop keeps the potential the carry
  earned. Anywhere else free, 0: a Kugel dropped beside the mug pays the carry back.
  """

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    object_name: str,
    mug_name: str,
    grasp_sensor_names: Sequence[Sequence[str]],
    gamma: float,
    std: float,
    xy_only: bool = False,
  ) -> torch.Tensor:
    obj: Entity = env.scene[object_name]
    mug: Entity = env.scene[mug_name]
    held = _grasped_any_bool(env, grasp_sensor_names)
    radial, z = _kugel_in_mug_frame(env, object_name, mug_name)
    if xy_only:
      d = radial
    else:
      command = env.command_manager.get_term(command_name)
      d = torch.norm(command.command - obj.data.root_link_pos_w, dim=-1)
    falling_in = (
      ~held
      & (radial < scene.MUG_INNER_RADIUS)
      & (z < scene.MUG_HALF_HEIGHT + _FALL_HEIGHT)
      & (mug_tilt_cos(mug) > _STANDING_COS)
    )
    phi = torch.where(falling_in, 1.0, held.float() * torch.exp(-d / std))
    return self._shape(env, phi, gamma)


def mug_tipped(env: ManagerBasedRlEnv, mug_name: str) -> torch.Tensor:
  """1.0 while the mug stands tilted past ~30 deg: knocked over, it cannot take the
  Kugel. A cost, not an ending. (B,)"""
  return (mug_tilt_cos(env.scene[mug_name]) <= _STANDING_COS).float()


class paid_within_budget:
  """``func``'s reward until it has paid ``budget`` seconds of its full reward this
  episode, then nothing. (B,)

  For ``lift``, which pays holding the Kugel near the goal: without an end to it, holding
  on there pays for good, as on the coaster, where 1% of the envs ever let go.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self.paid = torch.zeros(env.num_envs, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.paid[slice(None) if env_ids is None else env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    func: Callable[..., torch.Tensor],
    budget: float,
    **params,
  ) -> torch.Tensor:
    reward = func(env, **params) * (self.paid < budget).float()
    self.paid += reward * env.step_dt
    return reward
