"""S5.1 item 8: the trunk keep-out penalty and the 0.75 m reach termination.

Both come from Claudius's 82c4450. What these pin: every pick-and-place reward arm carries
the penalty (the variants replace the reward dict wholesale, so it has to be added back),
the penalty is exactly zero at the start pose (a constant from step one would be a tax, not
a signal), it does fire when a link is inside the box, and the reach termination fires on
distance and not before.
"""

import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from pal_mjlab.tasks.manipulation import mdp
from pal_mjlab.tasks.manipulation.pick_place import pick_place_env_cfg as pp
from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import (
  make_pick_place_env_cfg,
)
from pal_mjlab.tasks.manipulation.pick_place.rewards import (
  REWARD_VARIANTS,
  STACK_ONLY_ARMS,
)
from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS_LEFT_ARM

KEEPOUT = KANGAROO_GRIPPERS_LEFT_ARM.trunk_keepout
REACH = KANGAROO_GRIPPERS_LEFT_ARM.reach


@pytest.mark.parametrize("variant", sorted(REWARD_VARIANTS))
def test_every_reward_arm_carries_the_trunk_penalty(variant):
  # Plan S9's RG / SA read the base cube: they are built on the stack goal, which needs B's box.
  stack = (
    {"goal": "stack", "baseline": pp.BASELINES["B"]}
    if variant in STACK_ONLY_ARMS
    else {}
  )
  cfg = make_pick_place_env_cfg(
    KANGAROO_GRIPPERS_LEFT_ARM, gripper_name="left", reward=variant, **stack
  )
  term = cfg.rewards["trunk_clearance"]
  assert term.weight < 0
  assert "out_of_reach" in cfg.terminations


@pytest.fixture(scope="module")
def env():
  torch.manual_seed(0)
  cfg = make_pick_place_env_cfg(KANGAROO_GRIPPERS_LEFT_ARM, gripper_name="left")
  cfg.scene.num_envs = 2
  e = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  e.reset()
  e.step(torch.zeros(e.num_envs, e.action_manager.total_action_dim, device=e.device))
  yield e
  e.close()


def _penalty(env, **overrides) -> torch.Tensor:
  params = {
    "center": KEEPOUT.center,
    "half_extents": KEEPOUT.half_extents,
    "margin": KEEPOUT.margin,
    "reference_body": KEEPOUT.reference_body,
    **overrides,
  }
  asset = SceneEntityCfg("robot", body_names=KEEPOUT.body_name_regex)
  asset.resolve(env.scene)
  return mdp.keepout_box_penalty(env, asset_cfg=asset, **params)


def test_the_penalty_is_zero_at_the_start_pose(env):
  assert torch.all(_penalty(env) == 0.0)


def test_the_penalty_fires_inside_the_box(env):
  """Positive control: a box that swallows the arms scores one per watched link."""
  big = _penalty(env, half_extents=(2.0, 2.0, 2.0))
  n_links = len(env.scene["robot"].find_bodies(KEEPOUT.body_name_regex)[0])
  assert n_links == 12  # arm 2-7, both arms
  assert torch.allclose(big, torch.full_like(big, float(n_links)))


def test_reach_fires_on_distance_only(env):
  obj = env.scene["object"]
  params = {
    "entity_names": ("object",),
    "max_distance": REACH.max_distance,
    "reference_body": REACH.reference_body,
  }
  assert not mdp.entity_out_of_reach(env, **params).any()
  robot = env.scene["robot"]
  base = robot.data.body_link_pos_w[:, robot.body_names.index(REACH.reference_body)]
  pose = obj.data.root_link_pose_w.clone()
  pose[0, 0] = base[0, 0] + REACH.max_distance + 0.05
  pose[0, 1] = base[0, 1]
  obj.write_root_link_pose_to_sim(pose)
  env.sim.forward()
  out = mdp.entity_out_of_reach(env, **params)
  assert out.tolist() == [True, False]
