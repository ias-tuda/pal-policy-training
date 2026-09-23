"""Pick-and-place is possible through the action interface the policy trains on.

C1 and C2 (2026-09-22) spent ~750 M steps per run on a task that could not be done: the
action scales confined the jaw to 22 % of its range and the grasp point to ~12 cm of travel,
and the grasp sensor required contacts a held cube never makes. S1 had proved the *robot*
could reach the cube; nothing had proved the *policy* could. This is that proof, kept in the
suite so that a later scale, pose, sensor or reward change that makes the task impossible
again fails here rather than at iteration 15 000.

The scripted episode (`scripts/scripted_pick_place.py`) plans IK waypoints from the cube's
observed pose and feeds them through the env's own `JointPositionAction` as raw actions in
[-1, 1] -- same term, scales, clip and 50 Hz control rate as training. The plan's S5.0 gate:
the grasp sensor fires, the cube leaves the table, and it is placed within 2.5 cm and
released, in at least 9 of 10 spawns. The spawns are the spawn box's centre and its eight
corners (x, y +-2 cm, yaw +-10 deg) -- the hardest cases -- plus one random jitter.
"""

import importlib.util
import pathlib
import sys

import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import (
  make_pick_place_env_cfg,
)
from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS_LEFT_ARM

_SCRIPT = pathlib.Path(__file__).parent.parent / "scripts/scripted_pick_place.py"
_SPEC = importlib.util.spec_from_file_location("scripted_pick_place", _SCRIPT)
assert _SPEC and _SPEC.loader
scripted = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = scripted  # its dataclass looks itself up there
_SPEC.loader.exec_module(scripted)

REQUIRED = 9
SPAWNS = 10


@pytest.fixture(scope="module")
def episodes() -> list[dict]:
  return scripted.rollout(
    num_envs=SPAWNS, device=get_test_device(), seed=0, verbose=False
  )


def test_the_scripted_policy_picks_and_places(episodes):
  placed = [e["env"] for e in episodes if e["placed_and_lifted"]]
  assert len(placed) >= REQUIRED, (
    f"placed and lifted in {len(placed)}/{SPAWNS} spawns (envs {placed}); "
    f"per-episode: {episodes}"
  )


def test_every_spawn_is_grasped_and_lifted(episodes):
  """Stronger than the gate, and what C1/C2 never once achieved: the grasp sensor fires
  and the cube leaves the table in every spawn, corners included."""
  assert all(e["grasped"] for e in episodes), episodes
  assert all(e["lifted"] for e in episodes), episodes


def test_no_scripted_action_needed_clipping(episodes):
  """Every waypoint the task needs is inside the policy's [-1, 1] box. A clipped action
  would mean the script reached the goal with commands the policy cannot issue."""
  assert episodes[0]["clipped_env_steps"] == 0


def test_the_script_respects_the_arm_slew_limit(episodes):
  """S5.1 item 4: no scripted step asks the arm to move faster than the hardware can, so the
  feasibility result is not bought with commands the rate-limited action term would lag."""
  assert episodes[0]["over_slew_env_steps"] == 0


def test_the_jaw_spans_its_range_and_no_further():
  """Action +1 is the fully closed jaw; any action <= 0 is the open jaw, never a target
  below the open limit. mjlab's position actuators are `ctrllimited=False`, so without the
  clip action -1 would press the four-bar into its stop."""
  cfg = make_pick_place_env_cfg(KANGAROO_GRIPPERS_LEFT_ARM, gripper_name="left")
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  term = env.action_manager.get_term("joint_pos")
  jaw = [n.split("/", 1)[-1] for n in term.target_names].index(
    "gripper_left_finger_joint"
  )
  grip = KANGAROO_GRIPPERS_LEFT_ARM.gripper("left")
  for raw, expected in ((1.0, grip.closed_pos), (-1.0, grip.open_pos), (0.0, 0.0)):
    a = torch.zeros(env.action_space.shape, device=env.device)
    a[:, jaw] = raw
    term.process_actions(a)
    got = float(term._processed_actions[0, jaw])
    assert got == pytest.approx(expected, abs=1e-6), f"action {raw} -> target {got}"
  env.close()
