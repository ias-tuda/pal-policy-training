"""Plan S7-2, module V: a bound on the tool's speed, in the action term.

Halving every joint's slew (S7's speed bound A) cut the held cube's peak speed by only 15-30 %:
several joints add up at the tool. V bounds the tool: after the per-joint slew, the arm's step
in commanded target dq is scaled down uniformly until sum_i |dq_i| r_i <= v * dt, with r_i the
distance from joint i's axis to the grasp point; v is 0.3 m/s, and 0.1 m/s while the cube is
held within 5 cm of the table or of the target (values confirmed by Siwei 2026-09-25). B-SV =
baseline B + M7 + V.

What the plan asks before GPU time: the bound holds on random actions; the slow zone engages;
the scripted gate still passes under V. The deployer must mirror the bound, as it must the slew
(`pal_policy_inferer`), and needs a proxy for "held" to do it.

A note on what the bound costs. sum_i |dq_i| r_i is an upper bound on the tool's travel, and a
loose one when joints move against each other: on the scripted approach, where the wrist turns
~1.5 rad to look down, V stretched the move to 223 control steps (4.5 s) for ~13 cm of grasp-point
travel. The script is conservative on top (the slow cap whenever the jaw is closed, 20 % margin),
but a policy under V will be slower than "0.3 m/s" suggests.
"""

import importlib.util
import pathlib
import sys

import numpy as np
import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from pal_mjlab.tasks.manipulation import mdp
from pal_mjlab.tasks.manipulation.mdp.observations import _grasped_bool
from pal_mjlab.tasks.manipulation.pick_place import pick_place_env_cfg as pp
from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS_LEFT_ARM
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import CONTROL_DT

_SCRIPT = pathlib.Path(__file__).parent.parent / "scripts/scripted_pick_place.py"
_SPEC = importlib.util.spec_from_file_location("scripted_pick_place_v", _SCRIPT)
assert _SPEC and _SPEC.loader
scripted = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = scripted
_SPEC.loader.exec_module(scripted)

FRS = "R-C-strict+M1+M2+M3+M6+M7"
B = pp.BASELINES["B"]
V = pp.TOOL_SPEED_CAPS["V"]
ARM = tuple(f"arm_left_{i}_joint" for i in range(1, 8))


def _cfg(tool=V, num_envs=None):
  cfg = pp.make_pick_place_env_cfg(
    KANGAROO_GRIPPERS_LEFT_ARM,
    gripper_name="left",
    reward=FRS,
    action_slew_rate_scale=0.5,
    baseline=B,
    tool_speed_cap=tool,
  )
  if num_envs is not None:
    cfg.scene.num_envs = num_envs
  return cfg


def test_v_is_the_confirmed_cap():
  assert (V.speed, V.slow_speed, V.slow_zone) == (0.3, 0.1, 0.05)
  assert pp.TOOL_SPEED_CAPS["none"] is None


def test_v_changes_only_the_action_term():
  capped, plain = _cfg(V), _cfg(None)
  action, base = capped.actions["joint_pos"], plain.actions["joint_pos"]
  assert type(action) is mdp.ToolSpeedCappedJointPositionActionCfg
  assert type(base) is mdp.RateLimitedJointPositionActionCfg
  assert action.rate_limit == base.rate_limit, "the slew is unchanged under V"
  assert action.scale == base.scale and action.clip == base.clip
  assert action.capped_joints == (r"arm_left_[1-7]_joint",), "the jaw is not capped"
  assert action.tool_site == "tcp_left" and action.control_dt == CONTROL_DT
  assert (action.speed_limit, action.slow_speed_limit, action.slow_zone) == (
    0.3,
    0.1,
    0.05,
  )
  assert capped.rewards.keys() == plain.rewards.keys()
  assert capped.observations["actor"].terms.keys() == (
    plain.observations["actor"].terms.keys()
  )


def _lever_arms(env, joints) -> torch.Tensor:
  """|(tool - anchor) x axis| per env and arm joint, from the sim state. (B, 7)"""
  robot = env.scene["robot"]
  jids = robot.indexing.joint_ids[[robot.joint_names.index(j) for j in joints]]
  anchor = env.sim.data.xanchor[:, jids]
  axis = env.sim.data.xaxis[:, jids]
  tool = robot.data.site_pos_w[:, robot.site_names.index("tcp_left")].unsqueeze(1)
  return torch.linalg.cross(tool - anchor, axis, dim=-1).norm(dim=-1)


def test_the_bound_holds_on_random_actions():
  """Every step: the commanded arm step's sum_i |dq_i| r_i is at most v * dt, with r_i from the
  state the term saw; it scales the slew-limited step and keeps its direction; and it binds --
  random actions ask for more."""
  env = ManagerBasedRlEnv(cfg=_cfg(num_envs=8), device=get_test_device())
  env.reset()
  term = env.action_manager.get_term("joint_pos")
  names = [n.split("/", 1)[-1] for n in term.target_names]
  arm = [names.index(j) for j in ARM]
  gen = torch.Generator(device="cpu").manual_seed(0)
  binding = checked = 0
  for _ in range(60):
    previous = term._last_target.clone()
    r = _lever_arms(env, ARM)
    action = (torch.rand(env.num_envs, term.action_dim, generator=gen) * 2 - 1).to(
      env.device
    )
    wanted = torch.clamp(
      action * term._scale + term._offset, term._clip[..., 0], term._clip[..., 1]
    )
    slewed = previous + (wanted - previous).clamp(-term._rate, term._rate)
    env.step(action)
    fresh = env.episode_length_buf > 0  # not reset inside this step
    step = (term._last_target - previous)[:, arm]
    want = (slewed - previous)[:, arm]
    travel = (step.abs() * r).sum(dim=-1)
    uncapped = (want.abs() * r).sum(dim=-1)
    limit = term.speed_limit * CONTROL_DT
    assert bool((travel[fresh] <= limit[fresh] * (1 + 1e-4) + 1e-7).all()), travel
    shrink = (travel / uncapped.clamp_min(1e-12)).unsqueeze(-1)
    assert torch.allclose(step[fresh], (want * shrink)[fresh], atol=1e-6)
    binding += int((uncapped[fresh] > 1.01 * limit[fresh]).sum())
    checked += int(fresh.sum())
  env.close()
  assert binding > checked // 4, f"the cap bound on {binding} of {checked} env-steps"


# -- The scripted gate under V, and the slow zone on its states ---------------------------


@pytest.fixture(scope="module")
def gate():
  """The scripted pick-and-place under B + V from the S71 spawn box's corners (at -/+10 deg)
  and its centre (0 and +10 deg) to the target box's centre. The hook records, after each
  step, the state the next step's cap is computed from and the cap this step used."""
  box = pp.SPAWN_BOXES["S71"]
  spawns = scripted.box_corners(box) + [((0.42, 0.27), box.yaw)]
  trace = []

  def hook(phase, env):
    term = env.action_manager.get_term("joint_pos")
    obj = env.scene[pp.OBJECT]
    pos = obj.data.root_link_pos_w
    goal = env.command_manager.get_term("goal_pose").command
    half = env.sim.model.geom_size[:, obj.indexing.geom_ids[0], 2]
    top = mdp.surface_top_z(env, SceneEntityCfg(pp.TABLE))
    sensor = KANGAROO_GRIPPERS_LEFT_ARM.gripper("left").contact_sensor_name
    trace.append(
      {
        "phase": phase,
        "held": _grasped_bool(env, sensor).cpu().numpy(),
        "height": (pos[:, 2] - top - half).cpu().numpy(),
        "dxy": torch.norm((pos - goal)[:, :2], dim=-1).cpu().numpy(),
        "limit": term.speed_limit.cpu().numpy(),
      }
    )

  saved = scripted.CLEAR_HEIGHT
  # 8 cm: the held cube then leaves V's 5 cm slow zone on the carry, so both caps are seen
  # (at the corners check's 5 cm it never does). Plannable at the target box's centre.
  scripted.CLEAR_HEIGHT = 0.08
  try:
    rows = scripted.rollout(
      num_envs=len(spawns),
      device=get_test_device(),
      reward=FRS,
      verbose=False,
      hook=hook,
      action_slew_rate_scale=0.5,
      baseline=B,
      spawns=spawns,
      targets=[B.target_xy] * len(spawns),
      tool_speed_cap=V,
      episode_length_s=40.0,
    )
  finally:
    scripted.CLEAR_HEIGHT = saved
  return rows, trace


def test_the_scripted_gate_passes_under_v(gate):
  rows, _ = gate
  placed = sum(r["placed_and_lifted"] for r in rows)
  assert placed >= 9, rows
  assert rows[0]["clipped_env_steps"] == 0


def test_the_slow_zone_engages_where_the_plan_says(gate):
  """The cap each step used is the slow one exactly when, at the state that step started
  from, the cube was held within 5 cm of its resting height or horizontally of the goal."""
  _, trace = gate
  slow_held = fast_held = 0
  for before, now in zip(trace, trace[1:], strict=False):
    zone = before["held"] & ((before["height"] < 0.05) | (before["dxy"] < 0.05))
    want = np.where(zone, V.slow_speed, V.speed)
    assert np.allclose(now["limit"], want), (now["phase"], before, now["limit"])
    slow_held += int((before["held"] & zone).sum())
    fast_held += int((before["held"] & ~zone).sum())
  assert slow_held > 0 and fast_held > 0, (slow_held, fast_held)
