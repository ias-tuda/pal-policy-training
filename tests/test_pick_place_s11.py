"""Plan S11 (`pick-place-v1-objects-plan.md`, *S11 -- two arms*): step 0's model checks.

SEQ's codes are the one-arm env on BS11 (S7-8's plant, ABS-a, W2, PQ, AH10, the right home
mirrored, the tracker split) with ``TARGET=T11`` (L-T) or the stack goal at T (R-ST, R-ST-W); JOINT
is ``ARMS=both``. Built on CPU. What is checked: the right arm is the left's mirror image (FK, item 1)
and its mirrored home touches nothing; T stays put in the robot frame while the table moves; R-ST's
base spawns at T and is priced by its displacement; YG and the tracker split; JOINT's 73-D actor
and 16-D action; the right's stack terms wait for A (coupling 1) and a slid A never opens them;
BI-JM's placed-A start; the hand-off takes over the left arm alone, on its trigger, and flies it up,
out and home (coupling 2); HH registers two hands driven into each other (coupling 3's positive
control) and reads clear at home.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scene import Scene
from pal_mjlab.tasks.manipulation import mdp
from pal_mjlab.tasks.manipulation.mdp import handoff as ho
from pal_mjlab.tasks.manipulation.mdp.observations import (
  _WAS_LIFTED,
  latch_attr,
  yaw_error_mod_90,
)
from pal_mjlab.tasks.manipulation.pick_place import bimanual as bi
from pal_mjlab.tasks.manipulation.pick_place.kangaroo.env_cfg import (
  kangaroo_pick_place_env_cfg,
)
from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import (
  ROBOT_TARGETS,
  T11_XY,
)
from pal_mjlab.tasks.manipulation.pick_place.rewards import YG_YAW_STD
from pal_mjlab.tasks.manipulation.robots import (
  ARM_MIRROR_SIGNS,
  mirrored_right_home,
)

RCTP = "R-C-strict+M1+M2+M3+M6+J+M8"
#: BS11 for the one-arm codes (plan S11's table): BS + PQ + AH10 + the mirrored home + the split.
BS11 = {
  "DR": "DR-scene",
  "PPO": "PPO-stack",
  "SLEW": "stack",
  "BASELINE": "B",
  "SPAWN": "W2",
  "TOOL_SPEED": "none",
  "GAINS": "G20",
  "EFFORT": "CTP",
  "LAG": "FP+JS",
  "ACTION": "abs",
  "EPISODE_S": "10",
  "CRITIC_OBS": "q",
  "ARM_TABLE": "AH10",
  "ARM_HOME": "mirror",
  "TRACKER": "split",
  "TARGET": "T11",
}
CODES = {
  "L-T": {"REWARD": RCTP + "+YG"},
  "R-ST": {"REWARD": RCTP + "+SA", "GOAL": "stack"},
  "R-ST-W": {"REWARD": RCTP + "+SA", "GOAL": "stack", "TARGET": "T11W"},
  "BI-J": {
    "REWARD": "S11-BI",
    "ARMS": "both",
    "HANDOFF_TRAIN": "left",
    "START": "scratch",
    "EPISODE_S": "15",
  },
  "BI-JM": {
    "REWARD": "S11-BI",
    "ARMS": "both",
    "HANDOFF_TRAIN": "left",
    "START": "mixed",
    "EPISODE_S": "15",
  },
}
_ALL = {
  *BS11,
  "REWARD",
  "GOAL",
  "GATE",
  "ARMS",
  "HANDOFF_TRAIN",
  "START",
  "HANDOFF_ROUTE",
}
_ALL |= {"SHAPE", "OCC", "OBJ_OBS", "CMD_OBS"}


def _cfg(monkeypatch, code: str, num_envs: int = 4, **changes):
  for var in _ALL:
    monkeypatch.delenv(var, raising=False)
  for var, value in {**BS11, **CODES[code], **changes}.items():
    if value is None:
      monkeypatch.delenv(var, raising=False)
    else:
      monkeypatch.setenv(var, value)
  cfg = kangaroo_pick_place_env_cfg()
  cfg.scene.num_envs = num_envs
  cfg.seed = 0
  return cfg


def _env(monkeypatch, code: str, num_envs: int = 4, cfg=None, **changes):
  if cfg is None:
    cfg = _cfg(monkeypatch, code, num_envs, **changes)
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  env.reset()
  return env


def _step(env, steps: int, actions: torch.Tensor | None = None) -> None:
  with torch.inference_mode():
    for _ in range(steps):
      a = (
        torch.zeros(env.action_space.shape, device=env.device)
        if actions is None
        else actions
      )
      env.step(a)


# -- Item 1: symmetry, measured --------------------------------------------------------------


def test_the_right_arm_is_the_left_arms_mirror_image(monkeypatch):
  """q_right = s * q_left puts the right TCP at the left's (x, -y, z), its frame at M R M diag(-1,
  -1, 1) (plan S11 *Measured today*), and the joint ranges map onto each other."""
  cfg = _cfg(monkeypatch, "L-T", num_envs=1)
  m = Scene(cfg.scene, "cpu").compile()
  d = mujoco.MjData(m)

  def jid(n):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "robot/" + n)

  def sid(n):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "robot/" + n)

  left = [jid(f"arm_left_{i}_joint") for i in range(1, 8)]
  right = [jid(f"arm_right_{i}_joint") for i in range(1, 8)]
  s = np.asarray(ARM_MIRROR_SIGNS)
  rl, rr = m.jnt_range[left], m.jnt_range[right]
  assert np.allclose(np.sort(rl * s[:, None], axis=1), rr)
  mx = np.diag([1.0, -1.0, 1.0])
  flip = np.diag([-1.0, -1.0, 1.0])
  rng = np.random.default_rng(0)
  for _ in range(50):
    ql = rng.uniform(rl[:, 0], rl[:, 1])
    d.qpos[:] = m.qpos0
    for j, q in zip(left, ql, strict=True):
      d.qpos[m.jnt_qposadr[j]] = q
    for j, q in zip(right, s * ql, strict=True):
      d.qpos[m.jnt_qposadr[j]] = q
    mujoco.mj_kinematics(m, d)
    pl, pr = d.site_xpos[sid("tcp_left")], d.site_xpos[sid("tcp_right")]
    assert np.abs(pr - pl * np.array([1.0, -1.0, 1.0])).max() < 1e-4
    rot_l = d.site_xmat[sid("tcp_left")].reshape(3, 3)
    rot_r = d.site_xmat[sid("tcp_right")].reshape(3, 3)
    assert np.abs(rot_r - mx @ rot_l @ mx @ flip).max() < 1e-4


def test_the_mirrored_home_is_the_left_homes_mirror_and_touches_nothing(monkeypatch):
  env = _env(monkeypatch, "L-T", num_envs=2)
  robot = env.scene["robot"]
  home = mirrored_right_home()
  for i in range(1, 8):
    j = robot.joint_names.index(f"arm_right_{i}_joint")
    jl = robot.joint_names.index(f"arm_left_{i}_joint")
    assert float(robot.data.default_joint_pos[0, j]) == pytest.approx(
      home[f"arm_right_{i}_joint"]
    )
    assert float(robot.data.default_joint_pos[0, j]) == pytest.approx(
      ARM_MIRROR_SIGNS[i - 1] * float(robot.data.default_joint_pos[0, jl])
    )
  _step(env, 20)
  # The parked right arm holds its home and the right hand touches neither the table nor the floor.
  right = [robot.joint_names.index(f"arm_right_{i}_joint") for i in range(1, 8)]
  err = (robot.data.joint_pos[:, right] - robot.data.default_joint_pos[:, right]).abs()
  assert float(err.max()) < 0.05
  assert not bool(env.scene["left_hand_surface_contact"].data.found.any())


def test_unset_s11_selectors_leave_the_one_arm_cfg_alone(monkeypatch):
  """Without S11's selectors the right home is Claudius's and the goal table-relative."""
  for var in _ALL:
    monkeypatch.delenv(var, raising=False)
  for var, value in {**BS11, "REWARD": RCTP}.items():
    if var not in ("ARM_HOME", "TRACKER", "TARGET", "CRITIC_OBS", "ARM_TABLE"):
      monkeypatch.setenv(var, value)
  cfg = kangaroo_pick_place_env_cfg()
  robot = cfg.scene.entities["robot"]
  assert robot.init_state.joint_pos["arm_right_5_joint"] == pytest.approx(-0.382)
  assert cfg.commands["goal_pose"].frame == "surface"
  assert type(cfg.actions["joint_pos"]) is mdp.RateLimitedJointPositionActionCfg
  assert cfg.observations["actor"].terms["object_pos"].func is mdp.object_position_b


# -- T, L-T, R-ST -----------------------------------------------------------------------------


def test_t11_stays_in_the_robot_frame_while_the_table_moves(monkeypatch):
  env = _env(monkeypatch, "L-T", num_envs=16)
  goal = env.command_manager.get_term("goal_pose").command - env.scene.env_origins
  tx, ty = T11_XY
  half = ROBOT_TARGETS["T11"].half_xy
  assert float((goal[:, 0] - tx).abs().max()) <= half + 1e-6
  assert float((goal[:, 1] - ty).abs().max()) <= half + 1e-6
  table = env.scene["table"]
  gid = table.indexing.geom_ids[0]
  top = env.sim.data.geom_xpos[:, gid, 2] + env.sim.model.geom_size[:, gid, 2]
  assert torch.allclose(goal[:, 2], top + 0.025, atol=1e-4)
  # DR-scene did move the tables: the goals are not table-relative.
  table_xy = env.sim.data.geom_xpos[:, gid, :2]
  assert float(table_xy[:, 0].std()) > 0.003 or float(table_xy[:, 1].std()) > 0.003


def test_l_t_grades_the_yaw_and_splits_the_tracker(monkeypatch):
  cfg = _cfg(monkeypatch, "L-T")
  for name in ("release", "placed"):
    assert cfg.rewards[name].params["yaw_std"] == pytest.approx(YG_YAW_STD)
  actor = cfg.observations["actor"].terms
  assert actor["object_pos"].func is mdp.object_position_biased_b
  assert cfg.events["reset_tracker_bias"].params["keys"] == ("object",)
  assert sum(t.func is not None for t in actor.values()) == 8
  # The goal is given (the robot knows T), the cube is tracked.
  assert actor["target_pos"].func is mdp.goal_position_b


@pytest.mark.parametrize("code", ["R-ST", "R-ST-W"])
def test_r_st_spawns_the_base_at_t_and_prices_its_displacement(monkeypatch, code):
  env = _env(monkeypatch, code, num_envs=16)
  target = ROBOT_TARGETS["T11" if code == "R-ST" else "T11W"]
  base = env.scene["base_object"]
  xy = base.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
  assert float((xy[:, 0] - target.xy[0]).abs().max()) <= target.base_half_xy + 1e-4
  assert float((xy[:, 1] - target.xy[1]).abs().max()) <= target.base_half_xy + 1e-4
  yaw = yaw_error_mod_90(base.data.root_link_quat_w)
  assert float(yaw.max()) <= min(target.base_yaw, math.pi / 4) + 1e-4
  assert "base_moved" in env.cfg.rewards and "base_disturbance" not in env.cfg.rewards
  assert env.cfg.observations["actor"].terms["target_pos"].func is (
    mdp.stack_goal_position_tracked_b
  )
  # Push the base 3 cm: base_moved saturates at -2 per step.
  pose = base.data.root_link_pose_w.clone()
  pose[:, 0] += 0.03
  base.write_root_link_pose_to_sim(pose)
  pen = mdp.reference_displacement_penalty(env, "goal_pose", 0.02)
  assert torch.allclose(pen, torch.ones_like(pen), atol=1e-3)


# -- JOINT -------------------------------------------------------------------------------------


def test_joint_has_a_73d_actor_and_a_16d_action(monkeypatch):
  env = _env(monkeypatch, "BI-J", num_envs=2)
  obs = env.observation_manager.compute()
  assert obs["actor"].shape[-1] == 73
  assert env.action_space.shape[-1] == 16
  term = env.action_manager.get_term("joint_pos")
  names = [n.split("/")[-1] for n in term.target_names]
  assert set(names) == {
    *(f"arm_{s}_{i}_joint" for s in ("left", "right") for i in range(1, 8)),
    "gripper_left_finger_joint",
    "gripper_right_finger_joint",
  }
  # Both jaws are slew-limited and on the servo; both arms on T's band.
  for jaw in ("gripper_left_finger_joint", "gripper_right_finger_joint"):
    assert float(term._rate[0, names.index(jaw)]) == pytest.approx(0.03)
  assert len(term._servo_idx) == 2 and len(term._band_idx) == 14
  # A in W2, B in W2's mirror image.
  a = env.scene["object"].data.root_link_pos_w - env.scene.env_origins
  b = env.scene[bi.CUBE_B].data.root_link_pos_w - env.scene.env_origins
  assert bool((a[:, 1] > 0.15).all()) and bool((b[:, 1] < -0.15).all())


def _place_a(env, offset=(0.0, 0.0)) -> None:
  """A resting at the goal T (+ ``offset``), the left's lift and release latched there."""
  a = env.scene["object"]
  goal = env.command_manager.get_term("goal_pose").command
  pose = a.data.root_link_pose_w.clone()
  pose[:, :3] = goal
  pose[:, 0] += offset[0]
  pose[:, 1] += offset[1]
  pose[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
  a.write_root_link_pose_to_sim(pose)
  a.write_root_link_velocity_to_sim(torch.zeros_like(a.data.root_link_vel_w))


def _gate(env) -> torch.Tensor:
  p = env.cfg.rewards["placed"].params
  return mdp.base_placed_bool(
    env,
    p["command_name"],
    p["object_name"],
    p["grasp_sensor_name"],
    p["support_sensor_name"],
    p["jaw_cfg"],
    p["jaw_open"],
    p["threshold"],
    0.05,
  )


def test_the_rights_stack_terms_wait_for_a(monkeypatch):
  env = _env(monkeypatch, "BI-J", num_envs=2)
  terms = env.cfg.rewards
  for name in bi.GATED_RIGHT:
    term = terms[f"right_{name}"]
    assert term.func is mdp.when_base_placed and term.weight > 0, name
  # A slid onto T (never lifted): no gate, so no stack income for the right.
  _place_a(env)
  _step(env, 3)
  assert not bool(_gate(env).any())
  # Lifted and let go at T: the gate opens.
  getattr(env, latch_attr(_WAS_LIFTED, "left_pad_contact"))[:] = True
  _step(env, 3)
  assert bool(_gate(env).all())
  # A knocked 4 cm off T: closed again.
  _place_a(env, offset=(0.04, 0.0))
  _step(env, 3)
  assert not bool(_gate(env).any())
  # The wrapper is the inner term times the gate.
  ones = lambda env, **kw: torch.ones(env.num_envs, device=env.device)  # noqa: E731
  p = {k: v for k, v in terms["right_placed"].params.items() if k.startswith("gate_")}
  assert not bool(mdp.when_base_placed(env, ones, **p).any())


def test_bi_jm_starts_half_the_episodes_with_a_placed_and_the_left_home(monkeypatch):
  cfg = _cfg(monkeypatch, "BI-JM", num_envs=8)
  cfg.events["start_a_placed"].params["share"] = 1.0
  env = _env(monkeypatch, "BI-JM", cfg=cfg)
  term = env.action_manager.get_term("joint_pos")
  assert bool((term.phase[:, term.arm_index("left")] == ho.HOME).all())
  a = env.scene["object"].data.root_link_pos_w - env.scene.env_origins
  assert float((a[:, 0] - T11_XY[0]).abs().max()) <= bi.MIXED_A_HALF_XY + 1e-4
  assert bool(getattr(env, latch_attr(_WAS_LIFTED, "left_pad_contact")).all())
  _step(env, 2)
  assert float(_gate(env).float().mean()) > 0.8
  # The left's placed term pays from the first steps.
  assert (
    float(
      env.reward_manager._step_reward[
        :, list(env.reward_manager.active_terms).index("placed")
      ].mean()
    )
    > 0
  )


def test_the_hand_off_takes_the_left_arm_alone(monkeypatch):
  env = _env(monkeypatch, "BI-J", num_envs=2, HANDOFF_TRAIN="none")
  term = env.action_manager.get_term("joint_pos")
  left = term._arms[0]["idx"]
  right = term._arms[1]["idx"]
  term.start_handoff("left", torch.ones(2, dtype=torch.bool, device=env.device))
  a = torch.zeros(env.action_space.shape, device=env.device)
  b = a.clone()
  b[:, left] = 0.9  # only the left's columns differ
  b[:, right] = 0.0
  term.process_actions(a)
  t_a = term._processed_actions.clone()
  term.reset()
  term.start_handoff("left", torch.ones(2, dtype=torch.bool, device=env.device))
  term.process_actions(b)
  assert torch.allclose(term._processed_actions, t_a)
  c = a.clone()
  c[:, right] = 0.9
  term.reset()
  term.start_handoff("left", torch.ones(2, dtype=torch.bool, device=env.device))
  term.process_actions(c)
  assert not torch.allclose(term._processed_actions[:, right], t_a[:, right])


def test_the_left_hands_off_on_its_timeout(monkeypatch):
  import dataclasses

  cfg = _cfg(monkeypatch, "BI-J", num_envs=2)
  arms = list(cfg.actions["joint_pos"].arms)
  assert arms[0].trigger is not None and arms[1].trigger is None
  arms[0] = dataclasses.replace(
    arms[0], trigger=dataclasses.replace(arms[0].trigger, timeout_s=0.2)
  )
  cfg.actions["joint_pos"].arms = tuple(arms)
  env = _env(monkeypatch, "BI-J", cfg=cfg)
  term = env.action_manager.get_term("joint_pos")
  _step(env, 5)
  assert bool((term.phase[:, 0] == ho.POLICY).all())
  _step(env, 10)
  assert bool((term.phase[:, 0] > ho.POLICY).all())
  assert bool((term.phase[:, 1] == ho.POLICY).all())


def test_hh_reads_clear_at_home_and_registers_two_hands_driven_together(monkeypatch):
  import importlib.util
  import pathlib
  import sys

  env = _env(monkeypatch, "BI-J", num_envs=2, HANDOFF_TRAIN="none")
  clear = mdp.arm_arm_clearance(env)
  assert bool((clear > bi.HH_MARGIN).all()), clear
  assert not bool(mdp.arm_arm_clearance_penalty(env, bi.HH_MARGIN).any())
  # Positive control: the left hand at T by IK, the right at the mirror image of that pose --
  # the same point, since T is on the centre line. The hands overlap.
  path = (
    pathlib.Path(__file__).resolve().parents[1] / "scripts" / "scripted_pick_place.py"
  )
  spec = importlib.util.spec_from_file_location("scripted_pick_place", path)
  sp = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = sp
  spec.loader.exec_module(sp)
  kin = sp.Kin.from_model(env.sim.mj_model)
  goal = np.array([T11_XY[0], T11_XY[1], 0.92])
  ql = kin.ik(goal, sp.hand_x_for(kin, 0.0), kin.q_default)
  assert ql is not None
  qr = np.asarray(ARM_MIRROR_SIGNS) * ql
  robot = env.scene["robot"]
  q = robot.data.joint_pos.clone()
  for i in range(7):
    q[:, robot.joint_names.index(f"arm_left_{i + 1}_joint")] = float(ql[i])
    q[:, robot.joint_names.index(f"arm_right_{i + 1}_joint")] = float(qr[i])
  robot.write_joint_state_to_sim(q, torch.zeros_like(q))
  term = env.action_manager.get_term("joint_pos")
  names = [n.split("/")[-1] for n in term.target_names]
  target = q[:, [robot.joint_names.index(n) for n in names]]
  act = (target - term._offset) / term._scale
  env.sim.forward()
  assert bool((mdp.arm_arm_clearance(env) < 0.0).all())
  assert bool((mdp.arm_arm_clearance_penalty(env, bi.HH_MARGIN) == 1.0).all())
  with torch.inference_mode():
    env.step(act)
  assert bool(mdp.contact_penalty(env, bi.ARM_ARM_SENSOR, bi.HH_CONTACT_FORCE).all())


def test_the_mirror_adapter_is_exact_on_mirrored_states(monkeypatch):
  """Plan S11's `MIRROR=right` (`scripts/eval_chain.py`): on a state and its mirror image -- the right
  arm at s * the left's joints and velocities, B at A's mirror image -- the right's adapted
  observation is the left's, to the FK's mirror precision. (Closed loop the paths drift apart by
  ~0.3 mm: the model's right arm links carry the left's centre-of-mass offsets, so its gravity
  torques are not the mirror image -- not the adapter's error; report §4.1.)"""
  import importlib.util
  import pathlib
  import sys

  path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "eval_chain.py"
  spec = importlib.util.spec_from_file_location("eval_chain", path)
  ec = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = ec
  spec.loader.exec_module(ec)
  env = _env(
    monkeypatch, "BI-J", num_envs=4, DR="DR0", LAG="none", HANDOFF_TRAIN="none"
  )
  ad = ec.MirrorAdapter(env)
  robot = env.scene["robot"]
  s = torch.tensor(ARM_MIRROR_SIGNS, device=env.device)
  left = [robot.joint_names.index(f"arm_left_{i}_joint") for i in range(1, 8)]
  right = [robot.joint_names.index(f"arm_right_{i}_joint") for i in range(1, 8)]
  lim = robot.data.joint_pos_limits[0]
  g = torch.Generator().manual_seed(0)
  q = robot.data.joint_pos.clone()
  qd = torch.zeros_like(q)
  for k in range(env.num_envs):
    ql = lim[left, 0] + (lim[left, 1] - lim[left, 0]) * torch.rand(7, generator=g).to(
      env.device
    )
    vl = (torch.rand(7, generator=g) - 0.5).to(env.device)
    q[k, left], q[k, right] = ql, s * ql
    qd[k, left], qd[k, right] = vl, s * vl
  robot.write_joint_state_to_sim(q, qd)
  a, b = env.scene["object"], env.scene[bi.CUBE_B]
  pose = a.data.root_link_pose_w.clone()
  o = env.scene.env_origins
  rel = pose[:, :3] - o
  rel[:, 1] = -rel[:, 1]
  w, x, y, z = pose[:, 3:7].unbind(-1)
  b.write_root_link_pose_to_sim(
    torch.cat([rel + o, torch.stack([w, -x, y, -z], -1)], -1)
  )
  env.sim.forward()
  t = ad.terms({"critic": env.observation_manager.compute_group("critic")}, "critic")
  goal = t["target_pos"] * torch.tensor([1.0, -1.0, 1.0], device=env.device)
  goal[:, 2] -= 0.05
  diff = (ad.left(t) - ad.right({**t, "goal": goal}, base_term="goal")).abs()
  assert float(diff.max()) < 1e-4, diff.amax(0)
