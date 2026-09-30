"""Plan S7-8 (Siwei, 2026-09-30 17:19-17:43): the arm at the stack task's speeds, with absolute and
with delta actions, and the cube and the target drawn from one wide area.

Part A -- what is checked here, on S7-6's RCTP-FJ plant (G20, baseline B, CTP, DR-scene, LAG=FP+JS):

* ``SLEW=stack`` builds Claudius's ``Slow25`` rates per joint (arm 1-2 1/16 of the URDF speed,
  arm 3 1/16, arm 4-7 1/8, the jaw 0.03 rad per step); ``ACTION``, ``CMD_OBS`` and ``EPISODE_S`` are
  read and validated; unset reproduces S7-6's cfg exactly;
* neither action mode ever moves a joint's command by more than its per-step limit, the jaw
  included; a delta action with a = 0 holds its target, clips at its box and does not wind up;
* ``CMD_OBS`` swaps the ``actions`` term for the commanded target and nothing else: the actor is
  38-D and the action 8-D in ABS-a, ABS-c and DEL.

Part B -- ``SPAWN=WC`` / ``TARGET=WC``: every cube and every target lies inside Claudius's
``-Wide`` area about the (possibly shifted) table, they are at least 10 cm apart, and the distance
is logged; unset reproduces W2 and B's target box.
"""

import dataclasses

import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from pal_mjlab.tasks.manipulation import mdp, scene
from pal_mjlab.tasks.manipulation import tabletop_env_cfg as tt
from pal_mjlab.tasks.manipulation.mdp.events import _surface_centre_xy
from pal_mjlab.tasks.manipulation.pick_place import pick_place_env_cfg as pp
from pal_mjlab.tasks.manipulation.pick_place.kangaroo import env_cfg as kenv
from pal_mjlab.tasks.manipulation.pick_place.rewards import S52_BASE
from pal_mjlab.tasks.manipulation.robots import (
  KANGAROO_GRIPPERS_LEFT_ARM_WC,
  KANGAROO_GRIPPERS_LEFT_ARM_WIDE,
  kangaroo_grippers_left_arm,
)

RCTP = f"{S52_BASE}+M1+M2+M3+M6+J+M8"
CTP = pp.EFFORT_BOUNDS["CTP"]
#: S7-6's RCTP-FJ, as `pick_place.sbatch` exports it.
S76_ENV = {
  "REWARD": RCTP,
  "DR": "DR-scene",
  "SLEW": "0.5",
  "BASELINE": "B",
  "SPAWN": "W2",
  "GAINS": "G20",
  "EFFORT": "CTP",
  "LAG": "FP+JS",
}
NEW_VARS = ("ACTION", "CMD_OBS", "TARGET", "EPISODE_S")


def _env_vars(monkeypatch, **changes):
  for var, value in {**S76_ENV, **changes}.items():
    if value is None:
      monkeypatch.delenv(var, raising=False)
    else:
      monkeypatch.setenv(var, value)
  for var in NEW_VARS:
    if var not in changes:
      monkeypatch.delenv(var, raising=False)


def _cfg(
  *,
  wc: bool = False,
  action: str = "abs",
  cmd_obs: bool = False,
  stack: bool = True,
  num_envs: int | None = None,
):
  arm = KANGAROO_GRIPPERS_LEFT_ARM_WC if wc else KANGAROO_GRIPPERS_LEFT_ARM_WIDE
  cfg = pp.make_pick_place_env_cfg(
    kangaroo_grippers_left_arm("G20", arm, arm_effort_scale=CTP.effort_scale),
    gripper_name="left",
    reward=RCTP,
    action_slew_rate_scale=1.0 if stack else 0.5,
    baseline=pp.BASELINES["B"],
    spawn=pp.SPAWN_BOXES["WC" if wc else "W2"],
    dr="DR-scene",
    effort=CTP,
    latency=tt.LATENCY_MODELS["FP+JS"],
    target_area=pp.TARGET_AREAS["WC"] if wc else None,
    stack_speed=stack,
    action_mode=action,
    command_obs=cmd_obs,
  )
  if num_envs is not None:
    cfg.scene.num_envs = num_envs
  return cfg


# -- Configuration --------------------------------------------------------------------------


def test_the_stack_rates_are_claudius_slow25():
  """`origin/faster-moves` b236d62: 0.25 x STANDING_SLEW_RATE_SCALE (arm 1-3 0.25, arm 4-7 0.5) on the
  URDF speeds, i.e. 1/16 and 1/8; the jaw 0.15 x JAW_SLEW_LIMIT (0.20)."""
  rates = tt.stack_arm_rate_limit(KANGAROO_GRIPPERS_LEFT_ARM_WIDE.joint_velocity_limits)
  dt = tt.CONTROL_DT
  assert rates == {
    "arm_.*_1_joint": pytest.approx(1.95 / 16 * dt),
    "arm_.*_2_joint": pytest.approx(1.95 / 16 * dt),
    "arm_.*_3_joint": pytest.approx(2.35 / 16 * dt),
    "arm_.*_4_joint": pytest.approx(2.35 / 8 * dt),
    "arm_.*_5_joint": pytest.approx(2.35 / 8 * dt),
    "arm_.*_6_joint": pytest.approx(2.35 / 8 * dt),
    "arm_.*_7_joint": pytest.approx(2.35 / 8 * dt),
  }
  # rad/s, as the plan's table gives them.
  assert rates["arm_.*_1_joint"] / dt == pytest.approx(0.122, abs=1e-3)
  assert rates["arm_.*_3_joint"] / dt == pytest.approx(0.147, abs=1e-3)
  assert rates["arm_.*_7_joint"] / dt == pytest.approx(0.294, abs=1e-3)
  assert tt.STACK_JAW_STEP / dt == pytest.approx(1.5)
  assert tt.STACK_JAW_STEP == pytest.approx(0.15 * 0.20)


@pytest.mark.parametrize("action", ["abs", "delta"])
def test_stack_speed_sets_the_rates_on_the_action_term(action):
  term = _cfg(action=action).actions["joint_pos"]
  assert type(term).__name__ == (
    "DeltaRateLimitedJointPositionActionCfg"
    if action == "delta"
    else "RateLimitedJointPositionActionCfg"
  )
  assert term.rate_limit["gripper_left_finger_joint"] == pytest.approx(0.03)
  assert term.rate_limit["arm_.*_2_joint"] == pytest.approx(1.95 / 16 * tt.CONTROL_DT)
  assert set(term.rate_limit) == {f"arm_.*_{i}_joint" for i in range(1, 8)} | {
    "gripper_left_finger_joint"
  }
  # The plant's bounds and the jaw servo are still there.
  assert term.range_clip_joints and term.effort_band_joints and term.servo_joints


def test_unset_reproduces_s76(monkeypatch):
  """No S7-8 variable set: exactly what the S7-6 selectors built before this stage."""
  _env_vars(monkeypatch)
  got = kenv.kangaroo_pick_place_env_cfg()
  want = pp.make_pick_place_env_cfg(
    kangaroo_grippers_left_arm(
      "G20", KANGAROO_GRIPPERS_LEFT_ARM_WIDE, arm_effort_scale=CTP.effort_scale
    ),
    gripper_name="left",
    reward=RCTP,
    action_slew_rate_scale=0.5,
    baseline=pp.BASELINES["B"],
    spawn=pp.SPAWN_BOXES["W2"],
    dr="DR-scene",
    effort=CTP,
    latency=tt.LATENCY_MODELS["FP+JS"],
  )
  for field in ("actions", "observations", "commands", "events", "terminations"):
    assert getattr(got, field) == getattr(want, field), field
  assert got.episode_length_s == want.episode_length_s == 10.0
  assert set(got.rewards) == set(want.rewards)
  # The default arm is the W2 one, the target the box B has always drawn from.
  assert got.commands[tt.COMMAND].min_object_distance == 0.0
  assert got.commands[tt.COMMAND].offset_range == (0.02, 0.02)


def test_the_selectors_are_read_and_validated(monkeypatch):
  _env_vars(
    monkeypatch, SLEW="stack", ACTION="delta", SPAWN="WC", TARGET="WC", EPISODE_S="30"
  )
  cfg = kenv.kangaroo_pick_place_env_cfg()
  assert (
    type(cfg.actions["joint_pos"]).__name__ == "DeltaRateLimitedJointPositionActionCfg"
  )
  assert cfg.episode_length_s == 30.0
  assert cfg.commands[tt.COMMAND].min_object_distance == 0.10
  # DEL observes its target without being told to; the actor stays 38-D.
  assert cfg.observations["actor"].terms["actions"].func is mdp.commanded_target_rel
  assert cfg.observations["critic"].terms["actions"].func is mdp.commanded_target_rel
  assert list(cfg.observations["actor"].terms) == [
    "joint_pos", "joint_vel", "actions", "tcp_pos", "tcp_quat", "object_pos", "object_yaw",
    "target_pos",
  ]  # fmt: skip
  for var, value in (
    ("SLEW", "fast"),
    ("SLEW", "1.5"),
    ("ACTION", "velocity"),
    ("CMD_OBS", "target"),
    ("TARGET", "everywhere"),
    ("EPISODE_S", "0"),
    ("EPISODE_S", "nan"),
    ("EPISODE_S", "long"),
  ):
    _env_vars(monkeypatch, **{var: value})
    with pytest.raises(ValueError, match=var):
      kenv.kangaroo_pick_place_env_cfg()
  # A delta action integrates: it may not be shown the raw last action.
  _env_vars(monkeypatch, SLEW="stack", ACTION="delta", CMD_OBS="act")
  with pytest.raises(ValueError, match="delta"):
    kenv.kangaroo_pick_place_env_cfg()
  # ABS-a is the default for abs, ABS-c asks for the target.
  _env_vars(monkeypatch, SLEW="stack")
  abs_a = kenv.kangaroo_pick_place_env_cfg()
  assert abs_a.observations["actor"].terms["actions"].func is mdp.last_action
  _env_vars(monkeypatch, SLEW="stack", CMD_OBS="cmd")
  abs_c = kenv.kangaroo_pick_place_env_cfg()
  assert abs_c.observations["actor"].terms["actions"].func is mdp.commanded_target_rel
  assert list(abs_c.observations["actor"].terms) == list(
    abs_a.observations["actor"].terms
  )


def test_stack_speed_and_a_factor_do_not_mix_and_delta_has_no_tool_cap():
  with pytest.raises(ValueError, match="stack_speed"):
    pp.make_pick_place_env_cfg(
      kangaroo_grippers_left_arm("G20", KANGAROO_GRIPPERS_LEFT_ARM_WIDE),
      gripper_name="left",
      action_slew_rate_scale=0.5,
      stack_speed=True,
    )
  with pytest.raises(ValueError, match="tool speed"):
    pp.make_pick_place_env_cfg(
      kangaroo_grippers_left_arm("G20", KANGAROO_GRIPPERS_LEFT_ARM_WIDE),
      gripper_name="left",
      tool_speed_cap=pp.TOOL_SPEED_CAPS["V"],
      stack_speed=True,
      action_mode="delta",
    )


def test_cmd_obs_changes_only_the_actions_term():
  base, cmd = _cfg(cmd_obs=False), _cfg(cmd_obs=True)
  for group in ("actor", "critic"):
    got, want = cmd.observations[group].terms, base.observations[group].terms
    assert list(got) == list(want)
    for name in want:
      if name == "actions":
        assert got[name].func is mdp.commanded_target_rel
        assert want[name].func is mdp.last_action
      else:
        assert got[name] == want[name], (group, name)


def test_the_area_and_the_scales():
  wc = pp.SPAWN_BOXES["WC"]
  # Claudius's WIDE_OBJECT_POSE_RANGE about the table centre (0.55, 0): x -0.15..+0.05, y -0.04..+0.26.
  assert wc.x == pytest.approx((0.40, 0.60))
  assert wc.y == pytest.approx((-0.04, 0.26))
  assert not wc.widen_at_dr_scene
  assert pp._object_pose_range(wc) == {
    "x": pytest.approx((-0.15, 0.05)),
    "y": pytest.approx((-0.04, 0.26)),
    "yaw": pytest.approx((-0.7853981633974483, 0.7853981633974483)),
  }
  assert pp.TARGET_AREAS["B"] is None and pp.TARGET_AREAS["WC"] is wc
  assert pp.TARGET_MIN_SEPARATION == 0.10
  cmd = pp._command_cfg((0.0, 0.0), "DR-scene", target_area=wc)
  assert cmd.offset == pytest.approx((-0.05, 0.11))
  assert cmd.offset_range == pytest.approx((0.10, 0.15))
  assert cmd.min_object_distance == 0.10
  # The WC arm has W2's action space widened where the area needs it, never narrowed.
  w, c = (
    KANGAROO_GRIPPERS_LEFT_ARM_WIDE.action_scale,
    KANGAROO_GRIPPERS_LEFT_ARM_WC.action_scale,
  )
  assert set(w) == set(c)
  assert all(c[k] >= w[k] for k in w), (w, c)
  assert (
    dataclasses.replace(KANGAROO_GRIPPERS_LEFT_ARM_WC, action_scale=w)
    == KANGAROO_GRIPPERS_LEFT_ARM_WIDE
  )


# -- The action terms -----------------------------------------------------------------------

N_ENVS = 8


def _term(env):
  return env.action_manager.get_term("joint_pos")


def _run(env, actions, steps: int):
  """Step ``steps`` times; the commanded target after each step, which envs were still in the
  same episode (a reset puts the command back at the default pose) and which had T's effort band
  or C's range clip move the target (`RateLimitedJointPositionAction._bound`)."""
  term, out, alive, bound = _term(env), [], [], []
  for t in range(steps):
    _, _, terminated, truncated, _ = env.step(actions(t))
    out.append(term.commanded_target.clone())
    alive.append(~(terminated | truncated))
    bound.append(term.band_hit | term.range_hit)
  return torch.stack(out), torch.stack(alive), torch.stack(bound)


@pytest.mark.parametrize("action", ["abs", "delta"])
def test_no_command_ever_moves_more_than_its_step_limit(action):
  env = ManagerBasedRlEnv(
    cfg=_cfg(action=action, num_envs=N_ENVS), device=get_test_device()
  )
  try:
    env.reset()
    term = _term(env)
    rate = term._rate[0]
    assert torch.isfinite(rate).all(), "every joint has a limit, the jaw included"
    torch.manual_seed(0)
    shape = env.action_space.shape

    def saturated(t):
      # Random signs, resampled every 8 steps: full-scale reversals.
      if t % 8 == 0:
        env._s78_sign = torch.sign(torch.randn(shape, device=env.device))
      return 5.0 * env._s78_sign

    cmd, alive, bound = _run(env, saturated, 100)
    prev = torch.cat([term._default_target().unsqueeze(0), cmd[:-1]])
    step = (cmd - prev).abs()
    # Steps in the same episode as the one before, and not pulled back by T's band. The band keeps
    # the target within the PD's effort cap of the *measured* position, so a joint a contact holds
    # back sees its target retreat faster than the slew (S7-4); a command never advances faster.
    same = alive & torch.cat([torch.ones_like(alive[:1]), alive[:-1]])
    ok = (same & ~bound).unsqueeze(-1)
    assert int(ok.sum()) > 0.5 * ok.numel() / 8, "most steps are free of the band"
    excess = (step - rate).masked_fill(~ok, -1.0)
    assert float(excess.max()) <= 1e-7, (
      action,
      excess.max(dim=1).values.max(dim=0).values,
    )
    # And the limits are actually reached: the saturated actions do move at full speed.
    reached = (
      (step - rate).abs().masked_fill(~ok, 1.0).min(dim=0).values.min(dim=0).values
    )
    assert bool((reached < 1e-6).all()), reached
  finally:
    env.close()


def test_delta_holds_at_zero_clips_at_the_box_and_does_not_wind_up():
  env = ManagerBasedRlEnv(
    cfg=_cfg(action="delta", num_envs=4), device=get_test_device()
  )
  try:
    env.reset()
    term = _term(env)
    rate, lo, hi = term._rate[0], term._box_lo, term._box_hi
    zero = torch.zeros(env.action_space.shape, device=env.device)
    # a = 0 holds the target, from the default pose and after moving.
    start = term.commanded_target.clone()
    for _ in range(10):
      env.step(zero)
    assert torch.equal(term.commanded_target, start)
    push = zero.clone()
    push[:, 2] = 1.0  # arm 3, whose box is 0.65 rad wide: 0.65 / 0.00294 = 222 steps
    push[:, -1] = 1.0  # the jaw, 0.95 rad / 0.03 = 32 steps
    for _ in range(15):
      env.step(push)
    # float32: fifteen additions to a number near 1.8 round at 1e-7 each.
    assert torch.allclose(
      term.commanded_target[:, 2] - start[:, 2], 15 * rate[2].expand(4), atol=5e-6
    )
    held = term.commanded_target.clone()
    for _ in range(10):
      env.step(zero)
    assert torch.equal(term.commanded_target, held)
    # The jaw runs into its closed stop and stays there: it is never past the box...
    for _ in range(60):
      env.step(push)
      assert bool((term.commanded_target <= hi + 1e-7).all())
    jaw = -1
    assert torch.allclose(term.commanded_target[:, jaw], hi[:, jaw], atol=1e-6)
    # ...and one step back leaves the stop by exactly one step: no wind-up.
    back = zero.clone()
    back[:, jaw] = -1.0
    env.step(back)
    assert torch.allclose(
      term.commanded_target[:, jaw], hi[:, jaw] - rate[jaw], atol=1e-6
    )
    # Nor does an action far beyond 1 move it faster than a step.
    before = term.commanded_target.clone()
    env.step(zero + 100.0)
    assert bool(((term.commanded_target - before).abs() <= rate + 1e-7).all())
    assert bool((term.commanded_target >= lo - 1e-7).all())
  finally:
    env.close()


def test_delta_starts_from_the_default_pose_at_every_reset():
  env = ManagerBasedRlEnv(
    cfg=_cfg(action="delta", num_envs=4), device=get_test_device()
  )
  try:
    env.reset()
    term = _term(env)
    move = torch.ones(env.action_space.shape, device=env.device)
    for _ in range(20):
      env.step(move)
    assert not torch.allclose(term.commanded_target, term._default_target())
    env.reset()
    assert torch.allclose(term.commanded_target, term._default_target())
  finally:
    env.close()


@pytest.mark.parametrize(
  ("action", "cmd_obs"), [("abs", False), ("abs", True), ("delta", True)]
)
def test_the_actor_is_38d_and_the_action_8d(action, cmd_obs):
  env = ManagerBasedRlEnv(
    cfg=_cfg(action=action, cmd_obs=cmd_obs, num_envs=4), device=get_test_device()
  )
  try:
    obs, _ = env.reset()
    assert obs["actor"].shape == (4, 38), obs["actor"].shape
    assert env.action_space.shape == (4, 8)
    term = _term(env)
    move = 5.0 * torch.ones(env.action_space.shape, device=env.device)
    for _ in range(12):
      env.step(move)
    obs = env.get_observations()
    names = list(env.observation_manager.active_terms["actor"])
    dims = env.observation_manager.group_obs_term_dim["actor"]
    start = sum(d[-1] for d in dims[: names.index("actions")])
    seen = obs["actor"][:, start : start + 8]
    target = (
      term.commanded_target
      - env.scene["robot"].data.default_joint_pos[:, term._target_ids]
    )
    if cmd_obs:
      assert torch.allclose(seen, target, atol=1e-6)
      assert float(target.abs().max()) > 0.01
    else:
      # The raw last action, un-noised in the actor: what the policy sent, not what was commanded.
      assert torch.allclose(seen, term.raw_action, atol=1e-6)
  finally:
    env.close()


# -- The wide area --------------------------------------------------------------------------

AREA_X, AREA_Y = (-0.15, 0.05), (-0.04, 0.26)  # about the table centre


def _wc_env(num_envs: int = 256):
  return ManagerBasedRlEnv(
    cfg=_cfg(wc=True, action="delta", cmd_obs=True, num_envs=num_envs),
    device=get_test_device(),
  )


def _cube_and_goal(env):
  centre = _surface_centre_xy(env, SceneEntityCfg("table"))
  cube = env.scene[pp.OBJECT].data.root_link_pos_w[:, :2]
  goal = env.command_manager.get_term(tt.COMMAND).target_pos[:, :2]
  return centre, cube, goal


def test_every_cube_and_target_is_in_the_area_at_least_10_cm_apart():
  env = _wc_env()
  try:
    lo = torch.tensor([AREA_X[0], AREA_Y[0]], device=env.device)
    hi = torch.tensor([AREA_X[1], AREA_Y[1]], device=env.device)
    seen_cube, seen_goal, dists = [], [], []
    command = env.command_manager.get_term(tt.COMMAND)
    for _ in range(6):
      env.reset()
      env.sim.forward()
      centre, cube, goal = _cube_and_goal(env)
      # The table moved (DR-scene), and both are drawn about wherever it is.
      assert bool(
        ((centre - torch.tensor([0.55, 0.0], device=env.device)).abs() > 1e-4).any()
      )
      rel_c, rel_g = cube - centre, goal - centre
      eps = 1e-4
      assert bool(((rel_c >= lo - eps) & (rel_c <= hi + eps)).all()), (
        "cube outside the area"
      )
      assert bool(((rel_g >= lo - eps) & (rel_g <= hi + eps)).all()), (
        "target outside the area"
      )
      d = torch.norm(cube - goal, dim=-1)
      assert float(d.min()) >= 0.10 - 1e-4, float(d.min())
      assert torch.allclose(command.metrics["carry_distance"], d, atol=1e-4)
      seen_cube.append(rel_c)
      seen_goal.append(rel_g)
      dists.append(d)
    cube, goal, d = torch.cat(seen_cube), torch.cat(seen_goal), torch.cat(dists)
    # It really is the whole area, not a corner of it: both span (nearly) all of it...
    for pts in (cube, goal):
      assert bool((pts.min(0).values < lo + 0.02).all()) and bool(
        (pts.max(0).values > hi - 0.02).all()
      )
    # ...and the carries are of every length the area allows: 10 cm up to its diagonal (36 cm).
    assert float(d.max()) > 0.30 and float(d.min()) < 0.13
    # The area's far side is used as a target as well as a cube's spawn (not "always the same way").
    assert (
      float((goal[:, 0] - cube[:, 0]).min()) < -0.05
      and float((goal[:, 0] - cube[:, 0]).max()) > 0.05
    )
  finally:
    env.close()


def test_the_fallback_when_rejection_gives_up_is_still_10_cm_away(monkeypatch):
  monkeypatch.setattr(mdp.SurfaceOffsetCommand, "MAX_REDRAWS", 0)
  env = _wc_env(num_envs=512)
  try:
    env.reset()
    env.sim.forward()
    _, cube, goal = _cube_and_goal(env)
    assert float(torch.norm(cube - goal, dim=-1).min()) >= 0.10 - 1e-4
  finally:
    env.close()


def test_the_goal_follows_the_table_once_drawn():
  """The goal is table-relative, so a step later it is still where it was drawn about the table."""
  env = _wc_env(num_envs=8)
  try:
    env.reset()
    command = env.command_manager.get_term(tt.COMMAND)
    drawn = command.offsets.clone()
    for _ in range(3):
      env.step(torch.zeros(env.action_space.shape, device=env.device))
    assert torch.equal(command.offsets, drawn)
    centre = _surface_centre_xy(env, SceneEntityCfg("table"))
    assert torch.allclose(command.target_pos[:, :2], centre + drawn, atol=1e-6)
  finally:
    env.close()


def test_b_and_w2_are_untouched_by_the_area_code():
  """Without TARGET=WC the goal is drawn once, from B's box, at any distance from the cube."""
  env = ManagerBasedRlEnv(cfg=_cfg(stack=False, num_envs=64), device=get_test_device())
  try:
    env.reset()
    command = env.command_manager.get_term(tt.COMMAND)
    assert "carry_distance" not in command.metrics
    rel = command.offsets - torch.tensor(
      [0.46 - scene.TABLE_POS[0], 0.08 - scene.TABLE_POS[1]], device=env.device
    )
    assert bool((rel.abs() <= 0.02 + 1e-6).all())
  finally:
    env.close()


# -- The eval and the scripted rollout ------------------------------------------------------


def _load_script(name: str):
  import importlib.util
  import pathlib
  import sys

  path = pathlib.Path(__file__).parent.parent / "scripts" / f"{name}.py"
  spec = importlib.util.spec_from_file_location(f"{name}_s78", path)
  assert spec and spec.loader
  mod = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = mod
  spec.loader.exec_module(mod)
  return mod


def test_the_eval_reads_the_new_flag_and_bins_by_carry_and_spawn():
  ev = _load_script("eval_pick_place")
  args = ev.parse_args(["--checkpoint", "x.pt", "--handoff"])
  assert args.handoff_timeout == ev.HANDOFF_TIMEOUT_S == 8.0
  args = ev.parse_args(["--checkpoint", "x.pt", "--handoff", "--handoff-timeout", "21"])
  assert args.handoff_timeout == 21.0
  assert ev.CARRY_BINS == ((0.10, 0.20), (0.20, 0.30), (0.30, 9.99))
  # The old W2 box about the table centre is where `SPAWN=W2` draws the cube.
  w2 = pp.SPAWN_BOXES["W2"]
  assert w2.x == pytest.approx(
    (ev.TABLE_CENTRE_XY[0] + ev.W2_REL_X[0], ev.TABLE_CENTRE_XY[0] + ev.W2_REL_X[1])
  )
  assert w2.y == pytest.approx(
    (ev.TABLE_CENTRE_XY[1] + ev.W2_REL_Y[0], ev.TABLE_CENTRE_XY[1] + ev.W2_REL_Y[1])
  )


def test_the_delta_handoff_walks_the_target_home_at_the_stack_speed():
  ev = _load_script("eval_pick_place")
  env = ManagerBasedRlEnv(
    cfg=_cfg(action="delta", cmd_obs=True, num_envs=2), device=get_test_device()
  )
  try:
    env.reset()
    term = _term(env)
    zero = torch.zeros(env.action_space.shape, device=env.device)
    move = zero.clone()
    move[:, 0], move[:, 3], move[:, -1] = 1.0, -1.0, 1.0
    for _ in range(40):
      env.step(move)
    away = (term.commanded_target - term._default_target()).abs()
    assert float(away.max()) > 0.05
    steps = 0
    for _ in range(600):
      steps += 1
      a = ev.handoff_action(env, zero)
      assert bool((a.abs() <= 1.0 + 1e-7).all())
      env.step(a)
      if float((term.commanded_target - term._default_target()).abs().max()) < 1e-6:
        break
    assert float((term.commanded_target - term._default_target()).abs().max()) < 1e-6
    assert steps < 200, "40 steps out at full speed is back in about as many"
    # Once home, the hand-off holds: zero action.
    assert float(ev.handoff_action(env, zero).abs().max()) < 1e-6
  finally:
    env.close()
  # An absolute action's hand-off is action 0, as it always was.
  env = ManagerBasedRlEnv(cfg=_cfg(num_envs=2), device=get_test_device())
  try:
    env.reset()
    zero = torch.zeros(env.action_space.shape, device=env.device)
    assert ev.handoff_action(env, zero) is zero
  finally:
    env.close()


def test_the_scripted_rollout_takes_the_new_options():
  """The script's CLI reaches the env: stack speed, delta, JS, the wide goal."""
  import sys

  sp = _load_script("scripted_pick_place")
  argv = [
    "x", "rollout", "--stack-speed", "--action", "delta", "--lag", "JS", "--target", "WC",
    "--spawn", "WC", "--baseline", "B", "--dr", "DR-scene", "--gains", "G20",
    "--effort", "CTP", "--episode-s", "60",
  ]  # fmt: skip
  old, sys.argv = sys.argv, argv
  try:
    args = sp.parse_args()
  finally:
    sys.argv = old
  assert args.stack_speed and args.action == "delta" and args.lag == "JS"
  assert args.target == "WC" and args.episode_s == 60.0
