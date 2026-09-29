"""Plan S7-7 (Siwei, 2026-09-29): the cube goes out of the camera's view, and the actor is told.

Checked here, on S7-6's plant (G20, W2's arm, baseline B, CTP, DR-scene, ``LAG=FP+JS``):

* ``OCC`` / ``OBJ_OBS`` are read and validated, and unset build exactly S7-6's cfg;
* Claudius's line-of-sight distance (`90c84f5`), ported unchanged;
* the flag and the zero fill: a lost cube is exactly zero, the flag says so, the shapes are 39-D
  (O-Z, O-K), 38-D (O-S) and the critic reads the live cube plus the flag and the steps unseen;
* a lost cube is reported lost with the drawn lag, the same 5-8 steps as its pose;
* O-K rides a scripted carry within 5 mm, holds a resting cube exactly, lets go where the hand
  let go, and gives the belief up when the cube turns up somewhere else;
* O-S never changes after the first step and never shows the previous episode's cube.

The scripted scenes replace the sight line (a schedule), the TCP (a point that follows the cube)
and the jaw (a measured angle and a command): O-K's logic is under test, not MuJoCo's contacts.
"""

import dataclasses

import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from pal_mjlab.tasks.manipulation import mdp
from pal_mjlab.tasks.manipulation import tabletop_env_cfg as tt
from pal_mjlab.tasks.manipulation.mdp import occlusion as occ
from pal_mjlab.tasks.manipulation.mdp.observations import _to_base_frame
from pal_mjlab.tasks.manipulation.pick_place import pick_place_env_cfg as pp
from pal_mjlab.tasks.manipulation.pick_place.kangaroo import env_cfg as kenv
from pal_mjlab.tasks.manipulation.pick_place.rewards import S52_BASE
from pal_mjlab.tasks.manipulation.robots import (
  KANGAROO_GRIPPERS_LEFT_ARM_WIDE,
  kangaroo_grippers_left_arm,
)

R = f"{S52_BASE}+M1+M2+M3+M6+J+M8"
CTP = pp.EFFORT_BOUNDS["CTP"]
S76_ENV = {
  "REWARD": R,
  "DR": "DR-scene",
  "SLEW": "0.5",
  "BASELINE": "B",
  "SPAWN": "W2",
  "GAINS": "G20",
  "EFFORT": "CTP",
  "LAG": "FP+JS",
}
N_ENVS = 6


def _cfg(occlusion: occ.OccCfg | None, num_envs: int | None = None):
  cfg = pp.make_pick_place_env_cfg(
    kangaroo_grippers_left_arm(
      "G20", KANGAROO_GRIPPERS_LEFT_ARM_WIDE, arm_effort_scale=CTP.effort_scale
    ),
    gripper_name="left",
    reward=R,
    action_slew_rate_scale=0.5,
    baseline=pp.BASELINES["B"],
    spawn=pp.SPAWN_BOXES["W2"],
    dr="DR-scene",
    effort=CTP,
    latency=tt.LATENCY_MODELS["FP+JS"],
    occlusion=occlusion,
  )
  if num_envs is not None:
    cfg.scene.num_envs = num_envs
  return cfg


def _occ(fill: str, flag: bool | None = None, **kw) -> occ.OccCfg:
  """A deterministic occlusion config: the sight line is scripted and there is no random loss."""
  base = occ.occ_cfg("arm", fill, flag)
  assert base is not None
  return dataclasses.replace(base, dropout=None, **kw)


# -- Configuration --------------------------------------------------------------------------


def test_selectors_are_read_and_validated(monkeypatch):
  for var, value in S76_ENV.items():
    monkeypatch.setenv(var, value)
  for var in ("OCC", "OBJ_OBS", "OCC_FLAG"):
    monkeypatch.delenv(var, raising=False)
  unset = kenv.kangaroo_pick_place_env_cfg()
  monkeypatch.setenv("OCC", "none")
  monkeypatch.setenv("OBJ_OBS", "live")
  none = kenv.kangaroo_pick_place_env_cfg()
  assert unset.observations == none.observations, "OCC=none is S7-6's cfg"
  assert unset.actions == none.actions
  assert unset.observations == _cfg(None).observations

  monkeypatch.setenv("OCC", "hand")
  monkeypatch.setenv("OBJ_OBS", "estimate")
  cfg = kenv.kangaroo_pick_place_env_cfg()
  term = cfg.observations["actor"].terms["object_pos"]
  assert term.params["occ"].fill == "estimate" and term.params["occ"].view == "hand"
  assert "object_seen" in cfg.observations["actor"].terms
  monkeypatch.setenv("OCC_FLAG", "0")
  no_flag = kenv.kangaroo_pick_place_env_cfg()
  assert "object_seen" not in no_flag.observations["actor"].terms
  monkeypatch.delenv("OCC_FLAG")
  monkeypatch.setenv("OBJ_OBS", "snap")
  monkeypatch.setenv("OCC", "none")
  snap = kenv.kangaroo_pick_place_env_cfg()  # O-S is independent of the occlusion model
  assert snap.observations["actor"].terms["object_pos"].params["occ"].fill == "snap"
  # OCC=none with a filler: the machinery on, nothing hidden (a policy evaluated without OCC).
  monkeypatch.setenv("OBJ_OBS", "zero")
  clear = kenv.kangaroo_pick_place_env_cfg().observations["actor"].terms
  assert clear["object_pos"].params["occ"].view == "none"
  assert clear["object_pos"].params["occ"].dropout is None and "object_seen" in clear
  for bad in (("arm", "live"), ("nope", "zero"), ("arm", "nope")):
    monkeypatch.setenv("OCC", bad[0])
    monkeypatch.setenv("OBJ_OBS", bad[1])
    with pytest.raises(ValueError):
      kenv.kangaroo_pick_place_env_cfg()


def test_the_terms_of_each_arm():
  base = _cfg(None).observations
  for fill, flag in (
    ("zero", True),
    ("estimate", True),
    ("snap", False),
    ("hold", False),
  ):
    cfg = _cfg(_occ(fill))
    actor = cfg.observations["actor"].terms
    critic = cfg.observations["critic"].terms
    assert ("object_seen" in actor) == flag, fill
    # Same terms in the same order, the flag right after the cube's yaw.
    names = [n for n in actor if n != "object_seen"]
    assert names == list(base["actor"].terms)
    if flag:
      assert list(actor).index("object_seen") == list(actor).index("object_yaw") + 1
    # The actor's cube terms report through the sensor, noise inside it (no mjlab noise, no
    # term-level delay); the critic keeps S7-6's live cube and gains the flag and steps unseen.
    for name in ("object_pos", "object_yaw"):
      assert actor[name].noise is None
      assert (actor[name].delay_min_lag, actor[name].delay_max_lag) == (0, 0)
      assert critic[name] == base["critic"].terms[name]
      assert critic[name].params["lag"] is None
    assert {"object_seen", "object_unseen"} <= set(critic)
    assert actor["object_pos"].params["lag"] == (5, 8)


def test_the_sensor_carries_the_actors_noise():
  """The noise mjlab's model would have applied moves into the sensor with the same bounds."""
  base = _cfg(None).observations["actor"].terms
  cfg = _cfg(_occ("zero"))
  bias_pos, jit_pos, bias_yaw, jit_yaw = (
    cfg.observations["actor"].terms["object_pos"].params["occ"].noise
  )
  assert bias_pos == base["object_pos"].noise.bias_noise_cfg.n_max == 0.008
  assert jit_pos == base["object_pos"].noise.noise_cfg.n_max == 0.002
  assert bias_yaw == base["object_yaw"].noise.bias_noise_cfg.n_max == 0.03
  assert jit_yaw == base["object_yaw"].noise.noise_cfg.n_max == 0.01


@pytest.mark.parametrize(
  ("p0", "p1", "q0", "q1", "expected"),
  [
    ((0, 0, 0), (1, 0, 0), (0.5, 1, 0), (0.5, 2, 0), 1.0),  # apart, perpendicular
    ((0, 0, 0), (1, 0, 0), (0.5, -1, 1), (0.5, 1, 1), 1.0),  # crossing above
    ((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), 1.0),  # collinear, with a gap
    ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), 1.0),  # parallel
    ((0, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0), 0.0),  # a point on the end
    ((0, 0, 0), (1, 0, 0), (0.3, 0.2, 0), (0.3, 0.2, 0), 0.2),  # a point beside it
    ((0, 0, 0), (0, 0, 0), (0, 1, 0), (0, 2, 0), 1.0),  # the first a point
    ((0, 0, 0), (1, 0, 0), (-1, -1, 0), (-2, -1, 0), 2**0.5),  # both ends clamped
  ],
)
def test_segment_distance(p0, p1, q0, q1, expected):
  """Claudius's `tests/test_camera_view.py` (90c84f5), on the ported function."""
  t = lambda v: torch.tensor([v], dtype=torch.float64)  # noqa: E731
  got = occ.segment_distance(t(p0), t(p1), t(q0), t(q1))
  assert got.item() == pytest.approx(expected, abs=1e-9)


# -- The sensor on scripted scenes ----------------------------------------------------------

#: The jaw's scripted (measured angle, command), rad: open, then closed on a cube (held),
#: opening again.
JAW_OPEN, JAW_HELD = (0.05, 0.0), (0.30, 0.55)
#: The cube hovers 10 cm above the table so the arm never meets it (as S7-6's tracker test).
HOVER = 0.10
#: Where the (scripted) TCP is in the cube's frame while it is held: off the closing axis.
OFFSET = torch.tensor([0.0, 0.01, 0.02])


class Scene:
  """A scripted world for `CubeSensor`: the cube's pose, the sight line, the TCP and the jaw
  are functions of the step number, and the sensor reads them through patched hooks."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    monkeypatch,
    blocked,
    jaw,
    cube_x,
    carry=True,
    tcp_x=None,
  ):
    self.env = env
    self.obj = env.scene[pp.OBJECT]
    self.start = None
    self.t = 0
    self.blocked, self.jaw, self.cube_x, self.carry = blocked, jaw, cube_x, carry
    self.tcp_x = tcp_x
    scene = self

    def view(env, camera_body, camera_pos, segments, target_w, stop, margin, **_):
      hidden = scene.blocked(scene.t)
      return torch.full(
        (target_w.shape[0],), 1.0 if hidden else 0.0, device=target_w.device
      )

    def tcp(sensor, env):
      pos = scene.obj.data.root_link_pos_w.clone() - (
        OFFSET.to(env.device)
        if scene.carry
        else torch.tensor([0.0, 0.3, 0.0], device=env.device)
      )
      if (
        scene.tcp_x is not None and scene.start is not None
      ):  # a hand with a path of its own
        pos = scene.start[:, :3].clone() - OFFSET.to(env.device)
        pos[:, 0] += scene.tcp_x(scene.t)
        pos[:, 2] += HOVER
      quat = torch.zeros(env.num_envs, 4, device=env.device)
      quat[:, 0] = 1.0
      return pos, quat

    def jaw_state(sensor, env):
      q, cmd = scene.jaw(scene.t)
      full = lambda v: torch.full((env.num_envs,), v, device=env.device)  # noqa: E731
      return full(q), full(cmd)

    monkeypatch.setattr(occ, "camera_view_blocked", view)
    monkeypatch.setattr(occ.CubeSensor, "_tcp", tcp)
    monkeypatch.setattr(occ.CubeSensor, "_jaw", jaw_state)

  def reset(self, warm: int = 10) -> None:
    """Reset the env and let the scripted world settle: the cube hovers and has been seen there
    for longer than the tracker's lag, so no history holds the table pose or the real geometry."""
    self.env.reset()
    self.start = self.obj.data.root_link_pose_w.clone()
    for _ in range(warm):
      self.t = -1  # the schedules' t < 0: seen, jaw open, cube at rest
      self.step()
    self.t = 0

  def step(self) -> occ.CubeSensor:
    pose = self.start.clone()
    pose[:, 0] += self.cube_x(self.t)
    pose[:, 2] += HOVER
    self.obj.write_root_link_pose_to_sim(pose)
    # Hovering: no velocity gathers between the writes.
    self.obj.write_root_link_velocity_to_sim(
      torch.zeros(self.env.num_envs, 6, device=self.env.device)
    )
    self.env.step(torch.zeros(self.env.action_space.shape, device=self.env.device))
    self.t += 1
    return self.env._pal_cube_sensors[pp.OBJECT]

  def live(self) -> torch.Tensor:
    return self.obj.data.root_link_pos_w.clone()


def _env(occlusion: occ.OccCfg) -> ManagerBasedRlEnv:
  return ManagerBasedRlEnv(cfg=_cfg(occlusion, N_ENVS), device=get_test_device())


def _terms(env, name: str):
  """An actor term's function and params, as the observation manager calls them."""
  term = env.cfg.observations["actor"].terms[name]
  return term.func, term.params


def test_a_lost_cube_is_reported_lost_with_the_drawn_lag_and_zero_and_flagged(
  monkeypatch,
):
  env = _env(_occ("zero"))
  try:
    lost_from = 20
    scene = Scene(
      env, monkeypatch, lambda t: t >= lost_from, lambda t: JAW_OPEN, lambda t: 0.0
    )
    scene.reset()
    for _ in range(lost_from + 12):
      sensor = scene.step()
      t = scene.t - 1
      lag = sensor.tracker_lag
      # Reported lost as late as the pose: the same drawn 5-8 steps.
      want_lost = t >= lost_from + lag
      assert bool((~sensor.seen == want_lost).all()), (t, lag, sensor.seen)
      func, params = _terms(env, "object_pos")
      pos = func(env, **params)
      yfunc, yparams = _terms(env, "object_yaw")
      yaw = yfunc(env, **yparams)
      flag = mdp.object_seen(env, **_terms(env, "object_yaw")[1])
      lost = want_lost
      # A filler is not noised: exactly zero, and the flag says 0; visible cubes are noised.
      assert bool((pos[lost] == 0).all()) and bool((yaw[lost] == 0).all())
      assert bool((flag[lost] == 0).all()) and bool((flag[~lost] == 1).all())
      if bool((~lost).any()):
        assert bool((pos[~lost].abs().sum(-1) > 0).all())
    assert bool(((sensor.tracker_lag >= 5) & (sensor.tracker_lag <= 8)).all())
    # The critic reads the live cube, the flag and the steps unseen.
    critic = env.cfg.observations["critic"].terms
    assert critic["object_pos"].params["lag"] is None
    unseen = mdp.object_unseen_steps(env, **critic["object_unseen"].params)
    assert bool((unseen > 0).all()) and bool((unseen <= 1).all())
    assert env.observation_manager.group_obs_dim["actor"] == (39,)
    assert env.observation_manager.group_obs_dim["critic"] == (41,)
  finally:
    env.close()


def test_o_k_rides_a_scripted_carry_within_5_mm(monkeypatch):
  env = _env(_occ("estimate"))
  try:
    grasp_at, lost_from, n = 12, 40, 70
    scene = Scene(
      env,
      monkeypatch,
      lambda t: t >= lost_from,
      lambda t: JAW_HELD if t >= grasp_at else JAW_OPEN,
      lambda t: (
        0.004 * max(0, t - 20)
      ),  # the hand carries the cube 4 mm a step, once closed
    )
    scene.reset()
    errors, held = [], []
    for _ in range(n):
      sensor = scene.step()
      t = scene.t - 1
      errors.append((sensor.est_pos - scene.live()).norm(dim=-1))
      held.append(sensor.held.clone())
      if t == grasp_at + 6:
        assert bool(sensor.held.all()), (
          "a stalled, commanded-closed jaw around a near cube holds"
        )
    lag = sensor.tracker_lag
    for t in range(lost_from + 8, n):
      # The cube has been out of view for 8-.. steps and moved a further 4 mm per step: the
      # estimate rides with the TCP instead of holding the last pose (which would be > 3 cm off).
      assert bool((errors[t] < 0.005).all()), (t, errors[t])
    last_seen_error = torch.tensor(
      [0.004 * (n - 1 - (lost_from + int(k) - 1)) for k in lag]
    )
    assert bool((last_seen_error > 0.03).all()), (
      "holding the last pose would have missed"
    )
    assert bool(held[-1].all())
    out, _ = sensor.output(env)
    base = env.scene["robot"]
    expect = _to_base_frame(base, sensor.est_pos)
    assert torch.allclose(out - sensor._bias_pos, expect, atol=3e-3)
  finally:
    env.close()


def test_o_k_holds_a_resting_cube_exactly(monkeypatch):
  env = _env(_occ("estimate"))
  try:
    scene = Scene(
      env,
      monkeypatch,
      lambda t: t >= 15,
      lambda t: JAW_OPEN,
      lambda t: 0.0,
      carry=False,
    )
    scene.reset()
    for _ in range(50):
      sensor = scene.step()
      assert not bool(sensor.held.any()), "an open jaw holds nothing"
    assert torch.allclose(sensor.est_pos, scene.live(), atol=5e-3)
  finally:
    env.close()


def test_o_k_lets_go_where_the_hand_did_and_drops_a_belief_the_cube_contradicts(
  monkeypatch,
):
  env = _env(_occ("estimate"))
  try:
    release_at = 45
    scene = Scene(
      env,
      monkeypatch,
      lambda t: 12 <= t < 60,
      lambda t: JAW_OPEN if (t < 12 or t >= release_at) else JAW_HELD,
      lambda t: 0.004 * max(0, min(t, release_at) - 20),
    )
    scene.reset()
    for _ in range(58):
      sensor = scene.step()
    assert not bool(sensor.held.any()), "the jaw opened below theta_open"
    # Not held: the cube is where it was let go, the TCP's place then, and it stayed there.
    assert torch.allclose(sensor.est_pos, scene.live(), atol=0.02)
  finally:
    env.close()

  env = _env(_occ("estimate"))
  try:
    scene = Scene(
      env,
      monkeypatch,
      lambda t: 12 <= t < 40,
      lambda t: JAW_HELD if t >= 8 else JAW_OPEN,
      lambda t: 0.004 * max(0, min(t, 30) - 20),  # the cube drops out of the hand at 30
      tcp_x=lambda t: 0.004 * max(0, t - 20),  # ...and the hand carries on without it
    )
    scene.reset()
    for _ in range(38):
      sensor = scene.step()
    assert bool(sensor.held.all()), "held while out of view"
    for _ in range(14):
      sensor = scene.step()
    # Visible again, centimetres from where the hand would carry it: the belief is overridden.
    assert not bool(sensor.held.any())
  finally:
    env.close()


def test_o_s_never_changes_and_never_shows_the_previous_episode(monkeypatch):
  cfg = _occ("snap")
  env = _env(cfg)
  try:
    scene = Scene(
      env, monkeypatch, lambda t: False, lambda t: JAW_OPEN, lambda t: 0.01 * t
    )
    scene.reset()
    outs = []
    for _ in range(12):
      scene.step()
      func, params = _terms(env, "object_pos")
      yfunc, yparams = _terms(env, "object_yaw")
      outs.append((func(env, **params).clone(), yfunc(env, **yparams).clone()))
    for pos, yaw in outs[1:]:
      assert torch.equal(pos, outs[0][0]) and torch.equal(yaw, outs[0][1])
    first = outs[0][0].clone()
    assert "object_seen" not in env.cfg.observations["actor"].terms
    assert env.observation_manager.group_obs_dim["actor"] == (38,)
    scene.reset()
    scene.step()
    func, params = _terms(env, "object_pos")
    new = func(env, **params)
    assert not torch.allclose(new, first, atol=1e-3), (
      "a new episode's cube, not the last one's"
    )
    scene.step()
    assert torch.equal(func(env, **params), new)
  finally:
    env.close()


def test_hold_reports_the_last_reported_pose_while_lost(monkeypatch):
  env = _env(_occ("hold", flag=False))
  try:
    scene = Scene(
      env,
      monkeypatch,
      lambda t: t >= 15,
      lambda t: JAW_OPEN,
      lambda t: 0.004 * t,
      carry=False,
    )
    scene.reset()
    for _ in range(40):
      sensor = scene.step()
    lost = ~sensor.seen
    assert bool(lost.all())
    assert torch.allclose(sensor.pos_w, sensor.last_pos)
    # Frozen at where the cube was when last seen: not the live pose it has moved on to.
    assert bool(((sensor.pos_w - scene.live()).norm(dim=-1) > 0.05).all())
  finally:
    env.close()
