"""The cube's tracker latency, and what it must not also delay.

A vision tracker reports the cube 100-150 ms late -- 5-8 control steps at 50 Hz. The
actor's cube terms are jaw-relative, so the latency has to sit on the cube pose before
the vector is formed: mjlab's per-term ``delay_max_lag`` would buffer the finished vector
and stale the encoder side by the same 150 ms, which no robot does.
"""

import pal_mjlab.tasks  # noqa: F401
import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from pal_mjlab.tasks.manipulation import mdp
from pal_mjlab.tasks.manipulation.mdp.observations import _object_pose_w, _ObjectTracker
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import (
  OBJECT,
  OBJECT_MARKER_DELAY,
  OBJECT_TRACKER_LAG,
)

TASK = "Mjlab-Grasp-LeftArm-Standing-Gravcomp-ArmV36-Slow50-NoTable-Wide-Patient-LeanObs-Pal-Kangaroo-Grippers"

#: The same task with the tracker model off: the cube through the old marker delay.
CONTROL_TASK = TASK.replace("-Patient-LeanObs", "-Patient-NoTrackLag-LeanObs")

NUM_ENVS = 4


class _FakeEnv:
  """Enough of an env to drive the tracker with a pose that actually moves.

  The tracker reads one object's world pose and two counters; driving those directly is
  the only way to check a latency, since a cube resting on the table holds still and any
  lag fits a constant.
  """

  def __init__(self, num_envs: int = NUM_ENVS):
    self.num_envs = num_envs
    self.device = "cpu"
    self.common_step_counter = 0
    self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
    self._data = type(
      "Data",
      (),
      {
        "root_link_pos_w": torch.zeros(num_envs, 3),
        "root_link_quat_w": torch.zeros(num_envs, 4),
      },
    )()
    self._data.root_link_quat_w[:, 0] = 1.0
    self.scene = {OBJECT: type("Obj", (), {"data": self._data})()}

  def advance(self, pos_x: float) -> None:
    """One env step with the cube at a new, unique position."""
    self.common_step_counter += 1
    self.episode_length_buf += 1
    self._data.root_link_pos_w[:, 0] = pos_x

  def reset_envs(self, pos_x: float) -> None:
    self.episode_length_buf[:] = 0
    self._data.root_link_pos_w[:, 0] = pos_x


def test_actor_cube_terms_are_lagged_and_the_critic_is_not():
  cfg = load_env_cfg(TASK)
  actor, critic = cfg.observations["actor"].terms, cfg.observations["critic"].terms
  for name in ("jaw_to_object", "object_yaw_to_jaw", "object_to_goal"):
    assert actor[name].params["lag"] == OBJECT_TRACKER_LAG, name
    assert critic[name].params["lag"] is None, name
    # The latency lives on the pose, not on the finished vector: a term-level delay here
    # would stale the jaw side along with the cube.
    assert actor[name].delay_max_lag == 0, name


def test_notracklag_reproduces_the_runs_from_before_the_tracker_model():
  """The control has to be the *old* behaviour, not merely "no latency".

  Every run before the tracker model saw the cube through mjlab's per-term delay at the
  0-2 steps a marker tracker costs. A control with no latency at all would differ from
  those runs in two ways instead of one, and the comparison would say nothing.
  """
  control = load_env_cfg(CONTROL_TASK).observations
  for name in ("jaw_to_object", "object_yaw_to_jaw", "object_to_goal"):
    actor = control["actor"].terms[name]
    assert actor.params["lag"] is None, name
    assert (actor.delay_min_lag, actor.delay_max_lag) == OBJECT_MARKER_DELAY, name
    critic = control["critic"].terms[name]
    assert critic.params["lag"] is None, name
    assert (critic.delay_min_lag, critic.delay_max_lag) == (0, 0), name


def test_the_tracker_reports_a_pose_the_cube_held_that_many_steps_ago():
  env = _FakeEnv()
  lo, hi = OBJECT_TRACKER_LAG
  tracker = _ObjectTracker(env, OBJECT, OBJECT_TRACKER_LAG)
  history, reported = [], []
  for step in range(30):
    env.advance(float(step))  # a position that identifies the step it was taken at
    history.append(float(step))
    tracker.update(env)
    reported.append(tracker.pose[0][:, 0].clone())
  for env_id in range(env.num_envs):
    lags = {
      history.index(float(reported[t][env_id])) - t
      for t in range(hi, len(reported))
      if float(reported[t][env_id]) in history
    }
    assert len(lags) == 1, f"env {env_id}: latency is not constant within an episode"
    lag = -lags.pop()
    assert lo <= lag <= hi, f"env {env_id}: lag {lag} outside {OBJECT_TRACKER_LAG}"


def test_the_lag_is_drawn_per_env():
  """A single shared draw would let a policy learn one latency instead of tolerating any."""
  env = _FakeEnv(num_envs=64)
  tracker = _ObjectTracker(env, OBJECT, OBJECT_TRACKER_LAG)
  tracker.update(env)
  assert len(set(tracker._lag.tolist())) > 1


def test_a_reset_does_not_report_the_previous_episodes_cube():
  env = _FakeEnv()
  tracker = _ObjectTracker(env, OBJECT, OBJECT_TRACKER_LAG)
  for step in range(20):
    env.advance(float(step))
    tracker.update(env)
  env.reset_envs(99.0)
  tracker.update(env)
  # No history to be late with: the cube is reported where the reset put it.
  assert torch.allclose(tracker.pose[0][:, 0], torch.full((env.num_envs,), 99.0))


def test_the_pose_advances_once_per_env_step_however_many_terms_ask():
  env = _FakeEnv()
  tracker = _ObjectTracker(env, OBJECT, OBJECT_TRACKER_LAG)
  env.advance(1.0)
  tracker.update(env)
  before = tracker.pose[0].clone()
  for _ in range(3):  # the three jaw-relative terms, same step
    tracker.update(env)
  assert torch.equal(tracker.pose[0], before)


def test_history_is_deep_enough_for_its_own_lag():
  env = _FakeEnv()
  tracker = _ObjectTracker(env, OBJECT, OBJECT_TRACKER_LAG)
  assert tracker._pos.shape[0] == OBJECT_TRACKER_LAG[1] + 1


@pytest.fixture(scope="module")
def env():
  cfg = load_env_cfg(TASK)
  cfg.scene.num_envs = NUM_ENVS
  e = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    yield e
  finally:
    e.close()


def test_the_jaw_side_of_jaw_to_object_is_read_live(env):
  """With the cube at rest, the lagged vector must equal the live one at every step.

  The arm is moving throughout. A delay on the finished term would show the jaw where it
  was 5-8 steps ago and the two would diverge; a delay on the cube pose alone cannot.
  """
  # The manager resolves its own deep copy of each term cfg; the one on env.cfg still
  # has unresolved site ids.
  grasp_cfg = env.observation_manager.get_term_cfg("actor", "jaw_to_object").params[
    "grasp_cfg"
  ]
  action = torch.full((env.num_envs, env.action_manager.total_action_dim), 0.6)
  env.reset()
  moved = False
  previous = None
  for _ in range(20):
    env.step(action)
    lagged = mdp.jaw_to_object_b(
      env, OBJECT, grasp_cfg=grasp_cfg, lag=OBJECT_TRACKER_LAG
    )
    live = mdp.jaw_to_object_b(env, OBJECT, grasp_cfg=grasp_cfg, lag=None)
    assert torch.allclose(lagged, live, atol=2e-3)
    if previous is not None and (lagged - previous).abs().max() > 1e-3:
      moved = True
    previous = lagged
  assert moved, "the arm never moved; the test could not have caught a delayed jaw"


def test_one_object_is_one_tracker(env):
  """All three cube terms must agree on where the cube was, not draw a lag each."""
  first = _object_pose_w(env, OBJECT, OBJECT_TRACKER_LAG)
  again = _object_pose_w(env, OBJECT, OBJECT_TRACKER_LAG)
  assert torch.equal(first[0], again[0]) and torch.equal(first[1], again[1])
  with pytest.raises(ValueError, match="tracked two ways"):
    _object_pose_w(env, OBJECT, (1, 2))


def test_live_pose_bypasses_the_tracker(env):
  obj = env.scene[OBJECT]
  pos_w, _ = _object_pose_w(env, OBJECT, None)
  assert torch.equal(pos_w, obj.data.root_link_pos_w)


#: The dropout variant of ``TASK``.
DROP_TASK = TASK.replace("-Patient-LeanObs", "-Patient-Drop-LeanObs")


def test_drop_puts_the_dropout_on_every_actor_cube_term_and_not_the_critic():
  from pal_mjlab.tasks.manipulation.grasp.kangaroo import TRACKER_DROPOUT

  cfg = load_env_cfg(DROP_TASK)
  actor, critic = cfg.observations["actor"].terms, cfg.observations["critic"].terms
  for name in ("jaw_to_object", "object_yaw_to_jaw", "object_to_goal"):
    assert actor[name].params["dropout"] == TRACKER_DROPOUT, name
    assert critic[name].params.get("dropout") is None, name


def test_dropout_holds_the_last_reported_pose_then_resumes():
  torch.manual_seed(0)
  env = _FakeEnv(num_envs=256)
  lag = (0, 0)
  # Lost on the first chance, for exactly 3 steps.
  tracker = _ObjectTracker(env, OBJECT, lag, dropout=(1.0, 3, 3))
  tracker.update(env)
  reported = []
  for x in range(1, 9):
    env.advance(float(x))
    tracker.update(env)
    reported.append(tracker.pose[0][0, 0].item())
  # Step 1 starts a dropout holding the pose reported at step 1 through steps 1-3; step 4
  # reports live, and immediately starts the next one.
  assert reported == [1.0, 1.0, 1.0, 4.0, 4.0, 4.0, 7.0, 7.0]


def test_dropout_rate_matches_its_config():
  torch.manual_seed(0)
  env = _FakeEnv(num_envs=4096)
  tracker = _ObjectTracker(env, OBJECT, (0, 0), dropout=(0.01, 5, 50))
  tracker.update(env)
  lost = 0.0
  steps = 2000
  for x in range(1, steps + 1):
    env.advance(float(x))
    tracker.update(env)
    lost += (tracker.pose[0][:, 0] != float(x)).float().mean().item()
  # Renewal process: mean dropout 27.5 steps, then a geometric wait of mean ~100.
  expected = 27.5 / (27.5 + 1 / 0.01)
  assert lost / steps == pytest.approx(expected, rel=0.1)


def test_a_reset_ends_a_dropout():
  env = _FakeEnv()
  tracker = _ObjectTracker(env, OBJECT, (0, 0), dropout=(1.0, 50, 50))
  tracker.update(env)
  env.advance(1.0)
  tracker.update(env)
  env.advance(2.0)
  tracker.update(env)
  assert tracker.pose[0][0, 0].item() == 1.0
  env.reset_envs(9.0)
  tracker.update(env)
  # The next episode's cube, not the last one's held pose.
  assert tracker.pose[0][0, 0].item() == 9.0


def test_blind_after_grasp_reports_the_pose_from_the_grasp_on_until_a_reset(
  monkeypatch,
):
  """The hand around the mug hides it from the camera: from the step the jaw holds it the
  tracker reports where it last saw it, held or let go, until the next episode."""
  from pal_mjlab.tasks.manipulation.mdp import observations as obs

  env = _FakeEnv()
  held = torch.zeros(env.num_envs, dtype=torch.bool)
  monkeypatch.setattr(obs, "_grasped_any_bool", lambda env, names: held.clone())
  tracker = _ObjectTracker(env, OBJECT, (0, 0), blind_after=(("a", "b"),))
  for step in range(5):
    env.advance(float(step))
    tracker.update(env)
  assert torch.equal(tracker.pose[0][:, 0], torch.full((env.num_envs,), 4.0))
  held[:2] = True
  env.advance(5.0)
  tracker.update(env)
  held[:] = False  # let go: the tracker does not find it again
  for step in range(6, 10):
    env.advance(float(step))
    tracker.update(env)
  assert tracker.pose[0][:, 0].tolist() == [5.0, 5.0, 9.0, 9.0]
  env.reset_envs(99.0)
  tracker.update(env)
  assert torch.equal(tracker.pose[0][:, 0], torch.full((env.num_envs,), 99.0))


def test_a_blind_and_a_seeing_term_on_one_object_are_two_trackers(env):
  _object_pose_w(env, OBJECT, OBJECT_TRACKER_LAG)
  with pytest.raises(ValueError, match="tracked two ways"):
    _object_pose_w(env, OBJECT, OBJECT_TRACKER_LAG, blind_after=(("a", "b"),))
