"""The mug-on-coaster task: the mug starts upright beside the coaster, the goal on it."""

import pal_mjlab.tasks  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from pal_mjlab.tasks.manipulation import mdp, scene
from pal_mjlab.tasks.manipulation.coaster.coaster_env_cfg import (
  COASTER,
  COASTER_MIN_SEPARATION,
)

TASK = "Mjlab-MugCoaster-Standing-Pal-Kangaroo-Grippers"


def test_coaster_task_is_registered_with_the_place_ladder():
  cfg = load_env_cfg(TASK)
  assert load_rl_cfg(TASK).experiment_name == "kangaroo_mug_coaster"
  assert {"lift", "set_down", "release", "stacked", "clear"} <= set(cfg.rewards)
  assert cfg.rewards["release"].params["object_width"] == scene.MUG_WIDTH
  assert {"mug_dropped", "object_flipped", "out_of_view", "coaster_out_of_view"} <= set(
    cfg.terminations
  )
  assert "object_rot" in cfg.observations["actor"].terms
  assert "object_to_goal" in cfg.observations["actor"].terms


def test_mug_starts_upright_beside_the_coaster_with_the_goal_on_it():
  cfg = load_env_cfg(TASK)
  cfg.scene.num_envs = 16
  env = ManagerBasedRlEnv(
    cfg=cfg, device="cuda:0" if torch.cuda.is_available() else "cpu"
  )
  try:
    env.reset()
    mug = env.scene["object"]
    coaster = env.scene[COASTER]
    goal = env.command_manager.get_term("goal_pose").command
    mug_pos = mug.data.root_link_pos_w
    coaster_pos = coaster.data.root_link_pos_w

    apart = torch.norm(mug_pos[:, :2] - coaster_pos[:, :2], dim=-1)
    assert (apart >= COASTER_MIN_SEPARATION - 1e-4).all()
    assert torch.allclose(goal[:, :2], coaster_pos[:, :2], atol=1e-5)
    assert torch.allclose(
      goal[:, 2] - coaster_pos[:, 2],
      torch.full_like(goal[:, 2], scene.COASTER_HALF_HEIGHT + scene.MUG_HALF_HEIGHT),
      atol=1e-5,
    )
    # Upright: the mug's up axis along world z.
    assert (-mug.data.projected_gravity_b[:, 2] > 0.999).all()

    # Left alone, nothing falls, tips or leaves the view in the first second.
    action = torch.zeros(
      env.num_envs, env.action_manager.total_action_dim, device=env.device
    )
    for _ in range(50):
      _, _, terminated, _, _ = env.step(action)
      assert not terminated.any()
  finally:
    env.close()


def test_mug_set_on_the_coaster_stays_standing_on_it():
  # Set on the coaster, the mug has to stand on it, touching it, where the goal says.
  cfg = load_env_cfg(TASK)
  cfg.scene.num_envs = 16
  env = ManagerBasedRlEnv(
    cfg=cfg, device="cuda:0" if torch.cuda.is_available() else "cpu"
  )
  try:
    env.reset()
    mug = env.scene["object"]
    goal = env.command_manager.get_term("goal_pose").command.clone()
    pose = mug.data.root_link_pose_w.clone()
    pose[:, :3] = goal
    pose[:, 2] += 0.001
    mug.write_root_link_pose_to_sim(pose)
    mug.write_root_link_velocity_to_sim(torch.zeros(env.num_envs, 6, device=env.device))
    env.sim.forward()

    action = torch.zeros(
      env.num_envs, env.action_manager.total_action_dim, device=env.device
    )
    for _ in range(100):
      _, _, terminated, _, _ = env.step(action)
      assert not terminated.any()
    goal = env.command_manager.get_term("goal_pose").command
    error = goal - mug.data.root_link_pos_w
    assert (torch.norm(error[:, :2], dim=-1) < 0.005).all()
    assert (error[:, 2].abs() < 0.003).all()
    assert (-mug.data.projected_gravity_b[:, 2] > 0.995).all()
    assert env.scene["object_base_contact"].data.found.bool().any(dim=-1).all()
  finally:
    env.close()


def test_release_variant_swaps_the_top_rungs():
  from pal_mjlab.tasks.manipulation.coaster import coaster_env_cfg as c
  from pal_mjlab.tasks.manipulation.mug.mug_env_cfg import HAND_OBJECT_SENSOR

  cfg = load_env_cfg("Mjlab-MugCoaster-Standing-Release-Pal-Kangaroo-Grippers")
  r = cfg.rewards
  assert "release" not in r
  assert cfg.terminations["object_flipped"].func is mdp.object_tipped_free
  assert r["releasing"].weight == c.COASTER_RELEASING_REWARD
  for name in ("stacked", "clear", "releasing"):
    assert r[name].params["hand_sensor_name"] == HAND_OBJECT_SENSOR
  for name in ("lift", "set_down"):
    assert r[name].params["placed_term"] == "stacked"
    assert r[name].params["hold_time"] == c.COASTER_HOLD_TIME
  assert r["lift"].params["budget"] == c.COASTER_LIFT_BUDGET
  assert r["lift"].params["after_budget"] == c.COASTER_LIFT_AFTER_BUDGET
  assert "budget" not in r["set_down"].params
  assert r["set_down"].params["xy_std"] == c.COASTER_SET_DOWN_XY_STD
  assert HAND_OBJECT_SENSOR in [s.name for s in cfg.scene.sensors]
  assert "release" in load_env_cfg(TASK).rewards  # the plain task as it was
  assert r["knocked"].weight == c.COASTER_KNOCK_PENALTY
  assert set(r["knocked"].params["term_names"]) == {"object_flipped", "out_of_view"}
  assert set(r["knocked"].params["term_names"]) <= set(cfg.terminations)
  assert "knocked" not in load_env_cfg(TASK).rewards
  assert r["carry"].weight == c.COASTER_CARRY_REWARD
  assert r["centre"].weight == c.COASTER_CENTRE_REWARD
  assert r["centre"].params["std"] == c.COASTER_CENTRE_STD
  assert r["centre"].params["gamma"] == r["carry"].params["gamma"]
  assert (
    r["carry"].params["gamma"]
    == load_rl_cfg(
      "Mjlab-MugCoaster-Standing-Release-Pal-Kangaroo-Grippers"
    ).algorithm.gamma
  )


def test_carry_pays_the_way_to_the_coaster_and_a_drop_pays_it_back(monkeypatch):
  import math
  from types import SimpleNamespace

  from pal_mjlab.tasks.manipulation.mdp import coaster_rewards as cr

  state = {"held": True, "touch": False, "d": 0.3}
  monkeypatch.setattr(
    cr, "_grasped_any_bool", lambda env, n: torch.tensor([state["held"]])
  )
  monkeypatch.setattr(
    cr, "_on_base", lambda *a: (torch.tensor([state["touch"]]), None, None)
  )
  monkeypatch.setattr(cr, "mug_tilt_cos", lambda obj: torch.tensor([1.0]))
  obj = SimpleNamespace(data=SimpleNamespace(root_link_pos_w=torch.zeros(1, 3)))
  command = SimpleNamespace(command=torch.zeros(1, 3))
  env = SimpleNamespace(
    num_envs=1, device="cpu", step_dt=0.5, scene={"mug": obj},
    command_manager=SimpleNamespace(get_term=lambda n: command),
  )  # fmt: skip
  term = cr.coaster_carry_progress(None, env)

  def step(**kw):
    state.update(kw)
    command.command[0, 0] = state["d"]
    return term(env, "goal", "mug", (), "base", gamma=1.0, std=0.1).item() * 0.5

  def phi(d):
    return math.exp(-d / 0.1)

  assert step() == 0.0  # the first step sets phi
  assert math.isclose(step(d=0.1), phi(0.1) - phi(0.3), rel_tol=1e-5)  # carried closer
  assert math.isclose(step(), 0.0, abs_tol=1e-7)  # held still: nothing at gamma 1
  assert math.isclose(step(held=False), -phi(0.1), rel_tol=1e-5)  # dropped: paid back
  step(held=True, d=0.0)
  assert math.isclose(step(held=False, touch=True), 0.0, abs_tol=1e-7)  # let go on it


def test_knocking_the_mug_costs_the_penalty_once():
  from types import SimpleNamespace

  fired = {
    "object_flipped": torch.tensor([True, False, False]),
    "out_of_view": torch.tensor([False, True, False]),
  }
  env = SimpleNamespace(
    num_envs=3, device="cpu", step_dt=0.02,
    termination_manager=SimpleNamespace(get_term=fired.__getitem__),
  )  # fmt: skip
  paid = mdp.termination_fired(env, ("object_flipped", "out_of_view")) * env.step_dt
  # Times dt, as the reward manager scales it: the weight once, per event.
  assert torch.allclose(paid, torch.tensor([1.0, 1.0, 0.0]))


class _Latched:
  """Drive ``coaster_rewards`` with the contact, the goal error and the jaw set by hand."""

  def __init__(self, monkeypatch, cls):
    from pal_mjlab.tasks.manipulation.mdp import coaster_rewards as cr

    self.touch, self.held, self.hand = (torch.tensor([False]) for _ in range(3))
    self.xy, self.z, self.release = (torch.tensor([0.0]) for _ in range(3))
    monkeypatch.setattr(cr, "_on_base", lambda *a: (self.touch, self.xy, self.z))
    monkeypatch.setattr(cr, "_grasped_any_bool", lambda env, n: self.held)
    monkeypatch.setattr(cr, "_hand_touching", lambda env, n: self.hand)
    monkeypatch.setattr(cr, "_release_progress", lambda *a, **k: self.release)

    class _Env:
      num_envs = 1
      device = "cpu"
      step_dt = 0.5
      scene = {"mug": None}

    self.env = _Env()
    self.term = cls(None, self.env)

  def step(self, touch=False, held=False, hand=None, xy=0.0, z=0.0, release=0.0):
    self.touch[0], self.held[0], self.xy[0], self.z[0] = touch, held, xy, z
    self.hand[0] = held if hand is None else hand
    self.release[0] = release
    return self.term(self.env, "goal", "mug", (), "base", None, "hand").item()


def test_placed_survives_the_contact_flicker_and_needs_the_jaw_let_go(monkeypatch):
  from pal_mjlab.tasks.manipulation.mdp import coaster_rewards as cr

  m = _Latched(monkeypatch, cr.mug_placed_on_coaster)
  assert m.step(touch=True, release=1.0) == 0.0  # pushed on: never set down in the jaw
  m.step(touch=True, held=True)  # set down in the jaw
  # The contact drops out, the mug still at the goal: the set-down holds.
  assert m.step(touch=False, z=0.002, release=1.0) == 1.0
  # The jaw shut round it, out of contact: not let go.
  assert m.step(touch=True, hand=False, release=0.0) == 0.0
  assert m.step(touch=True, hand=True, release=1.0) == 0.0  # a finger on it
  m.step(held=True, z=0.03)  # picked up again: the set-down clears
  assert m.step(touch=True, release=1.0) == 0.0


def test_holding_at_the_goal_stops_paying_and_a_relift_does_not_refill_it(monkeypatch):
  from pal_mjlab.tasks.manipulation.mdp import coaster_rewards as cr

  m = _Latched(monkeypatch, cr.mug_placed_on_coaster)

  class _Rewards:
    def get_term_cfg(self, name):
      return type("Cfg", (), {"func": m.term})

  m.env.reward_manager = _Rewards()
  gate = cr.paid_until_placed(None, m.env)

  def pay():
    return gate(m.env, lambda env: torch.ones(1), "stacked", hold_time=1.0).item()

  assert pay() == 1.0
  m.step(touch=True, held=True)  # placed: step 1 of the two hold_time allows
  assert pay() == 1.0
  m.step(touch=True, held=True)
  assert pay() == 0.0
  m.step(held=True, z=0.05)  # lifted off again
  m.step(touch=True, held=True)
  assert pay() == 0.0
  m.term.reset()
  assert pay() == 1.0


def test_lift_drops_once_it_has_paid_its_budget_wherever_the_mug_is(monkeypatch):
  from pal_mjlab.tasks.manipulation.mdp import coaster_rewards as cr

  m = _Latched(monkeypatch, cr.mug_placed_on_coaster)

  class _Rewards:
    def get_term_cfg(self, name):
      return type("Cfg", (), {"func": m.term})

  m.env.reward_manager = _Rewards()
  gate = cr.paid_until_placed(None, m.env)

  def pay(value, after_budget=0.0):
    lift = lambda env: torch.tensor([value])  # noqa: E731
    return gate(
      m.env, lift, "stacked", hold_time=1.0, budget=1.0, after_budget=after_budget
    ).item()

  # Held off the goal, never set down: 0.25 s of the budget a step (0.5 for 0.5 s).
  for _ in range(4):
    assert pay(0.5) == 0.5
  assert pay(0.5) == 0.0  # 1 s paid of 1
  gate.reset()
  assert pay(0.5) == 0.5

  # With a share after it, the spent budget still pulls toward the goal, at that share.
  gate.reset()
  for _ in range(4):
    assert pay(0.5, after_budget=0.25) == 0.5
  assert pay(0.5, after_budget=0.25) == 0.125
  assert pay(1.0, after_budget=0.25) == 0.25


def test_replay_starts_episodes_set_down_where_the_policy_set_the_mug_down():
  from pal_mjlab.tasks.manipulation.coaster import coaster_env_cfg as c
  from pal_mjlab.tasks.manipulation.mdp import coaster_rewards as cr

  cfg = load_env_cfg("Mjlab-MugCoaster-Standing-Release-Replay-Pal-Kangaroo-Grippers")
  cfg.scene.num_envs = 16
  cfg.events["replay_placed"].params.update(probability=1.0, min_states=1)
  env = ManagerBasedRlEnv(
    cfg=cfg, device="cuda:0" if torch.cuda.is_available() else "cpu"
  )
  try:
    env.reset()
    mug = env.scene["object"]
    goal = env.command_manager.get_term("goal_pose").command.clone()
    pose = mug.data.root_link_pose_w.clone()
    pose[:, :3] = goal
    pose[:, 2] += 0.001
    mug.write_root_link_pose_to_sim(pose)
    mug.write_root_link_velocity_to_sim(torch.zeros(env.num_envs, 6, device=env.device))
    env.sim.forward()
    action = torch.zeros(
      env.num_envs, env.action_manager.total_action_dim, device=env.device
    )
    for _ in range(20):
      env.step(action)
    # The policy setting it down, by hand: the mug standing on the coaster, as taken.
    placed = env.reward_manager.get_term_cfg("stacked").func
    table = SceneEntityCfg(c.TABLE)
    placed.replay.capture(env, torch.ones(env.num_envs, dtype=torch.bool), table)
    coaster = env.scene[COASTER]
    taken = coaster.data.root_link_pos_w[:, :2].clone()
    on_coaster = mug.data.root_link_pos_w - coaster.data.root_link_pos_w

    env.reset()
    # Each env starts from one of the states taken, not a fresh spawn: the coaster where
    # one of them had it, and the mug standing on it as it stood.
    assert (
      torch.cdist(coaster.data.root_link_pos_w[:, :2], taken).min(-1).values < 1e-5
    ).all()
    now = mug.data.root_link_pos_w - coaster.data.root_link_pos_w
    assert (now[:, :2].norm(dim=-1) < 1e-3).all()
    assert (now[:, 2] - on_coaster[0, 2]).abs().max() < 1e-3
    for name in ("stacked", "clear", "releasing"):
      latch = env.reward_manager.get_term_cfg(name).func.latch
      assert isinstance(latch, cr._PlaceLatch)
      assert latch.set_down.all() and not latch.restored.any()
    for _ in range(50):
      _, _, terminated, _, _ = env.step(action)
      assert not terminated.any()
    goal = env.command_manager.get_term("goal_pose").command
    assert (
      torch.norm(goal[:, :2] - mug.data.root_link_pos_w[:, :2], dim=-1) < 0.005
    ).all()
  finally:
    env.close()


def test_success_metrics_read_the_end_state_term():
  for task, term in (
    ("Mjlab-MugCoaster-Standing-Release-Pal-Kangaroo-Grippers", "stacked"),
    ("Mjlab-MugUpright-Standing-Lower-Release-Pal-Kangaroo-Grippers", "stood_released"),
    ("Mjlab-MugTurn-Standing-Release-Pal-Kangaroo-Grippers", "turned"),
  ):
    m = load_env_cfg(task).metrics
    assert m["episode_success"].params["term_name"] == term
    assert m["episode_success"].reduce == "max"
    assert m["final_success"].reduce == "last"


def test_cube_stack_gets_the_release_ladder_with_the_cube_width():
  import math

  from mjlab.tasks.registry import load_rl_cfg
  from pal_mjlab.tasks.manipulation import scene
  from pal_mjlab.tasks.manipulation.stack.stack_env_cfg import BASE_OBJECT

  task = "Mjlab-Stack-Standing-Place-Release-Replay-Pal-Kangaroo-Grippers"
  cfg = load_env_cfg(task)
  place = load_env_cfg("Mjlab-Stack-Standing-Place-Pal-Kangaroo-Grippers")
  r = cfg.rewards
  assert "release" not in r and "knocked" not in r
  for name in ("stacked", "clear", "releasing"):
    assert r[name].params["object_width"] == scene.CUBE_WIDTH
  assert r["lift"].params["placed_term"] == "stacked"
  assert r["carry"].params["stand_tilt"] == math.pi
  assert cfg.events["replay_placed"].params["object_names"][1] == BASE_OBJECT
  assert cfg.metrics["episode_success"].params["term_name"] == "stacked"
  # Same actor, so the first-generation stack runs load; same experiment to find them.
  assert list(cfg.observations["actor"].terms) == list(
    place.observations["actor"].terms
  )
  assert load_rl_cfg(task).experiment_name == "kangaroo_stack_standing"
  assert r["carry"].params["gamma"] == load_rl_cfg(task).algorithm.gamma
  # The mug's own ladder keeps the mug's width.
  mug = load_env_cfg("Mjlab-MugCoaster-Standing-Release-Pal-Kangaroo-Grippers").rewards
  assert mug["stacked"].params["object_width"] == scene.MUG_WIDTH
  assert "stand_tilt" not in mug["carry"].params
