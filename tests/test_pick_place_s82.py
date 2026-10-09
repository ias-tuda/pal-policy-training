"""Plan S8.2 (`pick-place-v1-objects-plan.md`, *S8.2*): S8.1's ball on WC with ``FRICTION=low`` -- the table top's
and the object's sliding friction drawn per world from 0.3 (`env_cfg.FRICTION_CODES`), so the object-table contact
(MuJoCo: the larger of the two) spans 0.3-1.2 instead of S8.1's 0.8-1.2.

What is checked, on CPU: unset is S8.1's cfg exactly; ``low`` changes only the object's friction range and adds the
table's draw; it refuses a rung without DR-object; in an env the drawn coefficients land in their ranges, differ across
worlds, and some contacts fall below S8.1's floor of 0.8.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from pal_mjlab.tasks.manipulation.pick_place.kangaroo import env_cfg as kenv
from pal_mjlab.tasks.manipulation.pick_place.kangaroo.env_cfg import (
  kangaroo_pick_place_env_cfg,
)
from test_pick_place_s81 import _ALL, BALL_C, S81, _summary

LOW_TABLE, LOW_OBJECT = kenv.FRICTION_CODES["low"]


def _cfg(mp, num_envs: int = 4, **changes):
  for var in (*_ALL, "FRICTION"):
    mp.delenv(var, raising=False)
  for var, value in {**BALL_C, **S81, **changes}.items():
    if value is not None:
      mp.setenv(var, value)
  cfg = kangaroo_pick_place_env_cfg()
  cfg.scene.num_envs = num_envs
  cfg.seed = 0
  return cfg


def test_unset_and_default_are_s81(monkeypatch):
  base = _cfg(monkeypatch)
  for value in (None, "default"):
    cfg = _cfg(monkeypatch, FRICTION=value)
    assert _summary(cfg) == _summary(base)
    assert "table_friction" not in cfg.events
    assert cfg.events["object_friction"].params == base.events["object_friction"].params


def test_low_changes_only_the_two_friction_draws(monkeypatch):
  base, low = _cfg(monkeypatch), _cfg(monkeypatch, FRICTION="low")
  assert low.events["object_friction"].params["ranges"] == LOW_OBJECT
  table = low.events["table_friction"]
  assert table.params["ranges"] == LOW_TABLE
  assert table.params["asset_cfg"].name == "table"
  assert table.params["asset_cfg"].geom_names == ("table_top",)
  assert table.mode == "startup" and table.params["axes"] == [0]
  rest = {
    k: v
    for k, v in low.events.items()
    if k not in ("table_friction", "object_friction")
  }
  assert rest.keys() == {k for k in base.events if k != "object_friction"}
  for k, v in rest.items():
    assert dataclasses.asdict(v) == dataclasses.asdict(base.events[k]), k
  assert {**_summary(low), "events": None} == {**_summary(base), "events": None}


def test_low_needs_dr_object(monkeypatch):
  with pytest.raises(ValueError, match="DR-object"):
    _cfg(monkeypatch, FRICTION="low", DR="DR0")


def test_a_bad_code_raises(monkeypatch):
  with pytest.raises(ValueError):
    _cfg(monkeypatch, FRICTION="slippery")


def test_the_draws_land_in_range_and_reach_below_08(monkeypatch):
  cfg = _cfg(monkeypatch, num_envs=64, FRICTION="low")
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  try:
    env.reset()
    model = env.sim.model
    names = [env.sim.mj_model.geom(i).name for i in range(env.sim.mj_model.ngeom)]
    tg = names.index("table/table_top")
    og = names.index("object/object_geom")
    table = model.geom_friction[:, tg, 0]
    obj = model.geom_friction[:, og, 0]
    assert ((table >= LOW_TABLE[0] - 1e-6) & (table <= LOW_TABLE[1] + 1e-6)).all()
    assert ((obj >= LOW_OBJECT[0] - 1e-6) & (obj <= LOW_OBJECT[1] + 1e-6)).all()
    assert table.std() > 0.05 and obj.std() > 0.05
    contact = torch.maximum(table, obj)
    assert (contact < 0.8).float().mean() > 0.1, contact
  finally:
    env.close()


# -- RD ----------------------------------------------------------------------------------------

import collections  # noqa: E402

from pal_mjlab.tasks.manipulation import tabletop_env_cfg as tt  # noqa: E402
from pal_mjlab.tasks.manipulation.mdp.observations import (  # noqa: E402
  _RELEASE,
  _WAS_LIFTED,
)
from pal_mjlab.tasks.manipulation.pick_place import (
  pick_place_env_cfg as pp,  # noqa: E402
)
from pal_mjlab.tasks.manipulation.pick_place import rewards as rw  # noqa: E402
from pal_mjlab.tasks.manipulation.robots import (  # noqa: E402
  KANGAROO_GRIPPERS_LEFT_ARM_WC,
  kangaroo_grippers_left_arm,
)
from test_pick_place_s81 import RCTP, scripted  # noqa: E402

RD_ARM = f"{RCTP}+RD"


def test_rd_adds_only_its_term(monkeypatch):
  base = _summary(_cfg(monkeypatch))
  rd = _summary(_cfg(monkeypatch, REWARD=RD_ARM))
  extra = {k: v for k, v in rd["rewards"].items() if k not in base["rewards"]}
  assert extra == {"released_motion": ("released_object_motion", rw.RD_WEIGHT)}
  assert {**rd, "rewards": None} == {**base, "rewards": None}
  with pytest.raises(ValueError, match="RD needs J and M8"):
    rw._s52_arm(("M1", "M2", "M3", "M6", "J", "RD"))


def test_rd_charges_a_moving_free_ball_after_the_release_only(monkeypatch):
  cfg = _cfg(monkeypatch, num_envs=8, REWARD=RD_ARM)
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  try:
    env.reset()
    zero = torch.zeros(env.action_space.shape, device=env.device)
    for _ in range(3):
      env.step(zero)
    obj = env.scene["object"]
    term = env.reward_manager.get_term_cfg("released_motion")
    rd = lambda: term.func(env, **term.params)  # noqa: E731
    assert torch.all(rd() == 0.0)  # at rest, not released
    vel = torch.zeros_like(obj.data.root_link_vel_w)
    vel[:4, 0] = (
      0.15  # envs 0-3 rolling at 15 cm/s (the velocity shows after a physics step)
    )
    obj.write_root_link_velocity_to_sim(vel)
    env.step(zero)
    assert torch.all(rd() == 0.0)  # moving, but no release yet
    obj.write_root_link_velocity_to_sim(vel)
    getattr(env, _WAS_LIFTED)[:] = True
    latch = getattr(env, _RELEASE)
    latch.released[:] = True
    env.step(zero)
    assert bool(latch.released.all())
    v = rd()
    assert (v[:4] > 0.9).all(), v
    assert torch.all(v[4:] == 0.0), v
  finally:
    env.close()


CTP = pp.EFFORT_BOUNDS["CTP"]


def test_rd_stays_near_zero_along_a_clean_scripted_place_and_keeps_the_order():
  rec = collections.defaultdict(list)
  terms: dict = {}
  names = ("reach", "grasp", "lift", "goal", "goal_near", "release", "placed", "retreat", "pad_contact",
           "released_motion")  # fmt: skip

  def hook(phase, env):
    if not terms:
      terms.update(
        {
          n: c
          for n, c in zip(
            env.reward_manager.active_terms, env.reward_manager._term_cfgs, strict=True
          )
          if n in names
        }  # fmt: skip
      )
    rec[phase].append(
      {n: float(c.weight * c.func(env, **c.params)[0]) for n, c in terms.items()}
    )

  rows = scripted.rollout(
    device=get_test_device(),
    verbose=False,
    robot=kangaroo_grippers_left_arm(
      "G20", KANGAROO_GRIPPERS_LEFT_ARM_WC, arm_effort_scale=CTP.effort_scale
    ),
    baseline=pp.BASELINES["B"],
    spawn_box=pp.SPAWN_BOXES["WC"],
    target_area=pp.SPAWN_BOXES["WC"],
    dr="DR-scene",
    effort=CTP,
    latency=tt.LATENCY_MODELS["JS"],
    stack_speed=True,
    episode_length_s=70.0,
    num_envs=1,
    reward=RD_ARM,
    shape=pp.OBJECT_SHAPES["ball"],
    ball_offset=rw.EQ_OFFSET,
    spawns=[((0.45, 0.22), 0.0)],
    targets=[(0.46, 0.08)],
    hook=hook,
  )
  assert rows[0]["placed_and_lifted"], rows[0]
  placed = [t for p in ("open", "retreat") for t in rec[p] if t["placed"] > 0]
  assert placed
  # Once placed the ball is still: RD costs (almost) nothing there.
  assert max(-t["released_motion"] for t in placed) < 0.2
  total = lambda step: sum(step.values())  # noqa: E731
  order = [total(rec["descend"][-1]), total(rec["lift"][-1]), total(rec["steady"][-1]),
           total(rec["hold"][-1]), total(placed[-1])]  # fmt: skip
  assert all(a < b for a, b in zip(order, order[1:], strict=False)), order
