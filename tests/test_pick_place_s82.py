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
from test_pick_place_s81 import BALL_C, S81, _ALL, _summary

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
  rest = {k: v for k, v in low.events.items() if k not in ("table_friction", "object_friction")}
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
