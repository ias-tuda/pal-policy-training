"""Plan §12 (`pick-place-insert-plan.md`): the cross pin and the socket block, ``GOAL=insert`` -- step 0 item 1.

The env is `BS12` (plan §0.3: `BS2` with ``CMD_OBS=act``, the wide area, the raised table) with ``SHAPE=pin`` and an
insert ``GOAL``, built on CPU. What is checked: the block's boxes are the CAD's; mujoco_warp's box-box contacts of a
pin plate resting on the table are CPU MuJoCo's at the recorded poses that popped the pin (decision 0005, the patch);
the cfg's widths, codes and refusals; the reset puts the pin upright on WC and the block on WC clear of it, and
neither moves when nothing touches them; the actor's pin and block observations are the truth with the noise off,
the same for psi and psi + 90 deg and for a pin turned end for end; ``seated`` holds for a pin stood in the hole
(positive control) and not for one wedged 10 mm in or standing beside the block; `placed` needs the lift; pushing
the block costs; and the per-step reward rises along the insertion (L5) for IN and LAD-IN.
"""

from __future__ import annotations

import json
import math
import pathlib

import mujoco
import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul
from pal_mjlab.tasks.manipulation import mdp, scene
from pal_mjlab.tasks.manipulation.mdp import insert as ins_mdp
from pal_mjlab.tasks.manipulation.mdp.events import surface_top_z
from pal_mjlab.tasks.manipulation.mdp.observations import _placed_bool
from pal_mjlab.tasks.manipulation.pick_place import insert as ins
from pal_mjlab.tasks.manipulation.pick_place.kangaroo.env_cfg import (
  kangaroo_pick_place_env_cfg,
  kangaroo_pick_place_rl_cfg,
)
from pal_mjlab.tasks.manipulation.pick_place.pick_place_env_cfg import SPAWN_BOXES

DATA = pathlib.Path(__file__).parent / "data"
RCTP = "R-C-strict+M1+M2+M3+M6+J+M8"
#: `BS12` + IN (plan §12.1's batch, every code but its GOAL).
IN_ENV = {
  "REWARD": RCTP,
  "DR": "DR-scene",
  "PPO": "PPO-stack",
  "SLEW": "stack",
  "BASELINE": "B",
  "SPAWN": "WC",
  "TOOL_SPEED": "none",
  "GAINS": "G20",
  "EFFORT": "CTP",
  "LAG": "FP+JS",
  "ACTION": "abs",
  "CMD_OBS": "act",
  "TARGET": "WC",
  "EPISODE_S": "15",
  "CRITIC_OBS": "q",
  "ARM_TABLE": "AH10",
  "SHAPE": "pin",
  "GOAL": "insert",
  "TABLE_DZ": "22",
}
WC = SPAWN_BOXES["WC"]
GEO = {"block_height": scene.BLOCK_HEIGHT, "hole_d": scene.BLOCK_HOLE_D}
HALF = scene.PIN_LENGTH / 2


def _set_env(monkeypatch, **changes) -> None:
  for var, value in {**IN_ENV, **changes}.items():
    if value is None:
      monkeypatch.delenv(var, raising=False)
    else:
      monkeypatch.setenv(var, value)
  for var in ("OCC", "OBJ_OBS", "GATE", "BOX_FEET", "BOX_OBS", "CKPT_SPAWN", "TRACKER"):
    if var not in changes:
      monkeypatch.delenv(var, raising=False)


def _cfg(monkeypatch, num_envs: int = 4, **changes):
  _set_env(monkeypatch, **changes)
  cfg = kangaroo_pick_place_env_cfg()
  cfg.scene.num_envs = num_envs
  cfg.seed = 0
  return cfg


@pytest.fixture(scope="module")
def env():
  """IN on CPU, 8 envs, the actor's noise off."""
  mp = pytest.MonkeyPatch()
  cfg = _cfg(mp, num_envs=8)
  cfg.observations["actor"].enable_corruption = False
  e = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  e.reset()
  yield e
  e.close()
  mp.undo()


def _hold(env, steps: int) -> None:
  with torch.inference_mode():
    for _ in range(steps):
      env.step(torch.zeros(env.action_space.shape, device=env.device))


def _place(entity, pos: torch.Tensor, quat: torch.Tensor) -> None:
  entity.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1))
  entity.write_root_link_velocity_to_sim(torch.zeros_like(entity.data.root_link_vel_w))


def _yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
  z = torch.zeros_like(yaw)
  return quat_from_euler_xyz(z, z, yaw)


# -- The parts ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
  "hole_d, chamfer, ref",
  [
    (scene.BLOCK_HOLE_D, scene.BLOCK_CHAMFER, "s12_block_boxes_d38.json"),
    (scene.BLOCK_HOLE_D_TIGHT, scene.BLOCK_CHAMFER_TIGHT, "s12_block_boxes_d35.json"),
  ],
)
def test_the_block_is_the_cads_boxes(hole_d, chamfer, ref):
  """`scene.socket_block_boxes` is `cad/s12_insert/insert_geoms.block_boxes` (mm there) box for box."""
  want = json.loads((DATA / ref).read_text())
  got = scene.socket_block_boxes(hole_d=hole_d, chamfer=chamfer)
  assert [w[0] for w in want] == [g[0] for g in got]
  for (_, c, h, q), (_, gc, gh, gq) in zip(want, got, strict=True):
    assert gc == pytest.approx([x / 1e3 for x in c], abs=1e-9)
    assert gh == pytest.approx([x / 1e3 for x in h], abs=1e-9)
    assert gq == pytest.approx(q, abs=1e-9)


def test_the_parts_masses_and_collision_bits():
  block = scene.get_socket_block_spec().compile()
  pin = scene.get_pin_spec().compile()
  assert block.body_mass[1] == pytest.approx(scene.BLOCK_MASS)
  assert pin.body_mass[1] == pytest.approx(scene.PIN_MASS)
  ring = set(scene.socket_block_ring_geom_names("block"))
  for g in range(block.ngeom):
    name = mujoco.mj_id2name(block, mujoco.mjtObj.mjOBJ_GEOM, g)
    bits = (int(block.geom_contype[g]), int(block.geom_conaffinity[g]))
    assert bits == ((1, 1) if name in ring else (2, 1)), name
  table = scene.get_table_spec(0.837, scene.INSERT_TABLE_CONTYPE).compile()
  # The rule: (contype_a & conaffinity_b) | (contype_b & conaffinity_a).
  t, inner, one = (scene.INSERT_TABLE_CONTYPE, 1), (2, 1), (1, 1)
  hit = lambda a, b: bool((a[0] & b[1]) | (b[0] & a[1]))  # noqa: E731
  assert set(table.geom_contype) == {scene.INSERT_TABLE_CONTYPE}
  assert not hit(t, inner) and hit(t, one) and hit(inner, one)


# -- Decision 0005: box-box contacts of a pin plate resting on the table ---------------------------------


def _resting_model(pose: dict, plate: str) -> mujoco.MjModel:
  def quat(mat):
    q = [0.0] * 4
    import numpy as np

    qq = np.zeros(4)
    mujoco.mju_mat2Quat(qq, np.asarray(mat, float))
    q[:] = qq.tolist()
    return " ".join(map(str, q))

  t, p = pose["table/table_top"], pose[plate]
  xml = f"""<mujoco><option><flag nativeccd="disable"/></option><worldbody>
  <geom type="box" size="{" ".join(map(str, t["size"]))}" pos="{" ".join(map(str, t["pos"]))}" quat="{quat(t["mat"])}"/>
  <body pos="{" ".join(map(str, p["pos"]))}" quat="{quat(p["mat"])}"><freejoint/>
  <geom type="box" size="{" ".join(map(str, p["size"]))}"/></body></worldbody></mujoco>"""
  return mujoco.MjModel.from_xml_string(xml)


@pytest.mark.parametrize("world", ["0", "4", "5"])
@pytest.mark.parametrize("plate", ["object/object_geom", "object/object_geom_y"])
def test_a_resting_plate_gets_cpu_mujocos_box_box_contacts(world, plate):
  """At the reset poses that popped the pin (worlds 4 and 5 of the first IN smoke test, 2026-10-07; world 0 is a
  pose that did not), mujoco_warp under the task's contact options returns CPU MuJoCo's contacts: four, at zero
  depth. Unpatched mujoco_warp 3.11.0 returns a fifth at -512 / -361 / -471 mm (decision 0005)."""
  import mujoco_warp as mjw
  import warp as wp

  pose = json.loads((DATA / "s12_boxbox_poses.json").read_text())[world]
  m = _resting_model(pose, plate)
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  cpu = sorted(round(float(c.dist), 6) for c in d.contact[: d.ncon])
  with wp.ScopedDevice("cpu"):
    mm = mjw.put_model(m)
    dd = mjw.put_data(m, d, nconmax=32, njmax=128)
    mjw.forward(mm, dd)
    n = int(dd.nacon.numpy()[0])
    got = sorted(round(float(x), 6) for x in dd.contact.dist.numpy()[:n])
  assert len(cpu) == 4
  assert got == pytest.approx(cpu, abs=1e-5)


# -- The cfg -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("goal", sorted(ins.INSERT_GOALS))
def test_the_insert_cfgs(monkeypatch, goal):
  cfg = _cfg(monkeypatch, GOAL=goal)
  actor = list(cfg.observations["actor"].terms)
  assert actor[-5:] == ["object_pos", "object_yaw", "target_pos", "box_yaw", "pin_tilt"]
  critic = list(cfg.observations["critic"].terms)
  assert critic[-6:] == [
    "grasped",
    "release_grade",
    "release_latched",
    "box_yaw",
    "pin_tilt",
    "pin_seated",
  ]
  assert ins.BLOCK in cfg.scene.entities
  assert "nativeccd" in cfg.sim.mujoco.disableflags
  rw = cfg.rewards
  code = ins.INSERT_GOALS[goal]
  if code.ladder:
    assert rw["ladder"].weight == ins_mdp.LADDER_SCALE
    assert rw["release"].weight == 0.0 and rw["placed"].weight == 0.0
    assert not {"reach", "grasp", "lift", "goal", "goal_near", "retreat"} & set(rw)
  else:
    assert rw["reach"].func is mdp.insert_reach
    assert rw["goal"].func is mdp.insert_approach
    assert rw["depth"].weight == ins.DEPTH_WEIGHT
    assert ("align" in rw) == (code.align_weight > 0)
  assert rw["placed"].params["threshold"] == ins.AT_GOAL_TOL
  assert rw["placed"].params["tilt_std"] == ins.M6_TILT_STD
  hole = cfg.scene.entities[ins.BLOCK].spec_fn().compile()
  stave = mujoco.mj_name2id(hole, mujoco.mjtObj.mjOBJ_GEOM, "block_stave0")
  r_in = hole.geom_pos[stave, 0] - hole.geom_size[stave, 0]
  assert r_in == pytest.approx(code.hole_d / 2, abs=1e-9)
  rl = kangaroo_pick_place_rl_cfg()
  assert rl.widen_actor_on_resume and rl.widen_critic_on_resume


def test_the_pin_and_the_insert_go_together(monkeypatch):
  with pytest.raises(ValueError, match="SHAPE=pin"):
    _cfg(monkeypatch, SHAPE="cube")
  with pytest.raises(ValueError, match="insert GOAL"):
    _cfg(monkeypatch, GOAL="table")
  with pytest.raises(ValueError, match="TARGET"):
    _cfg(monkeypatch, TARGET="B")
  with pytest.raises(ValueError):
    _cfg(monkeypatch, REWARD="R-C-strict+M1+M2+M3+M6+J+M8+S")
  with pytest.raises(ValueError, match="BOX_OBS"):
    _cfg(monkeypatch, BOX_OBS="pose")


def test_the_pin_keeps_its_friction_and_dr_object_its_mass(monkeypatch):
  cfg = _cfg(monkeypatch)
  fr = cfg.events["object_friction"].params
  assert fr["ranges"] == (scene.PIN_BLOCK_FRICTION, scene.PIN_BLOCK_FRICTION)
  assert fr["asset_cfg"].geom_names == scene.pin_geom_names("object")
  assert "object_mass" in cfg.events


# -- The env ------------------------------------------------------------------------------------------


def test_the_reset_stands_the_pin_and_the_block_on_wc_clear_of_each_other(env):
  pin, block = env.scene["object"], env.scene[ins.BLOCK]
  top = surface_top_z(env, SceneEntityCfg("table"))
  table = env.scene["table"]
  tb = table.indexing.body_ids[table.body_names.index("table")]
  shift = env.sim.data.xpos[:, tb, :2] - env.scene.env_origins[:, :2]
  shift = shift - torch.tensor(scene.table_pos(0.837)[:2], device=env.device)
  for ent, half in ((pin, HALF), (block, 0.0)):
    xy = ent.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2] - shift
    assert bool(((xy[:, 0] >= WC.x[0] - 1e-6) & (xy[:, 0] <= WC.x[1] + 1e-6)).all())
    assert bool(((xy[:, 1] >= WC.y[0] - 1e-6) & (xy[:, 1] <= WC.y[1] + 1e-6)).all())
    assert torch.allclose(
      ent.data.root_link_pos_w[:, 2] - top, torch.full_like(top, half), atol=1e-4
    )
  d = torch.norm(
    pin.data.root_link_pos_w[:, :2] - block.data.root_link_pos_w[:, :2], dim=-1
  )
  assert bool((d >= ins.PIN_MIN_SEPARATION - 1e-6).all())
  assert bool((mdp.observations.object_tilt(env, "object") < 1e-4).all())


def test_nothing_moves_when_nothing_touches_it(env):
  """The pin stands and the block sits for 1 s with the arm at home (the box-box defect popped 2 of 8 pins in the
  first step, decision 0005)."""
  pin, block = env.scene["object"], env.scene[ins.BLOCK]
  p0, b0 = pin.data.root_link_pos_w.clone(), block.data.root_link_pos_w.clone()
  _hold(env, 50)
  assert torch.norm(pin.data.root_link_pos_w - p0, dim=-1).max() < 5e-4
  assert torch.norm(block.data.root_link_pos_w - b0, dim=-1).max() < 5e-4
  assert bool((mdp.observations.object_tilt(env, "object") < math.radians(0.5)).all())


def test_the_observed_pin_and_block_are_the_truth_with_the_noise_off(env):
  from pal_mjlab.tasks.manipulation.mdp.observations import _to_base_frame

  robot, pin, block = env.scene["robot"], env.scene["object"], env.scene[ins.BLOCK]
  _hold(env, 10)  # past the tracker's lag: nothing has moved
  obs = env.observation_manager.compute_group("actor")
  terms = env.observation_manager.active_terms["actor"]
  dims = env.observation_manager.group_obs_term_dim["actor"]
  cut, i = {}, 0
  for n, dd in zip(terms, dims, strict=True):
    cut[n] = obs[:, i : i + dd[0]]
    i += dd[0]
  assert torch.allclose(
    cut["object_pos"], _to_base_frame(robot, pin.data.root_link_pos_w), atol=1e-5
  )
  assert torch.allclose(
    cut["target_pos"], _to_base_frame(robot, block.data.root_link_pos_w), atol=1e-5
  )
  # upright to the 0.1 deg a pin settles by on its foot
  assert torch.allclose(cut["pin_tilt"], torch.zeros_like(cut["pin_tilt"]), atol=5e-3)
  for name, ent in (("object_yaw", pin), ("box_yaw", block)):
    psi = mdp.yaw_relative_to_base(env, ent.data.root_link_quat_w)
    want = torch.stack([torch.cos(4 * psi), torch.sin(4 * psi)], dim=-1)
    assert torch.allclose(
      cut[name], want, atol=1e-3
    )  # x 4 of the settling pin's ~2e-5 rad of rock


def test_the_pin_observations_are_the_same_a_quarter_turn_on_and_end_for_end(env):
  pin = env.scene["object"]
  pos = pin.data.root_link_pos_w.clone()
  q = pin.data.root_link_quat_w.clone()
  tilt = quat_from_euler_xyz(
    torch.full_like(pos[:, 0], 0.2),
    torch.full_like(pos[:, 0], -0.1),
    torch.zeros_like(pos[:, 0]),
  )
  base = quat_mul(tilt, q)
  quarter = quat_mul(base, _yaw_quat(torch.full_like(pos[:, 0], math.pi / 2)))
  flip = quat_mul(
    base,
    quat_from_euler_xyz(
      torch.full_like(pos[:, 0], math.pi),
      torch.zeros_like(pos[:, 0]),
      torch.zeros_like(pos[:, 0]),
    ),
  )
  out = []
  for qq in (base, quarter, flip):
    pin.write_root_link_pose_to_sim(torch.cat([pos, qq], dim=-1))
    env.sim.forward()
    out.append(mdp.pin_tilt_b(env, "object"))
  assert torch.allclose(out[0], out[1], atol=1e-5)
  assert torch.allclose(out[0], out[2], atol=1e-5)
  assert float(out[0].norm(dim=-1).min()) > 0.1


def _stand_in_block(env, bottom_above: float, dxy=(0.0, 0.0)) -> None:
  """The pin upright in (or beside) the block, its bottom ``bottom_above`` over the block's bottom face."""
  pin, block = env.scene["object"], env.scene[ins.BLOCK]
  b = block.data.root_link_pos_w
  pos = b.clone()
  pos[:, 0] += dxy[0]
  pos[:, 1] += dxy[1]
  pos[:, 2] += bottom_above + HALF
  _place(pin, pos, block.data.root_link_quat_w.clone())
  env.sim.forward()


def test_seated_holds_for_a_pin_stood_in_the_hole_and_not_for_one_wedged_or_beside(env):
  """Positive control: a pin stood in the hole on the table is seated and stays so; one held 10 mm into the hole
  (bottom 46 mm over the floor) and one standing on the table beside the block are not."""
  _stand_in_block(env, 0.0)
  _hold(env, 25)
  st = mdp.insert_state(env, "object", ins.BLOCK, **GEO)
  assert bool(st.seated().all()), st.height
  assert bool((st.depth > scene.BLOCK_HEIGHT - 0.003).all())
  _stand_in_block(env, scene.BLOCK_HEIGHT - 0.010)
  st = mdp.insert_state(env, "object", ins.BLOCK, **GEO)
  assert not bool(st.seated().any())
  assert torch.allclose(st.depth, torch.full_like(st.depth, 0.010), atol=1e-4)
  _stand_in_block(env, 0.0, dxy=(0.06, 0.0))
  st = mdp.insert_state(env, "object", ins.BLOCK, **GEO)
  assert not bool(st.seated().any())
  assert bool((st.axis_offset > 0.05).all())


def test_placed_needs_the_lift(env):
  """A pin stood seated without ever being lifted is not placed (L4), though it is seated."""
  _stand_in_block(env, 0.0)
  _hold(env, 5)
  term = env.reward_manager.get_term_cfg("placed")
  p = {
    k: v
    for k, v in term.params.items()
    if k
    in (
      "command_name",
      "object_name",
      "grasp_sensor_name",
      "threshold",
      "support_sensor_name",
      "jaw_cfg",
      "jaw_open",
    )
  }
  placed = _placed_bool(env, velocity_threshold=0.05, **p)
  assert not bool(placed.any())
  assert bool(mdp.insert_state(env, "object", ins.BLOCK, **GEO).seated().all())


def test_pushing_the_block_costs(env):
  block = env.scene[ins.BLOCK]
  cmd = env.command_manager.get_term("goal_pose")
  pos = block.data.root_link_pos_w.clone()
  start = cmd.box_start_xy.clone()
  pos[:, :2] = start + torch.tensor([0.02, 0.0], device=env.device)
  _place(block, pos, block.data.root_link_quat_w.clone())
  env.sim.forward()
  pen = mdp.box_displacement_penalty(env, "goal_pose", 0.02)
  assert torch.allclose(pen, torch.ones_like(pen), atol=1e-3)
  goal = cmd._goal()
  assert torch.allclose(goal[:, :2], pos[:, :2], atol=1e-4)
  assert torch.allclose(
    goal[:, 2] - pos[:, 2], torch.full_like(pos[:, 2], HALF), atol=1e-4
  )


# -- L5: the per-step reward rises along the insertion ----------------------------------------------------


def _g(x, s):
  return math.exp(-((x / s) ** 2))


def _in_income(stage: str, tilt_deg: float = 0.0) -> float:
  """IN's task terms at plan §12's L5 stages, from the term definitions (`mdp.insert`) and weights (`insert`):
  the pin held upright, its bottom on the hole's axis unless said; reach 1 while the hand is at the grasp point."""
  a, d, u = ins.ALIGN_WEIGHT, ins.DEPTH_WEIGHT, ins.UPRIGHT_WEIGHT
  w_goal, w_near = ins.APPROACH_WEIGHT, ins.NEAR_WEIGHT

  def approach(dist):
    return w_goal * _g(dist, ins.APPROACH_STD) + w_near * _g(dist, ins.NEAR_STD)

  def lift(h_centre, dist):
    return 2.0 * min(h_centre, 0.10) / 0.10 * (1 - _g(dist, ins.LIFT_FADE_STD))

  H = scene.BLOCK_HEIGHT
  if stage == "reach":
    return 1.0
  if stage == "carry":  # 10 cm up, 20 cm from the entry
    return 1 + 1 + lift(0.10 + HALF, 0.2) + approach(0.2)
  if stage == "over":  # bottom 3 cm over the entry, on the axis
    return 1 + 1 + lift(H + 0.03 + HALF, 0.03) + approach(0.03) + a + u
  if stage.startswith("in"):  # bottom `frac` of the block's height below the entry
    frac = float(stage[2:])
    return 1 + 1 + lift(H * (1 - frac) + HALF, 0.0) + approach(0.0) + a + u + d * frac
  if stage == "seated_held":
    return _in_income("in1.0")
  q = 0.25 + 0.75 * _g(math.radians(tilt_deg), ins.M6_TILT_STD)
  if stage == "released":  # jaw open, the hand still around the pin (reach ~1)
    return 1 + approach(0.0) + 4 * q + 8 * q
  if stage == "retreated":  # hand 10 cm over the pin's top
    return _g(0.10, 0.2) + approach(0.0) + 4 * q + 8 * q + 2.0
  raise ValueError(stage)


def test_l5_the_in_reward_rises_along_the_insertion():
  stages = ["reach", "carry", "over", "in0.1", "in0.5", "seated_held", "retreated"]
  inc = [_in_income(s) for s in stages]
  assert all(b > a for a, b in zip(inc, inc[1:], strict=False)), dict(
    zip(stages, inc, strict=True)
  )
  # The let-go out-earns the hold, also at a 3 deg let-go (M6's tilt grade): by >= 3 per step.
  assert _in_income("released", 3.0) > _in_income("seated_held") + 1.0
  assert _in_income("retreated", 3.0) > _in_income("seated_held") + 3.0


def test_l5_the_ladder_floors_and_shaping():
  floors = ins_mdp.LADDER_FLOORS
  # Every rung's best is below the next rung's worst, and every floor gap x the weight >= 2 x the largest per-step
  # penalty inserting pays (block_moved 2), S11.2's decision 1.
  for lo, hi in zip(floors, floors[1:], strict=False):
    assert lo + ins_mdp.LADDER_SHAPE < hi
    assert (hi - lo) * ins_mdp.LADDER_SCALE >= 2 * 2.0


def test_nothing_is_drawn_under_the_home_hand(monkeypatch):
  """The reset keeps the pin's and the block's centres out of `insert.HOME_HAND_KEEPOUT` (robot frame), where the
  start pose's fingers hang below a standing pin's top, wherever DR-scene puts the table; 3 resets x 64 envs."""
  from pal_mjlab.tasks.manipulation.mdp.events import in_keepout

  cfg = _cfg(monkeypatch, num_envs=64)
  e = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  try:
    for _ in range(3):
      e.reset()
      for name in ("object", ins.BLOCK):
        xy = e.scene[name].data.root_link_pos_w[:, :2] - e.scene.env_origins[:, :2]
        assert not bool(in_keepout(xy, ins.HOME_HAND_KEEPOUT).any()), name
  finally:
    e.close()
