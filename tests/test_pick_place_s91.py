"""Plan S9.1 (`pick-place-v1-objects-plan.md`, *S9.1*): step 0 item 3's build and tests, and decision
0003's box-box contact.

* Decision 0003: every tabletop task sends box-box pairs to mujoco_warp's ``box_box`` collider
  (``nativeccd`` off, `tabletop_env_cfg.SIM_DISABLE_FLAGS`); ``CONTACT=ccd`` gives back the GJK/EPA
  path every run before S9.1 trained on. The regression takes the red cube's poses on the base from a
  recorded press (S9 ST-GA s2's closed jaw pushing it 0.1-17 mm into the base,
  `tests/data/s91_stack_press.json`) and checks that mujoco_warp, under the task's options, returns
  CPU MuJoCo's contacts there; GJK/EPA, the positive control, does not. (The press itself is chaotic:
  replayed open-loop it sinks the cube 14 mm on the recorded model and ~1 mm on one compiled on another
  host, whose mesh hulls differ slightly, so it is not the test.)
* `BS9.1` (SC): the S11 stack at ``T11W`` with ``CMD_OBS=cmd``, the split tracker, `base_moved`, PQ,
  AH10 and no SA. SC-Y (``BASE_OBS=pose``): the base's tracked yaw last on the actor (live on the
  critic) and M6's yaw grade taken against the base -- a cube aligned with a base turned 30 deg has
  no yaw error. SC-SAw: SA with S9.1's widths, ~0 at the `OK` bounds' widths where SAW pays, and the
  per-step reward still rising strictly along a scripted stack (L5).
"""

from __future__ import annotations

import collections
import importlib.util
import json
import math
import pathlib
import sys

import mujoco
import numpy as np
import pytest
import torch
import warp as wp
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import Scene
from mjlab.utils.lab_api.math import quat_from_euler_xyz
from pal_mjlab.tasks.manipulation import mdp, scene
from pal_mjlab.tasks.manipulation import tabletop_env_cfg as tt
from pal_mjlab.tasks.manipulation.mdp.stack import (
  OK_DXY,
  OK_HEIGHT,
  OK_SPEED,
  OK_TILT,
)
from pal_mjlab.tasks.manipulation.pick_place import pick_place_env_cfg as pp
from pal_mjlab.tasks.manipulation.pick_place import rewards as rw
from pal_mjlab.tasks.manipulation.pick_place.kangaroo.env_cfg import (
  kangaroo_pick_place_env_cfg,
)
from pal_mjlab.tasks.manipulation.robots import (
  KANGAROO_GRIPPERS_LEFT_ARM_WIDE,
  kangaroo_grippers_left_arm,
)

RCTP = "R-C-strict+M1+M2+M3+M6+J+M8"
#: `BS9.1` (plan S9.1's table): DEP1's BS2 (BS + PQ + AH10, CMD_OBS=cmd) on the S11 stack at T11W.
BS91 = {
  "REWARD": RCTP,
  "DR": "DR-scene",
  "PPO": "PPO-stack",
  "SLEW": "stack",
  "BASELINE": "B",
  "SPAWN": "W2",
  "TOOL_SPEED": "none",
  "GAINS": "G20",
  "EFFORT": "CTP",
  "LAG": "FP+JS",
  "OCC": "none",
  "OBJ_OBS": "live",
  "ACTION": "abs",
  "CMD_OBS": "cmd",
  "TARGET": "T11W",
  "EPISODE_S": "10",
  "SHAPE": "cube",
  "GOAL": "stack",
  "GATE": "none",
  "CRITIC_OBS": "q",
  "ARM_TABLE": "AH10",
  "TRACKER": "split",
  "ARM_HOME": "mirror",
}
CODES = {
  "SC": {},
  "SC-Y": {"REWARD": RCTP + "+YG", "BASE_OBS": "pose"},
  "SC-SAw": {"REWARD": RCTP + "+SAW"},
}
_ALL = {*BS91, "BASE_OBS", "CONTACT", "HANDOFF_ROUTE", "ARMS", "HANDOFF_TRAIN", "START"}


def _cfg(monkeypatch, code: str = "SC", num_envs: int = 4, **changes):
  for var in _ALL:
    monkeypatch.delenv(var, raising=False)
  for var, value in {**BS91, **CODES[code], **changes}.items():
    if value is None:
      monkeypatch.delenv(var, raising=False)
    else:
      monkeypatch.setenv(var, value)
  cfg = kangaroo_pick_place_env_cfg()
  cfg.scene.num_envs = num_envs
  cfg.seed = 0
  return cfg


# -- Decision 0003: box-box contact ------------------------------------------------------------


def test_the_tabletop_default_sends_box_pairs_to_box_box(monkeypatch):
  assert "nativeccd" in tt.SIM_DISABLE_FLAGS
  cfg = _cfg(monkeypatch)
  assert "nativeccd" in cfg.sim.mujoco.disableflags
  model = Scene(cfg.scene, device="cpu").compile()
  cfg.sim.mujoco.apply(model)
  assert model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
  # The one opt-out: the physics every run before S9.1 trained on.
  old = _cfg(monkeypatch, CONTACT="ccd")
  assert "nativeccd" not in old.sim.mujoco.disableflags
  with pytest.raises(ValueError, match="CONTACT"):
    _cfg(monkeypatch, CONTACT="primitive")


_PRESS = pathlib.Path(__file__).parent / "data" / "s91_stack_press.json"


def _two_cubes(cfg) -> mujoco.MjModel:
  """The two task cubes alone, under the task's MuJoCo options (``cfg.sim.mujoco``)."""
  spec = mujoco.MjSpec()
  for name, z in (("base_object", 0.025), ("object", 0.075)):
    cube = scene.get_cube_spec(name=name)
    spec.worldbody.add_frame(pos=(0.0, 0.0, z)).attach_body(
      cube.body(name), f"{name}/", ""
    )
  model = spec.compile()
  cfg.sim.mujoco.apply(model)
  return model


def _manifolds(model: mujoco.MjModel, poses) -> list[tuple[int, float, int, float]]:
  """(CPU MuJoCo's contact count, its deepest distance, mujoco_warp's, its deepest) for the red cube
  at each pose in the base's frame."""
  import mujoco_warp as mjw

  data = mujoco.MjData(model)
  base = model.jnt_qposadr[model.body("base_object/base_object").jntadr[0]]
  red = model.jnt_qposadr[model.body("object/object").jntadr[0]]
  device = get_test_device()
  out = []
  with wp.ScopedDevice("cuda:0" if device.startswith("cuda") else "cpu"):
    m = mjw.put_model(model)
    for pose in poses:
      data.qpos[base : base + 7] = (0.0, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0)
      data.qpos[red : red + 3] = np.asarray(pose[:3]) + (0.0, 0.0, 0.025)
      data.qpos[red + 3 : red + 7] = np.asarray(pose[3:]) / np.linalg.norm(pose[3:])
      mujoco.mj_forward(model, data)
      d = mjw.put_data(model, data, nconmax=32, njmax=256)
      mjw.forward(m, d)
      host = mujoco.MjData(model)
      mjw.get_data_into(host, model, d)
      deepest = lambda dd: min(dd.contact[i].dist for i in range(dd.ncon))  # noqa: E731
      out.append((data.ncon, deepest(data), host.ncon, deepest(host)))
  return out


def test_box_pairs_get_cpu_mujocos_contacts(monkeypatch):
  """Decision 0003's regression. At the 58 poses of a recorded press (the closed jaw pushing the red
  cube 0.1-17 mm into the base) mujoco_warp under the task's options returns CPU MuJoCo's contacts:
  the same number at the same depth. Positive control: GJK/EPA (``CONTACT=ccd``) does not, at any of
  them -- with the two cubes alone it returns 2-8 contacts (measured 2026-10-05, 58 / 58 differ); in
  the recorded run, beside the robot's mesh pairs, it returned 1-2 where CPU MuJoCo returns 4-6, and
  the cube sank 11-17 mm. The count it returns depends on the scene, which is the defect."""
  poses = json.loads(_PRESS.read_text())["poses"]
  fixed = _manifolds(_two_cubes(_cfg(monkeypatch)), poses)
  for cpu_n, cpu_d, warp_n, warp_d in fixed:
    assert cpu_n > 0 and warp_n == cpu_n, fixed
    assert warp_d == pytest.approx(cpu_d, abs=1e-5)
  old = _manifolds(_two_cubes(_cfg(monkeypatch, CONTACT="ccd")), poses)
  differ = sum(warp_n != cpu_n for cpu_n, _, warp_n, _ in old)
  assert differ >= len(poses) // 2, old


# -- BS9.1 and its codes -----------------------------------------------------------------------


def test_sc_is_the_s11_stack_at_t11w_in_the_cmd_contract(monkeypatch):
  cfg = _cfg(monkeypatch)
  actor = cfg.observations["actor"].terms
  assert actor["target_pos"].func is mdp.stack_goal_position_tracked_b
  assert actor["object_pos"].func is mdp.object_position_biased_b
  assert "base_yaw" not in actor and "base_yaw" not in cfg.observations["critic"].terms
  assert "base_moved" in cfg.rewards and "stack_align" not in cfg.rewards
  assert "base_disturbance" not in cfg.rewards
  assert cfg.commands["goal_pose"].yaw_reference is None
  assert cfg.episode_length_s == 10.0
  assert type(cfg.actions["joint_pos"]) is mdp.RateLimitedJointPositionActionCfg


def test_unset_base_obs_leaves_r_st_w_unchanged(monkeypatch):
  """BASE_OBS unset (and the S9.1 code) builds S11's R-ST-W cfg: same terms, same sizes."""
  r_st_w = _cfg(monkeypatch, CMD_OBS=None, REWARD=RCTP + "+SA")
  same = _cfg(monkeypatch, CMD_OBS=None, REWARD=RCTP + "+SA", BASE_OBS="goal")
  for group in ("actor", "critic"):
    assert list(r_st_w.observations[group].terms) == list(
      same.observations[group].terms
    )
  assert sorted(r_st_w.rewards) == sorted(same.rewards)
  assert list(r_st_w.events) == list(same.events)


@pytest.mark.parametrize(
  "changes, match",
  [
    ({"REWARD": RCTP + "+YG"}, "BASE_OBS=pose"),
    ({"BASE_OBS": "pose", "TRACKER": "B"}, "TRACKER=split"),
    ({"BASE_OBS": "pose", "GOAL": "table", "TARGET": "T11"}, "GOAL=stack"),
    ({"BASE_OBS": "yaw"}, "BASE_OBS"),
  ],
)
def test_combinations_that_are_not_built_raise(monkeypatch, changes, match):
  with pytest.raises(ValueError, match=match):
    _cfg(monkeypatch, **changes)


def test_sa_and_saw_are_one_or_the_other():
  with pytest.raises(ValueError, match="pick one"):
    rw._s52_arm(("M1", "M2", "M3", "M6", "J", "M8", "SA", "SAW"))


@pytest.fixture(scope="module")
def sc_y_env():
  mp = pytest.MonkeyPatch()
  try:
    cfg = _cfg(mp, "SC-Y", num_envs=8)
    env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
    env.reset()
    yield env
    env.close()
  finally:
    mp.undo()


def test_sc_y_observes_the_base_yaw_last(sc_y_env):
  env = sc_y_env
  actor = env.observation_manager.active_terms["actor"]
  critic = env.observation_manager.active_terms["critic"]
  assert actor[-1] == "base_yaw" and critic[-1] == "base_yaw"
  dims = env.observation_manager.group_obs_term_dim["actor"]
  assert tuple(dims[-1]) == (2,)
  assert env.cfg.commands["goal_pose"].yaw_reference == "base_object"
  # The critic's is the live yaw: (cos 4 psi, sin 4 psi) of the base against the robot base.
  base = env.scene["base_object"]
  psi = mdp.yaw_relative_to_base(env, base.data.root_link_quat_w)
  live = mdp.box_yaw_b(env, "base_object")
  want = torch.stack([torch.cos(4 * psi), torch.sin(4 * psi)], dim=-1)
  assert torch.allclose(live, want, atol=1e-5)


def test_the_yaw_grade_is_taken_against_the_base(sc_y_env):
  """A cube turned 30 deg on a base turned 30 deg is aligned (plan S9.1 step 0 item 3); against the
  world's axes it would be 30 deg off."""
  env = sc_y_env
  n = env.num_envs
  yaw = torch.full((n,), math.radians(30.0), device=env.device)
  zeros = torch.zeros_like(yaw)
  q = quat_from_euler_xyz(zeros, zeros, yaw)
  for name in ("base_object", "object"):
    entity = env.scene[name]
    pose = entity.data.root_link_pose_w.clone()
    pose[:, 3:7] = q
    entity.write_root_link_pose_to_sim(pose)
  env.sim.forward()
  cube_q = env.scene["object"].data.root_link_quat_w
  ref = mdp.yaw_reference_quat(env, "goal_pose")
  assert ref is not None
  assert float(mdp.yaw_error_mod_90(cube_q, ref).abs().max()) < 1e-4
  assert torch.allclose(
    mdp.yaw_error_mod_90(cube_q), torch.full_like(yaw, math.radians(30.0)), atol=1e-4
  )


def test_saw_pays_where_the_policies_let_go_and_sa_does_not():
  """At STSA-c s1's hold over the base (step 0 item 4: 1.9 cm off, 2.5 cm up, 22 deg, 0.12 m/s) SA's
  Gaussians multiply to ~0 (L15); SAW's still pay a usable fraction."""
  dxy, height, tilt, speed = 0.019, 0.025, math.radians(22.0), 0.12
  sa = math.exp(
    -(
      (dxy / OK_DXY) ** 2
      + (height / OK_HEIGHT) ** 2
      + (tilt / OK_TILT) ** 2
      + (speed / OK_SPEED) ** 2
    )
  )
  w_dxy, w_height, w_tilt, w_yaw, w_speed = rw.SAW_WIDTHS
  assert w_yaw is None
  saw = math.exp(
    -(
      (dxy / w_dxy) ** 2
      + (height / w_height) ** 2
      + (tilt / w_tilt) ** 2
      + (speed / w_speed) ** 2
    )
  )
  assert sa < 1e-3 and saw > 0.03, (sa, saw)


# -- L5: SC-SAw's reward still rises along a scripted stack ------------------------------------

_SCRIPT = pathlib.Path(__file__).parent.parent / "scripts/scripted_pick_place.py"
_SPEC = importlib.util.spec_from_file_location("scripted_pick_place_s91", _SCRIPT)
assert _SPEC and _SPEC.loader
scripted = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = scripted
_SPEC.loader.exec_module(scripted)

ORDER = ("reach", "carry", "over_target", "held_on_target", "placed")
VERTICAL_LOWERING = tuple(
  step
  for phase in scripted.PHASES
  for step in (
    (("lower", "retreat", True, 30), ("lower_down", "place", True, 40))
    if phase[0] == "lower"
    else (phase,)
  )
)
TASK_TERMS = (
  "reach",
  "grasp",
  "lift",
  "goal",
  "goal_near",
  "release",
  "placed",
  "retreat",
  "stack_align",
)
CTP = pp.EFFORT_BOUNDS["CTP"]


def _robot():
  return kangaroo_grippers_left_arm(
    "G20", KANGAROO_GRIPPERS_LEFT_ARM_WIDE, arm_effort_scale=CTP.effort_scale
  )


def _stack_cfg(reward: str):
  cfg = pp.make_pick_place_env_cfg(
    _robot(),
    gripper_name="left",
    reward=reward,
    baseline=pp.BASELINES["B"],
    spawn=pp.SPAWN_BOXES["W2"],
    dr="DR-scene",
    effort=CTP,
    latency=tt.LATENCY_MODELS["FP+JS"],
    stack_speed=True,
    episode_length_s=10.0,
    goal="stack",
  )
  cfg.scene.num_envs = 1
  return cfg


@pytest.fixture(scope="module")
def saw_stack_episode():
  """One scripted stack (no lock) on the base in B's box, every task term of SC-SAw recorded at
  every step for env 0, weighted."""
  rec = collections.defaultdict(list)
  terms: dict = {}

  def hook(phase, env):
    if not terms:
      built = _stack_cfg(RCTP + "+SAW").rewards
      for cfg in built.values():
        for value in cfg.params.values():
          if isinstance(value, SceneEntityCfg):
            value.resolve(env.scene)
      terms.update({n: c for n, c in built.items() if n in TASK_TERMS})
    rec[phase].append(
      {n: float(c.weight * c.func(env, **c.params)[0]) for n, c in terms.items()}
    )

  rows = scripted.rollout(
    device=get_test_device(),
    verbose=False,
    robot=_robot(),
    baseline=pp.BASELINES["B"],
    spawn_box=pp.SPAWN_BOXES["W2"],
    dr="DR-scene",
    effort=CTP,
    latency=tt.LATENCY_MODELS["JS"],
    stack_speed=True,
    episode_length_s=70.0,
    goal="stack",
    num_envs=1,
    reward=RCTP + "+SAW",
    hook=hook,
    phases=VERTICAL_LOWERING,
  )
  assert rows[0]["placed_and_lifted"], rows[0]
  return rec


def test_saw_keeps_the_reward_rising_along_the_stack(saw_stack_episode):
  rec = saw_stack_episode
  total = lambda step: sum(step.values())  # noqa: E731
  placed = [t for t in rec["open"] + rec["retreat"] if t["placed"] > 0]
  assert placed, "the cube never counted as placed on the base"
  totals = {
    "reach": total(rec["descend"][-1]),
    "carry": total(rec["lift"][-1]),
    "over_target": total(rec["steady"][-1]),
    "held_on_target": total(rec["hold"][-1]),
    "placed": total(placed[-1]),
  }
  values = [totals[name] for name in ORDER]
  assert all(a < b for a, b in zip(values, values[1:], strict=False)), totals
  # SAW pays on the base, nothing before the carry, and in the carry (which ends over the goal, a few
  # centimetres up: SAW's 3 cm height width is the gradient it is for; measured 2.0 against a hold's
  # 5.9 at weight 6) under half the hold -- the ordering above is what binds.
  hold = max(t["stack_align"] for t in rec["hold"])
  assert hold > 0.3 * rw.SAW_WEIGHT
  for phase in ("approach", "descend", "close", "lift"):
    assert all(t["stack_align"] < 1e-3 * rw.SAW_WEIGHT for t in rec[phase]), phase
  assert max(t["stack_align"] for t in rec["carry"]) < 0.5 * hold
