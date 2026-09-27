"""Plan S7-2's corners check: the scripted pick-and-place, under baseline B, from every spawn
box to the target box's corners and with DR-scene's table shift, at >= 9/10 per box.

The plan made this check the arbiter of two things before any S7-2 run: P2's spawn boxes W1 and
W2 with P2's wider action scales (prep 2, "the corners test (>= 9/10 per box) on the fda9492
model") and the size of B's target box (prep 5a, "the scripted corners test over target box
+/- table shift (>= 9/10) deciding its final size"). The box the plan proposed, (0.42, 0.07)
+/-3 cm, did not pass: at the 0.815 m table the arm cannot place near the torso with its upper
arm clear, and 11-18 % of that box's targets with the table shift had no contact-free plan
(66 % for a cube at +10 deg; `artifacts/s7-2/ikmap/`). B's box is (0.46, 0.08) +/-2 cm (Siwei,
2026-09-26), and this is its check.

The ten cases per box (`scripted_pick_place.corner_cases`): each spawn corner at +/- the box's
largest yaw, paired with the target box's centre, its four corners with the table shifted
+/-2 cm the same way and +/-1 cm up or down -- the farthest targets relative to the robot --
its four corners unshifted, and its centre with the table shifted towards the torso. The
script carries 5 cm over the table (`CLEAR_HEIGHT`), not 10: at the raised table a 10 cm carry
folds the upper arm into the torso over this region, where S5.2's policies carried on 3-8 cm
arcs anyway. Its episode is 30 s: the script is slow and closed-loop on purpose, and what is
checked here is the action space, not the clock (`rollout(episode_length_s=...)`).

Measured 2026-09-26 (Mac, CPU): 9/10 on each of S71, W1 and W2.

Plan S7-3 (S7-2 on S8's G20 plant) runs the same check as its gate 4, on its own plant and, for
B-SV, under V: ``GAINS=G20`` builds each box's arm on that gain code, ``TOOL_SPEED=V`` adds the
tool speed cap. They are read from the variables the task reads, so unset they give S7-2's
check.
"""

import importlib.util
import os
import pathlib
import sys

import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import pytest
from conftest import get_test_device
from pal_mjlab.tasks.manipulation.pick_place import pick_place_env_cfg as pp
from pal_mjlab.tasks.manipulation.pick_place.kangaroo.env_cfg import SPAWN_ROBOTS
from pal_mjlab.tasks.manipulation.robots import kangaroo_grippers_left_arm

_SCRIPT = pathlib.Path(__file__).parent.parent / "scripts/scripted_pick_place.py"
_SPEC = importlib.util.spec_from_file_location("scripted_pick_place_s72", _SCRIPT)
assert _SPEC and _SPEC.loader
scripted = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = scripted
_SPEC.loader.exec_module(scripted)

REQUIRED = 9
#: The script's carry height over the grasp pose, m (see the module docstring).
CARRY_HEIGHT = 0.05
B = pp.BASELINES["B"]
#: The plant and the tool speed cap (plan S7-3, see the module docstring).
GAINS = os.environ.get("GAINS", "G0")
TOOL_SPEED = os.environ.get("TOOL_SPEED", "none")
#: Under V the script stretches every move to V's bound; the V gate gives it 40 s
#: (`test_pick_place_tool_speed.py`).
EPISODE_S = 40.0 if TOOL_SPEED == "V" else 30.0


@pytest.fixture(scope="module", params=sorted(pp.SPAWN_BOXES))
def corners(request):
  name = request.param
  cases = scripted.corner_cases(pp.SPAWN_BOXES[name], B.target_xy, B.target_half_xy)
  table = {}

  def hook(phase, env):
    if not table:  # each env's table, as the episode ran on it
      table["xy"] = scripted._surface_xy(env, pp.TABLE)
      table["z"] = scripted._surface_top(env, pp.TABLE)

  saved = scripted.CLEAR_HEIGHT
  scripted.CLEAR_HEIGHT = CARRY_HEIGHT
  try:
    rows = scripted.rollout(
      num_envs=len(cases),
      device=get_test_device(),
      reward="R-C-strict+M1+M2+M3+M6",
      verbose=False,
      action_slew_rate_scale=0.5,
      robot=kangaroo_grippers_left_arm(GAINS, SPAWN_ROBOTS[name]),
      baseline=B,
      spawn_box=pp.SPAWN_BOXES[name],
      dr="DR-scene",
      tool_speed_cap=pp.TOOL_SPEED_CAPS[TOOL_SPEED],
      spawns=[c[0] for c in cases],
      targets=[c[1] for c in cases],
      table_offsets=[c[2] for c in cases],
      episode_length_s=EPISODE_S,
      hook=hook,
    )
  finally:
    scripted.CLEAR_HEIGHT = saved
  return name, cases, rows, table


def test_the_corners_are_placed(corners):
  name, cases, rows, _ = corners
  placed = [r["env"] for r in rows if r["placed_and_lifted"]]
  failed = [
    (r["env"], cases[r["env"]], r["plan_error"], r["final_cube_to_goal_mm"])
    for r in rows
    if not r["placed_and_lifted"]
  ]
  assert len(placed) >= REQUIRED, f"{name}: {len(placed)}/{len(rows)}; failed {failed}"


def test_the_script_never_outran_the_slew(corners):
  _, _, rows, _ = corners
  assert rows[0]["over_slew_env_steps"] == 0


def test_the_tables_were_where_the_cases_put_them(corners):
  """Positive control on the shifts: each env's table top sat at its case's offset from the
  nominal pose while the episode ran -- DR-scene's own draw overwritten, not added to."""
  _, cases, _, table = corners
  x0, y0, _ = pp.scene.table_pos(B.table_top_z)
  for i, (_, _, shift) in enumerate(cases):
    assert table["xy"][i, 0] - x0 == pytest.approx(shift[0], abs=1e-5)
    assert table["xy"][i, 1] - y0 == pytest.approx(shift[1], abs=1e-5)
    assert table["z"][i] - B.table_top_z == pytest.approx(shift[2], abs=1e-5)
