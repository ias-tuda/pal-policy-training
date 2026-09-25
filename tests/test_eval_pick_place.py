"""The eval script's per-episode accounting.

`Tally` is where an evaluation goes quietly wrong: a latch that does not clear turns every
later episode into a success, and an off-by-one in the step counter mis-states time-to-place.
Neither needs a GPU or a checkpoint to test, so neither should wait for one.
"""

import importlib.util
import pathlib

import torch

_SPEC = importlib.util.spec_from_file_location(
  "eval_pick_place", pathlib.Path(__file__).parent.parent / "scripts/eval_pick_place.py"
)
assert _SPEC and _SPEC.loader
eval_pick_place = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(eval_pick_place)
Tally = eval_pick_place.Tally


def _flags(**overrides) -> dict[str, torch.Tensor]:
  base = dict.fromkeys(Tally.FIELDS, False)
  base.update(overrides)
  return {k: torch.tensor([bool(v)]) for k, v in base.items()}


def test_a_flag_seen_once_survives_to_the_end_of_the_episode():
  """Every rate here is "did this ever happen", so one step of contact is enough."""
  tally = Tally(1, "cpu")
  tally.observe(**_flags(grasped=True))
  for _ in range(5):
    tally.observe(**_flags())
  tally.finish(torch.tensor([0]))
  assert tally.done[0]["grasped"] is True
  assert tally.done[0]["lifted"] is False


def test_the_latches_clear_between_episodes():
  tally = Tally(1, "cpu")
  tally.observe(**_flags(grasped=True, placed=True))
  tally.finish(torch.tensor([0]))
  tally.observe(**_flags())
  tally.finish(torch.tensor([0]))
  assert tally.done[0]["grasped"] and tally.done[0]["placed"]
  assert not tally.done[1]["grasped"] and not tally.done[1]["placed"]
  assert tally.done[1]["steps_to_place"] == -1


def test_time_to_place_records_the_first_success_not_the_last():
  tally = Tally(1, "cpu")
  for _ in range(3):
    tally.observe(**_flags())
  tally.observe(**_flags(placed=True))  # step index 3
  tally.observe(**_flags(placed=True))
  tally.finish(torch.tensor([0]))
  assert tally.done[0]["steps_to_place"] == 3
  assert tally.done[0]["length"] == 5


def test_episodes_that_end_on_different_steps_are_kept_apart():
  """Envs finish at different times; one finishing must not reset or bank another."""
  tally = Tally(3, "cpu")
  three = lambda **kw: {  # noqa: E731
    k: torch.tensor(kw.get(k, [False, False, False])) for k in Tally.FIELDS
  }
  tally.observe(**three(grasped=[True, False, False]))
  tally.finish(torch.tensor([0]))
  tally.observe(**three(lifted=[False, True, False]))
  tally.finish(torch.tensor([1, 2]))

  assert tally.done[0]["grasped"] and not tally.done[0]["lifted"]
  assert tally.done[1]["lifted"], "env 1's flag was cleared by env 0 finishing"
  assert not tally.done[2]["lifted"]
  assert tally.done[0]["length"] == 1 and tally.done[1]["length"] == 2


def test_the_summary_reports_both_placed_rates():
  tally = Tally(1, "cpu")
  # One slid (placed, never lifted), one carried (placed and lifted), one failure.
  for placed, lifted in ((True, False), (True, True), (False, False)):
    tally.observe(**_flags(placed=placed, placed_and_lifted=placed and lifted))
    tally.finish(torch.tensor([0]))
  out = eval_pick_place.summarise(tally, step_dt=0.02, episodes=3)
  assert out["episodes"] == 3
  assert out["placed"] == 2 / 3
  assert out["placed_and_lifted"] == 1 / 3
  # The difference is the thing C0 found: places that were never picks.
  assert out["placed"] - out["placed_and_lifted"] == 1 / 3


def test_a_placement_that_ends_the_episode_is_still_counted():
  """Under a `success` termination the env resets in the step the cube comes to rest, so
  `observe` never sees it placed. `terminal` banks it from the termination -- and credits
  `placed and lifted` only if a lift was seen earlier in the episode."""
  tally = Tally(2, "cpu")
  tally.observe(
    **{k: torch.tensor([False, False]) for k in Tally.FIELDS}
    | {"lifted": torch.tensor([True, False])}
  )
  tally.terminal(torch.tensor([True, True]))
  tally.finish(torch.tensor([0, 1]))
  carried, slid = tally.done
  assert carried["placed"] and carried["placed_and_lifted"]
  assert slid["placed"] and not slid["placed_and_lifted"]
  assert carried["steps_to_place"] == 0


def test_time_to_place_is_none_when_nothing_was_placed():
  """A dead policy must not divide by zero on the way to reporting that it is dead."""
  tally = Tally(1, "cpu")
  tally.observe(**_flags())
  tally.finish(torch.tensor([0]))
  assert eval_pick_place.summarise(tally, 0.02, 1)["time_to_place_s"] is None


def _measure(tally, held, dxy, dz):
  tally.measure(torch.tensor(held), torch.tensor(dxy), torch.tensor(dz))


def test_transport_is_measured_horizontally_and_only_while_held():
  """S5.2 base selection: the horizontal distance to the target while held, and the height
  above it while held within 3 cm horizontally. A cube 6 cm straight above the target is 0 cm
  off horizontally -- the 3-D `position_error` misread exactly that as "no transport"."""
  tally = Tally(1, "cpu")
  _measure(tally, [False], [0.30], [0.00])  # not held: ignored
  _measure(tally, [True], [0.20], [0.05])  # carrying
  _measure(tally, [True], [0.02], [0.06])  # over the target, 6 cm up
  _measure(tally, [True], [0.00], [0.06])  # straight above it
  tally.observe(**_flags())
  tally.finish(torch.tensor([0]))
  row = tally.done[0]
  assert row["held_steps"] == 3
  assert abs(row["held_dxy_mean"] - (0.20 + 0.02 + 0.00) / 3) < 1e-6
  assert abs(row["held_dxy_min"] - 0.0) < 1e-6
  assert row["near_xy_while_held"] is True
  assert abs(row["hover_dz_mean"] - 0.06) < 1e-6  # the 0.05 m step was 20 cm out


def test_an_episode_that_never_holds_has_no_transport_numbers():
  tally = Tally(1, "cpu")
  _measure(
    tally, [False], [0.01], [0.0]
  )  # the cube sits on the target, but was never held
  tally.observe(**_flags())
  tally.finish(torch.tensor([0]))
  row = tally.done[0]
  assert row["held_steps"] == 0 and row["held_dxy_mean"] is None
  assert row["near_xy_while_held"] is False and row["hover_dz_mean"] is None


def test_transport_accumulators_clear_between_episodes_and_stay_per_env():
  tally = Tally(2, "cpu")
  _measure(tally, [True, True], [0.01, 0.10], [0.07, 0.02])
  tally.observe(**{k: torch.tensor([False, False]) for k in Tally.FIELDS})
  tally.finish(torch.tensor([0]))
  _measure(tally, [False, True], [0.50, 0.02], [0.0, 0.03])
  tally.observe(**{k: torch.tensor([False, False]) for k in Tally.FIELDS})
  tally.finish(torch.tensor([0, 1]))
  first, second, other = tally.done
  assert first["near_xy_while_held"] and abs(first["hover_dz_mean"] - 0.07) < 1e-6
  assert second["held_steps"] == 0, "env 0's accumulators survived its reset"
  assert other["held_steps"] == 2 and abs(other["held_dxy_min"] - 0.02) < 1e-6
  assert abs(other["hover_dz_mean"] - 0.03) < 1e-6  # only its near step counts


def test_the_summary_reports_transport_quality():
  tally = Tally(1, "cpu")
  for held, dxy, dz in (
    ([True], [0.02], [0.06]),
    ([True], [0.08], [0.06]),
    ([False], [0.0], [0.0]),
  ):
    _measure(tally, held, dxy, dz)
    tally.observe(**_flags())
    tally.finish(torch.tensor([0]))
  out = eval_pick_place.summarise(tally, 0.02, 3)
  assert out["episodes_held"] == 2
  assert (
    abs(out["within_3cm_xy_while_held"] - 1 / 3) < 1e-9
  )  # a fraction of all episodes
  assert abs(out["held_dxy_mean_m"] - 0.05) < 1e-6
  assert abs(out["hover_dz_mean_m"] - 0.06) < 1e-6


# -- Speed while held (plan S7's speed-bound read) ------------------------------------------

DT = 0.02


def _speed(tally, held, obj, tcp):
  tally.measure_speed(torch.tensor(held), torch.tensor(obj), torch.tensor(tcp), DT)


def _end(tally, env_ids):
  n = tally.num_envs
  tally.observe(**{k: torch.zeros(n, dtype=torch.bool) for k in Tally.FIELDS})
  tally.finish(torch.tensor(env_ids))


def test_peak_speed_counts_only_steps_held_at_both_ends():
  """The jaw closing and the release are not transport: a step counts only if the cube was
  held when it started and when it ended."""
  tally = Tally(1, "cpu")
  _speed(tally, [False], [[0.00, 0.0, 0.0]], [[0.00, 0.0, 0.1]])  # approach
  _speed(
    tally, [True], [[0.05, 0.0, 0.0]], [[0.05, 0.0, 0.1]]
  )  # grasp closes: not counted
  _speed(tally, [True], [[0.06, 0.0, 0.0]], [[0.07, 0.0, 0.1]])  # carry: 0.5 / 1.0 m/s
  _speed(tally, [True], [[0.07, 0.0, 0.0]], [[0.08, 0.0, 0.1]])  # carry: 0.5 / 0.5 m/s
  _speed(
    tally, [False], [[0.20, 0.0, 0.0]], [[0.20, 0.0, 0.1]]
  )  # released: not counted
  _end(tally, [0])
  row = tally.done[0]
  assert abs(row["peak_obj_speed_held"] - 0.5) < 1e-5
  assert abs(row["peak_tcp_speed_held"] - 1.0) < 1e-5


def test_the_jump_across_a_reset_is_not_a_speed():
  """The first observation after a reset is a teleport from the last one before it."""
  tally = Tally(1, "cpu")
  _speed(tally, [True], [[0.00, 0.0, 0.0]], [[0.0, 0.0, 0.0]])
  _speed(tally, [True], [[0.01, 0.0, 0.0]], [[0.0, 0.0, 0.0]])  # 0.5 m/s
  _end(tally, [0])
  _speed(
    tally, [True], [[0.40, 0.3, 0.0]], [[0.4, 0.3, 0.0]]
  )  # new episode, held at once
  _speed(tally, [True], [[0.40, 0.3, 0.0]], [[0.4, 0.3, 0.0]])  # still
  _end(tally, [0])
  first, second = tally.done
  assert abs(first["peak_obj_speed_held"] - 0.5) < 1e-5
  assert second["peak_obj_speed_held"] == 0.0 and second["peak_tcp_speed_held"] == 0.0


def test_an_episode_never_held_for_a_full_step_has_no_peak_speed():
  tally = Tally(2, "cpu")
  _speed(tally, [True, False], [[0.0, 0.0, 0.0]] * 2, [[0.0, 0.0, 0.0]] * 2)
  _speed(tally, [False, False], [[0.3, 0.0, 0.0]] * 2, [[0.3, 0.0, 0.0]] * 2)
  _end(tally, [0, 1])
  assert all(r["peak_obj_speed_held"] is None for r in tally.done)
  assert all(r["peak_tcp_speed_held"] is None for r in tally.done)


def test_the_summary_reports_peak_speeds_over_the_episodes_that_moved_a_held_cube():
  tally = Tally(1, "cpu")
  for step in (0.010, 0.024, None):  # 0.5 and 1.2 m/s peaks, then an episode never held
    held = step is not None
    _speed(tally, [held], [[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]])
    _speed(tally, [held], [[step or 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]])
    _end(tally, [0])
  out = eval_pick_place.summarise(tally, DT, 3)
  assert abs(out["peak_obj_speed_held_mean_mps"] - 0.85) < 1e-5
  assert abs(out["peak_obj_speed_held_max_mps"] - 1.2) < 1e-5
  assert out["peak_tcp_speed_held_max_mps"] == 0.0


# -- The release point (S5.2 report section 6: set down vs let go short) --------------------

GOAL = [[0.42, 0.07, 0.785]]


def _rel(tally, held, obj):
  tally.measure_speed(
    torch.tensor(held),
    torch.tensor(obj),
    torch.zeros(len(held), 3),
    DT,
    torch.tensor(GOAL * len(held)),
  )


def test_the_last_release_is_recorded_with_distance_height_and_speed():
  tally = Tally(1, "cpu")
  _rel(tally, [True], [[0.42, 0.20, 0.83]])  # carrying
  _rel(tally, [False], [[0.42, 0.18, 0.82]])  # a first, early let-go: 11 cm short
  _rel(tally, [True], [[0.42, 0.10, 0.80]])  # re-grasped, carried on
  _rel(tally, [True], [[0.42, 0.08, 0.79]])
  _rel(tally, [False], [[0.42, 0.07, 0.788]])  # the last let-go, on the target
  _end(tally, [0])
  row = tally.done[0]
  assert abs(row["release_dxy"] - 0.0) < 1e-6
  assert abs(row["release_dz"] - 0.003) < 1e-6
  assert abs(row["release_speed"] - 0.0102 / DT) < 1e-4  # |(0, -0.01, -0.002)| / dt


def test_an_episode_that_never_lets_go_has_no_release():
  tally = Tally(2, "cpu")
  _rel(tally, [True, False], [[0.4, 0.2, 0.8]] * 2)
  _rel(tally, [True, False], [[0.4, 0.2, 0.8]] * 2)
  _end(tally, [0, 1])
  assert all(r["release_dxy"] is None for r in tally.done)


def test_the_summary_reports_where_the_cube_was_let_go():
  tally = Tally(1, "cpu")
  for y in (0.07, 0.12):  # on the target, then 5 cm short
    _rel(tally, [True], [[0.42, y + 0.01, 0.80]])
    _rel(tally, [False], [[0.42, y, 0.79]])
    _end(tally, [0])
  out = eval_pick_place.summarise(tally, DT, 2)
  assert out["episodes_released"] == 2
  assert abs(out["release_within_3cm_xy"] - 0.5) < 1e-9
