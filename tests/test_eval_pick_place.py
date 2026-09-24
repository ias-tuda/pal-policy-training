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
