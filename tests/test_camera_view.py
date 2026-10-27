"""The segment distance behind ``mdp.camera_view_blocked``."""

import pytest
import torch
from pal_mjlab.tasks.manipulation.mdp.rewards import _segment_distance


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
  t = lambda v: torch.tensor([v], dtype=torch.float64)  # noqa: E731
  got = _segment_distance(t(p0), t(p1), t(q0), t(q1))
  assert got.item() == pytest.approx(expected, abs=1e-9)
