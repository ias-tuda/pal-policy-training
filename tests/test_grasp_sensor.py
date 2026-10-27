"""The grasp signal: both pads touching the object, read correctly off the contact sensor.

Until 2026-09-23 `_grasped_bool` applied `.all()` to the raw `found` tensor. mjlab lays that
out as `(B, P * num_slots)`, primary-major, and the sensor watched four geoms per gripper
(fingertip and inner finger, both sides) with two slots each -- so a "grasp" needed eight
simultaneous contacts, including two deep in the fingers that a 5 cm cube held at the pads
never reaches. Measured in S5.0: both fingertips on the cube at ~60 N, the cube lifted and
carried, and `grasped` False throughout.
"""

import types

import torch
from pal_mjlab.tasks.manipulation.mdp.observations import _grasped_bool
from pal_mjlab.tasks.manipulation.robots import KANGAROO_GRIPPERS

SLOTS = 2


def _env(found: list[list[float]]):
  """The smallest thing `_grasped_bool` reads: env.scene[name].data.found and cfg.num_slots."""
  sensor = types.SimpleNamespace(
    data=types.SimpleNamespace(found=torch.tensor(found)),
    cfg=types.SimpleNamespace(num_slots=SLOTS),
  )
  return types.SimpleNamespace(scene={"pads": sensor})


def test_both_pads_in_contact_is_a_grasp_whichever_slot_holds_it():
  # pad 0 in its first slot only, pad 1 in its second slot only
  assert _grasped_bool(_env([[1, 0, 0, 1]]), "pads").tolist() == [True]


def test_one_pad_is_not_a_grasp_however_many_contacts_it_has():
  assert _grasped_bool(_env([[2, 2, 0, 0]]), "pads").tolist() == [False]
  assert _grasped_bool(_env([[0, 0, 3, 3]]), "pads").tolist() == [False]


def test_no_contact_is_not_a_grasp():
  assert _grasped_bool(_env([[0, 0, 0, 0]]), "pads").tolist() == [False]


def test_batched():
  found = [[1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]]
  assert _grasped_bool(_env(found), "pads").tolist() == [True, False, False]


def test_the_grasp_sensor_watches_exactly_one_pad_per_jaw_side():
  """The sensor's geoms, matched against the robot's actual spec rather than assumed."""
  import re

  from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import (
    get_kangaroo_grippers_tabletop_spec,
  )

  spec = (
    get_kangaroo_grippers_tabletop_spec()
  )  # kept alive: its geoms are views into it
  names = [g.name for g in spec.geoms]
  for grip in KANGAROO_GRIPPERS.grippers:
    matched = sorted(n for n in names if re.fullmatch(grip.grasp_geom_regex, n))
    assert matched == [
      f"gripper_{grip.name}_fingertip_left_collision",
      f"gripper_{grip.name}_fingertip_right_collision",
    ], matched
