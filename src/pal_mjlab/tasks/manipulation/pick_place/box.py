"""Plan S10 (`pick-place-v1-objects-plan.md`, *S10 -- put the cube into a box*): ``GOAL``.

The cube is picked from W2 as in every pick-and-place run and put into an open box
(`scene.get_box_spec`, built to the same numbers) that sits where baseline B's target was:
(0.46, 0.08) +/- 2 cm table-relative, at yaw 0 +/- 5 deg, which the actor does not observe.
The goal is the box floor plus the cube's half-height (`mdp.BoxGoalCommand`); everything else
is the run's own reward variant (S10: `BS`'s RCTP) with these changes, all made here so that a
``GOAL`` code changes the same things on any variant:

* M6's release grade (`release`, `placed` and every other reader of the release latch) measures
  the let-go height above the **rim**, not above the cube's resting height
  (`events.surface_top_z` on the box's walls), and -- every code but ``box-ny`` -- multiplies a
  yaw grade G(cube yaw mod 90 deg, 10 deg) at the let-go;
* `box_moved`: -2 x min(box displacement / 2 cm, 1) per step -- pushing the box costs (plan S10
  step 0, item 6);
* the actor's goal comes through the box's tracker (the cube's lag, `LAG=FP`) and both tracked
  poses carry a shared 6 mm + own 4 mm bias (plan S10 item 5, Siwei 22:02); the cube's own 8 mm
  bias of `DR-obs` / baseline B is replaced by that split;
* `box_dropped` ends an episode whose box left the table;
* sensors for the box walls against the cube and against the hand, read by the eval, and by
  ``box-rim``'s penalty (-1 per step while the cube is held and either touches a wall).

``box-dr`` draws the box's inner width (7.8-8.2 cm), wall height (2.2-2.8 cm) and mass
(60-200 g) per world; the cube's mass (30-100 g) is `DR-object`'s, already in `DR-scene`.

Plan S10.1 (`pick-place-v1-objects-plan.md`, *S10.1*), two more selectors on any box code:

* ``TARGET=WC`` (an ``area``): the box is drawn from that area, table-relative, at a yaw in +/- 45 deg
  (every orientation of the 4-fold symmetric box), at least `BOX_MIN_SEPARATION` from the cube, and
  the cube is drawn only where such a box fits; M6's yaw grade (and the eval's let-go yaw) is then
  measured against the box, |wrap_90(psi_cube - psi_box)|. Without an area the box stays in
  baseline B's target box at 0 +/- 5 deg and the grade against the world's axes, as in S10 / DEP1.
* ``BOX_OBS=pose``: the actor's ``target_pos`` slot carries the box's own tracked position (its
  origin, the centre of its bottom face) instead of the goal point computed from it, and a
  ``box_yaw`` term, (cos 4 psi, sin 4 psi) of the box's tracked yaw relative to the base, is appended
  last (actor 40-D on `BS2`); the critic gets the same two, live (``box_yaw`` after PQ's inputs, so a
  W2 box checkpoint widens exactly on resume). ``goal`` (unset) is S10's goal point.

M8's ``h_clear`` "from the rim" (plan S10, BX-RIM) is not a change at this box: M8 measures the
tool point above the cube's top, and the 2.5 cm rim is 2.5 cm below a cube resting inside, so
the higher of the two -- what clearing both means -- is the cube's top, which M8 already uses.
"""

from __future__ import annotations

import dataclasses
import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from pal_mjlab.tasks.manipulation import mdp, scene
from pal_mjlab.tasks.manipulation.robot_cfg import GripperCfg
from pal_mjlab.tasks.manipulation.tabletop_env_cfg import (
  _DR_OBS_OBJECT_POS_NOISE,
  COMMAND,
  OBJECT,
  ROBOT,
  TABLE,
)

#: The box's scene entity name.
BOX = "box"
#: Its wall geoms (`scene.get_box_spec`): the rim is their top.
BOX_WALLS = scene.box_wall_geom_names(BOX)
#: The box's yaw at a reset: 0 +/- this, rad (plan S10 *The box*: fixed yaw, Siwei 13:20).
BOX_YAW = math.radians(5.0)

#: The tracker bias split (plan S9 item 3 / S10 item 5; 6 + 4 mm kept for now, Siwei 22:02):
#: per axis, uniform in +/- the shared part (one camera, every object) plus +/- each object's own.
TRACKER_SHARED_BIAS = 0.006
TRACKER_OWN_BIAS = 0.004

#: Sensors of the box walls against the cube and against the hand (the `arm_*_7_link` subtree).
BOX_WALL_OBJECT_SENSOR = "box_wall_object_contact"
BOX_WALL_HAND_SENSOR = "box_wall_hand_contact"

#: Plan S10.1: the box's yaw on an area, rad -- every orientation of a 4-fold symmetric box.
BOX_AREA_YAW = math.pi / 4
#: Plan S10.1: the least xy distance between the cube's and the box's centres on an area, m. The plan's
#: proposal, 16 cm: the box's half-diagonal at 45 deg (6.4 cm) + the cube's (3.5 cm) + an open hand past
#: the cube (~4 cm) + 2 cm; step 0 item 2 (the grasp clearance next to the box) sets it.
BOX_MIN_SEPARATION = 0.16
#: Plan S10.1's ``BOX_OBS=pose``: the box's tracked yaw, a per-episode bias (baseline B's cube yaw bias)
#: and a per-step jitter, both on the angle, rad.
BOX_YAW_BIAS = 0.03
BOX_YAW_NOISE = 0.02
#: What the actor observes of the box (``BOX_OBS``): the goal point from the box (S10) or its pose.
BOX_OBS_NAMES = ("goal", "pose")

#: BX-DR's ranges (plan S10's table).
BOX_DR_INNER = (0.078, 0.082)
BOX_DR_WALL_HEIGHT = (0.022, 0.028)
BOX_DR_MASS_KG = (0.06, 0.20)


@dataclasses.dataclass(frozen=True)
class BoxGoal:
  """What a ``GOAL`` box code adds to a pick-and-place cfg (`apply_box_goal`)."""

  yaw_std: float | None = math.radians(10.0)
  """M6's yaw grade at the let-go, G(|yaw mod 90 deg|, this), rad; None for none (BX-NY)."""

  rim_weight: float = 0.0
  """BX-RIM: per step while the cube is held and the hand or the cube touches a box wall."""

  dr: bool = False
  """BX-DR: the box's inner width, wall height and mass drawn per world."""

  moved_weight: float = -2.0
  """`box_moved`'s weight, at a displacement of `moved_scale` or more."""

  moved_scale: float = 0.02
  """The box displacement at which `box_moved` saturates, m."""

  feet: bool = True
  """The box stands on four corner feet (`scene.get_box_spec`); False is the flat floor every S10
  run of `0ba3f60` trained on, which bobs once rocked (S10 report §8). ``BOX_FEET=0`` selects it."""


#: The ``GOAL`` codes (plan S10's five experiments: BX, BX-FT -- BX resumed --, BX-NY, BX-RIM,
#: BX-DR). ``table`` is every run before S10: the goal on the table, no box in the scene.
BOX_GOALS: dict[str, BoxGoal | None] = {
  "table": None,
  "box": BoxGoal(),
  "box-ny": BoxGoal(yaw_std=None),
  "box-rim": BoxGoal(rim_weight=-1.0),
  "box-dr": BoxGoal(dr=True),
}

#: The reward terms that read the release latch (`mdp.observations._release_latch`): every one
#: of them must measure the release height from the same surface, the box's rim.
_LATCH_READERS = (
  mdp.released_at_goal_graded,
  mdp.object_placed_graded,
  mdp.set_down_placed,
  mdp.pad_contact_after_release,
  mdp.open_at_goal,
)
#: ... and the ones M6 grades, which take the yaw grade.
_GRADED = (mdp.released_at_goal_graded, mdp.object_placed_graded)


def rim_surface() -> SceneEntityCfg:
  """The surface the release height is measured from at the box: its walls' top, the rim."""
  return SceneEntityCfg(BOX, geom_names=BOX_WALLS)


def _wall_sensor(name: str, secondary: ContactMatch) -> ContactSensorCfg:
  return ContactSensorCfg(
    name=name,
    primary=ContactMatch(mode="geom", pattern=BOX_WALLS, entity=BOX),
    secondary=secondary,
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=1,
  )


def area_holds_the_box(area_x: tuple[float, float], area_y: tuple[float, float]) -> bool:
  """Whether every point of the area has a point of it at least `BOX_MIN_SEPARATION` away -- the
  centre's farthest corner, at the half-diagonal, is the worst case. Plan S10.1: WC (20 x 30 cm,
  18.0 cm) does; WCK (12 x 10 cm, 7.8 cm) does not."""
  return 0.5 * math.hypot(area_x[1] - area_x[0], area_y[1] - area_y[0]) >= BOX_MIN_SEPARATION


def apply_box_goal(
  cfg: ManagerBasedRlEnvCfg,
  gripper: GripperCfg,
  code: BoxGoal | None,
  target_xy: tuple[float, float],
  target_half_xy: float,
  table_top_z: float,
  threshold: float,
  area: tuple[tuple[float, float], tuple[float, float]] | None = None,
  pose_obs: bool = False,
) -> None:
  """Turn a built pick-and-place cfg's table goal into plan S10's box, in place; None leaves
  it exactly as it is. ``target_xy`` / ``target_half_xy`` are the baseline's target box (env
  frame, for the table at its nominal place) -- where the box is put down; ``threshold`` the
  goal tolerance the reward variant uses.

  Plan S10.1: ``area`` ((x lo, x hi), (y lo, y hi), env frame for the table at its nominal place)
  draws the box from that area at any yaw, clear of the cube, and grades the let-go's yaw against
  the box (the cube's own reset must be drawn where a box fits: `pick_place_env_cfg`'s
  ``goal_area``); ``pose_obs`` is ``BOX_OBS=pose`` (the module docstring). The critic's ``box_yaw``
  is appended by `append_critic_box_yaw`, after PQ's inputs."""
  if code is None:
    if area is not None or pose_obs:
      raise ValueError("a box area or BOX_OBS=pose needs a box GOAL")
    return
  cfg.scene.entities = {
    **cfg.scene.entities,
    BOX: scene.get_box_cfg(name=BOX, feet=code.feet),
  }
  table_cx, table_cy = scene.TABLE_POS[0], scene.TABLE_POS[1]
  if area is None:
    cfg.events["reset_box"] = EventTermCfg(
      func=mdp.reset_box_on_surface,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg(BOX),
        "surface_cfg": SceneEntityCfg(TABLE),
        "offset": (target_xy[0] - table_cx, target_xy[1] - table_cy),
        "half_range": (target_half_xy, target_half_xy),
        "yaw": BOX_YAW,
      },
    )
  else:
    (x0, x1), (y0, y1) = area
    if not area_holds_the_box((x0, x1), (y0, y1)):
      raise ValueError(
        f"the area x {area[0]} y {area[1]} cannot keep a box {BOX_MIN_SEPARATION} m from every "
        "cube in it (plan S10.1: WC can, WCK cannot)"
      )
    # After `reset_objects` (the dict keeps insertion order): it reads where the cube now is.
    cfg.events["reset_box"] = EventTermCfg(
      func=mdp.reset_box_in_area,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg(BOX),
        "surface_cfg": SceneEntityCfg(TABLE),
        "object_name": OBJECT,
        "area_x": (x0 - table_cx, x1 - table_cx),
        "area_y": (y0 - table_cy, y1 - table_cy),
        "yaw": BOX_AREA_YAW,
        "min_distance": BOX_MIN_SEPARATION,
      },
    )
  cfg.events["reset_tracker_bias"] = EventTermCfg(
    func=mdp.reset_tracker_bias,
    mode="reset",
    params={
      "shared": TRACKER_SHARED_BIAS,
      "own": TRACKER_OWN_BIAS,
      "keys": (OBJECT, BOX),
    },
  )
  cfg.commands[COMMAND] = mdp.BoxGoalCommandCfg(
    entity_name=OBJECT,
    box_name=BOX,
    floor=scene.BOX_FLOOR,
    difficulty="dynamic",
    resampling_time_range=(1e9, 1e9),
    success_threshold=threshold,
    debug_vis=True,
    yaw_reference=None if area is None else BOX,
  )

  # -- Observations: the goal through the box's tracker, the shared + own bias. --------------
  actor = cfg.observations["actor"].terms
  obj_term = actor["object_pos"]
  lag = obj_term.params.get("lag")
  if obj_term.func is not mdp.object_position_b:
    raise ValueError(
      "plan S10's tracker bias replaces the actor's object_pos term; it is "
      f"{obj_term.func.__name__} (occlusion is not built for the box)"
    )
  actor["object_pos"] = dataclasses.replace(
    obj_term,
    func=mdp.object_position_biased_b,
    params={**obj_term.params, "bias_key": OBJECT},
    noise=Unoise(n_min=-_DR_OBS_OBJECT_POS_NOISE, n_max=_DR_OBS_OBJECT_POS_NOISE),
  )
  if pose_obs:
    # Plan S10.1: the box itself, in the old goal's slot (a W2 box checkpoint resumes on it) and its
    # yaw appended last; the critic's slot gets the live box position.
    actor["target_pos"] = ObservationTermCfg(
      func=mdp.object_position_biased_b,
      params={"object_name": BOX, "bias_key": BOX, "lag": lag},
      noise=Unoise(n_min=-_DR_OBS_OBJECT_POS_NOISE, n_max=_DR_OBS_OBJECT_POS_NOISE),
    )
    actor["box_yaw"] = ObservationTermCfg(
      func=mdp.box_yaw_b,
      params={"box_name": BOX, "bias_key": BOX, "noise": BOX_YAW_NOISE, "lag": lag},
    )
    cfg.observations["critic"].terms["target_pos"] = ObservationTermCfg(
      func=mdp.object_position_b, params={"object_name": BOX}
    )
    cfg.events["reset_tracker_yaw_bias"] = EventTermCfg(
      func=mdp.reset_tracker_yaw_bias,
      mode="reset",
      params={"half": BOX_YAW_BIAS, "keys": (BOX,)},
    )
  else:
    goal_params = {"box_name": BOX, "object_name": OBJECT, "floor": scene.BOX_FLOOR}
    actor["target_pos"] = ObservationTermCfg(
      func=mdp.box_goal_position_tracked_b,
      params={**goal_params, "bias_key": BOX, "lag": lag},
      noise=Unoise(n_min=-_DR_OBS_OBJECT_POS_NOISE, n_max=_DR_OBS_OBJECT_POS_NOISE),
    )
    # The critic keeps the command's live goal (`goal_position_b`), which is the same point.

  # -- Rewards. ----------------------------------------------------------------------------
  for name, term in list(cfg.rewards.items()):
    if term.func not in _LATCH_READERS:
      continue
    params = {**term.params, "surface_cfg": rim_surface()}
    if term.func in _GRADED and code.yaw_std is not None:
      params["yaw_std"] = code.yaw_std
    cfg.rewards[name] = RewardTermCfg(func=term.func, weight=term.weight, params=params)
  cfg.rewards["box_moved"] = RewardTermCfg(
    func=mdp.box_displacement_penalty,
    weight=code.moved_weight,
    params={"command_name": COMMAND, "scale": code.moved_scale},
  )
  cfg.scene.sensors = (
    *cfg.scene.sensors,
    _wall_sensor(
      BOX_WALL_OBJECT_SENSOR, ContactMatch(mode="body", pattern=OBJECT, entity=OBJECT)
    ),
    _wall_sensor(
      BOX_WALL_HAND_SENSOR,
      ContactMatch(mode="subtree", pattern=gripper.hand_body, entity=ROBOT),
    ),
  )
  if code.rim_weight:
    cfg.rewards["rim_contact"] = RewardTermCfg(
      func=mdp.contact_while_held,
      weight=code.rim_weight,
      params={
        "grasp_sensor_name": gripper.contact_sensor_name,
        "sensor_names": (BOX_WALL_OBJECT_SENSOR, BOX_WALL_HAND_SENSOR),
      },
    )

  cfg.terminations["box_dropped"] = TerminationTermCfg(
    func=mdp.entity_below_height,
    params={"entity_name": BOX, "min_height": table_top_z - 0.1},
  )

  if code.dr:
    cfg.events["box_geometry"] = EventTermCfg(
      mode="startup",
      func=mdp.set_box_geometry,
      params={
        "asset_cfg": SceneEntityCfg(BOX),
        "walls": BOX_WALLS,
        "floor_name": f"{BOX}_floor",
        "wall": scene.BOX_WALL,
        "floor": scene.BOX_FLOOR,
        "inner_range": BOX_DR_INNER,
        "height_range": BOX_DR_WALL_HEIGHT,
        "feet": scene.box_foot_geom_names(BOX) if code.feet else (),
        "foot_inset": scene.BOX_FOOT_INSET,
      },
    )
    cfg.events["box_mass"] = EventTermCfg(
      mode="startup",
      func=dr.pseudo_inertia,
      params={
        "asset_cfg": SceneEntityCfg(BOX, body_names=(BOX,)),
        "alpha_range": tuple(math.log(m / scene.BOX_MASS) / 2 for m in BOX_DR_MASS_KG),
        "distribution": "uniform",
      },
    )


def append_critic_box_yaw(cfg: ManagerBasedRlEnvCfg) -> None:
  """Plan S10.1's ``BOX_OBS=pose``, the critic's half: the box's live yaw (cos 4 psi, sin 4 psi),
  appended after every other critic term -- PQ's included -- so that a resume from a W2 box
  checkpoint widens the critic by appended columns only (`runner.widen_input`)."""
  critic = cfg.observations["critic"]
  critic.terms = {
    **critic.terms,
    "box_yaw": ObservationTermCfg(func=mdp.box_yaw_b, params={"box_name": BOX}),
  }
