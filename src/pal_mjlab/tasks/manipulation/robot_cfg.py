"""The robot-side interface of the manipulation framework.

A manipulation task needs to ask a robot a small, fixed set of questions: where is the
grasp point, which two pads close on the object, which geoms are the friction surfaces,
which joint opens the jaw, and which joints may the policy command. Answering those in one
declarative object -- instead of writing ``arm_right_1_joint`` into a reward, or poking
``cfg.rewards["lift"].params["asset_cfg"].site_names`` from every task -- is what lets one
task config run on any robot and one robot serve any task.

To add a robot, write a single :class:`ManipulationRobotCfg` (see
``config/kangaroo/robot.py``). To add a task, write an env cfg that takes one and never
mentions a body part by name.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping

import mujoco
from mjlab.entity import EntityCfg


@dataclasses.dataclass(frozen=True)
class GripperCfg:
  """One parallel-jaw gripper, as manipulation terms see it."""

  name: str
  """Short identifier, e.g. ``"left"``. Used to name per-gripper sensors."""

  tcp_site: str
  """Site at the grasp centre, between the pads. Distance-to-object terms key off this."""

  pad_sites: tuple[str, str]
  """The two opposing pad sites. Their separation is the jaw opening; a grasp means the
  object lies between them. Order is irrelevant but must be exactly two."""

  pad_geom_regex: str
  """Collision geoms of the pads. Friction randomisation and the contact sensor use these."""

  finger_joint: str
  """The single actuated jaw joint. The rest of the linkage is passive."""

  open_pos: float
  """Joint position at which the jaw is fully open."""

  closed_pos: float
  """Joint position at which the jaw is fully closed."""

  hand_body: str
  """Body whose subtree is the hand, for the end-effector/ground collision sensor."""

  grasp_geom_regex: str
  """The two opposing pad geoms the grasp sensor watches -- exactly one per jaw side. A grasp
  is both of them touching the object. Narrower than ``pad_geom_regex`` on purpose: that one
  also names geoms deeper in the finger (friction surfaces), which a small object held at
  the pads never touches, and requiring *every* friction geom made the grasp unreachable."""

  def __post_init__(self) -> None:
    if len(self.pad_sites) != 2:
      raise ValueError(
        f"{self.name}: pad_sites must be exactly 2, got {self.pad_sites}"
      )

  @property
  def contact_sensor_name(self) -> str:
    return f"{self.name}_pad_contact"

  @property
  def ground_sensor_name(self) -> str:
    return f"{self.name}_hand_ground_contact"

  @property
  def surface_sensor_name(self) -> str:
    return f"{self.name}_hand_surface_contact"


@dataclasses.dataclass(frozen=True)
class KeepoutBoxCfg:
  """A box, fixed in one of the robot's own frames, that named links must stay out of.

  The trunk of a humanoid is the obvious instance: the arms can fold into the chest, and the
  trunk colliders only stop them once they touch. This is the volume they must keep clear,
  so the penalty starts *before* the contact. (S5.1 item 8, from Claudius's 82c4450.)
  """

  reference_body: str
  """Body whose frame the box is fixed in -- the trunk link, so it follows the torso."""

  body_name_regex: tuple[str, ...]
  """Links that must stay out, matched against body names. A link whose origin is inside
  the box at the start pose must not be listed: it would pay the full penalty from the
  first step of every episode (on Kangaroo, the shoulder link)."""

  center: tuple[float, float, float]
  """Box centre in ``reference_body``'s frame, metres."""

  half_extents: tuple[float, float, float]
  """Box half-extents, metres, taken from the trunk's collision geom."""

  margin: float = 0.05
  """Distance from the box surface at which the penalty starts, metres. It stands in for
  the watched links' own thickness -- the term scores link *origins* -- and must stay below
  the tightest clearance the start pose holds, or the penalty is on before the policy acts."""


@dataclasses.dataclass(frozen=True)
class ReachCfg:
  """How far from the robot an object can be and still be worth reaching for."""

  reference_body: str
  """Body the distance is measured from, horizontally. The base link."""

  max_distance: float
  """Planar distance beyond which an object is out of reach, metres."""


@dataclasses.dataclass(frozen=True)
class ManipulationRobotCfg:
  """A robot, described in the terms a manipulation task needs."""

  entity_cfg_fn: Callable[[], EntityCfg]
  """Builds the robot entity. Called once per env cfg construction."""

  grippers: tuple[GripperCfg, ...]
  """All grippers, in a stable order. ``default_gripper`` picks the one a single-arm task
  uses; a bimanual task may use several."""

  action_joint_regex: tuple[str, ...]
  """Actuator name patterns the policy may command. Deliberately narrower than ``.*``: on a
  humanoid the legs are held by a separate balance controller and must not be in the action
  space, or the exported ONNX will not match what ``pal_policy_deployer`` drives."""

  action_scale: float | dict[str, float]
  """Per-actuator action scale, as ``JointPositionActionCfg`` takes it."""

  viewer_body: str
  """Body the viewer anchors on."""

  fixed_base: bool = False
  """Weld the robot's root body to the world, holding it at its initial pose. Set for
  humanoids doing tabletop work: balance is out of scope for a manipulation policy and a
  falling robot drowns the reward signal."""

  root_free_joint: str = "origin"
  """Name of the root free joint. Used to locate the root body when ``fixed_base`` is set."""

  action_clip: dict[str, tuple[float, float]] | None = None
  """Per-actuator clamp on the *processed* joint target, as ``JointPositionActionCfg.clip``
  takes it. None leaves targets unclamped. mjlab builds position actuators with
  ``ctrllimited=False``, so a target past a joint limit is not clamped by the simulator --
  the actuator pushes the joint into its limit at up to its full effort."""

  trunk_keepout: KeepoutBoxCfg | None = None
  """Clearance volume around the trunk, penalised as ``trunk_clearance``; ``None`` for none."""

  joint_velocity_limits: Mapping[str, float] | None = None
  """Hardware joint speed ceilings in rad/s, keyed by actuator-name regex, or ``None``.

  `make_tabletop_env_cfg` turns them into a per-control-step slew limit on the commanded
  target (`mdp.RateLimitedJointPositionAction`). Taken from the robot's URDF, not tuned: a
  target that moves faster than the servo can follow trains against an actuator that does
  not exist. (S5.1 item 4, Claudius's 602cd9d.)"""

  reach: ReachCfg | None = None
  """Usable workspace radius; an object beyond it ends the episode as ``out_of_reach``.
  ``None`` never calls an object unreachable."""

  def __post_init__(self) -> None:
    if not self.grippers:
      raise ValueError("at least one gripper is required")
    names = [g.name for g in self.grippers]
    if len(set(names)) != len(names):
      raise ValueError(f"gripper names must be unique, got {names}")

  def gripper(self, name: str | None = None) -> GripperCfg:
    """The named gripper, or the first one if ``name`` is None."""
    if name is None:
      return self.grippers[0]
    for g in self.grippers:
      if g.name == name:
        return g
    raise KeyError(f"no gripper {name!r}; have {[g.name for g in self.grippers]}")

  def build_entity_cfg(self) -> EntityCfg:
    """The robot entity, with ``fixed_base`` applied."""
    cfg = self.entity_cfg_fn()
    if not self.fixed_base:
      return cfg
    inner = cfg.spec_fn
    assert inner is not None, "fixed_base requires a spec_fn"
    root_joint = self.root_free_joint

    def spec_fn() -> mujoco.MjSpec:
      spec = inner()
      root = next((j for j in spec.joints if j.name == root_joint), None)
      if root is None:
        raise ValueError(
          f"fixed_base: no free joint named {root_joint!r} in the robot spec; "
          f"have {[j.name for j in spec.joints][:5]}..."
        )
      spec.delete(root)
      return spec

    # With the free joint gone, mjlab anchors the entity on a mocap body that already
    # carries the spawn pose, and ``init_state.pos`` would then be applied a second time on
    # top of it -- Kangaroo ended up at 1.845 m (0.945 mocap + 0.9 init), feet dangling
    # 0.95 m above the floor. Zero the offset and let the mocap place the robot.
    init_state = dataclasses.replace(cfg.init_state, pos=(0.0, 0.0, 0.0))
    return dataclasses.replace(cfg, spec_fn=spec_fn, init_state=init_state)
