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

  joint_damping_range: Mapping[str, tuple[float, float]] | None = None
  """Absolute ``dof_damping`` ranges keyed by joint-name regex, sampled per env at startup.
  ``None`` disables the randomisation.

  Absolute for the same reason as ``joint_friction_range``, and it is a distinct quantity
  from the ``damping`` in ``_pal_impedance_params``: that one is the actuator's derivative
  gain (randomised separately by ``dr.pd_gains``), whereas this is joint-level viscous
  damping. The arms carry none of the latter at all, so a multiplicative range would be a
  silent no-op on exactly the joints that matter.
  """

  joint_friction_range: Mapping[str, tuple[float, float]] | None = None
  """Absolute ``dof_frictionloss`` ranges keyed by joint-name regex, sampled per env at
  startup. ``None`` disables the randomisation.

  Absolute rather than scaled because the arm actuators are built by
  ``_calc_actuator_params``, which sets armature/stiffness/damping but *no* frictionloss:
  their default is 0, so a multiplicative range would be a no-op and the arms would keep
  the frictionless joints they have today. Ranges are per joint group because the stiction
  that is realistic on a 50 N.m arm joint would seize the gripper four-bar.
  """

  fixed_base: bool = False
  """Weld the robot's root body to the world, holding it at its initial pose. Set for
  humanoids doing tabletop work: balance is out of scope for a manipulation policy and a
  falling robot drowns the reward signal."""

  root_free_joint: str = "origin"
  """Name of the root free joint. Used to locate the root body when ``fixed_base`` is set."""

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
