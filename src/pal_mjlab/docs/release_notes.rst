.. _release_notes:

Release Notes
=============

This page summarizes what has changed in ``pal_mjlab`` between releases, in
plain language, with a link back to the pull request (PR) that introduced
each change. If you are new to the project, this is the fastest way to
catch up on what has moved since the last tagged release.

.. contents:: Table of Contents
   :local:
   :depth: 2

Unreleased — since ``v1.0.0``
------------------------------

The changes below have landed on ``main`` since the ``v1.0.0`` tag
(2026-03-06) and are not yet part of a tagged release.

New Features & Tasks
^^^^^^^^^^^^^^^^^^^^^

- **Changed: the Kangaroo jaw is stable in the tabletop manipulation tasks.**
  Driven over its full range, the gripper's passive linkage joints vibrated at
  up to ~2000 rad/s until the simulation diverged, and at the old jaw gain a
  closing jaw squeezed a cube hard enough to buckle the linkage and throw the
  cube out. The tabletop robot model now gives the ten passive gripper joints
  more armature (1e-4 to 5e-3) and the jaw a gentler gain (stiffness 20 to 8,
  about 36 N per pad on a 5 cm cube). In ``Mjlab-PickPlace-Pal-Kangaroo-Grippers``
  the jaw's "closed" target is now 0.78 rad, the furthest the linkage can
  actually close, instead of the joint's 0.95 rad limit. The open gap and the
  pad-gap-against-angle curve are unchanged. Affects ``Mjlab-Lift-``,
  ``Mjlab-Stack-`` and ``Mjlab-PickPlace-Pal-Kangaroo-Grippers``; the velocity
  and reaching Grippers tasks are unchanged. Results trained before this change
  were on a different jaw.

- **Fixed: the grasp signal of every tabletop manipulation task could not
  register a grasp.** The grasp sensor watched four geoms per gripper
  (fingertip and inner finger, both sides) and ``is_grasped`` required every
  contact slot of all four at once. A 5 cm cube held at the pads touches only
  the two fingertips, so ``grasped`` stayed False while the cube was lifted
  and carried. It also meant a held cube counted as "released", so a
  ``success`` termination could fire mid-carry. A grasp is now both fingertip
  pads in contact (new ``GripperCfg.grasp_geom_regex``). Affects
  ``Mjlab-Lift-``, ``Mjlab-Stack-`` and ``Mjlab-PickPlace-Pal-Kangaroo-Grippers``;
  the observation's shape is unchanged.

- **Changed: ``Mjlab-PickPlace-Pal-Kangaroo-Grippers`` action scales.** The
  action is an absolute joint target, and the inherited scales let the policy
  command the jaw to only 22 % of its range and move the grasp point about
  12 cm, so the task could not be done. The arm scales are now sized from IK
  over the task's own waypoints (0.20-1.05 rad per joint). The jaw scale is its
  full 0.95 rad range, with the target clipped at the open limit through the
  new ``ManipulationRobotCfg.action_clip``. Lift and Stack are unchanged. A
  scripted pick-and-place through the policy's action interface is now a test
  (``tests/test_pick_place_scripted_feasibility.py``).

- **Fixed: the goal of every tabletop manipulation task was unreachable with
  more than one environment.** ``Mjlab-Lift-``, ``Mjlab-Stack-`` and
  ``Mjlab-PickPlace-Pal-Kangaroo-Grippers`` inherited the locomotion default of
  ``env_spacing=2.5``. mjlab simulates environments as batched worlds with
  identical geometry, but a scene carrying a terrain still lays ``env_origins``
  out on a grid of that spacing. The object placement event reads the table's
  position out of physics and ignores the grid, while the lifting command adds
  it to its target, so the goal ended up 1.6-1.9 m from the cube in every
  environment except the first. At one environment the grid is zero and nothing
  is wrong, which is why it survived every smoke test. The tabletop scene now
  uses ``env_spacing=0.0``.

- **Reward variants for the pick-and-place task, selected by name.**
  ``make_pick_place_env_cfg(reward=...)`` takes ``R-A`` (the framework's own
  reward, and the default), ``R-B`` (sparse: reach, goal, success only), ``R-C``
  (phase-gated, with a bounded lift height and a transport term that only pays
  once the object is off the table), ``R-D`` (eight bounding-box keypoints under
  a logistic kernel) or ``R-E`` (a per-step time cost with a larger terminal
  bonus and a shorter horizon). Every variant carries the same regularisers, so
  a comparison between them is a comparison of the thing that differs. Adds a
  hand-versus-table contact sensor and the reward terms the variants need.

- **A named domain-randomisation ladder for the tabletop manipulation tasks.**
  ``make_tabletop_env_cfg`` takes a ``dr_level`` (and the pick-and-place factory a
  ``dr``), one of ``DR0``, ``DR-object``, ``DR-contact``, ``DR-robot``, ``DR-obs``
  or ``DR-scene``, each adding to the one below: object friction and density,
  then gripper pad friction, then actuator gains and joint friction, then
  observation noise and encoder bias, then table height and a wider spawn box.
  The point is that a training run's config records its randomisation level
  instead of a report claiming it. Asking for an unknown level is an error rather
  than a silent fallback. Leaving ``dr_level`` unset keeps the previous behaviour
  exactly, so lift and stack are unchanged. Also adds
  ``mdp.SurfaceOffsetCommand``, a goal that sits on a surface whose height is read
  from physics -- needed once the table itself moves, since a goal with a fixed
  height is wrong by more than the success threshold.

- **New pick-and-place task for the Kangaroo:**
  ``Mjlab-PickPlace-Pal-Kangaroo-Grippers``. The robot takes a 5 cm cube from a
  lightly noised spot on the table and puts it down on a fixed target 20 cm
  away, using the left arm only -- eight actions, seven arm joints and the jaw --
  with the right arm held at its nominal pose and left out of the observation
  vector. Built on the same ``make_tabletop_env_cfg`` as lift and stack, with
  stack's release and success shaping and no second cube. The spawn and target
  are not arbitrary: they are the two points in the measured top-down grasp map
  of the table that are 20 cm apart with the most clearance to the edge of the
  reachable region. Domain randomisation is selected by name (``dr="DR0"``, the
  only level built so far), so a run's config records it.

- **The tabletop manipulation tasks now hold the pose they start in.** Two fixes.
  First, a ``hold_nominal_targets`` reset event commands every joint to its
  configured default: mjlab zeroes all joint targets on reset and only the joints
  an action term names get one written back, so the legs, the telescopic leg drive
  and the pelvis were commanded to zero for the whole episode. Without it
  ``leg_*_length_joint`` collapsed from 0.600 m to its 0.128 m mechanical minimum
  within 0.2 s of every reset. Second, the torso, both arms and both grippers are
  now gravity-compensated in simulation, leaving PAL's impedance gains untouched;
  those gains need a tenth of the torque gravity actually applies, and without
  compensation the left shoulder sagged 0.4 rad. Both apply only to
  ``Mjlab-Lift-Pal-Kangaroo-Grippers`` and ``Mjlab-Stack-Pal-Kangaroo-Grippers``;
  the velocity and reaching tasks are unaffected.

- **The tabletop manipulation tasks get a collision model above the elbow, and a
  start pose over the table.** ``Mjlab-Lift-Pal-Kangaroo-Grippers`` and
  ``Mjlab-Stack-Pal-Kangaroo-Grippers`` now instantiate a new
  ``get_kangaroo_grippers_tabletop_robot_cfg``. It adds collision boxes for the
  torso and for the upper arm, forearm and elbow of both arms, sized from the
  link meshes, and replaces the shipped ``arm_*_4_collision`` capsule, which
  enclosed arm 5-7 rather than arm 4. Before this the torso and the whole upper
  arm passed through the table. The robot also starts with the left elbow square
  and the open jaw above the table rather than hanging at its side. The velocity
  and reaching tasks are unaffected: they keep
  ``get_kangaroo_grippers_robot_cfg``, whose collision set and initial pose are
  unchanged.

- **Removing custom height targets and weights for foot_swing_height and
  foot_clearance rewards in Kangaroo rough velocity task.** Removing values
  optimized for rough terrains consisting mainly of structured stairs (where
  high clearance must be encouraged). Experimentally, it has been proven
  that they slow down the training process while not improving any of the metrics.
  (`#111 <https://github.com/pal-robotics/pal_mjlab/pull/111>`_)

- **New rough-terrain velocity tasks for the Kangaroo (full and lower
  body).** Adds ``Mjlab-Velocity-Rough-Pal-Kangaroo`` and
  ``Mjlab-Velocity-Rough-Pal-Kangaroo-Lower-Body`` on top of the baseline
  config: a PAL terrain generator (flat, pebbles, random boxes and
  inverted pyramid stairs at 0.3/0.4/0.5 m step widths), a critic-only
  1.2x0.6 m height scan. The terrain curriculum is disabled in play mode.
  (`#109 <https://github.com/pal-robotics/pal_mjlab/pull/109>`_)

- **Lower-body-specific noise and reward-weight tuning.** The IMU
  observation noise (``imu_projected_gravity``, ``base_lin_acc``) and the
  ``upright`` reward weight are now overridden specifically for the
  lower-body flat env config, tuned to match the microstrain IMU used on
  that robot, instead of changing the shared baseline config's values.
  (`#101 <https://github.com/pal-robotics/pal_mjlab/pull/101>`_)

- **Actuator parameters are now easy to tune.** Motor gains, armature,
  friction and effort limits for each Kangaroo actuator were previously
  hard-coded inline; they are now grouped into named parameter blocks in
  ``kangaroo_constants.py`` so you can change one actuator's behaviour
  without hunting through the file.
  (`#108 <https://github.com/pal-robotics/pal_mjlab/pull/108>`_)

- **Foot collision avoidance capsules added to the 4-DoF Kangaroo model.**
  Extra collision capsules around the feet stop the legs/feet from
  interpenetrating during testing, adding this reduces the foot stepping
  over each other.
  (`#104 <https://github.com/pal-robotics/pal_mjlab/pull/104>`_)

- **Dual-band velocity command sampling.** The velocity command generator
  can now avoid sampling commands whose magnitude falls in a "dead" low
  range, so training does not waste time near-zero velocity commands —
  particularly useful for rough-terrain/stairs training.
  Note: Command sampler is added but not yet used in the tasks
  (`#85 <https://github.com/pal-robotics/pal_mjlab/pull/85>`_)

- **New decoupled angular-velocity rewards.** Two new reward terms,
  ``track_body_ang_vel_z_exp`` (yaw tracking) and
  ``body_ang_vel_xy_l2_penalty`` (roll/pitch penalty), are available as a
  replacement for mjlab's combined ``track_angular_velocity`` /
  ``body_angular_velocity_penalty`` pair. Splitting them fixes a
  frame-mismatch bug (yaw used the body frame, the penalty used the world
  frame) and stops the roll/pitch restoring gradient from vanishing
  whenever yaw error is large — both terms can now be tuned independently.
  Note: The rewards are added, but are not used within the configurations yet.
  (`#81 <https://github.com/pal-robotics/pal_mjlab/pull/81>`_)

- **New tracking task for the Kangaroo lower body (+ 5-DoF variant).**
  Adds a motion-tracking task on top of the existing velocity-tracking
  task.
  (`#91 <https://github.com/pal-robotics/pal_mjlab/pull/91>`_)

- **New velocity-tracking task for the Kangaroo lower body.** The basic
  velocity-tracking environment for the lower-body robot.
  (`#76 <https://github.com/pal-robotics/pal_mjlab/pull/76>`_)

- **Reward set adapted for lower-body flat-ground locomotion.** Minimal
  reward changes needed to get the lower-body robot walking on flat
  ground.
  (`#87 <https://github.com/pal-robotics/pal_mjlab/pull/87>`_)

Model Updates
^^^^^^^^^^^^^

- Updated the Kangaroo 7-DoF and hands MuJoCo models with proper masses
  and inertias.
  (`#107 <https://github.com/pal-robotics/pal_mjlab/pull/107>`_)
- Updated the Kangaroo 4-DoF MuJoCo model with proper masses and inertias.
  (`#106 <https://github.com/pal-robotics/pal_mjlab/pull/106>`_)
- Replaced the TALOS collision meshes with capsule proxies, which are
  cheaper to simulate and collide.
  (`#102 <https://github.com/pal-robotics/pal_mjlab/pull/102>`_)
- Updated the Kangaroo lower-body model.
  (`#84 <https://github.com/pal-robotics/pal_mjlab/pull/84>`_)
- Set the density of the foot capsules to zero, so they no longer affect
  the robot's mass/inertia and act as pure collision geometry.
  (`#79 <https://github.com/pal-robotics/pal_mjlab/pull/79>`_)

Bug Fixes & Reliability
^^^^^^^^^^^^^^^^^^^^^^^^

- **Fixed a CUDA illegal-memory-access crash on reset with mjlab 1.5.1.**
  Training would crash with a CUDA illegal memory access error when
  environments reset. Worked around by pinning a specific mjlab commit;
  see also the memory note on this issue for how it was diagnosed.
  (`#94 <https://github.com/pal-robotics/pal_mjlab/pull/94>`_)

- **Improved policy repeatability across training runs.** Reduces the
  joint-velocity and IMU-projected-gravity observation noise so that
  repeated training runs converge to more similar behaviour, making it
  easier to compare experiments.
  (`#75 <https://github.com/pal-robotics/pal_mjlab/pull/75>`_)

Dependency Updates
^^^^^^^^^^^^^^^^^^

``pal_mjlab`` tracks ``mjlab`` closely. If you hit an environment or
dependency issue, check which of these bumps might be relevant first.

- Updated ``mjlab`` to ``1.6.0``.
  (`#103 <https://github.com/pal-robotics/pal_mjlab/pull/103>`_)
- Updated ``mjlab`` to ``1.5.3``.
  (`#99 <https://github.com/pal-robotics/pal_mjlab/pull/99>`_)
- Updated ``mjlab`` to ``1.5.2``.
  (`#95 <https://github.com/pal-robotics/pal_mjlab/pull/95>`_)
- Updated ``mjlab`` to ``1.5.1`` (superseded by the CUDA fix in
  `#94 <https://github.com/pal-robotics/pal_mjlab/pull/94>`_).
  (`#92 <https://github.com/pal-robotics/pal_mjlab/pull/92>`_)
- Updated ``mjlab`` to ``1.5.0``.
  (`#80 <https://github.com/pal-robotics/pal_mjlab/pull/80>`_)
- Updated ``mjlab`` to ``1.4.0``.
  (`#77 <https://github.com/pal-robotics/pal_mjlab/pull/77>`_)
- Updated ``mjlab`` to ``1.3.0``.
  (`#73 <https://github.com/pal-robotics/pal_mjlab/pull/73>`_)
- Updated ``mjlab`` to ``1.2.0``.
  (`#70 <https://github.com/pal-robotics/pal_mjlab/pull/70>`_)

Breaking Changes
^^^^^^^^^^^^^^^^

- **The Hands and Grippers rough tasks are no longer registered.**
  ``Mjlab-Velocity-Rough-Pal-Kangaroo-Hands`` and
  ``Mjlab-Velocity-Rough-Pal-Kangaroo-Grippers`` are commented out while
  the new rough config settles.
  (`#109 <https://github.com/pal-robotics/pal_mjlab/pull/109>`_)

- **Renamed** ``pal_kangaroo_rough_env_cfg`` **to**
  ``pal_kangaroo_baseline_env_cfg``. If you reference this config directly
  (rather than through a registered Gym task id), update the import.
  (`#86 <https://github.com/pal-robotics/pal_mjlab/pull/86>`_)

Documentation & CI
^^^^^^^^^^^^^^^^^^

- Added the first version of the ``pal_mjlab`` documentation (the site
  you are reading now).
  (`#82 <https://github.com/pal-robotics/pal_mjlab/pull/82>`_)
- Added a reward-testing CI workflow to catch regressions in reward
  functions.
  (`#78 <https://github.com/pal-robotics/pal_mjlab/pull/78>`_)
- Updated the CI versions.
  (`#83 <https://github.com/pal-robotics/pal_mjlab/pull/83>`_)

Maintenance
^^^^^^^^^^^

- Changed the default checkpoint ``save_interval`` to 500 iterations.
  (`#72 <https://github.com/pal-robotics/pal_mjlab/pull/72>`_)
- Removed trailing whitespace across the codebase.
  (`#93 <https://github.com/pal-robotics/pal_mjlab/pull/93>`_)

``v1.0.0``
----------

Initial tagged release of ``pal_mjlab``.
