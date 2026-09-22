# Project memory — Oppsie-for-Kangaroo

*Long-term project state. Read before starting work; update when a decision, milestone or
result lands. Not a conversation log — replace outdated entries rather than appending.
Keep under ~500 lines.*
*Last updated: 2026-09-22 (pick-and-place v0 stage S1 complete; stage report in
`../docs/reports/pick-place-v0/s1-report.md`).*

## 1. What we are building

TU Darmstadt fork of PAL Robotics' `pal_mjlab`: KANGAROO / Talos / TIAGo Pro in
[mjlab](https://github.com/mujocolab/mjlab) (Isaac Lab API on MuJoCo Warp), used as the code base
for the **Oppsie** work on the KANGAROO humanoid — robot models, task/env configs and the PPO
training stack.

**Open: what Oppsie is.** The method, its central premise and what a finished result looks like
are not written down in this repo. Fill this section — and the header of `CLAUDE.md` — the first
time it is stated. Do not infer it from the code.

Plans live in `oppsie-data/docs/` beside this repo, outside git (`docs/roadmap.md` will be the top of the tree; only the pick-and-place plan exists so far).

## 2. Where things stand

| Item | Status |
|---|---|
| Fork restarted at upstream `pal_mjlab` main `a461dc3` | ✅ commit `3b05cf6` |
| Gripper actuators on the KANGAROO grippers model | ✅ commit `7a0c7fb` (kangaroo_constants.py, kangaroo_grippers.xml) |
| Manipulation training framework (lift + stack, tabletop scene, MDP terms) | 🟡 commit `f33baec` — **first exercised 2026-09-21**: `Mjlab-Lift-Pal-Kangaroo-Grippers` builds and runs 500 steps on CPU under zero and random actions, no NaN, no auto-reset. Two defects found, see §5 |
| Pick-and-place v0 S1 (scene, start pose, reach map, collision model) | ✅ complete — `../docs/reports/pick-place-v0/s1-report.md` |
| Pick-and-place v0 S2-S5 | ❌ not started |
| Velocity, tracking, reaching tasks (inherited from upstream) | ✅ registered; no result measured *in this fork* |
| Any training run | ❌ none recorded |
| `../docs/` (roadmap, decisions, reports, status summary) | 🟡 only `pick-place-v0-plan.md` so far |
| Per-run cost on any host | ❌ not measured |

## 3. Design decisions taken

- **Fork rather than vendor.** `mjlab` stays an installed dependency (`mjlab>=1.6.0`, resolved by
  `uv.lock`); only `pal_mjlab` is forked, restarted cleanly at upstream `a461dc3` rather than
  carrying upstream history. Consequence: changes must survive an upstream merge — prefer new
  packages and config factories over edits to inherited files.
- **`ManipulationOnPolicyRunner` for the manipulation tasks**, not the velocity runner: it is the
  one that exports ONNX with the metadata `pal_policy_inferer` reads
  (`tasks/manipulation/lift/kangaroo/__init__.py`).
- **Branch policy: work on `siwei`, always commit, push only when asked** (user, 2026-09-21).
- **uv, not conda,** for this project's environment — `make sync`, `uv run`, with mutually
  exclusive `cpu` / `cu128` torch extras.
- **S1 tooling is kinematics-only and lives in a new top-level `scripts/`**
  (`scene_model.py`, `reach_map.py`, `nominal_pose.py`, `render_scene.py`,
  `fit_link_colliders.py`). They compile the task's scene with `Scene(cfg).compile()` instead
  of constructing `ManagerBasedRlEnv`: 5 s against ~60 s of warp kernel build, no GPU, and
  the same model with the task's own solver settings applied.
- **Tabletop manipulation gets its own robot cfg** (`get_kangaroo_grippers_tabletop_robot_cfg`,
  2026-09-22): collision boxes for the torso and for arm 2-4 of both arms, PAL's oversized
  `arm_*_4_collision` capsule replaced, four contact excludes, and a start pose with the left
  elbow at pi/2 and the open jaw over the table. The velocity and reaching tasks keep
  `get_kangaroo_grippers_robot_cfg` unchanged — that boundary is what
  `tests/test_kangaroo_collision_model.py` guards.
- **Pick-and-place v0 geometry (user decision, 2026-09-22):** object spawn centre
  (0.42, 0.27) +/- 2 cm, fixed target (0.42, 0.07, 0.785), 20 cm apart. Spawn clearance
  14.0 cm, target 4.5 cm — the target is 0.5 cm under the plan's 5 cm rule after S1.4;
  (0.36, 0.07) would give 8.2 cm at the same separation if it is ever revisited.

## 4. Load-bearing facts

- `origin` = `git@git.ias.informatik.tu-darmstadt.de:talos/TU_Darmstadt/pal_mjlab_kangaroo_talos.git`.
  Upstream `github.com/pal-robotics/pal_mjlab` is a *different* remote; `README.md` describes
  upstream, not this checkout.
- Branches on `origin` (2026-09-21): `main`, `KangarooFullEnv`, `my-new-branch`, `rack-hang`,
  `talos-reaching-real-nogripper-table-cubes`.
- **KANGAROO has four models here**: `kangaroo` (4 DoF/arm, fake forearm), `kangaroo_lower_body`,
  `kangaroo_hands` (5 DoF/arm, Seed Robotics), `kangaroo_grippers` (7 DoF/arm). A value tuned for
  one is inherited silently by the others.
- **Femur and knee joints are equality-driven** by the leg four-bar, and gripper finger joints by
  the jaw four-bar; `REGEX_ALL_ACTUATED_JOINTS` in `kangaroo_constants.py` excludes them, because
  the posture reward's per-joint std tables only cover legs, pelvis and arms.
- **Task ids are inconsistently spelled**: `Mjlab-Tracking-Flat-Pal-Kangaroo_Lower_Body`
  (underscores) vs `Mjlab-Velocity-Flat-Pal-Kangaroo-Lower-Body` (hyphens). Copy ids from
  `grep -rh 'task_id="' src/pal_mjlab/tasks`.
- Tasks self-register: each `tasks/<family>/<robot>/__init__.py` calls `register_mjlab_task`, and
  `tasks/__init__.py` auto-imports packages via `import_packages` with blacklist
  `["utils", ".mdp"]`. Entry point `mjlab.tasks -> pal_mjlab.tasks` in `pyproject.toml`.
- **Python floor is 3.10** (CI matrix 3.10–3.13) though `.python-version` is 3.13.
  Ruff with `indent-width = 2`, rules `E4,E7,E9,F,I,B`. CI also runs `pre-commit --all-files`.
- **`UV_FROZEN=1` in CI** — any dependency change must commit an updated `uv.lock`.
- `pyproject.toml` overrides `pillow>=12.2.0` to lift moviepy's precautionary `pillow<12` cap,
  which conflicts with mjviser's requirement. Do not "fix" it by downgrading pillow.
- `mujoco_warp_io.patch` (14 lines) wraps `warp_util.check_toolkit_driver()` in
  `mujoco_warp/_src/io.py` in try/except, downgrading a CUDA toolkit/driver mismatch to a warning.
  **Not applied automatically** — its applied/unapplied state is part of a result's provenance.
  The upstream commit it applies to is not recorded.
- `.github/workflows/ci.yaml` is GitHub Actions while `origin` is GitLab — **whether CI actually
  runs on pushes here is unverified.** Run `make test` and `make format` locally.
- Committed motions: `motions/kang_walk.csv`, `motions/talos_walk.csv`. NPZ outputs, `logs`,
  `wandb`, `artifacts/`, `motion.mp4` are gitignored.
- Motion conversion: `uv run -m pal_mjlab.scripts.csv_to_npz --input-file motions/<m>.csv
  --output-name <m> --input-fps 30 --output-fps 50 --robot-name <robot> --render True`.
- **The console script is `list-envs`, not `list_envs`.**
- **`MANIPULATION_COLLISION` filters out colliders that exist in the XML.** The grippers
  variant collides on feet, arm 5-7 and the gripper only — 43 of 110 geoms in the lift scene.
  `pelvis_2_collision` (torso capsule) and `arm_*_4_collision` are in `kangaroo_grippers.xml`
  and disabled by the policy, not missing. Arm 1-3 genuinely have no collider.
- **Collision geometry can only be changed in the `MjSpec`, before compile.** MuJoCo drops
  non-colliding geoms from the BVH, so setting `contype` on a compiled `MjModel` is a silent
  no-op. In mjlab: edit the entity's `CollisionCfg` before `Scene.compile()`.
- **Kangaroo left-arm workspace at table height (measured 2026-09-22, 48 IK seeds per cell):**
  top-down grasp of a cube on the table is possible on **353 of 1200** 2 cm cells with the
  S1.4 collision set (445 without it), a band at **x ∈ [0.26, 0.58], y ∈ [−0.29, 0.39]**. The
  table's far 27 cm in x is unreachable. The gripper pads reach **25.6 mm beyond `tcp_left`**,
  so a top-down grasp puts the TCP at z = 0.796, not at the cube centre 0.785.
- **A reach map is only as good as its IK restart count.** At 8 restarts per cell the same
  measurement returns 58 cells where 48 restarts return 88: it was measuring the solver. Quote
  `--seeds` with any workspace number.
- **The arm's real meshes come within 1 mm of the torso**, and within 9.2 mm for arm 5 vs
  arm 7 — those self-collisions are real. Arm 2 vs arm 4 (82.8 mm) and arm 4 vs arm 6
  (71.7 mm) cannot touch; their proxies only overlap at the joint, and they are excluded.

## 5. What doesn't work / open problems

1. **Nothing holds the non-commanded joints — the robot collapses at every reset.** `ctrl` is
   0 for all 30 actuators after `reset()`; the action term then writes only the 16 arm/jaw
   actuators, leaving both legs, both `leg_*_length_joint` and `pelvis_1/2` commanded to 0.
   `leg_*_length_joint` goes 0.600 → 0.128 m (its mechanical minimum) in **0.2 s**, the feet
   leave the floor after one control step, and `pelvis_1` sits at its hard limit ±0.349.
   This answers the plan's S4.1 question: **keyframe `ctrl` does not survive reset in mjlab
   1.6.0.** Fix is a `hold_nominal_targets` reset event (plan D3).
2. **PAL's arm gains cannot hold the arm against gravity.** At the tabletop start pose the
   holding torque is 16.19 Nm on `arm_left_1_joint` (stiffness 40 → **0.405 rad** of droop)
   and 10.30 Nm on `arm_left_4_joint` (stiffness 20 → **0.515 rad**). A
   `hold_nominal_targets` reset event does *not* fix this: with every actuator held at
   nominal the grasp point still falls from z = 0.886 to 0.793 in 10 s and `pelvis_1` still
   reaches its limit. S3's "max |Δq| < 0.02 rad" acceptance criterion is unmeetable as
   written. Options for S4: MuJoCo `gravcomp` on the arm/torso bodies; a
   `MANIPULATION_HOLD_GAINS` block (arm 1 would need ~800 Nm/rad, not credible against the
   50 Nm effort limit); or welding the torso instead of `base_link`.
3. **`tests/` still covers none of `tasks/manipulation/`'s MDP terms** (the new collision
   test covers the robot model only), and no `Mjlab-Lift-*` / `Mjlab-Stack-*` training run is
   recorded.
5. **`pyright` is disabled in CI** ("TODO: make pyright work", commented out in `ci.yaml`). Type
   errors will not be caught for you.
6. **No roadmap, decision records or status summary yet** in `../docs/`. Create them on
   the first real task.

## 6. Current best settings

- **Cost:** ❌ unmeasured on every host. Measure minutes per 1000 PPO iterations at a fixed
  `--env.scene.num-envs` once per host and record it here; all sizing keys off it.
- **Known-good configuration:** `make sync` (= `uv sync --all-extras --all-packages`);
  `make test` (= `uv run --extra cpu pytest`, `FORCE_CPU=1` to force CPU on a GPU box);
  `make format` (ruff format + `ruff check --fix`). Upstream README's suggested training scale is
  `--env.scene.num-envs 4096` — **unverified on our hosts.**

## 7. Experiments run

| Run | Task id / config / seed | Host | Result | Where recorded |
|---|---|---|---|---|
| S1 scene probe, 2026-09-21 | `Mjlab-Lift-Pal-Kangaroo-Grippers`, play cfg, `num_envs=1`, cpu, torch seed 0 | local Mac (CPU, no `mujoco_warp_io.patch`) | 500 steps survived under both `zero` and `random`; no NaN, no auto-reset; cube rests at z = 0.785; 0 robot-robot contacts; legs collapse (§5.1) | `../docs/reports/pick-place-v0/s1-checkin-1-look-pose-reach.md` |
| S1 reach map, 2026-09-22 | `scripts/reach_map.py --samples 50000 --seeds 48`, seed 0 | local Mac | 445 graspable cells pre-S1.4, 353 after | `s1-report.md`, `s1_reach_map_{baseline,tabletop}.npz` |
| S1.4 collision model, 2026-09-22 | `get_kangaroo_grippers_tabletop_robot_cfg` | local Mac | 36 → 50 active robot colliders, 0 robot-robot contacts at rest, 21% of the graspable table lost | `s1-report.md` §4 |
| *(no training run yet)* | | | | |

## 8. Compute state

| host | what is installed / checked out there | last verified |
|---|---|---|
| local Mac | this checkout, `main` @ `f33baec`, clean; git source of truth | 2026-09-21 |
| `Geoffery` | ~20 checkouts under `~/repo/humanoids/`; this repo's presence unverified | — |
| `blackwell` | shallow clone under `~/repo/humanoids/`; sm_120, driver-check patch likely needed | — |
| `limxH100` | not verified for this repo | — |
| `limx4090` | not verified for this repo | — |
| IAS Slurm (`mn.ias...`) | repos under `/home/ju/repo/humanoids`; this repo's presence unverified | — |
| Cluster42 | staging area on `Geoffery`; nothing deployed for this repo | — |

## 9. Next steps

0. **Pick-and-place v0: S1 is done** (`../docs/reports/pick-place-v0/s1-report.md`). Next is
   **S4/D3 before S3**: the holding problem in §5.2 above makes one of S3's acceptance
   criteria unmeetable, and every episode currently starts by collapsing. Then S2's DR0 and
   S3's env cfg, which consume the spawn/target above.
1. Write down what Oppsie is (§1 here and the header of `CLAUDE.md`), and its non-goals.
2. Create the `siwei` branch off `main` and work there.
3. Smoke-test the new manipulation tasks: `list_envs --keyword pal`, then `play
   Mjlab-Lift-Pal-Kangaroo-Grippers --agent zero` and `--agent random`; add a launch test under
   `tests/` mirroring `test_kangaroo_velocity_task_launch.py`.
4. Pick the training host, install the repo there with `uv`, and measure the per-iteration cost;
   fill §6.
5. Create `../docs/roadmap.md` and `../docs/status_summary.md` before the first real run.
