# Project memory — Oppsie-for-Kangaroo

*Long-term project state. Read before starting work; update when a decision, milestone or
result lands. Not a conversation log — replace outdated entries rather than appending.
Keep under ~500 lines.*
*Last updated: 2026-09-22 (pick-and-place v0 S1-S4 complete, S5 implemented and deployed to
the IAS cluster; no experiment launched. Reports in `../docs/reports/pick-place-v0/`).*

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
| Pick-and-place v0 S4 (holding the body) | ✅ complete — `../docs/reports/pick-place-v0/s4-report.md`, decision `../docs/decisions/0001-*.md` |
| Pick-and-place v0 S3 (env + DR0) | ✅ complete — `Mjlab-PickPlace-Pal-Kangaroo-Grippers` registered, `../docs/reports/pick-place-v0/s3-report.md`. Cost per iteration still unmeasured (no GPU here) |
| Pick-and-place v0 S2 (DR ladder) | ✅ complete — six rungs, `../docs/reports/pick-place-v0/s2-report.md`. Only DR0 and DR-object are in v0's scope; ranges above them are untuned |
| Pick-and-place v0 S5 implementation (5 reward variants, regularisers, sbatch) | ✅ `../docs/reports/pick-place-v0/s5-report-part1.md` |
| Pick-and-place v0 S5 experiments (C0-C3) | ❌ **not launched** — deployed to IAS and waiting on a go; cost still unmeasured |
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
- **Gravity compensation, not stiffer gains, for tabletop manipulation** (decision 0001,
  2026-09-22). PAL's impedance gains need 16.19 Nm at a stiffness of 40 on `arm_left_1_joint`
  at the tabletop pose — 0.405 rad of droop — so the torso, both arms and both grippers carry
  `body gravcomp="1"` and PAL's K/D are untouched. Raising stiffness 15x also passes the S4
  acceptance but discards the one actuator parameter traceable to the real robot. **The
  premise — that PAL's real controller compensates gravity below the impedance loop — is an
  inference, not a citation, and is what would overturn the decision.**
- **Tabletop scenes use `env_spacing=0.0`**, not the locomotion default. mjlab batches envs
  as identical worlds; a terrain-bearing scene still grids `env_origins`, and
  `reset_objects_on_surface` (reads physics) and `LiftingCommand` (adds env_origins) then
  disagree about where the goal is.
- **S5 reward variants are selected by name** (`make_pick_place_env_cfg(reward=...)`): R-A
  framework baseline, R-B sparse, R-C phase-gated, R-D keypoints, R-E time cost. All share the
  same regularisers. R-D uses Allshire et al.'s logistic kernel `K(x)=(e^{30x}+2+e^{-30x})^-1`
  over 8 bounding-box corners, not the plan's Gaussian sketch, because the variant exists to
  test the published idea.
- **Pick-and-place v0 geometry (user decision, 2026-09-22):** object spawn centre
  (0.42, 0.27) +/- 2 cm, fixed target (0.42, 0.07, 0.785), 20 cm apart. Spawn clearance
  14.0 cm, target 4.5 cm — the target is 0.5 cm under the plan's 5 cm rule after S1.4;
  (0.36, 0.07) would give 8.2 cm at the same separation if it is ever revisited. Both live in
  `pick_place_env_cfg.py` as `SPAWN_XY` / `TARGET_XY`.
- **DR level is a named argument, not prose** (`dr="DR0"`, `tabletop_env_cfg.apply_dr_level`).
  Six cumulative rungs: DR0, DR-object, DR-contact, DR-robot, DR-obs, DR-scene. The base
  tabletop cfg is *not* DR0 — it randomises pad friction at startup and jitters the commanded
  joints at reset — so "no randomisation" has to be asked for; `dr_level=None` (the default)
  keeps lift and stack exactly as they were. Asking for an unknown rung raises.
  `tests/test_manipulation_dr_ladder.py` asserts each rung varies what it claims and nothing
  else, against the compiled model rather than the config.

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
- **A validation at `num_envs=1` proves less than it looks.** `env_origins` is all-zero at one
  env and a grid of `env_spacing` above it, so anything keyed off it is silently right at one
  env and wrong at scale. This cost the tabletop tasks a goal 1.6-1.9 m from the cube at every
  `num_envs > 1` until 2026-09-22 (`env_spacing=0.0` now). Check per-world model fields,
  contact budgets and anything touching env origins at 4+ envs.
- **`reset_objects_on_surface`'s `pose_range` is relative to the table centre**, while
  `LiftingCommand`'s `target_position_range` is relative to the *env origin*. A spawn point
  measured in the env frame must be converted; the two are 0.55 m apart in x on this table.
- **Use `dr.pseudo_inertia`, not `dr.body_mass`, for a density change.** `body_mass` leaves
  `body_inertia` alone — a point mass at the COM — and mjlab warns as much at runtime. Its
  `alpha` is a log scale: mass and inertia both scale by `exp(2*alpha)`.
- **`root_link_pos_w` of a fixed-base entity is its *spawn* pose, not where its body is.**
  Randomise the table height with `dr.body_pos` and `data.geom_xpos` moves while
  `root_link_pos_w` stays put, so `EntityOffsetCommand` cannot track it. `mdp.SurfaceOffsetCommand`
  reads the surface out of physics instead, the same way `mdp.events.surface_top_z` does.
- **mjlab's actuator delay is a cfg field, not an event.** `delay_min_lag`/`delay_max_lag`
  and friends on `ActuatorCfg` (`mjlab/actuator/actuator.py:67-85`), with per-env lags via
  `Actuator.set_lags`. Switching it on needs a robot-cfg variant, so it is not in DR-robot.
- **`JointPositionAction` is absolute**: `target = default_joint_pos + scale * action`
  (`mjlab/envs/mdp/actions/actions.py:212-224`), and `clip_actions` defaults to `None`. With
  the manipulation scales (0.10 rad on arm 1-5, 0.05 on the wrist) an action of magnitude 1
  moves the Kangaroo's grasp point up to 16.3 cm — measured — so the start pose decides what
  an untrained policy can reach. From the S1.2 pose both the spawn and the target are inside
  the |action| <= 1 envelope.
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

1. *(fixed 2026-09-22, S4)* `ctrl` is 0 for all 30 actuators after `reset()` --
   `Entity.clear_state` zeroes `joint_pos_target` (`mjlab/entity/data.py:238`) and only action
   joints get one written back. `mdp.hold_nominal_targets` now writes the defaults back. Left
   alone it collapsed `leg_*_length_joint` 0.600 → 0.128 m in 0.2 s.
2. **`tests/` still covers none of `tasks/manipulation/`'s MDP terms** (the new tests cover
   the robot model and the hold/gravcomp boundary only), and no `Mjlab-Lift-*` /
   `Mjlab-Stack-*` training run is recorded.
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
| S5 reward variants, 2026-09-22 | `tests/test_pick_place_rewards.py`, 2 envs per arm, CPU | local Mac | all five arms build, step and evaluate finite terms; R-D differs from R-A in exactly one term | `s5-report-part1.md` |
| S2 DR ladder, 2026-09-22 | `tests/test_manipulation_dr_ladder.py`, 4 envs per rung, CPU | local Mac | every rung varies exactly the model fields it claims; DR0 varies none; object mass lands in 0.030-0.098 kg with inertia tracking it | `s2-report.md` |
| S3 env validation, 2026-09-22 | `Mjlab-PickPlace-Pal-Kangaroo-Grippers`, `num_envs=1`, CPU | local Mac | `--agent zero` 500 steps: left arm max \|dq\| 0.0005 rad (limit 0.02). `--agent random` 1000 steps: no auto-reset, no termination, no NaN. 11.8 control steps/s — a laptop number, not a cost | `s3-report.md` |
| S4 hold check, 2026-09-22 | `scripts/hold_check.py --steps 1000`, seed 0, 4 configurations | local Mac | before: pelvis 0.357 rad, right arm 0.246, leg length 0.491 m. After hold event + gravcomp: 0.007 / 0.019 / 0.004 — S4 acceptance PASS at steady state | `s4-report.md` |
| *(no training run yet)* | | | | |

## 8. Compute state

| host | what is installed / checked out there | last verified |
|---|---|---|
| local Mac | this checkout, `main` @ `f33baec`, clean; git source of truth | 2026-09-21 |
| `Geoffery` | ~20 checkouts under `~/repo/humanoids/`; this repo's presence unverified | — |
| `blackwell` | shallow clone under `~/repo/humanoids/`; sm_120, driver-check patch likely needed | — |
| `limxH100` | not verified for this repo | — |
| `limx4090` | not verified for this repo | — |
| IAS Slurm (`mn.ias...`) | **this repo deployed 2026-09-22**: `/home/ju/repo/humanoids/pal_mjlab_kangaroo_talos`, branch `siwei` @ `4257e90`, hash-verified; `uv` 0.12.17 at `~/.local/bin`; `uv sync --frozen --extra cu128` done (same versions as the Mac); `mujoco_warp_io.patch` NOT applied; `slurm_logs/` created. Deployed by git bundle over scp, not by pushing to origin. ~7 free GPUs matching `vram48gb\|rtx3090\|a5000` at deploy time | 2026-09-22 |
| Cluster42 | staging area on `Geoffery`; nothing deployed for this repo | — |

## 9. Next steps

0. **Pick-and-place v0: S1-S4 done, S5 implemented and deployed to IAS; nothing launched.**
   Next, in order: (a) **C0** — 1 seed, 256 envs, 200 iterations, ~1 GPU-hour, the exact
   `sbatch` line is in `s5-report-part1.md` §4. It is the first GPU run of this task and the
   only source of the per-iteration cost everything else is sized from; (b) size and launch
   **C1** (R-A, DR0, 3 seeds) from C0's numbers; (c) the plan's check-in before C2.
   Decide before launching: the wandb project name (`oppsie-pick-place-v0`, does not exist
   yet) and how C2 selects a reward arm (more task ids goes beyond D7).
   Create `../docs/status_summary.md` and `../docs/Mjlab-PickPlace-Pal-Kangaroo-Grippers.md`
   with C0's numbers — CLAUDE.md requires both and neither exists.
1. Write down what Oppsie is (§1 here and the header of `CLAUDE.md`), and its non-goals.
2. Create the `siwei` branch off `main` and work there.
3. Smoke-test the new manipulation tasks: `list_envs --keyword pal`, then `play
   Mjlab-Lift-Pal-Kangaroo-Grippers --agent zero` and `--agent random`; add a launch test under
   `tests/` mirroring `test_kangaroo_velocity_task_launch.py`.
4. Pick the training host, install the repo there with `uv`, and measure the per-iteration cost;
   fill §6.
5. Create `../docs/roadmap.md` and `../docs/status_summary.md` before the first real run.
