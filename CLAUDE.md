# Oppsie-for-Kangaroo

This repository is the TU Darmstadt fork of [`pal_mjlab`](https://github.com/pal-robotics/pal_mjlab)
— PAL Robotics' robots (KANGAROO, Talos, TIAGo Pro) implemented in
[mjlab](https://github.com/mujocolab/mjlab), i.e. the Isaac Lab API on MuJoCo Warp. It is the
code base for the **Oppsie** work on the KANGAROO humanoid: robot models, task/environment
configs, and the RL training stack used to produce and evaluate policies for that robot.

The central constraint: **this is a fork, not a rewrite.** The tree was restarted at upstream
`pal_mjlab` main `a461dc3` (commit `3b05cf6`, "Restart: fork pal_mjlab upstream main (a461dc3)
at the repo root"), and mjlab itself (`mjlab>=1.6.0`) is an installed dependency, not vendored.
Every design decision has to survive an upstream merge: prefer new task packages, new config
functions and new constants over edits to inherited files, and keep the upstream layout
(`src/pal_mjlab/{robots,tasks,scripts,docs}`) intact.

> **Gap to close (2026-09-21):** what *Oppsie* is — the method, its central technical premise,
> and what a finished result looks like — is not written down anywhere in this repo. Fill the
> paragraph above and the non-goals below the first time it is stated, and record it in
> `memory_oppsie_kangaroo.md` §1. Until then, do not infer the project's goal from the code.

Non-goals (provisional, confirm before relying on them): Talos and TIAGo Pro are carried as
inherited from upstream and are not being developed here; sim-to-real deployment on hardware;
upstreaming changes to `pal-robotics/pal_mjlab` (do not open anything there without asking).

Authoritative context lives in `../docs/` — the `docs/` folder beside this repo in `oppsie-data/`, outside git (see *Repository layout*); `src/pal_mjlab/docs/` is the
user-facing Sphinx documentation, a different thing. Long-term state lives in
`memory_oppsie_kangaroo.md` (see *Project memory*).

---

# High-Level Principles

Always prioritize:

1. Correctness over speed.
2. Reusing existing infrastructure instead of rewriting components.
3. Small, verifiable changes.
4. Reproducibility.
5. Minimal disruption to the existing codebase, and to code you do not own — upstream
   `pal_mjlab` files and installed `mjlab` above all.
6. Writing it down — a result without its provenance does not exist.

Whenever possible, preserve the current architecture, the configuration hierarchy
(`*_env_cfg()` factory functions taking `play=`), and compatibility with existing registered
task ids; avoid introducing unnecessary abstractions.

---

# Expected workflow for implementation tasks

1. Read the relevant code and docs.
2. Understand the current architecture and execution path.
3. Identify the components that can be reused.
4. Produce an implementation plan.
5. Implement minimal changes.
6. Validate.
7. Summarize: files changed, commands executed, validation performed, remaining limitations.
8. Update the relevant docs, the decision record if a decision was taken, and
   `memory_oppsie_kangaroo.md`.

**Never immediately begin editing files.**

---

# Working agreement — plan, implement, document, report

1. **A plan comes first.** Usually the user writes it (`../docs/*-plan.md`); follow that one.
   If no plan exists for the work being asked, produce one and get it approved before
   implementing.
2. **Follow the plan, and say where you deviate.** Deviations are allowed and often correct,
   but they are written down with the evidence that forced them — never applied silently.
3. **Document the implementation as it happens**, not afterwards from memory: what changed,
   what was run, what was measured, what broke and what the fix was. The failures are the
   valuable half — they are what stops the next person repeating them.
4. **Ask for input on important decisions.** Anything that changes scope, picks between
   approaches, commits significant time or compute, deletes data, or invalidates an earlier
   decision: stop, lay out the alternatives with evidence, recommend one, and ask. Routine
   judgment calls do not need a check-in.
5. **Record consequential decisions** in `../docs/decisions/NNNN-slug.md`: context → findings
   with file/line or measurement evidence → decision → consequences → confidence and what
   would change it. Supersede old records rather than editing the history out of them.
6. **Write a stage report at the end of each stage** — each milestone or experiment campaign —
   in `../docs/reports/`; raw artifacts go in `artifacts/<stage>/` (gitignored). Written for
   someone reading it cold in three months, it states: what the stage set out to do and which
   exit criteria were met, missed or dropped; the numbers and how they were measured; the
   decisions taken and where their records live; what broke; what is still open; and the
   provenance (branch, commit, task id, config, seed, host, wandb run, exact command).
7. **Update `memory_oppsie_kangaroo.md`** whenever a milestone, decision or result lands.

Never claim a step is done without saying how it was validated. If validation was not
possible, say so explicitly and why.

---

# Evidence Requirement

Do not assume anything about this project, its history, or its infrastructure.

Before answering questions or making claims about:

- what a component, script, config or parameter does;
- which branch or commit implements a behaviour;
- how a machine or service is accessed or configured;
- what has or has not been validated;

read the actual code and the docs, and cite the specific file, line, branch, commit or
measurement backing each claim.

If evidence cannot be found, say so explicitly instead of inferring or guessing. Flag
documentation gaps rather than filling them in from assumption.

Habits that catch most of these mistakes:

- **Do not generalise from one robot variant.** KANGAROO has four variants in this repo
  (`kangaroo`, `kangaroo_lower_body`, `kangaroo_hands`, `kangaroo_grippers`) plus Talos and
  TIAGo Pro. One env config is an example, not a rule. Check a second before writing it down.
- **Check which copy is live.** `mjlab` is an installed dependency resolved by `uv`, not a
  checkout in this tree: the version on `sys.path` in `.venv` is the one that matters, and it
  is pinned in `uv.lock`, not by the `>=1.6.0` floor in `pyproject.toml`. `mujoco_warp` is
  patched locally (see *External Dependencies*) — an unpatched host behaves differently.
- **A branch label is not a code state** on hosts that receive patch deploys. Verify by
  checksum against the source of truth (the Mac / `origin`).
- **A summary is not evidence.** `src/pal_mjlab/docs/release_notes.rst` and any status table
  get quoted without their sources being read; when a number matters, open the document or the
  wandb run that measured it.

Claims that have already been wrong here:

- **`README.md` describes the upstream project, not this fork.** Its clone URL
  (`github.com/pal-robotics/pal_mjlab.git`) is not this repo's `origin`, and its `your-org`
  wandb paths are placeholders. Do not quote it as fact about this checkout.
- **`.github/workflows/ci.yaml` is a GitHub Actions workflow, but `origin` is IAS GitLab.**
  Nothing in this tree shows that CI actually runs on pushes here. Treat "CI will catch it" as
  unverified — run `make test` and `make format` locally.
- **Task ids are not uniformly spelled.** `Mjlab-Tracking-Flat-Pal-Kangaroo_Lower_Body` uses
  underscores while `Mjlab-Velocity-Flat-Pal-Kangaroo-Lower-Body` uses hyphens. Copy ids from
  `grep -rh 'task_id="' src/pal_mjlab/tasks`, never from memory or from the pattern.
- **The console script is `list-envs`, not `list_envs`** — `uv run list_envs` is "command not
  found". (Checked 2026-09-21 against `.venv/bin/`; `play` and `train` are spelled as written.)
- **"The grippers variant has no torso collider" is wrong.** `kangaroo_grippers.xml:82` defines
  `pelvis_2_collision`, a torso capsule, and `:100/:181` define `arm_*_4_collision`. They are
  present in the XML and *filtered out* by `MANIPULATION_COLLISION`, which lists only feet,
  arm 5-7 and the gripper — `CollisionCfg.edit_spec` zeroes `contype`/`conaffinity` on every
  geom it does not match. "Missing geom" and "geom disabled by the variant's collision policy"
  need different fixes. (Measured 2026-09-21, `../docs/reports/pick-place-v0/`.)
- **Re-enabling a geom's `contype`/`conaffinity` on a *compiled* `MjModel` does not make it
  collide.** MuJoCo's compiler leaves non-colliding geoms out of the bounding-volume hierarchy,
  so the geom reports `contype=1` and broadphase never proposes it: 0 contacts in 100k poses
  that visibly intersect. Collision geometry has to be changed in the `MjSpec` — for mjlab, by
  editing the entity's `CollisionCfg` before `Scene.compile()`. Any "variant X costs nothing"
  result from this kind of experiment needs a positive control first.
- **"The actuator's `ctrlrange` clamps a joint target at the joint limit" is wrong in mjlab.**
  mjlab builds every position actuator with `ctrllimited=False` (`mjlab/utils/spec.py`), so a
  target past a limit makes the actuator push the joint into its stop at up to full effort. Clamp
  targets in the action term (`JointPositionActionCfg.clip`). (Measured 2026-09-23, S5.0.)
- **A contact sensor's `found` is not `(B, num_primaries)`.** It is `(B, P × num_slots)`,
  primary-major. `.all()` over the raw tensor demands every slot of every primary. Together with
  inner-finger geoms in the grasp sensor, that made `grasped` unreachable for a cube held at the
  pads. C1/C2's `grasp` = 0 had this cause as well as the action scale. Reduce over slots first.
  (Found 2026-09-23, S5.0.)
- **"At jaw scale 1.0 the grasp sensor fires" was wrong for a jaw closing on a free cube.** The
  jaw stops at 0.39 rad against the cube and the inner fingers never touch. A test that pins the
  object in place is not a grasp test. Use `scripts/scripted_pick_place.py rollout`.
- **"A jaw rate limit fails the grasp" (S6 bench) was a bench artefact.** `gripper_bench.py`
  applied the limit once per `set_jaw` call, and the `step`/`shake` stimuli call it once, so
  the jaw never closed. Fixed in `2348670`; with it, 0002 + a 0.2 rad/step slew grasps 64/64.
- **mjlab's `NoiseModelWithAdditiveBias` does not redraw its bias at reset, it adds to it**
  (default `operation="add"`), so the "per-episode" bias random-walks: 65 mm after 20 resets
  against an 8 mm range. Set `operation="abs"` on `bias_noise_cfg`. (Measured 2026-09-23.)
- **A string-keyed DR `ranges` dict is not scoped by `asset_cfg.joint_names`** in mjlab:
  each key is matched against every joint of the entity. `arm_.*` randomises a parked arm too.
- **"DR-scene places the cube on the moving table" was wrong until 2026-09-24.** Its goal,
  `mdp.SurfaceOffsetCommandCfg`, inherited mjlab `LiftingCommandCfg`'s non-None
  `object_pose_range`, so the command re-placed the cube on the floor (x 0.30-0.35, z 0.02-0.05)
  *after* the reset event had put it on the table. The DR-scene tests checked the goal and the
  table, never the cube. A command must not place objects here; check where the object is after
  `env.reset()`, not only that the placement event ran. (Found by plan S7's D5 test.)
- **"DR0 randomises nothing" is true of the model, not of the observations.** The actor's base
  observation noise (joint pos +/-0.002 rad, vel +/-0.2 rad/s, object pos +/-5 mm, yaw
  +/-0.02 rad; `tabletop_env_cfg.py`) is on at every rung in training; only `--play` turns it
  off. (Found 2026-09-24, plan S7's D4 alignment.)
- *(Add every further claim that turns out wrong. These lines are the most valuable in the file.)*

---

# Project memory

Long-term state lives in **one memory file per project**, at the repository root:

- **`memory_oppsie_kangaroo.md`** — Oppsie-for-Kangaroo: what is built, what works, what has
  been measured, what is next.

If a combined file is ever split, leave `memory.md` as a **pointer stub only** — never write
project state there.

Rules:

1. **Read `memory_oppsie_kangaroo.md` before starting any work.** This is in addition to, not
   instead of, the Evidence Requirement.
2. **Update it whenever an important decision is made, a milestone or notable progress is
   reached, or a result lands** — negative results included. Record enough provenance to
   re-find the evidence (notes path, branch/commit, wandb run id, artifact directory, host).
3. Cross-cutting facts (simulator version, robot model quirks, host environments) live in this
   file; per-task detail lives in `../docs/`.

A memory file is **not a conversation log**. It is concise structured knowledge that will be
useful for continuing the project months later. It should always answer:

- What are we building?
- What currently works?
- What doesn't work?
- What design decisions have been made, and why?
- What experiments have been run?
- What are the current best settings?
- What should be done next?

Only include what is still relevant. Prefer **replacing** outdated information over appending.
Refactor regularly: merge duplicated sections, delete obsolete entries, keep it under roughly
300–500 lines. Detail belongs in `../docs/`; the memory file carries the state and the pointer.

---

# Per-robot / per-task records and the status summary

The work spans several **(robot variant × task)** pairs, each trained and evaluated separately.
The registered ids, as of 2026-09-21 (`grep -rh 'task_id="' src/pal_mjlab/tasks`):

| family | ids |
|---|---|
| Velocity | `Mjlab-Velocity-{Flat,Rough}-Pal-Kangaroo[-Lower-Body|-Hands|-Grippers]`, `Mjlab-Velocity-{Flat,Rough}-Pal-Talos` |
| Tracking (imitation) | `Mjlab-Tracking-Flat-Pal-Kangaroo[-Hands]`, `Mjlab-Tracking-Flat-Pal-Kangaroo_Lower_Body`, `Mjlab-Tracking-Flat-Pal-Talos`, plus `-No-State-Estimation` variants |
| Reaching | `Mjlab-Reaching-Flat-Pal-Kangaroo[-Hands|-Grippers]`, `Mjlab-Reaching-Flat-Pal-Tiago-Pro` |
| Manipulation | `Mjlab-Lift-Pal-Kangaroo-Grippers`, `Mjlab-Stack-Pal-Kangaroo-Grippers` |

`../docs/status_summary.md` is the cross-task status matrix, and it is the file most likely to be
quoted without its sources being read.

**Whenever work is done on one of these tasks** — a run launched, finished, stopped or
invalidated; a motion retargeted or repaired; an evaluation scored; a gate passed or failed; a
defect found; a decision taken — update **both**:

1. **that task's own document** (`../docs/<task-id>.md`), with the full record: branch, commit,
   config function, seed, wandb run id/path, checkpoint, launch command, and the measurement
   itself; and
2. **the status summary** — the matrix cell, the task's section, and its "what still needs
   doing" list when the change opens or closes a task.

Updating one and not the other is precisely what makes the summary stale and its numbers unsafe
to quote.

Legend, held to strictly: ✅ done · 🟡 qualified · ⏳ in flight, outcome unrecorded · ❌ not
started · 🛑 run and invalidated. **A run that is launched but unread is ⏳ with its wandb run id
and ETA recorded — never ✅.**

Exploratory work that is not yet a per-task result stays in its own doc; it enters the matrix
when it settles and has been run across the variants it claims to cover.

---

# Repository layout

```
src/pal_mjlab/
  robots/<robot>/           # MuJoCo XMLs, STL assets, *_constants.py (EntityCfg, actuators)
  tasks/<family>/           # velocity, tracking, reaching, manipulation
    <robot>/__init__.py     # register_mjlab_task(...) — this is where task ids are defined
    <robot>/env_cfgs.py     # *_env_cfg(play: bool) factory functions
    <robot>/rl_cfg.py       # PPO runner config
    mdp/                    # rewards, observations, commands, events, terminations
  scripts/csv_to_npz.py     # motion retargeting CSV -> NPZ
  docs/                     # Sphinx site (user-facing) — release_notes.rst lives here
motions/                    # retargeted motion CSVs (committed); NPZ outputs are gitignored
tests/                      # pytest; runs on CPU via `make test`
static/                     # images used by README and docs
../docs/                    # plans, decision records, stage reports — canonical working context, OUTSIDE the repo
  roadmap.md                # goal, phases, milestones, open questions — start here
  decisions/                # decision records NNNN-slug.md
  reports/                  # stage reports
  status_summary.md         # cross-task status matrix
artifacts/<stage>/          # raw run artifacts (gitignored)
memory_oppsie_kangaroo.md   # long-term project state
mujoco_warp_io.patch        # local patch to an installed dependency (see below)
```

`../docs/` (i.e. `oppsie-data/docs/`, a sibling of this repo) is the canonical place for working context; it is not tracked by this repo. When `../docs/`
and `src/pal_mjlab/docs/` describe the same thing, **`../docs/` is authoritative for project state
and `src/pal_mjlab/docs/` is authoritative for user-facing behaviour** — the Sphinx pages are
published and must not carry internal run logs.

**Never commit:** trained checkpoints, ONNX exports, wandb run directories, `.npz` motion data,
rendered videos, or any `artifacts/` content. `.gitignore` already covers `logs`, `wandb`,
`artifacts/`, `*.npz`, `motion.mp4` — check it before claiming something is committed.

---

# Coding Style

Prefer

- extending existing classes and `*_env_cfg()` factories
- adding configuration flags and new named constant blocks in `<robot>_constants.py`
- adding helper functions in the task's `mdp/` module

instead of

- duplicate implementations
- parallel pipelines
- unnecessary inheritance

Hard conventions in this repo (`pyproject.toml`, `.pre-commit-config.yaml`):

- **ruff, `indent-width = 2`.** Two-space indentation, everywhere. `make format` runs
  `ruff format` then `ruff check --fix`; CI checks `ruff@0.14.14 check --diff`,
  `ruff format --diff` and `pre-commit run --all-files`.
- Lint rule set is `E4, E7, E9, F, I, B` — import sorting (`I`) and flake8-bugbear (`B`) are
  enforced, so unsorted imports and mutable default arguments fail the build.
- **Python 3.10 is the floor** (`requires-python = ">=3.10,<3.14"`, CI matrix 3.10–3.13) even
  though `.python-version` pins 3.13 locally. No `tomllib`, no `typing.Self`, no `StrEnum`, no
  3.11+ exception groups — they pass locally and break the 3.10 CI leg.

Keep functions focused. Avoid deeply nested logic. Document non-obvious algorithms — especially
anything involving **joint-name regexes, coordinate frames, quaternion conventions, units, and
which joints are actuated vs equality-driven**, where this project's bugs live.

Match the surrounding code: naming, comment density, idiom. New code should not be identifiable
as new by its style.

---

# Configuration Philosophy

Avoid hardcoding values. Prefer the config objects: env behaviour belongs in `env_cfgs.py`
factories, robot/actuator parameters in `<robot>_constants.py` named parameter blocks, PPO
settings in `rl_cfg.py`. Keep defaults backwards compatible so existing task ids stay
reproducible.

**Watch defaults that silently apply to the wrong variant.** This is the recurring failure mode
here: a value tuned for one KANGAROO variant is inherited by the others without complaint, and
nothing errors. Two live examples in `kangaroo_constants.py`:

- `REGEX_ALL_ACTUATED_JOINTS` explicitly excludes `leg_*_femur_joint`, `leg_*_knee_joint` (driven
  by the leg four-bar equalities) and `gripper_*_joint` (driven by the jaw four-bar) — because
  otherwise the grippers variant pulls 12 finger joints into the locomotion posture reward,
  whose per-joint std tables only cover legs, pelvis and arms.
- IMU noise and the `upright` reward weight are overridden **for the lower-body flat config
  specifically**, not in the shared baseline (release notes, PR #101).

When adding a parameter, either set it explicitly per variant or make it fail loudly on a
variant it was not tuned for. The startup asserts in `kangaroo_constants.py` (every XML path is
`assert p.exists()`) are the pattern to follow.

Machine-specific paths and environment belong in a per-host setup file, not in committed
scripts. **Credentials belong in neither** — not in source, git, a command line, a launcher
manifest, a log, or chat.

---

# Logging

Runs log to Weights & Biases (`--agent.logger wandb --agent.wandb-project ... --agent.run-name
... --agent.wandb-tags ...`). Log enough to reproduce behaviour:

- episode return and the per-term reward breakdown (which term is actually driving the policy)
- tracking error / command-following metrics for velocity and tracking tasks; success rate for
  lift and stack
- termination causes by category (fall, timeout, joint limit, contact)
- throughput: steps/s, iteration wall-clock, GPU memory and utilization
- anything that distinguishes "it ran" from "it ran correctly" — e.g. contact counts, `njmax`
  saturation, NaN/Inf guards

Avoid excessive logging inside hot loops. Make any batch/sweep launcher print an
**expected-vs-actual count** at the end: a loop that under-delivers reports no error, it just
returns fewer runs than asked for.

---

# Validation Requirements

Every implementation should be validated. In this repo that means, in rough order of cost:

- `make test` — `uv run --extra cpu pytest`, the CPU test suite (`tests/`, with
  `FORCE_CPU=1` available to force CPU on a GPU box). Existing tests cover velocity/tracking
  rewards, the dual-band velocity command, the Talos collision model and a task launch.
- `make format` — ruff format + lint, matching CI.
- `uv run list-envs --keyword pal` — the new task id is registered and importable.
- `uv run play <task-id> --agent zero` and `--agent random` — the env constructs, steps, and
  does not explode with zero/random actions.
- shape, dtype and unit checks at MDP term interfaces; joint-name regexes matched against the
  compiled model, not assumed.
- a disabled-path check: with the new flag off, behaviour is identical to before.
- a short training run (few hundred iterations, small `--env.scene.num-envs`) for anything that
  changes what is learned.

Do not claim completion without validation. If validation cannot be performed, explicitly
explain why.

**A training curve is evidence about training only.** Reward going up says nothing about
closed-loop behaviour — measure the thing you are claiming, with `play` and the task's own
metrics.

---

# Safety-Critical / High-Blast-Radius Code

Any modification affecting

- `src/pal_mjlab/robots/*/xmls/*.xml` and `*_constants.py` — the models every task instantiates
- shared MDP terms in `tasks/*/mdp/` that several env configs import
- reward weights, termination conditions, observation composition, action scaling — anything
  that changes what is learned
- `register_mjlab_task` ids (renaming one silently invalidates every recorded result against it)
- `uv.lock` / dependency pins — they change the simulator under every task at once

should be isolated, flag-gated where possible, default-off, and easy to review in a diff. Add a
new `*_env_cfg()` variant rather than mutating a shared baseline.

---

# Experiment Policy

> **Unmeasured (2026-09-21):** the cost of one unit of experiment here — minutes per 1000 PPO
> iterations at a given `--env.scene.num-envs`, per host — has not been recorded. Measure it
> once on each host you use and write it here; every sizing decision keys off this number.

Experiments are expensive. Before launching anything large:

- verify the task id (copy it from the source, see *Evidence Requirement*)
- verify the config function and any overrides on the command line
- verify branch, commit, and any patch applied on top (including `mujoco_warp_io.patch`)
- verify the output directory, and that it is on the large filesystem
- verify the wandb project / experiment / run name / tags
- verify the GPU, and that the machine is actually free
- verify `--agent.max-iterations` is the intended one, not a default
- verify the wall-clock limit covers the slowest node, not the average one

**Prefer smoke tests first:** smallest `--env.scene.num-envs`, one seed, a few iterations,
`--agent.logger` pointed at a scratch project.

**Never launch multi-day experiments automatically.** Announce long runs before starting them,
with the expected wall-clock.

**Before submitting to a shared machine or a scheduler, show the user** the resource block,
commit, task id, motion/registry path, checkpoint paths, output directory, wandb
project/run/tags, seed, and the full command — then submit only when explicitly asked to.

**After launching, do not declare it healthy until you have watched it work**: GPU visible,
environment and assets loaded, wandb run identity and `running` state confirmed by query (not by
console text), at least one real iteration completed, and the log free of tracebacks, NaN/Inf,
CUDA errors and instability terminations.

**Comparisons need the right arms.** A config-matched control answers "did the change do
anything"; the **current best / published** configuration answers "does it beat what we have" —
and only the second bears on a decision. Include both.

---

# Reproducibility

Whenever experiments are launched, record:

- git branch and commit hash of this repo, plus any uncommitted diff applied on top
- the resolved `mjlab` / `mujoco-warp` / `torch` versions actually installed
  (`uv pip list | grep -E 'mjlab|mujoco|torch|warp'`), and whether `mujoco_warp_io.patch` is applied
- task id, config function, every CLI override, and the seed
- host (and GPU id)
- wandb project / run name / run id
- the exact launch command
- output directory

Put this in the run's artifact directory **and** in the stage report **and** in the memory file.
Scrollback is not a record.

---

# Performance Philosophy

Profile before optimizing. Never assume a bottleneck. Measure:

- CPU and GPU utilization
- wall-clock split by stage — model compile / warp kernel build vs stepping vs PPO update
- throughput of the hot path (env steps/s, samples/s, iterations/s)
- policy update / reset / host-device transfer time separately
- scaling with `--env.scene.num-envs`

Support optimization decisions with measurements. **Bigger `num-envs` is not automatically
faster** — verify throughput rather than assuming, especially on the 80/95 GB cards where the
memory headroom invites a larger number than the CPU side can feed.

Warp kernel compilation is a real fixed cost on a cold cache (CI caches `~/.cache/warp` for this
reason). Time a second run, not the first, or you are timing the compiler.

---

# External Dependencies

`mjlab`, `mujoco-warp`, `torch` and the upstream `pal_mjlab` history are third-party code we do
not own. Treat them as read-mostly.

- Prefer **new files, config factories and flag-gated hooks** over edits to inherited upstream
  files; write a modified *copy* of an XML or config rather than editing the released one.
- If an upstream edit is unavoidable, keep it minimal, record it as a patch at the repo root
  next to `mujoco_warp_io.patch`, and note the upstream commit it applies to.
- **`mujoco_warp_io.patch`** patches `mujoco_warp/_src/io.py` to wrap
  `warp_util.check_toolkit_driver()` in a try/except and downgrade a CUDA toolkit/driver mismatch
  to a warning. It is not applied automatically by anything in this tree — **check whether the
  installed `mujoco_warp` is patched before blaming a host for a driver error, and record which
  state a result was measured under.** Its upstream commit is not recorded; record it the next
  time the patch is regenerated.
- Dependencies are managed by **uv**, not conda or pip. `uv.lock` is the source of truth and CI
  runs with `UV_FROZEN=1`, so any dependency change must commit an updated lock file.
- `torch` comes from mutually exclusive extras: **`--extra cpu`** (tests, laptops) or
  **`--extra cu128`** (GPU hosts). `make sync` runs `uv sync --all-extras --all-packages`.
- `pyproject.toml` carries `override-dependencies = ["pillow>=12.2.0"]` to lift moviepy's
  precautionary `pillow<12` cap, which conflicts with mjviser's `>=12.2`. Do not "fix" this by
  downgrading pillow.
- Pin and record upstream versions in every report and decision record.
- Bugs that belong upstream (mjlab, mujoco_warp, pal_mjlab) get a draft issue in `../docs/upstream/`
  — **do not file anything publicly, or open a PR against `pal-robotics/pal_mjlab`, without
  asking.**

---

# Git Policy

`origin` is **IAS GitLab**: `git@git.ias.informatik.tu-darmstadt.de:talos/TU_Darmstadt/pal_mjlab_kangaroo_talos.git`.
Upstream (`github.com/pal-robotics/pal_mjlab`) is a separate project; this fork restarted at its
main `a461dc3`. Other branches on `origin` as of 2026-09-21: `main`, `KangarooFullEnv`,
`my-new-branch`, `rack-hang`, `talos-reaching-real-nogripper-table-cubes`.

Never

- rewrite user commits
- force push
- discard uncommitted work
- change unrelated files

Always

- inspect the current branch
- inspect `git status`
- summarize changed files

**Committing and pushing — the policy for this repo:**

- **Work on the branch `siwei`.** Create it from the current `main` if it does not exist
  (`git switch -c siwei`) and stay on it; do not commit directly to `main`.
- **Always commit.** Every change is committed as it lands, with a message saying what changed
  and why — uncommitted work on a remote box is one disk failure from gone.
- **Push only when explicitly asked.** Do not push to `origin` on your own initiative, on any
  branch. This grants no license to rewrite or force-push published history.

**Before committing a dirty or detached worktree, find the branch that carried the work forward
and diff against *that* tip** — not against the worktree's own HEAD. A worktree sitting on a
merge-base will happily produce a diff that reverts every commit made since. Identical files show
up as changes; `shasum` them against `git show <branch>:<path>` to tell novel work from stale
copies.

Large artifacts do not belong in git — keep them on the machine that produced them and reference
them by path and host. Check what is gitignored before claiming something is committed: an
ignored `artifacts/` directory makes "committed and hashed" read as satisfied while nothing is
tracked.

---

# Build the docs after touching them

The Sphinx site lives in `src/pal_mjlab/docs/` (`docs` dependency group: `sphinx`,
`sphinx-book-theme`; build output `docs_build/` is gitignored). **If a change touches `.rst`
files, build them before committing** — a source edit that commits cleanly can still break the
output.

```bash
uv run --group docs sphinx-build -b html src/pal_mjlab/docs docs_build
```

Then read the build log and **report specifics, not "it built"**: the exact count of warnings
(expected 0), any unresolved `:ref:`/toctree entry, and any page that silently rendered empty.
Refer to sections **by label, not by number**.

`src/pal_mjlab/docs/release_notes.rst` is the user-facing changelog: add an entry under
*Unreleased* for any user-visible change (new task id, changed default, removed reward term),
in plain language, with the PR or commit it came from.

Build artifacts are gitignored; do not commit them.

---

# Compute Resources

These are the standing machines. **Every fact below carries the date it was verified.** Treat
anything older than a few months as a hypothesis: re-check it, then correct this file. A claim
about a machine must be traceable to a login, not to a memory.

| host | what it is | use it for | do not |
|---|---|---|---|
| **local Mac** | laptop, no GPU | editing, git source of truth, orchestration, docs builds | training, anything that needs a GPU |
| **`Geoffery`** | dev box, 1× RTX 4080 16 GB | implementation, debugging, smoke and integration tests, driving Cluster42 | long experiments — it is too small |
| **`blackwell`** | workstation, 1× RTX PRO 6000 Blackwell Max-Q 95 GiB | single-GPU training, motion/dataset generation, preflight, driving Cluster42 | assuming exclusive use — it is shared |
| **`limxH100`** | server, H100 80 GB | PPO training, sweeps, profiling, benchmarking | running anything on its jump host |
| **`limx4090`** | server, 8× RTX 4090 | parallel evaluation sweeps, multi-seed grids | uploading bulk data to it; treating it as a git source of truth |
| **IAS cluster** | Slurm, `mn.ias...` | queued single-GPU training, seed sweeps, long runs | multi-GPU jobs — the resource block is single-GPU |
| **Cluster42** | Determined, A100-SXM4-80GB × 73 agents | production experiments, concurrent runs, large sweeps | multi-node jobs; driving it from anywhere but `Geoffery`/`blackwell` |

**Environment note for this repo specifically:** the fleet notes below mention conda because
other projects use it. **This project uses `uv`** — `make sync` (`uv sync --all-extras
--all-packages`), then `uv run <entry point>`. Use `--extra cu128` on GPU hosts and
`--extra cpu` where there is no CUDA. Do not create a conda env for it; do not mix the two.

## How to log in

| target | command | user | note |
|---|---|---|---|
| `Geoffery` | `ssh Geoffery` | `siwei` | `ProxyJump ias`. Alias is spelled **Geoffery**; the machine calls itself *Geoffrey* |
| `blackwell` | `ssh blackwell` | `siwei` | `ProxyJump ias` → `blackwell.ias.informatik.tu-darmstadt.de` |
| IAS login/submit | `ssh ju@mn.ias.informatik.tu-darmstadt.de` | `ju` | **not** the `ias` alias — see below |
| `limxH100` | `ssh limxH100` | `max` | `ProxyJump limxH100-jump`; jump host is for proxying only |
| `limx4090` | `ssh limx4090` | `ubuntu` | direct |
| Cluster42 | `det` CLI only, **from `Geoffery` or `blackwell`** | `sj12xefi` | no SSH shell on the login node for this account |

**The `ias` alias is a trap.** `~/.ssh/config`'s `ias` points at
`kay.ias.informatik.tu-darmstadt.de` and exists **only** as the `ProxyJump` for `Geoffery` and
`blackwell`. The Slurm login node is a different machine, `mn.ias...`, reached by its full
hostname with user `ju`. The same identity file works for both. (Verified 2026-08-05.)

Note that `origin` for this repo is on the same IAS GitLab host family — a host that can reach
`git.ias.informatik.tu-darmstadt.de` over SSH can clone and push; a host that cannot needs the
bundle/patch routes below.

**Set up an SSH `ControlMaster`** for anything behind `ProxyJump` — otherwise every command
re-jumps and intermittently times out during banner exchange.

**Credentials are not in this file.** Passwords for hosts that need them live in the per-host
runbook kept locally and pushed nowhere.

## `Geoffery` — primary development machine

`130.83.164.53`, user `siwei`, via `ProxyJump ias`. Ubuntu, **one RTX 4080 (16 GB)** — enough for
smoke and integration checks, not for training. (Verified 2026-08-12.)

```bash
ssh Geoffery
cd ~/repo/humanoids/<checkout> && make sync      # uv, not conda, for this repo
```

- GitHub over SSH works from here, so plain `git fetch` / `git push` is fine — unlike the LimX
  boxes.
- Checkouts live under `~/repo/humanoids/` (~20 of them); the main one sits on an old branch with
  uncommitted work — **add a worktree, do not check out in it**.
- Holds the Cluster42 staging area (`/home/siwei/cluster42_staging/`) and a `det` client at
  `~/.venv-det/bin/det`.
- Test invocation for this repo: `make test` (= `uv run --extra cpu pytest`). `FORCE_CPU=1`
  forces the CPU path even with the 4080 visible.

Use for: implementation, debugging, lightweight testing, driving Cluster42.
Avoid: long experiments.

## `blackwell` — single-GPU workstation

`blackwell.ias.informatik.tu-darmstadt.de`, user `siwei`, via `ProxyJump ias`. **RTX PRO 6000
Blackwell Max-Q, 95 GiB VRAM, sm_120**, 64 cores, 125 GiB RAM. (Verified 2026-08-07.)

- **It is shared.** Peers deploy to the same worktrees mid-run. Do not assume a tree is clean at
  HEAD — take a `git diff` before and after your own work to establish what code state actually
  produced a result.
- The checkout is a **depth-1 shallow clone**: it shares no history with the local repo, so an
  incremental `git bundle <base>..<branch>` cannot apply. Ship a full bundle and
  `git clone -b <branch> f.bundle <newdir>` into a separate directory.
- Delete macOS `._*` AppleDouble files after untarring, or the worktree reads dirty.
- sm_120 is new enough that the CUDA toolkit/driver check is the first thing to fail — this is
  exactly what `mujoco_warp_io.patch` suppresses. Record whether it was applied.

## `limxH100` — large-scale training server

`172.19.160.21`, user `max`, via `ProxyJump limxH100-jump` (`36.103.167.225`). H100, 80 GB.

- **The jump host exists only to proxy.** Never run installation, training, evaluation or data
  processing on it — everything runs on the H100 host.
- **GitHub fetch does not work here** — the host's key authenticates as a different account with
  no access to the private repo. Use the git-bundle-over-scp route below. (Verified 2026-07-21.)
- The wandb login on the box is the experiment owner's account; use it as-is and do not edit
  `~/.netrc` without being asked.
- 80 GB of GPU memory invites a larger `--env.scene.num-envs` — **verify throughput rather than
  assuming larger is faster.**

Use for: PPO training, hyperparameter sweeps, profiling, benchmarking.

## `limx4090` — parallel evaluation box

`117.50.47.53`, user `ubuntu`, direct. **8× RTX 4090** (launchers address GPU ids `0`–`7`).

- **Uploads are the constraint: ~9–25 KB/s up, ~400 KB/s down** (measured 2026-08-13 and
  2026-09-18). Do not plan any workflow that pushes bulk data here — regenerate on the host.
- Cannot fetch from GitHub and **cannot push**. Deploy code as a gzipped `git diff` patch, or as
  a code-only `git archive`; commit on the Mac first, since `git diff` carries only committed
  changes.
- The checkouts are shallow, so delta bundles fail with "Repository lacks these prerequisite
  commits".
- **The branch label lies.** Trees here are the residue of patch deploys: dirty, detached,
  reporting an old commit while being byte-identical to a current tip. `git rev-parse` and
  `git status` are not evidence — **verify by `md5sum` against the Mac**, per file.
- **There is more than one environment and they differ in their pins.** Check which one a result
  was measured under before reusing or dismissing the host.
- **Python block-buffers stdout when redirected to a file**: a backgrounded sweep can log nothing
  for ten minutes and be perfectly healthy. Judge progress by written artifacts and by the wandb
  run state, not by the log tail.
- It is a **deployment, not a source of truth**. To bring work back:
  `git fetch ssh://limx4090/home/ubuntu/repo/humanoids/<repo> <branch>:<branch>` from the Mac.

## IAS cluster — Slurm

Login/submit node `mn.ias.informatik.tu-darmstadt.de`, user `ju`, home `/home/ju`, repos under
`/home/ju/repo/humanoids`. Shared BeeGFS filesystem (94 TB, 52 TB free as of 2026-07-23) — no
separate scratch path needed. wandb login present via `~/.netrc`. (Verified 2026-08-05.)

- Partition `main`: **MaxTime 3 days, DefaultTime 1 hour.**
- **Node throughput varies enough to change job sizing** — up to ~1.5× between nodes of the
  supposedly same class. Size the wall-clock off the *slowest* node you have seen, not the
  average: a sweep submitted at the average time limit lost three of five runs to the wall clock.

See *Job submission — IAS* below.

## Cluster42 — Determined

Master `https://login01.ai.tu-darmstadt.de:8080`, Determined 0.35.0, user `sj12xefi`, resource
pool `42_Compute`. 73 agents, A100-SXM4-80GB, 8 slots per node. (Verified 2026-08-08; access rule
2026-08-14.)

- **Drive it from `Geoffery` or `blackwell` only** — standing instruction.
- **Access is `det`-only.** Port 22 on the login node answers and offers password auth, but this
  account cannot use it. Never plan `rsync`/`scp` straight to the shared filesystem.
- Login is interactive only; the token caches to `~/.config/determined/auth.json` (**not**
  `~/.determined/`), and each host caches its own.
- **The workspace has a slots limit of 8** — one node. Plan single-node runs.
- **Storage: use the workspace fileset, never the user home.** The user fileset has a hard **2 GB**
  quota; the workspace fileset is provisioned at **~14.6 TB**. `df` reports ~328 TB free for every
  fileset — that is filesystem capacity, not your quota. A full fileset fails *silently and
  short*: `dd` reports "0 bytes copied" and rsync dies with `error in file IO (code 11)`, which
  looks like a network fault and is not.
- **A task sees only what you bind-mount.**

See *Job submission — Cluster42* below.

## Rules for all hosts

- **Record which host produced every number.** "Measured on `<host>`" is part of a result.
- **Never assume two hosts have the same environment** — check `uv.lock` resolution and whether
  `mujoco_warp_io.patch` is applied.
- When a host's setup is worked out, write it into its runbook: the recipe that works, *and* what
  broke with its fix.
- **Check free disk before launching anything that writes bulk data** (see *Disk Hygiene*).
- **An unvalidated host is probed and documented before it is used** — GPU, driver, CUDA, disk,
  existing environments, and what is already running under whose name.
- On shared machines, **check running processes before launching** and pin yourself to free
  devices. A solo smoke test under-predicts a shared launch: the same job measured alone can take
  up to ~4× longer beside someone else's, because partitioning GPUs does not partition CPU.
- **A GPU id is not an offset into a job list.** The round-robin idiom `i=$g` is correct only when
  the GPU list starts at 0. Index by worker position and carry the GPU id separately.
- **Warp/torch grab CUDA at import.** Setting `CUDA_VISIBLE_DEVICES` inside `main()` is a silent
  no-op — set it in the environment before the process starts.

## Moving code and data between hosts

| route | code | data |
|---|---|---|
| Mac → `Geoffery` / `blackwell` | `git push`/`fetch` to IAS GitLab; full bundle + `git clone` on blackwell (shallow clone) | `scp`/`rsync` |
| Mac → `limxH100` | **git bundle over `scp`**, then `git fetch /path/to.bundle <branch>:<tmp>` | `scp` |
| Mac → `limx4090` | **gzipped `git diff` patch** (~30 s), or code-only `git archive` | avoid — regenerate on host |
| `limx4090` → Mac | `git fetch ssh://limx4090/...` | `scp` (downloads are fast) |
| `blackwell` ↔ `limx4090` | none — no key, and non-22 ports deliver 0 bytes | **none.** Relay through the laptop sustains ~6–8 KB/s |
| host → Cluster42 | bundle through the `det shell` tunnel | `rsync` through the `det shell` tunnel, ~88 MB/s |

Rules that come out of this:

- **Regenerating on the target usually beats transferring to it.** Motion NPZs and rendered
  videos are cheap to rebuild from the committed CSVs with `csv_to_npz`; a multi-day transfer is
  not.
- **Verify every transfer by hash, never by exit status.** A failed `scp` has left a truncated
  file whose checksum silently differed.
- macOS `rsync` is **openrsync** and rejects `--append-verify`; macOS also has no `timeout`.
- **A file digest is not a content comparison.** Two motion NPZs can differ by SHA and be
  bit-identical trajectories, differing only in a provenance string. Hash the *arrays* (sorted
  keys, dtype + shape + bytes) before concluding anything needs regenerating.

---

# Job submission — IAS cluster (Slurm)

Submit from `mn.ias.informatik.tu-darmstadt.de`. Partition `main`: **MaxTime 3 days,
DefaultTime 1 hour.**

## Required resource block

Every job uses this block. It is a **single-GPU** allocation (2 CPUs, 5 GB/CPU = 10 GB RAM). It
is user-specified — **do not change it without asking.** A multi-seed sweep is *N* separate
single-GPU jobs, not one multi-GPU job.

```bash
#SBATCH --partition=main
#SBATCH -C 'vram48gb|rtx3090|a5000'
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5G
```

## Job template

```bash
#!/bin/bash
#SBATCH --partition=main
#SBATCH -C 'vram48gb|rtx3090|a5000'
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5G
#SBATCH --time=42:00:00          # ALWAYS set this; the default is 1 h. Size off the
                                 # SLOWEST node you have measured, not the average.
#SBATCH --job-name=<name>
#SBATCH --output=/home/ju/repo/humanoids/<repo>/slurm_logs/<name>_%j.log
# NOTE: override --job-name and --output as sbatch CLI flags at submission time;
# they take precedence over these lines (see gotcha 1).

set -euo pipefail

REPO=/home/ju/repo/humanoids/<repo>
EXPECTED_COMMIT=<full-40-char-sha>
SEED="${SEED:?SEED must be set, e.g. sbatch --export=ALL,SEED=4 ...}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-<default>}"

cd "$REPO"
actual_commit="$(git rev-parse HEAD)"
if [ "$actual_commit" != "$EXPECTED_COMMIT" ]; then
  echo "Refusing to run: expected commit $EXPECTED_COMMIT, found $actual_commit" >&2
  exit 1
fi

exec uv run --extra cu128 train <task-id> \
  --env.scene.num-envs "$NUM_ENVS" \
  --env.seed "$SEED" \
  --agent.seed "$SEED" \
  --agent.max-iterations "$MAX_ITERATIONS" \
  --agent.logger wandb \
  --agent.wandb-project <project> \
  --agent.experiment-name <experiment> \
  --agent.run-name "<run>_seed${SEED}" \
  --agent.wandb-tags '("ias","<tag>","<tag>")'
```

The **commit guard is the point of the template** — it refuses to run rather than silently
training the wrong code. Keep it, and bump `EXPECTED_COMMIT` deliberately with a comment saying
what changed between the old and new SHA. For a tracking task, add
`--registry-name <org>/csv_to_npz/<motion>`.

## Submitting and monitoring

```bash
mkdir -p /home/ju/repo/humanoids/<repo>/slurm_logs      # must exist BEFORE submitting

for s in 1 2 3 4 5; do
  sbatch --export=ALL,SEED=$s \
         --job-name=<name>_s$s \
         --output=/home/ju/repo/humanoids/<repo>/slurm_logs/<name>_s${s}_%j.log \
         scripts/<template>.sbatch
done
```

```bash
squeue -u ju                                   # my queue
scontrol show job <jobid>                      # why pending / where it landed
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ExitCode
sinfo -O NodeList,Gres,GresUsed,StateLong      # what is actually free
scancel <jobid>
tail -f /home/ju/repo/humanoids/<repo>/slurm_logs/<name>_<jobid>.log
```

## Gotchas

1. **`#SBATCH` lines are parsed before shell expansion.** A `${SEED}` in `#SBATCH --output=...`
   produces a file literally named `${SEED}`. Pass `--job-name` and `--output` as `sbatch` CLI
   flags instead.
2. **The default time limit is 1 hour**, and the maximum on `main` is 3 days. A job without an
   explicit `--time` dies mid-run.
3. **Size the wall-clock off the slowest node.** Nodes of the nominally same class have measured
   ~1.5× apart; a sweep submitted at the average lost 3 of 5 runs to the wall clock.
4. **A killed run leaves only its last checkpoint** if saves overwrite one file. For a long run,
   either checkpoint densely or keep iteration-numbered checkpoints.
5. **The log directory must exist before submission** — otherwise the job fails at start with no
   useful message.
6. **Verify a new CLI flag is live before burning many hours**: check the startup log echoes it,
   and that the first artifact it should produce appears after the first save interval.
7. **Warp kernel compilation happens on the compute node**, on a cold `~/.cache/warp`. Budget it,
   and do not read the first iteration's timing as steady-state throughput.
8. **Record the job id** alongside the commit, task id and config. Once the scrollback is gone,
   scheduler accounting is the only record.

---

# Job submission — Cluster42 (Determined)

Run every `det` call from `Geoffery` or `blackwell`. Read the local Cluster42 runbook first; this
is the shape, not a substitute for it.

## Login

```bash
export DET_MASTER=https://login01.ai.tu-darmstadt.de:8080
det user login sj12xefi        # interactive only — non-interactive fails on a termios error
```

## Task config template

```yaml
bind_mounts:
  - host_path: /pfss/mlde/workspaces/mlde_wsp_<WORKSPACE>
    container_path: /pfss/wsp          # the large fileset — mount it in EVERY task
    read_only: false
  - host_path: /pfss/mlde/users/<user>
    container_path: /pfss/mlde/users/<user>
    read_only: false

environment:
  image: <registry>/<image>:<tag>
  force_pull_image: true
  environment_variables:
    HOME: /pfss/mlde/users/<user>
    UV_CACHE_DIR: /pfss/wsp/.cache/uv
    WANDB_CACHE_DIR: /pfss/mlde/users/<user>/.cache/wandb
    WANDB_CONFIG_DIR: /pfss/mlde/users/<user>/.config/wandb

resources:
  resource_pool: 42_Compute
  slots: 1                              # workspace limit is 8
```

## Submitting

```bash
det command run \
  --config-file <config>.yaml \
  --workspace-name <WORKSPACE> \        # REQUIRED — see gotcha 1
  --detach \
  bash -lc '
    set -euo pipefail
    set -a; source /pfss/mlde/users/<user>/.secrets/<creds>.env; set +a
    cd /pfss/wsp/repos/<repo>
    test "$(git rev-parse HEAD)" = "<exact-commit>"
    uv sync --extra cu128 --frozen
    exec uv run --extra cu128 train <task-id> <fully-resolved-flags>
  '
```

```bash
det workspace list; det project list <WORKSPACE>; det agent list
det task list; det command list --workspace-name <WORKSPACE>
det task logs <task-id>
```

## Getting data in — the `det` shell tunnel

Tasks cannot reach your hosts (`:22` is closed from inside), and you cannot SSH to the shared
filesystem. Data goes in through a Determined shell whose SSH is tunnelled over the master —
measured at **~88 MB/s**.

```yaml
# shell.yaml — slots: 0 costs no GPU
description: <name>-data-staging-shell
resources:
  slots: 0
  resource_pool: 42_Compute
bind_mounts:
  - host_path: /pfss/mlde/workspaces/mlde_wsp_<WORKSPACE>
    container_path: /pfss/wsp
    read_only: false
```

```bash
det shell start --config-file shell.yaml --workspace-name <WORKSPACE> --detach
det shell show-ssh-command <shell-id>     # reuse its ProxyCommand as rsync's -e transport
# ... rsync -a --partial --info=progress2 -e "$SSHCMD" <files> <user>@<shell-id>:/pfss/wsp/...
det shell kill <shell-id>                 # when the transfer finishes
```

## Gotchas

1. **`det command run` needs `--workspace-name`.** Without it the CLI silently uses workspace 1
   and fails with "resource pool 42_Compute does not exist or is not available to workspace id 1"
   — which reads like a pool problem and is not.
2. **Drop `-tt` from the printed ssh command** when reusing it for rsync; it forces a pty and
   breaks rsync.
3. **Check rsync's own exit status**, not the wrapper's.
4. **Anything not bind-mounted is invisible** inside the task.
5. **Experiment context is capped** (~96 MB); the repo's STL assets alone are large — keep the
   code on the shared filesystem and reach it by bind mount, not by shipping it with the task.
6. **Tokens expire** and are cached per host; expect periodic re-login.

---

# Machines with no scheduler

`Geoffery`, `blackwell`, `limxH100`, `limx4090`: there is no queue, so **you are the queue.**

- Check what is already running (`nvidia-smi`, `ps`) and under whose name, before launching.
- Pin yourself to free devices explicitly, in the environment before the process starts.
- Start long runs under `tmux` (or `nohup`) so a dropped connection does not kill them.
- Report the launcher PID and the child PID after launching, and verify the wandb run is
  `running` by querying the API — not by reading console text.
- **`pgrep -cf train` counts the watching shell itself.** Use `"[t]rain"` or wait on an artifact.

---

# Disk Hygiene

Machines fill up. Reclaiming space is routine work, but deletion is irreversible and most of what
is on disk cannot be re-downloaded — only re-computed or re-collected.

## Check before you launch

- free space on the target host (`df -h`);
- an estimate of what the run will write (checkpoints × save interval × iterations, plus videos);
- that the output directory is on the large filesystem, not a small `$HOME`.

A run that dies at 95 % full has wasted its whole wall-clock.

## Sweep periodically

- stale worktrees whose experiment is finished and recorded;
- superseded motion NPZs and rendered `motion.mp4` files (cheap to regenerate from the CSVs);
- rebuildable caches: `~/.cache/uv`, `~/.cache/warp`, pip/conda caches — always safe;
- local `wandb/` run directories for runs confirmed finished upstream.

Report what was freed and what was kept.

## Before deleting anything

1. **Check for live processes.** Never delete a dataset, cache or worktree a running job
   references — check `ps`, `nvidia-smi`, and the run's own config.
2. **Check for uncommitted work.** `git status` every worktree; save real diffs as patches
   (`git diff --binary`) before removal.
3. **Check for unique git history.** A standalone clone may hold commits no other checkout has.
4. **Check for unique checkpoints, and classify by content, not by filename.** An extension glob
   is not a checkpoint filter — open the file and look at its top-level keys before deleting.
   Policy weights and ONNX exports are small and irreplaceable; rollout dumps are not.
5. **Preserve first, delete second.** Copy weights, patches and small result files
   (JSON/MD/CSV) to an archive, mirror it off the host, and verify by checksum before removing
   the source.

## Ask before deleting research data

Caches and confirmed-redundant copies may be deleted directly. Anything that is a **single copy
of experimental data** — retargeted motions not derivable from a committed CSV, evaluation
artifacts, trained checkpoints — requires explicit instruction, even when a cleanup was requested
in general terms. Present the candidates with sizes and what regenerating them would cost, and
let the user decide.

Prefer deleting the bulk file and keeping its sidecars: manifests, metadata and summaries are
small, and they are what makes the deleted data re-findable.

---

# When Unsure

If the requested change is ambiguous, do not guess. Instead:

1. explain the ambiguity;
2. propose alternatives;
3. recommend the minimal-risk implementation;
4. ask, if the alternatives lead to materially different work.
