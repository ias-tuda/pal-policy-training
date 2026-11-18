# Kangaroo loco-manipulation for SKEMA — report

| code | title | status | date | plan § | report § |
|---|---|---|---|---|---|
| P1 | Lower-body walk/stand policy with scripted arms | running | 2026-11-17 | §P1 | §P1 |
| P2 | WC-c pick-place on the free base, balanced by P1 | planned | — | §P2 | §P2 |
| P3 | Further skills (place, handover) on P1 | planned | — | §P3 | §P3 |
| P4 | SKEMA skill wrappers (walk_to θ, pick θ, place θ) | planned | — | §P4 | §P4 |

*Where it stands (2026-11-17):* P1 is running.

## P1 — Lower-body walk/stand policy with scripted arms

Status: running.

Done:
- New package `src/pal_mjlab/tasks/locomanip/` (no upstream or shared files edited):
  - `mdp/arm_actions.py` (scripted arm action term, `action_dim` 0)
  - `kangaroo/env_cfgs.py`, `kangaroo/rl_cfg.py`, and registration in `kangaroo/__init__.py`

Validation:
- `uv run list-envs --keyword LocoManip` lists the task.
- An env smoke test with 16 envs, zero actions and level 1 passed:
  - action manager: `joint_pos` 14 + `arms` 0
  - actor obs 114, critic obs 132
  - arm targets move (std 0.06–0.30 rad) and the joints track them (max error 0.1–0.9 rad,
    including the jaw step)
  - no NaN
  - zero-action legs fall, as expected
- `ruff format` / `ruff check` pass on the package.
- A 30-iteration training run at 4096 envs completed at 2.55 s/iteration on the RTX 4070.
- `make test` has not been run yet. It runs `uv run --extra cpu`, which would re-sync this venv to CPU torch under the running job, so it waits until the run ends.

Gate eval at checkpoint 4500 (`scripts/locomanip_eval.py`, play config, 64 envs, 20 s,
arms at level 1, seed 0; JSON in `artifacts/P1-s1/eval_4500*.json`; measured on the local
RTX 4070 while training ran):

| check | controller | result |
|---|---|---|
| stand (bar 1) | twist = 0 | 64/64 upright; drift mean 2.5 cm, max 24.1 cm (fails the < 10 cm bar on the worst envs) |
| walk_to (bar 2) | pure P (k=1) | 0 falls; 2/64 within 5 cm / 5°; median 11.8 cm / 4.6° |
| walk_to | P + speed floor 0.15 m/s | 14/64; median 13.3 cm / 2.9° |
| walk_to | P + speed floor 0.25 m/s, w floor 0.2 | 33/64; median 4.1 cm / 2.9°; max 23.7 cm |

Read: balance under full arm motion and payload is fine. Precision is not. The policy
ignores small twist commands, so a pure P-law stalls short of the target. It also drifts
when commanded to stand or turn in place, which accounts for the 10–24 cm tail. Two
responses: the speed floor in the walk_to controller (now the evaluator's `--v-min/--w-min`),
and P1.1 below.

### P1.1 — low-speed variant

`Mjlab-LocoManip-Velocity-Flat-Pal-Kangaroo-Grippers-LowSpeed`
(`pal_kangaroo_locomanip_lowspeed_flat_env_cfg`):
- The fork's `DualBandVelocityCommandCfg`, with 40 % of moving commands inside
  |v| < 0.3 m/s, |wz| < 0.3 rad/s.
- Tracking std sqrt(0.1) / sqrt(0.25).
- Air-time reward from 0.05 m/s commands on, air time 0.2–0.45 s. The base config uses
  0.5 m/s. These are the values of PAL's shipped lower-body walking policy
  (`stable/lower_body_loco` 639.0.0). PAL's Mar 2026 sweep v200–205 lowered the same
  threshold. It is the only reward threshold where my config differs from theirs.

PAL baseline videos (2026-11-17), from `scripts/pal_onnx_video.py`: PAL's shipped ONNX
policies run in `Mjlab-Velocity-Flat-Pal-Kangaroo-Lower-Body`. Actor obs, joint order and
action order match the ONNX metadata.
- `artifacts/pal_baseline/pal_v315_omni.mp4`: stable lower_body_loco 315, all directions;
  scripted stand / ±x / ±y / turn / forward+turn / 0.1 m/s. No falls.
- `artifacts/pal_baseline/pal_v639_forward.mp4`: stable 639 (B12A2), trained for forward
  0.2–0.3 m/s only, obs history 3. No falls.

P1-s1 was stopped at iteration 5003 (`model_5000.pt` kept as the baseline) so that P1.1
could start at once.

P1.1 result (checkpoint 5000, 2026-11-17 19:55): collapsed to standing.
- Stand bar met (64/64 upright, drift max 7 cm), but only because the policy never steps.
- walk_to: median error 1.43 m, so it hardly moves.
- Speed probe (16 envs, 3 s per command, mean over the last 3 s), commanded → actual:

  | command | P1-s1 model_5000 | P1.1 model_5000 |
  |---|---|---|
  | vx 0.1 / 0.3 / 0.6 | 0.00 / 0.00 / 0.58 | 0.00 / 0.00 / 0.00 |
  | vy 0.3 | 0.05 | 0.00 |
  | wz 0.4 | 0.12 | 0.22 |

  P1 also has a dead band: it walks above ~0.5 m/s, the old air-time `command_threshold`,
  and stands still below it.
- In P1.1 the air-time reward stayed ≈ 0. The strict tracking std (exp(-0.36/0.1) ≈ 0.03 at
  0.6 m/s error) plus 35 % standing and 40 % slow commands give no gradient towards
  stepping before the policy can walk. PAL applies the strict std to walking policies.
- Stopped at ~5200 iterations.

P1.2: the LowSpeed task, fine-tuned from P1-s1 `model_5000.pt` (`--agent.resume`, arms at
level 1, 3000 more iterations, seed 2). Command, code snapshot and log are in
`artifacts/P1.2-s1/`.

Prior work checked (2026-11-17): `pal_policy_deployer` holds ~200 PAL loco ONNX policies
(Sai Kishor Kothakota, Louis Le Lay, Feb–Mar 2026). They were trained with the velocity
task that `pal_kangaroo_flat_env_cfg` in this fork descends from; it was migrated on
2026-04-15. They were not used directly, for three reasons:
- They control arm joints 1–5, so the arms are not free for manipulation.
- They use the base Kangaroo model without grippers.
- Their observation layout predates the migration.

Public repo github.com/pal-robotics/pal_policy_deployer (Oct 2026) has a model registry
(`models.yaml`, stable = shipped). Stable walking policies come in two kinds:
- `lower_body_loco` v306–639: legs + pelvis, trained on `kangaroo_lower_body.xml`, which
  has no arms at all.
- `full_body_4dof_loco` v265/v700: the policy also drives the 4-DoF arms.

Neither fits a legs-only policy with free 7-DoF arms and grippers, so none is used as is.
The IAS branch `deployer_ias_kang` (May 2026) adds IAS retrains of loco v74/v77 and
sim-vs-real observation comparisons; it has no lower-body policy.

Smoke test, 512 envs: 33 % standing, and 40 % of moving commands have |vx| < 0.3.
The run is queued to start when P1-s1 exits (waiter PID 1052364).
Files are in `artifacts/P1.1-s1/` (command, code snapshot, log, `started_at.txt`).

Provenance:
- base commit 63ba43ab + the uncommitted package (snapshot in
  `artifacts/P1-s1/locomanip_src.tgz`)
- mjlab 1.6.0, mujoco/mujoco-warp 3.11.0, torch 2.10.0, warp 1.15.0
- host: local workstation, RTX 4070 12 GB
- wandb project `kangaroo-locomanip`, run `P1-scriptarms-s1` (id `vfvrla1a`, state running at launch; 2026-11-17 11:30). PID 991572
- command: `artifacts/P1-s1/command.txt`, log: `artifacts/P1-s1/train.log`

### P1.2 result (2026-11-18)
Training finished 22:28 at iteration 7999 (`logs/rsl_rl/kangaroo_locomanip/2026-11-17_19-59-37_P1.2-finetune-lowspeed-s1`).
64 envs, 20 s, seed 0, `scripts/locomanip_eval.py`; JSONs are in `artifacts/P1.2-s1/`.

| checkpoint | stand drift max | walk_to success (5 cm, 5°), v_min 0.2 / w_min 0.3 | pos err median / max | yaw err max |
|---|---|---|---|---|
| 6000 | 8.7 cm | **75 %** | 2.8 / 18 cm | 5.4° |
| 7000 | — | 48 % | 5.1 / 20 cm | 3.2° |
| 7999 | 8.7 cm | 69 % | 3.0 / 15 cm | 20.6° |

The 6000 stand number is from the default eval, the 7999 one from `eval_7999.json`. No envs fell in any run.

Speed probe, commanded → actual:

| command | P1-s1 5000 | P1.2 6000 | P1.2 7999 | PAL v315 (shipped) |
|---|---|---|---|---|
| vx 0.1 | 0.00 | 0.00 | 0.00 | 0.00 |
| vx 0.3 | 0.00 | 0.22 | 0.25 | 0.00 |
| vx 0.6 | 0.58 | 0.55 | 0.55 | 0.59 |
| vy 0.3 | 0.05 | 0.34 | 0.26 | 0.17 |
| wz 0.4 | 0.12 | 0.18 | 0.04 | 0.03 |

- The forward dead band dropped from about 0.5 to about 0.2 m/s, and sideways stepping now works. Here P1.2 is better than PAL's shipped v315, which has the same 0.5 m/s dead band.
- Turning on the spot is weak and got worse after 6000.
- A walk_to controller needs a speed floor of about 0.2 m/s / 0.3 rad/s. The pure P-law gives 2–9 % success.
- The best checkpoint is **6000**. Bar 1 (stand drift < 10 cm) is met. Bar 2 (walk_to < 5 cm / 5°) is met by 75 % of targets; the misses stall up to 18 cm short.

## P2 — WC-c pick-place on the free base, balanced by P1

Status: planned — no results yet; plan §P2

### P2a — scripted pick-place arm motion replayed while standing (2026-11-18)
No WC-c checkpoint is on this workstation; its runs are on the IAS cluster. The stand-in
is the repo's scripted IK pick-place (`scripts/scripted_pick_place.py rollout --gains G20`):
the same plant, real reach / grasp / lift / carry / place motions, 10/10 placed. Its
joint targets were recorded through `rollout(hook=...)`: 10 episodes, 399 steps each, left
arm + left jaw only (moving ranges 0.47–1.54 rad); cube 50 g.

Replay on P1.2 `model_6000`: 64 envs, zero twist command, 30 s. Each env loops one recorded
episode, with min-jerk ramps from and back to the home pose and a random start phase. The
startup DR adds a 0–1 kg hand payload.

| speed | upright | drift at end, mean / max | peak drift max |
|---|---|---|---|
| ×1 (as recorded) | 64/64 | 4.2 / 12.5 cm | 12.8 cm |
| ×2 | 64/64 | 4.8 / 14.7 cm | 14.7 cm |

No falls, even at double speed. The drift runs a little above the random-arm stand test
(8.7 cm max over 20 s), but this test is 30 s. Conclusion: P1.2 can carry pick-place arm
motions without retraining. For P2b, the base moves up to ~13 cm during a pick, so the arm
controller has to work in the base frame, closed loop, not from a fixed world pose.
Files: `scripts/locomanip_p2/record_pp.py`, `replay_stand.py`; the recorded targets
(`pp_g20_targets.npz`) stay in `artifacts/P2a/` (not committed).

### P2b — a learned grasp policy on the P1.2-balanced robot, then walk_to + grasp (2026-11-18)
There is still no WC-c checkpoint here, so the arm policy is the standing left-arm grasp policy
**vyq9vrpy** `model_4998.pt` (sha256 `658cb435…`). Its task is
`Mjlab-Grasp-LeftArm-Standing-Gravcomp-ArmV36-Slow50-NoTable-Smooth-Wide-Patient-LeanObs-Pal-Kangaroo-Grippers`.
It was trained at commit `e61bd8e9`, which is not on this branch. The scripts run on that code:
`PYTHONPATH=<worktree of e61bd8e9>/src`.

The grasp robot stands on 30× stiffened leg PD and runs at step_dt 0.02. It has 8 actions
(7 for the left arm, 1 for the jaw) and a 38-dim observation. It uses v36 arm gains with body gravcomp.

**Combined env** (`scripts/locomanip_p2/stand_grasp.py`): this is the grasp task's play config with these changes.
- The legs are back on the walking gains. The pelvis is on G20, and the right arm too (P1.2's
  gains): the grasp policy does not drive the right arm.
- The env adds an action term `legs` (P1.2's 14 targets), a `twist` command and a `loco`
  observation group (P1.2's 114 dims). P1.2's jaw observation is shifted to its own default.
- `settled_stance` is removed.
- Body gravcomp stays on the left arm and gripper only. The free joint's gravcomp share is
  clamped to 0 (see "what broke").

Both actors run from their checkpoints, with the MLP and the normaliser. They match the rsl_rl
runner's policy to about 1e-6.

| setup (32 envs) | grasp success | early terminations | base drift mean/max |
|---|---|---|---|
| solo, its own task (rsl_rl runner, 10 s) | **87.5 %** | 4 | max 1.1 cm |
| P1.2 legs, first combined version | 6 % | 4 | 13 / 53 cm |
| + right arm G20, free-joint gravcomp removed | 9 % | 5 | 12 / 36 cm |
| + table +3 / +4.5 cm | 22 / 34 % | 6 / 11 | 11 / 44, 13 / 46 cm |
| **+ table +6 / +7.5 cm** | **44 %** | 12 / 17 | 11 / 39, 10 / 31 cm |
| + table +9 cm | 34 % | 21 | 6 / 36 cm |

P1.2 stands the base about 4.5 cm higher than the grasp robot does, with pitch +1.5° and roll
−4.5°. The grasp policy works in the base frame, so the cube looks lower than any cube it saw
in training (the table-height DR is only ±1.5 cm). Raising the table makes up for this.
Removing the gain/damping/friction DR does not help: 0/32 on the plant before the table raise.

**Demo with two policy slots** (`scripts/locomanip_p2/locomanip_demo.py`), as the user asked.
- Lower body: P1.2 is always on. Skills write only its twist: `walk_to` writes the
  P-controller with a speed floor (0.2 m/s, 0.3 rad/s), every other phase writes zero.
- Upper body (left arm and jaw): a swappable slot, `home` (hold the default) or `grasp` (vyq9vrpy).
- Sequence per env: `walk_to(0, 0, 0)` from (−1.2, 0.4 m, 20°), then 1 s settle, then the grasp
  for up to 12 s. The table is at +6 cm. `out_of_reach` is removed. A termination ends the env (no retry).

| 32 envs, seed 0 | result |
|---|---|
| walk_to within 5 cm / 5° | 22/32, median 3.3 s, median error 2.9 cm / 1.4° |
| grasp success (counted over all 32) | **12/32 (37.5 %)**, median 2.8 s after the grasp starts |
| failures | table contact in the grasp 11, table contact in walk_to 1, arm–arm self-collision 1 |

Videos: seed 0 (arrives in 3.4 s, lifts the cube 6.0 s later) and seed 2 (arrives, misses the
grasp). Both are rendered single-env and are not committed. Copies of all session videos are in
`~/Documents/Office/Images and Videos/HybridT-Paper/SKEMA-GeneralMotions/kangaroo-locomanip_2026-11-18/`.

**What broke**
- **Gravcomp lifts the floating base.** mujoco_warp (`passive.py:581`) applies body gravcomp
  to every DOF of the chain whose joint has no `actgravcomp`, and that includes the free
  joint. The result is an external lift of the base. With full gravcomp the stand drift was
  21.7 / 48.4 cm in 10 s; without it, 2.5 / 3.6 cm. Fix: gravcomp on the left arm only, plus
  `actgravcomp=True` and `actfrcrange≈0` on the free joint (in the spec, before compile).
- **The right arm sagged onto the table.** On v36 gains without gravcomp it fell 7 cm in
  0.2 s. Fix: G20 for the right arm.
- **`settled_stance` with an extra action term stores a collapsed stance** (leg length
  0.117 m). The legs term outputs 0 while the stance settles. So "grasp plant + legs term"
  is not a valid control. Valid controls are the solo task, and the env without the legs term (15/16).
- **The first table sweep ran with the sagging right arm**, and it is discarded.
- **The demo's twist stayed at 0.** `build_cfg` set `rel_standing_envs = 1`, so the command
  term zeroed the twist every step. Fix: set it to 0 in the demo.

**Conclusion and next step.** The loco-manipulation loop works end to end: P1.2 always on, and
the upper slot switched by the skill. The grasp, however, reaches only about 40–44 % against
87.5 % solo. The main failure is table contact while the base sways about 10 cm and rolls
4–5°. The next stage fine-tunes the grasp policy on the P1.2-balanced robot (plan §P2, next step).

Provenance: branch `skema-locomanip`. The grasp code is `e61bd8e9` (branch
`faster-moves-backup`). P1.2 is `locomanip_docs/checkpoints/p1.2_model_6000.pt`. Host: the local
workstation (1 GPU). The commands are in the scripts' docstrings, for example
`stand_grasp.py <grasp ckpt> <p12 ckpt> 32 --seconds 15 --table-dz 0.06` and
`locomanip_demo.py <grasp ckpt> <p12 ckpt> 32`.

## P3 — Further skills (place, handover) on P1

Status: planned — no results yet; plan §P3

## P4 — SKEMA skill wrappers (walk_to θ, pick θ, place θ)

Status: planned — no results yet; plan §P4
