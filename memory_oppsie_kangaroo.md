# Project memory — Oppsie-for-Kangaroo

*Long-term project state. Read before starting work; update when a decision, milestone or
result lands. Not a conversation log — replace outdated entries rather than appending.
Keep under ~500 lines.*
*Last updated: 2026-09-30 23:35 (**v1 S8 (ball) built, step 0 run, submitted** — `siwei-s8-ball` @ `a9ec317`, jobs 160633–160643 queued, BALL-FT held for S7-8's read; see its row and §9). 2026-09-29 (pick-and-place v0: **S7-6 code on `siwei-s7-6`** 18:12 — S7-5's setup + the real robot's latencies, `LAG=FP+JS` (FoundationPose 5–8 steps on the cube pose, jaw servo lag; from Claudius's `origin/faster-moves`), not launched, plan after S7-5's read. Earlier: **S7-5 launched** 13:40 — jobs 160231–160240 from IAS `-s75` @ `83e9929` (`siwei-s7-5`): RCTP + O (open at goal, θ_hold 0.40) + Z (xy/z 2/2 cm) ×5, RCTP ×4 new seeds, RCTP s2 resumed with O+Z; rocking trace: no press, s2 squeezes to 0.37 rad. Earlier: **S7-4 read** — only RCTPS passes (S unused), RCTP s2 collapsed, press gone under CTP; X pick open. 2026-09-28: **S7-4 launched** — jobs 160053–160062 from IAS `-s74` @ `5b4bf36` (`siwei-s7-4`): X0 = W2-B-G20 + J + M8, EFFORT none/C/CTP, V0.5, S. 2026-09-27: **S7-3 launched** — S7-2's ten runs on S8's G20 plant, jobs
159831–159840 from IAS `-s73` @ `e4cd58a` (branch `siwei-s7-3` = `siwei-s7-2` + `siwei-s8`); **S7-2 read** at
1/2/3/5 k — B places without drift, M7 not learned, V holds at 3.6 s, W2 64/64; decisions for X open. 2026-09-26:
**S7-2 launched** (159038–159047, `-s72` @ `f3bcfd0`); **S8 read at 5 k** (`siwei-s8`: `GAINS`; G20 4/5 seeds place
faster than FR, s3 collapsed); `siwei` fast-forwarded to `4bf8a69` (= `siwei-s7-m6`) 2026-09-25. Earlier: **S7-1
read**: **FR is the base for S7-2 and X**. Reports in
`../docs/reports/pick-place-v0/`).*

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
| Pick-and-place v0 S5 **C0** | ✅ **done 2026-09-22** — jobs 157183-157186, all 200/200 on IAS. Throughput and divergence rate measured, see §6/§7 |
| Pick-and-place v0 S5 **C1** | 🛑 cancelled 2026-09-23 and **invalidated** (157188-157190) — the task was not possible through the action space, §5 −1 |
| Pick-and-place v0 S5 **C2** | 🛑 cancelled 2026-09-23 and **invalidated** (157239-157241), same cause |
| Pick-and-place v0 **S5.0** (make the grasp possible) | ✅ **done 2026-09-23** @ `a566363`: gate 1 scripted pick-and-place 9/10 (10/10 grasped + lifted, now a test); gate 2 reach 43.5 cm; **gate 3 NaN 0.011 %** (IAS job 157575, gate 0.1 %) — `../docs/reports/pick-place-v0/s5-0-report.md` |
| Pick-and-place v0 **S5.1 parallel items** (5, 9, +) | ✅ **done 2026-09-23** on branch `siwei-s5.1-parallel` (`50fed23`, `bbe1cca`, `2348670`), merged into `siwei` — `../docs/reports/pick-place-v0/s5-1-parallel-report.md`. Item 5 **decided (user, 2026-09-23): decision 0002 is the jaw**; Claudius's jaw and his damping/slew add-on not adopted |
| Pick-and-place v0 **S5.1 chain** (8, 2, 3, 1, 4, 6) | ✅ **merged into `siwei` @ `d9ea988`** (fast-forward 2026-09-24, `make test` 182 passed, ruff clean; branch `siwei-s5.1-chain`, code head `b0f8b85`; worktrees removed, their artifacts moved to `../results/S5.1-bench/`, `../results/S5.1-chain/`) — `../docs/reports/pick-place-v0/s5-1-report.md`. Gate 1 10/10, `make test` 182, NaN gate 157646 0.021 % |
| Pick-and-place v0 **c1a screen** | 🛑 cancelled 2026-09-24 10:36 at 12.3–15.2 k (superseded by S5.2). **Final 64-episode eval: grasped/lifted 100 % on 10/10 runs; within 3 cm *horizontally* while held 100 %; placing ≈ 0.** R-A holds ~1 cm above the target, the R-C family hovers 5–8 cm above it; both are per-step optima. **S5.2 base = R-C-strict + PPO-stack** (runner-up R-C). `s5-1-report.md` §6; data in `../results/S5.1-c1a/` |
| Pick-and-place v0 **S5.2** (reward arms for *placing*) | ✅ **read at 3–4 k** (`siwei-s5.2` @ `dad44f1`, eval @ `3323747`; **merged into `siwei` 2026-09-25**, fast-forward `cba928c` → `3323747`, `make test` 256 passed, ruff clean): base R-C-strict + modules M1 (no success termination, per-step `placed` ×8, release ×4), M2 (goal on the `was_lifted` latch ×7 + `goal_near` 3·G_0.03), M3 (lift × (1 − G_0.05(d_g))), M4 (success ×1000, keep termination); all release/placed/success gated on `was_lifted`. Only the full fix (+M1+M2+M3) is strictly increasing over the task stages (1.00 → 22.93 at scripted states); `make test` 252 passed. **Batch** launched 2026-09-24 12:26: 157927–157933, plus 157968–157970, which replace 157934–157936 (held by Slurm after a gn04 start failure; cancelled and resubmitted on the user's instruction at 14:24, `--exclude=gn04`); config verified from each run's `params/env.yaml`. **1 k early read: the full fix places 63/64 on both seeds** (64-episode eval), a real, very fast carry (probe + video); the other arms still hover. `../docs/reports/pick-place-v0/s5-2-report.md`. **4 k (64 eps each): full fix 64/64 on all 3 seeds, 0.61–0.67 s; seed 1 sets it down, seeds 2–3 mostly let go short and the cube slides in (1.2–1.5 m/s). +M1/+M1+M3 place by dropping from 3–8 cm, 3–7 k later; +M3+M4 and the control 0/64.** Base for S7 |
| Pick-and-place v0 **S7-1** (speed bound, release quality, DR ladder; plan S7) | ✅ **read 2026-09-25** (`../docs/reports/pick-place-v0/s7-report.md` §5.5, §7; 64-episode evals at 1/2/3/5(/8) k with the release point, `../results/S7/eval/`): **FA** (slew 0.5) places like the S5.2 full fix (62–64/64 from 2 k) but 0.1–0.3 s slower, held-cube peak 15–30 % lower (0.9 vs 1.0–1.3 m/s; halving joint slew does not halve tool speed), and **still drifts to letting go before arrival by 3 k** (held at goal 0.83 → 0.16, 0.84 → 0.00). **FB** (M5 penalty) same peak speed, releases worst. **FR = FA + M6** (`87b50d6`) **sets the cube down on both seeds at every read** (held at goal 0.95–1.00, released 0.3–0.7 cm out, q 0.91–0.98; to 3.5 k / 5 k) at FA's rate and time → **by the plan's rule S7-2 and X build on FR**; watch its release speed (0.10 → 0.44 m/s). **DR:** placing holds to D4; D5 first to drop (60/64 at 5 k, 61/64 at 8 k). `non_finite_state` ≤ 0.073 %. Early stops (plan rule): D4 158546, FR s1 158572 (13:55); D2/FA s2/D1 met it while the session was paused. Video `../videos/S7-M6/fa_vs_fr_slowmo_x4.mp4` |
| Pick-and-place v0 **S7-2** (baseline B, M7, V, W1/W2; plan S7-2) | ✅ **read 2026-09-27 at 1/2/3/5 k** (40 × 64 eps, local CPU, `../results/S7-2/eval/`, report §10); at 5 k: **B 63–64/64 in 0.87–0.91 s, releases 0.6–0.8 cm off at 0.08–0.11 m/s** (FR's set-down survives B + DR-scene); **B-S (M7) 62–63/64 but set-downs 0–5 %, dwell 0 — M7 not learned**; **B-SV peak cube v ≤ 0.29 m/s, 63–64/64 in 3.6 s; s1 drops from 8 cm** (M7 floors every non-set-down release, so a drop pays what a touchdown release pays); W1-B 61–63/64 0.93–0.95 s; **W2-B 64/64 ×2 0.95–0.97 s**. Across reads: **B does not drift** (0.08–0.15 m/s every read, FR in S7-1 drifted to 0.32–0.44); B-S s1 drifts 0.05 → 0.32; **B-SV s2 switches drop → lowering after 2 k, set-downs 3 → 14 → 23 %, still improving**; the amended early-stop rule would have stopped it at 2 k (reads neither release height nor M7 pass share). Open for Siwei: M7 redesign (keep M6's grade + set-down bonus?), V into X, W2 into X. `non_finite_state` ≤ 0.019 %. D5 diagnosis: failures not at torso-side table shifts, 3/4 fast releases. Launched 2026-09-26 19:00, completed 22:18–02:02, jobs 159038–159047 from IAS `-s72` @ `f3bcfd0` (`siwei-s7-2`) (wandb `mjnoc503` `gn0hc5he` B; `x7kg0zwt` `jtwy6ojh` B-S; `rzi0paxg` `mhx93lbx` B-SV; `4r5td7kk` `w2ubleym` W1-B; `czuojwnd` `c3ak7nor` W2-B); 5 k, reads at 1/2/3/5 k. **Baseline B** = FR on the fda9492 model (arm 5/7 about +z, obs in `base_link`) + table 0.815 m + 10 s + target (0.46, 0.08) ±2 cm table-relative + 38-D actor (`grasped` critic-only) + jaw slew 0.04, at DR-scene. **The plan's target box (0.42, 0.07) ±3 cm failed the corners check** (near the torso the arm cannot place at 0.815 m) — moved by Siwei's decision 2026-09-26. Gates: corners S71 9/10, W1 9/10, W2 10/10; V gate 10/10; jaw-slew bench 64/64, 0 drops; M7 tests; `make test` on `f3bcfd0` **420 passed** (30:37, Mac CPU, 2026-09-26). `../docs/reports/pick-place-v0/s7-2-report.md` |
| Pick-and-place v0 **S8** (stiff pelvis/arm gains *instead of* gravity compensation; plan S8) | 🟡 **read 2026-09-26 — no code meets the plan's bar; check-in pending.** G20: 4/5 seeds place 62–64/64 in 0.52–0.57 s (FR 0.74–0.78), s3 collapsed (11/64, std 0.02); G20s: 63–64/64 ×3 but ~2 m/s carries, elbow saturated ≤ 10 %, ~50 N set-down presses; in-batch G0 reproduces FR (and misses the plan's time bar itself, 0.89 s). The same SLEW gives a ~40 % faster tool on G20 (the soft G0 arm lags its target 0.06 rad); the body holds far better on G20 (≤ 0.024 vs G0's pelvis 0.054–0.087 rad). Eval `--plant` @ `38ca8c5`. Earlier: (`siwei-s8` @ `824aadc`, from `siwei` @ `4bf8a69`; IAS `-s8`). `GAINS` = G0 (today: SysID + gravcomp) / G15 (S4's ×15) / G20 (SysID ×20 = `impedance_control_v37`) / G20s (G20's K, D ×√20); every stiff code has no gravcomp anywhere (`kangaroo_constants.TABLETOP_GAINS`); `make test` 353. **Gates (local CPU):** hold at zero action — G20 elbows **0.0208 rad** (gate 0.02; the plan's own K ≥ 545 vs v37's 523.5), G20s 0.0208 + reset overshoot 0.029, G15 0.038; random left arm — no NaN, G20 holds the pelvis 7× better than G0, G20s rings (4.7 rad/s, elbow saturated 18 %); gripper unchanged without compensation, but under a slew-limited shake G20/G20s lose held cubes over 20 s (48 % / 100 %; 0 at 5 s); scripted 10/10 on every code. **Decision (user, 19:29): G20 + G20s train, G15 dropped.** Gate 5 (IAS, 200 it): G20 0.021 %, G20s 0.000 % `non_finite_state`. **Batch 158811–158820:** G20 ×5, G20s ×3, G0 ×2 (seeds 3–4), 5 k, reads at 1/2/3/5 k |
| Pick-and-place v0 **S7-3** (S7-2 on the G20 plant; plan S7-3) | ⏳ **launched 2026-09-27 20:55**, jobs 159831–159840 from IAS `-s73` @ `e4cd58a` (`siwei-s7-3` = `siwei-s7-2` + `siwei-s8` merged at `ca6d490`), all `running` (wandb `tzxft24a` `t1bbb03t` B; `8pto2td7` `ooqupp6f` B-S; `lvrug3n8` `5vt5ht3y` B-SV; `jhyftu4d` `fi9u8mea` W1-B; `rtqyiybl` `o6ndrzon` W2-B); each = its S7-2 run (159038–159047) + `GAINS=G20`; 5 k, ETA ~03:00 2026-09-28. `kangaroo_grippers_left_arm(gains, base)` puts a W box's wide-scale arm on the plant. **Gates:** hold 0.0208 rad (as S8); jaw shake 5 s 0 drops / 20 s 47 %; NaN 0.004 % (159508) / 0.000 % (159509); `make test` 442; **corners S71 8/10 at DR seed 0** (9/10 seed 1), W1 9/10, W2 9/10, S71 under V 9/10 — the miss is the scripted re-plan's contact check on a drooping arm, not the action space; **waived by Siwei 2026-09-27** ("launch all 10"). `../docs/reports/pick-place-v0/s7-3-report.md` |
| Pick-and-place v0 **S7-5** (O: open at the goal, Z: xy/z goal tolerance, on RCTP; plan S7-5) | ⏳ **launched 2026-09-29 13:40**, jobs 160231–160240 from IAS `-s75` @ `83e9929` (`siwei-s7-5`), all `running` (RCTPOZ s1–s5 `61bagz43` `mu2bqta0` `egpmkqwp` `un7l1iuf` `ln4rqgou`; RCTP s3–s6 `n0nglkrt` `nxl82l4g` `q2gbctmw` `dgbudpwq`; s2OZ `hl04i4jn` = S7-4 job 160058's `model_4999` + 1 k), 5 k, `--time 12:00:00`, ETA ~19:30–21:30. Arm `R-C-strict+M1+M2+M3+M6+J+M8+O+Z`, EFFORT=CTP. **Deviations:** θ_hold 0.40 (plan 0.30; Siwei, on the trace); O/Z are reward modules, not selectors; Z also on M1's `release`. Gates: `test_pick_place_s75.py` 10 passed, `make test` 497; NaN 0.015 % (160230). Prep 3 trace (`scripts/trace_pick_place.py`, `../results/S7-5/trace/`): RCTP s2 holds at the goal with **no press** (commanded TCP above measured, no joint at cap, hinge 0, 0–8 N), jaw squeezed to 0.37 rad (s1 0.30), 4/8 pinched tilted 12–17°. Unread. Plan *S7-5 as implemented* |
| Pick-and-place v0 **S7-8** (stack-task speeds, absolute and delta actions; cube and goal from one wide area; plan S7-8, Siwei 2026-09-30) | ✅ **built, submitted 2026-09-30 18:45, all 18 runs completed and read 2026-10-01** (5 k success after return: ABS-a 78 / 78, ABS-c 64 / 64 / 44 / 69, DEL 47 / 0 / 77 / 56, WC-ABS-c 61 / 0, WCK-ABS-c 39 / 3, DEL on WC / WCK 0 / 4; control S7-6 72 / 92 / 0 / 91; held-cube speed 0.18–0.33 m/s vs 0.9–1.0; bar not met; S7-8 report §9–§11). `siwei-s7-8` @ `9217a60` (from `siwei-s7-7`; not pushed): **`SLEW=stack`** (arm 1–3 at 1/16 of the URDF speed, 4–7 at 1/8, jaw 0.03 rad/step; Claudius's `Slow25`), **`ACTION=abs\|delta`** (`mdp.DeltaRateLimitedJointPositionAction`: target += step · clip(a), box-clipped, bounded integrator, `scale` = step), **`CMD_OBS=act\|cmd`** (commanded target replaces `actions`; 38-D actor), **`SPAWN=WC\|WCK`** / **`TARGET=B\|WC\|WCK`** (cube and goal from one area, ≥ 10 cm apart: `SurfaceOffsetCommand.min_object_distance`, `reset_objects_on_surface(goal_area=…)`), `EPISODE_S`. **Step 0:** Claudius's whole `-Wide` area is 35 % plannable by the scripted planner (plan: 98 %) → cut to `WCK` = x 0.40–0.52, y 0.16–0.26 (120 cm²); scripted stack-speed rollouts on random draws: W2 51 / 64, WCK 49 / 64, WC 8 / 64 (RL places W2 64 / 64: the script is stricter); peak held-cube speed 0.07–0.09 m/s; `EPISODE_S` 55. Runs: RCTP on S7-6's plant, 5 k, 2048 envs: WCK-ABS-c ×2 (160562–3), WCK-DEL ×2 (160564–5), ABS-c ×4 (160566–9), DEL ×4 (160570–3), ABS-a ×2 (160574–5), plus the whole `WC` area WC-ABS-c ×2 (160595–6), WC-DEL ×2 (160597–8) added at Siwei's request. Plan *S7-8 as implemented*, [S7-8 report](../docs/reports/pick-place-v0/s7-8-report.md), `../results/S7-8/`. |
| Pick-and-place **v1 S8 (ball)** (a 6 cm sphere on S7-8's plant; the new `pick-place-v1-objects-plan.md`, *not* the v0 S8 (gains); Siwei 2026-09-30) | 🟡 **built, step 0 run, submitted 2026-09-30 23:30 — gates 160633–160635 done (0 NaN) and runs 160636–160643 running since 2026-10-01 06:23–07:10; read at 2.5–4 k (not final): success after return BALL 92/98 %, BALL-DR 95/83 %, BALL-EQ 0 (stuck)/77 %, MIX 75/62 %; EQ no benefit; report §7.** `siwei-s8-ball` @ **`a9ec317`** (from `siwei-s7-8`; not pushed): `SHAPE` = `cube` (unset = S7-8 exactly) \| `ball` \| `ball-dr` \| `mix` (`pick_place_env_cfg.OBJECT_SHAPES`), `mdp/shapes.py` (`object_half_height`: a sphere's `geom_size` is `(r, 0, 0)` so the cube's `[..., 2]` read put the goal in the table; `PerShape`, `BySize`, `set_object_shape`), reward module `EQ` (`mdp.grasp_height_match`), yaw observation constant `[1, 0]` for a ball, eval reads `roll_after_let_go_median_m` / `release_over_3cm_high`. **Measured (gripper bench, G20):** a 60 mm ball is touched at **0.275 rad**, held at 0.309 (a face-to-face cube 0.359; the plan's 0.13–0.15 was a wrong interpolation) → J's θ_open **0.225, per radius 0.286 / 0.161 at r = 27.5 / 32.5 mm** (`BALL_JAW_OPEN_BY_RADIUS`); no ball lost with the open-jaw pad midpoint 20 mm below to 0 mm of its centre under a 30 s shake at the stack speeds, all lost from +5 mm → **`EQ_OFFSET` −10 mm** (plan: 0); rolling friction **6e-4** (released at 0.05 m/s stops < 1 cm; the table's 1e-4 rolls it 9–13 cm). Scripted stack-speed pick-and-place **60 / 64** (cube control 50 / 64; equator grasp 4 / 12), 31.9 s, 0 clipped steps; `EPISODE_S` 55; Geoffery NaN gates BALL 0, BALL-DR 0, MIX 1 event / 200 iterations. Runs at the plan's default BS (ABS-c, W2 + B) **before S7-8's read** (Siwei): BALL ×2, BALL-EQ ×2, BALL-DR ×2, MIX ×2; **BALL-FT ×2 held** (needs S7-8's best ABS-c W2 checkpoint). `tests/test_pick_place_s8.py` 33. Plan *S8 as implemented*, [S8 ball report](../docs/reports/pick-place-v0/s8-ball-report.md), `../results/S8-ball/`. |
| Pick-and-place v0 **S7-7** (the cube out of the camera's view; plan S7-7, Siwei 2026-09-29) | ✅ **read 2026-09-30 evening** (7-7.0 ✅, 7-7.1 ✅ 38 evals, **7-7.2 ✅ all ten runs + S7-6's RCTP-FJ control read**, report §8–§9, `../results/S7-7/matrix_73.md`): at 5 k under `OCC=arm` O-Z 88/0/0/86 %, O-K 0/83/0/92 %, O-S 0/84 % vs control (RCTP-FJ s1–s4) live 72/91/0/92 %, under `OCC=arm` 17–36 %; re-approach O-K 3 %, O-Z 13 %, control 36 %. Learners pass against the control under `OCC`; **4 of 8 O-Z/O-K seeds stall at 0 %** (control 1/4) → mean 43/44 % < control live 64 %. 7-7.3 (GRU) not triggered. **Open (Siwei): the stall rate — more seeds or resume from a learner.** `siwei-s7-7` (from `siwei-s7-6`; not pushed): `mdp/occlusion.py` `CubeSensor` on S7-6's `_ObjectTracker` — Claudius's line of sight (`camera_view_blocked`, `90c84f5`) + random loss (`eb05a76`) reported with the pose's lag; selectors **`OCC`** `none|arm|hand`, **`OBJ_OBS`** `live|hold|zero|estimate|estimate+|snap`, `OCC_FLAG`; O-K's jaw-only held rule + TCP-riding estimate; noise inside the sensor; critic live + flag + steps unseen. `tests/test_pick_place_s77.py` 18 passed. **Render check** (20 episodes, segmentation from the torso_front mount): model ↔ render 87/85/94/100 % (approach/grasp/carry/retreat); finger and near-face switches change nothing → `OCC=arm`. **Held rule** recalibrated (F1 0.29 → 0.62: stall 1.5 rad/s not 0.15 — the jaw rocks). **7-7.1** (RCTP s1 / s3, 64×64): live 95/94 %, under loss O-H 62/45, O-K 69/55, O-Z 6/14, O-S 0 %; jaw re-closes in 20–48 % of episodes; **O-K+ under `LAG=FP` without occlusion 94/92 % vs live 66/73 %**. **7-7.2**: jobs 160378–160387 (O-Z ×4 `zero`, O-K ×4 `estimate`, O-S ×2 `snap`; `LAG=FP+JS`, RCTP, 5 k; control = S7-6's RCTP-FJ 160367–160370), gate 160377; `-s77` @ `d844b35`. Plan *S7-7 as implemented*, [S7-7 report](../docs/reports/pick-place-v0/s7-7-report.md), `../results/S7-7/`. |
| Pick-and-place v0 **S7-6** (S7-5 + the real robot's latencies; plan S7-6, added 2026-09-29 18:12, Siwei) | ✅ **launched and read 2026-09-30** (jobs 160367–160376, gate 160354, IAS `-s76`; `../results/S7-6/s76_read.txt`, `evalonly_summary.txt`): success after return at 5 k (own LAG) RCTP-FJ 72/92/30/91 % (s3 collapses after 2 k, s4 learns late), RCTP-F 89/84 %, RCTP-J 86/92 % (s2 dips to 8 % at 2 k), FJ-R resume s1/s3 88/80 % at 7 k. Lag-free S7-4 s1 / S7-5 s3 evaluated under FP+JS drop from 98/97 % to 52–66 %. No separate report. Original record: Until S7-5 training ran at 50 Hz with the cube 0–2 steps (0–40 ms) late on the actor (`_DR_OBS_OBJECT_LAG`) and **no jaw lag**; the robot: FoundationPose 100–150 ms, jaw 0.13–0.17 s lag / 1.3–1.7 rad/s (Claudius, 2026-09-29). Selector **`LAG`** = `none` (default = S7-5) \| `FP` (`mdp._ObjectTracker`: one shared tracker, 5–8 steps per env per episode on the cube *pose*, replacing the term-level 0–2; critic live) \| `JS` (`RateLimitedJointPositionAction` servo on the jaw: first-order lag tau 0.11–0.19 s, ≤ 1.25–1.75 rad/s, per physics step; the command is unchanged, the deployer must not add it) \| `FP+JS`; in `pick_place.sbatch` (`_lag<LAG>` name/tag only when set). Ported from `origin/faster-moves` 1d484c4 / 71e1147, adapted; dropouts (eb05a76, occlusion) not ported. `tests/test_pick_place_s76.py` 11 passed; every other test file passed one file per process on the cloud CPU (x86, 7 GB: the whole suite in one process is OOM-killed there; velocity launch without the Rough cases) except `test_pick_place_m7::test_the_set_down_passes_and_is_graded`, which fails identically on unchanged `siwei-s7-5` there (grade 0.7987 vs > 0.8 — a CPU-numerics edge, not S7-6). Before launch: NaN gate with `LAG=FP+JS`, scripted pick-and-place / hand-off under JS. Plan *S7-6* |
| Pick-and-place v0 **S7-4** (J, M8, effort bound C/TP, V0.5, S; plan S7-4) | ✅ **read 2026-09-29** at 1/2/3/5 k (`--plant --handoff`, 64 eps; `../results/S7-4/eval/`, `../docs/reports/pick-place-v0/s7-4-report.md`): all runs open the jaw and lift clear; press gone under CTP (0 % elbow at limit after let-go); only **RCTPS** passes the bar on both seeds (95/97 %, 0.80 s) but S is unused (0 % set-downs); RCTP s1 95 %, **RCTP s2 collapsed**; RCTPV drops from ~7.5 cm; R/RC 0–6 % elbow saturation. Control (prep f): S7-2/S7-3 place 95–100 % by contact but 0–100 % with the jaw open, success 0–3 %. Recommended X = RCTP ×3 (open for Siwei). Launched 2026-09-28 22:57, jobs 160053–160062 from IAS `-s74` @ `5b4bf36` (`siwei-s7-4`), all `running` (wandb R `wvux7qiz` `qf00rgce`; RC `v605mh1s` `6on5wgp5`; RCTP `sefxxlb4` `ahtya5xy`; RCTPV `rs25t1rm` `73mbgnh8`; RCTPS `jqzakkvd` `balocfmi`), 5 k, `--time 12:00:00`. Gates: `test_pick_place_s74.py` 26 passed; W2 corners on G20+CTP ≥ 9/10, T binds in scripted approach 4/10 and carry 1/10; NaN 0.004 % / 0.000 % (160049/160050). **Deviations:** T's band is q + (D/K)q̇ ± τ_cmd/K (the plan's q ± τ_cmd/K capped moves at (τ_cmd−τ_g)/D ≈ 0.2 rad/s on G20); J = entering 'jaw < 0.15 and pads off' (edge/yawed grasps hold at ~0.10 rad). P weight −250. Prep (f) eval job 160063. Plan *S7-4 as implemented* |
| Pick-and-place v0 **PAL sim2sim** (S5.2 policies in PAL's physics simulator) | 🟡 **done 2026-09-25 on `Geoffery`** with pal-policy-deployment @ `8e7024a` + branch `siwei/pick-place-sim2sim` (joint-position node, virtual cube; not pushed) and `scripts/sim2sim/` (`siwei-sim2sim`). Full fix @ 20 k, table + static stand-in cube: **placed 8/9, 9/9, 5/9** (seeds 1–3; MuJoCo 9/9 ×3), grasp 0.18–0.24 s, place 0.60/0.64 s; open-loop replay: joints within 0.015 rad (arm 1–4) / 0.045 rad (wrist) of MuJoCo. No cube stop: seed 1 0/9. `../docs/reports/pick-place-v0/sim2sim-pal-report.md` |
| D10 -- arm/rung selection | ✅ one task id; `REWARD`/`DR` env vars validated at registration (`e27a9b5`) |
| Checkpointing | ✅ `save_interval` 1000 (was 100); `--agent.save-only-last` collapses to one rolling file (`d299961`) |
| Velocity, tracking, reaching tasks (inherited from upstream) | ✅ registered; no result measured *in this fork* |
| Any training run to completion | ❌ none — C0 is a 200-iteration smoke test, not a policy |
| `../docs/` (decisions, reports, status summary, per-task doc) | 🟡 all exist except `roadmap.md` |
| Per-run cost | 🟡 measured on **IAS only** (§6); ❌ on every other host |

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
  (`mjlab/envs/mdp/actions/actions.py:157`), and `clip_actions` defaults to `None`. A scale
  is the half-width of the commandable box, not a step size. Pick-place's scales since S5.0:
  arm 1-7 **0.20 / 0.55 / 0.40 / 0.55 / 0.55 / 0.50 / 1.05**, sized by
  `scripts/scripted_pick_place.py derive` (1.5 x the largest IK excursion over the task's
  waypoints); jaw **0.95** = full range. TCP envelope at |a| <= 1, commanded / physically
  reached: **59.5 / 43.5 cm** (was 13.9 / 14.4 cm at 0.10/0.05; neither S3's 16.3 nor the
  per-task doc's 12.1 reproduces). `scripts/action_envelope.py`.
- **The full-range jaw shakes the four-bar 3.3x harder** (local, 32 envs x 1000 random-action
  steps): passive-joint peak |qvel| 307 rad/s (median 223) vs 93 (76) at jaw scale 0.20;
  0 divergences either way. Expect the NaN rate to rise; S5.0 gate 3 measures it.
- **mjlab position actuators do not clamp their target** (`ctrllimited=False`,
  `mjlab/utils/spec.py`). A target past a joint limit presses into the stop at up to full
  effort. Pick-place clips the jaw target to [0, 0.95] via `ManipulationRobotCfg.action_clip`
  -> `JointPositionActionCfg.clip`.
- **Contact sensor `found` is `(B, P x num_slots)`, primary-major.** Reduce over slots
  before asking "is primary i in contact". The grasp is now **both fingertip pads**
  (`GripperCfg.grasp_geom_regex`), any slot; see §5 −1.
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

- **Checkpoints trained on the pre-`fda9492` MJCF do not match PAL's robot** (every `siwei`,
  `siwei-s5.2`, `siwei-s7` policy as of 2026-09-25): `arm_*_5_joint` and `arm_*_7_joint` turn
  about −z here and +z in PAL's URDF — FK `base_link → arm_left_7_link` is 148 mm / 180° apart,
  0.001 mm / 0.0001° with those two signs flipped — and observations are in the fixed-base root
  link, 0.9 m below `base_link` (`_to_base_frame` uses the root link; `fda9492` moves them to
  `base_link`). A deployment needs both mappings; the sim2sim profile carries them
  (`joint_signs`, `base_in_obs_frame_*`).
- **PAL's physics simulator has no free bodies** (static Bullet world objects only), so a cube
  cannot be simulated there. Its gripper settles 0.0387 rad more closed than commanded, and its
  jaw velocity chatters at ±2 rad/s at 1 kHz (0.1 N·m URDF effort limit in a limit cycle) —
  the S5.2 policies' jaw command flips with it. PAL's `DirectImpedanceControl` has no torque
  clamp at the training effort limits.
- **The S5.2 full fix grasps by collision**: the fingertips reach the cube at ~1.4 m/s and the
  cube, pinned on the table, stops the arm (MuJoCo contact probe, no hand–table contact); seed 1
  runs arm 6 9–18 % above its rated 2.35 rad/s in both simulators.

## 5. What doesn't work / open problems

-1. **THE PICK-AND-PLACE TASK WAS NOT PHYSICALLY POSSIBLE (found 2026-09-23).** C1 and C2 --
   six runs, ~750 M steps each -- all report `Episode_Reward/grasp` exactly 0.0000 at every
   logged point and plateau by iteration ~1500. Cause: **`JointPositionAction` scale is an
   absolute bound, not a per-step increment** (`target = raw*scale + offset`,
   `actions.py:157`, `use_default_offset=True`). So:
   - the jaw reaches only **0.206 of its 0-0.95 rad range**; with the cube held between the
     pads and the jaw fully commanded, the grasp sensor **never fires** (min pad gap 7.64 cm
     on a 5 cm cube). At scale 1.0 it **does** fire (6.85 cm). Geometry is fine; the scale
     was the limit.
   - the arm reaches a TCP envelope of only **12.1 cm** against a **13.5 cm** cube distance.
   The source comment ("0.2 rad *per step* ... covers the 0.95 rad range in five steps")
   assumed incremental control. **C1/C2 are invalid as reward comparisons.**
   **Raising the jaw scale is not free:** it was lowered from 0.50 to 0.20 precisely because
   0.50 peaked the four-bar at 129 rad/s, and that four-bar is what the NaN dumps name.
   Expect the divergence rate to rise from its measured 0.031 %.
   **A second, independent blocker (found 2026-09-23, S5.0): the grasp sensor could not
   register a grasp at all.** It watched 4 geoms (fingertip + inner finger, both sides) x 2
   slots and `_grasped_bool` required all 8; a cube held at the pads touches only the two
   fingertips (measured `found` = [0,0,2,2,0,0,2,2], ~60 N per pad, cube lifted and carried,
   `grasped` False). The "fires at scale 1.0" line above was a pinned-cube test and does not
   hold for a jaw closing on a free cube (it stops at 0.39 rad). **Both fixed in S5.0**
   (approved check-in for the sensor): scales from IK, jaw 0.95 + clip, grasp = both
   fingertip pads. Also fixed: `eval_pick_place.Tally` never counted a placement that ended
   its own episode (`Tally.terminal`).

0. **mujoco_warp does not reset a diverged world; MuJoCo's C engine does** (`mj_checkPos` /
   `mj_checkVel` / `mj_checkAcc` reset `mjData` on a bad state; no such symbol exists in
   `mujoco_warp/_src/`). A blown-up world therefore stays blown up and feeds NaN to the
   learner forever. Handled by `mdp.non_finite_state` (commit `e84d270`); rate measured at
   **0.031 % of episodes**, see §6.
   - **Check what mjlab already does before writing a guard.** `RewardManager.compute`
     sanitises every reward term unconditionally (`reward_manager.py:129`), and
     `ObservationGroupCfg.nan_policy` implements warn/sanitize/error — it just defaults to
     `"disabled"`, which is why the NaN reached rsl_rl and raised. A first version of the fix
     duplicated the reward half for nothing (`9adc904`, removed in `3cc2b1b`).
   - **What diverges is the gripper jaw four-bar, measured, not guessed** (job 157187,
     `--enable-nan-guard`, `scripts/nan_dump_report.py`): the two passive
     `gripper_left_fingertip_*_joint` hit 71 and 63 rad/s and cross 20 rad/s 32 steps before
     the actuated jaw joint, while `arm_left_5_joint` never exceeds 2.8 rad/s. The velocity
     trace oscillates and grows over ~0.18 s — a stiff linkage ringing, not a contact
     impulse. Consistent with the four-bar's 1e-4 armature. **S6 bench (2026-09-23,
     `scripts/gripper_bench.py`, `../docs/reports/pick-place-v0/s6-report.md`) reproduces
     it** (fingertip joints lead) and finds the fix: passive armature 5e-3 + jaw kp 20 -> 8
     + jaw clip 0.78 = peak 2013 -> 8.3 rad/s, CPU divergences 4 -> 0 in 2.56 M physics
     steps. **Decision 0002, LANDED 2026-09-23 at `72d02fb`** (tabletop only: passive
     armature 5e-3, tabletop jaw kp 8, `_JAW_CLOSED` 0.78 = jaw scale = clip). Also measured: softer equalities
     (plan's L4 suspect) are *worse*; kp 20 buckles the linkage on a centred cube (~117 N
     per pad); the jaw physically stops at 0.824 rad, not 0.95 (fingertip-joint limits);
     **S5.0's clip [0, 0.95] on today's model raises CPU divergences 4 -> 13** (p ~ 0.025).
   - **S5.1 bench (2026-09-23) vs Claudius's four-bar fix** (passive damping 0.05 + jaw
     slew + kp 20 + clip 0.95): 0 CPU divergences in 10.24 M steps like 0002 (S5.0 config:
     42), **but on a held cube kp 20 buckles the linkage (100-125 mm) and throws it**, with
     or without his damping/slew; kp 8 alone does not fix it — the passive armature does.
     0002 + his damping + slew ("hybrid") grasps 64/64, lowest ringing (5.5 rad/s).
   - **`gripper_bench.py` applied the jaw rate limit once per `set_jaw` call** until
     `2348670`, so `step`/`shake` never closed a rate-limited jaw. "Rate limit fails the
     grasp" (§9 before, S6 L6, 0002's rate0.1 scan row) was that artefact.
   - **mjlab 1.6.0 `NoiseModelWithAdditiveBias` random-walks its bias**: `reset` does
     `bias = cfg.apply(bias)` and the default op is "add" (15 mm after 1 episode, 65 mm after
     20, bound 8 mm). Use `operation="abs"` on `bias_noise_cfg`. Claudius's branch is
     affected. Draft issue `../docs/upstream/mjlab-additive-bias-random-walk.md`.
   - **mjlab's `dr.encoder_bias` offsets the actuator target, not the observation**
     (`JointPositionAction.apply_actions` subtracts it; only `joint_pos_rel(biased=True)` reports it, and
     our `joint_pos` term is not `biased`). D4/D5's ±5 mrad was a tracking offset the policy saw through.
     Baseline B adds an observation-side bias (±0.01, `abs`) and keeps the event. (S7-2, 2026-09-25.)
   - **A fixed entity's first body is its mocap anchor** (`table/mocap_base`), not `table/table`:
     writing the anchor's `model.body_pos` moves the table only at the next reset and on top of DR-scene's
     draw, which lives on `table/table`'s local offset. Index bodies by name. (S7-2 corners check.)
   - **At the 0.815 m table the arm cannot place near the torso** (script IK, hand straight down): the
     plan's target (0.42, 0.07) ±3 cm was 11–18 % unplannable with the table shift (66 % for a cube at
     +10°). B's target is (0.46, 0.08) ±2 cm (0–2 %). `../results/S7-2/ikmap/`.
   - **On G20 the scripted pick-and-place is pessimistic near the planner's edge** (S7-3 gate 4): `replan_place`
     commands the place pose shifted by the carry-pose tracking bias and demands contact-free IK *at the command*;
     the uncompensated arm droops (at a far-corner carry: arm 1 −0.024, elbow 0.018 rad; TCP 1.1 cm back, 2.0 cm low
     vs G0), so near (0.50, 0.04) the command leaves the contact-free region and the corrections are skipped. It is
     draw-dependent (DR's K × 0.84 on the shoulder): the case places alone and at DR seed 1. 0 clipped actions.
   - **A scripted rollout that times out reports the respawned cube.** The corners check under V at 40 s gave 0/10
     on G0 and G20 alike, "~19 cm off" = the spawn–target distance: every case ran out of episode (V + a 5 cm carry
     in V's slow zone: approach 223, carry 374 steps). Check `ended_by` / `length` before reading a miss. 120 s now.
   - **mjlab string-keyed DR `ranges` ignore `asset_cfg.joint_names`** — the keys match every
     joint of the entity. `tabletop_env_cfg._on_joints` restricts them to the action joints.
   - **DR-robot's `dr.pd_gains` randomises all 30 actuators, not the driven joints** (found
     2026-09-25, plan S8.0): it selects by `asset_cfg.actuator_ids`, which a
     `SceneEntityCfg(joint_names=...)` leaves at `slice(None)`. S7-1's D3–D5 ran with it; S7-2's B
     keeps it on purpose; not fixed. Scope it (`actuator_names=`) before S8.3.
   - **`actgravcomp` alone is unbounded under mjlab position actuators.** mjlab puts the
     effort limit on the actuator `forcerange` (PD only); MuJoCo clamps an actgravcomp joint
     only by the *joint* `actuatorfrcrange`, which mjlab leaves unset (46 N·m through a 1 N·m
     limit). Set both (`kangaroo_constants._route_gravcomp_through_actuators`).
   - **Claudius's trunk/arm 1-4 colliders are inert on his branch** (his robot keeps
     `MANIPULATION_COLLISION`, which zeroes unlisted geoms). Live here since S5.1 item 8: 49
     robot colliders, 0 at rest, none colliding by construction over 2000 random poses.
   - **With live mesh colliders the pick-place target is only reachable elbow-out**: the
     start-pose-seeded IK folds arm 3-4 64 mm into the torso at `carry`; 2/87 random-seed
     solutions are contact-free. `scripted_pick_place._solve` now demands contact-free
     waypoints *and* segments. The home pose (gripper looking forward) makes every top-down
     grasp turn the wrist ~1.5 rad: wrist scale 2.35.
   - **`Metrics/goal_pose/position_error` is 3-D.** For c1a's R-C family it read 5–7 cm while
     the horizontal error was 0.6–1.3 cm: the rest was the hover height. Use `eval_pick_place.py`'s
     transport block (`af35bb2`), which is horizontal and taken while held.
   - **Per env-step the aligned config costs the same on GPU** as the 0002 config (0.101 s,
     ~20 200 env-steps/s on a 3090); PPO-stack's 32 steps/env is the whole iteration increase.
   - **`Metrics/goal_pose/episode_success` is NOT the placed rate and overstates it.** It is
     `at_goal` *latched* -- a pure distance test, no release, no settle -- so a cube shoved
     through the goal region counts forever. `placed` needs `at_goal ∧ released ∧ settled`
     at the same instant. Measured 2026-09-22 (job 157248, C1 seed 1 @ 2300 iters): 2.6 %
     `episode_success` against **0.0 % placed over 200 episodes**, with `grasped` at 0 %.
     Report the placed rate from `scripts/eval_pick_place.py`; use `episode_success` only as
     a between-arm comparator.
   - **Contact sensors are empty until the first step after a reset**, and a teleport does
     not refresh them. Reading `object_table_contact` straight after `reset()` reports "not
     on the surface" for a cube visibly resting on one, which makes `object_lifted` collapse
     to `grasped`. Any test that places an object and reads a contact term must step first.
   - **`ruff format` can silently defeat a string-replacement edit.** Adding
     `support_sensor_name` to `terminations.object_placed` updated the signature but not the
     reflowed call body, so the parameter was accepted and ignored. Caught only because the
     plan mandated a behavioural test. Check the call site, not just the signature.
   - **Time-to-first-failure with n=1 is a badly biased rate estimate.** Three such
     observations gave "one per 0.5 M world-steps"; measuring properly once runs survived
     gave **one per 2.76 M** — 5.5x out.

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

7. **IAS operations (2026-09-24).**
   - *Where progress is:* the job's stdout is block-buffered, so `Learning iteration` lines in
     `slurm_logs/` lag by hundreds of iterations. Read progress from the run's tfevents
     (`~/s52_read.py` on the login node, run under the checkout's `uv run --extra cu128`).
   - *A job that never starts:* a job that fails at start with `user env retrieval failed` is
     requeued **held** (Priority 0, `Reason=user_env_retrieval_failed_requeued_held`) and
     never runs until `scontrol release`. Watch for it; squeue shows it only as PENDING.
     157934–157936 hit this on gn04 at 12:25:25.
   - *Releasing a held job:* on 2026-09-24 the permission guard refused `scontrol release`/`update`
     and the user chose cancel and resubmit (`slurm_logs/resubmit_s52.sh`, `--exclude=gn04`). On
     2026-09-25 S7-1's 158540 hit the same on **gn03** (1 s after dispatch);
     `scontrol update jobid=158540 ExcNodeList=gn03,gn04` + `scontrol release 158540` worked and
     kept the job id. Not a gn04 problem, then: any node can do it.
   - *Login (2026-09-25):* `ssh -i ~/.ssh/id_ed25519 ju@mn.ias...` works; the `ias` alias's RSA
     identity (`id_rsa_ju`) is **refused** by `mn` ("Permission denied (publickey)"), contrary to
     `CLAUDE.md`'s "the same identity file works for both".
   - *Resuming (2026-09-25, `34ddbee`):* `RESUME=<path of model_<it>.pt>` in `pick_place.sbatch`
     (any checkout; the run dir is linked in, names regex-escaped: run names carry `+`);
     `MAX_ITERATIONS` is then the iterations *added*; `ManipulationRunner.learn` restarts
     `common_step_counter`, which the reward curriculum reads (it restarted at 0 before).
   - *S7-1 readers on the login node:* `~/s71_read.py` (tfevents; jobs from `~/s71_jobs.txt`),
     `~/s71_cfgcheck.py <checkout>` (resolved config: terms, gates, slew as `rate_limit`, DR
     events, save interval), `~/s71_wandb.py` (run state by API query).
   - *gn03 again (2026-09-26):* S7-2's 159038–159040 were dispatched to gn03 and held
     (`user_env_retrieval_failed_requeued_held`) within 3 s; released in place with
     `ExcNodeList=gn03,gn04`. **Submit with `--exclude=gn03,gn04` by default.**
   - *S7-2 readers:* `~/s72_cfgcheck.py <checkout>` (plain python3 over `params/*.yaml`),
     `~/s72_wandb.py` (under the checkout's `~/.local/bin/uv run --extra cu128`: `uv` is not on a
     non-interactive ssh's `PATH`); reads locally with
     `../results/S7-2/eval_one.sh <code> <seed> <it>` + `read_table.py` (amended early-stop rule).
   - *Checking the config:* the resolved config is in the run's `params/env.yaml`; verify a
     new arm from it, not from stdout.
   - *Silent logs (2026-09-26):* S7-2's gn05/gn09 jobs (159038–159041) print nothing after wandb's
     setup lines while training normally (tfevents and wandb advance); gn11's print every iteration.
     Judge progress by wandb `_step` or tfevents, never by the Slurm log. Cause not established.

## 6. Current best settings

- **Cost — IAS cluster, `Mjlab-PickPlace-Pal-Kangaroo-Grippers`, measured 2026-09-22**
  (at iteration 199, so warp's ~3.5 min cold compile is excluded; `num_steps_per_env=24`):

  | envs | GPU | steps/s | s/iter | 20 k iters |
  |---|---|---|---|---|
  | 256 | — | 5 786 | 1.14 | 6.3 h |
  | 512 | RTX 3090 | 10 146 | 1.21 | 6.7 h |
  | 1024 | RTX 3090 | 15 320 | 1.60 | 8.9 h |
  | 2048 | RTX A5000 | 18 595 | 2.64 | 14.7 h |
  | 2048 | **RTX 3090** | **19 930 - 21 847** | **2.21** | **~12 h** |

  **The GPU is a bigger factor than `num_envs` co-location.** The three C1 seeds hit
  19 930 / 20 349 / 21 847 steps/s; the two sharing gn03 are not the slow pair. Compare rows
  only within a GPU -- the A5000 row understates 2048 envs by ~18 %.
  **4096 envs OOMs** (`Warp CUDA error 2` at graph launch) on the 24 GB cards
  `-C 'vram48gb|rtx3090|a5000'` selects, at both solver settings. Doubling `num_envs` buys
  1.75x / 1.51x / 1.21x — 2 CPUs cannot feed the GPU faster. Still ❌ on every other host.
- **Solver divergence rate:** 8 events over 22.1 M world-steps, four configurations =
  **0.031 % of episodes, 95 % CI [0.013 %, 0.061 %]**, one per 2.76 M world-steps. The plan's
  abort gate is 0.1 % and this clears it. Raising `iterations`/`ls_iterations` to 100/50 is
  **not** shown to help: the one run that tried it was on different hardware and its lower
  count (1 vs 3) is consistent with the same Poisson rate.
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
| **C0, 2026-09-22** | `Mjlab-PickPlace-Pal-Kangaroo-Grippers`, R-A/DR0, seed 1, 200 iters, jobs **157183-157186** @ `9adc904` | IAS (gn02/gn03/gn08; 3090 + A5000), `mujoco_warp_io.patch` NOT applied | **all four 200/200.** Throughput and divergence rate in §6. The same 1024-env config died at iteration 10 before the guard (157178). `Episode_Reward/grasp` = 0.0000 in every run while `success` fires — the policy slides the cube; see below | wandb `20260922_oppsie` tag `c0r`; `../docs/Mjlab-PickPlace-Pal-Kangaroo-Grippers.md` |
| S5.0 scale derivation, 2026-09-23 | `scripts/scripted_pick_place.py derive`, 9 spawns (centre + 8 corners) x 6 waypoints + target | local Mac | arm scales 0.20/0.55/0.40/0.55/0.55/0.50/1.05; none capped; wrist roll needs most (35 deg to square the pads to a -10 deg cube) | `s5-0-report.md` §2, `../results/S5.0/derive.json` |
| **S5.0 scripted feasibility, 2026-09-23** | `scripts/scripted_pick_place.py rollout --num-envs 10`, R-A/DR0, seed 0, CPU | local Mac | **9/10 placed ∧ lifted, 10/10 grasped and lifted**, 0 clipped actions. Miss: corner (0.44, 0.29, −10°), released 51 mm off. Needed the grasp-sensor fix; open-loop place 1/10, closed-loop on cube pose 9/10 | `s5-0-report.md` §5, `../results/S5.0/rollout_corners.json` |
| S5.0 envelope + shake, 2026-09-23 | `scripts/action_envelope.py --scales {old,new}`, 32 envs, seed 0, CPU | local Mac | reached 14.4 → **43.5 cm**; four-bar peak 93 → **307 rad/s**; 0 `non_finite_state` in 32 k env-steps each | `s5-0-report.md` §6-7, `../results/S5.0/envelope_*` |
| **S5.0 gate 3, 2026-09-23** | `Mjlab-PickPlace-Pal-Kangaroo-Grippers`, R-A/DR0, seed 1, 2048 envs, 200 iters, job **157575** @ `a566363` (jaw 0.95, S5.0 scales, pre-S6 four-bar) | IAS gn06 A5000, patch NOT applied | 200/200 in 11:30, 2.88 s/iter. **2 `non_finite_state` (iters 130, 169) in ~18 700 episodes = 0.011 %**, CI [0.0013, 0.039] % — no rise vs C0 despite 3.3x local shake excitation (policy noise is gentler than uniform random). `object_dropped` now ~46 % of terminations | wandb `20260922_oppsie` `loi5qo2r`; `s5-0-report.md` §8 |
| **S5.1 four-bar bench, 2026-09-23** | `scripts/gripper_bench.py` (fixed @ `2348670`), 256 worlds x 20 s + random at 1024 worlds (10.24 M steps) + 64 x 5 s shake ablation; S5.0 model @ `f44f68a` | local Mac, CPU | divergences in 10.24 M: S5.0 42, 0002 / CK-tabletop / CK-stack / hybrid 0. Held-cube grasp: 0002 and hybrid 64/64, 0 drops (5 s); Claudius's kp-20 jaw buckles or does not hold | `s5-1-parallel-report.md` §3, `../results/S5.1-bench/` (worktree) |
| **S5.1 chain gate 1, 2026-09-23** | `scripted_pick_place.py rollout --num-envs 10` @ `40ed408` | local Mac | 10/10 placed and lifted, 0 clipped, 0 over-slew | `s5-1-report.md` §2 |
| **S5.1 NaN gate, 2026-09-23** | R-A/DR0 s1, 2048 envs, 200 iters, PPO-stack, job **157646** @ `b0f8b85` | IAS gn02 3090 | 3 `non_finite_state` / ~16 300 episodes = 0.021 % (CI 0.004-0.054); 3.24 s/iter | wandb `k3ylpaa1`; `s5-1-report.md` §3 |
| **c1a screen, 2026-09-23** | 10 jobs 157648-157657 @ `b0f8b85` (R-A x2, R-C x2, R-C-strict x3, lift30, 2x PPO-A) | IAS | 🛑 cancelled 2026-09-24 10:36 at 12.3–15.2 k. Final 64-episode eval: grasped/lifted 100 % on 10/10, placing ≈ 0 (R-C s2 1/64, lift30 5/64); R-A holds ~1 cm above the target, the R-C family hovers 5–8 cm | `run_log.md`, `s5-1-report.md` §6, `../results/S5.1-c1a/` |
| **S5.2 ordering, 2026-09-24** | `tests/test_pick_place_s52_ordering.py`, scripted episode, carry 7 cm, CPU @ `dad44f1` | local Mac | full fix: reach 1.00 < carry 4.67 < over 9.32 < held on target 11.96 < released 14.95 < placed 22.93; the arms without M2 keep the cliff (held on target < over) by design; M2 weight 4 dips at ~5 cm, 7 is monotone (+0.001 minimum step) | `s5-2-report.md` §2–3 |
| **S5.2 batch, 2026-09-24** | 10 jobs 157927-157936 @ `dad44f1`: control, +M1 x2, +M1+M3 x2, +M1+M2+M3 x3, +M3+M4 x2; DR0, PPO-stack, 2048 envs, 20 k | IAS gn01/gn02/gn05 3090 | ⏳ running at ~3.0 s/it (wandb `vj54i7b7` `ux98sm8b` `2ss2aikm` `225pqag0` `zs79myfk` `agip67ks` `urkljqie`; resubmitted 157968 `ymw6jrv8`, 157969 `qiwv3laq`, 157970 `t280ohu9` for 157934–157936, which Slurm held). Read at 3–4 k ≈ 15:00–16:30 | `run_log.md`, `s5-2-report.md` §4 |
| **S5.2 1 k early read, 2026-09-24** | full fix 157932/157933 `model_1000.pt`, `eval_pick_place.py` 64 eps + `transport_probe.py` 8 envs + 2-episode video @ `dad44f1` | local Mac, CPU | **placed∧was_lifted 63/64 on both seeds**, 0 slid, time to place 0.69 / 0.79 s; carried off the table 12–16 held steps on a 3–8 cm arc, released 0.4–3.9 cm from the goal; cube peaks ~1.2 m/s (M1 rewards speed). `non_finite_state` 0.03–0.06 % vs ≈ 0.001 % for the other arms | `s5-2-report.md` §5, `../results/S5.2/{eval,probe}`, `../videos/S5.2/` |
| **S5.2 3–4 k read, 2026-09-24 19:00** | all 10 runs' `model_4000.pt`, `eval_pick_place.py` @ `3323747` (adds peak cube/TCP speed while held), 64 eps each, CPU | local Mac | full fix 64/64 ×3 (0.61–0.67 s; s1 sets down, s2/s3 let go short); +M1 0/64 and 64/64, +M1+M3 64/64 and 63/64 (drop from 3–8 cm); +M3+M4 0/64 ×2; control 0/64. Peak held-cube speed 0.86–1.26 m/s mean, 1.46 max; `non_finite_state` ≤ 0.072 % | `s5-2-report.md` §6, `../results/S5.2/eval/` |
| **PAL sim2sim, 2026-09-25** | full fix s1–s3 `model_19999` (+ s1 `model_8000`, `+M1+M3` s1, `+M1` s2), 9 episode specs × 8 s, conditions TC/T/F/TCraw + open-loop replay; matched MuJoCo reference per checkpoint | `Geoffery`, kangaroo:5-dev-ias `sha256:5348a188…`, 3 containers | TC placed 8/9, 9/9, 5/9 (virtual-cube ceiling 9, 9, 2); seed 1 by world: none 0/9, table 3/9, table + stand-in 8/9; `+M1+M3` 9/9, `+M1` 7/9; replay within 0.015/0.045 rad; sustained torque ≤ 0.94 of the training limits, impact impulses up to 1.5× | `../docs/reports/pick-place-v0/sim2sim-pal-report.md`, `Geoffery:~/pal_sim2sim/runs/pick_place/` |
| Pre-guard failures, 2026-09-22 | same task, jobs 157122-157179 | IAS | 157122 `uv: not found` (non-login shell); 157123 NaN @ iter 180; 157124 completed; 157176/7 **OOM at 4096**; 157178 NaN @ 10; 157179 NaN @ 15 | same |

## 8. Compute state

| host | what is installed / checked out there | last verified |
|---|---|---|
| local Mac | git source of truth: `siwei` (S7-2, S7-3, S8, sim2sim merged 2026-09-28) pushed to `origin/siwei` on the user's instruction 2026-09-28. All other worktrees removed 2026-09-28. **Raw artifacts live outside the repo**: `../results/<task>/`, videos in `../videos/<task>/` (S5.0 (+ `eval_smoke/`), S5.1-bench, S5.1-c1a, S5.1-chain, S5.1-merge, S5.2, S6, S7, S7-2, S7-3, S7-4, S7-M6, S7-merge, S8, sim2sim); the main checkout's `artifacts/` moved there 2026-09-29, `artifacts/s7-4` is a symlink to `../results/S7-4` while S7-4 runs; reports and this file point at the new paths | 2026-09-29 |
| `Geoffery` | ~20 checkouts under `~/repo/humanoids/`; **this repo: `pal_mjlab_kangaroo_talos-s77` (2026-09-29, `siwei-s7-7`, uv cu128, RTX 4080; S7-7's evaluations: `~/s77/run_eval.sh`, `~/s77/ckpt/`, `~/s77/eval/`; sync by `git fetch <bundle> siwei-s7-7 && git merge --ff-only FETCH_HEAD`; ssh via `~/.ssh/cm/g.sh` — plain `ssh Geoffery` times out on the gateway ~1 call in 3, and never `pkill -f` a pattern that is in your own ssh command line)**. **PAL sim2sim**: `~/pal_sim2sim/` (catkin `src/` = pal-policy-deployment `siwei/pick-place-sim2sim` + `models/` + `worlds/`; `runs/pick_place/<tag>/`), containers `pal_sim2sim`, `_2`, `_3` from `kangaroo:5-dev-ias` (unprivileged; `sim2sim_container.sh`). Registry login in `~/.docker-ias` (token); the default Docker Desktop store is GPG-locked over SSH | 2026-09-25 |
| `blackwell` | shallow clone under `~/repo/humanoids/`; sm_120, driver-check patch likely needed | — |
| `limxH100` | not verified for this repo | — |
| `limx4090` | not verified for this repo | — |
| IAS Slurm (`mn.ias...`) | `/home/ju/repo/humanoids/pal_mjlab_kangaroo_talos`, branch `siwei` fast-forwarded to **`a566363`** 2026-09-23 (bundle `~/siwei_a566363.bundle`, md5-verified); `uv sync --frozen --extra cu128` current (mjlab 1.6.0, mujoco/mujoco-warp 3.11.0, warp 1.15.0, torch 2.10.0); `mujoco_warp_io.patch` NOT applied; `slurm_logs/` exists. First deployed 2026-09-22 at `e84d270` | 2026-09-23 |
| IAS `-align` checkout | `/home/ju/repo/humanoids/pal_mjlab_kangaroo_talos-align`, cloned from bundle `~/align_s51.bundle` (sha256-verified) on branch `siwei-s5.1-chain` @ `b0f8b85`; own `uv sync --frozen --extra cu128` (same pins), own `slurm_logs/`; jobs pass `REPO=` to `pick_place.sbatch`. The main checkout is not touched by S5.1. Kept as c1a's record (c1a cancelled 2026-09-24) | 2026-09-24 |
| IAS `-s52` checkout | `/home/ju/repo/humanoids/pal_mjlab_kangaroo_talos-s52` on `siwei-s5.2` @ `dad44f1` (full 40-char sha checked by the sbatch commit guard); own `uv sync --frozen --extra cu128` (same pins), `slurm_logs/submit_s52.sh`. Readers on the login node: `~/s52_read.py` (tfevents), `~/s52_cfgcheck.py` (resolved config) | 2026-09-24 |
| IAS `-s7` checkout | `/home/ju/repo/humanoids/pal_mjlab_kangaroo_talos-s7` on `siwei-s7` @ `b50efcb`, deployed 2026-09-25 04:18 (full bundle, sha256-checked, own `uv sync --frozen --extra cu128`, same pins); `slurm_logs/submit_s71.sh`. **Runs S7-1 (158540–158547): do not fast-forward it while they run** | 2026-09-25 |
| IAS `-s8` checkout | `/home/ju/repo/humanoids/pal_mjlab_kangaroo_talos-s8` on `siwei-s8` @ `824aadc` (all S8 jobs ended 2026-09-26 00:47), cloned 2026-09-25 19:23 from `~/s8_824aadc.bundle` (sha256 matched), own `uv sync --frozen --extra cu128` (same pins); `slurm_logs/submit_s8.sh` (`PHASE=nan|batch RUNS="<code>:<seed> …"`); readers `~/s8_read.py`, `~/s8_cfgcheck.py` (prints the plant's K/D and the code), `~/s8_wandb.py`; jobs in `~/s8_jobs.txt`. **Runs S8 (158806–158807 done, 158811–158820): do not redeploy while they run** | 2026-09-25 |
| IAS `-s7m6` checkout | `/home/ju/repo/humanoids/pal_mjlab_kangaroo_talos-s7m6` on `siwei-s7-m6` @ `87b50d6` (M6), deployed 2026-09-25 10:09 (full bundle, sha256-checked, own uv sync, same pins); `slurm_logs/submit_s71_fr.sh`. **Runs FR (158572–158573): do not fast-forward it while they run** | 2026-09-25 |
| IAS `-s72` checkout | `/home/ju/repo/humanoids/pal_mjlab_kangaroo_talos-s72` on `siwei-s7-2` @ `f3bcfd0`, deployed 2026-09-26 11:10 at `2d727b8` (full bundle, sha256-checked, own `uv sync --frozen --extra cu128`, same pins), fast-forwarded 18:59 (incremental bundle, sha256-checked); `slurm_logs/submit_s72.sh`, `s72_cfgcheck_0926*.txt`. **Runs S7-2 (159038–159047): do not fast-forward it while they run** | 2026-09-26 |
| IAS `-s77` checkout | `/home/ju/repo/humanoids/pal_mjlab_kangaroo_talos-s77` on `siwei-s7-7` @ `d844b35`, cloned 2026-09-29 from `-s75` + `~/s77_ias_incr.bundle` (sha256 `fb334842…`) + `~/s77_ias_incr2.bundle` (`7cfd94c3…`), own `uv sync --frozen --extra cu128` (mjlab 1.6.0, mujoco/mujoco-warp 3.11.0, warp 1.15.0, torch 2.10.0); `slurm_logs/submit_s77.sh`; jobs `~/s77_jobs.txt`. Runs S7-7's 160377–160387 | 2026-09-29 |
| IAS `-s78` checkout | `/home/ju/repo/humanoids/pal_mjlab_kangaroo_talos-s78` on `siwei-s7-8` @ `9217a60`, cloned 2026-09-30 from `-s77` + `~/s78_final2.bundle` (sha256 `3cd1a899…`), own `uv sync --frozen --extra cu128`; `slurm_logs/submit_s78.sh`; jobs `~/s78_jobs.txt`; readers `~/s78_wandb.py`, `~/s78_gate.py`. Runs S7-8's 160542–160575 | 2026-09-30 |
| IAS `-s8ball` checkout | `/home/ju/repo/humanoids/pal_mjlab_kangaroo_talos-s8ball` on `siwei-s8-ball` @ `a9ec317`, cloned 2026-09-30 from `-s78` (own uv venv, `uv sync --frozen --extra cu128`); bundle `../results/S8-ball/s8_final.bundle`; jobs `~/s8_jobs.txt`, submit script `~/submit_s8.sh` (`../results/S8-ball/submit_s8.sh`). Geoffery: `-s77` checkout on `siwei-s8-ball`, `~/s8/{gate,step0,run_eval}.sh`. |
| IAS `-s75` checkout | `/home/ju/repo/humanoids/pal_mjlab_kangaroo_talos-s75` on `siwei-s7-5` @ `83e9929`, cloned 2026-09-29 from the `-s74` checkout + `~/s75_83e9929_incr.bundle` (sha256 `abcff11e…` matched; changed files checksum-equal to the Mac); own `uv sync --extra cu128`, pins as `-s74`, `mujoco_warp_io.patch` not applied; `slurm_logs/submit_s75.sh`; readers `~/s75_read.py`, `~/s75_wandb.py`; jobs `~/s75_jobs.txt`. **Runs S7-5 (160231–160240): do not touch while they run.** s2OZ's run dir is symlinked from `-s74`'s | 2026-09-29 |
| IAS `-s74` checkout | `/home/ju/repo/humanoids/pal_mjlab_kangaroo_talos-s74` on `siwei-s7-4` @ `5b4bf36`, cloned 2026-09-28 from `~/s74_d415d98.bundle` (sha256 `67c692f8…` matched), fast-forwarded by `~/s74_5b4bf36_incr.bundle` (`8da9e004…`); own uv sync (pins as `-s73`, `mujoco_warp_io.patch` not applied); `slurm_logs/submit_s74.sh`, `s74_prepf_list.txt`; readers `~/s74_read.py`, `~/s74_cfgcheck.py` (+ `_0928.txt`), `~/s74_wandb.py`; jobs `~/s74_jobs.txt`. **Runs S7-4 (160053–160062) and eval 160063: do not fast-forward it while they run** | 2026-09-28 |
| IAS `-s73` checkout | `/home/ju/repo/humanoids/pal_mjlab_kangaroo_talos-s73` on `siwei-s7-3` @ `e4cd58a`, cloned 2026-09-27 14:26 at `3396b99` from `~/s73_3396b99.bundle` (sha256 `aab0c048…` matched), own `uv sync --frozen --extra cu128` (same pins), fast-forwarded ~15:08 (incremental bundle, sha256 `7e22b969…`); `slurm_logs/submit_s73.sh` (`PHASE=nan|batch`), `s73_*_0927.txt`; readers `~/s73_read.py`, `~/s73_cfgcheck.py` (S7-2's config check + S8's plant), `~/s73_wandb.py`; jobs in `~/s73_jobs.txt`. **Runs S7-3 (159831–159840): do not fast-forward it while they run** | 2026-09-27 |
| Cluster42 | staging area on `Geoffery`; nothing deployed for this repo | — |

## 9. Next steps

- **S8 ball (submitted 2026-09-30 23:30, queued):** when 160633–160635 (gates, `non_finite_state` ≤ 0.1 %) and 160636–160643 start, do the launch checks (GPU, wandb `running` by API — tags `shapeball` / `shapeball-dr` / `shapemix`, `pad_height` in the reward table of BALL-EQ, `params/env.yaml`); read at 1 / 2 / 3 / 5 k with `../results/S8-ball/run_eval.sh` (`EPISODE_S=55`, `SHAPE=…`, `--plant --handoff`; new reads: where the ball is let go, roll after the let-go, drops in the carry, stuck at 3 k, MIX per shape). **BALL-FT ×2** after S7-8's read: `FT_CKPT=<model_4999.pt of the best ABS-c W2 run> PHASE=batch bash submit_s8.sh BALL-FT`. If S7-8 prefers DEL or `WCK`, relaunch the eight with `ACTION` / `SPAWN` / `TARGET` changed. Open: the real ball (mass, material, rolling friction), the real jaw's touch angle, the eval's `grasp_pad_dz_median_m` is the instantaneous pad height (open-jaw value minus ~10 mm).
- **S7-8 (read 2026-10-01; next: resume WC-ABS-c s1 / DEL s1 / s4 to 10 k, more `WC-ABS-c` seeds, placement precision; the gates / launch checks below are history):** gates 160542 / 160543 / 160560 / 160561 (`non_finite_state` share of terminations; bar ≤ 0.1 % of episodes at 10 s episodes ≈ 0.5 % at 55 s), then launch checks for 160562–160575 (wandb tags `actdelta` / `cmdobscmd` / `tgtWCK` / `ep55`, first iteration, no traceback); read WCK-ABS-c at the newest checkpoint in the morning with `../results/S7-8/run_eval.sh` (`EPISODE_S=55`, hand-off timeout 46 s) and sim2sim on Geoffery before the robot; 1 / 2 / 3 / 5 k reads for all; bar (draft): success ≥ 90 % on ≥ 3 of 4 seeds of ABS-c or DEL within 5 points of the control, peak held-cube speed ≤ 0.2 m/s, re-approaches ≤ 5 %, no seed stuck at 3 k. Open: the full `WC` area (Siwei), the DEL deployer (integrator + box), γ.
- **S7-7 (submitted 2026-09-29 23:09, queued):** when 160377 (gate, `non_finite_state` ≤ 0.1 %) and 160378–160387 start, do the launch checks (GPU, wandb `running` by API — tags `occarm` / `objestimate` / `objzero` / `objsnap` —, first iteration, no traceback); read at 1/2/3/5 k with `eval_pick_place.py --plant --handoff` on Geoffery, **with and without `OCC`** (`OCC=arm OBJ_OBS=<arm>` and `OCC=none OBJ_OBS=<arm>` = nothing hidden, flag 1; O-S: `OCC=none OBJ_OBS=snap`); the S7-6 control's read as the `OCC=none` reference; bar: under `OCC` within 3 points of the same policy without, without `OCC` not below the control. 7-7.3 (GRU) only if O-K misses; an `estimate+` arm if O-K shows the lag-removal effect (7-7.1: 94/92 % vs 66/73 % under `FP`). Open: FoundationPose's lost-cube output (Siwei → Claudius), the camera orientation, the deployer-side parity test.
- **S7-6 (code only, `siwei-s7-6`):** Siwei writes the S7-6 plan after S7-5's read — which arm(s) under `LAG=FP+JS` (and whether `FP` / `JS` alone as controls), from scratch or resumed, ~10 k + 200-episode eval + decision record (the old X). Gates first: 200-it NaN run with `LAG=FP+JS`; scripted pick-and-place and the eval's hand-off under JS.
- **S7-5 (running, jobs 160231–160240, ETA ~19:30–21:30 2026-09-29; s2OZ ~15:00):** read at 1/2/3/5 k with `eval_pick_place.py --plant --handoff` on `siwei-s7-5` (both tolerances, tilt at let-go); stuck = placed(J) < 10 % at a read; bar: no RCTPOZ run stuck at 3 k, every seed ≥ 95 % S7-4 success at 5 k, no loss vs RCTP in release height / set-down force. Then S7-6 = X (~10 k, 200-episode eval, decision record superseding 0001).
- **S7-4 read (2026-09-29, superseded by S7-5):** Siwei picks X (recommended RCTP ×3; RCTPS is the plan-rule pass but S is inert; V's slow zone invites a 7.5 cm drop); then S7-5 (~10 k, 200-episode eval, decision record superseding 0001).
- **S7-3 (running, jobs 159831–159840, ETA ~03:00 2026-09-28):** read at 1 / 2 / 3 / 5 k with S7-2's list + `--plant`
  (eval on `siwei-s7-3`), re-evaluate S7-2's checkpoints on the same eval, then the S7-2 vs S7-3 comparison (plan
  S7-3, *Comparison*), which decides S7-4's plant. Open: the scripted re-plan's contact check at the reached pose.
- **S8 (read, check-in pending):** no code met the plan's bar (G20: one collapsed seed of 5;
  G20s: violent). Plan-strict: G0 stays and 0001 gets an addendum. Alternative: keep G20 as a
  candidate — more seeds for the collapse rate, S8.3 at DR-robot (after scoping DR-robot's
  `pd_gains`), and a slew re-derived for the stiff plant. 1/2/3 k evals not run (checkpoints in
  `../results/S8/ckpt/`, `../results/S8/schedule.sh jobs_early.txt`).

0. ~~Read the c1a screen~~ done: 3–4 k read, then the final read after cancelling at 12–15 k
   (`s5-1-report.md` §6). It is the input to S5.2.
1. ~~Merge `siwei-s5.1-chain`~~ done, `d9ea988`. ~~S5.2~~ read: the full fix is S7's base (`s5-2-report.md` §6).
   ~~Merge `siwei-s5.2`~~ done (fast-forward, `3323747`, 2026-09-25). ~~S7-1~~ read: S7-2 and X build on FR.
   ~~P2 / S7-2~~ launched 2026-09-26, ~~read~~ 2026-09-27 (S7-2 report §10; reads re-runnable with
   `../results/S7-2/eval_all.sh` + `read_table.py`), ~~D5 diagnosis~~ done (§8). **Next: Siwei's decisions for X**
   — M7's design (as specified it floors every non-set-down release; options in the report's *Open*), V into X
   (3.6 s per pick-and-place), W2 into X; S7-3 (G20 plant) runs in another session (`-s73`, `siwei-s7-3`) and
   re-evaluates S7-2's checkpoints with its merged eval. Also: prep 7 (grip force vs cube size), asking PAL for
   the gripper's force, merging `siwei-s7-2` / `siwei-s8` into `siwei` (the user's call).
2. **The jaw misses its own 0.01 % NaN bar** on both 0002 configs (157635 0.026 %, 157646
   0.021 %); the plan's 0.1 % gate passes. Decision 0002 records it; not blocking the screen.
3. Write down **what Oppsie is** (§1 here and `CLAUDE.md`'s header gap note), and its
   non-goals. Still the oldest open item in the repo.
4. Create `../docs/roadmap.md` — named in `CLAUDE.md` as the entry point, still missing.
5. Measure per-iteration cost on `Geoffery` / `blackwell` / `limxH100`; §6 covers IAS only.
6. `tests/` covers the manipulation MDP terms only through the pick-place suites; no
   `Mjlab-Lift-*` / `Mjlab-Stack-*` training run is recorded, and their `env_spacing` bug
   means any pre-`2f4671a` result against them would have been invalid anyway.
