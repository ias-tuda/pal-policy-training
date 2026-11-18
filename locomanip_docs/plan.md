# Kangaroo loco-manipulation for SKEMA — plan

| code | title | status | date | plan § | report § |
|---|---|---|---|---|---|
| P1 | Lower-body walk/stand policy with scripted arms | running | 2026-11-17 | §P1 | §P1 |
| P2 | WC-c pick-place on the free base, balanced by P1 | planned | — | §P2 | §P2 |
| P3 | Further skills (place, handover) on P1 | planned | — | §P3 | §P3 |
| P4 | SKEMA skill wrappers (walk_to θ, pick θ, place θ) | planned | — | §P4 | §P4 |

*Where it stands (2026-11-17):* P1 is implemented and training locally (RTX 4070).

Goal: a sim video of Kangaroo walking to a table, standing, and doing pick-and-place, with
SKEMA choosing the stance pose (walk_to θ = x, y, yaw) and the manipulation parameters.
Approach: decoupled (HOMIE-style). The legs learn to balance under arm motion they do not
control. The arms are WC-c (~95 % fixed-base success) or scripted. The earlier attempt
(motionplanning-sampling `mjlab_kangaroo/`) failed because its balance policy only ever saw
a carry pose (RESULTS.md: 0/64 retained lifts).

Branch `skema-locomanip`, from `origin/kangaroo-pick-and-place` @ 63ba43ab. Worktree
`OopsieData/pal_mjlab_locomanip`. Separate from the Oopsie task register and dashboard.

## P1 — Lower-body walk/stand policy with scripted arms

- Task `Mjlab-LocoManip-Velocity-Flat-Pal-Kangaroo-Grippers`
  (`tasks/locomanip/kangaroo/env_cfgs.py`). It is built on `pal_kangaroo_flat_env_cfg`.
- Plant: the tabletop grippers model with G20 gains, free base. This is WC-c's plant, so P2
  needs no plant change.
- Policy: legs + pelvis, 14 actions (the Lower-Body action set).
- Arms and jaws: `ScriptedArmJointPositionAction`. It gives random minimum-jerk segments
  around the manipulation home pose, with half-widths 0.8/0.8/0.6 rad (shoulder-elbow /
  arm 5 / wrist), 30 % holds and a random open/close of the jaw. Curriculum: amplitude
  level from 0.1 to 1 over 4000 iterations.
- Payload: a point mass of 0–1 kg on each hand. Standing envs: 35 %.
- Posture reward: legs + pelvis only.
- Batch: 1 seed, 4096 envs, 10k iterations, local RTX 4070 (~2.55 s/it, ~7 h).
- Exit bars:
  1. In play, at amplitude level 1 with a standing command, ≥ 95 % of 64 envs stay upright
     for 20 s. Drift is < 10 cm.
  2. walk_to final error is < 5 cm / 5° on 64 random targets within 2 m. The P-controller
     acts on the twist command.
  3. Replayed WC-c arm trajectories, standing: no falls (needs the P1b recorder below).
- P1b (if bar 3 fails or for the gate): record arm joint trajectories from WC-c rollouts
  and add a replay mode to the scripted action, mixed 50/50 with random segments.
- Follow-ups not in P1: upstream leg SysID actuators (#119, #121) are not ported. The leg
  gains stay at the fork's values; revisit them for sim-to-real.

## P2 — WC-c pick-place on the free base, balanced by P1

Run WC-c with P1 on the legs in a combined play env (table + cube, free base). Measure the
success rate against fixed-base WC-c. If it is < ~90 % of the fixed-base rate, fine-tune P1
with WC-c driving the arms (P1 frozen → then joint fine-tune).

Status 2026-11-18: P2a (scripted arm replay, no falls) and P2b (vyq9vrpy grasp + walk_to
demo, 37.5 %) are done; see report §P2. Next: raise the grasp success near the table.

## P3 — Further skills (place, handover) on P1

The same as P2 for the existing fork tasks (pick_place target variants, MugKugel handoff).

## P4 — SKEMA skill wrappers (walk_to θ, pick θ, place θ)

walk_to(θ = x, y, yaw) is the P-controller over P1. Manipulation skills keep their task θ.
Wrap them in motionplanning-sampling. Then record the demo.
