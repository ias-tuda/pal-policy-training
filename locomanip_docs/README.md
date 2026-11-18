# Kangaroo loco-manipulation — handover

Goal: a demo where KANGAROO **walks to a pose, stops, and manipulates while standing**.
Walking is one SKEMA skill, `walk_to(x, y, yaw)`; manipulation skills run while the robot
stands. The split follows HOMIE: a lower-body RL policy keeps balance and walks, while the
arms are driven by something else (now a random script, later the WC-c pick-place policy).

Status (2026-11-18): **stage P1 (walk/stand policy) is good enough to build on.** P2–P4 have
not started. The details, all measurements, and the failed runs are in
[`report.md`](report.md). The stage plan is in [`plan.md`](plan.md).

## What is here

| path | what |
|---|---|
| `src/pal_mjlab/tasks/locomanip/kangaroo/env_cfgs.py` | env configs: `pal_kangaroo_locomanip_flat_env_cfg` (P1) and `..._lowspeed_flat_env_cfg` (P1.1/P1.2, slow commands + stepping reward from 0.05 m/s) |
| `src/pal_mjlab/tasks/locomanip/kangaroo/__init__.py` | task ids `Mjlab-LocoManip-Velocity-Flat-Pal-Kangaroo-Grippers[-LowSpeed]` |
| `src/pal_mjlab/tasks/locomanip/mdp/arm_actions.py` | `ScriptedArmJointPositionAction`: arms and grippers follow random minimum-jerk targets. It is an action term with 0 policy dims. |
| `scripts/locomanip_eval.py` | gate tests: `stand` (20 s, zero command, arms moving) and `walk_to` (random target ≤ 2 m, P-controller) on 64 envs |
| `scripts/locomanip_video.py` | demo video: 3 walk_to targets, then standing with arms moving |
| `scripts/pal_onnx_video.py` | reference: runs PAL's shipped walking ONNX policies (from `pal_policy_deployer`) |

Design:
- The policy drives the legs and pelvis only (14 actions). Arms and grippers are not in its
  action space.
- Robot: the grippers tabletop model with G20 arm gains, the same plant WC-c was trained on.
  This lets WC-c drive the arms later without a model change.
- Robustness training: 0–1 kg random payload per hand, 35 % standing commands.

## Best policy

**P1.2, checkpoint `model_6000.pt`**: LowSpeed task, fine-tuned from P1 `model_5000`.

- wandb: `claudik/kangaroo-locomanip`, run `l2pfbq7x` (P1.2). The P1 parent is `vfvrla1a`.
- local copy: `logs/rsl_rl/kangaroo_locomanip/2026-11-17_19-59-37_P1.2-finetune-lowspeed-s1/model_6000.pt`
  on the workstation where it was trained. Checkpoints are not in git; ask for the file or
  take it from wandb.

| test (64 envs, 20 s) | result |
|---|---|
| stand with arms moving, max drift | 8.7 cm, 0 falls |
| walk_to within 5 cm / 5° | 75 % (median 2.8 cm), 0 falls |
| slowest speed it actually walks | ~0.2 m/s forward, 0.3 m/s sideways works |

**Important for whoever writes the `walk_to` skill:** the policy ignores commands below about
0.2 m/s. A plain P-controller therefore stalls short of the target (2–9 % success). Use a speed
floor, `v_min = 0.2 m/s` and `w_min = 0.3 rad/s`, until inside the tolerance, as
`locomanip_eval.py --v-min 0.2 --w-min 0.3` and `locomanip_video.py` do.

## Commands

```bash
make sync   # uv, GPU torch: --extra cu128

T=Mjlab-LocoManip-Velocity-Flat-Pal-Kangaroo-Grippers-LowSpeed
CK=logs/rsl_rl/kangaroo_locomanip/<run>/model_6000.pt

# evaluate (stand + walk_to), ~5 min on one GPU
uv run python scripts/locomanip_eval.py --task $T --checkpoint $CK --v-min 0.2 --w-min 0.3 --out eval.json

# demo video (48 s)
uv run python scripts/locomanip_video.py --task $T --checkpoint $CK --out demo.mp4

# interactive viewer
uv run play $T --checkpoint-file $CK

# P1.2 training as run: fine-tune from the P1 checkpoint, 3000 iterations, ~2.5 h on 4096 envs
uv run train $T --env.scene.num-envs 4096 --env.seed 2 --agent.seed 2 \
  --env.actions.arms.level-ramp-steps 0 --agent.resume True \
  --agent.load-run <P1 run dir> --agent.load-checkpoint model_5000.pt \
  --agent.max-iterations 3000 --agent.logger wandb --agent.wandb-project kangaroo-locomanip
# P1 from scratch: same, with task Mjlab-LocoManip-Velocity-Flat-Pal-Kangaroo-Grippers,
# no --agent.resume and the default arm ramp, 5000 iterations (~3.5 h)
```

Do not run `make test` while training: it re-syncs the venv to CPU torch.

## Lessons (read before changing rewards)

- **Training the low-speed config from scratch (P1.1) collapsed to standing still.** The strict
  tracking std, the stepping reward raised to air time ≥ 0.2 s, and many zero/slow commands
  together make "never step" the best answer for a policy that cannot walk yet. Fine-tuning a
  policy that already walks (P1.2) works.
- **The base config's `air_time.command_threshold = 0.5`** pays for stepping only above
  0.5 m/s. So P1 (and PAL's shipped v315) do not walk below about 0.5 m/s.
- PAL's own policies (`pal_policy_deployer`) were checked. None fits "legs only, with free 7-DoF
  arms and grippers": the lower-body ones were trained on a model without arms, and the
  full-body ones also control the arms.

## Known weaknesses / next steps

1. **Turning on the spot is slow**: 0.4 rad/s commanded gives 0.18 rad/s. It got worse after
   checkpoint 6000. The remaining 25 % of walk_to misses stall up to 18 cm short. Suggested
   fix (P1.3): a short fine-tune from 6000 with more pure-turn commands and a higher weight on
   angular tracking.
2. **P2**: replace the scripted arms with the WC-c pick-place policy (same plant), on the free
   base, balanced by this policy. First test: WC-c arm trajectories replayed during `stand`.
3. **P4**: SKEMA wrappers `walk_to(x, y, yaw)` (controller as above), `pick(θ)`, `place(θ)`.
4. Only flat ground in sim; no sim-to-sim or hardware tests yet.
