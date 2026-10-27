# Pending runs

## Stack `-Place` (cancelled 2026-10-01, before any useful training)

Task `Mjlab-Stack-Standing-Place-Pal-Kangaroo-Grippers` (commit 6731d16), experiment
`kangaroo_stack_standing`. Cancelled minutes after start to free GPUs for the mug runs;
their log dirs (`2026-10-01_09-20-4*_stack-std-place-*`) hold nothing worth keeping.

| Cancelled job | Run label | Seed | Start from | Iterations |
|---|---|---|---|---|
| 160600 | `stack-std-place-ft1200` | 456 | `2026-09-29_20-36-20_stack-std-scratch_s456/model_1200.pt` | +3000 |
| 160601 | `stack-std-place-ft1200` | 123 | `2026-09-29_20-36-22_stack-std-scratch_s123/model_1200.pt` | +3000 |
| 160602 | `stack-std-place-scratch` | 123 | scratch | 5000 (task default) |
| 160603 | `stack-std-place-scratch` | 456 | scratch | 5000 (task default) |

Relaunch, on the cluster in `~/OopsieData/pal_mjlab_kangaroo_talos` with a clean tree:

```bash
TASK=Mjlab-Stack-Standing-Place-Pal-Kangaroo-Grippers
for src in 2026-09-29_20-36-20_stack-std-scratch_s456:456 \
           2026-09-29_20-36-22_stack-std-scratch_s123:123; do
  run=${src%%:*}; s=${src##*:}
  TASK=$TASK SEEDS=$s MAX_ITERS=3000 RUN_LABEL=stack-std-place-ft1200 \
    bash scripts/slurm_train.sh --agent.resume True --agent.load-run $run \
    --agent.load-checkpoint model_1200.pt
done
TASK=$TASK SEEDS="123 456" RUN_LABEL=stack-std-place-scratch bash scripts/slurm_train.sh
```

Why: the first-generation stack policies grasp (~90%) and hold the cube on the base
(~82%) but almost never open the jaw at the goal, and end stacked ~0% of the time;
xy precision also degraded over training (6.6 mm at 1200 → 24 mm at 5000). `-Place`
adds a held set-down rung, a release rung, pays the stack only after a set-down, and
pays the hand clearing it.
