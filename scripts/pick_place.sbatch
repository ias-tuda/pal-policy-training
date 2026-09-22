#!/bin/bash
# Pick-and-place v0 training on the IAS cluster.
#
# One job is one seed. A multi-seed campaign is N of these, not one multi-GPU job:
# the resource block below is single-GPU and is user-specified -- do not change it
# without asking (CLAUDE.md, *Job submission -- IAS cluster*).
#
#   sbatch --export=ALL,SEED=1,NUM_ENVS=256,MAX_ITERATIONS=200,RUN_TAG=c0 \
#          --job-name=pp_c0_s1 \
#          --output=/home/ju/repo/humanoids/pal_mjlab_kangaroo_talos/slurm_logs/pp_c0_s1_%j.log \
#          --time=04:00:00 \
#          scripts/pick_place.sbatch
#
# The log directory must exist before submitting, and --job-name / --output must be
# passed as CLI flags: #SBATCH lines are parsed before shell expansion, so a ${SEED}
# in one produces a file literally named '${SEED}'.
#SBATCH --partition=main
#SBATCH -C 'vram48gb|rtx3090|a5000'
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5G

set -euo pipefail

# A batch step runs a non-login shell, so ~/.profile is never sourced and uv -- which
# installs itself to ~/.local/bin -- is not on PATH. Without this the job dies at exec
# with status 127 after the commit guard has already passed, which is a confusing way to
# find out. (Job 157122 did exactly that.)
export PATH="$HOME/.local/bin:$PATH"

REPO=/home/ju/repo/humanoids/pal_mjlab_kangaroo_talos
TASK=Mjlab-PickPlace-Pal-Kangaroo-Grippers

# The point of the guard is that the job refuses to run rather than silently training
# the wrong code. Bump it deliberately, with a note saying what changed.
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT must be set to the full 40-char SHA}"

SEED="${SEED:?SEED must be set, e.g. sbatch --export=ALL,SEED=1 ...}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-20000}"
RUN_TAG="${RUN_TAG:-c1}"
REWARD="${REWARD:-R-A}"
DR="${DR:-DR0}"

WANDB_PROJECT="${WANDB_PROJECT:-oppsie-pick-place-v0}"
EXPERIMENT="${EXPERIMENT:-kangaroo_pick_place}"

# Extra flags for a diagnostic run, e.g. --enable-nan-guard True. Word-split on purpose;
# keep them simple. Whatever goes here is echoed into the log, so a run that was not the
# plain configuration says so in its own provenance.
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "$REPO"
actual_commit="$(git rev-parse HEAD)"
if [ "$actual_commit" != "$EXPECTED_COMMIT" ]; then
  echo "Refusing to run: expected commit $EXPECTED_COMMIT, found $actual_commit" >&2
  exit 1
fi

# The reward variant and the DR level are not CLI flags on the task: the registered id
# carries R-A at DR0, which is what C0 and C1 run. C2 needs a way to select the other
# arms, and that is a decision for the C1 check-in -- refuse rather than train R-A
# under a run name that claims otherwise.
if [ "$REWARD" != "R-A" ] || [ "$DR" != "DR0" ]; then
  echo "Refusing to run: this task id is R-A at DR0; asked for $REWARD at $DR." >&2
  echo "Selecting a variant is a C2 question and is not wired to the CLI yet." >&2
  exit 1
fi

echo "=== provenance ==="
echo "commit      $actual_commit"
echo "task        $TASK"
echo "seed        $SEED"
echo "num_envs    $NUM_ENVS"
echo "iterations  $MAX_ITERATIONS"
echo "reward/DR   $REWARD / $DR"
echo "extra args  ${EXTRA_ARGS:-(none)}"
echo "node        $(hostname)   gpu $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo '?')"
uv pip list 2>/dev/null | grep -E '^(mjlab|mujoco|mujoco-warp|warp-lang|torch) ' || true
# Whether mujoco_warp_io.patch is applied is part of a result's provenance: the patch
# wraps check_toolkit_driver() in a try/except, so an unpatched tree calls it bare.
if grep -qE '^[[:space:]]+warp_util[.]check_toolkit_driver[(][)]' \
     .venv/lib/python*/site-packages/mujoco_warp/_src/io.py 2>/dev/null; then
  echo "patch       mujoco_warp_io.patch NOT applied"
else
  echo "patch       mujoco_warp_io.patch applied, or upstream moved -- check"
fi
echo "=================="

# shellcheck disable=SC2086  # EXTRA_ARGS is meant to word-split.
exec uv run --extra cu128 train "$TASK" \
  $EXTRA_ARGS \
  --env.scene.num-envs "$NUM_ENVS" \
  --env.seed "$SEED" \
  --agent.seed "$SEED" \
  --agent.max-iterations "$MAX_ITERATIONS" \
  --agent.logger wandb \
  --agent.wandb-project "$WANDB_PROJECT" \
  --agent.experiment-name "$EXPERIMENT" \
  --agent.run-name "pp_${RUN_TAG}_${REWARD}_${DR}_n${NUM_ENVS}_seed${SEED}" \
  --agent.wandb-tags "(\"ias\",\"pick-place-v0\",\"$RUN_TAG\",\"$REWARD\",\"$DR\")"
