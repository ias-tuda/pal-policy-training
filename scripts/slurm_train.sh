#!/usr/bin/env bash
# slurm_train.sh — submit one SLURM job per seed, each running an mjlab training run.
#
# Usage:
#   bash scripts/slurm_train.sh [EXTRA_TYRO_OVERRIDES...]
#
# Extra positional arguments are forwarded verbatim to `train`, so any tyro flag
# works, e.g.:
#   bash scripts/slurm_train.sh --agent.algorithm.entropy-coef 0.01
#
# Environment variables:
#   TASK            mjlab task id (default: Mjlab-Stack-Pal-Kangaroo-Grippers)
#   SEEDS           Space-separated list of integer seeds (default: 42). One job per seed.
#   NUM_ENVS        Parallel envs (default: 4096). The manipulation env cfgs default to
#                   num_envs=1, so this MUST be passed or the job trains on one env.
#   MAX_ITERS       --agent.max-iterations. Unset -> the task's own default (20000).
#   RUN_LABEL       Job name / run name / wandb group prefix (default: derived from TASK)
#   WANDB_PROJECT   --agent.wandb-project (default: pal-mjlab)
#   WANDB_GROUP     Shared wandb group for all seeds of this launch
#                   (default: RUN_LABEL_<timestamp>); exported as WANDB_RUN_GROUP.
#   WANDB_MODE      Passed through if set (e.g. offline, disabled).
#   VIDEO           1 -> record training videos (--video True). Default 0.
#   TIME            SLURM wall-clock limit (default: 24:00:00)
#   CPUS            CPUs per task (default: 4)
#   MEM_PER_CPU     Memory per CPU in MB (default: 8192 -> 32 GB total at CPUS=4)
#   GPU             GPU gres spec passed to --gres=gpu: (default: 1)
#   PARTITION       SLURM partition name (default: main)
#   EXCLUDE         Comma-separated nodes to exclude (default: dgx-station)
#   GPU_CONSTRAINT  SLURM --constraint feature for GPU VRAM (default: vram24gb)
#   MAIL            e-mail for SLURM notifications (optional)
#   PROJECT_DIR     Absolute path to repo root (default: this script's parent dir)
#   DRY_RUN         1 -> print the generated job script instead of submitting it
#   ALLOW_DIRTY     1 -> bypass the clean-tree / lock-file checks
#
# Examples:
#   bash scripts/slurm_train.sh
#   SEEDS="42 123 456" bash scripts/slurm_train.sh
#   NUM_ENVS=2048 MAX_ITERS=5000 RUN_LABEL=stack-short bash scripts/slurm_train.sh
#   DRY_RUN=1 bash scripts/slurm_train.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Configuration (overridable via env)
# ---------------------------------------------------------------------------
TASK="${TASK:-Mjlab-Stack-Pal-Kangaroo-Grippers}"
SEEDS="${SEEDS:-42}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-}"
# Default label: the task id minus the boilerplate, lowercased
# (Mjlab-Stack-Pal-Kangaroo-Grippers -> stack-kangaroo-grippers).
DEFAULT_LABEL="$(echo "${TASK#Mjlab-}" | sed 's/-Pal-/-/' | tr '[:upper:]' '[:lower:]')"
RUN_LABEL="${RUN_LABEL:-${DEFAULT_LABEL}}"
WANDB_PROJECT="${WANDB_PROJECT:-pal-mjlab}"
WANDB_GROUP="${WANDB_GROUP:-${RUN_LABEL}_$(date +%Y%m%d_%H%M%S)}"
WANDB_MODE="${WANDB_MODE:-}"
VIDEO="${VIDEO:-0}"
TIME="${TIME:-24:00:00}"
CPUS="${CPUS:-4}"
MEM_PER_CPU="${MEM_PER_CPU:-8192}"
GPU="${GPU:-1}"
PARTITION="${PARTITION:-main}"
EXCLUDE="${EXCLUDE:-dgx-station}"
GPU_CONSTRAINT="${GPU_CONSTRAINT:-vram24gb}"
MAIL="${MAIL:-}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DRY_RUN="${DRY_RUN:-0}"

# The job prepends .venv/bin to PATH rather than resolving the env at runtime, so the
# compute node needs no network. That means `uv sync` has to have happened already.
if [ ! -x "${PROJECT_DIR}/.venv/bin/train" ]; then
    echo "ERROR: ${PROJECT_DIR}/.venv/bin/train not found." >&2
    echo "       Run 'make sync' (uv sync --all-extras --all-packages) on a login node first." >&2
    exit 1
fi

# Refuse to launch from a dirty tree so the recorded git hash matches the code.
# shellcheck source=scripts/require_clean_tree.sh
source "${SCRIPT_DIR}/require_clean_tree.sh"
require_clean_tree "${PROJECT_DIR}"

# Serialize extra tyro overrides into a single string for embedding in the job script.
EXTRA_ARGS="${*}"

LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}/slurm"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
read -ra SEED_ARRAY <<< "$SEEDS"
N_SEEDS="${#SEED_ARRAY[@]}"

echo "========================================"
echo "  mjlab SLURM training launcher"
echo "========================================"
echo "  Task:         $TASK"
echo "  Seeds:        $SEEDS"
echo "  Jobs:         $N_SEEDS  (one per seed)"
echo "  Num envs:     $NUM_ENVS"
echo "  Max iters:    ${MAX_ITERS:-<task default>}"
echo "  Run label:    $RUN_LABEL"
echo "  WandB proj:   $WANDB_PROJECT"
echo "  WandB group:  $WANDB_GROUP"
echo "  WandB mode:   ${WANDB_MODE:-<default>}"
echo "  Video:        $VIDEO"
echo "  Wall time:    $TIME"
echo "  GPUs:         $GPU"
echo "  CPUs/task:    $CPUS"
echo "  Mem/CPU:      ${MEM_PER_CPU}M"
echo "  Partition:    $PARTITION"
echo "  Exclude:      $EXCLUDE"
echo "  GPU constr.:  $GPU_CONSTRAINT"
echo "  Project dir:  $PROJECT_DIR"
echo "  Extra args:   ${EXTRA_ARGS:-<none>}"
echo "========================================"
echo ""

# ---------------------------------------------------------------------------
# Submit one job per seed
# ---------------------------------------------------------------------------
JOB_IDS=()

for SEED in "${SEED_ARRAY[@]}"; do
    # Assemble the full `train` argv here rather than emitting conditional continuation
    # lines inside the heredoc: an unset MAX_ITERS would otherwise leave a dangling
    # backslash and a blank line, breaking the command.
    TRAIN_ARGS="\"${TASK}\""
    TRAIN_ARGS+=" --env.scene.num-envs ${NUM_ENVS}"
    TRAIN_ARGS+=" --agent.seed ${SEED}"
    TRAIN_ARGS+=" --agent.run-name \"${RUN_LABEL}_s${SEED}\""
    TRAIN_ARGS+=" --agent.wandb-project \"${WANDB_PROJECT}\""
    TRAIN_ARGS+=" --agent.wandb-tags \"('${RUN_LABEL}','seed${SEED}')\""
    TRAIN_ARGS+=" --log-root \"${LOG_DIR}/rsl_rl\""
    [ -n "$MAX_ITERS" ] && TRAIN_ARGS+=" --agent.max-iterations ${MAX_ITERS}"
    [ "$VIDEO" = "1" ] && TRAIN_ARGS+=" --video True"
    [ -n "$EXTRA_ARGS" ] && TRAIN_ARGS+=" ${EXTRA_ARGS}"

    JOB_SCRIPT=$(mktemp /tmp/slurm_mjlab_XXXXXX.sh)

    cat > "$JOB_SCRIPT" <<SLURM
#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --gres=gpu:${GPU}
#SBATCH --time=${TIME}
#SBATCH --mem-per-cpu=${MEM_PER_CPU}
#SBATCH --job-name="${RUN_LABEL}-${SEED}"
#SBATCH --output=${LOG_DIR}/slurm/${RUN_LABEL}-${SEED}-%j.out
$([ -n "$PARTITION" ] && echo "#SBATCH --partition=${PARTITION}")
$([ -n "$EXCLUDE" ] && echo "#SBATCH --exclude=${EXCLUDE}")
$([ -n "$GPU_CONSTRAINT" ] && echo "#SBATCH --constraint=${GPU_CONSTRAINT}")
$([ -n "$MAIL" ] && echo "#SBATCH --mail-type=END,FAIL" && echo "#SBATCH --mail-user=${MAIL}")

set -euo pipefail

echo "Job \$SLURM_JOB_ID starting on \$(hostname) at \$(date)"
nvidia-smi || true

cd "${PROJECT_DIR}"

# uv's .venv, populated on a login node by 'make sync'. Prepending it to PATH keeps the
# job offline: no resolve, no download, no interpreter fetch at job start.
export PATH="${PROJECT_DIR}/.venv/bin:\$PATH"

# Headless rendering on the compute node. mjlab's train sets MUJOCO_EGL_DEVICE_ID itself
# to match the local rank.
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

# RslRlBaseRunnerCfg has no group field, but wandb.init reads WANDB_RUN_GROUP, which is
# what ties the seeds of one launch together in the UI.
export WANDB_RUN_GROUP="${WANDB_GROUP}"
$([ -n "$WANDB_MODE" ] && echo "export WANDB_MODE=${WANDB_MODE}")

train ${TRAIN_ARGS}

echo "Job \$SLURM_JOB_ID done at \$(date)"
SLURM

    if [ "$DRY_RUN" = "1" ]; then
        echo "--- DRY RUN: job script for seed ${SEED} ---"
        cat "$JOB_SCRIPT"
        echo "--- end ---"
        rm "$JOB_SCRIPT"
        continue
    fi

    JOB_ID=$(sbatch --parsable "$JOB_SCRIPT")
    JOB_IDS+=("$JOB_ID")
    echo "Submitted seed $SEED  ->  SLURM job ID: $JOB_ID"
    rm "$JOB_SCRIPT"
done

if [ "$DRY_RUN" = "1" ]; then
    echo ""
    echo "DRY_RUN=1 — nothing submitted."
    exit 0
fi

echo ""
echo "========================================"
echo "  All $N_SEEDS jobs submitted."
echo "  Monitor with:  squeue -j ${JOB_IDS[*]}"
echo "  Cancel all:    scancel ${JOB_IDS[*]}"
echo "  Logs:          ${LOG_DIR}/slurm/${RUN_LABEL}-*-<jobid>.out"
echo "========================================"
