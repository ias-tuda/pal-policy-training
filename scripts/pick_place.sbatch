#!/bin/bash
# Pick-and-place v0 training on the IAS cluster.
#
# One job is one seed. A multi-seed campaign is N of these, not one multi-GPU job:
# the resource block below is single-GPU and is user-specified -- do not change it
# without asking (CLAUDE.md, *Job submission -- IAS cluster*).
#
#   sbatch --export=ALL,SEED=1,NUM_ENVS=256,MAX_ITERATIONS=200,RUN_TAG=c0 \
#          --job-name=pp_c0_s1 \
#          --output=$HOME/repo/humanoids/pal_mjlab_kangaroo_talos/slurm_logs/pp_c0_s1_%j.log \
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

REPO=$HOME/repo/humanoids/pal_mjlab_kangaroo_talos
TASK=Mjlab-PickPlace-Pal-Kangaroo-Grippers

# The point of the guard is that the job refuses to run rather than silently training
# the wrong code. Bump it deliberately, with a note saying what changed.
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT must be set to the full 40-char SHA}"

SEED="${SEED:?SEED must be set, e.g. sbatch --export=ALL,SEED=1 ...}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-20000}"
RUN_TAG="${RUN_TAG:-c1}"
# D10: one task id; the arm and the rung are selected by these two variables, which
# `pick_place/kangaroo/env_cfg.py` reads at registration and validates against the names in
# `rewards.py` / the DR ladder. They must be *exported*, not just set: a value supplied via
# --export is already in the environment, but a default assigned here would otherwise be a
# plain shell variable and the training process would silently get R-A/DR0.
REWARD="${REWARD:-R-A}"
DR="${DR:-DR0}"
# The PPO recipe, the same way (S5.1 item 9, `rl_cfg.PPO_RECIPES`). Its default is C1r/C2r's
# recipe, not C1's; PPO=PPO-A gives back what C0-C2 ran.
PPO="${PPO:-PPO-stack}"
export REWARD DR PPO

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

# Validate the names before the job takes a GPU for the ~4 min warp compile. The authority
# is the registry itself -- this imports the same module `train` will, so there is one list
# of valid names and it cannot drift from this script.
if ! uv run --extra cu128 python -c \
     "from pal_mjlab.tasks.manipulation.pick_place.kangaroo.env_cfg import kangaroo_pick_place_env_cfg as f, kangaroo_pick_place_rl_cfg as g; f(); g()" \
     >/dev/null 2>"$REPO/.reward_check.$$"; then
  echo "Refusing to run: REWARD=$REWARD DR=$DR PPO=$PPO did not validate." >&2
  tail -3 "$REPO/.reward_check.$$" >&2
  rm -f "$REPO/.reward_check.$$"
  exit 1
fi
rm -f "$REPO/.reward_check.$$"

echo "=== provenance ==="
echo "commit      $actual_commit"
echo "task        $TASK"
echo "seed        $SEED"
echo "num_envs    $NUM_ENVS"
echo "iterations  $MAX_ITERATIONS"
echo "reward/DR   $REWARD / $DR"
echo "ppo         $PPO"
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
  --agent.run-name "pp_${RUN_TAG}_${REWARD}_${DR}_${PPO}_n${NUM_ENVS}_seed${SEED}" \
  --agent.wandb-tags "(\"ias\",\"pick-place-v0\",\"$RUN_TAG\",\"$REWARD\",\"$DR\",\"$PPO\")"
