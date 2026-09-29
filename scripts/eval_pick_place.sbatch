#!/bin/bash
# Evaluate a list of pick-and-place checkpoints on one IAS GPU, one after the other
# (`scripts/eval_pick_place.py`). Plan S7-4 prep (f) re-reads S7-2's and S7-3's 5 k checkpoints
# with the jaw-release (J) reads; the same script serves S7-4's own reads.
#
#   sbatch --export=ALL,EXPECTED_COMMIT=<sha>,LIST=<path>,REPO=<checkout> \
#          --job-name=pp_eval --output=<checkout>/slurm_logs/pp_eval_%j.log --time=03:00:00 \
#          scripts/eval_pick_place.sbatch
#
# LIST has one checkpoint per line: `<json out> <checkpoint> [KEY=VALUE ...]`, the KEY=VALUEs
# being the task's selectors that run trained with (REWARD, DR, BASELINE, SPAWN, TOOL_SPEED,
# GAINS, EFFORT, LAG, SLEW, PPO) -- read at registration, so each eval is its own process. Blank
# lines and lines starting with # are skipped. EVAL_ARGS (e.g. "--plant --handoff") go to
# every eval. The expected-vs-actual count is printed at the end.
#SBATCH --partition=main
#SBATCH -C 'vram48gb|rtx3090|a5000'
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5G

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

REPO="${REPO:-/home/ju/repo/humanoids/pal_mjlab_kangaroo_talos}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT must be set to the full 40-char SHA}"
LIST="${LIST:?LIST must name the checkpoint list}"
EPISODES="${EPISODES:-64}"
NUM_ENVS="${NUM_ENVS:-64}"
EVAL_ARGS="${EVAL_ARGS:-}"

cd "$REPO"
actual_commit="$(git rev-parse HEAD)"
if [ "$actual_commit" != "$EXPECTED_COMMIT" ]; then
  echo "Refusing to run: expected commit $EXPECTED_COMMIT, found $actual_commit" >&2
  exit 1
fi
echo "=== provenance ==="
echo "commit      $actual_commit"
echo "list        $LIST ($(grep -cvE '^[[:space:]]*(#|$)' "$LIST") checkpoints)"
echo "episodes    $EPISODES over $NUM_ENVS envs, seed 0"
echo "eval args   ${EVAL_ARGS:-(none)}"
echo "node        $(hostname)   gpu $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo '?')"
echo "=================="

want=0
ok=0
while read -r out ckpt rest; do
  case "$out" in ''|'#'*) continue ;; esac
  want=$((want + 1))
  mkdir -p "$(dirname "$out")"
  echo "--- $ckpt ($rest) -> $out"
  # shellcheck disable=SC2086  # the selectors and EVAL_ARGS are meant to word-split.
  if env $rest uv run --extra cu128 python scripts/eval_pick_place.py \
       --checkpoint "$ckpt" --episodes "$EPISODES" --num-envs "$NUM_ENVS" \
       --device cuda:0 --seed 0 --json "$out" $EVAL_ARGS; then
    ok=$((ok + 1))
  else
    echo "!!! eval failed: $ckpt" >&2
  fi
done < "$LIST"
echo "evaluated $ok / $want expected"
