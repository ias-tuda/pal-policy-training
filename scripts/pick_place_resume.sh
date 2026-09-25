#!/bin/bash
# Resume flags for `train`, from a checkpoint file: sourced by `pick_place.sbatch` when RESUME
# is set, and runnable on its own to see what it would do.
#
#   RESUME=/abs/path/<run dir>/model_4999.pt REPO=<checkout> EXPERIMENT=<name> \
#     bash scripts/pick_place_resume.sh          # prints the flags, one per line
#
# mjlab's train resumes with `--agent.resume True --agent.load-run <regex>
# --agent.load-checkpoint <regex>`, both matched against `<REPO>/logs/rsl_rl/<EXPERIMENT>/`
# only (`mjlab.utils.os.get_checkpoint_path`). So:
# * a run directory that lives in another checkout -- the one the run started from, which
#   must not be touched while it has jobs -- is linked into this one's log root first;
# * both names are regex-escaped and anchored: run names carry `+` (`R-C-strict+M1+M2+M3`),
#   which unescaped is a quantifier and matches nothing, and an unanchored name would also
#   match a longer one;
# * the checkpoint is named exactly, not the "latest" the default regex would pick
#   (`model_900.pt` sorts after `model_1000.pt`).
# The resumed run continues the policy, optimiser, learning rate and iteration count
# (rsl_rl `OnPolicyRunner.load`), in a new run directory and a new wandb run; MAX_ITERATIONS
# is then the number of iterations *added*.
set -euo pipefail
: "${RESUME:?RESUME must be the path of a model_<iter>.pt}"
: "${REPO:?REPO must be set}" "${EXPERIMENT:?EXPERIMENT must be set}"
if [ ! -f "$RESUME" ]; then
  echo "Refusing to run: RESUME=$RESUME is not a file" >&2
  exit 1
fi
_resume_run_dir=$(cd "$(dirname "$RESUME")" && pwd -P)
_resume_run=$(basename "$_resume_run_dir")
_resume_ckpt=$(basename "$RESUME")
_resume_root="$REPO/logs/rsl_rl/$EXPERIMENT"
mkdir -p "$_resume_root"
if [ ! -e "$_resume_root/$_resume_run" ]; then
  ln -s "$_resume_run_dir" "$_resume_root/$_resume_run"
elif [ "$(cd "$_resume_root/$_resume_run" && pwd -P)" != "$_resume_run_dir" ]; then
  echo "Refusing to run: $_resume_root/$_resume_run exists and is not $_resume_run_dir" >&2
  exit 1
fi
_resume_escape() { printf '%s' "$1" | sed -e 's/[][\\.*^$(){}?+|/]/\\&/g'; }
RESUME_ARGS=(
  --agent.resume True
  --agent.load-run "^$(_resume_escape "$_resume_run")\$"
  --agent.load-checkpoint "^$(_resume_escape "$_resume_ckpt")\$"
)
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  printf '%s\n' "${RESUME_ARGS[@]}"
fi
