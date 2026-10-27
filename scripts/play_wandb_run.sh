#!/usr/bin/env bash
# Download the latest (or --checkpoint ITER) checkpoint for a cluster W&B run and play its training task.
set -euo pipefail

if [[ $# -lt 1 || "$1" == --help || "$1" == -h ]]; then
    echo "Usage: $0 RUN_ID [--checkpoint ITER] [PLAY_ARGS...]"
    echo "Example: $0 kofy81xq --checkpoint 1500 --num-envs 1"
    echo "Uses the latest model_*.pt unless --checkpoint selects model_<ITER>.pt."
    echo "Viewer defaults to viser; override with --viewer native or --viewer auto."
    if [[ $# -lt 1 ]]; then exit 2; fi
    exit 0
fi

run_id="$1"
shift
checkpoint_args=()
passthrough=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint=*) checkpoint_args=("$1"); shift ;;
        --checkpoint)
            if [[ $# -lt 2 ]]; then echo "Error: --checkpoint requires a value." >&2; exit 2; fi
            checkpoint_args=(--checkpoint "$2"); shift 2 ;;
        *) passthrough+=("$1"); shift ;;
    esac
done
set -- "${passthrough[@]}"
viewer_args=(--viewer viser)
for arg in "$@"; do
    if [[ "$arg" == --viewer || "$arg" == --viewer=* ]]; then
        viewer_args=()
        break
    fi
done
exec bash "$(dirname "${BASH_SOURCE[0]}")/download_wandb_run.sh" "$run_id" --play "${checkpoint_args[@]}" "${viewer_args[@]}" "$@"
