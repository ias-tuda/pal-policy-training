#!/usr/bin/env bash
# Download the latest (or --checkpoint ITER) checkpoint for a cluster W&B run and export it to ONNX.
set -euo pipefail

if [[ $# -lt 1 || "$1" == --help || "$1" == -h ]]; then
    echo "Usage: $0 RUN_ID [--checkpoint ITER] [EXPORT_ARGS...]"
    echo "Example: $0 edf7d80j --checkpoint 1500"
    echo "Uses the latest model_*.pt unless --checkpoint selects model_<ITER>.pt."
    echo "Writes model_<iter>.onnx next to the downloaded checkpoint; override with -o PATH."
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
exec bash "$(dirname "${BASH_SOURCE[0]}")/download_wandb_run.sh" "$run_id" --export "${checkpoint_args[@]}" "$@"
