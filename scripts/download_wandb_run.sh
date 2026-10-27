#!/usr/bin/env bash
# Resolve a W&B run ID to its training log directory, then rsync the cluster data.
# Usage: scripts/download_wandb_run.sh RUN_ID [DESTINATION] [--play|--export [--checkpoint ITER] [ARGS...]]
# --checkpoint selects model_<ITER>.pt (ITER or the file name) instead of the latest one.
# Example: scripts/download_wandb_run.sh 94e3kesr

set -euo pipefail

readonly REMOTE="kienle@mn.ias.informatik.tu-darmstadt.de"
readonly REMOTE_PROJECT="/home/kienle/OopsieData/pal_mjlab_kangaroo_talos"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 RUN_ID [DESTINATION] [--play|--export [--checkpoint ITER] [ARGS...]]" >&2
    exit 2
fi

run_id="$1"
shift
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
destination="$project_root/wandb"
if [[ $# -gt 0 && "$1" != --play && "$1" != --export ]]; then
    destination="$1"
    shift
fi
# action: none (full download), play, or export (one checkpoint only, latest by default).
action=none
requested_checkpoint=""
if [[ $# -gt 0 ]]; then
    if [[ "$1" != --play && "$1" != --export ]]; then
        echo "Error: expected --play or --export after destination." >&2
        exit 2
    fi
    action="${1#--}"
    shift
    if [[ $# -gt 0 && "$1" == --checkpoint=* ]]; then
        requested_checkpoint="${1#--checkpoint=}"
        shift
    elif [[ $# -gt 0 && "$1" == --checkpoint ]]; then
        if [[ $# -lt 2 ]]; then
            echo "Error: --checkpoint requires a value." >&2
            exit 2
        fi
        requested_checkpoint="$2"
        shift 2
    fi
fi
if [[ -n "$requested_checkpoint" ]]; then
    if [[ "$requested_checkpoint" =~ ^[0-9]+$ ]]; then
        requested_checkpoint="model_${requested_checkpoint}.pt"
    elif [[ ! "$requested_checkpoint" =~ ^model_[0-9]+\.pt$ ]]; then
        echo "Error: --checkpoint must be an iteration number or model_<ITER>.pt." >&2
        exit 2
    fi
fi

if [[ ! "$run_id" =~ ^[[:alnum:]]{8}$ ]]; then
    echo "Error: RUN_ID must be an eight-character W&B run ID." >&2
    exit 2
fi

mkdir -p "$destination"
destination="$(cd "$destination" && pwd)"

# The W&B metadata stored on the cluster provides the run name and start time.
# Use those to resolve the corresponding timestamped logs/rsl_rl/<task>/<run> folder;
# the files transferred below are the training data, not the W&B run directory.
run_info="$(ssh "$REMOTE" python3 - "$REMOTE_PROJECT" "$run_id" "$action" "$requested_checkpoint" <<'PY'
import datetime as dt
import json
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
run_id = sys.argv[2]
runs = sorted((root / "wandb").glob(f"run-*-{run_id}/files/wandb-metadata.json"))
if not runs:
    sys.exit(0)
metadata = json.loads(runs[-1].read_text())
args = metadata.get("args", [])
try:
    run_name = args[args.index("--agent.run-name") + 1]
except (ValueError, IndexError):
    sys.exit(0)
started = dt.datetime.fromisoformat(metadata["startedAt"].replace("Z", "+00:00"))
started = started.astimezone().replace(tzinfo=None)
candidates = []
for task_dir in (root / "logs/rsl_rl").iterdir():
    if not task_dir.is_dir():
        continue
    for run_dir in task_dir.iterdir():
        if not run_dir.is_dir() or not run_dir.name.endswith("_" + run_name):
            continue
        try:
            timestamp = dt.datetime.strptime(run_dir.name[:19], "%Y-%m-%d_%H-%M-%S")
        except ValueError:
            continue
        candidates.append((abs((timestamp - started).total_seconds()), run_dir))
if candidates:
    distance, match = min(candidates, key=lambda item: item[0])
    if distance <= 600:
        print(match)
        if sys.argv[3] != "none":
            tasks = [arg for arg in args if arg.startswith("Mjlab-")]
            if len(tasks) != 1:
                sys.exit("Error: could not identify a unique training task in W&B metadata.")
            checkpoints = [
                (int(m.group(1)), path.name)
                for path in match.glob("model_*.pt")
                if path.is_file() and (m := re.fullmatch(r"model_(\d+)\.pt", path.name))
            ]
            if not checkpoints:
                sys.exit("Error: this run has no model_*.pt checkpoints yet.")
            requested = sys.argv[4] if len(sys.argv) > 4 else ""
            if requested and requested not in {name for _, name in checkpoints}:
                available = ", ".join(str(it) for it, _ in sorted(checkpoints))
                sys.exit(f"Error: {requested} not found; available iterations: {available}")
            print(tasks[0])
            print(requested or max(checkpoints)[1])
PY
)"
remote_run="${run_info%%$'\n'*}"
if [[ -z "$remote_run" ]]; then
    echo "Error: could not map W&B run '$run_id' to a training log directory on $REMOTE." >&2
    echo "The run must have cluster W&B metadata and matching logs/rsl_rl output." >&2
    exit 1
fi

relative_run="${remote_run#"$REMOTE_PROJECT"/}"
target_dir="$destination/$relative_run"
mkdir -p "$target_dir"
echo "Downloading cluster training data: $remote_run"
rsync_filters=()
if [[ "$action" != none ]]; then
    mapfile -t fields <<< "$run_info"
    task="${fields[1]}"
    checkpoint="${fields[2]}"
    rsync_filters=(--include="/$checkpoint" --exclude='/model_*.pt')
fi
rsync -a --partial --append-verify --info=progress2 -e ssh \
    "${rsync_filters[@]}" \
    "$REMOTE:$remote_run/" "$target_dir/"
echo "Download complete: $target_dir"
if [[ "$action" != none ]]; then
    if [[ ! -s "$target_dir/$checkpoint" ]]; then
        echo "Error: downloaded checkpoint is missing or empty: $target_dir/$checkpoint" >&2
        exit 1
    fi
    cd "$project_root"
    if [[ "$action" == play ]]; then
        echo "Playing $task with $checkpoint"
        exec uv run play "$task" --checkpoint-file "$target_dir/$checkpoint" "$@"
    fi
    echo "Exporting $task from $checkpoint"
    exec uv run export-policy "$task" "$target_dir/$checkpoint" "$@"
fi
