#!/bin/bash
# Apply this repo's local patches to the installed mujoco_warp of a venv (decision 0005: mujoco_warp_box_box_depth.patch).
#   bash scripts/apply_mujoco_warp_patches.sh [<venv dir>]        # default: ./.venv
# Idempotent: a patch whose marker is already in the file is skipped; otherwise it must apply cleanly (patch --dry-run)
# or the script stops without touching anything. Do not run it on a venv a running job imports from.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
VENV=${1:-$REPO/.venv}
SITE=$(ls -d "$VENV"/lib/python3*/site-packages | head -1)
[ -d "$SITE/mujoco_warp" ] || { echo "no mujoco_warp under $SITE" >&2; exit 1; }
apply() {  # patch file, file it changes (relative to site-packages), marker
  local p=$REPO/$1 f=$SITE/$2 marker=$3
  if grep -q "$marker" "$f"; then echo "already applied: $1 ($f)"; return 0; fi
  patch --dry-run -p1 -d "$SITE" < "$p" > /dev/null || { echo "does not apply cleanly: $1 -> $f" >&2; exit 1; }
  patch -p1 -d "$SITE" < "$p" > /dev/null
  grep -q "$marker" "$f" && echo "applied: $1 ($f)"
}
apply mujoco_warp_box_box_depth.patch mujoco_warp/_src/collision_primitive.py "Local patch (oppsie, 2026-10-08)"
