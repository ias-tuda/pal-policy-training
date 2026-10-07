#!/bin/bash
# Decision 0005 (plan §12): apply mujoco_warp_boxbox.patch to ONE venv's mujoco_warp 3.11.0.
#
#   scripts/apply_mujoco_warp_boxbox_patch.sh .venv          # this worktree's venv
#
# uv installs packages as hardlinks into its cache, so an in-place edit would change every venv that shares the
# cache entry (other worktrees, other sessions). The file is copied to a fresh inode first; its link count is
# checked before the patch is applied. Idempotent: a venv that already carries the patch is left alone.
set -euo pipefail
VENV=${1:?usage: $0 <venv dir>}
HERE=$(cd "$(dirname "$0")/.." && pwd)
PATCH="$HERE/mujoco_warp_boxbox.patch"
SITE=$(ls -d "$VENV"/lib/python3.*/site-packages | head -1)
F="$SITE/mujoco_warp/_src/collision_primitive_core.py"
test -f "$F" || { echo "no mujoco_warp in $VENV" >&2; exit 1; }
if grep -q BOX_BOX_EDGE_TOL "$F"; then echo "already patched: $F"; exit 0; fi
# The base the patch was made against (mujoco_warp 3.11.0, collision_primitive_core.py).
BASE_MD5=9c54b45f9014a828087cb59e2ebff93d
got=$( (md5sum "$F" 2>/dev/null || md5 -r "$F") | cut -d' ' -f1)
test "$got" = "$BASE_MD5" || { echo "unexpected base $got (want $BASE_MD5): not mujoco_warp 3.11.0's file" >&2; exit 1; }
tmp="$F.decision0005.$$"
cp "$F" "$tmp" && rm -f "$F" && mv "$tmp" "$F"
links=$(stat -c %h "$F" 2>/dev/null || stat -f %l "$F")
test "$links" = 1 || { echo "$F still has $links links" >&2; exit 1; }
(cd "$SITE" && patch -p1 --no-backup-if-mismatch < "$PATCH")
grep -q BOX_BOX_EDGE_TOL "$F" && echo "patched: $F"
