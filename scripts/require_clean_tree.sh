#!/usr/bin/env bash
# require_clean_tree.sh — assert the running state is reproducible before launching
# SLURM jobs:
#   1. the git working tree is clean, so the git hash recorded by the run
#      (mjlab calls runner.add_git_repo_to_log, which wandb picks up) faithfully
#      represents the code that executes; and
#   2. uv.lock is in sync with pyproject.toml, so the .venv the job uses matches
#      the recorded dependencies.
#
# Usage (source it, then call):
#   source "$(dirname "$0")/require_clean_tree.sh"
#   require_clean_tree "${PROJECT_DIR}"
#
# Set ALLOW_DIRTY=1 to bypass both checks (with a warning) — e.g. for quick debug
# launches where reproducibility doesn't matter.

require_clean_tree() {
    local dir="${1:-.}"

    if ! git -C "$dir" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo "ERROR: '$dir' is not inside a git work tree — cannot verify code state." >&2
        exit 1
    fi

    # --porcelain catches staged + unstaged + untracked changes.
    local tree_status
    tree_status="$(git -C "$dir" status --porcelain)"

    if [[ -n "$tree_status" ]]; then
        if [[ -n "${ALLOW_DIRTY:-}" ]]; then
            echo "WARNING: working tree is DIRTY but ALLOW_DIRTY is set — continuing." >&2
            echo "         The recorded git hash will NOT represent the running code." >&2
            git -C "$dir" status --short >&2
        else
            echo "ERROR: working tree is not clean — commit or stash changes so the git" >&2
            echo "       hash represents the code being run. (Set ALLOW_DIRTY=1 to bypass.)" >&2
            echo "" >&2
            git -C "$dir" status --short >&2
            exit 1
        fi
    else
        echo "Clean working tree at $(git -C "$dir" rev-parse --short HEAD) — git hash represents the code."
    fi

    require_uv_locked "$dir"
}

# require_uv_locked — assert uv.lock is up to date with pyproject.toml.
# `uv lock --check` re-solves and exits non-zero if the lock is stale, i.e. the
# env `uv sync` would install differs from the lock. Same DIRTY semantics as the
# git check: ALLOW_DIRTY downgrades the failure to a warning.
require_uv_locked() {
    local dir="${1:-.}"

    if ! command -v uv >/dev/null 2>&1; then
        echo "WARNING: uv not on PATH — skipping lock-file check." >&2
        return 0
    fi

    if ( cd "$dir" && uv lock --check ) >/dev/null 2>&1; then
        echo "uv.lock is in sync with pyproject.toml — env matches the lock file."
        return 0
    fi

    if [[ -n "${ALLOW_DIRTY:-}" ]]; then
        echo "WARNING: uv.lock is OUT OF DATE with pyproject.toml but ALLOW_DIRTY is set — continuing." >&2
        echo "         The installed dependencies may not match uv.lock." >&2
        return 0
    fi
    echo "ERROR: uv.lock is not up to date with pyproject.toml — run 'uv lock' and commit" >&2
    echo "       the updated uv.lock so dependencies match the recorded state." >&2
    echo "       (Set ALLOW_DIRTY=1 to bypass.)" >&2
    exit 1
}
