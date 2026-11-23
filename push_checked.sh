#!/usr/bin/env bash
# Checked push: copy whatever result files exist, commit, push, then verify the remote has the commit.
# Usage: push_checked.sh "<wait-for marker>" "<log file>" "<commit message>" file1 [file2 ...]
# Files are paths relative to the experiment dir; results/*.json go to the repo's results/, everything else to its root.
set -u
cd "$(dirname "$0")"
MARK="$1"; LOG="$2"; MSG="$3"; shift 3
REPO=/c/Users/aayus/Projects/soft-spec-decoding
until grep -q "$MARK" "$LOG" 2>/dev/null; do sleep 30; done
added=()
for f in "$@"; do
  if [ -f "$f" ]; then
    case "$f" in results/*) cp "$f" "$REPO/results/" ; added+=("results/$(basename "$f")");; *) cp "$f" "$REPO/"; added+=("$(basename "$f")");; esac
  else
    echo "MISSING (not pushed): $f"
  fi
done
cd "$REPO"
if [ ${#added[@]} -eq 0 ]; then echo "PUSH FAILED: no files to add"; exit 1; fi
git add -- "${added[@]}" || { echo "PUSH FAILED: git add"; exit 1; }
if git diff --cached --quiet; then echo "nothing new to commit (already pushed)"; exit 0; fi
git commit -q -m "$MSG" || { echo "PUSH FAILED: commit"; exit 1; }
git push -q origin main || { echo "PUSH FAILED: push"; exit 1; }
git fetch -q origin main && [ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ] && echo "PUSHED AND VERIFIED $(git rev-parse --short HEAD)" || echo "PUSH FAILED: remote does not match"
