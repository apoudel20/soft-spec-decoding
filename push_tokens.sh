#!/usr/bin/env bash
# Wait for generation, export tokens to the repo, push them with the gate scripts, verify the remote.
cd "$(dirname "$0")"
until grep -q "=== gen_more done" logs/gen_more.log 2>/dev/null; do sleep 30; done
.venv/Scripts/python export_tokens.py || { echo "PUSH FAILED: export"; exit 1; }
REPO=/c/Users/aayus/Projects/soft-spec-decoding
cp common.py cache_states.py ae_gate.py gen_q3_more.py run_gate.sh export_tokens.py "$REPO/"
cd "$REPO"
git add tokens common.py cache_states.py ae_gate.py gen_q3_more.py run_gate.sh export_tokens.py || { echo "PUSH FAILED: add"; exit 1; }
git commit -q -m "Token data (2,000 train + 120 dev Qwen3-8B answers) and stage-2 gate scripts" || { echo "PUSH FAILED: commit"; exit 1; }
git push -q origin main || { echo "PUSH FAILED: push"; exit 1; }
git fetch -q origin main && [ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ] && echo "PUSHED AND VERIFIED $(git rev-parse --short HEAD)" || echo "PUSH FAILED: remote does not match"
