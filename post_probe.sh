#!/usr/bin/env bash
# After post_eval.sh: rerun the failure analysis with per-token rejection-sampling acceptance, then push.
cd "$(dirname "$0")"
REPO=/c/Users/aayus/Projects/soft-spec-decoding
until grep -q "=== post-eval done" logs/post_eval.log 2>/dev/null; do sleep 30; done
./.venv/Scripts/python failures.py --sets gsm8k:40 mtbench:40 humaneval:40 --block d3_soft > logs/failures_rs.log 2>&1 && echo "failures rs ok" || { echo "FAILURES RS FAILED"; tail -5 logs/failures_rs.log; }
cp results/failures_summary.json "$REPO/results/"; cp failures.py post_probe.sh "$REPO/"
( cd "$REPO" && git add results/failures_summary.json failures.py post_probe.sh && git commit -q -m "Failure analysis: per-category acceptance under T=1 rejection sampling" && git push -q origin main && echo pushed ) || echo "nothing pushed"
echo "=== post-probe done $(date +%H:%M)"
