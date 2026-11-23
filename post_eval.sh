#!/usr/bin/env bash
# Runs after run_eval.sh: push raw eval results, then the planner probe (smoke test first), then push again.
# Safe to run unattended. Only results/*.json and scripts are copied to the public repo.
cd "$(dirname "$0")"
P=./.venv/Scripts/python
REPO=/c/Users/aayus/Projects/soft-spec-decoding
sync_push() {
  cp results/*.json "$REPO/results/" 2>/dev/null
  for f in probes.py post_eval.sh run_eval.sh failures.py arm0.py; do cp "$f" "$REPO/$f"; done
  ( cd "$REPO" && git add results/*.json *.py *.sh && git commit -q -m "$1" && git push -q origin main && echo "pushed: $1" ) || echo "nothing to push for: $1"
}
until grep -q "=== all evals done" logs/eval_queue.log 2>/dev/null; do sleep 30; done
sync_push "Raw evaluation results (greedy and T=1), before write-up"
echo "=== probes smoke $(date +%H:%M)"
if $P probes.py --n_train 30 --per_seq 8 --sets gsm8k:40 --out probes_smoke > logs/probes_smoke.log 2>&1; then
  echo "=== probes full $(date +%H:%M)"
  $P probes.py --n_train 1500 --per_seq 48 --out probes > logs/probes.log 2>&1 && echo "probes ok" || { echo "PROBES FAILED"; tail -5 logs/probes.log; }
else
  echo "PROBES SMOKE FAILED"; tail -8 logs/probes_smoke.log
fi
rm -f results/probes_smoke.json
sync_push "Planner go/no-go probe: target vs drafter future knowledge"
echo "=== post-eval done $(date +%H:%M)"
