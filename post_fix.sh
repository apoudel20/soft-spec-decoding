#!/usr/bin/env bash
# Runs after post_planner.sh: rerun the planner probe with the fixed ridge regression
# (the first run penalised the intercept with lam*N, lam=10, so its R^2 and cosine numbers are invalid).
cd "$(dirname "$0")"
P=./.venv/Scripts/python
REPO=/c/Users/aayus/Projects/soft-spec-decoding
until grep -q "=== post-planner done" logs/post_planner.log 2>/dev/null; do sleep 60; done
echo "=== probes rerun $(date +%H:%M)"
if $P probes.py --n_train 1500 --per_seq 48 --out probes > logs/probes_fixed.log 2>&1; then
  echo "probes ok"
  cp results/probes.json "$REPO/results/"
  cp probes.py post_fix.sh "$REPO/"
  ( cd "$REPO" && git add results/probes.json probes.py post_fix.sh && git commit -q -m "Planner probe rerun with fixed ridge (unpenalised intercept, penalty chosen on held-out train rows)" && git push -q origin main && echo "pushed" ) || echo "push failed"
else
  echo "PROBES FAILED"; tail -5 logs/probes_fixed.log
fi
echo "=== post-fix done $(date +%H:%M)"
