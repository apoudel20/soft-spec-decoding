#!/usr/bin/env bash
# Planner rerun with the bounded, gated plan input (the first run's plan vector grew to 4-10x the token
# embedding norm and drowned the drafter's inputs, so its results are invalid). Waits for the probe rerun.
# Three arms answer the decision rules: target features, predicted plan, oracle near-future summary.
cd "$(dirname "$0")"
P=./.venv/Scripts/python
REPO=/c/Users/aayus/Projects/soft-spec-decoding
until grep -q "=== post-fix done" logs/post_fix.log 2>/dev/null; do sleep 30; done
mkdir -p results/planner_v1_invalid && mv results/planner.json results/planner_v1_invalid/ 2>/dev/null
echo "=== planner2 smoke $(date +%H:%M)"
if $P planner.py train --plan pred --steps 3 --name plan_smoke > logs/planner2_smoke.log 2>&1 && \
   $P planner.py eval --ckpts d4_soft plan_smoke --sets gsm8k:40 --stride 16 --out planner_smoke >> logs/planner2_smoke.log 2>&1; then
  grep meanL logs/planner2_smoke.log
  rm -f ckpt/plan_smoke.pt ckpt/plan_smoke_log.json results/planner_smoke.json
  for p in feat pred oracle_near; do
    echo "=== train plan_$p $(date +%H:%M)"
    $P planner.py train --plan $p --steps 1200 > logs/plan2_$p.log 2>&1 && tail -n 3 logs/plan2_$p.log | head -n 1 || { echo "FAILED plan_$p"; tail -5 logs/plan2_$p.log; }
  done
  echo "=== planner2 eval $(date +%H:%M)"
  $P planner.py eval --ckpts d4_soft plan_feat plan_pred plan_oracle_near --out planner > logs/planner2_eval.log 2>&1 && grep -E "meanL|wrote" logs/planner2_eval.log || { echo "PLANNER EVAL FAILED"; tail -5 logs/planner2_eval.log; }
else
  echo "PLANNER2 SMOKE FAILED"; tail -10 logs/planner2_smoke.log
fi
cp results/planner.json "$REPO/results/" 2>/dev/null; cp planner.py post_planner2.sh "$REPO/"
( cd "$REPO" && git add results/planner.json planner.py post_planner2.sh && git commit -q -m "Planner rerun with bounded, gated plan input (first run invalid: plan vector drowned the token inputs)" && git push -q origin main && echo pushed ) || echo "nothing pushed"
echo "=== post-planner2 done $(date +%H:%M)"
