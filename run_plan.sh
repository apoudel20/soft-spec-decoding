#!/usr/bin/env bash
# Sketch A on DFlash: smoke, train mid and oracle plan heads, evaluate against plain DFlash. Runs after run_dflash.sh.
cd "$(dirname "$0")"
P=./.venv/Scripts/python
until grep -q "=== dflash done" logs/run_dflash.log 2>/dev/null || grep -q "FAILED" logs/run_dflash.log 2>/dev/null; do sleep 20; done
echo "=== plan smoke $(date +%H:%M)"
if $P dflash_plan.py train --plan mid --n_train 3 --blocks 2 > logs/plan_smoke.log 2>&1 && \
   $P dflash_plan.py eval --plans none mid --sets math --stride 64 --out dflash_plan_smoke >> logs/plan_smoke.log 2>&1; then
  grep -E "^step|^math" logs/plan_smoke.log | tail -3
  rm -f results/dflash_plan_smoke.json
  for p in mid oracle; do
    echo "=== train $p $(date +%H:%M)"
    $P dflash_plan.py train --plan $p > logs/plan_$p.log 2>&1 && tail -n 2 logs/plan_$p.log | head -n 1 || { echo "TRAIN $p FAILED"; tail -6 logs/plan_$p.log; }
  done
  echo "=== plan eval $(date +%H:%M)"
  $P dflash_plan.py eval --plans none mid oracle > logs/plan_eval.log 2>&1 && grep -E "^math|^code|^chat" logs/plan_eval.log || { echo "PLAN EVAL FAILED"; tail -8 logs/plan_eval.log; }
else
  echo "PLAN SMOKE FAILED"; grep -v -E "Loading|it/s" logs/plan_smoke.log | tail -15
fi
echo "=== plan done $(date +%H:%M)"
