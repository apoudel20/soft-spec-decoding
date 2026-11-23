#!/usr/bin/env bash
# After post_probe.sh: latent planner experiment (smoke test, 4 trainings of ~28 min, offline eval), then push.
cd "$(dirname "$0")"
P=./.venv/Scripts/python
REPO=/c/Users/aayus/Projects/soft-spec-decoding
until grep -q "=== post-probe done" logs/post_probe.log 2>/dev/null; do sleep 30; done
echo "=== planner smoke $(date +%H:%M)"
if $P planner.py train --plan pred --steps 3 --name plan_smoke > logs/planner_smoke.log 2>&1 && \
   $P planner.py eval --ckpts d4_soft plan_smoke --sets gsm8k:40 --stride 16 --out planner_smoke >> logs/planner_smoke.log 2>&1; then
  rm -f ckpt/plan_smoke.pt ckpt/plan_smoke_log.json results/planner_smoke.json
  for p in feat pred oracle_near oracle_far; do
    echo "=== train plan_$p $(date +%H:%M)"
    $P planner.py train --plan $p --steps 1200 > logs/plan_$p.log 2>&1 || { echo "FAILED plan_$p"; tail -5 logs/plan_$p.log; }
  done
  echo "=== planner eval $(date +%H:%M)"
  $P planner.py eval --out planner > logs/planner_eval.log 2>&1 && grep -E "meanL|wrote" logs/planner_eval.log || { echo "PLANNER EVAL FAILED"; tail -5 logs/planner_eval.log; }
else
  echo "PLANNER SMOKE FAILED"; tail -10 logs/planner_smoke.log
fi
cp results/planner.json "$REPO/results/" 2>/dev/null; cp planner.py post_planner.sh "$REPO/"
( cd "$REPO" && git add results/planner.json planner.py post_planner.sh 2>/dev/null; git commit -q -m "Latent planner experiment: target features vs plan head vs oracle future summaries" && git push -q origin main && echo pushed ) || echo "nothing pushed"
echo "=== post-planner done $(date +%H:%M)"
