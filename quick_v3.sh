#!/usr/bin/env bash
# Can the drafter use a TRUE future summary at all? v3 plan input, oracle_near, first 400 steps of the
# 1200-step schedule. Compare to d4_soft at step 400 (0.7983 / 0.7385) and to the v2 run, whose plan input
# never opened (0.8032 / 0.7465, same script and averaging). Then run the Qwen3.5 scale probe.
cd "$(dirname "$0")"
P=./.venv/Scripts/python
until grep -q "=== post-planner2 done" logs/post_planner2.log 2>/dev/null; do sleep 30; done
echo "=== v3 oracle_near quick test $(date +%H:%M)"
$P planner_v3.py train --plan oracle_near --steps 1200 --stop 400 --name plan3_oracle_near_q > logs/plan3_oracle_near_q.log 2>&1 \
  && grep -E "^step (100|200|300|400) " logs/plan3_oracle_near_q.log || { echo "V3 QUICK FAILED"; tail -8 logs/plan3_oracle_near_q.log; }
echo "=== v3 quick done $(date +%H:%M)"
bash run_q35.sh
