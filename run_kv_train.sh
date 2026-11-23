#!/usr/bin/env bash
# Trained future-context ceiling: smoke, train horizon 16 and 64 adapters, evaluate, push.
cd "$(dirname "$0")"
P=./.venv/Scripts/python
REPO=/c/Users/aayus/Projects/soft-spec-decoding
echo "=== kv train smoke $(date +%H:%M)"
if $P kv_oracle_train.py train --horizon 16 --n_train 3 --blocks 2 > logs/kvt_smoke.log 2>&1; then
  grep -E "^step" logs/kvt_smoke.log | tail -1
  for h in 16 64; do
    echo "=== train horizon $h $(date +%H:%M)"
    $P kv_oracle_train.py train --horizon $h > logs/kvt_$h.log 2>&1 && tail -n 2 logs/kvt_$h.log | head -n 1 || { echo "TRAIN $h FAILED"; tail -6 logs/kvt_$h.log; }
  done
  echo "=== kv eval $(date +%H:%M)"
  $P kv_oracle_train.py eval > logs/kvt_eval.log 2>&1 && grep -E "^math|^code|^chat" logs/kvt_eval.log || { echo "KV EVAL FAILED"; tail -8 logs/kvt_eval.log; }
else
  echo "KV SMOKE FAILED"; grep -v -E "Loading|it/s" logs/kvt_smoke.log | tail -12
fi
cp results/kv_oracle.json results/kv_oracle_trained.json "$REPO/results/" 2>/dev/null; cp kv_oracle.py kv_oracle_train.py run_kv_train.sh "$REPO/"
( cd "$REPO" && git add results/kv_oracle.json results/kv_oracle_trained.json kv_oracle.py kv_oracle_train.py run_kv_train.sh 2>/dev/null; git commit -q -m "Future-context ceiling for DFlash: untrained and trained adapters" && git push -q origin main && echo pushed ) || echo "nothing pushed"
echo "=== kv done $(date +%H:%M)"
