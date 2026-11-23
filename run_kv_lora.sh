#!/usr/bin/env bash
# Go/no-go: LoRA-fine-tuned DFlash with and without the true future. Smoke, two trainings, eval. Push is separate (push_checked.sh).
cd "$(dirname "$0")"
P=./.venv/Scripts/python
echo "=== lora smoke $(date +%H:%M)"
if $P kv_lora.py train --variant fut16 --n_train 3 --blocks 2 > logs/kvl_smoke.log 2>&1; then
  grep -E "^step" logs/kvl_smoke.log | tail -1
  for v in control fut16; do
    echo "=== train $v $(date +%H:%M)"
    $P kv_lora.py train --variant $v > logs/kvl_$v.log 2>&1 && tail -n 2 logs/kvl_$v.log | head -n 1 || { echo "TRAIN $v FAILED"; tail -6 logs/kvl_$v.log; }
  done
  echo "=== lora eval $(date +%H:%M)"
  $P kv_lora.py eval > logs/kvl_eval.log 2>&1 && grep -E "^math|^code|^chat" logs/kvl_eval.log || { echo "LORA EVAL FAILED"; tail -8 logs/kvl_eval.log; }
else
  echo "LORA SMOKE FAILED"; grep -v -E "Loading|it/s" logs/kvl_smoke.log | tail -12
fi
echo "=== lora done $(date +%H:%M)"
