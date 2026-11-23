#!/usr/bin/env bash
# Sketches B-E on DFlash, after Sketch A (run_plan.sh). GPU smoke first, then train each, then one joint eval.
cd "$(dirname "$0")"
P=./.venv/Scripts/python
until grep -q "=== dflash done" logs/run_dflash2.log 2>/dev/null; do sleep 30; done
echo "=== kv oracle $(date +%H:%M)"
$P kv_oracle.py > logs/kv_oracle.log 2>&1 && grep -E "^math|^code|^chat" logs/kv_oracle.log || { echo "KV ORACLE FAILED"; tail -6 logs/kv_oracle.log; }
echo "=== sketches gpu smoke $(date +%H:%M)"
if $P sketches.py train --sketch E_cnn --n_train 3 --blocks 2 > logs/sketch_smoke.log 2>&1; then
  rm -f ckpt/sketch_E_cnn.pt
  for sk in B C D E_cnn E_probe; do
    echo "=== train $sk $(date +%H:%M)"
    $P sketches.py train --sketch $sk > logs/sketch_$sk.log 2>&1 && tail -n 2 logs/sketch_$sk.log | head -n 1 || { echo "TRAIN $sk FAILED"; tail -6 logs/sketch_$sk.log; }
  done
  echo "=== sketches eval $(date +%H:%M)"
  $P sketches.py eval > logs/sketches_eval.log 2>&1 && grep -E "^math|^code|^chat" logs/sketches_eval.log || { echo "SKETCH EVAL FAILED"; tail -8 logs/sketches_eval.log; }
else
  echo "SKETCH GPU SMOKE FAILED"; grep -v -E "Loading|it/s" logs/sketch_smoke.log | tail -15
fi
echo "=== sketches done $(date +%H:%M)"
