#!/bin/bash
# Conditional-VAE plan (learned prior, KL charge) after queue 7: 256-token plan at beta 1 and 0.3, 64-token at beta 1.
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
until grep -q "queue 7 done" logs/run_plan7.log 2>/dev/null; do sleep 60; done
for a in "--G 32 --c 16 --beta 1" "--G 32 --c 16 --beta 0.3" "--G 8 --c 16 --beta 1"; do
  echo "=== start $a $(date +%H:%M)"
  $PY cvae_plan.py $a
done
echo "=== cvae queue done $(date +%H:%M)"
