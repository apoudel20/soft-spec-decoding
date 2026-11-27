#!/bin/bash
# Decoder trained on prior samples in half of the rounds (prior learns from the token loss). After the first CVAE queue.
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
until grep -q "cvae queue done" logs/run_cvae.log 2>/dev/null; do sleep 60; done
for a in "--G 32 --c 16 --beta 1 --mix 0.5" "--G 8 --c 16 --beta 1 --mix 0.5"; do
  echo "=== start $a $(date +%H:%M)"
  $PY cvae_plan.py $a
done
echo "=== cvae queue 2 done $(date +%H:%M)"
