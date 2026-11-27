#!/bin/bash
# Fixes from the 27 Nov Neon audit: evaluation-only reruns, then the masked-KL mixing run, a control trained only at
# offset 0, and progressive drafting with valid offsets only and o mod 8.
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
echo "=== reeval start $(date +%H:%M)"
$PY plan_reeval.py
for a in "cvae_plan.py --G 32 --c 16 --beta 0.3 --mix 0.5" "plan_gist.py --c 0 --poff 0" "plan_gist.py --G 32 --c 16 --sigma 0.25 --oblock --rel --ofix"; do
  echo "=== start $a $(date +%H:%M)"
  $PY $a
done
echo "=== audit fix queue done $(date +%H:%M)"
