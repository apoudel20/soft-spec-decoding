#!/bin/bash
# beta 1 collapsed (KL 0.003 bits). After the beta 0.3 run: prior mixing at beta 0.3 (256 and 64 tokens), and beta 0.1.
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
until grep -q "cvae G32c16b0.3 done" logs/run_cvae.log 2>/dev/null; do sleep 30; done
for a in "--G 32 --c 16 --beta 0.3 --mix 0.5" "--G 32 --c 16 --beta 0.1" "--G 8 --c 16 --beta 0.3 --mix 0.5"; do
  echo "=== start $a $(date +%H:%M)"
  $PY cvae_plan.py $a
done
echo "=== cvae queue 3 done $(date +%H:%M)"
