#!/bin/bash
# Prior mixing collapsed the plan (KL 0.005 bits); the 64-token run uses beta 0.3 without mixing instead.
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
until grep -q "cvae G32c16b0.1 done" logs/run_cvae3.log 2>/dev/null; do sleep 30; done
echo "=== start --G 8 --c 16 --beta 0.3 $(date +%H:%M)"
$PY cvae_plan.py --G 8 --c 16 --beta 0.3
echo "=== cvae queue 4 done $(date +%H:%M)"
