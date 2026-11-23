#!/usr/bin/env bash
# Sketch A step 1: DFlash baseline + exactness on Qwen3-8B, after the fair probe frees the GPU.
cd "$(dirname "$0")"
P=./.venv/Scripts/python
until grep -q "=== plan done" logs/run_plan.log 2>/dev/null; do sleep 30; done
until grep -q "^exit" logs/dflash_download.log 2>/dev/null; do sleep 10; done
grep -q "^exit 0" logs/dflash_download.log || { echo "DFLASH DOWNLOAD FAILED"; exit 1; }
echo "=== dflash smoke $(date +%H:%M)"
if $P dflash_eval.py --n 1 --max_new 32 --sets math --out dflash_eval_smoke > logs/dflash_smoke.log 2>&1; then
  grep -E "^math|loaded" logs/dflash_smoke.log
  rm -f results/dflash_eval_smoke.json
  echo "=== dflash full $(date +%H:%M)"
  $P dflash_eval.py --n 10 --max_new 160 > logs/dflash_eval.log 2>&1 && grep -E "^math|^code|^chat" logs/dflash_eval.log || { echo "DFLASH EVAL FAILED"; tail -8 logs/dflash_eval.log; }
else
  echo "DFLASH SMOKE FAILED"; grep -v -E "Loading|it/s" logs/dflash_smoke.log | tail -15
fi
echo "=== dflash done $(date +%H:%M)"
