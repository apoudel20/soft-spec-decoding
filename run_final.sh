#!/usr/bin/env bash
# After the scale queue: one apples-to-apples benchmark (DFlash, chain, both samplers' trees), then examples.
cd "$(dirname "$0")"
P=.venv/Scripts/python
until grep -q "=== scale done" logs/run_scale2.log 2>/dev/null; do sleep 60; done
sleep 30
echo "=== final benchmark $(date +%H:%M)"
$P e2e.py --sets mtbench gsm8k humaneval --n 20 --max_new 256 --drafters dflash gister_v2 gister_tree16d_n2 "gister_tree16d_n2@stage2_gister_flow_big" --out e2e_final > logs/e2e_final.log 2>&1
grep -vE "^W1" logs/e2e_final.log | grep -E "AR|tau"
echo "=== examples $(date +%H:%M)"
$P examples.py --s2 stage2_gister_flow_big --k 8 --rounds 12 > logs/examples.log 2>&1 || $P examples.py --s2 stage2_gister_flow --k 8 --rounds 12 > logs/examples.log 2>&1
grep -E "mean kept|===|Error" logs/examples.log
echo "=== final done $(date +%H:%M)"
