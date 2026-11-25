#!/usr/bin/env bash
# One job at a time on the GPU: stage-1 token-loss runs, fair DFlash baseline, stage 2 (det, flow), the all-layer sweep, then the long cache.
cd "$(dirname "$0")"
L=logs/run_fresh.log
while ps -p 8677 > /dev/null 2>&1; do sleep 30; done
grep -h GATE logs/ae_ce1_z2048.log >> $L
.venv/Scripts/python ae_gate.py --name ce1_hf_z2048 --zdim 2048 --ce 1 --headfeat > logs/ae_ce1_hf_z2048.log 2>&1; grep -h GATE logs/ae_ce1_hf_z2048.log >> $L
echo "=== fresh gate done $(date)" >> $L
echo "=== stage2 start $(date)"
.venv/Scripts/python dflash_tree_gap.py --hf_labels --out dflash_fair > logs/dflash_fair.log 2>&1; tail -n 4 logs/dflash_fair.log
BEST=$(.venv/Scripts/python -c "
import json,os
c={n:json.load(open(f'results/ae_gate_{n}.json'))['sets']['dev_chat']['agree'] for n in ['z2048_full','ce1_z2048','ce1_hf_z2048'] if os.path.exists(f'results/ae_gate_{n}.json')}
print(max(c,key=c.get))")
echo "stage-1 model for stage 2: $BEST"
.venv/Scripts/python stage2.py --ae $BEST --mode det > logs/stage2_det.log 2>&1; grep -E "^dev_" logs/stage2_det.log
.venv/Scripts/python stage2.py --ae $BEST --mode flow > logs/stage2_flow.log 2>&1; grep -E "^dev_" logs/stage2_flow.log
echo "=== stage2 done $(date)"
.venv/Scripts/python layer_sweep.py --smoke > logs/layer_sweep_smoke.log 2>&1 && .venv/Scripts/python layer_sweep.py > logs/layer_sweep.log 2>&1; tail -n 3 logs/layer_sweep.log
echo "=== sweep done $(date)"
.venv/Scripts/python cache_states.py --long > logs/cache_long.log 2>&1; tail -n 1 logs/cache_long.log
.venv/Scripts/python ae_gate.py --cache cache_long --name long_ce1_hf_z2048 --zdim 2048 --ce 1 --headfeat > logs/ae_long_ce1_hf_z2048.log 2>&1; grep -h GATE logs/ae_long_ce1_hf_z2048.log >> $L
echo "=== fresh done $(date)" >> $L
