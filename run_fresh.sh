#!/usr/bin/env bash
# Token-targeted stage 1: top-1 cross-entropy, then plus the head feature, then the long-answer cache and a rerun on it.
cd "$(dirname "$0")"
echo "=== fresh start $(date)"
.venv/Scripts/python ae_gate.py --name ce1_z2048 --zdim 2048 --ce 1 > logs/ae_ce1_z2048.log 2>&1; grep -h GATE logs/ae_ce1_z2048.log
.venv/Scripts/python ae_gate.py --name ce1_hf_z2048 --zdim 2048 --ce 1 --headfeat > logs/ae_ce1_hf_z2048.log 2>&1; grep -h GATE logs/ae_ce1_hf_z2048.log
echo "=== fresh gate done $(date)"
.venv/Scripts/python cache_states.py --long > logs/cache_long.log 2>&1; tail -n 1 logs/cache_long.log
.venv/Scripts/python ae_gate.py --cache cache_long --name long_ce1_hf_z2048 --zdim 2048 --ce 1 --headfeat > logs/ae_long_ce1_hf_z2048.log 2>&1; grep -h GATE logs/ae_long_ce1_hf_z2048.log
echo "=== fresh done $(date)"
