#!/bin/bash
# Progressive drafting with the plan shifted so the drafted block comes first, then the stage-1 VAE runs on 6M rows.
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
echo "=== start rel $(date +%H:%M)"
$PY plan_gist.py --G 32 --c 16 --sigma 0.25 --oblock --rel
TC="--train_cache cache_pb,cache_pb2,cache_pb3 --zdim 2048 --steps 36000"
echo "=== ae big_6m_z2048 start $(date +%H:%M)"
$PY ae_gate.py --name big_6m_z2048 --d 1024 --depth 6 $TC
echo "=== ae big_6m_z2048 done $(date +%H:%M)"
echo "=== ae base_6m_z2048 start $(date +%H:%M)"
$PY ae_gate.py --name base_6m_z2048 --d 512 --depth 4 $TC
echo "=== ae base_6m_z2048 done $(date +%H:%M)"
echo "=== queue 7 done $(date +%H:%M)"
