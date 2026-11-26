#!/bin/bash
# Relaunch after the 06:50 shutdown. Plan length at a fixed per-block budget (16 numbers, noise 0.25), control,
# high-capacity plan; then stage-1 VAE on 6M rows: d1024x6 (bigger) and d512x4 (same data, original size).
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
for a in "--G 32 --c 16 --sigma 0.25" "--G 64 --c 16 --sigma 0.25" "--G 8 --c 16 --sigma 0.25" "--c 0" "--G 32 --c 64 --sigma 0.05"; do
  echo "=== start $a $(date +%H:%M)"
  $PY plan_gist.py $a || { echo "=== retry $a at B 64"; $PY plan_gist.py $a --B 64; }
done
TC="--train_cache cache_pb,cache_pb2,cache_pb3 --zdim 2048 --steps 36000"
echo "=== ae big_6m_z2048 start $(date +%H:%M)"
$PY ae_gate.py --name big_6m_z2048 --d 1024 --depth 6 $TC
echo "=== ae big_6m_z2048 done $(date +%H:%M)"
echo "=== ae base_6m_z2048 start $(date +%H:%M)"
$PY ae_gate.py --name base_6m_z2048 --d 512 --depth 4 $TC
echo "=== ae base_6m_z2048 done $(date +%H:%M)"
echo "=== queue 4 done $(date +%H:%M)"
