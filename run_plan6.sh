#!/bin/bash
# Reordered after the control showed predicted plans below the no-plan drafter: gist-to-next-gist prediction and
# progressive offsets first, then the stage-1 VAE runs on 6M rows. Starts when the c64 run in run_plan4.log ends.
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
until grep -q "plan pw_G32c64s0.05 done" logs/run_plan4.log 2>/dev/null; do sleep 30; done
B="--G 32 --c 16 --sigma 0.25"
for a in "$B --phist" "$B --oblock --phist" "$B --oblock"; do
  echo "=== start $a $(date +%H:%M)"
  $PY plan_gist.py $a
done
TC="--train_cache cache_pb,cache_pb2,cache_pb3 --zdim 2048 --steps 36000"
echo "=== ae big_6m_z2048 start $(date +%H:%M)"
$PY ae_gate.py --name big_6m_z2048 --d 1024 --depth 6 $TC
echo "=== ae big_6m_z2048 done $(date +%H:%M)"
echo "=== ae base_6m_z2048 start $(date +%H:%M)"
$PY ae_gate.py --name base_6m_z2048 --d 512 --depth 4 $TC
echo "=== ae base_6m_z2048 done $(date +%H:%M)"
echo "=== queue 6 done $(date +%H:%M)"
