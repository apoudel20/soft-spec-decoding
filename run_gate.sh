#!/usr/bin/env bash
# Stage-2 entry gate queue: wait for the extra answers, cache states, train and test 4 autoencoders, push.
cd "$(dirname "$0")"
PY=.venv/Scripts/python
LOG=logs/run_gate.log
echo "=== queue start $(date)" >> $LOG
until grep -q "=== gen_more done" logs/gen_more.log 2>/dev/null; do sleep 60; done
echo "=== gen_more seen $(date)" >> $LOG
$PY cache_states.py >> $LOG 2>&1 || { echo "FAILED cache_states" >> $LOG; exit 1; }
for cfg in "z1024_full 1024 full" "z512_full 512 full" "z2048_full 2048 full" "z1024_final 1024 final"; do
  set -- $cfg
  $PY ae_gate.py --name $1 --zdim $2 --rows $3 >> logs/ae_$1.log 2>&1 || echo "FAILED ae $1" >> $LOG
  grep -h "GATE\|=== ae" logs/ae_$1.log >> $LOG
done
echo "=== gate queue done $(date)" >> $LOG
