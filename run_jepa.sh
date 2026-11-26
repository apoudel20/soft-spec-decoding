#!/bin/bash
# JEPA gist over the target's states for the 64 tokens after the drafted chunk, then drafters that read it.
# Starts after run_far.sh finishes so the two queues never share the GPU.
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
until grep -q "far queue done" logs/run_far.log 2>/dev/null; do sleep 60; done
echo "=== jepa train jg64 $(date +%H:%M)"
$PY jepa_gist.py train --name jg64 --far 64 --steps 4000 --B 64 || exit 1
echo "=== jepa export jg64 $(date +%H:%M)"
$PY jepa_gist.py export --name jg64 || exit 1
COMMON="--ae gist_hf_z2048 --mode det --direct --raw --train_cache cache_pb --steps 10000 --gist_npy jg64"
echo "=== stage2 smoke with jepa gist $(date +%H:%M)"
$PY stage2.py $COMMON --gist_kind orc --smoke || exit 1
echo "=== stage2 trained with oracle jepa gist $(date +%H:%M)"
$PY stage2.py $COMMON --gist_kind orc
echo "=== stage2 trained with predicted jepa gist $(date +%H:%M)"
$PY stage2.py $COMMON --gist_kind pred
echo "=== jepa queue done $(date +%H:%M)"
