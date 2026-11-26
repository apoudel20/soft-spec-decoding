#!/bin/bash
# Oracle gist test: does a compressed view of the next 64 to 256 tokens after the drafted chunk help draft it?
# Same network and data in every arm. Control: gist input zeroed.
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
COMMON="--ae gist_hf_z2048 --mode det --direct --raw --train_cache cache_pb --steps 10000"
echo "=== control (gist zeroed) $(date +%H:%M)"
$PY stage2.py $COMMON --far 64 --far_zero
echo "=== oracle gist, 64 tokens $(date +%H:%M)"
$PY stage2.py $COMMON --far 64
echo "=== oracle gist, 256 tokens $(date +%H:%M)"
$PY stage2.py $COMMON --far 256
echo "=== far queue done $(date +%H:%M)"
