#!/usr/bin/env bash
# Main training queue (about 4 h on an RTX 4070 Ti). Each run logs to logs/<name>.log.
cd "$(dirname "$0")"
P=./.venv/Scripts/python
run() { name=$1; shift; echo "=== $name $(date +%H:%M)"; $P train.py "$@" --name "$name" > "logs/$name.log" 2>&1 && tail -n 2 "logs/$name.log" | head -n 1 || { echo "FAILED $name"; tail -n 5 "logs/$name.log"; }; }
run c_soft --arm C --inp soft --steps 1500
run c_hard --arm C --inp hard --steps 1500
run d3_soft --arm D --label d3 --inp soft --steps 1200
run d1_soft --arm D --label d1 --inp soft --steps 1200
run d4_soft --arm D --label d4 --inp soft --steps 1200
run d1_hard --arm D --label d1 --inp hard --steps 1200
run d3_hard --arm D --label d3 --inp hard --steps 1200
run d4_hard --arm D --label d4 --inp hard --steps 1200
run d2_soft --arm D --label d2 --inp soft --steps 1200
echo "=== all done $(date +%H:%M)"
