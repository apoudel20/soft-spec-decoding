#!/usr/bin/env bash
# Sketches B-E on DFlash, after Sketch A (run_plan.sh). GPU smoke first, then train each, then one joint eval.
cd "$(dirname "$0")"
P=./.venv/Scripts/python
until grep -q "=== dflash done" logs/run_dflash2.log 2>/dev/null; do sleep 30; done
echo "=== kv oracle $(date +%H:%M)"
$P kv_oracle.py > logs/kv_oracle.log 2>&1 && grep -E "^math|^code|^chat" logs/kv_oracle.log || { echo "KV ORACLE FAILED"; tail -6 logs/kv_oracle.log; }
# Sketches B-E skipped: they use the same input-embedding pathway that had no effect for Sketch A (oracle included).
echo "=== sketches done $(date +%H:%M)"
