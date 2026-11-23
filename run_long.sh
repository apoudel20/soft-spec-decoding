#!/usr/bin/env bash
# After the gate queue: extend capped answers to 1,024 tokens (smoke test first), export tokens, push, verify.
cd "$(dirname "$0")"
PY=.venv/Scripts/python
LOG=logs/run_long.log
until grep -q "=== gate queue done" logs/run_gate.log 2>/dev/null; do sleep 60; done
echo "=== long start $(date)" >> $LOG
$PY extend_long.py --smoke --cap 420 >> $LOG 2>&1 || { echo "FAILED extend smoke" >> $LOG; exit 1; }
rm -f data/*_long_smoke.pt
$PY extend_long.py --cap 1024 >> $LOG 2>&1 || { echo "FAILED extend" >> $LOG; exit 1; }
$PY export_tokens.py >> $LOG 2>&1 || { echo "FAILED export" >> $LOG; exit 1; }
REPO=/c/Users/aayus/Projects/soft-spec-decoding
cp extend_long.py export_tokens.py run_long.sh "$REPO/"
cd "$REPO"
git add tokens extend_long.py export_tokens.py run_long.sh && git commit -q -m "Long answers: capped answers continued to 1,024 tokens" && git push -q origin main \
  && git fetch -q origin main && [ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ] \
  && echo "PUSHED AND VERIFIED $(git rev-parse --short HEAD)" >> "$OLDPWD/$LOG" || echo "PUSH FAILED long" >> "$OLDPWD/$LOG"
echo "=== long done $(date)" >> "$OLDPWD/$LOG"
