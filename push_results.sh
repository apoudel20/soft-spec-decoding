#!/usr/bin/env bash
# After the whole chain (Sketch A, DFlash baseline, sketches B-E) finishes, copy results and code to the public repo and push.
cd "$(dirname "$0")"
REPO=/c/Users/aayus/Projects/soft-spec-decoding
until grep -q "=== sketches done" logs/run_sketches.log 2>/dev/null; do sleep 60; done
cp results/dflash_plan.json results/dflash_eval.json results/sketches_eval.json results/kv_oracle.json "$REPO/results/" 2>/dev/null
cp kv_oracle.py dflash_model.py dflash_eval.py dflash_plan.py sketches.py run_plan.sh run_dflash.sh run_sketches.sh push_results.sh "$REPO/"
cd "$REPO" && git add results/dflash_plan.json results/dflash_eval.json results/sketches_eval.json results/kv_oracle.json kv_oracle.py \
  dflash_model.py dflash_eval.py dflash_plan.py sketches.py run_plan.sh run_dflash.sh run_sketches.sh push_results.sh 2>/dev/null
git commit -q -m "DFlash baseline and sketch A-E results (raw, before write-up)" && git push -q origin main && echo "pushed $(date +%H:%M)" || echo "nothing pushed"
