#!/usr/bin/env bash
# Scale probe on Qwen3.5-9B vs Qwen3.5-0.8B, after the planner rerun frees the GPU.
# 1) generate the 9B's own greedy answers with Ollama, 2) probe every second layer of both models, 3) push.
cd "$(dirname "$0")"
P=./.venv/Scripts/python
REPO=/c/Users/aayus/Projects/soft-spec-decoding
until grep -q "=== post-planner2 done" logs/post_planner2.log 2>/dev/null; do sleep 30; done
until grep -q "^exit" logs/q35_download.log 2>/dev/null; do sleep 30; done
grep -q "^exit 0" logs/q35_download.log || { echo "DOWNLOAD FAILED"; tail -5 logs/q35_download.log; exit 1; }

echo "=== ollama smoke $(date +%H:%M)"
$P -c "
import gen_q35 as g
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(g.HF_ID)
p = tok.apply_chat_template([{'role': 'user', 'content': 'What is 12 times 7?'}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
print(repr(g.ollama(p, 40)))
" 2>&1 | grep -v -E "triton|Redirects" || { echo "OLLAMA SMOKE FAILED"; exit 1; }

echo "=== generate $(date +%H:%M)"
$P gen_q35.py --n_train 600 --n_eval 40 > logs/gen_q35.log 2>&1 || { echo "GEN FAILED"; tail -8 logs/gen_q35.log; exit 1; }
grep -E "sequences|done in" logs/gen_q35.log

echo "=== probe smoke $(date +%H:%M)"
if $P probes_q35.py --n_train 20 --per_seq 4 --sets gsm8k:40 --layer_step 8 --out probes_q35_smoke > logs/probes_q35_smoke.log 2>&1; then
  rm -f results/probes_q35_smoke.json
  echo "=== probe full $(date +%H:%M)"
  $P probes_q35.py --n_train 600 --per_seq 24 --out probes_q35 > logs/probes_q35.log 2>&1 && echo "probes ok" || { echo "PROBES FAILED"; tail -8 logs/probes_q35.log; }
else
  echo "PROBE SMOKE FAILED"; tail -12 logs/probes_q35_smoke.log
fi
cp results/probes_q35.json "$REPO/results/" 2>/dev/null; cp gen_q35.py probes_q35.py run_q35.sh "$REPO/"
( cd "$REPO" && git add results/probes_q35.json gen_q35.py probes_q35.py run_q35.sh 2>/dev/null; git commit -q -m "Scale probe: Qwen3.5-9B vs Qwen3.5-0.8B knowledge of the 9B's own future, every second layer" && git push -q origin main && echo pushed ) || echo "nothing pushed"
echo "=== q35 done $(date +%H:%M)"
