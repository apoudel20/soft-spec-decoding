#!/usr/bin/env bash
# Layer sweep: where does Qwen3-8B keep the gist of its own future, and how much more does it know than Qwen3-0.6B?
# 1) generate the 8B's own greedy answers with Ollama (train prompts + held-out dev prompts), 2) probe every
# second layer of both models, 3) push results.
cd "$(dirname "$0")"
P=./.venv/Scripts/python
REPO=/c/Users/aayus/Projects/soft-spec-decoding
until grep -q "^exit" logs/ollama_pull.log 2>/dev/null; do sleep 10; done
grep -q "^exit 0" logs/ollama_pull.log || { echo "OLLAMA PULL FAILED"; tail -5 logs/ollama_pull.log; exit 1; }

echo "=== ollama smoke $(date +%H:%M)"
$P -c "
import gen_q3 as g
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(g.HF_ID)
p = tok.apply_chat_template([{'role': 'user', 'content': 'What is 12 times 7?'}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
print(repr(g.ollama(p, 40)))
" 2>&1 | grep -v -E "triton|Redirects" || { echo "OLLAMA SMOKE FAILED"; exit 1; }

echo "=== generate $(date +%H:%M)"
$P gen_q3.py --n_train 600 --n_eval 40 > logs/gen_q3.log 2>&1 || { echo "GEN FAILED"; tail -8 logs/gen_q3.log; exit 1; }
grep -E "sequences|done in" logs/gen_q3.log

echo "=== probe smoke $(date +%H:%M)"
if $P probes_q3.py --n_train 20 --per_seq 4 --sets math:40 --layer_step 12 --out probes_q3_smoke > logs/probes_q3_smoke.log 2>&1; then
  rm -f results/probes_q3_smoke.json
  echo "=== probe full $(date +%H:%M)"
  $P probes_q3.py --n_train 600 --per_seq 24 --out probes_q3 > logs/probes_q3.log 2>&1 && echo "probes ok" || { echo "PROBES FAILED"; tail -8 logs/probes_q3.log; }
else
  echo "PROBE SMOKE FAILED"; tail -12 logs/probes_q3_smoke.log
fi
cp results/probes_q3.json "$REPO/results/" 2>/dev/null; cp gen_q3.py probes_q3.py run_q3.sh "$REPO/"
( cd "$REPO" && git add results/probes_q3.json gen_q3.py probes_q3.py run_q3.sh 2>/dev/null; git commit -q -m "Layer sweep: Qwen3-8B vs Qwen3-0.6B knowledge of the 8B's own future, every second layer" && git push -q origin main && echo pushed ) || echo "nothing pushed"
echo "=== q3 done $(date +%H:%M)"
