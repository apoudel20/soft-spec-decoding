#!/usr/bin/env bash
# Main evaluation queue. Greedy on all arms, then T=1 on a subset. Batch size 1, 192 new tokens.
# Eval subsets: MT-Bench 30 (first turns, category-stratified), GSM8K test first 30, HumanEval first 30.
# The best Arm D drafter for C+D is chosen on dev prompts only.
cd "$(dirname "$0")"
P=./.venv/Scripts/python
F="ResourceTracker|resource_tracker|_recursion|Redirects|^Traceback \(most recent call last\):$"
EV="mtbench:30 gsm8k:30 humaneval:30"
run() { out=$1; shift; echo "=== $out $(date +%H:%M)"; $P eval.py --max_new 192 --out "$out" "$@" 2>&1 | grep -v -E "$F" | grep -E "==|tau=|wrote|Error|error"; }

# 1. dev selection of the Arm D drafter (dev prompts only)
run dev_select_d --sets dev_chat:10 dev_math:10 dev_code:10 --methods \
  "D:ckpt=d1_soft,B=8,K=2" "D:ckpt=d3_soft,B=8,K=2" "D:ckpt=d4_soft,B=8,K=2" "D:ckpt=d2_soft,B=8,K=2" \
  "D:ckpt=d1_hard,B=8,K=2" "D:ckpt=d3_hard,B=8,K=2" "D:ckpt=d4_hard,B=8,K=2"
BEST=$($P -c "
import json; r=json.load(open('results/dev_select_d.json'))['sets']
specs=[k for k in next(iter(r.values()))['methods'] if k!='ar']
best=max(specs, key=lambda s: sum(v['methods'][s]['tau'] for v in r.values()))
print(best.split('ckpt=')[1].split(',')[0])")
echo "best dev drafter: $BEST"

# 2. greedy: baselines and Arm A
run eval_greedy_base --sets $EV --methods "sd:g=3" "sd:g=5" "sd:g=7" "jacobi:H=8" \
  "softjacobi:H=4,S=8" "softjacobi:H=4,S=8,tail=detached"
# 3. greedy: Arm D (all labels x soft/hard inputs, K and B sweeps on the dev-selected drafter)
run eval_greedy_D --sets $EV --methods \
  "D:ckpt=d1_soft,B=8,K=2" "D:ckpt=d3_soft,B=8,K=2" "D:ckpt=d4_soft,B=8,K=2" "D:ckpt=d2_soft,B=8,K=2" \
  "D:ckpt=d1_hard,B=8,K=2" "D:ckpt=d3_hard,B=8,K=2" "D:ckpt=d4_hard,B=8,K=2" \
  "D:ckpt=$BEST,B=8,K=1" "D:ckpt=$BEST,B=8,K=3" "D:ckpt=$BEST,B=4,K=2" "D:ckpt=$BEST,B=16,K=2"
# 4. greedy: Arm C vs C-hard, H x S sweep (detached tail), plus attached layout and sharpening ablations
CS=""
for ck in c_soft c_hard; do
  inp=soft; [ $ck = c_hard ] && inp=hard
  for hs in "2,4" "2,8" "2,16" "4,4" "4,8" "4,16" "8,8" "8,16"; do
    H=${hs%,*}; S=${hs#*,}; CS="$CS C:ckpt=$ck,inp=$inp,H=$H,S=$S"
  done
done
run eval_greedy_C --sets $EV --methods $CS "C:ckpt=c_soft,H=4,S=8,tail=attached" "C:ckpt=c_hard,inp=hard,H=4,S=8,tail=attached" \
  "C:ckpt=c_soft,H=4,S=8,sharpen=0.5"
# 5. greedy: C+D
run eval_greedy_CD --sets $EV --methods "CD:ckpt=c_soft,dckpt=$BEST,H=4,S=8,K=2" "CD:ckpt=c_hard,inp=hard,dckpt=$BEST,H=4,S=8,K=2" \
  "CD:ckpt=c_soft,dckpt=$BEST,H=8,S=16,K=2"
# 6. temperature 1 with rejection sampling (subset)
run eval_t1 --temp 1.0 --sets $EV --methods "sd:g=5" "jacobi:H=8" "D:ckpt=$BEST,B=8,K=2" "D:ckpt=d1_soft,B=8,K=2" \
  "D:ckpt=d3_hard,B=8,K=2" "C:ckpt=c_soft,H=4,S=8" "C:ckpt=c_hard,inp=hard,H=4,S=8" "CD:ckpt=c_soft,dckpt=$BEST,H=4,S=8,K=2"
echo "=== all evals done $(date +%H:%M)"
