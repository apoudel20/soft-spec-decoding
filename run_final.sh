#!/usr/bin/env bash
# After the scale queue: one apples-to-apples benchmark (DFlash, chain, both samplers' trees), then examples.
cd "$(dirname "$0")"
P=.venv/Scripts/python
until grep -q "=== scale done" logs/run_scale2.log 2>/dev/null; do sleep 60; done
sleep 30
echo "=== final benchmark $(date +%H:%M)"
$P e2e.py --sets mtbench gsm8k humaneval --n 20 --max_new 256 --drafters dflash gister_v2 gister_tree16d_n2 "gister_tree16d_n2@stage2_gister_flow_big" --out e2e_final > logs/e2e_final.log 2>&1
grep -vE "^W1" logs/e2e_final.log | grep -E "AR|tau"
echo "=== examples $(date +%H:%M)"
$P examples.py --s2 stage2_gister_flow_big --k 8 --rounds 12 > logs/examples.log 2>&1 || $P examples.py --s2 stage2_gister_flow --k 8 --rounds 12 > logs/examples.log 2>&1
grep -E "mean kept|===|Error" logs/examples.log
echo "=== final done $(date +%H:%M)"
echo "=== context ablation $(date +%H:%M)"
$P stage2.py --ae gist_hf_z2048 --mode det --raw --hist 8 --steps 6000 > logs/stage2_raw_h8.log 2>&1
grep -E "^dev_.*\{|===" logs/stage2_raw_h8.log | cut -c1-200
echo "=== ablation done $(date +%H:%M)"
echo "=== full-context test $(date +%H:%M)"
FC="--ae gist_hf_z2048 --mode flow --raw --ctx_pos 56 --ctx_layers 30,36 --train_cache cache_pb --d 1024 --depth 12 --dec_w 0 --t_shift 3 --nfe 2 --init stage2_gister_v2"
$P stage2.py $FC --smoke --bestk 4 > logs/stage2_ctx_smoke.log 2>&1
if grep -q "=== stage2" logs/stage2_ctx_smoke.log; then
  ($P wandb_follow.py --log logs/stage2_ctx56.log --name gister_flow_ctx56 --idle_min 240 --notes "Full-context test: 158M sampler from v2 init, 10k steps on cache_pb, plus raw states of the 56 positions before the chunk (layers 30, 36). Compare with the 158M sampler without context (stage2_eval_..._nfe2)." > logs/follow_ctx56.log 2>&1 &)
  $P stage2.py $FC --steps 10000 --bestk 8,16 > logs/stage2_ctx56.log 2>&1
  grep -E "^dev_.*\{|===" logs/stage2_ctx56.log | cut -c1-400
else
  echo "CTX SMOKE FAILED"; grep -vE "^W1" logs/stage2_ctx_smoke.log | tail -8
fi
echo "=== full-context done $(date +%H:%M)"
