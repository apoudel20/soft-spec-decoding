#!/usr/bin/env bash
# Scale step: 4M more on-policy positions (6M total) and a ~285M sampler, then trees end to end vs the 158M sampler.
cd "$(dirname "$0")"
P=.venv/Scripts/python
until grep -q "=== cache done" logs/cache_pb2.log 2>/dev/null; do sleep 20; done
sleep 30
echo "=== cache_pb2 done $(date +%H:%M)"
OFF=$($P -c "import numpy as np;print((len(np.load('data/cache_pb/train_meta.npz')['seq'])+len(np.load('data/cache_pb2/train_meta.npz')['seq']))/1e6)")
echo "pb3 offset $OFF M tokens"
$P cache_states.py --pb_tokens 2.0 --pb_offset $OFF --pb_name cache_pb3 > logs/cache_pb3.log 2>&1
grep -E "^train|=== cache" logs/cache_pb3.log | cut -c1-120
echo "=== sampler 285M $(date +%H:%M)"
F="--ae gist_hf_z2048 --mode flow --raw --train_cache cache_pb,cache_pb2,cache_pb3 --d 1280 --depth 14 --dec_w 0 --t_shift 3 --nfe 2"
($P wandb_follow.py --log logs/stage2_flow_big.log --name gister_flow_285M_6M --notes "Scale step: ~285M flow sampler (d1280 x 14) from scratch on 6M on-policy positions (cache_pb + pb2 + pb3), t_shift 3, nfe 2. lat = velocity MSE." > logs/follow_flow_big.log 2>&1 &)
$P stage2.py $F --steps 24000 --bestk 8,16 > logs/stage2_flow_big.log 2>&1
grep -E "^dev_.*\{|===|Error" logs/stage2_flow_big.log | cut -c1-500
cp ckpt/stage2_gist_hf_z2048_flow_raw_cache_pb+cache_pb2+cache_pb3_d1280x14_s24000_nfe2.pt ckpt/stage2_gister_flow_big.pt
echo "=== trees big vs small $(date +%H:%M)"
$P e2e.py --sets mtbench gsm8k humaneval --n 20 --max_new 256 --drafters gister_v2 gister_tree16d_n2 "gister_tree16d_n2@stage2_gister_flow_big" "gister_tree8d_n2@stage2_gister_flow_big" --out e2e_tree_big > logs/e2e_tree_big.log 2>&1
grep -vE "^W1" logs/e2e_tree_big.log | grep -E "AR|tau"
echo "=== scale done $(date +%H:%M)"
