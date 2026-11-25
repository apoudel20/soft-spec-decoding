#!/usr/bin/env bash
# Overnight queue, one GPU job at a time.
cd "$(dirname "$0")"
P=.venv/Scripts/python
echo "=== tree rerun $(date +%H:%M)"
$P e2e.py --sets mtbench gsm8k --n 20 --max_new 256 --drafters gister_v2 gister_tree8d_n2 gister_tree16d_n2 --out e2e_tree_fast > logs/e2e_tree_fast.log 2>&1
grep -vE "^W1" logs/e2e_tree_fast.log | grep -E "AR|tau"
echo "=== sampler long $(date +%H:%M)"
F="--ae gist_hf_z2048 --mode flow --raw --train_cache cache_pb --d 1024 --depth 12 --dec_w 0 --t_shift 3"
($P wandb_follow.py --log logs/stage2_flow_long.log --name gister_flow_long --notes "Gister sampler continued 30k steps from the 10k-step sampler, nfe 2 at eval. lat = velocity MSE." > logs/follow_flow_long.log 2>&1 &)
$P stage2.py $F --nfe 2 --lr 2e-4 --steps 30000 --init stage2_gister_flow_v1 --bestk 8,16 > logs/stage2_flow_long.log 2>&1
grep -E "^dev_.*\{|===" logs/stage2_flow_long.log | cut -c1-500
cp ckpt/stage2_gist_hf_z2048_flow_raw_cache_pb_d1024x12_s30000_nfe2.pt ckpt/stage2_gister_flow.pt
echo "=== tree with long sampler $(date +%H:%M)"
$P e2e.py --sets mtbench gsm8k humaneval --n 20 --max_new 256 --drafters gister_v2 gister_tree8d_n2 gister_tree16d_n2 --out e2e_tree_long > logs/e2e_tree_long.log 2>&1
grep -vE "^W1" logs/e2e_tree_long.log | grep -E "AR|tau"
echo "=== night done $(date +%H:%M)"
