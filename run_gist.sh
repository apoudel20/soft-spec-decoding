#!/usr/bin/env bash
# Decisive comparison, one job at a time: gist latent (target states + the head's top-1 reading in, exact tokens out)
# vs token latent, both through stage 2.
cd "$(dirname "$0")"
.venv/Scripts/python stage2.py --tok tokvae_z128 --mode det --steps 6000 > logs/stage2_tok.log 2>&1; grep -E "^dev_" logs/stage2_tok.log
.venv/Scripts/python ae_gate.py --name gist_hf_z2048 --zdim 2048 --ce 1 --headfeat --cos_w 0.1 --mid_w 0.1 --steps 8000 > logs/ae_gist_hf_z2048.log 2>&1; grep -E "^dev_|GATE" logs/ae_gist_hf_z2048.log | cut -c1-90
.venv/Scripts/python stage2.py --ae gist_hf_z2048 --mode det --steps 6000 > logs/stage2_gist.log 2>&1; grep -E "^dev_" logs/stage2_gist.log
echo "=== gist done $(date)"
