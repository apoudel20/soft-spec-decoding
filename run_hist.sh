#!/usr/bin/env bash
# Stage 2 reading the last 8 chunk latents (64 positions): gist vs token latent.
cd "$(dirname "$0")"
.venv/Scripts/python stage2.py --ae gist_hf_z2048 --mode det --steps 6000 --hist 8 > logs/stage2_gist_h8.log 2>&1; grep -E "^dev_.*\{" logs/stage2_gist_h8.log
.venv/Scripts/python stage2.py --tok tokvae_z128 --mode det --steps 6000 --hist 8 > logs/stage2_tok_h8.log 2>&1; grep -E "^dev_.*\{" logs/stage2_tok_h8.log
echo "=== hist done $(date)"
