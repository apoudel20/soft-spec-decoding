#!/usr/bin/env bash
# Prompt/prefix-conditioned autoencoder gate: k=8, z2048, previous 8 positions as context, plus the matched no-context baseline.
cd "$(dirname "$0")"
echo "=== ctx start $(date)"
.venv/Scripts/python ae_gate.py --name ctx8_z2048 --zdim 2048 --ctx 8 > logs/ae_ctx8_z2048.log 2>&1; tail -n 1 logs/ae_ctx8_z2048.log
.venv/Scripts/python ae_gate.py --name ctx8_nocond_z2048 --zdim 2048 --ctx 8 --nocond > logs/ae_ctx8_nocond_z2048.log 2>&1; tail -n 1 logs/ae_ctx8_nocond_z2048.log
grep -h "GATE" logs/ae_ctx8_z2048.log logs/ae_ctx8_nocond_z2048.log
echo "=== ctx done $(date)"
