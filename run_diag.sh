#!/usr/bin/env bash
# Post-gate diagnostic (does not count as a gate pass): is the ~94/93/78% reconstruction ceiling the recipe or a limit?
# (1) plain autoencoder, no VAE noise, no latent dropout, beta 0, same size and steps as z2048_full
# (2) bigger decoder (d 1024, depth 6) trained 3x longer, regularised as in the gate
cd "$(dirname "$0")"
PY=.venv/Scripts/python
LOG=logs/run_diag.log
until grep -q "=== long done" logs/run_long.log 2>/dev/null; do sleep 60; done
echo "=== diag start $(date)" >> $LOG
$PY ae_gate.py --name diag_plain_z2048 --zdim 2048 --det --beta 0 --zdrop 0 >> logs/ae_diag_plain_z2048.log 2>&1 || echo "FAILED diag plain" >> $LOG
grep -h "GATE" logs/ae_diag_plain_z2048.log >> $LOG
$PY ae_gate.py --name diag_big_z2048 --zdim 2048 --d 1024 --depth 6 --steps 36000 >> logs/ae_diag_big_z2048.log 2>&1 || echo "FAILED diag big" >> $LOG
grep -h "GATE" logs/ae_diag_big_z2048.log >> $LOG
echo "=== diag done $(date)" >> $LOG
