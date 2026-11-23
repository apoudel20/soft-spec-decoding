#!/usr/bin/env bash
# Publishing that follows the queues: HF checkpoints + W&B after the gate, refreshed HF dataset after the long answers.
cd "$(dirname "$0")"
PY=.venv/Scripts/python
LOG=logs/publish.log
until grep -q "=== gate queue done" logs/run_gate.log 2>/dev/null; do sleep 60; done
sleep 120   # let push_checked finish its git work first
$PY hf_upload_ckpt.py >> $LOG 2>&1 && echo "HF CKPT OK" >> $LOG || echo "FAILED hf ckpt" >> $LOG
$PY wandb_sync.py >> $LOG 2>&1 && echo "WANDB OK" >> $LOG || echo "FAILED wandb (login?)" >> $LOG
until grep -q "=== long done" logs/run_long.log 2>/dev/null; do sleep 60; done
$PY hf_upload.py >> $LOG 2>&1 && echo "HF DATASET OK" >> $LOG || echo "FAILED hf dataset" >> $LOG
$PY wandb_sync.py >> $LOG 2>&1 || true
echo "=== publish done $(date)" >> $LOG
