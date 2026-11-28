#!/bin/bash
# Decoder fine-tuned on predicted plans (dec), predictor and decoder fine-tuned together (e2e), and the no-plan
# control fine-tuned with the same recipe (ctrl). 64-token and 256-token plans.
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
for a in "plan_pw_G8c16s0.25 --arm dec" "plan_pw_G8c0 --arm ctrl" "plan_pw_G8c16s0.25 --arm e2e" "plan_pw_G32c16s0.25 --arm dec" "plan_pw_G32c16s0.25 --arm e2e"; do
  echo "=== start $a $(date +%H:%M)"
  $PY ft_pred.py --ckpt $a
done
echo "=== ft_pred queue done $(date +%H:%M)"
