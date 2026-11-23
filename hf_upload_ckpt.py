"""Publish the stage-2 autoencoder checkpoints and their gate results as a public Hugging Face model repo.

Run: python hf_upload_ckpt.py [--repo ablastvacant/soft-spec-decoding-ae-gate]
"""
import argparse
import glob
import json
import os

from huggingface_hub import HfApi

from common import CKPT, RES

CARD = """---
license: apache-2.0
tags:
- speculative-decoding
- qwen3
- autoencoder
base_model: Qwen/Qwen3-8B
---

# Chunk autoencoders over Qwen3-8B hidden states (stage-2 gate)

VAEs that compress k=8 consecutive positions of Qwen3-8B (4-bit nf4) hidden states into one latent and decode them
back, trained so the frozen Qwen3-8B LM head gives the same tokens. Part of a drafter design for lossless
speculative decoding: https://github.com/apoudel20/soft-spec-decoding (`ae_gate.py` defines the model and the
loading code, `cache_states.py` produces the inputs).

Input grid rows: hidden states after layers 18, 24, 30 and the final normed state ("full"), or the final row only
("final"). Each checkpoint is a dict `{"model": state_dict, "args": training args}`.

Gate fixed before training (dev sets, teacher-forced, against the target's own top-1): agreement at least 99% chat
and 99.5% math and code, mean TV at most 0.01 chat and 0.005 math and code, and at most a 1 point drop with latent
noise at 0.1x the per-dim spread.

| Run | Latent | Rows | Math agree | Code agree | Chat agree | Gate |
|---|---|---|---|---|---|---|
{rows}

Results JSONs with training curves, TV, KL, margin buckets and noise tests are in `results/`.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="ablastvacant/soft-spec-decoding-ae-gate")
    args = ap.parse_args()
    rows, pairs = [], []
    for p in sorted(glob.glob(os.path.join(RES, "ae_gate_*.json"))):
        name = os.path.basename(p)[len("ae_gate_"):-5]
        if name.endswith("_smoke"):
            continue
        r = json.load(open(p))
        s = r["sets"]
        rows.append(f"| {name} | {r['args']['zdim']} | {r['args']['rows']} | {s['dev_math']['agree']:.4f} | "
                    f"{s['dev_code']['agree']:.4f} | {s['dev_chat']['agree']:.4f} | {'pass' if r['pass'] else 'fail'} |")
        pairs.append((p, f"results/{os.path.basename(p)}"))
        ck = os.path.join(CKPT, f"ae_{name}.pt")
        if os.path.exists(ck):
            pairs.append((ck, f"ae_{name}.pt"))
    if not rows:
        raise SystemExit("no results to upload")
    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True, private=False)
    stage = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_stage")
    os.makedirs(stage, exist_ok=True)
    card = os.path.join(stage, "MODEL_README.md")
    open(card, "w", encoding="utf-8").write(CARD.replace("{rows}", "\n".join(rows)))
    api.upload_file(path_or_fileobj=card, path_in_repo="README.md", repo_id=args.repo)
    for src, dst in pairs:
        api.upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id=args.repo)
        print("uploaded", dst, flush=True)
    print("REPO FILES", json.dumps(api.list_repo_files(args.repo)))
    print(f"https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
