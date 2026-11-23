"""Generate more Qwen3-8B greedy training answers, disjoint from data/q3_train.pt.

Re-creates the exact sampling used for q3_train.pt (random.Random(0), same order of calls) to know which prompts
were used, then samples n more prompts from the rest with the same mix (1/2 chat, 1/4 math, 1/4 code).
Output: data/q3_train_more.pt, same item format {"prompt": ids, "resp": ids}.
"""
import argparse
import os
import random

import torch
from transformers import AutoTokenizer

from common import DATA, train_prompts
from gen_q3 import HF_ID, run, unload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1400)
    ap.add_argument("--max_new", type=int, default=320)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    out = os.path.join(DATA, "q3_train_more.pt")
    if os.path.exists(out):
        print("exists", out); return
    tok = AutoTokenizer.from_pretrained(HF_ID)
    tp = train_prompts(2000)
    by = {k: [t for kk, t in tp if kk == k] for k in ("chat", "math", "code")}
    rng, n0 = random.Random(0), 600
    used = set(rng.sample(by["chat"], n0 // 2) + rng.sample(by["math"], n0 // 4) + rng.sample(by["code"], n0 - n0 // 2 - n0 // 4))
    rest = {k: [t for t in v if t not in used] for k, v in by.items()}
    r2, n = random.Random(1), args.n
    texts = r2.sample(rest["chat"], n // 2) + r2.sample(rest["math"], n // 4) + r2.sample(rest["code"], n - n // 2 - n // 4)
    assert not (set(texts) & used)
    print({k: len(v) for k, v in rest.items()}, "sampling", len(texts), flush=True)
    items = run(tok, texts, args.max_new, args.workers)
    torch.save(items, out)
    print(f"train_more: {len(items)} sequences, {sum(len(x['resp']) for x in items)} resp tokens", flush=True)
    unload()
    print("=== gen_more done", flush=True)


if __name__ == "__main__":
    main()
