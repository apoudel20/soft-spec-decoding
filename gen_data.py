"""Generate self-distilled training data: the target's own responses to train prompts.

Half the prompts are answered greedily, half with temperature-1 sampling (no top-p/top-k/penalty).
Output: data/train.pt = list of {"src", "mode", "prompt": [ids], "resp": [ids]} and data/train_meta.json.

Run: python gen_data.py --n_chat 4000 --n_math 2500 [--limit 64 for a speed test]
"""
import argparse
import json
import os
import random
import time

import torch
from transformers import AutoModelForCausalLM

from common import DATA, chat_ids, train_prompts
from model import EOS_IDS, TARGET, load_tokenizer, model_dir

MAX_PROMPT = 160
MAX_RESP = 352


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_chat", type=int, default=4000)
    ap.add_argument("--n_math", type=int, default=2500)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="train")
    args = ap.parse_args()
    random.seed(0)
    tok = load_tokenizer()
    ps = train_prompts(args.n_chat)
    chat = [p for p in ps if p[0] == "chat"]
    math = [p for p in ps if p[0] == "math"]
    code = [p for p in ps if p[0] == "code"]
    random.shuffle(math)
    ps = chat + math[: args.n_math] + code
    random.shuffle(ps)
    items = []
    for i, (src, text) in enumerate(ps):
        ids = chat_ids(tok, text)
        if len(ids) <= MAX_PROMPT:
            items.append({"src": src, "mode": "greedy" if i % 2 == 0 else "sample", "prompt": ids})
    if args.limit:
        items = items[: args.limit]
    print(f"{len(items)} prompts after length filter", {s: sum(it['src'] == s for it in items) for s in ('chat', 'math', 'code')}, flush=True)

    hf = AutoModelForCausalLM.from_pretrained(model_dir(TARGET), dtype=torch.bfloat16).cuda().eval()
    pad = 151643
    t0 = time.time()
    done = 0
    for mode in ("greedy", "sample"):
        group = sorted([it for it in items if it["mode"] == mode], key=lambda it: len(it["prompt"]))
        for b in range(0, len(group), args.bs):
            batch = group[b:b + args.bs]
            L = max(len(it["prompt"]) for it in batch)
            ids = torch.full((len(batch), L), pad, dtype=torch.long)
            att = torch.zeros((len(batch), L), dtype=torch.long)
            for j, it in enumerate(batch):
                ids[j, L - len(it["prompt"]):] = torch.tensor(it["prompt"])
                att[j, L - len(it["prompt"]):] = 1
            kw = dict(do_sample=False) if mode == "greedy" else dict(do_sample=True, temperature=1.0)
            with torch.no_grad():
                out = hf.generate(ids.cuda(), attention_mask=att.cuda(), max_new_tokens=MAX_RESP, pad_token_id=pad,
                                  eos_token_id=list(EOS_IDS), top_p=None, top_k=None, repetition_penalty=1.0, **kw)
            for j, it in enumerate(batch):
                r = out[j, L:].tolist()
                cut = next((k for k, t in enumerate(r) if t in EOS_IDS), None)
                it["resp"] = r[: cut + 1] if cut is not None else r
            done += len(batch)
            el = time.time() - t0
            print(f"{done}/{len(items)}  {el / 60:.1f} min elapsed, eta {el / done * (len(items) - done) / 60:.1f} min", flush=True)
    torch.save(items, os.path.join(DATA, args.out + ".pt"))
    lens = [len(it["resp"]) for it in items]
    meta = {"n": len(items), "mean_resp_len": sum(lens) / len(lens), "truncated": sum(l >= MAX_RESP for l in lens),
            "by_src": {s: sum(it["src"] == s for it in items) for s in ("chat", "math", "code")}, "minutes": (time.time() - t0) / 60}
    json.dump(meta, open(os.path.join(DATA, args.out + "_meta.json"), "w"), indent=1)
    print(meta)


if __name__ == "__main__":
    main()
