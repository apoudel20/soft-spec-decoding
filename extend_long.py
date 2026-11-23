"""Extend answers that stopped at the 320-token generation cap, up to `--cap` tokens, without regenerating.

Greedy decoding is deterministic, so continuing from prompt + answer-so-far (raw mode) gives the same text a longer
run would have produced. Only answers that hit the old cap are continued. The new tokens are re-tokenised with
the HF tokenizer and appended. Writes new files (data/<name>_long.pt), the originals are left untouched.

Run:    python extend_long.py [--cap 1024] [--smoke]
Output: data/q3_train_long.pt, data/q3_train_more_long.pt, data/q3_dev_<set>_40_long.pt
"""
import argparse
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import AutoTokenizer

from common import DATA
from gen_q3 import HF_ID, MODEL, OLLAMA, unload

OLD_CAP = 320


def cont(text, n):
    body = {"model": MODEL, "prompt": text, "raw": True, "stream": False, "keep_alive": "10m",
            "options": {"temperature": 0, "top_k": 1, "top_p": 1.0, "presence_penalty": 0.0, "repeat_penalty": 1.0,
                        "num_predict": n, "num_ctx": 2048, "seed": 0}}
    req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        return json.loads(r.read())["response"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(HF_ID)
    names = ["q3_dev_math_40", "q3_dev_code_40", "q3_dev_chat_40", "q3_train", "q3_train_more"]
    t0 = time.time()
    for name in names:
        src, dst = os.path.join(DATA, name + ".pt"), os.path.join(DATA, name + ("_long_smoke" if args.smoke else "_long") + ".pt")
        if os.path.exists(dst) or not os.path.exists(src):
            print("skip", name, flush=True)
            continue
        items = torch.load(src)
        todo = [i for i, it in enumerate(items) if len(it["resp"]) >= OLD_CAP - 1]
        if args.smoke:
            items, todo = [items[i] for i in todo[:2]], [0, 1][:len(todo)]
        print(f"{name}: {len(todo)} of {len(items)} hit the cap", flush=True)
        out = [dict(it) for it in items]

        def one(i):
            it = items[i]
            text = tok.decode(it["prompt"] + it["resp"])
            new = cont(text, args.cap - len(it["resp"]))
            out[i] = {"prompt": it["prompt"], "resp": it["resp"] + tok(new, add_special_tokens=False)["input_ids"],
                      "extended": True}

        with ThreadPoolExecutor(args.workers) as ex:
            for n, _ in enumerate(ex.map(one, todo)):
                if n % 50 == 0:
                    print(f"  {n}/{len(todo)}  {(time.time() - t0) / 60:.1f} min", flush=True)
        torch.save(out, dst)
        L = [len(x["resp"]) for x in out]
        print(f"{name}: mean answer {sum(L) / len(L):.0f} tokens, at new cap {sum(l >= args.cap - 1 for l in L)}", flush=True)
        if args.smoke:
            for x in out[:1]:
                print(repr(tok.decode(x["resp"][OLD_CAP - 20:OLD_CAP + 30])), flush=True)
            break
    unload()
    print("=== extend done", flush=True)


if __name__ == "__main__":
    main()
