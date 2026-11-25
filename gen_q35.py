"""Generate Qwen3.5-9B's own greedy responses through the local Ollama server, for the scale probe.

The probe asks whether a model knows its own future, so the text has to be the 9B's own output.
Ollama (llama.cpp, Q4_K_M) is used only because it generates much faster than HF on this GPU.
Prompts use the HF chat template with thinking disabled and are sent raw, so the template matches.
Responses are re-tokenised with the HF tokenizer.

Output: data/q35_train.pt (mixed chat/math/code training prompts, disjoint from eval) and
        data/q35_ref_<set>_<n>.pt for the eval prompt subsets. Each item is {"prompt": ids, "resp": ids}.
Run:    python gen_q35.py --n_train 600 --n_eval 40
"""
import argparse
import json
import os
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import AutoTokenizer

from common import DATA, eval_prompts, train_prompts

HF_ID = "Qwen/Qwen3.5-9B"
OLLAMA = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen3.5:9b"


def ollama(prompt, max_new, keep_alive="10m"):
    body = {"model": MODEL, "prompt": prompt, "raw": True, "stream": False, "keep_alive": keep_alive,
            "options": {"temperature": 0, "top_k": 1, "top_p": 1.0, "presence_penalty": 0.0,
                        "repeat_penalty": 1.0, "num_predict": max_new, "num_ctx": 1024, "seed": 0}}
    req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["response"]


def unload():
    req = urllib.request.Request(OLLAMA, data=json.dumps({"model": MODEL, "keep_alive": 0}).encode(),
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=120).read()


def run(tok, texts, max_new, workers):
    prompts = [tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False) for t in texts]
    keep = [i for i, p in enumerate(prompts) if len(tok(p, add_special_tokens=False)["input_ids"]) <= 400]
    out = [None] * len(prompts)
    t0 = time.time()

    def one(i):
        out[i] = ollama(prompts[i], max_new)

    with ThreadPoolExecutor(workers) as ex:
        for n, _ in enumerate(ex.map(one, keep)):
            if n % 50 == 0:
                print(f"  {n}/{len(keep)}  {(time.time() - t0) / 60:.1f} min", flush=True)
    items = []
    for i in keep:
        resp = tok(out[i], add_special_tokens=False)["input_ids"]
        if len(resp) >= 8:
            items.append({"prompt": tok(prompts[i], add_special_tokens=False)["input_ids"], "resp": resp})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=600)
    ap.add_argument("--n_eval", type=int, default=40)
    ap.add_argument("--max_new", type=int, default=320)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(HF_ID)
    rng = random.Random(0)
    t0 = time.time()
    for s in ("gsm8k", "mtbench", "humaneval"):
        path = os.path.join(DATA, f"q35_ref_{s}_{args.n_eval}.pt")
        if not os.path.exists(path):
            items = run(tok, eval_prompts(s, args.n_eval), args.max_new, args.workers)
            torch.save(items, path)
            print(f"{s}: {len(items)} sequences, mean resp {sum(len(x['resp']) for x in items) / len(items):.0f} tokens", flush=True)
    path = os.path.join(DATA, "q35_train.pt")
    if not os.path.exists(path):
        tp = train_prompts(2000)
        by = {k: [t for kk, t in tp if kk == k] for k in ("chat", "math", "code")}
        n = args.n_train
        texts = rng.sample(by["chat"], n // 2) + rng.sample(by["math"], n // 4) + rng.sample(by["code"], n - n // 2 - n // 4)
        items = run(tok, texts, args.max_new, args.workers)
        torch.save(items, path)
        print(f"train: {len(items)} sequences", flush=True)
    unload()
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
