"""Stage 0 and Stage 1: how much is better tree selection worth?

Target Qwen3-8B (4-bit NF4), drafter Qwen3-0.6B (bf16), greedy, held-out dev prompts (chat, math, code).

Stage 0 (hardware knee): time one target forward of T new tokens on top of a ~300-token cache,
for T in 1..128. Shows how many tree nodes one verification pass can check for about the price of one.

Stage 1 (selection gap): at sampled positions of the target's own greedy continuation, the drafter grows a
forest by beam expansion (top-8 children per node, keep the 64 best paths per depth by cumulative drafter
log-prob, depth 8, up to ~460 nodes). Then, for node budgets B, compare
  drafter  : the B nodes with the highest cumulative drafter log-prob (EAGLE-2-style selection; ancestor-closed)
  oracle   : the best any selection could do from this forest = the length of the target's actual greedy path
             that the forest contains (needs at most 8 nodes)
Accepted length = the longest prefix of the target's greedy continuation present in the selected nodes
(the bonus token is not counted). Under greedy decoding this is exactly what tree verification accepts.
The gap (oracle / drafter - 1) bounds what any smarter selection, such as a target-guided one, can gain from
this forest. Go if the gap is at least 20% at B=32, kill if under 10% (from the review's staged plan).

Run: python selection_gap.py --n 30 --max_new 160 --stride 8
Output: results/selection_gap.json
"""
import argparse
import json
import math
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common import RES, dev_prompts

TARGET_ID, DRAFT_ID = "Qwen/Qwen3-8B", "Qwen/Qwen3-0.6B"
BUDGETS = [8, 16, 32, 64, 128]


def chat(tok, text):
    s = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
    return tok(s, add_special_tokens=False)["input_ids"]


@torch.no_grad()
def stage0(m, prefix_ids):
    ids = torch.tensor([prefix_ids], device="cuda")
    out = m(input_ids=ids, use_cache=True)
    times = {}
    for T in [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]:
        new = torch.randint(100, 20000, (1, T), device="cuda")
        ts = []
        for rep in range(12):
            cache = out.past_key_values
            extra = cache.get_seq_length() - len(prefix_ids)
            if extra > 0:
                cache.crop(-extra)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            m(input_ids=new, past_key_values=cache, use_cache=True)
            torch.cuda.synchronize()
            if rep >= 2:
                ts.append(time.perf_counter() - t0)
        times[T] = round(1000 * sorted(ts)[len(ts) // 2], 2)
        print(f"  T={T:4d}  {times[T]} ms", flush=True)
    return times


@torch.no_grad()
def forest(d, ctx, K=8, W=64, D=8):
    """Beam forest from context ids. Returns dict path(tuple) -> cumulative drafter log-prob."""
    nodes = {}
    beams = [((), 0.0)]
    for depth in range(D):
        seqs = [ctx + list(p) for p, _ in beams]
        L = max(len(s) for s in seqs)
        x = torch.full((len(seqs), L), 0, dtype=torch.long, device="cuda")
        att = torch.zeros((len(seqs), L), dtype=torch.long, device="cuda")
        for i, s in enumerate(seqs):  # left-pad so the last position is aligned
            x[i, L - len(s):] = torch.tensor(s, device="cuda")
            att[i, L - len(s):] = 1
        lg = d(input_ids=x, attention_mask=att, logits_to_keep=1).logits[:, -1].float()
        lp = torch.log_softmax(lg, -1)
        top = lp.topk(K, -1)
        cand = []
        for i, (p, c) in enumerate(beams):
            for j in range(K):
                cand.append((p + (int(top.indices[i, j]),), c + float(top.values[i, j])))
        cand.sort(key=lambda z: -z[1])
        beams = cand[:W]
        for p, c in beams:
            nodes[p] = c
    return nodes


def accepted(selected, path):
    n = 0
    for k in range(1, len(path) + 1):
        if tuple(path[:k]) in selected:
            n = k
        else:
            break
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--max_new", type=int, default=160)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--out", default="selection_gap")
    args = ap.parse_args()
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(TARGET_ID)
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    t = AutoModelForCausalLM.from_pretrained(TARGET_ID, quantization_config=q, dtype=torch.bfloat16, device_map="cuda")
    t.eval()
    print(f"target loaded, GPU {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)
    res = {"args": vars(args), "sets": {}}
    seqs = {}
    for s in ("math", "code", "chat"):
        seqs[s] = []
        for text in dev_prompts(s, args.n):
            p = chat(tok, text)
            if len(p) > 300:
                continue
            g = t.generate(torch.tensor([p], device="cuda"), max_new_tokens=args.max_new, do_sample=False,
                           temperature=None, top_p=None, top_k=None)
            seqs[s].append({"prompt": p, "resp": g[0, len(p):].tolist()})
        print(f"{s}: {len(seqs[s])} target continuations  {(time.time() - t0) / 60:.1f} min", flush=True)
    longest = max((x for v in seqs.values() for x in v), key=lambda x: len(x["prompt"]))
    print("stage 0: target forward time vs new tokens", flush=True)
    res["stage0_ms"] = stage0(t, (longest["prompt"] + longest["resp"])[:300])
    del t
    torch.cuda.empty_cache()
    d = AutoModelForCausalLM.from_pretrained(DRAFT_ID, dtype=torch.bfloat16, device_map="cuda")
    d.eval()
    for s, items in seqs.items():
        acc = {B: [] for B in BUDGETS}
        orc, sizes = [], []
        for it in items:
            full = it["prompt"] + it["resp"]
            for pos in range(len(it["prompt"]), len(full) - 8, args.stride):
                path = full[pos:pos + 8]
                nodes = forest(d, full[:pos])
                ranked = sorted(nodes, key=lambda p: -nodes[p])
                for B in BUDGETS:
                    acc[B].append(accepted(set(ranked[:B]), path))
                orc.append(accepted(set(nodes), path))
                sizes.append(len(nodes))
        r = {"positions": len(orc), "forest_nodes_mean": sum(sizes) / len(sizes),
             "oracle_mean": sum(orc) / len(orc)}
        g = torch.Generator().manual_seed(0)
        O = torch.tensor(orc, dtype=torch.float)
        for B in BUDGETS:
            r[f"drafter_B{B}_mean"] = sum(acc[B]) / len(acc[B])
            r[f"gap_B{B}"] = r["oracle_mean"] / max(r[f"drafter_B{B}_mean"], 1e-9) - 1
            Dv = torch.tensor(acc[B], dtype=torch.float)
            idx = torch.randint(0, len(O), (2000, len(O)), generator=g)  # bootstrap over positions
            gaps = (O[idx].mean(1) / Dv[idx].mean(1).clamp_min(1e-9) - 1).sort().values
            r[f"gap_B{B}_ci95"] = [round(float(gaps[50]), 4), round(float(gaps[1949]), 4)]
        r["per_position"] = {"oracle": orc, **{f"drafter_B{B}": acc[B] for B in BUDGETS}}
        res["sets"][s] = r
        print(s, {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()},
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
        json.dump(res, open(os.path.join(RES, args.out + ".json"), "w"), indent=1)
    res["minutes"] = (time.time() - t0) / 60
    json.dump(res, open(os.path.join(RES, args.out + ".json"), "w"), indent=1)
    print("wrote", os.path.join(RES, args.out + ".json"))


if __name__ == "__main__":
    main()
