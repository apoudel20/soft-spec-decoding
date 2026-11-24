"""Cache Qwen3-8B (4-bit) hidden states for the autoencoder gate test.

For every response position p (state at p predicts token p+1, from the last prompt position to the second-last
token) store the hidden state after layers 18, 24, 30 and the final normed state (what the LM head reads) as fp16
rows in data/cache/<split>_L<layer>.npy. Also stores, per position, the sequence id, the next token, the target's
top-1 and the top-1/top-2 logit margin and flip radius (margin / ||W_top1 - W_top2||, i.e. how far the final state
must move to change the top-1 token), and per-dimension stats on the train split (mean, std, max |x|).

Splits: train (q3_train + q3_train_more), dev_math, dev_code, dev_chat.
Run:    python cache_states.py [--smoke] [--long]
Output: data/cache/* (data/cache_long/* with --long), results/state_stats.json
"""
import argparse
import json
import os

import numpy as np
import torch

from common import DATA, RES, load_items
from dflash_eval import load_target

LAYERS = [18, 24, 30, 36]


def splits(smoke, long=False):
    x = "_long" if long else ""
    s = {f"dev_{k}": load_items(f"q3_dev_{k}_40{x}") for k in ("math", "code", "chat")}
    tr = load_items(f"q3_train{x}")
    more = load_items(f"q3_train_more{x}")
    if more:
        tr = tr + more
    s["train"] = tr
    if smoke:
        s = {k: v[:2] for k, v in s.items()}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--long", action="store_true", help="use the answers continued to 1,024 tokens (*_long)")
    ap.add_argument("--pb_tokens", type=float, default=0,
                    help="if > 0: train split only, from PerfectBlend Qwen3-8B answers, this many million response tokens")
    args = ap.parse_args()
    out = os.path.join(DATA, ("cache_smoke" if args.smoke else "cache") + ("_long" if args.long else ""))
    if args.pb_tokens:
        out = os.path.join(DATA, "cache_pb" + ("_smoke" if args.smoke else ""))
    os.makedirs(out, exist_ok=True)
    t = load_target()
    W = t.lm_head.weight.detach()
    assert W.dtype == torch.bfloat16 and W.dim() == 2, (W.dtype, W.shape)
    torch.save(W.cpu(), os.path.join(out, "lm_head.pt"))
    D = W.shape[1]
    stats = {"layers": LAYERS, "splits": {}}
    if args.pb_tokens:
        from transformers import AutoTokenizer
        from dflash_eval import TARGET_ID
        from gister_train import load_convs
        allc = load_convs(AutoTokenizer.from_pretrained(TARGET_ID), 200 if args.smoke else 60000, own=False)
        seqs, tot = [], 0
        for c in allc:
            seqs.append(c)
            tot += len(c["resp"])
            if tot >= args.pb_tokens * 1e6:
                break
        todo = {"train": seqs}
    else:
        todo = splits(args.smoke, args.long)
    for name, seqs in todo.items():
        N = sum(len(it["resp"]) for it in seqs)
        mm = {l: np.lib.format.open_memmap(os.path.join(out, f"{name}_L{l}.npy"), "w+", np.float16, (N, D)) for l in LAYERS}
        meta = {k: np.zeros(N, dt) for k, dt in [("seq", np.int32), ("next", np.int32), ("top1", np.int32),
                                                  ("margin", np.float32), ("radius", np.float32), ("hnorm", np.float32)]}
        acc = {l: [torch.zeros(D, dtype=torch.float64, device="cuda"), torch.zeros(D, dtype=torch.float64, device="cuda"),
                   torch.zeros(D, device="cuda")] for l in LAYERS}
        o, checked = 0, False
        for si, it in enumerate(seqs):
            ids = torch.tensor([it["prompt"] + it["resp"]], device="cuda")
            n0, n = len(it["prompt"]), ids.shape[1]
            with torch.no_grad():
                r = t(ids, output_hidden_states=True)
                hs = r.hidden_states
                fin = hs[36][0]
                if not checked:   # the final row must be exactly what the LM head reads
                    a = (fin @ W.T).float()
                    err = (a - r.logits[0].float()).abs().max().item()
                    if err > 1e-2:
                        fin2 = t.model.norm(hs[36])[0]
                        err2 = ((fin2 @ W.T).float() - r.logits[0].float()).abs().max().item()
                        raise SystemExit(f"final hidden state is not head input (err {err}, after norm {err2})")
                    checked = True
                sl = slice(n0 - 1, n - 1)
                m = n - n0
                for l in LAYERS:
                    x = (fin if l == 36 else hs[l][0])[sl]
                    mm[l][o:o + m] = x.to(torch.float16).cpu().numpy()
                    if name == "train":
                        xf = x.double()
                        acc[l][0] += xf.sum(0); acc[l][1] += (xf * xf).sum(0)
                        acc[l][2] = torch.maximum(acc[l][2], x.float().abs().amax(0))
                lg = r.logits[0, sl].float()
                top = lg.topk(2, -1)
                dW = (W[top.indices[:, 0]] - W[top.indices[:, 1]]).float().norm(dim=-1)
                marg = top.values[:, 0] - top.values[:, 1]
                meta["seq"][o:o + m] = si
                meta["next"][o:o + m] = ids[0, n0:].cpu().numpy()
                meta["top1"][o:o + m] = top.indices[:, 0].cpu().numpy()
                meta["margin"][o:o + m] = marg.cpu().numpy()
                meta["radius"][o:o + m] = (marg / dW).cpu().numpy()
                meta["hnorm"][o:o + m] = fin[sl].float().norm(dim=-1).cpu().numpy()
            o += m
            if si % 200 == 0:
                print(f"  {name} {si}/{len(seqs)}", flush=True)
        for l in LAYERS:
            mm[l].flush()
        np.savez(os.path.join(out, f"{name}_meta.npz"), **meta)
        rel = meta["radius"] / meta["hnorm"]
        st = {"positions": N, "top1_eq_next": float((meta["top1"] == meta["next"]).mean()),
              "margin_quantiles": {q: float(np.quantile(meta["margin"], q)) for q in (0.01, 0.05, 0.1, 0.5)},
              "rel_flip_radius_quantiles": {q: float(np.quantile(rel, q)) for q in (0.01, 0.05, 0.1, 0.5)},
              "frac_margin_below_0.5": float((meta["margin"] < 0.5).mean())}
        if name == "train":
            st["layers"] = {}
            for l in LAYERS:
                mean = acc[l][0] / N
                std = (acc[l][1] / N - mean * mean).clamp_min(0).sqrt()
                mx = acc[l][2]
                torch.save({"mean": mean.float().cpu(), "std": std.float().cpu(), "maxabs": mx.cpu()},
                           os.path.join(out, f"stats_L{l}.pt"))
                med = mx.median()
                big = (mx > 20 * med).nonzero().flatten().tolist()
                st["layers"][l] = {"maxabs": float(mx.max()), "median_maxabs": float(med), "massive_dims": big,
                                   "median_std": float(std.median())}
        stats["splits"][name] = st
        print(name, json.dumps({k: v for k, v in st.items() if k != "layers"}), flush=True)
        if "layers" in st:
            print("layers", json.dumps(st["layers"]), flush=True)
    if not args.smoke and not args.pb_tokens:
        json.dump(stats, open(os.path.join(RES, "state_stats_long.json" if args.long else "state_stats.json"), "w"), indent=1)
    print("=== cache done", flush=True)


if __name__ == "__main__":
    main()
