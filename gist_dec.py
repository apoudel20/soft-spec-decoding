"""Gist-conditioned block decoder, trained from scratch: does a gist of the next 64 positions help drafting?

At anchor row i (the last verified position) the decoder reads
  context : the target's final state and PCA'd planning states (layers 18, 24, 30) at rows i-7..i
  token   : the token just accepted (row i+1's input), embedded with the frozen output-layer row
  gist    : 8 block latents of rows i+1..i+64 from the gist VAE (arm "gist"), or 8 learned constants (arm "none")
and drafts 15 tokens at once through the frozen Qwen3-8B output layer, trained with KL to the target's own
distributions at rows i+1..i+15. Both arms have the same architecture, data and steps.

Score on held-out dev (teacher-forced, non-overlapping anchors): accepted per round = leading drafted tokens equal
to the target's own top-1, plus the bonus token (out of 16), the same unit as the DFlash numbers.
The gist arm is also scored with gists shuffled across anchors (does the gain come from the actual future?).

Gate, fixed before running: proceed to a gist predictor if the true gist raises accepted per round by >= 10% over
the no-gist arm on at least two of math, code, chat. Close the gist route if the gain is < 3% on all three.
This gist comes from the true future, so it is an upper bound on what a predicted gist could give.

Run:    python gist_dec.py --arm gist|none [--smoke]
Output: results/gist_dec_<arm>.json, results/gist_gate.json (once both arms exist), ckpt/gist_dec_<arm>.pt
"""
import argparse
import json
import os
import queue
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import CKPT, DATA, RES

C, NQ, NB = 8, 15, 8


def tblock(d, n, heads):
    l = nn.TransformerEncoderLayer(d, heads, 4 * d, 0.0, batch_first=True, norm_first=True, activation="gelu")
    return nn.TransformerEncoder(l, n, enable_nested_tensor=False)


class GistDecoder(nn.Module):
    def __init__(self, D, K, zq, arm, d=768, depth=6, heads=12):
        super().__init__()
        self.arm = arm
        self.ctx = nn.Linear(D + 3 * K, d)
        self.ctx_e = nn.Parameter(torch.randn(C, d) * 0.02)
        self.tok = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, d))
        self.gist = nn.Linear(zq, d)
        self.gist_e = nn.Parameter(torch.randn(NB, d) * 0.02)
        self.gist_const = nn.Parameter(torch.randn(NB, d) * 0.02)
        self.q = nn.Parameter(torch.randn(NQ, d) * 0.02)
        self.body = tblock(d, depth, heads)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, D))

    def forward(self, ctx, tok, gist):
        B = ctx.shape[0]
        g = self.gist(gist) + self.gist_e if self.arm == "gist" else self.gist_const.expand(B, -1, -1)
        h = torch.cat([self.ctx(ctx) + self.ctx_e, self.tok(tok)[:, None], g, self.q.expand(B, -1, -1)], 1)
        return self.out(self.body(h)[:, -NQ:])                 # (B, NQ, D) in the output layer's input space


class Data:
    def __init__(self, cache, split, sfx, stats36, W):
        m = np.load(os.path.join(cache, f"{split}_meta.npz"))
        self.seq, self.next, self.top1 = m["seq"], m["next"], m["top1"]
        self.fin = np.load(os.path.join(cache, f"{split}_L36.npy"), mmap_mode="r")
        self.pca = np.load(os.path.join(cache, f"{split}_pca{sfx}.npy"))
        self.G = np.load(os.path.join(cache, f"{split}_gist{sfx}.npy"))
        ok = np.load(os.path.join(cache, f"{split}_gist_ok{sfx}.npy"))
        n = len(self.seq)
        back = np.zeros(n, bool)
        back[C - 1:] = self.seq[C - 1:] == self.seq[:n - C + 1]
        self.anchors = np.nonzero(ok & back)[0]
        self.mean, self.std = stats36
        self.W = W

    def get(self, an, shuffle_from=None):
        rows = an[:, None] + np.arange(-(C - 1), NQ + 1)[None]          # i-7 .. i+15 : 23 rows
        flat = rows.reshape(-1)
        srt = np.argsort(flat)
        a = np.empty((len(flat), self.fin.shape[1]), np.float16)
        a[srt] = self.fin[flat[srt]]
        fin = torch.from_numpy(a.astype(np.float32)).view(len(an), -1, self.fin.shape[1])
        ctx_f = (fin[:, :C] - self.mean) / self.std
        mid = torch.from_numpy(self.pca[an[:, None] + np.arange(-(C - 1), 1)[None]].astype(np.float32)).flatten(2)
        ctx = torch.cat([ctx_f, mid], -1)
        tok = self.W[torch.from_numpy(self.next[an].astype(np.int64))].float()
        g_src = an if shuffle_from is None else shuffle_from
        gist = torch.from_numpy(self.G[g_src].astype(np.float32))
        tgt = fin[:, C:C + NQ]                                         # rows i+1..i+15, final normed states
        lab = torch.from_numpy(self.top1[an[:, None] + np.arange(1, NQ + 1)[None]].astype(np.int64))
        return ctx, tok, gist, tgt, lab


def prefetch(data, B, steps, rng):
    q = queue.Queue(maxsize=6)

    def work():
        for _ in range(steps):
            q.put(data.get(rng.choice(data.anchors, B)))
        q.put(None)
    threading.Thread(target=work, daemon=True).start()
    while True:
        b = q.get()
        if b is None:
            return
        yield b


def accepted(pred, lab):
    eq = (pred == lab).int()
    return eq.cumprod(1).sum(1) + 1                                   # leading matches + bonus token


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["gist", "none"])
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = "cpu" if args.smoke else "cuda"
    sfx = "_smoke" if args.smoke else ""
    cache = os.path.join(DATA, "cache")
    if args.smoke:
        args.steps, args.B = 6, 2
    W = torch.load(os.path.join(cache, "lm_head.pt")).to(dev)
    s36 = torch.load(os.path.join(cache, "stats_L36.pt"))
    stats36 = (s36["mean"], s36["std"].clamp_min(1e-3 * s36["std"].median().item()))
    tr = Data(cache, "dev_chat" if args.smoke else "train", sfx, stats36, W.cpu())
    D, K, zq = W.shape[1], tr.pca.shape[2], tr.G.shape[2]
    model = GistDecoder(D, K, zq, args.arm, **({"d": 64, "depth": 1, "heads": 4} if args.smoke else {})).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 500) * max(0.1, 0.5 * (1 + np.cos(np.pi * min(1, s / args.steps)))))
    print(f"arm {args.arm}  params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M  train anchors {len(tr.anchors)}", flush=True)

    def logp(h):
        return (h.to(torch.bfloat16) @ W.T).float().log_softmax(-1)

    t0, log = time.time(), []
    model.train()
    for step, (ctx, tok, gist, tgt, lab) in enumerate(prefetch(tr, args.B, args.steps, np.random.default_rng(0))):
        ctx, tok, gist, tgt, lab = (x.to(dev) for x in (ctx, tok, gist, tgt, lab))
        with torch.autocast(dev, torch.bfloat16, enabled=dev == "cuda"):
            h = model(ctx, tok, gist).float()
        lq = logp(h.flatten(0, 1))
        with torch.no_grad():
            lp = logp(tgt.flatten(0, 1))
        kl = (lp.exp() * (lp - lq)).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        kl.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.steps - 1:
            with torch.no_grad():
                acc = accepted(lq.argmax(-1).view(-1, NQ), lab).float().mean().item()
            r = {"step": step, "kl": kl.item(), "accepted_train": acc, "min": (time.time() - t0) / 60}
            log.append(r)
            print(json.dumps(r), flush=True)
            if not np.isfinite(kl.item()):
                raise SystemExit("non-finite loss")
    model.eval()
    res = {"args": vars(args), "train_log": log, "sets": {}}
    rng = np.random.default_rng(1)
    for sp in ("dev_math", "dev_code", "dev_chat"):
        dv = Data(cache, sp, sfx, stats36, W.cpu())
        keep, last = [], -10 ** 9
        for a in dv.anchors:
            if a >= last + NQ + 1:
                keep.append(a)
                last = a
        an = np.array(keep)
        out = {"true": [], "shuffled": []}
        top1 = []
        with torch.no_grad():
            perm = rng.permutation(an)
            for j in range(0, len(an), 64):
                b = an[j:j + 64]
                for kind in (["true", "shuffled"] if args.arm == "gist" else ["true"]):
                    ctx, tok, gist, tgt, lab = dv.get(b, None if kind == "true" else perm[j:j + 64])
                    with torch.autocast(dev, torch.bfloat16, enabled=dev == "cuda"):
                        h = model(ctx.to(dev), tok.to(dev), gist.to(dev)).float()
                    pred = logp(h.flatten(0, 1)).argmax(-1).view(-1, NQ).cpu()
                    out[kind].append(accepted(pred, lab))
                    if kind == "true":
                        top1.append((pred == lab).float().mean(0))
        r = {"anchors": len(an), "accepted": torch.cat(out["true"]).float().mean().item(),
             "slot_top1": torch.stack(top1).mean(0).tolist()}
        if out["shuffled"]:
            r["accepted_shuffled_gist"] = torch.cat(out["shuffled"]).float().mean().item()
        res["sets"][sp] = r
        print(sp, json.dumps({k: v for k, v in r.items() if k != "slot_top1"}), flush=True)
    os.makedirs(CKPT, exist_ok=True)
    if not args.smoke:
        torch.save({"model": model.state_dict(), "args": vars(args)}, os.path.join(CKPT, f"gist_dec_{args.arm}.pt"))
    out_p = os.path.join(RES, f"gist_dec_{args.arm}{sfx}.json")
    json.dump(res, open(out_p, "w"), indent=1)
    other = os.path.join(RES, f"gist_dec_{'none' if args.arm == 'gist' else 'gist'}{sfx}.json")
    if os.path.exists(other):
        g = json.load(open(out_p if args.arm == "gist" else other))
        n = json.load(open(other if args.arm == "gist" else out_p))
        gain = {sp: g["sets"][sp]["accepted"] / n["sets"][sp]["accepted"] - 1 for sp in g["sets"]}
        verdict = ("proceed to a gist predictor" if sum(v >= 0.10 for v in gain.values()) >= 2
                   else "close the gist route" if all(v < 0.03 for v in gain.values()) else "in between: report as is")
        gate = {"gain_true_gist_over_none": gain, "verdict": verdict,
                "accepted_gist": {sp: g["sets"][sp]["accepted"] for sp in g["sets"]},
                "accepted_none": {sp: n["sets"][sp]["accepted"] for sp in n["sets"]},
                "accepted_shuffled_gist": {sp: g["sets"][sp].get("accepted_shuffled_gist") for sp in g["sets"]}}
        json.dump(gate, open(os.path.join(RES, f"gist_gate{sfx}.json"), "w"), indent=1)
        print("GIST GATE", json.dumps(gate), flush=True)
    print(f"=== gist_dec {args.arm} done", flush=True)


if __name__ == "__main__":
    main()
