"""Trained ceiling: can a frozen DFlash drafter use the target's true future, given a learned adapter?

kv_oracle.py appended the raw mean of the target's future features as one extra context position without
training, and acceptance dropped sharply: DFlash never saw such an input. Here a small adapter maps the mean of the
true future context features (next 16 or 64 positions, layers 1, 9, 17, 25, 33) to the extra context position:
    token = g * f + U(V f),   f = mean future features (5 x 4096 dims)
with g (a per-feature scale) and U both initialised to zero, so the extra position starts as all zeros, which
DFlash barely notices, and training starts from plain DFlash. Only the adapter is trained; target and drafter are
frozen. This measures how much knowing the future could help this drafter at best through its own context channel.

Run:  python kv_oracle_train.py train --horizon 16   (or 64)
      python kv_oracle_train.py eval
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import CKPT, DATA, RES
from dflash_eval import load_drafter, load_target
from dflash_plan import B, CTX_LAYERS, ORACLE_N
from kv_oracle import ctx_feats


class Adapter(nn.Module):
    def __init__(self, D, r=256):
        super().__init__()
        self.g = nn.Parameter(torch.zeros(D))
        self.V = nn.Linear(D, r, bias=False)
        self.U = nn.Linear(r, D, bias=False)
        nn.init.zeros_(self.U.weight)

    def forward(self, f):
        return self.g * f + self.U(self.V(f))


def logits_with_token(t, d, ids, ctx, s, tok):
    blk = torch.full((1, B), d.mask_token_id, dtype=torch.long, device="cuda")
    blk[0, 0] = ids[s]
    emb = t.model.embed_tokens(blk)
    if tok is None:
        th = ctx[None, :s]
        pos = torch.arange(0, s + B, device="cuda")[None]
    else:
        th = torch.cat([ctx[:s], tok[None].to(ctx.dtype)])[None]
        pos = torch.cat([torch.arange(0, s, device="cuda"), torch.tensor([s - 1], device="cuda"),
                         torch.arange(s, s + B, device="cuda")])[None]
    h = d(target_hidden=th, noise_embedding=emb, position_ids=pos, is_causal=False)
    return t.lm_head(h[:, 1:, :]).float()[0]


def future(ctx, s, n):
    return ctx[s + 1:s + 1 + n].float().mean(0)


def train(args):
    t, d = load_target(), load_drafter()
    for p in d.parameters():
        p.requires_grad_(False)
    D = ctx_dim = 4096 * len(CTX_LAYERS)
    ad = Adapter(D).cuda()
    opt = torch.optim.AdamW(ad.parameters(), lr=args.lr, weight_decay=0.0)
    items = [x for x in torch.load(os.path.join(DATA, "q3_train.pt")) if len(x["resp"]) > B + ORACLE_N]
    rng = random.Random(0)
    rng.shuffle(items)
    items = items[: args.n_train]
    total = len(items)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1, (i + 1) / 30) * 0.5 * (1 + math.cos(math.pi * min(1, i / total))))
    log, t0 = [], time.time()
    for step, it in enumerate(items, 1):
        ids = torch.tensor(it["prompt"] + it["resp"], device="cuda")
        n0 = len(it["prompt"])
        ctx = ctx_feats(t, ids)
        lo, hi = n0 + 1, len(ids) - B - ORACLE_N // 4
        starts = rng.sample(range(lo, hi), min(args.blocks, hi - lo)) if hi > lo else []
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for s in starts:
            lg = logits_with_token(t, d, ids, ctx, s, ad(future(ctx, s, args.horizon)))
            loss = F.cross_entropy(lg, ids[s + 1:s + B]) / len(starts)
            loss.backward()
            tot += loss.item()
        opt.step()
        sched.step()
        log.append(tot)
        if step % 25 == 0 or step == total:
            print(f"step {step}/{total}  loss {sum(log[-25:]) / len(log[-25:]):.4f}  |g| {ad.g.norm().item():.3f}  "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
    torch.save({"state": ad.state_dict(), "horizon": args.horizon, "loss": log}, os.path.join(CKPT, f"kv_oracle_{args.horizon}.pt"))
    print("saved", flush=True)


@torch.no_grad()
def evaluate(args):
    t, d = load_target(), load_drafter()
    D = 4096 * len(CTX_LAYERS)
    ads = {}
    for h in args.horizons:
        ck = torch.load(os.path.join(CKPT, f"kv_oracle_{h}.pt"))
        a = Adapter(D).cuda()
        a.load_state_dict(ck["state"])
        ads[h] = a
    res = {"args": vars(args), "sets": {}}
    for s_name in args.sets:
        seqs = torch.load(os.path.join(DATA, f"q3_dev_{s_name}_40.pt"))
        acc = {"none": [], **{f"fut{h}_trained": [] for h in ads}}
        for it in seqs:
            ids = torch.tensor(it["prompt"] + it["resp"], device="cuda")
            n0 = len(it["prompt"])
            if len(ids) < n0 + B + ORACLE_N // 4 + 2:
                continue
            ctx = ctx_feats(t, ids)
            for s in range(n0 + 1, len(ids) - B - ORACLE_N // 4, args.stride):
                lab = ids[s + 1:s + B]
                acc["none"].append(int((logits_with_token(t, d, ids, ctx, s, None).argmax(-1) == lab).int().cumprod(0).sum()) + 1)
                for h, a in ads.items():
                    pred = logits_with_token(t, d, ids, ctx, s, a(future(ctx, s, h))).argmax(-1)
                    acc[f"fut{h}_trained"].append(int((pred == lab).int().cumprod(0).sum()) + 1)
        res["sets"][s_name] = {k: {"blocks": len(v), "mean_accepted": sum(v) / len(v)} for k, v in acc.items()}
        res["sets"][s_name]["per_block"] = acc
        print(s_name, {k: round(sum(v) / len(v), 3) for k, v in acc.items()}, flush=True)
        json.dump(res, open(os.path.join(RES, args.out + ".json"), "w"), indent=1)
    print("wrote", os.path.join(RES, args.out + ".json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--horizons", type=int, nargs="+", default=[16, 64])
    ap.add_argument("--n_train", type=int, default=300)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sets", nargs="+", default=["math", "code", "chat"])
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--out", default="kv_oracle_trained")
    a = ap.parse_args()
    train(a) if a.mode == "train" else evaluate(a)
