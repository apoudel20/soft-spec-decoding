"""Go/no-go for a plan-conditioned drafter: does DFlash gain from the target's true future when the drafter itself
can learn to use it?

A frozen DFlash ignored the true future both through its input embeddings (dflash_plan.py) and through a trained
adapter on an extra context position (kv_oracle_train.py). Here the drafter is fine-tuned with LoRA (rank 16 on every
linear layer of its 5 blocks) in two otherwise identical runs:
  control : no extra input. Measures what fine-tuning on the 8B's own answers does by itself.
  fut16   : plus one extra context position built by an adapter from the mean of the target's true next-16
            context features (the block being drafted). An upper bound on what any plan could add.
If fut16 clearly beats control, a drafter trained to use a plan has headroom and a plan-conditioned drafter is worth
building. If not, future plans do not help a strong block drafter at this scale.

Run:  python kv_lora.py train --variant control   (and --variant fut16)
      python kv_lora.py eval
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
from kv_oracle_train import Adapter, future, logits_with_token


class LoRALinear(nn.Module):
    def __init__(self, base, r=16, alpha=32):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (F.linear(F.linear(x.float(), self.A), self.B) * self.scale).to(x.dtype)


def add_lora(d, r=16):
    for layer in d.layers:
        for mod in (layer.self_attn, layer.mlp):
            for name, child in list(mod.named_children()):
                if isinstance(child, nn.Linear):
                    setattr(mod, name, LoRALinear(child, r))
    return [p for n, p in d.named_parameters() if n.endswith(".A") or n.endswith(".B")]


def lora_state(d):
    return {n: p.detach().cpu() for n, p in d.named_parameters() if n.endswith(".A") or n.endswith(".B")}


def train(args):
    t, d = load_target(), load_drafter()
    for p in d.parameters():
        p.requires_grad_(False)
    params = add_lora(d)
    for p in params:
        p.requires_grad_(True)
    ad = None
    if args.variant == "fut16":
        ad = Adapter(4096 * len(CTX_LAYERS)).cuda()
        params += list(ad.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    items = [x for x in torch.load(os.path.join(DATA, "q3_train.pt")) if len(x["resp"]) > B + ORACLE_N]
    rng = random.Random(0)
    rng.shuffle(items)
    items = items[: args.n_train]
    total = len(items)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1, (i + 1) / 30) * 0.5 * (1 + math.cos(math.pi * min(1, i / total))))
    d.train()
    log, t0 = [], time.time()
    for step, it in enumerate(items, 1):
        ids = torch.tensor(it["prompt"] + it["resp"], device="cuda")
        n0 = len(it["prompt"])
        ctx = ctx_feats(t, ids)
        lo, hi = n0 + 1, len(ids) - B - ORACLE_N // 4
        starts = random.Random(step).sample(range(lo, hi), min(args.blocks, hi - lo)) if hi > lo else []
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for s in starts:
            tok = None if ad is None else ad(future(ctx, s, 16))
            lg = logits_with_token(t, d, ids, ctx, s, tok)
            loss = F.cross_entropy(lg, ids[s + 1:s + B]) / len(starts)
            loss.backward()
            tot += loss.item()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        log.append(tot)
        if step % 25 == 0 or step == total:
            print(f"step {step}/{total}  loss {sum(log[-25:]) / len(log[-25:]):.4f}  {(time.time() - t0) / 60:.1f} min", flush=True)
    torch.save({"lora": lora_state(d), "adapter": None if ad is None else ad.state_dict(), "loss": log},
               os.path.join(CKPT, f"kv_lora_{args.variant}.pt"))
    print("saved", flush=True)


@torch.no_grad()
def evaluate(args):
    t = load_target()
    res = {"args": vars(args), "sets": {}}
    variants = {}
    for v in ["control", "fut16"]:
        d = load_drafter()
        add_lora(d)
        ck = torch.load(os.path.join(CKPT, f"kv_lora_{v}.pt"))
        d.load_state_dict({k: x.cuda() for k, x in ck["lora"].items()}, strict=False)
        d.eval()
        ad = None
        if ck["adapter"] is not None:
            ad = Adapter(4096 * len(CTX_LAYERS)).cuda()
            ad.load_state_dict(ck["adapter"])
        variants[v] = (d, ad)
    base = load_drafter()
    for s_name in args.sets:
        seqs = torch.load(os.path.join(DATA, f"q3_dev_{s_name}_40.pt"))
        acc = {"none": [], "control": [], "fut16": []}
        for it in seqs:
            ids = torch.tensor(it["prompt"] + it["resp"], device="cuda")
            n0 = len(it["prompt"])
            if len(ids) < n0 + B + ORACLE_N // 4 + 2:
                continue
            ctx = ctx_feats(t, ids)
            for s in range(n0 + 1, len(ids) - B - ORACLE_N // 4, args.stride):
                lab = ids[s + 1:s + B]
                acc["none"].append(int((logits_with_token(t, base, ids, ctx, s, None).argmax(-1) == lab).int().cumprod(0).sum()) + 1)
                for v, (d, ad) in variants.items():
                    tok = None if ad is None else ad(future(ctx, s, 16))
                    pred = logits_with_token(t, d, ids, ctx, s, tok).argmax(-1)
                    acc[v].append(int((pred == lab).int().cumprod(0).sum()) + 1)
        res["sets"][s_name] = {k: {"blocks": len(v), "mean_accepted": sum(v) / len(v)} for k, v in acc.items()}
        res["sets"][s_name]["per_block"] = acc
        print(s_name, {k: round(sum(v) / len(v), 3) for k, v in acc.items()}, flush=True)
        json.dump(res, open(os.path.join(RES, args.out + ".json"), "w"), indent=1)
    print("wrote", os.path.join(RES, args.out + ".json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("--variant", choices=["control", "fut16"], default="control")
    ap.add_argument("--n_train", type=int, default=300)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--sets", nargs="+", default=["math", "code", "chat"])
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--out", default="kv_lora")
    a = ap.parse_args()
    train(a) if a.mode == "train" else evaluate(a)
