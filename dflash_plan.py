"""Sketch A on DFlash: does a plan vector from the target's tapped layers raise block acceptance?

Setup. Target Qwen3-8B (4-bit), drafter z-lab/Qwen3-8B-DFlash-b16, frozen. DFlash already conditions on the
target's hidden states from layers 1, 9, 17, 25, 33 at every context position. We add one plan vector to the
embeddings of all 16 slots of the block. Only the plan head is trained.

Plan sources (--plan):
  none   : no plan vector (plain DFlash; nothing to train).
  mid    : the target's hidden states at layers 18 and 20 at the last verified position (where the layer sweep
           found the most about the length and structure of the rest of the answer).
  oracle : the mean of the target's layer-20 states over the true next 64 positions. Not available at inference;
           it is the ceiling on what a plan of this kind could add.
Plan head: LayerNorm, then a zero-initialised linear map to the drafter width, with the output norm capped at the
median token-embedding norm (so training starts exactly at plain DFlash and the plan can never drown the inputs).

Data. The 8B's own greedy answers (data/q3_train.pt from gen_q3.py) for training, held-out dev answers
(data/q3_dev_<set>_40.pt) for evaluation. A block starts at response position s: slot 0 holds token s, slots 1..15
are masks and are trained (cross-entropy) to predict tokens s+1..s+15, given the target's hidden states for
positions < s, exactly as in DFlash's decoding loop. Evaluation reports mean accepted length per block
(accepted drafts + 1), which equals tau under greedy chain verification.

Run:  python dflash_plan.py train --plan mid      (also --plan oracle)
      python dflash_plan.py eval --plans none mid oracle
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

CTX_LAYERS = [1, 9, 17, 25, 33]
PLAN_LAYERS = [18, 20]
ORACLE_LAYER, ORACLE_N = 20, 64
B = 16


class PlanHead(nn.Module):
    """Direction from a learned projection, strength from a learned scalar that starts at 5% of a token-embedding
    norm. (v1 used a zero-initialised projection with a norm cap: Adam moves every weight by about the learning rate,
    so the output hit the cap within a few steps and the drafter got worse. A scalar strength avoids that, and
    starting it above zero avoids the deadlock of a zero gate.)"""

    def __init__(self, din, dout, enorm, alpha0=0.05):
        super().__init__()
        self.ln = nn.LayerNorm(din, elementwise_affine=False)
        self.proj = nn.Linear(din, dout)
        self.alpha = nn.Parameter(torch.tensor(alpha0))
        self.enorm = enorm

    def forward(self, z):
        v = F.normalize(self.proj(self.ln(z.float())), dim=-1)
        return self.alpha.clamp(-1, 1) * self.enorm * v


@torch.no_grad()
def target_states(t, ids):
    hs = t(input_ids=ids[None], output_hidden_states=True).hidden_states
    ctx = torch.cat([hs[l + 1][0] for l in CTX_LAYERS], dim=-1)          # (L, 5*4096)
    mid = torch.cat([hs[l + 1][0] for l in PLAN_LAYERS], dim=-1).float()  # (L, 2*4096)
    orc = hs[ORACLE_LAYER + 1][0].float()                                 # (L, 4096)
    return ctx, mid, orc


def plan_input(kind, s, mid, orc):
    if kind == "mid":
        return mid[s - 1]
    if kind == "oracle":
        return orc[s + 1:s + 1 + ORACLE_N].mean(0)
    return None


def block_logits(t, d, ids, ctx, s, plan_vec):
    blk = torch.full((1, B), d.mask_token_id, dtype=torch.long, device="cuda")
    blk[0, 0] = ids[s]
    emb = t.model.embed_tokens(blk)
    if plan_vec is not None:   # slot 0 holds the real last token and is left untouched
        emb = torch.cat([emb[:, :1], emb[:, 1:] + plan_vec.to(emb.dtype)[None, None]], dim=1)
    pos = torch.arange(0, s + B, device="cuda")[None]
    h = d(target_hidden=ctx[None, :s], noise_embedding=emb, position_ids=pos, is_causal=False)
    return t.lm_head(h[:, 1:, :]).float()[0]   # (15, V): slot i predicts token s+i


def starts_for(n_prompt, L, k, rng):
    lo, hi = n_prompt + 1, L - B - ORACLE_N // 4
    if hi <= lo:
        return []
    return rng.sample(range(lo, hi), min(k, hi - lo))


def train(args):
    t, d = load_target(), load_drafter()
    for p in d.parameters():
        p.requires_grad_(False)
    enorm = t.model.embed_tokens.weight.float().norm(dim=1).median().item()
    din = {"mid": 4096 * len(PLAN_LAYERS), "oracle": 4096}[args.plan]
    head = PlanHead(din, d.config.hidden_size, enorm).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.0)
    items = [x for x in torch.load(os.path.join(DATA, "q3_train.pt")) if len(x["resp"]) > B + ORACLE_N]
    rng = random.Random(0)
    rng.shuffle(items)
    items = items[: args.n_train]
    total = len(items) * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1, (i + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(1, i / total))))
    log, t0, step = [], time.time(), 0
    for ep in range(args.epochs):
        for it in items:
            ids = torch.tensor(it["prompt"] + it["resp"], device="cuda")
            ctx, mid, orc = target_states(t, ids)
            loss_sum, nb = 0.0, 0
            opt.zero_grad(set_to_none=True)
            for s in starts_for(len(it["prompt"]), len(ids), args.blocks, rng):
                lg = block_logits(t, d, ids, ctx, s, head(plan_input(args.plan, s, mid, orc)))
                loss = F.cross_entropy(lg, ids[s + 1:s + B]) / args.blocks
                loss.backward()
                loss_sum += loss.item() * args.blocks
                nb += 1
            opt.step()
            sched.step()
            step += 1
            log.append(loss_sum / max(nb, 1))
            if step % 25 == 0 or step == total:
                v = torch.linalg.vector_norm(head.proj.weight).item()
                print(f"step {step}/{total}  loss {sum(log[-25:]) / len(log[-25:]):.4f}  |W| {v:.3f}  "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
    torch.save({"plan": args.plan, "state": head.state_dict(), "din": din, "enorm": enorm, "loss": log},
               os.path.join(CKPT, f"dflash_plan_{args.plan}.pt"))
    print("saved", flush=True)


@torch.no_grad()
def evaluate(args):
    t, d = load_target(), load_drafter()
    heads = {}
    for p in args.plans:
        if p == "none":
            heads[p] = None
            continue
        ck = torch.load(os.path.join(CKPT, f"dflash_plan_{p}.pt"))
        h = PlanHead(ck["din"], d.config.hidden_size, ck["enorm"]).cuda()
        h.load_state_dict(ck["state"])
        heads[p] = h
    res = {"args": vars(args), "sets": {}}
    for s_name in args.sets:
        seqs = torch.load(os.path.join(DATA, f"q3_dev_{s_name}_40.pt"))
        acc = {p: [] for p in args.plans}
        for it in seqs:
            ids = torch.tensor(it["prompt"] + it["resp"], device="cuda")
            if len(ids) < len(it["prompt"]) + B + ORACLE_N // 4 + 2:
                continue
            ctx, mid, orc = target_states(t, ids)
            for s in range(len(it["prompt"]) + 1, len(ids) - B - ORACLE_N // 4, args.stride):
                lab = ids[s + 1:s + B]
                for p, h in heads.items():
                    pv = None if h is None else h(plan_input(p, s, mid, orc))
                    pred = block_logits(t, d, ids, ctx, s, pv).argmax(-1)
                    acc[p].append(int((pred == lab).int().cumprod(0).sum()) + 1)
        res["sets"][s_name] = {p: {"blocks": len(v), "mean_accepted": sum(v) / len(v)} for p, v in acc.items()}
        print(s_name, {p: round(r["mean_accepted"], 3) for p, r in res["sets"][s_name].items()}, flush=True)
        json.dump(res, open(os.path.join(RES, args.out + ".json"), "w"), indent=1)
    print("wrote", os.path.join(RES, args.out + ".json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("--plan", choices=["mid", "oracle"], default="mid")
    ap.add_argument("--plans", nargs="+", default=["none", "mid", "oracle"])
    ap.add_argument("--n_train", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sets", nargs="+", default=["math", "code", "chat"])
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--out", default="dflash_plan")
    a = ap.parse_args()
    train(a) if a.mode == "train" else evaluate(a)
