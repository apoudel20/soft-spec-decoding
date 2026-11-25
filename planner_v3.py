"""Latent planner experiment: does a target-side summary of the future help the block drafter?

Same block drafter recipe as the best Arm D drafter (Qwen2.5-0.5B, gated LoRA, B=16 training
blocks, two passes with soft self-conditioning, TV loss to the target's distribution). A vector z
is projected into the drafter's embedding space and added to every block slot (both passes). The
projection is zero-initialised, so training starts from the plain drafter.

--plan feat         z = the target's last-layer hidden state at t-1. In deployment this is the
                    newest target state available when the block starting at x_t is drafted.
                    The DFlash/EAGLE-style "send the features" baseline.
--plan pred         z = PlanHead(h_{t-1}), trained jointly with an auxiliary loss to regress the
                    mean target hidden state over t+1..t+N (a future summary, FSP-like), plus
                    the drafter loss. Your latent planner, with an explicit plan objective.
--plan oracle_near  z = the true mean target hidden state over t+1..t+N. Ceiling for a perfect
                    plan (it contains the block being drafted).
--plan oracle_far   z = the true mean over t+B+1..t+B+N, skipping the block. Tests whether
                    knowing the further future helps the near tokens at all.

All variants stay lossless in deployment: z only steers the drafter, and every token is still
verified against the unchanged target. This script measures offline accepted length per block
(greedy, as in arm0.py), not wall-clock.

Run: python planner.py train --plan pred --name plan_pred
     python planner.py eval --ckpts d4_soft plan_feat plan_pred plan_oracle_near plan_oracle_far
"""
import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import CKPT, DATA, RES
from model import DRAFTER, TARGET, QwenLM
from train import batch_iter, gather_rows, loss_rows, pack_mask, sample_starts, topk_probs


class PlanMods(nn.Module):
    """Maps a plan vector z to an input offset for the drafter's block slots.

    v3: zero-initialised projection with a norm cap at the drafter's median token-embedding norm,
    e_plan = enorm * v / max(1, |v|) with v = proj(LN(z)). Training starts exactly at the plain drafter,
    the projection gets gradient from step 0, and the offset can never exceed a token embedding.
    v1 (uncapped) grew to 4-10x the embedding norm and drowned the inputs. v2 (unit direction behind a
    tanh gate initialised at 0) deadlocked: at gate 0 the projection gets no gradient, so it stayed
    random and the gate never opened, even for the oracle."""

    def __init__(self, dt, dd, pred, dev):
        super().__init__()
        self.proj = nn.Linear(dt, dd).to(dev)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.head = nn.Sequential(nn.Linear(dt, dt), nn.GELU(), nn.Linear(dt, dt)).to(dev) if pred else None

    def forward(self, z, enorm):
        v = self.proj(nrm(z))
        return enorm * v / v.norm(dim=-1, keepdim=True).clamp_min(1.0)


def emb_norm(m):
    return m.E.float().norm(dim=1).median().item()


def nrm(x):
    return F.layer_norm(x.float(), x.shape[-1:])


def plan_vectors(kind, hn_t, starts, B, N):
    """hn_t (Nb, L, dt) target final hidden; starts (Nb, nb). Returns raw feature, near and far summaries."""
    h = hn_t.float()
    Nb, L, dt = h.shape
    C = torch.cat([h.new_zeros(Nb, 1, dt), h.cumsum(1)], 1)  # C[:, i] = sum of h[:, :i]

    def mean_range(a, b):
        a, b = a.clamp(0, L), b.clamp(0, L)
        s = gather_rows(C, b) - gather_rows(C, a)
        return s / (b - a).clamp_min(1)[..., None]

    feat = gather_rows(h, (starts - 1).clamp_min(0))
    near = mean_range(starts + 1, starts + 1 + N)
    far = mean_range(starts + B + 1, starts + B + 1 + N)
    return feat, near, far


def z_for(kind, pm, feat, near, far):
    """Returns (z, aux_loss) for a plan kind."""
    if kind == "feat":
        return feat, None
    if kind == "pred":
        zp = pm.head(nrm(feat))
        return zp, F.mse_loss(nrm(zp), nrm(near).detach())
    if kind == "oracle_near":
        return near, None
    if kind == "oracle_far":
        return far, None
    raise ValueError(kind)


def train(args):
    items = [it for it in torch.load(os.path.join(DATA, "train.pt")) if len(it["resp"]) >= 4]
    d = QwenLM(DRAFTER)
    d.use_graphs = False
    d.add_lora(32, 64)
    t = QwenLM(TARGET)
    t.use_graphs = False
    d.plan = PlanMods(t.d, d.d, args.plan == "pred", d.dev)
    params = [p for p in d.parameters() if p.requires_grad]
    enorm = emb_norm(d)
    B, nb, N = 16, 8, 8
    rng = random.Random(0)
    torch.manual_seed(0)
    opt = torch.optim.AdamW(params, lr=2e-4, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(1, s / args.steps))))
    it = batch_iter(items, N, 0)
    M = nb * B
    log, t0 = [], time.time()
    for step in range(args.steps):
        if args.stop and step > args.stop:
            break
        ids, lens, rs = next(it)
        with torch.no_grad():
            _, kvs = d.full(ids, collect_kv=True)
            hn_t = t.full(ids)[0]
        starts = sample_starts(lens, rs, nb, 0, rng)
        pos = (starts[:, :, None] + torch.arange(B, device="cuda")).view(N, M)
        mask = pack_mask(starts, B, ids.shape[1])
        valid = (pos + 1 <= (lens.cuda()[:, None] - 1)).float().reshape(-1)
        lab_h = gather_rows(hn_t, pos).reshape(-1, t.d)
        lab = lambda idx: t.logits(lab_h[idx])
        feat, near, far = plan_vectors(args.plan, hn_t, starts, B, args.N)
        z, aux = z_for(args.plan, d.plan, feat, near, far)
        e_plan = d.plan(z, enorm).repeat_interleave(B, 1)  # (N, M, dd)
        x0 = d.rows32(gather_rows(ids, starts))
        first = (torch.arange(M, device="cuda") % B == 0)[None, :, None]
        e1 = torch.where(first, x0.repeat_interleave(B, 1), d.e_mask.expand(N, M, -1)) + e_plan
        h1 = d.packed(e1, pos, kvs, mask, ckpt=True)
        loss1 = loss_rows(d, h1.reshape(-1, d.d), lab, "tv", valid=valid)
        with torch.no_grad():
            lg1 = torch.stack([d.logits(h1[n].detach()) for n in range(N)])
            gi, gp = topk_probs(lg1)
            g = d.embed_soft(gi, gp)
            del lg1
        e2 = torch.where(first, x0.repeat_interleave(B, 1), torch.roll(g, 1, dims=1) + d.e_flag) + e_plan
        h2 = d.packed(e2, pos, kvs, mask, ckpt=True)
        loss2 = loss_rows(d, h2.reshape(-1, d.d), lab, "tv", valid=valid)
        loss = loss1 + loss2 + (args.aux * aux if aux is not None else 0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        log.append((loss1.item(), loss2.item(), aux.item() if aux is not None else 0.0))
        if step % 50 == 0 or step == args.steps - 1:
            el = time.time() - t0
            a = [sum(x[i] for x in log[-50:]) / len(log[-50:]) for i in range(3)]
            print(f"step {step} loss1 {a[0]:.4f} loss2 {a[1]:.4f} aux {a[2]:.4f} plan/emb {e_plan.detach().norm(dim=-1).mean().item() / enorm:.3f}  {el / 60:.1f} min, eta {el / (step + 1) * (args.steps - step - 1) / 60:.1f} min", flush=True)
    sd = d.trainable_state()
    sd["__meta__"] = {"r": 32, "alpha": 64, "inp": "soft", "label": "d4", "plan": args.plan, "N": args.N, "steps": args.steps}
    torch.save(sd, os.path.join(CKPT, args.name + ".pt"))
    json.dump({"args": vars(args), "loss": log}, open(os.path.join(CKPT, args.name + "_log.json"), "w"))
    print("saved", args.name)


@torch.no_grad()
def evaluate(args):
    from arm0 import summarize
    t = QwenLM(TARGET)
    t.use_graphs = False
    seqs = {}
    for s in args.sets:
        name, _, n = s.partition(":")
        seqs[name] = torch.load(os.path.join(DATA, f"ref_{name}_{n}.pt"))
    tgt_h = {k: [t.full(torch.tensor(s["prompt"] + s["resp"], device="cuda")[None])[0][0].float() for s in v] for k, v in seqs.items()}
    dt = t.d
    del t
    torch.cuda.empty_cache()
    B = args.B
    res = {"args": vars(args), "results": {}}
    for ck in args.ckpts:
        d = QwenLM(DRAFTER)
        d.use_graphs = False
        sd = torch.load(os.path.join(CKPT, ck + ".pt"), map_location="cpu")
        plan = sd.get("__meta__", {}).get("plan", "none")
        N = sd.get("__meta__", {}).get("N", 64)
        if plan != "none":
            d.add_lora(32, 64)
            d.plan = PlanMods(dt, d.d, plan == "pred", d.dev)
        d.load_trainable(sd)
        for name, ss in seqs.items():
            hits, Ls = [], []
            for si, s in enumerate(ss):
                ids = torch.tensor(s["prompt"] + s["resp"], device="cuda")
                r = len(s["prompt"])
                if len(ids) - B - 1 <= r:
                    continue
                starts = torch.arange(r, len(ids) - B - 1, args.stride, device="cuda")
                _, kvs = d.full(ids[None], collect_kv=True)
                nb = len(starts)
                st = starts[None]
                pos = (st[:, :, None] + torch.arange(B, device="cuda")).view(1, nb * B)
                mask = pack_mask(st, B, len(ids))
                tgt = ids[(pos + 1).clamp_max(len(ids) - 1)][0].view(nb, B)
                first = (torch.arange(nb * B, device="cuda") % B == 0)[None, :, None]
                x_in = d.rows32(ids[pos[0]])[None]
                if plan != "none":
                    feat, near, far = plan_vectors(plan, tgt_h[name][si][None], st, B, N)
                    z, _ = z_for(plan, d.plan, feat, near, far)
                    e_plan = d.plan(z, emb_norm(d)).repeat_interleave(B, 1)
                else:
                    e_plan = 0.0
                e_tf = torch.where(first, x_in, x_in + d.e_flag) + e_plan
                p_tf = d.logits(d.packed(e_tf, pos, kvs, mask, ckpt=False)[0]).argmax(-1).view(nb, B)
                e = torch.where(first, x_in, d.e_mask.expand_as(x_in)) + e_plan
                lg = d.logits(d.packed(e, pos, kvs, mask, ckpt=False)[0])
                gi, gp = topk_probs(lg)
                e = torch.where(first, x_in, torch.roll(d.embed_soft(gi, gp), 1, dims=0)[None] + d.e_flag) + e_plan
                p_fr = d.logits(d.packed(e, pos, kvs, mask, ckpt=False)[0]).argmax(-1).view(nb, B)
                hits.append(p_tf == tgt)
                Ls.append((p_fr == tgt).int().cumprod(1).sum(1))
            r_ = summarize(torch.cat(hits), torch.cat(Ls).cpu(), B)
            res["results"][f"{name}/{ck}"] = r_
            print(f"{name:10s} {ck:18s} plan={plan:11s} meanL={r_['mean_L']:.3f}  a1={r_['a_teacher_forced'][0]:.3f}  "
                  f"c={[round(x, 3) for x in r_['c_conditional']]}", flush=True)
        del d
        torch.cuda.empty_cache()
    json.dump(res, open(os.path.join(RES, args.out + ".json"), "w"), indent=1)
    print("wrote", os.path.join(RES, args.out + ".json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("--plan", choices=["feat", "pred", "oracle_near", "oracle_far"], default="pred")
    ap.add_argument("--N", type=int, default=64, help="future summary horizon in tokens")
    ap.add_argument("--aux", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--stop", type=int, default=0, help="stop after this step, keeping the --steps LR schedule")
    ap.add_argument("--name", default=None)
    ap.add_argument("--ckpts", nargs="+", default=["d4_soft", "plan_feat", "plan_pred", "plan_oracle_near", "plan_oracle_far"])
    ap.add_argument("--sets", nargs="+", default=["gsm8k:40", "mtbench:40", "humaneval:40"])
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out", default="planner")
    a = ap.parse_args()
    if a.mode == "train":
        a.name = a.name or f"plan3_{a.plan}"
        train(a)
    else:
        evaluate(a)
