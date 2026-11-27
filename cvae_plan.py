"""Paragraph plan as a conditional VAE. plan_gist.py trains the plan encoder only to help a decoder that already reads
the context, so the plan learns what the context lacks, and no predictor from that context can supply it. Here the
predictor is a learned prior and the training loss charges the plan for every nat the prior cannot predict.

  posterior q(z | next G*8 target tokens)   PlanEnc over head-row embeddings -> G x c Gaussian (mean, log-variance)
  prior     p(z | target states at the current chunk)   -> G x c Gaussian
  decoder   p(drafted block | z, current chunk states)  the 23M decoder of plan_gist.py
  loss      NLL(drafted block | z ~ q) + beta * KL(q || p) / 8      (both per token; beta = 1 is the ELBO)
The KL weight rises from 0 to beta over --warm steps. At drafting time z comes from the prior, so the prior's samples
give the tree branches. The drafted block is the plan's first block (offset 0).

Reported on dev (accepted per round, at most 9): post_mean (oracle), post_sample, prior_mean, prior_sample, best of K
prior samples against the K most likely drafts of the decoder at the prior mean, trie nodes, and the KL in bits.
Control without a plan: results/plan_pw_G8c0.json.

Run:    python cvae_plan.py --G 32 --c 16 --beta 1
Output: results/cvae_<tag>.json, ckpt/cvae_<tag>.pt
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ae_gate import CKPT, Data, tblock
from common import DATA, RES
from plan_gist import K, LAYERS, Decoder, RawIn, kbest, load_stats, loader, pack, round_ok, tree_stats


class Post(nn.Module):
    def __init__(self, G, D, c, d=512, depth=4):
        super().__init__()
        self.G = G
        self.inp = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, d))
        self.pos = nn.Parameter(torch.randn(G * K, d) * 0.02)
        self.pad = nn.Parameter(torch.zeros(d))
        self.q = nn.Parameter(torch.randn(G, d) * 0.02)
        self.body = tblock(d, depth)
        self.out = nn.Linear(d, 2 * c)

    def forward(self, e, m):
        h = torch.where(m[:, :, None], self.inp(e), self.pad) + self.pos
        h = self.body(torch.cat([h, self.q.expand(len(e), -1, -1)], 1))[:, -self.G:]
        mu, lv = self.out(h).float().chunk(2, -1)
        return mu, lv.clamp(-8, 4)


class Prior(nn.Module):
    def __init__(self, G, D, c, d=512, depth=4):
        super().__init__()
        self.raw = RawIn(D, d)
        self.q = nn.Parameter(torch.randn(G, d) * 0.02)
        self.body = tblock(d, depth)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * c))

    def forward(self, raw):
        h = torch.cat([self.raw(raw), self.q.expand(len(raw), -1, -1)], 1)
        mu, lv = self.out(self.body(h)[:, -len(self.q):]).float().chunk(2, -1)
        return mu, lv.clamp(-8, 4)


def gkl(mq, lq, mp, lp):                        # KL(q || p) per round, nats, summed over the plan
    return 0.5 * (lp - lq + (lq.exp() + (mq - mp) ** 2) / lp.exp() - 1).sum((1, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--G", type=int, default=32)
    ap.add_argument("--c", type=int, default=16)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--warm", type=int, default=3000)
    ap.add_argument("--train_cache", default="cache_pb")
    ap.add_argument("--dev_cache", default="cache")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--B", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bestk", default="4,8")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.B = 30, 16
    dev = "cuda"
    G, c = args.G, args.c
    tag = f"G{G}c{c}b{args.beta:g}" + (f"_st{args.steps}" if args.steps != 10000 else "") + ("_smoke" if args.smoke else "")
    print(f"=== cvae {tag}: plan {G * K} tokens, {G} x {c}, beta {args.beta}", flush=True)

    dev_cache = os.path.join(DATA, args.dev_cache)
    stats = load_stats(dev_cache)
    W = torch.load(os.path.join(dev_cache, "lm_head.pt")).to(dev)
    D = W.shape[1]
    fmean, fstd = stats[-1][0].to(dev), stats[-1][1].to(dev)

    def logp(h):
        return ((h.float() * fstd + fmean).to(torch.bfloat16) @ W.T).float().log_softmax(-1)

    def split(cache, sp):
        dt = Data(os.path.join(DATA, cache), sp, LAYERS, K, stats)
        T1 = np.load(os.path.join(DATA, cache, f"{sp}_meta.npz"))["top1"].astype(np.int64)
        st = np.arange(len(dt.seq))
        return dt, T1, st[round_ok(dt.seq, st, 2 * K)]

    t0 = time.time()
    tr, trT1, trst = split(args.train_cache, "train")
    q, p, dec = Post(G, D, c).to(dev), Prior(G, D, c).to(dev), Decoder(G, c, D).to(dev)
    count = lambda m: sum(x.numel() for x in m.parameters()) / 1e6
    print(f"params posterior {count(q):.1f}M, prior {count(p):.1f}M, decoder {count(dec):.1f}M", flush=True)
    params = list(q.parameters()) + list(p.parameters()) + list(dec.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / 500) * max(0.1, 0.5 * (1 + np.cos(np.pi * min(1, s / max(1, args.steps))))))
    log = []
    for step, b in enumerate(loader(tr, trT1, trst, args.B, args.steps, G, 0.0, 0)):
        raw, raw_a, ids, m, lab, o, _, _ = [x.to(dev, non_blocking=True) for x in b]
        with torch.autocast("cuda", torch.bfloat16):
            mq, lq = q(W[ids], m)
            mp, lp = p(raw)
            z = mq + (0.5 * lq).exp() * torch.randn_like(mq)
            nll = F.nll_loss(logp(dec(z, o, raw)).flatten(0, 1), lab.flatten())
        kl = gkl(mq, lq, mp, lp).mean()
        beta = args.beta * min(1.0, step / max(1, args.warm))
        loss = nll + beta * kl / K
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sch.step()
        if step % 500 == 0 or step == args.steps - 1:
            r = {"step": step, "nll": nll.item(), "kl_bits": kl.item() / np.log(2), "beta": beta,
                 "post_std": mq.std(0).mean().item(), "min": (time.time() - t0) / 60}
            log.append(r)
            print(json.dumps(r), flush=True)
            if not np.isfinite(loss.item()):
                raise SystemExit("non-finite loss")
    if not args.smoke:
        torch.save({"post": q.state_dict(), "prior": p.state_dict(), "dec": dec.state_dict(), "args": vars(args)},
                   os.path.join(CKPT, f"cvae_{tag}.pt"))

    for mod in (q, p, dec):
        mod.eval()
    res = {"args": vars(args), "train_log": log, "sets": {}}
    g = torch.Generator(device=dev).manual_seed(0)
    bk = [int(x) for x in args.bestk.split(",")]
    for sp in ["dev_math", "dev_code", "dev_chat"]:
        dt, T1, st = split(args.dev_cache, sp)
        keep, last = [], -10 ** 9
        for s in st:
            if s >= last + 2 * K:
                keep.append(s)
                last = s
        rounds = np.array(keep)[:256] if args.smoke else np.array(keep)
        out, ex, kls = {}, {}, []
        with torch.no_grad():
            for i in range(0, len(rounds), 64):
                s = rounds[i:i + 64]
                raw, raw_a, ids, m, lab, oo, _, _ = [x.to(dev) for x in pack(dt, T1, s, s, G)]
                B = len(s)
                with torch.autocast("cuda", torch.bfloat16):
                    mq, lq = q(W[ids], m)
                    mp, lp = p(raw)
                    kls.append((gkl(mq, lq, mp, lp) / np.log(2)).cpu())
                    nz = lambda mu, lv: mu + (0.5 * lv).exp() * torch.randn(mu.shape, device=dev, generator=g)
                    ys = {"post_mean": mq, "post_sample": nz(mq, lq), "prior_mean": mp, "prior_sample": nz(mp, lp)}
                    for kind, y in ys.items():
                        lpk = logp(dec(y, oo, raw))
                        out.setdefault(kind, []).append(((lpk.argmax(-1) == lab).int().cumprod(1).sum(1) + 1).float().cpu())
                        if kind == "prior_mean":
                            for Kb in bk:
                                for nm, v in zip(["best", "mean", "distinct", "nodes"], tree_stats(kbest(lpk, Kb).cpu(), lab.cpu())):
                                    ex.setdefault(f"kbest{Kb}_{nm}", []).append(v)
                    for Kb in bk:
                        zk = nz(mp.repeat_interleave(Kb, 0), lp.repeat_interleave(Kb, 0))
                        hk = dec(zk, oo.repeat_interleave(Kb, 0), raw.repeat_interleave(Kb, 0))
                        tk = torch.cat([logp(x).argmax(-1) for x in hk.split(128)])
                        for nm, v in zip(["best", "mean", "distinct", "nodes"], tree_stats(tk.view(B, Kb, K).cpu(), lab.cpu())):
                            ex.setdefault(f"prior{Kb}_{nm}", []).append(v)
        r = {"rounds": int(len(rounds)), "kl_bits": torch.cat(kls).mean().item()}
        r.update({f"accepted_{kk}": torch.cat(v).mean().item() for kk, v in out.items()})
        r.update({kk: torch.cat(v).mean().item() for kk, v in ex.items()})
        res["sets"][sp] = r
        print(sp, json.dumps(r), flush=True)
    json.dump(res, open(os.path.join(RES, f"cvae_{tag}.json"), "w"), indent=1)
    print(f"=== cvae {tag} done {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
