"""DFlash-backed stage 2: the frozen DFlash drafter is the forecaster, the stage-1 gist latent is the output space.

Input per round: DFlash's 8 block outputs for the next chunk (dflash_feats.py), i.e. its guesses of the target's
final states at positions q..q+7, plus the frozen LM-head row of each slot's own top-1 (as stage 1's --headfeat).
Heads (all read the 8 slots jointly with a small transformer):
  latent : slots -> gist latent z (nq x zq) -> frozen stage-1 decoder -> frozen LM head. Loss: token CE against the
           target's own top-1 + lat_w * MSE to the true gist latent of the chunk. This is the two-stage design with
           DFlash as stage 2's backbone.
  refine : control without the latent. slots -> per-slot correction added to DFlash's own output -> LM head. Same
           size, same data. If latent does not beat refine, the gain (if any) is from fine-tuning, not the gist.
Reported on dev, on exactly the rounds stage2.py scores (so the numbers sit in the same table):
  chain  : DFlash's own top-1 chain, capped at 8 drafted tokens (+ bonus), max 9
  pred   : the head's draft
  oracle : the true gist latent decoded (stage-1 ceiling)

Run:    python dflash_s2.py --head latent|refine [--steps 8000] [--smoke]
Output: results/dflash_s2_<head>.json
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ae_gate import CKPT, tblock
from common import DATA, RES
from stage2 import load_stage1

K = 8
FE = os.path.join(DATA, "dflash_feats")


class Head(nn.Module):
    def __init__(self, D, kind, nq, zq, d=512, depth=6):
        super().__init__()
        self.kind, self.nq = kind, nq
        self.inp = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, d))
        self.hf = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, d))
        self.pe = nn.Parameter(torch.randn(K, d) * 0.02)
        self.body = tblock(d, depth)
        if kind == "latent":
            self.q = nn.Parameter(torch.randn(nq, d) * 0.02)
            self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, zq))
        else:
            self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, D))
            nn.init.zeros_(self.out[1].weight)
            nn.init.zeros_(self.out[1].bias)

    def forward(self, h, hrow):                 # h (B, 8, D) DFlash outputs, hrow (B, 8, D) head rows of their top-1
        x = self.inp(h) + self.hf(hrow) + self.pe
        if self.kind == "latent":
            return self.out(self.body(torch.cat([self.q.expand(len(h), -1, -1), x], 1))[:, :self.nq])
        return h + self.out(self.body(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="latent", choices=["latent", "refine"])
    ap.add_argument("--ae", default="gist_hf_z2048")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--B", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--lat_w", type=float, default=1.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps = 30
    dev = "cuda"
    ae, a, cache, layers, stats, W = load_stage1(args.ae, dev)
    R, nq = len(layers), ae.nq
    fin = slice((R - 1) * K, R * K)
    fmean, fstd = stats[-1][0].to(dev), stats[-1][1].to(dev)

    def logits(x):                              # (..., D) head inputs -> (..., V) log-probs
        return (x.to(torch.bfloat16) @ W.T).float().log_softmax(-1)

    def dec_final(z):
        with torch.autocast("cuda", torch.bfloat16):
            return ae.decode(z).float()[:, fin] * fstd + fmean

    S = {}
    for sp in ["train", "dev_math", "dev_code", "dev_chat"]:
        mt = np.load(os.path.join(cache, f"{sp}_meta.npz"))
        S[sp] = {"H": np.load(os.path.join(FE, f"{sp}_h.npy"), mmap_mode="r"),
                 "rows": np.load(os.path.join(FE, f"{sp}_rows.npy")),
                 "top1": mt["top1"].astype(np.int64), "seq": mt["seq"],
                 "Z": torch.load(os.path.join(DATA, "zcache", f"{args.ae}_{sp}.pt"))}
    tr = S["train"]
    sub = tr["Z"][np.random.default_rng(1).choice(tr["rows"], 50000, replace=False)].float()
    zm, zs = sub.mean(0).to(dev), sub.std(0).clamp_min(1e-4).to(dev)

    def batch(sp, idx):
        s = S[sp]
        h = torch.from_numpy(np.ascontiguousarray(s["H"][np.sort(idx)])).to(dev).float()
        r = s["rows"][np.sort(idx)]
        lab = torch.from_numpy(s["top1"][r[:, None] + np.arange(K)]).to(dev)
        z = (s["Z"][r].float().to(dev) - zm) / zs
        hrow = W[logits(h).argmax(-1)].float()
        return h, hrow, lab, z

    net = Head(W.shape[1], args.head, nq, ae.zq, args.d, args.depth).to(dev)
    print(f"head params {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M, train rounds {len(tr['rows'])}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / 500) * max(0.1, 0.5 * (1 + np.cos(np.pi * min(1, s / args.steps)))))

    def draft(h, hrow):                         # -> head inputs (B, 8, D) and normalised latent (or None)
        with torch.autocast("cuda", torch.bfloat16):
            o = net(h, hrow).float()
        if args.head == "latent":
            return dec_final(o * zs + zm), o
        return o, None

    rng = np.random.default_rng(0)
    t0, log = time.time(), []
    net.train()
    for step in range(args.steps):
        h, hrow, lab, z = batch("train", rng.choice(len(tr["rows"]), args.B, replace=False))
        x, zo = draft(h, hrow)
        ce = F.nll_loss(logits(x).flatten(0, 1), lab.flatten())
        lat = F.mse_loss(zo, z) if zo is not None else torch.zeros((), device=dev)
        loss = ce + args.lat_w * lat
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.steps - 1:
            r = {"step": step, "ce": ce.item(), "lat": lat.item(), "min": (time.time() - t0) / 60}
            log.append(r)
            print(json.dumps(r), flush=True)
    tag = f"{args.head}{args.tag}{'_smoke' if args.smoke else ''}"
    if not args.smoke:
        torch.save({"model": net.state_dict(), "args": vars(args), "zm": zm.cpu(), "zs": zs.cpu()},
                   os.path.join(CKPT, f"dflash_s2_{tag}.pt"))

    net.eval()
    res = {"args": vars(args), "train_log": log, "sets": {}}
    for sp in ["dev_math", "dev_code", "dev_chat"]:
        s = S[sp]
        rows, seq = s["rows"], s["seq"]
        have = set(rows.tolist())
        keep, last = [], -10 ** 9                # the rounds stage2.py scores: chunk r-8 valid, rounds 16 apart
        for i, r in enumerate(rows):
            if r - K in have and seq[r - K] == seq[r] and r >= last + 2 * K:
                keep.append(i)
                last = r
        keep = np.array(keep)
        out = {"chain": [], "pred": [], "oracle": []}
        slot = {"chain": [], "pred": []}
        with torch.no_grad():
            for i in range(0, len(keep), 128):
                h, hrow, lab, z = batch(sp, keep[i:i + 128])
                x, _ = draft(h, hrow)
                for kind, hx in (("chain", h), ("pred", x), ("oracle", dec_final(z * zs + zm))):
                    eq = (logits(hx).argmax(-1) == lab).int()
                    out[kind].append(eq.cumprod(1).sum(1) + 1)
                    if kind in slot:
                        slot[kind].append(eq.float().mean(0))
        r = {"rounds": len(keep), **{f"accepted_{k}": torch.cat(v).float().mean().item() for k, v in out.items()},
             **{f"slot_{k}": torch.stack(v).mean(0).tolist() for k, v in slot.items()}, "max_per_round": K + 1}
        res["sets"][sp] = r
        print(sp, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if not k.startswith("slot")}),
              flush=True)
    json.dump(res, open(os.path.join(RES, f"dflash_s2_{tag}.json"), "w"), indent=1)
    print(f"=== dflash_s2 {tag} done {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
