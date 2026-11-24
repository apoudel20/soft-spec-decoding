"""Stage-2 entry gate: can a VAE compress k future Qwen3-8B states into one chunk latent and decode them back
so the frozen LM head gives the same tokens?

Input grid: rows = cached layers (18, 24, 30, final normed) or the final row only, columns = k consecutive
response positions. Each cell is z-scored per dimension with train stats. Massive-activation dims (max |x| >
20x the median dim's) are pinned to their mean and left out of the loss. Encoder: transformer over the R*k cells
plus nq query tokens -> nq latent tokens (total latent `zdim`). Decoder: transformer over the latent tokens plus
R*k output queries -> cells. Loss: KL(target || decoded) through the frozen LM head on the final row, + (1 - cos)
on the final row, + 0.1 * masked MSE on middle rows, + beta * VAE KL with a per-dim floor (CALM), latent dropout.

Gate (fixed before running): top-1 agreement with the target >= 99% chat, >= 99.5% math and code. Mean TV <=
0.01 chat, <= 0.005 math and code. Agreement drop <= 1 point with latent noise at 0.1 x the per-dim spread.

Run:    python ae_gate.py --name z1024_full --zdim 1024 [--rows final] [--smoke]
Output: results/ae_gate_<name>.json, ckpt/ae_<name>.pt
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

from common import DATA, RES

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt")
GATE = {"dev_chat": (0.99, 0.01), "dev_math": (0.995, 0.005), "dev_code": (0.995, 0.005)}


def tblock(d, n):
    l = nn.TransformerEncoderLayer(d, 8, 4 * d, 0.0, batch_first=True, norm_first=True, activation="gelu")
    return nn.TransformerEncoder(l, n, enable_nested_tensor=False)


class AE(nn.Module):
    def __init__(self, D, R, k, zdim, d=512, nq=4, depth=4):
        super().__init__()
        self.R, self.k, self.nq, self.zq = R, k, nq, zdim // nq
        self.inp = nn.Linear(D, d)
        self.cell_e = nn.Parameter(torch.randn(R * k, d) * 0.02)
        self.q = nn.Parameter(torch.randn(nq, d) * 0.02)
        self.enc = tblock(d, depth)
        self.mu, self.lv = nn.Linear(d, self.zq), nn.Linear(d, self.zq)
        self.zin = nn.Linear(self.zq, d)
        self.lat_e = nn.Parameter(torch.randn(nq, d) * 0.02)
        self.cell_d = nn.Parameter(torch.randn(R * k, d) * 0.02)
        self.dec = tblock(d, depth)
        self.nout = nn.LayerNorm(d)
        self.out = nn.Linear(d, D)

    def encode(self, x):                       # x (B, R*k, D) z-scored
        B = x.shape[0]
        h = torch.cat([self.q.expand(B, -1, -1), self.inp(x) + self.cell_e], 1)
        h = self.enc(h)[:, :self.nq]
        return self.mu(h), self.lv(h).clamp(-10, 5)

    def decode(self, z):                       # z (B, nq, zq)
        B = z.shape[0]
        h = torch.cat([self.zin(z) + self.lat_e, self.cell_d.expand(B, -1, -1)], 1)
        return self.out(self.nout(self.dec(h)[:, self.nq:]))


class Data:
    def __init__(self, cache, split, layers, k, stats):
        meta = np.load(os.path.join(cache, f"{split}_meta.npz"))
        self.seq, self.margin = meta["seq"], meta["margin"]
        self.mm = [np.load(os.path.join(cache, f"{split}_L{l}.npy"), mmap_mode="r") for l in layers]
        n = len(self.seq)
        ok = np.zeros(n, bool)
        ok[:n - k + 1] = self.seq[:n - k + 1] == self.seq[k - 1:]
        self.starts = np.nonzero(ok)[0]
        self.k, self.stats = k, stats

    def get(self, st):
        idx = (st[:, None] + np.arange(self.k)[None]).reshape(-1)
        srt = np.argsort(idx)
        xs = []
        for m, (mean, std, mask) in zip(self.mm, self.stats):
            a = np.empty((len(idx), m.shape[1]), np.float16)
            a[srt] = m[idx[srt]]
            x = (torch.from_numpy(a).float() - mean) / std
            x[:, mask] = 0
            xs.append(x.view(len(st), self.k, -1))
        return torch.stack(xs, 1).flatten(1, 2), idx       # (B, R*k, D), row-major


def prefetch(data, B, steps, rng):
    q = queue.Queue(maxsize=6)

    def work():
        for _ in range(steps):
            q.put(data.get(rng.choice(data.starts, B)))
        q.put(None)
    threading.Thread(target=work, daemon=True).start()
    while True:
        b = q.get()
        if b is None:
            return
        yield b


def nonoverlap(starts, k):
    keep, s = [], -10 ** 9
    for a in starts:
        if a >= s + k:
            keep.append(a)
            s = a
    return np.array(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--zdim", type=int, default=1024)
    ap.add_argument("--rows", default="full", choices=["full", "final"])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--B", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta", type=float, default=1e-3)
    ap.add_argument("--floor", type=float, default=0.5)
    ap.add_argument("--zdrop", type=float, default=0.15)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--det", action="store_true", help="plain autoencoder: decode mu, no sampling noise")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cache = os.path.join(DATA, "cache_smoke" if args.smoke else "cache")
    if args.smoke:
        args.steps, args.B = 30, 8
    layers = [18, 24, 30, 36] if args.rows == "full" else [36]
    stats = []
    for l in layers:
        s = torch.load(os.path.join(cache, f"stats_L{l}.pt"))
        mx = s["maxabs"]
        # pin massive dims only in middle rows: the final row is the head's input, so nothing there is pinned
        mask = (mx > 20 * mx.median()) if l != 36 else torch.zeros_like(mx, dtype=torch.bool)
        std = s["std"].clamp_min(1e-3 * s["std"].median().item())
        stats.append((s["mean"], std, mask))
    fstat = [t.cuda() for t in stats[-1]]
    W = torch.load(os.path.join(cache, "lm_head.pt")).cuda()
    D, R = W.shape[1], len(layers)
    tr = Data(cache, "train", layers, args.k, stats)
    model = AE(D, R, args.k, args.zdim, d=args.d, depth=args.depth).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / 500) * max(0.1, 0.5 * (1 + np.cos(np.pi * min(1, s / args.steps)))))
    keep = torch.stack([~s[2] for s in stats]).cuda().float()      # (R, D) dims kept in the loss
    fin = slice((R - 1) * args.k, R * args.k)

    def unz(x):                                  # final-row z -> normed state the LM head reads
        mean, std, mask = fstat
        h = x * std + mean
        return torch.where(mask, mean.expand_as(h), h)

    def logp(h):
        return (h.to(torch.bfloat16) @ W.T).float().log_softmax(-1)

    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M  train windows {len(tr.starts)}  "
          f"masked dims {[int(s[2].sum()) for s in stats]}", flush=True)
    t0, log = time.time(), []
    model.train()
    for step, (x, _) in enumerate(prefetch(tr, args.B, args.steps, np.random.default_rng(0))):
        x = x.cuda(non_blocking=True)
        with torch.autocast("cuda", torch.bfloat16):
            mu, lv = model.encode(x)
            z = mu if args.det else mu + torch.randn_like(mu) * (0.5 * lv).exp()
            z = F.dropout(z, args.zdrop)
            y = model.decode(z).float()
        fh, ft = unz(y[:, fin]), unz(x[:, fin])
        lq = logp(fh.flatten(0, 1))
        with torch.no_grad():
            lp = logp(ft.flatten(0, 1))
        kl = (lp.exp() * (lp - lq)).sum(-1).mean()
        cos = 1 - F.cosine_similarity(fh, ft, dim=-1).mean()
        loss = kl + cos
        if R > 1:
            e = (y[:, :fin.start] - x[:, :fin.start]).pow(2).view(x.shape[0], R - 1, args.k, D)
            mid = (e * keep[:R - 1, None, :]).sum() / (keep[:R - 1].sum() * args.k * x.shape[0])
            loss = loss + 0.1 * mid
        mu32, lv32 = mu.float(), lv.float()
        klz = (0.5 * (mu32 ** 2 + lv32.exp() - 1 - lv32)).mean((0, 1)).clamp_min(args.floor).sum()
        loss = loss + args.beta * klz
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.steps - 1:
            r = {"step": step, "kl": kl.item(), "cos": cos.item(), "klz": klz.item(),
                 "agree_train": (lq.argmax(-1) == lp.argmax(-1)).float().mean().item(), "min": (time.time() - t0) / 60}
            log.append(r)
            print(json.dumps(r), flush=True)
            if not np.isfinite(loss.item()):
                raise SystemExit("non-finite loss")
    if not args.smoke:
        os.makedirs(CKPT, exist_ok=True)
        torch.save({"model": model.state_dict(), "args": vars(args)}, os.path.join(CKPT, f"ae_{args.name}.pt"))

    model.eval()
    res = {"args": vars(args), "train_log": log, "sets": {}}
    g = torch.Generator(device="cuda").manual_seed(0)
    for split in ("dev_math", "dev_code", "dev_chat"):
        dv = Data(cache, split, layers, args.k, stats)
        st_all = nonoverlap(dv.starts, args.k)
        with torch.no_grad():
            mus = []
            for i in range(0, len(st_all), 64):
                x, _ = dv.get(st_all[i:i + 64])
                with torch.autocast("cuda", torch.bfloat16):
                    mus.append(model.encode(x.cuda())[0].float())
            spread = torch.cat(mus).std(0, keepdim=True)
            A = {"agree": [], "tv": [], "kl": [], "margin": []}
            N = {"0.05": [], "0.1": [], "0.3": [], "posterior": []}
            for i in range(0, len(st_all), 64):
                x, idx = dv.get(st_all[i:i + 64])
                x = x.cuda()
                lp = logp(unz(x[:, fin]).flatten(0, 1))
                with torch.autocast("cuda", torch.bfloat16):
                    mu, lv = model.encode(x)
                    zs = {"mean": mu, "posterior": mu + torch.randn(mu.shape, device="cuda", generator=g) * (0.5 * lv).exp()}
                    for sg in (0.05, 0.1, 0.3):
                        zs[str(sg)] = mu + sg * spread * torch.randn(mu.shape, device="cuda", generator=g)
                    outs = {kk: model.decode(v).float() for kk, v in zs.items()}
                for kk, y in outs.items():
                    lq = logp(unz(y[:, fin]).flatten(0, 1))
                    ag = (lq.argmax(-1) == lp.argmax(-1)).float().cpu()
                    if kk == "mean":
                        p, q = lp.exp(), lq.exp()
                        A["agree"].append(ag)
                        A["tv"].append((0.5 * (p - q).abs().sum(-1)).cpu())
                        A["kl"].append((p * (lp - lq)).sum(-1).cpu())
                        A["margin"].append(torch.from_numpy(dv.margin[idx]))
                    else:
                        N[kk].append(ag)
        A = {kk: torch.cat(v) for kk, v in A.items()}
        m = A["margin"]
        buckets = {"<0.5": m < 0.5, "0.5-2": (m >= 0.5) & (m < 2), ">=2": m >= 2}
        agree = A["agree"].mean().item()
        r = {"positions": len(m), "agree": agree, "tv": A["tv"].mean().item(), "kl": A["kl"].mean().item(),
             "agree_by_margin": {b: [A["agree"][mk].mean().item() if mk.any() else None, int(mk.sum())]
                                 for b, mk in buckets.items()},
             "agree_noise": {kk: torch.cat(v).mean().item() for kk, v in N.items()}}
        ga, gt = GATE[split]
        r["gate"] = {"agree_ok": agree >= ga, "tv_ok": r["tv"] <= gt,
                     "noise_ok": agree - r["agree_noise"]["0.1"] <= 0.01}
        r["gate"]["pass"] = all(r["gate"].values())
        res["sets"][split] = r
        print(split, json.dumps(r), flush=True)
    res["pass"] = all(v["gate"]["pass"] for v in res["sets"].values())
    out = os.path.join(RES, f"ae_gate_{args.name}{'_smoke' if args.smoke else ''}.json")
    json.dump(res, open(out, "w"), indent=1)
    print("GATE", "PASS" if res["pass"] else "FAIL", args.name, flush=True)
    print(f"=== ae {args.name} done", flush=True)


if __name__ == "__main__":
    main()
