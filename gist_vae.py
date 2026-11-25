"""Gist VAE: the target's planning states for the next 64 positions -> 8 block latents (one per 8 positions).

Input per window: PCA'd states of layers 18, 24, 30 at 64 consecutive response positions (192 cells). Encoder:
transformer over the cells plus 8 query tokens -> 8 latents (mean and log-variance, zq dims each). Decoder: the 8
latents plus 192 output queries -> cells. Loss: MSE in PCA space (unit-variance components) + beta * KL with a
per-dim floor. The gist is meant to be lossy, so it is judged by reconstruction R^2, overall and per block.

After training, the encoder mean is computed for every anchor row i whose next 64 rows (i+1..i+64) are in the same
answer: data/cache/<split>_gist.npy (N, 8, zq) fp16 and <split>_gist_ok.npy (N,) bool.

Run: python gist_vae.py [--smoke]
Output: results/gist_vae.json, ckpt/gist_vae.pt, data/cache/*_gist*.npy
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from common import CKPT, DATA, RES

H, NB = 64, 8


def tblock(d, n, heads=8):
    l = nn.TransformerEncoderLayer(d, heads, 4 * d, 0.0, batch_first=True, norm_first=True, activation="gelu")
    return nn.TransformerEncoder(l, n, enable_nested_tensor=False)


class GistVAE(nn.Module):
    def __init__(self, K, zq=256, d=384, depth=4, L=3):
        super().__init__()
        self.L, self.zq = L, zq
        self.inp = nn.Linear(K, d)
        self.cell_e = nn.Parameter(torch.randn(H * L, d) * 0.02)
        self.q = nn.Parameter(torch.randn(NB, d) * 0.02)
        self.enc = tblock(d, depth)
        self.mu, self.lv = nn.Linear(d, zq), nn.Linear(d, zq)
        self.zin = nn.Linear(zq, d)
        self.lat_e = nn.Parameter(torch.randn(NB, d) * 0.02)
        self.cell_d = nn.Parameter(torch.randn(H * L, d) * 0.02)
        self.dec = tblock(d, depth)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, K))

    def encode(self, x):                     # x (B, H*L, K)
        h = torch.cat([self.q.expand(x.shape[0], -1, -1), self.inp(x) + self.cell_e], 1)
        h = self.enc(h)[:, :NB]
        return self.mu(h), self.lv(h).clamp(-10, 5)

    def decode(self, z):
        h = torch.cat([self.zin(z) + self.lat_e, self.cell_d.expand(z.shape[0], -1, -1)], 1)
        return self.out(self.dec(h)[:, NB:])


def windows(P, starts):                      # rows s..s+63 -> (B, H*L, K), position-major
    idx = starts[:, None] + np.arange(H)[None]
    x = torch.from_numpy(P[idx].astype(np.float32))          # (B, H, L, K)
    return x.flatten(1, 2)


def valid_starts(seq):
    n = len(seq)
    ok = np.zeros(n, bool)
    ok[:n - H + 1] = seq[:n - H + 1] == seq[H - 1:]
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zq", type=int, default=256)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta", type=float, default=1e-3)
    ap.add_argument("--floor", type=float, default=0.5)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = "cpu" if args.smoke else "cuda"
    sfx = "_smoke" if args.smoke else ""
    cache = os.path.join(DATA, "cache")
    if args.smoke:
        args.steps, args.B = 20, 4
    tr_split = "dev_chat" if args.smoke else "train"
    P = {sp: np.load(os.path.join(cache, f"{sp}_pca{sfx}.npy")) for sp in ["dev_math", "dev_code", "dev_chat"] + ([] if args.smoke else ["train"])}
    seq = {sp: np.load(os.path.join(cache, f"{sp}_meta.npz"))["seq"] for sp in P}
    K = P[tr_split].shape[2]
    model = GistVAE(K, args.zq, d=64 if args.smoke else 384, depth=1 if args.smoke else 4).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 300) * max(0.1, 0.5 * (1 + np.cos(np.pi * min(1, s / args.steps)))))
    st_tr = np.nonzero(valid_starts(seq[tr_split]))[0]
    rng = np.random.default_rng(0)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M  train windows {len(st_tr)}", flush=True)
    t0, log = time.time(), []
    for step in range(args.steps):
        x = windows(P[tr_split], rng.choice(st_tr, args.B)).to(dev)
        with torch.autocast(dev, torch.bfloat16, enabled=dev == "cuda"):
            mu, lv = model.encode(x)
            z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
            y = model.decode(z).float()
        mse = (y - x).pow(2).mean()
        mu32, lv32 = mu.float(), lv.float()
        klz = (0.5 * (mu32 ** 2 + lv32.exp() - 1 - lv32)).mean((0, 1)).clamp_min(args.floor).sum()
        loss = mse + args.beta * klz
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.steps - 1:
            r = {"step": step, "mse": mse.item(), "klz": klz.item(), "min": (time.time() - t0) / 60}
            log.append(r)
            print(json.dumps(r), flush=True)
    model.eval()
    res = {"args": vars(args), "train_log": log, "sets": {}}
    with torch.no_grad():
        for sp in ["dev_math", "dev_code", "dev_chat"]:
            st = np.nonzero(valid_starts(seq[sp]))[0][::8]
            se, sb = 0.0, np.zeros(NB)
            n = 0
            for a in range(0, len(st), 128):
                x = windows(P[sp], st[a:a + 128]).to(dev)
                with torch.autocast(dev, torch.bfloat16, enabled=dev == "cuda"):
                    y = model.decode(model.encode(x)[0]).float()
                e = (y - x).pow(2).view(len(x), NB, H // NB * model.L, K)
                se += e.sum().item()
                sb += e.sum((0, 2, 3)).cpu().numpy()
                n += len(x)
            var = P[sp].astype(np.float32).var(0).mean()
            r2 = 1 - se / (n * H * model.L * K) / var
            r2b = 1 - sb / (n * H // NB * model.L * K) / var
            res["sets"][sp] = {"windows": n, "r2": float(r2), "r2_by_block": [float(v) for v in r2b]}
            print(sp, json.dumps(res["sets"][sp]), flush=True)
        # precompute gist means for every anchor i: window rows i+1..i+64
        for sp, Ps in P.items():
            N = len(Ps)
            ok = np.zeros(N, bool)
            vs = valid_starts(seq[sp])
            ok[:-1] = vs[1:] & (seq[sp][:-1] == seq[sp][1:])
            G = np.lib.format.open_memmap(os.path.join(cache, f"{sp}_gist{sfx}.npy"), "w+", np.float16, (N, NB, args.zq))
            anchors = np.nonzero(ok)[0]
            for a in range(0, len(anchors), 512):
                an = anchors[a:a + 512]
                x = windows(Ps, an + 1).to(dev)
                with torch.autocast(dev, torch.bfloat16, enabled=dev == "cuda"):
                    G[an] = model.encode(x)[0].float().cpu().numpy().astype(np.float16)
            G.flush()
            np.save(os.path.join(cache, f"{sp}_gist_ok{sfx}.npy"), ok)
            print(sp, "gist anchors", int(ok.sum()), flush=True)
    if not args.smoke:
        os.makedirs(CKPT, exist_ok=True)
        torch.save({"model": model.state_dict(), "args": vars(args), "K": K}, os.path.join(CKPT, "gist_vae.pt"))
        json.dump(res, open(os.path.join(RES, "gist_vae.json"), "w"), indent=1)
    print("=== gist_vae done", flush=True)


if __name__ == "__main__":
    main()
