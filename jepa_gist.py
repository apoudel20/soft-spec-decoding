"""Gist of the far future, learned by prediction in latent space (JEPA-style, VICReg regularisation).

For a round whose current chunk is rows s..s+7, the drafter drafts rows s+8..s+15. The gist covers the stretch after
that: rows s+16 .. s+16+F-1 of the target's states (layers --far_layers). Two networks:
  E  far encoder: far window -> G gist tokens of gdim numbers (the oracle gist, only available offline)
  P  predictor:   context window rows s-C+8 .. s+7 (layers --ctx_layers) -> the same G x gdim (available at drafting time)
Loss: VICReg on the flattened gists, invariance MSE(P, E) plus variance and covariance terms on both, so E cannot
collapse to a constant that P predicts trivially. No token loss anywhere.

Caveat: the target's state at row q has read every token up to q, so far-window states also carry the drafted tokens.
The token-oracle runs (stage2.py --far, built from token ids after the draft) are the leak-free comparison.

Run:    python jepa_gist.py train --name jg64      then   python jepa_gist.py export --name jg64
Output: ckpt/jepa_<name>.pt, data/gist/<name>_<cache>_<split>_{orc,pred}.npy  (N, G, gdim) float16, row = chunk start
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

LAYERS = [18, 24, 30, 36]


class WinEnc(nn.Module):
    def __init__(self, n, D, d, G, gdim, depth):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, d))
        self.pos = nn.Parameter(torch.randn(n, d) * 0.02)
        self.pad = nn.Parameter(torch.zeros(d))
        self.q = nn.Parameter(torch.randn(G, d) * 0.02)
        self.body = tblock(d, depth)
        self.out = nn.Linear(d, gdim)

    def forward(self, x, m):                    # x (B, n, D) normalised states, m (B, n) bool inside the answer
        h = torch.where(m[:, :, None], self.inp(x), self.pad) + self.pos
        h = self.body(torch.cat([h, self.q.expand(len(x), -1, -1)], 1))[:, -len(self.q):]
        return self.out(h)


class Split:
    def __init__(self, cache, split, layers):
        mt = np.load(os.path.join(cache, f"{split}_meta.npz"))
        self.seq = mt["seq"]
        self.mm = {l: np.load(os.path.join(cache, f"{split}_L{l}.npy"), mmap_mode="r") for l in layers}
        self.stats = {}
        for l in layers:
            s = torch.load(os.path.join(cache, f"stats_L{l}.pt"))
            mx = s["maxabs"]
            mask = (mx > 20 * mx.median()) if l != 36 else torch.zeros_like(mx, dtype=torch.bool)
            self.stats[l] = (s["mean"], s["std"].clamp_min(1e-3 * s["std"].median().item()), mask)

    def window(self, s, off, L, layers):        # (B, len(layers)*L, D), (B, len(layers)*L) for rows s+off .. s+off+L-1
        rows = s[:, None] + off + np.arange(L)[None]
        n = len(self.seq)
        rc = np.clip(rows, 0, n - 1)
        ok = (rows >= 0) & (rows < n) & (self.seq[rc] == self.seq[s][:, None])
        uniq, inv = np.unique(rc.reshape(-1), return_inverse=True)     # read each row once (export windows overlap)
        inv = torch.from_numpy(inv.reshape(-1))
        xs = []
        for l in layers:
            mean, std, mask = self.stats[l]
            x = (torch.from_numpy(np.ascontiguousarray(self.mm[l][uniq])).float() - mean) / std
            x[:, mask] = 0
            xs.append(x[inv].view(len(s), L, -1))
        m = torch.from_numpy(ok).repeat(1, len(layers))
        return torch.cat(xs, 1), m


def vicreg(p, e, w_inv=25.0, w_var=25.0, w_cov=1.0):
    p, e = p.flatten(1).float(), e.flatten(1).float()
    inv = F.mse_loss(p, e)
    def var(z):
        return F.relu(1 - torch.sqrt(z.var(0) + 1e-4)).mean()
    def cov(z):
        z = z - z.mean(0)
        c = (z.T @ z) / (len(z) - 1)
        return (c.pow(2).sum() - c.diagonal().pow(2).sum()) / z.shape[1]
    return w_inv * inv + w_var * (var(p) + var(e)) + w_cov * (cov(p) + cov(e)), inv


def build(args, D):
    nf, nc = len(args.far_layers) * args.far, len(args.ctx_layers) * args.ctx
    E = WinEnc(nf, D, args.d, args.G, args.gdim, args.depth)
    P = WinEnc(nc, D, args.d, args.G, args.gdim, args.depth)
    return E, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "export"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--train_cache", default="cache_pb")
    ap.add_argument("--dev_cache", default="cache")
    ap.add_argument("--far", type=int, default=64)
    ap.add_argument("--far_off", type=int, default=16)
    ap.add_argument("--far_layers", default="24,36")
    ap.add_argument("--ctx", type=int, default=64, help="context rows ending at the current chunk's last row")
    ap.add_argument("--ctx_layers", default="30,36")
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--gdim", type=int, default=64)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    args.far_layers = [int(x) for x in args.far_layers.split(",")]
    args.ctx_layers = [int(x) for x in args.ctx_layers.split(",")]
    dev = "cuda"
    used = sorted(set(args.far_layers) | set(args.ctx_layers))
    ck_path = os.path.join(CKPT, f"jepa_{args.name}.pt")
    ctx_off = 8 - args.ctx                      # context rows s+8-C .. s+7

    if args.cmd == "train":
        if args.smoke:
            args.steps = 30
        tr = Split(os.path.join(DATA, args.train_cache), "train", used)
        D = tr.mm[used[0]].shape[1]
        E, P = build(args, D)
        E, P = E.to(dev), P.to(dev)
        params = list(E.parameters()) + list(P.parameters())
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1, (s + 1) / 300) * max(0.1, 0.5 * (1 + np.cos(np.pi * min(1, s / max(1, args.steps))))))
        n = len(tr.seq)
        ok = np.zeros(n, bool)
        ok[:n - 16] = tr.seq[:n - 16] == tr.seq[16:]      # the drafted chunk is inside the answer
        starts = np.nonzero(ok)[0]
        rng = np.random.default_rng(0)
        t0, log = time.time(), []
        print(f"params E {sum(p.numel() for p in E.parameters()) / 1e6:.1f}M, P {sum(p.numel() for p in P.parameters()) / 1e6:.1f}M", flush=True)
        for step in range(args.steps):
            s = rng.choice(starts, args.B)
            xf, mf = tr.window(s, args.far_off, args.far, args.far_layers)
            xc, mc = tr.window(s, ctx_off, args.ctx, args.ctx_layers)
            keep = mf.any(1)                    # rounds with at least one far row in the answer
            with torch.autocast("cuda", torch.bfloat16):
                e = E(xf[keep].to(dev), mf[keep].to(dev))
                p = P(xc[keep].to(dev), mc[keep].to(dev))
            loss, inv = vicreg(p, e)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            if step % 250 == 0 or step == args.steps - 1:
                with torch.no_grad():
                    ef = e.flatten(1).float()
                    r = {"step": step, "loss": loss.item(), "inv": inv.item(),
                         "e_std": ef.std(0).mean().item(), "min": (time.time() - t0) / 60}
                log.append(r)
                print(json.dumps(r), flush=True)
        if True:
            torch.save({"E": E.state_dict(), "P": P.state_dict(), "args": vars(args), "log": log}, ck_path)
        print(f"=== jepa train {args.name} done", flush=True)
        return

    ck = torch.load(ck_path, map_location="cpu")
    a = argparse.Namespace(**ck["args"])
    out_dir = os.path.join(DATA, "gist")
    os.makedirs(out_dir, exist_ok=True)
    res = {"args": ck["args"], "sets": {}}
    for cache_name, splits in [(a.dev_cache, ["dev_math", "dev_code", "dev_chat"]), (a.train_cache, ["train"])]:
        for sp in splits:
            S = Split(os.path.join(DATA, cache_name), sp, used)
            D = S.mm[used[0]].shape[1]
            E, P = build(a, D)
            E.load_state_dict(ck["E"])
            P.load_state_dict(ck["P"])
            E, P = E.to(dev).eval(), P.to(dev).eval()
            n = len(S.seq) if not args.smoke else 2048
            paths = {kind: os.path.join(out_dir, f"{args.name}_{cache_name}_{sp}_{kind}.npy") for kind in ["orc", "pred"]}
            mm = {kind: np.lib.format.open_memmap(pth + ".tmp", "w+", np.float16, (n, a.G, a.gdim))
                  for kind, pth in paths.items()}
            t0, inv_sum, cnt = time.time(), 0.0, 0
            with torch.no_grad():
                for i in range(0, n, 256):
                    s = np.arange(i, min(n, i + 256))
                    xf, mf = S.window(s, a.far_off, a.far, a.far_layers)
                    xc, mc = S.window(s, ctx_off, a.ctx, a.ctx_layers)
                    with torch.autocast("cuda", torch.bfloat16):
                        e = E(xf.to(dev), mf.to(dev)).float()
                        p = P(xc.to(dev), mc.to(dev)).float()
                    e = e * mf.any(1).to(dev)[:, None, None]
                    mm["orc"][s] = e.half().cpu().numpy()
                    mm["pred"][s] = p.half().cpu().numpy()
                    inv_sum += F.mse_loss(p, e, reduction="sum").item()
                    cnt += e.numel()
                    if i % (256 * 400) == 0:
                        print(sp, i, n, f"{(time.time() - t0) / 60:.1f} min", flush=True)
            for kind, pth in paths.items():
                mm[kind].flush()
                del mm[kind]
                if not args.smoke:
                    os.replace(pth + ".tmp", pth)
            res["sets"][f"{cache_name}/{sp}"] = {"rows": n, "inv_mse": inv_sum / cnt}
            print(sp, json.dumps(res["sets"][f"{cache_name}/{sp}"]), flush=True)
    if not args.smoke:
        json.dump(res, open(os.path.join(RES, f"jepa_{args.name}_export.json"), "w"), indent=1)
    print(f"=== jepa export {args.name} done", flush=True)


if __name__ == "__main__":
    main()
