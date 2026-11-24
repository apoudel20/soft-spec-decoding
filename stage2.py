"""Stage 2 of the two-stage drafter: a search network that maps the current chunk latent z_t to the next one z_t+1.

Stage 1 (ae_gate.py, frozen here) encodes a chunk of k=8 consecutive target states (layers x positions grid) into a
latent and decodes it back, the decoded final-layer row going through the frozen LM head.

At drafting time the last k verified positions p-7..p are known, so z_t = E(states p-7..p). The search network
predicts z_t+1 = the latent of states p+1..p+8. Decoding it and applying the head gives draft tokens for positions
p+2..p+9 (state q predicts token q+1). Token p+1 comes from the target's own state at p. Teacher-forced score per
round: leading drafted tokens equal to the target's own top-1, plus the target's bonus token, so at most k+1 = 9.

Modes: det (predict z_t+1 directly) and flow (rectified flow from noise to z_t+1, conditioned on z_t, Euler
sampling with --nfe steps). Loss: MSE in normalised latent space + dec_w * KL(target || decoded) through the frozen
decoder and head on the next chunk's final row ("trained to match the true planning latents of the next k").

Reported on dev (non-overlapping rounds): accepted per round for
  pred    : the search network's z_t+1
  oracle  : E(true next chunk), the stage-1 ceiling for this design
  copy    : z_t reused as z_t+1, a trivial floor
DFlash reference (same 40-answer dev sets, teacher-forced chain, 16 max): see results/dflash_tree_gap.json.

Control (--tok tokvae_z128): the same search network on a token-chunk VAE latent (tok_vae.py), which only knows
the tokens. If the target-state gist does not beat it, the gist carries nothing extra for drafting.

History (--hist H): the search network reads the last H chunk latents (H*8 positions, e.g. 8 blocks = 64 tokens)
instead of only z_t. Missing history at the start of an answer is zeros.

Run:    python stage2.py --ae z2048_full --mode det|flow [--smoke]   or   python stage2.py --tok tokvae_z128
Output: results/stage2_<ae>_<mode>.json, ckpt/stage2_<ae>_<mode>.pt
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ae_gate import AE, CKPT, Data, tblock
from common import DATA, RES


def load_stage1(name, dev):
    ck = torch.load(os.path.join(CKPT, f"ae_{name}.pt"), map_location="cpu")
    a = ck["args"]
    assert not a.get("ctx") or a.get("nocond"), "context-conditioned stage-1 models are not supported here"
    cache = os.path.join(DATA, a.get("cache", "cache"))
    layers = [18, 24, 30, 36] if a["rows"] == "full" else [36]
    stats = []
    for l in layers:
        s = torch.load(os.path.join(cache, f"stats_L{l}.pt"))
        mx = s["maxabs"]
        mask = (mx > 20 * mx.median()) if l != 36 else torch.zeros_like(mx, dtype=torch.bool)
        stats.append((s["mean"], s["std"].clamp_min(1e-3 * s["std"].median().item()), mask))
    W = torch.load(os.path.join(cache, "lm_head.pt")).to(dev)
    D, R = W.shape[1], len(layers)
    ae = AE(D, R, a["k"], a["zdim"], d=a.get("d", 512), depth=a.get("depth", 4), headfeat=a.get("headfeat", False))
    ae.load_state_dict(ck["model"])
    ae = ae.to(dev).eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae, a, cache, layers, stats, W


class Search(nn.Module):
    def __init__(self, nq, zq, d=512, depth=6, H=1):
        super().__init__()
        self.nq = nq
        self.cin, self.xin = nn.Linear(zq, d), nn.Linear(zq, d)
        self.pos = nn.Parameter(torch.randn((H + 1) * nq, d) * 0.02)
        self.temb = nn.Sequential(nn.Linear(64, d), nn.SiLU(), nn.Linear(d, d))
        self.body = tblock(d, depth)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, zq))

    def forward(self, zc, xt, t):              # zc, xt (B, nq, zq) normalised, t (B,)
        f = torch.exp(torch.linspace(0, np.log(1000), 32, device=t.device))
        te = torch.cat([torch.sin(t[:, None] * f), torch.cos(t[:, None] * f)], -1)
        h = torch.cat([self.cin(zc), self.xin(xt) + self.temb(te)[:, None]], 1) + self.pos
        return self.out(self.body(h)[:, -self.nq:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae", default="z2048_full")
    ap.add_argument("--tok", default="", help="use a token-VAE checkpoint (ckpt/<name>.pt) instead of --ae")
    ap.add_argument("--mode", default="det", choices=["det", "flow"])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--B", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--dec_w", type=float, default=1.0)
    ap.add_argument("--nfe", type=int, default=8)
    ap.add_argument("--hist", type=int, default=1, help="number of past chunk latents the search network reads")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = "cuda"
    if args.smoke:
        args.steps, args.B = 30, 16
    if args.tok:
        from tok_vae import TokVAE, load_embed
        ck = torch.load(os.path.join(CKPT, f"{args.tok}.pt"), map_location="cpu")
        a = ck["args"]
        cache = os.path.join(DATA, "cache")
        W = torch.load(os.path.join(cache, "lm_head.pt")).to(dev)
        Emb = load_embed().to(dev)
        ae = TokVAE(W.shape[1], a["zdim"]).to(dev).eval()
        ae.load_state_dict(ck["model"])
        for p_ in ae.parameters():
            p_.requires_grad_(False)
        a["k"], layers, stats = 8, [36], None
        args.ae = args.tok
    else:
        ae, a, cache, layers, stats, W = load_stage1(args.ae, dev)
    k, R, nq = a["k"], len(layers), ae.nq
    fin = slice((R - 1) * k, R * k)
    if not args.tok:
        fmean, fstd = stats[-1][0].to(dev), stats[-1][1].to(dev)

    def unz(x):
        return x * fstd + fmean

    def logp(h):
        return (h.to(torch.bfloat16) @ W.T).float().log_softmax(-1)

    def encode(x):
        if args.tok:
            with torch.autocast("cuda", torch.bfloat16):
                return ae.encode(Emb[x])[0].float()
        hf = W[logp(unz(x[:, fin]).flatten(0, 1)).argmax(-1)].view(len(x), k, -1).float() if ae.hf is not None else None
        with torch.autocast("cuda", torch.bfloat16):
            return ae.encode(x, None, hf)[0].float()

    def dec_final(z):                           # latent -> head inputs (B, k, D)
        with torch.autocast("cuda", torch.bfloat16):
            if args.tok:
                return ae.decode(z).float()
            return unz(ae.decode(z).float()[:, fin])

    # encode every valid chunk once with the frozen stage-1 encoder
    t0 = time.time()
    Z, meta = {}, {}
    T1, VAL = {}, {}
    for sp in ["train", "dev_math", "dev_code", "dev_chat"]:
        mt = np.load(os.path.join(cache, f"{sp}_meta.npz"))
        T1[sp] = mt["top1"].astype(np.int64)
        if args.tok:
            from tok_vae import chunks
            dt = type("D", (), {})()
            dt.seq = mt["seq"]
            _, dt.starts = chunks(sp)
        else:
            dt = Data(cache, sp, layers, k, stats)
        st = dt.starts[:4000] if args.smoke else dt.starts
        mu = torch.zeros(len(dt.seq), nq, ae.zq, dtype=torch.float16)
        valid = np.zeros(len(dt.seq), bool)
        with torch.no_grad():
            for i in range(0, len(st), 512):
                b = st[i:i + 512]
                if args.tok:
                    x = torch.from_numpy(T1[sp][b[:, None] + np.arange(k)[None]])
                else:
                    x, _, _ = dt.get(b)
                mu[b] = encode(x.to(dev)).half().cpu()
        valid[st] = True
        n = len(dt.seq)
        pair = np.zeros(n, bool)
        pair[:n - k] = valid[:n - k] & valid[k:] & (dt.seq[:n - k] == dt.seq[k:])
        Z[sp], meta[sp] = mu, (dt, np.nonzero(pair)[0])
        VAL[sp] = valid
        print(sp, "chunks", len(st), "pairs", int(pair.sum()), f"{(time.time() - t0) / 60:.1f} min", flush=True)
    tr_mu, (tr_data, tr_pairs) = Z["train"], meta["train"]
    sub = tr_mu[np.random.default_rng(1).choice(tr_pairs, min(50000, len(tr_pairs)), replace=False)].float()
    zm, zs = sub.mean(0).to(dev), sub.std(0).clamp_min(1e-4).to(dev)

    net = Search(nq, ae.zq, args.d, args.depth, args.hist).to(dev)

    def ctx(sp, s):                             # (B, hist*nq, zq) normalised, oldest first, zeros where missing
        seq = meta[sp][0].seq
        idx = s[:, None] - k * np.arange(args.hist - 1, -1, -1)[None]
        ok = (idx >= 0) & VAL[sp][np.clip(idx, 0, None)] & (seq[np.clip(idx, 0, None)] == seq[s][:, None])
        z = (Z[sp][np.clip(idx, 0, None)].float().to(dev) - zm) / zs
        z = z * torch.from_numpy(ok).to(dev)[:, :, None, None]
        return z.flatten(1, 2)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / 500) * max(0.1, 0.5 * (1 + np.cos(np.pi * min(1, s / args.steps)))))
    print(f"search params {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M", flush=True)

    def predict(zc_n, g=None):                  # normalised z_t -> normalised z_t+1
        B = zc_n.shape[0]
        shp = (B, nq, ae.zq)
        if args.mode == "det":
            return net(zc_n, torch.zeros(shp, device=dev), torch.zeros(B, device=dev))
        x = torch.randn(shp, device=dev, generator=g)
        for i in range(args.nfe):
            t = torch.full((B,), i / args.nfe, device=dev)
            x = x + net(zc_n, x, t) / args.nfe
        return x

    rng = np.random.default_rng(0)
    log = []
    net.train()
    for step in range(args.steps):
        s = rng.choice(tr_pairs, args.B)
        zc = ctx("train", s)
        z1 = (tr_mu[s + k].float().to(dev) - zm) / zs
        lab = torch.from_numpy(T1["train"][(s + k)[:, None] + np.arange(k)[None]]).to(dev)
        with torch.autocast("cuda", torch.bfloat16):
            if args.mode == "det":
                z1h = net(zc, torch.zeros_like(z1), torch.zeros(args.B, device=dev)).float()
                lat = F.mse_loss(z1h, z1)
            else:
                x0 = torch.randn_like(z1)
                t = torch.rand(args.B, device=dev)
                xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * z1
                v = net(zc, xt, t).float()
                lat = F.mse_loss(v, z1 - x0)
                z1h = xt + (1 - t)[:, None, None] * v
        loss = lat
        kl = torch.zeros((), device=dev)
        if args.dec_w:                          # decoded next chunk must give the target's own next 8 tokens
            lq = logp(dec_final(z1h * zs + zm).flatten(0, 1))
            kl = F.nll_loss(lq, lab.flatten())
            loss = loss + args.dec_w * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.steps - 1:
            r = {"step": step, "lat": lat.item(), "kl": kl.item(), "min": (time.time() - t0) / 60}
            log.append(r)
            print(json.dumps(r), flush=True)
            if not np.isfinite(loss.item()):
                raise SystemExit("non-finite loss")
    tag = f"{args.ae}_{args.mode}{f'_h{args.hist}' if args.hist > 1 else ''}{'_smoke' if args.smoke else ''}"
    if not args.smoke:
        torch.save({"model": net.state_dict(), "args": vars(args), "zm": zm.cpu(), "zs": zs.cpu()},
                   os.path.join(CKPT, f"stage2_{tag}.pt"))

    net.eval()
    res = {"args": vars(args), "stage1": a, "train_log": log, "sets": {}}
    g = torch.Generator(device=dev).manual_seed(0)
    for sp in ["dev_math", "dev_code", "dev_chat"]:
        dt, pairs = meta[sp]
        top1 = T1[sp]                                                       # the target's own top-1 per row
        keep, last = [], -10 ** 9
        for s in pairs:                          # non-overlapping rounds
            if s >= last + 2 * k:
                keep.append(s)
                last = s
        pairs = np.array(keep)
        out = {"pred": [], "oracle": [], "copy": []}
        slot = []
        with torch.no_grad():
            for i in range(0, len(pairs), 128):
                s = pairs[i:i + 128]
                zc = Z[sp][s].float().to(dev)
                zh = ctx(sp, s)
                lab = torch.from_numpy(top1[(s + k)[:, None] + np.arange(k)[None]].astype(np.int64))
                zp = predict(zh, g) * zs + zm
                for kind, z in (("pred", zp), ("oracle", Z[sp][s + k].float().to(dev)), ("copy", zc)):
                    tok = logp(dec_final(z).flatten(0, 1)).argmax(-1).view(len(s), k).cpu()
                    eq = (tok == lab).int()
                    out[kind].append(eq.cumprod(1).sum(1) + 1)
                    if kind == "pred":
                        slot.append(eq.float().mean(0))
        r = {"rounds": len(pairs), **{f"accepted_{kk}": torch.cat(v).float().mean().item() for kk, v in out.items()},
             "slot_top1_pred": torch.stack(slot).mean(0).tolist(), "max_per_round": k + 1}
        res["sets"][sp] = r
        print(sp, json.dumps({kk: v for kk, v in r.items() if kk != "slot_top1_pred"}), flush=True)
    json.dump(res, open(os.path.join(RES, f"stage2_{tag}.json"), "w"), indent=1)
    print(f"=== stage2 {tag} done", flush=True)


if __name__ == "__main__":
    main()
