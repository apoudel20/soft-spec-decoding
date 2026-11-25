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


class MultiData:
    """Several state caches read as one: rows are concatenated, answer ids offset so they stay distinct."""

    def __init__(self, parts):
        self.parts = parts
        self.off = np.cumsum([0] + [len(p.seq) for p in parts])
        so = np.cumsum([0] + [int(p.seq.max()) + 1 for p in parts])
        self.seq = np.concatenate([p.seq + o for p, o in zip(parts, so[:-1])])

    def get(self, rows):
        rows = np.asarray(rows)
        out = None
        for i, p in enumerate(self.parts):
            m = (rows >= self.off[i]) & (rows < self.off[i + 1])
            if not m.any():
                continue
            x = p.get(rows[m] - self.off[i])[0]
            if out is None:
                out = torch.zeros((len(rows),) + tuple(x.shape[1:]), dtype=x.dtype)
            out[torch.from_numpy(m)] = x
        return out, None, None


class ZStore:
    """Encoded latents for one or more caches, read from disk-backed arrays (np.memmap) instead of RAM.
    Indexing with an int array of any shape returns a float16 torch tensor of shape idx.shape + (nq, zq)."""

    def __init__(self, arrs):
        self.arrs = arrs
        self.off = np.cumsum([0] + [len(a) for a in arrs])
        self.tail = tuple(arrs[0].shape[1:])

    def __len__(self):
        return int(self.off[-1])

    def __getitem__(self, idx):
        idx = np.asarray(idx)
        flat = idx.reshape(-1)
        out = np.empty((len(flat),) + self.tail, np.float16)
        for i, a in enumerate(self.arrs):
            m = (flat >= self.off[i]) & (flat < self.off[i + 1])
            if m.any():
                out[m] = a[flat[m] - self.off[i]]
        return torch.from_numpy(out.reshape(idx.shape + self.tail))


class Search(nn.Module):
    def __init__(self, nq, zq, d=512, depth=6, H=1, raw=0, D=4096, out_dim=None):
        super().__init__()
        self.nq = nq
        self.cin, self.xin = nn.Linear(zq, d), nn.Linear(zq, d)
        self.pos = nn.Parameter(torch.randn((H + 1) * nq, d) * 0.02)
        self.temb = nn.Sequential(nn.Linear(64, d), nn.SiLU(), nn.Linear(d, d))
        if raw:                                 # the target's raw states (layers x positions) of the current chunk
            self.rin = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, d))
            self.rpos = nn.Parameter(torch.randn(raw, d) * 0.02)
        self.body = tblock(d, depth)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, out_dim or zq))

    def forward(self, zc, xt, t, raw=None):    # zc, xt (B, nq, zq) normalised, t (B,), raw (B, R*k, D)
        f = torch.exp(torch.linspace(0, np.log(1000), 32, device=t.device))
        te = torch.cat([torch.sin(t[:, None] * f), torch.cos(t[:, None] * f)], -1)
        h = torch.cat([self.cin(zc), self.xin(xt) + self.temb(te)[:, None]], 1) + self.pos
        if raw is not None:
            h = torch.cat([self.rin(raw) + self.rpos, h], 1)
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
    ap.add_argument("--lat_w", type=float, default=1.0, help="weight of the latent MSE (0 = token loss only)")
    ap.add_argument("--nfe", type=int, default=8)
    ap.add_argument("--hist", type=int, default=1, help="number of past chunk latents the search network reads")
    ap.add_argument("--raw", action="store_true",
                    help="also feed the target's raw states (all cached layers) at the current chunk's positions")
    ap.add_argument("--direct", action="store_true",
                    help="ablation: no latent. Predict the next 8 head-input states directly and decode with the LM "
                         "head (token loss only); the current gist and raw states are still the input")
    ap.add_argument("--ctx_pos", type=int, default=0,
                    help="also feed raw states of this many positions before the chunk (zeros where outside the answer)")
    ap.add_argument("--ctx_layers", default="30,36", help="cached layers used for the --ctx_pos context rows")
    ap.add_argument("--train_cache", default="", help="train split from data/<this> (e.g. cache_pb); dev stays on the stage-1 cache")
    ap.add_argument("--init", default="", help="start from ckpt/<this>.pt (same architecture)")
    ap.add_argument("--t_shift", type=float, default=1.0,
                    help="flow: shift timesteps towards noise (SD3/RAE style; about 3 for a 2,048-dim latent)")
    ap.add_argument("--bestk", default="", help="flow eval: comma list of K; report best-of-K sampled drafts")
    ap.add_argument("--eval_only", action="store_true", help="no training; evaluate --init")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.eval_only:
        args.steps = 0
    assert not (args.raw and args.tok), "--raw needs a state-cache stage-1 model"
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
    def load_part(cdir, sp, zc_path):            # one cache split -> (Data, top1, encoded latents, chunk starts)
        mt = np.load(os.path.join(cdir, f"{sp}_meta.npz"))
        top1 = mt["top1"].astype(np.int64)
        if args.tok:
            from tok_vae import chunks
            dt = type("D", (), {})()
            dt.seq = mt["seq"]
            _, dt.starts = chunks(sp)
        else:
            dt = Data(cdir, sp, layers, k, stats)
        st = dt.starts[:4000] if args.smoke else dt.starts
        npy = zc_path[:-3] + ".npy"                   # disk-backed latents: a 2M-row cache is 8 GB
        if not args.smoke and os.path.exists(npy):
            mu = np.load(npy, mmap_mode="r")
        elif not args.smoke and os.path.exists(zc_path):
            old = torch.load(zc_path).numpy()
            np.save(npy, old)
            del old
            mu = np.load(npy, mmap_mode="r")
        else:
            shape = (len(dt.seq), nq, ae.zq)
            if args.smoke:
                mu = np.zeros(shape, np.float16)
            else:
                os.makedirs(os.path.dirname(npy), exist_ok=True)
                mu = np.lib.format.open_memmap(npy + ".tmp", "w+", np.float16, shape)
            with torch.no_grad():
                for i in range(0, len(st), 512):
                    b = st[i:i + 512]
                    if args.tok:
                        x = torch.from_numpy(top1[b[:, None] + np.arange(k)[None]])
                    else:
                        x, _, _ = dt.get(b)
                    mu[b] = encode(x.to(dev)).half().cpu().numpy()
            if not args.smoke:
                mu.flush()
                del mu
                os.replace(npy + ".tmp", npy)
                mu = np.load(npy, mmap_mode="r")
        return dt, top1, mu, st

    for sp in ["train", "dev_math", "dev_code", "dev_chat"]:
        names = args.train_cache.split(",") if (sp == "train" and args.train_cache) else [None]
        parts = [load_part(os.path.join(DATA, nm) if nm else cache, sp,
                           os.path.join(DATA, "zcache", f"{args.ae}_{sp}{'_' + nm if nm else ''}.pt")) for nm in names]
        if len(parts) == 1:
            dt, T1[sp], mu, st = parts[0]
            mu = ZStore([mu])
        else:                                    # several train caches: concatenate rows, keep answers distinct
            dt = MultiData([p_[0] for p_ in parts])
            T1[sp] = np.concatenate([p_[1] for p_ in parts])
            mu = ZStore([p_[2] for p_ in parts])
            st = np.concatenate([p_[3] + o for p_, o in zip(parts, dt.off[:-1])])
        valid = np.zeros(len(dt.seq), bool)
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

    nraw = (R * k + len(args.ctx_layers.split(",")) * args.ctx_pos) if args.raw else 0
    nq_out = k if args.direct else nq           # direct: one output query per drafted position
    assert not (args.direct and args.mode != "det"), "--direct is a deterministic ablation"
    net = Search(nq, ae.zq, args.d, args.depth, args.hist, nraw, W.shape[1],
                 out_dim=W.shape[1] if args.direct else None).to(dev)
    if args.direct:                             # output queries: k positions instead of nq latent tokens
        net.nq = k
        net.pos = nn.Parameter(torch.randn(args.hist * nq + k, args.d, device=dev) * 0.02)
    if args.init:                               # shapes that changed (e.g. rpos with --ctx_pos) start fresh
        sd, own = torch.load(os.path.join(CKPT, f"{args.init}.pt"), map_location="cpu")["model"], net.state_dict()
        keep = {kk: v for kk, v in sd.items() if kk in own and own[kk].shape == v.shape}
        skipped = sorted(set(own) - set(keep))
        net.load_state_dict(keep, strict=False)
        print(f"init from {args.init}: {len(keep)} tensors loaded, fresh: {skipped}", flush=True)

    cl = [layers.index(int(x)) for x in args.ctx_layers.split(",")] if args.ctx_pos else []

    def ctxraw(sp, s):                          # (B, len(cl)*C, D): raw states of the C positions before chunk s
        dt = meta[sp][0]
        assert not isinstance(dt, MultiData), "--ctx_pos supports a single train cache"
        C = args.ctx_pos
        rows = s[:, None] + np.arange(-C, 0)[None]
        rc = np.clip(rows, 0, None)
        ok = torch.from_numpy((rows >= 0) & (dt.seq[rc] == dt.seq[s][:, None]))          # same answer only
        out = []
        for li in cl:
            mean, std, mask = dt.stats[li]
            flat = rc.reshape(-1)
            srt = np.argsort(flat)
            a = np.empty((len(flat), dt.mm[li].shape[1]), np.float16)
            a[srt] = dt.mm[li][flat[srt]]
            x = (torch.from_numpy(a).float() - mean) / std
            x[:, mask] = 0
            out.append(x.view(len(s), C, -1) * ok[:, :, None])
        return torch.cat(out, 1)

    def rawx(sp, s):                            # (B, R*k [+ len(cl)*C], D) normalised target states, or None
        if not args.raw:
            return None
        x = meta[sp][0].get(s)[0]
        if args.ctx_pos:
            x = torch.cat([x, ctxraw(sp, s)], 1)
        return x.to(dev)

    def ctx(sp, s):                             # (B, hist*nq, zq) normalised, oldest first, zeros where missing
        seq = meta[sp][0].seq
        idx = s[:, None] - k * np.arange(args.hist - 1, -1, -1)[None]
        ok = (idx >= 0) & VAL[sp][np.clip(idx, 0, None)] & (seq[np.clip(idx, 0, None)] == seq[s][:, None])
        z = (Z[sp][np.clip(idx, 0, None)].float().to(dev) - zm) / zs
        z = z * torch.from_numpy(ok).to(dev)[:, :, None, None]
        return z.flatten(1, 2)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / 500) * max(0.1, 0.5 * (1 + np.cos(np.pi * min(1, s / max(1, args.steps))))))
    print(f"search params {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M", flush=True)

    def tmap(u):                                # data at t=1; shift spends more steps near noise
        sig = 1 - u
        return 1 - args.t_shift * sig / (1 + (args.t_shift - 1) * sig)

    def to_head(o):                             # stage-2 output -> head inputs (B, k, D)
        return unz(o.float()) if args.direct else dec_final(o * zs + zm)

    def predict(zc_n, g=None, raw=None):        # normalised z_t -> normalised z_t+1 (a sample in flow mode)
        B = zc_n.shape[0]
        shp = (B, nq_out, ae.zq)
        raw = raw.float() if raw is not None else None
        if args.mode == "det":
            return net(zc_n, torch.zeros(shp, device=dev), torch.zeros(B, device=dev), raw)
        x = torch.randn(shp, device=dev, generator=g)
        grid = tmap(torch.linspace(0, 1, args.nfe + 1, device=dev))
        for i in range(args.nfe):
            t = grid[i].expand(B)
            x = x + net(zc_n, x, t, raw) * (grid[i + 1] - grid[i])
        return x

    rng = np.random.default_rng(0)
    log = []
    net.train()
    for step in range(args.steps):
        s = rng.choice(tr_pairs, args.B)
        zc = ctx("train", s)
        rx = rawx("train", s)
        z1 = (tr_mu[s + k].float().to(dev) - zm) / zs
        lab = torch.from_numpy(T1["train"][(s + k)[:, None] + np.arange(k)[None]]).to(dev)
        with torch.autocast("cuda", torch.bfloat16):
            if args.mode == "det":
                z1h = net(zc, torch.zeros(args.B, nq_out, ae.zq, device=dev), torch.zeros(args.B, device=dev), rx).float()
                lat = torch.zeros((), device=dev) if args.direct else F.mse_loss(z1h, z1)
            else:
                x0 = torch.randn_like(z1)
                t = tmap(torch.rand(args.B, device=dev))
                xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * z1
                v = net(zc, xt, t, rx).float()
                lat = F.mse_loss(v, z1 - x0)
                z1h = xt + (1 - t)[:, None, None] * v
        loss = args.lat_w * lat
        kl = torch.zeros((), device=dev)
        if args.dec_w:                          # decoded next chunk must give the target's own next 8 tokens
            lq = logp(to_head(z1h).flatten(0, 1))
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
    tag = f"{'eval_' if args.eval_only else ''}{args.ae}_{args.mode}{'_direct' if args.direct else ''}{'_raw' if args.raw else ''}{f'_ctx{args.ctx_pos}' if args.ctx_pos else ''}{f'_{args.train_cache.replace(",", "+")}' if args.train_cache else ''}{f'_lat{args.lat_w:g}' if args.lat_w != 1 else ''}{f'_h{args.hist}' if args.hist > 1 else ''}{f'_d{args.d}x{args.depth}' if (args.d, args.depth) != (512, 6) else ''}{f'_s{args.steps}' if args.steps != 6000 else ''}{f'_nfe{args.nfe}' if args.mode == 'flow' and args.nfe != 8 else ''}{'_smoke' if args.smoke else ''}"
    if not args.smoke and not args.eval_only:
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
                po = predict(zh, g, rawx(sp, s))
                heads = {"pred": to_head(po), "oracle": dec_final(Z[sp][s + k].float().to(dev)), "copy": dec_final(zc)}
                for kind, hx in heads.items():
                    tok = logp(hx.flatten(0, 1)).argmax(-1).view(len(s), k).cpu()
                    eq = (tok == lab).int()
                    out[kind].append(eq.cumprod(1).sum(1) + 1)
                    if kind == "pred":
                        slot.append(eq.float().mean(0))
        r = {"rounds": len(pairs), **{f"accepted_{kk}": torch.cat(v).float().mean().item() for kk, v in out.items()},
             "slot_top1_pred": torch.stack(slot).mean(0).tolist(), "max_per_round": k + 1}
        if args.bestk and args.mode == "flow":   # K sampled futures per round: best one = ceiling of a K-branch tree
            for K in [int(x) for x in args.bestk.split(",")]:
                best, single, distinct = [], [], []
                with torch.no_grad():
                    for i in range(0, len(pairs), 32):
                        s = pairs[i:i + 32]
                        zh, rx = ctx(sp, s), rawx(sp, s)
                        zp = predict(zh.repeat_interleave(K, 0), g,
                                     rx.repeat_interleave(K, 0) if rx is not None else None) * zs + zm
                        tok = logp(dec_final(zp).flatten(0, 1)).argmax(-1).view(len(s), K, k).cpu()
                        lab = torch.from_numpy(top1[(s + k)[:, None] + np.arange(k)[None]].astype(np.int64))
                        acc = (tok == lab[:, None]).int().cumprod(2).sum(2) + 1          # (B, K)
                        best.append(acc.max(1).values)
                        single.append(acc.float().mean(1))
                        distinct.append(torch.tensor([len({tuple(x) for x in tok[j].tolist()}) for j in range(len(s))]))
                r[f"accepted_best{K}"] = torch.cat(best).float().mean().item()
                r[f"accepted_single_k{K}"] = torch.cat(single).mean().item()
                r[f"distinct_drafts_k{K}"] = torch.cat(distinct).float().mean().item()
        res["sets"][sp] = r
        print(sp, json.dumps({kk: v for kk, v in r.items() if kk != "slot_top1_pred"}), flush=True)
    json.dump(res, open(os.path.join(RES, f"stage2_{tag}.json"), "w"), indent=1)
    print(f"=== stage2 {tag} done", flush=True)


if __name__ == "__main__":
    main()
