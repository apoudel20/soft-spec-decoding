"""Paragraph plan for drafting. The plan is G ordered block latents over the next G*8 tokens, and its first block is
the one drafted this round (rows s+8..s+15 for a round whose verified chunk is rows s..s+7). The decoder drafts an
8-token block from the plan, the target's states at the current chunk, and the offset o of that block inside the
plan, so one plan can serve several rounds in order (progressive drafting).

Networks, trained together:
  E     plan encoder (oracle, offline): the target's own next G*8 tokens, embedded with the frozen LM-head rows,
        -> G x c numbers scaled to unit average power. A Gaussian channel y = mu + sigma * noise caps the plan at
        G * c * 0.5 * log2(1 + 1 / sigma^2) bits, so with small c the plan cannot be an exact code of the block.
  Dec   decoder: [G plan tokens + offset | 32 raw states (4 layers x 8 rows) | 8 queries] -> 8 head inputs -> head.
        Loss: NLL of the target's own top-1 at the drafted rows.
  P     deterministic predictor: raw states at the plan's anchor -> mu (MSE to stop-grad mu).
  Flow  rectified-flow sampler over mu conditioned on the same raw states. Samples give different plans, hence
        different drafts (a draft tree whose branches come from plans).
The anchor s0 = s - o is where the plan was made. With o > 0 the plan is stale by o tokens, which is the
progressive setting. Half the training rounds use o = 0, the rest o uniform in 1..G*8-8 (same answer only).
--c 0 is the control: the same decoder with no plan.

Reported on dev, non-overlapping rounds, accepted tokens per round (bonus included, at most 9):
  oracle (E(true future) through the channel), oracle_clean (no channel noise), zero (plan input zeroed),
  det (P), flow (one sample), at offsets 0/16/32/48 (and 128/248/504 when the plan is long enough), plan made at the anchor;
  at offset 0: best of K flow samples and best of the K most likely drafts under the decoder's own per-slot
  distribution (exact K-best of the slot product, the per-position tree baseline), with trie node counts.

Run:    python plan_gist.py --c 4 [--smoke]
Output: results/plan_<tag>.json, ckpt/plan_<tag>.pt
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

from ae_gate import CKPT, Data, tblock
from common import DATA, RES

LAYERS = [18, 24, 30, 36]
K = 8


def load_stats(cache):
    out = []
    for l in LAYERS:
        s = torch.load(os.path.join(cache, f"stats_L{l}.pt"))
        mx = s["maxabs"]
        mask = (mx > 20 * mx.median()) if l != 36 else torch.zeros_like(mx, dtype=torch.bool)
        out.append((s["mean"], s["std"].clamp_min(1e-3 * s["std"].median().item()), mask))
    return out


def temb(t, d=64):
    f = torch.exp(torch.linspace(0, np.log(1000), d // 2, device=t.device))
    return torch.cat([torch.sin(t[:, None] * f), torch.cos(t[:, None] * f)], -1)


class RawIn(nn.Module):
    def __init__(self, D, d, n=4 * K):
        super().__init__()
        self.rin = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, d))
        self.rpos = nn.Parameter(torch.randn(n, d) * 0.02)

    def forward(self, raw):
        return self.rin(raw) + self.rpos


class PlanEnc(nn.Module):
    def __init__(self, G, D, c, d=512, depth=4):
        super().__init__()
        self.G = G
        self.inp = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, d))
        self.pos = nn.Parameter(torch.randn(G * K, d) * 0.02)
        self.pad = nn.Parameter(torch.zeros(d))
        self.q = nn.Parameter(torch.randn(G, d) * 0.02)
        self.body = tblock(d, depth)
        self.down = nn.Linear(d, c)

    def forward(self, e, m):                    # e (B, G*K, D) head rows of the future tokens, m (B, G*K) in answer
        h = torch.where(m[:, :, None], self.inp(e), self.pad) + self.pos
        h = self.body(torch.cat([h, self.q.expand(len(e), -1, -1)], 1))[:, -self.G:]
        x = self.down(h).float()               # unit average power over the whole plan (no tanh: it saturated and collapsed)
        return x * torch.rsqrt(x.pow(2).mean((1, 2), keepdim=True) + 1e-6)


class Decoder(nn.Module):
    def __init__(self, G, c, D, d=512, depth=6, rel=False, omod=False):
        super().__init__()
        self.G, self.rel, self.omod = G, rel, omod
        self.up = nn.Linear(max(c, 1), d)
        self.gpos = nn.Parameter(torch.randn(G, d) * 0.02)
        self.oemb = nn.Embedding(G * K - K + 1, d)      # the drafted block can sit anywhere inside the plan
        self.raw = RawIn(D, d)
        self.q = nn.Parameter(torch.randn(K, d) * 0.02)
        self.body = tblock(d, depth)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, D))

    def forward(self, y, o, raw):               # y (B, G, c) plan, o (B,) offset, raw (B, 32, D)
        if self.rel:                            # shift the plan so the block holding the drafted tokens comes first
            idx = torch.arange(self.G, device=y.device)[None] + torch.div(o, K, rounding_mode="floor")[:, None]
            ok = (idx < self.G)[:, :, None].to(y.dtype)
            y = y.gather(1, idx.clamp(max=self.G - 1)[:, :, None].expand(-1, -1, y.shape[2])) * ok
        p = self.up(y) + self.gpos + self.oemb(o % K if self.omod else o)[:, None]
        h = torch.cat([p, self.raw(raw), self.q.expand(len(raw), -1, -1)], 1)
        return self.out(self.body(h)[:, -K:])


class PrevIn(nn.Module):
    """The plan of the 8G already verified tokens before the drafted block, as G extra tokens."""

    def __init__(self, G, c, d):
        super().__init__()
        self.pin = nn.Linear(c, d)
        self.ppos = nn.Parameter(torch.randn(G, d) * 0.02)

    def forward(self, prev):
        return self.pin(prev) + self.ppos


class PlanPred(nn.Module):
    def __init__(self, G, c, D, d=512, depth=4, hist=False):
        super().__init__()
        self.raw = RawIn(D, d)
        self.prev = PrevIn(G, c, d) if hist else None
        self.q = nn.Parameter(torch.randn(G, d) * 0.02)
        self.body = tblock(d, depth)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, c))

    def forward(self, raw, prev=None):
        h = torch.cat([self.raw(raw), self.q.expand(len(raw), -1, -1)], 1)
        if self.prev is not None:
            h = torch.cat([self.prev(prev), h], 1)
        return self.out(self.body(h)[:, -len(self.q):])


class PlanFlow(nn.Module):
    def __init__(self, G, c, D, d=512, depth=4, hist=False):
        super().__init__()
        self.raw = RawIn(D, d)
        self.prev = PrevIn(G, c, d) if hist else None
        self.xin = nn.Linear(c, d)
        self.gpos = nn.Parameter(torch.randn(G, d) * 0.02)
        self.te = nn.Sequential(nn.Linear(64, d), nn.SiLU(), nn.Linear(d, d))
        self.body = tblock(d, depth)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, c))
        self.G = G

    def forward(self, raw, x, t, prev=None):
        h = torch.cat([self.raw(raw), self.xin(x) + self.gpos + self.te(temb(t))[:, None]], 1)
        if self.prev is not None:
            h = torch.cat([self.prev(prev), h], 1)
        return self.out(self.body(h)[:, -self.G:])

    @torch.no_grad()
    def sample(self, raw, c, nfe, g, prev=None):
        x = torch.randn(len(raw), self.G, c, device=raw.device, generator=g)
        for i in range(nfe):
            t = torch.full((len(raw),), i / nfe, device=raw.device)
            x = x + self.forward(raw, x, t, prev).float() / nfe
        return x


def round_ok(seq, s, span):                     # rows s .. s+span-1 inside one answer
    n = len(seq)
    ok = s + span - 1 < n
    return ok & (seq[np.clip(s, 0, n - 1)] == seq[np.clip(s + span - 1, 0, n - 1)])


def pack(dt, T1, s, s0, G):
    seq, n = dt.seq, len(dt.seq)
    off = s0 != s
    x = dt.get(np.concatenate([s, s0[off]]))[0]
    raw = x[:len(s)]
    raw_a = raw.clone()
    if off.any():
        raw_a[torch.from_numpy(off)] = x[len(s):]
    rows = (s0 + K)[:, None] + np.arange(G * K)[None]
    rc = np.clip(rows, 0, n - 1)
    m = (rows < n) & (seq[rc] == seq[s0][:, None])
    lab = T1[(s + K)[:, None] + np.arange(K)[None]]
    pr = (s0 + K - G * K)[:, None] + np.arange(G * K)[None]       # the 8G tokens before the plan, ending at the bonus
    pc = np.clip(pr, 0, n - 1)
    pm = (pr >= 0) & (seq[pc] == seq[s0][:, None])
    return (raw, raw_a, torch.from_numpy(T1[rc]), torch.from_numpy(m), torch.from_numpy(lab), torch.from_numpy(s - s0),
            torch.from_numpy(T1[pc]), torch.from_numpy(pm))


def loader(dt, T1, starts, B, steps, G, poff, seed, oblock=False, ofix=False):
    q = queue.Queue(maxsize=6)
    rng = np.random.default_rng(seed)
    seq = dt.seq
    first = np.r_[0, np.nonzero(seq[1:] != seq[:-1])[0] + 1]        # first row of each answer
    room = np.arange(len(seq)) - first[np.searchsorted(first, np.arange(len(seq)), side="right") - 1]

    def work():
        for _ in range(steps):
            s = rng.choice(starts, B)
            if ofix:                            # only offsets that stay inside the answer (audit C2)
                hi = np.minimum(room[s], G * K - K)
                if oblock:
                    o = K * (rng.random(B) * (hi // K + 1)).astype(np.int64)
                else:
                    o = np.where((rng.random(B) < poff) & (hi >= 1), 1 + (rng.random(B) * np.maximum(hi, 1)).astype(np.int64), 0)
            elif oblock:                        # the drafted block equally often at each block of the plan
                o = K * rng.integers(0, G, B)
            else:
                o = np.where(rng.random(B) < poff, rng.integers(1, G * K - K + 1, B), 0)
            s0 = s - o
            bad = (s0 < 0) | (seq[np.clip(s0, 0, None)] != seq[s])
            s0[bad] = s[bad]
            q.put(pack(dt, T1, s, s0, G))
        q.put(None)
    threading.Thread(target=work, daemon=True).start()
    while True:
        b = q.get()
        if b is None:
            return
        yield b


def kbest(lp, Kb):                              # lp (B, K, V) per-slot log-probs -> (B, Kb, K) top sequences
    tv, ti = lp.topk(Kb, -1)                    # the Kb best of a product of slots only use each slot's top Kb
    sc, seqs = tv[:, 0], ti[:, 0, :, None]
    for j in range(1, K):
        tot = (sc[:, :, None] + tv[:, j, None, :]).flatten(1)
        sc, idx = tot.topk(Kb, -1)
        pa, ch = idx // Kb, idx % Kb
        seqs = torch.cat([seqs.gather(1, pa[:, :, None].expand(-1, -1, seqs.shape[2])),
                          ti[:, j].gather(1, ch)[:, :, None]], 2)
    return seqs


def tree_stats(tok, lab):                       # tok (B, Kb, K), lab (B, K) -> best accepted, mean, distinct, nodes
    acc = (tok == lab[:, None]).int().cumprod(2).sum(2) + 1
    tl = tok.tolist()
    nodes = [len({tuple(d[:j]) for d in r for j in range(1, K + 1)}) for r in tl]
    distinct = [len({tuple(d) for d in r}) for r in tl]
    return acc.max(1).values.float(), acc.float().mean(1), torch.tensor(distinct).float(), torch.tensor(nodes).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--G", type=int, default=8, help="plan blocks of 8 tokens (8 = 64 tokens)")
    ap.add_argument("--c", type=int, default=4, help="numbers per block latent; 0 = control with no plan")
    ap.add_argument("--sigma", type=float, default=0.5, help="channel noise on the unit-power plan")
    ap.add_argument("--warm", type=int, default=3000, help="steps over which channel noise rises from 0.1 sigma to sigma")
    ap.add_argument("--ofix", action="store_true",
                    help="sample only offsets inside the answer; with --rel the decoder sees only o mod 8")
    ap.add_argument("--rel", action="store_true", help="decoder reads the plan shifted to the drafted block")
    ap.add_argument("--oblock", action="store_true", help="train offsets uniform over the plan's blocks (0, 8, ..)")
    ap.add_argument("--phist", action="store_true",
                    help="predictor and flow also read the plan of the previous 8G verified tokens (gist to next gist)")
    ap.add_argument("--raw_drop", type=float, default=0.0, help="fraction of training rounds with the raw states zeroed")
    ap.add_argument("--e_pred", type=float, default=0.0,
                    help="weight pulling the plan towards what P predicts (the encoder learns a predictable plan)")
    ap.add_argument("--poff", type=float, default=0.5, help="fraction of training rounds with a stale plan (o > 0)")
    ap.add_argument("--train_cache", default="cache_pb")
    ap.add_argument("--dev_cache", default="cache")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--tail", type=int, default=2000, help="extra steps training only P and Flow on the frozen plan")
    ap.add_argument("--B", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--nfe", type=int, default=8)
    ap.add_argument("--bestk", default="4,8")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.tail, args.B = 30, 10, 16
    dev = "cuda"
    G, c = args.G, args.c
    bits = G * c * 0.5 * np.log2(1 + 1 / args.sigma ** 2)
    tag = "pw_" + f"G{G}c{c}" + (f"s{args.sigma:g}" if c else "") + (f"_ep{args.e_pred:g}" if args.e_pred else "") + \
          (f"_poff{args.poff:g}" if args.poff != 0.5 else "") + (f"_rd{args.raw_drop:g}" if args.raw_drop else "") + ("_ob" if args.oblock else "") + ("_rel" if args.rel else "") + ("_of" if args.ofix else "") + ("_ph" if args.phist else "") + (f"_st{args.steps}" if args.steps != 10000 else "") + \
          ("_smoke" if args.smoke else "")
    print(f"=== plan {tag}: plan {G * K} tokens, {G} x {c}, channel bound {bits:.1f} bits", flush=True)

    dev_cache = os.path.join(DATA, args.dev_cache)
    stats = load_stats(dev_cache)
    W = torch.load(os.path.join(dev_cache, "lm_head.pt")).to(dev)
    D = W.shape[1]
    fmean, fstd = stats[-1][0].to(dev), stats[-1][1].to(dev)

    def logp(h):                                # decoder output (z-scored final-layer state) -> log-probs
        return ((h.float() * fstd + fmean).to(torch.bfloat16) @ W.T).float().log_softmax(-1)

    def split(cache, sp):
        dt = Data(os.path.join(DATA, cache), sp, LAYERS, K, stats)
        T1 = np.load(os.path.join(DATA, cache, f"{sp}_meta.npz"))["top1"].astype(np.int64)
        st = np.arange(len(dt.seq))
        return dt, T1, st[round_ok(dt.seq, st, 2 * K)]

    t0 = time.time()
    tr, trT1, trst = split(args.train_cache, "train")
    print(f"train rounds {len(trst)}", flush=True)

    E = PlanEnc(G, D, c).to(dev) if c else None
    dec = Decoder(G, c, D, rel=args.rel, omod=args.ofix and args.rel).to(dev)
    P = PlanPred(G, c, D, hist=args.phist).to(dev) if c else None
    fl = PlanFlow(G, c, D, hist=args.phist).to(dev) if c else None
    count = lambda m: sum(p.numel() for p in m.parameters()) / 1e6 if m is not None else 0
    print(f"params E {count(E):.1f}M, decoder {count(dec):.1f}M, P {count(P):.1f}M, flow {count(fl):.1f}M", flush=True)
    main_p = list(dec.parameters()) + (list(E.parameters()) if E is not None else [])
    pf_p = (list(P.parameters()) + list(fl.parameters())) if c else []
    total = args.steps + (args.tail if c else 0)

    def cos(n):
        return lambda s: min(1, (s + 1) / 500) * max(0.1, 0.5 * (1 + np.cos(np.pi * min(1, s / max(1, n)))))
    opt_m = torch.optim.AdamW(main_p, lr=args.lr, weight_decay=0.01)
    sch_m = torch.optim.lr_scheduler.LambdaLR(opt_m, cos(args.steps))
    if c:
        opt_p = torch.optim.AdamW(pf_p, lr=args.lr, weight_decay=0.01)
        sch_p = torch.optim.lr_scheduler.LambdaLR(opt_p, cos(total))

    def plan(ids, m):
        return E(W[ids], m).float()

    log = []
    for step, b in enumerate(loader(tr, trT1, trst, args.B, total, G, args.poff, 0, args.oblock, args.ofix)):
        raw, raw_a, ids, m, lab, o, pids, pm = [x.to(dev, non_blocking=True) for x in b]
        tail = step >= args.steps
        nll = pl = fll = torch.zeros((), device=dev)
        with torch.autocast("cuda", torch.bfloat16):
            if c:
                with torch.set_grad_enabled(not tail):
                    mu = plan(ids, m)
                sig = args.sigma * min(1.0, 0.1 + 0.9 * step / max(1, args.warm))
                y = mu + sig * torch.randn_like(mu)
            else:
                y = torch.zeros(len(raw), G, 1, device=dev)
            if not tail:
                rin = raw
                if args.raw_drop:
                    rin = raw * (torch.rand(len(raw), 1, 1, device=dev) >= args.raw_drop)
                nll = F.nll_loss(logp(dec(y, o, rin)).flatten(0, 1), lab.flatten())
            if c:
                tgt = mu.detach()
                prev = None
                if args.phist:
                    with torch.no_grad():
                        prev = plan(pids, pm)
                ph = P(raw_a, prev).float()
                pl = F.mse_loss(ph, tgt)
                x0 = torch.randn_like(tgt)
                t = torch.rand(len(tgt), device=dev)
                xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * tgt
                fll = F.mse_loss(fl(raw_a, xt, t, prev).float(), tgt - x0)
        loss = nll + pl + fll
        if c and args.e_pred and not tail:
            loss = loss + args.e_pred * F.mse_loss(mu, ph.detach())
        opt_m.zero_grad(set_to_none=True)
        if c:
            opt_p.zero_grad(set_to_none=True)
        loss.backward()
        if not tail:
            torch.nn.utils.clip_grad_norm_(main_p, 1.0)
            opt_m.step()
            sch_m.step()
        if c:
            torch.nn.utils.clip_grad_norm_(pf_p, 1.0)
            opt_p.step()
            sch_p.step()
        if step % 500 == 0 or step == total - 1:
            r = {"step": step, "nll": nll.item(), "p_mse": pl.item(), "flow": fll.item(),
                 "mu_std": mu.std(0).mean().item() if c else 0, "min": (time.time() - t0) / 60}
            log.append(r)
            print(json.dumps(r), flush=True)
            if not np.isfinite(loss.item()):
                raise SystemExit("non-finite loss")
    if not args.smoke:
        torch.save({"E": E.state_dict() if E else None, "dec": dec.state_dict(), "P": P.state_dict() if P else None,
                    "flow": fl.state_dict() if fl else None, "args": vars(args)}, os.path.join(CKPT, f"plan_{tag}.pt"))

    for mod in [E, dec, P, fl]:
        if mod is not None:
            mod.eval()
    res = {"args": vars(args), "bits_bound": bits, "train_log": log, "sets": {}}
    g = torch.Generator(device=dev).manual_seed(0)
    bk = [int(x) for x in args.bestk.split(",")]
    for sp in ["dev_math", "dev_code", "dev_chat"]:
        dt, T1, st = split(args.dev_cache, sp)
        keep, last = [], -10 ** 9
        for s in st:                            # non-overlapping rounds
            if s >= last + 2 * K:
                keep.append(s)
                last = s
        rounds = np.array(keep)[:256] if args.smoke else np.array(keep)
        r = {"rounds": int(len(rounds))}
        for o in [x for x in [0, 16, 32, 48, 128, 248, 504] if x <= G * K - K]:
            s0 = rounds - o
            okk = (s0 >= 0) & (dt.seq[np.clip(s0, 0, None)] == dt.seq[rounds])
            s, s0 = rounds[okk], s0[okk]
            out = {}
            ex = {}
            with torch.no_grad():
                for i in range(0, len(s), 64):
                    raw, raw_a, ids, m, lab, oo, pids, pm = [x.to(dev) for x in pack(dt, T1, s[i:i + 64], s0[i:i + 64], G)]
                    B = len(raw)
                    noise = lambda: args.sigma * torch.randn(B, G, max(c, 1), device=dev, generator=g)
                    with torch.autocast("cuda", torch.bfloat16):
                        ys = {"zero": torch.zeros(B, G, max(c, 1), device=dev)}
                        if c:
                            mu = plan(ids, m)
                            prev = plan(pids, pm) if args.phist else None
                            ph = P(raw_a, prev).float()
                            fs = fl.sample(raw_a, c, args.nfe, g, prev)
                            ys.update({"oracle": mu + noise(), "oracle_clean": mu, "det": ph + noise(),
                                       "det_clean": ph, "flow": fs + noise()})
                        for kind, y in ys.items():
                            lp = logp(dec(y, oo, raw))
                            tok = lp.argmax(-1)
                            out.setdefault(kind, []).append(((tok == lab).int().cumprod(1).sum(1) + 1).float().cpu())
                            if o == 0 and kind == ("det_clean" if c else "zero"):
                                for Kb in bk:  # per-position baseline: K most likely drafts of this decoder
                                    st_ = tree_stats(kbest(lp, Kb).cpu(), lab.cpu())
                                    for nm, v in zip(["best", "mean", "distinct", "nodes"], st_):
                                        ex.setdefault(f"kbest{Kb}_{nm}", []).append(v)
                        if c and o == 0:
                            for Kb in bk:      # K sampled plans -> K drafts
                                ysk = fl.sample(raw_a.repeat_interleave(Kb, 0), c, args.nfe, g,
                                                prev.repeat_interleave(Kb, 0) if prev is not None else None)
                                ysk = ysk + args.sigma * torch.randn(ysk.shape, device=dev, generator=g)
                                hk = dec(ysk, oo.repeat_interleave(Kb, 0), raw.repeat_interleave(Kb, 0))
                                tk = torch.cat([logp(x).argmax(-1) for x in hk.split(128)])
                                st_ = tree_stats(tk.view(B, Kb, K).cpu(), lab.cpu())
                                for nm, v in zip(["best", "mean", "distinct", "nodes"], st_):
                                    ex.setdefault(f"flow{Kb}_{nm}", []).append(v)
            sfx = "" if o == 0 else f"_o{o}"
            r.update({f"accepted_{kk}{sfx}": torch.cat(v).mean().item() for kk, v in out.items()})
            r.update({f"{kk}": torch.cat(v).mean().item() for kk, v in ex.items()})
            r[f"rounds{sfx}"] = int(len(s))
        res["sets"][sp] = r
        print(sp, json.dumps(r), flush=True)
    json.dump(res, open(os.path.join(RES, f"plan_{tag}{'' if not args.smoke else ''}.json"), "w"), indent=1)
    print(f"=== plan {tag} done {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
