"""Training for Arm C (soft-input target) and Arm D (parallel block drafter).

Both arms use the same trick. The trainable parameters (gated LoRA, e_flag, e_mask) only act on
"block" rows, and block rows attend to a context KV that the frozen base model computes on the
true sequence. So one no-grad base pass per sequence gives the context KV for every split point,
and many blocks per sequence are packed into one forward pass with a custom attention mask.

Arm C (--arm C): the target with S soft slots at positions p0..p0+S-1 after a true hard prefix.
  Slot i holds a guess distribution q for token x_{p0+i}: input E^T q + e_flag (C) or
  E[argmax q] + e_flag (C-hard, --inp hard). Fresh slots get e_mask. Guesses are drawn from a
  mixture: the base predictive distribution at a random temperature, a noisy one-hot of the true
  token, and a misaligned (shifted) base distribution, which mimics stale guesses. With prob
  --rollout the model's own outputs from a first no-grad pass are fed back (as in deployment).
  Loss: KL(p_base(. | true prefix) || p_model(. | soft inputs)) at every slot.

Arm D (--arm D): the 0.5B drafter as a parallel block drafter. Slot 0 = x_t (true token), slots
  1..B-1 = e_mask in pass 1. In pass 2 slot i gets pass 1's distribution for x_{t+i}, as
  E^T q + e_flag (soft) or E[argmax q] + e_flag (hard). Loss = loss(pass 1) + loss(pass 2) with
  label variant D1 (CE on target-generated tokens), D2 (KL to the frozen drafter's own AR
  distribution), D3 (forward KL to the target) or D4 (TV to the target).

Run: python train.py --arm C --inp soft --steps 2000 --name c_soft
     python train.py --arm D --label d3 --inp soft --steps 2000 --name d3_soft
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from common import CKPT, DATA
from model import DRAFTER, TARGET, QwenLM

K = 64  # top-k used for every soft input (matches decode.TOPK)
PAD = 151643


def batch_iter(items, n, seed, max_len=512):
    rng = random.Random(seed)
    while True:
        b = rng.sample(items, n)
        seqs = [(it["prompt"] + it["resp"])[:max_len] for it in b]
        rs = [len(it["prompt"]) for it in b]
        L = max(len(s) for s in seqs)
        ids = torch.full((n, L), PAD, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s)
        yield ids.cuda(), torch.tensor([len(s) for s in seqs]), torch.tensor(rs)


def sample_starts(lens, rs, nb, lo_off, rng):
    """nb block starts per sequence inside the response region."""
    st = []
    for L, r in zip(lens.tolist(), rs.tolist()):
        a, b = r + lo_off, max(r + lo_off, L - 2)
        st.append([rng.randint(a, b) for _ in range(nb)])
    return torch.tensor(st, device="cuda")  # (N, nb)


def pack_mask(starts, blen, Lc):
    """Bool mask (N, 1, M, Lc+M): block row (b, i) sees context j < start_b and rows (b, i' <= i)."""
    N, nb = starts.shape
    M = nb * blen
    blk = torch.arange(M, device="cuda") // blen
    idx = torch.arange(M, device="cuda") % blen
    ctx = torch.arange(Lc, device="cuda")[None, None, :] < starts[:, blk][:, :, None]  # (N, M, Lc)
    own = (blk[:, None] == blk[None, :]) & (idx[None, :] <= idx[:, None])  # (M, M)
    return torch.cat([ctx, own[None].expand(N, -1, -1)], -1)[:, None]


def gather_rows(x, pos):
    """x (N, L, ...) gathered at pos (N, M), clamped to valid range."""
    pos = pos.clamp(0, x.shape[1] - 1)
    return torch.gather(x, 1, pos.view(*pos.shape, *([1] * (x.dim() - 2))).expand(*pos.shape, *x.shape[2:]))


def chunked(fn, *xs, chunk=256):
    outs = [fn(*[x[i:i + chunk] for x in xs]) for i in range(0, xs[0].shape[0], chunk)]
    return torch.cat(outs) if outs[0].dim() else sum(outs)


def loss_rows(model, hn, lab, kind, tok=None, valid=None, chunk=192):
    """Mean loss over rows. hn: model final hidden (R, d) with grad. lab: label logits source:
    for kl/tv, a function rows->label logits (no grad); for ce, token ids (R,)."""
    R = hn.shape[0]
    total = hn.new_zeros((), dtype=torch.float32)
    for i in range(0, R, chunk):
        sl = slice(i, i + chunk)

        def f(h, idx):
            lg = model.logits(h)
            lp = F.log_softmax(lg, -1)
            if kind == "ce":
                l = -lp.gather(-1, tok[idx][:, None])[:, 0]
            else:
                with torch.no_grad():
                    pl = F.log_softmax(lab(idx), -1)
                if kind == "kl":
                    l = (pl.exp() * (pl - lp)).sum(-1)
                else:  # tv
                    l = 0.5 * (pl.exp() - lp.exp()).abs().sum(-1)
            return (l * valid[idx]).sum()

        idx = torch.arange(i, min(R, i + chunk), device="cuda")
        total = total + torch.utils.checkpoint.checkpoint(f, hn[sl], idx, use_reentrant=False)
    return total / valid.sum().clamp_min(1)


def topk_probs(logits, temp=1.0):
    p = torch.softmax(logits.float() / temp, -1)
    v, i = p.topk(K, -1)
    return i, v


# ------------------------------- Arm C ---------------------------------------------------------
def c_guesses(m, hn_base, ids, starts, S, rng):
    """Guess distributions (ids, probs) of shape (N, M, K) for slots at positions starts+i, plus a
    fresh mask (N, M) for slots with no guess."""
    N, nb = starts.shape
    M = nb * S
    pos = (starts[:, :, None] + torch.arange(S, device="cuda")).view(N, M)
    with torch.no_grad():
        # (a) base predictive dist for x_p = logits at p-1, random temperature per block
        temps = torch.empty(N, nb, 1, device="cuda").uniform_(0.7, 1.5).expand(N, nb, S).reshape(N, M, 1)
        h_prev = gather_rows(hn_base, pos - 1)
        ia, pa = [], []
        for n in range(N):
            lg = m.logits(h_prev[n]) / temps[n]
            i_, p_ = topk_probs(lg)
            ia.append(i_), pa.append(p_)
        ia, pa = torch.stack(ia), torch.stack(pa)
        # (c) misaligned: base dist of a neighbouring position (p-2 or p), mimics stale/shifted guesses
        shift = torch.where(torch.rand(N, M, device="cuda") < 0.5, -2, 0)
        h_sh = gather_rows(hn_base, pos + shift)
        ic, pc = [], []
        for n in range(N):
            i_, p_ = topk_probs(m.logits(h_sh[n]))
            ic.append(i_), pc.append(p_)
        ic, pc = torch.stack(ic), torch.stack(pc)
        # (b) noisy one-hot of the true token
        x = gather_rows(ids, pos)
        lam = torch.rand(N, M, 1, device="cuda")
        noise_i = torch.randint(0, m.V, (N, M, K - 1), device="cuda")
        noise_p = torch.softmax(torch.randn(N, M, K - 1, device="cuda") * 2, -1)
        ib = torch.cat([x[..., None], noise_i], -1)
        pb = torch.cat([lam, (1 - lam) * noise_p], -1)
        typ = torch.multinomial(torch.tensor([0.35, 0.35, 0.30], device="cuda"), N * M, replacement=True).view(N, M, 1)
        gi = torch.where(typ == 0, ia, torch.where(typ == 1, ib, ic))
        gp = torch.where(typ == 0, pa, torch.where(typ == 1, pb, pc))
        # fresh tail: slots >= f are fresh, f ~ U{0..S} per block
        f = torch.randint(0, S + 1, (N, nb, 1), device="cuda")
        fresh = (torch.arange(S, device="cuda")[None, None] >= f).view(N, M)
    return pos, gi, gp, fresh


def c_embed(m, gi, gp, fresh, inp):
    if inp == "soft":
        e = m.embed_soft(gi, gp)
    else:
        e = m.rows32(gi[..., 0])
    e = torch.where(fresh[..., None], m.e_mask, e)
    return e + m.e_flag


def train_c(args, items):
    m = QwenLM(TARGET)
    m.use_graphs = False
    m.add_lora(args.r, args.alpha)
    params = [p for p in m.parameters() if p.requires_grad]
    S, nb, N = args.S, args.nb, args.bs
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1, s / args.steps))))
    it = batch_iter(items, N, args.seed)
    log = []
    t0 = time.time()
    for step in range(args.steps):
        ids, lens, rs = next(it)
        with torch.no_grad():
            hn_base, kvs = m.full(ids, collect_kv=True)
        starts = sample_starts(lens, rs, nb, 1, rng)
        pos, gi, gp, fresh = c_guesses(m, hn_base, ids, starts, S, rng)
        mask = pack_mask(starts, S, ids.shape[1])
        valid = (pos <= (lens.cuda()[:, None] - 1)).float()
        if rng.random() < args.rollout:
            # Deployment (detached tail): a round's soft inputs are the previous round's soft outputs,
            # computed when the committed prefix was s = a+1 tokens shorter. Mimic it: pass 1 (no grad)
            # runs the block s positions earlier with mixture guesses; pass-2 slot j takes pass-1 row
            # j+s-1 (which predicted position starts+j); slots beyond pass-1 coverage are fresh.
            s = rng.randint(1, 4)
            st1 = starts - s
            pos1, gi1, gp1, fresh1 = c_guesses(m, hn_base, ids, st1, S, rng)
            with torch.no_grad():
                h1 = m.packed(c_embed(m, gi1, gp1, fresh1, args.inp), pos1, kvs, pack_mask(st1, S, ids.shape[1]), ckpt=False)
                i1, p1 = [], []
                for n in range(N):
                    i_, p_ = topk_probs(m.logits(h1[n]))
                    i1.append(i_), p1.append(p_)
                i1 = torch.stack(i1).view(N, nb, S, K)
                p1 = torch.stack(p1).view(N, nb, S, K)
                j = torch.arange(S, device="cuda")
                src = (j + s - 1).clamp_max(S - 1)
                gi = i1[:, :, src].reshape(N, nb * S, K)
                gp = p1[:, :, src].reshape(N, nb * S, K)
                fresh = (j + s - 1 >= S)[None, None].expand(N, nb, S).reshape(N, nb * S)
        emb = c_embed(m, gi, gp, fresh, args.inp)
        hn = m.packed(emb, pos, kvs, mask, ckpt=True)
        lab_h = gather_rows(hn_base, pos).reshape(-1, m.d)
        loss = loss_rows(m, hn.reshape(-1, m.d), lambda idx: m.logits(lab_h[idx]), "kl", valid=valid.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        log.append(loss.item())
        if step % 25 == 0 or step == args.steps - 1:
            el = time.time() - t0
            print(f"step {step} loss {sum(log[-25:]) / len(log[-25:]):.4f}  {el / 60:.1f} min, eta {el / (step + 1) * (args.steps - step - 1) / 60:.1f} min  mem {torch.cuda.max_memory_allocated() / 1e9:.1f}GB", flush=True)
    return m, log


# ------------------------------- Arm D ---------------------------------------------------------
def train_d(args, items):
    d = QwenLM(DRAFTER)
    d.use_graphs = False
    d.add_lora(args.r, args.alpha)
    t = QwenLM(TARGET) if args.label in ("d3", "d4") else None
    if t is not None:
        t.use_graphs = False
    params = [p for p in d.parameters() if p.requires_grad]
    B, nb, N = args.B, args.nb, args.bs
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1, s / args.steps))))
    it = batch_iter(items, N, args.seed)
    kind = {"d1": "ce", "d2": "kl", "d3": "kl", "d4": "tv"}[args.label]
    log = []
    t0 = time.time()
    M = nb * B
    for step in range(args.steps):
        ids, lens, rs = next(it)
        with torch.no_grad():
            hn_d, kvs = d.full(ids, collect_kv=True)
            hn_t = t.full(ids)[0] if t is not None else None
        starts = sample_starts(lens, rs, nb, 0, rng)  # slot 0 = x_t at position t >= response start
        pos = (starts[:, :, None] + torch.arange(B, device="cuda")).view(N, M)
        mask = pack_mask(starts, B, ids.shape[1])
        valid = (pos + 1 <= (lens.cuda()[:, None] - 1)).float().reshape(-1)
        nxt = gather_rows(ids, pos + 1).reshape(-1)
        if args.label == "d2":
            lab_h, lab_m = gather_rows(hn_d, pos).reshape(-1, d.d), d
        elif t is not None:
            lab_h, lab_m = gather_rows(hn_t, pos).reshape(-1, t.d), t
        else:
            lab_h, lab_m = None, None
        lab = (lambda idx: lab_m.logits(lab_h[idx])) if lab_h is not None else None
        x0 = d.rows32(gather_rows(ids, starts))  # (N, nb, d)
        first = (torch.arange(M, device="cuda") % B == 0)[None, :, None]
        e1 = torch.where(first, x0.repeat_interleave(B, 1), d.e_mask.expand(N, M, -1))
        h1 = d.packed(e1, pos, kvs, mask, ckpt=True)
        loss1 = loss_rows(d, h1.reshape(-1, d.d), lab, kind, tok=nxt, valid=valid)
        with torch.no_grad():
            lg1 = []
            for n in range(N):
                lg1.append(d.logits(h1[n].detach()))
            lg1 = torch.stack(lg1)  # (N, M, V): row i predicts x_{t+i+1} -> input for slot i+1
            if args.inp == "soft":
                gi, gp = topk_probs(lg1)
                g = d.embed_soft(gi, gp)
            else:
                g = d.rows32(lg1.argmax(-1))
            del lg1
        e2 = torch.where(first, x0.repeat_interleave(B, 1), torch.roll(g, 1, dims=1) + d.e_flag)
        h2 = d.packed(e2, pos, kvs, mask, ckpt=True)
        loss2 = loss_rows(d, h2.reshape(-1, d.d), lab, kind, tok=nxt, valid=valid)
        loss = loss1 + loss2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        log.append((loss1.item(), loss2.item()))
        if step % 25 == 0 or step == args.steps - 1:
            el = time.time() - t0
            l1 = sum(x[0] for x in log[-25:]) / len(log[-25:])
            l2 = sum(x[1] for x in log[-25:]) / len(log[-25:])
            print(f"step {step} loss1 {l1:.4f} loss2 {l2:.4f}  {el / 60:.1f} min, eta {el / (step + 1) * (args.steps - step - 1) / 60:.1f} min  mem {torch.cuda.max_memory_allocated() / 1e9:.1f}GB", flush=True)
    return d, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["C", "D"], required=True)
    ap.add_argument("--inp", choices=["soft", "hard"], default="soft")
    ap.add_argument("--label", choices=["d1", "d2", "d3", "d4"], default="d3")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--nb", type=int, default=8, help="blocks per sequence")
    ap.add_argument("--S", type=int, default=16, help="Arm C soft slots per block (covers S<=16 at test time)")
    ap.add_argument("--B", type=int, default=16, help="Arm D block length (covers B<=16 at test time)")
    ap.add_argument("--rollout", type=float, default=0.5)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="train")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    items = torch.load(os.path.join(DATA, args.data + ".pt"))
    items = [it for it in items if len(it["resp"]) >= 4]
    print(f"{len(items)} training sequences", flush=True)
    m, log = (train_c if args.arm == "C" else train_d)(args, items)
    sd = m.trainable_state()
    sd["__meta__"] = {"r": args.r, "alpha": args.alpha, **vars(args)}
    torch.save(sd, os.path.join(CKPT, args.name + ".pt"))
    json.dump({"args": vars(args), "loss": log}, open(os.path.join(CKPT, args.name + "_log.json"), "w"))
    print("saved", os.path.join(CKPT, args.name + ".pt"))


if __name__ == "__main__":
    main()
