"""Decoding loops, batch size 1. All share `verify`, the standard speculative verification step.

Conventions
- `seq` is prompt + committed tokens. The target cache always holds KV for seq[:-1]; the last
  committed token x_last is fed at slot 0 of the next chunk, so slot j's output is the target
  distribution for the token after slot j.
- temp == 0 means greedy. temp > 0 means sampling from softmax(logits / temp) with standard
  rejection sampling, which keeps the output distribution exactly the target's.
- Guess distributions are stored as top-k (ids, probs). For sampling, drafts are drawn from the
  renormalised top-k distribution and that same distribution is used as q in verification.
"""
import time

import torch

from model import EOS_IDS

TOPK = 64


class Stats:
    def __init__(self, maxoff=40):
        self.rounds = 0  # target decode passes (prefill excluded)
        self.committed = 0  # tokens committed by decode passes
        self.dpasses = 0  # drafter forward passes
        self.att = [0] * maxoff  # offset k attempted (all earlier drafts accepted)
        self.acc = [0] * maxoff  # offset k accepted

    def add(self, n_drafts, a, n_commit):
        self.rounds += 1
        self.committed += n_commit
        for k in range(min(n_drafts, a + 1)):
            self.att[k] += 1
            if k < a:
                self.acc[k] += 1

    def merge(self, o):
        self.rounds += o.rounds
        self.committed += o.committed
        self.dpasses += o.dpasses
        self.att = [x + y for x, y in zip(self.att, o.att)]
        self.acc = [x + y for x, y in zip(self.acc, o.acc)]

    def tau(self):
        return self.committed / max(1, self.rounds)

    def per_pos(self, n):
        cond = [self.acc[k] / self.att[k] if self.att[k] else float("nan") for k in range(n)]
        uncond = [self.acc[k] / max(1, self.rounds) for k in range(n)]
        return cond, uncond


def to_topk(logits, k=TOPK):
    p = torch.softmax(logits.float(), -1)
    v, i = p.topk(k, -1)
    return i, v


def dense(ids, probs, V):
    q = torch.zeros(V, device=probs.device)
    q.scatter_(0, ids, probs / probs.sum())
    return q


def verify(p_logits, drafts, qs, temp, gen):
    """p_logits (n+1, V): target logits at slots 0..n; drafts (n,) ids; qs: n dense draft dists (sampling only).
    Returns (a, toks): number of accepted drafts and the a+1 committed tokens."""
    n = drafts.shape[0]
    if temp == 0:
        tgt = p_logits.argmax(-1)
        if n == 0:
            return 0, [int(tgt[0])]
        a = int((drafts == tgt[:n]).int().cumprod(0).sum())
        return a, drafts[:a].tolist() + [int(tgt[a])]
    p = torch.softmax(p_logits.float() / temp, -1)
    if n == 0:
        return 0, [int(torch.multinomial(p[0], 1, generator=gen))]
    Q = torch.stack(qs)
    ar = torch.arange(n, device=p.device)
    px, qx = p[ar, drafts], Q[ar, drafts]
    u = torch.rand(n, device=p.device, generator=gen)
    acc = (u * qx <= px).int()
    a = int(acc.cumprod(0).sum())
    if a < n:
        r = (p[a] - Q[a]).clamp_min(0)
        s = r.sum()
        r = r / s if s > 0 else p[a]
        tok = int(torch.multinomial(r, 1, generator=gen))
    else:
        tok = int(torch.multinomial(p[n], 1, generator=gen))
    return a, drafts[:a].tolist() + [tok]


def _pick(logits, temp, gen):
    if temp == 0:
        return int(logits.argmax())
    return int(torch.multinomial(torch.softmax(logits.float() / temp, -1), 1, generator=gen))


class Run:
    """Holds the committed sequence and stop logic."""

    def __init__(self, prompt, max_new):
        self.seq = list(prompt)
        self.out = []
        self.max_new = max_new
        self.done = False

    def commit(self, toks):
        n = 0
        for t in toks:
            self.seq.append(t)
            self.out.append(t)
            n += 1
            if t in EOS_IDS or len(self.out) >= self.max_new:
                self.done = True
                break
        return n


def _prefill(m, prompt, extra):
    """Eager prefill of the target's persistent cache (no graph: prompt lengths vary)."""
    c = m.get_cache()
    assert len(prompt) + extra <= c.limit
    lg = m.step(c, torch.tensor(prompt, device=m.dev), lo=-1, graph=False)
    c.n = len(prompt)
    return c, lg[-1]


def _prefill_draft(dm, prompt):
    """Drafter cache holds prompt[:-1]; the rest is fed as pending tokens in the first round."""
    c = dm.get_cache()
    if len(prompt) > 1:
        dm.step(c, torch.tensor(prompt[:-1], device=dm.dev), lo=-1, graph=False)
    c.n = len(prompt) - 1
    return c


def _timed(fn):
    def wrap(*a, **k):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out, st = fn(*a, **k)
        torch.cuda.synchronize()
        return {"tokens": out, "time": time.perf_counter() - t0, "stats": st}
    return wrap


@_timed
@torch.no_grad()
def gen_ar(tm, prompt, max_new, temp=0, gen=None, record_gap=False):
    """Plain autoregressive decoding. With record_gap, st.gaps[i] is the top1-top2 logit gap at step i
    (used to tell numerical near-ties from real mismatches in the exactness check)."""
    st = Stats()
    st.gaps = []
    tc, lg = _prefill(tm, prompt, max_new + 8)
    run = Run(prompt, max_new)

    def pick(l):
        if record_gap:
            t2 = l.topk(2).values
            st.gaps.append(float(t2[0] - t2[1]))
        return _pick(l, temp, gen)

    run.commit([pick(lg)])
    while not run.done:
        lg = tm.step(tc, torch.tensor([run.seq[-1]], device=tm.dev))
        tc.n += 1
        st.add(0, 0, run.commit([pick(lg[-1])]))
    return run.out, st


@_timed
@torch.no_grad()
def gen_sd(tm, dm, prompt, max_new, gamma=5, temp=0, gen=None):
    """Classic speculative decoding: the small model drafts gamma tokens autoregressively."""
    st = Stats()
    tc, lg = _prefill(tm, prompt, max_new + gamma + 8)
    dc = _prefill_draft(dm, prompt)
    run = Run(prompt, max_new)
    run.commit([_pick(lg, temp, gen)])
    while not run.done:
        Ls = len(run.seq)
        pend = run.seq[dc.n:]
        dl = dm.step(dc, torch.tensor(pend, device=dm.dev), lo=-1)
        dc.n += len(pend)
        st.dpasses += 1
        drafts, qs = [], []
        for k in range(gamma):
            if temp == 0:
                d = dl[-1].argmax()
            else:
                q = torch.softmax(dl[-1].float() / temp, -1)
                d = torch.multinomial(q, 1, generator=gen)[0]
                qs.append(q)
            drafts.append(d)
            if k < gamma - 1:
                dl = dm.step(dc, d.view(1))
                dc.n += 1
                st.dpasses += 1
        drafts = torch.stack(drafts)
        x = torch.tensor([run.seq[-1]], device=tm.dev)
        tl = tm.step(tc, torch.cat([x, drafts]))
        a, toks = verify(tl, drafts, qs, temp, gen)
        tc.n += 1 + a
        dc.n = Ls + min(a, gamma - 1)
        st.add(gamma, a, run.commit(toks))
    return run.out, st


@torch.no_grad()
def block_propose(dm, dc, seq, B, iters=2, inp="soft"):
    """Parallel block drafter (Arm D). Returns logits (B, V); row i predicts the token at offset i+1
    after x_last. Iteration 1 uses the learned mask embedding at offsets 1..B-1; later iterations
    feed back the previous iteration's distributions (soft: E^T q, hard: E[argmax q]) plus e_flag."""
    pend = seq[dc.n:]
    ctx, x = pend[:-1], pend[-1]
    ex = dm.embed_ids(torch.tensor([x], device=dm.dev))
    parts = [ex, dm.e_mask.to(ex.dtype).expand(B - 1, -1)]
    if ctx:
        parts.insert(0, dm.embed_ids(torch.tensor(ctx, device=dm.dev)))
    lg = dm.step(dc, embeds=torch.cat(parts), gate=len(ctx), lo=len(ctx))
    dc.n += len(ctx)
    for _ in range(1, iters):
        if inp == "soft":
            ids, p = to_topk(lg[: B - 1])
            rows = dm.embed_soft(ids, p) + dm.e_flag
        else:
            rows = dm.rows32(lg[: B - 1].argmax(-1)) + dm.e_flag
        lg = dm.step(dc, embeds=torch.cat([ex.float(), rows]), gate=0)
    return lg


@_timed
@torch.no_grad()
def gen_block(tm, dm, prompt, max_new, B=8, iters=2, inp="soft", temp=0, gen=None):
    """Speculative decoding with the parallel block drafter (Arm D). Target is frozen."""
    st = Stats()
    tc, lg = _prefill(tm, prompt, max_new + B + 8)
    dc = _prefill_draft(dm, prompt)
    run = Run(prompt, max_new)
    run.commit([_pick(lg, temp, gen)])
    while not run.done:
        dl = block_propose(dm, dc, run.seq, B, iters, inp)
        st.dpasses += iters
        if temp == 0:
            drafts, qs = dl.argmax(-1), []
        else:
            Q = torch.softmax(dl.float() / temp, -1)
            drafts = torch.multinomial(Q, 1, generator=gen)[:, 0]
            qs = list(Q)
        x = torch.tensor([run.seq[-1]], device=tm.dev)
        tl = tm.step(tc, torch.cat([x, drafts]))
        a, toks = verify(tl, drafts, qs, temp, gen)
        tc.n += 1 + a
        st.add(B, a, run.commit(toks))
    return run.out, st


@_timed
@torch.no_grad()
def gen_softtail(tm, prompt, max_new, H=4, S=8, temp=0, inp="soft", lora=False, flag=False,
                 stale="reuse", sharpen=1.0, dm=None, d_iters=2, d_inp="soft", tail="attached", gen=None):
    """One target pass per round over [x_last | H hard drafts | S soft tail].

    - Plain Jacobi: S=0, lora=False (guesses are the target's own previous outputs).
    - Training-free soft Jacobi: S>0, lora=False, flag=False.
    - Arm C: lora=True, flag=True, inp='soft'; C-hard: same with inp='hard'.
    - Arm C+D: dm = block drafter that fills guess slots the target has no guess for yet.
    Hard drafts are argmax (greedy) or a sample (temp>0) of the current guess distribution.
    Only hard-slot KV is ever kept, and the LoRA (gate=H+1) touches only soft rows.

    tail="attached" (the plan's literal layout): soft slots guess offsets H+1..H+S and attend to
      the hard drafts, so after a rejection at offset a+1 every later output is stale.
    tail="detached": soft slots guess offsets 1..S (S >= H), sit at those RoPE positions and cannot
      attend to the hard-draft rows, only to the committed prefix (incl. x_last) and earlier soft
      slots. Their outputs never condition on an unverified token. Hard rows are unaffected.
    """
    st = Stats()
    det = tail == "detached"
    assert not det or S >= H
    n_g = S if det else H + S
    tc, lg = _prefill(tm, prompt, max_new + n_g + 8)
    dc = _prefill_draft(dm, prompt) if dm is not None else None
    run = Run(prompt, max_new)
    run.commit([_pick(lg, temp, gen)])
    dev = tm.dev
    G_ids = torch.zeros(n_g, TOPK, dtype=torch.long, device=dev)
    G_p = torch.zeros(n_g, TOPK, device=dev)
    nv = 0  # guesses valid for offsets 1..nv (always a contiguous prefix)
    fresh = getattr(tm, "e_mask", tm.E_mean).float() if lora else tm.E_mean
    e_flag = tm.e_flag if flag else None
    while not run.done:
        if dc is not None and nv < n_g:
            dl = block_propose(dm, dc, run.seq, n_g, d_iters, d_inp)
            st.dpasses += d_iters
            G_ids[nv:], G_p[nv:] = to_topk(dl[nv:])
            nv = n_g
        # hard drafts
        qs = []
        if temp == 0:
            h = G_ids[:H, 0].clone()
        else:
            P = torch.where((torch.arange(H, device=dev) < nv)[:, None], G_p[:H], torch.ones_like(G_p[:H]))
            idx = torch.multinomial(P / P.sum(-1, keepdim=True), 1, generator=gen)[:, 0] if H else None
            h = G_ids[torch.arange(H, device=dev), idx] if H else G_ids[:0, 0]
        prev = run.seq[-1]
        for j in range(H):
            if j >= nv:  # no guess yet: copy the previous token (deterministic q = one-hot)
                h[j] = h[j - 1] if j > 0 else prev
        if temp > 0:
            for j in range(H):
                if j < nv:
                    qs.append(dense(G_ids[j], G_p[j], tm.V))
                else:
                    q = torch.zeros(tm.V, device=dev)
                    q[h[j]] = 1.0
                    qs.append(q)
        # soft tail rows
        rows = []
        if S:
            gi, gp = (G_ids, G_p) if det else (G_ids[H:], G_p[H:])
            if inp == "soft":
                pp = gp if sharpen == 1.0 else gp.pow(1.0 / sharpen)
                rows = tm.embed_soft(gi, pp)
            else:
                rows = tm.rows32(gi[:, 0])
            k = torch.arange(n_g - S, n_g, device=dev)
            rows = torch.where((k < nv)[:, None], rows, fresh[None])
            if e_flag is not None:
                rows = rows + e_flag
        x = torch.tensor([prev], device=dev)
        hx = tm.rows32(torch.cat([x, h]))
        emb = torch.cat([hx, rows]) if S else hx
        if det:
            n0 = tc.n
            rope = torch.cat([tm.ar[n0:n0 + H + 1], tm.ar[n0 + 1:n0 + 1 + S]])
            tl = tm.step(tc, embeds=emb, gate=(H + 1) if lora else None, rope=rope, hide=(H + 1, 1, H + 1))
        else:
            tl = tm.step(tc, embeds=emb, gate=(H + 1) if lora else None)
        a, toks = verify(tl[: H + 1], h, qs, temp, gen)
        tc.n += 1 + a
        n_new = n_g - a
        if det:  # soft row i guesses offset i+1 and predicts offset i+2; new offset m = old a+1+m
            G_ids[:n_new], G_p[:n_new] = to_topk(tl[H + 1 + a:H + 1 + S])
            nv = n_new
        elif stale == "reset" and a < H:
            nv = 0
        else:
            G_ids[:n_new], G_p[:n_new] = to_topk(tl[a + 1:])
            nv = n_new
        st.add(H, a, run.commit(toks))
    return run.out, st
