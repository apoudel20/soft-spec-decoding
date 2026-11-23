"""Planner go/no-go: does the target know its own future better than the drafter does?

For both Qwen2.5-1.5B (target) and Qwen2.5-0.5B (drafter), we take frozen hidden states at a
response position t (a middle layer and the last layer) and fit simple probes for:
  token+k   the token at t+k (k = 1, 2, 4, 8, 16, 32), linear softmax probe over the 4096 most
            frequent response tokens plus "other". Metric: top-1 accuracy on the eval sequences.
  bag+N     the mean target-embedding of tokens t+1..t+N (N = 32, 128): a model-neutral summary
            of "what comes next". Ridge regression. Metric: cosine similarity to the truth, and
            R^2 relative to predicting the training mean.
  remaining log(1 + tokens left in the response). Ridge regression. Metric: R^2.
Baselines without any hidden state: unigram (most frequent token) for token+k, and the same
ridge probes fit on the current token's embedding only ("embedding-only").

Train on the self-distilled training sequences; test on the target's greedy outputs for the
eval prompts (data/ref_<set>_<n>.pt). Nothing here is tuned on the eval results.
The probes are linear, so a gap between models measures linearly readable information only.

Run: python probes.py --n_train 1500 --per_seq 48
Output: results/probes.json
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from common import DATA, RES
from model import DRAFTER, TARGET, QwenLM, rms_norm, rotate_half

KS = [1, 2, 4, 8, 16, 32]
NS = [32, 128]
VOCAB = 4096


@torch.no_grad()
def hidden_at(m, ids, layers):
    """Residual-stream hidden states after the given layers (last layer is final-normed). ids (1, L)."""
    h = m.embed_ids(ids)
    L = ids.shape[1]
    pos = torch.arange(L, device=m.dev)
    cos, sin = m.rope(pos)
    cos, sin = cos[None, None], sin[None, None]
    out = {}
    for li, Lr in enumerate(m.layers):
        x = rms_norm(h, Lr.ln1, Lr.eps)
        B, T, _ = x.shape
        q = Lr.q(x, None).view(B, T, Lr.nh, Lr.hd).transpose(1, 2)
        k = Lr.k(x, None).view(B, T, Lr.nkv, Lr.hd).transpose(1, 2)
        v = Lr.v(x, None).view(B, T, Lr.nkv, Lr.hd).transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        h = h + Lr.o(a.transpose(1, 2).reshape(B, T, -1), None)
        x = rms_norm(h, Lr.ln2, Lr.eps)
        h = h + Lr.down(F.silu(Lr.gate(x, None)) * Lr.up(x, None), None)
        if li in layers:
            out[li] = (rms_norm(h, m.norm_w, m.cfg["rms_norm_eps"]) if li == len(m.layers) - 1 else h)[0]
    return out


def sample_positions(seqs, per_seq, rng, need=1):
    out = []
    for si, s in enumerate(seqs):
        r, L = len(s["prompt"]), len(s["prompt"]) + len(s["resp"])
        cand = list(range(r, L - need))
        if not cand:
            continue
        for t in (rng.sample(cand, min(per_seq, len(cand))) if per_seq else cand):
            out.append((si, t))
    return out


@torch.no_grad()
def features(m, seqs, positions, layers):
    by = {}
    for si, t in positions:
        by.setdefault(si, []).append(t)
    feats = {li: [] for li in layers}
    order = []
    for si, ts in by.items():
        s = seqs[si]
        ids = torch.tensor(s["prompt"] + s["resp"], device="cuda")[None]
        hs = hidden_at(m, ids, layers)
        tt = torch.tensor(ts, device="cuda")
        for li in layers:
            feats[li].append(hs[li][tt].float().cpu())
        order += [(si, t) for t in ts]
    return {li: torch.cat(v) for li, v in feats.items()}, order


def targets(seqs, order, vocab_map, E):
    """Per-position probe targets."""
    tok = {k: [] for k in KS}
    bag = {n: [] for n in NS}
    rem, cur = [], []
    for si, t in order:
        full = seqs[si]["prompt"] + seqs[si]["resp"]
        L = len(full)
        for k in KS:
            tok[k].append(vocab_map.get(full[t + k], VOCAB) if t + k < L else -100)
        for n in NS:
            fut = full[t + 1:min(L, t + 1 + n)]
            bag[n].append(E[torch.tensor(fut)].float().mean(0) if fut else torch.zeros(E.shape[1]))
        rem.append(math.log1p(L - 1 - t))
        cur.append(E[full[t]].float())
    return ({k: torch.tensor(v) for k, v in tok.items()}, {n: torch.stack(v) for n, v in bag.items()},
            torch.tensor(rem), torch.stack(cur))


def ridge(Xtr, Ytr, Xte, lam=10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A = (Xtr - mu) / sd
    B = (Xte - mu) / sd
    A1 = torch.cat([A, torch.ones(len(A), 1)], 1).double().cuda()
    B1 = torch.cat([B, torch.ones(len(B), 1)], 1).double().cuda()
    Yc = Ytr.double().cuda()
    W = torch.linalg.solve(A1.T @ A1 + lam * len(A1) * torch.eye(A1.shape[1], device="cuda", dtype=torch.float64), A1.T @ Yc)
    return (B1 @ W).float().cpu()


def r2(pred, y, ytr_mean):
    return 1 - ((pred - y) ** 2).sum().item() / ((y - ytr_mean) ** 2).sum().item()


def token_probe(Xtr, ytr, Xte, yte, epochs=4):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A = ((Xtr - mu) / sd).cuda()
    B = ((Xte - mu) / sd).cuda()
    ytr, yte = ytr.cuda(), yte.cuda()
    keep = ytr >= 0
    A, ytr = A[keep], ytr[keep]
    W = torch.nn.Linear(A.shape[1], VOCAB + 1).cuda()
    opt = torch.optim.AdamW(W.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(A), device="cuda")
        for i in range(0, len(A), 1024):
            b = perm[i:i + 1024]
            loss = F.cross_entropy(W(A[b]), ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        ok = yte >= 0
        pred = W(B[ok]).argmax(-1)
        return (pred == yte[ok]).float().mean().item(), (pred == yte[ok])[yte[ok] < VOCAB].float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=1500)
    ap.add_argument("--per_seq", type=int, default=48)
    ap.add_argument("--sets", nargs="+", default=["gsm8k:40", "mtbench:40", "humaneval:40"])
    ap.add_argument("--out", default="probes")
    args = ap.parse_args()
    rng = random.Random(0)
    t0 = time.time()
    train = [it for it in torch.load(os.path.join(DATA, "train.pt")) if len(it["resp"]) > 40]
    rng.shuffle(train)
    train = train[: args.n_train]
    tests = {}
    for s in args.sets:
        name, _, n = s.partition(":")
        tests[name] = torch.load(os.path.join(DATA, f"ref_{name}_{n}.pt"))
    # vocabulary of frequent response tokens (from training data only)
    cnt = {}
    for it in train:
        for x in it["resp"]:
            cnt[x] = cnt.get(x, 0) + 1
    top = sorted(cnt, key=cnt.get, reverse=True)[:VOCAB]
    vmap = {x: i for i, x in enumerate(top)}
    tr_pos = sample_positions(train, args.per_seq, rng)
    te_pos = {k: sample_positions(v, 0, rng) for k, v in tests.items()}

    res = {"args": vars(args), "n_train_positions": len(tr_pos), "models": {}}
    E_ref = None
    for mname, mid in (("target_1.5B", TARGET), ("drafter_0.5B", DRAFTER)):
        m = QwenLM(mid)
        m.use_graphs = False
        if E_ref is None:
            E_ref = m.E.float().cpu()  # model-neutral summary space: the target's embedding matrix
        nl = len(m.layers)
        layers = [nl // 2, nl - 1]
        Ftr, otr = features(m, train, tr_pos, layers)
        Ttr = targets(train, otr, vmap, E_ref)
        Fte = {}
        for k, v in tests.items():
            Fte[k] = (features(m, v, te_pos[k], layers), )
        del m
        torch.cuda.empty_cache()
        out = {}
        for li in layers:
            lname = f"layer{li + 1}of{nl}"
            out[lname] = {}
            for sname, seqs in tests.items():
                (Fs, ote), = Fte[sname]
                Tte = targets(seqs, ote, vmap, E_ref)
                r = {}
                for k in KS:
                    acc, acc_in = token_probe(Ftr[li], Ttr[0][k], Fs[li], Tte[0][k])
                    r[f"token+{k}"] = round(acc, 4)
                for n in NS:
                    p = ridge(Ftr[li], Ttr[1][n], Fs[li])
                    r[f"bag+{n}_cos"] = round(F.cosine_similarity(p, Tte[1][n], -1).mean().item(), 4)
                    r[f"bag+{n}_r2"] = round(r2(p, Tte[1][n], Ttr[1][n].mean(0)), 4)
                p = ridge(Ftr[li], Ttr[2][:, None], Fs[li])[:, 0]
                r["remaining_r2"] = round(r2(p, Tte[2], Ttr[2].mean()), 4)
                out[lname][sname] = r
                print(mname, lname, sname, r, f"{(time.time() - t0) / 60:.1f} min", flush=True)
        res["models"][mname] = out
    # baselines without hidden states
    base = {}
    Ttr = targets(train, tr_pos, vmap, E_ref)
    for sname, seqs in tests.items():
        ote = te_pos[sname]
        Tte = targets(seqs, ote, vmap, E_ref)
        b = {}
        for k in KS:
            y = Tte[0][k]
            y = y[y >= 0]
            b[f"token+{k}_unigram"] = round((y == 0).float().mean().item(), 4)
        for n in NS:
            p = ridge(Ttr[3], Ttr[1][n], Tte[3])
            b[f"bag+{n}_cos_embedding_only"] = round(F.cosine_similarity(p, Tte[1][n], -1).mean().item(), 4)
            b[f"bag+{n}_r2_embedding_only"] = round(r2(p, Tte[1][n], Ttr[1][n].mean(0)), 4)
        p = ridge(Ttr[3], Ttr[2][:, None], Tte[3])[:, 0]
        b["remaining_r2_embedding_only"] = round(r2(p, Tte[2], Ttr[2].mean()), 4)
        base[sname] = b
        print("baseline", sname, b, flush=True)
    res["baselines"] = base
    res["minutes"] = (time.time() - t0) / 60
    json.dump(res, open(os.path.join(RES, args.out + ".json"), "w"), indent=1)
    print("wrote", os.path.join(RES, args.out + ".json"))


if __name__ == "__main__":
    main()
