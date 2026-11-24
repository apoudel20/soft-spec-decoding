"""Quick check on the existing state cache: how many of the target's next 8 tokens can one state at position p give?

For each cached layer (18, 24, 30, final 36) and a baseline (the input token's embedding at p, no target compute),
train the layer_sweep.py probe (MLP trunk, one output per offset, frozen LM head) on 60k train rows and score on
dev: top-1 accuracy per offset against the target's own top-1, and run = leading correct offsets + 1 (out of 9).
Offset 1 from the final layer is exact by construction (that is the target's own next token).

Run:    python cache_probe.py
Output: results/cache_probe.json
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import DATA, RES
from layer_sweep import K, Probe
from tok_vae import load_embed

C = os.path.join(DATA, "cache")


def rows(split, n=None, seed=0):
    m = np.load(os.path.join(C, f"{split}_meta.npz"))
    seq, top1, nxt = m["seq"], m["top1"].astype(np.int64), m["next"].astype(np.int64)
    N = len(seq)
    ok = np.zeros(N, bool)
    ok[1:N - K + 1] = (seq[1:N - K + 1] == seq[K:]) & (seq[1:N - K + 1] == seq[:N - K])
    r = np.nonzero(ok)[0]
    if n:
        r = np.sort(np.random.default_rng(seed).choice(r, n, replace=False))
    lab = top1[r[:, None] + np.arange(K)[None]]
    tok = nxt[r - 1]                                  # the input token at p
    return r, torch.from_numpy(lab), torch.from_numpy(tok)


def main():
    t0 = time.time()
    W = torch.load(os.path.join(C, "lm_head.pt")).cuda()
    E = load_embed()
    r_tr, y_tr, tok_tr = rows("train", 60000)
    dev = {s: rows(f"dev_{s}") for s in ("math", "code", "chat")}
    res = {}
    for key in ["emb", 18, 24, 30, 36]:
        def feats(split, r, tok):
            if key == "emb":
                return E[tok].float()
            X = np.load(os.path.join(C, f"{split}_L{key}.npy"), mmap_mode="r")
            return torch.from_numpy(X[r].astype(np.float32))
        xtr = feats("train", r_tr, tok_tr)
        mu, sd = xtr.mean(0), xtr.std(0).clamp_min(1e-3)
        xtr = ((xtr - mu) / sd).clamp(-20, 20)
        torch.manual_seed(0)
        pr = Probe(W.shape[1]).cuda()
        opt = torch.optim.AdamW(pr.parameters(), lr=1e-3, weight_decay=0.01)
        steps = 1500
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 100) * 0.5 * (1 + np.cos(np.pi * s / steps)))
        g = torch.Generator().manual_seed(0)
        for step in range(steps):
            b = torch.randint(0, len(xtr), (256,), generator=g)
            with torch.autocast("cuda", torch.bfloat16):
                lg = pr(xtr[b].cuda()) @ W.T
            loss = F.cross_entropy(lg.float().flatten(0, 1), y_tr[b].cuda().flatten())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        out = {}
        for s, (r, y, tok) in dev.items():
            x = ((feats(f"dev_{s}", r, tok) - mu) / sd).clamp(-20, 20)
            pred = []
            with torch.no_grad():
                for i in range(0, len(x), 512):
                    with torch.autocast("cuda", torch.bfloat16):
                        pred.append((pr(x[i:i + 512].cuda()) @ W.T).argmax(-1).cpu())
            eq = torch.cat(pred) == y
            out[s] = {"rows": len(y), "acc_by_offset": [round(a, 4) for a in eq.float().mean(0).tolist()],
                      "run": (eq.int().cumprod(1).sum(1).float() + 1).mean().item()}
        res[str(key)] = out
        print(key, json.dumps({s: [round(v["run"], 3), v["acc_by_offset"][:4]] for s, v in out.items()}),
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
    json.dump(res, open(os.path.join(RES, "cache_probe.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
