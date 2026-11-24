"""Where does Qwen3-8B hold its next 8 tokens? A sweep over every second layer (0..36) of the target.

For each layer l, the target's state at one position p is fed to a small nonlinear probe (MLP trunk, one output
vector per offset, frozen LM head on top) that predicts the target's own top-1 tokens at positions p+1..p+8
(teacher forced, labels = the HF target's own top-1, as in stage2.py). Probes are trained on 400 training answers
and scored on the 120 dev answers.

Reported per layer: top-1 accuracy per offset, and "run" = mean leading run of correct offsets + 1 (the number a
drafter reading only that one state would get per round, at most 9). Also reported separately at boundary
positions (the token at p contains a newline), where the planning literature finds future information.
Baseline: the same probe on the input token embedding at p (no target computation).

Run:    python layer_sweep.py [--smoke]
Output: results/layer_sweep.json, data/sweep/*.npy
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import DATA, RES, load_items
from dflash_eval import load_target

K = 8


class Probe(nn.Module):
    def __init__(self, D, h=1024):
        super().__init__()
        self.trunk = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, h), nn.GELU(), nn.Linear(h, h), nn.GELU())
        self.heads = nn.Parameter(torch.randn(K, h, D) * h ** -0.5)
        self.norm = nn.LayerNorm(D)

    def forward(self, x):                       # (B, D) -> (B, K, D) head inputs
        return self.norm(torch.einsum("bh,khd->bkd", self.trunk(x), self.heads))


def extract(t, items, layers, stride, tok_is_nl):
    feats = {l: [] for l in layers}
    labs, bnd, setid = [], [], []
    emb = t.get_input_embeddings().weight
    feats["emb"] = []
    for it in items:
        ids = torch.tensor(it["prompt"] + it["resp"], device="cuda")
        n0 = len(it["prompt"])
        if len(ids) < n0 + K + 2:
            continue
        with torch.no_grad():
            out = t(ids[None], output_hidden_states=True)
            top1 = out.logits[0].argmax(-1)
            pos = torch.arange(n0 - 1, len(ids) - K, stride, device="cuda")
            for l in layers:
                feats[l].append(out.hidden_states[l][0, pos].half().cpu())
            feats["emb"].append(emb[ids[pos]].half().cpu())
            labs.append(top1[pos[:, None] + torch.arange(K, device="cuda")[None]].cpu())
            bnd.append(tok_is_nl[ids[pos]].cpu())
        del out
    return {k: torch.cat(v) for k, v in feats.items()}, torch.cat(labs), torch.cat(bnd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=400)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--B", type=int, default=256)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    layers = list(range(0, 37, 2))
    if args.smoke:
        args.n_train, args.steps, layers = 4, 20, [18, 36]
    from transformers import AutoTokenizer
    from dflash_eval import TARGET_ID
    tk = AutoTokenizer.from_pretrained(TARGET_ID)
    t = load_target()
    V = t.lm_head.weight.shape[0]
    nl = torch.zeros(V, dtype=torch.bool)
    for i in range(len(tk)):
        if "\n" in tk.decode([i]):
            nl[i] = True
    nl = nl.cuda()
    tr_items = load_items("q3_train")
    rng = np.random.default_rng(0)
    tr_items = [tr_items[i] for i in rng.permutation(len(tr_items))[:args.n_train]]
    dev_sets = {s: load_items(f"q3_dev_{s}_40") for s in ("math", "code", "chat")}
    if args.smoke:
        dev_sets = {s: v[:2] for s, v in dev_sets.items()}
    t0 = time.time()
    Xtr, Ytr, _ = extract(t, tr_items, layers, args.stride, nl)
    Xdv, Ydv, Bdv, Sdv = {}, [], [], []
    for j, (s, items) in enumerate(dev_sets.items()):
        X, Y, Bn = extract(t, items, layers, args.stride, nl)
        for kk, v in X.items():
            Xdv.setdefault(kk, []).append(v)
        Ydv.append(Y)
        Bdv.append(Bn)
        Sdv.append(torch.full((len(Y),), j))
    Xdv = {kk: torch.cat(v) for kk, v in Xdv.items()}
    Ydv, Bdv, Sdv = torch.cat(Ydv), torch.cat(Bdv), torch.cat(Sdv)
    print(f"train rows {len(Ytr)}  dev rows {len(Ydv)}  boundary dev rows {int(Bdv.sum())}  "
          f"extract {(time.time() - t0) / 60:.1f} min", flush=True)
    W = t.lm_head.weight.detach()
    del t
    torch.cuda.empty_cache()
    res = {"args": vars(args), "layers": {}}
    for key in ["emb"] + layers:
        xtr = Xtr[key].float()
        mu, sd = xtr.mean(0), xtr.std(0).clamp_min(1e-3)
        xtr = ((xtr - mu) / sd).clamp(-20, 20)
        xdv = ((Xdv[key].float() - mu) / sd).clamp(-20, 20)
        torch.manual_seed(0)
        pr = Probe(W.shape[1]).cuda()
        opt = torch.optim.AdamW(pr.parameters(), lr=1e-3, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 100) * 0.5 * (1 + np.cos(np.pi * s / args.steps)))
        g = torch.Generator().manual_seed(0)
        for step in range(args.steps):
            b = torch.randint(0, len(xtr), (args.B,), generator=g)
            with torch.autocast("cuda", torch.bfloat16):
                lg = pr(xtr[b].cuda()) @ W.T
            loss = F.cross_entropy(lg.float().flatten(0, 1), Ytr[b].cuda().flatten())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        pred = []
        with torch.no_grad():
            for i in range(0, len(xdv), 512):
                with torch.autocast("cuda", torch.bfloat16):
                    pred.append((pr(xdv[i:i + 512].cuda()) @ W.T).argmax(-1).cpu())
        eq = (torch.cat(pred) == Ydv)
        run = eq.int().cumprod(1).sum(1).float() + 1

        def summ(m):
            return {"rows": int(m.sum()), "acc_by_offset": eq[m].float().mean(0).tolist(), "run": run[m].mean().item()}
        r = {"all": summ(torch.ones(len(eq), dtype=torch.bool)), "boundary": summ(Bdv),
             "by_set": {s: summ(Sdv == j) for j, s in enumerate(dev_sets)}, "train_loss": loss.item()}
        res["layers"][str(key)] = r
        print(key, "run", round(r["all"]["run"], 3), "acc", [round(a, 3) for a in r["all"]["acc_by_offset"]],
              "| boundary run", round(r["boundary"]["run"], 3), f"{(time.time() - t0) / 60:.1f} min", flush=True)
    json.dump(res, open(os.path.join(RES, f"layer_sweep{'_smoke' if args.smoke else ''}.json"), "w"), indent=1)
    print("=== layer sweep done", flush=True)


if __name__ == "__main__":
    main()
