"""Fairer layer probe: does Qwen3-8B know more about its own upcoming content than Qwen3-0.6B?

The first sweep (probes_q3.py) gave the 8B a 4,096-wide state against the 0.6B's 1,024 with only ~14k training
rows, and the 8B summary probe went below zero on chat, which looks like overfitting. Here both models are
compared on equal terms:
  - more training rows (64 positions per answer, ~38k rows),
  - a wider ridge penalty grid (up to 10x the row count), chosen on held-out training rows as before,
  - each layer's state reduced by PCA (fit on training rows only) to the same width k for both models,
    k in {64, 256, 1024}, plus the full width for reference.
Only a few layers per model are probed: around the peaks found in the first sweep.
Targets: mean embedding of the next 16, 64 and 128 tokens, and log remaining length. Tested on held-out dev answers.

Run:    python probes_q3_fair.py
Output: results/probes_q3_fair.json
"""
import argparse
import json
import os
import random
import time

import torch

import probes as PR
from common import DATA, RES
from probes import r2, ridge, sample_positions, targets
from probes_q3 import SMALL_ID, TARGET_ID, features, load

PR.NS = [16, 64, 128]
LAMS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
LAYERS = {TARGET_ID: [12, 18, 24, 28, 32, 36], SMALL_ID: [10, 14, 18, 22, 26, 28]}


def pca_fit(X, k):
    mu = X.mean(0)
    U, S, V = torch.pca_lowrank((X - mu).cuda(), q=k, center=False, niter=4)
    return mu, V[:, :k].cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=600)
    ap.add_argument("--per_seq", type=int, default=64)
    ap.add_argument("--sets", nargs="+", default=["math:40", "code:40", "chat:40"])
    ap.add_argument("--ks", nargs="+", type=int, default=[64, 256, 1024, 0])  # 0 = full width
    ap.add_argument("--out", default="probes_q3_fair")
    args = ap.parse_args()
    rng = random.Random(0)
    t0 = time.time()
    train = [it for it in torch.load(os.path.join(DATA, "q3_train.pt")) if len(it["resp"]) > 40]
    rng.shuffle(train)
    train = train[: args.n_train]
    tests = {}
    for s in args.sets:
        name, _, n = s.partition(":")
        tests[name] = torch.load(os.path.join(DATA, f"q3_dev_{name}_{n}.pt"))
    tr_pos = sample_positions(train, args.per_seq, rng)
    te_pos = {k: sample_positions(v, 0, rng) for k, v in tests.items()}
    small = load(SMALL_ID)
    E_ref = small.model.embed_tokens.weight.float().cpu()
    del small
    torch.cuda.empty_cache()
    vmap = {}
    res = {"args": vars(args), "n_train_rows": len(tr_pos), "lams": LAMS, "models": {}}
    for mid in (SMALL_ID, TARGET_ID):
        m = load(mid)
        layers = LAYERS[mid]
        nl = len(m.model.layers)
        print(f"== {mid}: probing {layers} of {nl}", flush=True)
        Ftr, otr = features(m, train, tr_pos, layers)
        Ttr = targets(train, otr, vmap, E_ref)
        Fte = {k: features(m, v, te_pos[k], layers) for k, v in tests.items()}
        del m
        torch.cuda.empty_cache()
        out = {}
        for li in layers:
            X = Ftr[li]
            out[f"layer{li}of{nl}"] = {}
            for k in args.ks:
                if k:
                    mu, V = pca_fit(X, k)
                    proj = lambda Z: (Z - mu) @ V
                else:
                    proj = lambda Z: Z
                Xp = proj(X)
                rk = {}
                for sname, (Fs, ote) in Fte.items():
                    Tte = targets(tests[sname], ote, vmap, E_ref)
                    Zs = proj(Fs[li])
                    r = {}
                    for n in PR.NS:
                        p = ridge(Xp, Ttr[1][n], Zs, lams=LAMS)
                        r[f"bag+{n}_r2"] = round(r2(p, Tte[1][n], Ttr[1][n].mean(0)), 4)
                    p = ridge(Xp, Ttr[2][:, None], Zs, lams=LAMS)[:, 0]
                    r["remaining_r2"] = round(r2(p, Tte[2], Ttr[2].mean()), 4)
                    rk[sname] = r
                out[f"layer{li}of{nl}"][f"k{k or 'full'}"] = rk
                print(mid, li, f"k={k or 'full'}", {s: (v["bag+16_r2"], v["bag+64_r2"], v["bag+128_r2"], v["remaining_r2"]) for s, v in rk.items()},
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
            res["models"][mid] = out
            json.dump(res, open(os.path.join(RES, args.out + ".json"), "w"), indent=1)
    res["minutes"] = (time.time() - t0) / 60
    json.dump(res, open(os.path.join(RES, args.out + ".json"), "w"), indent=1)
    print("wrote", os.path.join(RES, args.out + ".json"))


if __name__ == "__main__":
    main()
