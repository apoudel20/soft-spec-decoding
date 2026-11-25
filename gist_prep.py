"""Reduce the target's planning states (layers 18, 24, 30) to 512 PCA dims per layer and keep them in one array.

Per layer: z-score with the train stats, zero the massive-activation dims, fit PCA on a random sample of train rows,
standardise each component to unit variance. Output per split: data/cache/<split>_pca.npy, shape (N, 3, 512),
fp16, rows aligned with the state cache. Also data/cache/pca_L<l>.pt and results/gist_prep.json (variance kept).

Run: python gist_prep.py [--smoke]
"""
import argparse
import json
import os

import numpy as np
import torch

from common import DATA, RES

LAYERS = [18, 24, 30]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--fit_rows", type=int, default=40000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cache = os.path.join(DATA, "cache")
    splits = ["dev_math", "dev_code", "dev_chat"] + ([] if args.smoke else ["train"])
    fit_split = "dev_chat" if args.smoke else "train"
    if args.smoke:
        args.fit_rows, args.k = 2000, 64
    rng = np.random.default_rng(0)
    info = {"k": args.k, "layers": {}}
    comps = {}
    for l in LAYERS:
        s = torch.load(os.path.join(cache, f"stats_L{l}.pt"))
        mean, std, mx = s["mean"].numpy(), s["std"].numpy(), s["maxabs"]
        mask = (mx > 20 * mx.median()).numpy()
        std = np.maximum(std, 1e-3 * np.median(std))
        X = np.load(os.path.join(cache, f"{fit_split}_L{l}.npy"), mmap_mode="r")
        idx = np.sort(rng.choice(len(X), min(args.fit_rows, len(X)), replace=False))
        Z = (X[idx].astype(np.float32) - mean) / std
        Z[:, mask] = 0
        Zt = torch.from_numpy(Z)
        mu = Zt.mean(0)
        U, S, V = torch.pca_lowrank(Zt - mu, q=args.k, center=False, niter=4)
        scale = S / np.sqrt(len(Z) - 1)
        tot = ((Zt - mu) ** 2).sum(1).mean().item()
        kept = (scale ** 2).sum().item() / tot
        comps[l] = (mean, std, mask, mu.numpy(), V.numpy().astype(np.float32), scale.numpy())
        torch.save({"mean": mean, "std": std, "mask": mask, "mu": mu.numpy(), "V": V.numpy(), "scale": scale.numpy()},
                   os.path.join(cache, f"pca_L{l}{'_smoke' if args.smoke else ''}.pt"))
        info["layers"][l] = {"variance_kept": kept, "masked_dims": int(mask.sum())}
        print(f"L{l}: {args.k} comps keep {kept:.3f} of variance", flush=True)
    for sp in splits:
        N = len(np.load(os.path.join(cache, f"{sp}_L18.npy"), mmap_mode="r"))
        out = np.lib.format.open_memmap(os.path.join(cache, f"{sp}_pca{'_smoke' if args.smoke else ''}.npy"), "w+",
                                        np.float16, (N, 3, args.k))
        for j, l in enumerate(LAYERS):
            mean, std, mask, mu, V, scale = comps[l]
            X = np.load(os.path.join(cache, f"{sp}_L{l}.npy"), mmap_mode="r")
            for a in range(0, N, 20000):
                Z = (X[a:a + 20000].astype(np.float32) - mean) / std
                Z[:, mask] = 0
                out[a:a + 20000, j] = (((Z - mu) @ V) / scale).astype(np.float16)
        out.flush()
        print(sp, N, flush=True)
    if not args.smoke:
        json.dump(info, open(os.path.join(RES, "gist_prep.json"), "w"), indent=1)
    print("=== gist_prep done", flush=True)


if __name__ == "__main__":
    main()
