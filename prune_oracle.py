"""Does a plan that prunes the drafter's vocabulary help? Upper bounds before training any planner.

The planner's job here is not to give exact tokens but the set of tokens the next N positions will use (a bag of
words over a paragraph, which compresses far better than exact text). DFlash's per-slot distributions are then
restricted to that set. Measured on dev, on the same rounds as stage2.py / dflash_s2.py, against the target's own
top-1, with up to 8 drafted tokens (max 9 accepted):
  chain              : DFlash top-1 per slot, restricted to the bag
  tree16 / tree64    : best-first tree over the top-4 per slot (inside the bag), 16 / 64 prefixes verified
  top4_oracle        : deepest true prefix inside the per-slot top-4 (perfect selection)
Bags:
  none               : no restriction (plain DFlash)
  future_N           : the target's top-1 tokens at the next N positions (N = 16, 64, 256). Includes the 8 being
                       drafted, so small N is a leaky ceiling. Large N is what a paragraph-level gist could give.
  later_N            : the same but skipping the 8 drafted positions (rows r+8 .. r+8+N): only the vocabulary of
                       what comes after the draft, so no direct leak of the drafted tokens.
  context            : every token already in the prompt and answer so far (no planner needed, deployable)

Run:    python prune_oracle.py
Output: results/prune_oracle.json
"""
import json
import os

import numpy as np
import torch

from cache_states import splits
from common import DATA, RES
from dflash_tree_gap import best_prefixes, accepted_in

K = 8
FE = os.path.join(DATA, "dflash_feats")


def rounds(rows, seq):
    have, keep, last = set(rows.tolist()), [], -10 ** 9
    for i, r in enumerate(rows):
        if r - K in have and seq[r - K] == seq[r] and r >= last + 2 * K:
            keep.append(i)
            last = r
    return np.array(keep)


def main():
    W = torch.load(os.path.join(DATA, "cache", "lm_head.pt")).cuda()
    V = W.shape[0]
    items = splits(False)
    res = {"sets": {}}
    for sp in ["dev_math", "dev_code", "dev_chat"]:
        mt = np.load(os.path.join(DATA, "cache", f"{sp}_meta.npz"))
        seq, top1 = mt["seq"], mt["top1"].astype(np.int64)
        H = np.load(os.path.join(FE, f"{sp}_h.npy"), mmap_mode="r")
        rows = np.load(os.path.join(FE, f"{sp}_rows.npy"))
        keep = rounds(rows, seq)
        first = {}                               # first cache row of each answer
        for i, a in enumerate(seq):
            first.setdefault(int(a), i)
        bags = ["none", "future_16", "future_64", "future_256", "later_64", "later_256", "context"]
        acc = {b: {m: [] for m in ("chain", "tree16", "tree64", "top4_oracle")} for b in bags}
        size = {b: [] for b in bags}
        for i0 in range(0, len(keep), 64):
            idx = keep[i0:i0 + 64]
            r = rows[idx]
            h = torch.from_numpy(np.ascontiguousarray(H[idx])).cuda().to(torch.bfloat16)
            lp = torch.log_softmax((h @ W.T).float(), -1)                  # (B, 8, V)
            lab = top1[r[:, None] + np.arange(K)]
            for b in bags:
                mask = torch.ones(len(r), V, dtype=torch.bool, device="cuda")
                if b != "none":
                    mask.zero_()
                    for n, rr in enumerate(r):
                        a = int(seq[rr])
                        end = first[a] + int((seq == a).sum())             # one past the answer's last row
                        if b == "context":
                            it = items[sp][a]
                            j = rr - first[a]
                            toks = it["prompt"] + it["resp"][:j]
                        else:
                            N = int(b.split("_")[1])
                            lo = rr + (K if b.startswith("later") else 0)
                            toks = top1[lo:min(lo + N, end)].tolist()
                        if toks:
                            mask[n, torch.tensor(toks, device="cuda")] = True
                    size[b] += mask.sum(1).tolist()
                lpm = lp.masked_fill(~mask[:, None], -1e9)
                top = lpm.topk(4, -1)
                lpt, idt = top.values.cpu(), top.indices.cpu()
                for n in range(len(r)):
                    y = lab[n].tolist()
                    c = idt[n, :, 0].tolist()
                    m = 0
                    while m < K and c[m] == y[m]:
                        m += 1
                    acc[b]["chain"].append(m + 1)
                    for bud in (16, 64):
                        acc[b][f"tree{bud}"].append(accepted_in(best_prefixes(lpt[n], idt[n], bud), y))
                    m = 0
                    while m < K and y[m] in idt[n, m].tolist():
                        m += 1
                    acc[b]["top4_oracle"].append(m + 1)
        out = {b: {**{m: float(np.mean(v)) for m, v in acc[b].items()},
                   "bag_size": float(np.mean(size[b])) if size[b] else float(V)} for b in bags}
        res["sets"][sp] = {"rounds": len(keep), **out}
        print(sp, len(keep), flush=True)
        for b in bags:
            print(f"  {b:11s}", {k: round(v, 2) for k, v in out[b].items()}, flush=True)
        json.dump(res, open(os.path.join(RES, "prune_oracle.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
