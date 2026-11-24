"""How much could a tree on DFlash, and better selection inside it, add?

DFlash drafts a block in one pass: slot i gets its own distribution over tokens. Plain DFlash checks one chain (the
top token in every slot). A tree keeps the top K tokens per slot and checks the most probable prefixes. Here, on
held-out dev answers with teacher-forced context, we measure mean accepted length (accepted + 1) for
  chain        : DFlash's single chain (top-1 per slot),
  tree, budget : the `budget` most probable prefixes (product of per-slot probabilities), budgets 32/64/128/256,
  oracle       : the deepest true prefix present anywhere in the top-K tree (best any selection could do).
The gap between a budgeted tree and the oracle is the room left for a smarter, for example target-guided, selector.

Run:    python dflash_tree_gap.py
Output: results/dflash_tree_gap.json
"""
import argparse
import heapq
import json
import os

import torch

from common import DATA, RES
from dflash_eval import load_drafter, load_target
from dflash_plan import B, ORACLE_N
from kv_oracle import ctx_feats
from kv_oracle_train import logits_with_token

BUDGETS = [32, 64, 128, 256]


def best_prefixes(lp_top, idx_top, budget):
    """Best-first expansion over per-slot top-K choices. Returns the set of selected prefixes (tuples of token ids)."""
    sel, heap = set(), [(-lp_top[0, k].item(), (idx_top[0, k].item(),)) for k in range(idx_top.shape[1])]
    heapq.heapify(heap)
    while heap and len(sel) < budget:
        nlp, pre = heapq.heappop(heap)
        sel.add(pre)
        d = len(pre)
        if d < idx_top.shape[0]:
            for k in range(idx_top.shape[1]):
                heapq.heappush(heap, (nlp - lp_top[d, k].item(), pre + (idx_top[d, k].item(),)))
    return sel


def accepted_in(sel, lab):
    n = 0
    for i in range(len(lab)):
        if tuple(lab[:i + 1]) in sel:
            n = i + 1
        else:
            break
    return n + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--sets", nargs="+", default=["math", "code", "chat"])
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--out", default="dflash_tree_gap")
    ap.add_argument("--hf_labels", action="store_true",
                    help="score against the HF target's own top-1 (as stage2.py does), not the answer text; also report chain capped at 8 drafted")
    args = ap.parse_args()
    t, d = load_target(), load_drafter()
    res = {"args": vars(args), "sets": {}}
    for s_name in args.sets:
        seqs = torch.load(os.path.join(DATA, f"q3_dev_{s_name}_40.pt"))
        acc = {"chain": [], **{f"tree{b}": [] for b in BUDGETS}, "oracle": []}
        if args.hf_labels:
            acc["chain_cap8"] = []
        for it in seqs:
            ids = torch.tensor(it["prompt"] + it["resp"], device="cuda")
            n0 = len(it["prompt"])
            if len(ids) < n0 + B + ORACLE_N // 4 + 2:
                continue
            with torch.no_grad():
                ctx = ctx_feats(t, ids)
                if args.hf_labels:
                    hf_top1 = t(ids[None]).logits[0].argmax(-1)
                for s in range(n0 + 1, len(ids) - B - ORACLE_N // 4, args.stride):
                    lg = logits_with_token(t, d, ids, ctx, s, None)          # (15, V)
                    lp = torch.log_softmax(lg, -1)
                    top = lp.topk(args.K, -1)
                    lab = hf_top1[s:s + B - 1].tolist() if args.hf_labels else ids[s + 1:s + B].tolist()
                    chain = top.indices[:, 0].tolist()
                    n = 0
                    while n < len(lab) and chain[n] == lab[n]:
                        n += 1
                    acc["chain"].append(n + 1)
                    if args.hf_labels:
                        acc["chain_cap8"].append(min(n, 8) + 1)
                    lpt, idt = top.values.cpu(), top.indices.cpu()
                    for b in BUDGETS:
                        acc[f"tree{b}"].append(accepted_in(best_prefixes(lpt, idt, b), lab))
                    n = 0
                    while n < len(lab) and lab[n] in idt[n].tolist():
                        n += 1
                    acc["oracle"].append(n + 1)
        res["sets"][s_name] = {k: {"blocks": len(v), "mean_accepted": sum(v) / len(v)} for k, v in acc.items()}
        print(s_name, {k: round(sum(v) / len(v), 3) for k, v in acc.items()}, flush=True)
        json.dump(res, open(os.path.join(RES, args.out + ".json"), "w"), indent=1)
    print("wrote", os.path.join(RES, args.out + ".json"))


if __name__ == "__main__":
    main()
