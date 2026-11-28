"""Decoder fine-tuned on predicted plans (27 Nov follow-up to the plan_pw runs).

The plan_pw decoders were trained on the oracle plan (encoder over the true future tokens). At drafting time they
receive the predictor's plan, which they never saw in training. Here the encoder is dropped and the decoder is
fine-tuned on what it gets at drafting time.

  --arm dec   freeze the predictor P, fine-tune only the decoder on P's plan (unit average power, no noise)
  --arm e2e   fine-tune P and the decoder together on the token loss (P becomes part of one direct drafter)
  --arm ctrl  the no-plan control (plan_pw_G8c0), decoder fine-tuned with the same recipe, the step-matched baseline

All arms start from saved checkpoints, train with offset 0 only, and are scored on the same dev rounds as
plan_reeval.py (one draft, and the K most likely drafts for K = 4, 8, 12, 16 with trie nodes), before and after.
Run:    python ft_pred.py --ckpt plan_pw_G8c16s0.25 --arm dec [--steps 4000] [--smoke]
Output: results/ft_pred_<ckpt>_<arm>.json
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from ae_gate import CKPT, Data
from common import DATA, RES
from plan_gist import K, LAYERS, Decoder, PlanPred, kbest, load_stats, loader, pack, round_ok, tree_stats

KS = [4, 8, 12, 16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arm", choices=["dec", "e2e", "ctrl"], required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--B", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--train_cache", default="cache_pb")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.B = 30, 16
    dev = "cuda"
    tag = f"{args.ckpt}_{args.arm}" + (f"_st{args.steps}" if args.steps != 4000 else "") + ("_smoke" if args.smoke else "")
    ck = torch.load(os.path.join(CKPT, f"{args.ckpt}.pt"), map_location="cpu")
    a = ck["args"]
    G, c = a["G"], a["c"]
    assert (c == 0) == (args.arm == "ctrl"), "ctrl arm needs the c = 0 checkpoint, the others a plan checkpoint"
    assert not a.get("phist") and not a.get("rel"), "only plain plan_pw checkpoints"
    print(f"=== ft_pred {tag}: G {G} c {c}, {args.steps} steps", flush=True)

    cache = os.path.join(DATA, "cache")
    stats = load_stats(cache)
    W = torch.load(os.path.join(cache, "lm_head.pt")).to(dev)
    D = W.shape[1]
    fmean, fstd = stats[-1][0].to(dev), stats[-1][1].to(dev)

    def logp(h):
        return ((h.float() * fstd + fmean).to(torch.bfloat16) @ W.T).float().log_softmax(-1)

    dec = Decoder(G, c, D).to(dev)
    dec.load_state_dict(ck["dec"])
    P = None
    if c:
        P = PlanPred(G, c, D).to(dev)
        P.load_state_dict(ck["P"])
        P.requires_grad_(args.arm == "e2e")
    count = lambda m: sum(p.numel() for p in m.parameters()) / 1e6 if m is not None else 0
    print(f"params decoder {count(dec):.1f}M, P {count(P):.1f}M", flush=True)

    def plan_in(raw_a, B):
        if not c:
            return torch.zeros(B, G, 1, device=dev)
        ph = P(raw_a).float()
        return ph * torch.rsqrt(ph.pow(2).mean((1, 2), keepdim=True) + 1e-6)

    g = torch.Generator(device=dev).manual_seed(0)

    def evaluate():
        dec.eval()
        if P is not None:
            P.eval()
        res = {}
        for sp in ["dev_math", "dev_code", "dev_chat"]:
            dt = Data(cache, sp, LAYERS, K, stats)
            T1 = np.load(os.path.join(cache, f"{sp}_meta.npz"))["top1"].astype(np.int64)
            st = np.arange(len(dt.seq))
            st = st[round_ok(dt.seq, st, 2 * K)]
            keep, last = [], -10 ** 9
            for s in st:
                if s >= last + 2 * K:
                    keep.append(s)
                    last = s
            rounds = np.array(keep)[:128] if args.smoke else np.array(keep)
            acc, ex = [], {}
            with torch.no_grad():
                for i in range(0, len(rounds), 32):
                    s = rounds[i:i + 32]
                    raw, raw_a, ids, m, lab, oo, _, _ = [x.to(dev) for x in pack(dt, T1, s, s, G)]
                    with torch.autocast("cuda", torch.bfloat16):
                        lp = logp(dec(plan_in(raw_a, len(s)), oo, raw))
                    acc.append(((lp.argmax(-1) == lab).int().cumprod(1).sum(1) + 1).float().cpu())
                    for Kb in KS:
                        st_ = tree_stats(kbest(lp, Kb).cpu(), lab.cpu())
                        ex.setdefault(f"kbest{Kb}_best", []).append(st_[0])
                        ex.setdefault(f"kbest{Kb}_nodes", []).append(st_[3])
            r = {"rounds": int(len(rounds)), "one_draft": torch.cat(acc).mean().item()}
            r.update({k: torch.cat(v).mean().item() for k, v in ex.items()})
            res[sp] = r
            print(sp, json.dumps({k: round(v, 3) for k, v in r.items()}), flush=True)
        dec.train()
        if P is not None and args.arm == "e2e":
            P.train()
        return res

    t0 = time.time()
    out = {"args": vars(args), "src_args": a, "params": {"decoder": count(dec), "P": count(P)}}
    print("--- before", flush=True)
    out["before"] = evaluate()

    dt = Data(os.path.join(DATA, args.train_cache), "train", LAYERS, K, stats)
    T1 = np.load(os.path.join(DATA, args.train_cache, "train_meta.npz"))["top1"].astype(np.int64)
    st = np.arange(len(dt.seq))
    st = st[round_ok(dt.seq, st, 2 * K)]
    params = list(dec.parameters()) + (list(P.parameters()) if args.arm == "e2e" else [])
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    n = args.steps
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * max(0.1, 0.5 * (1 + np.cos(np.pi * min(1, s / n)))))
    dec.train()
    if args.arm == "e2e":
        P.train()
    elif P is not None:
        P.eval()
    log = []
    for step, b in enumerate(loader(dt, T1, st, args.B, n, G, 0.0, 1)):
        raw, raw_a, ids, m, lab, o, _, _ = [x.to(dev, non_blocking=True) for x in b]
        with torch.autocast("cuda", torch.bfloat16):
            with torch.set_grad_enabled(args.arm == "e2e"):
                y = plan_in(raw_a, len(raw))
            loss = F.nll_loss(logp(dec(y, o, raw)).flatten(0, 1), lab.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sch.step()
        if step % 500 == 0 or step == n - 1:
            r = {"step": step, "nll": loss.item(), "min": (time.time() - t0) / 60}
            log.append(r)
            print(json.dumps(r), flush=True)
            if not np.isfinite(loss.item()):
                raise SystemExit("non-finite loss")
    out["train_log"] = log
    print("--- after", flush=True)
    out["after"] = evaluate()
    if not args.smoke:
        torch.save({"dec": dec.state_dict(), "P": P.state_dict() if P is not None else None, "args": vars(args),
                    "src_args": a}, os.path.join(CKPT, f"ft_pred_{tag}.pt"))
    json.dump(out, open(os.path.join(RES, f"ft_pred_{tag}.json"), "w"), indent=1)
    print(f"=== ft_pred {tag} done {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
