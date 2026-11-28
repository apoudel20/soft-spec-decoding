"""Diagnostics for the ft_pred fine-tunes (review follow-up, 27 Nov).

For each fine-tuned decoder, on the same dev rounds as ft_pred.py / plan_reeval.py:
  - one draft and the 8 most likely drafts with the predictor's plan (as in ft_pred.py),
  - the same with the plan set to zero, to measure whether the fine-tuned decoder still uses the plan,
  - token loss on the dev rounds,
  - paired bootstrap 95% intervals of (arm - control) over rounds, control = ft_pred_plan_pw_G8c0_ctrl.
Run:    python ft_pred_diag.py
Output: results/ft_pred_diag.json
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from ae_gate import CKPT, Data
from common import DATA, RES
from plan_gist import K, LAYERS, Decoder, PlanPred, kbest, load_stats, pack, round_ok, tree_stats

ARMS = ["plan_pw_G8c0_ctrl", "plan_pw_G8c16s0.25_dec", "plan_pw_G8c16s0.25_e2e",
        "plan_pw_G32c16s0.25_dec", "plan_pw_G32c16s0.25_e2e"]
SETS = ["dev_math", "dev_code", "dev_chat"]


def rounds_of(dt):
    st = np.arange(len(dt.seq))
    st = st[round_ok(dt.seq, st, 2 * K)]
    keep, last = [], -10 ** 9
    for s in st:
        if s >= last + 2 * K:
            keep.append(s)
            last = s
    return np.array(keep)


def boot(d, n=10000, seed=0):
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    return [float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    dev = "cuda"
    cache = os.path.join(DATA, "cache")
    stats = load_stats(cache)
    W = torch.load(os.path.join(cache, "lm_head.pt")).to(dev)
    D = W.shape[1]
    fmean, fstd = stats[-1][0].to(dev), stats[-1][1].to(dev)

    def logp(h):
        return ((h.float() * fstd + fmean).to(torch.bfloat16) @ W.T).float().log_softmax(-1)

    data = {}
    for sp in SETS:
        dt = Data(cache, sp, LAYERS, K, stats)
        T1 = np.load(os.path.join(cache, f"{sp}_meta.npz"))["top1"].astype(np.int64)
        data[sp] = (dt, T1, rounds_of(dt))

    per = {}
    out = {"arms": {}}
    for arm in ARMS:
        ck = torch.load(os.path.join(CKPT, f"ft_pred_{arm}.pt"), map_location="cpu")
        a = ck["src_args"]
        G, c = a["G"], a["c"]
        dec = Decoder(G, c, D).to(dev).eval()
        dec.load_state_dict(ck["dec"])
        P = None
        if c:
            P = PlanPred(G, c, D).to(dev).eval()
            P.load_state_dict(ck["P"])
        modes = ["plan", "zero"] if c else ["plan"]
        res = {}
        for sp in SETS:
            dt, T1, rounds = data[sp]
            acc = {m_: [] for m_ in modes}
            k8 = {m_: [] for m_ in modes}
            nll = {m_: [] for m_ in modes}
            with torch.no_grad():
                for i in range(0, len(rounds), 32):
                    s = rounds[i:i + 32]
                    raw, raw_a, ids, m, lab, oo, _, _ = [x.to(dev) for x in pack(dt, T1, s, s, G)]
                    for m_ in modes:
                        if c and m_ == "plan":
                            ph = P(raw_a).float()
                            y = ph * torch.rsqrt(ph.pow(2).mean((1, 2), keepdim=True) + 1e-6)
                        else:
                            y = torch.zeros(len(s), G, max(c, 1), device=dev)
                        with torch.autocast("cuda", torch.bfloat16):
                            lp = logp(dec(y, oo, raw))
                        acc[m_].append(((lp.argmax(-1) == lab).int().cumprod(1).sum(1) + 1).float().cpu())
                        k8[m_].append(tree_stats(kbest(lp, 8).cpu(), lab.cpu())[0].float())
                        nll[m_].append(F.nll_loss(lp.flatten(0, 1), lab.flatten(), reduction="none").view(len(s), -1).mean(1).cpu())
            r = {}
            for m_ in modes:
                A, B8, L = torch.cat(acc[m_]).numpy(), torch.cat(k8[m_]).numpy(), torch.cat(nll[m_]).numpy()
                per[(arm, sp, m_)] = (A, B8)
                r[m_] = {"one_draft": float(A.mean()), "kbest8_best": float(B8.mean()), "dev_nll": float(L.mean())}
            r["rounds"] = int(len(rounds))
            res[sp] = r
            print(arm, sp, json.dumps(r), flush=True)
        out["arms"][arm] = res

    out["paired_vs_ctrl"] = {}
    for arm in ARMS[1:]:
        out["paired_vs_ctrl"][arm] = {}
        for sp in SETS:
            A, B8 = per[(arm, sp, "plan")]
            Ac, Bc = per[(ARMS[0], sp, "plan")]
            Az, Bz = per[(arm, sp, "zero")]
            out["paired_vs_ctrl"][arm][sp] = {
                "one_draft_minus_ctrl": boot(A - Ac), "kbest8_minus_ctrl": boot(B8 - Bc),
                "one_draft_plan_minus_zero": boot(A - Az)}
            print(arm, sp, json.dumps(out["paired_vs_ctrl"][arm][sp]), flush=True)
    out["note"] = "mean and 95% paired bootstrap interval over dev rounds (10k resamples). Training-run seed variance is not included."
    json.dump(out, open(os.path.join(RES, "ft_pred_diag.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
