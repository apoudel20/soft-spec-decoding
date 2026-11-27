"""Evaluation-only fixes from the 27 Nov audit, on saved checkpoints (offset 0, the same dev rounds).

  control (plan_pw_G8c0): the K most likely drafts for K = 4, 8, 12, 16, so trees can be compared at matched nodes.
  plan_pw runs: the predicted plan rescaled to unit average power (the power the decoder was trained on), with and
     without channel noise. The original eval fed it at about 1 - p_mse of that power.
  cvae runs: K-best at a prior sample and at the prior mean rescaled to the expected sample norm (the original
     K-best used the raw prior mean, which the decoder never saw in training); best of K prior samples for
     K = 4, 8, 12, 16.
Output: results/plan_reeval.json
"""
import json
import os

import numpy as np
import torch

from ae_gate import CKPT, Data
from common import DATA, RES
from cvae_plan import Post, Prior
from plan_gist import K, LAYERS, Decoder, PlanEnc, PlanPred, kbest, load_stats, pack, round_ok, tree_stats

dev = "cuda"
cache = os.path.join(DATA, "cache")
stats = load_stats(cache)
W = torch.load(os.path.join(cache, "lm_head.pt")).to(dev)
D = W.shape[1]
fmean, fstd = stats[-1][0].to(dev), stats[-1][1].to(dev)
g = torch.Generator(device=dev).manual_seed(0)
KS = [4, 8, 12, 16]


def logp(h):
    return ((h.float() * fstd + fmean).to(torch.bfloat16) @ W.T).float().log_softmax(-1)


def rounds_of(sp):
    dt = Data(cache, sp, LAYERS, K, stats)
    T1 = np.load(os.path.join(cache, f"{sp}_meta.npz"))["top1"].astype(np.int64)
    st = np.arange(len(dt.seq))
    st = st[round_ok(dt.seq, st, 2 * K)]
    keep, last = [], -10 ** 9
    for s in st:
        if s >= last + 2 * K:
            keep.append(s)
            last = s
    return dt, T1, np.array(keep)


def acc1(lp, lab):
    return ((lp.argmax(-1) == lab).int().cumprod(1).sum(1) + 1).float().cpu()


def add_tree(ex, name, tok, lab):
    for nm, v in zip(["best", "nodes"], [tree_stats(tok.cpu(), lab.cpu())[i] for i in (0, 3)]):
        ex.setdefault(f"{name}_{nm}", []).append(v)


def sample_trees(ex, name, zfun, dec, oo, raw, lab, B):
    for Kb in KS:
        zk = zfun(Kb)
        hk = dec(zk, oo.repeat_interleave(Kb, 0), raw.repeat_interleave(Kb, 0))
        tk = torch.cat([logp(x).argmax(-1) for x in hk.split(128)]).view(B, Kb, K)
        add_tree(ex, f"{name}{Kb}", tk, lab)


res = {}
for tag in ["plan_pw_G8c0", "plan_pw_G8c16s0.25", "plan_pw_G32c16s0.25", "cvae_G32c16b0.3", "cvae_G32c16b0.1", "cvae_G8c16b0.3"]:
    ck = torch.load(os.path.join(CKPT, f"{tag}.pt"), map_location="cpu")
    a = ck["args"]
    G, c = a["G"], a["c"]
    cv = tag.startswith("cvae")
    dec = Decoder(G, c, D).to(dev).eval()
    dec.load_state_dict(ck["dec"])
    if cv:
        q, p = Post(G, D, c).to(dev).eval(), Prior(G, D, c).to(dev).eval()
        q.load_state_dict(ck["post"])
        p.load_state_dict(ck["prior"])
    elif c:
        E, P = PlanEnc(G, D, c).to(dev).eval(), PlanPred(G, c, D).to(dev).eval()
        E.load_state_dict(ck["E"])
        P.load_state_dict(ck["P"])
    res[tag] = {}
    for sp in ["dev_math", "dev_code", "dev_chat"]:
        dt, T1, rounds = rounds_of(sp)
        out, ex = {}, {}
        with torch.no_grad():
            for i in range(0, len(rounds), 32):
                s = rounds[i:i + 32]
                raw, raw_a, ids, m, lab, oo, _, _ = [x.to(dev) for x in pack(dt, T1, s, s, G)]
                B = len(s)
                with torch.autocast("cuda", torch.bfloat16):
                    if not c:
                        lp = logp(dec(torch.zeros(B, G, 1, device=dev), oo, raw))
                        out.setdefault("one_draft", []).append(acc1(lp, lab))
                        for Kb in KS:
                            add_tree(ex, f"kbest{Kb}", kbest(lp, Kb), lab)
                    elif not cv:
                        sig = a["sigma"]
                        ph = P(raw_a).float()
                        php = ph * torch.rsqrt(ph.pow(2).mean((1, 2), keepdim=True) + 1e-6)
                        nz = lambda: sig * torch.randn(ph.shape, device=dev, generator=g)
                        for kind, y in {"det_raw": ph, "det_unitpower": php, "det_unitpower_noise": php + nz()}.items():
                            out.setdefault(kind, []).append(acc1(logp(dec(y, oo, raw)), lab))
                        out.setdefault("power_det", []).append(ph.pow(2).mean((1, 2)).cpu())
                    else:
                        mp, lv = p(raw)
                        sd = (0.5 * lv).exp()
                        nm = ((mp.pow(2) + sd.pow(2)).sum((1, 2), keepdim=True) / mp.pow(2).sum((1, 2), keepdim=True)).sqrt()
                        zs = mp + sd * torch.randn(mp.shape, device=dev, generator=g)
                        for kind, y in {"prior_mean": mp, "prior_mean_normmatched": mp * nm, "prior_sample": zs}.items():
                            lp = logp(dec(y, oo, raw))
                            out.setdefault(kind, []).append(acc1(lp, lab))
                            if kind != "prior_mean":
                                for Kb in KS:
                                    add_tree(ex, f"kbest{Kb}_at_{kind}", kbest(lp, Kb), lab)
                        sample_trees(ex, "prior", lambda Kb: mp.repeat_interleave(Kb, 0) + sd.repeat_interleave(Kb, 0)
                                     * torch.randn((B * Kb,) + mp.shape[1:], device=dev, generator=g), dec, oo, raw, lab, B)
        r = {"rounds": int(len(rounds))}
        r.update({k: torch.cat(v).mean().item() for k, v in out.items()})
        r.update({k: torch.cat(v).mean().item() for k, v in ex.items()})
        res[tag][sp] = r
        print(tag, sp, json.dumps({k: round(v, 3) for k, v in r.items()}), flush=True)
    json.dump(res, open(os.path.join(RES, "plan_reeval.json"), "w"), indent=1)
print("=== reeval done", flush=True)
