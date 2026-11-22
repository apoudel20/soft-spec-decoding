"""Arm 0 diagnostic: per-position draft accuracy, teacher-forced vs free-running.

Runs offline on the target's own greedy outputs for the eval prompts. For greedy decoding, a draft
at offset k is accepted iff it equals the target's greedy token, so these sequences are the truth.

For each drafter and block offset k = 1..B, over block starts t in the response:
  a_k   teacher-forced accuracy: the in-block prefix x_{t+1..t+k-1} is the true one
  b_k   free-running P(L >= k): the whole block is drafted at once (L = accepted length)
  c_k   P(L >= k | L >= k-1): conditional acceptance under free running
  ind_k a_1 * ... * a_k: what b_k would be if errors were independent with the teacher-forced rates
Reading (PLAN.md): high a but collapsing c means incoherent joint drafts; low a means conditioning
or capacity is the bottleneck.

Drafters:
  base         Qwen2.5-0.5B, autoregressive. Free-running with all earlier drafts correct is
               teacher forcing, so b_k = P(k consecutive teacher-forced hits) exactly.
  <ckpt>[:K]   Arm D block drafter checkpoint, free-running with K iterations (default 1 and 2).
               Teacher forcing feeds the true tokens as hard guesses (E[x] + e_flag).
Run: python arm0.py --sets gsm8k:40 mtbench:40 humaneval:40 --drafters base d3_soft --out arm0
"""
import argparse
import json
import os

import torch

from common import CKPT, DATA, RES, chat_ids, eval_prompts
from decode import gen_ar
from model import DRAFTER, TARGET, QwenLM, load_tokenizer
from train import pack_mask, topk_probs


def ref_sequences(sname, n, tok, max_new=256):
    path = os.path.join(DATA, f"ref_{sname}_{n}.pt")
    if os.path.exists(path):
        return torch.load(path)
    t = QwenLM(TARGET)
    out = []
    for text in eval_prompts(sname, n):
        p = chat_ids(tok, text)
        if len(p) > 400:
            continue
        r = gen_ar(t, p, max_new, 0)
        out.append({"prompt": p, "resp": r["tokens"]})
    torch.save(out, path)
    del t
    torch.cuda.empty_cache()
    return out


def summarize(tf_hits, L, B):
    """tf_hits (n, B) bool teacher-forced hits per start; L (n,) free-running accepted length."""
    a = tf_hits.float().mean(0).tolist()
    b = [(L >= k).float().mean().item() for k in range(1, B + 1)]
    c = [b[0]] + [b[k] / b[k - 1] if b[k - 1] > 0 else float("nan") for k in range(1, B)]
    ind, acc = [], 1.0
    for x in a:
        acc *= x
        ind.append(acc)
    return {"a_teacher_forced": a, "b_P_L_ge_k": b, "c_conditional": c, "independent_pred": ind,
            "mean_L": L.float().mean().item(), "n_starts": int(L.numel())}


@torch.no_grad()
def diag_base(d, seqs, B, stride):
    hits, Ls = [], []
    for s in seqs:
        ids = torch.tensor(s["prompt"] + s["resp"], device="cuda")
        hn, _ = d.full(ids[None])
        pred = torch.cat([d.logits(c).argmax(-1) for c in hn[0].split(256)])  # pred[p] = guess for x_{p+1}
        ok = pred[:-1] == ids[1:]  # ok[p] -> x_{p+1} predicted correctly from the true prefix
        r = len(s["prompt"])
        for t in range(r, len(ids) - B - 1, stride):
            h = ok[t:t + B]
            hits.append(h)
            Ls.append(int(h.int().cumprod(0).sum()))
    return summarize(torch.stack(hits), torch.tensor(Ls), B)


@torch.no_grad()
def diag_block(d, seqs, B, stride, iters, inp):
    hits, Ls = [], []
    for s in seqs:
        ids = torch.tensor(s["prompt"] + s["resp"], device="cuda")
        r = len(s["prompt"])
        starts = torch.arange(r, len(ids) - B - 1, stride, device="cuda")
        if len(starts) == 0:
            continue
        _, kvs = d.full(ids[None], collect_kv=True)
        nb = len(starts)
        st = starts[None]
        pos = (st[:, :, None] + torch.arange(B, device="cuda")).view(1, nb * B)
        mask = pack_mask(st, B, len(ids))
        tgt = ids[(pos + 1).clamp_max(len(ids) - 1)][0].view(nb, B)  # slot i should predict x_{t+i+1}
        first = (torch.arange(nb * B, device="cuda") % B == 0)[None, :, None]
        x_in = d.rows32(ids[pos[0]])[None]  # true token at each slot position
        # teacher forced: slots hold the true tokens as hard guesses
        e_tf = torch.where(first, x_in, x_in + d.e_flag)
        p_tf = d.logits(d.packed(e_tf, pos, kvs, mask, ckpt=False)[0]).argmax(-1).view(nb, B)
        # free running: pass 1 with e_mask, then iters-1 refinement passes
        e = torch.where(first, x_in, d.e_mask.expand_as(x_in))
        lg = d.logits(d.packed(e, pos, kvs, mask, ckpt=False)[0])
        for _ in range(1, iters):
            if inp == "soft":
                gi, gp = topk_probs(lg)
                g = d.embed_soft(gi, gp)
            else:
                g = d.rows32(lg.argmax(-1))
            e = torch.where(first, x_in, torch.roll(g, 1, dims=0)[None] + d.e_flag)
            lg = d.logits(d.packed(e, pos, kvs, mask, ckpt=False)[0])
        p_fr = lg.argmax(-1).view(nb, B)
        hits.append(p_tf == tgt)
        Ls.append((p_fr == tgt).int().cumprod(1).sum(1))
    return summarize(torch.cat(hits), torch.cat(Ls).cpu(), B)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--drafters", nargs="+", default=["base"])
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok = load_tokenizer()
    data = {}
    for s in args.sets:
        name, _, n = s.partition(":")
        data[name] = ref_sequences(name, int(n), tok)
    d = QwenLM(DRAFTER)
    d.use_graphs = False
    res = {"args": vars(args), "sets": {k: len(v) for k, v in data.items()}, "results": {}}
    for spec in args.drafters:
        ck, _, its = spec.partition(":")
        if ck != "base":
            meta = d.load_trainable(os.path.join(CKPT, ck + ".pt"))
            inp = meta.get("inp", "soft")
        for name, seqs in data.items():
            if ck == "base":
                res["results"][f"{name}/base"] = r = diag_base(d, seqs, args.B, args.stride)
                print(f"{name:10s} base       a={[round(x, 3) for x in r['a_teacher_forced']]} b={[round(x, 3) for x in r['b_P_L_ge_k']]} meanL={r['mean_L']:.2f}", flush=True)
                continue
            for K in ([int(its)] if its else [1, 2]):
                key = f"{name}/{ck}:K{K}"
                res["results"][key] = r = diag_block(d, seqs, args.B, args.stride, K, inp)
                print(f"{name:10s} {ck}:K{K}  a={[round(x, 3) for x in r['a_teacher_forced']]} b={[round(x, 3) for x in r['b_P_L_ge_k']]} c={[round(x, 3) for x in r['c_conditional']]} meanL={r['mean_L']:.2f}", flush=True)
        json.dump(res, open(os.path.join(RES, args.out + ".json"), "w"), indent=1)
    print("wrote", os.path.join(RES, args.out + ".json"))


if __name__ == "__main__":
    main()
