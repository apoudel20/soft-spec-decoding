"""How exactly does the drafter fail?

Also records, per token, the expected acceptance of the base drafter under T=1 rejection sampling
(sum_v min(p, q) at that position), to compare with greedy exact-match hit rates by category.
 Token-level failure analysis plus a temperature sweep.

Runs offline on the target's own greedy outputs for the eval prompts (data/ref_<set>_<n>.pt, made by
arm0.py). The target's greedy token is the truth: a draft is accepted iff it equals it.

Two views:
  1. Teacher-forced (the "knowledge gap"): at every response position, the drafter sees the true
     prefix and predicts the next token. Drafters: the base 0.5B (autoregressive), and the slot-0
     prediction of a block drafter checkpoint.
  2. Free-running block (the "coherence gap"): a block drafter drafts B tokens at once (K passes);
     we record the first offset where it fails and what token it got wrong.

For every prediction we record the target token, the drafter token, their categories
(whitespace, punctuation, markdown/latex, number, function word, content word), the target's
probability of the drafter's token, the target's own top-1/top-2 margin, and the context.

Temperature sweep: expected per-token acceptance under standard rejection sampling,
sum_v min(p_T(v), q_T(v)), with both models at temperature T, teacher-forced (top-256 support).

Outputs: results/failures_summary.json and results/failures_examples.jsonl (every prediction).
Run: python failures.py --sets gsm8k:40 mtbench:40 humaneval:40 --block d3_soft
"""
import argparse
import json
import os
import re
import string

import torch

from common import CKPT, DATA, RES
from model import DRAFTER, TARGET, QwenLM, load_tokenizer
from train import pack_mask, topk_probs

TOPN = 256
TEMPS = [0.3, 0.5, 0.7, 1.0, 1.3]
FUNC = set("""a an the of to in on at by for with from as is are was were be been being it its this that these those
and or but if then so than not no do does did have has had i you he she we they me my your our their his her them
which who whom what when where why how there here can could will would should may might must shall also just very
into over under about after before between each all any some such only own same both""".split())
MD_CHARS = set("*#`|>_~")


def category(s):
    if s.strip() == "":
        return "newline" if "\n" in s else "whitespace"
    t = s.strip()
    if any(c in MD_CHARS for c in t) or t.startswith("\\") or "$" in t or t in ("-", "--", "---", "\\(", "\\["):
        return "markdown/latex"
    if re.search(r"\d", t):
        return "number"
    if all(c in string.punctuation for c in t):
        return "punctuation"
    if t.lower() in FUNC:
        return "function word"
    return "content word"


@torch.no_grad()
def topn_logits(m, ids):
    hn, _ = m.full(ids[None])
    vals, idx = [], []
    for c in hn[0].split(256):
        lg = m.logits(c)
        v, i = lg.topk(TOPN, -1)
        vals.append(v.half().cpu())
        idx.append(i.int().cpu())
    return torch.cat(vals), torch.cat(idx)  # row p predicts token p+1


def overlap(tv, ti, dv, di, T):
    """sum_v min(p_T, q_T) over the union of two top-N supports (rows aligned)."""
    pt = torch.softmax(tv.float() / T, -1)
    pd = torch.softmax(dv.float() / T, -1)
    out = []
    for r in range(pt.shape[0]):
        a = dict(zip(ti[r].tolist(), pt[r].tolist()))
        s = 0.0
        for k, q in zip(di[r].tolist(), pd[r].tolist()):
            if k in a:
                s += min(a[k], q)
        out.append(s)
    return out


@torch.no_grad()
def block_preds(d, ids, starts, B, iters, inp):
    """Free-running block drafts, one block per start: (nb, B) drafted ids for offsets 1..B."""
    _, kvs = d.full(ids[None], collect_kv=True)
    nb = len(starts)
    st = starts[None]
    pos = (st[:, :, None] + torch.arange(B, device="cuda")).view(1, nb * B)
    mask = pack_mask(st, B, len(ids))
    first = (torch.arange(nb * B, device="cuda") % B == 0)[None, :, None]
    x_in = d.rows32(ids[pos[0]])[None]
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
    return lg.argmax(-1).view(nb, B)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["gsm8k:40", "mtbench:40", "humaneval:40"])
    ap.add_argument("--block", default="d3_soft")
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=60, help="characters of context kept per example")
    args = ap.parse_args()
    tok = load_tokenizer()
    dec = lambda i: tok.decode([int(i)])
    seqs = {}
    for s in args.sets:
        name, _, n = s.partition(":")
        seqs[name] = torch.load(os.path.join(DATA, f"ref_{name}_{n}.pt"))

    # target top-N logits at every position (then free the target)
    t = QwenLM(TARGET)
    t.use_graphs = False
    tl = {k: [topn_logits(t, torch.tensor(s["prompt"] + s["resp"], device="cuda")) for s in v] for k, v in seqs.items()}
    del t
    torch.cuda.empty_cache()

    d = QwenLM(DRAFTER)
    d.use_graphs = False
    dl_base = {k: [topn_logits(d, torch.tensor(s["prompt"] + s["resp"], device="cuda")) for s in v] for k, v in seqs.items()}
    meta = d.load_trainable(os.path.join(CKPT, args.block + ".pt"))
    inp = meta.get("inp", "soft")

    rows = []
    summary = {"args": vars(args), "sets": {}}
    for name, ss in seqs.items():
        temp_acc = {"base": {T: [] for T in TEMPS}}
        for si, s in enumerate(ss):
            full = s["prompt"] + s["resp"]
            r = len(s["prompt"])
            ids = torch.tensor(full, device="cuda")
            tv, ti = tl[name][si]
            dv, di = dl_base[name][si]
            # block drafter slot-0 (teacher-forced next token) and free-running blocks
            starts = torch.arange(r, len(full) - args.B - 1, device="cuda") if len(full) - args.B - 1 > r else None
            bp = block_preds(d, ids, starts, args.B, args.K, inp) if starts is not None else None
            for T in TEMPS:
                temp_acc["base"][T] += overlap(tv[r - 1:-1], ti[r - 1:-1], dv[r - 1:-1], di[r - 1:-1], T)
            text = tok.decode(full[r:])
            pre_chars = [len(tok.decode(full[r:p])) for p in range(r, len(full))]
            for p in range(r, len(full)):
                x = full[p]
                pt = torch.softmax(tv[p - 1].float(), -1)
                tgt_top1, tgt_top2 = ti[p - 1][0].item(), ti[p - 1][1].item()
                tmap = dict(zip(ti[p - 1].tolist(), pt.tolist()))
                margin = pt[0].item() - pt[1].item()
                ctx = text[max(0, pre_chars[p - r] - args.ctx):pre_chars[p - r]]
                preds = {"base": di[p - 1][0].item()}
                if bp is not None and p - 1 - r < len(starts) and p - 1 >= r:
                    preds["block_slot0"] = bp[p - 1 - r, 0].item()  # block starting at x_{p-1}
                rs = overlap(tv[p - 1:p], ti[p - 1:p], dv[p - 1:p], di[p - 1:p], 1.0)[0]  # base drafter, T=1
                for who, g in preds.items():
                    rows.append({"rs_accept": round(rs, 4) if who == "base" else None,"set": name, "seq": si, "pos": p - r, "who": who, "view": "teacher_forced",
                                 "ok": g == x, "target": dec(x), "draft": dec(g), "cat_target": category(dec(x)),
                                 "cat_draft": category(dec(g)), "p_target_of_draft": round(tmap.get(g, 0.0), 4),
                                 "target_top1_prob": round(pt[0].item(), 4), "target_margin": round(margin, 4),
                                 "target_is_greedy": tgt_top1 == x, "ctx": ctx})
            # free-running: first failure per block
            if bp is not None:
                tgt = torch.tensor([[full[min(t0 + k + 1, len(full) - 1)] for k in range(args.B)] for t0 in starts.tolist()])
                okm = (bp.cpu() == tgt)
                for bi, t0 in enumerate(starts.tolist()):
                    L = int(okm[bi].int().cumprod(0).sum())
                    if L < args.B:
                        p = t0 + L + 1
                        g = bp[bi, L].item()
                        x = full[p]
                        pt = torch.softmax(tv[p - 1].float(), -1)
                        tmap = dict(zip(ti[p - 1].tolist(), pt.tolist()))
                        rows.append({"set": name, "seq": si, "pos": p - r, "who": f"{args.block}:K{args.K}",
                                     "view": "free_running", "offset": L + 1, "ok": False, "target": dec(x),
                                     "draft": dec(g), "cat_target": category(dec(x)), "cat_draft": category(dec(g)),
                                     "p_target_of_draft": round(tmap.get(g, 0.0), 4),
                                     "target_top1_prob": round(pt[0].item(), 4),
                                     "target_margin": round(pt[0].item() - pt[1].item(), 4),
                                     "ctx": text[max(0, pre_chars[p - r] - args.ctx):pre_chars[p - r]]})
        summary["sets"][name] = {"temp_expected_acceptance_base": {str(T): sum(v) / len(v) for T, v in temp_acc["base"].items()}}

    # aggregate
    def agg(sel):
        out = {}
        cats = sorted({r["cat_target"] for r in sel})
        n = len(sel)
        for c in cats:
            rc = [r for r in sel if r["cat_target"] == c]
            fails = [r for r in rc if not r["ok"]]
            out[c] = {"share_of_tokens": len(rc) / max(1, n), "n": len(rc), "accuracy": 1 - len(fails) / max(1, len(rc)),
                      "share_of_failures": len(fails) / max(1, sum(not r["ok"] for r in sel))}
            rsv = [r["rs_accept"] for r in rc if r.get("rs_accept") is not None]
            if rsv:  # expected acceptance under T=1 rejection sampling vs greedy exact match, same tokens
                out[c]["rs_accept_T1"] = sum(rsv) / len(rsv)
                rsf = [r["rs_accept"] for r in fails if r.get("rs_accept") is not None]
                out[c]["rs_accept_T1_on_greedy_misses"] = sum(rsf) / len(rsf) if rsf else None
        fails = [r for r in sel if not r["ok"]]
        if fails:
            out["_failure_flavour"] = {
                "same_word_diff_spacing_or_case": sum(f["target"].strip().lower() == f["draft"].strip().lower() for f in fails) / len(fails),
                "draft_has_target_prob_ge_0.2": sum(f["p_target_of_draft"] >= 0.2 for f in fails) / len(fails),
                "target_near_tie_margin_lt_0.2": sum(f["target_margin"] < 0.2 for f in fails) / len(fails),
                "target_confident_top1_ge_0.9": sum(f["target_top1_prob"] >= 0.9 for f in fails) / len(fails),
            }
        out["_overall_accuracy"] = 1 - len(fails) / max(1, n)
        return out

    for name in seqs:
        for who in sorted({r["who"] for r in rows if r["set"] == name}):
            sel = [r for r in rows if r["set"] == name and r["who"] == who]
            summary["sets"][name][who] = agg(sel)
    json.dump(summary, open(os.path.join(RES, "failures_summary.json"), "w"), indent=1)
    with open(os.path.join(RES, "failures_examples.jsonl"), "w", encoding="utf8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({k: {w: (v[w]["_overall_accuracy"], v[w].get("_failure_flavour")) for w in v if w != "temp_expected_acceptance_base"} for k, v in summary["sets"].items()}, indent=1))
    print(json.dumps({k: v["temp_expected_acceptance_base"] for k, v in summary["sets"].items()}, indent=1))
    print("rows", len(rows))


if __name__ == "__main__":
    main()
