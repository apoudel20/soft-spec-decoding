"""Run decoding methods on prompt sets, batch size 1, and write aggregate metrics to results/<out>.json.

Method specs (comma-separated key=val after a colon):
  ar
  sd:g=5                                   classic SD, Qwen2.5-0.5B drafts g tokens
  jacobi:H=8                               frozen-target Jacobi (Arm A)
  softjacobi:H=4,S=8                       training-free soft-tail Jacobi, frozen target (Arm A variant)
  C:ckpt=c_soft,H=4,S=8[,tail=attached|detached][,stale=reset][,sharpen=0.5]
                                           Arm C (soft; default tail=detached); inp=hard for C-hard
  D:ckpt=d3_soft,B=8,K=2[,inp=soft]        Arm D block drafter + frozen target
  CD:ckpt=c_soft,dckpt=d3_soft,H=4,S=8,K=2 Arm C+D
Sets: mtbench:N, gsm8k:N, humaneval:N, dev_chat:N, dev_math:N, dev_code:N

Example: python eval.py --sets gsm8k:5 --methods ar "sd:g=5" "jacobi:H=8" --out smoke
"""
import argparse
import json
import os
import time

import torch

from common import CKPT, RES, chat_ids, dev_prompts, eval_prompts
from decode import Stats, gen_ar, gen_block, gen_sd, gen_softtail
from model import DRAFTER, TARGET, QwenLM, load_tokenizer


def parse(spec):
    name, _, rest = spec.partition(":")
    kv = {}
    for part in filter(None, rest.split(",")):
        k, _, v = part.partition("=")
        kv[k] = v
    return name, kv


def load_set(s, tok, max_prompt=400):
    name, _, n = s.partition(":")
    n = int(n) if n else None
    texts = dev_prompts(name[4:], n or 40) if name.startswith("dev_") else eval_prompts(name, n)
    out = []
    for t in texts:
        ids = chat_ids(tok, t)
        if len(ids) > max_prompt:  # keep the KV cache small; count skipped prompts
            continue
        out.append(ids)
    return name, out, len(texts)


class Models:
    def __init__(self):
        self.t = QwenLM(TARGET)
        self.d = None
        self.t_ck = self.d_ck = None

    def drafter(self, ck=None):
        if self.d is None:
            self.d = QwenLM(DRAFTER)
        if ck and ck != self.d_ck:
            self.d.load_trainable(os.path.join(CKPT, ck + ".pt"))
            self.d_ck = ck
        return self.d

    def target(self, ck=None):
        if ck and ck != self.t_ck:
            self.t.load_trainable(os.path.join(CKPT, ck + ".pt"))
            self.t_ck = ck
        return self.t


def run_method(M, spec, prompt, max_new, temp, gen):
    name, kv = parse(spec)
    i = lambda k, d: int(kv.get(k, d))
    if name == "ar":
        return gen_ar(M.target(), prompt, max_new, temp, gen)
    if name == "sd":
        return gen_sd(M.target(), M.drafter(), prompt, max_new, i("g", 5), temp, gen)
    if name == "jacobi":
        return gen_softtail(M.target(), prompt, max_new, H=i("H", 8), S=0, temp=temp, gen=gen)
    if name == "softjacobi":
        return gen_softtail(M.target(), prompt, max_new, H=i("H", 4), S=i("S", 8), temp=temp,
                            sharpen=float(kv.get("sharpen", 1.0)), stale=kv.get("stale", "reuse"),
                            tail=kv.get("tail", "attached"), gen=gen)
    if name in ("C", "CD"):
        dm = M.drafter(kv["dckpt"]) if name == "CD" else None
        return gen_softtail(M.target(kv["ckpt"]), prompt, max_new, H=i("H", 4), S=i("S", 8), temp=temp,
                            inp=kv.get("inp", "soft"), lora=True, flag=True, stale=kv.get("stale", "reuse"),
                            sharpen=float(kv.get("sharpen", 1.0)), dm=dm, d_iters=i("K", 2),
                            d_inp=kv.get("dinp", "soft"), tail=kv.get("tail", "detached"), gen=gen)
    if name == "D":
        return gen_block(M.target(), M.drafter(kv["ckpt"]), prompt, max_new, B=i("B", 8), iters=i("K", 2),
                         inp=kv.get("inp", "soft"), temp=temp, gen=gen)
    raise ValueError(spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = load_tokenizer()
    M = Models()
    gen = torch.Generator(device="cuda")
    res = {"args": vars(args), "sets": {}, "time": time.strftime("%Y-%m-%d %H:%M")}
    def captures():
        return getattr(M.t, "n_captures", 0) + (getattr(M.d, "n_captures", 0) if M.d is not None else 0)

    for s in args.sets:
        sname, prompts, n_total = load_set(s, tok)
        # Warm-up (untimed): capture the CUDA graphs each method needs on two dev prompts, so
        # graph-capture time does not land inside timed runs. Remaining captures are reported.
        for wp in [chat_ids(tok, t) for t in dev_prompts("chat", 2)]:
            for spec in ["ar"] + args.methods:
                for t in ({0.0, args.temp}):
                    run_method(M, spec, wp, 96, t, gen.manual_seed(0))
        print(f"== {sname}: {len(prompts)} prompts ({n_total - len(prompts)} skipped as too long)", flush=True)
        agg = {}
        ref = []
        for pi, p in enumerate(prompts):
            gen.manual_seed(args.seed * 100003 + pi)
            r = gen_ar(M.target(), p, args.max_new, args.temp, gen, record_gap=(args.temp == 0))
            ref.append(r)
        ar_time = sum(r["time"] for r in ref)
        ar_tok = sum(len(r["tokens"]) for r in ref)
        agg["ar"] = {"tau": 1.0, "time": ar_time, "tokens": ar_tok, "tok_per_s": ar_tok / ar_time, "speedup": 1.0}
        for spec in args.methods:
            if spec == "ar":
                continue
            st, tt, ntok, exact, gaps = Stats(), 0.0, 0, 0, []
            cap0 = captures()
            for pi, p in enumerate(prompts):
                gen.manual_seed(args.seed * 100003 + pi + 7)
                r = run_method(M, spec, p, args.max_new, args.temp, gen)
                st.merge(r["stats"])
                tt += r["time"]
                ntok += len(r["tokens"])
                if args.temp == 0:
                    a, b = r["tokens"], ref[pi]["tokens"]
                    if a == b:
                        exact += 1
                    else:
                        k = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
                        gaps.append(ref[pi]["stats"].gaps[k] if k < len(ref[pi]["stats"].gaps) else None)
            nmax = 16
            cond, unc = st.per_pos(nmax)
            row = {"tau": st.tau(), "rounds": st.rounds, "drafter_passes_per_round": st.dpasses / max(1, st.rounds),
                   "time": tt, "tokens": ntok, "tok_per_s": ntok / tt,
                   "speedup": (ar_time / tt) if args.temp == 0 else (ntok / tt) / (ar_tok / ar_time),
                   "acc_cond": [round(x, 4) for x in cond if x == x], "acc_uncond": [round(x, 4) for x in unc],
                   "graph_captures_in_timed_runs": captures() - cap0}
            if args.temp == 0:
                row["exact_match"] = f"{exact}/{len(prompts)}"
                row["divergence_gaps"] = [round(g, 3) if g is not None else None for g in gaps]
            agg[spec] = row
            print(f"  {spec:45s} tau={row['tau']:.3f} speedup={row['speedup']:.2f}x tok/s={row['tok_per_s']:.1f}"
                  + (f" exact={row['exact_match']} gaps={row['divergence_gaps']}" if args.temp == 0 else ""), flush=True)
        res["sets"][sname] = {"n": len(prompts), "n_total": n_total, "methods": agg}
        json.dump(res, open(os.path.join(RES, args.out + ".json"), "w"), indent=1)
    print("wrote", os.path.join(RES, args.out + ".json"))


if __name__ == "__main__":
    main()
