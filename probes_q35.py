"""Scale probe: does Qwen3.5-9B know its own future better than Qwen3.5-0.8B, in any layer?

Same probes as probes.py (token t+k, mean-embedding summary of the next 32/128 tokens, remaining
length, all linear, with the corrected ridge), but swept over every second layer of both models.
The text is the 9B's own greedy output (gen_q35.py). The 9B runs in 4-bit NF4 (bitsandbytes), the
0.8B in bf16, both through HF transformers. The summary space is the 0.8B's input embedding matrix,
which is the same for both models' probes.

Run:    python probes_q35.py --n_train 600 --per_seq 24
Output: results/probes_q35.json
"""
import argparse
import json
import os
import random
import time

import torch
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, Qwen3_5ForCausalLM

from common import DATA, RES
from probes import KS, NS, VOCAB, r2, ridge, sample_positions, targets, token_probe

TARGET_ID, SMALL_ID = "Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-0.8B"


def load(mid):
    if mid == TARGET_ID:
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        m = Qwen3_5ForCausalLM.from_pretrained(mid, quantization_config=q, dtype=torch.bfloat16, device_map="cuda")
    else:
        m = Qwen3_5ForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda")
    del m.lm_head  # only hidden states are needed
    torch.cuda.empty_cache()
    m.eval()
    return m


@torch.no_grad()
def features(m, seqs, positions, layers):
    by = {}
    for si, t in positions:
        by.setdefault(si, []).append(t)
    feats = {li: [] for li in layers}
    order = []
    for si, ts in by.items():
        s = seqs[si]
        ids = torch.tensor(s["prompt"] + s["resp"], device="cuda")[None]
        hs = m.model(input_ids=ids, output_hidden_states=True).hidden_states  # hs[i] = after layer i, last is final-normed
        tt = torch.tensor(ts, device="cuda")
        for li in layers:
            feats[li].append(hs[li][0, tt].float().cpu())
        order += [(si, t) for t in ts]
    return {li: torch.cat(v) for li, v in feats.items()}, order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=600)
    ap.add_argument("--per_seq", type=int, default=24)
    ap.add_argument("--sets", nargs="+", default=["gsm8k:40", "mtbench:40", "humaneval:40"])
    ap.add_argument("--models", nargs="+", default=[SMALL_ID, TARGET_ID])
    ap.add_argument("--layer_step", type=int, default=2)
    ap.add_argument("--out", default="probes_q35")
    args = ap.parse_args()
    rng = random.Random(0)
    t0 = time.time()
    train = [it for it in torch.load(os.path.join(DATA, "q35_train.pt")) if len(it["resp"]) > 40]
    rng.shuffle(train)
    train = train[: args.n_train]
    tests = {}
    for s in args.sets:
        name, _, n = s.partition(":")
        tests[name] = torch.load(os.path.join(DATA, f"q35_ref_{name}_{n}.pt"))
    cnt = {}
    for it in train:
        for x in it["resp"]:
            cnt[x] = cnt.get(x, 0) + 1
    vmap = {x: i for i, x in enumerate(sorted(cnt, key=cnt.get, reverse=True)[:VOCAB])}
    tr_pos = sample_positions(train, args.per_seq, rng)
    te_pos = {k: sample_positions(v, 0, rng) for k, v in tests.items()}
    small = load(SMALL_ID)
    E_ref = small.model.embed_tokens.weight.float().cpu()  # shared summary space for both models
    del small
    torch.cuda.empty_cache()

    res = {"args": vars(args), "n_train_seqs": len(train), "n_train_positions": len(tr_pos),
           "n_test_seqs": {k: len(v) for k, v in tests.items()}, "models": {}}
    for mid in args.models:
        m = load(mid)
        nl = len(m.model.layers)
        layers = sorted(set(list(range(args.layer_step, nl + 1, args.layer_step)) + [nl]))
        print(f"== {mid}: {nl} layers, probing {layers}, GPU {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)
        Ftr, otr = features(m, train, tr_pos, layers)
        Ttr = targets(train, otr, vmap, E_ref)
        Fte = {k: features(m, v, te_pos[k], layers) for k, v in tests.items()}
        del m
        torch.cuda.empty_cache()
        out = {}
        for li in layers:
            lname = f"layer{li}of{nl}"
            out[lname] = {}
            for sname, (Fs, ote) in Fte.items():
                Tte = targets(tests[sname], ote, vmap, E_ref)
                r = {}
                for k in KS:
                    acc, _ = token_probe(Ftr[li], Ttr[0][k], Fs[li], Tte[0][k])
                    r[f"token+{k}"] = round(acc, 4)
                for n in NS:
                    p = ridge(Ftr[li], Ttr[1][n], Fs[li])
                    r[f"bag+{n}_r2"] = round(r2(p, Tte[1][n], Ttr[1][n].mean(0)), 4)
                p = ridge(Ftr[li], Ttr[2][:, None], Fs[li])[:, 0]
                r["remaining_r2"] = round(r2(p, Tte[2], Ttr[2].mean()), 4)
                out[lname][sname] = r
            print(mid, lname, {s: (v["token+2"], v["token+8"], v["bag+32_r2"], v["bag+128_r2"]) for s, v in out[lname].items()},
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
        res["models"][mid] = out
        json.dump(res, open(os.path.join(RES, args.out + ".json"), "w"), indent=1)
    base = {}
    Ttr = targets(train, tr_pos, vmap, E_ref)
    for sname, seqs in tests.items():
        Tte = targets(seqs, te_pos[sname], vmap, E_ref)
        b = {}
        for k in KS:
            y = Tte[0][k]
            b[f"token+{k}_unigram"] = round((y[y >= 0] == 0).float().mean().item(), 4)
        for n in NS:
            p = ridge(Ttr[3], Ttr[1][n], Tte[3])
            b[f"bag+{n}_r2_embedding_only"] = round(r2(p, Tte[1][n], Ttr[1][n].mean(0)), 4)
        p = ridge(Ttr[3], Ttr[2][:, None], Tte[3])[:, 0]
        b["remaining_r2_embedding_only"] = round(r2(p, Tte[2], Ttr[2].mean()), 4)
        base[sname] = b
        print("baseline", sname, b, flush=True)
    res["baselines"] = base
    res["minutes"] = (time.time() - t0) / 60
    json.dump(res, open(os.path.join(RES, args.out + ".json"), "w"), indent=1)
    print("wrote", os.path.join(RES, args.out + ".json"))


if __name__ == "__main__":
    main()
