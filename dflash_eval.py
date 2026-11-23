"""Sketch A, step 1: the DFlash baseline on Qwen3-8B, with an exactness check.

Target Qwen3-8B (4-bit NF4), drafter z-lab/Qwen3-8B-DFlash-b16 (bf16, vendored in dflash_model.py), greedy.
For each held-out dev prompt we produce
  ref  : the target's own greedy output (HF generate),
  spec : DFlash speculative output (block of 16, one denoising pass, chain verification),
and report accepted tokens per target pass (tau), wall-clock tokens per second for both, and whether the two
outputs are token-identical. Any mismatch is reported with the target's top-1/top-2 logit gap at the first
differing position, since 4-bit kernels can flip near-ties between a 1-token and a 16-token forward.

Run:    python dflash_eval.py --n 10 --max_new 160
Output: results/dflash_eval.json
"""
import argparse
import json
import os
import time

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Qwen3Config

from common import RES, dev_prompts
from dflash_model import DFlashDraftModel

TARGET_ID = "Qwen/Qwen3-8B"
DRAFT_ID = "z-lab/Qwen3-8B-DFlash-b16"


def load_target():
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    t = AutoModelForCausalLM.from_pretrained(TARGET_ID, quantization_config=q, dtype=torch.bfloat16, device_map="cuda")
    return t.eval()


def load_drafter():
    from huggingface_hub import snapshot_download
    path = snapshot_download(DRAFT_ID, allow_patterns=["config.json", "model.safetensors"])
    cfg = Qwen3Config.from_pretrained(path)
    cfg._attn_implementation = "sdpa"
    d = DFlashDraftModel(cfg)
    sd = load_file(os.path.join(path, "model.safetensors"))
    missing, unexpected = d.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert not [m for m in missing if "rotary" not in m], missing
    return d.to("cuda", torch.bfloat16).eval()


def chat(tok, text):
    s = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(s, add_special_tokens=False, return_tensors="pt").input_ids.cuda()


@torch.no_grad()
def gap_at(t, ids, pos):
    lg = t(ids[:, :pos]).logits[0, -1].float()
    top = lg.topk(2).values
    return round(float(top[0] - top[1]), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max_new", type=int, default=160)
    ap.add_argument("--sets", nargs="+", default=["math", "code", "chat"])
    ap.add_argument("--out", default="dflash_eval")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(TARGET_ID)
    t = load_target()
    d = load_drafter()
    print(f"loaded, GPU {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)
    stop = [tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")]
    res = {"args": vars(args), "sets": {}}
    # warm-up so CUDA and bnb kernels are initialised before timing
    w = chat(tok, "Say hi.")
    t.generate(w, max_new_tokens=8, do_sample=False, temperature=None, top_p=None, top_k=None)
    d.spec_generate(t, w, 8, stop, 0.0)
    for s in args.sets:
        rows = []
        for text in dev_prompts(s, args.n):
            ids = chat(tok, text)
            torch.cuda.synchronize(); a = time.perf_counter()
            ref = t.generate(ids, max_new_tokens=args.max_new, do_sample=False, temperature=None, top_p=None, top_k=None,
                             eos_token_id=stop)[0, ids.shape[1]:]
            torch.cuda.synchronize(); b = time.perf_counter()
            out, acc = d.spec_generate(t, ids, args.max_new, stop, 0.0)
            torch.cuda.synchronize(); c = time.perf_counter()
            spec = out[0, ids.shape[1]:]
            n = min(len(ref), len(spec))
            same = bool(torch.equal(ref[:n], spec[:n]))
            row = {"ref_len": len(ref), "spec_len": len(spec), "identical": same, "tau": sum(acc) / len(acc),
                   "blocks": len(acc), "ar_s": b - a, "spec_s": c - b}
            if not same:
                k = int((ref[:n] != spec[:n]).nonzero()[0])
                row["first_diff"] = k
                row["gap"] = gap_at(t, torch.cat([ids[0], ref[:k]])[None], ids.shape[1] + k)
            rows.append(row)
        tok_ar = sum(r["ref_len"] for r in rows) / sum(r["ar_s"] for r in rows)
        tok_sp = sum(r["spec_len"] for r in rows) / sum(r["spec_s"] for r in rows)
        res["sets"][s] = {"n": len(rows), "tau": sum(r["tau"] * r["blocks"] for r in rows) / sum(r["blocks"] for r in rows),
                          "ar_tok_s": tok_ar, "spec_tok_s": tok_sp, "speedup": tok_sp / tok_ar,
                          "identical": sum(r["identical"] for r in rows), "rows": rows}
        r = res["sets"][s]
        print(f"{s}: tau {r['tau']:.2f}  AR {tok_ar:.1f} tok/s  DFlash {tok_sp:.1f} tok/s  speedup {r['speedup']:.2f}x  "
              f"identical {r['identical']}/{r['n']}  gaps {[x.get('gap') for x in rows if 'gap' in x]}", flush=True)
        json.dump(res, open(os.path.join(RES, args.out + ".json"), "w"), indent=1)
    print("wrote", os.path.join(RES, args.out + ".json"))


if __name__ == "__main__":
    main()
