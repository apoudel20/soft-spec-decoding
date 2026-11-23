"""Breakdown of the generated Qwen3-8B data: sources, prompt and answer lengths, cap hits, chat request types.

Source is recovered from the prompt text (math prompts end with the GSM8K suffix, code prompts with the MBPP
template). Chat request type is a rough rule on the request's opening words, reported as such.
Run:    python data_stats.py
Output: results/data_stats.json
"""
import json
import os
import re

import numpy as np
import torch
from transformers import AutoTokenizer

from common import DATA, RES

CAP = 320
TYPES = [("Write or create", r"^(write|create|compose|draft|generate|develop|design|produce|craft|build|make)\b"),
         ("Explain or describe", r"^(explain|describe|discuss|analy[sz]e|summari[sz]e|elaborate|outline|compare|evaluate|examine|provide|give|list|tell)\b"),
         ("Question", r"^(what|how|why|who|which|when|where|is|are|can|could|do|does|did|should|would|will)\b"),
         ("Rewrite or edit", r"^(rewrite|paraphrase|translate|edit|revise|rephrase|convert|continue|transfer|generate according)"),
         ]


def user_text(tok, ids):
    s = tok.decode(ids)
    m = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", s, re.S)
    return (m.group(1) if m else s).strip()


def kind(text):
    if text.endswith("put your final answer within \\boxed{}."):
        return "math"
    if text.endswith("Reply with the code in a Python code block."):
        return "code"
    return "chat"


def chat_type(text):
    first = text.strip().split("\n")[0].lower()
    for name, pat in TYPES:
        if re.search(pat, first):
            return name
    return "Other (context first, then a request)"


def summarize(v):
    v = np.array(v)
    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)), "p10": float(np.quantile(v, 0.1)),
            "p90": float(np.quantile(v, 0.9)), "min": int(v.min()), "max": int(v.max())}


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    files = {"train": "q3_train.pt", "train_more": "q3_train_more.pt",
             "dev_math": "q3_dev_math_40.pt", "dev_code": "q3_dev_code_40.pt", "dev_chat": "q3_dev_chat_40.pt"}
    out = {"cap": CAP, "sets": {}}
    bins = list(range(0, CAP + 20, 20))
    for name, f in files.items():
        p = os.path.join(DATA, f)
        if not os.path.exists(p):
            out["sets"][name] = None
            continue
        items = torch.load(p)
        by = {}
        for it in items:
            t = user_text(tok, it["prompt"])
            k = kind(t)
            d = by.setdefault(k, {"prompt": [], "resp": [], "types": {}, "examples": []})
            d["prompt"].append(len(it["prompt"]))
            d["resp"].append(len(it["resp"]))
            if k == "chat":
                ct = chat_type(t)
                d["types"][ct] = d["types"].get(ct, 0) + 1
            if len(d["examples"]) < 4:
                body = t.replace("\nPlease reason step by step, and put your final answer within \\boxed{}.", "")
                body = body.replace("\nWrite a Python function for this. Reply with the code in a Python code block.", "")
                d["examples"].append({"prompt": body[:220], "answer": tok.decode(it["resp"])[:260],
                                      "answer_tokens": len(it["resp"])})
        s = {}
        for k, d in by.items():
            r = np.array(d["resp"])
            s[k] = {"count": len(r), "prompt_tokens": summarize(d["prompt"]), "answer_tokens": summarize(r),
                    "total_answer_tokens": int(r.sum()), "frac_at_cap": float((r >= CAP - 1).mean()),
                    "hist": np.histogram(np.minimum(r, CAP - 1), bins=bins)[0].tolist(), "types": d["types"],
                    "examples": d["examples"]}
        out["sets"][name] = s
        print(name, {k: (v["count"], round(v["answer_tokens"]["mean"]), round(v["frac_at_cap"], 3)) for k, v in s.items()}, flush=True)
    out["bins"] = bins
    json.dump(out, open(os.path.join(RES, "data_stats.json"), "w"), indent=1)
    print("wrote results/data_stats.json")


if __name__ == "__main__":
    main()
