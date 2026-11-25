"""Qualitative examples for the write-up: what Gister's sampled futures look like, round by round.

For a few benchmark prompts, runs the sampled-tree loop (K samples + v2's chain) and records per round: the text so
far (tail), each drafted continuation, how many of its tokens the target accepted, and the text actually kept.
Output: results/examples.json
Run:    python examples.py [--s2 stage2_gister_flow_big] [--k 8] [--rounds 12]
"""
import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoTokenizer, DynamicCache

from common import RES, eval_prompts
from dflash_eval import TARGET_ID, chat, load_target
from e2e import GistDrafter, build_tree, keep_path


@torch.inference_mode()
def run(t, tok, dr, ids, rounds):
    n0 = ids.shape[1]
    cache = DynamicCache()
    dr.reset()
    o = t(ids, past_key_values=cache, use_cache=True, logits_to_keep=1, output_hidden_states=True)
    dr.observe(o.hidden_states, n0)
    bonus = o.logits[0, -1].argmax()
    out, start, log = [], n0, []
    for _ in range(rounds):
        d = dr.draft(bonus, start)
        d = d if d.dim() == 2 else d[None]
        toks, par, dep, kids = build_tree(bonus, d)
        n = len(toks)
        anc = np.zeros((n, n), bool)
        for i in range(n):
            if par[i] >= 0:
                anc[i] = anc[par[i]]
            anc[i, i] = True
        mask = torch.ones(1, 1, n, start + n, dtype=torch.bool)
        mask[0, 0, :, start:] = torch.from_numpy(anc)
        o = t(torch.tensor([toks], device="cuda"), attention_mask=mask.cuda(),
              position_ids=torch.tensor([[start + x for x in dep]], device="cuda"),
              past_key_values=cache, use_cache=True, output_hidden_states=True)
        post = o.logits[0].argmax(-1).tolist()
        path, cur = [0], 0
        while post[cur] in kids[cur]:
            cur = kids[cur][post[cur]]
            path.append(cur)
        kept = [toks[i] for i in path]
        drafts = []
        for row in d.tolist():                         # accepted length of each individual draft
            a = 0
            for j, x in enumerate(row):
                if j + 1 < len(path) and toks[path[j + 1]] == x:
                    a += 1
                else:
                    break
            drafts.append({"text": tok.decode(row), "accepted": a})
        log.append({"context_tail": tok.decode((ids[0].tolist() + out)[-40:]), "bonus": tok.decode([int(bonus)]),
                    "drafts": drafts, "kept": tok.decode(kept), "kept_tokens": len(kept), "tree_nodes": n})
        out += kept
        bonus = torch.tensor(post[cur], device="cuda")
        idx = torch.tensor(path, device="cuda")
        keep_path(cache, start, idx)
        start += len(path)
        dr.observe(o.hidden_states, None, idx)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2", default="stage2_gister_flow_big")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=12)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(TARGET_ID)
    t = load_target()
    dr = GistDrafter(t, s2=args.s2, k_samples=args.k, det_s2="stage2_gister_v2")
    dr.main = dict(dr.main, nfe=2)
    res = {"args": vars(args), "examples": []}
    for s in ("gsm8k", "mtbench", "humaneval"):
        for text in eval_prompts(s, 2):
            log = run(t, tok, dr, chat(tok, text), args.rounds)
            res["examples"].append({"set": s, "prompt": text, "rounds": log})
            print(s, "mean kept", round(float(np.mean([r["kept_tokens"] for r in log])), 2), flush=True)
    json.dump(res, open(os.path.join(RES, "examples.json"), "w"), indent=1, ensure_ascii=False)
    print("=== examples done", flush=True)


if __name__ == "__main__":
    main()
