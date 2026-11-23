"""Zero-training ceiling: can DFlash use future information if it arrives through its own context channel?

Sketch A added a plan vector to the block's input embeddings, and even an oracle plan did not help. DFlash was
trained to read the target through a different channel: the target's hidden states (layers 1, 9, 17, 25, 33) at
every context position become extra keys and values in every drafter layer. Here we append ONE extra context
position whose features are the mean of the target's true future features, with no training at all:
  fut16 : mean over the next 16 positions (roughly the block being drafted),
  fut64 : mean over the next 64 positions (a longer-horizon gist).
The extra position is given the rotary position of the last real context token. If acceptance rises, the channel
can carry future information and the next step is learning to predict that position. If not, plans cannot help
this drafter through either channel.

Run:    python kv_oracle.py
Output: results/kv_oracle.json
"""
import argparse
import json
import os

import torch

from common import DATA, RES
from dflash_eval import load_drafter, load_target
from dflash_plan import B, CTX_LAYERS, ORACLE_N


@torch.no_grad()
def ctx_feats(t, ids):
    hs = t(input_ids=ids[None], output_hidden_states=True).hidden_states
    return torch.cat([hs[l + 1][0] for l in CTX_LAYERS], dim=-1)


@torch.no_grad()
def accepted(t, d, ids, ctx, s, extra):
    blk = torch.full((1, B), d.mask_token_id, dtype=torch.long, device="cuda")
    blk[0, 0] = ids[s]
    emb = t.model.embed_tokens(blk)
    if extra is None:
        th = ctx[None, :s]
        pos = torch.arange(0, s + B, device="cuda")[None]
    else:
        th = torch.cat([ctx[:s], extra[None].to(ctx.dtype)])[None]
        pos = torch.cat([torch.arange(0, s, device="cuda"), torch.tensor([s - 1], device="cuda"),
                         torch.arange(s, s + B, device="cuda")])[None]
    h = d(target_hidden=th, noise_embedding=emb, position_ids=pos, is_causal=False)
    pred = t.lm_head(h[:, 1:, :]).argmax(-1)[0]
    return int((pred == ids[s + 1:s + B]).int().cumprod(0).sum()) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["math", "code", "chat"])
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="kv_oracle")
    args = ap.parse_args()
    t, d = load_target(), load_drafter()
    res = {"args": vars(args), "sets": {}}
    for s_name in args.sets:
        seqs = torch.load(os.path.join(DATA, f"q3_dev_{s_name}_40.pt"))[: args.n]
        acc = {"none": [], "fut16": [], "fut64": []}
        for it in seqs:
            ids = torch.tensor(it["prompt"] + it["resp"], device="cuda")
            n0 = len(it["prompt"])
            if len(ids) < n0 + B + ORACLE_N // 4 + 2:
                continue
            ctx = ctx_feats(t, ids)
            for s in range(n0 + 1, len(ids) - B - ORACLE_N // 4, args.stride):
                acc["none"].append(accepted(t, d, ids, ctx, s, None))
                acc["fut16"].append(accepted(t, d, ids, ctx, s, ctx[s + 1:s + 1 + 16].float().mean(0)))
                acc["fut64"].append(accepted(t, d, ids, ctx, s, ctx[s + 1:s + 1 + ORACLE_N].float().mean(0)))
        res["sets"][s_name] = {k: {"blocks": len(v), "mean_accepted": sum(v) / len(v)} for k, v in acc.items()}
        res["sets"][s_name]["per_block"] = acc
        print(s_name, {k: round(sum(v) / len(v), 3) for k, v in acc.items()}, flush=True)
        json.dump(res, open(os.path.join(RES, args.out + ".json"), "w"), indent=1)
    print("wrote", os.path.join(RES, args.out + ".json"))


if __name__ == "__main__":
    main()
