"""Cache the frozen DFlash drafter's block outputs, aligned with the stage-1 state cache, for a DFlash-backed stage 2.

For a cache chunk starting at row r (answer a, response index j, target state at absolute position q = n0 - 1 + j),
DFlash is run with block start s = q: context = the target's states at positions < q (layers 1, 9, 17, 25, 33),
block = [token q, mask x 15]. Its outputs at block slots 1..8 are its guesses of the target's final states for
positions q..q+7, i.e. they draft tokens q+1..q+8, the same 8 tokens the stage-1 gist latent of chunk r decodes to.
So DFlash's first 8 drafted tokens and stage 2's 8 drafted tokens are the same positions.

All blocks of one answer go through DFlash in one packed forward (each block attends to its own context prefix
and its own 16 slots). The first answer of each split is checked against the one-block-at-a-time reference.

Rows: every valid chunk start on dev, every 4th on train.
Output: data/dflash_feats/<split>_h.npy (N, 8, 4096) fp16 and <split>_rows.npy (N,) cache rows.
Run:    python dflash_feats.py [--smoke]
"""
import argparse
import os
import time

import numpy as np
import torch

from cache_states import splits
from common import DATA
from dflash_eval import load_drafter, load_target
from dflash_plan import CTX_LAYERS
from kv_oracle_train import logits_with_token

K, BS = 8, 16


def packed(t, d, ids, th, starts):
    """DFlash outputs (len(starts), 8, D) for block starts `starts`, all in one forward."""
    L, n = len(ids), len(starts)
    blk = torch.full((n, BS), d.mask_token_id, dtype=torch.long, device="cuda")
    blk[:, 0] = ids[starts]
    emb = t.model.embed_tokens(blk.view(1, -1))
    st = torch.as_tensor(starts, device="cuda")
    pos = torch.cat([torch.arange(L, device="cuda"), (st[:, None] + torch.arange(BS, device="cuda")).view(-1)])[None]
    qb = torch.arange(n, device="cuda").repeat_interleave(BS)                 # block id of each query
    ctx_ok = torch.arange(L, device="cuda")[None] < st[qb][:, None]          # (nq, L)
    own = qb[:, None] == qb[None]                                            # (nq, nq)
    mask = torch.cat([ctx_ok, own], 1)[None, None]
    h = emb
    pe = d.rotary_emb(h, pos)
    for layer in d.layers:
        h = layer(hidden_states=h, target_hidden=th[None], attention_mask=mask, position_ids=pos,
                  position_embeddings=pe)
    h = d.norm(h)[0].view(n, BS, -1)
    return h[:, 1:1 + K]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--train_stride", type=int, default=4)
    args = ap.parse_args()
    out = os.path.join(DATA, "dflash_feats" + ("_smoke" if args.smoke else ""))
    os.makedirs(out, exist_ok=True)
    t, d = load_target(), load_drafter()
    W = t.lm_head.weight
    t0 = time.time()
    for name, items in splits(False).items():
        meta = np.load(os.path.join(DATA, "cache", f"{name}_meta.npz"))
        seq, top1 = meta["seq"], meta["top1"]
        if args.smoke:
            items = items[:3]
        stride = args.train_stride if name == "train" else 1
        sel, o = [], 0                           # (answer, j, row) for every selected chunk start
        for a, it in enumerate(items):
            m = len(it["resp"])
            assert (seq[o:o + m] == a).all()
            sel += [(a, j, o + j) for j in range(0, m - K + 1, stride)]
            o += m
        rows = np.array([r for _, _, r in sel])
        np.save(os.path.join(out, f"{name}_rows.npy"), rows)
        H = np.lib.format.open_memmap(os.path.join(out, f"{name}_h.npy"), "w+", np.float16, (len(sel), K, W.shape[1]))
        agree, tot, i0 = 0, 0, 0
        by_a = {}
        for idx, (a, j, r) in enumerate(sel):
            by_a.setdefault(a, []).append((idx, j, r))
        for a, lst in by_a.items():
            it = items[a]
            ids = torch.tensor(it["prompt"] + it["resp"], device="cuda")
            n0 = len(it["prompt"])
            with torch.no_grad():
                hs = t(input_ids=ids[None], output_hidden_states=True).hidden_states
                ctx = torch.cat([hs[l + 1][0] for l in CTX_LAYERS], dim=-1)
                th = d.hidden_norm(d.fc(ctx))
                starts = np.array([n0 - 1 + j for _, j, _ in lst])
                for c in range(0, len(lst), 256):
                    h = packed(t, d, ids, th, starts[c:c + 256])
                    H[[x[0] for x in lst[c:c + 256]]] = h.to(torch.float16).cpu().numpy()
                    tok = (h @ W.T).argmax(-1).cpu().numpy()
                    lab = top1[np.array([x[2] for x in lst[c:c + 256]])[:, None] + np.arange(K)]
                    agree += (tok == lab).sum()
                    tot += lab.size
                if a == 0:                       # packed forward must match the one-block reference
                    for s in starts[[0, len(starts) // 2, -1]]:
                        ref = logits_with_token(t, d, ids, ctx, int(s), None)[:K].argmax(-1)
                        got = (packed(t, d, ids, th, np.array([s]))[0] @ W.T).argmax(-1)
                        mine = (packed(t, d, ids, th, starts)[list(starts).index(s)] @ W.T).argmax(-1)
                        print("check s", int(s), "ref==single", (ref == got).float().mean().item(),
                              "ref==packed", (ref == mine).float().mean().item(), flush=True)
            if a % 200 == 0:
                print(name, a, len(items), f"slot agree {agree / max(tot, 1):.3f}", f"{(time.time() - t0) / 60:.1f} min",
                      flush=True)
        H.flush()
        print("done", name, len(sel), f"per-slot top-1 agreement {agree / tot:.4f}", f"{(time.time() - t0) / 60:.1f} min",
              flush=True)


if __name__ == "__main__":
    main()
