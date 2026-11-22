"""Check that train.py's packed block forward matches decode.py's cached inference path.

Arm D: packed blocks (pass 1 with e_mask, pass 2 soft self-conditioning) vs decode.block_propose.
Arm C: packed soft block vs gen_softtail's chunk [x_last | H hard | S soft] with gate=H+1.
Differences should be bf16 layout noise only (same argmax almost everywhere).
Run: python consistency_check.py --dckpt dbg_d [--cckpt dbg_c]
"""
import argparse
import os

import torch

from common import CKPT, DATA
from decode import block_propose, to_topk
from model import DRAFTER, TARGET, QwenLM
from train import pack_mask, topk_probs


def check_d(ck, seq, t, B=8):
    d = QwenLM(DRAFTER)
    d.load_trainable(os.path.join(CKPT, ck + ".pt"))
    ids = torch.tensor(seq, device="cuda")
    # inference path
    c = d.get_cache()
    d.step(c, ids[: t - 1], graph=False)
    c.n = t - 1
    inf = block_propose(d, c, seq[: t + 1], B, iters=2, inp="soft")  # pend = [x_{t-1}, x_t]: ctx x_{t-1}, slot0 x_t
    # packed path, block at position t
    _, kvs = d.full(ids[None], collect_kv=True)
    st = torch.tensor([[t]], device="cuda")
    pos = (st[:, :, None] + torch.arange(B, device="cuda")).view(1, B)
    mask = pack_mask(st, B, len(ids))
    first = (torch.arange(B, device="cuda") == 0)[None, :, None]
    x0 = d.rows32(ids[t:t + 1])[None]
    with torch.no_grad():
        e1 = torch.where(first, x0, d.e_mask.expand(1, B, -1))
        lg1 = d.logits(d.packed(e1, pos, kvs, mask, ckpt=False)[0])
        gi, gp = topk_probs(lg1)
        e2 = torch.where(first, x0, torch.roll(d.embed_soft(gi, gp), 1, 0)[None] + d.e_flag)
        lg2 = d.logits(d.packed(e2, pos, kvs, mask, ckpt=False)[0])
    print(f"[D] max|logit diff|={(inf - lg2).abs().max().item():.3f}  argmax agree={(inf.argmax(-1) == lg2.argmax(-1)).float().mean().item():.3f}")


def check_c(ck, seq, t, H=4, S=8):
    from train import c_embed
    m = QwenLM(TARGET)
    m.load_trainable(os.path.join(CKPT, ck + ".pt"))
    ids = torch.tensor(seq, device="cuda")
    g = torch.Generator(device="cuda").manual_seed(0)
    gi = torch.randint(0, m.V, (S, 64), device="cuda", generator=g)
    gp = torch.softmax(torch.randn(S, 64, device="cuda", generator=g) * 3, -1)
    fresh = torch.zeros(S, dtype=torch.bool, device="cuda")
    fresh[S - 2:] = True
    # inference: prefix x_<t in cache, chunk = [x_t | x_{t+1..t+H} hard | S soft]
    c = m.get_cache()
    m.step(c, ids[:t], graph=False)
    c.n = t
    soft = torch.where(fresh[:, None], m.e_mask.float(), m.embed_soft(gi, gp)) + m.e_flag
    emb = torch.cat([m.rows32(ids[t:t + H + 1]), soft])
    inf = m.step(c, embeds=emb, gate=H + 1)[H + 1:]
    # packed: soft block starts at p0 = t+H+1
    _, kvs = m.full(ids[None], collect_kv=True)
    st = torch.tensor([[t + H + 1]], device="cuda")
    pos = (st[:, :, None] + torch.arange(S, device="cuda")).view(1, S)
    with torch.no_grad():
        e = c_embed(m, gi[None], gp[None], fresh[None], "soft")
        pk = m.logits(m.packed(e, pos, kvs, pack_mask(st, S, len(ids)), ckpt=False)[0])
    print(f"[C] max|logit diff|={(inf - pk).abs().max().item():.3f}  argmax agree={(inf.argmax(-1) == pk.argmax(-1)).float().mean().item():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dckpt")
    ap.add_argument("--cckpt")
    ap.add_argument("--data", default="fake")
    a = ap.parse_args()
    it = torch.load(os.path.join(DATA, a.data + ".pt"))[0]
    seq = it["prompt"] + it["resp"]
    t = len(it["prompt"]) + 20
    if a.dckpt:
        check_d(a.dckpt, seq, t)
    if a.cckpt:
        check_c(a.cckpt, seq, t)
