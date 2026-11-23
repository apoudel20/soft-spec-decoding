"""Sketches B-E on a frozen DFlash drafter (Sketch A is dflash_plan.py).

All variants add learned vectors to the 16 slot embeddings of a DFlash block. Each vector is produced from the
target's hidden states at the last verified position (or, for D, from a latent generated out of noise), with
zero-initialised projections and a per-slot norm cap at the median token-embedding norm, so training starts
exactly at plain DFlash. Only the new heads are trained; target and drafter are frozen.

  B        three heads read layers 9, 18 and 28 and steer slots 1-5, 6-10 and 11-15 respectively.
  C        four alternative plans from layers 18 and 20; trained winner-take-all (loss of the best plan);
           evaluated as best of four, which is what verifying four candidate blocks would accept.
  D        a small flow-matching denoiser generates a plan latent from noise in 2 Euler steps, conditioned on
           layers 18 and 20, trained toward the target's true future summary (mean layer-20 state over the next
           64 positions); a plan head maps the latent to a vector added to every slot.
  E_cnn    coarse to fine: one code for the whole block from layer 9, one per 4-slot segment from layer 18,
           one per slot from layer 28 (shallow layers steer the coarse level).
  E_probe  the same with the assignment flipped: coarse from 28, segments from 18, slots from 9.

Run:  python sketches.py train --sketch B      (C, D, E_cnn, E_probe)
      python sketches.py eval --sketches none B C D E_cnn E_probe
      python sketches.py smoke                  (CPU only: tiny random target and drafter, checks every path)
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import CKPT, DATA, RES

CTX_LAYERS = [1, 9, 17, 25, 33]
NEED = [9, 18, 20, 28]
B = 16
ORACLE_N = 64
K_C = 4
SEGS = [(1, 5), (5, 9), (9, 13), (13, 16)]


def cap(v, enorm):
    return enorm * v / v.norm(dim=-1, keepdim=True).clamp_min(1.0)


def zero_linear(i, o):
    l = nn.Linear(i, o)
    nn.init.zeros_(l.weight)
    nn.init.zeros_(l.bias)
    return l


class Head(nn.Module):
    """LayerNorm then a zero-initialised linear map."""

    def __init__(self, din, dout):
        super().__init__()
        self.ln = nn.LayerNorm(din, elementwise_affine=False)
        self.proj = zero_linear(din, dout)

    def forward(self, z):
        return self.proj(self.ln(z.float()))


class SketchB(nn.Module):
    def __init__(self, H, enorm):
        super().__init__()
        self.enorm = enorm
        self.heads = nn.ModuleDict({str(L): Head(H, H) for L in (9, 18, 28)})

    def forward(self, st, s):
        out = torch.zeros(1, B, self.heads["9"].proj.out_features, device=st[9].device)
        for L, (a, b) in ((9, (1, 6)), (18, (6, 11)), (28, (11, 16))):
            out[0, a:b] = cap(self.heads[str(L)](st[L][s - 1]), self.enorm)
        return out, None


class SketchC(nn.Module):
    def __init__(self, H, enorm):
        super().__init__()
        self.enorm, self.H = enorm, H
        self.head = Head(2 * H, K_C * H)
        nn.init.normal_(self.head.proj.weight, std=1e-3)  # break symmetry between the K plans

    def forward(self, st, s):
        z = torch.cat([st[18][s - 1], st[20][s - 1]])
        v = cap(self.head(z).view(K_C, self.H), self.enorm)
        return v[:, None, :].expand(K_C, B, self.H), None


class SketchD(nn.Module):
    def __init__(self, H, enorm, hid=1024):
        super().__init__()
        self.enorm, self.H = enorm, H
        self.cond = nn.Sequential(nn.LayerNorm(2 * H, elementwise_affine=False), nn.Linear(2 * H, hid))
        self.den = nn.Sequential(nn.Linear(H + hid + 1, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, H))
        self.ln_t = nn.LayerNorm(H, elementwise_affine=False)
        self.plan = Head(H, H)

    def velocity(self, zt, tt, c):
        return self.den(torch.cat([zt, c, tt[..., None]], -1))

    def sample(self, c, steps=2):
        z = torch.randn(self.H, device=c.device)
        for i in range(steps):
            tt = torch.tensor(i / steps, device=c.device)
            z = z + self.velocity(z, tt, c) / steps
        return z

    def forward(self, st, s, target_future=None):
        c = self.cond(torch.cat([st[18][s - 1], st[20][s - 1]]).float())
        aux = None
        if target_future is not None:      # flow matching toward the normalised true future summary
            z1 = self.ln_t(target_future.float())
            z0 = torch.randn_like(z1)
            tt = torch.rand((), device=z1.device)
            zt = (1 - tt) * z0 + tt * z1
            aux = F.mse_loss(self.velocity(zt, tt, c), z1 - z0)
            z = z1                            # teacher-forced latent for the plan head during training
        else:
            z = self.sample(c)
        v = cap(self.plan(z), self.enorm)
        return v[None, None, :].expand(1, B, self.H), aux


class SketchE(nn.Module):
    def __init__(self, H, enorm, assign):
        super().__init__()
        self.enorm, self.H = enorm, H
        self.L = {"cnn": {"coarse": 9, "seg": 18, "tok": 28}, "probe": {"coarse": 28, "seg": 18, "tok": 9}}[assign]
        self.coarse = Head(H, H)
        self.seg = nn.ModuleList([Head(H, H) for _ in SEGS])
        self.tok = Head(H, H)
        self.slot_scale = nn.Parameter(torch.ones(B, H) * 0.0)   # per-slot modulation of the token-level code

    def forward(self, st, s):
        out = self.coarse(st[self.L["coarse"]][s - 1])[None].expand(B, self.H).clone()
        segv = torch.zeros(B, self.H, device=out.device)
        for (a, b), h in zip(SEGS, self.seg):
            segv[a:b] = h(st[self.L["seg"]][s - 1])
        tokv = self.tok(st[self.L["tok"]][s - 1])[None] * (1 + self.slot_scale)
        out = out + segv + tokv
        out[0] = 0
        return cap(out, self.enorm)[None], None


def build(name, H, enorm):
    return {"B": lambda: SketchB(H, enorm), "C": lambda: SketchC(H, enorm), "D": lambda: SketchD(H, enorm),
            "E_cnn": lambda: SketchE(H, enorm, "cnn"), "E_probe": lambda: SketchE(H, enorm, "probe")}[name]()


@torch.no_grad()
def target_states(t, ids):
    hs = t(input_ids=ids[None], output_hidden_states=True).hidden_states
    ctx = torch.cat([hs[l + 1][0] for l in CTX_LAYERS], dim=-1)
    st = {L: hs[L + 1][0].float() for L in NEED}
    return ctx, st


def block_logits(t, d, ids, ctx, s, add):
    """add: (K, 16, H) slot offsets or None. Returns logits (K, 15, V)."""
    K = 1 if add is None else add.shape[0]
    blk = torch.full((K, B), d.mask_token_id, dtype=torch.long, device=ids.device)
    blk[:, 0] = ids[s]
    emb = t.model.embed_tokens(blk)
    if add is not None:
        emb = emb + add.to(emb.dtype)
    pos = torch.arange(0, s + B, device=ids.device)[None].expand(K, -1)
    h = d(target_hidden=ctx[None, :s].expand(K, -1, -1), noise_embedding=emb, position_ids=pos, is_causal=False)
    return t.lm_head(h[:, 1:, :]).float()


def starts_for(n_prompt, L, k, rng):
    lo, hi = n_prompt + 1, L - B - ORACLE_N // 4
    return rng.sample(range(lo, hi), min(k, hi - lo)) if hi > lo else []


def train(args, t, d, items, enorm):
    H = d.config.hidden_size
    m = build(args.sketch, H, enorm).to(items[0][0].device if isinstance(items[0], tuple) else "cuda")
    dev = next(m.parameters()).device
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=0.0)
    total = len(items) * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1, (i + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(1, i / total))))
    rng, log, t0, step = random.Random(0), [], time.time(), 0
    for ep in range(args.epochs):
        for ids, n_prompt in items:
            ctx, st = target_states(t, ids)
            opt.zero_grad(set_to_none=True)
            tot, nb = 0.0, 0
            for s in starts_for(n_prompt, len(ids), args.blocks, rng):
                fut = st[20][s + 1:s + 1 + ORACLE_N].mean(0) if args.sketch == "D" else None
                add, aux = m(st, s, fut) if args.sketch == "D" else m(st, s)
                lg = block_logits(t, d, ids, ctx, s, add)
                lab = ids[s + 1:s + B]
                ce = torch.stack([F.cross_entropy(lg[k], lab) for k in range(lg.shape[0])])
                loss = ce.min() if args.sketch == "C" else ce.mean()
                if aux is not None:
                    loss = loss + aux
                (loss / args.blocks).backward()
                tot += loss.item()
                nb += 1
            opt.step()
            sched.step()
            step += 1
            log.append(tot / max(nb, 1))
            if step % 25 == 0 or step == total:
                print(f"step {step}/{total}  loss {sum(log[-25:]) / len(log[-25:]):.4f}  {(time.time() - t0) / 60:.1f} min", flush=True)
    return m, log


@torch.no_grad()
def evaluate(args, t, d, sets, heads):
    res = {}
    for s_name, seqs in sets.items():
        acc = {p: [] for p in heads}
        for ids, n_prompt in seqs:
            if len(ids) < n_prompt + B + ORACLE_N // 4 + 2:
                continue
            ctx, st = target_states(t, ids)
            for s in range(n_prompt + 1, len(ids) - B - ORACLE_N // 4, args.stride):
                lab = ids[s + 1:s + B]
                for p, h in heads.items():
                    add = None if h is None else h(st, s)[0]
                    pred = block_logits(t, d, ids, ctx, s, add).argmax(-1)
                    a = (pred == lab[None]).int().cumprod(1).sum(1).max().item() + 1   # best of K (K=1 except C)
                    acc[p].append(a)
        res[s_name] = {p: {"blocks": len(v), "mean_accepted": sum(v) / max(len(v), 1)} for p, v in acc.items()}
        print(s_name, {p: round(r["mean_accepted"], 3) for p, r in res[s_name].items()}, flush=True)
    return res


def load_real():
    from dflash_eval import load_drafter, load_target
    t, d = load_target(), load_drafter()
    for p in d.parameters():
        p.requires_grad_(False)
    enorm = t.model.embed_tokens.weight.float().norm(dim=1).median().item()
    return t, d, enorm


def tiny_models():
    """Random tiny target (36 layers, width 64) and DFlash drafter on CPU, for smoke tests only."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from dflash_model import DFlashDraftModel
    torch.manual_seed(0)
    tc = Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=36, num_attention_heads=2,
                     num_key_value_heads=1, head_dim=32, max_position_embeddings=1024)
    t = Qwen3ForCausalLM(tc).eval()
    dc = Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=2,
                     num_key_value_heads=1, head_dim=32, max_position_embeddings=1024)
    dc.block_size, dc.num_target_layers = B, 36
    dc.dflash_config = {"mask_token_id": 151669, "target_layer_ids": CTX_LAYERS}
    dc.layer_types = ["full_attention"] * 2
    dc._attn_implementation = "sdpa"
    d = DFlashDraftModel(dc).eval()
    for p in list(d.parameters()) + list(t.parameters()):
        p.requires_grad_(False)
    return t, d, t.model.embed_tokens.weight.norm(dim=1).median().item()


def real_items(name):
    xs = torch.load(os.path.join(DATA, name))
    return [(torch.tensor(x["prompt"] + x["resp"], device="cuda"), len(x["prompt"])) for x in xs]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval", "smoke"])
    ap.add_argument("--sketch", choices=["B", "C", "D", "E_cnn", "E_probe"], default="B")
    ap.add_argument("--sketches", nargs="+", default=["none", "B", "C", "D", "E_cnn", "E_probe"])
    ap.add_argument("--n_train", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sets", nargs="+", default=["math", "code", "chat"])
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--out", default="sketches_eval")
    args = ap.parse_args()
    if args.mode == "smoke":
        t, d, enorm = tiny_models()
        g = torch.Generator().manual_seed(1)
        fake = [(torch.randint(0, 151000, (140,), generator=g), 20) for _ in range(2)]
        heads = {"none": None}
        for sk in ["B", "C", "D", "E_cnn", "E_probe"]:
            args.sketch, args.blocks, args.epochs = sk, 2, 1
            m, log = train(args, t, d, fake, enorm)
            assert all(math.isfinite(x) for x in log), (sk, log)
            m.eval()
            heads[sk] = m
            print(f"smoke train {sk}: ok, loss {log[-1]:.3f}", flush=True)
        args.stride = 40
        r = evaluate(args, t, d, {"fake": fake}, heads)
        print("SMOKE OK", flush=True)
    elif args.mode == "train":
        t, d, enorm = load_real()
        items = [x for x in real_items("q3_train.pt") if len(x[0]) - x[1] > B + ORACLE_N]
        random.Random(0).shuffle(items)
        m, log = train(args, t, d, items[: args.n_train], enorm)
        torch.save({"sketch": args.sketch, "state": m.state_dict(), "enorm": enorm, "loss": log},
                   os.path.join(CKPT, f"sketch_{args.sketch}.pt"))
        print("saved", flush=True)
    else:
        t, d, enorm = load_real()
        heads = {}
        for p in args.sketches:
            if p == "none":
                heads[p] = None
                continue
            ck = torch.load(os.path.join(CKPT, f"sketch_{p}.pt"))
            m = build(p, d.config.hidden_size, ck["enorm"]).cuda()
            m.load_state_dict(ck["state"])
            heads[p] = m.eval()
        sets = {s: real_items(f"q3_dev_{s}_40.pt") for s in args.sets}
        res = {"args": vars(args), "sets": evaluate(args, t, d, sets, heads)}
        json.dump(res, open(os.path.join(RES, args.out + ".json"), "w"), indent=1)
        print("wrote", os.path.join(RES, args.out + ".json"))
