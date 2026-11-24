"""Split-design stage 1: can a VAE compress the target's next 8 tokens into one small latent and decode them exactly?

Input: the HF target's own top-1 tokens at 8 consecutive response positions (from the state cache meta), embedded
with the target's frozen input embedding. Encoder: small transformer -> nq latent tokens (zdim total). Decoder:
small transformer -> 8 vectors -> frozen LM head. Loss: cross-entropy on the 8 tokens + beta * KL with a per-dim
floor, latent dropout and posterior noise (CALM recipe), so the latent is smooth enough for stage-2 diffusion.

Gate (the stage-1 bar, fixed before running): exact token agreement >= 99.5% math and code, >= 99% chat, with an
agreement drop <= 1 point under latent noise at 0.1 x the per-dim spread.

Run:    python tok_vae.py --zdim 128 [--steps 4000]
Output: results/tok_vae_z<zdim>.json
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from ae_gate import CKPT, nonoverlap, tblock
from common import DATA, RES

K = 8
GATE = {"dev_chat": 0.99, "dev_math": 0.995, "dev_code": 0.995}


def load_embed():
    snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/"))[0]
    idx = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(snap, idx["model.embed_tokens.weight"]), "pt") as f:
        return f.get_tensor("model.embed_tokens.weight")


class TokVAE(nn.Module):
    def __init__(self, D, zdim, nq=4, d=512, depth=4):
        super().__init__()
        self.nq, self.zq = nq, zdim // nq
        self.inp = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, d))
        self.pe = nn.Parameter(torch.randn(K, d) * 0.02)
        self.q = nn.Parameter(torch.randn(nq, d) * 0.02)
        self.enc = tblock(d, depth)
        self.mu, self.lv = nn.Linear(d, self.zq), nn.Linear(d, self.zq)
        self.zin = nn.Linear(self.zq, d)
        self.lat_e = nn.Parameter(torch.randn(nq, d) * 0.02)
        self.oq = nn.Parameter(torch.randn(K, d) * 0.02)
        self.dec = tblock(d, depth)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, D), nn.LayerNorm(D))

    def encode(self, e):
        h = self.enc(torch.cat([self.q.expand(len(e), -1, -1), self.inp(e) + self.pe], 1))[:, :self.nq]
        return self.mu(h), self.lv(h).clamp(-10, 5)

    def decode(self, z):
        h = torch.cat([self.zin(z) + self.lat_e, self.oq.expand(len(z), -1, -1)], 1)
        return self.out(self.dec(h)[:, self.nq:])          # (B, K, D) head inputs


def chunks(split):
    m = np.load(os.path.join(DATA, "cache", f"{split}_meta.npz"))
    seq, top1 = m["seq"], m["top1"]
    n = len(seq)
    ok = np.zeros(n, bool)
    ok[:n - K + 1] = seq[:n - K + 1] == seq[K - 1:]
    return top1.astype(np.int64), np.nonzero(ok)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zdim", type=int, default=128)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--B", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--beta", type=float, default=1e-3)
    ap.add_argument("--floor", type=float, default=0.5)
    ap.add_argument("--zdrop", type=float, default=0.15)
    args = ap.parse_args()
    E = load_embed().cuda()
    W = torch.load(os.path.join(DATA, "cache", "lm_head.pt")).cuda()
    tr_tok, tr_st = chunks("train")
    model = TokVAE(E.shape[1], args.zdim).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / 300) * max(0.05, 0.5 * (1 + np.cos(np.pi * min(1, s / args.steps)))))
    rng = np.random.default_rng(0)
    t0, log = time.time(), []
    for step in range(args.steps):
        st = rng.choice(tr_st, args.B)
        y = torch.from_numpy(tr_tok[st[:, None] + np.arange(K)[None]]).cuda()
        with torch.autocast("cuda", torch.bfloat16):
            mu, lv = model.encode(E[y])
            z = F.dropout(mu + torch.randn_like(mu) * (0.5 * lv).exp(), args.zdrop)
            lg = model.decode(z) @ W.T
        ce = F.cross_entropy(lg.float().flatten(0, 1), y.flatten())
        mu32, lv32 = mu.float(), lv.float()
        klz = (0.5 * (mu32 ** 2 + lv32.exp() - 1 - lv32)).mean((0, 1)).clamp_min(args.floor).sum()
        loss = ce + args.beta * klz
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.steps - 1:
            r = {"step": step, "ce": ce.item(), "klz": klz.item(),
                 "agree_train": (lg.argmax(-1) == y).float().mean().item(), "min": (time.time() - t0) / 60}
            log.append(r)
            print(json.dumps(r), flush=True)
    os.makedirs(CKPT, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args)}, os.path.join(CKPT, f"tokvae_z{args.zdim}.pt"))
    model.eval()
    res = {"args": vars(args), "train_log": log, "sets": {}}
    g = torch.Generator(device="cuda").manual_seed(0)
    for sp in ("dev_math", "dev_code", "dev_chat"):
        tok, st = chunks(sp)
        st = nonoverlap(st, K)
        y = torch.from_numpy(tok[st[:, None] + np.arange(K)[None]]).cuda()
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            mu, lv = model.encode(E[y])
            spread = mu.float().std(0, keepdim=True)
            ag = {}
            for kk, z in {"mean": mu, "noise0.1": mu + 0.1 * spread * torch.randn(mu.shape, device="cuda", generator=g),
                          "noise0.3": mu + 0.3 * spread * torch.randn(mu.shape, device="cuda", generator=g),
                          "posterior": mu + torch.randn(mu.shape, device="cuda", generator=g) * (0.5 * lv).exp()}.items():
                pred = (model.decode(z) @ W.T).argmax(-1)
                ag[kk] = (pred == y).float().mean().item()
                if kk == "mean":
                    chunk_exact = (pred == y).all(1).float().mean().item()
        r = {"positions": int(y.numel()), "agree": ag["mean"], "chunk_exact": chunk_exact, "agree_noise": ag,
             "pass": ag["mean"] >= GATE[sp] and ag["mean"] - ag["noise0.1"] <= 0.01}
        res["sets"][sp] = r
        print(sp, json.dumps(r), flush=True)
    res["pass"] = all(v["pass"] for v in res["sets"].values())
    json.dump(res, open(os.path.join(RES, f"tok_vae_z{args.zdim}.json"), "w"), indent=1)
    print("GATE", "PASS" if res["pass"] else "FAIL", f"z{args.zdim}", f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
