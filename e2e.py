"""End-to-end lossless speculative decoding with every drafter, on the same loop, timed against plain greedy decoding.

Loop (greedy): prefill the prompt; each round the drafter proposes n tokens after the bonus token b, the target
verifies [b, d1..dn] in one forward, the longest matching prefix is kept plus the target's own next token as the new
bonus, and the target KV cache is cropped. Output equals the target's greedy output up to 4-bit kernel numerics
(batched verification vs one-token decoding can flip near-ties). tau = tokens kept per round (accepted + bonus).

Drafters:
  dflash        : DFlash as released, 15 drafted tokens per round (block 16). Same steps as its spec_generate.
  dflash8       : DFlash capped at 8 drafted tokens (the same draft length as our designs).
  dflash_latent : DFlash block outputs -> dflash_s2 latent head -> gist latent -> stage-1 decoder -> LM head (8).
  dflash_refine : DFlash block outputs -> dflash_s2 refine head (control, no latent) -> LM head (8).
  gist          : our two-stage design without DFlash: stage-1 gist of the last 8 target states (layers 18, 24, 30,
                  36) + those raw states -> stage-2 search network -> next gist latent -> stage-1 decoder (8).

Run:    python e2e.py --sets mtbench gsm8k humaneval --n 20 --max_new 256 [--drafters ...] [--smoke]
Output: results/e2e.json (per set: tau, tokens/s, speedup vs greedy, identical outputs)
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import AutoTokenizer, DynamicCache

from ae_gate import CKPT
from common import RES, eval_prompts
from dflash_eval import TARGET_ID, chat, load_drafter, load_target
from dflash_model import crop_to, extract_context_feature
from dflash_s2 import Head
from stage2 import Search, load_stage1

K = 8


class DFlashDrafter:
    def __init__(self, t, d, n_draft=15, head=None, dec=None):
        self.t, self.d, self.n, self.head, self.dec = t, d, n_draft, head, dec
        self.W = t.lm_head.weight

    def reset(self):
        self.cache, self.th = DynamicCache(), None

    def observe(self, hs, n):
        self.th = extract_context_feature(hs, self.d.target_layer_ids)[:, :n]

    def draft(self, bonus, start):
        d, bs = self.d, self.d.block_size
        blk = torch.full((1, bs), d.mask_token_id, dtype=torch.long, device="cuda")
        blk[0, 0] = bonus
        pos = torch.arange(self.cache.get_seq_length(), start + bs, device="cuda")[None]
        h = d(target_hidden=self.th, noise_embedding=self.t.model.embed_tokens(blk), position_ids=pos,
              past_key_values=self.cache, use_cache=True, is_causal=False)[:, -bs + 1:]
        crop_to(self.cache, start)
        if self.head is None:
            return self.t.lm_head(h)[0, :self.n].argmax(-1)
        h8 = h[:, :K].float()
        hrow = self.W[(h8.to(torch.bfloat16) @ self.W.T).argmax(-1)].float()
        with torch.autocast("cuda", torch.bfloat16):
            o = self.head(h8, hrow).float()
        x = self.dec(o) if self.dec is not None else o
        return (x.to(torch.bfloat16) @ self.W.T)[0].argmax(-1)


class GistDrafter:
    LAYERS = [18, 24, 30, 36]

    def __init__(self, t, ae_name="gist_hf_z2048", s2="stage2_gist_hf_z2048_det_raw"):
        self.t, self.W = t, t.lm_head.weight
        self.ae, a, _, layers, stats, _ = load_stage1(ae_name, "cuda")
        assert layers == self.LAYERS
        self.st = [(m.cuda(), s.cuda(), mk.cuda()) for m, s, mk in stats]
        ck = torch.load(os.path.join(CKPT, f"{s2}.pt"), map_location="cpu")
        sa = ck["args"]
        D, R = self.W.shape[1], len(layers)
        self.net = Search(self.ae.nq, self.ae.zq, sa["d"], sa["depth"], sa.get("hist", 1),
                          R * K if sa.get("raw") else 0, D).cuda().eval()
        self.net.load_state_dict(ck["model"])
        self.raw = sa.get("raw", False)
        self.zm, self.zs = ck["zm"].cuda(), ck["zs"].cuda()

    def reset(self):
        self.buf = None

    def observe(self, hs, n):
        x = torch.stack([hs[l][0, :n] for l in self.LAYERS])                 # (R, n, D)
        self.buf = x if self.buf is None else torch.cat([self.buf, x], 1)[:, -K:]

    def draft(self, bonus, start):
        b = self.buf
        if b.shape[1] < K:                                                     # very short prompt: repeat the first
            b = torch.cat([b[:, :1].expand(-1, K - b.shape[1], -1), b], 1)
        xs = []
        for r, (m, s, mk) in enumerate(self.st):
            x = (b[r].float() - m) / s
            x[:, mk] = 0
            xs.append(x)
        x = torch.stack(xs)[None].flatten(1, 2)                                # (1, R*K, D) layer-major, as the cache
        fin = b[-1].float()
        hf = self.W[(fin.to(torch.bfloat16) @ self.W.T).argmax(-1)].float()[None]
        with torch.autocast("cuda", torch.bfloat16):
            z = self.ae.encode(x, None, hf)[0].float()
        zc = (z - self.zm) / self.zs
        z1 = self.net(zc, torch.zeros_like(zc), torch.zeros(1, device="cuda"), x if self.raw else None)
        with torch.autocast("cuda", torch.bfloat16):
            fo = self.ae.decode(z1 * self.zs + self.zm).float()[:, -K:]
        m, s, _ = self.st[-1]
        return ((fo * s + m).to(torch.bfloat16) @ self.W.T)[0].argmax(-1)


@torch.inference_mode()
def spec(t, drafter, ids, max_new, stop):
    n0 = ids.shape[1]
    cache = DynamicCache()
    drafter.reset()
    o = t(ids, past_key_values=cache, use_cache=True, logits_to_keep=1, output_hidden_states=True)
    drafter.observe(o.hidden_states, n0)
    bonus = o.logits[0, -1].argmax()
    out, taus, start = [], [], n0
    while len(out) < max_new:
        dr = drafter.draft(bonus, start)
        blk = torch.cat([bonus[None], dr])[None]
        o = t(blk, position_ids=torch.arange(start, start + blk.shape[1], device="cuda")[None],
              past_key_values=cache, use_cache=True, output_hidden_states=True)
        post = o.logits[0].argmax(-1)
        acc = int((dr == post[:-1]).cumprod(0).sum())
        new = blk[0, :acc + 1].tolist()
        out += new
        taus.append(acc + 1)
        bonus = post[acc]
        start += acc + 1
        crop_to(cache, start)
        drafter.observe(o.hidden_states, acc + 1)
        if any(s in new for s in stop):
            break
    out = out[:max_new]
    for i, x in enumerate(out):
        if x in stop:
            out = out[:i + 1]
            break
    return out, taus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["mtbench", "gsm8k", "humaneval"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--drafters", nargs="+", default=["dflash", "dflash8", "dflash_latent", "dflash_refine", "gist"])
    ap.add_argument("--out", default="e2e")
    ap.add_argument("--check_ref", action="store_true", help="also run DFlash's own spec_generate and compare")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n, args.max_new = 2, 48
    tok = AutoTokenizer.from_pretrained(TARGET_ID)
    t, d = load_target(), load_drafter()
    stop = [tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")]
    dr = {}
    for name in args.drafters:
        if name == "dflash":
            dr[name] = DFlashDrafter(t, d, 15)
        elif name == "dflash8":
            dr[name] = DFlashDrafter(t, d, K)
        elif name.startswith("dflash_"):
            kind = name.split("_")[1]
            ck = torch.load(os.path.join(CKPT, f"dflash_s2_{kind}.pt"), map_location="cpu")
            ha = ck["args"]
            ae, _, _, layers, stats, _ = load_stage1(ha["ae"], "cuda")
            head = Head(t.lm_head.weight.shape[1], kind, ae.nq, ae.zq, ha["d"], ha["depth"]).cuda().eval()
            head.load_state_dict(ck["model"])
            dec = None
            if kind == "latent":
                zm, zs = ck["zm"].cuda(), ck["zs"].cuda()
                fm, fs = stats[-1][0].cuda(), stats[-1][1].cuda()

                def dec(o, ae=ae, zm=zm, zs=zs, fm=fm, fs=fs):
                    with torch.autocast("cuda", torch.bfloat16):
                        return ae.decode(o * zs + zm).float()[:, -K:] * fs + fm
            dr[name] = DFlashDrafter(t, d, K, head, dec)
        elif name == "gist":
            dr[name] = GistDrafter(t)
    w = chat(tok, "Say hi.")                                   # warm-up
    t.generate(w, max_new_tokens=8, do_sample=False, temperature=None, top_p=None, top_k=None)
    for x in dr.values():
        spec(t, x, w, 16, stop)
    res = {"args": vars(args), "gpu": torch.cuda.get_device_name(), "sets": {}}
    for s in args.sets:
        rows = []
        for text in eval_prompts(s, args.n):
            ids = chat(tok, text)
            torch.cuda.synchronize(); a = time.perf_counter()
            ref = t.generate(ids, max_new_tokens=args.max_new, do_sample=False, temperature=None, top_p=None,
                             top_k=None, eos_token_id=stop)[0, ids.shape[1]:].tolist()
            torch.cuda.synchronize()
            row = {"ar_s": time.perf_counter() - a, "ar_len": len(ref), "d": {}}
            if args.check_ref:
                o, acc = d.spec_generate(t, ids, args.max_new, stop, 0.0)
                row["ref_spec"] = {"out": o[0, ids.shape[1]:].tolist(), "taus": acc}
            for name, x in dr.items():
                torch.cuda.synchronize(); a = time.perf_counter()
                out, taus = spec(t, x, ids, args.max_new, stop)
                torch.cuda.synchronize()
                n = min(len(out), len(ref))
                row["d"][name] = {"s": time.perf_counter() - a, "len": len(out), "taus": taus,
                                  "identical": out[:n] == ref[:n],
                                  "first_diff": next((i for i in range(n) if out[i] != ref[i]), None)}
                if args.check_ref and name == "dflash":
                    row["d"][name]["same_as_spec_generate"] = out == row["ref_spec"]["out"][:len(out)]
            rows.append(row)
        ar = sum(r["ar_len"] for r in rows) / sum(r["ar_s"] for r in rows)
        sm = {"n": len(rows), "ar_tok_s": ar}
        for name in dr:
            taus = [x for r in rows for x in r["d"][name]["taus"]]
            tps = sum(r["d"][name]["len"] for r in rows) / sum(r["d"][name]["s"] for r in rows)
            sm[name] = {"tau": float(np.mean(taus)), "tok_s": tps, "speedup": tps / ar,
                        "identical": sum(r["d"][name]["identical"] for r in rows)}
        res["sets"][s] = {"summary": sm, "rows": rows}
        print(s, f"AR {ar:.1f} tok/s", flush=True)
        for name in dr:
            v = sm[name]
            print(f"  {name:14s} tau {v['tau']:.2f}  {v['tok_s']:.1f} tok/s  speedup {v['speedup']:.2f}x  "
                  f"identical {v['identical']}/{len(rows)}", flush=True)
        json.dump(res, open(os.path.join(RES, args.out + ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
