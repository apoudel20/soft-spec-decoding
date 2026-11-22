"""Exactness harness.

1. Parity: our Qwen2 forward vs HF transformers (final hidden states must be bitwise equal).
2. Gated LoRA, same layout: with a random non-zero adapter and arbitrary soft-tail contents,
   logits and KV at every hard row of the chunk [x_last | H hard | S soft] must be bitwise
   identical to the base model's on the same layout.
3. Cross-layout (informational): one chunk vs token-by-token. Not bitwise equal in general;
   reports max |diff| and argmax agreement.
4. Rejection sampler: empirical output distribution of `verify` matches p (chi-square).

Run: python test_exactness.py [--adapter ckpt/xxx.pt] [--hf]
"""
import argparse
import sys

import torch

from model import TARGET, QwenLM, load_tokenizer, model_dir
from common import chat_ids


def prompt_ids(tok, text):
    return torch.tensor(chat_ids(tok, text), device="cuda")


def test_hf_parity(m, ids):
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(model_dir(TARGET), dtype=torch.bfloat16).cuda().eval()
    with torch.no_grad():
        hh = hf(ids[None], output_hidden_states=True).hidden_states[-1][0]
    ours = m.full(ids[None])[0][0]
    ok = torch.equal(ours, hh)
    print(f"[parity] final hidden bitwise equal to HF: {ok}  max|diff|={(ours.float() - hh.float()).abs().max().item():.3g}")
    del hf
    torch.cuda.empty_cache()
    return ok


def run_chunk(m, ids, H, S, soft_embeds, gate, detached=False):
    c = m.get_cache()
    m.step(c, ids[:-1], lo=-1, graph=False)
    c.n = len(ids) - 1
    g = torch.Generator(device="cuda").manual_seed(0)
    hard = torch.randint(0, 150000, (H,), device="cuda", generator=g)
    emb = torch.cat([m.embed_ids(ids[-1:]), m.embed_ids(hard), soft_embeds.to(torch.bfloat16)], 0)
    n0 = c.n
    if detached:  # soft rows at RoPE positions n0+1.., hidden from the hard-draft rows
        rope = torch.cat([m.ar[n0:n0 + H + 1], m.ar[n0 + 1:n0 + 1 + S]])
        lg = m.step(c, embeds=emb, gate=gate, rope=rope, hide=(H + 1, 1, H + 1))
    else:
        lg = m.step(c, embeds=emb, gate=gate)
    kv = (c.k[:, :, :, : n0 + H + 1].clone(), c.v[:, :, :, : n0 + H + 1].clone())
    return lg, kv


def test_gated_exact(m, ids, adapter=None):
    ok_all = True
    for H, S in [(2, 4), (4, 8), (8, 16), (1, 1), (0, 16), (8, 8)]:
        rnd = torch.Generator(device="cuda").manual_seed(H * 100 + S)
        # base: LoRA off, soft rows get some arbitrary embeddings
        base_soft = torch.randn(S, m.d, device="cuda", generator=rnd) * 0.02
        lg_b, kv_b = run_chunk(m, ids, H, S, base_soft, gate=None)
        # adapted: LoRA gated on the soft tail, soft rows get E^T q + e_flag
        q = torch.softmax(torch.randn(S, 64, device="cuda", generator=rnd) * 3, -1)
        qi = torch.randint(0, m.V, (S, 64), device="cuda", generator=rnd)
        soft = m.embed_soft(qi, q) + m.e_flag
        for det in (False, True):
            lg_c, kv_c = run_chunk(m, ids, H, S, soft, gate=H + 1, detached=det)
            hard_ok = torch.equal(lg_b[: H + 1], lg_c[: H + 1])
            kv_ok = torch.equal(kv_b[0], kv_c[0]) and torch.equal(kv_b[1], kv_c[1])
            soft_diff = (lg_b[H + 1:] - lg_c[H + 1:]).abs().max().item() if S else 0.0
            print(f"[gated {'detached' if det else 'attached'} H={H} S={S}] hard-row logits bitwise equal: {hard_ok}  "
                  f"kept KV bitwise equal: {kv_ok}  (soft rows differ by up to {soft_diff:.3g}, as they should)")
            ok_all &= hard_ok and kv_ok and soft_diff > 0
    return ok_all


def test_cross_layout(m, ids):
    lg_chunk = m.step(m.get_cache(), ids, graph=False)
    c2 = m.get_cache()
    rows = []
    for i in range(len(ids)):
        rows.append(m.step(c2, ids[i:i + 1]))  # graph path, T=1
        c2.n += 1
    lg_tok = torch.cat(rows)
    d = (lg_chunk - lg_tok).abs().max().item()
    agree = (lg_chunk.argmax(-1) == lg_tok.argmax(-1)).float().mean().item()
    print(f"[cross-layout] chunk vs token-by-token: bitwise equal={torch.equal(lg_chunk, lg_tok)}  "
          f"max|diff|={d:.3g}  argmax agreement={agree:.4f}  (informational)")


def test_rejection_sampler(n=200000, V=12):
    from decode import verify
    g = torch.Generator(device="cuda").manual_seed(0)
    p_logits = torch.randn(2, V, device="cuda", generator=g) * 1.5
    q = torch.softmax(torch.randn(V, device="cuda", generator=g) * 1.5, -1)
    p = torch.softmax(p_logits[0], -1)
    counts = torch.zeros(V, device="cuda")
    drafts = torch.multinomial(q.expand(n, V), 1, replacement=True, generator=g)[:, 0]
    for i in range(0, n, 1):
        if i >= 20000:
            break
        a, toks = verify(p_logits, drafts[i:i + 1], [q], temp=1.0, gen=g)
        counts[toks[0]] += 1
    tot = counts.sum()
    exp = p * tot
    chi2 = ((counts - exp) ** 2 / exp).sum().item()
    tv = 0.5 * (counts / tot - p).abs().sum().item()
    # chi-square with V-1=11 dof: 99.9% quantile about 31.3
    ok = chi2 < 31.3
    print(f"[sampler] first committed token ~ p: chi2={chi2:.1f} (dof {V - 1}, pass<31.3) TV={tv:.4f} -> {ok}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="trained Arm C adapter; random LoRA if omitted")
    ap.add_argument("--hf", action="store_true", help="also run HF parity (loads a second model copy)")
    args = ap.parse_args()
    tok = load_tokenizer()
    m = QwenLM(TARGET)
    ids = prompt_ids(tok, "Write a haiku about speculative decoding, then explain it.")
    ok = True
    if args.hf:
        ok &= test_hf_parity(m, ids)
    test_cross_layout(m, ids)
    if args.adapter:
        m.load_trainable(args.adapter)
    else:
        m.add_lora(32, 64)
        with torch.no_grad():  # make the random adapter non-trivial
            for L in m.layers:
                for t in ("q", "k", "v", "o", "gate", "up", "down"):
                    getattr(L, t).B.normal_(0, 0.02)
            m.e_flag.normal_(0, 0.02)
    ok &= test_gated_exact(m, ids)
    ok &= test_rejection_sampler()
    print("ALL EXACTNESS TESTS PASSED" if ok else "SOME EXACTNESS TESTS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
