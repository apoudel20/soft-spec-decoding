"""Where does a Gister tree round spend its time? Times draft, verify and cache trimming separately, alone on the GPU.

Run: python profile_tree.py
"""
import time

import torch
from transformers import AutoTokenizer, DynamicCache

from common import eval_prompts
from dflash_eval import TARGET_ID, chat, load_target
from dflash_model import crop_to
from e2e import GistDrafter, build_tree, keep_path


def timed(fn, reps=10):
    torch.cuda.synchronize()
    a = time.perf_counter()
    for _ in range(reps):
        r = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - a) / reps * 1000, r


@torch.inference_mode()
def main():
    tok = AutoTokenizer.from_pretrained(TARGET_ID)
    t = load_target()
    ids = chat(tok, eval_prompts("mtbench", 1)[0])
    out = t.generate(ids, max_new_tokens=200, do_sample=False, temperature=None, top_p=None, top_k=None)
    ids = out[:, :ids.shape[1] + 200]                           # a realistic context of prompt + 200 tokens
    start = ids.shape[1]
    for name, kw in [("chain v2", dict(s2="stage2_gister_v2")),
                     ("flow K=4 nfe8", dict(s2="stage2_gister_flow", k_samples=4)),
                     ("flow K=8+det nfe8", dict(s2="stage2_gister_flow", k_samples=8, det_s2="stage2_gister_v2"))]:
        dr = GistDrafter(t, **kw)
        dr.reset()
        cache = DynamicCache()
        o = t(ids, past_key_values=cache, use_cache=True, logits_to_keep=1, output_hidden_states=True)
        dr.observe(o.hidden_states, start)
        bonus = o.logits[0, -1].argmax()
        ms_d, d = timed(lambda: dr.draft(bonus, start))
        d = d if d.dim() == 2 else d[None]
        toks, par, dep, kids = build_tree(bonus, d)
        n = len(toks)
        anc = torch.zeros(n, n, dtype=torch.bool, device="cuda")
        for i in range(n):
            if par[i] >= 0:
                anc[i] = anc[par[i]]
            anc[i, i] = True
        mask = torch.cat([torch.ones(n, start, dtype=torch.bool, device="cuda"), anc], 1)[None, None]
        x = torch.tensor([toks], device="cuda")
        pos = (start + torch.tensor(dep, device="cuda"))[None]

        def verify():
            r = t(x, attention_mask=mask, position_ids=pos, past_key_values=cache, use_cache=True,
                  output_hidden_states=True)
            crop_to(cache, start)
            return r
        ms_v, _ = timed(verify)
        t(x, attention_mask=mask, position_ids=pos, past_key_values=cache, use_cache=True)
        idx = torch.tensor([0, 1, 2], device="cuda")
        ms_k, _ = timed(lambda: keep_path(cache, start, idx), reps=1)
        print(f"{name:20s} nodes {n:3d}  draft {ms_d:6.1f} ms  verify {ms_v:6.1f} ms  keep_path {ms_k:5.1f} ms", flush=True)
    one = DynamicCache()
    t(ids, past_key_values=one, use_cache=True, logits_to_keep=1)

    def step():
        r = t(ids[:, -1:], position_ids=torch.tensor([[start]], device="cuda"), past_key_values=one, use_cache=True)
        crop_to(one, start)
        return r
    ms_1, _ = timed(step)
    print(f"plain decode step (1 token): {ms_1:.1f} ms", flush=True)


if __name__ == "__main__":
    main()
