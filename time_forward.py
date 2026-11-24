"""How fast can the 4-bit target produce training states? Times one forward at several sizes, alone on the GPU.

Reports tokens/s for: one sequence with output_hidden_states, and a padded batch of sequences with hooks that keep
only layers 18, 24, 30 and the final norm (what Gister needs). Decides between live harvesting and a disk cache.
"""
import time

import torch

from dflash_eval import load_target

LAYERS = [18, 24, 30]


def main():
    t = load_target()
    keep = {}
    hooks = [t.model.layers[l - 1].register_forward_hook(lambda m, i, o, l=l: keep.__setitem__(l, o[0] if isinstance(o, tuple) else o))
             for l in LAYERS]
    hooks.append(t.model.norm.register_forward_hook(lambda m, i, o: keep.__setitem__(36, o)))
    V = t.config.vocab_size
    print(f"GPU after load {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
    for B, L in [(1, 512), (1, 1536), (4, 512), (8, 512), (4, 1024), (16, 256)]:
        ids = torch.randint(0, V - 1000, (B, L), device="cuda")
        with torch.no_grad():
            t.model(input_ids=ids)                       # warm-up for this shape
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            a = time.perf_counter()
            for _ in range(3):
                t.model(input_ids=ids)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - a) / 3
        print(f"B={B:2d} L={L:5d}: {dt:.2f} s/forward, {B * L / dt:,.0f} tok/s, peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB",
              flush=True)
        keep.clear()


if __name__ == "__main__":
    main()
