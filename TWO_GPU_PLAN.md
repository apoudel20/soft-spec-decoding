# Two-GPU plan for Gister (for Bijin and his agent)

Written 2026-11-24 evening. Read AGENT_NOTES.md first for the design and every result so far.

## Where things stand

Gister is our drafter: a stage-1 gist VAE over Qwen3-8B's hidden states (layers 18, 24, 30, final) plus a stage-2
search network that predicts the next gist latent, decoded through the target's frozen LM head. It runs end to end
(`e2e.py`) and is faster than plain decoding, but it is well behind DFlash. Accepted tokens per round, out of 9,
same dev rounds:

| | Math | Code | Chat |
|---|---|---|---|
| Gister v1 (gist + raw states, 22M) | 2.80 | 2.23 | 1.43 |
| DFlash chain, capped at 8 | 5.88 | 5.59 | 2.68 |
| Stage-1 ceiling (true next gist decoded) | 8.51 | 8.47 | 7.63 |

Stage 1 is fine. Stage 2 (forecasting the next gist) is the bottleneck.

## What we learned about the hardware (12 GB card)

- The 4-bit target alone takes 6.4 GB and runs at about 3,000 tokens/s in forward passes (`time_forward.py`).
- Running the target and stage-2 training in the same process does not fit. Windows then spills GPU memory into
  system RAM and everything slows about 50x without an error. `gister_train.py` (live harvesting) hit this.
- So work in two phases: the target writes states to disk (`cache_states.py`), then stage 2 trains with no target
  loaded (`stage2.py`). Disk cost is 32 KB per position (4 layers, fp16). Never run two GPU jobs at once.

## Aayush's machine (running now)

1. `cache_states.py --pb_tokens 2.0`: 2M response positions of Qwen3-8B's own non-thinking answers from
   `heiheiha798/perfectblend-regen-qwen3-8b-non-thinking` shard 0, into `data/cache_pb/` (about 65 GB). That is 4x
   v1's training data.
2. `stage2.py --ae gist_hf_z2048 --raw --train_cache cache_pb --d 1024 --depth 12 --steps 20000`: the 154M
   predictor on it. Tests whether data fixes the overfitting seen at 154M on the old 512K positions.
3. `e2e.py --sets mtbench gsm8k humaneval --n 20 --max_new 256 --drafters dflash dflash8 gister gister_...`: τ
   and wall-clock speedup for Gister v1, v2 and DFlash on one loop.

## Suggested for Bijin's GPU (independent of the above, no shared files)

Pick one. Each ends in a results JSON and a checkpoint. Push code and `results/*.json` to a branch
`bijin/<task>`, checkpoints to the HF model repo (below), not to git.

**A. Sampling stage 2, done properly (highest value).** Aayush wants Gister to sample latent futures instead of
averaging them. Our first flow-matching run (`stage2.py --mode flow`) collapsed to the copy floor (1.21 / 1.13 /
1.06) because its token loss was computed on the one-step estimate `xt + (1 - t) v`, which nearly sees the answer
at high t. Fix it:
- train the velocity with plain flow-matching MSE only, or put the token loss on a full multi-step sample with the
  network unrolled (or on the one-step estimate only at low t),
- sample t from a logit-normal and shift it towards noise for a 2,048-dim latent (the RAE paper's dimension shift),
- give it the raw target states (`--raw`), which is what helped the deterministic version.
Then measure what sampling buys: draw K = 4, 8, 16 latents per round, decode each to 8 tokens, and report
accepted for the best of K (the ceiling for a latent-sampled draft tree) next to the deterministic chain. If
best-of-8 beats the deterministic chain by a lot, sampled trees are Gister's path past DFlash's per-slot trees.

**B. Make the latent predictable (joint fine-tune).** Stage 1 was trained only to reconstruct, so the gist holds
detail the past cannot predict. Fine-tune the stage-1 encoder together with stage 2: loss = stage-1 reconstruction
(exact tokens through the head) + stage-2 prediction loss on the next chunk. The exact-token loss stops collapse.
Gate: stage-1 agreement stays at or above 99.0 / 98.5 / 96.5 while stage-2 accepted rises.

## Setup on Bijin's machine

- `python cache_states.py` builds the original cache (`data/cache/`, about 18 GB, about 20 minutes). The dev splits
  there are what every number above is scored on.
- Checkpoints: HF model repo `ablastvacant/soft-spec-decoding-ae-gate`, folder `gister/`:
  `ae_gist_hf_z2048.pt` (stage 1) and `stage2_gist_hf_z2048_det_raw.pt` (stage-2 v1). Put them in `ckpt/`.
- Encoded latents for a cache are rebuilt automatically in `data/zcache/` on the first `stage2.py` run.

## How the two machines coordinate

They run different experiments, not one training job. Two home GPUs cannot share one training run over the
internet at useful speed. Results meet in `results/*.json` on GitHub and in this file. Whoever finishes first
updates the "Where things stand" table.
