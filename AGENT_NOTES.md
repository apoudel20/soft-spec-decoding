# Agent notes: where the project is and what to do next

Last updated 2026-11-24. Read this first, then RESULTS.md (the "Autoencoder gate" section at the end).

## Goal

Lossless speculative decoding on Qwen3-8B (4-bit nf4, HF transformers). Baseline drafter is the released DFlash
drafter `z-lab/Qwen3-8B-DFlash-b16`. DFlash is only the baseline to beat. We are not trying to tune DFlash itself.

Unit of comparison everywhere: accepted tokens per round, out of 16 (leading drafted tokens equal to the target's
own top-1, plus the bonus token), teacher-forced, scored against the HF nf4 target's own top-1.

## The design being built (Aayush's whiteboard design)

Two stages, trained in order.

**Stage 1: autoencoder over planning latents (VAE, or RAE as an alternative).**
- Expose the target's hidden states at later layers (18, 24, 30, final normed state 36), cached once over a
  calibration set.
- Arrange them as planning embeddings (reshape, or a grid of layers x positions with positional encoding).
- Train a VAE or RAE to reconstruct them, like an image autoencoder.
- Reuse the target's frozen LM head and embedding layer. A reconstruction pushed through the LM head must give the
  same outputs as the original.

**Stage 2: latent diffusion drafter (trained after stage 1 is frozen).**
- A "search" network takes the current compressed latent z_t and diffuses to the next latent z_t+1.
- z_t+1 goes through the stage-1 decoder and the LM head to give draft tokens.
- It is trained to match the target's true planning latents for the next k tokens.

**Decided by Aayush (2026-11-24):**
1. k = 8 positions per latent.
2. The prompt goes in as a condition. It is implemented as the target's states at the 8 positions before the chunk
   (all cached layers), fed to both the encoder and the decoder (`ae_gate.py --ctx 8`). The target's state at a
   position already summarises the prompt and all text before it, and a drafter has these states at drafting time.
3. Stage-1 bar is exact match: the gate below (>= 99.5% math and code, >= 99% chat top-1 agreement) stays.
4. The decoder outputs the 8 final-layer states of the chunk (the final row of the grid), and each goes through the
   frozen LM head. Middle layers are rebuilt too, but only the final row feeds the head.

Running now: `run_ctx.sh` (ctx8 z2048, and the matched no-context baseline on the same chunks with `--nocond`).
Do not start stage 2 until stage 1 passes or Aayush says otherwise.

## What has been run (all on the capped answers, 512,380 train positions)

Stage-1 gate: chunk VAE over layers x 8 positions, loss = KL through the frozen head on the final row + cosine +
small masked MSE on the middle rows + KL with a floor. Gate fixed before running: top-1 agreement with the target
>= 99.5% on math and code, >= 99% on chat.

| Config | Math | Code | Chat | Note |
|---|---|---|---|---|
| z512 full grid | 93.9 | 92.5 | 77.8 | |
| z1024 full grid | 94.0 | 92.6 | 77.6 | |
| z2048 full grid | 94.6 | 93.0 | 78.3 | best |
| z1024 final row only | 93.7 | 92.3 | 76.8 | middle rows barely matter |
| z2048 plain AE (no noise, no KL) | 93.3 | 92.6 | 75.0 | VAE noise was not the problem |
| z2048 big decoder (161M, 36k steps) | 86.0 | 84.6 | 61.6 | train 94.5%, so it overfits |

All fail. Misses concentrate where the target itself is unsure (top-1/top-2 margin < 0.5: about 40 to 50%
agreement, margin >= 2: 87 to 97%). The bigger decoder overfitting suggests more data helps more than more
parameters, which is one reason the answers were extended to 1,024 tokens (below).

Earlier experiments (plan vector added to DFlash, true future fed to DFlash, layer sweeps, tree headroom) are in
RESULTS.md. None of them raised acceptance except draft trees, which are known prior work.

## Data

- `tokens/*.jsonl`: the target's own greedy answers (prompt and response token ids). 2,000 train answers and 40
  dev answers each for math, code, chat. `*_long` versions continue capped answers to 1,024 tokens. Details in
  `tokens/README.md`.
- HF dataset (same tokens): `ablastvacant/soft-spec-decoding-qwen3-8b-answers`.
- HF model repo with the gate checkpoints and results: `ablastvacant/soft-spec-decoding-ae-gate`.
- The hidden-state cache is not in git or on HF (about 18 GB for the capped answers). Build it locally (task 1).

## Tasks that do not wait on the design decisions

1. **Build the state cache on your GPU.**
   `python cache_states.py` (capped answers, writes `data/cache/`, about 18 GB) and
   `python cache_states.py --long` (1,024-token answers, writes `data/cache_long/`, expect roughly twice the size).
   Run `--smoke` first. Needs the HF target `Qwen/Qwen3-8B` in 4-bit (bitsandbytes), about 6 GB of VRAM.
2. **Logit-lens check for the decoder-output question.** For layers 18, 24 and 30 in the cache, apply the target's
   final RMSNorm and the LM head, and report how often the result equals the target's top-1 (stored in
   `<split>_meta.npz` as `top1`), per dev split. This tells us how much of the next token a middle layer already
   holds, which decides option (a) vs (b) above.
3. **Rerun the best gate config on the long cache** once task 1 is done, to see whether more data closes the gap:
   run `python ae_gate.py --cache cache_long --name z2048_long --zdim 2048`. Compare with the z2048 row above.

## Conventions

- Lossless only. Everything is scored against the target's own top-1, never against reference text.
- Fix a pass/fail gate in the script docstring before a run, and smoke test (`--smoke`) before any long run.
- Never commit `data/`, `ckpt/`, `logs/`, venvs or keys. Only code and `results/*.json`.
- Report numbers as measured, including failures.
