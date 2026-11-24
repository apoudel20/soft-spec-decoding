# Agent notes: where the project is and what to do next

Last updated 2026-11-24 (after the stage-2 history runs). Read this first. RESULTS.md has the earlier experiments.

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

Nothing is running. The GPU is idle. Both stages are built and run end to end (below).

## Where it stands (short version)

- **Stage 1 nearly passes** with the "gist" autoencoder: 99.4 / 99.0 / 97.3% exact top-1 on math / code / chat.
- **Stage 2 works but is weak**: 2.2 / 1.8 / 1.3 accepted per round (out of 9) against DFlash's 7.5 / 6.8 / 3.0
  (out of 16).
- **The gist beats a tokens-only latent** by 13 to 24% in every stage-2 setting. This is the first result that
  supports the design over a plain token-chunk autoencoder.
- **The bottleneck is the stage-2 predictor**, not stage 1 and not context. A perfect predictor (the true next
  latent) gives 8.5 / 8.5 / 7.6, far above DFlash. The current one is 19M parameters trained for 13 minutes.
- Open decision for Aayush: scale stage 2 up, or ship DFlash + draft trees (chat 2.96 to 4.00) as the near-term
  deliverable while the research continues.

## Stage 1 results (capped answers, 512,380 train positions)

Gate fixed before running: top-1 agreement with the target >= 99.5% on math and code, >= 99% on chat.

First recipe: chunk VAE over layers x 8 positions, loss = KL through the frozen head on the final row + cosine +
small masked MSE on the middle rows + KL with a floor.

| Config | Math | Code | Chat | Note |
|---|---|---|---|---|
| z512 full grid | 93.9 | 92.5 | 77.8 | |
| z1024 full grid | 94.0 | 92.6 | 77.6 | |
| z2048 full grid | 94.6 | 93.0 | 78.3 | |
| z1024 final row only | 93.7 | 92.3 | 76.8 | middle rows barely matter |
| z2048 plain AE (no noise, no KL) | 93.3 | 92.6 | 75.0 | VAE noise was not the problem |
| z2048 big decoder (161M, 36k steps) | 86.0 | 84.6 | 61.6 | train 94.5%, so it overfits |
| ctx8 z2048 (8 prefix positions as condition) | 92.5 | 91.0 | 72.9 | context made it worse (overfits) |
| ctx8 nocond (matched chunks, no context) | 94.1 | 92.6 | 77.6 | |
| gist_tok z2048 (`--ce 1 --cos_w 0.1 --mid_w 0.1`) | 94.1 | 92.7 | 78.3 | token loss alone is not enough |
| **gist_hf z2048 (+ `--headfeat`)** | **99.4** | **99.0** | **97.3** | near pass |

What fixed it: `--headfeat` gives the encoder W[top1], the frozen LM-head row of each final state's own top-1.
That is a function of the input states only, so nothing leaks from the labels. Why it was needed: final-layer
states are high dimensional (PCA to 1,024 dims still only reaches 90 / 89 / 67% agreement), so matching the state
by cosine cannot give exact tokens. The encoder has to be told which token the state picks.

Control: `tok_vae.py` (a VAE over the target's 8 top-1 tokens, frozen embedding in, frozen head out, z128, 4k
steps) gets 98.4 / 97.4 / 94.8%. It is undertrained and is only a control, not the design.

Prompt conditioning (`--ctx 8`) did not help stage 1, so the final stage-1 model (`gist_hf_z2048`) has no
context. The context instead goes to stage 2 (history, below).

## Stage 2 results

`stage2.py`. A transformer reads the latent of the current chunk (or the last H chunk latents with `--hist H`)
and predicts the next chunk's latent. Loss: latent MSE + cross-entropy of the decoded next chunk (frozen stage-1
decoder, frozen head) against the target's own next 8 top-1 tokens. Deterministic mode (`--mode det`), 6,000 steps.

Scoring: the target's own state at position p gives token p+1. Stage 2 drafts p+2..p+9, so the max is 9 accepted
per round (8 drafted + bonus). Teacher-forced, non-overlapping rounds, against the HF target's top-1.

| Latent | History | Math | Code | Chat |
|---|---|---|---|---|
| token VAE | 1 chunk (8 positions) | 1.75 | 1.46 | 1.18 |
| token VAE | 8 chunks (64 positions) | 1.88 | 1.48 | 1.18 |
| gist_hf | 1 chunk | 2.17 | 1.80 | 1.34 |
| gist_hf | 8 chunks | 2.30 | 1.81 | 1.31 |
| oracle (true next gist latent) | | 8.51 | 8.47 | 7.63 |
| copy baseline | | about 1.0 | about 1.0 | about 1.0 |

- First drafted token correct: 51 to 55% on math, 24 to 26% on chat.
- Latent MSE (normalised) stays at 0.89 to 0.97, so the predictor explains only about 10% of the next latent.
- More history barely helps. The predictor (size, data, training length) is the bottleneck.
- Flow mode (`--mode flow`) is implemented and smoke tested but not run at full length.

DFlash reference (teacher-forced chain, out of 16): math 7.46, code 6.77, chat 2.96, scored against the Ollama
text, not the HF top-1. Tree with 64 branches reaches 4.00 on chat. The fair rerun against the HF top-1
(`dflash_tree_gap.py --hf_labels`) has not been run yet (task 2 below).

Earlier experiments (plan vector added to DFlash, true future fed to DFlash, layer sweeps, tree headroom) are in
RESULTS.md. None of them raised acceptance except draft trees, which are known prior work.

## Data

- `tokens/*.jsonl`: the target's own greedy answers (prompt and response token ids). 2,000 train answers and 40
  dev answers each for math, code, chat. `*_long` versions continue capped answers to 1,024 tokens. Details in
  `tokens/README.md`.
- HF dataset (same tokens): `ablastvacant/soft-spec-decoding-qwen3-8b-answers`.
- HF model repo with the gate checkpoints and results: `ablastvacant/soft-spec-decoding-ae-gate`.
- The hidden-state cache is not in git or on HF (about 18 GB for the capped answers). Build it locally (task 1).

## Reproduce

```
python cache_states.py                                   # data/cache/, about 18 GB, run --smoke first
python ae_gate.py --name gist_hf_z2048 --zdim 2048 --ce 1 --headfeat --cos_w 0.1 --mid_w 0.1 --steps 8000
python stage2.py --ae gist_hf_z2048 --mode det --steps 6000 [--hist 8]
python tok_vae.py --zdim 128 && python stage2.py --tok tokvae_z128 --mode det --steps 6000   # control
```

`run_gist.sh` and `run_hist.sh` run these in order. Results land in `results/ae_gate_*.json`,
`results/stage2_*.json`, `results/tok_vae_z128.json`. Checkpoints go to `ckpt/` (not in git).

## Tasks for Bijin

1. **Build the state cache on your GPU.**
   `python cache_states.py` (capped answers, writes `data/cache/`, about 18 GB) and
   `python cache_states.py --long` (1,024-token answers, writes `data/cache_long/`, expect roughly twice the size).
   Run `--smoke` first. Needs the HF target `Qwen/Qwen3-8B` in 4-bit (bitsandbytes), about 6 GB of VRAM.
   The long cache now matters: stage 2 is data and capacity limited.
2. **Fair DFlash baseline.** `python dflash_tree_gap.py --hf_labels --out dflash_fair` rescores DFlash against the
   HF target's own top-1 (same labels as stage 2) and adds a cap-at-8 chain number for a like-for-like comparison.
   Not run yet, so smoke test it first.
3. **Stage 2 at scale**, if Aayush picks that path. `stage2.py` reads whichever cache its stage-1 checkpoint was
   trained on, so first retrain stage 1 with `--cache cache_long --name gist_hf_long` (other flags as above). Then
   train a bigger search network on it
   (`--d 1024 --depth 12`), longer training (`--steps 30000`), then `--mode flow`. Watch whether slot-1 accuracy
   and the latent MSE move. If 30k steps on the long cache does not get chat above 2, report that before going
   bigger.

## Conventions

- Lossless only. Everything is scored against the target's own top-1, never against reference text.
- Fix a pass/fail gate in the script docstring before a run, and smoke test (`--smoke`) before any long run.
- Never commit `data/`, `ckpt/`, `logs/`, venvs or keys. Only code and `results/*.json`.
- Report numbers as measured, including failures.
