# Agent notes: where the project is and what to do next

Last updated 2026-11-25. Read this first, then TWO_GPU_PLAN.md (what each GPU is doing next). RESULTS.md has every number.


> History note (2026-11-25): commits after 2987111 were rewritten to remove personal details. If you cloned before
> that, run `git fetch && git reset --hard origin/main` (after saving any local work).

## Status, 25 Nov

- Current results are in RESULTS.md, section "Gister, audited state (25 Nov)". The deck copy is in `deck/slides/`.
- Best offline result: 285M flow sampler, best of 16, 4.18 / 3.06 / 1.92 accepted per round (max 9). Adding the raw
  states of 56 earlier positions to the 158M sampler gives 4.25 / 2.83 / 1.82.
- End to end, the Gister chain runs 1.34x / 2.05x / 1.40x (MT-Bench / GSM8K / HumanEval). Trees are slower than the
  chain on this GPU. DFlash runs 2.69x / 4.73x / 7.89x.
- The stage-1 latent mostly encodes the 8 tokens, because its gate rewards exact reconstruction. The proposed next
  step is a latent over 64 to 256 tokens trained by latent prediction, starting with an oracle test (RESULTS.md,
  "Proposed next step").

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

- **Stage 1 nearly passes on top-1** with the "gist" autoencoder: 99.4 / 99.0 / 97.3% exact top-1 on math / code / chat.
  It **fails the gate's distribution check** (TV 0.06 / 0.06 / 0.20 against 0.005 to 0.01), so every Gister claim is for greedy decoding only.
- **Stage 2 works but is weak**: 2.2 / 1.8 / 1.3 accepted per round (out of 9) against DFlash's 7.5 / 6.8 / 3.0
  (out of 16).
- **The gist beats a tokens-only latent** by 13 to 24% in every stage-2 setting. This is the first result that
  supports the design over a plain token-chunk autoencoder.
- **The bottleneck is the stage-2 predictor**, not stage 1 and not context. A perfect predictor (the true next
  latent) gives 8.5 / 8.5 / 7.6, far above DFlash. The current one is 19M parameters trained for 13 minutes.
- Open decision for Aayush: scale stage 2 up, or ship DFlash + draft trees (chat 2.96 to 4.00) as the near-term
  deliverable while the research continues.

## Naming and scope (Aayush, 2026-11-24)

**Gister** (working title) is our design: stage-1 gist VAE over the target's states + stage-2 search network that
predicts the next gist latent, decoded through the frozen LM head. No DFlash inside it. We are building our own
drafter, not improving DFlash or EAGLE. DFlash, EAGLE-3/4, DFlash2 and DSpark are baselines only. The DFlash-backed
heads below are a side experiment. Gister is trained per target (here Qwen3-8B), like EAGLE and DFlash.

## Latest (afternoon of 2026-11-24)

- **Fair DFlash offline row** (`dflash_s2.py`, same rounds and HF top-1 labels as `stage2.py`, chain capped at 8
  drafted, max 9): 5.88 / 5.59 / 2.68. Gister's best is 2.80 / 2.23 / 1.43.
- **DFlash-backed side experiment** (`dflash_feats.py` caches DFlash block outputs aligned with the state cache,
  `dflash_s2.py --head latent` maps them to the gist latent and decodes with stage 1): 5.30 / 4.51 / 2.25, below
  DFlash's own chain. `--head refine` (control, no latent) is the comparison.
- **Vocabulary-pruning oracle** (`prune_oracle.py`): restricting DFlash to the true tokens of the next 256 positions
  lifts chat chain only 2.69 to 3.12. Restricting to anything that can miss a drafted token halves acceptance. Hard
  pruning is not worth building.
- **End-to-end loop** (`e2e.py`): one lossless greedy loop for every drafter (Gister, DFlash, DFlash capped at 8,
  DFlash-backed heads), verified by the target, timed against plain greedy decoding on MT-Bench, GSM8K, HumanEval.
  `--check_ref` compares the DFlash path with DFlash's own `spec_generate`.
- **DFlash measured here** (RTX 4070 Ti, nf4 target, `dflash_eval.py`, own dev sets): speedup 4.75x / 3.80x /
  2.03x, tau 5.41 / 4.71 / 2.37.
- **Why the 154M predictor overfit**: data. DFlash trained on about 800K target-regenerated conversations (up to
  3,072 tokens, 6 epochs). Gister had 2,000 answers (512K positions). Next run trains stage 2 on target forward
  passes over far more on-policy text, computed on the fly (no disk cache needed).
- **On-policy data found**: `heiheiha798/perfectblend-regen-qwen3-8b-non-thinking` (Qwen3-8B, thinking off, 8
  shards, about 177K conversations per shard). Shard 0 is in `data/ext/perfectblend/` (not in git).
  `inference-optimization/Qwen3-8B-Regenerated-Collection` is larger but mixes thinking modes (unchecked).

Published baselines for the comparison table (not apples to apples with our 4-bit, 4070 Ti setup):

| Drafter | Target | Size | tau MT-Bench | Speedup | Source |
|---|---|---|---|---|---|
| DFlash b16 | Qwen3-8B | 5 layers (~1B) | 4.24 | 2.75x (H200) | DFlash paper T1 |
| EAGLE-3 tree 60 | Qwen3-8B | 1 layer | 3.26 | 1.90x (H200) | DFlash paper T1 |
| DSpark | Qwen3-8B | 5 layers | 3.72 (T=1) | not reported | DSpark paper (2607.05147) |
| EAGLE-4 | Qwen3.8-27B only | 3B | not reported | 3.59x SPEED-Bench (4x B300) | EAGLE-Corp card |
| DFlash2 | Qwen3.8-27B (z-lab), Qwen3-8B (RedHatAI) | not reported | not reported | 2.80x (EAGLE-4 card) | |

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
| gist_hf, 154M (`--d 1024 --depth 12`, 20k steps) | 1 chunk | 2.30 | 2.19 | 1.30 |
| gist_hf + raw target states (`--raw`) | 1 chunk | **2.80** | **2.23** | **1.43** |
| gist_hf + raw, token loss only (`--raw --lat_w 0`) | 1 chunk | 2.79 | 2.20 | 1.40 |
| gist_hf, flow mode (naive, see below) | 1 chunk | 1.21 | 1.13 | 1.06 |
| oracle (true next gist latent) | | 8.51 | 8.47 | 7.63 |
| copy baseline | | about 1.0 | about 1.0 | about 1.0 |

- First drafted token correct: 51 to 55% on math, 24 to 26% on chat.
- Latent MSE (normalised) stays at 0.89 to 0.97, so the predictor explains only about 10% of the next latent.
- More history barely helps. The predictor (size, data, training length) is the bottleneck.
- 154M overfits: train token loss falls from 4.8 to 1.45 while dev stays flat. Size alone does not help on 480k pairs.
- Raw target states (`--raw`, 4 layers x the 8 current positions) are the one input change that helps: +29% math,
  +24% code, +7% chat. Slot-1 accuracy rises to 64% math, 32% chat.
- Dropping the latent MSE changes nothing, so regressing in latent space is not what holds chat back.
- Flow mode collapsed to the copy floor. Its token loss uses the one-step estimate xt + (1-t)v, which nearly sees the
  answer at high t, so it is not a fair test of sampling. Needs a proper recipe (loss on sampled outputs, timestep
  shift for a 2,048-dim latent) before it means anything.
- Encoded latents are cached in `data/zcache/` after the first run, which saves about 5 minutes per run.

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
