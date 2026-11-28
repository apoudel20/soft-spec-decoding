# Results

**Read this first (updated 2026-11-25).** The project now centres on **Gister** (Qwen3-8B 4-bit target, one RTX
4070 Ti). Its results are in the sections "Stage-2 gate", "Gist autoencoder and stage 2" and "Gister" near the end.
The early sections below are the original Qwen2.5-1.5B study (target Qwen2.5-1.5B-Instruct, drafter
Qwen2.5-0.5B-Instruct), kept as the record. Its pending follow-ups were completed or superseded on Qwen3-8B.

Scope and caveats for every Gister number:
- **Greedy decoding only.** Stage 1 matches the target's top-1 token 99.4 / 99.0 / 97.3% (math / code / chat), which
  is itself below the gate's 99.5 / 99.5 / 99.0% (`agree_ok: false` on all three sets). The gate also required the decoded distribution to stay close to the target's (total variation
  <= 0.005 to 0.01). Measured TV is 0.060 / 0.064 / 0.202 and KL 0.20 / 0.23 / 0.84, so the gate fails
  (`results/ae_gate_gist_hf_z2048.json`, `pass: false`). Greedy verification uses only the top-1 token, so the
  greedy results stand, but sampling at T > 0 would need a far tighter stage 1.
- **Two label conventions.** `dflash_tree_gap.json` and the earliest DFlash rows score against the Ollama-written
  text with up to 16 drafted. Every stage-2, Gister and `e2e.py` number scores against the HF target's own top-1
  with up to 8 drafted (max 9). Compare within a convention.
- **Wall-clock runs share one consumer GPU.** Plain greedy speed varies between runs (21 / 16 / 19 tok/s in
  `e2e.json`, 12 / 9 tok/s in `e2e_tree_fast.json` when desktop apps shared the GPU). Speedups are only compared
  within a run. An earlier "+12% tree speedup on GSM8K" came from a throttled run and is withdrawn: on an idle GPU
  (`e2e_final.json`) every tree is slower than the chain.
- **Draft latency is unoptimised.** On an idle GPU Gister drafts in about 11 ms (chain) and 15 to 49 ms (trees, before
  the bf16-drafting fix) against a 4-bit target decode step of about 39 to 45 ms (`profile_tree_idle.json`). On an
  optimised bf16 engine a Qwen3-8B step is far shorter, so the drafter would need CUDA graphs, fewer layers or a
  one-step sampler to keep any speedup.

## Relation to prior work

This is a small-scale study (1.5B target, 30 prompts per set), and most positive results here reproduce known
findings. The literature review in `reports/` was done after the experiments started. It should have come first.

| Finding here | Prior work |
|---|---|
| The target refining guesses for its own future (Arm C) beats frozen-target Jacobi | Shown at larger scale with bigger gains by Samragh et al. 2025 (arXiv 2507.11851) |
| A parallel block drafter beats classic SD on math | Known. DFlash, PARD and Medusa-style drafters do this at 7B to 8B scale |
| Target-distribution labels (TV) raise acceptance under sampling | Expected from DistillSpec (2023) |
| Block-drafter acceptance decays with offset | Known problem in parallel and non-autoregressive decoding |
| Specific future tokens are not linearly decodable beyond a few positions | Consistent with Future Lens and later probing work |
| Soft (distribution) inputs give the target nothing over hard guesses | Agrees with bijmuj/EmbeddingSuperPosition (drafter inputs) |

The only untested idea with a possible novelty claim is the latent planner. It is a target-produced, long-horizon
plan trained with an explicit future-prediction objective and fed to a block drafter. ConFu (arXiv 2603.08899),
SD² and FSP already pass target-derived future vectors to drafters, so the open question is narrow. Does an
explicit plan objective beat passing the target's hidden state? Decision rules, fixed before the results:

1. If the oracle (true future summary) doesn't raise acceptance, plans can't help at this scale. Stop.
2. If the oracle helps but the predicted plan only matches target features, prior work covers it. Stop.
3. Only if the predicted plan clearly beats target features is it worth scaling to a 4B to 8B target, Spec-Bench,
   and released EAGLE-3 and DFlash baselines.

## Summary

| Question (PLAN.md decision row) | Answer at this scale |
|---|---|
| C > B and C > C-hard? | **No.** Arm B was skipped, so C-hard is the hard-guess baseline. Soft and hard guesses give the same τ on GSM8K and HumanEval (2.33 vs 2.33, 2.36 vs 2.36). Soft is ahead by 4% on MT-Bench (1.79 vs 1.72) and by 1 to 7% at T=1. That is too small to call real signal. |
| C ≈ C-hard > B? | **Yes, with Jacobi standing in for B.** Letting the target refine its own future guesses lifts τ from 1.07 (frozen-target Jacobi) to 2.33 on GSM8K. The win comes from refinement in the verification pass, not from softness. |
| C ≈ B? | Not tested, because Arm B was skipped. |
| D3/D4 > D1 with soft inputs? | **Greedy no, T=1 yes for D4.** Greedy τ is flat across labels (4.58 to 4.61 on GSM8K). At T=1, D4 (target TV labels) beats D1 (hard CE) by 21% on GSM8K, 11% on MT-Bench and 6% on HumanEval. |
| Coherence collapse that D can't fix? | **Partly fixed, not solved.** Per-position acceptance of the best block drafter falls from 0.90 to 0.53 across the block, while classic SD stays flat near 0.93. A second self-conditioning pass recovers about half the gap. This row points to latent block modeling, which is what the planner experiment tests. |

Best wall-clock speedup per set (greedy, vs AR with the same kernels):

| Set | Best overall | Classic SD best | Arm C best (no drafter model) |
|---|---|---|---|
| GSM8K | 1.52x, D (d2_soft, B=8, K=2) | 1.37x (γ=7) | 1.39x (C-hard, H=4, S=8) |
| MT-Bench | 1.13x, SD γ=3 and D (d4_soft, K=1) tied | 1.13x (γ=3) | 1.05x (C soft, H=4, S=8) |
| HumanEval | 1.54x, SD γ=7 | 1.54x (γ=7) | 1.41x (C soft, H=4, S=8) |

Speedups vary by a few percent between runs. Treat differences under about 3% as noise.

## Setup actually used

- Eval subsets are 30 prompts each from MT-Bench first turns, GSM8K `test` and HumanEval, chosen by fixed stride.
  Generation is capped at 192 new tokens.
- Dev sets for any selection are 10 prompts each from UltraChat `test_sft`, the last 500 GSM8K `train` questions
  (held out of training) and MBPP `validation`. The only selection made was the C+D drafter (d4_soft), and it
  was made on dev.
- Training data is 7296 target-generated responses (half greedy, half T=1) to 4000 UltraChat, 2500 GSM8K and MBPP
  `train` plus `test` prompts. Prompts are up to 160 tokens and responses up to 352.
- Arm C trained for 1500 steps and Arm D for 1200 steps, with gated LoRA r=32.

Deviations from PLAN.md:

- Decode uses CUDA graphs. Eager PyTorch was CPU-bound at about 20 ms per forward for either model, which hides
  all speedups. Every arm uses the same graphs.
- The eval cap is 192 new tokens instead of 256, and training prompts go up to 160 tokens instead of 128.
- Arm B (CLLM fine-tune) was skipped. C-hard is the hard-guess baseline.
- Arm C uses a **detached tail**. The plan-literal layout (soft rows after the hard drafts, attending to them) goes
  stale after any rejection and gives τ about 1.06 (table below). In the detached layout the soft rows sit at
  their own guessed positions and the hard-draft rows can't see them. It is exact, and the test suite covers it.

## Exactness

- The self-written forward pass is bitwise equal to HF `transformers`.
- Gated LoRA is bitwise exact within a layout. Hard-row logits and kept KV match the base target for every
  tested (H, S), in both attached and detached layouts.
- The rejection sampler passes a chi-square test against the target distribution.
- **Greedy outputs.** Methods that verify with the same chunk width get identical token-match counts, and the
  counts change only when the width changes. Soft and hard variants always tie, and classic SD shows the same
  divergences (17/30 to 20/30 on GSM8K depending on γ). All divergences from AR happen at near-ties. The top-1/top-2 gap at the first divergent token is at most 0.155 logits and
  usually under 0.06. These come from bf16 numerics across different chunk shapes, and no method's verification
  rule produced them.

## Arm 0: teacher-forced vs whole-block accuracy

The table shows the block of 8 at the target's own greedy continuation, over 40 prompts per set.

| Drafter | Set | Teacher-forced a (mean) | P(L ≥ 8) | Mean accepted L of 8 |
|---|---|---|---|---|
| Base 0.5B, AR | GSM8K | 0.94 | 0.65 | 6.25 |
| Base 0.5B, AR | MT-Bench | 0.83 | 0.33 | 4.19 |
| Base 0.5B, AR | HumanEval | 0.97 | 0.83 | 7.16 |
| D3 soft block drafter, K=1 | GSM8K | 0.91 | 0.01 | 2.64 |
| D3 soft block drafter, K=2 | GSM8K | 0.91 | 0.14 | 4.25 |
| D3 soft block drafter, K=1 | MT-Bench | 0.77 | 0.00 | 1.50 |
| D3 soft block drafter, K=2 | MT-Bench | 0.77 | 0.02 | 2.30 |
| D3 soft block drafter, K=1 | HumanEval | 0.95 | 0.00 | 2.43 |
| D3 soft block drafter, K=2 | HumanEval | 0.95 | 0.09 | 4.27 |

Block drafters keep high teacher-forced accuracy at every offset. Their free-running conditional acceptance
collapses. On GSM8K it goes from 0.93 at offset 1 to 0.48 at offset 8 with K=1, and to 0.64 with K=2. That is
the coherence gap. D1, D3 and D4 are nearly identical here (meanL within 0.1). Full curves are in
`results/arm0_*.json`.

## Baselines and Arm A (greedy)

| Method | GSM8K | MT-Bench | HumanEval |
|---|---|---|---|
| AR tok/s | 99.2 | 95.2 | 96.0 |
| SD γ=3 | 3.59 / 1.30x | 3.03 / 1.13x | 3.78 / 1.29x |
| SD γ=5 | 4.94 / 1.35x | 3.85 / 1.11x | 5.46 / 1.50x |
| SD γ=7 | 6.14 / 1.37x | 4.41 / 1.03x | 6.95 / 1.54x |
| Jacobi H=8 (frozen target) | 1.07 / 0.86x | 1.06 / 0.90x | 1.04 / 0.82x |
| Soft Jacobi H=4, S=8, training-free | 1.07 / 0.83x | 1.06 / 0.86x | 1.04 / 0.80x |
| Soft Jacobi, detached tail | 1.08 / 0.82x | 1.07 / 0.83x | 1.06 / 0.84x |

Cells are τ / wall-clock speedup. A frozen target can't use soft or hard future guesses, so training-free Jacobi
gives almost nothing at this scale.

## Arm D: block drafter (greedy, B=8)

| Drafter | GSM8K | MT-Bench | HumanEval |
|---|---|---|---|
| D1 hard CE, soft feedback | 4.61 / 1.48x | 3.11 / 0.99x | 4.69 / 1.49x |
| D1 hard CE, hard feedback | 4.14 / 1.39x | 2.83 / 0.92x | 4.25 / 1.35x |
| D2 self-distillation, soft | 4.59 / 1.52x | 3.10 / 1.01x | 4.77 / 1.52x |
| D3 target forward KL, soft | 4.58 / 1.49x | 3.10 / 0.99x | 4.77 / 1.52x |
| D3 target forward KL, hard | 4.16 / 1.41x | 2.83 / 0.91x | 4.31 / 1.36x |
| D4 target TV, soft | 4.61 / 1.51x | 3.20 / 1.03x | 4.70 / 1.50x |
| D4 target TV, hard | 4.53 / 1.49x | 3.09 / 1.00x | 4.71 / 1.47x |
| D4 soft, K=1 (one pass) | 3.29 / 1.50x | 2.49 / 1.13x | 3.17 / 1.41x |
| D4 soft, K=3 | 5.33 / 1.33x | 3.63 / 0.92x | 5.63 / 1.39x |
| D4 soft, B=4 | 3.85 / 1.33x | 2.99 / 1.02x | 4.02 / 1.35x |
| D4 soft, B=16 | 4.70 / 1.37x | 3.23 / 0.91x | 4.71 / 1.33x |

K is the number of drafter passes per block. "Soft" and "hard" refer to how the first pass's guesses are fed
into the second pass (Eᵀq vs argmax embedding).

- Soft feedback beats hard feedback by about 10% τ for D1 and D3. With TV labels (D4) the gap shrinks to 1 to 4%.
- Labels barely matter under greedy decoding. D2 (distilling the base drafter's own distribution) is as good as
  learning from the target, so the gain over hard CE comes from soft supervision in general, not from target
  knowledge specifically.
- Per-position conditional acceptance for D4 soft on GSM8K is 0.90, 0.89, 0.81, 0.76, 0.73, 0.67, 0.67, 0.53.
  Classic SD γ=7 stays at 0.91 to 0.95.

## Arm C and C-hard (greedy, detached tail)

H is the number of hard draft rows the target writes for itself and S is the number of soft (or hard, for
C-hard) guess rows it refines each round.

| H, S | C soft: GSM8K | C-hard: GSM8K | C soft: MT-Bench | C-hard: MT-Bench | C soft: HumanEval | C-hard: HumanEval |
|---|---|---|---|---|---|---|
| 2, 4 | 2.11 / 1.27x | 2.09 / 1.20x | 1.72 / 1.03x | 1.66 / 0.99x | 2.17 / 1.29x | 2.16 / 1.35x |
| 2, 8 | 2.16 / 1.27x | 2.14 / 1.27x | 1.74 / 1.03x | 1.69 / 1.00x | 2.26 / 1.36x | 2.22 / 1.38x |
| 2, 16 | 2.15 / 1.12x | 2.16 / 1.13x | 1.72 / 0.89x | 1.68 / 0.87x | 2.24 / 1.19x | 2.24 / 1.22x |
| 4, 4 | 2.15 / 1.28x | 2.11 / 1.27x | 1.74 / 1.03x | 1.68 / 1.00x | 2.20 / 1.36x | 2.17 / 1.30x |
| 4, 8 | 2.33 / 1.31x | 2.33 / 1.39x | 1.79 / 1.05x | 1.72 / 0.99x | 2.36 / 1.41x | 2.36 / 1.39x |
| 4, 16 | 2.35 / 1.22x | 2.31 / 1.21x | 1.78 / 0.92x | 1.71 / 0.87x | 2.33 / 1.24x | 2.36 / 1.24x |
| 8, 8 | 2.38 / 1.26x | 2.32 / 1.23x | 1.79 / 0.94x | 1.72 / 0.88x | 2.38 / 1.29x | 2.34 / 1.23x |
| 8, 16 | 2.38 / 1.19x | 2.34 / 1.21x | 1.79 / 0.94x | 1.72 / 0.87x | 2.35 / 1.27x | 2.36 / 1.26x |

Ablations at H=4, S=8:

| Variant | GSM8K | MT-Bench | HumanEval |
|---|---|---|---|
| Plan-literal attached tail, soft | 1.08 / 0.64x | 1.06 / 0.63x | 1.04 / 0.63x |
| Plan-literal attached tail, hard | 1.08 / 0.62x | 1.06 / 0.63x | 1.04 / 0.65x |
| Soft, sharpened q (T=0.5) | 2.21 / 1.28x | 1.67 / 0.95x | 2.23 / 1.36x |

- H1 holds. Arm C more than doubles τ over frozen Jacobi with no separate drafter model.
- H2 fails on the target side. Soft guesses tie hard guesses except for a small MT-Bench edge.
- τ saturates near 2.4. The hard drafts each round are written before the target has seen that round's bonus
  token, so the first draft is really a two-ahead prediction. Its conditional acceptance is only 0.62 on GSM8K
  and 0.46 on MT-Bench. This is a structural limit of drafting inside the verification pass.
- Larger S costs time (wider chunks) and doesn't raise τ past S=8.

## C+D (greedy)

The best dev drafter (d4_soft) seeds Arm C's tail.

| Method | GSM8K | MT-Bench | HumanEval |
|---|---|---|---|
| C soft + d4_soft, H=4, S=8 | 2.40 / 0.84x | 1.82 / 0.70x | 2.43 / 0.77x |
| C-hard + d4_soft, H=4, S=8 | 2.35 / 0.81x | 1.74 / 0.70x | 2.43 / 0.77x |
| C soft + d4_soft, H=8, S=16 | 2.40 / 0.67x | 1.80 / 0.64x | 2.39 / 0.69x |

Negative. Seeding raises τ by about 3% and the drafter passes make it slower than AR. The target's tail
re-derives what the drafter proposed rather than building on it.

## T=1 with rejection sampling

| Method | GSM8K | MT-Bench | HumanEval |
|---|---|---|---|
| AR tok/s | 97.2 | 97.7 | 100.0 |
| SD γ=5 | 4.58 / 1.26x | 3.71 / 0.98x | 5.02 / 1.34x |
| Jacobi H=8 | 1.09 / 0.78x | 1.09 / 0.73x | 1.06 / 0.78x |
| D4 TV, soft feedback | 4.24 / 1.29x | 2.90 / 0.96x | 4.24 / 1.32x |
| D1 hard CE, soft feedback | 3.49 / 1.08x | 2.61 / 0.83x | 4.00 / 1.24x |
| D3 forward KL, hard feedback | 3.20 / 1.03x | 2.35 / 0.76x | 3.76 / 1.17x |
| C soft, H=4, S=8 | 2.08 / 1.21x | 1.62 / 0.85x | 2.18 / 1.17x |
| C-hard, H=4, S=8 | 2.02 / 1.15x | 1.60 / 0.87x | 2.04 / 1.12x |
| C soft + d4_soft | 2.09 / 0.78x | 1.66 / 0.65x | 2.30 / 0.76x |

- H3 holds under sampling. TV labels raise acceptance where it matters, since expected acceptance is 1 − TV(p, q).
  D4 soft matches classic SD wall-clock on GSM8K and HumanEval with a single 0.5B model running two passes per block.
- Soft vs hard guesses for Arm C stay within 1 to 7%.

## Failure analysis

From `failures.py` on the base and block drafters (`results/failures_summary.json`, examples in
`results/failure_examples_sample.json`, browsable in the shared failure explorer page).

- Most misses are content and function words (GSM8K 77%, MT-Bench 87%, HumanEval 52%). Formatting and
  whitespace are about 8%, 3% and 26%. HumanEval formatting misses are mostly code newlines and indentation.
- Teacher-forced misses are mostly near-ties. The target gives the drafted token p ≥ 0.2 in 34 to 49% of them,
  and is at least 0.9 sure of its own choice in only 4 to 10%.
- Free-running block misses are different. The target is at least 0.9 sure in 33 to 83% of them, so they are
  plain errors on easy tokens caused by the drafter's own earlier guesses.
- Scaling both models' temperature barely moves expected acceptance (GSM8K 0.938 at T=0.3, 0.927 at T=1.3).

**Rejection sampling on the same misses.** For each position on the target's greedy path, this is the expected
acceptance of the base drafter's token under T=1 speculative sampling, Σ min(p, q). It's split by token category,
both over all positions and over only the positions where greedy drafting failed.

| Category | GSM8K all | GSM8K greedy misses | MT-Bench all | MT-Bench greedy misses | HumanEval all | HumanEval greedy misses |
|---|---|---|---|---|---|---|
| Content word | 0.91 | 0.52 | 0.78 | 0.49 | 0.95 | 0.49 |
| Function word | 0.89 | 0.52 | 0.81 | 0.54 | 0.97 | 0.45 |
| Number | 0.98 | 0.39 | 0.96 | 0.40 | 0.99 | 0.42 |
| Punctuation | 0.95 | 0.53 | 0.88 | 0.56 | 0.97 | 0.54 |
| Newline | 0.94 | 0.53 | 0.92 | 0.52 | 0.89 | 0.58 |
| Whitespace | 0.97 | 0.61 | 0.97 | 0.56 | 0.99 | 0.70 |
| Markdown / LaTeX | 0.97 | 0.59 | 0.93 | 0.58 | 0.94 | 0.53 |

About half of every greedy miss is recovered under sampling, because the drafter and target overlap in
probability mass even when their argmaxes differ. Whitespace and markdown misses recover the most (0.53 to 0.70).
Number misses recover the least (about 0.40), since a wrong digit is usually a real error. This only applies to
sampled decoding. The output then matches the target's T=1 distribution, and greedy output stays defined by
exact argmax match.

## Planner go/no-go probe

Linear probes on hidden states at the middle and last layer predict token t+k over a 4096-token vocabulary
plus "other". They were trained on 71,955 positions from training sequences and tested on all response positions
of the eval references.

| Probe (last layer) | k=1 | k=2 | k=4 | k=8 | k=16 | k=32 |
|---|---|---|---|---|---|---|
| Target 1.5B, GSM8K | 0.87 | 0.56 | 0.23 | 0.08 | 0.05 | 0.03 |
| Drafter 0.5B, GSM8K | 0.85 | 0.51 | 0.21 | 0.07 | 0.05 | 0.04 |
| Target 1.5B, MT-Bench | 0.68 | 0.30 | 0.12 | 0.07 | 0.06 | 0.06 |
| Drafter 0.5B, MT-Bench | 0.63 | 0.27 | 0.11 | 0.08 | 0.06 | 0.06 |
| Target 1.5B, HumanEval | 0.82 | 0.39 | 0.15 | 0.08 | 0.06 | 0.05 |
| Drafter 0.5B, HumanEval | 0.80 | 0.34 | 0.12 | 0.07 | 0.06 | 0.05 |

The most-common-token baseline is 0.04 on GSM8K and MT-Bench, and 0.01 on HumanEval. The target's edge over the
drafter is 2 to 5 points at k=2 to 4 and gone by k=8. Middle layers are much worse than last layers at k ≤ 4. At k ≥ 8 they are about equal and sometimes
slightly better (MT-Bench k=16, 0.075 vs 0.064), all close to the baseline.
Specific tokens beyond a few positions are not linearly readable from either model, so a useful plan at this
scale has to be a summary of the future rather than its tokens.

**Future summaries.** Ridge probes predict the mean target-embedding of the next 32 or 128 response tokens, and
log remaining length. The penalty was chosen on held-out training rows. An earlier run penalised the intercept,
so its numbers were invalid and were replaced.

| Probe, R² on eval references | Next 32 tokens | Next 128 tokens | Remaining length |
|---|---|---|---|
| Target 1.5B last layer, GSM8K | 0.37 | 0.41 | 0.30 |
| Drafter 0.5B last layer, GSM8K | 0.34 | 0.39 | 0.27 |
| Current-token embedding only, GSM8K | 0.15 | 0.17 | 0.08 |
| Target 1.5B last layer, MT-Bench | 0.22 | 0.23 | 0.09 |
| Drafter 0.5B last layer, MT-Bench | 0.20 | 0.22 | 0.04 |
| Target 1.5B last layer, HumanEval | 0.30 | 0.28 | 0.23 |
| Drafter 0.5B last layer, HumanEval | 0.27 | 0.27 | 0.26 |

Both models' hidden states carry a real linear signal about the gist of the next 32 to 128 tokens, about twice
the embedding-only baseline. The target's edge over the drafter is small, at 0.01 to 0.03 R². Middle layers
give about the same summary R² and are better for remaining length (target GSM8K 0.52). At this scale the
drafter already holds most of the linearly readable plan the target has, which lowers the ceiling for a
target-provided plan. These are linear probes, so a nonlinear plan head could find more.

## Latent planner

**No valid result. Paused pending a literature review.** A second run (v2, `results/planner.json` as pushed at
02:27) bounded the plan input behind a tanh gate initialised at zero. That deadlocked, because at zero the
projection gets no gradient, so the gate never opened in any variant, including the oracle. Its accepted lengths
(GSM8K 4.01 to 4.11 vs 4.15) only show seed noise of about ±0.1. A third version (capped zero-init projection)
was stopped at step 200 of a quick oracle check, where it was slightly worse than no plan. Prior work read
since then (`research_notes/Prior work read in full.md`) shows the remaining idea is narrower than first
thought.

**First run invalid.** In the first run (`results/planner.json` as pushed on 23 Nov at 00:55)
every plan variant, including the oracle, scored far below the plain drafter (GSM8K mean accepted length 1.2
to 1.7 vs 4.15). The cause was a bug in the plan input. An unbounded, zero-initialised projection grew under
Adam to 4 to 10 times the norm of a token embedding (1.7 to 4.4 vs 0.45) and drowned the drafter's token inputs.
Those numbers say nothing about the idea. The rerun caps the plan offset at the median token-embedding norm
behind a learned gate that starts at zero, so each variant starts exactly at the plain drafter.

The best Arm D recipe (TV labels, soft feedback) gets a vector added to its block inputs. That vector
comes from target features, from a plan head trained to predict a summary of the next 64 tokens, or from oracle
summaries of the true future (near, and skipping the drafted block). All runs use the same steps and data as d4_soft.

## Selection gap (Qwen3-8B target, Qwen3-0.6B drafter)

How much could a better choice of which drafted nodes the target checks be worth? `selection_gap.py`, results in
`results/selection_gap.json`. Target Qwen3-8B in 4-bit NF4, drafter Qwen3-0.6B in bf16, greedy, 30 held-out dev
prompts each for math (GSM8K train holdout), code (MBPP validation) and chat (UltraChat test_sft), 160 new tokens.
At every 8th position of the target's own continuation the drafter grows a beam forest (top-8 children, 64 paths
kept per depth, depth 8, 456 nodes). We compare checking the B nodes with the highest cumulative drafter
log-probability against the best any choice could do from the same forest (the depth of the target's true path
inside the forest). Accepted length is out of 8.

| Set | Positions | Best possible | Drafter, 8 nodes | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|
| Chat | 540 | 5.21 | 2.62 (+99%) | 3.28 (+59%) | 3.83 (+36%) | 4.30 (+21%) | 4.70 (+11%) |
| Code | 494 | 7.24 | 3.95 (+83%) | 5.15 (+41%) | 6.05 (+20%) | 6.59 (+10%) | 6.90 (+5%) |
| Math | 569 | 7.67 | 5.03 (+52%) | 6.11 (+26%) | 6.79 (+13%) | 7.17 (+7%) | 7.43 (+3%) |

95% bootstrap intervals for the gap at 32 nodes: chat 31 to 41%, code 17 to 23%, math 11 to 16%.

- The ceiling clears the pre-set go line (20% at 32 nodes) on chat and code, and sits between the go and kill lines on
  math. These are upper bounds on what any selector could recover, not what a gist-guided one will.
- On chat the forest misses the target's path often (best possible 5.2 of 8), so how the forest is grown matters too.
- Stage 0 (hardware knee) was not informative: with bitsandbytes 4-bit weights a target forward costs about 100 ms
  for anything from 1 to 128 new tokens, because dequantisation dominates. Which budget column applies in practice
  needs a measurement with real inference kernels.

## Layer sweep (Qwen3-8B vs Qwen3-0.6B)

`gen_q3.py`, `probes_q3.py`, results in `results/probes_q3.json`. The 8B's own greedy answers (Ollama Q4_K_M,
thinking off) to 600 training prompts and 40 held-out dev prompts per set (math, code, chat, never the eval sets).
Linear probes on every second layer of Qwen3-8B (4-bit NF4) and Qwen3-0.6B, 24 positions per training answer.
Ridge R² for the mean embedding of the next 16, 64 or 128 tokens and for log remaining length.

| Probe (best layer, its depth) | Math 8B | Math 0.6B | Code 8B | Code 0.6B | Chat 8B | Chat 0.6B |
|---|---|---|---|---|---|---|
| Next 64 tokens summary | 0.32 (78%) | 0.33 (64%) | 0.16 (94%) | 0.20 (79%) | 0.07 (100%) | 0.19 (71%) |
| Remaining length | 0.72 (50%) | 0.68 (64%) | 0.45 (50%) | 0.38 (57%) | 0.46 (56%) | 0.44 (64%) |
| Next token (top-1 acc.) | 0.82 | 0.77 | 0.77 | 0.69 | 0.62 | 0.53 |

Embedding-only baseline for the 64-token summary: math 0.12, code 0.04, chat 0.04.

This first sweep was underpowered: on chat the 8B summary probe was below zero at most layers, consistent with
a 4,096-wide state overfitting about 14,000 rows. It is superseded by the fair rerun below.

### Fair rerun

`probes_q3_fair.py`, results in `results/probes_q3_fair.json` (best per cell in `probes_q3_fair_best.json`).
Same answers, 64 positions per training answer (36,719 rows), ridge penalty grid up to 10 times the row count,
and each layer reduced by PCA (fit on training rows) to the same width for both models. Six layers per model.

| Equal width 1,024, best layer | Math 8B | Math 0.6B | Code 8B | Code 0.6B | Chat 8B | Chat 0.6B |
|---|---|---|---|---|---|---|
| Next 16 tokens | 0.37 | 0.35 | 0.29 | 0.26 | 0.17 | 0.14 |
| Next 64 tokens | 0.38 | 0.35 | 0.25 | 0.23 | 0.23 | 0.20 |
| Next 128 tokens | 0.40 | 0.38 | 0.23 | 0.20 | 0.25 | 0.23 |
| Remaining length | 0.74 | 0.69 | 0.49 | 0.41 | 0.50 | 0.47 |

- The 8B is ahead in every cell, at widths 256 and 1,024 and at full width. The content edge is +0.02 to +0.03 R²,
  the length edge +0.03 to +0.08 (up to +0.13 at full width on code).
- Remaining length peaks at layer 18 of 36 in the 8B. Content summaries peak at layers 24 to 36.
- So the target does know more about its own future, and which layer to tap depends on what is predicted.

## Sketch A on DFlash (plan vector into the block inputs)

`dflash_plan.py`, results in `results/dflash_plan.json`. Target Qwen3-8B (4-bit), drafter z-lab/Qwen3-8B-DFlash-b16
frozen. A learned plan vector is added to the input embeddings of slots 1 to 15 of each 16-token block: its direction
from a linear map of the target's layer 18 and 20 states at the last verified position, its length a learned scalar
that starts at 5% of a token-embedding norm. Trained for 549 steps on the 8B's own answers (8 blocks per answer).
The oracle variant uses the mean layer-20 state over the true next 64 positions instead. Evaluated on held-out dev
answers, every 16th position, teacher-forced context, greedy chain acceptance.

| Set | Blocks | Plain DFlash | + mid-layer plan | + oracle plan |
|---|---|---|---|---|
| Math | see json | 7.46 | 7.49 (+0.4%) | 7.44 (-0.3%) |
| Code | see json | 6.71 | 6.91 (+3.0%) | 6.83 (+1.8%) |
| Chat | see json | 2.94 | 2.95 (+0.3%) | 2.95 (+0.4%) |

No meaningful effect. The learned strength grew from 0.05 to 0.125, so the head was used, but neither the realistic
plan nor the oracle moves acceptance. Because the oracle fails too, this points at the injection pathway (input
embeddings) rather than at the value of future information. DFlash reads the target through context keys and values,
which `kv_oracle.py` tests next without training.

A first version of the head (zero-initialised projection with a norm cap) damaged the drafter within 3 steps
(6.96 to 4.82 accepted on the smoke set) because Adam moved all weights at once and the output hit the cap. The
smoke test caught it before the full run.

## Can DFlash use the target's future?

All on held-out dev answers, Qwen3-8B (4-bit) target, z-lab DFlash, accepted tokens per 16-token block. The "true
future" is the mean of the target's context features (layers 1, 9, 17, 25, 33) over the next 16 positions, i.e. the
block being drafted, added as one extra context position.

| What DFlash is given | Drafter | Math | Code | Chat | Script / results |
|---|---|---|---|---|---|
| Nothing extra | as released | 7.46 | 6.71 | 2.94 | |
| True future, raw | frozen | 5.44 | 4.11 | 2.04 | `kv_oracle.py`, `kv_oracle.json` |
| True future, trained adapter (300 answers) | frozen | 7.43 | 6.71 | 2.94 | `kv_oracle_train.py`, `kv_oracle_trained.json` |
| Nothing extra, LoRA r16 (control, 300 answers) | fine-tuned | 7.44 | 6.87 | 2.97 | `kv_lora.py`, `kv_lora.json` |
| True future, LoRA r16 + adapter (300 answers) | fine-tuned | 7.48 | 6.81 | 2.95 | `kv_lora.py`, `kv_lora.json` |

Paired difference, true future minus control (bootstrap 95% CI over blocks): math +0.04 (-0.02 to +0.12), code -0.06
(-0.16 to +0.03), chat -0.02 (-0.05 to +0.00). Final training losses are identical (2.585 vs 2.586).

Even the true states of the block being drafted, summarised into one vector, do not help DFlash at this training
budget, frozen or fine-tuned. Fed raw without training they hurt. Together with Sketch A this is a negative result
for conditioning a strong block drafter on a single plan vector. Much longer training or a richer plan (per-position
rather than averaged) could still behave differently; neither was tested.

## Tree headroom on DFlash

`dflash_tree_gap.py`, results in `results/dflash_tree_gap.json`. One DFlash pass gives a distribution per slot; a tree
keeps the top 4 tokens per slot and checks the most probable prefixes (product of per-slot probabilities). Accepted
tokens per pass on held-out dev answers, teacher-forced context.

| Set | Chain | Tree 32 | Tree 64 | Tree 128 | Tree 256 | Best possible (top 4 per slot) |
|---|---|---|---|---|---|---|
| Math | 7.46 | 8.63 | 9.03 | 9.31 | 9.52 | 10.49 |
| Code | 6.77 | 8.21 | 8.68 | 9.10 | 9.37 | 10.72 |
| Chat | 2.96 | 3.85 | 4.00 | 4.13 | 4.20 | 4.43 |

- A tree adds 16 to 35% accepted tokens over the single chain at 32 to 64 nodes. Known: DDTree.
- Room left for a better selector at 64 nodes: +16% math, +24% code, +11% chat (chat is limited by the candidates).
- Closest prior work for a learned selector: GRAFT (2608.20375) re-scores DDTree edges on DFlash with a scorer that
  sees only the parent token, child token and depth (frozen target embeddings), distilled from target traces, and
  reports +19% throughput over DDTree on Qwen3-8B. It uses no target hidden state or forecast at scoring time.

## Stage-2 gate: chunk autoencoder over Qwen3-8B states

`cache_states.py`, `ae_gate.py`, results in `results/ae_gate_*.json`. A VAE compresses k=8 consecutive positions of
Qwen3-8B (4-bit) hidden states (layers 18, 24, 30 and the final normed state, or the final row only) into one latent
and decodes them back. Score: how often the decoded final state, through the frozen LM head, picks the target's own
top-1, on held-out dev windows. Gate fixed before training: 99.5% math and code, 99% chat.

| Latent | Rows | Math | Code | Chat | Gate |
|---|---|---|---|---|---|
| 512 | full grid | 93.9% | 92.5% | 77.8% | fail |
| 1024 | full grid | 94.0% | 92.6% | 77.6% | fail |
| 2048 | full grid | 94.6% | 93.0% | 78.3% | fail |
| 1024 | final row only | 93.7% | 92.3% | 76.8% | fail |

- All four fail by a wide margin. By the pre-set rule this design does not proceed as is.
- Latent size and the middle layers barely matter. Training and dev agreement are about equal (underfit), and the
  decoders ignore latent noise, which points at the recipe (small decoder, 12k steps, VAE noise and latent dropout)
  more than at a hard limit.
- Diagnostics (do not reopen the gate). A plain autoencoder (z2048, no sampling noise, no KL) gets 93.3% / 92.6% /
  75.0%, so the VAE noise was not what held it back. A larger decoder (161M parameters, 36k steps) reaches 94.5% on
  train but only 86.0% / 84.6% / 61.6% on dev, so it overfits. More data looks more useful than more parameters,
  which is why the answers were continued to 1,024 tokens (`tokens/*_long.jsonl`, `cache_states.py --long`).
- Data note: the 4-bit HF target and the Ollama build that wrote the answers pick a different next token at 4% / 6%
  / 13% of math / code / chat positions. Earlier offline teacher-forced numbers in this file were scored against the
  Ollama text, which lowers their absolute values, chat most. Comparisons within each table are unaffected.

### Gist autoencoder and stage 2 (later runs)

- Adding the head feature (`--headfeat`, the frozen head row of each final state's own top-1, fed to the encoder)
  plus a top-1 cross-entropy loss lifts stage 1 to 99.4% / 99.0% / 97.3% (`ae_gate_gist_hf_z2048.json`). Below the
  gate on all three sets (99.5 / 99.5 / 99.0%). Prefix context (`--ctx 8`) made stage 1 worse (92.5 / 91.0 / 72.9).
- A token-chunk VAE control (`tok_vae_z128.json`, tokens only) gets 98.4% / 97.4% / 94.8%.
- Stage 2 (`stage2.py`, deterministic, 6,000 steps) predicts the next chunk latent. Accepted per round out of 9,
  math / code / chat: gist 2.17 / 1.80 / 1.34, token latent 1.75 / 1.46 / 1.18, true next latent 8.51 / 8.47 /
  7.63. With 64 positions of history: gist 2.30 / 1.81 / 1.31, token 1.88 / 1.48 / 1.18 (`stage2_*.json`).
- The gist beats the token latent in every setting. Both are far below DFlash (7.46 / 6.77 / 2.96 out of 16). The
  predictor is the limit: it explains about 10% of the next latent's variance, and more history does not help.

## Gister (the two-stage design, named 2026-11-24)

Offline, accepted per round out of 9 (8 drafted + bonus), same dev rounds, labels = the HF target's own top-1:

| Stage 2 | Math | Code | Chat |
|---|---|---|---|
| gist only, 20M | 2.17 | 1.80 | 1.34 |
| gist only, 154M, 20k steps (overfits: train CE 1.45, dev flat) | 2.30 | 2.19 | 1.30 |
| gist + raw target states (`--raw`), 20M = **Gister v1** | **2.80** | **2.23** | **1.43** |
| same, token loss only (`--lat_w 0`) | 2.79 | 2.20 | 1.40 |
| flow mode, naive token loss (collapsed) | 1.21 | 1.13 | 1.06 |
| DFlash chain capped at 8 (`dflash_s2.py`) | 5.88 | 5.59 | 2.68 |
| stage-1 ceiling (true next gist) | 8.51 | 8.47 | 7.63 |

Side experiments on DFlash (not Gister):
- DFlash block outputs mapped to the gist latent and decoded by stage 1 (`dflash_s2.py --head latent`): 5.30 / 4.51 /
  2.25. A refine head without the latent: 5.61 / 5.27 / 2.45. Both lose to DFlash's own chain.
- Vocabulary pruning (`prune_oracle.py`): restricting DFlash to the true tokens of the next 256 positions lifts chat
  chain 2.69 to 3.12 and tree-64 3.69 to 4.30. Any bag that can miss a drafted token roughly halves acceptance.

End to end (`e2e.py`, one lossless loop for every drafter, reproduces DFlash's `spec_generate` round for round;
Qwen3-8B nf4, RTX 4070 Ti, greedy). Smoke test, 2 prompts, 48 new tokens:

| | MT-Bench tau | speedup | GSM8K tau | speedup |
|---|---|---|---|---|
| Gister v1 | 1.45 | 1.38x | 1.90 | 1.74x |
| DFlash | 2.65 | 2.28x | 4.71 | 3.79x |

DFlash on our own dev sets (`dflash_eval.py`, 10 prompts, 160 tokens): tau 5.41 / 4.71 / 2.37, speedup 4.75x /
3.80x / 2.03x. The full benchmark (20 prompts per set, 256 tokens, with Gister v2) is in `results/e2e.json` once run.

Gister v2: stage 2 at 154M trained on 2M positions of Qwen3-8B's own non-thinking PerfectBlend answers
(`cache_states.py --pb_tokens 2.0`, `stage2.py --train_cache cache_pb`), 4x v1's data. Offline 3.08 / 2.22 / 1.45
(v1 2.80 / 2.23 / 1.43). Training loss levelled at about 3.8 to 4.1 nats after more than one pass over the 2M
PerfectBlend positions; data helped math only. That fits a deterministic predictor averaging over futures, but the flat
code and chat results are also confounded with PerfectBlend's domain mix. The sampler (flow matching from v2) is next.

Full end-to-end benchmark (`e2e.py`, 20 prompts per set, 256 new tokens, `results/e2e.json`):

| | MT-Bench tau | speedup | GSM8K tau | speedup | HumanEval tau | speedup |
|---|---|---|---|---|---|---|
| Gister v2 | 1.60 | 1.48x | 2.39 | 2.24x | 1.66 | 1.52x |
| Gister v1 | 1.42 | 1.25x | 2.08 | 1.93x | 1.48 | 1.41x |
| DFlash capped at 8 | 3.07 | 2.49x | 4.91 | 4.43x | 7.27 | 6.63x |
| DFlash (15 drafted) | 3.15 | 2.49x | 5.51 | 5.17x | 10.01 | 8.26x |

Plain greedy ran at 21.2 / 15.9 / 19.2 tok/s. The low GSM8K baseline inflates every GSM8K speedup by the same factor.
Drafters with the same verification width give identical text (dflash8, gister, gister_v2 in
`e2e.json`). Wider verification (DFlash's 16 tokens, trees of 30 to 120 nodes) diverges differently at near-ties
under 4-bit batched numerics, so the identical counts differ by drafter. For the width-9 drafters, 15 / 9 / 5 prompts
differ from plain greedy (counts match across those drafters; pairwise agreement was checked only in the smoke test).

Gister sampler (`stage2.py --mode flow --raw --train_cache cache_pb --d 1024 --depth 12 --dec_w 0 --t_shift 3
--nfe 8 --init stage2_gister_v2 --steps 10000 --bestk 4,8,16`), offline, out of 9:

| | Math | Code | Chat |
|---|---|---|---|
| v2 deterministic | 3.08 | 2.22 | 1.45 |
| sampler, one sample | 2.71 | 1.83 | 1.36 |
| sampler, best of 4 | 3.36 | 2.38 | 1.58 |
| sampler, best of 8 | 3.67 | 2.59 | 1.70 |
| sampler, best of 16 | 4.00 | 2.86 | 1.84 |

Samples are distinct (8 draws give 7.6 to 8 different drafts). Best-of-K beats the deterministic predictor from K=4
and grows per doubling of K (8 to 16, 2 flow steps) by +0.31 math, +0.25 code, +0.13 chat. Best-of-K is the ceiling of a K-branch tree, not a
speedup: verification of the tree is not built yet. Still far below DFlash's single chain (5.88 / 5.59 / 2.68).

Flow steps at sampling time (offline, best of 8, out of 9): 8 steps 3.67 / 2.59 / 1.70, 2 steps 3.72 / 2.64 / 1.72,
1 step 3.50 / 2.49 / 1.63. One step makes each sample more accurate but less diverse. Two steps is the setting.

Sampled trees end to end (`e2e.py`, K sampled futures plus v2's chain merged into one token tree, verified in one
target pass with a tree attention mask; a one-branch tree reproduces the chain path exactly). 20 prompts per set,
256 new tokens, `results/e2e_tree.json`:

| | MT-Bench tau | speedup | GSM8K tau | speedup | HumanEval tau | speedup |
|---|---|---|---|---|---|---|
| Gister v2 chain | 1.60 | 1.54x | 2.39 | 2.31x | 1.66 | 1.60x |
| tree, 8 samples + chain, 2 steps | 1.89 | 1.31x | 2.94 | 2.28x | 1.99 | 1.53x |
| tree, 16 samples + chain, 2 steps | 2.01 | 1.33x | 3.12 | 2.24x | 2.08 | 1.48x |
| tree, 8 samples + chain, 1 step | 1.80 | 1.36x | 2.79 | 2.29x | 1.90 | 1.62x |

This `e2e_tree.json` run was taken while desktop apps shared the GPU (plain greedy 10 tok/s against 21), which
distorts the tree-vs-chain speed comparison. Its tau values stand; its speedups are superseded by `e2e_final.json` in
the audited section below.

## Gister, audited state (25 Nov)

This section replaces earlier Gister statements where they differ. Its claims were checked in three external reviews
(`deck/neon_audit_b1_code.md`, `deck/neon_audit_b2_claims.md`, `deck/neon_audit_b3_novelty.md`), and the issues they
raised are fixed here. All results are greedy decoding on the Qwen3-8B 4-bit target, one RTX 4070 Ti.

**Stage 1 fails its gate on both checks.** Top-1 agreement 99.41 / 99.01 / 97.32% is below the 99.5 / 99.5 / 99.0%
bar on all three sets, and total variation 0.060 / 0.064 / 0.202 is far above 0.005 / 0.005 / 0.01
(`ae_gate_gist_hf_z2048.json`, `agree_ok` and `tv_ok` false everywhere). Everything below is scoped to greedy
decoding. Stage 1 only reached this level once the encoder was given the LM-head row of each state's own top-1
token (`--headfeat`), and the middle layers barely mattered (z1024 final row only 93.7 / 92.3 / 76.8 vs full grid
94.0 / 92.6 / 77.6). So the latent is closer to "token chunk plus the final-layer state" than to a multi-layer plan.

**Stage 2, offline (accepted per round out of 9, HF top-1 labels, same dev rounds):**

| Stage 2 | Math | Code | Chat |
|---|---|---|---|
| deterministic, 154M, 2M positions (v2) | 3.08 | 2.22 | 1.45 |
| v1: deterministic + raw states, 20M, 512K positions | 2.80 | 2.23 | 1.43 |
| v1 + 7 prior gist chunks (`--hist 8`, 64 positions in total) | 2.81 | 2.15 | 1.39 |
| v1 inputs, no latent: 8 tokens predicted directly (`--direct`) | 2.73 | 2.16 | 1.39 |
| flow sampler 158M / 2M positions, warm-started from v2, 2 steps: one sample | 2.85 | 1.95 | 1.41 |
| best of 8 | 3.72 (+21%) | 2.64 (+19%) | 1.72 (+18%) |
| best of 16 | 4.04 (+31%) | 2.89 (+30%) | 1.85 (+27%) |
| flow sampler 285M / 6M positions, from scratch, 2 steps: one sample | 3.04 | 2.08 | 1.46 |
| best of 8 | 3.94 (+28%) | 2.73 (+23%) | 1.77 (+22%) |
| best of 16 | 4.18 (+36%) | 3.06 (+38%) | 1.92 (+32%) |
| flow sampler 158M / 2M positions + raw states of 56 earlier positions, warm-started from v2, batch 64, 2 steps: one sample | 3.00 | 2.00 | 1.43 |
| best of 8 | 4.01 | 2.66 | 1.75 |
| best of 16 | 4.25 | 2.83 | 1.82 |
| DFlash chain, capped at 8 drafted (`dflash_s2_latent.json`, `accepted_chain`) | 5.88 | 5.59 | 2.68 |
| DFlash tree, 64 nodes (`prune_oracle.json`, `none.tree64`) | 7.00 | 6.94 | 3.69 |
| stage-1 reconstruction ceiling (true next gist decoded, not a drafter) | 8.51 | 8.47 | 7.63 |

Percentages are against the deterministic v2. Per doubling of K (8 to 16): +0.31 / +0.25 / +0.13 (158M),
+0.24 / +0.33 / +0.15 (285M). Distinct drafts out of 8: 7.3 to 8.0. Drafted tokens accepted (tau minus the free bonus
token): Gister 285M best of 8 is 2.94 / 1.73 / 0.77, which is 49% / 29% / 29% of DFlash's 64-node tree and 60% / 38% /
46% of DFlash's capped chain. Best of K is the ceiling for a K-branch tree, not a measured speedup.

The 158M to 285M step changes several things at once (158M to 285M parameters through width and depth, 2M to 6M
positions, 30k to 24k total steps, warm start vs from scratch), so it is one confounded scaling point: +3 to 7% per
metric. The gist-history ablation shows no gain. **Direct-token control** (`stage2_gist_hf_z2048_det_direct_raw.json`,
same inputs, size and steps as v1, predicting the 8 head-input states directly with no latent): 2.73 / 2.16 / 1.39
against 2.80 / 2.23 / 1.43 for v1 with the gist latent. With the latent, acceptance is about 3% higher. **Full-context test**
(`stage2_gist_hf_z2048_flow_raw_ctx56_cache_pb_d1024x12_s20000_nfe2.json`, raw states of the 56 earlier positions,
batch 64, 20k steps): best of 16 is 4.25 / 2.83 / 1.82 against 4.04 / 2.89 / 1.85 for the 158M sampler without it,
+5% on math and no change on code or chat. Batch size differs, so this is not a controlled comparison.

**End to end on an idle GPU** (`e2e_final.json`, 20 prompts per set, 256 new tokens, plain greedy 22.5 / 21.6 /
22.3 tok/s, bf16 drafting for every drafter, no extra round after end of sequence):

| | MT-Bench tau | speedup | GSM8K tau | speedup | HumanEval tau | speedup | ms per round |
|---|---|---|---|---|---|---|---|
| DFlash (released, 15 drafted) | 3.15 | 2.69x | 5.52 | 4.73x | 10.01 | 7.89x | 52 to 57 |
| Gister v2 chain | 1.60 | 1.34x | 2.38 | 2.05x | 1.67 | 1.40x | 53 to 54 |
| Gister tree, 4 samples + chain (285M sampler) | 1.89 | 1.08x | 2.96 | 1.75x | 2.00 | 1.16x | 77 to 78 |
| Gister tree, 8 samples + chain (285M sampler) | 1.99 | 1.07x | 3.13 | 1.77x | 2.11 | 1.14x | 82 to 83 |

- Gister's chain and DFlash take the same time per round. The difference is in accepted tokens.
- Trees raise tau 18 to 31% but cost 45 to 56% more per round (33 to 37 nodes for 4 samples, 54 to 62 for 8), so
  every tree is slower than the chain here.
  Profiled on the idle GPU (`profile_tree.py`, 09:00 run): plain decode step 39 ms; chain draft 11 ms + verify 40 ms;
  tree of 67 nodes draft 30 ms + verify 62 ms; 120 nodes draft 49 ms + verify 78 ms (285M sampler, before the
bf16-drafting fix, fixed context of prompt + 200 tokens; the end-to-end rows above include the fix). Verification is not flat in
  tree size on an idle GPU (the earlier flat profile was taken under desktop contention), and the 4-bit bitsandbytes
  kernels make all of these numbers specific to this setup. bf16 or FP8 engines change both draft and verify costs.
- Identical outputs: compared with target-only greedy decoding, outputs match on 4 to 18 of 20 prompts per set for
  Gister and 4 to 15 for DFlash. In the earlier DFlash check every first difference was at a near-tie, which batched 4-bit verification resolves
  differently from single-token steps (counts in the JSON).

**Data hygiene.** PerfectBlend prompts used for training (first 14,000 conversations of shard 0) were checked
against all benchmark and dev prompts: no exact or near (first 120 characters) matches for MT-Bench, GSM8K test,
HumanEval, MBPP validation or UltraChat test; 1 of 40 GSM8K-train holdout prompts shares its first 120 characters.

**Positioning (from the novelty review).** The closest prior work is ChainFlow (VAE over target hidden states, flow
to 8 future positions, frozen LM head, greedy tree), FlashMap (flow from multi-layer Qwen3-8B states, several noise
draws merged into a tree), MFD and DBLast (latent-conditioned coherent multi-future trees), CALM (one continuous latent
per token chunk) and DFlash with DDTree. Gister's specific combination is sampling several continuous latents of an
8-token chunk of the target's own future states and decoding each jointly. We have not compared it with a per-position product tree at
equal node count. That comparison is one of the proposed experiments.

**Proposed next step.** The stage-1 gate rewards exact reconstruction of 8 tokens, and the resulting latent mostly
encodes those tokens. The proposal is a latent over the next 64 to 256 tokens, trained by prediction in latent space
(JEPA-style, EMA target encoder, variance regularization), sampled with flow matching, and read by a decoder trained
with it from the start. The first test trains that decoder with the latent of the true future and compares it with the
same decoder without the latent. Earlier plan experiments (sections above) added one averaged future vector to DFlash, frozen
and fine-tuned with LoRA r16 on 300 answers, and saw no gain even with the true future. They do not test a decoder
trained from scratch with a latent of the future. Test 1 protocol: train the latent encoder first and freeze it, then
train two decoders from scratch, one given z of the true future and one given z = 0. Continue only if the first accepts
at least 20% more tokens on code and chat. Estimated compute for test 1: about one week on 4 A100- or H100-class GPUs,
including data generation. Caching 4 layers for 100M positions takes about 3.2 TB, so larger runs compute states on
the fly.

## Long-horizon gist and paragraph plan (25 to 26 Nov)

These runs start the proposed next step at small scale on one RTX 4070 Ti. Unless noted, the drafter is the 24M
direct drafter of `stage2.py --mode det --direct --raw` (8 output queries over the target's states at the current
chunk), trained 10k steps on `cache_pb`, and the score is accepted tokens per round out of 9 on the dev sets,
math / code / chat.

**Gist of the stretch after the draft.** The gist covers rows s+16 onward, so it holds no drafted token.
`stage2.py --far`: the target's top-1 tokens over that stretch, embedded with the frozen head rows, compressed to
8 × 64 numbers and fed to the drafter as 8 extra tokens. `jepa_gist.py`: an encoder over the target's states at
those rows (layers 24 and 36) and a predictor from the 64 rows of context (layers 30 and 36), trained with VICReg
(invariance 25, variance 25, covariance 1, 4k steps). The state gist can carry drafted tokens indirectly, since the
states after the draft have read them.

| Gist given to the drafter | Math | Code | Chat |
|---|---|---|---|
| none (same network, gist zeroed) | 3.02 | 2.10 | 1.48 |
| true next 64 tokens after the draft | 3.19 | 2.17 | 1.50 |
| true next 256 tokens after the draft | 3.09 | 2.18 | 1.46 |
| JEPA encoder of the true future states | 3.02 | 2.08 | 1.49 |
| JEPA predictor from context | 2.96 | 2.13 | 1.46 |

The largest gain is 6% on math with the true 64 tokens, below the 20% bar. The JEPA predictor's error is 0.027 on
train and 0.052 to 0.071 on dev. Its encoder did not collapse (per-dimension spread 0.70 at step 1,000). This setup
only tests a gist used as an extra drafter input. It does not test a gist that stage 2 predicts and the decoder
drafts from block by block.

**Paragraph plan** (`plan_gist.py`). The plan is G ordered block latents over the next 8·G tokens, and its first
block is the block drafted this round. An encoder over the target's own next tokens (head-row embeddings) gives
G × c numbers, scaled to unit average power, and a Gaussian channel with noise σ limits it to at most
G·c·½·log₂(1 + 1/σ²) bits. The 23M decoder drafts 8 tokens from the plan, the current chunk's states and the offset
of the drafted block inside the plan, so one plan can serve later rounds. A deterministic predictor and a
rectified-flow sampler learn the plan from the states at the plan's anchor. Evaluation covers the oracle plan, the
predicted plan, a zeroed plan, plans made 16 to 504 tokens earlier, and best of K sampled plans against the K most
likely drafts of the decoder's own per-slot distribution, with trie node counts.

- The first run (64 tokens, c = 4, tanh output) collapsed. The encoder's output spread fell to 0 by step 2,000 as
  tanh saturated, and the oracle plan then scored the same as no plan (2.99 / 2.16 / 1.49). Its numbers say nothing
  about plans. Power normalisation and a noise warmup replaced the tanh.
- 2,000-step checks after the fix (undertrained, for debugging only). With c = 4 and σ = 0.5 the oracle plan gives
  2.33 against 2.25 without a plan on math. With c = 64 and σ = 0.05 it gives 2.63 against 2.02 on math (+30%), +4%
  on code and none on chat. Best of 8 sampled plans is level with or below the 8 most likely per-position drafts
  (math 2.49 at 29 trie nodes against 2.54 at 31 with c = 4, 2.51 at 45 against 2.53 at 33 with c = 64). Zeroing the raw states in half of the training rounds weakened the decoder
  and did not make it use the plan.
- Full runs (10k decoder steps plus 2k predictor steps, `results/plan_pw_*.json`). Offset 0, drafted block = first
  plan block. "Zeroed" is the same decoder with its plan input set to zero. The control is the same 23M decoder
  trained without a plan. Trees: best of 8 flow samples against the decoder's own 8 most likely drafts, with trie
  nodes in brackets (math).

| Run | Oracle plan | Predicted plan (det) | Zeroed | Best of 8 flow plans | 8 most likely drafts |
|---|---|---|---|---|---|
| control, no plan | | | 3.02 / 2.16 / 1.48 | | 3.46 (29) |
| 64 tokens, c = 16, σ = 0.25 | 5.51 / 4.10 / 2.03 | 2.67 / 1.98 / 1.42 | 2.22 / 1.85 / 1.41 | 3.18 (49) | 3.13 (35) |
| 256 tokens, c = 16, σ = 0.25 | 4.79 / 3.29 / 1.73 | 2.73 / 2.10 / 1.48 | 2.49 / 2.03 / 1.46 | 3.10 (47) | 3.18 (33) |
| 512 tokens, c = 16, σ = 0.25, B 64 | 3.97 / 2.42 / 1.55 | 2.45 / 1.95 / 1.45 | 2.42 / 1.89 / 1.42 | 2.77 (47) | 2.79 (36) |
| 256 tokens, c = 64, σ = 0.05 | 5.89 / 4.69 / 2.46 | 2.57 / 1.96 / 1.43 | 2.17 / 1.94 / 1.41 | 2.79 (52) | 2.94 (33) |
| 256 tokens, predictor reads the previous plan | 4.91 / 3.29 / 1.75 | 2.79 / 2.04 / 1.48 | 2.61 / 2.09 / 1.47 | 3.13 (46) | 3.17 (34) |

- The oracle plan beats the control by 17 to 117% in the batch-128 runs (5 to 32% for the 512-token run at batch 64,
  which saw half the training rounds). Every predicted plan is below the control on math and code and at most level
  with it on chat (1.48 against 1.48). The table's predicted-plan column adds channel noise and feeds the predictor's
  output as is, at 0.21 to 0.45 of the power the decoder was trained on. Rescaled to unit power without noise
  (`plan_reeval.py`, `results/plan_reeval.json`) it gives 2.71 / 1.98 / 1.43 (64 tokens) and 2.76 / 2.06 / 1.47
  (256 tokens), still below the control.
- The plan length comparison is confounded. A longer plan at the same c per block has more total capacity, a longer
  attention sequence for the same 10k steps, and (512 tokens) half the batch. The oracle at offset 0 falls from 5.51
  (64 tokens) to 4.79 (256) and 3.97 (512) on math.
- Predicting the next plan from the plan of the previous 256 verified tokens (the gist-to-gist step) does not help:
  2.79 / 2.04 / 1.48 against 2.73 / 2.10 / 1.48. The previous window is left-padded near the start of an answer, while
  the encoder was trained on right-padded future windows, so this test is weaker than it looks. The predictor errors
  (0.60, 0.66) come from two runs with different encoders and are not directly comparable.
- Plans made earlier (offsets 16 to 248) give at most small gains, also for the oracle (for example 3.07 against 2.95
  zeroed at offset 16 with block-aligned offsets, 3.12 against 3.03 with the plan shifted to the drafted block). Two
  sampling problems make this test inconclusive. An offset that reaches before the start of the answer falls back to
  0, so offset 0 is heavily over-represented (only 7 to 19% of dev rounds have room for offset 248), and the decoder
  learns a separate embedding for every offset, each seen in well under 1% of rounds. The encoder's block queries
  also attend to the whole window. With block-aligned offsets the oracle at offset 0 reaches 7.80 / 7.01 / 5.25, which
  shows a tight code of the drafted block at offset 0 only.
- Rerun with offsets sampled only inside the answer and the shifted decoder given only the offset within a block
  (`--oblock --rel --ofix`, `plan_pw_G32c16s0.25_ob_rel_of.json`). The plan now helps about equally at every offset,
  so a plan made 16 to 128 tokens earlier works as well as a fresh one (math, oracle against zeroed: 3.17 / 2.88 at
  offset 0, 3.18 / 2.92 at 16, 3.25 / 3.01 at 48, 3.36 / 3.14 at 128). But the oracle's gain falls to about 0.3 on
  math and about 0.04 on code and chat, since each block must now serve as the drafted block. The predicted plan gives
  3.02 / 2.14 / 1.46 at offset 0 and 3.15 / 2.24 / 1.48 at offset 48, level with the control.
- Reading. The plan encoder is trained only to help a decoder that already reads the context, so it learns what the
  context does not contain, mostly the identity of the drafted tokens. A predictor from the same context cannot
  supply that. `cvae_plan.py` makes the predictor a learned prior and charges the plan for each nat the prior cannot
  predict (KL weight β, ELBO at β = 1).

**Decoder fine-tuned on predicted plans** (`ft_pred.py`, `results/ft_pred_*.json`). The decoders above were trained
on the oracle plan and see the predictor's plan only at drafting time. Each arm starts from a saved checkpoint and
trains 4,000 more steps (batch 128, learning rate 1e-4, offset 0) on `cache_pb`, with the plan the decoder gets at
drafting time. `dec` freezes the predictor and fine-tunes the decoder on its plan (unit average power, no channel
noise). `e2e` fine-tunes the predictor (15M) and the decoder (23M) together on the token loss. `ctrl` fine-tunes the
no-plan control with the same recipe, so all arms have the same number of decoder steps. Scored on the
`plan_reeval.py` dev rounds at offset 0, one draft and the K most likely drafts with trie nodes (math).

| Arm | One draft | 8 most likely drafts, math (nodes) | 16 most likely drafts, math (nodes) |
|---|---|---|---|
| control, fine-tuned | 3.02 / 2.22 / 1.47 | 3.46 (29) | 3.65 (51) |
| 64 tokens, before fine-tuning | 2.71 / 1.98 / 1.43 | 3.17 (35) | 3.31 (63) |
| 64 tokens, `dec` | 3.07 / 2.20 / 1.45 | 3.52 (29) | 3.67 (50) |
| 64 tokens, `e2e` | 3.09 / 2.20 / 1.45 | 3.49 (29) | 3.69 (50) |
| 256 tokens, before fine-tuning | 2.76 / 2.06 / 1.47 | 3.20 (34) | 3.31 (61) |
| 256 tokens, `dec` | 3.04 / 2.22 / 1.47 | 3.42 (29) | 3.64 (50) |
| 256 tokens, `e2e` | 3.01 / 2.24 / 1.48 | 3.44 (29) | 3.64 (51) |

- Training on the predicted plan brings the plan decoders up to the control and no further. All arms are within 0.08
  of the control on every set, at matched trie nodes as well. One seed per arm.
- Training the predictor and decoder together adds 15M trainable parameters and gives the same result as freezing
  the predictor, so extra capacity on this path does not help at 4,000 steps either.
- The plan arms start far from the control, since their decoders were trained on oracle plans (token loss 8.11
  against 3.49 on the first batch), and end at 3.47 to 3.49 against 3.35 on the final batch (the same batches for
  all 64-token arms). They may not have converged in 4,000 steps.
- All arms run at offset 0, so the predictor reads the same 32 states as the decoder and its plan can add capacity
  or structure but no new information. Within these checkpoints and this budget it adds neither.

**Plan as a conditional VAE** (`cvae_plan.py`, `results/cvae_*.json`). Posterior over the next 256 target tokens,
Gaussian prior from the current chunk's states, the same 23M decoder trained on posterior samples, loss
NLL + β·KL/8 with the KL weight rising over 3k steps, 10k steps, offset 0. At drafting time the plan comes from the
prior. Trees: best of K prior samples, with trie nodes in brackets (math).

| β | KL per round | True-future plan (posterior mean) | Prior mean | Prior sample | Best of 4 prior samples | Best of 8 prior samples |
|---|---|---|---|---|---|---|
| 1 | 0.003 bits | 2.91 / 2.01 / 1.43 | 2.91 / 2.01 / 1.43 | 3.06 / 2.08 / 1.46 | 3.14 (10) | 3.15 (11) |
| 0.3 | 8 to 8.5 bits | 3.05 / 2.08 / 1.48 | 2.69 / 1.98 / 1.44 | 2.90 / 2.13 / 1.45 | 3.12 (20) | 3.28 (31) |
| 0.1 | 34 to 44 bits | 4.10 / 2.64 / 1.55 | 2.71 / 2.06 / 1.46 | 2.79 / 2.11 / 1.46 | 3.15 (22) | 3.32 (38) |
| 0.3, decoder also reads prior samples in half the rounds | 0.005 bits | 2.64 / 1.99 / 1.44 | 2.64 / 1.99 / 1.44 | 2.94 / 2.09 / 1.46 | | 3.09 (12) |
| 0.3, 64-token plan | 15 to 19 bits | 4.09 / 2.75 / 1.60 | 2.86 / 2.10 / 1.48 | 2.82 / 2.05 / 1.46 | 3.27 (22) | 3.45 (39) |

The control without a plan gives 3.02 / 2.16 / 1.48 for one draft. Its K most likely drafts give 3.22 / 2.29 / 1.51
(K = 4, 18 / 18 / 19 nodes), 3.47 / 2.38 / 1.54 (K = 8, 29 / 31 / 32 nodes), 3.54 / 2.43 / 1.57 (K = 12, 40 / 42 / 45
nodes) and 3.61 / 2.48 / 1.58 (K = 16, 51 / 53 / 57 nodes). The control was trained with offsets (half of the rounds
at offsets up to 56), the CVAE runs at offset 0 only. A control trained at offset 0 only (`plan_pw_G8c0_poff0.json`)
gives 3.03 / 2.10 / 1.46 for one draft and 3.43 / 2.37 / 1.53 for its 8 most likely drafts (29 / 31 / 32 nodes),
within 0.06 of the first control, so the comparisons below hold for both. Reevaluated on the same rounds
(`plan_reeval.py`), the first control gives 3.01 / 2.15 / 1.48 and 3.47 for its 8 most likely drafts, against
3.02 / 2.16 / 1.48 and 3.46 from its own run.

- At β = 1 the posterior collapses onto the prior and the plan carries no information. In the β = 0.3 run where the
  decoder also read prior samples in half of the rounds, the KL was applied to all rounds although the posterior was
  used in only half, which doubled its weight on the posterior. That run also collapsed. Rerun with the KL masked to
  the posterior rounds (`cvae_G32c16b0.3_mix0.5km.json`) it collapses again (KL 0.008 to 0.009 bits, prior sample
  3.07 / 2.12 / 1.48), so reading prior samples is enough to make the decoder stop using the plan. Smaller β lets the plan carry more (8 to 8.5 bits at β = 0.3, 34 to 44 bits at β = 0.1), and the gap between
  the true-future plan and the prior grows with it.
- The decoder is trained on samples. At β = 0.3 (256 tokens) a latent mean scores lower than a sample (posterior
  sample 3.23 / 2.30 / 1.50 against posterior mean 3.05 / 2.08 / 1.48), and rescaling the mean to the sample norm
  does not close the gap. At β = 0.1 and for the 64-token plan the posterior mean scores higher (math 4.10 against
  3.95, 4.09 against 3.67).
- A single prior draft does not beat the control on math or code. The prior samples of the collapsed β = 1 model and
  of the masked-KL mixing run are above it on math only (3.06 and 3.07 against 3.02).
- Trees at matched trie nodes (`results/plan_reeval.json`, best accepted per round interpolated at 20 / 30 / 40 nodes,
  math): control 3.27 / 3.47 / 3.54. Best of K prior samples: 3.20 / 3.29 / 3.35 (256 tokens, β = 0.3), 3.16 / 3.25 /
  3.36 (β = 0.1), 3.26 / 3.34 / 3.44 (64 tokens, β = 0.3). On code the control gives 2.30 / 2.37 / 2.42 and the best
  plan tree 2.32 / 2.35 / 2.39. On chat the plan trees are 0.01 to 0.04 above the control (1.55 / 1.56 / 1.57 against
  1.51 / 1.54 / 1.56), within the spread between runs. The per-position tree of the control is better on math and
  level on code and chat. Plan trees start at 22 to 26 nodes (K = 4), so their values at 20 nodes are extrapolated.
- Within one decoder, with the K most likely drafts taken at a prior sample (the decoder's training input), sampling
  the plan gives the better tree for the 64-token plan (math 3.34 against 3.15 at 30 nodes, code 2.35 against 2.32)
  and for β = 0.1 (math 3.25 against 3.09). For the 256-token plan at β = 0.3 the two are level (3.29 against 3.31).
  The earlier within-decoder comparison took the K most likely drafts at the prior mean, which the decoder never saw,
  and overstated the gain.
- Sampled-plan trees run the decoder and the LM head once per sample (8 times for K = 8), the per-position tree once.
- The K most likely drafts of a product of per-slot distributions maximise sequence probability, not expected
  accepted length, so they are not the strongest per-position tree.
- With a collapsed plan, z acts as an input noise the decoder learned to follow. Its 8 samples give 3.15 on math at
  11 nodes, against 3.22 for the control's 4 most likely drafts at 18 nodes.

**Stage 1 with more data.** The larger stage-1 model (d1024 × 6, 36k steps) overfit the 512k-row cache (section
above). Both sizes were retrained on the 5.9M windows of `cache_pb`, `cache_pb2` and `cache_pb3` (`ae_gate.py
--train_cache`, z2048, 36k steps, B 96), with the same dev sets and gate (`results/ae_gate_*_6m_z2048.json`).

| Stage 1 (states only, all four layers) | Math | Code | Chat |
|---|---|---|---|
| d512 × 4, 512k rows, 12k steps | 94.6% | 93.0% | 78.3% |
| d512 × 4, 5.9M windows | 96.7% | 95.4% | 85.4% |
| d1024 × 6, 5.9M windows, latent mean | 88.8% | 85.3% | 69.8% |
| d1024 × 6, 5.9M windows, posterior sample | 94.3% | 91.7% | 78.2% |

- Training on 5.9M windows for 36k steps, against 512k rows for 12k steps, lifts the original model by 2.1 to 7.1
  points, chat most. Data scale and 3x longer training are confounded here. All runs remain below the gate.
- The larger model underfits its training data (final token KL 0.345 against 0.211 for the original model, training
  agreement 90.9% against 92.3%), which points at its training recipe (same learning rate and noise settings as the
  small model). Its latent mean also scores lower than a posterior sample, while for the original model the mean is
  better. In a separate check on the first 600 dev windows (not in the JSON), rescaling the mean to the expected
  sample norm gave 94.1% (math) and 78.7% (chat), equal to a sample.
- Agreement is 99.4 / 99.0 / 96.7% where the target's top-1 logit leads by 2 or more, and 60 / 55 / 47% where it
  leads by less than 0.5. By error count, 29 to 39% of errors fall below a margin of 0.5, 47 to 52% between 0.5 and 2,
  and 14 to 20% at 2 or more.
- If per-token errors were independent, 96.7% and 85.4% would allow 1 + p + ... + p⁸ ≈ 7.9 (math) and 5.2 (chat)
  accepted per round from the true latent. For the head-row model the same formula gives 8.8 / 8.7 / 8.1 against a
  measured 8.5 / 8.5 / 7.6, so the real ceiling is likely lower. Both are reconstruction bounds, since stage 1
  encodes the drafted positions' own states. One predicted draft reaches 1.4 to 3.1.
## What the results say

1. **Target-side refinement.** Arm C uses the spare width of the verification pass to reach τ 2.3 without a second
   model. Soft inputs add nothing measurable over hard guesses, in line with the EmbeddingSuperPosition result on
   drafter inputs.
2. **Soft supervision.** Soft feedback between drafter passes adds about 10% τ, and target TV labels add 6 to 21% at
   T = 1.
3. **Block drafter errors.** Teacher-forced accuracy is flat across offsets, while free-running acceptance decays with
   offset. Errors come more from consistency across the block than from per-token knowledge.
4. **Combining C and D** gives no gain at this scale.
