# Results

Target Qwen2.5-1.5B-Instruct, drafter Qwen2.5-0.5B-Instruct, one RTX 4070 Ti, batch size 1, bf16.
All numbers come from `results/*.json` and the tables below are printed by `make_tables.py`.

**Status.** Arms 0, A, C, C-hard, D and C+D are done, greedy and T=1. Two follow-ups are still running and their
sections are marked pending. They are the corrected future-summary probes and the latent planner experiment.

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

**First run invalid, rerun pending.** In the first run (`results/planner.json` as pushed on 23 Nov at 00:55)
every plan variant, including the oracle, scored far below the plain drafter (GSM8K mean accepted length 1.2
to 1.7 vs 4.15). The cause was a bug in the plan input. An unbounded, zero-initialised projection grew under
Adam to 4 to 10 times the norm of a token embedding (1.7 to 4.4 vs 0.45) and drowned the drafter's token inputs.
Those numbers say nothing about the idea. The rerun caps the plan offset at the median token-embedding norm
behind a learned gate that starts at zero, so each variant starts exactly at the plain drafter.

The best Arm D recipe (TV labels, soft feedback) gets a vector added to its block inputs. That vector
comes from target features, from a plan head trained to predict a summary of the next 64 tokens, or from oracle
summaries of the true future (near, and skipping the drafted block). All runs use the same steps and data as d4_soft.

## What the results say

1. **Refinement works, softness doesn't (target side).** Arm C turns the spare width of the verification
   pass into τ 2.3 with no second model, but soft inputs add nothing measurable over hard guesses. This
   agrees with the EmbeddingSuperPosition result on drafter inputs.
2. **Soft supervision helps the drafter, and target TV labels help under sampling.** Soft feedback between
   drafter passes is worth about 10% τ, and TV labels are worth 6 to 21% at T=1.
3. **The block drafter's limit is coherence across the block, not per-token knowledge.** Teacher-forced
   accuracy is flat, but free-running acceptance decays with offset. That is the gap the latent planner targets.
4. **Combining C and D doesn't pay** at this scale.
