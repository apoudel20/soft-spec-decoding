# Soft Speculative Decoding: Minimal Experiment Plan

## Question

Speculative decoding treats the drafter as a detached "parasite": it proposes hard
tokens, the frozen target accepts or rejects them, and everything after a rejection
is thrown away.

We want to test whether a **symbiotic** setup does better. In that setup:

1. The **target** is lightly fine-tuned to accept *distributions* (soft tokens) at
   speculative positions and to refine them. It spends the spare compute of its
   memory-bound verification pass drafting its own future.
2. The **drafter** is trained on soft inputs, with the target's per-token
   distributions as soft labels.

The setup must stay **lossless** with respect to the original target model.

## Hypotheses

- **H1 (refinement):** A target allowed to refine guesses for future positions during
  verification raises accepted tokens per target forward pass over a frozen-target
  Jacobi baseline.
- **H2 (softness):** Soft inputs (Eᵀq) beat hard argmax inputs under the same
  fine-tuning. This is the key claim: the uncertainty in a distribution carries
  information that a single hard guess throws away.
- **H3 (drafter-side):** A drafter trained with soft inputs and the target's
  distributions as labels beats one trained on hard labels. This part does not need
  a VAE or latent space.

If H1 holds but H2 fails, the gain comes from the target refining its own future,
not from passing distributions. That is still a useful result.

## Setup

| Item | Choice (swap freely) |
|---|---|
| Target | A small open instruct model, roughly 1–2B parameters, from a family with a smaller sibling |
| Drafter | A DFlash-style block drafter if code is available; otherwise an EAGLE-style head or the smaller sibling model |
| Train data | Target-generated responses to ShareGPT/UltraChat prompts (self-distilled, so the labels match what the target actually says) |
| Eval | MT-Bench (chat), GSM8K (math), HumanEval (code). Code and math are where any blur hurts. |
| Decoding | Greedy first; then temperature 1 with standard rejection sampling |
| Hardware | One GPU is enough at this scale |

## Metrics

- **τ:** accepted tokens per target forward pass. This is the primary metric.
- **Per-position acceptance:** acceptance at block offset 1..B. It shows where
  blocks break down.
- **Wall-clock speedup** against plain autoregressive decoding, with the same
  kernels and batch size 1.
- **Exactness check:** greedy outputs must be token-identical to the base target.
  With sampling, the output distribution must match the base target. Test this on
  held-out prompts by comparing next-token distributions at committed positions.

## Arms

### Arm 0: Diagnostic (run first, about a day)

For the existing drafter, measure:

- (a) per-position draft accuracy with the true in-block prefix teacher-forced;
- (b) actual accepted length when the whole block is drafted at once.

Reading the result:

- **High (a), collapsing (b):** the per-position marginals are fine, but they are
  incoherent jointly. Trees and joint (latent) modeling are worth it.
- **Low (a):** the drafter's conditioning or capacity is the bottleneck. Arm D
  matters most.

### Arm A: Jacobi / lookahead with a frozen target (baseline)

Use plain Jacobi iteration on hard guesses of a window of n future tokens. Commit the
longest prefix where the target's output matches its input. No training.

### Arm B: CLLM-style fine-tune with hard guesses

Fine-tune the target so that Jacobi iteration on hard guesses converges in fewer
steps (the consistency-LLM objective). This is the strongest existing
"change the target" baseline.

### Arm C: Soft-input target (the symbiote)

**Input layout per round, in one forward pass:**

```
[ committed prefix (hard) | H drafted tokens to verify (hard) | S soft tail positions ]
```

- Each soft position's input is `Eᵀq + e_soft`, where `q` is the current
  distribution guess for that position and `e_soft` is a learned flag embedding.
- The H hard draft tokens are verified exactly as in normal speculative decoding.
- The target's outputs at the soft tail become refined distributions. They supply
  the next round's hard draft (argmax or sample) and its new soft tail. Fresh
  positions start from uniform, or from the drafter's distributions.

**Exactness by construction (gated LoRA):**

- LoRA adapters are applied only at positions flagged as soft.
- Attention is causal, so a position whose whole prefix is hard never sees soft
  keys or values. Its computation is exactly the base model's.
- Every committed token goes through standard verification at a hard position, so
  the output matches the base target. Numerics only match if the batch and sequence
  layout is identical, so the exactness check compares against the base model run
  with the same layout.

**Training:**

- Data: target-generated sequences. Pick a split point, mark the next H tokens as
  hard (ground truth, optionally with some corruption) and the following S as soft.
- Build soft inputs from a mix of:
  - the drafter's actual distributions (matches deployment);
  - the ground-truth one-hot corrupted with noise or label smoothing, at random
    strength (teaches refinement from any confidence level);
  - uniform (a cold start).
- Loss at soft positions:
  `KL( p_base(· | true prefix) ‖ p_model(· | soft inputs) )`.
  The labels come from the frozen base model run on the true prefix.
- The loss only touches the LoRA, and the base behavior on hard prefixes is
  unchanged by construction.

**Ablations, which carry the experiment:**

- **C-hard:** same adapter, same layout, but the soft tail gets `E[argmax q]` instead
  of `Eᵀq`. This isolates H2.
- **Temperature-sharpened q:** probes how much of the distribution actually matters.
- **S ∈ {4, 8, 16}, H ∈ {2, 4, 8}:** the trade-off between verification and
  refinement width.
- **Seeding:** with or without a small drafter seeding the soft tail.

### Arm D: Drafter-side soft training (cheapest; run in parallel with C)

Keep the target frozen. Only the drafter changes.

- **Soft inputs (self-conditioning):** at each denoising step, or for each later
  draft position, feed the drafter its own previous distributions as `Eᵀq` instead
  of a mask or hard token. This is the "average embeddings" idea.
- **Label variants:**
  - D1: hard ground-truth cross-entropy (baseline).
  - D2: the drafter's own earlier outputs as soft labels (self-distillation; a
    control, expected to give little gain).
  - D3: the target's per-token distribution, forward KL.
  - D4: the target's per-token distribution, TV loss. Expected acceptance is
    `1 − TV(p, q)`, so TV is the quantity actually being optimized.
- **Cross:** soft vs hard inputs × D1/D3/D4.

### Arm C+D: Both together

Use the best Arm D drafter to seed Arm C's soft tail.

## Decision criteria

| Outcome | Conclusion |
|---|---|
| C > B and C > C-hard (τ, matched wall-clock) | Distributions as target inputs carry real signal. Scale it up. |
| C ≈ C-hard > B | Refinement in the verification pass is the win; softness isn't. Simplify to hard inputs. |
| C ≈ B | The symbiote framing doesn't beat CLLM at this scale. Write up the negative result. |
| D3/D4 > D1 with soft inputs | Drafter-side soft training is enough. A VAE or latent space is unnecessary for this goal. |
| Arm 0 shows coherence collapse that D can't fix | Revisit latent block modeling, or tree verification over the marginals. |

## Order and rough timeline

1. Arm 0 diagnostic and the exactness test harness (about 2–3 days).
2. Arms A and D in parallel; D is mostly drafter fine-tuning (about 1 week).
3. Arm B (about 3–4 days, if CLLM code can be reused).
4. Arm C and its ablations (about 1–2 weeks; the decode loop is the fiddly part).
5. C+D, wall-clock benchmarking, write-up.

## Risks

- **Several target passes per round:** if C needs multiple passes to beat a
  one-pass EAGLE/DFlash setup, the τ gains won't show up as wall-clock gains.
  Always report both.
- **Kernel support:** mixed hard and soft inputs plus position-gated LoRA may break
  fused attention kernels. Prototype with eager attention and measure overhead
  separately.
- **Stale soft tail:** after a rejection at offset k < H, the soft tail was refined
  on a wrong prefix. Measure whether reusing it as a prior helps or hurts, compared
  with resetting it.

## Related work to read first

Speculative sampling (Leviathan et al.; Chen et al.), DistillSpec, Medusa / Medusa-2,
EAGLE / EAGLE-2, SpecInfer (tree verification), lookahead decoding, CLLMs, multi-token
prediction (Gloeckle et al.; DeepSeek-V3 MTP), Coconut, Soft Thinking, self-conditioning
in diffusion (Analog Bits), DFlash.
