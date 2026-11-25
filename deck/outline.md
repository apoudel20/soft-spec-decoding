# Gister: compute ask (outline)

Ten slides, one claim each. Every number traces to a file in `results/`. `{{...}}` = filled after the 11:05 run.
Framing: a working proof of concept for coherent multi-future tree drafting, not "almost beating DFlash".

## 1. The ask
- Cluster time to scale one mechanism that already works on a single consumer GPU.
- Concretely: a ~1B stage-2 predictor with full-context conditioning, on 100M+ on-policy positions.
- What we bring: a working lossless pipeline, measured scaling signal, an honest gap analysis.

## 2. Problem and idea
- Speculative decoding: a cheap drafter proposes tokens, the target verifies them in one pass, output unchanged.
- Gister drafts in a compressed latent of the target's own future states (a "gist"), and samples several futures.
- Distinct from DFlash / DSpark (per-slot block drafts), EAGLE (autoregressive features) and ChainFlow
  (per-token deterministic latents): one sampled latent per 8-token chunk, from 4 target layers, decoded jointly.
- Figure: pipeline diagram (website, "One Gister round with a sampled tree").

## 3. Stage 1 is near-lossless on top-1 (greedy scope)
- Exact top-1 tokens: 99.4 / 99.0 / 97.3% (math / code / chat) from an 8-token x 4-layer gist.
- Ceiling: decoding the true next gist gives 8.51 / 8.47 / 7.63 accepted per round out of 9.
- Caveat (on the slide): the gate's distribution check fails (TV 0.06 / 0.06 / 0.20 vs 0.005 to 0.01, KL up to
  0.84). Every claim is for greedy decoding. Sampling at T > 0 needs a tighter stage 1.

## 4. The bottleneck is stage 2, and why
- Deterministic predictor: 2.17 -> 2.80 (raw target states) -> 3.08 (154M, 2M positions) on math. Code and chat flat.
- Theory: a deterministic predictor regresses to the mean of possible futures. Data stopped the overfitting, then
  hit that floor (loss ~4.0 nats on unseen text).
- Structural limit (on the slide): stage 2 reads only the last 8 positions' states. DFlash reads the whole context.

## 5. Sampling works as a mechanism
- Flow-matching sampler, 2 Euler steps. 8 samples give 7.6 to 8 distinct drafts per round.
- Best of K over deterministic: +7 to 9% (K=4), +17 to 19% (K=8), +27 to 30% (K=16), about +0.3 tokens per
  doubling of K, in every domain.
- Figure: best-of-K curve per domain, deterministic line for reference.

## 6. Lossless tree verification, end to end
- K sampled drafts + the deterministic chain merged into one token tree, verified in one target pass
  (tree attention mask). A one-branch tree reproduces the chain path exactly. Output identical across drafters.
- Real tau rises 18 to 31% over the chain. Verifying ~100 nodes costs the same as one decode step.
- Speed on an idle GPU: {{e2e_final: chain vs tree speedups, MT-Bench / GSM8K / HumanEval}}.
- Earlier +12% GSM8K tree win came from a throttled run; {{confirmed / not confirmed}} on the idle GPU.

## 7. Where we stand vs baselines (honest table)
- Same machine, same loop: {{e2e_final table: DFlash, Gister chain, tree (158M), tree (285M)}}.
- At equal tree budget offline: Gister ~half of DFlash on math and chat, under 40% on code.
- Published (other hardware): DFlash 4.86x avg / 2.75x MT-Bench (H200); EAGLE-3 1.90x MT-Bench; DSpark tau 3.72
  MT-Bench (T=1). EAGLE-4 / DFlash 2 have no Qwen3-8B author checkpoints.

## 8. One scaling point (tonight)
- 158M sampler on 2M positions vs 285M on 6M positions: {{best-of-8 / best-of-16, both samplers}},
  {{tree tau and speedup, both samplers}}.
- Caveat (on the slide): the 285M sampler trained from scratch, the 158M one from a deterministic warm start, so
  the comparison understates or confounds scale.
- Context ablation (64 positions of gist history + raw states): {{stage2_raw_h8 result}}.

## 9. Examples
- 2 to 3 real rounds: the context, 3 to 4 sampled drafts, which branch the target accepted.
  {{from results/examples.json, check args.s2 == stage2_gister_flow_big}}.

## 10. Plan and compute
- Experiments, in order: (1) full-context conditioning (cross-attend to target states at all positions),
  (2) data scale with rolling caches (100M+ positions), (3) warm-started 0.5B to 1B sampler,
  (4) drafter latency (one-step sampler, CUDA graphs, bf16 target), (5) joint stage-1/2 training.
- Why a cluster: the target's forward pass here runs ~3,000 tokens/s, so 800M positions is ~3 days of GPU time
  before training. A 1B predictor barely fits 12 GB and trains at ~1.5 to 2 s/step.
- Risks stated: drafter latency on fast engines (27 to 67 ms eager vs 10 to 15 ms bf16 target steps), greedy-only
  scope until stage 1 is tightened.
- Deliverables: per-experiment results in `results/`, W&B runs, checkpoints on HF, weekly write-up.
