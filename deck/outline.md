# Gister: compute ask (outline)

Ten slides, one claim each. Every number traces to `results/` (see RESULTS.md, "Gister, audited state").
Framing: a measured proof of concept for drafting by sampling latent futures. It works as a mechanism and improves
with scale, it is not yet faster than the chain on this hardware, and it is well behind DFlash. The ask targets the
measured bottlenecks. Scope: greedy decoding only.
Pending at writing (fill in at 4:15 if done): full-context test, direct-token control, examples.

## 1. The ask
- Cluster time to test one idea properly: draft by sampling the target's own future states in a compressed latent.
- First experiments: full-context conditioning, a matched-budget comparison against per-slot trees, data scale
  (100M+ on-policy positions), a warm-started 0.5B to 1B sampler, drafter latency on a bf16 engine.
- What exists already: a lossless pipeline, audited results, three independent reviews and their fixes.

## 2. Idea and closest prior work
- Speculative decoding: a cheap drafter proposes tokens, the target verifies them in one pass, output unchanged.
- Gister: compress 8 tokens of the target's own hidden states into one latent, sample several future latents,
  decode each into a whole 8-token continuation, verify them together as one tree.
- Closest prior work: ChainFlow (VAE over target states, flow to 8 positions, deterministic), FlashMap (flow from
  multi-layer states, several noise draws as a tree), MFD / DBLast (coherent multi-future trees), CALM (one latent
  per token chunk), DFlash + DDTree (per-slot trees). Gister's specific combination: several sampled continuous latents
  of a chunk of the target's own future states, each decoded jointly.
- Figure: one Gister round with a sampled tree (website).

## 3. Stage 1: a strong ceiling, gate not passed
- Exact top-1 tokens 99.4 / 99.0 / 97.3% (math / code / chat) from one 2,048-dim latent per 8 tokens.
- Decoding the true next latent would give 8.5 / 8.5 / 7.6 accepted per round out of 9: the space is good to draft in.
- Gate failed on both checks: top-1 below 99.5 / 99.5 / 99.0, total variation 0.06 / 0.06 / 0.20 (bar 0.005 to
  0.01). Greedy decoding only. Works only with the top-1 token's head row as encoder input; middle layers add little,
  so the latent is closer to a token chunk plus the final state than a multi-layer plan.

## 4. Stage 2 is the bottleneck
- Deterministic predictor: 2.17 -> 2.80 (raw target states) -> 3.08 (154M, 2M positions) on math; code and chat flat.
- A deterministic predictor must bet on one future; with branching futures it averages them.
- It reads only the last 8 positions' states (DFlash reads every earlier position). 64 positions of gist history did
  not help (2.81 / 2.15 / 1.39). Full-context test with raw states: {{ctx56 result or "running"}}.

## 5. Sampling works as a mechanism
- Flow-matching sampler, 2 steps: 7.3 to 8.0 distinct drafts out of 8.
- Best of 8 over deterministic: +21 / +19 / +18% (158M), +28 / +23 / +22% (285M). Best of 16: up to +38%.
- Per doubling of K: +0.31 / +0.25 / +0.13 tokens (158M).
- Figure: best-of-K curves per domain, both samplers, deterministic line.

## 6. One scaling point
- 158M on 2M positions -> 285M on 6M positions: every metric up 3 to 7% (best of 16: 4.04 -> 4.18 math,
  2.89 -> 3.06 code, 1.85 -> 1.92 chat).
- Confounded: parameters, data, shape and warm start all changed. Gains are modest; with the 8-position context and
  the history null, context is the likelier limit than size.

## 7. End to end, idle GPU, same loop for everything
- Lossless greedy loop, verified to match DFlash's own decoder round for round.
- Speedup over plain decoding (MT-Bench / GSM8K / HumanEval): DFlash 2.69x / 4.73x / 7.89x; Gister chain 1.34x /
  2.05x / 1.40x; Gister tree (8 samples) 1.07x / 1.77x / 1.14x.
- Gister's chain and DFlash cost the same per round (about 53 ms). DFlash wins on accepted tokens.
- Trees raise tau 18 to 32% but cost 45 to 55% more per round here: sampling 8 to 16 latents and verifying 60 to 120
  nodes is not free on an idle GPU with 4-bit kernels.

## 8. Where Gister stands, honestly
- Drafted tokens accepted (excluding the free bonus token), best of 8 vs DFlash's 64-node tree: 49% / 29% / 29%.
- Published (other hardware): DFlash 4.86x average on Qwen3-8B (H200); EAGLE-3 1.90x MT-Bench; DSpark tau 3.72
  MT-Bench (T=1); DFlash 2 (RedHatAI, Qwen3-8B) block 8, vendor numbers only.
- No claim of beating DFlash or per-slot trees yet.

## 9. Examples
- 2 or 3 real rounds: context, several sampled drafts, the branch the target kept. {{from results/examples.json}}

## 10. Plan, compute, risks
1. Full-context conditioning: cross-attention to the target's states at every earlier position.
2. Matched-budget comparison: sampled latent tree vs DFlash/DDTree per-slot tree at the same node count.
3. Direct-token control at scale: does the latent beat predicting tokens directly? {{direct result or "running"}}
4. Data: 100M+ on-policy positions with rolling caches (here: ~3 days of target forward passes per 800M positions).
5. Model: warm-started 0.5B to 1B sampler (here: barely fits 12 GB).
6. Latency: one-step sampler, CUDA graphs, bf16 target, so tree drafting costs close to the chain's 11 ms.
- Risks: trees may not pay off even with better samples; stage 1 must be tightened before sampling at T > 0.
- Deliverables: results JSONs, W&B runs, checkpoints on HF, weekly write-up.
