# Feature-level drafters and latent-sampled coherent branches (prior work for a latent-autoencoder drafter on frozen Qwen3-8B)

Proposed design for reference: (A) autoencoder over the target's next 1..k hidden states, decoded through the frozen LM head with shared embeddings, (P) a predictor of that latent from context, (S) several latent samples giving coherent multi-token branches merged into a draft tree. Our own measurement: DFlash per-slot top-4 oracle 4.43 accepted vs chain 2.96 on open chat.

Verification status: items marked [abstract] were checked only against the arXiv abstract page. Items marked [html] were checked against the arXiv HTML full text via a summarizing fetch, so exact table numbers should be spot-checked before quoting in a paper. Items marked [unverified] come from prior knowledge and were not re-checked this session.

## 1. Feature-level drafters (EAGLE family, HASS, GRIFFIN, Medusa, Hydra, newer): what they predict, LM-head reuse, feature vs token loss, layer fusion, multi-step error

### Takeaway
The EAGLE line started with regression onto the target's top hidden state plus a token loss through the frozen LM head (EAGLE-1), then EAGLE-3 dropped the feature loss because it capped how much the drafter gained from more data. EAGLE-3 fused low, mid and high layers and used training-time test to fix the step-2+ distribution shift that appears once the feature anchor is removed. Every strong 2025-2026 drafter either autoregresses at token level with train-time rollout or drafts a block in parallel. A drafter that keeps a hidden-state target (Lyanna, ChainFlow) has to add token feedback to recover dependencies.

### Cited Findings
**EAGLE-1 (arXiv 2401.15077)** [abstract]
- Predicts "second-to-top-layer" features, meaning the hidden state fed to the LM head, and decodes them through the target's LM head. The feature-level autoregression is "more straightforward" than token-level, but sampling uncertainty limits it. The fix is to feed the token sequence advanced by one step together with the features. — [arXiv 2401.15077](https://arxiv.org/abs/2401.15077)
- Lossless ("preserving the distribution"). LLaMA2-Chat 70B: 2.7x-3.5x latency speedup, doubled throughput. Also Vicuna, Mixtral 8x7B. — [arXiv 2401.15077](https://arxiv.org/abs/2401.15077)
- EAGLE-1 on Vicuna-13B MT-bench (T=0): 3.07x, tau 3.98 (EAGLE-3 paper's table). — [EAGLE-3 html](https://arxiv.org/html/2503.01840)

**EAGLE-2 (arXiv 2406.16858)** [via EAGLE-3 paper html, own abstract not fetched]
- Replaces EAGLE's static tree with a dynamic, context-aware tree. It estimates acceptance from draft confidence and reranks or prunes at the end of drafting. — [EAGLE-3 html](https://arxiv.org/html/2503.01840)
- MT-bench T=0: Vicuna-13B 4.26x (tau 4.83), LLaMA-3.1-8B-Instruct 3.16x (tau 4.05), LLaMA-3.3-70B 2.83x (tau 3.67), DS-R1-Distill-LLaMA-8B 2.92x (tau 3.80). — [EAGLE-3 html](https://arxiv.org/html/2503.01840)

**EAGLE-3 (arXiv 2503.01840)** [html]
- Removes the feature-prediction loss and predicts tokens directly. The authors' reason is that the feature loss is an extra constraint that limits expressiveness, so EAGLE gained little from more training data. — [arXiv 2503.01840](https://arxiv.org/abs/2503.01840)
- Once the feature constraint is removed, step-1 acceptance rises, but the step-1 output no longer approximates the true next feature. The step-2 input is then out of distribution and step-2 acceptance becomes "very low". Training-time test (TTT) fixes this by feeding the drafter's own outputs back in during training, with a modified attention mask (lower-triangular for the native step, diagonal for simulated steps). — [EAGLE-3 html](https://arxiv.org/html/2503.01840)
- Multi-layer fusion: the low, mid and high-level target features (three k-dim vectors) are concatenated to 3k and projected to k by an FC layer. The paper text does not name the exact layer indices. [unverified: the public EAGLE-3 code is commonly reported to use layers 2, L/2 and L-3, not checked this session] — [EAGLE-3 html](https://arxiv.org/html/2503.01840)
- Drafter is one decoder layer. Its output goes through the LM head to sample tokens, and the sampled token's embedding is the next input. The paper does not say whether the head is the frozen target head. [unverified: the public EAGLE-3 checkpoints train their own reduced-vocab head (draft vocab ~32k with a d2t map), so EAGLE-3 does not strictly reuse the frozen head] — [EAGLE-3 html](https://arxiv.org/html/2503.01840)
- MT-bench T=0: Vicuna-13B 5.58x (tau 6.65), LLaMA-3.1-8B 4.40x (tau 6.13), LLaMA-3.3-70B 4.11x (tau 5.63), DS-R1-Distill-LLaMA-8B 4.05x (tau 5.58). Up to 6.5x (HumanEval), about 1.4x over EAGLE-2, 1.38x throughput at batch 64 in SGLang. Lossless. — [EAGLE-3 html](https://arxiv.org/html/2503.01840), [abstract](https://arxiv.org/abs/2503.01840)
- On Qwen3 targets in 2026 block-drafter papers, EAGLE-3 reaches only 3.4-4.2x (HF, batch 1) and 2.75x in SGLang on Qwen3-4B, against 4.06-4.38x for the DFlash chain. — [MFD html](https://arxiv.org/html/2611.13017)

**HASS (arXiv 2408.15766, ICLR 2025)** [abstract]
- Targets two mismatches in hidden-state or KV-reusing drafters: inconsistent context between training and decoding, and mismatched training vs decoding objectives. The fixes are "harmonized objective distillation" and "harmonized context alignment". No extra inference overhead. Four LLaMA models, 2.81x-4.05x averaged over three datasets, 8%-20% over EAGLE-2. — [arXiv 2408.15766](https://arxiv.org/abs/2408.15766)
- [unverified] Objective distillation is a top-K ranking-aware distillation loss. Context alignment trains on drafter-generated features over several steps, a precursor of TTT. EAGLE-3's html says HASS keeps feature prediction and top-layer inputs. — [EAGLE-3 html](https://arxiv.org/html/2503.01840)

**GRIFFIN (arXiv 2502.11018)** [abstract]
- Targets token misalignment between training and decoding. It uses token-alignable training with loss masking that excludes highly misaligned tokens, and a draft model that adds input tokens to correct inconsistencies in generated features. LLaMA, Vicuna, Qwen, Mixtral. Over 8% acceptance-length gain and over 7% speedup vs SOTA (EAGLE-3 / HASS family). Code released. — [arXiv 2502.11018](https://arxiv.org/abs/2502.11018)

**Medusa (arXiv 2401.10774)** [abstract]
- Extra heads on the last hidden state each predict a future position independently. Candidates come from a tree via tree attention. Medusa-1 has a frozen backbone, is lossless and gives over 2.2x. Medusa-2 trains jointly, gives 2.3-3.6x and is not described as lossless. "Typical acceptance" is a lossy acceptance scheme. — [arXiv 2401.10774](https://arxiv.org/abs/2401.10774)

**Hydra (arXiv 2402.05109)** [abstract]
- Medusa heads predict each candidate token "without regard" to the other candidates. Hydra heads condition each speculated token on the earlier candidate tokens (sequential dependence). Hydra++ reaches up to 1.31x Medusa throughput and 2.70x autoregressive. — [arXiv 2402.05109](https://arxiv.org/abs/2402.05109)

**Newer hidden-state drafters (2026)**
- **Lyanna (arXiv 2602.21224)** [html]: a single trainable layer autoregresses only over hidden states. Tokens enter the chain only at step 1, and token information is added later as an additive logit bias (RMSNorm of a token-info embedding). LM head calls are batched into one GEMM, cutting draft forward latency by 60.9%. Because hidden states do not depend on sampled tokens, rejected drafts can be re-sampled from the same states. Loss is hidden-state MSE plus CE. Conditional acceptance at steps 1-3 is 91/80/70%, and 90/73/64% without token info. Up to 3.3x over standard SPS, about 1.4x over EAGLE at batch 2 (1.2x at batch 16). Tested on LLaMA-2-7B and Vicuna-7B only. Losslessness is not explicitly claimed (greedy eval). — [arXiv 2602.21224](https://arxiv.org/html/2602.21224v1)
- **DART (arXiv 2601.19278)** [abstract]: from target hidden states, a single custom layer predicts logits for several future masked positions in parallel. The tree is pruned with N-gram semantic-continuity constraints. 2.03x-3.44x, about 30% over EAGLE-3 on average. Target models are not named in the abstract. — [arXiv 2601.19278](https://arxiv.org/abs/2601.19278)
- **KVShot / "When Hidden States Drift" (arXiv 2604.26412)** [abstract]: on Qwen3-8B, hidden-state drafters show "long-range decay" in acceptance that persists even with TTT. The explanation is that the target's hidden state is a query-shaped compressed summary that drops information later steps need. Reusing the target KV cache improves long-range acceptance, but end-to-end speedup stays marginal. Bottlenecks are shallow drafters estimating target queries badly and sparse gradients to the draft-side KV projections. Recommends block-wise training over TTT. — [arXiv 2604.26412](https://arxiv.org/abs/2604.26412)

### Inferences
- The proposed design reintroduces a feature target (the latent of future hidden states), which is the constraint EAGLE-3 removed. The EAGLE-3 argument is about a per-step regression loss used as an autoregressive input anchor, not about a latent that is decoded through the frozen head. It still warns that a pure reconstruction or regression objective may cap scaling. Train the predictor with a token-level (CE or acceptance) loss through the frozen head, and keep latent regression auxiliary at most.
- Frozen-head reuse is less common than it looks. EAGLE-1 and Medusa-1 reuse the frozen target head, while EAGLE-3 in practice uses its own trimmed head [unverified]. A frozen full 151k-vocab Qwen3 head costs one large GEMM per drafted position. Lyanna's batching of all positions into one GEMM is the relevant trick.
- KVShot is direct evidence, on the same Qwen3-8B target, that the top hidden state alone is lossy for long horizons. A latent of future hidden states inherits this limit unless the predictor also attends to the target KV or multi-layer features (EAGLE-3-style low/mid/high fusion).
- Closeness to the design: EAGLE-1 covers P (single step, frozen head). Lyanna covers P (multi-step hidden-state chain, head decode, re-sampling from the same states, a weak form of S). None of these has an autoencoder stage A.

### Gaps
- Exact EAGLE-3 fused layer indices are not in the paper text. Code-level indices were not verified.
- HASS acceptance lengths and exact loss definitions were not fetched (abstract only).
- No source found that ablates "keep the feature loss plus TTT" versus EAGLE-3's token-only loss directly. EAGLE-3 does not report it.
- EAGLE-2's own abstract (2406.16858) was not fetched. Its numbers are taken from the EAGLE-3 table.
- NextLat is covered elsewhere and was not re-researched.

## 2. Latent-variable or stochastic block drafters giving jointly coherent multi-token candidates, and evidence that per-position independence limits trees

### Takeaway
Two late-2026 preprints are close to our stage S on the same target family. Multi-Future Drafting (MFD, 2611.13017) adds K learned mode embeddings to a DFlash-style block drafter and builds a lossless mixture-DDTree. It proves that product (per-slot) trees must contain "chimera" prefixes and measures 27-33% chimera nodes on Qwen3-4B/8B. DBLast (2608.05448) adds a categorical latent with a shared LM head on Qwen3-4B/8B and gains up to +12.1% acceptance length, with the largest gains on the least deterministic targets. Both independently confirm our DFlash top-4 finding.

### Cited Findings
**Multi-Future Drafting: Mixture Block Drafters (arXiv 2611.13017, Nov 2026)** [html]
- Problem statement: DDTree ranks prefixes by the product of per-position marginals. When the continuation has several futures, the product "splices" them into chimeras, prefixes the target never produces. On 628 prompts with Qwen3-4B and Qwen3-8B, 27-33% of nodes in a 64-node product tree are chimeras, and the best path passes through one in 61-65% of rounds. — [MFD html](https://arxiv.org/html/2611.13017)
- Lemma 1: a product tree holding two futures that differ in m positions must also hold 2^(m-1)-1 chimeras, independent of training. Prop. 2: with two futures, D=15 and B=256, the optimal tree accepts all 15 tokens, while any product tree accepts at most 12. — [MFD html](https://arxiv.org/html/2611.13017)
- Method: K learned mode embeddings (K=8 main) injected at the drafter input. All modes are drafted in one forward pass, and each mode gives a product distribution. A mixture DDTree keeps the B best prefixes over all modes, scoring each by its best mode. Lossless, with verification unchanged. Overhead is K+1 vectors (about 23K parameters at K=8, h=2560). Training is winner-take-all: the mode with the highest coverage of the target block gets 1-eps (eps=0.1). A gate predicts the best mode. — [MFD html](https://arxiv.org/html/2611.13017)
- Results (HF, batch 1, 5-task mean): Qwen3-8B T=0, MFD 6.51x (tau 9.47) vs D-PACE+DDTree 6.04x (8.60), released DDTree 5.73x (8.18), DARTree 6.42x (10.09), DFlash chain 4.26x. Qwen3-4B T=1, MFD 5.96x (8.63) vs 5.39x. 4.8-7.4% faster than the identically trained single-mode tree at T=0 and 5.7-7.6% at T=1. Up to 7.78x with 1.42M prompts. — [MFD html](https://arxiv.org/html/2611.13017)
- SGLang (Qwen3-4B, T=0, batch 1): MFD 5.93x (tau 9.62), DDTree 4.97x, DFlash chain 4.06x, EAGLE-3 2.75x. MFD falls behind DDTree from batch 8. — [MFD html](https://arxiv.org/html/2611.13017)
- Key ablations for our design:
  - Hedging needs a branching verifier. A chain following the gate's mode is worse than the single-mode drafter, because the gate picks the best mode only 38% of the time at K=4 (chance 25%). Oracle chains would be 20-40% longer.
  - Where the latent enters matters. Input injection gives +5.0 (B=64, K=4), last-layer injection +1.5, and a low-rank output map -0.6. Input embeddings are about 3x as effective as a 26M-parameter full-rank output expansion.
  - Averaging the loss over modes collapses to a single mode, so winner-take-all is needed.
  - A mixture tree beats both a tree per mode and a committed-mode tree.
  - Mode argmax chains disagree in 88% of blocks at T=0 (91% at T=1).
  - MFD trees cover 8.3% more mass at B=64, but MFD trees are still 22-24% chimeras.
  — [MFD html](https://arxiv.org/html/2611.13017)
- Related work it cites (not individually verified): D-PACE, DFlash 2, JetSpec, Domino, DARTree, UBTree, TreeSpark, TreeFlash, DSpark, Draxler et al., FlashMap (Shen et al.), multiple-choice learning. — [MFD html](https://arxiv.org/html/2611.13017)

**DBLast: Dependent Block Drafting for Stochastic Speculative Decoding (arXiv 2608.05448)** [html]
- Proposal: q(x_1:b|y) = sum_z q(z|y) prod_i q_i(x_i|y,z). K=1 recovers the independent DFlash baseline. A linear "category hidden expander" gives branch states h_i,z = h_i + g_z(h_i), a shared LM head produces per-branch logits, and a category-prior head sits on the block anchor. Everything runs in one forward pass. Overhead at K=4 is +5.2% parameters (Qwen3-4B) and +6.7% (Qwen3-8B), with +1.4-1.6% latency per iteration. — [DBLast html](https://arxiv.org/html/2608.05448)
- Objective: marginal block NLL, plus an acceptance-oriented surrogate (AL) summing log importance ratio plus log expected accepted length over retained prefixes. The authors say AL is neither unbiased nor a bound. — [DBLast html](https://arxiv.org/html/2608.05448)
- Qwen3-8B average accepted length (bonus excluded; GSM8K / MT-Bench / HumanEval; up to 256 tokens), DFlash vs DBLast: high determinism 4.55 vs 4.81 (+5.6%), medium 4.19 vs 4.57 (+9.2%), low (T=1.5) 3.70 vs 4.15 (+12.1%). AL alone gives +5.2%, so dependency and AL are complementary. GSM8K 8B low-det: 4.71 vs 5.35. — [DBLast html](https://arxiv.org/html/2608.05448)
- K ablation (Qwen3-4B, T=1.5) with AL: 4.17 / 4.31 / 4.39 / 4.41 for K=1/2/4/8. Most of the gain arrives by K=4. — [DBLast html](https://arxiv.org/html/2608.05448)
- Inference samples z and decodes that branch greedily (a chain, not a tree). Lossless when verification uses the same conditional that produced the draft (prefix-marginalized over consistent branches). Example: on creative writing, independent DFlash proposes one degenerate block ("the the") at q=1.00, while DBLast proposes four coherent alternatives with prior 0.20-0.31. — [DBLast html](https://arxiv.org/html/2608.05448)
- Training uses only default-temperature target data. Soft-mixture training vs greedy-branch evaluation is an acknowledged gap. — [DBLast html](https://arxiv.org/html/2608.05448)

**Other**
- DARTree (alphaXiv 2608.13524, per search snippet only) is training-free. It extends a pretrained AR correction head from chains to trees for "more coherent paths". MFD reports DARTree at 6.42x / tau 10.09 on Qwen3-8B T=0, with tree construction at 2.6-2.7 ms per round vs about 0.35 ms for MFD. — [DARTree](https://www.alphaxiv.org/abs/2608.13524), [MFD html](https://arxiv.org/html/2611.13017)
- "Accelerating Speculative Decoding with Block Diffusion Draft Trees" (arXiv 2604.12989) appeared in search. It is likely the DDTree paper (covered elsewhere) and was not fetched. — [arXiv 2604.12989](https://arxiv.org/html/2604.12989)
- Hydra (above) is the earliest explicit statement that independent per-position heads (Medusa) lose accuracy versus sequentially dependent heads. — [arXiv 2402.05109](https://arxiv.org/abs/2402.05109)

### Inferences
- Our 4.43 vs 2.96 oracle-top-4 gap on open chat fits MFD's chimera result and DBLast's low-determinism gains. Open chat is the low-determinism regime where both report the biggest gains.
- MFD already implements stage S with discrete latents on Qwen3-8B, losslessly, at 6.51x. A continuous VAE latent (our design) must beat a K=8 discrete-mode mixture, or show it composes with one. The novelty left is continuous sampled latents over target hidden states, decoded through the frozen head, plus the autoencoder stage A. Neither MFD nor DBLast reconstructs target hidden states.
- MFD's ablation that the latent must enter at the drafter input, not at the output side, is a direct design constraint. Decoding a sampled latent only at the head (output-side) underperformed. The latent should condition the whole predictor stack.
- Winner-take-all (MFD) and AL surrogates (DBLast) both beat plain likelihood for multi-modal drafting. A VAE trained by ELBO alone may collapse to the mean mode, which matches MFD's "averaging collapses" finding.
- Gate or prior accuracy is low (38% at K=4), so value comes from putting several samples into one tree, not from picking one sample.
- MFD loses to plain DDTree at batch 8 and up. Latent branching is a low-batch technique.

### Gaps
- No VAE-style continuous-latent block drafter with a tree was found apart from ChainFlow (section 3). That drafter uses a VAE for compression, not for sampled diversity.
- UBTree, TreeSpark, TreeFlash, DSpark, FlashMap, Draxler et al., JetSpec, Domino and D-PACE are named in MFD but were not fetched. No arXiv IDs were obtained.
- Sequoia and dynamic-tree comparisons for latent-sampled trees were not searched separately this session.
- MFD and DBLast are 2026 preprints that have not been peer-reviewed. Their numbers come from summarizing fetches of the HTML and should be re-checked against the tables.

## 3. Drafters that condition on or predict a compressed future plan of the target

### Takeaway
ChainFlow (arXiv 2611.08669) is the closest prior work to the full design. A VAE compresses the target's hidden states to a latent, a few-step flow-matching predictor transports an extrapolated latent to the next 8 future latents, decoding goes through the target's frozen LM head, and token-feedback heads plus beam search build a chain or tree. It reports modest speedups (1.87-2.12x) and is not shown to be bitwise lossless. MFD's discrete "mode" is the other plan-like latent ("names the future").

### Cited Findings
- **ChainFlow: Few-Step Latent Flow Matching for Speculative Decoding (arXiv 2611.08669)** [html summary plus search snippet]
  - A VAE compresses the target's hidden states to a latent, so drafting cost scales with latent width, not target size. — [ChainFlow html](https://arxiv.org/html/2611.08669)
  - A conditional flow network transports an estimate of the next 8 latents (extrapolated from recent context) onto the target's future latents in one Euler step. Decoded latents go through the target's frozen LM head, so per-position categoricals feed standard verification. — [search snippet for ChainFlow](https://arxiv.org/pdf/2611.08669)
  - Lightweight token-feedback heads (a "path head" and a "Markov head") restore dependencies between drafted tokens. Beam search emits chain or tree drafts (an 8x4 tree is reported). — [ChainFlow html](https://arxiv.org/html/2611.08669)
  - Qwen3.5 targets in vLLM: 39% faster than EAGLE-3 and 11% faster than DFlash at matched drafter size and data. The 8x4 tree gives about 1.87x over AR at 4B (English), and pooled speedup peaks at 2.12x at 27B. The tree helps only up to about 4 concurrent requests. — [ChainFlow html](https://arxiv.org/html/2611.08669)
  - Limitations: greedy-only tree verification with a patched engine. Output differs from plain greedy on 4-13% of prompts (attributed to fp16 near-ties), so equivalence is not established. Only Qwen3.5 targets, and some ablations only at 0.8B. — [ChainFlow html](https://arxiv.org/html/2611.08669)
- MFD frames its latent as one that "names the future": given the mode, positions are independent, and uncertainty about the future becomes tree breadth. — [MFD](https://arxiv.org/abs/2611.13017)
- KVShot argues that the target's current hidden state is an insufficient summary for long-horizon drafting on Qwen3-8B. — [arXiv 2604.26412](https://arxiv.org/abs/2604.26412)

### Inferences
- ChainFlow covers stages A (VAE over hidden states), P (flow predictor for 8 future latents) and frozen-head decoding. It does not use sampled latents for coherent branches. Its tree comes from beam search over per-position categoricals plus token-feedback heads, so stage S as proposed (several latent samples, each a coherent branch) appears open. Flow matching naturally supports drawing several noise samples, and the paper does not report doing it [inferred from summary, verify in the full text].
- ChainFlow's need for token-feedback heads is a third independent signal, after Lyanna's token-info bias and EAGLE-1's shifted tokens, that decoding future hidden states position-wise through the frozen head loses inter-token dependency. Latent sampling has to supply this coupling or the drafter needs a feedback head.
- ChainFlow's speedups are well below MFD's 6.5x on Qwen3-8B, though the frameworks and targets differ (vLLM and Qwen3.5 vs HF and Qwen3). A latent-AE drafter should be benchmarked against DFlash+DDTree and MFD on Qwen3-8B with the same harness.

### Gaps
- ChainFlow's latent width, compression ratio, reconstruction loss, whether embeddings are shared, its Qwen3.5 model sizes beyond 0.8B/4B/27B, and per-task acceptance lengths were not readable (the PDF fetch returned binary, and the HTML summary lacked these). Read the saved PDF or the HTML directly.
- Whether ChainFlow samples several flow trajectories is not confirmed.
- ConFu, SD², FSP, LLM-JEPA, NextLat, DDTree and GRAFT are covered elsewhere and were not re-researched here.
