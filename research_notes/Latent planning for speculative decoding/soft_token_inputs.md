# Soft-token / embedding-superposition inputs to LMs and their use in speculative drafting

Scope note. Reasoning-accuracy results (Sections 1, 2) are kept separate from drafting/acceptance results (Section 3). All numbers below were read from the cited page in this session unless listed under Gaps. Citation format is "(Source)" at the end of each bullet.

## 1. Soft Thinking, Coconut, Mixture of Inputs, CoT2, ciphers, and 2025-2026 critiques (reasoning accuracy, not drafting)

### Takeaway
Pure probability-weighted embedding inputs (E^T q) fed to an untrained model are at best neutral and often harmful, and vanilla Soft Thinking behaves almost exactly like greedy decoding. Gains appear only when (a) a discrete token anchor is kept (Mixture of Inputs, about +1.3 to +2.4 avg points), (b) randomness is reinjected (Gumbel/Dirichlet), or (c) the model is trained to consume the continuous input (Coconut, soft-token RL).

### Cited Findings

Soft Thinking (Zhang et al., arXiv:2505.15778)
- Training-free method that feeds "concept tokens", probability-weighted mixtures of token embeddings, instead of sampled tokens. The abstract reports pass@1 up to +2.48 points and token usage reduced by up to 22.4% vs standard CoT on math and coding benchmarks ([arXiv:2505.15778](https://arxiv.org/abs/2505.15778)).

Critique, "LLMs are Single-threaded Reasoners" / "Stochastic Soft Thinking" (Wu et al., arXiv:2508.03440, v4, ICLR 2026)
- Probing finds the JS divergence between the Soft Thinking output and the output from feeding only the top-1 token is "highly concentrated ... around 0", while divergence to the top-2-token path "frequently approaches the maximum value". Logit-lens prominence of the top-1 path rises to 1.0 through layers, and the authors call the forward pass "a pruner" ([arXiv:2508.03440v4](https://arxiv.org/html/2508.03440v4)).
- Vanilla Soft Thinking's ROUGE-L to the greedy trace is reported as "significantly higher" than discrete token thinking's (no numeric value in the text) ([arXiv:2508.03440v4](https://arxiv.org/html/2508.03440v4)).
- Average accuracy over AIME24, AIME25, MATH500, AMC23, GPQA-D, HumanEval, MBPP, LiveCodeBench ([arXiv:2508.03440v4](https://arxiv.org/html/2508.03440v4)):
  - DeepSeek-R1-Distill-Qwen-32B. Token greedy 71.57, Token sampling 78.50, Soft vanilla 72.13, Soft+Dirichlet 78.36, Soft+Gumbel 79.55.
  - QwQ-32B. Token greedy 82.64, Token sampling 82.35, Soft vanilla 80.06, Soft+Dirichlet 81.39, Soft+Gumbel 83.04.
  - Skywork-OR1-32B. Token greedy 76.85, Token sampling 82.99, Soft vanilla 79.21, Soft+Dirichlet 83.12, Soft+Gumbel 83.41.
- So vanilla Soft Thinking sits near greedy (DeepSeek 72.13 vs 71.57) and below sampling. Soft+Gumbel beats sampling by only +1.05 (DeepSeek), +0.69 (QwQ), +0.42 (Skywork) average points ([arXiv:2508.03440v4](https://arxiv.org/html/2508.03440v4)).
- The paper argues a convex mixture of embeddings "remains within the manifold" of the input space and so alleviates OOD issues, but does not test this empirically. It truncates to top-k/top-p because the full-vocabulary tail adds noise ([arXiv:2508.03440v4](https://arxiv.org/html/2508.03440v4)).

Mixture of Inputs (Zhuang, Liu, Singh, Shang, Gao, "Text Generation Beyond Discrete Token Sampling", arXiv:2505.14827)
- Input weights are the Bayesian posterior mean w = (H*p + (beta+1-H)*y)/(beta+1), where p is the next-token distribution, y the one-hot sampled token, and H the normalized entropy. High entropy pushes toward p, low entropy toward the hard token ([arXiv:2505.14827v3](https://arxiv.org/html/2505.14827v3)).
- Average over AIME, CountDown4, GPQA-D, LiveCodeBench, Standard / Direct Mixture (pure E^T p) / MoI ([arXiv:2505.14827v3](https://arxiv.org/html/2505.14827v3)):
  - QwQ-32B. 72.86 / 60.96 / 74.15.
  - Nemotron-Super-49B. 53.09 / 46.97 / 55.45.
  - Gemma-3-27B. 40.09 / 41.36 / 41.65.
  - DAPO-Qwen-32B. 58.28 / 47.90 / 60.33.
- Pure distribution input collapses on code. LiveCodeBench drops 76.32 to 53.42 (QwQ), 39.92 to 16.04 (Nemotron), 54.01 to 23.87 (DAPO) ([arXiv:2505.14827v3](https://arxiv.org/html/2505.14827v3)).
- Reported MoI average gain is 1.8% over 16 model-task pairs, with per-task tuned hyperparameters for AIME and CountDown4 ([arXiv:2505.14827v3](https://arxiv.org/html/2505.14827v3)).
- Feeding back mixed output hidden states instead of input embeddings gave outputs the models could not interpret without retraining ([arXiv:2505.14827v3](https://arxiv.org/html/2505.14827v3)).

Coconut, Chain of Continuous Thought (Hao et al., arXiv:2412.06769)
- Feeds the last hidden state back as the next input embedding, with a multi-stage curriculum. The authors claim continuous thoughts "can encode multiple alternative next steps", enabling BFS-like search ([arXiv:2412.06769](https://arxiv.org/abs/2412.06769)).
- GPT-2 results, accuracy % (tokens). GSM8k: CoT 42.9 (25.0), No-CoT 16.5 (2.2), iCoT 30.0, Coconut 34.1 (8.2), Coconut w/o curriculum 14.4. ProntoQA: CoT 98.8, Coconut 99.8. ProsQA: CoT 77.5 (49.4), Coconut 97.0 (14.2), Coconut w/o curriculum 76.1, "pause as thought" 96.6 ([arXiv:2412.06769v2](https://arxiv.org/html/2412.06769v2)).
- Coconut loses to explicit CoT on GSM8k (34.1 vs 42.9). Its win is on search-heavy ProsQA, and "pause as thought" (no continuous content) reaches 96.6 there, close to Coconut's 97.0 ([arXiv:2412.06769v2](https://arxiv.org/html/2412.06769v2)).

Soft-token RL (Butt et al., "Soft Tokens, Hard Truths", arXiv:2509.19170, ICLR 2026)
- Trains continuous CoT with RL using embedding mixtures plus noise. The best setup is soft/fuzzy training with hard discrete decoding at inference, which matches hard training at pass@1 and beats it at pass@32 ([arXiv:2509.19170](https://arxiv.org/abs/2509.19170), [Graphcore blog](https://graphcore-research.github.io/soft-tokens-hard-truths)).
- SofT-GRPO (arXiv:2511.06411) reports small pass@1 gains and larger pass@32 gains over discrete GRPO (search snippet only, not verified in full text) ([arXiv:2511.06411](https://arxiv.org/html/2511.06411v2)).

### Inferences
- The consistent pattern is that soft inputs act as a diversity or exploration tool, not as a better point estimate. Under greedy verification (lossless greedy speculative decoding), the property that matters is matching the argmax, where soft inputs collapse to the top-1 path and add nothing.
- The MoI "Direct Mixture" rows are the closest literature analogue to feeding E^T q into a frozen model. They lose 6 to 12 average points on 3 of 4 models. This lines up with the friend's Phase 1 result that training-free input-side priors lower tau.
- The surviving input recipe is "hard anchor plus entropy-weighted soft residual" (MoI), not pure soft.

### Gaps
- CoT2 / continuous CoT (Gozeten et al., 2025) and "Let Models Speak Ciphers" (Pham et al., arXiv:2310.06272, ICLR 2024) were not fetched this session, so no numbers are given. Ciphers is about multi-agent debate with embedding-expectation messages, as I recall, but this is unverified.
- Soft Thinking's per-model and per-benchmark numbers and its Cold Stop details were not retrieved (abstract only).
- No paper found that isolates mode averaging in E^T q with a controlled bimodal test.

## 2. Self-conditioning in diffusion and diffusion LMs

### Takeaway
In diffusion LMs, feeding the model its own previous clean-state predictions is one of the most reliable quality levers. Reported generative-perplexity reductions are about 45 to 61%. Crucially, these models are trained with the feedback channel, so this is evidence that trained soft feedback helps iterative refinement, not that frozen models can use soft inputs.

### Cited Findings
- SCMDM, "Simple Self-Conditioning Adaptation for Masked Diffusion Models" (Cardei, Ta, Fioretto, arXiv:2604.26985). Post-trains a pretrained absorbing-mask MDM to condition on its previous clean-state predictions. OWT generative perplexity falls 42.89 to 23.72. The authors argue the standard 50% self-conditioning dropout is suboptimal in post-training and that full specialization to refinement works better ([arXiv:2604.26985](https://arxiv.org/abs/2604.26985)).
- Loopholing Discrete Diffusion (Jo, Yoon, Deschenaux, Gulcehre, Ahn, arXiv:2510.19304, ICLR 2026). Names the "sampling wall": categorical sampling collapses distributional information into one-hot vectors so later steps lose it. Adds a deterministic latent pathway carried across steps, trained with self-conditioning that avoids unrolling the full trajectory. Reports up to 61% lower generative perplexity than prior baselines and gains on Countdown and Game of 24 ([arXiv:2510.19304](https://arxiv.org/abs/2510.19304)).
- PreDiff-LM (arXiv:2607.25157). Ablation says removing self-conditioning raises PPL@50% mask from 69.0 to 214.0 (search snippet, full text not read) ([arXiv:2607.25157](https://arxiv.org/pdf/2607.25157)).
- Soft-masking for diffusion LMs (arXiv:2510.17206) propagates soft token information across steps and reports better perplexity and MAUVE when continuing pretraining of a 169M model (search snippet only) ([arXiv:2510.17206](https://arxiv.org/pdf/2510.17206v1)).
- FastDiSS (arXiv:2604.05551) notes that self-conditioning has a train/inference mismatch that hurts few-step sampling in continuous diffusion LMs (search snippet only) ([arXiv:2604.05551](https://arxiv.org/html/2604.05551)).

### Inferences
- Loopholing's latent pathway is a hidden-state carry, not E^T q. Its framing (one-hot sampling throws away information) is the strongest conceptual argument for H2. The evidence supports "carry rich state across refinement passes", which can be hidden state rather than a token-embedding mixture.
- A block drafter with a second self-conditioning pass is structurally a 2-step diffusion LM with self-conditioning. The large two-pass gain you observe is consistent with this literature, but nothing here says soft beats hard as the feedback content. Most of these works compare self-conditioning against none, not soft against argmax feedback.

### Gaps
- Analog Bits (Chen et al., arXiv:2208.04202), CDCD (Dieleman et al., arXiv:2211.15089), Plaid (Gulrajani & Hashimoto, arXiv:2305.18619), SSD-LM (Han et al., arXiv:2210.17432), TESS (Mahabadi et al., arXiv:2305.08379), and ReMDM were not fetched this session, so I give no numbers for them. These are the canonical self-conditioning references and a writer should not quote numbers for them from these notes.
- I found no paper with a clean ablation of soft-probability vs argmax-token self-conditioning in a masked diffusion LM at matched training. This is the exact experiment H2 needs.
- SCMDM's page does not say whether the fed-back predictions are probabilities, argmax tokens, or embeddings.

## 3. Speculative decoding with soft, expected, or superposed inputs (drafting/acceptance results)

### Takeaway
No work found shows soft token distributions as drafter inputs raising acceptance length. The friend's repo shows they do not, across DFlash, EAGLE-3, and gated-LoRA self-speculation on Qwen3. The one positive 2026 result, Parallel Reconsideration (arXiv:2611.05724), gives +5.3 to 10.2% tau from a second drafter pass conditioned on the hidden feature, and feeding the committed hard token there instead costs 4.1% tau.

### Cited Findings

Friend's repo, EmbeddingSuperPosition (Qwen3-4B thinking mode, GSM8K, greedy, RTX 4070) ([GitHub](https://github.com/bijmuj/EmbeddingSuperPosition))
- Phase 1, training-free on DFlash b16 (8 prompts, baseline tau 3.073). prev_draft input alpha 0.1 gave 3.062 (-0.012). two_pass input alpha 0.3, T=1 gave 2.763 (-0.309). two_pass_hidden gave 2.524 (-0.551). An oracle one-hot fed at the input gave 1.803 (-1.297), because DFlash saw only MASK at positions 1 and beyond in training. Oracle output-side fusion (lambda 1) gave 7.907 (+4.804).
- Complementarity analysis. Stale priors are right mostly where the fresh draft already failed. At depth 1, prev_draft adds a unique correct answer only 1 to 3% of the time. Target-stale logits carry more unique information than the drafter's own stale marginals.
- Phase 2/3, fine-tuned DFlash (20 held-out prompts). Stock 3.262, compute-matched control FT 3.832, prev_draft FT 3.815 (-0.019 vs control), two_pass FT 3.769 (-0.064 vs control, draft time 13.1 ms vs 6.6 ms), supertoken pre-drafter FT 3.776 (-0.060).
- EAGLE-3 fine-tuned, tree d7 k10 n60. Base 3.33, hard true-token feedback control 5.42 (2.84x), supertoken feedback 4.86 (2.55x). Supertoken training lowered tau by about 0.2 to 0.56 in every decoding mode.
- Gated-LoRA self-speculation on Qwen3-1.7B. Supertoken inputs gave tau x1.012 +/- 0.005 vs hard/mask baseline, with throughput x0.998.
- Sampling (T>0). Super-trained drafter tau relative to hard-trained was x0.91 at T=0.6 and x0.85 at T=1.0.

Parallel Reconsideration (Le et al., arXiv:2611.05724, Nov 2026) ([arXiv:2611.05724](https://arxiv.org/html/2611.05724v1))
- An EAGLE-3.1 single-layer drafter gets a gated second pass on uncertain positions, conditioned on fc(f0, Q(f0)), where f0 is the first-pass feature and Q a learned self-query. The re-sampled token becomes an extra tree candidate.
- Averages across GSM8K, MATH-500, AIME25, HumanEval, MBPP, LiveCodeBench, MT-Bench. Llama-3.1-8B T=0 B=48 tau 5.05 to 5.37 (+6.3%), speedup 2.18x to 2.28x. Qwen3-8B T=0 B=24 tau 4.00 to 4.36 (+9.0%), 1.92x to 2.06x. Gemma-3-12B T=0 B=24 tau 4.80 to 5.27 (+9.8%). Gemma-3-12B T=1 B=48 tau 5.10 to 5.62 (+10.2%), 2.71x to 2.96x.
- Ablation (Llama-3.1-8B, B=48, T=0). Replacing the feature-derived self-query with the committed token embedding e(t_p) cuts tau by 4.1% and speedup by 4.6%. The authors argue a committed token pushes the drafter to advance along one branch, while reconsideration needs to stay in place.
- Compute control. A second EAGLE decoder layer instead gives only +2.0% tau (4.95 to 5.05) and speedup drops 2.11x to 1.96x.
- No soft-distribution (E^T q) feedback variant is tested.

Superposed Decoding (Shen et al., arXiv:2405.18400, NeurIPS 2024) ([arXiv:2405.18400](https://arxiv.org/abs/2405.18400))
- Feeds a weighted sum of the latest token embeddings of k drafts as one input, expands to k^2 candidates, keeps the top k, and filters with n-gram interpolation. Reports at least 2.44x faster for k >= 3 than producing k drafts separately, coherence and factuality on par with nucleus and greedy, and user preference in a compute-normalized setting.
- This is a multi-draft generation method, not speculative decoding with a verifier. It relies on n-gram filtering to clean up superposition errors.

Related, not soft inputs
- OWL (arXiv:2510.07535) appends a learned [SPEC] token embedding to the target input to give the drafter an extra estimate and reports higher acceptance length (search snippet) ([arXiv:2510.07535](https://arxiv.org/pdf/2510.07535)).

### Inferences
- Across the friend's repo and Parallel Reconsideration, the second-pass benefit comes from conditioning on rich continuous state (a learned feature/query) plus extra compute, scored as an extra tree candidate. It does not come from injecting a token-embedding mixture. Soft marginals as the second-pass input (friend's two_pass FT) were -0.064 tau vs control at about 2x draft cost.
- Your two-pass observation is likely the same effect. Without a matched-compute, matched-training ablation of the pass-2 input (mask vs argmax embedding vs E^T q vs pass-1 hidden state vs MoI hybrid), the gain cannot be credited to softness.
- Output-side fusion is where priors helped in principle (oracle +4.8 tau). Input-side oracle hurt a frozen drafter. If priors are used at all, logit-level fusion with target-stale logits is the better-motivated channel.

### Gaps
- No paper found that feeds E^T q or a probability distribution into a drafter or target and reports improved acceptance. Searches for "soft draft", "probabilistic drafts", and "mixture embeddings parallel decoding" returned nothing on point.
- Parallel Reconsideration's pass-count ablation (Appendix D.3) was not in the portion read.

## 4. Mode-averaging risk: ambiguous/OOD blended embeddings vs useful uncertainty

### Takeaway
Evidence points to mode averaging being resolved by the model collapsing onto the top component (greedy-like), not to confusion from a blended point. Pure blends hurt badly on code and in frozen drafters. Uncertainty becomes useful only with a hard anchor, with noise, or with training.

### Cited Findings
- Wu et al. find the forward pass prunes soft inputs to the top-1 path (JS divergence to top-1 result near 0, logit-lens prominence of top-1 rising to 1.0) ([arXiv:2508.03440v4](https://arxiv.org/html/2508.03440v4)).
- Their case study shows semantically incoherent continuations after a blended input (for example "I" following "let"), consistent with blended inputs sometimes producing bad successors ([arXiv:2508.03440v4](https://arxiv.org/html/2508.03440v4)).
- Pure distribution inputs (MoI Direct Mixture) cut LiveCodeBench by 22.9 points (QwQ-32B), 23.9 (Nemotron), and 30.1 (DAPO), while helping Gemma-3-27B slightly on average (41.36 vs 40.09) ([arXiv:2505.14827v3](https://arxiv.org/html/2505.14827v3)).
- In favor of useful uncertainty, entropy-gated mixing with a hard anchor (MoI) adds +1.29 to +2.36 average points on 4 models ([arXiv:2505.14827v3](https://arxiv.org/html/2505.14827v3)). Soft-token RL preserves diversity (pass@32) and base-model behaviour out of domain ([arXiv:2509.19170](https://arxiv.org/abs/2509.19170)). Loopholing argues one-hot sampling destroys information that later steps need ([arXiv:2510.19304](https://arxiv.org/abs/2510.19304)).
- In drafting, supertoken feedback in training hurt EAGLE-3 more as temperature rose (x0.91 at T=0.6, x0.85 at T=1.0), which is where distributions are most multimodal ([GitHub](https://github.com/bijmuj/EmbeddingSuperPosition)).
- Parallel Reconsideration says hard token feedback commits to one branch, and its uncertainty is expressed by adding a second candidate to the tree rather than blending ([arXiv:2611.05724](https://arxiv.org/html/2611.05724v1)).

### Inferences
- For drafting, multimodality is better served by branching (tree candidates, multi-draft) than by blending into one input vector. Every positive drafting result found handles uncertainty through extra tree candidates.
- The convex-hull "stays on manifold" argument (Wu et al.) is asserted, not tested, and is contradicted in practice by the Direct Mixture code collapse and the frozen-drafter oracle result.

### Gaps
- No controlled study found that measures OOD-ness of E^T q vectors (for example nearest-neighbour distance or hidden-state statistics) as a function of distribution entropy or number of modes.

## 5. Verdict for H2 (soft distributions as inputs beat hard argmax for drafting/refinement)

### Takeaway
Replace H2. As stated, the evidence is against it. Every drafting test of soft token inputs found (DFlash, EAGLE-3, gated-LoRA self-spec) is neutral to negative at matched training, and the literature's soft-input gains are small, reasoning-only, and depend on noise or a hard anchor. The surviving hypothesis is that a second refinement pass conditioned on the drafter's continuous pass-1 state, emitted as extra tree candidates, raises tau at matched compute.

### Cited Findings
- Soft priors never beat the hard/mask baseline by more than about 1% tau and never improved throughput ([GitHub](https://github.com/bijmuj/EmbeddingSuperPosition)).
- Feature-conditioned second pass gives +5.3 to 10.2% tau and +3.5 to 9.2% speedup, and committed-token feedback is 4.1% worse in tau ([arXiv:2611.05724](https://arxiv.org/html/2611.05724v1)).
- Hard-anchor plus entropy-weighted soft (MoI) is the only soft-input variant with consistent gains, on reasoning accuracy ([arXiv:2505.14827v3](https://arxiv.org/html/2505.14827v3)).
- Self-conditioning in masked diffusion LMs gives large generative-perplexity gains when trained in ([arXiv:2604.26985](https://arxiv.org/abs/2604.26985), [arXiv:2510.19304](https://arxiv.org/abs/2510.19304)).

### Inferences
- Recommended replacement, H2'. The two-pass gain comes from self-conditioning on continuous pass-1 state (hidden feature or learned query), not from softness of token inputs. The test is a matched-compute, matched-data ablation of the pass-2 input with these arms: (a) mask only (extra compute, no feedback), (b) argmax token embedding, (c) E^T q with top-k truncation, (d) MoI hybrid (argmax anchor plus entropy-weighted soft), (e) pass-1 hidden state or feature (Loopholing / Parallel Reconsideration style). Report tau and tok/s, and emit pass-2 tokens as extra tree candidates where possible.
- Prediction from the evidence. (e) is at least as good as (b), (c) is about equal to (b) under greedy (collapse), and (d) is the only soft variant with a plausible small edge, mostly at T>0. If (a) matches (b) through (e), the gain is pure extra compute.
- Train with self-conditioning as in diffusion LMs (SCMDM suggests full rather than 50%-dropout self-conditioning in post-training). Use the friend's compute-matched control FT as the bar, since thinking-mode data adaptation alone gave +0.584 tau there.
- Drop pure E^T q input injection into frozen drafters. Keep output-side fusion with target-stale logits as a secondary probe only.

### Gaps
- Whether the two-pass gain persists for a DFlash-style block drafter (as opposed to EAGLE-3.1 trees) under matched compute is untested in the found literature. This is the experiment your own data is positioned to answer.
