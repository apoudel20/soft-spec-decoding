# World-model-style latent prediction for language, and its use for decoding speed, drafting and planning

Read-depth legend used below.
- [full] means the paper's HTML or PDF body was read (sometimes only the first ~100k characters).
- [abstract] means only the arXiv abstract (via the arXiv API) was read.
- [snippet] means only search-engine snippets of the abstract or body were seen. Treat numbers from these as provisional.

Target use kept in mind throughout: a small drafter-side model that rolls a compact latent of a large model's future forward to guide drafting.

## Q1. Belief-state and next-latent models for text (BST, NextLat, JTP, next-latent prediction, self-predictive representations, latent dynamics MLPs)

### Takeaway
A small cluster of Microsoft Research and related work (BST, JTP, NextLat, Semformer, latent lookahead) trains the LM's hidden state to be a belief state or to predict future latents. All of it is evaluated at small scale (57M to 162M, Semformer 125M) on synthetic planning tasks plus TinyStories or FineWeb sanity checks, and none of it is used for decoding speed. NextLat is the only one with an explicit learned latent dynamics MLP rolled forward several steps, and it is used only as a training regularizer.

### Cited Findings

Belief State Transformer (BST), arXiv 2410.23506, Hu et al., ICLR 2025 [snippet]
- Mechanism. A forward encoder reads a prefix and a backward encoder reads a suffix. The model is trained to predict the next token after the prefix and the previous token before the suffix. The authors credit success to learning a compact belief state. v3 (Sept 2025) is an "updated report with new improvements and authors." — [arXiv 2410.23506](https://arxiv.org/abs/2410.23506), [ICLR 2025 proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/01b3dea1871f7cea1e0e6be1f2f085bc-Abstract-Conference.html)
- Results. On star-graph path finding, removing either the belief-state objective or the backward encoder drops performance to near random among valid paths. On TinyStories it beats fill-in-the-middle for goal-conditioned generation. The authors list scaling as open. — [arXiv 2410.23506](https://arxiv.org/abs/2410.23506)
- Cost. In NextLat's TinyStories comparison, BST needs 114M training params for a 57M GPT-sized model and trains at 0.19 it/s vs 3.72 it/s for GPT (about 20x slower). Inference uses 57M or 114M params. — [NextLat 2511.05963](https://arxiv.org/html/2511.05963v1) [full]

Joint multi-token prediction (JTP), arXiv 2503.21801, Ahn, Lamb, Langford (Microsoft Research), March 2025 [snippet]
- Mechanism. Predicts the joint distribution of several future tokens over a sliding window. A teacher-forced representation bottleneck forces the hidden state to carry the information needed for the joint prediction. The authors claim this yields a short-horizon belief state, which independent-head MTP does not. — [arXiv 2503.21801](https://arxiv.org/html/2503.21801v1)
- Results. In a GPT-2-scale sanity check (162M, FineWeb) JTP gave slightly higher next-token loss than the GPT baseline while showing multi-token prediction ability. The authors call this preliminary. — [arXiv 2503.21801 PDF](https://arxiv.org/pdf/2503.21801)
- NextLat notes JTP's belief-state claim lacks formal conditions and gives a proposition (under k-observability) for when JTP forms belief states. — [NextLat 2511.05963](https://arxiv.org/pdf/2511.05963)

NextLat, arXiv 2511.05963 (already known, comparison only) [full]
- Mechanism. A latent dynamics model p_psi (simple MLPs) maps (current final-layer hidden state, next token) to the next hidden state. Loss is next-token CE plus smooth-L1 on multi-step recursive latent rollouts of horizon d (teacher-forced tokens, stop-gradient targets) plus a KL between the frozen LM head's distribution on true vs predicted hidden states. p_psi is training-only, so inference is unchanged. Theorem 3.2 argues the objective pushes hidden states toward belief states. — [NextLat](https://arxiv.org/html/2511.05963v1)
- Sizes and cost on TinyStories. GPT 57M, NextLat 66M train / 57M inference, 3.26 it/s (d=1) and 1.89 it/s (d=8) vs GPT 3.72. JTP 60M at 3.33 / 2.61. MTP 64M / 114M at 3.12 / 1.81. — [NextLat](https://arxiv.org/html/2511.05963v1)
- Countdown accuracy at horizon d = 1 / 4 / 8. NextLat 54.2 / 57.4 / 57.8, MTP 39.2 / 49.7 / 57.3, JTP 39.0 / 49.4 / 55.0, BST 42.3, GPT 33.1. Path-Star: NextLat near 100% on all graph sizes, BST fails on G(7,7). Manhattan taxi: 98.7% valid trajectories vs GPT 97.0%, detour robustness 95% vs 85%. — [NextLat](https://arxiv.org/html/2511.05963v1)
- The paper does not discuss decoding speed or drafting. — [NextLat](https://arxiv.org/html/2511.05963v1)
- Self-predictive representation lineage it cites from RL. SPR (Schwarzer et al., arXiv 2007.05929), DeepMDP (Gelada et al., ICML 2019), Zhang et al. (arXiv 2006.10742), EfficientZero (Ye et al., NeurIPS 2021), TD-MPC (Hansen et al., arXiv 2203.04955), Ni et al. ICLR 2024, Tang et al. ICML 2023. — [NextLat related work](https://arxiv.org/html/2511.05963v1)

Semformer, arXiv 2409.11143, Yin et al., EMNLP 2024 [snippet]
- Mechanism. Trainable planning tokens are inserted after the prefix and trained to regress latent representations of the upcoming response produced by an autoencoder that is used only during training. — [arXiv 2409.11143](https://arxiv.org/html/2409.11143v1)
- Results. Near-perfect accuracy on graph path-finding where standard training shortcuts. Pause tokens alone fail. A 125M model was pretrained and evaluated on perplexity, in-context learning and summarization fine-tuning. — [arXiv 2409.11143](https://arxiv.org/abs/2409.11143)

Latent Lookahead Training, arXiv 2603.20219, Noci et al., March 2026 [abstract]
- Mechanism. At selected positions the model recursively feeds hidden states back into the context for tau steps before committing to a token. The latent steps are supervised against the next tau ground-truth tokens. — [arXiv 2603.20219](https://arxiv.org/html/2603.20219v1)
- Results. Reported to substantially outperform AR and non-AR baselines on maze solving, Sudoku and ProsQA. The abstract gives no tau values or model sizes. — [arXiv 2603.20219](https://arxiv.org/html/2603.20219v1)

Concept-level prediction inside a token LM
- CoCoMix, arXiv 2502.08524, Tack et al. (Meta), Feb 2025. Predicts continuous concepts taken from a pretrained sparse autoencoder and mixes them into the hidden state alongside tokens. Reported more sample-efficient than next-token prediction, knowledge distillation and pause tokens, and concepts are steerable. — [arXiv 2502.08524](https://arxiv.org/abs/2502.08524) [abstract]
- Learning to Plan for LM from Unlabeled Data, arXiv 2404.00614, Cornille et al., 2024. A separate planner predicts future "abstract writing actions" (centroids of clustered sentence embeddings) and the LM is conditioned on them. Improves LM performance, especially text structure. Planner is separate and shareable. — [arXiv 2404.00614](https://arxiv.org/abs/2404.00614) [abstract]

### Inferences
- NextLat's p_psi is the closest existing component to the target drafter. It is a small MLP that rolls a large model's final hidden state forward given tokens. But it is trained only to regularize the big model and is discarded, and it needs the next token as input, so it is a token-conditioned transition model, not a free-running latent forecaster.
- Cornille et al. is the closest existing "separate side model predicts a compact future latent and the LM is conditioned on it" design. It is a coarse, discrete (cluster-centroid) latent at the sentence level, which is a plausible format for a drafter "gist."
- All belief-state results are at 57M to 162M scale on synthetic or toy text. There is no evidence yet that these objectives give usable belief states in multi-billion-parameter LMs.

### Gaps
- I did not read the BST or JTP bodies, so the exact JTP loss, the BST v3 changes, and any larger-scale BST results are unconfirmed.
- I found no paper that applies self-predictive representation learning (SPR/BYOL-style) to a pretrained LLM as a post-hoc side model, other than NextLat's training-time regularizer.
- Latent lookahead tau values, model size and cost are not in the abstract.

## Q2. JEPA and Dreamer analogues for text, latent planning, latent reasoning, hierarchical latent generation

### Takeaway
Text JEPA work is young and is running into a specific failure mode. Squared-error latent prediction averages over multiple valid continuations and collapses. The sentence-level and paragraph-level latent generators that do work (LCM, PLANNER, Latent Diffusion for Language, CALM) all use a generative (diffusion, energy or VAE) head over the latent rather than a point regressor. Latent reasoning methods (Coconut, CODI, CCoT, CoLaR, Soft Thinking, PLaT) do roll a latent forward, but to replace chain-of-thought tokens, not to draft output text.

### Cited Findings

JEPA for text
- LLM-JEPA (known, comparison only), arXiv 2509.14252, Huang, LeCun, Balestriero, 2025. Adds a JEPA loss over paired textual views on top of next-token prediction. — [cited in NextLat](https://arxiv.org/html/2511.05963v1); a follow-up ER-JEPA adds experience replay — [arXiv 2609.36952](https://arxiv.org/pdf/2609.36952) [snippet]
- The JEPA Paradox in Language, arXiv 2607.23531, Dinh et al., July 2026. Argues squared-error latent prediction mismatches language because masked text has multiple valid completions. Formalizes three conditions (predictability, non-collapse, low conditional variance), and matched I-JEPA vs T-JEPA experiments over five seeds show failures cause collapse and poor transfer. Concludes text JEPA objectives must preserve multiple plausible completions. — [arXiv 2607.23531](https://arxiv.org/abs/2607.23531) [abstract]
- VL-JEPA, arXiv 2512.10942, Chen et al. (Meta), Dec 2025. Predicts continuous embeddings of target text instead of generating tokens. Beats a matched token-space VLM with 50% fewer trainable params and cuts decoding operations by 2.85x through selective decoding (decode to text only when needed). 1.6B params. — [arXiv 2512.10942](https://arxiv.org/html/2512.10942v1) [abstract]

Sentence- and chunk-level latent generation
- Large Concept Models, arXiv 2412.08821, LCM team (Meta FAIR), Dec 2024. Autoregressive next-sentence prediction in the SONAR embedding space (200 languages). Variants tried include MSE regression, diffusion and quantized SONAR. 1.6B models on about 1.3T tokens, one architecture scaled to 7B on about 2.7T tokens. Reports strong zero-shot cross-lingual generalization. — [arXiv 2412.08821](https://arxiv.org/abs/2412.08821) [abstract]
- CALM, arXiv 2510.27688, Shao et al. (WeChat AI), Oct 2025 [full]. A context-free VAE compresses K tokens into one continuous vector (latent dim 128 for K=4, autoencoder about 75M params, over 99.9% token reconstruction). A Transformer predicts the next vector with a single-step energy-based head (about 10% of params). K in {1,2,4,8}, main results K=4. — [arXiv 2510.27688](https://arxiv.org/html/2510.27688v1)
- CALM results (BrierLM on WikiText-103, Pile 230B tokens). Transformer-S 281M 6.05, M 465M 7.07, L 849M 8.98. CALM-M 371M 5.72, L 735M 6.58, XL 1.82B 8.53. CALM-M is comparable to Transformer-S at 44% fewer train FLOPs and 34% fewer inference FLOPs. K=8 degrades. Energy head beat diffusion and flow-matching heads on ceiling. Discrete token input (4.70) beat continuous vector input (3.25). Only FLOPs are reported, no wall-clock. Limitations include a context-free autoencoder and the need for likelihood-free sampling. — [arXiv 2510.27688](https://arxiv.org/html/2510.27688v1)
- SentenceVAE, arXiv 2408.00655, An et al., 2024 [snippet]. A sentence encoder compresses a sentence into one token and a sentence decoder reconstructs it, so the LLM runs sentence-by-sentence. v5 abstract reports 204% to 365% inference speedup, perplexity at 46% to 75% of the original and lower memory, on the Wanjuan dataset. v1 and v5 abstracts describe the claims differently. — [arXiv 2408.00655](https://arxiv.org/pdf/2408.00655)

Paragraph-level latent diffusion
- PLANNER, arXiv 2306.02531, Zhang et al. (Apple), June 2023. A latent diffusion planning module generates semantic paragraph embeddings coarse-to-fine and an autoregressive decoder realizes them. Evaluated on semantic generation, text completion and summarization. Motivated by AR repetition and error accumulation. — [arXiv 2306.02531](https://arxiv.org/abs/2306.02531) [abstract]
- Latent Diffusion for Language Generation, arXiv 2212.09462, Lovelace et al., Dec 2022. Continuous diffusion in the latent space of a pretrained encoder-decoder autoencoder, decoded with the pretrained decoder. — [arXiv 2212.09462](https://arxiv.org/abs/2212.09462) [abstract]
- Diffusion Guided Language Modeling, arXiv 2408.04220, Lovelace et al., Aug 2024. A guided diffusion model produces a latent proposal that steers an AR LM toward attributes, and new attributes need only a logistic-regression classifier. — [arXiv 2408.04220](https://arxiv.org/abs/2408.04220) [abstract]

Latent reasoning (continuous thought)
- Coconut, arXiv 2412.06769, Hao et al. (Meta), Dec 2024. Feeds the last hidden state back as the next input embedding instead of decoding a token. Continuous thoughts can encode several alternative next steps (BFS-like). Beats CoT on planning-heavy logic tasks with a better accuracy-efficiency trade-off. — [arXiv 2412.06769](https://arxiv.org/abs/2412.06769) [abstract]
- CODI, arXiv 2502.21074, 2025. Self-distills explicit CoT into six continuous thoughts. About 2.7x speedup (3.1x compression) on compact CoTs and about 5.9x (7.8x compression) on verbose CoTs on GSM8k-Aug variants, and outperforms Coconut across compression ratios. — [arXiv 2502.21074](https://arxiv.org/pdf/2502.21074) [snippet]
- Compressed Chain of Thought (CCoT), arXiv 2412.13171, Cheng et al., Dec 2024. Variable-length continuous contemplation tokens that compress reasoning chains, works with off-the-shelf decoders, adjustable reasoning budget. — [arXiv 2412.13171](https://arxiv.org/abs/2412.13171) [abstract]
- CoLaR, arXiv 2505.16552, 2025. Merges embeddings of consecutive tokens by a compression factor. Reports 14.1% higher accuracy than latent baselines at matched compression and 53.3% shorter chains with 4.8% degradation vs explicit CoT. — [arXiv 2505.16552](https://arxiv.org/html/2505.16552) [snippet]
- Soft Thinking, arXiv 2505.15778, Zhang et al., May 2025. Training-free. Concept tokens are probability-weighted mixtures of token embeddings. Up to +2.48 pass@1 and up to 22.4% fewer tokens vs standard CoT on math and code. — [arXiv 2505.15778](https://arxiv.org/abs/2505.15778) [abstract]
- PLaT, arXiv 2601.21358, Wang et al., Jan 2026. Separates a Planner that evolves latent planning states from a Decoder that produces text readouts on demand, with answer-aware stopping. Llama-1B GSM8K Pass@128 80.59% vs CODI 72.37%, but lower Pass@1 in several math settings. — [arXiv 2601.21358](https://arxiv.org/html/2601.21358v1) [abstract]

Survey pointer
- "Alternatives to Next Token Prediction in Text Generation, A Survey," arXiv 2509.24435, groups these into multi-token prediction, plan-then-generate, latent reasoning, continuous generation, diffusion and concept/sentence-level modeling (LCM, SentenceVAE), plus JEPA (I-JEPA, V-JEPA, data2vec, LLM-JEPA). — [arXiv 2509.24435](https://arxiv.org/pdf/2509.24435) [only bibliography keys were readable]

### Inferences
- For a drafter that forecasts a large model's future latent, the JEPA Paradox and CALM results together argue against a plain MSE regressor if the latent must be decoded into one concrete continuation. Either predict a distribution over latents (energy, diffusion or flow head as in CALM, LCM and PLANNER), predict a coarse discrete code (Cornille centroids), or use the latent only as soft guidance where averaging is harmless.
- CALM's finding that discrete-token input beat continuous-vector input suggests the drafter should be fed tokens plus the large model's hidden state, not its own predicted latents alone. This matches EAGLE's choice of feeding the token sequence advanced by one step (see Q3).
- PLaT's planner/decoder split is architecturally the nearest latent-reasoning analogue to "roll a latent forward, then read out text." Its weaker Pass@1 is a warning that latent rollouts can increase diversity more than they increase top-1 accuracy.
- No Dreamer-style text model (learned latent transition plus imagined rollouts for planning or value estimation over generation) turned up in this search beyond the RL-derived lineage NextLat cites.

### Gaps
- I did not read the LCM paper body, so its per-variant numbers (MSE vs diffusion LCM) and any generation-speed figures are unconfirmed.
- I did not verify the LLM-JEPA arXiv ID against arXiv directly. It comes from NextLat's bibliography.
- No dedicated "language world model" paper with latent imagination for decoding was found in the searches run.

## Q3. Use of latent models for faster decoding (self-speculative drafting, latent-conditioned MTP, planned-latent parallel decoding, sentence drafting then filling), lossless or lossy

### Takeaway
The only widely used lossless method that drafts in latent space is the EAGLE family, which autoregresses the target's second-to-top-layer features and verifies tokens exactly. Everything that drafts at the sentence, step or "meaning" level (SentenceVAE, Lookahead Reasoning, SemanticSpec, FLy) is lossy because it accepts semantically equivalent rather than exactly matching output. Models that generate in a chunk latent from scratch (CALM, LCM) are new models, not accelerators of an existing target. No paper found uses a small side model that rolls a compact latent of a large model's future forward to steer a token drafter.

### Cited Findings

Lossless
- EAGLE, arXiv 2401.15077, Li et al., Jan 2024. Autoregression at the feature (second-to-top layer) level, with feature uncertainty resolved by also feeding the token sequence advanced one step. LLaMA2-Chat 70B gets 2.7x to 3.5x latency speedup and doubled throughput while preserving the output distribution. — [arXiv 2401.15077](https://arxiv.org/abs/2401.15077) [abstract]
- EAGLE-style drift. At later draft steps the target's hidden states do not exist, so the drafter feeds back its own predicted features, which drift from the target's. EAGLE-3 drops strict feature matching and trains on its own multi-step outputs. HASS and POSS address the same multi-position problem. — [POSS arXiv 2506.03566](https://arxiv.org/pdf/2506.03566) [snippet]. Note that the EAGLE-3 and HASS descriptions come from secondary summaries in search results and the IDs were not verified here.
- DART, arXiv 2601.19278, 2026. A single layer reads the target's hidden states and predicts logits for several future masked positions in one forward pass, avoiding autoregressive drafting. — [arXiv 2601.19278](https://arxiv.org/html/2601.19278v1) [snippet]
- Multi-token prediction, arXiv 2404.19737, Gloeckle et al. (Meta), 2024. n independent heads on a shared trunk. 13B models solve 12% more HumanEval and 17% more MBPP problems, and 4-token models are up to 3x faster at inference via self-speculative decoding. — [arXiv 2404.19737](https://arxiv.org/abs/2404.19737) [abstract]
- Your LLM Knows the Future, arXiv 2507.11851, Samragh et al. (Apple), July 2025. Finds correct future tokens often lie within the top-200 logits of a standard AR model. Gated LoRA plus a small sampler head fine-tune for MTP with original weights frozen. Reports nearly 5x faster code and math and about 2.5x faster chat, with no quality loss (self-speculative with verification). — [Apple ML Research](https://machinelearning.apple.com/research/prediction-potential), [arXiv 2507.11851](https://arxiv.org/abs/2507.11851v1) [snippet]

Lossy
- SentenceVAE (arXiv 2408.00655). Sentence-level generation, 204% to 365% speedup, perplexity changes. Lossy because it replaces token-level generation of the original model. — [arXiv 2408.00655](https://arxiv.org/pdf/2408.00655) [snippet]
- Lookahead Reasoning, arXiv 2506.19830, Fu et al., NeurIPS 2025. A small drafter proposes future reasoning steps, a 7B LLM-as-judge (or embedding model) accepts semantically correct steps, and token-level speculative decoding runs inside each step so the speedups multiply. GSM8K/AIME speedup rises from 1.4x to 2.1x "while preserving answer quality." Lossy (semantic acceptance). — [arXiv 2506.19830](https://arxiv.org/abs/2506.19830v1) [snippet]
- SemanticSpec, arXiv 2602.03708, Dong et al., Feb 2026. Verifies whole semantic sequences using a probe on hidden states that estimates meaning likelihood. Up to 2.7x on DeepSeek-R1-32B and 2.1x on QwQ-32B. Lossy. — [arXiv 2602.03708](https://arxiv.org/abs/2602.03708) [abstract]
- FLy (Training-Free Loosely Speculative Decoding), arXiv 2511.22972, ICLR 2026. Accepts semantically correct drafts beyond exact match using an entropy gate and deferred token window. Lossy. — [arXiv 2511.22972](https://arxiv.org/abs/2511.22972) [snippet]
- Latent reasoning speedups (CODI 2.7x to 5.9x, Soft Thinking up to 22.4% fewer tokens, CoLaR 53.3% shorter chains) come from shortening or replacing the visible chain of thought. Final answers can change, so these are lossy with respect to the original model's output. — see Q2 sources.
- VL-JEPA's 2.85x reduction in decoding operations comes from decoding text only when the predicted embedding changes, which is selective rather than verified. — [arXiv 2512.10942](https://arxiv.org/html/2512.10942v1) [abstract]

Different-model latent generators (not accelerators of an existing target)
- CALM (K=4) cuts generative steps by K and inference FLOPs by 34% at Transformer-S-level quality. No wall-clock numbers. — [arXiv 2510.27688](https://arxiv.org/html/2510.27688v1) [full]
- LCM generates sentence embeddings autoregressively then decodes with SONAR. — [arXiv 2412.08821](https://arxiv.org/abs/2412.08821) [abstract]

Guidance rather than speed
- Diffusion Guided LM (2408.04220) and PLANNER (2306.02531) use a planned latent to steer an AR decoder for attribute control and diversity, not speed. — [arXiv 2408.04220](https://arxiv.org/abs/2408.04220), [arXiv 2306.02531](https://arxiv.org/abs/2306.02531) [abstract]
- Cornille et al. (2404.00614) conditions the LM on a separately predicted future writing-action latent. — [arXiv 2404.00614](https://arxiv.org/abs/2404.00614) [abstract]

### Inferences
- The target design (small drafter rolls a compact latent of the large model's future, then drafts tokens conditioned on it, and the large model verifies) would be lossless under standard speculative verification and sits between EAGLE (feature-level, one step at a time, full-width features) and Cornille / Semformer (coarse future latent conditioning the generator). That combination appears unoccupied in what I found.
- EAGLE's lesson is that feature prediction alone is too uncertain and needs the sampled token as input. A gist latent would plausibly act as an extra conditioning signal for an EAGLE-like drafter rather than a replacement for token feedback.
- The lossy semantic-acceptance methods (Lookahead Reasoning, SemanticSpec, FLy) show that sentence- or step-level drafting can give 2x to 2.7x on 32B reasoning models. A latent gist could be used to pre-filter or rank whole-step drafts before verification, keeping exact verification for a lossless variant.

### Gaps
- I did not find any paper that conditions a multi-token or speculative drafter on a forecast latent of the target's future (for example a predicted future sentence embedding). This may exist under different terminology.
- EAGLE-3, Medusa, HASS, DLoop and Carryover details were not verified from primary sources in this session.
- SentenceVAE model sizes and the exact baseline for its speedup were not read.

## Q4. Evidence on how far ahead a learned latent can usefully predict text (horizon vs accuracy)

### Takeaway
Direct horizon curves are scarce and short. A single GPT-J-6B hidden state can match the model's own token two to four positions ahead with at most about 44% to 48% top-1 agreement using a learned-prompt probe, and simple linear probes fall from 29% to 16% across three extra positions. NextLat reports the strongest linear-probe signal out to 20 tokens on TinyStories, and CALM shows that a 4-token chunk latent works while 8 tokens degrades at sub-2B scale.

### Cited Findings
- Future Lens, arXiv 2311.04897, Pal et al., CoNLL 2023, GPT-J-6B [full]. Precision@1 agreement with the model's own future token, where N=1 is the token after the immediate next. Linear on hidden state 29.2 / 19.0 / 15.8 for N=1/2/3. Linear in vocab space 27.5 / 19.4 / 14.7. Fixed prompt 20.8 / 30.0 / 36.5. Learned prompt 48.4 / 43.7 / 46.9. Bigram baseline 20.1 at N=1. Learned prompt peaks at middle layers (layer 14 used for analysis). N above 3 not tested. — [ar5iv 2311.04897](https://ar5iv.labs.arxiv.org/html/2311.04897), [ACL Anthology](https://aclanthology.org/2023.conll-1.37)
- NextLat on TinyStories. Linear probes show NextLat matches GPT on next-token prediction and keeps the strongest predictive signal up to 20 tokens ahead, while MTP and JTP degrade sharply with offset. Countdown accuracy rises with rollout horizon d (1 to 8) for all multi-token methods. — [NextLat](https://arxiv.org/html/2511.05963v1) [full, exact probe numbers not extracted]
- CALM. Chunk size K=1 to 2 nearly halves compute with marginal loss, K=4 beats the baseline frontier, K=8 drops noticeably, which the authors attribute to model capacity. Latent dim 128 was best for K=4 (32 too compact). — [arXiv 2510.27688](https://arxiv.org/html/2510.27688v1) [full]
- Apple MTP finding. Correct future token sequences often appear within the top-200 logits of a standard AR model, which is weak but useful evidence that future information is present before any fine-tuning. — [arXiv 2507.11851](https://arxiv.org/html/2507.11851v1) [snippet]
- Gloeckle et al. found 4-token prediction gave up to 3x inference speedup, implying usable acceptance out to about 4 tokens at 7B to 13B scale. — [arXiv 2404.19737](https://arxiv.org/abs/2404.19737) [abstract]
- Lookahead Reasoning notes that the chance that a long run of drafted tokens is fully correct drops quickly with length, which motivates step-level drafting. — [arXiv 2506.19830](https://arxiv.org/abs/2506.19830v1) [snippet]

### Inferences
- For exact token drafting, the evidence supports a useful horizon of roughly 2 to 4 tokens from a single latent at 6B to 13B scale. Beyond that, precise token recovery from one vector is weak.
- For coarse "gist" guidance (topic, structure, step type), NextLat's 20-token probe signal and the sentence- and paragraph-level planners suggest longer horizons are informative, but only as distributional or semantic information, not exact tokens. This argues for the drafter latent to target a semantic or chunk summary of the next 8 to 32 tokens and leave exact tokens to the token drafter and verifier.
- The non-monotonic Future Lens learned-prompt numbers (46.9 at N=3 above 43.7 at N=2) suggest the probe partly exploits frequent multi-token patterns, so these figures likely overstate general predictability.

### Gaps
- No study found reports horizon vs accuracy for a learned latent dynamics model on a multi-billion-parameter LM beyond 4 tokens.
- NextLat's exact per-offset probe accuracies and Medusa or EAGLE per-position acceptance rates were not extracted in this session.
- No paper found measures how well a predicted future latent (as opposed to the true hidden state) supports drafting, which is the quantity the target design depends on.
