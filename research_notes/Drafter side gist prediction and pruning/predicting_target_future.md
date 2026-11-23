# Predicting a target model's future outputs, tokens and internal states (evidence base for drafter-side gist prediction)

Read-depth legend: [full] = full text or HTML fetched and numbers extracted. [abstract] = only abstract/metadata read. [snippet] = only seen in search-result summaries, not opened. Treat [snippet] items as leads to verify.

## Q1. Future-token and future-content decoding from hidden states (how far ahead, how it scales)

### Takeaway
A model's own hidden state at one position carries decodable information about tokens 2 to 4 positions ahead (about 44 to 48% top-1 with a learned decoder on GPT-J-6B) and about paragraph-level gist roughly equivalent to 5 to 10 tokens of real lookahead (Llama-3.2-3B). Prompt-end activations also encode global response attributes (length, structure, content choices), and this signal grows with model size within families. But decodability does not imply the model causally uses the plan, and the strongest signal sits at boundary tokens (newline, "\n\n") and in middle layers (about 60 to 80% depth).

### Cited Findings
- Future Lens [full]: GPT-J-6B, predicting the token N positions beyond the next from a single hidden state. Best Precision@1 for N=1/2/3: bigram baseline 20.1 (N=1 only), direct vocab projection 27.5/19.4/14.7, linear hidden-state approximation 29.2/19.0/15.8, fixed-prompt causal intervention 20.8/30.0/36.5, learned-prompt causal intervention 48.4/43.7/46.9. Learned prompt peaks around layer 14 (middle layers). So a linear map is weak, and a learned soft-prompt decoder that re-runs the model is much stronger — [Pal et al. 2023, arXiv 2311.04897](https://ar5iv.labs.arxiv.org/html/2311.04897); [abstract](https://arxiv.org/abs/2311.04897)
- Wu, Morris and Levine (COLM 2024) [abstract]: with "myopic" training (no gradients to past timesteps), synthetic tasks show clear "pre-caching" (computing features for future use), while natural language modeling looks more like "breadcrumbs" (features useful now happen to help later). Pre-caching increases with model scale. No numbers read — [arXiv 2404.00859](https://arxiv.org/abs/2404.00859)
- Anthropic "On the Biology of a Large Language Model" (Claude 3.5 Haiku, 2025) [full, first ~55% of page]: the rhyme target word is planned at the newline token before the line starts. Planned-word features were seen in about half the poems investigated. Injecting planned-word features ("rabbit", "green") at the newline made the model end the line with the injected word in 70% of 25 sampled poems, and the model rewrote the intermediate words to reach the target. Steering only worked at the newline planning position. The model holds multiple candidate targets at once. Attribution graphs gave insight for about a quarter of prompts tried — [Transformer Circuits 2025](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)
- "Where's the Plan?" (Ma and Rui, May 2026, ICML 2026 Mech Interp workshop) [abstract; ID verified as arXiv 2605.07984, v2 11 Jun 2026]: rhyming couplets across Qwen3, Gemma-3, Llama-3 at more than ten scales. Future-rhyme information is linearly decodable at the line boundary and the probe signal strengthens with scale in all three families. But activation patching shows only Gemma-3-27B causally relies on it (the causal driver shifts to the line boundary around layer 30). Five attention heads recover about 90% of rhyme-routing capacity at the newline. Other models show near-zero causal effect at the boundary despite strong probe signal — [arXiv 2605.07984](https://arxiv.org/abs/2605.07984)
- Emergent Response Planning in LLMs (Dong et al., ICML 2025) [full HTML, but exact numbers are only in figures]: one-hidden-layer MLP probes on prompt-end hidden states (before any response token) predict response length and reasoning-step count (Spearman/Kendall/Pearson), character choice in stories (4-class F1), multiple-choice answer (5-class F1), answer confidence and factual consistency (binary F1). Models: Llama-2-7B-Chat, Llama-3-8B-Instruct, Mistral-7B-Instruct, Qwen2-7B-Instruct and base versions. Probe width saturates at 128 or less (structure about 16, content about 32, behavior about 8 hidden units), i.e. the plan is low-dimensional. Structure attributes peak in middle layers, content in later layers. Within a family, larger instruct models plan more strongly (Llama-2 7B/13B/70B, Llama-3 8B/70B, Qwen-2 7B/72B, Qwen-2.5 1.5B/32B/72B), but the trend does not hold across families. Probe accuracy is high at the start of the response, dips mid-response and recovers near the end. Probes are far more accurate than the model's own verbalized prediction of its length or character choice — [arXiv 2502.06258](https://arxiv.org/html/2502.06258v3)
- ParaScopes (Pochinkov et al., Oct 2025) [full HTML]: decodes the next paragraph from residual-stream activations of Llama-3.2-3B-Instruct. A linear map from residual diffs of the final 12 layers into a SONAR text-autoencoder embedding (trained on 100k samples) gets cosine 0.55 to the true next paragraph (Qwen3-Embedding-0.6B) vs random 0.20, "cheat-5" (5 real future tokens given) 0.50, regenerating from the same prompt 0.82. Subject match (judge score >=2): 76% for the embedding-map ParaScope vs 51% for cheat-10 and 43% for feeding the residual vector into a blank copy of the model. Fine details are poorly recovered (3% at score >=2 vs 17% for cheat-5). A linear map and an MLP perform nearly the same (cosine 0.51 vs 0.50 with all-mpnet). Middle layers at 60 to 80% depth carry most of the signal. Plans form or refresh at "\n\n" (similarity to next topic: 0.12 before transition, 0.48 at, 0.65 after). A document-level outline probe was weak (subject match 54 to 61%, detail 24%), suggesting limited long-range planning in a 3B model. Headline: about 5 to 10 tokens of lookahead-equivalent — [arXiv 2511.00180](https://arxiv.org/html/2511.00180v1); earlier write-up [LessWrong](https://www.lesswrong.com/posts/9NqgYesCutErskdmu/parascopes-do-language-models-plan-the-upcoming-paragraph)
- Your LLM Knows the Future (Samragh et al., Apple, Jul 2025) [abstract]: argues vanilla AR LLMs contain substantial future-token information. Masked-input multi-token prediction with gated LoRA gives nearly 5x faster code/math and about 2.5x faster chat with no quality loss — [arXiv 2507.11851](https://arxiv.org/abs/2507.11851)
- FIRP (2024) [snippet]: predicts intermediate hidden states of future tokens and decodes them for speculative decoding — [arXiv 2410.20488](https://www.arxiv.org/pdf/2410.20488)

### Inferences
- For a gist drafter, the evidence supports predicting a coarse semantic summary (topic, subject, structure, length) of the next paragraph, not lexical detail. ParaScopes shows a linear map into a sentence-autoencoder space already captures subject-level gist better than 10 real tokens of context.
- The best read-out points are boundary tokens (newline, paragraph break, prompt end) and middle-to-late-middle layers of the target. A drafter-side gist head should be conditioned on target states at those positions rather than every token.
- The plan is low-dimensional (MLP probes saturate at 8 to 128 units), which favours a compact gist vector.
- "Decodable" is weaker than "used". For a drafter this matters less (we only need prediction, not a causal plan), but it means target states encode information about the future that the target may not follow strictly, so gist targets should come from the target's actual sampled continuations.
- Horizon evidence: token-level decoding works for 2 to 4 tokens; paragraph-level semantic gist (tens to around 100 tokens) works at coarse grain; document-level outline (hundreds of tokens) is weak in a 3B model. Scaling results suggest it improves with target size.

### Gaps
- No found paper gives a clean scaling law for k-ahead token decoding accuracy vs model size. Future Lens is GPT-J-6B only. Wu et al. numbers not read.
- Anthropic follow-up planning work after the 2025 biology paper was not searched.
- No exact probe correlation values for Dong et al. (only in figures).

## Q2. Predicting a model's behaviour or outputs from its activations or from another model

### Takeaway
Probes on the model's own activations predict future behaviour better than either the model's verbal self-report or another model trained on its outputs. Self-prediction beats cross-prediction by about 13 to 17 points on short properties, but neither works for long-output properties such as length or sentiment of a whole review. Small models can still steer or approximate large-model outputs at the logit level (proxy-tuning closes 88% of a tuning gap).

### Cited Findings
- Looking Inward (Binder et al., Oct 2024) [full HTML, first ~66%]: on held-out tasks, Llama-70B self-prediction 48.5% vs GPT-4o cross-predicting Llama-70B 31.8% (35.2% plateau in a data-scaling appendix). GPT-4o self-prediction 49.4% vs Llama-70B cross-predicting GPT-4o 36.6%. Self-prediction training raised GPT-4o from 32.6% to 49.4%. Cross-prediction plateaus around 10,000 training samples, below self-prediction. Predicted above baseline: second character of output, even/odd, option in a set, starting with a vowel. Not above baseline: third word, review sentiment, main character name, response length. Authors suggest self-simulation fails for outputs too long to simulate in one forward pass — [arXiv 2410.13787](https://arxiv.org/html/2410.13787)
- Dong et al.: probes on hidden states are far more accurate than verbalized self-prediction for response length and character choice — [arXiv 2502.06258](https://arxiv.org/html/2502.06258v3)
- Output-length prediction from the serving model's own hidden states: EGTP (Xie et al., ICLR 2026) [full HTML] pools final-layer hidden states weighted by next-token entropy and predicts 20 length bins plus regression. 29.16% average MAE reduction over the best baseline, including BERT/DistilBERT/OPT-based separate predictors (SSJF, S3, LTR-C) and TRAIL (MLP on LLM embeddings). Example: Qwen2.5-3B ForeLen MAE 110.75 vs TRAIL 146.42. Progressive prediction during generation cuts MAE from 95.24 to 80.85 on the RL task. About 0.66 ms and 5 to 7 MB vs 2 to 12 ms and 238 to 270 MB for auxiliary models. Models: Qwen2.5 0.5B to 7B, Llama-3.2 1B/3B. On Qwen2.5-0.5B, layer 12 (MAE 70.26) was only slightly better than final layer (73.93) — [arXiv 2602.11812](https://arxiv.org/html/2602.11812v2)
- TRAIL / "Don't Stop Me Now" (Shahout et al., 2024) [snippet]: predicts output length from intermediate-layer embeddings into k=10 bins — [arXiv 2410.01035](https://arxiv.org/pdf/2410.01035). An ACL SRW 2025 paper uses Llama-3-8B-Instruct layers 8 to 15 for remaining-length regression [snippet] — [ACL SRW 2025](https://preview.aclanthology.org/landing_page/2025.acl-srw.61.pdf). "Robust Length Prediction" treats the target as a distribution, using last-layer states [snippet] — [arXiv 2604.07931](https://arxiv.org/pdf/2604.07931)
- Predicting reasoning outcome early [snippets only]: linear probes after the first few reasoning tokens predict final correctness with AUC above 0.8 on MATH-like data — [arXiv 2511.14773](https://arxiv.org/html/2511.14773v1). Probes predict zero-shot CoT success before completion at 60 to 76.4% accuracy — [arXiv 2505.24362](https://arxiv.org/html/2505.24362v2). Reasoning models' hidden states predict correctness of intermediate answers before they are written, usable for early exit — [arXiv 2504.05419](https://arxiv.org/abs/2504.05419v1). Activation probes decode answers from partial traces, early exit saves 80% of tokens on MMLU-Redux and 30% on GPQA-Diamond — [arXiv 2603.05488](https://arxiv.org/html/2603.05488v3)
- Proxy-tuning (Liu et al., 2024) [abstract]: applying the logit difference of tuned vs untuned 7B proxies to Llama2-70B closes 88% of the gap to the true Llama2-70B-chat across knowledge, reasoning and safety benchmarks — [arXiv 2401.08565](https://arxiv.org/abs/2401.08565)

### Inferences
- A drafter that sees the target's activations (white-box) should beat a drafter that only sees the target's text outputs (black-box cross-prediction). Binder et al. show text-only cross-prediction plateaus early.
- Length and coarse outcome are reliably predictable from target states at prompt end, and progressive re-prediction improves accuracy. A gist head should be refreshed periodically (e.g. at boundaries), not computed once.
- Small separate models (BERT-class) are worse length predictors than tiny heads on the target's states, which argues for feeding target states into the drafter rather than relying on the drafter's own states.

### Gaps
- Exact mode-baseline numbers for Binder et al. are only in figures.
- I did not find work on tuned/logit-lens translators trained on one model and applied to another of a different size. Weak-to-strong generalization (Burns et al.) was not re-read.
- Early-outcome papers were not opened, so their models and horizons are unverified.

## Q3. Training small models to predict a larger model's hidden states

### Takeaway
Within a model family, an affine map between residual streams of different sizes is nearly lossless on round trips but small-to-large replacement is bottlenecked by the small model (+27 to +39% CE loss when the small model's stream drives the large model). So a small model can recover much of a larger model's representation geometry, but not its full content.

### Cited Findings
- Model stitching (Chen, Merullo, Stolfo, Pavlick, NeurIPS 2025) [full HTML]: affine maps between residual streams of within-family models. Next-token CE on OpenWebText: Pythia 70M (3.60) -> 160M (3.14) stitched gives 3.99 (+27%); GPT-2 small (3.07) -> medium (2.77) gives 3.08 (+11%); Gemma-2 2B (2.52) -> 9B (2.36) gives 3.28 (+39%). Round trips A->B->A add under 1% loss. Zero-shot SAE transfer raises fraction of unexplained variance (e.g. Gemma-2 2B->9B FUV 0.21 -> 0.42). Initializing a large-model SAE from a transferred small-model SAE saves about 30 to 50% FLOPs. Steering vectors transfer bimodally (en-fr 2B->9B: 0.69 vs ground truth 0.09, but en-hu 9B->2B: 0.06 vs 0.72) — [arXiv 2506.06609](https://arxiv.org/html/2506.06609v3)

### Inferences
- A 0.5 to 2B drafter in the same family as the target likely shares enough representation geometry that a linear or affine adapter from target states into drafter space (or the reverse) is viable. Expect coarse semantic features to transfer and fine detail to be lost, which suits gist prediction.
- The small-to-large bottleneck means the drafter should consume target states as input where possible, rather than try to regenerate them from its own states.

### Gaps
- Classic hidden-state distillation (TinyBERT, MiniLM, layer-mapping distillation for decoder LLMs) was not researched in this pass. No numbers on how much of a 7B to 70B model's hidden state a 0.5 to 1B model can regress.
- No found paper trains a small model to predict a large model's future hidden states (as opposed to current states). EAGLE-style feature-level drafters are the closest and are covered elsewhere.

## Q4. Response-level gist prediction (embedding or summary of the whole response before generating it)

### Takeaway
Direct evidence exists that a linear map from residual activations into a sentence-autoencoder embedding predicts the next paragraph's gist (cosine 0.55 vs 0.82 ceiling and 0.20 random, beating 5 real tokens of lookahead), and that prompt-end activations predict whole-response attributes. Document-level outlines are weakly predictable in small models.

### Cited Findings
- ParaScopes linear map into SONAR embedding: cosine 0.55, subject match 76%, beating cheat-10 (51%) on subject, losing on fine details — [arXiv 2511.00180](https://arxiv.org/html/2511.00180v1)
- ParaScopes outline probe (document-level, 4 to 5 bullets): subject match 54 to 61%, detail 24%, read as limited long-range planning in Llama-3.2-3B — [arXiv 2511.00180](https://arxiv.org/html/2511.00180v1)
- Dong et al. prompt-end probes predict whole-response length, reasoning steps, story character choice and MC answer, and the signal scales with model size within family — [arXiv 2502.06258](https://arxiv.org/html/2502.06258v3)
- Self-prediction of whole-response properties (length, sentiment of a written review) fails to beat baseline when done by the model verbally — [arXiv 2410.13787](https://arxiv.org/html/2410.13787)

### Inferences
- Target gist for a drafter could be defined as a SONAR-style or small sentence-embedding of the next 16 to around 128 tokens. ParaScopes gives a concrete accuracy reference (cosine about 0.5 to 0.55 from a 3B model's states), and larger targets probably encode more given within-family scaling.
- The ceiling (regeneration from the same prompt, cosine 0.82) reflects sampling variance. Gist targets should be compared against this ceiling, not against 1.0.
- Gist beyond one paragraph (hundreds of tokens) has weak evidence so far.

### Gaps
- No found work trains a separate small model to predict a large model's response embedding from the large model's states. Meta's Large Concept Model (sentence-embedding prediction) was not checked in this pass.
- No accuracy-vs-horizon curve (16, 32, 64, 128, 256 tokens) exists in the sources read.
