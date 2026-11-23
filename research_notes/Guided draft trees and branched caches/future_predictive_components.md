# Locating LLM components that predict the model's own future output (for a Qwen3.5-9B gist drafter)

Read-depth legend. [full-HTML] means the arXiv HTML body was read via fetch (summarised by a fetch model, so check key numbers before citing in a paper). [abstract] means only the abs page was read. [snippet] means only search-result text was seen.

## Where does future/plan information live (layers, heads, positions), and does it scale?

### Takeaway
Future information is reliably *decodable* at boundary tokens (end of prompt, "\n", "\n\n") in middle-to-late layers (roughly 50-80% depth), and decodability grows with scale. *Causal use* of that boundary encoding is much rarer. In rhyme planning only Gemma-3-27B (and Claude 3.5 Haiku) actually route the plan through the newline. Qwen3 models at every size tested (0.6B-32B) instead condition on the rhyme word directly, with near-zero causal effect at the newline despite strong probe signal.

### Cited Findings
- "Where's the Plan?" (Ma and Rui, arXiv 2605.07984, ICML 2026 MechInterp workshop) tests rhyming couplets on Qwen3, Gemma-3 and Llama-3 at more than ten scales with linear probes, activation patching and two-stage path patching. Future-rhyme info is linearly decodable at the line-boundary newline in all three families, and probe signal strengthens with scale. [full-HTML] — [arXiv 2605.07984](https://arxiv.org/html/2605.07984)
- Qwen3 patching results (corrupt-rhyme rate, all-layer patch). At the last word it is 1% (0.6B), 15% (1.7B), 42% (4B), 53% (8B), 63% (14B) and 76% (32B). At the newline it is 7%, 2%, 7%, 2%, 7% and 1%. Per-layer newline peak CI upper bounds are at most 0.21 for every Qwen3 size. Last-word and newline steering show the same pattern. The newline-minus-first-generated-position probe gap includes zero for Qwen3 0.6B-8B and is clearly nonzero only at 32B. Qwen3 tokenises ",\n" as one token, so the last word is at position -1. [full-HTML] — [arXiv 2605.07984](https://arxiv.org/html/2605.07984)
- Gemma-3-27B "handoff". Last-word patching drops sharply around layer 30 while newline patching rises to a peak of 0.63 [0.48, 0.78] at layer 33 (all-layer newline 67%, last-word 85%). Five heads (30.4, 28.14, 28.15, 30.5, 28.29) recover a corrupt-rhyme rate of 57%, about 90% of the full-residual 63%. Naive top-5 head patching reaches 46% (about 73%). Single heads show no measurable signal, and top-5 MLP patches at the newline have zero effect. Smaller Gemma-3 sizes show no newline effect. [full-HTML] — [arXiv 2605.07984](https://arxiv.org/html/2605.07984)
- Data cost of that method. Probes use 1,200 synthetic couplets (10 epochs of AdamW). Patching uses 5 prompt pairs x 20 samples per cell (N=100), which the authors say is far cheaper than steering or transcoder methods. No GPU-hour figures are given. [full-HTML] — [arXiv 2605.07984](https://arxiv.org/html/2605.07984)
- Anthropic "On the Biology of a LLM" (Claude 3.5 Haiku, 2025) uses attribution graphs over a roughly 30M-feature cross-layer transcoder. Planned rhyme-word features (for example "rabbit" or "habit") are active at the newline before the line is written. Several candidates are held at once, and backward planning shapes the intermediate words. Injecting a planned-word feature at the newline made the line end in that word in 70% of 25 poems. Steering at other positions had no effect. Planned-word features appeared in only about half of the poems examined, and the method gave useful insight on roughly a quarter of prompts. [full-HTML, first ~100k chars] — [Transformer Circuits](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)
- ParaScopes (Pochinkov et al., arXiv 2511.00180) reads the residual stream at "\n\n" paragraph boundaries in Llama-3.2-3B-Instruct, with token-wise analysis on Gemma-2-9B. A linear map from the last 12 layers' sublayer outputs to SONAR embeddings ("TAE ParaScope") gets 76% subject match (cheat-10 51%, random 4%) and cosine 0.55 (cheat-5 0.50, regenerated 0.82). Overall the decoders are equivalent to about 5 tokens of revealed future context. Fine details are weak (details>=2 match 3% for TAE, 15% for continuation-patching). Layers at about 60-80% depth matter most (cosine delta 0.15-0.25 for layers ~25-35 of 42, under 0.05 for layers 0-25). The paper reads planning as fairly myopic, updating at paragraph boundaries (similarity to the next topic 0.12 before the transition and 0.48 at it). Whole-document outline decoding is weak (24% minimal detail match). [full-HTML] — [arXiv 2511.00180](https://arxiv.org/html/2511.00180)
- Emergent Response Planning (Dong et al., arXiv 2502.06258, ICML 2025) probes prompt-final representations of Llama-2-7B, Llama-3-8B, Mistral-7B and Qwen2-7B (base and instruct). Scaling runs cover Llama-2-chat 7/13/70B, Llama-3-Instruct 8/70B, Qwen-2-Instruct 7/72B and Qwen-2.5-Instruct 1.5/32/72B. Probes predict response length, reasoning steps, story character choice, MC answer, confidence and factual consistency. Structure attributes peak in middle layers, content attributes in later layers, and behaviour attributes are flat after the first few layers. Instruct models plan better than base models. Larger models plan better within a family but not across families. During generation, accuracy for confidence and character choice is high early, dips mid-response and rises near the end. Exact numbers appear only in figures. [full-HTML] — [arXiv 2502.06258](https://arxiv.org/html/2502.06258)
- Wu, Morris and Levine (arXiv 2404.00859, COLM 2024) compare "pre-caching" (computing features at t mainly for future use) with "breadcrumbs" (features useful now happen to help later) using myopic training that blocks gradients to past positions. GPT-2-small shows a small myopia gap (val CE 3.28 vanilla vs 3.40 myopic), consistent with breadcrumbs. On Pythia 70M-2.8B the gap grows with size and becomes non-negligible by 2.8B, so pre-caching increases with scale. Synthetic tasks show clear pre-caching. [full-HTML] — [arXiv 2404.00859](https://arxiv.org/html/2404.00859)
- Future Lens (Pal et al., arXiv 2311.04897, CoNLL 2023, GPT-J-6B) reports Precision@1 from a single hidden state for tokens N+1/N+2/N+3. A learned 10-token soft prompt gets 48.4/43.7/46.9, a fixed-prompt transplant 20.8/30.0/36.5, and a linear map to the future hidden state 29.2/19.0/15.8 (bigram baseline 20.1 at N+1). The learned prompt peaks at middle layers (layer 14 of 28 in the token-type analysis). [full-HTML] — [arXiv 2311.04897](https://arxiv.org/html/2311.04897)
- Remaining length. "How Much is Left?" (Merzouk et al., arXiv 2607.05316) finds total response length linearly decodable from the prompt's last hidden state in three 7-8B models, beating a constant-median baseline in every cell. The authors state it is "decodable, not necessarily used causally". [abstract] — [arXiv 2607.05316](https://arxiv.org/abs/2607.05316)
- Length from hidden states is also used for scheduling in "Predicting LLM Output Length via Entropy-Guided Representations" (arXiv 2602.11812, ICLR 2026, ForeLen benchmark). [snippet] — [arXiv 2602.11812](https://arxiv.org/html/2602.11812)

### Inferences
- For Qwen specifically, the best prior is that *probes will find gist signal at boundary tokens* but Qwen may not causally use it (2605.07984 found no newline handoff in Qwen3 up to 32B). For a drafter this is fine, since a drafter only needs decodability, not causal use. It means patching results should not be used to choose the readout site for Qwen.
- Candidate readout positions to sweep. (1) The last prompt token or the assistant-turn start token. (2) Each "\n" and "\n\n" during generation, since ParaScopes and ERP both show plans refreshing at boundaries. (3) The current token, for short horizons.
- Candidate layers. Middle to late, about 50-80% depth, which is roughly layers 16-26 of Qwen3.5-9B's 32. Structure and length signals are likely to peak earlier than content and topic signals (per ERP).
- Realistic ceiling for 16-128 token gist from one residual vector is "topic or subject right, details wrong" (ParaScopes is about 5 tokens of context equivalent at 3B). Scale helps (ERP, Wu et al.), so 9B should be somewhat better than 3B, but this is not measured.

### Gaps
- No found study measures future-gist decodability or plan use specifically in Qwen3.5 (hybrid DeltaNet). The planning study on Qwen3 is rhyme-only.
- 2605.07984 gives no numeric per-layer probe accuracies for Qwen3.
- ERP exact accuracy and Spearman numbers are in figures only and were not extracted.

## Recurrent-state (DeltaNet / Mamba) models and future information

### Takeaway
There is almost no direct work on future-token probing in recurrent or hybrid models. Early 2026 work shows the GDN recurrent state can be probed, decomposed with SAEs and steered, and one paper reports that on Qwen3.5-4B the recurrent state rather than the KV cache controls continuation mode.

### Cited Findings
- WriteSAE (Young, arXiv 2605.12770) trains SAEs on the matrix-valued recurrent state, covering Gated DeltaNet, Mamba-2 and RWKV-7. SAE atoms beat deleting the native write on 92.4% of positions, and a closed form using forget gate, read query and output embedding predicts the logit change with R^2 = 0.98 for GDN. It reports cache-level steering in a recurrent layer. No Qwen model is named on the abstract page. [abstract] — [arXiv 2605.12770](https://arxiv.org/abs/2605.12770)
- An EMNLP 2026 Findings paper (split-prefill, keeping only the KV cache or only the recurrent state) reportedly finds on Qwen3.5-4B that the recurrent state controls the continuation's target mode even when the KV cache holds the instruction. [snippet only, the PDF could not be parsed] — [ACL Anthology 2026.findings-emnlp.2061](https://aclanthology.org/2026.findings-emnlp.2061.pdf)
- S0 Tuning (arXiv 2604.01168) adapts hybrid recurrent-attention models via the initial state and includes an appendix probe of whether the recurrent state encodes task info more strongly than the residual stream. [snippet] — [arXiv 2604.01168](https://arxiv.org/pdf/2604.01168)
- State dynamics of Qwen3-Next's linear-attention layers show unbounded state growth that can interfere over long contexts (arXiv 2602.02195). [snippet] — [arXiv 2602.02195](https://arxiv.org/pdf/2602.02195)
- Qwen3-Next uses a 3:1 pattern of Gated DeltaNet to gated attention. [snippet] — [Raschka architecture gallery](https://sebastianraschka.com/llm-architecture-gallery/hybrid-attention/)

### Inferences
- In Qwen3.5-9B, two kinds of site are worth probing beyond the residual stream. One is the DeltaNet state matrices at the end of the prompt, a compressed summary already shaped for future reads. The other is the outputs of the 8 full-attention layers (every 4th layer, so probably layers 3, 7, ..., 31; check `layer_types` in config.json). Heads in full-attention layers are the only "head-level" attention sites comparable to the Gemma 28.x/30.x heads.
- The split-prefill result suggests the recurrent state may carry the "mode or plan" more than the KV cache in Qwen3.5. If confirmed, readouts from DeltaNet states could rival residual readouts for gist. This is a hypothesis to test.

### Gaps
- No paper found that probes DeltaNet or Mamba state for future tokens, length or topic. The search found only adjacent work such as "Lost in State Space" (arXiv 2605.00253), which probes Mamba-130M for classification ([arXiv 2605.00253](https://arxiv.org/pdf/2605.00253)).
- The split-prefill paper's title, authors and numbers were not verified.

## Methods for searching components, and what fits one 12 GB GPU with a 4-bit 9B model

### Takeaway
The cheap, high-value pipeline is: cache activations once, sweep linear or ridge probes over (layer x position), then rank SAE features by predictive power at the best sites, then optionally run patching on a few hundred pairs. Attribution graphs with transcoders are not available for Qwen3.5-9B and are too heavy for 12 GB.

### Cited Findings
- Activation patching and path patching localised a 5-head plan circuit with only N=100 samples per cell, and the authors stress that this needs far less data than transcoder methods. [full-HTML] — [arXiv 2605.07984](https://arxiv.org/html/2605.07984)
- Attribution graphs need a trained cross-layer transcoder (about 30M features for Claude 3.5 Haiku). Attention patterns are frozen, so attention-mediated mechanisms may be missed, error nodes are uninterpretable, and interventions sometimes needed unusually strong perturbations. [full-HTML] — [Transformer Circuits](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)
- circuit-tracer (safety-research, BlackboxNLP 2025) is an open pipeline for finding, visualising and intervening on feature circuits, tested on models up to 14B. Public Qwen transcoders exist for Qwen3-4B (Hanna and Piotrowski, `mwhanna/qwen3-4b-transcoders`, 163,840 features per layer, on Neuronpedia) and Qwen3-0.6B/1.7B Base (BlueLight AI cross-layer transcoders). [snippet] — [ACL Anthology BlackboxNLP](https://preview.aclanthology.org/ingest-emnlp/2025.blackboxnlp-1.14.pdf); [Neuronpedia Qwen3-4B](https://www.neuronpedia.org/qwen3-4b/10-transcoder-hp); [LessWrong BlueLight CLTs](https://www.lesswrong.com/posts/cW9AdDm2DZtbXBnvt/cross-layer-transcoders-for-the-qwen3-llm-family)
- Training-free readouts. ParaScopes' "continuation" method patches the "\n\n" residual into a blank context and generates up to 128 tokens, and Future Lens's fixed-prompt transplant needs no training. A learned soft prompt (Future Lens) is the strongest single-state decoder at 48% P@1. [full-HTML] — [arXiv 2511.00180](https://arxiv.org/html/2511.00180); [arXiv 2311.04897](https://arxiv.org/html/2311.04897)
- Sublayer-diff inputs. ParaScopes feeds normalised MLP and attention sublayer outputs from the final 12 layers (24 vectors) into one linear map, a cheap way to pool across layers. [full-HTML] — [arXiv 2511.00180](https://arxiv.org/html/2511.00180)

### Inferences (practical guide, author's synthesis)
- Memory. A 4-bit 9B model is about 5-6 GB. One Qwen-Scope SAE layer is W_enc + W_dec at 65536x4096 each, about 268M params per matrix, about 1.07 GB per matrix in fp32 (about 0.54 GB in bf16). Load SAEs one or two layers at a time in bf16 and encode cached activations offline instead of hooking all 32 layers live.
- Step 1, cache. Run about 2-5k prompts and generate 128 tokens each. Save residuals at a handful of positions (end of prompt, every "\n" and "\n\n", every 16th token) for all 32 layers in fp16. Each vector is 8 KB, so 5k prompts x 10 positions x 32 layers is about 13 GB on disk, which is manageable. Also save full-attention layer per-head outputs (o_proj inputs) for the 8 attention layers, and optionally the final DeltaNet states at end of prompt.
- Step 2, targets. These are a bag-of-tokens over the next 16/32/64/128 tokens (multi-label BCE or KL to the empirical distribution), a sentence embedding of the continuation (as in ParaScopes/SONAR or any small embedder), remaining length (log, regression), and structural labels (list, code, prose).
- Step 3, sweep. Ridge or logistic probes per (layer, position) with held-out prompts. This takes minutes on CPU or GPU. Plot a heatmap and expect a peak around 50-80% depth.
- Step 4, SAE feature ranking at the top 2-3 layers. Encode with Qwen-Scope (TopK=100) and rank features by mutual information or AUROC against each target, or fit an L1-sparse probe on the 65k SAE codes. Compare against the dense probe to see how much predictive power survives sparsification.
- Step 5, heads. For the 8 attention layers, run per-head linear probes on head outputs, and do attribution (gradient x activation of the probe target, or of the logit of a future token, through o_proj). This is the Qwen analogue of finding Gemma's 28.x/30.x heads.
- Step 6, causal check (optional for a drafter). Use cheap activation patching between prompt pairs with different gists at the top sites. It runs in 4-bit inference and costs only forward passes. Expect weak causal effects in Qwen per 2605.07984, which does not invalidate the readout.
- Skip for 12 GB. Training transcoders or crosscoders on 9B, and circuit-tracer on Qwen3.5-9B (no transcoders published, and the hybrid DeltaNet layers are unlikely to be supported). Qwen3-4B with the mwhanna transcoders is a feasible dense proxy for attribution-graph exploration.

### Gaps
- No "future-token SAE" (an SAE trained to reconstruct or predict future activations) was found in this pass. Its existence is unverified.
- Whether circuit-tracer supports Qwen3.5 hybrid layers was not checked in its README.

## Qwen-specific interpretability (Qwen-Scope SAEs and planning in Qwen)

### Takeaway
Qwen-Scope (arXiv 2605.11887) releases TopK residual-stream SAEs for every layer of 7 Qwen3 and Qwen3.5 backbones, including all 32 layers of Qwen3.5-9B-Base (65,536 features, k=100). They are trained on in-house pretraining data, and no FVU or CE-delta numbers were visible. Planning evidence in Qwen is mixed. Qwen2/2.5 prompt states encode response attributes (ERP), but Qwen3 shows no causal newline plan.

### Cited Findings
- Model card for SAE-Res-Qwen3.5-9B-Base-W64K-L0_100. It hooks the residual stream for layers 0-31 with one file per layer (`layer{n}.sae.pt`). It has hidden 4096, width 65536 (16x), TopK with exactly 100 active features, and a .pt dict holding `W_enc` (65536x4096), `W_dec` (4096x65536), `b_enc` and `b_dec`. The loading example uses HF Transformers plus a forward hook, computes pre-activations, keeps the top 100 values and zeros the rest. There is a Gradio demo `app.py`. It is under the Qwen license, with a use restriction against non-scientific purposes. It is part of a 16-item Qwen-Scope collection. — [HF model card](https://huggingface.co/Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100)
- Qwen-Scope paper (arXiv 2605.11887). It has 14 SAE groups across Qwen3-1.7B, Qwen3-8B, Qwen3.5-2B, Qwen3.5-9B, Qwen3.5-27B (instruct variant), Qwen3-30B-A3B and Qwen3.5-35B-A3B. It uses TopK with k=50 or 100, 16x width for dense models (up to 64x for MoE), an auxiliary dead-feature loss of weight 1/32 and filtering of extreme-norm activations. It reports "negligible" dead features. Use cases include steering (suppressing Chinese feature 6159 to stop code-switching), toxicity classification (F1 > 0.90 in English with a few features), benchmark redundancy analysis and SAE-guided SFT/RL. No planning or output-prediction application appeared in the first 100k chars. [full-HTML, partial] — [arXiv 2605.11887](https://arxiv.org/html/2605.11887)
- Qwen3 at 0.6B-32B shows no newline planning handoff (see the first section). The rhyme-word site becomes causal from about 4B. — [arXiv 2605.07984](https://arxiv.org/html/2605.07984)
- Qwen-2-Instruct (7B/72B) and Qwen-2.5-Instruct (1.5B/32B/72B) appear in ERP's scaling study, with planning improving with size within a family. — [arXiv 2502.06258](https://arxiv.org/html/2502.06258)

### Inferences
- The SAEs were trained on the *base* model. Applying them to Qwen3.5-9B-Instruct activations (and to a 4-bit quantised model) will add reconstruction error. Measure FVU on your own activations before relying on feature rankings.
- Because the L0 is fixed by TopK, feature-level probes see exactly 100 active features per token. This makes sparse probes easy (100 nonzeros) but may split gist concepts across many features.

### Gaps
- Qwen-Scope reconstruction-quality numbers (FVU, CE delta) and training-token counts were not found in the read portion. The final ~69k chars of the paper were unread.
- Whether Qwen-Scope SAEs are on Neuronpedia or loadable in SAELens directly was not verified. The card gives only a raw-PyTorch loader.

## Nonlinear readouts versus linear probes, and their risks

### Takeaway
For future-*token* prediction, a nonlinear or model-in-the-loop decoder beats linear by a lot (Future Lens learned prompt 48 vs linear 29 P@1 at N+1, and 47 vs 16 at N+3). For *gist* (paragraph embedding), an MLP gave no gain over linear in ParaScopes. The main risks are probes that learn the task from the prompt (memorisation or shortcuts) and "decodable but not used" signals.

### Cited Findings
- Future Lens. The learned soft-prompt decoder beats linear maps at every offset (48.4/43.7/46.9 vs 29.2/19.0/15.8 P@1), and the gap widens with horizon. [full-HTML] — [arXiv 2311.04897](https://arxiv.org/html/2311.04897)
- ParaScopes. The MLP variant was not better than the linear TAE map (Kendall tau = 0.82 between them). [full-HTML] — [arXiv 2511.00180](https://arxiv.org/html/2511.00180)
- ERP uses one-hidden-layer ReLU MLP probes (hidden sizes grid-searched 1-1024) and reports no linear baseline. Cross-dataset accuracy drops but stays above baseline. [full-HTML] — [arXiv 2502.06258](https://arxiv.org/html/2502.06258)
- Decodable but not used. Qwen3, Llama-3 and most Gemma-3 sizes have strong newline probe signal with near-zero causal effect. [full-HTML] — [arXiv 2605.07984](https://arxiv.org/html/2605.07984). The length-probe authors explicitly flag "decodable, not necessarily used causally". [abstract] — [arXiv 2607.05316](https://arxiv.org/abs/2607.05316)
- Correlational probing of hybrid GDN activations cannot separate redundant correlated signals from causally relevant ones, which motivates causal interventions on 0.8B/27B hybrid GDN models (Research Square preprint, unverified). [snippet] — [Research Square rs-11201314](https://www.researchsquare.com/article/rs-11201314)

### Inferences
- For a drafter, "decodable but not used" is acceptable. The real risk is shortcut learning. A probe at end of prompt can predict the topic of the continuation from the prompt's own topic. Always report a baseline probe on the prompt's mean-pooled embedding (or a small sentence embedder of the prompt) and a "cheat-K" baseline (reveal K true future tokens), as ParaScopes does.
- Use prompt-disjoint splits (and dataset-disjoint splits for generalisation), plus control tasks with random labels, to bound probe capacity.
- Attention-pooling probes over several positions or layers are a reasonable middle ground (ParaScopes pools 24 sublayer outputs linearly). Expect nonlinear gains mainly on token-level targets and little gain on embedding-level gist.
- Bag-of-future-tokens targets behave like Future Lens's harder regime. Consider a small learned decoder (soft prompt into the drafter, or a tiny MLP head) rather than a pure linear map.

### Gaps
- No direct head-to-head of linear versus MLP versus attention-pooling probes for 16-128 token gist on a 7-10B model was found.
- No quantified study of probe memorisation for future-output probes was located in this pass.
