# Interpretability evidence for look-ahead and planning in LLM hidden states

Scope note. Sources were read from arXiv abstract and HTML pages, transformer-circuits.pub and search snippets on 2026-11-22. Many papers put exact numbers only in figures, which the fetch tool could not read. Where that happened it is flagged. "Linear probe" and "learned decoder" (MLP probe, learned soft prompt, trained map into a text autoencoder) are kept separate throughout.

## Future Lens (Pal et al. 2023): predicting tokens N ahead from one hidden state

### Takeaway
On GPT-J-6B, a learned soft-prompt decoder applied to a single middle-layer hidden state recovers the model's own token at N=1 to 3 positions ahead with about 44 to 48% precision@1. Linear probes fall from 29% to 16% over the same horizon. Token-level look-ahead is real but short, and it needs a nonlinear decoder.

### Cited Findings
- Model is GPT-J-6B (28 layers). The task is predicting the model's own token at positions t+2 and beyond from one hidden state at position t. The abstract claims "more than 48% accuracy" at some layers ([arXiv 2311.04897](https://arxiv.org/abs/2311.04897)).
- Precision@1 from Table 2, at N=1 / N=2 / N=3 ([HTML](https://arxiv.org/html/2311.04897)):
  - Learned prompt causal intervention (learned decoder) scores 48.4 / 43.7 / 46.9.
  - Fixed prompt causal intervention (no training) scores 20.8 / 30.0 / 36.5.
  - Linear model from hidden state (linear probe) scores 29.2 / 19.0 / 15.8.
  - Linear model from vocab space (linear probe) scores 27.5 / 19.4 / 14.7.
- The bigram baseline is 20.1% at N=1. Surprisal for the learned prompt is 4.5 / 4.4 / 3.9 and for the linear HS model is 14.1 / 13.2 / 13.1 ([HTML](https://arxiv.org/html/2311.04897)).
- Learned-prompt accuracy "peaks at the middle-layer hidden states", with layer 14 of 28 the best ([HTML](https://arxiv.org/html/2311.04897)).
- Accuracy depends strongly on the model's confidence. At N=1 the learned prompt gets 26% when the model's confidence is 0 to 30%, 57% at 30 to 60%, 77% at 60 to 90% and 95% at 90 to 100% ([HTML](https://arxiv.org/html/2311.04897)).
- For named entities the learned prompt gets 44%, 42% and 37% at N=1, 2, 3 ([HTML](https://arxiv.org/html/2311.04897)).

### Inferences
- The future tokens that can be decoded are mostly the ones the model is already confident about, such as continuations of multi-token entities. Those are the same tokens a small drafter tends to get right anyway, so the extra signal for speculative decoding is smaller than the headline number.
- The paper only tests up to N=3. It says nothing about horizons of 8 or more.

### Gaps
- Table 2 has an unlabeled "Lens" column (97.0 / 97.0 / 98.0 / 85.7). Its meaning could not be confirmed, so it is not interpreted here.
- No comparison across model sizes. The paper uses GPT-J-6B only.

## "Do language models plan for future tokens?" (Wu et al. 2024): pre-caching vs breadcrumbs

### Takeaway
In natural-language modeling at GPT-2-small scale, most of what is useful for the future is a byproduct of next-token computation ("breadcrumbs"). Deliberate pre-caching shows up clearly in synthetic tasks, and its share grows with scale across Pythia 70M to 2.8B.

### Cited Findings
- Pre-caching means the model computes features at step t that do not help the current prediction but help later. Breadcrumbs means the features most useful now are already the ones useful later. The test uses myopic training, where gradients to past positions are blocked ([arXiv 2404.00859](https://arxiv.org/abs/2404.00859)).
- Models used ([HTML v2](https://arxiv.org/html/2404.00859v2)):
  - 2-layer transformers (d=128) on a synthetic task.
  - The GPT-2 small (124M) architecture trained from scratch on MS MARCO.
  - The Pythia suite from 14M to 2.8B, fine-tuned from pretrained checkpoints.
- On GPT-2 small, validation cross-entropy is 3.28 for the vanilla model and 3.40 for the myopic one, a myopia gap of 0.12 nats. The transformer bigram baseline is 5.33 ([HTML v2](https://arxiv.org/html/2404.00859v2)).
- On Pythia, the Figure 9 caption says "Starting from the 70M model, we see that the gap increases with parameter count". The authors call pre-caching non-negligible at 2.8B. The abstract says "pre-caching increases with model scale" ([HTML v2](https://arxiv.org/html/2404.00859v2)).
- On the synthetic task, normalized Huber loss for vanilla vs myopic is 0.0030 vs 1.03 at p=0.3 (a trivial model scores 1.26). This is clear pre-caching when the task requires it ([HTML v2](https://arxiv.org/html/2404.00859v2)).
- On Pythia-410M, the myopic gradient component is consistently larger than the future component. Training is dominated by next-token prediction ([HTML v2](https://arxiv.org/html/2404.00859v2)).

### Inferences
- A 0.12-nat myopia gap at 124M is an upper bound on how much future-directed computation helps at that scale. It is small. The gap grows from 70M to 2.8B, which makes a 1.5B target somewhat more future-directed than a 0.5B drafter, but there is no sign of a qualitative jump in that range.

### Gaps
- The text does not give the Pythia per-size gaps in nats, only the Figure 9 trend.

## Anthropic circuit tracing and open-model follow-ups (rhyme planning)

### Takeaway
Claude 3.5 Haiku plans a rhyme word at the newline before the line it ends, about one line ahead, and injecting the planned-word feature steers the ending 70% of the time. Open-model replications from 2026 disagree. Planning-like behavior appears from about 1B up, but a causally active newline planning site appears only in the largest models (Gemma-3-27B, Gemma2-9B). Qwen3 0.6B to 32B and Llama-3 up to 70B instead condition on the last word token.

### Cited Findings
- Claude 3.5 Haiku represents the planned rhyme word at the newline token that ends line 1, before line 2 starts. It holds several candidates at once (for example "rabbit" and "habit"). It plans forward (picking the target) and backward (building the line to reach it) ([Biology of an LLM](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)).
- Planned-word features appeared in "about half of the poems we investigated" ([Biology](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)).
- Across 25 poems, injecting the "rabbit" or "green" feature at the newline made the model end the line with that word in 70% of cases. Interventions changed the output only at the newline position. Probabilities used 500 samples per experiment ([Biology](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)).
- The planned-word features are ordinary word features, not special planning features ([Biology](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)).
- Counterexample: in a jailbreak case the model seemed "to not know what it plans to say until it actually says it" ([Biology](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)).
- Ma and Rui, "Where's the Plan?" (arXiv 2605.07984, ICML 2026), tested Qwen3 0.6B to 32B, Gemma-3 1B to 27B and Llama-3 1B to 70B on rhyming couplets ([HTML](https://arxiv.org/html/2605.07984)):
  - On general Pile text, linear probe accuracy for the future token falls with distance. "The k=8 probe overlaps the unigram baseline at every layer."
  - On couplets the rhyme word comes on average 8 tokens after the newline. The Gemma-3 newline-vs-first-position top-5 probe gap rises from 0.11 at 1B to 0.38 at 27B.
  - For Qwen3 0.6B to 8B and Llama-3 1B to 8B, the confidence interval of that gap includes zero.
  - Activation patching finds a causal newline site only in Gemma-3-27B, with a corrupt-rhyme rate of 0.63 [0.48, 0.78] at layer 33. Five attention heads (L30H4, L28H14, L28H15, L30H5, L28H29) reproduce 57% of it.
  - Qwen3-32B and Llama-3.1-70B show near-zero newline patching and high last-word patching.
  - For Qwen, last-word patching becomes effective only from 4B (peak 0.48 [0.25, 0.72]). The 0.6B and 1.7B intervals nearly span zero.
  - The authors' conclusion is that "probe signal and causal use are dissociable". Limits are one task and 5 prompt pairs.
- Maar, Paperno, McDougall and Nanda, "What's the plan?" (arXiv 2601.20164), covers 23 models from 1B to 32B in Gemma2, Gemma3, Qwen3 and Llama3.x. They claim "implicit planning is a universal mechanism, present in smaller models than previously thought, starting from 1B parameters", but "weaker in smaller models". Planning metrics rise with size and with instruction tuning. Clean newline steering works only for Gemma2 9B and Gemma3 27B, while last-word steering works for all models. In Gemma2 9B, patching heads L30H03 and L31H15 recovers 59 to 93% of the steering effect ([HTML](https://arxiv.org/html/2601.20164)).
- Other 2026 preprints, seen only in search snippets and not read in full:
  - One claims a replication of the planning site on Gemma 2 2B and Llama 3.2 1B with open cross-layer transcoders, and 78% rhyming with a priming couplet vs 0% without ([arXiv 2604.15010](https://arxiv.org/html/2604.15010)).
  - "Planning or Improvisation?" reports that newline-site planning is not recovered with open CLTs ([arXiv 2609.18440](https://arxiv.org/pdf/2609.18440)).
  - The same stress-test paper cites Hanna and Ameisen (2026) as reproducing rhyme planning on Qwen3 with CLTs.
  - The first and third claims conflict with Ma and Rui's null Qwen3 newline result.

### Inferences
- The planning in these studies reaches about one line ahead (about 8 tokens). The plan is one target word, not a sequence.
- At the Qwen 0.5B to 4B scale, the evidence for a dedicated planning site is null or weak. What the model "knows" about a rhyme target is mostly recomputed from the last word of the previous line, which a drafter can see too.

### Gaps
- No attribution-graph study was found on Qwen2.5-0.5B or 1.5B.
- The Hanna and Ameisen Qwen3 result was not verified directly.

## Probing future and global properties: length, structure, answers, correctness, next paragraph

### Takeaway
Coarse whole-response properties can be decoded from the prompt's last hidden state, mostly with small MLP or linear probes: length, number of reasoning steps, the multiple-choice answer, the story character, the topic of the next paragraph and eventual correctness. Specific content beyond about 5 to 10 tokens cannot.

### Cited Findings
- Dong et al., "Emergent Response Planning in LLMs" (arXiv 2502.06258) ([abs](https://arxiv.org/abs/2502.06258), [HTML](https://arxiv.org/html/2502.06258)):
  - They probe prompt hidden states for structure (response length, reasoning steps), content (story character choice, final multiple-choice answer) and behavior (confidence, factual consistency).
  - Probes are 1-hidden-layer ReLU MLPs (learned decoders, hidden size 1 to 1024), not linear. Performance plateaus at hidden size of about 128. Structure attributes converge around 16, content around 32 and behavior around 8.
  - Main models are Llama-2-7B, Llama-3-8B, Mistral-7B and Qwen2-7B (base and chat). The scaling study adds Qwen-2.5-Instruct 1.5B, 32B and 72B, among others.
  - Length and step regression correlations are "strong" for fine-tuned models and "weaker but positive" for base models. Classification is "significantly above random baseline". Exact numbers are only in figures.
  - Structure attributes peak in middle layers, content in later layers, and behavior is flat after the first few layers.
  - Verbalized self-estimates are "near or worse-than-random" for base models.
- Merzouk et al., "How Much is Left? LLMs Linearly Encode Their Remaining Output Length" (arXiv 2607.05316, NeurIPS 2026). Using linear probes on three open 7 to 8B models and seven datasets, "total response length is linearly decodable from the prompt's last hidden state alone". The probes beat a statistical baseline and the authors frame this as "approximate estimation" ([abs](https://arxiv.org/abs/2607.05316)). Numbers were not on the abstract page.
- In length-prediction work for serving, Shahout et al. (2024, TRAIL) used intermediate-layer embeddings and found layers 8 to 15 most informative. Xie et al. (ICLR 2026, arXiv 2602.11812) re-predict remaining length at each step from the target's own hidden states ([search results](https://arxiv.org/html/2602.11812)). Both were seen only through snippets.
- Pochinkov et al., "ParaScopes" (arXiv 2511.00180), decode the next paragraph from the residual stream at the "\n\n" token on Llama-3.2-3B-Instruct ([HTML](https://arxiv.org/html/2511.00180v1)):
  - Cosine similarity to the true paragraph (Qwen3-Embed): random 0.20, continuation ParaScope 0.43, TAE (SONAR-map) ParaScope 0.55, Cheat-5 (giving the model 5 true tokens) 0.50, regenerate 0.82.
  - With all-mpnet embeddings, the SONAR linear map scores 0.51 and the SONAR MLP 0.50, so the linear and MLP maps are equal.
  - An LLM judge rates subject match at TAE 76%, Continuation 43%, Cheat-10 51%, random 4%. At moderate detail depth, TAE gets only 3% and Continuation 15%.
  - The authors summarize this as "equivalent to 5+ tokens of future context" and "a limited 'lookahead' of a 5-10 tokens". Layers at 60 to 80% depth carry most of the signal. Only one model was tested.
- Zhang et al., "Reasoning Models Know When They're Right" (arXiv 2504.05419), probes R1-Distill-Qwen 1.5B, 7B and 32B, R1-Distill-Llama 8B and 70B, and QwQ-32B ([HTML](https://arxiv.org/html/2504.05419v1)):
  - Probes are MLPs, but the grid search often picked linear (d=0).
  - All probes score ROC-AUC above 0.7. R1-Distill-Qwen-32B scores over 0.9 on AIME. R1-Distill-Llama-8B scores 0.82 on GSM8K, 0.84 on MATH and 0.67 on KnowLogic.
  - ECE ranges from 0.01 to 0.14. R1-Distill-Qwen-1.5B scores 0.04, 0.04, 0.14 and 0.09 on GSM8K, MATH, AIME and KnowLogic.
  - Correctness can be predicted before the intermediate answer is written. Peak accuracy is 79%, and accuracy is 14 points lower at the 10% position of the chunk.
  - Llama-3.1-8B-Instruct (short CoT) is "much worse".
- Afzal et al., "Knowing Before Saying" (ACL Findings 2025, arXiv 2505.24362), predict zero-shot CoT success "before generating a single token" with 60 to 76.4% accuracy ([search results](https://arxiv.org/html/2505.24362v2)).
- David (arXiv 2511.14773) finds that hidden states after about 4 reasoning tokens linearly predict eventual correctness with AUC up to 0.8+ ([search results](https://arxiv.org/html/2511.14773v1)).
- As a chess analogue, Jenner et al. (arXiv 2406.00877) find that in Leela Chess Zero a probe predicts the optimal move "2 turns ahead" with 92% accuracy, on board states where there is a single best line. Activations on the squares of future moves are causally important ([abs](https://arxiv.org/abs/2406.00877)).

### Inferences
- What can be decoded at long horizons is the low-dimensional structure of the response: length, step count, answer identity, topic and correctness. The probes saturate at 8 to 32 hidden units, which suggests these "plans" are only a few dimensions. That fits a small conditioning latent rather than a rich long-horizon token plan.
- Correctness and confidence probes measure the model's own uncertainty, not its future tokens. They could be useful for adaptive draft length or early stopping, but they do not tell the drafter what to write.

### Gaps
- Exact R², Spearman and per-attribute numbers for Dong et al. and Merzouk et al. are only in figures or the full PDF.
- "Language models (mostly) know what they know" (Kadavath et al. 2022, arXiv 2207.05221) predates the 2023 window and was not re-read.
- No study was found that probes future tokens at horizons of 32, 64 or 256 tokens directly, linear or learned.

## Scaling: does future decodability improve with model size?

### Takeaway
Yes, within a family, but slowly. The jumps that are documented happen at 9B to 27B, not between 0.5B and 4B. No study compares Qwen2.5-0.5B with 1.5B directly on future-token decodability.

### Cited Findings
- Dong et al. find that within a family, larger models plan structure and content more strongly. This includes Qwen-2.5-Instruct 1.5B vs 32B vs 72B. The trend does not hold across families ([HTML](https://arxiv.org/html/2502.06258)).
- Wu et al. find the myopia gap grows with parameter count from Pythia 70M to 2.8B ([HTML v2](https://arxiv.org/html/2404.00859v2)).
- Ma and Rui find no newline planning gap for Qwen3 0.6B to 8B (CI includes zero). Last-word causal planning starts only at Qwen3-4B, and the newline site appears only in Gemma-3-27B ([HTML](https://arxiv.org/html/2605.07984)).
- Maar et al. find planning metrics increase with size and instruction tuning, and planning is "weaker in smaller models" ([HTML](https://arxiv.org/html/2601.20164)).
- Zhang et al. find correctness probes work even for R1-Distill-Qwen-1.5B, with ECE of 0.04 on MATH ([HTML](https://arxiv.org/html/2504.05419v1)).

### Inferences
- The gap between Qwen2.5-1.5B and 0.5B in "knowledge of own future" is likely small at the token level. A move to Qwen3-4B is the smallest size where some causal planning signal (last-word) shows up in the Qwen family. Even there, the signal is about one line ahead.
- The key quantity for speculative decoding is the target's future information minus what the drafter computes itself from the same prefix. No paper measures that directly.

### Gaps
- No cross-size study measures token-level look-ahead (Future Lens style) at fixed horizons.

## Which representation of the future is most decodable?

### Takeaway
From most to least decodable: coarse structural or behavioral scalars (length, step count, correctness, answer class), then semantic or topic embeddings of the next paragraph, then specific tokens within about 3 to 8 positions, then exact tokens further out (near baseline at k=8 on general text).

### Cited Findings
- Specific tokens score 44 to 48% precision@1 at N=1 to 3 with a learned decoder and 16 to 29% with a linear probe on GPT-J-6B ([Future Lens](https://arxiv.org/html/2311.04897)). The k=8 linear probe is at the unigram baseline on general text across Qwen3, Gemma-3 and Llama-3 ([Where's the Plan](https://arxiv.org/html/2605.07984)).
- A semantic embedding of the next paragraph reaches cosine similarity 0.55 vs 0.50 for 5 true tokens. A linear map is as good as an MLP (0.51 vs 0.50). The decoded content is topic-level, and detail recovery is poor ([ParaScopes](https://arxiv.org/html/2511.00180v1)).
- Structural attributes (length, steps) are decodable from the prompt with tiny probes and peak in middle layers. Content attributes peak late ([Dong et al.](https://arxiv.org/html/2502.06258)). Total length is linearly decodable ([Merzouk et al.](https://arxiv.org/abs/2607.05316)).
- Single target words (rhyme, answer noun) are decodable and causally used about one line ahead in some models ([Biology](https://transformer-circuits.pub/2025/attribution-graphs/biology.html), [Maar et al.](https://arxiv.org/html/2601.20164)).

### Inferences
- A plan latent for a drafter should target a semantic or topic embedding plus a few structural scalars, not token identities past about 8 tokens.

### Gaps
- No paper found compares bag-of-words targets against exact tokens at matched horizons.

## Verdict: how strong is the signal for a long-horizon plan latent?

### Takeaway
The signal is moderate for coarse whole-response properties and for topic over about the next paragraph. It is weak for token content beyond about 5 to 10 tokens. At the 0.5B to 4B scale it is weaker still, and there is little evidence that it differs much between 0.5B and 1.5B.

### Cited Findings
- Token look-ahead decays to the unigram baseline by k=8 on general text ([Where's the Plan](https://arxiv.org/html/2605.07984)). It is about 47% at N=3 with a learned decoder on a 6B model ([Future Lens](https://arxiv.org/html/2311.04897)).
- The paragraph plan is worth about 5 to 10 tokens of context on a 3B model ([ParaScopes](https://arxiv.org/html/2511.00180v1)).
- Whole-response length, steps, answer and correctness are decodable from the prompt ([Dong et al.](https://arxiv.org/html/2502.06258), [Merzouk et al.](https://arxiv.org/abs/2607.05316), [Zhang et al.](https://arxiv.org/html/2504.05419v1)).
- Decodability does not mean the model uses the information causally ([Where's the Plan](https://arxiv.org/html/2605.07984), [Merzouk et al.](https://arxiv.org/abs/2607.05316)).

### Inferences
- Realistic horizons:
  - About 8 tokens: there is usable token-level signal, mostly on high-confidence spans.
  - 32 to 64 tokens: there is topic or semantic signal, but no token identity.
  - 256 tokens and whole response: only length, structure, answer and correctness scalars.
- For the Qwen2.5-1.5B target and 0.5B drafter, a plan latent is most likely to help through structural conditioning (length, step boundaries, answer commitment) and through topic over the next paragraph. Gains in acceptance length beyond about 8 tokens per step should be expected to be small.
- The recommended first experiment is cheap. Fit Future Lens style linear and MLP probes on both models at k = 1, 2, 4, 8, 16, 32 tokens and on paragraph embeddings. Then compare the target-minus-drafter gap before building the full system.

### Gaps
- No direct measurement exists of target-minus-drafter future information. No study was found on Qwen2.5-0.5B or 1.5B specifically.
