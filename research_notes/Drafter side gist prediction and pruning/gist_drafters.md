# Drafter-side gist prediction: drafters that predict the target's future representations, concepts or sentence-level gist

Reading depth legend. [full] = paper HTML read for the stated details this session. [abs] = abstract or landing page only. [snip] = search-engine summary only, primary text not opened. Items marked "prior knowledge, not re-verified" are not cited and sit in Inferences or Gaps.

Lossless = exact target verification (greedy match or standard speculative-sampling rejection), output distribution preserved. Lossy = semantic or judge-based acceptance, output can differ from the target.

## 1. Feature-level drafters (EAGLE family, HASS, GRIFFIN, Hydra, Falcon, successors). What is predicted, with what loss, and what did EAGLE-3 find about dropping feature regression?

### Takeaway
EAGLE-1/2 drafters autoregress on the target's second-to-top-layer feature (one step ahead, conditioned on the token shifted by one) and are trained with feature regression plus token loss. EAGLE-3 found that the feature-regression term is a constraint that caps expressiveness and data scaling. Dropping it, predicting tokens directly, and simulating multi-step drafting in training ("training-time test") lifted LLaMA-3.1-8B acceptance length from 4.05 to 5.37 on MT-bench before adding multi-layer feature fusion, which took it to 6.13. After EAGLE-3 the drafter no longer predicts a future target feature. Target features act only as conditioning input.

### Cited Findings

EAGLE-1 (arXiv 2401.15077) [abs]
- Drafts at feature level. It predicts the target's second-to-top-layer features, not tokens directly, and feeds the token sequence advanced by one step to resolve sampling uncertainty in feature autoregression. Horizon is one feature per drafter step, rolled out autoregressively — [arXiv abs](https://arxiv.org/abs/2401.15077)
- 2.7x to 3.5x latency speedup on LLaMA2-Chat 70B, doubled throughput, preserves output distribution (lossless) — [arXiv abs](https://arxiv.org/abs/2401.15077)
- In EAGLE-3's Table 1, EAGLE-1 on Vicuna-13B at T=0 averaged 3.05x speedup with τ = 3.96 — [EAGLE-3 HTML](https://arxiv.org/html/2503.01840v3)

EAGLE-2 (arXiv 2406.16858) [via other papers]
- Keeps the EAGLE-1 drafter design and changes only to dynamic, confidence-based draft trees instead of a static tree — [HASS search summary](https://arxiv.org/html/2408.15766v3)
- T=0 averages from EAGLE-3 Table 1. Vicuna-13B 4.22x / τ 4.83. LLaMA-3.1-8B 3.23x / τ 4.11. LLaMA-3.3-70B 2.85x / τ 3.78. DeepSeek-R1-Distill-LLaMA-8B 3.26x / τ 3.92 — [EAGLE-3 HTML](https://arxiv.org/html/2503.01840v3)

EAGLE-3 (arXiv 2503.01840, NeurIPS 2025) [full]
- The authors argue that feature prediction acts as "an additional constraint, which limits the expressiveness of the draft model" and makes it hard to benefit from more data. They remove it and directly predict tokens while simulating multi-step generation in training (training-time test, TTT) — [EAGLE-3 HTML](https://arxiv.org/html/2503.01840v3)
- Removing the feature loss raises first-draft-token acceptance substantially (their Fig. 4). Later-step acceptance stays low until TTT exposes the drafter to its own outputs. Fig. 1 shows a rising data-scaling curve that the authors say prior feature-regression drafters did not show. No numeric EAGLE-vs-EAGLE-3 scaling values are given in text — [EAGLE-3 HTML](https://arxiv.org/html/2503.01840v3)
- Ablation on LLaMA-3.1-8B (MT-bench speedup/τ, GSM8K speedup/τ). EAGLE-2 3.16x/4.05, 3.39x/4.24. Remove feature constraint 3.82x/5.37, 3.77x/5.22. Add fused low/mid/high features (full EAGLE-3) 4.40x/6.13, 4.48x/6.23 — [EAGLE-3 HTML](https://arxiv.org/html/2503.01840v3)
- With the feature loss gone, the drafter can consume low-, mid- and high-level target features. The three k-dim vectors are concatenated (3k) and reduced by an FC layer back to k. Drafter core is a single Transformer decoder layer plus the fusion FC and a token+feature FC. No parameter count stated. Exact layer indices not given — [EAGLE-3 HTML](https://arxiv.org/html/2503.01840v3)
- T=0 averages over five tasks. Vicuna-13B 5.51x / τ 6.62 (peak HumanEval 6.47x / τ 7.54). LLaMA-3.1-8B 4.44x / 6.23. LLaMA-3.3-70B 4.12x / 5.88. DeepSeek-R1-Distill-LLaMA-8B 4.16x / 5.84. Comparison rows on Vicuna-13B: SpS 1.92x/2.24, PLD 1.74x/1.80, Medusa 2.12x/2.51, Lookahead 1.62x/1.67, Hydra 2.80x/3.50 — [EAGLE-3 HTML](https://arxiv.org/html/2503.01840v3)
- Up to 6.5x speedup, about 1.4x over EAGLE-2, and 1.38x throughput gain at batch 64 in SGLang — [EAGLE-3 search summary](https://arxiv.org/html/2503.01840v1)
- Lossless. It "uses strict speculative sampling acceptance conditions". The paper excludes Medusa-style typical acceptance from T=1 comparisons because that is not lossless — [EAGLE-3 HTML](https://arxiv.org/html/2503.01840v3)

HASS (arXiv 2408.15766, ICLR 2025) [snip]
- Two components. "Harmonized objective distillation" is a ranking-distillation loss focused on the target's top-probability tokens. "Harmonized context alignment" trains in multiple steps so the drafter sees its own (decoding-time) features as context — [HASS v3](https://arxiv.org/html/2408.15766v3)
- Reports 8% to 16% acceptance-length gain over EAGLE-2 on LLaMA2-Chat 7/13B and LLaMA3-Instruct 8/70B, 2.81x to 4.05x wall-clock speedup. v1 said 8% to 12% and 2.81x to 3.65x, so numbers differ by version — [HASS ICLR](https://proceedings.iclr.cc/paper_files/paper/2025/file/575286a73f238b6516ce0467d67eadb2-Paper-Conference.pdf); [HASS v1](https://arxiv.org/html/2408.15766v1)

GRIFFIN (arXiv 2502.11018, NeurIPS 2025) [snip]
- Targets two train/decode mismatches. Feature misalignment is drafter features at decoding diverging from training features (what HASS addresses). Token misalignment is ground-truth tokens diverging from drafted tokens. HASS's token misalignment reaches about 35% at later training steps — [GRIFFIN v3](https://arxiv.org/html/2502.11018v3)
- Fixes are token-alignable training, a dynamic loss mask that only counts aligned tokens, and a Token-Guided Fusion (TGF) drafter that re-injects input tokens to correct features — [GRIFFIN v3](https://arxiv.org/html/2502.11018v3)
- v2 reports average acceptance length +20% over EAGLE-2 and +8% over HASS on Vicuna-7B. Exact v3 numbers not retrieved. Code at github.com/hsj576/GRIFFIN — [GRIFFIN v2 PDF](https://arxiv.org/pdf/2502.11018v2.pdf)

Hydra (arXiv 2402.05109) [snip]
- Medusa heads are sequentially independent. Hydra heads condition each head on earlier drafted tokens, a drop-in replacement. Hydra++ reaches up to 1.31x over Medusa and 2.71x over autoregressive decoding — [Hydra HTML](https://arxiv.org/html/2402.05109v2)
- Predicts tokens from the target's last hidden state, not future features. Horizon is the number of heads.

Falcon (arXiv 2412.12639, AAAI 2025) [snip]
- Semi-autoregressive drafter that emits a block of tokens per forward pass. Coupled Sequential Glancing Distillation strengthens intra-block token dependencies. Custom decoding tree. Lossless speedup 2.91x to 3.51x on Vicuna and LLaMA2-Chat — [Falcon abs](https://arxiv.org/abs/2412.12639v2)

2025 to 2026 successors and analyses found [snip unless noted]
- DFlash (2602.06036) block diffusion drafter conditioned on fused multi-layer target features. Details in section 4 — [DFlash HTML](https://arxiv.org/html/2602.06036v1)
- DFlare (2606.02091) extends DFlash with layer-wise fusion where each draft layer gets its own weighted mix of target hidden states, and reports gains over DFlash — [DFlare HTML](https://arxiv.org/html/2606.02091v1); [HF paper page](https://huggingface.co/papers/2606.02091.md)
- DLoop (2610.07659) runs several drafting stages before one verification. Each stage consumes the final-layer hidden states the earlier stages produced, and training exposes the drafter to its own hidden states for unverified tokens. This is the closest found to "drafter predicts and rolls forward future hidden states" after EAGLE-3 — [DLoop PDF](https://arxiv.org/pdf/2610.07659)
- "Attention Drift" (2605.09992) reports that drafter hidden states grow in magnitude with speculation depth, and post-normalisation between steps limits this. Relevant to any drafter that feeds its own predicted features back in — [Attention Drift](https://arxiv.org/html/2605.09992)
- "When Hidden States Drift: Can KV Caches Rescue Long-Range Speculative Decoding?" (2604.26412). Title only, not read — [PDF](https://arxiv.org/pdf/2604.26412)
- P-EAGLE (Hui et al., 2026) is cited by H-Spec (2609.24197) as predicting multiple future positions in parallel from a shared learnable hidden state. arXiv ID of P-EAGLE itself not found — [H-Spec HTML](https://arxiv.org/html/2609.24197v1)
- Speculative Pipeline Decoding (2605.30852) conditions the drafter on pre-step features from a pipelined target — [PDF](https://arxiv.org/pdf/2605.30852)
- Multi-Future Drafting (2611.13017), mixture block drafters with tree verification. Title and snippet only — [PDF](https://arxiv.org/pdf/2611.13017)
- REAL Loss (2611.14399) optimises expected acceptance length directly. Title only — [PDF](https://arxiv.org/pdf/2611.14399)

### Inferences
- The trajectory is clear. EAGLE-1/2 make the drafter a feature predictor (k-ahead feature regression with horizon 1 per step). HASS and GRIFFIN keep that framing and fix exposure bias. EAGLE-3 and DFlash abandon future-feature targets and use target features only as conditioning, with token CE as the sole loss. Any new "gist head" that regresses onto a future target representation runs against EAGLE-3's ablation unless the gist target is coarse enough (sentence-level, low-dim) to avoid the expressiveness constraint, or is used only as an auxiliary signal.
- EAGLE-3's ablation isolates "remove feature loss + TTT" (+1.32 τ) from "multi-layer fusion" (+0.76 τ) on MT-bench, so both are meaningful. It does not test feature loss together with fusion, so whether a weak auxiliary feature loss hurts a fused drafter is untested.
- Prior knowledge, not re-verified this session. EAGLE-1 trains with Smooth-L1 feature regression plus a token cross-entropy term with a small weight (about 0.1), and the drafter is one decoder layer (about 0.24B params for a 7B target). Check the EAGLE-1 full text before quoting.

### Gaps
- EAGLE-1 loss weights and drafter parameter count. Only the abstract was read.
- GRIFFIN v3 and HASS per-model τ tables were not opened.
- No Spec-Bench leaderboard snapshot retrieved this session.
- Clover/Clover-2, ReDrafter, Gumiho, Kangaroo and DeepSeek-V3 MTP were not checked this session.

## 2. Long-horizon or gist prediction used for drafting (beyond ConFu, SD², NextLat). Is there a drafter that first predicts a sentence/segment embedding or plan, then drafts tokens conditioned on it?

### Takeaway
I found no published lossless drafter with an explicit two-level design (plan or sentence-embedding head, then a token head conditioned on it) beyond the known ConFu, SD² and NextLat. The nearest work splits into two groups. Lossy step-level speculation (SpecReason, Lookahead Reasoning, ConfSpec, SemanticSpec) accepts whole steps or semantic sequences, but its "gist" is draft text judged by the target, not a predicted embedding. Lossless multi-step hidden-state drafters (DLoop, P-EAGLE, DFlash) predict multiple future positions without an explicit segment-level latent.

### Cited Findings
- Several targeted searches ("sentence-level plan embedding then tokens", "hierarchical ... latent plan token head") returned no matching paper. Results were general SD surveys, SpecDec (2203.16487, block masked drafting in MT) and EAGLE-style feature drafters — [search hit: SpecDec](https://arxiv.org/pdf/2203.16487); [search hit: Hierarchy Drafting, NAACL Findings 2025](https://aclanthology.org/2025.findings-naacl.216.pdf)
- SemanticSpec, "Beyond Tokens: Semantic-Aware Speculative Decoding for Efficient Inference by Probing Internal States" (EMNLP 2026, no arXiv ID on the page) [abs]. It verifies whole semantic sequences instead of tokens, and estimates semantic probabilities by probing the model's hidden states for how likely it is to generate sequences with a given meaning. Up to 2.7x on DeepSeek-R1-32B and 2.1x on QwQ-32B across four benchmarks. Losslessness not stated, but semantic equivalence acceptance is lossy by construction — [ACL Anthology](https://aclanthology.org/2026.emnlp-main.123/)
- SpecReason (2504.07891) [snip]. A small model drafts reasoning steps and the base model scores each step's utility against a threshold (3, 5 or 7 of 9). That gives 1.5x to 2.5x over vanilla LRM inference with +1.0% to 9.9% accuracy, and an extra 19.4% to 44.2% latency cut when stacked with token-level SD. Lossy, threshold-controlled — [SpecReason HTML](https://arxiv.org/html/2504.07891v2)
- Lookahead Reasoning is grouped with SpecReason as a segment-level semantic method with sequential verification. Mechanism details not retrieved — [SpecGuard 2604.15244](https://www.alphaxiv.org/abs/2604.15244.md)
- ConfSpec (2602.18447) replaces the LLM judge with confidence-gated cascaded verification, up to 2.24x while matching target accuracy. SpecGuard (2604.15244) uses model-internal signals for step-level acceptance. RSD (Reward-guided SD) uses a process reward model to gate steps. All lossy — [ConfSpec](https://www.arxiv.org/pdf/2602.18447); [SpecGuard](https://arxiv.org/pdf/2604.15244.pdf)
- NextLat (2511.05963), for comparison only. It learns latent dynamics that can be "recursively composed in latent space, enabling variable-length self-speculative decoding" — [NextLat PDF](https://arxiv.org/pdf/2511.05963)
- DLoop (2610.07659) is a lossless multi-stage drafter rolling forward its own hidden states before one verification. Horizon spans several drafting stages. The predicted states condition later token drafting, which makes it the closest lossless analogue to "predict future representation, then draft" — [DLoop PDF](https://arxiv.org/pdf/2610.07659)
- DFlash's authors justify their design with the claim that AR target hidden states already encode information about several upcoming tokens, citing Samragh et al. (2507.11851) — [DFlash HTML](https://arxiv.org/html/2602.06036v1)

### Inferences
- The two-level "gist head + token head" lossless drafter looks open as of Nov 2026, apart from ConFu, SD² and NextLat. The closest building blocks are DFlash-style KV injection of a conditioning vector (a predicted gist could be injected the same way as target context features) and DLoop-style multi-stage rollout with self-generated hidden states.
- Step-level lossy methods show that segment-level acceptance can yield 1.5x to 2.7x on reasoning models, but they spend target compute on judging or probing. A gist predictor that steers token drafting while keeping exact verification would keep losslessness and stack with these.

### Gaps
- SemanticSpec mechanism details (what is drafted, probe design, exact acceptance rule) are not on the landing page. No arXiv ID found.
- Lookahead Reasoning arXiv ID and acceptance mechanism were not retrieved. Prior knowledge suggests 2506.19830, not verified.
- The search did not cover Chinese-venue or workshop papers. A 2026 plan-conditioned drafter may exist under different terms such as "intent", "outline" or "skeleton".

## 3. Concept/sentence-level LMs that could serve as the gist predictor (LCM/SONAR, CoCoMix, next-sentence embedding prediction, CALM, latent diffusion). Any used for speed?

### Takeaway
LCM, SONAR-LLM and CoCoMix provide sentence-embedding or concept predictors, but none was used as a speculative-decoding drafter in what I found. LCM's speed argument is qualitative (shorter sequence of sentence embeddings). CALM (2510.27688) is the one concept-level model explicitly built for speed. It predicts one continuous vector per K tokens and cuts generative steps about K-fold, but it is a standalone lossy LM, not a drafter with exact verification.

### Cited Findings
- Large Concept Models (2412.08821, Meta) [snip]. Autoregressive next-sentence prediction in SONAR embedding space, where a sentence is a "concept". SONAR supports text in about 200 languages — [LCM HTML](https://arxiv.org/html/2412.08821v2); [Meta page](https://ai.meta.com/research/publications/large-concept-models-language-modeling-in-a-sentence-representation-space/)
- Speed claims around LCM are qualitative. It reduces latency by predicting a much shorter trajectory of sentence-level embeddings. No throughput benchmarks were found in the results — [SONAR-LLM HTML](https://arxiv.org/html/2508.05305v1)
- SONAR-LLM (2508.05305) predicts the next sentence embedding and decodes it through a frozen SONAR decoder, trained with token-level CE through that decoder (so likelihood-based, unlike LCM's MSE/diffusion objectives) — [SONAR-LLM HTML](https://arxiv.org/html/2508.05305v1)
- CoCoMix (2502.08524, ICLR 2026) [snip]. A pretrained SAE extracts concepts from hidden states and the most influential are selected by attribution. The LM learns to predict them from its hidden state with CE, compresses them into one continuous concept vector, and interleaves it with token hidden states. A 1.38B model matches NTP with 21.5% fewer training tokens. Concept vectors can be inspected and steered. It is a training-efficiency method, not an inference-speed method — [CoCoMix HTML](https://arxiv.org/html/2502.08524v1); [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/59056767478c7df64e6250eadfeb0a04-Abstract-Conference.html)
- CALM, Continuous Autoregressive Language Models (2510.27688) [snip]. An autoencoder compresses K tokens into one vector with over 99.9% reconstruction accuracy. The LM does likelihood-free next-vector prediction with a lightweight generative head conditioned on the last hidden state, reducing steps about K-fold, and claims strong-baseline performance at lower compute. It introduces the BrierLM metric. A context-aware autoencoder is left as future work — [CALM abs](https://arxiv.org/abs/2510.27688v1); [HF model card](https://huggingface.co/cccczshao/CALM-Autoencoder)

### Inferences
- For a gist predictor inside a lossless drafter, CoCoMix-style SAE concept prediction (CE over a discrete concept set, cheap head on the hidden state) and CALM-style K-token vectors are the most directly usable. Both are predicted from a single hidden state, which is what an EAGLE/DFlash-style drafter already receives. LCM/SONAR embeddings come from an external encoder, so the drafter would need to predict into a space the target does not natively use.
- CALM's K-token vector is effectively a "chunk gist" with near-lossless reconstruction. As a drafter, decoding a predicted vector into K tokens and verifying them exactly would make it a block drafter. That usage was not found in the literature.

### Gaps
- No measured LCM latency numbers retrieved. The LCM paper itself was not opened.
- No paper found that uses LCM, SONAR-LLM, CoCoMix or CALM as a speculative drafter.
- Latent diffusion LMs that predict sentence latents (for example PLANNER, or 2025-2026 latent-diffusion text models) were not searched this session.

## 4. Block/diffusion drafters (DFlash, PARD, DiffuSpec, etc.). What conditioning vectors do they take?

### Takeaway
DFlash conditions a 5-layer block-diffusion drafter on hidden states from 5 target layers. These are fused by a projection and injected into the K/V of every draft layer, persisting in the draft KV cache. It reaches τ 6.49 and 4.86x on Qwen3-8B versus EAGLE-3's τ 2.96 and 1.76x, losslessly. Large-dLLM drafters (DiffuSpec, SpecDiff-2) and PARD condition on context tokens rather than target features, as far as the DFlash related-work section indicates.

### Cited Findings
- DFlash (2602.06036) [full] — [DFlash HTML](https://arxiv.org/html/2602.06036v1)
  - Conditioning. Hidden states from 5 target layers sampled uniformly between layer 2 and the third-to-last layer are concatenated and projected into a "target context feature". This is projected into the K and V of every draft layer, stored in the draft KV cache and reused across drafting iterations. The authors contrast this with EAGLE-3, which feeds features only as input and so lets them dilute with depth.
  - Drafter has 5 layers (8 for Qwen3-Coder). Embedding and LM head are shared and frozen from the target, so only draft layers train. No parameter count given.
  - Block size 16 for Qwen3, 10 for LLaMA-3.1.
  - Loss is position-weighted CE, w_k = exp(−(k−1)/γ) with γ = 7, 5 or 4 for blocks 16, 10 or 8. Token CE only, no feature regression. 512 random anchor positions per sequence, 6 epochs, AdamW at 6e-4.
  - Targets are LLaMA-3.1-8B-Instruct, Qwen3-4B, Qwen3-8B and Qwen3-Coder-30B-A3B.
  - Qwen3-8B, thinking off, greedy. DFlash 4.86x / τ 6.49 (peak 6.08x on MATH-500). EAGLE-3 (16 draft tokens) 1.76x / τ 2.96. EAGLE-3 (60) 2.02x / τ 3.40. About 2.4x over EAGLE-3 greedy and about 2.2x at T=1. Thinking on gives 3.9x to 4.6x. SGLang Qwen3-8B up to 5.1x at concurrency 1. LLaMA-3.1-8B in SGLang gives 2.2x to 2.8x versus 1.5x to 2.0x for EAGLE-3.
  - Lossless ("over 6x lossless acceleration") via standard target verification.
- DFlare (2606.02091) gives each draft layer a layer-wise learned mixture of target hidden states, improving on DFlash's single fused vector — [HF paper page](https://huggingface.co/papers/2606.02091.md)
- DFlash related work describes DiffuSpec (2510.02358) and SpecDiff-2 (2511.00606) as using large pretrained dLLMs as drafters, TiDAR (2511.08923) as jointly training diffusion and AR objectives, and PARD (2504.18583) as training small AR models to do parallel drafting. DFlash does not attribute target-hidden-state conditioning to any of these — [DFlash HTML](https://arxiv.org/html/2602.06036v1)
- Samragh et al. (2507.11851) train a LoRA adapter for parallel drafting and observe that hidden features encode future-token information — [DFlash HTML](https://arxiv.org/html/2602.06036v1)
- DART (2601.19278) is a "diffusion-inspired" SD method. Title and snippet only — [DART HTML](https://arxiv.org/html/2601.19278v1)
- NVIDIA NeMo AutoModel ships a DFlash speculative component, and an MLX reimplementation fuses hidden states from several target layers through a projection — [NeMo docs](https://docs.nvidia.com/nemo/automodel/latest/nemo-automodel/nemo_automodel/components/speculative/dflash); [MLX port](https://huggingface.co/tritesh/dflash-mlx-universal/blob/main/dflash_mlx/model.py)

### Inferences
- DFlash's KV-injection slot is a natural place for a predicted gist vector. Its conditioning is the target's current context (no future prediction), and the 16-token block is a horizon at which a segment-level gist would plausibly help most, since later block positions are down-weighted in the loss precisely because they are hard.
- The big τ jump from EAGLE-3 (AR, 1 layer) to DFlash (parallel, 5 layers, deep KV conditioning) suggests drafter depth and conditioning pathway matter more than predicting future features.

### Gaps
- PARD, DiffuSpec, SpecDiff-2 and TiDAR papers were not opened. Their exact conditioning (pure tokens versus any target features) is inferred from DFlash's related-work description only.
- DFlash drafter parameter count is not given in the paper.
- The z-lab/dflash and SafeAILab/EAGLE GitHub repos were not inspected this session.
