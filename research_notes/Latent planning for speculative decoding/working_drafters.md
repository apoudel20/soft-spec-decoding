# Speculative decoding methods that demonstrably work (2023 to 2026), as replacements for weak parts of H1 and H3

Context for the report writer. Our setup is target Qwen2.5-1.5B, drafter Qwen2.5-0.5B, lossless speculative decoding. H1 means the target refines its own future guesses inside the verification pass via gated LoRA on extra positions (prior art Samragh et al. 2025, arXiv 2507.11851). H3 means a drafter trained with self-conditioning plus target soft labels (KL or TV distillation). Our own measurement shows that parallel block drafters have high teacher-forced per-position accuracy, but free-running acceptance collapses with offset (a coherence gap). One self-conditioning pass recovers about half of it.

Notation. tau = mean accepted tokens per target forward pass (includes the bonus token unless noted). "Speedup" = wall-clock latency ratio vs autoregressive decoding unless marked "tokens/step only". All numbers below were read from the primary paper or leaderboard in this session unless listed under Gaps.

## Q1. EAGLE-1/2/3 (feature-level drafting, dynamic trees, training-time test)

### Takeaway
EAGLE-3 (arXiv 2503.01840) is the strongest verified single-drafter baseline among autoregressive drafters. It reaches tau 6.62 and 5.51x on Vicuna-13B at T=0 and stays lossless at T=1 (tau 5.67, 4.65x). Its two decisive changes are dropping the feature-regression loss and "training-time test", in which the drafter is trained on its own sampled outputs. The paper reports that this keeps per-step acceptance almost flat as the number of self-generated inputs grows, while EAGLE-1 acceptance drops. That flat curve is exactly the free-running coherence gap we measured, closed by self-conditioning during training.

### Cited Findings
- EAGLE-3 abstract reports speedup up to 6.5x, about 1.4x over EAGLE-2, and a 1.38x throughput gain in SGLang at batch size 64. It "abandons feature prediction in favor of direct token prediction" and uses multi-layer feature fusion via "training-time test" ([arXiv 2503.01840](https://arxiv.org/abs/2503.01840)).
- EAGLE-3 Table 1, mean over MT-bench, HumanEval, GSM8K, Alpaca and CNN/DM, T=0, Vicuna-13B. SpS 1.92x (tau 2.24), PLD 1.74x (1.80), Medusa 2.12x (2.51), Lookahead 1.62x (1.67), Hydra 2.80x (3.50), EAGLE 3.05x (3.96), EAGLE-2 4.22x (4.83), EAGLE-3 5.51x (6.62) ([arXiv 2503.01840 HTML](https://arxiv.org/html/2503.01840)).
- Same table, other targets at T=0. LLaMA-3.1-8B EAGLE-2 3.23x (tau 4.11) vs EAGLE-3 4.44x (6.23). LLaMA-3.3-70B 2.85x (3.78) vs 4.12x (5.88). DeepSeek-R1-Distill-LLaMA-8B 3.26x (3.92) vs 4.16x (5.84) ([arXiv 2503.01840 HTML](https://arxiv.org/html/2503.01840)).
- T=1 (lossless sampling, Medusa excluded as lossy). Vicuna-13B SpS 1.60x (1.84), EAGLE 2.44x (3.45), EAGLE-2 3.76x (4.41), EAGLE-3 4.65x (5.67). LLaMA-3.1-8B EAGLE-2 2.80x (3.62) vs EAGLE-3 3.45x (4.92). LLaMA-3.3-70B 2.65x (3.54) vs 3.95x (5.66) ([arXiv 2503.01840 HTML](https://arxiv.org/html/2503.01840)).
- Per task, Vicuna-13B, T=0, EAGLE-3. MT-bench 5.58x, HumanEval 6.47x, GSM8K 5.32x, Alpaca 5.16x, CNN/DM 5.01x ([arXiv 2503.01840 HTML](https://arxiv.org/html/2503.01840)).
- Ablation on LLaMA-3.1-8B (MT-bench speedup/tau, GSM8K speedup/tau). EAGLE-2 3.16x/4.05, 3.39x/4.24. Removing the feature constraint (which enables training-time test) 3.82x/5.37, 3.77x/5.22. Adding fused low/mid/high features 4.40x/6.13, 4.48x/6.23. The paper does not isolate training-time test from the loss change ([arXiv 2503.01840 HTML](https://arxiv.org/html/2503.01840)).
- Training-time test mechanism. The drafter generates predictions and feeds them back as inputs during training, with attention masks adjusted to mimic the tree context at inference. Figure 7 shows EAGLE acceptance dropping significantly as the count of self-estimated inputs grows, while EAGLE-3 stays almost unchanged (no per-position table given) ([arXiv 2503.01840 HTML](https://arxiv.org/html/2503.01840)).
- EAGLE-3 scaling. Speedup rises with training data (ShareGPT about 68K plus UltraChat-200K about 464K, about 8x EAGLE's data), whereas the original EAGLE did not scale ([arXiv 2503.01840 HTML](https://arxiv.org/html/2503.01840)).
- Serving. SGLang H100, LLaMA-3.1-8B, MT-bench, batch 2 to 64. EAGLE-3 1.81x down to 1.38x, EAGLE falls below 1.0x from batch 24. SGLang batch 1 throughput 158.34 tok/s (no spec), 244.10 (EAGLE-2), 373.25 (EAGLE-3). vLLM batch 2 to 56 EAGLE-3 1.75x to 1.01x ([arXiv 2503.01840 HTML](https://arxiv.org/html/2503.01840)).
- Independent leaderboard (Spec-Bench, greedy, FP16, batch 1, A100, Vicuna-13B-v1.3). EAGLE3 3.02x overall (tau 5.71), EAGLE2 2.46x (4.43), Hydra 2.20x (3.35), EAGLE 2.16x (3.64), Medusa 1.80x (2.39), SpS 1.54x (2.19), Lookahead 1.30x (1.63) ([Spec-Bench Leaderboard](https://github.com/hemingkx/Spec-Bench/blob/main/Leaderboard.md)).
- Spec-Bench A100 Vicuna-7B-v1.3. EAGLE2 2.36x (tau 4.34), EAGLE 2.10x (3.58), Hydra 2.07x (3.26), Medusa 1.71x (2.31), SpS 1.52x (2.28), Lookahead 1.34x (1.64) ([Spec-Bench Leaderboard](https://github.com/hemingkx/Spec-Bench/blob/main/Leaderboard.md)).
- HASS (harmonized speculative sampling) fixes the same training/decoding context mismatch ("harmonized context alignment") plus an objective mismatch ("harmonized objective distillation"). It reports 2.81x to 4.05x wall-clock speedup on four LLaMA models over three datasets, 8% to 20% over EAGLE-2 ([arXiv 2408.15766](https://arxiv.org/abs/2408.15766)).

### Inferences
- The independent Spec-Bench numbers are roughly 55% of the EAGLE-3 paper's self-reported speedups on the same Vicuna-13B target (3.02x vs 5.51x). Tau is closer (5.71 vs 6.62). Expect paper-reported wall-clock numbers to shrink under third-party harnesses.
- EAGLE-3's training-time test is the strongest published evidence that training a drafter on its own outputs closes an offset-dependent acceptance decay. It directly supports the self-conditioning half of H3. Our "one pass recovers half" result suggests we are under-doing it. EAGLE-3 unrolls the full draft depth during training, not one pass.
- The EAGLE-3 ablation shows that conditioning on multi-layer target features (not top layer only) adds about 0.8 tau. Our 0.5B drafter as a standalone model does not see target features at all. That is a larger likely loss than the choice of KL vs TV.

### Gaps
- EAGLE-1 (arXiv 2401.15077) and EAGLE-2 (arXiv 2406.16858) abstracts were not re-read this session. Their numbers above come from the EAGLE-3 paper's re-runs and from Spec-Bench, not from their original papers.
- No EAGLE-3 number exists in these sources for a target as small as 1.5B. Draft overhead is relatively larger on small targets.

## Q2. Medusa, Medusa-2, Hydra, ReDrafter, MTP heads (Gloeckle 2024, DeepSeek-V3), Falcon and other semi-autoregressive drafters

### Takeaway
Independent heads (Medusa) are the weakest learned drafters (tau about 2.3 to 2.5). Every method that adds sequential dependence among draft positions gains measurably. Hydra adds up to 1.31x over Medusa, Falcon's glancing distillation lifts tau from 2.78 to 3.34, and the Apple sampler head conditions each token on the previous sampled one. This is the cleanest literature evidence that the within-block coherence gap is real and fixable with cheap sequential conditioning.

### Cited Findings
- Medusa-1 (frozen backbone) reports over 2.2x speedup "without compromising generation quality". Medusa-2 (heads fine-tuned jointly with backbone) reports 2.3x to 3.6x. The abstract does not claim Medusa-2 or its "typical acceptance" scheme is lossless ([arXiv 2401.10774](https://arxiv.org/abs/2401.10774)). The EAGLE-3 paper excludes Medusa from T=1 results as lossy ([arXiv 2503.01840 HTML](https://arxiv.org/html/2503.01840)).
- Hydra makes each draft head depend on the preceding candidate tokens, reporting throughput up to 1.31x over Medusa and 2.70x over autoregressive decoding (Hydra++) ([arXiv 2402.05109](https://arxiv.org/abs/2402.05109)). On Spec-Bench A100 Vicuna-13B, Hydra reaches tau 3.35 vs Medusa 2.39 ([Spec-Bench Leaderboard](https://github.com/hemingkx/Spec-Bench/blob/main/Leaderboard.md)).
- ReDrafter uses an RNN draft conditioned on the target's hidden states, beam search, dynamic tree attention to dedupe prefixes, and knowledge distillation from the target. It reports up to 2.8x on Vicuna MT-Bench in PyTorch on H100 and up to 2.3x in MLX on Apple Silicon ([arXiv 2403.09919](https://arxiv.org/abs/2403.09919)).
- Gloeckle et al. 4-token-prediction models are "up to 3 times faster at inference, even with large batch sizes" (13B models). They also solve about 12% more HumanEval and 17% more MBPP problems ([arXiv 2404.19737](https://arxiv.org/abs/2404.19737)).
- DeepSeek-V3 MTP module used for speculation. Second-token acceptance is 85% to 90% across topics, giving 1.8x TPS ([DeepSeek-V3 report, arXiv 2412.19437](https://arxiv.org/pdf/2412.19437)). A later DeepSeek paper states 80% to 90% ([arXiv 2505.09343](https://arxiv.org/pdf/2505.09343)). These two primary sources differ slightly on the range.
- An independent study reportedly measured a median second-position acceptance of 62.8% on plain text and found realized speedup below accepted-token progress. Later work notes that recursively reusing the single shipped MTP module gives poor acceptance beyond the first extra token. Both claims come from a search summary and were not read in full this session ([arXiv 2609.35188](https://arxiv.org/pdf/2609.35188), [FastMTP arXiv 2509.18362](https://arxiv.org/html/2509.18362v1)).
- Falcon (semi-autoregressive drafter, Coupled Sequential Glancing Distillation). It argues standard semi-AR training ignores within-block mutual information and replaces spans of draft inputs with ground-truth tokens on a Hamming-distance schedule. It uses a relaxed causal mask in which tokens see their own k-token block and predecessors, with SpecInfer-style tree verification ([arXiv 2412.12639 HTML](https://arxiv.org/html/2412.12639)).
- Falcon greedy speedups (MT-Bench / HumanEval / GSM8K). Vicuna-7B 3.10x / 3.21x / 2.92x vs EAGLE 2.82x / 2.95x / 2.72x. Vicuna-13B 2.97x / 3.12x / 3.13x. LLaMA2-Chat-13B 3.11x / 3.51x / 3.10x vs EAGLE 2.95x / 3.25x / 2.93x ([arXiv 2412.12639 HTML](https://arxiv.org/html/2412.12639)).
- Falcon acceptance, Vicuna-7B MT-Bench. Falcon alpha 77.64%, tau 3.34. EAGLE alpha 74.59%, tau 3.94. Medusa alpha 60.53%, tau 1.51. Falcon wins on speedup despite lower tau because it drafts k tokens per drafter pass ([arXiv 2412.12639 HTML](https://arxiv.org/html/2412.12639)).
- Falcon CSGD ablation, Vicuna-7B MT-Bench. Without tree attention CSGD raises alpha 62.86 to 68.68 and tau 1.64 to 2.13 (2.06x to 2.27x). With tree attention, alpha 74.37 to 77.64, tau 2.78 to 3.34 (2.65x to 3.10x) ([arXiv 2412.12639 HTML](https://arxiv.org/html/2412.12639)).

### Inferences
- The progression Medusa, then Hydra, then Falcon or the Apple sampler head is a controlled natural experiment showing that cross-position conditioning inside a draft block is worth roughly +0.5 to +1.0 tau. It directly targets our coherence gap.
- Falcon's CSGD is a training-side fix (expose the drafter to correct partial blocks and train the rest). Our self-conditioning pass is the inference-side mirror (feed the drafter its own guesses). The two are complementary and both are cheap.
- For H3, Medusa-2 style joint fine-tuning of the target is not lossless in general. Any H1 variant that changes target weights must keep the gate so the verification logits stay exactly the base model's.

### Gaps
- Gloeckle's exact self-speculative tau and per-model speedups were not extracted (only the abstract's "up to 3x").
- Hydra model names and per-benchmark numbers were not in the abstract. Use the Spec-Bench figures.
- Other semi-AR drafters named in the brief (Clover arXiv 2405.00263, Amphista arXiv 2406.13170, Gumiho arXiv 2503.10135) were not verified this session.

## Q3. Parallel and block drafters (DFlash, PARD, ParallelSpec, PASS, BiTA, SDSAT, diffusion drafters, Apple quadratic mask) and within-block coherence

### Takeaway
Parallel block drafters work at the state of the art when the drafter is conditioned on the target's internal features. DFlash (arXiv 2602.06036, ICML 2026) reaches tau 6.49 and 4.86x on Qwen3-8B at T=0, and tau 5.48 and 4.03x at T=1, about 2.4x EAGLE-3's speedup in the same harness. It gets coherence from bidirectional attention within the block, multi-layer target features injected into every drafter layer's K/V, and an exponentially decaying position-weighted loss. PARD shows a target-independent mask-token drafter beats a vanilla small AR drafter on Qwen2.5 targets. The Apple method's best result is 5.2 to 5.35 tokens/step (not wall-clock) on GSM8K and HumanEval, averaging 3.17 across 13 tasks. Its coherence fix is a small sampler MLP that conditions each position on the previously sampled token.

### Cited Findings
- DFlash architecture. A lightweight block-diffusion drafter (default 5 layers) takes hidden features from 5 target layers sampled uniformly between layer 2 and the third-to-last, concatenated and projected. That "target context feature" is injected into the Key and Value projections of every draft layer and cached. The authors contrast this with EAGLE-3, where target features enter only as inputs and "dilute with depth" ([arXiv 2602.06036 HTML](https://arxiv.org/html/2602.06036v1)).
- DFlash within-block decoding. "All masked positions within a block are decoded in parallel in a single forward pass." Training uses bidirectional attention within the block plus attention to injected context, with no cross-block attention. Default block size is 16 (10 for LLaMA-3.1) ([arXiv 2602.06036 HTML](https://arxiv.org/html/2602.06036v1)).
- DFlash loss. Cross-entropy with position weights w_k = exp(-(k-1)/gamma), gamma 7 for block 16, 5 for block 10, 4 for block 8. Training uses 512 random anchor positions per sequence, about 800K samples whose responses were regenerated by the target, 6 epochs, LR 6e-4. Embedding and LM head are shared with the target and frozen ([arXiv 2602.06036 HTML](https://arxiv.org/html/2602.06036v1)).
- DFlash Table 1 (thinking off, max 2048 tokens, Transformers backend, H200, averaged over GSM8K, MATH-500, AIME25, HumanEval, MBPP, LiveCodeBench, MT-Bench). Qwen3-4B T=0, DFlash 4.91x (tau 6.54) vs EAGLE-3(16) 1.81x (3.05) and EAGLE-3(60) 2.08x (3.48). Qwen3-8B T=0, DFlash 4.86x (6.49) vs EAGLE-3(16) 1.76x (2.96) and EAGLE-3(60) 2.02x (3.40). Qwen3-8B T=1, DFlash 4.03x (5.48) vs EAGLE-3(60) 1.88x (3.26). Qwen3-4B T=1, DFlash 4.24x (5.69) ([arXiv 2602.06036 HTML](https://arxiv.org/html/2602.06036v1)).
- DFlash reasoning mode, Qwen3-8B, T=0. GPQA 4.17x (tau 5.17), MATH-500 4.64x (5.82), AIME25 4.51x (5.74). SGLang on B200 reaches up to 5.1x on Qwen3-8B MATH-500 at concurrency 1. The headline peak is 6.1x ([arXiv 2602.06036 HTML](https://arxiv.org/html/2602.06036v1)). The abstract claims "over 6x lossless acceleration" and up to 2.5x over EAGLE-3 ([search result for arXiv 2602.06036](https://arxiv.org/abs/2602.06036v1)).
- DFlow (follow-up to DFlash) reuses the verifier's hidden states at rejected positions to guide the next drafting round at no extra target compute. It adds a "self-condition" training strategy that feeds verifier representations from earlier predictions into later ones. It reports consistently better acceptance length than DFlash on Qwen3 models (exact numbers not in abstract) ([arXiv 2609.06498](https://arxiv.org/abs/2609.06498)).
- DScale (follow-up) keeps the DFlash drafter unchanged and adds a 112K-parameter predictor for dynamic verify-length allocation. It reports +43.9% (Qwen3-8B) and +48.8% (Qwen3-4B) geometric-mean throughput over DFlash at concurrency 8 to 32 on A100-40GB ([arXiv 2609.37532](https://arxiv.org/abs/2609.37532)).
- PARD inserts mask tokens so one drafter pass yields K tokens, each depending only on the prefix and masks ("the predictions at each step do not depend on each other"). Conditional Drop-token training cuts training cost about 3x. A shared mask ID beat distinct IDs (221.97 vs 218.05 tok/s) and allows inference K above training K ([arXiv 2504.18583 HTML](https://arxiv.org/html/2504.18583)).
- PARD wall-clock on vLLM, A100-40GB, averaged over HumanEval, GSM8K and SpecBench. Qwen2.5-3B AR 130.28 tok/s, vanilla small-AR draft (VSD) 1.55x, PARD 2.65x. Qwen2.5-7B VSD 1.97x, PARD 3.18x. Qwen2.5-14B VSD 2.97x, PARD 3.71x. LLaMA3.1-8B EAGLE 1.58x, EAGLE-3 2.65x, VSD 1.84x, PARD 3.00x. LLaMA3.3-70B EAGLE-3 3.22x, PARD 3.49x ([arXiv 2504.18583 HTML](https://arxiv.org/html/2504.18583)).
- PARD per-position acceptance, LLaMA3.1-8B. HumanEval 1-alpha / 4-alpha is EAGLE 0.83/0.72, EAGLE-3 0.87/0.85, PARD 0.93/0.90. GSM8K is 0.79/0.66, 0.82/0.79, 0.88/0.85. PARD drafter memory traffic stays 2.48 GB regardless of draft length, vs 5.94 to 11.88 GB for EAGLE/EAGLE-3 at k=4 to 8. Speedup falls from 3.63x at batch 1 to 1.33x at batch 16 ([arXiv 2504.18583 HTML](https://arxiv.org/html/2504.18583)).
- Apple "Your LLM Knows the Future". Tulu3-8B SFT with frozen backbone, gated LoRA rank 128, k=8 mask tokens. Average tokens/step is 1.67 with 1 mask and 3.17 with 8 masks across 13 tasks. GSM8K 5.22, HumanEval 5.35, MMLU 2.38, AlpacaEval 2.52, PopQA 1.91. Averages for 2 to 7 masks are 2.14, 2.48, 2.69, 2.88, 2.99, 3.10 ([arXiv 2507.11851 HTML](https://arxiv.org/html/2507.11851)).
- Apple metrics and losslessness. The paper reports generated tokens per generation step only, not wall-clock. Verification is token matching against standard AR output. The paper does not say whether sampling-based decoding is supported ([arXiv 2507.11851 HTML](https://arxiv.org/html/2507.11851)). The abstract's "nearly 5x" (code/math) and "almost 2.5x" (chat/knowledge) are therefore tokens/step ([arXiv 2507.11851](https://arxiv.org/abs/2507.11851)).
- Apple coherence mechanisms. The sampler head concatenates the previous sampled token's embedding with the current hidden state and passes them through a 2-layer MLP, so each token is "conditioned on previously sampled ones". Quadratic decoding beats linear, with a gap that grows with mask count. A latent consistency loss also helps (ablation values in figures only). Ranks above 128 reduce speedup ([arXiv 2507.11851 HTML](https://arxiv.org/html/2507.11851)).
- ParallelSpec uses a single parallel drafter predicting multiple future tokens. It reports up to 62% latency reduction over baseline methods and 2.84x overall on Llama-2-13B (third-party criteria). It is a v1 "work in progress" ([arXiv 2410.05589](https://arxiv.org/abs/2410.05589)).
- DiffuSpec (training-free, pretrained diffusion LM as drafter) adds causal-consistency path search and an adaptive draft-length controller, reporting up to 3x wall-clock ([arXiv 2510.02358](https://arxiv.org/html/2510.02358v1)). SpecDiff showed discrete diffusion drafters can work ([arXiv 2408.05636](https://arxiv.org/pdf/2408.05636)). SpecDiff-2 adds train-time alignment and reports up to 5.5x average, with the aligned drafter over 40% better than the base SpecDiff drafter ([arXiv 2511.00606](https://arxiv.org/pdf/2511.00606v1)). DEER reportedly reaches acceptance lengths up to 32 tokens ([arXiv 2512.15176](https://www.arxiv.org/pdf/2512.15176)). These DLM-drafter details came from a search summary, and DFlash describes DiffuSpec and SpecDiff-2 as using large (for example 7B) drafters ([arXiv 2602.06036](https://arxiv.org/pdf/2602.06036)).

### Inferences
- DFlash is the single most relevant replacement for our block drafter. It shows that a parallel one-pass drafter can beat AR feature drafters by a wide margin once it (a) reads multi-layer target hidden states through K/V injection, (b) uses bidirectional attention inside the block, and (c) down-weights far positions in the loss. Our Qwen2.5-0.5B drafter conditions only on tokens. That probably explains more of the free-running collapse than within-block dependence alone.
- DFlow's "self-condition" training on verifier representations from earlier rounds is the closest published analogue of a combined H1+H3. The target's pass over rejected/future positions feeds the drafter's next round without changing target weights. This keeps losslessness trivially and needs no gated LoRA. It is a strong candidate to replace the H1 mechanism.
- The Apple method's 3.17 average tokens/step on an 8B model is well below DFlash's tau 6.5 on Qwen3-8B (different models and tasks, so this is not a controlled comparison). Its headline 5x is tokens/step on math/code only. H1 as specified is therefore not the frontier. Its useful contribution is the sampler-head idea, which is a cheap coherence fix.
- PARD gives the most relevant small-model data point. On Qwen2.5-3B, a vanilla small AR drafter gets only 1.55x while PARD gets 2.65x. For our 1.5B target with a 0.5B drafter (drafter about one third of target size), sequential AR drafting is expensive, so one-pass block drafting is structurally favored.
- PARD shows independent mask predictions still give 4-alpha of 0.85 to 0.90 on code/math with K=8. That suggests the coherence gap is task dependent and smallest on templated domains. Measure it on chat as well as GSM8K.

### Gaps
- PASS (arXiv 2311.13581), BiTA (arXiv 2401.12522) and SDSAT (arXiv 2403.18647) were not fetched this session, so no verified numbers.
- DFlash does not report per-position acceptance or an ablation of bidirectional vs causal within-block attention in the extracted text. That is the exact ablation we need.
- DFlow's numeric gains over DFlash were not in the abstract.
- No paper found reports results for a Qwen2.5-1.5B target specifically. PARD's smallest Qwen2.5 target is 3B.

## Q4. Jacobi, lookahead decoding, CLLMs, and self-speculative layer skipping (LayerSkip, Draft&Verify, SWIFT, Kangaroo)

### Takeaway
Training-free and self-speculative methods are consistently the weakest on wall-clock. Lookahead decoding reaches tau about 1.6 and 1.13x to 1.34x on Spec-Bench. SWIFT reports 1.3x to 1.6x, Kangaroo up to 1.68x, and LayerSkip up to 2.16x. CLLMs reach 2.4x to 3.4x but require fine-tuning the target. None of these is a good replacement for a learned drafter at our scale.

### Cited Findings
- Lookahead decoding is "an exact, parallel decoding algorithm" with no draft model. It reports up to 1.8x on MT-bench and 4x with strong scaling on multiple GPUs for code completion ([arXiv 2402.02057](https://arxiv.org/abs/2402.02057)).
- Spec-Bench independent numbers for Lookahead. A100 Vicuna-7B 1.34x (tau 1.64), Vicuna-13B 1.30x (1.63), Vicuna-33B 1.25x (1.61), RTX 3090 Vicuna-7B 1.13x ([Spec-Bench Leaderboard](https://github.com/hemingkx/Spec-Bench/blob/main/Leaderboard.md)). EAGLE-3 paper re-run on Vicuna-13B gives 1.62x (tau 1.67) ([arXiv 2503.01840 HTML](https://arxiv.org/html/2503.01840)).
- CLLMs (Jacobi decoding plus consistency fine-tuning of the target) report 2.4x to 3.4x generation speed "while preserving generation quality" on domain-specific and open-domain benchmarks. The abstract does not use the word lossless ([arXiv 2403.00835](https://arxiv.org/abs/2403.00835)).
- LayerSkip (early exit plus self-speculation with remaining layers) reports up to 2.16x on CNN/DM summarization, 1.82x on coding and 2.0x on TOPv2 semantic parsing ([arXiv 2404.16710](https://arxiv.org/abs/2404.16710)).
- SWIFT (on-the-fly layer skipping, plug-and-play, no training or auxiliary model) reports 1.3x to 1.6x and states it preserves the original output distribution ([arXiv 2410.06916](https://arxiv.org/abs/2410.06916)).
- Kangaroo (shallow sub-network plus a 67M-parameter adapter, early stop when confidence falls below a threshold) reports up to 1.68x on Spec-Bench. Its adapter is 88.7% smaller than Medusa-1's 591M ([arXiv 2404.18911](https://arxiv.org/abs/2404.18911)).
- Prompt lookup / PLD on Spec-Bench gets 1.56x to 1.66x overall but 2.30x to 2.66x on summarization, where outputs copy the input ([Spec-Bench Leaderboard](https://github.com/hemingkx/Spec-Bench/blob/main/Leaderboard.md)).

### Inferences
- CLLMs show a target can be trained to make its own multi-token guesses consistent, but they modify the target. A CLLM-style objective applied only through a gated adapter (as in H1) would keep exactness, although no paper verified this combination.
- Layer-skip self-speculation on a 1.5B model has few layers to skip. The 1.3x to 2.2x reported on larger models is unlikely to beat a dedicated drafter here.

### Gaps
- Draft&Verify (arXiv 2309.08168) was not fetched. Its commonly cited "up to 1.99x" figure is unverified this session.
- CLLM tokens-per-step and exact models were not on the abstract page.

## Q5. Distillation for drafters (DistillSpec, online speculative decoding) and which losses help acceptance

### Takeaway
Distillation helps by about 10% to 45% wall-clock over an undistilled draft. The data the loss is computed on matters more than the divergence. On-policy (drafter-generated) or target-generated sequences with white-box logits clearly beat fixed data. The best divergence depends on task and decoding mode, and directly optimizing TVD was not best in most DistillSpec settings despite its theoretical link to acceptance. For H3, keep soft-label KD, but make the data on-policy and treat the divergence as a tuned hyperparameter, not a central claim.

### Cited Findings
- DistillSpec tested FKL, RKL, JSD and TVD. Best on XSum greedy was mixed draft+target data with JSD slightly ahead. GSM8K greedy favored FKL with draft-generated data. GSM8K at T=1 favored RKL with target-generated data. The authors conclude the divergence should be tuned per task and decoding strategy. Directly optimizing TVD "was not best in most settings" ([arXiv 2310.08461 HTML](https://arxiv.org/html/2310.08461)).
- DistillSpec data findings. Model-generated data beat fixed ground truth, and supervised KD on fixed data ranked lowest. Draft-generated data was comparable to or better than target-generated data at much lower cost. SeqKD (target samples with one-hot labels) performed poorly, so white-box logits matter ([arXiv 2310.08461 HTML](https://arxiv.org/html/2310.08461)).
- DistillSpec speedup is about 10% to 45% over standard SD, largest under greedy. A GSM8K-distilled draft transferred to 23 BigBenchHard tasks, raising speedup from 1.93x to 2.21x (greedy) and 1.78x to 2.02x (sampling). Models were T5-XL 3B target with T5-Small 77M draft, and a 234M/33M GPT-like pair on LM1B ([arXiv 2310.08461 HTML](https://arxiv.org/html/2310.08461)).
- DistillSpec theory. Token-level acceptance equals 1 - TVD(p, q). Theorem 4.1 bounds sequence acceptance by the on-policy KD loss. The appendix proof's final bound carries an extra factor of 2 relative to the theorem statement ([arXiv 2310.08461 HTML](https://arxiv.org/html/2310.08461)).
- Online speculative decoding (continual KD of the draft on live query traffic) reports token acceptance gains of 0.1 to 0.65 and 1.42x to 2.17x latency reduction ([arXiv 2310.07177](https://arxiv.org/abs/2310.07177)).
- HASS's "harmonized objective distillation" aligns the drafter's training objective with the decoding objective and contributes to its 8% to 20% gain over EAGLE-2 ([arXiv 2408.15766](https://arxiv.org/abs/2408.15766)).
- Practice in current strong drafters. DFlash trains on about 800K samples whose responses are regenerated by the target, with plain position-weighted CE ([arXiv 2602.06036 HTML](https://arxiv.org/html/2602.06036v1)). ReDrafter uses KD from the target ([arXiv 2403.09919](https://arxiv.org/abs/2403.09919)). Falcon combines Smooth L1 feature regression with a distillation loss ([arXiv 2412.12639 HTML](https://arxiv.org/html/2412.12639)).

### Inferences
- "KL vs TV" is unlikely to be a publishable lever by itself. DistillSpec already found no universal winner. The stronger H3 framing is on-policy, self-conditioned drafter rollouts scored with target soft labels. That merges EAGLE-3's training-time test (on-policy inputs) with DistillSpec's on-policy KD (on-policy targets).
- For lossless sampling at T=1 the relevant quantity is TVD on the drafter's own visited states. DistillSpec's T=1 winner (RKL) is mode-seeking, which plausibly suits a small drafter that cannot cover the target. Test FKL, RKL and TVD under both greedy and T=1 rather than picking one.
- DFlash's exp(-(k-1)/gamma) position weighting is a cheap, verified trick to add to whichever divergence we use. Early positions dominate tau because acceptance is a prefix product.

### Gaps
- DistillSpec's per-divergence acceptance-rate tables (Table C.1) were not in the read portion of the page.
- Online speculative decoding's models and exact loss were not on the abstract page.
- Goel et al. "Direct alignment of draft model" (arXiv 2403.00858, TVD++ loss) was not verified this session.

## Q6. Tree verification (SpecInfer, Sequoia, EAGLE-2 dynamic trees) and gains over chain drafts

### Takeaway
Trees reliably add acceptance length at batch size 1. EAGLE to EAGLE-2 (dynamic, confidence-pruned tree) adds about 0.9 tau and 1.2x speedup on Vicuna-13B, and Falcon's tree attention adds about 1.2 tau. Sequoia's DP-optimized trees report up to 4.04x on Llama2-7B. Gains shrink at high batch where verification compute is no longer free.

### Cited Findings
- Vicuna-13B T=0, EAGLE 3.05x (tau 3.96) vs EAGLE-2 with dynamic draft trees 4.22x (tau 4.83). At T=1, 2.44x (3.45) vs 3.76x (4.41) ([arXiv 2503.01840 HTML](https://arxiv.org/html/2503.01840)).
- Falcon, Vicuna-7B MT-Bench, adding tree attention raises tau from 2.13 to 3.34 and speedup from 2.27x to 3.10x (with CSGD) ([arXiv 2412.12639 HTML](https://arxiv.org/html/2412.12639)).
- Sequoia uses dynamic programming for an optimal speculation tree. It reports up to 4.04x (Llama2-7B), 3.73x (Llama2-13B) and 2.27x (Vicuna-33B) on A100. Offloaded exact Llama2-70B runs at 0.56 s/token on L40, 9.96x over its own offloading baseline. It claims robustness across temperatures (no figures in abstract) ([arXiv 2402.12374](https://arxiv.org/abs/2402.12374)).
- At serving scale, EAGLE-3's SGLang speedup drops from 1.81x at batch 2 to 1.38x at batch 64. PARD falls from 3.63x at batch 1 to 1.33x at batch 16 ([arXiv 2503.01840 HTML](https://arxiv.org/html/2503.01840), [arXiv 2504.18583 HTML](https://arxiv.org/html/2504.18583)). DScale's gains over DFlash come purely from smarter verify-length allocation at concurrency 8 to 32 ([arXiv 2609.37532](https://arxiv.org/abs/2609.37532)).

### Inferences
- The Apple "quadratic" decoding is effectively a structured tree over future offsets. Its reported gain over linear decoding fits the tree literature. For H1, comparing against a proper dynamic tree (EAGLE-2 style) is the fair baseline.
- For a 1.5B target at batch 1 the target pass is cheap and memory-bound, so a modest tree (tens of nodes) should be nearly free. This favors tree or quadratic verification over a single chain.

### Gaps
- SpecInfer (arXiv 2305.09781) was not fetched this session, so its 1.5x to 2.8x (distributed) and 2.6x to 3.5x (offloading) figures are unverified here.
- No source gave a controlled chain-vs-tree ablation for a parallel block drafter like DFlash.

## Q7. Verdicts for our hypotheses

### Takeaway
Keep the self-conditioning core of H3, but replace "one self-conditioning pass" with EAGLE-3-style multi-step training-time test, and make KD on-policy with a tuned divergence plus DFlash position weighting. Replace H1's gated-LoRA-in-the-target mechanism with target-feature conditioning of the drafter (DFlash K/V injection) plus verifier-to-drafter information flow from rejected/future positions (DFlow). Add a cheap sequential sampler head (Apple, Hydra) or glancing training (Falcon) to close the remaining within-block coherence gap.

### Cited Findings
- Self-conditioning during training keeps acceptance almost flat with offset (EAGLE-3 Fig. 7), and removing the feature constraint to enable it gives +1.2 to +1.3 tau on LLaMA-3.1-8B ([arXiv 2503.01840 HTML](https://arxiv.org/html/2503.01840)).
- Multi-layer target-feature conditioning adds about +0.8 to +1.0 tau on top (EAGLE-3 ablation). Injecting it into every drafter layer's K/V yields tau 6.5 for a one-pass block drafter on Qwen3-8B ([arXiv 2503.01840 HTML](https://arxiv.org/html/2503.01840), [arXiv 2602.06036 HTML](https://arxiv.org/html/2602.06036v1)).
- Verifier representations at rejected positions, fed to the next drafting round with "self-condition" training, improve acceptance over DFlash at no extra target compute ([arXiv 2609.06498](https://arxiv.org/abs/2609.06498)).
- Sequential conditioning across draft positions. Sampler head ([arXiv 2507.11851 HTML](https://arxiv.org/html/2507.11851)), Hydra up to 1.31x over Medusa ([arXiv 2402.05109](https://arxiv.org/abs/2402.05109)), Falcon CSGD tau 2.78 to 3.34 ([arXiv 2412.12639 HTML](https://arxiv.org/html/2412.12639)).
- The Apple H1 prior art reports tokens/step only, with greedy token matching, averaging 3.17 tokens/step over 13 tasks on Tulu3-8B ([arXiv 2507.11851 HTML](https://arxiv.org/html/2507.11851)).
- KD divergence has no universal winner, and on-policy data matters most ([arXiv 2310.08461 HTML](https://arxiv.org/html/2310.08461)).
- On Qwen2.5 targets, a one-pass parallel drafter beats a vanilla small AR drafter (3B target 2.65x vs 1.55x) ([arXiv 2504.18583 HTML](https://arxiv.org/html/2504.18583)).

### Inferences
- **H1 (target refines its own guesses via gated LoRA on extra positions). Replace the mechanism, keep the idea.** The verified version of "the target helps draft its own future" is target hidden states flowing into a separate drafter (DFlash) and reuse of verifier states at rejected positions (DFlow). Both are lossless by construction because target weights never change, and both report stronger numbers than the Apple method. If H1 is kept, it needs wall-clock numbers, a sampling-mode exactness argument (Apple only describes token matching), and a sampler head. Without those it is a reproduction of 2507.11851.
- **H3 (self-conditioning plus target soft labels). Keep, strengthen.** Change "one self-conditioning pass" to unrolled training-time test over the full block (EAGLE-3) or glancing schedules (Falcon). Our measured "recovers about half" is consistent with doing only one pass. Compute KD on the drafter's own rollouts (DistillSpec on-policy). Treat FKL/RKL/TVD as an ablation, not a headline. Add exp(-(k-1)/gamma) position weighting.
- **Direct fixes for the within-block coherence gap, in order of evidence strength.** (1) Condition the drafter on multi-layer target features via K/V injection (DFlash). (2) Train on own outputs across the full block (EAGLE-3 training-time test, DFlow self-condition). (3) Make positions depend on earlier sampled draft tokens through a tiny sampler MLP (Apple) or sequential heads (Hydra). (4) Glancing distillation (Falcon). (5) Bidirectional attention within the block (DFlash, not separately ablated).
- **Scale caution.** Our 0.5B drafter is about one third of the 1.5B target. Published wins come from drafters of 1 to 5 layers (DFlash, EAGLE-3) or with PARD-style one-pass drafting. A full 0.5B model run autoregressively will be cost-limited no matter how good its acceptance is. A DFlash-style few-layer drafter sharing Qwen2.5-1.5B's embeddings and LM head is the most directly transferable design.
- **Baselines to report.** EAGLE-3 and DFlash (both have open code) plus vanilla 0.5B drafting, in tau and in wall-clock at T=0 and T=1, on GSM8K, HumanEval and MT-Bench. These are the benchmarks shared by most papers above.

### Gaps
- No source tested any of these methods on a Qwen2.5-1.5B target, so transfer of the absolute numbers to our scale is unverified.
- No source isolates the bidirectional-vs-causal within-block attention choice, which would directly measure the coherence gap in a DFlash-type drafter.
