# Search-space pruning and candidate restriction for speculative-decoding drafters

Scope: methods that prune or constrain what a drafter (or decoder) can propose, before or during drafting, as prior art for "predict a gist of the target's upcoming text, then prune the drafter's candidate space before it drafts."

Read-depth legend used below:
- [full] = full HTML/paper text read via fetch this session
- [abstract] = arXiv abstract page or abstract-level text only
- [snippet] = search-result summaries of the paper (closer to abstract-only, treat numbers as author claims)
- [prior] = arXiv ID and facts from prior knowledge, not re-verified this session. The report writer should verify before citing.

Static vs context-dependent tags: STATIC = one subset fixed offline. PER-STEP = subset recomputed every draft step. PER-EVENT/SEGMENT = subset updated on triggers or over windows.

## Q1. Drafter vocabulary pruning (FR-Spec, VocabTrim, dynamic or context-dependent vocab subsets, LM-head pruning)

### Takeaway
Static frequency-based drafter-vocab trimming (FR-Spec, VocabTrim, Ben Shoham 2026) is lossless and gives modest gains (about 1.1x over EAGLE-2, up to ~16 to 20% throughput) at a cost of roughly 2 to 7% acceptance length. A 2025 to 2026 wave (DynaSpec, CORAL router, SpecVocab, NanoSpec, EvoSpec) makes the subset context-dependent, mostly per step, using a learned router or a low-rank predictor or in-context history, and recovers nearly all full-vocab acceptance. None of them conditions the subset on a predicted semantic plan or gist of upcoming text. That specific mechanism appears to be open.

### Cited Findings

STATIC methods
- FR-Spec (arXiv 2502.14856, ACL 2025, Tsinghua/thunlp). Restricts the EAGLE-2 drafter's LM head to a frequency-ranked token subset while verification uses the full vocabulary. Cuts LM-head compute by 75%. Mathematically lossless (output distribution unchanged). Average 1.12x speedup over EAGLE-2. Llama-3-8B (128k vocab) is the motivating model. Ablation: 8k subset noticeably shortens acceptance length, 64k subset keeps ~97.7% of full-vocab acceptance length. Code github.com/thunlp/FR-Spec. [snippet] — [arXiv 2502.14856](https://arxiv.org/abs/2502.14856), [ACL Anthology](https://aclanthology.org/2025.acl-long.198)
- Independent measurement of FR-Spec acceptance loss. On Spec-Bench average with EAGLE-3, mean accepted length 3.78 (full vocab) vs 3.54 (FR-Spec), i.e. about 6% loss. On 70B specialised domains 3.08 vs 2.96. On LMSYS multi-turn traces FR-Spec 3.41. [full, from EvoSpec tables] — [EvoSpec 2605.27390](https://arxiv.org/html/2605.27390v2)
- VocabTrim (arXiv 2506.22694, Goel et al., ICML 2025 workshop listing). Training-free. Rebuilds drafter LM head over frequent tokens chosen by sampling the target on a calibration set. LM head can be >30% of a small drafter's parameters with the Llama-3 vocab. Reduces LM-head output dimension by up to 75% with "negligible" acceptance impact (v1). Reported elsewhere as up to 16% speedup on Spec-Bench for Llama-3. Lossless because target verifies. STATIC. [snippet] — [arXiv 2506.22694](https://arxiv.org/abs/2506.22694v2), [follow-up summary in 2603.05210](https://arxiv.org/html/2603.05210)
- Balancing Coverage and Draft Latency in Vocabulary Trimming (arXiv 2603.05210, Ofir Ben Shoham, Mar 2026). Treats draft vocab selection as constrained optimisation: coverage over assistant responses in training data vs FLOPs-estimated LM-head latency, searched with a Tree-structured Parzen Estimator under a minimum-coverage constraint. Up to 97% draft-vocab reduction. Domain-specific tasks: up to 16% latency reduction, 20% throughput gain. Out-of-distribution: up to 6.7% throughput. Acceptance rates and losslessness not stated on the abstract page. STATIC (inferred) and explicitly domain-specific. [abstract] — [arXiv 2603.05210](https://arxiv.org/abs/2603.05210)

CONTEXT-DEPENDENT methods (most relevant to the gist idea)
- DynaSpec (arXiv 2510.13847, Zhang, Ullah, Schultheis, Babbar, v3 Feb 2026). Lightweight coarse-grained meta-classifiers route each context to a few token clusters. The union of top-k clusters forms the drafter shortlist. Routing runs in parallel with drafting on a separate CUDA stream. Target verifies over full vocab, "preserving exactness" (lossless). Llama-3-8B: 98.4% of full-vocab mean accepted length vs 93.6% for fixed shortlists. Up to 2.23x throughput on a rare-token dataset vs 1.91x for static. Baselines FR-Spec and VocabTrim, which it calls corpus-dependent and suppressive of rare/domain tokens. Granularity (per step vs per segment) not stated in the abstract, described only as "context-aware". This is the closest prior art in form: a learned predictor picks a token subset from context before drafting. [abstract] — [arXiv 2510.13847](https://arxiv.org/abs/2510.13847)
- CORAL (arXiv 2502.16880, Weng et al., ACL 2025). Besides cross-step representation alignment, splits the drafter LM head into N equal groups (v = V/N each) and uses a router to activate only some groups. Router costs a slight drop in acceptance length due to imperfect routing, so it is used only for large-vocab, smaller models (Qwen2.5, Llama3 at 7B and 14B). Overall CORAL speedups 2.50x to 4.07x, above EAGLE-2 and HASS (speedup is from the whole method, not router alone). Lossless (standard verification). PER-STEP routing (inferred from design). [snippet] — [arXiv 2502.16880](https://arxiv.org/html/2502.16880v3), [ACL Anthology](https://aclanthology.org/2025.acl-long.278)
- Note on citation confusion. EvoSpec cites "CORAL (Xie et al., 2024) arXiv 2410.09675" as order-agnostic refinement with LM-head routing, while NanoSpec cites CORAL as Weng et al. 2025 with a trained cluster router. The router-over-LM-head-groups paper found this session is 2502.16880 (Weng et al.). Treat 2410.09675 as possibly a different paper or a mis-citation. — [EvoSpec](https://arxiv.org/html/2605.27390v2), [NanoSpec](https://arxiv.org/html/2605.26444)
- SpecVocab, "Speculative Decoding with a Speculative Vocabulary" (arXiv 2602.13836, Williams, Kwon, Li, Kouris, Venieris, Findings of ACL 2026). PER-STEP. Projects the draft hidden state to a low dimension d' (1/8 or 1/16 of draft dim), computes approximate logits over the full vocab, keeps top-k (default k = 2048), then computes exact logits only for those. Ranking module trained by distillation from the target (full-vocab softmax term weighted λ = 0.1 best). Models Qwen3 4B/8B and OLMo 2 1B/7B on Spec-Bench, baselines re-implemented on EAGLE-3 with 32K vocab. Qwen3 8B: acceptance 5.01 vs 4.78 (EAGLE-3 repro) and 4.84 (VocabTrim). Throughput 245.2 vs 235.1 tok/s. Up to 8.1% throughput over EAGLE-3 (OLMo 2 7B), 3.1 to 5.4% over VocabTrim. Note acceptance goes up vs the 32k-vocab EAGLE-3 baseline because out-of-vocab target tokens become reachable. Lossless implied, not explicitly stated. [full] — [arXiv 2602.13836](https://arxiv.org/html/2602.13836v2)
- NanoSpec (arXiv 2605.26444, Chen, Xu, Zhang, Wang, Xu, Ma, v2 Jun 2026). Training-free, PER-STEP in-context active vocab. Candidate stream seeded with prompt tokens plus target top-3 at each prompt position, then appends all unique draft-tree tokens and target top-3 verification candidates. Active set = unique tokens in last 3072 stream entries (recency eviction). Average active vocab <3k tokens (vs 128k Llama-3, 32k FR-Spec). Weight rows gathered asynchronously on a second CUDA stream, ~24 MB overhead. Lossless. EAGLE-2 backbone, Llama-3.1-8B: 392.7 tok/s (2.25x AR, τ 3.59) vs FR-Spec 369.7 (2.12x, τ 3.62) vs DynaSpec 367.7 (2.13x, τ 3.73). Llama-3.2-1B: 1.35x vs FR-Spec 1.17x and DynaSpec 1.15x. EAGLE-3 Llama-3.1-8B: 430.0 vs 396.8 (FR-Spec) vs 362.2 (EAGLE-3). Draft time −51.6% vs EAGLE-2. Qwen-2-7B τ 3.48 vs full-vocab 3.65. Code (HumanEval): 381.1 tok/s vs DynaSpec 372.4, FR-Spec 336.3, EAGLE-2 332.0. Gains on code are moderate because outputs are short and the candidate stream needs ~5 steps to warm up. No formatting-specific analysis. [full] — [arXiv 2605.26444](https://arxiv.org/html/2605.26444)
- EvoSpec (arXiv 2605.27390, Zhang et al., Xidian). Active vocab = static top-32k plus a dynamic buffer of up to 256 tokens. On a miss (verified target token outside active vocab), CPU-side async expansion retrieves "long-tail" tokens via HNSW semantic neighbours over the target LM-head embeddings plus a token co-occurrence graph, ARC eviction. Also online LoRA adaptation of the drafter by KL distillation on top-K target logits. PER-EVENT. Qwen3-4B with EAGLE-3 and Qwen3-1.7B drafts, Llama-3.1-70B/8B, Llama-3.2-3B/1B. Specialised domains (code, law, medicine) ~1.13x relative speedup (baseline ambiguous between FR-Spec and EAGLE-3 in Table 2), CORAL 1.10x. Spec-Bench MAL 3.65 (1024 dynamic) vs FR-Spec 3.54 vs full 3.78. LMSYS traces 3.67 vs FR-Spec 3.41. Losslessness not formally proved, main results greedy (T=0). Relevant to the gist idea because it uses semantic embedding neighbourhoods to expand the candidate set. [full, truncated at appendix H] — [arXiv 2605.27390](https://arxiv.org/html/2605.27390v2)

Adjacent LM-head pruning (not drafters)
- Dynamic Vocabulary Pruning in Early-Exit LLMs (arXiv 2410.18952, Vincenti et al., NeurIPS 2024 ENLSP). Maps hidden states to full vocab only at the first exit(s), takes top-K, and uses that pruned vocab for confidence estimation at later exits. PER-TOKEN, context-dependent. Lossy in the usual early-exit sense (no verification). [snippet] — [arXiv 2410.18952](https://arxiv.org/abs/2410.18952)
- SpecEE (arXiv 2504.08850) frames the vocabulary as the search space for early-exit predictors and says the LM head is ~20% of inference overhead at Llama2 vocab size, proposing a lightweight predictor. [snippet] — [arXiv 2504.08850](https://arxiv.org/pdf/2504.08850v1.pdf)

### Inferences
- The acceptance cost of static trimming is about 2 to 7% of τ at 32k to 64k subsets on Llama-3 (FR-Spec 97.7% at 64k, ~93.6 to 93.7% in DynaSpec's and EvoSpec's measurements). Context-dependent methods close most of that gap (98.4% DynaSpec, NanoSpec within ~0.03 τ of FR-Spec at <3k tokens).
- Speedup gains from vocab pruning are bounded by the LM-head share of draft time. They are larger for small drafters/targets (Llama-3.2-1B gains are relatively bigger in NanoSpec) and for large-vocab models (Llama-3, Qwen).
- The predictor families already explored are: cluster routers (DynaSpec, CORAL), low-rank logit approximators (SpecVocab), in-context recency sets (NanoSpec), and embedding-neighbour retrieval on miss (EvoSpec). A gist predictor that outputs a semantic summary of upcoming text and maps it to a token subset would be a new member, and DynaSpec and SpecVocab are the baselines it must beat. The key differentiator would be per-segment (gist horizon) prediction amortised over many draft steps rather than per-step routing.
- All target-verified variants are lossless by construction, since only the proposal distribution changes. Losslessness under sampling (T>0) requires the drafter's q to be renormalised over the subset and used consistently in the rejection test. This is standard but worth stating in the writeup.

### Gaps
- DynaSpec read was abstract-only. Its routing granularity, cluster count, benchmark names and speedup baseline were not confirmed.
- FR-Spec and VocabTrim per-benchmark acceptance and speedup tables were not read in full this session.
- No paper found that picks the vocab subset from a predicted plan, topic or summary of upcoming text.
- No found work reports vocab-pruning results specifically on markdown or formatting-heavy text.

## Q2. Token-tree and candidate pruning, early-exit drafting, draft-length prediction, predicting acceptance before drafting

### Takeaway
Tree pruning is dominated by confidence-as-acceptance-proxy (EAGLE-2), optimised static or per-step tree shapes (Sequoia, OPT-Tree, DySpec), and learned per-token acceptance classifiers (C2T, SpecDec++ acceptance head). These prune breadth and depth of the draft but never the token vocabulary itself. All are lossless.

### Cited Findings
- EAGLE-2 builds a dynamic draft tree using drafter confidence as an approximation of acceptance rate, growing only promising branches. Lossless. [snippet, survey] — [Survey 2411.13157](https://arxiv.org/pdf/2411.13157). EAGLE-2 arXiv ID 2406.16858 [prior].
- OPT-Tree (arXiv 2406.17276, TACL 2025). Adaptive draft-tree structure constructed and adjusted every decoding step to maximise expected acceptance length. Lossless. PER-STEP. [snippet] — [arXiv 2406.17276](https://arxiv.org/pdf/2406.17276)
- Sequoia applies global dynamic programming to construct an optimal tree (hardware-aware, static per setting). Lossless. [snippet, via C2T] — [C2T 2502.13652](https://arxiv.org/html/2502.13652). Sequoia arXiv ID 2402.12374 [prior].
- DySpec proposes a dynamic token tree structure. [snippet] — [DySpec PDF](https://www.wict.pku.edu.cn/docs/20250717175203347541.pdf). arXiv ID 2410.11744 [prior].
- C2T (arXiv 2502.13652). Classifier-based tree construction. A small classifier predicts per-token confidence from features, pruning first by predicted confidence then top-K. Argues that when tree size is matched, EAGLE-2's acceptance-length benefit is only 11%. Lossless. PER-STEP learned pruning. [snippet] — [arXiv 2502.13652](https://arxiv.org/html/2502.13652)
- Dynamic Depth Decoding (arXiv 2409.00142) extends dynamic tree depth for EAGLE-style drafters. [snippet, title only] — [arXiv 2409.00142](https://arxiv.org/pdf/2409.00142)
- Talon (arXiv 2601.07353) is listed as confidence-aware adaptive token trees. Not opened. [snippet, title only] — search result list in [survey query](https://arxiv.org/pdf/2411.13157)
- SpecDec++ (arXiv 2405.19715, Huang, Guo, Wang, COLM 2025 / ICML 2024 workshop). Draft length as an MDP with a threshold policy. A trained acceptance-prediction head on the draft model stops speculation when predicted P(at least one rejection) exceeds a threshold. Llama-2-chat 7B/70B: Alpaca 2.04x vs 1.90x fixed-K, GSM8K 2.26x (+9.4%), HumanEval 2.23x (+11.1%). Lossless. Code github.com/Kaffaljidhmah2/SpecDec_pp. [snippet] — [arXiv 2405.19715](https://arxiv.org/abs/2405.19715)
- Self-speculative / early-exit drafting (Kangaroo 2404.18911, LayerSkip 2404.16710, SWIFT 2410.06916) use shallow layers of the target as the drafter. These IDs are [prior] and not verified this session.

### Inferences
- "Predicting acceptance before drafting" exists at token or block granularity (SpecDec++ head, C2T classifier, EAGLE-2 confidence) but these predict whether a drafted token will be accepted, not which region of vocabulary the target will use. A gist predictor would be complementary and could also feed a draft-length policy (long drafts when gist confidence is high).
- Tree pruning and vocab pruning compose: vocab pruning lowers per-node cost and tree pruning lowers node count. NanoSpec and FR-Spec already run on EAGLE-2/3 trees.

### Gaps
- Exact speedup numbers for EAGLE-2, Sequoia, OPT-Tree, DySpec were not re-read this session.
- No data found on "adaptive draft length" methods beyond SpecDec++ (e.g. DISCO, AdaEAGLE were not checked).

## Q3. Retrieval and n-gram candidate sets, suffix-automaton drafters, mixture or routing of drafters

### Takeaway
Retrieval drafters restrict candidates to continuations seen in the prompt, generated text, or a datastore. They are lossless, training-free, and strongest on repetitive or formatting-heavy output (code, SQL, agentic loops). Routing between drafters per round or per token is now an active 2025 to 2026 topic.

### Cited Findings
- SAM-Decoding (arXiv 2411.10666, Hu et al.). Two suffix automata, static over a corpus and dynamic over prompt plus generated text, drafts from longest-suffix matches. No draft model or GPU needed for drafting. O(1) average update and retrieval (v3). Spec-Bench: 2.27x over AR when combined with Token Recycling, 2.49x with EAGLE-2 (v2). Claims ~18% faster than PLD and REST on Spec-Bench and up to 1.3x over baselines on HumanEval (v3). Lossless. Per-context (dynamic automaton grows each step). [snippet] — [arXiv 2411.10666](https://arxiv.org/html/2411.10666v3)
- SuffixDecoding (arXiv 2411.04975, Oliaro, Jia, Campos, Qiao, Snowflake/CMU). Model-free, suffix trees over previous outputs and the current request, builds speculation trees. Up to 4.5x speculative speedup on agentic workloads (v2). Text-to-SQL production app: up to 2.9x higher output throughput and 3x lower latency vs model-based SD (v1). Matches SD on open-ended chat. In Snowflake ArcticInference. Lossless. [snippet] — [arXiv 2411.04975](https://arxiv.org/html/2411.04975v2)
- Hierarchical Drafting, HD (arXiv 2502.05609, Findings of NAACL 2025). Training-free, draws drafts from three databases ordered by temporal locality (context, model, statistics). Lossless. [snippet] — [arXiv 2502.05609](https://arxiv.org/html/2502.05609)
- Graft, "Draft Less, Retrieve More" (arXiv 2605.20104). Hybrid tree mixing parametric drafter and retrieval proposals, all verified under the standard rule. [snippet] — [arXiv 2605.20104](https://arxiv.org/html/2605.20104v1)
- NanoSpec cites prompt lookup decoding (Saxena, 2023, GitHub, no arXiv) and LAD (Wang et al., 2025) as evidence of temporal locality in generated tokens, which motivates in-context vocabularies. [full] — [NanoSpec](https://arxiv.org/html/2605.26444)
- MetaSD (arXiv 2604.05417, ACL 2026 Findings). Multiple drafters in one loop, drafter choice as a multi-armed bandit with block divergence reward. [snippet] — [arXiv 2604.05417](https://arxiv.org/html/2604.05417v1)
- HedgeSpec / Not-a-Bandit (arXiv 2510.20064, ICLR 2026). Drafter selection as full-information online learning. Feedback for all drafters computed without extra target calls. Works with single, multi-draft and tree methods. [snippet] — [arXiv 2510.20064](https://arxiv.org/html/2510.20064v1)
- WhiFlash (arXiv 2606.07710, ACL 2026 Findings). Token-level routing between an AR drafter and a diffusion drafter, with runtime-informed or offline-learned neural policy. [snippet] — [arXiv 2606.07710](https://arxiv.org/pdf/2606.07710)
- SpecRouter (NeurIPS 2026 listing). Learned cost-aware selection among heterogeneous drafters. Details not visible. [snippet, title only] — [NeurIPS 2026](https://neurips.cc/virtual/2026/178993)
- REST (2311.08252), Lookahead decoding n-gram pools (2402.02057), Token Recycling (2408.08696), Spec-Bench (2401.07851). IDs [prior], not re-read.

### Inferences
- Retrieval drafters are an extreme form of context-dependent candidate restriction: the candidate set is the set of continuations of the current suffix in a datastore. A gist predictor could act as a retrieval key (choose which datastore segment or which document to match against) rather than as a vocab mask.
- Drafter routers (MetaSD, HedgeSpec, WhiFlash) decide which proposal mechanism to use per round. A gist signal ("upcoming text is a code block / table / list") is a natural routing feature, which none of these papers uses as far as the snippets show.
- Formatting-heavy text: evidence is strongest for SuffixDecoding (SQL, agentic) and NanoSpec/SAM on HumanEval. Markdown specifically is not benchmarked in anything found.

### Gaps
- No per-task breakdown for markdown or formatted chat outputs found.
- REST and Lookahead numbers not re-read.
- No paper found that combines drafter routing with retrieval drafters using a learned content predictor.

## Q4. Plan- or topic-conditioned restriction, and its use inside lossless speculative decoding

### Takeaway
Plan conditioning exists in language modelling (Cornille et al. planners, planning tokens), in lossy step-level speculation (Lookahead Reasoning, ConfSpec), and in hard constraint masks (grammar-guided decoding applied to drafters). I found no lossless speculative-decoding work where a predicted plan, outline, keyword set or semantic summary restricts the drafter's candidate vocabulary. This is the gap the gist idea would occupy.

### Cited Findings
- Learning to Plan for Language Modeling from Unlabeled Data (arXiv 2404.00614, Cornille, Moens, Mai, KU Leuven, 2024). A planner trained self-supervised predicts the next abstract writing action from context, and the LM is finetuned to condition on it. Improves LM performance, especially text structure. Plans one sentence ahead. Not a speculative-decoding method. [snippet] — [arXiv 2404.00614](https://arxiv.org/abs/2404.00614)
- Learning to Plan Long-Term for Language Modeling (arXiv 2409.00070) extends this to multi-step plans with multiple hypotheses. End-to-end Planner Training for Language Modeling (arXiv 2410.12492, Cornille, Mai, Sun, Moens) trains planner and LM jointly. [snippet] — [arXiv 2409.00070](https://arxiv.org/html/2409.00070v1), [arXiv 2410.12492](https://export.arxiv.org/pdf/2410.12492)
- Scaling Speculative Decoding with Lookahead Reasoning (arXiv 2506.19830, Fu, Ge, Shao, Deng, Zhang, Jun 2025). Step-level speculation: a drafter proposes whole reasoning steps, accepted if a judge or embedding model deems them semantically equivalent to the target's step. Raises SD speedup from 1.4x to 2.1x on GSM8K, AIME and others while preserving answer quality. LOSSY at the token level (semantic acceptance). Code github.com/hao-ai-lab/LookaheadReasoning. [snippet] — [arXiv 2506.19830](https://arxiv.org/abs/2506.19830v1)
- ConfSpec (arXiv 2602.18447) adds confidence-gated verification to step-level speculative reasoning and argues verification is the bottleneck. LOSSY. [snippet] — [arXiv 2602.18447](https://arxiv.org/html/2602.18447v1)
- Grammar-constrained decoding inside SD. Unconstrained drafters waste drafts on grammar-violating tokens, so TensorRT-LLM applies XGrammar or LLGuidance masks to draft logits. The grammar state advances per draft token and rolls back on rejection, tree drafts need traversal of grammar states, and with EAGLE-3 a d2t mapping converts target-vocab masks to the draft vocab. Lossless with respect to the constrained target distribution. Context-dependent PER-STEP mask, but from a given grammar, not a predicted one. [full-ish, blog] — [NVIDIA TensorRT-LLM blog](https://nvidia.github.io/TensorRT-LLM/1.3.0rc26/blogs/tech_blog/blog12_Combining_Guided_Decoding_and_Speculative_Decoding.html)
- FactorSpec (arXiv 2611.02297) describes a grammar object with fill/accept/rollback where the grammar supplies a mask row per draft node, and serves exact relational constraints as speculative drafts. [snippet] — [arXiv 2611.02297](https://arxiv.org/pdf/2611.02297)
- Searches for "keyword-guided drafter", "plan-conditioned drafting" and "summary-conditioned speculative decoding" returned no matching papers. Closest hierarchical SD results (HiSpec 2510.01336, HD 2502.05609) are hierarchical in drafter/verifier stacking, not in semantic planning. [snippet] — [HiSpec](https://arxiv.org/html/2510.01336v2), [HD](https://arxiv.org/html/2502.05609)
- Skeleton-of-Thought (2307.15337) and planning tokens (2310.05707) are outline-then-expand and plan-token methods that change the output (lossy). IDs [prior], not re-read.

### Inferences
- The gist idea sits at the intersection of (a) context-dependent drafter vocab (DynaSpec, SpecVocab, NanoSpec, EvoSpec), (b) planners that predict abstract future content (Cornille et al.), and (c) step-level semantic speculation (Lookahead Reasoning). Unlike (c), it keeps exact token-level verification, so it would be lossless. Unlike (a), its predictor operates at segment horizon and in semantic space.
- Grammar-mask SD is the strongest existence proof that a context-dependent hard mask on the drafter is compatible with lossless verification and with tree drafts, and it already has rollback machinery that a gist-mask could reuse.
- Risk, by analogy to FR-Spec's 8k ablation: if the gist-derived subset is too small or the gist is wrong, acceptance collapses on rare tokens. EvoSpec-style miss-triggered expansion or a fallback to full vocab on low gist confidence is a likely necessity.
- Strongest baselines for a paper on this idea: FR-Spec (static), DynaSpec (learned router), SpecVocab (per-step low-rank predictor), NanoSpec (training-free in-context), all on EAGLE-2/3 with Llama-3 and Qwen3 on Spec-Bench plus HumanEval.

### Gaps
- No lossless SD paper found using a predicted plan/topic/keyword set to restrict drafter candidates. Absence is based on four targeted searches, not an exhaustive review.
- Keyword-constrained decoding lines (e.g. NeuroLogic, directed beam search) were not checked this session.
- No measurement found of how much the target's next-k-token vocabulary is predictable from a semantic summary (e.g. coverage of a gist-derived subset), which would be the key feasibility number.
