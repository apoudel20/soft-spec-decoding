# Guided draft-tree construction, multi-draft exact sampling, and target-steered drafting

Scope note. Searched 2026-11-23. Read depth is marked per item.
- [full] = HTML full text fetched and read this session (via a summarising fetch tool, so table values are second-hand but taken from the paper's own tables).
- [abstract/snippet] = only search snippets or abstract pages seen.
- [prior] = from background knowledge, not re-opened this session. arXiv IDs for [prior] items are ones I am confident of, but numbers attached to them should be re-checked before citing.

## Q1. Tree construction methods, and does any condition tree shape on target-provided information?

### Takeaway
Almost every inference-time tree builder (static Medusa/SpecInfer trees, EAGLE-2 confidence expansion, Sequoia/OPT-Tree/C2T/DySpec budget optimisers, RL controllers, hardware-aware expanders) shapes the tree from drafter-side scores, acceptance history, or hardware cost. The closest work to "target steers the tree" is Exit-Guided Speculative Decoding (2610.11750), but it uses the frozen target only at training time to teach the drafter where target mass escapes the tree. I found no method that, at inference time, uses a target-emitted forward-looking signal (future vector, plan, gist, hidden-state uncertainty) to allocate per-branch tree budget. That remains an open slot.

### Cited Findings

Static and confidence-based trees
- Medusa: multiple decoding heads on the target, static hand-designed sparse tree, verified with tree attention. Medusa-1 is lossless under typical/greedy verification. Medusa-2 "typical acceptance" is lossy. arXiv 2401.10774 [prior, ID confirmed as cited in ConFu] — [arXiv](https://arxiv.org/abs/2401.10774)
- SpecInfer: token trees built from one or more small speculative models, tree-based parallel verification, lossless multi-step speculative sampling. arXiv 2305.09781 [prior] — [arXiv](https://arxiv.org/abs/2305.09781)
- EAGLE-2: context-aware dynamic draft tree. It expands and reranks nodes by the drafter's own (well-calibrated) confidence, an approximation of acceptance rate. Lossless. The shape signal is drafter-side only. arXiv 2406.16858 [prior, ID confirmed as cited in ConFu] — [arXiv](https://arxiv.org/abs/2406.16858)
- EAGLE-3 (2503.01840) keeps EAGLE-2 style dynamic trees and changes the drafter to multi-layer feature fusion plus training-time test [prior, ID confirmed as cited in ConFu] — [ConFu related work](https://arxiv.org/html/2603.08899v1)

Optimal-budget and dynamic-structure trees
- Sequoia: dynamic-programming search for the tree that maximises expected accepted tokens under a fixed node budget, using positional acceptance-rate vectors measured offline. Hardware-aware tree size. Lossless sampling/verification robust across temperatures. arXiv 2402.12374 [prior] — [arXiv](https://arxiv.org/abs/2402.12374)
- OPT-Tree: per-step adaptive tree that maximises the mathematical expectation of acceptance length using drafter probabilities. arXiv 2406.17276 [prior]. Used as a baseline by GRAFT (2608.20375) [full] — [GRAFT](https://arxiv.org/html/2608.20375)
- DySpec: dynamic token tree built greedily from draft distribution, motivated by a link between draft probability and acceptance rate. arXiv 2410.11744 [snippet] — [arXiv PDF](https://arxiv.org/pdf/2410.11744)
- C2T (classifier-based tree construction): a small classifier using features such as joint and entropy of draft probabilities prunes candidate tokens dynamically. arXiv 2502.13652 [prior] — [arXiv](https://arxiv.org/abs/2502.13652)
- CAST (inference-cost-aware dynamic trees, 2510.26577), TALON (confidence-aware adaptive token trees, 2601.07353), ECHO (elastic SD, 2604.09603) are listed as tree-construction related work in GRAFT [full, citation only] — [GRAFT](https://arxiv.org/html/2608.20375)
- SMART (2604.09731, Apr 2026): training-free, hardware-aware rule that expands a node only when its marginal benefit-to-cost ratio exceeds the tree-level speedup. Plug-in for MSD and EAGLE. On DFlash trees, average speedup 1.39x to 1.60x at T=0 [abstract/snippet] — [arXiv](https://arxiv.org/pdf/2604.09731)
- BASTION (2605.29727): hardware-calibrated online controller choosing tree size per step from expected acceptance gain vs verification latency, block-diffusion drafter [snippet] — [arXiv](https://arxiv.org/html/2605.29727v1)
- SpecBlock (2605.07243): co-trained rank head assigns per-position branching during drafting, replacing fixed top-k [snippet] — [arXiv](https://arxiv.org/abs/2605.07243)
- Draft Less, Retrieve More (2605.20104): prunes low-confidence draft branches and splits the node budget between drafted and retrieved candidates [snippet] — [arXiv](https://arxiv.org/pdf/2605.20104)
- SAGE (2602.00523, VLMs): starts from static tree and adjusts depth and width by draft prediction entropy each iteration [snippet] — [arXiv](https://arxiv.org/html/2602.00523v1)

Learned / RL tree policies
- Re-SpS (2601.12212): claims to be the first RL framework for draft-tree hyperparameters. Policy sets total-token limit, depth and expansion factor per draft-verify round, with feature reuse to keep overhead low [snippet] — [arXiv](https://arxiv.org/html/2601.12212v1)
- Learning to Draft, LTD (2603.01639, ICLR 2026): two co-adaptive RL policies controlling tree depth and verification size, reward = measured throughput per cycle [snippet] — [arXiv](https://arxiv.org/pdf/2603.01639)
- RADAR (2512.14069): draft-tree generation as an MDP trained with offline RL on acceptance-length distributions from ShareGPT [snippet] — [arXiv](https://arxiv.org/pdf/2512.14069v1)
- Group Tree Optimization, GTO (2509.22134): training-side. Optimises the drafter against the tree decoding policy (scalar tree reward) rather than a single greedy path [snippet] — [arXiv](https://arxiv.org/html/2509.22134v2)

Closest to target-guided allocation
- Exit-Guided Speculative Decoding / ExitTrain (2610.11750, Oct 2026, Hu, Ren, Lu, Zhou) [full]. Problem statement: draft-score tree builders spend budget on branches the target rarely follows and under-expand target-likely branches. Mechanism: during training, build the inference-time tree (DDTree best-first builder, B=64) with the current drafter, freeze its shape, have the frozen target evaluate every node, and compute an exit weight w(v)=P(v)·rho(v) where P is target prefix probability and rho is target mass not covered by v's drafted children. Drafter loss adds an exit-weighted log-likelihood term. Target quantities are detached. The authors call it a "tree-allocation surrogate", not an exact gradient. Also proposes TEV, a verifier that cuts verifier-stage latency about 15%. — [arXiv](https://arxiv.org/html/2610.11750)
- ExitTrain results: about 13% longer output blocks, about 14% end-to-end over DDTree. Qwen3-4B/8B tau +8.8% to +13.8%, speedup +11.5% to +13.8%. LLaMA-3.1-8B tau +17.7%/+10.4% at T=0/1. Gemma-4-12B-IT +5.6%/+4.8%. Qwen3-Coder-30B-A3B +5.2%/+4.3%. Exit-weighted beats token-only fine-tuning by about 5.9%/7.9% tau and prefix-weighted by 7.8%/4.4% (Qwen3-4B). Baselines are DFlash, DDTree, Draft-OPD, not EAGLE-2/3. Claims exactness (Theorem 1 gives a fixed-tree ceiling on expected accepted length, TEV matches the target exit/bonus law) — [arXiv](https://arxiv.org/html/2610.11750)
- Related target-signal training methods cited by ExitTrain: VAT (2608.30135, reweights simulated first-rejection positions), Draft-OPD (2605.29343, position-wise on-policy distillation), DDTree (2604.12989, block-diffusion draft trees), DARTree (2608.13524, autoregressive draft trees) [citation only] — [ExitTrain](https://arxiv.org/html/2610.11750)
- GRAFT (2608.20375) [full]. TDES: a learned parent-child compatibility term added to DFlash scores, a bi-tower over frozen target token embeddings plus depth embedding, distilled from target traces (KL, margin, BCE), about 0.3 ms/round at inference using only token ids. SABA: per-round tree budget by maximising expected gain over cost, with signals = DFlash draft confidence, cosine similarity of adjacent DFlash hidden states ("representation degeneration"), and acceptance history. Parameters fit offline. Results 2.13x to 6.36x over AR on Qwen3-4B/8B/Coder-30B-A3B, one A100. GRAFT sometimes has lower mean accepted tokens than DDTree but higher speedup (Qwen3-8B GSM8K MAT 8.06 vs 8.18, speedup 5.28 vs 4.33). No explicit losslessness statement on the page — [arXiv](https://arxiv.org/html/2608.20375)
- JetSpec (2606.18394): parallel draft head reading frozen target hidden states, builds a tree under a fixed node budget. This is drafter conditioning on target states (EAGLE-style), not target-directed allocation [snippet] — [arXiv](https://arxiv.org/pdf/2606.18394)

### Inferences
- The field's budget-allocation signals cluster into four groups. (a) Drafter confidence or entropy: EAGLE-2, OPT-Tree, DySpec, C2T, SAGE, TALON. (b) Offline acceptance statistics: Sequoia, RADAR. (c) Hardware cost and throughput: Sequoia, SMART, BASTION, LTD, CAST. (d) Target-derived signals used only at training time: ExitTrain, GRAFT TDES, GTO, VAT. No paper found uses a target-emitted forward-looking quantity at inference to say which subtree deserves budget.
- ExitTrain is the strongest prior-art threat for the framing "draft-score builders allocate budget badly relative to the target". A target-steered tree proposal should cite it and position itself as inference-time, per-context steering versus ExitTrain's amortised, training-time correction.
- GRAFT shows a cheap target-derived prior (frozen target embeddings) can reshape trees at about 0.3 ms per round. That suggests a target-provided gist vector consumed by a light scorer is feasible on overhead grounds.
- Because the target already runs a forward pass on every tree node during verification, a target-side "where next" signal can ride along at roughly zero extra target passes. ConFu (Q3) already does this, but feeds the drafter rather than the tree allocator.

### Gaps
- Numeric results for Sequoia, OPT-Tree, C2T, DySpec, Medusa and EAGLE-2 were not re-read this session. Do not quote speedups without checking.
- Full text of Re-SpS, LTD, RADAR, SpecBlock and BASTION was not read. Their state features might include target hidden states. Snippets did not say so.
- I found no paper titled "learned draft tree" or "Tree search for speculative decoding" matching the prompt's wording. The RL papers above are the nearest hits.

## Q2. Multi-draft exact sampling: how much do extra drafts add, and how does it interact with temperature?

### Takeaway
Lossless multi-draft verification (SpecTr OT, RRS with or without replacement, SpecHub, Hu et al.'s optimal/greedy scheme) adds measurable but modest gains. The verifier choice moves acceptance a few points, and drawing drafts without replacement is the main lever. The wall-clock gain over RRS inside an EAGLE tree is roughly 1.0x to 1.17x. At T=0 multi-draft verification collapses to "is the target argmax among the candidates", so its whole theoretical apparatus is a sampling (T>0) phenomenon. Under greedy, the extra drafts only help as coverage, which a top-k tree already provides.

### Cited Findings
- SpecTr (2310.15141, NeurIPS 2023): speculative decoding as optimal transport with membership cost, generalising maximal coupling to k candidates. Optimal LP is exponential in k. Practical selection is (1-1/e)-optimal multiplicatively. v2 abstract reports 2.13x wall clock, a further 1.37x over standard SD. An earlier version said 3x and 1.36x, so versions conflict [abstract] — [arXiv](https://arxiv.org/abs/2310.15141v2)
- Multi-candidate speculative sampling (2401.06706) and Recursive Speculative Decoding (2402.14160, sampling without replacement plus recursive rejection) are the standard RRS baselines [prior] — [arXiv 2401.06706](https://arxiv.org/abs/2401.06706); [arXiv 2402.14160](https://arxiv.org/abs/2402.14160)
- SpecHub (2411.05289, EMNLP 2024): reduces OTM to a compact LP over a sparse joint distribution, linear overhead. Raises second-draft acceptance by 1 to 5% and yields 0.05 to 0.27 more tokens per step than RRS and 0.02 to 0.16 more than RRS without replacement. Claims the same batch efficiency with half the tree nodes. Supports only two drafts [abstract] — [arXiv](https://arxiv.org/abs/2411.05289)
- Towards Optimal Multi-draft SD (2502.18779, ICLR 2025, Hu, Zheng et al.) [full]: computes the optimal acceptance rate via the OT dual, shows sampling without replacement beats with replacement, and gives a greedy draft construction plus a verifier that reaches the bound — [arXiv](https://arxiv.org/html/2502.18779)
- Its Table 1 (T=0.7, 3 drafts) acceptance %, EAGLE-0.24B to Vicuna-7B, Alpaca/CNN-DM/WMT: RRS with replacement 63.4/56.7/32.9, RRS without replacement 70.9/63.4/36.9, greedy 72.6/65.7/39.5. EAGLE-0.26B to Qwen2-7B: 59.6/46.7/38.3 with replacement, 68.3/52.4/43.9 without, greedy 69.9/54.0/45.5. Greedy vs the without-replacement optimum is about -0.1 to +1.4 points, so it is small. Without vs with replacement is large, up to about 7 points — [arXiv](https://arxiv.org/html/2502.18779)
- Its Table 2 (Vicuna-7B, EAGLE framework, MT-Bench), speedups relative to RRS with replacement: T=0.1, 4 drafts 3 steps, greedy 88.8% acceptance, 1.17x. EAGLE sparse tree 1.08x. T=0.6 up to 1.09x. T=1.0, 4 drafts, 1.08x. EAGLE sparse tree 0.97x. SpecHub vs greedy at 2 drafts shows no significant difference (84.0 vs 84.7 at T=0.1) — [arXiv](https://arxiv.org/html/2502.18779)
- The same paper notes that with-replacement methods effectively use one draft at T=0, and that the effect of temperature is non-monotonic, with the gap to optimum widening as T rises. It does not report a single-draft (n=1) baseline in its tables — [arXiv](https://arxiv.org/html/2502.18779)
- UniVer (2605.04543) frames multi-step and multi-draft SD in a unified view. List-level distribution coupling (2506.05632) is another coupling-based multi-draft approach [snippet, titles only] — [UniVer](https://arxiv.org/pdf/2605.04543); [List coupling](https://arxiv.org/pdf/2506.05632)
- Block verification (2403.10444) verifies the draft block jointly instead of token by token, never worse than token-level and lossless [prior] — [arXiv](https://arxiv.org/abs/2403.10444)
- Hierarchical Speculative Decoding, HSD (2601.05724, ICLR 2026 oral): provably lossless hierarchical branch resampling for sequence-level verification that addresses "joint intractability". Average +6.7% decoding speed, up to +12.3% [abstract] — [arXiv](https://arxiv.org/html/2601.05724)
- Traversal verification (2505.12398) verifies trees leaf-to-root for higher acceptance while staying lossless [prior] — [arXiv](https://arxiv.org/abs/2505.12398)

### Inferences
- The marginal value of a better multi-draft verifier is about 1 to 7 acceptance points and up to about 1.17x wall clock in EAGLE trees. That is smaller than the typical gain from better drafters (ConFu +8 to 11%, ExitTrain +13%). Tree shape and draft quality likely matter more than the verification rule.
- For a target-steered tree, the verifier must be a lossless tree verifier (SpecInfer-style multi-step RRS, without-replacement, or the 2502.18779 greedy scheme). Any tree shape, however it was chosen, stays lossless as long as the choice does not depend on the token being sampled at verification time. A target gist computed from already-verified context satisfies this.
- At T=0 the multi-draft verifier question disappears and only coverage matters. Steering is then purely about putting the target's argmax path inside the tree. At T>0 the steering signal should target probability mass (as ExitTrain's rho does), not just the argmax.

### Gaps
- I found no clean published number for "k drafts per position vs 1 draft" at matched node budget in a modern EAGLE-style setting. 2502.18779 does not tabulate n=1.
- SpecTr block efficiency as a function of k was not found in the abstract. Full text would be needed.

## Q3. Target-guided drafting: does the target send something back to steer the next round?

### Takeaway
Yes, two lossless methods do this. ConFu (2603.08899) has the target emit a "future prediction vector" through contemplate tokens during verification. SD2 (2511.09844) builds a steering vector from the verifier's hidden state at the first rejected token. Both feed the signal to the drafter as conditioning and leave the tree budget and shape unchanged. Neither uses it to decide which branches to grow. That gap is exactly the proposed idea.

### Cited Findings
- ConFu (2603.08899, Qin, Goel, Gagrani, Garrepalli, Lee, Sun, ICLR 2026 workshop) [full]. About 16 learnable soft-prompt tokens are prepended to the frozen target's KV cache. A "contemplate token" is appended, and only it attends to the soft prompts. Its hidden state is the future vector f. A dynamic MoE router picks contemplate embeddings conditioned on the last accepted token's hidden state. The drafter (one-layer EAGLE-style) gets f as one extra input token, fixed for the whole drafting pass — [arXiv](https://arxiv.org/html/2603.08899v1)
- ConFu tree and cost: tree size unchanged (30 or 60 nodes). One contemplate token is inserted after every draft node during verification, so the target processes about 2T tokens per step with an adjusted tree mask. The paper calls the overhead "modest" and does not measure it separately — [arXiv](https://arxiv.org/html/2603.08899v1)
- ConFu results on SpecBench, H100, batch 1, vs EAGLE-3: Llama-3.1-8B T=0 30 nodes, tau 4.59 to 5.01 and speedup 2.36x to 2.69x. Llama-3.2-3B, tau 4.00 to 4.41 and 1.83x to 2.11x. Overall 8 to 11% gains, largest at low temperature. Ablation: no MoE 4.97, no MoE and no replication 4.81. The paper's own text claims +12.8% for the 3B setting, which does not match its table (about +10%). Losslessness is asserted in the impact statement, not proved separately — [arXiv](https://arxiv.org/html/2603.08899v1); [ICML 2026 poster](https://icml.cc/virtual/2026/poster/64165)
- SD2, Steering Pretrained Drafters (2511.09844, AAAI 2026): the verification step also produces a steering vector from the verifier's hidden states at the first rejected token, which guides the drafter in the next round. Steering alone reportedly takes accepted tokens per round from about 1.0 to about 2.0 (+100%) for an unaligned pretrained drafter. Unfreezing the drafter does better. Requires white-box access to verifier hidden states [abstract/snippet] — [arXiv](https://arxiv.org/html/2511.09844)
- HASS (2408.15766) and Griffin (2502.11018) fix EAGLE's training/inference mismatch (harmonised representations, token alignment). This is training-side alignment, not a runtime feedback channel [citation in ConFu] — [ConFu related work](https://arxiv.org/html/2603.08899v1)
- DREAM (2505.19201): drafting with refined target features (multi-layer target features with attention) [snippet, title only] — [arXiv](https://arxiv.org/pdf/2505.19201)
- Draft-length control is drafter-side in most work. SVIP stops drafting by draft entropy as a rejection proxy (2411.18462, EMNLP 2025). AdaSD (2512.11280) uses draft entropy for length plus a JS-distance acceptance criterion, which makes it lossy. PEARL (2408.11850) adapts length via parallel pre-verify/post-verify. The search found no method where target entropy sets draft length or tree width [snippet] — [SVIP](https://aclanthology.org/2025.emnlp-main.844.pdf); [AdaSD](https://arxiv.org/pdf/2512.11280); [PEARL](https://arxiv.org/pdf/2408.11850)
- Multi-Drafter SD with Alignment Feedback (ACL Findings 2026) picks among drafters with a multi-armed bandit on acceptance feedback, a different feedback loop [snippet] — [ACL Anthology](https://aclanthology.org/2026.findings-acl.1629.pdf)
- "Your LLM Knows the Future" (2507.11851) shows the target itself has latent multi-token prediction capacity (mask-token fine-tuning) [snippet, title only] — [arXiv](https://arxiv.org/pdf/2507.11851)

### Inferences
- ConFu is the direct ancestor of "target predicts its own future gist". It already computes a per-node future vector during verification (one contemplate token per draft node), so it holds a gist for every tree node, not only the accepted one. The proposed extension is to use those per-node vectors, or a gist at the accepted frontier, to score and allocate the next tree's branches, which ConFu does not do.
- SD2 shows the rejection-point hidden state is informative enough to double acceptance for a misaligned drafter. Its natural use for tree shaping (for example, widen at the rejection-like positions) is unexplored in what I found.
- Both methods stay lossless because the signal only affects what is proposed, never the accept/reject rule. A gist-steered tree inherits this.

### Gaps
- Full SD2 text not read. Model sizes, temperatures and exact speedups are unverified.
- No method found that sends a discrete "plan" or text-level gist back with verification. All found signals are continuous vectors.

## Q4. Hierarchical or two-level trees (semantic/segment level over token level), and are any lossless?

### Takeaway
Step- or sentence-level speculation exists for reasoning models (Lookahead Reasoning, ConfSpec), but it is lossy. It accepts semantically equivalent steps, not distribution-exact ones. Skeleton-of-Thought is a lossy prompting-level parallelism. Lossless "hierarchical" work (HSD, block or traversal verification) is hierarchical in the verification math over token sequences, not a semantic tree of segments. I found no lossless two-level tree where semantic branches have token subtrees as leaves.

### Cited Findings
- Lookahead Reasoning (2506.19830, NeurIPS 2025, Fu, Ge, Shao, Deng, Zhang): the drafter proposes several future reasoning steps (split on "\n\n"). The target expands them in one batched pass, a semantic verifier keeps acceptable steps, and the target regenerates the rest. It is combinable with token-level SD, improving SD speedup from 1.4x to 2.1x on GSM8K, AIME and others while "preserving answer quality". This is lossy (semantic equivalence) [abstract] — [arXiv](https://arxiv.org/abs/2506.19830v1)
- ConfSpec (2602.18447, ACL 2026): confidence-gated cascade where the drafter judges step equivalence and only uncertain cases go to the target. Lossy [abstract] — [arXiv](https://arxiv.org/html/2602.18447v1)
- HSD (2601.05724, ICLR 2026): lossless hierarchical branch resampling for sequence-level verification, +6.7% average speed [abstract] — [arXiv](https://arxiv.org/html/2601.05724)
- Skeleton-of-Thought (2307.15337): the model writes a skeleton, then expands points in parallel. It changes the output, so it is lossy and not speculative decoding [prior] — [arXiv](https://arxiv.org/abs/2307.15337)

### Inferences
- A lossless two-level design is possible if the "semantic level" only shapes which token subtrees are drafted (proposal side) while verification stays token-exact. For example, the target gist selects two or three continuation modes, and each mode gets a token subtree. That combines Lookahead-style branching on direction with SpecInfer-style exact verification. I found no paper doing this.
- Lookahead Reasoning's 1.4x to 2.1x gain comes mostly from accepting non-identical steps. A lossless variant would only get the coverage part of that gain.

### Gaps
- Not checked: whether HSD's "branches" are segments of the draft chain or something closer to a semantic hierarchy. The abstract does not say.
- No search hit for "tree of thoughts for decoding speed" as a speedup (rather than quality) method.
