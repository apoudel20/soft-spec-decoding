# ChainFlow (arXiv 2611.08669) and DynaSpec (arXiv 2510.13847) as prior art for drafter-side gist prediction and pruning

Read status. Both papers were read through their arXiv HTML pages (https://arxiv.org/html/2611.08669, v1, and https://arxiv.org/html/2510.13847, v3). Each page went through a fetch tool that summarizes, so short quoted phrases below are the tool's direct extractions and longer content is close paraphrase. Items marked [verbatim] were returned as direct quotes. Items marked [paraphrase] come from the extraction but were not quoted. Items marked [abs] come only from the arXiv abstract page. Items marked [HF card] come from the Hugging Face model card. The PDFs were not opened separately. Figures (for example DynaSpec Fig. 3) could not be read.

## ChainFlow: what is compressed, what is predicted, how tokens are drafted and verified, results

### Takeaway
ChainFlow trains a transformer VAE that compresses the target's final-layer hidden states roughly per token (2560 to 640 dims for Qwen3.5-4B, and 4096/5120 to 1024 for 9B/27B). A one-step flow-matching drafter predicts K=8 future latents from C=8 context latents. Those latents are decoded to hidden states and pushed through the target's frozen lm_head over a fixed 62,642-token shortlist. The 8,192 candidates kept each round are picked from the decoded future hiddens, so the predicted future does prune the token search. The result is 1.87x on Qwen3.5-4B (1.11x over DFlash and 1.39x over EAGLE-3) and 2.12x on 27B, at batch size 1 only. Tree verification is greedy-only and not bitwise lossless in practice.

### Cited Findings
**Identity**
- Title "ChainFlow: Few-Step Latent Flow Matching for Speculative Decoding". Authors Selim Aktas, Yusuf Erdem Erdem, Ahmed Zeer (corresponding), Himmet Toprak Kesgin, Muzaffer Kaan Yuce, Mehmet Fatih Amasyali, of Koç University and Yıldız Technical University. Version v1. [paraphrase] — [arXiv HTML](https://arxiv.org/html/2611.08669)
- The abstract claims "ChainFlow outperforms EAGLE-3 and DFlash in vLLM by 39% and 11%, respectively" on Qwen3.5 at matched drafter size and training data, and says gains widen under sequential language adaptation. [verbatim fragment] — [arXiv HTML](https://arxiv.org/html/2611.08669)

**What the VAE compresses**
- ChainFlow "compresses the target's final hidden states" into a latent space. The encoder and decoder are "two-layer bidirectional transformers over windows of up to 16 states". The drafter uses the posterior mean, so encoding is deterministic. KL weight β = 3e-5. [verbatim fragments + paraphrase] — [arXiv HTML](https://arxiv.org/html/2611.08669)
- Hidden size D is 2560 (4B), 4096 (9B) and 5120 (27B). Latent width Z is 640 (4B) and 1024 (9B, 27B), which is 4x, 4x and 5x channel compression. [paraphrase] — [arXiv HTML](https://arxiv.org/html/2611.08669)
- Tokens per latent. The paper writes future latents as z ∈ R^(K×Z), one row per drafted position, so there is one latent per token and compression is along the channel dimension only. The 16-state window is the attention context. This reading is the extraction tool's inference from the notation. The paper does not state it outright. — [arXiv HTML](https://arxiv.org/html/2611.08669)
- The HF card says the VAE is "unfrozen and trained jointly with the flow for acceptance rather than reconstruction, with a small reconstruction anchor". [HF card] — [Flow-Drafter-4B](https://huggingface.co/selimaktas/Flow-Drafter-4B)

**What the flow drafter predicts**
- A conditional flow network predicts the next K = 8 future latents, using C = 8 context latents as cross-attention memory. [paraphrase] — [arXiv HTML](https://arxiv.org/html/2611.08669)
- The source distribution is a linear extrapolation of the last two context latents (Eq. 2), not Gaussian noise. Training uses N = 2 Euler steps and serving uses N = 1. [paraphrase] — [arXiv HTML](https://arxiv.org/html/2611.08669)
- There are two "chunk experts", each covering 4 consecutive draft positions, with no routing. [paraphrase] — [arXiv HTML](https://arxiv.org/html/2611.08669)
- Drafter sizes are about 0.53B (4B target), 1.16B (9B) and 1.64B (27B). [paraphrase] — [arXiv HTML](https://arxiv.org/html/2611.08669)

**How latents become draft tokens, and the candidate restriction**
- Decoded hidden states go through the target's frozen lm_head. A path head over the last p = 8 parent tokens and a rank-256 Markov head restore dependence on the chosen prefix. [paraphrase] — [arXiv HTML](https://arxiv.org/html/2611.08669)
- The head is restricted to "a shortlist of the 62,642 tokens that occur in the training corpus". [verbatim] — [arXiv HTML](https://arxiv.org/html/2611.08669)
- "One matrix product of h^1:K against the shortlist selects a shared set of M=8192 candidate ids." The candidate set is selected from the predicted future hidden states across all K positions. The scoring and aggregation rule is not specified. [verbatim] — [arXiv HTML](https://arxiv.org/html/2611.08669)
- "two-pass rescoring of the top M=8192 candidates raises throughput to 611.7 tokens/s (13%)" versus 540.4 for shortlist-only, with τ about 2.78 either way. [verbatim + paraphrase] — [arXiv HTML](https://arxiv.org/html/2611.08669)
- Section 4.5 mentions a "per-language shortlist" for Turkish, which is not reconciled with the single 62,642-token shortlist. [paraphrase] — [arXiv HTML](https://arxiv.org/html/2611.08669)

**Drafting structure and verification**
- The chain length is 3. The headline tree is 8×4 (8 nodes per depth, depth 4, 32 nodes), verified in one target pass with an ancestor attention mask. [paraphrase] — [arXiv HTML](https://arxiv.org/html/2611.08669)
- "In both modes a token is emitted only if the target's own logits select it, up to fp16 ties." "Chains are verified by standard rejection; tree verification ... supports greedy decoding only." [verbatim] — [arXiv HTML](https://arxiv.org/html/2611.08669)
- In Limitations, "4–13% of prompts produce different outputs from plain greedy decoding", attributed to "fp16 near-ties during batched verification". The measurements "do not establish bitwise equivalence to plain decoding." [verbatim] — [arXiv HTML](https://arxiv.org/html/2611.08669). The HF card instead calls verification "lossless" [HF card] — [Flow-Drafter-4B](https://huggingface.co/selimaktas/Flow-Drafter-4B)
- Evaluation ran on "one NVIDIA B300 GPU with vLLM 0.31.0" with a patched engine for trees. [verbatim] — [arXiv HTML](https://arxiv.org/html/2611.08669)

**Results**

Table 1 covers Qwen3.5-4B in English at batch size 1, pooled over 8 domains (mean τ / speedup).

| Method | Mean τ | Speedup |
|---|---|---|
| EAGLE-3 | 2.49 | 1.35x |
| DFlash | 2.35 | 1.69x |
| ChainFlow chain | 2.01 | 1.23x |
| ChainFlow tree 8×4 | 2.78 | 1.87x |
| ChainFlow tree 5×7 | 2.73 | 1.75x |

Source: [arXiv HTML](https://arxiv.org/html/2611.08669)

- The tree leads DFlash in every domain. The smallest margin is translation (1.39x vs 1.38x). — [arXiv HTML](https://arxiv.org/html/2611.08669)
- Table 3 (scaling, 8×4 tree) reports τ / chain speedup / tree speedup of 2.78 / 1.23x / 1.87x for 4B, 2.66 / 1.30x / 1.78x for 9B and 2.73 / 1.61x / 2.12x for 27B. The flow pass takes 0.30 to 0.38 ms versus target AR steps of 3.1, 4.2 and 11.2 ms. — [arXiv HTML](https://arxiv.org/html/2611.08669)
- At 27B, per-domain tree speedup ranges from 3.23x on math (highest) to 1.74x on translation (lowest). — [arXiv HTML](https://arxiv.org/html/2611.08669)
- Turkish after adaptation reaches 1.81x (4B), 1.87x (9B) and 2.25x (27B). An unadapted English drafter is a net loss on Turkish at 4B and 9B. — [arXiv HTML](https://arxiv.org/html/2611.08669)
- Table 9 (concurrency, 4B) shows 1.85x at 1 request, 1.12x at 4, 0.94x at 8 and 0.45x at 128. EAGLE-3 and DFlash stay above 1x at every level shown. — [arXiv HTML](https://arxiv.org/html/2611.08669)

**Training**
- English training uses 45,887 prompts (16.6M tokens) across 8 domains. Turkish, German and French get 8,000 rows each, added sequentially without English replay, plus 2,088 Turkish replay rows. Responses are the target's own greedy continuations. Optimization is AdamW at 5e-4 constant for 15 epochs with the VAE at 1/20 of that LR and batches of about 6K windows. No GPU-hours are reported. — [arXiv HTML](https://arxiv.org/html/2611.08669)
- The HF card lists different training data (gsm8k, nemotron-math, nemotron-stem, alpaca-code, dolly-chat). Its acceptance figures are much higher (for example gsm8k τ 5.25 at 4B), and it gives a 9B mean vLLM speedup of 1.27x. These are presumably a different checkpoint and benchmark (RedHatAI speculator_benchmarks) than the paper. [HF card] — [Flow-Drafter-4B](https://huggingface.co/selimaktas/Flow-Drafter-4B)

**Ablations**
- Token feedback (0.8B target, GSM8K) gives τ 2.68 / 0.96x for context-only flow, 2.93 / 1.01x with the Markov head, and 3.21 / 1.00x for AR. — [arXiv HTML](https://arxiv.org/html/2611.08669)
- Two experts plateau at the same level as four. A noise source collapses at N=1 (τ 1.97) while extrapolation gives τ 2.49. Served N=2 raises τ from 2.78 to 2.83 but lowers throughput from 611.7 to 580.1 tok/s. The 8×4 tree is fastest (1.87x) and 5×7 slowest (1.75x). — [arXiv HTML](https://arxiv.org/html/2611.08669)

**Limitations (author-stated)**
- Gains shrink with concurrency. Draft length and topology are fixed. The compact VAE latent may cap acceptance. Only Qwen3.5 was tested (hybrid attention needed engine work). Several ablations use only a 0.8B target. Drafters may be undertrained. Tree mode is greedy-only on a patched engine and not bitwise identical. — [arXiv HTML](https://arxiv.org/html/2611.08669)
- Related work does not mention DynaSpec, FR-Spec, VocabTrim or semantic-planning drafters. It cites latent flow and diffusion work (Rombach, Esser, Dao), LangFlow and continuous AR models. [paraphrase] — [arXiv HTML](https://arxiv.org/html/2611.08669)

### Inferences
- ChainFlow's "latent" is a per-token, channel-compressed copy of the target's last-layer hidden state, not a multi-token semantic gist. It predicts the target's future representations (8 positions ahead) on the drafter side, which is the closest prior art to "predict the target's future".
- Pruning by the predicted future exists only as an efficiency trick. The single matmul of all K predicted hiddens against the shortlist selects 8,192 candidates shared across positions, which is a dynamic vocabulary cut driven by the predicted future. It is not framed or analyzed as gist-conditioned search-space pruning, and the selection rule is not specified.
- The fixed 62,642-token shortlist is static (corpus occurrence), which is FR-Spec-like.
- The headline speedup holds only at batch size 1. Above about 8 concurrent requests the drafter is a net slowdown.

### Gaps
- The exact scoring and aggregation rule for the M=8192 candidates (union of per-position top-k, or max over positions) and the two-pass rescoring algorithm are not described in the HTML.
- Whether any latent ever summarizes more than one token is not stated explicitly. The one-latent-per-token reading is inferred from notation.
- No GitHub repository was found. The HF cards reference a "chained-flow" project and its `load_tree_module`, but no URL was given. The HF model cards exist for 4B, 9B and 27B.
- There are no training compute figures and no sampling (temperature > 0) results for trees.
- The discrepancy between the paper and the HF card on data and acceptance numbers is unresolved.

## DynaSpec: routing, clusters, granularity, shortlist size, results, losslessness

### Takeaway
DynaSpec partitions the vocabulary offline with spherical k-means on the LM-head weights. A two-layer MLP router, fed only the current token embedding and the drafter's previous hidden state, picks the top-k clusters at every draft step, with a budget that is larger at early positions and decays later. The drafter's LM head is computed only over the union of those clusters (about 28K tokens for Llama-3-8B, about 20K for Qwen-2-7B). The target verifies over the full vocabulary, so decoding stays exact. The router has no access to any future or semantic summary.

### Cited Findings
**Identity**
- Title "DynaSpec: Context-aware Dynamic Speculative Sampling for Large-Vocabulary Language Models". Authors Jinbin Zhang, Nasib Ullah, Erik Schultheis, Rohit Babbar. Labeled "A Preprint" with no venue. — [arXiv HTML v3](https://arxiv.org/html/2510.13847)
- Versions are v1 (11 Oct 2025), v2 (10 Nov 2025) and v3 (3 Feb 2026). [abs] — [arXiv abs](https://arxiv.org/abs/2510.13847)

**Clusters**
- The vocabulary is partitioned offline into M ≪ |V| clusters by spherical k-means on column-normalized LM-head weights. "We do not enforce balance (i.e., |Cm| may vary)." [verbatim fragment] — [arXiv HTML](https://arxiv.org/html/2510.13847)
- The value of M is not stated anywhere in the HTML. — [arXiv HTML](https://arxiv.org/html/2510.13847)

**Router (meta-classifier)**
- The router r_θ is "a two-layer feed-forward neural network with ReLU activation". Its inputs are the current token embedding and the previous drafter hidden state. It selects the top-k clusters and the shortlist is their union. [verbatim + paraphrase] — [arXiv HTML](https://arxiv.org/html/2510.13847)
- It is trained with multi-label per-cluster BCE (sigmoid). Labels come from EAGLE-2 runs on prompts "e.g., ShareGPT and UltraChat". A positive is any cluster holding a token from the plausible next-token set ("top-L tokens under the full-vocabulary drafter"). "Intuitively, a cluster is labeled positive if it contains at least one high-probability token for that context." [verbatim] — [arXiv HTML](https://arxiv.org/html/2510.13847)
- The router runs on a separate CUDA stream, overlapping the drafter's attention and MLP. Index selection and the head GEMM are fused in a custom kernel. — [arXiv HTML](https://arxiv.org/html/2510.13847)

**Granularity and budget**
- Routing runs at every draft step. In multi-token or tree contexts it takes the union of clusters over tokens. The cluster budget k_c(t) is position-aware (Eq. 3) with k0 > k1, "hyperparameters controlling the initial cluster budgets". Values are not given. — [arXiv HTML](https://arxiv.org/html/2510.13847)
- Average shortlist size is about 28K for Llama-3-8B (Table 2 labels it "DynaSpec-27k") versus FR-Spec's fixed 32K, and about 20K for Qwen-2-7B. — [arXiv HTML](https://arxiv.org/html/2510.13847)

**Drafter, models, setup**
- The drafter is EAGLE-style with one transformer block and reuses the target's frozen embedding and LM head. Targets are Llama-3-8B-Instruct (128K vocab) and Qwen-2-7B (152K vocab). Evaluation covers six Spec-Bench subtasks (80 prompts each) plus HumanEval (164), on a single A6000, implemented in the FR-Spec C++/CUDA framework (https://github.com/thunlp/FR-Spec). No DynaSpec-specific repo is given. — [arXiv HTML](https://arxiv.org/html/2510.13847)

**Results**
- Table 1 mean acceptance length (Full / FR-Spec-32k / DynaSpec) is 3.89 / 3.64 / 3.83 for Llama-3-8B and 3.59 / 3.37 / 3.53 for Qwen-2-7B. Llama Code is 3.92 / 3.31 / 3.86. DynaSpec recovers 98.4% of full-vocab acceptance versus 93.6% for FR-Spec on Llama. — [arXiv HTML](https://arxiv.org/html/2510.13847)
- Table 2 (Llama-3-8B, tok/s, average) shows target 42.18 (1.00x), EAGLE-2 full vocab 81.47 (1.93x), FR-Spec-32k 89.83 (2.13x) and DynaSpec-27k 94.98 (2.25x). Per task DynaSpec gets MT 92.97, Conv 107.55, RAG 85.55, Math 110.96, QA 87.11, Summ 85.71 and Code 95.01, where FR-Spec gets 81.60 on Code. — [arXiv HTML](https://arxiv.org/html/2510.13847)
- The abstract claims "up to 2.23x throughput gain" versus "1.91x for static approaches", which does not match Table 2 (2.25x and 2.13x average). — [arXiv abs](https://arxiv.org/abs/2510.13847) vs [arXiv HTML](https://arxiv.org/html/2510.13847)
- No Qwen throughput is reported. — [arXiv HTML](https://arxiv.org/html/2510.13847)

**Losslessness and theory**
- "the target model still verifies over the full vocabulary, preserving exactness" under standard accept/reject. [verbatim] — [arXiv HTML](https://arxiv.org/html/2510.13847)
- Theorems 5.1 and 5.3 and Corollary 5.2 bound overlap and expected accepted length. On HumanEval (500 contexts) the shortlist coverage p_c(S) is 0.8544 for FR-Spec and 0.87435 for DynaSpec. The lower-bound terms are -0.0325 and -0.0001, both negative, so they are not informative. — [arXiv HTML](https://arxiv.org/html/2510.13847)

**Ablations and limitations**
- With a position-aware versus fixed budget (Fig. 3), the position-aware budget gives higher acceptance at similar mean shortlist size (28.7k vs 29.2k). Exact values were not readable. A position-aware FR-Spec variant (Table 4) raises acceptance from 3.74 to 3.81 at a vocab of 31,739 versus 32,768. — [arXiv HTML](https://arxiv.org/html/2510.13847)
- There is no dedicated limitations section. The paper notes that a dynamic indexed head adds gathered-matmul overhead and can be slower than a dense fixed head. — [arXiv HTML](https://arxiv.org/html/2510.13847)
- Internal inconsistencies remain. Section 6.2's Qwen "Conversation 3.52 → 3.76" matches the RAG column of Table 1. Table 4 FR-Spec-F Code is 4.11 versus 3.31 in Table 1. — [arXiv HTML](https://arxiv.org/html/2510.13847)

### Inferences
- DynaSpec is pure next-token vocabulary pruning. The router predicts which clusters the current step's next token falls in, from local state. It is conditioned on no forward-looking or semantic summary, and it does not predict anything about the target's future beyond one token.
- Its pruning cuts the drafter's LM-head cost (vocab projection), not the tree or search space over sequences. The gain over FR-Spec is modest (2.25x vs 2.13x on Llama, single A6000, 8B targets).

### Gaps
- M (number of clusters), k0, k1, L, the router's hidden width and its training cost are not stated in the HTML. They may be in the PDF appendix, which was not opened.
- Exact Fig. 3 values are unavailable. No Qwen throughput is reported. v1 and v2 differences were not checked.

## Coverage of "predict the target's future gist on the drafter side, then prune the drafter's search space with it"

### Takeaway
ChainFlow covers the "predict the target's future on the drafter side" half. It predicts K=8 future target hidden states, channel-compressed through a VAE, and incidentally uses them to choose a shared 8,192-token candidate set. DynaSpec covers the "prune the drafter's search space with a learned router" half, but only from current local state for the next token. Neither predicts a compressed multi-token gist, and neither uses one to prune tree branches or vocabulary across a span.

### Cited Findings
- ChainFlow predicts future latents of the target's final hidden states, 8 positions ahead, and decodes them through the target lm_head. — [arXiv HTML](https://arxiv.org/html/2611.08669)
- ChainFlow selects M=8192 shared candidates by one matmul of the predicted future hiddens h^1:K against a static 62,642-token shortlist. — [arXiv HTML](https://arxiv.org/html/2611.08669)
- DynaSpec's router uses only the current token embedding and the previous drafter hidden state to choose clusters per draft step. — [arXiv HTML](https://arxiv.org/html/2510.13847)
- DynaSpec keeps exactness via full-vocab target verification. ChainFlow's tree mode is greedy-only and not bitwise identical in served runs. — [DynaSpec](https://arxiv.org/html/2510.13847); [ChainFlow](https://arxiv.org/html/2611.08669)

### Inferences
- Already covered:
  - ChainFlow predicts target-side future representations with a small drafter over multiple positions in 1 flow step.
  - ChainFlow compresses target hidden states with a VAE (per token, 4 to 5x channel compression).
  - ChainFlow uses predicted futures to define a reduced, shared candidate set per round.
  - DynaSpec learns context-dependent vocabulary pruning via clustered LM-head routing, with position-dependent budgets and lossless verification.
- Not covered:
  - Neither compresses many future tokens into one gist vector (temporal compression).
  - Neither uses a gist explicitly as a pruning signal, such as conditioning a router like DynaSpec's on a predicted future summary, or masking or pruning tree branches inconsistent with the gist.
  - Neither prunes the sequence-level or tree search space, as opposed to the vocabulary.
  - Neither analyzes coverage or recall of gist-based pruning against acceptance.
  - Neither preserves sampling-mode losslessness with tree pruning. ChainFlow trees are greedy-only.
  - Neither evaluates at batch sizes above 1 to 8. ChainFlow regresses at concurrency 8 or more.
- A natural combination to position against both: a DynaSpec-style cluster router conditioned on a ChainFlow-style predicted future latent. Neither paper does this, and ChainFlow does not cite DynaSpec or FR-Spec.

### Gaps
- Neither paper's appendix PDF was read directly. Unstated hyperparameters (DynaSpec M, k0, k1, L) and the ChainFlow candidate-selection rule may be specified there.
- No ChainFlow code repo was located to confirm the candidate-selection implementation.
