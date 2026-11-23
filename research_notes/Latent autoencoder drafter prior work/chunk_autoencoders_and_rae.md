# Continuous token-chunk autoencoders (CALM and similar), Representation Autoencoders (RAE), and hidden-state autoencoders as prior work for a latent drafter

Proposed design (for reference). Frozen Qwen3-8B target. Cache hidden states at layers 18, 24, 30 and final for positions t..t+k, arrange them as a layers x positions grid with layer and position embeddings, autoencode them (VAE, or RAE-style with the frozen target as encoder and only a decoder trained), pass the decoded final-layer row through the frozen LM head for a token KL loss, train a predictor of the next chunk latent, and sample several latents to build a draft tree that the target verifies exactly.

Stage tags used below: [AE] autoencoder, [PRED] latent predictor, [TREE] sampling or tree construction.

## Q1. CALM, "Continuous Autoregressive Language Models" (Tencent WeChat AI)

### Takeaway
CALM (arXiv 2510.27688, Shao, Li, Meng, Zhou, 31 Oct 2025) compresses K=4 tokens into a 128-dim VAE latent at over 99.9% token reconstruction accuracy using a small (about 75M) autoencoder. Robustness comes from KL with clipping plus dropout and token masking. An energy-score head trained without likelihoods predicts the next latent. It covers [AE] and [PRED] for a token-input autoencoder, but not hidden states and not speculative decoding.

### Cited Findings
- arXiv 2510.27688, submitted 31 Oct 2025, authors Chenze Shao, Darren Li, Fandong Meng, Jie Zhou. It replaces next-token prediction with next-vector prediction and cuts generative steps by a factor of K. — [arXiv abs](https://arxiv.org/abs/2510.27688)
- Autoencoder architecture. Tokens are embedded, passed through a position-wise FFN, flattened, and linearly compressed from Kd to d. A second FFN and a projection produce the latent. The decoder mirrors this and outputs logits through the tied input embedding. d=512, about 75M parameters, trained on a 15B-token Pile subset for 30k steps at a 512k-token batch. — [arXiv HTML](https://arxiv.org/html/2510.27688v1)
- K is in {1, 2, 4, 8} and the latent dim is 32K, so l=128 at K=4. An earlier deterministic version used l=10 at K=4. — [arXiv HTML](https://arxiv.org/html/2510.27688v1)
- Reconstruction accuracy is over 99.9% token-level at K=4. No per-K accuracy table was found in the portion read. — [arXiv HTML](https://arxiv.org/html/2510.27688v1)
- Robustness tricks:
  - Diagonal Gaussian posterior with KL weight beta=0.001.
  - KL clipping with a per-dimension floor lambda_KL=0.5. Without it, 71 of 128 dimensions collapsed.
  - Dropout of 0.15 on the latent and 0.15 token masking on inputs, both disabled after AE training.
  - Posterior std converges to about 0.3.
  - Source: [arXiv HTML](https://arxiv.org/html/2510.27688v1)
- Energy loss. A Monte Carlo energy score with alpha=1, N=8 model samples and M=100 target samples from the posterior. alpha<1 fails to train, and alpha=2 collapses to BrierLM 0. — [arXiv HTML](https://arxiv.org/html/2510.27688v1)
- Generative head:
  - Input is the transformer hidden state plus a uniform noise vector in [-0.5, 0.5], each linearly projected to d.
  - L residual MLP blocks, with L equal to a quarter of the backbone layers. Each block is about 6d^2 parameters (two linears plus SwiGLU).
  - A final linear outputs the l-dim latent. The head is about 10% of total parameters.
  - Source: [arXiv HTML](https://arxiv.org/html/2510.27688v1)
- Input feedback. The predicted latent is decoded to K tokens, which are re-embedded and compressed by a 2-layer MLP into the next input. Feeding the raw latent back scored worse (BrierLM 3.25 vs 4.70). — [arXiv HTML](https://arxiv.org/html/2510.27688v1)
- Backbones (layers/hidden/FFN): S 12/768/2048, M 16/1024/2752, L 16/1536/4096, XL 16/2560/6880. — [arXiv HTML](https://arxiv.org/html/2510.27688v1)
- Results at K=4 (params, train FLOPs 1e20, inference FLOPs per token 1e8, BrierLM):

  | Model | Params | Train FLOPs (1e20) | Inference FLOPs/token (1e8) | BrierLM |
  |---|---|---|---|---|
  | Transformer-S | 281M | 6.6 | 4.4 | 6.05 |
  | Transformer-M | 465M | 11.9 | 7.9 | 7.07 |
  | Transformer-L | 849M | 22.5 | 15.0 | 8.98 |
  | CALM-M | 371M | 3.7 | 2.9 | 5.72 |
  | CALM-L | 735M | 7.7 | 4.6 | 6.58 |
  | CALM-XL | 1.82B | 19.5 | 9.4 | 8.53 |

  CALM-M is close to Transformer-S with about 44% fewer training FLOPs and 34% fewer inference FLOPs. K=8 lost performance, which the authors attribute to capacity. — [arXiv HTML](https://arxiv.org/html/2510.27688v1)
- BrierLM:
  - Brier(P, y) = 2P(y) - sum P(x)^2, estimated from two samples as 1{x1=y} + 1{x2=y} - 1{x1=x2}.
  - Brier-n treats n-grams as atomic outcomes. BrierLM = 100 x the geometric mean of Brier-1..4.
  - Correlation with cross-entropy across checkpoints is Pearson -0.966 and Spearman -0.991.
  - Source: [arXiv HTML](https://arxiv.org/html/2510.27688v1)
- Temperature sampling:
  - An exact rejection algorithm (Algorithm 1) splits 1/T into integer and fractional parts.
  - A batch approximation (Algorithm 2) is asymptotically unbiased. It needs about N=100 samples to match T=0.6 and about N=200 for T=0.5.
  - Source: [arXiv HTML](https://arxiv.org/html/2510.27688v1)
- Speculative decoding appears only in related work. CALM does not evaluate it. — [arXiv HTML](https://arxiv.org/html/2510.27688v1)
- An author blog post exists. — [Blog](https://shaochenze.github.io/blog/2025/CALM/)

### Inferences
- Closeness to the proposal. CALM's AE input is token embeddings, not hidden states, and its decoder is a trained one tied to the embeddings, not the target's frozen LM head. Its chunk latent and its sample-able generative head (noise input, energy loss) are a direct template for [AE] and [PRED].
- CALM's head can draw many latent samples cheaply. That makes it a natural [TREE] source: sample N latents, decode each to K tokens, and merge them into a tree. CALM itself never does this.
- CALM's lesson that a raw latent is a poor next input (3.25 vs 4.70 BrierLM) suggests conditioning the drafter on target hidden states or token embeddings, not on its own predicted latent alone.
- The KL floor and latent dropout are the concrete tricks to copy so the decoder tolerates predictor error. Imperfect predicted latents are exactly the drafter's case.

### Gaps
- Per-K reconstruction accuracy (K=1, 2, 8) was not found in the portion of the paper read (first 100k of 148k characters).
- No published results apply CALM to LLMs above 1.82B parameters, to hidden states, or to speculative decoding.

## Q2. Other token-chunk, sentence and context autoencoders (latent size vs k vs reconstruction)

### Takeaway
Near-lossless reconstruction is achievable at modest ratios. Examples: CALM at 4 tokens to 128 dims, ICAE at 4x on Llama-7B with 99.1 to 99.8 BLEU, and TextLDM at 1:1 tokens to 64 or 128 channels with 97.6 to 100% accuracy. The extreme case is per-sample optimization, where Llama-3.1-8B decodes 1568 tokens from one input vector. Trained encoders at high ratios (500x) are lossy. Sentence spaces like SONAR are reported as fragile for generation.

### Cited Findings
- **Cramming 1568 tokens** (arXiv 2502.13063, Kuratov, Arkhipov, Bulatov, Burtsev). [AE] decoder side only.
  - Method. Per-sample optimized [mem] input vectors are fed to a frozen LLM, which decodes the text. AdamW lr 0.01, up to 5000 steps.
  - Max tokens from 1 vector at 0.99 accuracy on PG-19:

    | Model | Max tokens from 1 vector |
    |---|---|
    | Pythia-160M | 80 |
    | Pythia-1.4B | 160 |
    | Llama-3.2-1B | 512 |
    | Llama-3.2-3B | 1024 |
    | Llama-3.1-8B | 1568 |

    Random text gives lower limits (792 for Llama-3.1-8B).
  - Llama-3.2-1B decodes 7168 tokens from 16 vectors.
  - Capacity is governed by the text's uncertainty (unconditional cross-entropy), not its length. The theoretical bound is L <= d_model*b/log2|V|.
  - Source: [arXiv abs](https://arxiv.org/abs/2502.13063), [arXiv HTML](https://arxiv.org/html/2502.13063)
- **ICAE** (arXiv 2307.06945, Ge, Hu, Wang, Wang, Chen, Wei, Microsoft). [AE] for context compression.
  - Setup. A LoRA encoder (r=128 on Q and V, about 1% extra parameters) writes k=128 memory slots for 512 tokens (4x). The frozen LLM decodes.
  - Reconstruction BLEU/loss (pretrained, 512 to 128):

    | Target LLM | BLEU (%) | Loss |
    |---|---|---|
    | Llama-7B | 99.1 | 0.017 |
    | Llama-2-7B | 99.5 | 0.009 |
    | Llama-2-13B | 99.8 | 0.004 |

  - Near-perfect reconstruction up to about 300 tokens. At 500 tokens, median BLEU is above 0.98 and median EM is near 0.6.
  - Continuation perplexity goes from 9.01 to 9.50 at 4x.
  - Source: [arXiv abs](https://arxiv.org/abs/2307.06945), [arXiv HTML](https://arxiv.org/html/2307.06945)
- **500xCompressor** (arXiv 2408.03094, Li, Su, Collier). [AE] lossy.
  - Compresses contexts into as few as 1 token at 6x to 480x, with about 0.3% extra parameters.
  - KV values preserve information better than embeddings at high ratios.
  - QA retains 62.26 to 72.89% of uncompressed performance.
  - Source: [arXiv abs](https://arxiv.org/abs/2408.03094)
- **Large Concept Models / SONAR** (arXiv 2412.08821, Meta). [AE] frozen sentence encoder plus [PRED].
  - The concept is one sentence embedded in SONAR (up to 200 languages).
  - Variants: Base-LCM (MSE regression, regresses to averages), diffusion LCMs (One-Tower and Two-Tower, clearly better on pretraining metrics), and Quant-LCM (RVQ).
  - Ablations ran at about 1.6B parameters, with scale-up to 7B on about 2.7T tokens.
  - The SONAR space is fragile. Some embeddings decode poorly and predictions can leave the manifold. Fragility-weighted training helped.
  - Source: [arXiv HTML](https://arxiv.org/html/2412.08821)
- **TextLDM** (arXiv 2605.07748, Jiang et al.). [AE] plus diffusion [PRED].
  - Tokenizer and latent. Qwen3 tokenizer, one latent per token (no sequence compression), latent channels 32 to 192, TextVAE of 350 to 690M parameters.
  - REPA alignment of intermediate encoder features to a frozen Qwen3-1.7B. The 3rd-to-last layer beat the last layer.
  - Loss is CE + 1e-3 KL + 1.0 REPA.
  - Reconstruction is 97.55 to 97.57% (WikiSource), 97.78 to 97.80% (Wikipedia), 99.58 to 99.60% (TinyStories) and 99.98 to 100% (LM1B). REPA does not change it.
  - REPA does improve diffusion. For the 328M DiT on WikiSource, ROUGE-1 is 32.6 vs 27.8 and MAUVE 20.4 vs 2.5.
  - Diffusion settings are a logit-normal timestep schedule (std 1.5) and 50 Euler steps.
  - Source: [arXiv HTML](https://arxiv.org/html/2605.07748v1)
- **LDLM** (arXiv 2605.07933, Meshchaninov, ..., Korotin, Vetrov). [AE] on frozen LM hidden states plus [PRED].
  - Architecture. The frozen GPT-2 third-to-last-layer hidden states feed trainable 6-layer Perceiver encoder and decoder modules. A separate stop-gradient token decoder and a 12-layer DiT complete the model.
  - OWT-128 Gen-PPL / MAUVE by latent space:

    | Latent space | Gen-PPL | MAUVE |
    |---|---|---|
    | Frozen hidden states (TEncDM-like) | 149.7 | 36.6 |
    | Separately trained AE (COSMOS-like) | 98.5 | 53.4 |
    | Joint training (LDLM) | 66.3 | 89.1 |
    | Real text | 25.7 | n/a |

  - Source: [arXiv HTML](https://arxiv.org/html/2605.07933v1)
- **AURORA-LM** (arXiv 2608.02602). [AE] learned query encoder-decoder plus a block-causal flow-matching denoiser.
  - Block Q=16. The L model has a 1B denoiser.
  - It deliberately does not inherit a pretrained encoder.
  - OWT Gen-PPL 23.56 and MAUVE 0.890 for the S model.
  - Tokens per latent and reconstruction accuracy were not stated on the project page.
  - Source: [Project page](https://aurora-lm-project.github.io/)

### Inferences
- Rough capacity guide for k in 4 to 16 tokens:
  - 128 continuous dims suffice for 4 tokens at 99.9% (CALM).
  - An 8B LM's single 4096-dim input vector can carry far more (Cramming) when optimized per sample. So capacity is not the bottleneck for a chunk of 4 to 16 hidden-state positions.
  - The bottleneck is smoothness and predictability of the latent (LCM fragility, LDLM's frozen-vs-trained gap, TextLDM's REPA effect).
- The LDLM table is the most direct evidence on the RAE-style choice for language. Using frozen LM hidden states directly as the diffusion space was markedly worse than a trained or jointly trained latent. This argues for at least a trained compressing encoder on top of the frozen target features, not a pure RAE.
- TextLDM finding that a late-but-not-final layer is the better alignment target fits the plan to include mid layers (18, 24, 30).

### Gaps
- Not fetched in this pass, so these details are unverified recollection and must be checked before citing:
  - Gist tokens (arXiv 2304.08467, Mu et al., reported up to 26x prompt compression).
  - PLANNER (arXiv 2306.02531, Apple, paragraph-level latent diffusion with a fixed number of latent codes).
  - LD4LG (arXiv 2212.09462, Lovelace et al., Perceiver compression to a small fixed latent set).
  - xRAG (arXiv 2405.13792, one-token retrieval embedding).
  - COSMOS and TEncDM (arXiv IDs not verified).
- SONAR's own autoencoding BLEU numbers were not found in the portion of the LCM paper read.
- 500xCompressor's base LLM and regeneration numbers were not in the abstract.

## Q3. RAE, "Diffusion Transformers with Representation Autoencoders", and follow-ups

### Takeaway
RAE (arXiv 2510.11690, Zheng, Ma, Tong, Xie, NYU, ICLR 2026) pairs a frozen DINOv2-B encoder (256 tokens x 768) with a trained ViT decoder. It reconstructs better than SD-VAE (rFID 0.49 vs 0.62). Diffusion in this 768-dim token space needs three fixes plus a wide shallow head: DiT width at least the token dim, a dimension-dependent noise-schedule shift, and decoder noise augmentation. The head is DiT-DH. Language follow-ups that use a fully frozen LLM as the encoder are rare. The closest found are LDLM and TextLDM, and LDLM's evidence is negative for pure frozen features.

### Cited Findings
- arXiv 2510.11690, 13 Oct 2025, Boyang Zheng, Nanye Ma, Shengbang Tong, Saining Xie. Accepted to ICLR 2026. — [arXiv](https://arxiv.org/abs/2510.11690v1), [ICLR page](https://iclr.cc/virtual/2026/poster/10011890), [Project](https://rae-dit.github.io/)
- Encoders are DINOv2-B (patch 14, dim 768, default), SigLIP2-B and MAE-B. The decoder is a ViT-XL by default, trained with L1 + LPIPS + GAN losses. — [arXiv HTML](https://arxiv.org/html/2510.11690v1)
- rFID: DINOv2-B 0.49, SigLIP2-B 0.53, MAE-B 0.16, SD-VAE 0.62. Linear probe accuracy: DINOv2-B 84.5% vs SD-VAE 8.0%. Decoder cost: ViT-B 22.2 GFLOPs (rFID 0.58) vs SD-VAE decoder 310.4 GFLOPs. — [arXiv HTML](https://arxiv.org/html/2510.11690v1)
- Standard DiT fails on RAE latents. DiT-S gets gFID 215.76 (vs 51.74 on SD-VAE) and DiT-XL gets 23.08 (vs 7.13). — [arXiv HTML](https://arxiv.org/html/2510.11690v1)
- Width rule:
  - A DiT of width d only fits a single image when d >= the token dim n. Extra depth at width 384 did not help.
  - Theorem. If d < n, the loss is lower-bounded by the sum of eigenvalues lambda_{d+1..n}.
  - Overfit losses with DINOv2-B: DiT-S 5.2e-1, DiT-B 2.4e-2, DiT-L 1.3e-3.
  - Source: [arXiv HTML](https://arxiv.org/html/2510.11690v1)
- Noise shift. t_m = alpha t_n / (1 + (alpha-1) t_n), with alpha = sqrt(m/n), base n=4096 and m = tokens x dim. gFID goes from 23.08 to 4.81 with the shift. The fetch summarizer computed alpha of about 6.9 for 256x768, which the paper does not state. — [arXiv HTML](https://arxiv.org/html/2510.11690v1)
- Decoder noise augmentation. The decoder trains on z + n with n ~ N(0, sigma^2 I) and sigma ~ |N(0, tau^2)|. gFID improves from 4.81 to 4.28 while rFID worsens from 0.49 to 0.57. The tau value is ablated in an appendix that was not read. — [arXiv HTML](https://arxiv.org/html/2510.11690v1)
- DiT-DH. The base DiT is followed by a wide, shallow head of 2 layers at width 2048.
  - FID on DINOv2 S/B/L: DiT-XL 3.50/4.28/6.09 vs DiT-DH-XL 2.42/2.16/2.73.
  - DiT-DH hurts on low-dim SD-VAE (11.70 vs 7.13).
  - Source: [arXiv HTML](https://arxiv.org/html/2510.11690v1)
- Headline results on ImageNet:
  - 256 without guidance: 1.51 gFID at 800 epochs, 2.16 at 80 epochs.
  - With AutoGuidance: 1.13 at 256 and 1.13 at 512.
  - About 47x faster training convergence than SiT-XL and 16x faster than REPA-XL.
  - The ICLR page lists 1.18 at 256, which conflicts with 1.13, so the number may differ by version.
  - Sources: [arXiv HTML](https://arxiv.org/html/2510.11690v1), [ICLR page](https://iclr.cc/virtual/2026/poster/10011890)
- Follow-up: "Scaling Text-to-Image Diffusion Transformers with Representation Autoencoders" (arXiv 2601.16208, Xie co-author). It uses a SigLIP-2 RAE in a MetaQuery-style LLM pipeline and reports faster convergence and better quality than VAE baselines. — [arXiv HTML](https://arxiv.org/html/2601.16208)
- Another follow-up: Multiplayer Interactive World Models with RAEs (arXiv 2607.05352). Found in search but not read. — [arXiv PDF](https://arxiv.org/pdf/2607.05352)

### Inferences
- How the RAE rules translate to the proposal:
  - Latent width. A Qwen3-8B hidden state is 4096-dim, and a 4-layer x (k+1)-position grid has m = 4 x (k+1) x 4096 dims (for example about 131k at k=7). By RAE's rule, a predictor diffusing directly in that space needs width at least 4096 per token. That is near the target's own width and defeats a cheap drafter.
  - Compression. This favors compressing to a small latent (CALM: 128 dims for 4 tokens, ChainFlow: 1024 for Qwen3.5-9B), or using a DiT-DH-style wide shallow head only at the output.
  - Noise shift. If diffusion or flow is used, apply RAE's dimension-dependent shift, with alpha = sqrt(m/4096) where m is the latent dimensionality.
  - Decoder noise. RAE's decoder noise augmentation and CALM's latent dropout and KL floor serve the same purpose. They make the decoder robust to predictor error, and the proposal should adopt one of them.
- The pure RAE option (frozen target as encoder, train only the decoder) has a structural problem for language. The target's hidden states already contain the information, so the decoder is trivial for the final-layer row (the LM head itself). The hard part, predicting future states, is left entirely to the predictor in a high-dimensional, possibly non-smooth space. LDLM's result (frozen hidden states 149.7 Gen-PPL vs joint 66.3) is evidence that this hurts generative modeling in text.

### Gaps
- No paper was found that applies the full RAE recipe (frozen LLM encoder, trained decoder only, width rule, noise shift, DiT-DH) to language modeling or to LLM hidden states.
- The RAE tau value and the appendix ablations were not read.

## Q4. Autoencoding an LLM's own hidden states and decoding through the frozen LM head

### Takeaway
ChainFlow (arXiv 2611.08669, Nov 2026) is the closest existing work, close enough that the proposal needs to position against it. It trains a VAE over windows of a frozen Qwen3.5 target's final hidden states and predicts future latents with 1-2 step rectified flow. It decodes to hidden states, applies the frozen lm_head for logits, and drafts a tree verified by the target. It beats EAGLE-3 by about 39% and DFlash by about 11% at 4B. It differs from the proposal in four ways: final layer only, no multi-layer grid, beam-search trees rather than multiple latent samples, and greedy verification only.

### Cited Findings
- **ChainFlow** (arXiv 2611.08669, Aktas, Erdem, Zeer, Kesgin, Yuce, Amasyali, v1 5 Nov 2026). Covers [AE] + [PRED] + [TREE].
  - Targets. Qwen3.5-4B, 9B and 27B with D=2560, 4096 and 5120. Latent widths Z are 640, 1024 and 1024. Drafters are about 529M, 1.16B and 1.64B parameters, excluding the shared frozen embedding and lm_head. Only the target's final hidden states are cached. Multi-layer fusion is not described.
  - VAE. A transformer VAE with 2-layer bidirectional encoder and decoder over windows of up to 16 states. KL beta = 3e-5. It is pretrained on cached states, then fine-tuned jointly with the drafter at 1/20 of the learning rate. The drafter uses the posterior mean.
  - Flow. Rectified flow from a trajectory-extrapolation source, z0(i) = z_t + i(z_t - z_{t-1}). Context is C=8 latents through cross-attention and K=8 draft positions. Training uses 2 Euler steps and serving uses 1. Two chunk experts each have 8 blocks.
  - Feedback heads. A path head over the frozen embeddings of the last 8 parent tokens, and a rank-256 Markov transition bias. Logits come from the frozen lm_head on the decoded hidden states.
  - Loss weights: flow 0.2, hidden-state error 0.25, CE through lm_head 0.3, coverage hinge 0.3, acceptance surrogate 0.15, VAE reconstruction 0.5.
  - Trees. Built by beam search over candidate tokens (shortlist M=8192), with an 8x4 tree of 32 nodes. They are not built from multiple latent samples. Verification is greedy only, with an ancestor attention mask.
  - Results:
    - 4B, pooled English: 1.39x over EAGLE-3 and 1.11x over DFlash (tree).
    - 27B: 2.12x pooled, 3.23x on math.
    - Setup: B300, vLLM, batch 1.
  - Source: [arXiv abs](https://arxiv.org/abs/2611.08669), [arXiv HTML](https://arxiv.org/html/2611.08669)
- **LDLM** encodes frozen GPT-2 hidden states (third-to-last layer) with a trained Perceiver into diffusion latents and decodes back to hidden states, then to tokens with a separate decoder. It does not use the frozen LM head directly. — [arXiv HTML](https://arxiv.org/html/2605.07933v1)
- **TextLDM** aligns VAE features to frozen Qwen3-1.7B hidden states (REPA) rather than autoencoding them. — [arXiv HTML](https://arxiv.org/html/2605.07748v1)
- **ReDrafter** (arXiv 2403.09919) conditions an RNN drafter on the target's last hidden state, with no latent autoencoder. — [arXiv PDF](https://arxiv.org/pdf/2403.09919)

### Inferences
- What remains open relative to ChainFlow:
  1. Multi-layer grid input (layers 18, 24, 30, final) with layer and position embeddings. ChainFlow uses the final layer only.
  2. Explicit token-KL distillation through the frozen LM head. ChainFlow uses CE plus hidden-state error, and a KL to target logits is a small variation.
  3. Trees built from multiple sampled latents, possibly with a CALM-style energy head. ChainFlow decodes the posterior mean deterministically and branches over tokens by beam search.
  4. Lossless sampling-mode verification. ChainFlow is greedy only.
  5. An RAE-style decoder-only ablation.
- Items 3 and 4 are the clearest novelty. Item 1 is an incremental change, and EAGLE-3 already fuses low, mid and high layers, though that is outside this note's sources.
- ChainFlow's latent sizes (Z = D/4 at 4B, D/4 at 9B, D/5 at 27B, per position) give a calibrated starting point for a Qwen3-8B latent (D=4096, so Z of about 1024 per position).

### Gaps
- ChainFlow's VAE reconstruction accuracy, latent-dim ablation, VAE vs no-VAE ablation, and acceptance lengths were not on the pages read. Its ablations were mostly run on a 0.8B target.
- "Frozen LLMs are Native Decoders for High-Norm Semantic ..." (ACL 2026 long, aclanthology 2026.acl-long.1717) appeared in search and looks relevant, but the PDF could not be parsed, so its content is unverified. — [ACL Anthology PDF](https://aclanthology.org/2026.acl-long.1717.pdf)
- "Dynamic Semantic Compression for Efficient Latent-Space Inference in LLMs" (arXiv 2609.15338) and "Enabling Agents to Communicate Entirely in Latent Space" (ACL 2026) appeared in search but were not read.
- No CALM follow-up applying chunk latents to speculative decoding was found besides ChainFlow, which cites Shao et al.
