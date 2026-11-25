[gemini-4-neon via interactions, thinking_level=high, 176s]
[tokens: input 194,810 (cached 0), output+thinking 17,988; call $0.8542; total $4.6166 of $10.00]
[interaction_id: ChA1ZTZjNzJmNGZhNjY4NGY2EAgaAjBkKgRtYWlu]

### Executive Verdict

**Is the claimed distinction real and precisely stated?**
**Partially real in narrow architectural intersection, but imprecisely stated and materially overstated in the deck and project notes.**

Stripped of framing, Gister is the **intersection of four already-published ingredients**:
1. **ChainFlow (arXiv 2611.08669)**: VAE over frozen target hidden states + few-step flow matching to predict the next 8 positions + decoding through the frozen LM head + greedy tree verification.
2. **CALM (arXiv 2510.27688)**: compressing a multi-token chunk into a single continuous VAE latent and decoding the chunk jointly.
3. **FlashMap (arXiv 2610.24365)**: few-step rectified flow conditioned on multi-layer Qwen3-8B target states, with multiple independent noise draws merged into a verified draft tree.
4. **MFD (arXiv 2611.13017)** / **DBLast (arXiv 2608.05448)**: latent-conditioned coherent multi-future mixture trees on Qwen3-8B to eliminate chimera branches in factorized block drafters.

Your own earlier report (`reports/Latent autoencoder drafter prior work.md`) already stated the exact truth that `deck/outline.md` and `AGENT_NOTES.md` now obscure: *the only unoccupied cell is sampling multiple continuous latents of target hidden states into a coherent tree verified losslessly at $T>0$.* **Gister does not even occupy the $T>0$ half of that cell** because Stage 1 failed its distribution gate (TV 0.06–0.20 vs. $\le 0.005\text{–}0.01$).

**Highest-overlap prior works (where reviewers will call Gister incremental):**
1. **FlashMap (2610.24365)** — highest overlap on *mechanism and motivation* (multi-layer Qwen3-8B conditioning + few-step flow + multi-sample coherent trees). Incremental critique: *"Why add a lossy Stage-1 hidden-state VAE when FlashMap flows directly over token blocks, works at $T>0$, and beats DFlash (6.14x vs. Gister's 2.24x)?"*
2. **ChainFlow (2611.08669)** — highest overlap on *architecture* (VAE over target hidden states + flow predictor for 8 steps + frozen LM head + greedy tree). Incremental critique: *"Pooling 8 per-token latents into one 2048-d chunk vector and drawing $K$ noise samples instead of extrapolating + beam search is a direct modification of ChainFlow, currently performing worse."*
3. **MFD (2611.13017)** — highest overlap on *the stated headline framing* ("coherent multi-future tree drafting" on Qwen3-8B). Incremental critique: *"Each decoded latent $z_j$ still outputs per-slot marginals through the LM head, so $K$ samples form a mixture of product distributions—the exact probabilistic object of MFD, replacing $K=8$ cheap discrete mode embeddings (23K params, $\tau=9.47$, 6.51x) with a heavy VAE + flow stack ($\tau=2.0\text{–}3.1$)."*
4. **CALM (2510.27688)** — highest overlap on *Stage-1 representation*, especially once you account for `--headfeat` (see Finding #3).

---

### Findings Ranked by Severity

---

#### 1. [CRITICAL] Complete omission of the three closest competitors (FlashMap, MFD, DBLast) and CALM from the pitch deck and active baseline list
* **Locations:**
  * `deck/outline.md` (Slide 2, Slide 7, Slide 10)
  * `AGENT_NOTES.md` ("Naming and scope", lines 61–66; "Published baselines", lines 88–94)
* **Issue:** `deck/outline.md` frames Gister as *"a working proof of concept for coherent multi-future tree drafting"* and positions it only against DFlash, DSpark, EAGLE, and ChainFlow. It **never mentions FlashMap, MFD, DBLast, DDTree, or CALM**—even though your own report (`reports/Latent autoencoder drafter prior work.md`) explicitly concluded that FlashMap and MFD already own coherent multi-future tree drafting on Qwen3-8B and that Gister *"would read as 'continuous-latent MFD over target hidden states' and would need to beat MFD, FlashMap's multi-sample tree and DDTree in one harness on Qwen3-8B."* Presenting a compute ask that omits the exact papers already solving "coherent multi-future tree drafting" at 6.1x–6.5x on Qwen3-8B looks either uninformed or evasive.
* **Concrete fix / rewording:**
  * Add FlashMap (2610.24365), MFD (2611.13017), DFlash+DDTree (2604.12989), and CALM (2510.27688) to Slide 2, Slide 7, and `AGENT_NOTES.md`'s baseline table.
  * Replace Slide 2's distinction bullet with:
    > *"Positioned at the intersection of ChainFlow (VAE + flow over target hidden states, deterministic beam tree), CALM (single continuous latent per token chunk), and FlashMap / MFD (multi-sample / multi-mode coherent draft trees on Qwen3-8B). Unlike FlashMap (token-simplex flow) and MFD (K discrete mode embeddings), Gister tests whether sampling a chunk-level continuous latent of target states is a better space for coherent tree branching."*

---

#### 2. [CRITICAL] The core empirical claim ("Sampling works as a mechanism" for coherent multi-future trees) lacks the mandatory control—a factorized product tree at matched node budget
* **Locations:**
  * `deck/outline.md` (Slide 5, Slide 6)
  * `RESULTS.md` ("Gister", lines 333–356)
* **Issue:** Slide 5–6 treats +18% to +31% $\tau$ (sampled tree vs. Gister's deterministic chain) as proof that *multi-latent sampling* works as a coherent tree-drafting mechanism. **It proves nothing of the sort.** A standard factorized product tree (DDTree-style top-$k$ per slot, with zero latent sampling and zero cross-slot coherence) already yields **+16% to +35% $\tau$** over a single chain on DFlash (`RESULTS.md`, "Tree headroom on DFlash", 32–64 nodes). Because Gister's deterministic chain is much weaker than DFlash and sits on an averaging floor, *any* multi-branch tree of ~100 nodes will beat it by 20–30%. You have not measured whether Gister's $K$-sample tree beats a matched-node-budget **product tree decoded from Gister v2's deterministic output**, nor whether it beats **MFD-style discrete mode embeddings** on the same 154M backbone. Worse, `deck/outline.md` Slide 10 asks for cluster compute to scale to 1B / 100M+ tokens *before* running the exact kill-test your own report (`reports/Latent autoencoder drafter prior work.md`, "A one-week offline test settles the open piece") set as a prerequisite.
* **Concrete fix / rewording:**
  * Do not claim "sampling works for coherent multi-future drafting" from a tree-vs-chain comparison. Reword Slide 5–6 to: *"Flow sampling produces diverse non-collapsing trajectories (7.6–8/8 distinct) and recovers +18–31% $\tau$ over our deterministic chain. Whether continuous latent sampling beats a matched-budget factorized product tree or MFD-style discrete modes on the same backbone is not yet tested and is Gate 1 before scaling."*
  * Move the matched-budget comparison (mean product tree vs. flow-sampled tree vs. MFD discrete modes) to step (0) of Slide 10, ahead of cluster scaling.

---

#### 3. [CRITICAL] The "multi-layer target hidden-state gist" claim is undermined by your own Stage-1 and Stage-2 ablations (`--headfeat`, middle-layer null, `--lat_w 0`, and confounded token-VAE control)
* **Locations:**
  * `deck/outline.md` (Slide 2: *"from 4 target layers, decoded jointly"*; Slide 3: *"from an 8-token x 4-layer gist"*)
  * `AGENT_NOTES.md` ("Where it stands", line 55: *"The gist beats a tokens-only latent by 13 to 24%... supports the design over a plain token-chunk autoencoder"*; "Stage 1 results", lines 113–121)
  * `RESULTS.md` ("Stage-2 gate", lines 269–285; "Gister", lines 294–296)
* **Issue:** Four results in your own tables severely weaken the claim that Gister's Stage-1 latent is a meaningful *multi-layer hidden-state* representation rather than a heavier, lossier CALM-style token-chunk autoencoder:
  1. **Middle layers did not help Stage 1:** `z1024 full grid` (94.0 / 92.6 / 77.6%) vs. `z1024 final row only` (93.7 / 92.3 / 76.8%)—virtually identical (`AGENT_NOTES.md` literally notes *"middle rows barely matter"*). No final-row-only ablation was run *with* `--headfeat`.
  2. **Stage 1 failed until token identity (`W[top1]`) was handed to the encoder:** Without `--headfeat`, the 4-layer hidden-state VAE stalled at 94.6 / 93.0 / 78.3%. Feeding `W[top1]` (the frozen LM-head row of the argmax token) is what pushed it to 99.4 / 99.0 / 97.3%. That means the encoder relies directly on the discrete top-1 token's embedding, making Stage 1 much closer to a token-chunk VAE (CALM) than a pure hidden-state compressor.
  3. **The "gist beats token-VAE by 13–24%" control is invalid:** `tok_vae.py` was trained with **`zdim=128` (16x smaller!)** and **4k steps (half as long)**—explicitly called *"undertrained"* in `AGENT_NOTES.md`—and never tested with `--raw` target-state conditioning. Claiming this supports hidden-state gists over a token-chunk autoencoder will be rejected on sight.
  4. **In Stage 2, latent-space supervision (`--lat_w`) had zero effect once raw states were added:** `gist + raw` scored 2.80 / 2.23 / 1.43 vs. `--lat_w 0` (token CE loss only) at 2.79 / 2.20 / 1.40 (`RESULTS.md`), while raw multi-layer states (`--raw`, which DFlash/FlashMap/EAGLE-3 already use as conditioning) provided the real jump (+29% math, +24% code).
* **Concrete fix / rewording:**
  * Drop "from 4 target layers" as a differentiating feature of the *latent* (keep multi-layer states honestly described as *Stage-2 conditioning*, where they actually helped, and credit EAGLE-3/DFlash/FlashMap).
  * Remove the claim in `AGENT_NOTES.md` that the 13–24% gap over `tok_vae_z128` validates hidden-state latents over token-chunk AEs; explicitly label that control mismatched (`z=128, 4k steps` vs. `z=2048, 8k steps`).
  * State explicitly on Slide 3 that Stage 1 required `W[top1]` (`--headfeat`) to reach 97.3–99.4% top-1 agreement.

---

#### 4. [HIGH] Double standard on "lossless" / greedy exactness vs. ChainFlow
* **Locations:**
  * `deck/outline.md` (Slide 1, Slide 6: *"Lossless tree verification, end to end... Output identical across drafters"*)
  * `RESULTS.md` ("Gister", lines 329–331)
  * `reports/Latent autoencoder drafter prior work.md` (summary & table)
  * `reports/Drafter side gist prediction and pruning.md` (section 1 & table)
* **Issue:** Both literature reports criticize ChainFlow as *"greedy only and not bit-exact (4% to 13% of prompts diverge from plain greedy decoding, attributed to fp16 near-ties)"* and list exactness (especially at $T>0$) as a key gap Gister can fill. In reality:
  1. Gister is **also greedy-only** (Stage 1 fails the TV/KL gate by 6x–20x, so $T>0$ exactness is untested and impractical with the current AE).
  2. In `RESULTS.md` (`e2e.json`), **Gister diverges from plain greedy on 15/20 (75%) MT-Bench, 9/20 (45%) GSM8K, and 5/20 (25%) HumanEval prompts** due to 4-bit batched-verify numerics—the exact same class of low-precision batched-verify mismatch ChainFlow has, at a *higher* rate! Slide 6 says *"Output identical across drafters"* (which only means Gister and DFlash match *each other* when verified with the same 4-bit batched kernel), omitting the 25–75% divergence from plain AR greedy.
* **Concrete fix / rewording:**
  * Stop listing ChainFlow's greedy-only / fp16-near-tie divergence as a differentiator until Gister actually passes the $T>0$ gate and runs an exact $T=1$ rejection-sampling test.
  * On Slide 6, write: *"Greedy tree verification matches batched-verify target logits identically across all drafters (25–75% of prompts diverge from batch-1 AR greedy at 4-bit NF4 near-ties, identical to DFlash under the same verifier; $T>0$ sampling is not yet supported)."*

---

#### 5. [HIGH] Wrong and unverified descriptions of EAGLE-3, DSpark, DFlash 2, and EAGLE-4
* **Locations:**
  * `deck/outline.md` (Slide 2: *"Distinct from DFlash / DSpark (per-slot block drafts), EAGLE (autoregressive features)..."*; Slide 7)
  * `AGENT_NOTES.md` ("Naming and scope", lines 64–65; "Published baselines", lines 88–94)
* **Issue:** Four factual or substantiation errors in how baselines are characterized:
  1. **EAGLE-3 is NOT "(autoregressive features)":** Your own notes (`gist_drafters.md`, `working_drafters.md`, `feature_drafters_and_latent_branches.md`) emphasize that **EAGLE-3 explicitly abandoned future-feature prediction** in favor of direct token prediction with multi-layer feature fusion and training-time test. Only EAGLE-1/2 predicted features.
  2. **DSpark is NOT merely "(per-slot block drafts)":** Your own notes (`shared_geometry.md` line 154) admit DSpark (2607.05147) was read from *search snippets only*, and `latent_diffusion_text_and_drafters.md` (citing `2610.40847`) notes that **DSpark uses Markov-head and recurrent-head variants** to model within-block token dependencies—the opposite of independent per-slot drafting.
  3. **DFlash 2 and EAGLE-4 were never read:** Zero files in `reports/` or `research_notes/` analyze DFlash 2 or EAGLE-4's architectures (`AGENT_NOTES.md` has only two sparse rows from an HF card, including the suspicious typo/model name `"Qwen3.8-27B"`). Claiming Gister is distinct from methods you have not read is an immediate credibility hit.
  4. **Suspected table conflation for EAGLE-3 in `AGENT_NOTES.md` and Slide 7:** `AGENT_NOTES.md` line 91 lists `EAGLE-3 tree 60 | Qwen3-8B | tau MT-Bench: 3.26 | Speedup: 1.90x (H200) | DFlash paper T1`. Yet `working_drafters.md` line 113 records DFlash Table 1's Qwen3-8B EAGLE-3(60) numbers as **3.40 / 2.02x ($T=0$, 7-task avg)** and **3.26 / 1.88x ($T=1$, 7-task avg)**. Re-verify whether `3.26 / 1.90x` is truly MT-Bench $T=0$ or a misread of the $T=1$ average row.
* **Concrete fix / rewording:**
  * Slide 2: Change `EAGLE (autoregressive features)` to `EAGLE-1/2 (AR feature prediction) and EAGLE-3 (AR direct token prediction with fused target features)`.
  * Remove DSpark from the `(per-slot block drafts)` parenthetical until `2607.05147` is read in full (or describe it accurately as *"block drafter with Markov/recurrent token heads"* once verified).
  * Drop claims of architectural distinction from DFlash 2 and EAGLE-4 until their papers/technical reports are read; in Slide 7 simply note *"DFlash 2 and EAGLE-4 lack public technical descriptions / Qwen3-8B author checkpoints."*
  * Re-check DFlash Table 1 (`2602.06036`) for EAGLE-3(60) MT-Bench $T=0$ vs. 7-task average $T=1$, and fix `Qwen3.8-27B` in `AGENT_NOTES.md` if it is a typo for `Qwen3.5-27B`.

---

#### 6. [HIGH] Understating ChainFlow's overlap & stating an unverified inference as fact
* **Locations:**
  * `deck/outline.md` (Slide 2: *"ChainFlow (per-token deterministic latents)"*)
  * `research_notes/Drafter side gist prediction and pruning/chainflow_dynaspec.md` (lines 25–27, 81)
* **Issue:** Calling ChainFlow simply `(per-token deterministic latents)` hides that ChainFlow already builds the **same core stack** as Gister: a VAE over frozen Qwen target hidden states, flow matching to predict 8 future latents, frozen LM-head decoding, and a 32-node greedy draft tree. Moreover, `chainflow_dynaspec.md` explicitly warns that *"one latent per token... is the extraction tool's inference from the notation. The paper does not state it outright,"* and notes that ChainFlow's VAE encoder/decoder are bidirectional transformers over 16-state windows with two 4-position chunk experts—so its latents are already contextually coupled across chunks, and its compression ratio (1024 dims/token at 9B) differs from Gister's (2048 dims / 8 tokens = 256 dims/token) by only 4x in total latent dimensionality.
* **Concrete fix / rewording:**
  * Verify ChainFlow's exact tensor shapes directly from the PDF/LaTeX source (not a summarizing fetch).
  * Reword Slide 2 to: *"ChainFlow (VAE + flow over target final-layer hidden states with frozen LM head, but per-position latents, deterministic extrapolation, and token beam-search trees rather than chunk-level stochastic multi-sample trees)."*

---

#### 7. [MEDIUM] "Decoded jointly through the frozen LM head" is imprecise and invites a reviewer misunderstanding
* **Locations:**
  * Prompt claim & `deck/outline.md` (Slide 2: *"one sampled latent per 8-token chunk, from 4 target layers, decoded jointly"*)
  * `AGENT_NOTES.md` ("The design being built", item 4)
* **Issue:** The frozen LM head is a linear projection applied **independently per position** to each of the 8 reconstructed final-layer states. It does *not* perform joint decoding across tokens (unlike an AR or Markov head). What is joint is that all 8 reconstructed states are deterministic functions of a *single* sampled chunk latent $z$, inducing cross-position coherence conditional on $z$—the exact same factorization $q(x_{1:8} \mid z) = \prod_{i=1}^8 q_i(x_i \mid z)$ used by MFD, DBLast, and CALM's chunk decoder. Saying "decoded jointly through the frozen LM head" sounds either confused about how the LM head works or like you are claiming sequential coupling inside the head.
* **Concrete fix / rewording:**
  * Replace *"decoded jointly through the frozen LM head"* with *"decoded in one pass into 8 correlated final-layer states and projected position-wise through the frozen LM head, so cross-token coherence comes from the shared chunk latent $z$ (analogous to CALM's chunk decoder and MFD's mode-conditioned block)."*

---

#### 8. [MEDIUM] Internal contradictions and unverified attributions across the research notes
* **Locations:**
  * `research_notes/Drafter side gist prediction and pruning/latent_world_models_text.md` (lines 11, 29–32) vs. `research_notes/Prior work read in full.md` (line 45)
  * `AGENT_NOTES.md` (line 90: DFlash `~1B` attributed to `DFlash paper T1`) vs. `research_notes/Drafter side gist prediction and pruning/gist_drafters.md` (line 116: *"No parameter count given"*)
  * `research_notes/Guided draft trees and branched caches/guided_tree_construction.md` (line 75: ConFu "ICLR 2026 workshop" + "ICML 2026 poster") vs. `research_notes/Latent planning for speculative decoding/latent_planning_novelty.md` (line 53: *"Venue not stated on page"*)
  * `research_notes/Drafter side gist prediction and pruning/gist_drafters.md` (line 69: SemanticSpec `2026.emnlp-main.123`, no arXiv ID) vs. `research_notes/Latent planning for speculative decoding/latent_planning_novelty.md` (line 41: `arXiv 2602.03708`, `2026.findings-emnlp.1239.pdf`)
* **Issue:**
  * `latent_world_models_text.md` still states NextLat is *"training-only, so inference is unchanged... does not discuss decoding speed or drafting"* (from v1), contradicting `Prior work read in full.md` (later version drafts up to 3.3x / 4.9 accepted tokens).
  * DFlash's `~1B` parameter count came from FlashMap's Table (`~1049M`), not from DFlash Table 1 (`AGENT_NOTES.md` misattributes the source).
  * ConFu's venue and SemanticSpec's ACL Anthology URL/track conflict across notes.
* **Concrete fix / rewording:**
  * Add a one-line header note in `latent_world_models_text.md` noting NextLat v1 vs. NeurIPS 2026 version.
  * Attribute DFlash's `~1.05B` parameter count to FlashMap (2610.24365) or directly from the HF checkpoint config (`z-lab/Qwen3-8B-DFlash-b16`), not DFlash Table 1.
  * Standardize ConFu and SemanticSpec citations against their canonical arXiv/proceedings pages before any external write-up.

---

#### 9. [LOW] Framing mismatch between Stage-1/Stage-2 reality and "Gister" naming
* **Locations:**
  * `AGENT_NOTES.md` ("Naming and scope"), `deck/outline.md` (Slide 2–3)
* **Issue:** "Gist" implies a high-level semantic summary (like ParaScopes/FSP/LCM over 32–128+ tokens), whereas Gister's Stage-1 vector is an **exact 8-token lossless-top-1 compression** (dimension 2048—larger than a 0.6B model's residual stream!) that requires `W[top1]` to reproduce exact token identities. A reviewer who reads "gist" expects semantic compression over a long span, then sees an 8-token exact-token bottleneck (`z=2048`) and sees a framing/mechanism mismatch—the same mismatch your own report warned against (*"the paper is about sampled futures and should drop the planning framing"*).
* **Concrete fix / rewording:**
  * Define "chunk latent" plainly in technical sections (8-token compressed state vector, $z \in \mathbb{R}^{2048}$) and avoid implying semantic/long-horizon planning.

---

### Strongest Honest Novelty Claim

> **Gister tests whether sampling multiple continuous chunk-level latents (one vector per 8-token block, compressed via a target-state autoencoder and decoded through the frozen LM head) produces more coherent, higher-coverage lossless draft trees than (a) factorized per-slot product trees (DFlash+DDTree), (b) discrete mixture-of-modes trees (MFD), and (c) direct token-space flow trees (FlashMap)—and whether compressing target hidden states adds value over compressing tokens alone (CALM).**

Why this is the strongest *honest* claim:
- It concedes what is already owned by prior work (hidden-state VAE + flow drafting: ChainFlow; chunk VAE: CALM; multi-sample flow trees: FlashMap; coherent mixture trees: MFD; multi-layer target conditioning: EAGLE-3 / DFlash).
- It pinpoints the **exact scientific hypothesis** that remains genuinely open and testable: *continuous hidden-state chunk latent vs. discrete modes (MFD) vs. token-simplex flow (FlashMap) vs. token-chunk VAE (CALM)* as the representation for multi-future tree drafting.
- It stays honest about scope (**greedy-only** until Stage 1 passes the TV/KL distribution gate; **batch-1** regime like ChainFlow and MFD).

---

### The Single Experiment That Would Most Strengthen It

**Run the controlled 4-arm matched-backbone, matched-budget tree ablation (offline, 1 GPU, ~2–3 days) BEFORE requesting cluster compute.**

Hold the **exact same 154M Stage-2 backbone, same 2M-position PerfectBlend cache, same `--raw` multi-layer target conditioning, same training steps, and same tree node budgets ($B \in \{16, 32, 64, 112\}$)** fixed, and compare four arms on math / code / chat (measuring $\tau$, chimera rate as defined in MFD, and per-offset acceptance):

1. **Arm A — Deterministic Product Tree (DDTree-style control):** Take your existing `Gister v2` deterministic predictor, decode its 8-slot logits, and build a best-first product tree of budget $B$. *(Tests whether sampling actually beats factorized branching on the same predictor.)*
2. **Arm B — Gister Flow-Sampled Tree (current method):** Your existing 2-step flow sampler over `gist_hf_z2048`, $K$ samples merged to budget $B$.
3. **Arm C — MFD-Style Discrete-Mode Baseline (no VAE, no flow):** Same 154M backbone + $K=8$ learned mode embeddings injected at the input (winner-take-all CE loss directly through the frozen LM head, no Stage-1 AE), merged into a mixture tree of budget $B$. *(Tests whether the Stage-1 VAE + continuous flow adds anything over MFD's 23K-parameter discrete modes.)*
4. **Arm D — Matched Token-Chunk VAE Flow Control (CALM-style AE):** Retrain `tok_vae` at matched **`z=2048` and 8k steps** (tokens only, no hidden states), then run the exact same Stage-2 flow sampler + `--raw` conditioning as Arm B. *(De-confounds whether Stage 1's target hidden states beat a token-only chunk autoencoder once latent width, training steps, and `--raw` conditioning are matched.)*

*(Optional 5th arm if time permits: **Arm E — FlashMap-style direct flow** on the same backbone without Stage 1, or compare Arm C vs. Arm B directly against the released DFlash+DDTree ceiling.)*

**Why this single experiment makes or breaks the project:**
- If **Arm B beats Arm A (by $\ge 5\%$ on chat), beats or matches Arm C, and beats Arm D**, you have a clean, unassailable, publication-grade novelty proof—and a rock-solid justification for cluster compute to fix full-context conditioning and scale data.
- If **Arm A or Arm C beats Arm B**, you immediately learn that the Stage-1 VAE + continuous flow is a net tax (confirming your own report's hypothesis) and save weeks of cluster compute by pivoting to full-context discrete-mode / direct-flow drafting instead.
- If **Arm D matches Arm B**, you know the "target hidden states in Stage 1" claim is dead (because `--headfeat` was doing the work) and should simplify Stage 1 to a pure CALM-style token-chunk VAE.
