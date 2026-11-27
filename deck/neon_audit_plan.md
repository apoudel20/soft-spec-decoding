### Summary Verdict

Core data-alignment, leakage, KL, and trie-counting logic in `plan_gist.py` and `cvae_plan.py` are clean. However, the write-up contains **direct numerical contradictions against its own tables and JSONs**, **major train/eval and loss bugs** that invalidate two key comparisons (the CVAE within-decoder tree win and the `--mix` / `--oblock` experiments), **unfair tree/training confounds**, and **large blocks of results unsupported by the provided audit bundle**.

---

## Verified Clean (Checked, No Bug Found)

- **Row / label / plan-window alignment** (`plan_gist.py:134-154`): `raw` reads rows `s..s+7`, `lab` is `T1[s+8..s+15]`, plan window `rows` is `T1[s0+8 .. s0+8+8G-1]`, and `prev` (`pr`) is `T1[s0+8-8G .. s0+7]` (ending at the verified bonus token `T1[s0+7]` predicted by row `s0+7`). `round_ok` enforces `s..s+15` in one sequence; dev rounds stride by `2*K = 16` (non-overlapping).
- **No future-token leakage into predicted/prior plans**: `P` and `Flow` (`plan_gist.py:255-262, 300-302`) read only `raw_a` (`s0..s0+7`) and optionally `prev` (`<= s0+7`). `Prior` (`cvae_plan.py:111, 151`) reads only `raw` (`s..s+7`).
- **Exactness of `kbest` (`plan_gist.py:173-182`), `tree_stats` (`plan_gist.py:185-189`), and `gkl` (`cvae_plan.py:61-62, 119-121`)**: `kbest` returns the exact top-`Kb` full 8-token sequences under the factorised slot product; `tree_stats` counts unique non-empty drafted prefixes (excluding root); `gkl` is exact diagonal-Gaussian KL in nats summed over `(G, c)`, scaled by `1/8` to match per-token NLL (`β = 1` is the exact ELBO), and converted via `/ ln 2`.

---

## Critical Findings

### C1. CVAE within-decoder `prior8` vs `kbest8` win is an artifact of evaluating `kbest8` on an OOD `prior_mean`
- **Where**: `cvae_plan.py:155-161`, `results/cvae_G32c16b0.3.json`, `deck/audit_plan_section.md` (CVAE bullet 3).
- **Bug / Confound**: `dec` is trained exclusively on latent *samples* `z = mu + sigma * eps` (`cvae_plan.py:112`), never on latent means. At `G=32, β=0.3` (512 latent dims carrying only ~8 bits total, so posterior/prior variance dominates the mean), feeding the zero-noise `prior_mean` puts `dec` out of distribution and craters single-draft acceptance relative to `prior_sample`:
  - `dev_math`: `accepted_prior_mean = 2.687` vs `accepted_prior_sample = 2.898` (**−0.211 tokens**)
  - `dev_code`: `accepted_prior_mean = 1.978` vs `accepted_prior_sample = 2.131` (**−0.153 tokens**)
  - The same OOD drop occurs for `post_mean` vs `post_sample` (`3.050 vs 3.226` math; `2.084 vs 2.298` code) and even for the collapsed `β=1` run (`2.91 vs 3.06` math).
- Yet `cvae_plan.py:158` computes `kbest4`/`kbest8` **only at `prior_mean`**, whereas `prior4`/`prior8` draw **`prior_sample`**! Comparing the actual *tree lift* over each method's own single-draft baseline in `cvae_G32c16b0.3.json`:
  - Math: `kbest8` lifts `2.687 → 3.082` (**+0.395**); `prior8` lifts `2.898 → 3.282` (**+0.385**).
  - Code: `kbest8` lifts `1.978 → 2.169` (**+0.191**); `prior8` lifts `2.131 → 2.349` (**+0.218**, at `33.7 vs 31.8` nodes).
  - Chat: `kbest8` lifts `1.443 → 1.536` (**+0.093**, 32.3 nodes); `prior8` lifts `1.450 → 1.548` (**+0.098**, 36.9 nodes).
- Thus the headline claim *"Within one decoder, best of K prior samples beats that decoder's own K most likely drafts at a similar node count (β = 0.3: 3.28 against 3.08 on math, 2.35 against 2.17 on code, at 31 to 34 nodes)"* is invalid: virtually the entire +0.20 math gap is the OOD penalty on `prior_mean`. (`kbest` must be evaluated at `prior_sample` or a norm-matched `prior_mean`, or `dec` trained with mean/sample mixing.)

### C2. `loader` silently collapses out-of-bounds offsets to `o = 0`, invalidating `--oblock` and progressive-drafting tests
- **Where**: `plan_gist.py:160-166` (plus `plan_gist.py:72, 79`), `results/plan_pw_G32c16s0.25.json`, `deck/audit_plan_section.md` (Paragraph plan bullet 7).
- **Bug**:
  ```python
  if oblock:
      o = K * rng.integers(0, G, B)
  else:
      o = np.where(rng.random(B) < poff, rng.integers(1, G * K - K + 1, B), 0)
  s0 = s - o
  bad = (s0 < 0) | (seq[np.clip(s0, 0, None)] != seq[s])
  s0[bad] = s[bad]
  ```
  Whenever `s - o` precedes the start of the current sequence, `s0` is reset to `s`, silently turning that example into **`o = 0`**.
- From `plan_pw_G32c16s0.25.json`, dev sequences are mostly shorter than 248 tokens: only 51%/42%/59% of rounds survive `o = 128`, and only **10%/7%/19%** survive `o = 248` (`rounds_o248 = 68/33/143` vs `655/450/743`). Assuming similar lengths in `cache_pb`:
  1. **With `--oblock`**, offsets are **not** *"drawn uniformly over blocks"*: half or more of all `o > 0` draws fail `bad` and pile onto `o = 0`, massively over-representing `o = 0` while starving large offsets.
  2. **With default `poff = 0.5` (`G = 32`)**, `o = 0` gets `> 70%` of training rounds, while each of the 248 non-zero offsets gets `< 0.2%` of rounds — and `Decoder.oemb = nn.Embedding(249, d)` (`plan_gist.py:79`) gives every integer offset an **unshared** embedding! Combined with global power normalisation across `(G, c)` (`plan_gist.py:72`), `PlanEnc` pours signal into block 0, and `Decoder` learns via `oemb(0)` vs `oemb(o>0)` to use the plan only at `o = 0` and ignore it at `o > 0`. This shows up directly in `plan_pw_G32c16s0.25.json`: at `o = 0`, `zero = 2.49` vs `oracle = 4.79`; at `o = 16`, `zero = 2.99` ≈ `oracle = 2.96` ≈ control `3.05`!
  3. Even with `--rel` (`plan_gist.py:85-89`), `self.oemb(o)` is still added, so the decoder still distinguishes `o = 0` from `o > 0`.
- The write-up's claim *"With offsets drawn uniformly over blocks..."* is false, and attributing the failure solely to unconstrained attention in `PlanEnc` misses the primary sampling and embedding bugs.

### C3. Direct numerical contradictions between prose claims and the tables / JSONs
- **Where**: `deck/audit_plan_section.md` vs `results/plan_pw_G8c0.json`, `results/plan_pw_G32c16s0.25.json`, `results/cvae_G8c16b0.3.json`.
- **Contradictions**:
  1. **"The oracle plan beats the control by 37 to 117%."** False. Against control (`3.02 / 2.16 / 1.48`), the verified `plan_pw_G32c16s0.25.json` chat oracle is `1.732 / 1.478 - 1 =` **+17.2%** (not ≥ 37%), and the 512-token row in the same table is **+31% / +12% / +5%**. The true range across the table is **5% to 117%** (or **17% to 117%** excluding the `B=64` run). `37%` is only the `64-token, c=16` chat gain (`2.03/1.48 - 1`).
  2. **"No prior-mean or prior-sample draft beats the control's single draft."** Directly contradicted by the CVAE table's own `β = 1` row (`Prior sample = 3.06` on math > control `3.02`), and by `cvae_G8c16b0.3.json` `dev_chat.accepted_prior_mean = 1.4818` > control `1.4778` (table rounds both to `1.48`).
  3. **"Every predicted plan is below the control."** False on chat: `plan_pw_G32c16s0.25.json` has `dev_chat.accepted_det = 1.4805` vs control `1.4778` (the table itself prints `1.48` vs `1.48`, and likewise for the `phist` row).
  4. **"Plans made earlier (offsets 16 to 248) never help, also for the oracle."** Contradicted by the next two sentences (`3.07 vs 2.95` for `oblock` and `3.12 vs 3.03` for `rel` at `o = 16`, both also above control `3.05` at `o = 16`), and by `plan_pw_G32c16s0.25.json` `dev_code.accepted_oracle_o248 = 2.485` vs `zero = 2.394` (though `N = 33`).

---

## High Findings

### H1. `cvae_plan.py --mix` doubles effective KL pressure on the posterior, confounding the collapse claim
- **Where**: `cvae_plan.py:114-121`, `deck/audit_plan_section.md` (CVAE bullet 1).
- **Bug**: When `--mix 0.5` replaces `z` with a prior sample `zp` on 50% of the batch, the posterior `q` gets **zero** reconstruction (`nll`) gradient on those 50% of examples, yet `kl = gkl(mq, lq, mp, lp).mean()` still applies 100% of `beta * kl / K` across the entire batch! That doubles the effective KL weight on `q` (`beta / (1 - mix) = 0.6`), directly inducing collapse.
- **Overclaim**: *"whenever the decoder also trains on prior samples, the posterior collapses onto the prior"* generalizes from a single buggy `mix=0.5` run; KL should be masked by `(1 - use_p)` (or rescaled) before concluding that prior mixing causes collapse.

### H2. `plan_gist.py` eval adds channel noise to `det`, `flow`, and `flowK` (while `kbest` uses `det_clean`), and fails to power-normalise `P`'s shrunken predictions
- **Where**: `plan_gist.py:72, 303, 309, 319`, `results/plan_pw_G32c16s0.25.json`, `deck/audit_plan_section.md` (Paragraph plan description & table).
- **Train/eval mismatches**:
  1. `plan_gist.py:303, 319` adds fresh Gaussian noise `sigma * randn` to `det`, `flow`, and every sample in `flowK`. Thus `flow8` mixes flow-sampler diversity with random channel noise (inflating `flow8_nodes` to 47–52 and hurting single-sample `accepted_flow = 2.456 < accepted_zero = 2.492`), and `flow_clean` is never evaluated.
  2. The table reports noisy `accepted_det` (`2.73 / 2.10 / 1.48`) while `kbest` (`plan_gist.py:309`) is computed on noise-free `det_clean` (`2.75 / 2.10 / 1.47`) — an undocumented inconsistency.
  3. `PlanEnc` normalises `mu` to unit average power (`E[mu^2] = 1`, line 72), so `dec` only sees inputs of power `1 + σ²`. `P` minimizes MSE to `mu`, so its output `ph = E[mu | raw_a]` shrinks to power `1 - p_mse ≈ 0.40` (at `p_mse = 0.60`), and is passed unnormalised to `dec` — an OOD norm mismatch analogous to the Stage-1 issue discussed later in the same document.

### H3. Tree comparisons are not node-matched, omit ~8× drafter compute overhead, and understate CVAE node overhead
- **Where**: `plan_gist.py:317-321`, `cvae_plan.py:162-164`, `results/plan_pw_G8c0.json`, `results/cvae_G8c16b0.3.json`, `deck/audit_plan_section.md` (Paragraph plan & CVAE bullets).
- **Confounds & Mismatch**:
  1. Comparing `K=8` sampled-plan trees (39–52 nodes) against `kbest8` (29–32 nodes) is **not node-matched**; giving the control a matching ~40–46 node budget (e.g., `kbest12` or a prefix-probability tree) would raise the control above `3.46 / 2.38 / 1.55`.
  2. **Omitted drafter-side latency/compute**: `kbest8` runs the 23M `Decoder` + LM-head **once**; `prior8` runs `Prior` (14M) + **8× `Decoder` + 8× LM-head** (`152k` vocab); `flow8` runs **64 forward passes of `PlanFlow`** (`nfe=8 × K=8`) + **8× `Decoder` + 8× LM-head**. For speculative decoding, ignoring an 8×+ drafter cost plus 31–79% more target verification nodes overstates the competitiveness of sampled-plan trees.
  3. **Numerical mismatch**: For the 64-token CVAE (`cvae_G8c16b0.3.json`) vs control (`plan_pw_G8c0.json`), node overhead is **+31.1%** on math (`38.55/29.40`), **+41.7%** on code (`43.39/30.63`), and **+42.7%** on chat (`46.30/32.44`) — i.e., **31 to 43%**, not *"30 to 40% more nodes"*.

### H4. Stage-1 conclusions are confounded by a 3× step increase (12k → 36k) and an incomplete diagnosis of `d1024 × 6`
- **Where**: `results/ae_gate_base_6m_z2048.json`, `results/ae_gate_big_6m_z2048.json`, `deck/audit_plan_section.md` (Stage 1 table & bullets 1–2).
- **Confounds & Unsupported claims**:
  1. **Data vs steps confound**: Row 1 (`d512 × 4, 512k rows`) trained for **12k steps**, whereas Row 2 (`5.9M windows`) trained for **36k steps** (`ae_gate_base_6m_z2048.json` `args.steps = 36000`). In `base_6m_z2048`'s own `train_log`, `kl` drops from `0.396` at step 12k to `0.211` at step 36k (`agree_train`: `87.2% → 92.3%`). Attributing the +2 to +7 point gain purely to *"More data"* is invalid without a 36k-step 512k-row or 12k-step 5.9M-window ablation.
  2. **Norm-matched mean explanation**: First, `94.1%` (math) and `78.7%` (chat) are **not in the JSONs** (and code is omitted). Second, rescaling `mu` to match `E[||z||]` inflates the signal vector `mu` by a large scalar rather than matching the training distribution `mu + noise`; moreover, `d512 × 4` uses the exact same `zdim=2048, floor=0.5, beta=0.001, zdrop=0.15` and reaches the exact same `klz = 256.0`, yet its latent mean (`96.7 / 95.4 / 85.4%`) *beats* its posterior sample (`96.1 / 94.5 / 84.1%`)! Third, `ae_gate_big_6m_z2048.json` shows `d1024 × 6` is **severely underfitting on train itself** (`step 35999`: `kl = 0.345` vs `base`'s `0.211`, `agree_train = 90.9%` vs `92.3%`), which is the actual evidence supporting an optimization/recipe issue — yet the text never cites the training loss.

### H5. Large portions of the write-up cannot be verified from the provided files
- **Unsupported claims / missing artifacts**:
  - **Entire "Gist of the stretch after the draft" subsection**: `stage2.py`, `jepa_gist.py`, all 5 table rows, VICReg weights/steps, JEPA train/dev errors (`0.027`, `0.052–0.071`), and spread `0.70`.
  - **Paragraph plan**: tanh collapse run (`2.99/2.16/1.49`), all 2k-step debug runs (`c=4`, `c=64`, `raw_drop`), 4 of 6 rows in the full plan table (`G=8 c=16`, `G=64 c=16 B=64`, `G=32 c=64 s=0.05`, `G=32 phist`), `p_mse = 0.66`, and the `oblock` (`7.80/7.01/5.25`, `3.07 vs 2.95`) and `rel` (`5.44`, `3.12 vs 3.03`) runs.
  - **CVAE**: 3 of 5 table rows (`β=1`, `β=0.1`, `β=0.3 mix=0.5`) and `β=0.1` code/chat tree scores (`2.40 / 1.59`).
  - **Stage 1**: `ae_gate.py`, 512k-row baseline (`94.6/93.0/78.3%`), exact "5.9M windows" count, and norm-matched mean numbers (`94.1%`, `78.7%`).

---

## Medium Findings

### M1. `--phist` applies a right-padding-trained `PlanEnc` to left-padded histories, and compares `p_mse` across different latent spaces
- **Where**: `plan_gist.py:69, 149-151, 255, 300`, `deck/audit_plan_section.md` (Paragraph plan bullet 6).
- **Bug / Confound**: `PlanEnc` is trained only on future windows `ids` (valid at the start, right-padded near sequence end, with block 0 carrying the `o=0` signal). Under `--phist`, `plan(pids, pm)` applies `PlanEnc` with `no_grad` to **left-padded** past windows (80–92% of dev rounds occur before token 248!), placing the most recent context at block `G-1` and padding block 0.
- Additionally, `p_mse` (`0.60 → 0.66`) is compared across two separately trained runs whose encoders `E` define different latent spaces with potentially different variance/predictability; note also that on math the table shows `det` actually *rose* from `2.73` to `2.79` (while `zero` rose `2.49 → 2.61` and code fell `2.10 → 2.04`).

### M2. Control `plan_pw_G8c0.json` was trained with `poff = 0.5` while all CVAE runs used `poff = 0.0`
- **Where**: `results/plan_pw_G8c0.json` (`args.poff = 0.5`), `plan_gist.py:79, 89`, `cvae_plan.py:108`.
- **Confound**: Even with `c = 0`, `Decoder.forward` adds `self.oemb(o)` to the 8 zeroed plan tokens. In the control, `o` varies across 57 offsets on half the steps (injecting irrelevant offset embeddings), whereas every CVAE run trains exclusively at `o = 0`. A clean control for CVAE should use `--poff 0`.

### M3. "A longer plan lowers the oracle at offset 0. The oracle's gain sits in the drafted block." is confounded
- **Where**: `deck/audit_plan_section.md` (Paragraph plan bullets 4–5).
- **Confound**: Increasing `G` from 8 → 32 → 64 **does not dilute a fixed bit budget** — it multiplies the channel capacity by 4× and 8× (`262 → 1046 → 2093` bits). The drop from `5.51 → 4.79 → 3.97` is confounded by (i) the 512-token run using `B = 64` (half as many training examples), (ii) a 4×–8× longer attention sequence in `PlanEnc` and `Decoder` trained for the same 10k steps, and (iii) different `o = 0` fallback rates under `poff = 0.5`.

### M4. Stage-1 "Errors sit at near-tie positions" misrepresents the error distribution
- **Where**: `results/ae_gate_base_6m_z2048.json` (`agree_by_margin`), `deck/audit_plan_section.md` (Stage 1 bullet 3).
- **Analysis**: Computing actual error counts from `agree_by_margin` in `ae_gate_base_6m_z2048.json`:
  - `dev_math` (353 errors): `<0.5` = 137 (**38.8%**), `0.5–2` = 165 (**46.7%**), `>=2` = 51 (**14.4%**).
  - `dev_code` (338 errors): `<0.5` = 97 (**28.7%**), `0.5–2` = 174 (**51.5%**), `>=2` = 67 (**19.8%**).
  - `dev_chat` (1,738 errors): `<0.5` = 676 (**38.9%**), `0.5–2` = 818 (**47.1%**), `>=2` = 244 (**14.0%**).
- Across all three sets, the **plurality/majority of errors (47–52%) sits in the omitted `0.5–2` margin bucket**, and **14–20% sit on high-confidence `>=2` margin positions** (≥7.4:1 logit odds), not `<0.5` near-ties.

### M5. Non-sequitur in `--oblock` interpretation and unmentioned offset survival bias
- **Where**: `deck/audit_plan_section.md` (Paragraph plan bullet 7), `results/plan_pw_G8c0.json`, `results/plan_pw_G32c16s0.25.json`.
- **Issues**:
  1. Claiming *"the oracle at offset 0 reaches 7.80 / 7.01 / 5.25, so each block latent becomes a close code of its own tokens"* does not follow: high `o = 0` acceptance only proves the plan encodes block 0 tightly at `o = 0`; the `3.07` score at `o = 16` shows the remaining block latents did *not* become close codes of their own tokens.
  2. Because larger offsets filter dev rounds to later positions in longer sequences (`plan_pw_G8c0.json` math `accepted_zero`: `3.02` at `o=0` → `3.05` at `o=16` → `3.14` at `o=32` → `3.18` at `o=48`, with `rounds` dropping `655 → 535`, and to `68/33/143` at `o=248`), cross-offset scores are only comparable within the same offset subset (and noisy at `o=248`).

### M6. Selective reporting in the CVAE table and bullets
- **Where**: `results/cvae_G32c16b0.3.json`, `results/cvae_G8c16b0.3.json`, `deck/audit_plan_section.md`.
- **Issues**:
  1. For `G=32, β=0.3`, the table reports `post_mean` (`3.05 / 2.08 / 1.48`) while omitting `post_sample` (`3.23 / 2.30 / 1.50`), which is substantially higher due to the mean-vs-sample mismatch.
  2. Table row `β=0.3` reports KL as `8 bits`, whereas `dev_code.kl_bits = 8.52` (should be `8 to 8.5 bits`, matching the range convention in rows 3 and 5).
  3. The within-decoder bullet cites math and code (`31 to 34 nodes`) for `G=32, β=0.3`, omitting chat (`prior8 = 1.55` at **37 nodes** vs `kbest8 = 1.54` at 32 nodes) and omitting `G=8, β=0.3` where `prior8` uses **24–41% more nodes** than `kbest8` (`39 vs 31`, `43 vs 32`, `46 vs 33`).

### M7. `kbest` is a suboptimal per-position tree baseline
- **Where**: `plan_gist.py:173-182`.
- **Note**: `kbest` maximizes full 8-token sequence probability rather than expected accepted prefix length (branching late at slot 7 before branching at slot 0 when slot 7 has slightly smaller log-ratio). Even this suboptimal per-position baseline matches or beats the sampled-plan trees at fewer nodes — worth noting so the baseline isn't mistaken for an upper bound on per-position drafting trees.

---

## Low Findings

- **L1. Rounding error**: In `results/plan_pw_G32c16s0.25.json`, `dev_math.kbest8_nodes = 33.499237`, which rounds to **33**, not **34** in the Paragraph plan table.
- **L2. Unqualified math-only node counts for control**: In the CVAE prose (*"3.26 / 2.29 / 1.52 for its 4 most likely drafts (18 nodes) and 3.46 / 2.38 / 1.55 for its 8 most likely drafts (29 nodes)"*), `18` and `29` are math-only (`dev_code` is `18.3 / 30.6`, `dev_chat` is `19.1 / 32.4`).
- **L3. Early collapse check for JEPA**: Reporting *"per-dimension spread 0.70 at step 1,000"* for a 4,000-step run does not establish non-collapse at step 4,000.
- **L4. Minor code / description nits**: `plan_gist.py:331` has a redundant ternary `'' if not args.smoke else ''`; the description *"its first block is the block drafted this round"* only holds at `o = 0`; `cvae_plan.py` allocates unused `oemb` rows `1..G*K-K`.

---

## Suggested Accurate Wording Replacements

| Current wording | Accurate replacement |
|---|---|
| *"The oracle plan beats the control by 37 to 117%. Every predicted plan is below the control."* | *"The oracle plan beats the control by 17 to 117% across the B = 128 runs (5 to 31% for the half-batch 512-token run). Every predicted plan is below the control on math and code, and at most level with it on chat (1.48 vs 1.48)."* |
| *"Plans made earlier (offsets 16 to 248) never help, also for the oracle. With offsets drawn uniformly over blocks the oracle at offset 0 reaches 7.80 / 7.01 / 5.25, so each block latent becomes a close code of its own tokens..."* | *"In the default runs, plans made earlier (offsets 16 to 248) do not help even for the oracle, largely because out-of-bounds offsets fall back to o = 0 and each non-zero offset uses an unshared, rarely trained embedding. With block-multiple offsets (`--oblock`, still heavily skewed toward o = 0 by the boundary fallback), the oracle at o = 0 reaches 7.80 / 7.01 / 5.25, but at o = 16 gives only a small gain over zeroed (3.07 vs 2.95 on math; 3.12 vs 3.03 with `--rel`). Because of the o = 0 fallback, global power normalisation, and untied encoder queries, this test of progressive drafting is inconclusive."* |
| *"At β = 1, and whenever the decoder also trains on prior samples, the posterior collapses..."* | *"At β = 1, and in our `--mix 0.5` run (where KL was still applied to 100% of examples, doubling effective KL pressure on the posterior), the posterior collapses onto the prior..."* |
| *"No prior-mean or prior-sample draft beats the control's single draft."* | *"No prior-mean draft beats the control on math or code (and only ties it on chat); the only prior-sample draft above the control is the collapsed β = 1 model on math (3.06 vs 3.02), which remains below it on code and chat."* |
| *"Within one decoder, best of K prior samples beats that decoder's own K most likely drafts at a similar node count (β = 0.3: 3.28 against 3.08 on math, 2.35 against 2.17 on code, at 31 to 34 nodes)."* | *"At G = 32, β = 0.3, `kbest8` was evaluated at the out-of-distribution `prior_mean` (single-draft 2.69 / 1.98) while `prior8` used `prior_sample` (2.90 / 2.13); relative to their respective single-draft baselines, tree gains are nearly identical (+0.39 vs +0.40 on math, +0.22 vs +0.19 on code)."* |
| *"...matches it on all three sets (3.45 / 2.40 / 1.57 against 3.46 / 2.38 / 1.55) with 30 to 40% more nodes."* | *"...matches the control's 8-draft tree on all three sets (3.45 / 2.40 / 1.57 vs 3.46 / 2.38 / 1.55), while requiring 31 to 43% more trie nodes (39/43/46 vs 29/31/32) and 8× more decoder forward passes."* |
| *"More data lifts the original model by 2 to 7 points, chat most."* | *"Training on 5.9M windows for 36k steps (vs 512k rows for 12k steps) lifts the original model by 2.1 to 7.1 points, chat most; data scale and 3× longer training are confounded here."* |
| *"Errors sit at near-tie positions."* | *"Agreement is high when the target's top-1 logit margin is ≥ 2 (99.4 / 99.0 / 96.7%) and drops sharply below 0.5 (60 / 55 / 47%); by error volume, 47 to 52% of errors occur in the 0.5–2 margin band, 29 to 39% below 0.5, and 14 to 20% at margin ≥ 2."* |