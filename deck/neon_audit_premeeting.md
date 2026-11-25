[gemini-4-neon via interactions, thinking_level=high, 205s]
[tokens: input 132,774 (cached 0), output+thinking 21,605; call $0.7224; total $8.2303 of $10.00]
[interaction_id: ChA2ZDI3ZmIzZjZhYmVjOTVmEAgaAjIzKgRtYWlu]

### Critical

1. **`deck/slides/plan.html` (note) & `RESULTS.md` (`Proposed next step`) — Misstates your own prior negative result**
   - **Issue:** Claims earlier plan experiments `"added a single averaged future vector to a frozen DFlash"`. In `RESULTS.md` (`Can DFlash use the target's future?`), you **also fine-tuned DFlash with LoRA r16 + adapter** (`kv_lora.py`) and got no gain (`math +0.04, code -0.06, chat -0.02`, identical training losses `2.585 vs 2.586`). A professor who checks `RESULTS.md` will view omitting the fine-tuned control as cherry-picking.
   - **Fix:** Change to `"added a single averaged future vector to DFlash (frozen, and fine-tuned with LoRA r16 on 300 answers) and saw no gain. They do not test a decoder trained from scratch with a learned multi-token latent."`

2. **`deck/slides/bottleneck.html` (table row 3) — Off-by-8× factual error**
   - **Issue:** Row 3 says `"+ 64 previous latents"`. `--hist 8` is **7 previous latents** (8 latents total = 64 **positions**, not 64 latents, which would be 512 positions).
   - **Fix:** Change to `"+ 7 previous latents (64 positions total)"`.

3. **`deck/website_gister_section.html` (Figures 2 & 3 SVGs + captions + offline note) — Multiple false / stale claims**
   - **Issue:**
     - **Fig. 2 SVG** says `"costs about one decode step"` (contradicting `profile_tree_idle.json`: verify is `40 → 78 ms` vs `39 ms` decode, and contradicting its own caption `"+45 to 55% round time"`) and `"158M today · 285M tonight"`.
     - **Fig. 3 SVG** says `"Tonight's scale step"`, `"2M (today) + 2M (tonight)"` (4M, not 6M), `"about 7 to 8 am"`; **Fig. 3 caption** says `"about twice the data ... (6M positions)"`—`2M → 6M` is **3×**, not 2×.
     - **Offline table note** says `"about 100x more text"`—v2 used `2M` (`4×` v1's `512K`) and 285M used `6M` (`12×` v1), not `100×`.
   - **Fix:** Update Fig. 2 SVG (`verify scales with tree size: 40–78 ms`, `158M / 285M sampler`), rewrite Fig. 3 SVG & caption to past tense with `3× data (2M → 6M positions)` and `1.8× params (158M → 285M)` (noting it is confounded), and replace `"about 100x more text"` with `"4× (v2) to 12× (285M) more positions than v1"`.

---

### High

4. **`deck/slides/cover.html` (lead) & `deck/outline.md` (#1) — Overclaim on identical output**
   - **Issue:** `"Output is identical to the target’s greedy decoding."` In `results/e2e_final.json` (linked directly on the cover slide), Gister matches plain target-only greedy on only **4–5/20 MT-Bench** and **7–11/20 GSM8K** prompts due to 4-bit batched numerics. A skeptical professor checking the repo will flag this immediately.
   - **Fix:** Change to `"Uses exact greedy verification (same rule as DFlash; under 4-bit batched numerics, wider verification shifts near-ties relative to batch-1 decode)."` Mention `Qwen3-8B (4-bit)` on the cover.

5. **`deck/slides/e2e.html` (lead) & `deck/website_gister_section.html` (e2e note) — Wrong node count (`60 to 120 nodes`)**
   - **Issue:** Both say `"verifying 60 to 120 nodes"`. In `e2e_final.json`, the rows shown (`4 samples + chain` and `8 samples + chain`) verify **33 to 37 nodes** and **54 to 62 nodes** (`120 nodes` was the K=16 run not in this table).
   - **Fix:** Change `"60 to 120 nodes"` to `"33 to 62 nodes"`. Also add `"Every tree is slower than the chain on this GPU"` visibly on `e2e.html` (not only in `<aside>`), and add the 4-bit target-speed caveat (`~45 ms/step` 4-bit dequantization makes chain drafting look cheaper than it would on a bf16/vLLM engine).

6. **`deck/slides/sampling.html` (chart + text) — Confounded hero comparison**
   - **Issue:** Plots `Deterministic, 154M` (2M positions) against `Flow sampler 285M` (6M positions) and claims `"Best of 16 exceeds the deterministic predictor by 36 / 38 / 32%"`. That mixes sampling with 1.8× model size and 3× data. The matched `158M / 2M` sampler gets **+31 / +30 / +27%** (`4.04 / 2.89 / 1.85`).
   - **Fix:** Either plot the matched `158M / 2M` sampler on Slide 5 (leaving `285M / 6M` for Slide 6), or label the bars `"154M, 2M pos"` vs `"285M, 6M pos"` and state the matched `158M` gain (`+31 / +30 / +27%`) explicitly.

7. **`deck/slides/plan.html` (Test 1 & Box 4) — Unspecified `oracle z` mechanism + missing concrete compute ask / storage reality check**
   - **Issue:**
     - **Test 1 (`Oracle z`)** is the #1 thing a professor will attack: if `z` is trained with JEPA (no token reconstruction), how do you get `oracle z` in Test 1 before training JEPA? If you train the encoder jointly with the decoder, that's just reconstruction again; if you mean-pool target states, that's the averaged vector that failed on DFlash.
     - **Box 4 (`800M positions is about 3 days`)**: caching 4 layers at 32 KB/pos for 800M positions is **~25 TB** (3.2 TB for 100M), so it must be computed on the fly or compressed—and the slide never states the **actual cluster ask** (GPUs / GPU-hours for Test 1 vs Tests 2–4) or a **quantitative go/no-go bar** for Test 1.
   - **Fix:** Specify Test 1's exact protocol (e.g., train JEPA encoder+predictor first, freeze encoder, feed true-future `z` to a from-scratch decoder vs `z = 0`, with a quantitative go/no-go threshold), clarify on-the-fly vs cached states for 100M+ positions, and state the concrete GPU-hour ask for Test 1 (gate) vs Tests 2–4.

8. **`deck/slides/standing.html` (Card 1 vs `<aside>`) — Self-contradiction on `"equal node budget"`**
   - **Issue:** Card 1 compares Gister best-of-8 vs DFlash 64-node tree labelled **`equal node budget, offline`**, while `<aside>` and `RESULTS.md` say you have *not* compared against per-position trees at equal node count (Test 3). Moreover, `best of 8` is an offline upper bound over 8 branches (≤64 slots), and comparing Gister (~285M) vs DFlash (~1B) confounds tree construction with drafter model.
   - **Fix:** Change Card 1 sub-label to `"≤64 token slots each, offline; different drafters (285M vs ~1B)"` and add `"capped at 8 drafted, bonus excluded"` + `"Gister 285M"` to Card 2. Clarify Test 3 tests sampled-latent trees vs per-position trees *with matched drafter / equal verified nodes*.

9. **`deck/slides/examples.html` (both cards) — Garbled context-to-draft boundary & GSM8K-only selection**
   - **Issue:** Card 1 reads `"Let's break this down step"` → `"step.⏎⏎..."` (missing `"by"` / bonus token); Card 2 reads `"Janet's ducks lay **1"` → `"eggs per day**"` (missing `"6"` / bonus token!). A professor will think drafting is off-by-one. Also, both examples are GSM8K markdown templates (`⏎⏎---⏎⏎### Step`), with no code or chat example. (`results/examples.json` is not attached to verify raw strings.)
   - **Fix:** Show the full boundary including the bonus token (e.g., `…step by` | `step.…`, `…lay **16` | `eggs per day**…`), define `8 ✓` visibly on the slide, and swap one GSM8K card for a HumanEval or MT-Bench example.

---

### Medium

10. **`deck/slides/idea.html` (`Related work`) — Contradicts `RESULTS.md` on ChainFlow**
    - **Issue:** `idea.html` describes ChainFlow as `"one deterministic 8-position draft"`, whereas `RESULTS.md` describes it as `"flow to 8 future positions, frozen LM head, greedy tree"`. If ChainFlow already uses flow + tree, the claimed differentiation (`"combines sampled latents ... with joint decoding"`) looks weak.
    - **Fix:** Reconcile `idea.html` with `RESULTS.md` (and your novelty audit), add years/arXiv IDs, and state precisely what differs (e.g., chunk-level continuous latent vs per-position, and that no improvement over prior work is claimed yet—put `<aside>`'s honesty on the slide).

11. **`deck/slides/bottleneck.html` — Missing direct-token ablation (`--direct`)**
    - **Issue:** After Slide 3 admits `"The latent mostly encodes the 8 tokens"`, a professor will immediately ask if predicting tokens directly works just as well. You ran `--direct` (`2.73 / 2.16 / 1.39` vs `2.80 / 2.23 / 1.43`, ~3% with latent), yet it is missing from the deck. Also `"model size add little"` understates `2.80 → 3.08` (+10% math) while omitting the PerfectBlend domain-mix confound on code/chat.
    - **Fix:** Add the `--direct` row (`2.73 / 2.16 / 1.39`) to the table and refine the lead (`"Latent history adds nothing; scaling to 154M/2M helps math (+10%) only (confounded with domain mix)"`).

12. **`deck/slides/scaling.html` (note & lead) — Incomplete confounds & omitted GSM8K-train overlap**
    - **Issue:** Note lists `"no overlap with MT-Bench, GSM8K test, HumanEval, MBPP validation or UltraChat test"`—quietly omitting **GSM8K-train holdout**, which is the actual `Math` dev set in this table and has **1/40** prompt prefix overlap (`RESULTS.md`). Also omits step-count confound (`10k` vs `24k` vs `20k`) and that `ctx56` only uses layers `30, 36` (not all 4). Lead says `"no change on code or chat"`, whereas best-of-16 drops slightly (`2.89 → 2.83`, `-2%`; `1.85 → 1.82`, `-1.5%`).
    - **Fix:** Mention `"1/40 overlap on the GSM8K-train dev set"`, add `"steps (10k/24k/20k)"` and `"layers 30, 36"` for `ctx56`, and change `"no change"` to `"−1 to −2% on code and chat"`.

13. **`deck/slides/sampling.html` (text) — Wrong `0.13` lower bound for the 285M chart**
    - **Issue:** `"Each doubling of K adds 0.13 to 0.33 tokens."` For the `285M` model plotted, `8 → 16` adds **`0.15 to 0.33`** (`+0.24 / +0.33 / +0.15`); `0.13` is from the `158M` model (`+0.31 / +0.25 / +0.13`), which isn't on this slide. Also only one doubling (`8 → 16`) was measured for 285M.
    - **Fix:** Match the range to whichever model is plotted (`"Doubling K from 8 to 16 adds 0.15 to 0.33 tokens"` for 285M, or `0.13 to 0.31` for 158M).

14. **`deck/website_gister_section.html` (tables & status) — Internal mismatches and missing stage-1 top-1 failure**
    - **Issue:**
      - End-to-end table lists `Gister chain (v2)` size as `32M + 158M` (v2 is `154M`, not `158M`); `v1` is listed as `22M` vs `20M` in `RESULTS.md`/`bottleneck.html`.
      - Offline table mixes **8-step (`nfe=8`)** 158M rows (`4.00 / 2.86 / 1.84`) with **2-step (`nfe=2`)** 285M and `ctx56` rows, while the status table compares `ctx56` against `4.04 / 2.89 / 1.85` (`nfe=2`), which isn't in the table!
      - Offline note says `"The context run is warm-started and uses batch 64"`—the 158M baseline was *also* warm-started; the confounds are `batch 64`, `20k steps`, and `layers 30,36`.
      - Status table & Fig. 1 caption omit that Stage 1 **failed top-1** (`99.4 / 99.0 / 97.3% < 99.5 / 99.5 / 99.0%`), not just TV. Status table also uses stale `7.6 to 8` distinct drafts (`nfe=8`) instead of `7.3 to 8.0` (`nfe=2`), and `"matches DFlash's implementation exactly"` overstates 4-bit exactness.
    - **Fix:** Standardize offline table to `nfe=2` (`2.85/1.95/1.41`, `3.72/2.64/1.72`, `4.04/2.89/1.85`) or label `nfe=8`, fix `154M` / `20M` sizes, add Stage-1 top-1 gate failure, update distinct drafts to `7.3 to 8.0`, and fix the `ctx56` note and `25 Nov 2026` eyebrow date.

15. **`deck/slides/e2e.html` & `deck/slides/standing.html` — Key caveats hidden in `<aside>`**
    - **Issue:** On `e2e.html`, the withdrawn `+12%` tree speedup is only in `<aside>` (fine if you never sent the old number to the professor, worth keeping visible if they saw the website earlier). On `standing.html`, `"not directly comparable"` is only in `<aside>`, the table lacks arXiv IDs, and the `DFlash 2 (RedHatAI)` row carries no numbers.
    - **Fix:** Add `"not directly comparable (bf16/H200 vs 4-bit/4070 Ti)"` and arXiv IDs visibly on `standing.html`; either include RedHatAI's vendor numbers or drop the empty row.

---

### Low

16. **Inconsistent key for `ctx56` `"one sample"` (`deck/slides/scaling.html` & `RESULTS.md`)**
    - Rows 1–2 (`158M`, `285M`) use `accepted_single_k8` (`2.85/1.95/1.41`, `3.04/2.08/1.46`), while Row 3 (`ctx56`) uses `accepted_single_k16` (`3.00/2.00/1.43` instead of `accepted_single_k8` = `3.01/2.00/1.42`).
    - **Fix:** Use `accepted_single_k8` consistently (`3.01 / 2.00 / 1.42`).

17. **Minor precision & clarity nits**
    - `stage1.html`: `"Reaching 99% required..."` (chat only reached `97.3%`; also note the `z = 1,024` ablation did *not* use `--headfeat`/CE, otherwise the 94% vs 99% jump looks like latent dim).
    - `idea.html`: Clarify `"the last 8 verified positions"` (sliding window) rather than `"8 consecutive positions"`, and mention trees include `+ v2 chain` (matching `e2e.html`).
    - `e2e.html` & `RESULTS.md`: `"18 to 32%"` / `"45 to 55%"` are `17.5–31.3%` (`18–31%`) and `45.0–55.5%` (`45–56%`) from exact JSON; adding `τ` to `e2e.html` lets a reader verify the acceptance gain on-slide.
    - `RESULTS.md`: Still has two small leftovers from Milestone 1 (`Gist autoencoder and stage 2` says `"not over it on chat"`, and `e2e_final` bullet 3 says `"drafters with the same verification width agree on every prompt"` though `e2e_final` has no equal-width pair).

---

### Tone Check & Likely First Reviewer Attacks
- **Tone:** Clean and unpromotional overall; the only overstatements are `"Output is identical..."` (`cover.html`), `"about 100x more text"` and `"costs about one decode step"` (`website_gister_section.html`), and `"matches DFlash's implementation exactly"` (status table).
- **Expect these 3 questions first:** (1) On Slide 10, how is `oracle z` defined without collapsing into either an autoencoder or the averaged vector that failed on fine-tuned DFlash? (2) Why fund this direction when Gister is 29–49% of DFlash, trees lose on wall-clock, and ChainFlow/FlashMap already do latent-flow trees—and what exact GPU-hours does Test 1 need? (3) Why does the cover claim identical greedy output when `e2e_final.json` has `identical: 4/20` on MT-Bench?
