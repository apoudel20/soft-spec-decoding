[gemini-4-neon via interactions, thinking_level=high, fetched after a local print error]
[interaction_id: ChA2ZmY4NzU5NTZiM2FiY2MxEAgaAmJmKgRtYWlu]

### Executive Verdict

Most raw table entries extracted directly from the attached JSONs are transcription-accurate, **but the write-ups and compute-pitch deck contain several direct contradictions with the JSONs, a broken wall-clock benchmark claim, and a major missing control that a professor allocating cluster compute will catch immediately.**

Specifically:
1. Two documents explicitly claim outputs are **identical across drafters**, while `e2e_summaries.json` shows `identical` counts dropping from `11/20` to `7/20` across drafters.
2. Slide 5 attaches **8-step** sampler numbers (including a `K=4` stat that was never run at 2 steps) to a **"2 Euler steps"** headline, and both `RESULTS.md` and Slide 5 overstate chat's per-doubling gain by more than 2× (`+0.3` claimed vs `+0.12–0.14` actual).
3. The sole positive tree speedup (**+12% on GSM8K** in `e2e_tree_fast.json`) is physically impossible without mid-run GPU un-throttling (`tree16` with 97 nodes ran 6.7 ms/round *faster* than `tree8` with 56 nodes).
4. The claimed **`27 ms` chain drafting / `96 ms` target step** directly contradicts the round latencies implied by `e2e_summaries.json` (`~47–63 ms` idle AR step, only `~4.0–4.6 ms` chain overhead).
5. There is **no ablation proving the stage-1 gist VAE helps** over predicting 8 tokens directly from `--raw` target states—and every indirect control (`--lat_w 0`, `dflash_s2_latent` vs `refine`, worse cosine reconstruction with `--headfeat`) currently points the other way.

---

### CRITICAL Findings

#### C1. False "output identical across drafters" claim — directly contradicted by `e2e_summaries.json`
- **Locations:**
  - `RESULTS.md` (*Gister*, `e2e.json` paragraph): *"All drafters give identical text on every prompt. Differences from plain greedy (15 / 9 / 5 prompts) are 4-bit batched-verify numerics, the same for every drafter."*
  - `deck/outline.md` (Slide 6): *"Output identical across drafters."*
- **Evidence (`results/e2e_summaries.json`):**
  - `e2e.sets.mtbench.{dflash,dflash8,gister,gister_v2}.identical`: `4` vs `5, 5, 5` (16 diffs for `dflash`, not 15).
  - `e2e.sets.gsm8k.{dflash,dflash8,gister,gister_v2}.identical`: `7` vs `11, 11, 11` (13 diffs for `dflash`, not 9).
  - `e2e_tree.sets.gsm8k.{gister_v2,gister_tree8d_n2,gister_tree16d_n2,gister_tree8d_n1}.identical`: `11`, `9`, `7`, `8`.
  - `e2e_tree.sets.humaneval.*.identical`: `15`, `16`, `18`, `18`.
  - `e2e_tree_fast.sets.gsm8k.*.identical`: `11`, `9`, `7`.
  - (Also `dflash_eval.sets.chat.identical`: `0/10`.)
- **Fix:** Restrict the identical-text statement strictly to drafters sharing the exact same verification width/mask (`dflash8`, `gister`, `gister_v2` in `e2e.json`). Explicitly report that `dflash` (width 16) and every tree size (widths 43–113) diverge differently under 4-bit NF4 batched numerics, and verify/report the top-1/top-2 logit margin at first divergence (as done in the 1.5B study) to prove the verification loop itself is bug-free.

#### C2. Slide 5 mixes 8-step numbers under a "2 Euler steps" label, cites a non-existent `K=4` stat, and overstates chat scaling by >2×
- **Locations:**
  - `deck/outline.md` (Slide 5): *"Flow-matching sampler, 2 Euler steps. 8 samples give 7.6 to 8 distinct drafts per round. Best of K over deterministic: +7 to 9% (K=4), +17 to 19% (K=8), +27 to 30% (K=16), about +0.3 tokens per doubling of K, in every domain."*
  - `RESULTS.md` (*Gister*, sampler paragraph): *"grows about +0.3 tokens per doubling of K, in every domain."*
- **Evidence:**
  - `results/stage2_eval_gist_hf_z2048_flow_raw_cache_pb_d1024x12_s0_nfe2.json` (`nfe = 2`):
    - `args.bestk = "8,16"` — **no `K=4` run exists for 2 Euler steps.**
    - `sets.dev_math.distinct_drafts_k8 = 7.466` (`7.5`, not `7.6`; code `7.98`, chat `8.00`).
    - Relative to deterministic v2 (`stage2_gist_hf_z2048_det_raw_cache_pb_d1024x12_s20000.json`, `3.076 / 2.224 / 1.455`):
      - `accepted_best8` (`3.724 / 2.640 / 1.720`): **+21.0% / +18.7% / +18.2%** (not `+17 to 19%`).
      - `accepted_best16` (`4.035 / 2.891 / 1.851`): **+31.2% / +30.0% / +27.2%** (not `+27 to 30%`).
  - `results/stage2_gist_hf_z2048_flow_raw_cache_pb_d1024x12_s10000.json` (`nfe = 8`):
    - `distinct_drafts_k8` is `7.60 / 8.00 / 8.00`, `K=4` is `+9.4% / +6.9% / +8.5%`, `K=8` is `+19.3% / +16.3% / +16.6%` (`+16` to `19%`, not `+17` to `19%`), `K=16` is `+29.9% / +28.6% / +26.5%`.
  - **Per-doubling gain on chat (`sets.dev_chat`):**
    - `nfe = 8`: `K=4→8` is `1.696 - 1.579 = +0.12`; `K=8→16` is `1.840 - 1.696 = +0.14`.
    - `nfe = 2`: `K=8→16` is `1.851 - 1.720 = +0.13`.
    - (Code is `+0.21` to `+0.27`; only math is `+0.31` to `+0.33`.)
- **Fix:** Use `nfe=2` numbers consistently on Slide 5 (run `bestk=4,8,16` for `nfe=2` or drop `K=4`), fix distinct drafts (`7.5 to 8.0`), fix percentages (`+18 to 21%` at K=8, `+27 to 31%` at K=16), and replace "+0.3 in every domain" in both files with "+0.31 math, +0.25 code, +0.13 chat per doubling."

#### C3. The "+12% GSM8K tree speedup" (`e2e_tree_fast.json`) is invalid mid-run throttling noise, and GPU contention *does* distort the chain-vs-tree comparison
- **Locations:**
  - `RESULTS.md` (*Scope and caveats* #3): *"The tree's +12% on GSM8K comes from the throttled run and is re-measured on an idle GPU in `e2e_final.json`."*
  - `RESULTS.md` (*Gister*, `e2e_tree` paragraph): *"Desktop apps shared the GPU during this run (plain greedy 10 tok/s against 21 earlier), which lowers absolute speeds but not the comparison."*
  - `deck/outline.md` (Slide 6): *"Earlier +12% GSM8K tree win came from a throttled run..."*
- **Evidence (`results/e2e_summaries.json`):**
  - Derived ms/round (`tau / tok_s * 1000`) in `e2e_tree_fast.sets.gsm8k`:
    - `gister_v2`: `2.387 / 21.127 = 113.0 ms/round` (`speedup: 2.269x`)
    - `gister_tree8d_n2` (`tree_nodes: 55.6`): `2.936 / 21.223 = 138.3 ms/round` (`speedup: 2.279x`)
    - `gister_tree16d_n2` (`tree_nodes: 96.6`): `3.118 / 23.681 = 131.7 ms/round` (`speedup: 2.543x`, `+12.1%`)
    - **A 16-sample, 97-node tree ran 6.7 ms/round FASTER than an 8-sample, 56-node tree**, proving GPU contention lifted mid-run during `tree16`!
  - In the *same* `e2e_tree_fast` run on `mtbench` (`gister_v2: 88.5 ms`, `tree8: 114.2 ms`, `tree16: 126.2 ms`), trees were **8.8% and 12.3% SLOWER** than the chain (`1.40x` and `1.34x` vs `1.53x`)—and `e2e_tree_fast` is never tabulated in `RESULTS.md`.
  - In `e2e_tree.sets.gsm8k`, `tree16` was slower per round than `tree8` (`139.6` vs `129.0 ms`) and slower than the chain (`2.24x` vs `2.31x`).
  - Furthermore, slowing the GPU-bound target step by 2× (10 vs 21 tok/s) artificially reduces the relative overhead of CPU/Python tree bookkeeping, so GPU contention **does** distort the tree-vs-chain comparison.
- **Fix:** Remove the "+12% GSM8K" claim entirely (document it as an invalid measurement where `tree16` beat `tree8` in per-round latency), delete the claim that GPU sharing doesn't affect the comparison, and report clean idle-GPU results once `e2e_final.json` is run (verifying monotonic ms/round with tree size and consistent `ar_tok_s` across sets).

#### C4. Missing foundational ablation: zero evidence that the stage-1 gist VAE helps over predicting tokens directly from `--raw` target states
- **Locations:** `deck/outline.md` (Slides 1, 2, 10), `RESULTS.md` (*Gist autoencoder and stage 2* & *Gister*), `AGENT_NOTES.md` (lines 45, 51–55).
- **Evidence:**
  - `ae_gate_gist_tok_z2048.json` (no `--headfeat`) got `94.1 / 92.7 / 78.3%`, no better than `ae_gate_z2048_full.json` (`94.6 / 93.0 / 78.3%`). Only feeding `W[top1]` (`ae_gate_gist_hf_z2048.json`) reached `99.4 / 99.0 / 97.3%`, while hidden-state cosine error **worsened** from `0.234` (`z2048_full.train_log[-1].cos`) to `0.330` (`gist_hf_z2048.train_log[-1].cos`). Stage 1 is largely encoding the discrete argmax token IDs, not target states.
  - Dropping the latent MSE loss completely (`stage2_gist_hf_z2048_det_raw_lat0.json`, `lat_w: 0`) leaves acceptance virtually unchanged (`2.79 / 2.20 / 1.40` vs `2.80 / 2.23 / 1.43` in `det_raw.json`).
  - In the DFlash side experiment, routing through the gist latent **hurt** on every domain compared to a direct refine head (`dflash_s2_latent.json`: `5.30 / 4.51 / 2.25` vs `dflash_s2_refine.json`: `5.61 / 5.27 / 2.45`).
  - The only "gist beats tokens" experiment (`stage2_gist_hf_z2048_det*.json` vs `stage2_tokvae_z128_det*.json`) is doubly confounded: `tok_vae` used a **16× smaller latent** (`zdim: 128`, `steps: 4000`, admitted "undertrained" in `AGENT_NOTES.md` line 114) and **never received `--raw` target states as input**.
- **Fix:** Before asking for cluster compute to scale stage 2 to 1B parameters, run a clean control with the exact same `--raw` input predicting the next 8 tokens directly (no stage-1 VAE, or a matched `z=2048` token VAE + `--raw`). Caveat the `z128` vs `z2048` comparison explicitly in all docs.

#### C5. Claimed `27 ms` chain drafting / `96 ms` target step directly contradicts `e2e_summaries.json`
- **Locations:**
  - `RESULTS.md` (*Scope and caveats* #4): *"Gister drafts in eager PyTorch (27 ms chain, 35 to 67 ms trees) against a 4-bit target step of about 96 ms."*
  - `deck/outline.md` (Slide 10): *"drafter latency on fast engines (27 to 67 ms eager vs 10 to 15 ms bf16 target steps)"*
- **Evidence (`results/e2e_summaries.json`):**
  - No profiling JSON exists in `results/` for `27 ms`, `35–67 ms`, or `96 ms`.
  - On an idle GPU (`e2e.sets`): AR step (`1000 / ar_tok_s`) is **`47.1 ms` (MT-Bench), `63.0 ms` (GSM8K), `52.1 ms` (HumanEval)** (`43.4–47.9 ms` in `dflash_eval.sets`). `gister_v2` **total** round time (`tau / tok_s * 1000`) is **`51.1 / 67.3 / 56.7 ms`**—an overhead of only **`4.0–4.6 ms`** over an AR step. A `27 ms` chain draft is mathematically incompatible with a `51.1 ms` total round time!
  - On the throttled GPU (`e2e_tree.sets`): AR step is **`96.3 / 100.2 / 100.1 ms`** (so `96 ms` is the throttled step!), and `gister_v2` chain round is `100.6 / 103.6 / 103.6 ms`—still only **`3.4–4.3 ms`** above AR.
  - (Also note: `e2e_tree.sets` shows `tree8d_n2` adds `29–42 ms` over AR and `tree16d_n2` adds `39–49 ms` over AR, on the throttled GPU.)
- **Fix:** Re-profile on an idle GPU, commit the profiling JSON, reconcile the `27 ms` vs `~4 ms` chain discrepancy, and explicitly analyze the cluster bf16 latency budget—including the cost of running the stage-1 decoder + 152K-vocab LM head on `16 × 8 = 128` positions per round for a 16-branch tree.

---

### HIGH Findings

#### H1. Stage-1 top-1 gate failure (`agree_ok: false` on all 3 domains) is downplayed or omitted
- **Locations:**
  - `RESULTS.md` (*Scope & caveats* #1; *Gist autoencoder and stage 2*): *"Near the gate, not over it on chat."*
  - `AGENT_NOTES.md` (line 42): *"Stage 1 nearly passes on top-1... It fails the gate's distribution check"*
  - `TWO_GPU_PLAN.md` (line 17): *"Stage 1 is fine."*
  - `deck/outline.md` (Slide 3): caveat mentions only the distribution check failing.
- **Evidence (`results/ae_gate_gist_hf_z2048.json`):**
  - `sets.dev_math`: `agree = 0.99414` (< 0.995), `gate.agree_ok = false`, `gate.pass = false`.
  - `sets.dev_code`: `agree = 0.99008` (< 0.995, 2× allowed error), `gate.agree_ok = false`, `gate.pass = false`.
  - `sets.dev_chat`: `agree = 0.97315` (< 0.990, 2.7× allowed error), `gate.agree_ok = false`, `gate.pass = false`.
- **Fix:** State clearly in all four docs that `gist_hf_z2048` failed **both** `agree_ok` (on math, code, and chat) and `tv_ok` (on all three). Remove *"Stage 1 is fine"* from `TWO_GPU_PLAN.md`.

#### H2. Slide 7's "~half of DFlash on math and chat, under 40% on code" is inflated by the free `+1` bonus token
- **Location:** `deck/outline.md` (Slide 7): *"At equal tree budget offline: Gister ~half of DFlash on math and chat, under 40% on code."*
- **Evidence:**
  - Equal ~64-node budget offline (HF top-1, max 9, same 655/450/743 dev rounds):
    - DFlash `tree64` (`prune_oracle.json -> sets.*.none.tree64`): `7.003 / 6.938 / 3.693` -> **`6.00 / 5.94 / 2.69` drafted tokens accepted (`tau - 1`)**.
    - Gister 158M `best8` (`stage2_eval_..._nfe2.json -> accepted_best8`): `3.724 / 2.640 / 1.720` (`53% / 38% / 47%` of raw `tau`, but only **`45% / 28% / 27%`** of drafted tokens accepted!).
    - Gister 285M `best8` (`stage2_..._d1280x14_s24000_nfe2.json -> accepted_best8`): `3.940 / 2.731 / 1.774` (`56% / 39% / 48%` of raw `tau`, but only **`49% / 29% / 29%`** of drafted tokens accepted!).
  - On chat, even the trivial copy baseline (`tau = 1.02`) is `28%` of DFlash by raw `tau`.
- **Fix:** Report **both** `tau` and drafted tokens accepted (`tau - 1`) on Slide 7 so the true gap on chat (~27–29% of DFlash's drafted tokens) and code (~28–29%) is transparent.

#### H3. Finished 285M / 6M-position scaling run is missing from `RESULTS.md`, unfilled in `deck/outline.md`, and shows a weak +3–6% gain across 4 confounded variables
- **Locations:** `RESULTS.md` (omitted), `deck/outline.md` (Slides 1, 7, 8).
- **Evidence:** `results/stage2_gist_hf_z2048_flow_raw_cache_pb+cache_pb2+cache_pb3_d1280x14_s24000_nfe2.json` vs `results/stage2_eval_gist_hf_z2048_flow_raw_cache_pb_d1024x12_s0_nfe2.json`:
  - `accepted_best8`: Math `3.724 -> 3.940` (`+5.8%`, +0.22), Code `2.640 -> 2.731` (`+3.5%`, +0.09), Chat `1.720 -> 1.774` (`+3.1%`, +0.05).
  - `accepted_best16`: Math `4.035 -> 4.176` (`+3.5%`, +0.14), Code `2.891 -> 3.060` (`+5.8%`, +0.17), Chat `1.851 -> 1.923` (`+3.9%`, +0.07).
  - Four variables changed simultaneously: model size (`1024×12` -> `1280×14`), data (`cache_pb` -> `cache_pb,cache_pb2,cache_pb3`), steps (`10k` -> `24k`), and init (`stage2_gister_v2` -> from scratch).
- **Fix:** Add these numbers to `RESULTS.md` and Slide 8, explicitly note all four confounded variables, and address directly why a `+0.05` token chat gain for 3× data / 1.8× params justifies scaling to 100M+ positions (e.g., arguing the bottleneck is the 8-token context window, not data/params alone—once `stage2_raw_h8` confirms it).

#### H4. Cross-convention and oracle-vs-drafter comparisons in `AGENT_NOTES.md` and `RESULTS.md`
- **Locations:**
  - `AGENT_NOTES.md` (line 43): *"2.2 / 1.8 / 1.3 accepted per round (out of 9) against DFlash's 7.5 / 6.8 / 3.0 (out of 16)."*
  - `AGENT_NOTES.md` (line 47): *"A perfect predictor (the true next latent) gives 8.5 / 8.5 / 7.6, far above DFlash."*
  - `RESULTS.md` (*Gist autoencoder and stage 2*): *"Both are far below DFlash (7.46 / 6.77 / 2.96 out of 16)."*
- **Evidence:**
  - `stage2_gist_hf_z2048_det.json` uses **HF top-1 labels, max 9**; `dflash_tree_gap.json` (`7.46 / 6.77 / 2.96`) uses **Ollama text labels, max 16**. Fair HF-top-1 max-9 DFlash numbers already exist in `dflash_s2_latent.json` (`5.88 / 5.59 / 2.68`).
  - Comparing a ground-truth **oracle** (`8.51 / 8.47 / 7.63`, which even `stage2_tokvae_z128_det.json` nearly matches at `8.50 / 8.21 / 7.60`) against a non-oracle drafter (`DFlash`) is apples-to-oranges; DFlash's oracle ceiling is `9.0` (or `16.0`).
- **Fix:** Compare Stage-2/Gister strictly against HF-top-1 cap-8 DFlash (`5.88 / 5.59 / 2.68` chain; `7.00 / 6.94 / 3.69` tree-64 in `prune_oracle.json`), and frame `8.51 / 8.47 / 7.63` strictly as the stage-1 reconstruction ceiling out of 9.

#### H5. Overclaim that Gister v2 hit the "averaging floor" on "unseen text" (ignores epoch count and domain shift)
- **Locations:**
  - `RESULTS.md` (*Gister*, v2 paragraph): *"Training loss levelled near 4.0 on unseen text, so data helps math only: the deterministic predictor sits on the averaging floor."*
  - `deck/outline.md` (Slide 4): *"Data stopped the overfitting, then hit that floor (loss ~4.0 nats on unseen text)."*
  - `TWO_GPU_PLAN.md` (line 40): *"training loss 4.00 on mostly unseen text"*
- **Evidence (`results/stage2_gist_hf_z2048_det_raw_cache_pb_d1024x12_s20000.json`):**
  - `args.steps = 20000`, `args.B = 128` -> `2,560,000` training samples drawn from `2.0M` positions (`>= 1.28` epochs even at 1 pair/position, or `~10` epochs if chunk-aligned). Late-step loss (`3.67–4.09` nats) is not on unseen text.
  - `args.train_cache = "cache_pb"` (PerfectBlend) replaces v1's in-domain math/code/chat cache. Flat `dev_code` (`2.23 -> 2.22`) and `dev_chat` (`1.43 -> 1.45`) are confounded by domain shift.
- **Fix:** Report *"~3.8–4.1 nats after >1 pass over 2M PerfectBlend positions"* and note that the flat code/chat result is confounded with PerfectBlend's domain mix.

#### H6. Unbacked / not-yet-run JSONs behind key pitch claims
- **Locations:** `deck/outline.md` (Slides 6–10), `RESULTS.md` (Scope #3–4, smoke test, and all Qwen2.5-1.5B + early Qwen3-8B sections), `TWO_GPU_PLAN.md` (line 21).
- **Evidence:** No attached JSON exists for `e2e_final.json`, `stage2_raw_h8*.json`, `examples.json`, `profile_tree*.json`, `time_forward*.json` (`3,000 tok/s`, `6.4 GB`), the 2-prompt `e2e` smoke test, `1.5–2 s/step` (though `d1280x14_s24000` `train_log` implies `0.47 s/step` at 285M), or the earlier Qwen2.5-1.5B / Qwen3-8B probe/oracle sections.
- **Fix:** Run and commit `e2e_final.json`, `stage2_raw_h8.json`, `examples.json`, `profile_tree.json`, and `time_forward.json` before sending the deck; clearly separate verified numbers from estimates.

#### H7. Slide 6 presents 4-bit NF4 flat verification cost as a general property
- **Location:** `deck/outline.md` (Slide 6): *"Verifying ~100 nodes costs the same as one decode step."*
- **Evidence:** `RESULTS.md` (*Selection gap*) explicitly caveats that flat latency up to 128 tokens occurs *"with bitsandbytes 4-bit weights... because dequantisation dominates. Which budget column applies in practice needs a measurement with real inference kernels."*
- **Fix:** Add the 4-bit bitsandbytes caveat directly on Slide 6, noting that bf16/FP8 cluster engines will incur a real verification penalty at ~100 tree nodes.

---

### MEDIUM Findings

#### M1. Gist vs token-VAE relative gain is `11 to 24%`, not `13 to 24%`
- **Location:** `AGENT_NOTES.md` (line 45): *"The gist beats a tokens-only latent by 13 to 24% in every stage-2 setting."*
- **Evidence:** `stage2_gist_hf_z2048_det_h8.json` (`dev_chat.accepted_pred = 1.3136`) vs `stage2_tokvae_z128_det_h8.json` (`dev_chat.accepted_pred = 1.1844`): `+10.9%` (`11%`, not `13%`).
- **Fix:** Change to `11 to 24%` and note the `z2048` vs `z128` confound.

#### M2. Calling 154M gist-only dev results "flat" ignores a +21% code gain
- **Locations:** `RESULTS.md` (*Gister* table, line 320); `AGENT_NOTES.md` (line 143).
- **Evidence:** `stage2_gist_hf_z2048_det.json` (`2.166 / 1.804 / 1.336`) vs `stage2_gist_hf_z2048_det_d1024x12_s20000.json` (`2.299 / 2.189 / 1.303`): dev code rose **+21.3%** (`1.80 -> 2.19`) and math +6.1%; only chat fell slightly (`1.34 -> 1.30`).
- **Fix:** Replace "dev flat" with "dev chat drops slightly (1.34 -> 1.30), math +6%, code +21%."

#### M3. Slide 4's "Code and chat flat" misstates the `--raw` gain on code
- **Location:** `deck/outline.md` (Slide 4): *"Deterministic predictor: 2.17 -> 2.80 (raw target states) -> 3.08 (154M, 2M positions) on math. Code and chat flat."*
- **Evidence:** `det.json` -> `det_raw.json` -> `det_raw_cache_pb_d1024x12_s20000.json`: code jumped `1.80 -> 2.23` (**+24%**) from `--raw` and was only flat from v1 to v2 (`2.23 -> 2.22`); chat went `1.34 -> 1.43 -> 1.45`.
- **Fix:** State that `--raw` lifted all domains (+29% / +24% / +7%), and only the v1→v2 step was flat on code and chat.

#### M4. Inaccurate explanation of GSM8K speedups in `e2e.json`
- **Location:** `RESULTS.md` (*Gister*, `e2e.json` paragraph): *"Plain greedy ran at 21.2 / 15.9 / 19.2 tok/s. The low GSM8K baseline inflates every GSM8K speedup by the same factor."*
- **Evidence (`results/e2e_summaries.json -> e2e`):** `speedup / tau` for `gister_v2` is `0.922` (MT-Bench), `0.936` (GSM8K), `0.917` (HumanEval). Both AR (`63.0 ms`) and `gister_v2` (`67.3 ms`) slowed together on GSM8K; speedups were not inflated by `21.2 / 15.9 = 1.33×`.
- **Fix:** State that both AR and speculative steps ran ~25–30% slower on GSM8K, keeping `speedup / tau` consistent (~0.92–0.94).

#### M5. `RESULTS.md` Scope #2 misstates `e2e.py` draft cap
- **Location:** `RESULTS.md` (*Scope and caveats* #2): *"Every stage-2, Gister and `e2e.py` number scores against the HF target's own top-1 with up to 8 drafted (max 9)."*
- **Evidence:** `e2e_summaries.json -> e2e.sets.humaneval.dflash.tau = 10.006` (drafts 15, max 16).
- **Fix:** Exclude the full `dflash` (15-drafted) row in `e2e.json` and `dflash_eval.json`.

#### M6. 3× training-set size contradiction in `AGENT_NOTES.md`
- **Locations:** `AGENT_NOTES.md` line 68 (*"2,000 answers (512K positions)"*) vs line 161 (*"2,000 train answers and 40 dev answers each for math, code, chat"*).
- **Evidence:** Dev sets in `ae_gate_*.json` have `10,584 / 7,360 / 11,920` positions over 40 answers each (~249 tokens/answer). `512,380` train positions is ~2,059 answers **total**, not 2,000 per domain (~6,000 answers).
- **Fix:** Correct line 161 to give the true per-domain train counts (~667 each, ~2,000 total).

#### M7. Inconsistent, unlogged model parameter counts across docs
- **Locations:**
  - Gister v1 (`d=512, depth=6`): `20M` (`RESULTS.md`) vs `22M` (`TWO_GPU_PLAN.md`); gist-only `d=512, depth=6`: `20M` (`RESULTS.md`) vs `19M` (`AGENT_NOTES.md`).
  - Gister v2 / flow (`d=1024, depth=12`): `154M` (`RESULTS.md`, Slide 4) vs `158M` (Slides 7–8).
  - None of `ae_gate_*.json` or `stage2_*.json` log `n_params` (including `161M` and `285M`).
- **Fix:** Log exact `n_params` in every JSON and align all docs/slides.

#### M8. Systematic `accepted_pred` vs `accepted_single_k*` discrepancy in flow JSONs (higher value quietly reported)
- **Location:** `RESULTS.md` (*Gister*, sampler table): *"sampler, one sample: 2.71 / 1.83 / 1.36."*
- **Evidence:** In `stage2_gist_hf_z2048_flow_raw_cache_pb_d1024x12_s10000.json`, `accepted_pred` is **`2.632 / 1.780 / 1.349`**, while `accepted_single_k8` is `2.710 / 1.827 / 1.365`. The same systematic `+0.04–0.08` math gap appears in `nfe1`, `nfe2`, `nfe4`, and `d1280x14`.
- **Fix:** Audit why `accepted_pred` differs systematically from `accepted_single_k*` in `stage2.py` and label the reported column accurately (e.g., mean single draw across `K=8`).

#### M9. Silent omission of `nfe=4` (which beats `nfe=2` on math)
- **Location:** `RESULTS.md` (*Gister*, flow steps line): reports 8, 2, and 1 steps only.
- **Evidence:** `results/stage2_eval_gist_hf_z2048_flow_raw_cache_pb_d1024x12_s0_nfe4.json` gets `accepted_best8 = 3.762 / 2.622 / 1.713` (`3.76 > 3.72` on math).
- **Fix:** Add `4 steps: 3.76 / 2.62 / 1.71` and justify choosing 2 steps on compute-per-step grounds.

#### M10. Tree tau range (`18 to 31%`) and round latency (`122 to 146 ms`) exclude the 1-step tree in the same table
- **Locations:** `RESULTS.md` (*Gister*, `e2e_tree` paragraph), `deck/outline.md` (Slide 6).
- **Evidence (`e2e_summaries.json -> e2e_tree`):** `gister_tree8d_n1` raises tau by **11.9% / 17.0% / 14.8%** (only 2-step trees are `17.6–30.6%`), and its HumanEval round cost is `1.9050 / 16.2062 * 1000 = 117.5 ms` (`118 ms`, not `122 ms`).
- **Fix:** Specify *"2-step trees raise tau 18 to 31% (1-step: 12 to 17%)"* and *"118 to 146 ms."*

#### M11. Unmentioned deployment cost & single-sample regression of flow trees on Slides 5–6
- **Locations:** `deck/outline.md` (Slides 5 & 6).
- **Evidence:** Single-sample flow (`nfe2`: `2.78–2.85 / 1.95–1.97 / 1.41`) is **worse** than deterministic v2 (`3.08 / 2.22 / 1.45`), and `e2e_tree.json` uses `tree8d`/`tree16d` (deterministic v2 chain **plus** K sampled branches), which requires holding/running **two** 154–158M stage-2 models at inference.
- **Fix:** Disclose the single-sample regression on Slide 5 and report tree results without the separate deterministic checkpoint (or disclose the 2× drafter memory/compute footprint).

#### M12. Slide 10 proposes "rolling caches (100M+ positions)" on a cluster without addressing TB storage or online harvesting
- **Location:** `deck/outline.md` (Slide 10).
- **Evidence:** `TWO_GPU_PLAN.md` line 25 states `32 KB/position` (`3.2 TB` per 100M positions, `25.6 TB` for 800M) and notes caching was only a 12 GB VRAM workaround.
- **Fix:** Propose online in-memory harvesting on 80 GB cluster GPUs (or explicitly budget multi-TB scratch storage and justify why caching beats online forward passes).

---

### LOW Findings

#### L1. Stale statements, placeholders, and internal contradictions
- `RESULTS.md` line 255: placeholder `see json` left in the `dflash_plan` table (`Blocks` column).
- `RESULTS.md` line 343: *"is in `results/e2e.json` once run"* — already run and tabulated 7 lines below.
- `RESULTS.md` lines 397–405 (*What the results say*): entirely stale (summarizes only Qwen2.5-1.5B, omits Qwen3-8B/Gister).
- `AGENT_NOTES.md` line 15 (*"out of 16"* everywhere) contradicts `k=8` (max 9) throughout the rest of the file; lines 35–38 (`--ctx 8` decided) contradict line 116 (`--ctx 8` dropped); lines 43–49, 60, 154, 175–183 are pre-v2/pre-flow/pre-`refine` stale (Task 2 claims HF-label DFlash hasn't been run yet despite `dflash_s2.py` and `prune_oracle.py`).
- `TWO_GPU_PLAN.md` lines 27–43: stale live status (*"running now"*, *"step 16,500 of 20,000"*).
- **Fix:** Clean up placeholders, update concluding/status sections, or label `AGENT_NOTES.md` / `TWO_GPU_PLAN.md` as dated historical logs.

#### L2. Minor numerical/citation discrepancies and typos
- **Unexplained DFlash chain shifts across identical dev splits:** `dflash_s2_latent.json` has `accepted_chain = 5.881 / 5.591 / 2.677` while `prune_oracle.json` (`none.chain`) has `5.916 / 5.598 / 2.688` on the same `655 / 450 / 743` rounds; similarly `dflash_plan`/`kv_oracle` report `7.46 / 6.71 / 2.94` vs `dflash_tree_gap.json`'s `7.46 / 6.77 / 2.96`.
- **Single-minibatch train metric:** `RESULTS.md` (line 297) and `AGENT_NOTES.md` (line 104) cite `94.5% on train` for `ae_gate_diag_big_z2048.json`—that is `train_log[-1].agree_train = 0.9453`, a single minibatch (adjacent logs fluctuate `93.2%–95.7%`).
- **Train vs dev latent MSE:** `AGENT_NOTES.md` line 141 (*"Latent MSE stays at 0.89 to 0.97"*) cites training `lat` loss (no dev latent MSE is logged in any JSON), and `stage2_tokvae_z128_det.json` ends at `0.988`.
- **Missing `--raw` flag note:** `RESULTS.md` line 325 puts the collapsed naive flow row (`stage2_gist_hf_z2048_flow.json`, `raw: false`) right under the `--raw` rows without noting `raw: false`.
- **External baseline notes:** `AGENT_NOTES.md` lines 84–85 write `Qwen3.8-27B` (probable typo) and list a RedHatAI Qwen3-8B checkpoint for DFlash2, whereas `deck/outline.md` Slide 7 says *"no Qwen3-8B author checkpoints"* and cites *"DFlash 4.86x avg"* (absent from `AGENT_NOTES.md`'s table).
- **Fix:** Explain or align the small baseline shifts, qualify single-batch train numbers, mark `raw: false` on the naive flow row, and reconcile external baseline citations/typos.