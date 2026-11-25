[gemini-4-neon via interactions, thinking_level=high, 221s]
[tokens: input 45,356 (cached 0), output+thinking 26,052; call $0.5268; total $7.5079 of $10.00]
[interaction_id: ChA3NmM1OWU3NmJiMGM0MmQ0EAgaAjgwKgRtYWlu]

### Summary of Audit Status

All raw metrics in **"Gister, audited state (25 Nov)"** match the attached JSONs and logs (`ae_gate_gist_hf_z2048.json`, `stage2_*_nfe2.json`, `stage2_*_h8.json`, `e2e_summaries.json`, `profile_tree_idle.json`, `e2e_final.log`, `run_final.log`). No critical fabrications were found.

Of the 8 earlier audit items: **greedy-only scope**, **withdrawn +12% claim**, **per-doubling gains**, **tau-minus-bonus comparison**, and **prior work named** are reflected in the audited section. Three items are **incomplete or contradicted by leftover text earlier in `RESULTS.md`**: stage-1 dual-gate failure, identical-output wording, and idle-vs-contended timings/conclusions.

---

### High

1. **Mismatched baseline & mislabelling in audited Stage-2 table (`_h8` row)**
   - **Location:** `RESULTS.md` → `Gister, audited state (25 Nov)` → Stage-2 offline table, row 2 (`same, plus 64 positions of gist history (_h8, 20M, old cache)`).
   - **Issue:** Placed directly under `v2 (154M, 2M positions)` (`3.08 / 2.22 / 1.45`) and labelled `"same"`, even though `_h8` (`stage2_gist_hf_z2048_det_raw_h8.json`) is `d=512, depth=6` (20M), `steps=6000`, `train_cache=""` (old cache). Its matched baseline—**Gister v1 (`20M, old cache, hist=1`: `2.80 / 2.23 / 1.43`)**—is missing from this table, so the reader sees an apparent −0.27 math drop vs v2 rather than the true ablation delta (`+0.01 / −0.08 / −0.04` vs v1). Also, `hist=8` is 8 chunks (64 positions total = 56 prior + 8 current, vs 8 current for `hist=1`).
   - **Fix:** Add the Gister v1 row (`deterministic, 20M, old cache (v1)`: `2.80 / 2.23 / 1.43`) right above `_h8` and change `"same, plus..."` to `"v1 + 7 prior gist chunks (hist=8, 64 total positions)"`.

2. **Top-of-file caveat still cites stale, GPU-contended latencies**
   - **Location:** `RESULTS.md` → `Scope and caveats` (top), bullet 4 (`Draft latency is unoptimised`).
   - **Issue:** Cites `27 ms chain, 35 to 67 ms trees` against a `4-bit target step of about 96 ms`. These are from the contended/throttled run (~10 tok/s AR) and directly contradict `profile_tree_idle.json` (plain decode `38.9 ms`; chain draft `11.2 ms`; tree draft `15.4–49.1 ms`) and `e2e_final.json` (~45 ms AR step; `53–54 ms` chain round; `78–83 ms` tree round).
   - **Fix:** Update bullet 4 to the idle-GPU numbers (`~11 ms` chain draft, `15–49 ms` tree draft before bf16 drafting, vs `~39–45 ms` 4-bit target decode step).

3. **False identical-output statement in `e2e.json` section; unsupported breadth in audited section**
   - **Location:** (a) `RESULTS.md` → `Gister (the two-stage design)` → paragraph below `e2e.json` table; (b) `RESULTS.md` → `Gister, audited state (25 Nov)` → `e2e_final` bullet 3.
   - **Issue:**
     - (a) Still says `"Differences from plain greedy (15 / 9 / 5 prompts) are 4-bit batched-verify numerics, the same for every drafter."` False: in `e2e_summaries.json` (`e2e`), `dflash` differs on **16 / 13 / 5** prompts (`identical` = `4 / 7 / 15`), contradicting the immediately preceding sentence.
     - (b) In `e2e_final`, **no two drafters share the same verification width** (`9`, `16`, `~33–37`, `~54–62`), so `"drafters with the same verification width agree on every prompt"` is not shown by `e2e_final.json` (only `e2e.json` had three width-9 drafters, and `e2e_summaries.json` only logs `identical`-vs-AR counts, not pairwise text equality). Moreover, `"diverges at near-ties"` is asserted for Qwen3-8B 4-bit without a logged logit-gap check (unlike the 1.5B study), despite high divergence rates (`15–16/20` on MT-Bench, `9–13/20` on GSM8K).
   - **Fix:** Delete `"the same for every drafter"` in (a) and specify `15 / 9 / 5` applies to width-9 drafters only. In (b), attribute equal-width agreement to `e2e.json`, state whether pairwise agreement was actually checked (vs matching AR counts), and qualify `"at near-ties"` unless a Qwen3-8B logit-gap check exists.

---

### Medium

4. **Stage-1 `agree_ok` failure still obscured in two earlier sections**
   - **Location:** (a) `RESULTS.md` → `Scope and caveats` bullet 1; (b) `RESULTS.md` → `Gist autoencoder and stage 2 (later runs)` bullet 1.
   - **Issue:** (a) `"matches the target's top-1 token 99.4 / 99.0 / 97.3% ... but its pre-registered gate also required ... TV <= 0.005 to 0.01"` implies top-1 passed and only TV failed. (b) `"Near the gate, not over it on chat"` implies math and code passed. Both conflict with `ae_gate_gist_hf_z2048.json` (`agree_ok: false` and `tv_ok: false` on all three sets).
   - **Fix:** State in both places that top-1 agreement also falls short of `99.5 / 99.5 / 99.0%` on all three sets (`agree_ok: false` and `tv_ok: false`).

5. **Idle-GPU profile omits the pre-bf16-fix caveat, creating a 10 ms discrepancy with `e2e_final`**
   - **Location:** `RESULTS.md` → `Gister, audited state (25 Nov)` → `e2e_final` bullet 2.
   - **Issue:** Quotes `profile_tree_idle.json` (`30 ms draft + 62 ms verify = 92 ms` for 67 nodes) directly under the `e2e_final` table (`82–83 ms/round` for K=8+chain) without mentioning `profile_tree_idle.json`'s explicit `source` caveat: `"before the bf16-drafting fix in e2e.py"` (and fixed `ctx+200`, 67 nodes, 285M sampler, vs `54–62` mean nodes in `e2e_final`).
   - **Fix:** Add `"(285M sampler, before the bf16-drafting fix, fixed ctx+200)"` so the 92 ms sum doesn't contradict the 82–83 ms table row.

6. **Stale ablation status relative to `logs/run_final.log`**
   - **Location:** `RESULTS.md` → `Gister, audited state (25 Nov)` → paragraph below Stage-2 table (`Two ablations are running...`).
   - **Issue:** `logs/run_final.log` shows `full-context test` **completed at 11:06** (`=== full-context done 11:06`, though with no metrics printed to `run_final.log`), and only `direct-token ablation` started at 11:06. (Also note `=== examples 09:46` crashed with `UnicodeEncodeError: '\u178b'`.)
   - **Fix:** Inspect the full-context output/JSON and either report its result or note why it produced no log output; update status to only the direct-token ablation running.

7. **Contradictory / broken prose left in `e2e_tree.json` section**
   - **Location:** `RESULTS.md` → `Gister (the two-stage design)` → paragraph below `e2e_tree.json` table.
   - **Issue:** Still claims desktop GPU contention `"lowers absolute speeds but not the comparison"`—directly disproven by `e2e_tree` / `e2e_tree_fast` vs `e2e_final` (contention made trees tie or beat chain). Also, the spliced edit `"on an idle GPU it does not (see the audited section below) (profile_tree.py), so the overhead is drafting and tree bookkeeping, being cut."` is a non-sequitur, and `"Trees raise tau 18% to 31%"` ignores the 1-step tree row (`+12%` MT-Bench, `+15%` HumanEval).
   - **Fix:** Replace that paragraph with a concise note that `e2e_tree` was GPU-contended (distorting tree-vs-chain speedups) and is superseded by `e2e_final` below.

---

### Low

8. **Confounded scaling factor list omits training steps and double-counts model size**
   - **Location:** `RESULTS.md` → `Gister, audited state (25 Nov)` → `"changes four things at once (parameters, 3x data, width and depth, warm start vs from scratch)"`.
   - **Issue:** `"parameters"` and `"width and depth"` are the same change, while **training steps (`10k → 24k`)** also changed (`steps: 24000` in the 285M JSON) and isn't listed.
   - **Fix:** List `(model size 158M→285M, 3x data, 2.4x steps 10k→24k, warm start vs from scratch)`.

9. **Minor rounding, key-naming, and verification nits**
   - **Location:** `RESULTS.md` → `Gister, audited state (25 Nov)`.
   - **Issues:**
     - `e2e_final` bullet 2: `"18 to 32%"` tau gain and `"45 to 55%"` ms/round cost are actually **`17.5% to 31.3%`** (`18–31%`) and **`45.0% to 55.5%`** (`45–56%`) from exact JSON values; `tree4d` ms/round is `77.5–78.2` (written `77 to 78`).
     - Stage-2 table: `"one sample"` uses `accepted_single_k8` (`2.85 / 1.95 / 1.41` and `3.04 / 2.08 / 1.46`), whereas `accepted_pred` is `2.78 / 1.97 / 1.41` and `3.00 / 2.07 / 1.46`—name the key. Chat `%` deltas (`+18% / +27% / +22% / +32%`) use unrounded v2 (`~1.454`), which looks 1 pt off if checked against table `1.45` (`+19% / +28% / +22% / +33%`).
     - `prune_oracle.json` (`none.tree64`) and `dflash_s2_latent.json` are not attached here, and the table doesn't explicitly confirm `none.tree64` uses the same 8-drafted cap (max 9) as Gister.
     - Earlier sampler section quotes the 2-step per-doubling gains (`+0.31/+0.25/+0.13`) directly beneath the 8-step table (whose diffs are `+0.33/+0.27/+0.14`) before introducing 2-step sampling.
     - `Data hygiene`: clarify that `"first 14,000 conversations of shard 0"` covers all three caches (`cache_pb`, `cache_pb2`, `cache_pb3`, 6M positions) rather than only `cache_pb`.
     - `e2e_tree_big` (idle GPU, showing K=16 trees fall below 1x AR: `0.74x` MT-Bench, `0.77x` HumanEval) is in `e2e_summaries.json` but unmentioned.
     - `Positioning` omits years/arXiv IDs for ChainFlow, FlashMap, MFD, DBLast, CALM; `What the results say` at the bottom still summarizes only the old 1.5B study.
   - **Fix:** Tighten exact percentages (`18–31%`, `45–56%`), name `accepted_single_k8`, state the `prune_oracle` cap and 6M-cache coverage, add a one-line mention of `e2e_tree_big` K=16 (<1x AR), add citation IDs in Positioning, and label `What the results say` as `(1.5B study)`.
