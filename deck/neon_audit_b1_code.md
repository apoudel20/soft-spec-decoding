[gemini-4-neon via interactions, thinking_level=high, 514s]
[tokens: input 54,384 (cached 0), output+thinking 58,448; call $1.0399; total $5.6564 of $10.00]
[interaction_id: ChA4NzUzYzBlNDQ3YTU1MDNkEAgaAjBjKgRtYWlu]

Below is an adversarial audit of all scripts in the repository, ranked by impact on reported numbers.

---

### CRITICAL

1. **Gister's stage-2 Search network runs in FP32 (no `bfloat16` autocast) in `e2e.py` and `profile_tree.py`, severely understating Gister's `tok/s`, `speedup`, and `draft ms` vs BF16 DFlash**
   - **Where**: `e2e.py:108-112`, `e2e.py:121-131`, `e2e.py:145-156` (inherited by `profile_tree.py:41`)
   - **Evidence**: DFlash is explicitly cast to BF16 (`dflash_eval.py:45`: `d.to("cuda", torch.bfloat16).eval()`). In `GistDrafter.draft()`, `ae.encode` (`e2e.py:145-146`) and `ae.decode` (`e2e.py:154-155`) are wrapped in `torch.autocast("cuda", torch.bfloat16)`, and `DFlashDrafter.draft()` wraps `self.head` in `autocast(bfloat16)` (`e2e.py:62-63`). However, `GistDrafter._sample()` (`e2e.py:151`) is called **outside** any `autocast` block, and `Search` weights are FP32 (`Search(...).cuda().eval()` at `e2e.py:108-109`, with TF32 never enabled). A 158M–285M stage-2 transformer running in FP32 incurs 2× weight bandwidth (~1.14 GB per NFE step for 285M) and non-Tensor-Core FP32 matmuls, biasing every reported Gister throughput, speedup, and draft latency measurement downward. (`stage2.py:269-279` `predict()` also runs eval in FP32 while training uses BF16 autocast.)
   - **Fix**: Cast `Search` (and `AE`) to `bfloat16` at load time in `e2e.py` (or wrap `self._sample(...)` in `with torch.autocast("cuda", torch.bfloat16):` and convert weights to `bfloat16` so decode isn't reading FP32 weights). Wrap `stage2.py:predict()` in `torch.autocast("cuda", torch.bfloat16)` as well.

2. **`stage2.py --eval_only --init <ckpt>` ignores the checkpoint's saved `zm`/`zs` normalization stats, silently corrupting offline dev numbers**
   - **Where**: `stage2.py:225-234` (contrast with saving at `stage2.py:309` and loading at `e2e.py:111`)
   - **Evidence**: `stage2.py` saves `"zm"` and `"zs"` in every checkpoint (`stage2.py:309`), and `e2e.py:111` loads them. In `stage2.py`, lines 225–227 always recompute `zm, zs` from `Z["train"]` (which defaults to `data/cache`, i.e., `q3_train`) even when `--eval_only --init <ckpt>` is passed. Evaluating a checkpoint trained on `cache_pb` or `cache_pb,cache_pb2,cache_pb3` (such as `stage2_eval_..._nfe2` referenced in `run_final.sh:25`) without matching `--train_cache` silently normalizes/denormalizes latents with the wrong mean and std, invalidating `accepted_pred`, `accepted_bestK`, and `slot_top1_pred`. It also relies on the user manually re-specifying matching `mode`, `t_shift`, `d`, `depth`, `hist`, `raw`, `ctx_pos`.
   - **Fix**: Under `args.eval_only` (when `args.init` is set), load `zm = ck["zm"].to(dev)` and `zs = ck["zs"].to(dev)` from the checkpoint and assert/restore the architectural and flow hyperparameters (`mode`, `t_shift`, `d`, `depth`, `hist`, `raw`, `ctx_pos`) from `ck["args"]`.

3. **`profile_tree.py` silently forces `nfe=2` on rows labeled `nfe8` and times without warm-up**
   - **Where**: `profile_tree.py:28-35`, `profile_tree.py:41-63`
   - **Evidence**: Rows are labeled `"flow K=4 nfe8"` and `"flow K=8+det nfe8"` (`profile_tree.py:29-30`), then lines 34–35 unconditionally overwrite `dr.main = dict(dr.main, nfe=2)` for every flow model. Any latency breakdown reporting `"nfe8"` from this script is actually measuring 2 Euler steps (~4× less sampler work). In addition, neither `draft`, `verify` (with 4D tree attention mask), nor `keep_path` (`reps=1` cold!) is warmed up before `timed()`, polluting timings with first-call kernel/allocator overhead.
   - **Fix**: Specify `nfe` explicitly per config in the benchmark list (remove the blanket `nfe=2` override), add explicit warm-up calls before every `timed(...)`, and run `keep_path` for `reps >= 10` (re-populating the cache outside the timed region or timing the slice-copy directly).

---

### HIGH

4. **`e2e.py:spec()` executes a wasted extra draft+verify round whenever EOS is emitted as the bonus token or at `len(out) == max_new - 1`, distorting `tau`, `tok/s`, `speedup`, and `tree_nodes`**
   - **Where**: `e2e.py:194-246`
   - **Evidence**: `spec()` never appends the newly computed `bonus` until the *next* loop iteration, and its stop checks (`e2e.py:225`, `239`) only inspect `new` (which excludes the new `bonus`). Whenever the target emits `<|im_end|>` / EOS as the bonus token (very common whenever the drafter misses EOS) or when `len(out) == max_new - 1`, `spec()` runs a full extra draft + verify round past EOS / `max_new`, appends that garbage round to `taus` and `drafter.nodes`, charges its wall-clock time to `s`, and then truncates `out` back to `stop` / `max_new`. (By contrast, `dflash_model.py:265-271` writes `bonus` to `output_ids` and breaks immediately on stop-token bonus.) Also, when `stop` occurs mid-`new`, `taus[-1]` counts accepted tokens *after* EOS.
   - **Fix**:
     1. After prefill (`e2e.py:193`), if `int(bonus) in stop or max_new <= 1`: return `[int(bonus)][:max_new], []`.
     2. In each round, if `stop` appears in `new`, truncate `new` (and `taus[-1]`) at the first stop token and `break`; if `len(out) >= max_new`, `break`; otherwise if `int(bonus) in stop or len(out) + 1 >= max_new`, `out.append(int(bonus))` and `break` without running another round.

5. **Missing `gister_train.py` breaks `cache_states.py --pb_tokens` / `run_scale2.sh`, leaves PerfectBlend train/dev/eval contamination unauditable, and uses fragile float shard offsets**
   - **Where**: `cache_states.py:55-68`, `run_scale2.sh:8-10`
   - **Evidence**: `cache_states.py:58` imports `from gister_train import load_convs`, which does not exist in the repo. This means (a) `cache_pb`, `cache_pb2`, `cache_pb3` (the 6M-token training data for the 158M/285M Gister models) cannot be regenerated, (b) there is no verifiable deduplication in the repo ensuring PerfectBlend prompts/answers do not overlap with GSM8K test/dev, MBPP val, UltraChat test, MT-Bench, or HumanEval, and (c) `run_scale2.sh:8` computes `OFF` as float millions of tokens and `cache_states.py:61` checks `skip < args.pb_offset * 1e6`; float rounding or non-exact offsets can silently duplicate or skip boundary conversations across `cache_pb*` shards.
   - **Fix**: Add `load_convs` to the repo with explicit prompt-level deduplication against all `dev_prompts` and `eval_prompts`, and shard by exact integer conversation indices (`--pb_start_conv` / `--pb_end_conv`) rather than float millions of tokens.

6. **Silent `bfloat16 -> float16` overflow risk for massive activations in `cache_states.py`**
   - **Where**: `cache_states.py:100-106`, `ae_gate.py:93-95`, `stage2.py:246-248`
   - **Evidence**: Qwen3-8B hidden states are `bfloat16` (range up to `3.4e38`) and middle layers (18, 24, 30) exhibit massive activations (`ae_gate.py:158` pins dims with `max |x| > 20 * median`). `cache_states.py:102` casts directly to `torch.float16` (max finite `65,504`) without checking finiteness, while computing `stats_L{l}.pt` from the uncast `bfloat16` tensor `x`. Any activation `> 65,504` silently writes `inf` into the `.npy` cache; if an unmasked dim overflows on dev (or is read without applying `mask`), downstream tensors silently become `inf`/`nan`.
   - **Fix**: Assert `torch.isfinite(x16).all()` during caching (or clamp to `[-65504, 65504]` with an explicit warning count, or store `bfloat16` as raw `uint16` memmaps).

7. **`dflash_s2.py` crashes on current `zcache` (`.pt` vs `.npy`), and `stage2.py` silently reuses stale `zcache` files after stage-1 retraining**
   - **Where**: `dflash_s2.py:98`, `stage2.py:186-192`
   - **Evidence**: `stage2.py:186` switched `zcache` to `.npy` memmaps and no longer writes `.pt` files, while `dflash_s2.py:98` hardcodes `torch.load(os.path.join(DATA, "zcache", f"{args.ae}_{sp}.pt"))`, failing with `FileNotFoundError` for any newly cached AE. Meanwhile, `stage2.py:187` reuses `zcache/{args.ae}_{sp}.npy` unconditionally without verifying it matches the current `ckpt/ae_{args.ae}.pt`; retraining an AE under the same `--name` silently pairs old cached encoder latents with the new decoder.
   - **Fix**: Update `dflash_s2.py` to load `.npy` (with `.pt` fallback), and include a short SHA256 hash of `ckpt/ae_{args.ae}.pt` in the `zcache` filename (or validate stored mtime/hash metadata).

8. **`identical` check in `e2e.py` and `dflash_eval.py` reports false positives when output lengths differ**
   - **Where**: `e2e.py:310-313`, `dflash_eval.py:89-95`
   - **Evidence**: Both slice to `n = min(len(out), len(ref))` and compute `identical = (out[:n] == ref[:n])` (`torch.equal(ref[:n], spec[:n])`). If speculative decoding terminates early or continues past AR's stop token (one is a strict prefix of the other), it is counted as `identical = True` with `first_diff = None`.
   - **Fix**: Use `out == ref` (`len(ref) == len(spec) and bool(torch.equal(ref, spec))`), and report `first_diff = n` when prefixes match but lengths differ.

---

### MEDIUM

9. **Warm-up tree node counts leak into the first eval prompt's `tree_nodes` in `e2e.py`**
   - **Where**: `e2e.py:200-201`, `e2e.py:295-296`, `e2e.py:314-316`
   - **Evidence**: Warm-up calls `spec(t, x, w, 16, stop)`, appending warm-up tree sizes to `x.nodes`, without clearing `x.nodes` before the benchmark loop begins. The first prompt of the first dataset (`mtbench`) therefore includes warm-up rounds in its `nodes` list and biases `sm["mtbench"]["tree_nodes"]`.
   - **Fix**: Clear `self.nodes.clear()` inside `GistDrafter.reset()` (or return `nodes` from `spec()` instead of mutating state on the drafter).

10. **Unweighted per-batch averaging biases `slot_top1_pred` in `stage2.py` and `slot_chain` / `slot_pred` in `dflash_s2.py`**
    - **Where**: `stage2.py:330-332`, `dflash_s2.py:149-151`
    - **Evidence**: Both append `eq.float().mean(0)` per batch and then compute `torch.stack(...).mean(0)`. The last partial batch (potentially a handful of rounds vs 128) receives equal weight to every full batch. (`accepted_*` is unaffected because it uses `torch.cat`.)
    - **Fix**: Append `eq.float()` and aggregate with `torch.cat(...).mean(0).tolist()`.

11. **VAE free-bits KL averages across the `nq=4` latent tokens before clamping and summing in `ae_gate.py` and `tok_vae.py`**
    - **Where**: `ae_gate.py:194`, `tok_vae.py:93`
    - **Evidence**: `mu32` has shape `(B, nq, zq)`. `(0.5 * (...)).mean((0, 1)).clamp_min(args.floor).sum()` averages over both batch (`dim 0`) and latent tokens (`dim 1`), allowing up to `nq - 1` tokens to collapse to zero KL while summing over `zq = zdim / 4` instead of `zdim` (4× smaller than per-dim KL).
    - **Fix**: Change `.mean((0, 1))` to `.mean(0)` so free-bits clamping and summing apply per dimension across all `(nq, zq)`.

12. **`stage2.py` output `tag` omits `dec_w`, `t_shift`, `init`, and `ctx_layers`, causing silent checkpoint and result overwrites**
    - **Where**: `stage2.py:307-310`, `stage2.py:345`
    - **Evidence**: Two runs differing only in `--dec_w` (`0` vs `1`), `--t_shift` (`1` vs `3`), `--ctx_layers`, or `--eval_only --init <ckptA vs ckptB>` write to the exact same `ckpt/stage2_<tag>.pt` and `results/stage2_<tag>.json`, silently clobbering earlier experiments.
    - **Fix**: Include `dec_w`, `t_shift`, `ctx_layers`, and `init` (or add an explicit `--tag` argument) in `tag`.

13. **Empty or `< 4`-token `later_N` bags near answer ends in `prune_oracle.py` cause `topk(4)` to pick arbitrary `-1e9` tokens**
    - **Where**: `prune_oracle.py:73-83`
    - **Evidence**: On rounds where `rr + K >= end` (final chunks of answers), `later_64` / `later_256` yields `toks = []`, leaving `mask[n]` all `False` and `lpm[n]` uniformly `-1e9`. `topk(4, -1)` then selects arbitrary token IDs (`0, 1, 2, 3`). Whenever `0 < len(set(toks)) < 4`, out-of-bag tokens with `-1e9` likewise leak into `top4_oracle` and `best_prefixes`.
    - **Fix**: When `mask[n].sum() == 0`, record 1 (bonus-only, 0 drafted matches) without calling `topk`; when `0 < k_avail < 4`, slice `lpt`/`idt` to `min(4, k_avail)` before evaluating `best_prefixes` and `top4_oracle`.

14. **`GistDrafter` in `e2e.py` ignores `sa["ae"]`, `sa["hist"]`, and `sa["ctx_pos"]` from stage-2 checkpoints**
    - **Where**: `e2e.py:83-112`
    - **Evidence**: `ae_name` defaults to `"gist_hf_z2048"` and is never checked against `ck["args"]["ae"]` of `s2` or `det_s2`, so loading a stage-2 model trained on another AE silently decodes with the wrong stage-1 decoder. Checkpoints trained with `--hist > 1` or `--ctx_pos > 0` (such as `run_final.sh:22`) fail with shape mismatches rather than being handled or cleanly rejected.
    - **Fix**: Default `ae_name` from `s2`'s `ck["args"]["ae"]`, assert `det_s2` uses the same `ae`, and either assert `hist == 1 and ctx_pos == 0` with a clear message or implement multi-chunk / context buffering in `GistDrafter`.

15. **Silent checkpoint fallback in `run_final.sh` and missing EOS stopping in `examples.py`**
    - **Where**: `run_final.sh:11`, `examples.py:27-69`
    - **Evidence**: `run_final.sh:11` chains `examples.py --s2 stage2_gister_flow_big ... || examples.py --s2 stage2_gister_flow ...`, silently falling back to the smaller 158M sampler if the 285M run fails. Inside `examples.py:run()`, the loop runs a fixed `rounds` iterations with no stop-token check, so short answers keep generating past `<|im_end|>` and pollute both the printed `mean kept` and `results/examples.json`.
    - **Fix**: Remove the `||` fallback in `run_final.sh` and break in `examples.py:run()` once a stop token is generated.

16. **`dflash_tree_gap.py` evaluates different rounds and (by default) different labels than `stage2.py` / `dflash_s2.py`**
    - **Where**: `dflash_tree_gap.py:63-74` (cited as the DFlash reference in `stage2.py:18`)
    - **Evidence**: `dflash_tree_gap.py` defaults to `--hf_labels=False` (scoring against saved AR text `ids` instead of 4-bit teacher-forced `hf_top1`), skips sequences shorter than `n0 + 34`, starts at `n0 + 1` (row `j = 2`), and drops the final 32 tokens (leftover `ORACLE_N // 4` from `dflash_plan.py`), whereas `stage2.py` / `dflash_s2.py` / `prune_oracle.py` score `hf_top1` on `j = 8 .. len(resp) - 8`.
    - **Fix**: Select rounds using the same `data/cache/{sp}_meta.npz` `rounds()` logic as `dflash_s2.py` / `prune_oracle.py` and default `--hf_labels=True`.

---

### LOW

17. **Dead, stale, or broken scripts and helpers that can confuse readers or cause subtle errors**
    - **`common.py:19-23` `chat_ids()`**: Dead (unused anywhere in the repo) and **omits `enable_thinking=False`**, unlike `dflash_eval.py:48` `chat()`. Reusing it silently puts Qwen3-8B into thinking mode (`<think>...</think>`), causing a major train/eval format mismatch. Remove or align it with `enable_thinking=False`.
    - **`dflash_plan.py`, `kv_oracle.py`, `kv_oracle_train.py`**: Stale "Sketch A" experiments (hardcoding `.pt` paths instead of `common.load_items()`, stepping `opt.step()` with zero grad when `starts` is empty at `dflash_plan.py:118` and `kv_oracle_train.py:94`, and silently truncating the 64-token oracle window near sequence ends). Active scripts (`dflash_feats.py`, `dflash_tree_gap.py`) still import `B`, `CTX_LAYERS`, `ORACLE_N`, `ctx_feats`, and `logits_with_token` from them. Move those shared helpers into `dflash_model.py` or `dflash_eval.py` and archive/remove the stale Sketch A scripts.
    - **`run_scale2.sh`, `run_final.sh`**: One-off Windows-venv (`.venv/Scripts/python`) queue scripts without `set -euo pipefail`, invoking missing `wandb_follow.py` and polling untracked `logs/cache_pb2.log`.
    - **`dflash_s2.py:32, 100`**: `--smoke` reads full `data/dflash_feats` instead of `data/dflash_feats_smoke`, and `choice(tr["rows"], 50000, replace=False)` crashes if `len(tr["rows"]) < 50000` (use `min(50000, len(tr["rows"]))`).
    - **`tok_vae.py:32-36, 105-115`**: `load_embed()` hardcodes `~/.cache/huggingface/hub/...` (ignoring `HF_HOME`/`HF_HUB_CACHE`), and dev evaluation runs the entire dev split in a single unbatched forward (OOM risk on longer caches like `cache_long`).
    - **`stage2.py:12, 296`**: Docstring and log key call the decoded loss `"kl"` (`KL(target || decoded)`), whereas the implementation computes hard top-1 cross-entropy `F.nll_loss(lq, lab.flatten())`. Rename the log key/docstring to `ce` for accuracy.
