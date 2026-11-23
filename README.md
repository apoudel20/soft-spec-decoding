# Soft speculative decoding experiments

Can a target LLM and its drafter cooperate on distributions ("soft tokens") instead of hard
guesses, while the output stays exactly the target's? Code for the experiments in
[PLAN.md](PLAN.md). Results will be in RESULTS.md.

**Status: work in progress.** The harness, exactness tests and training code are done, and
training runs are in progress. No conclusions yet.

Related negative result: [bijmuj/EmbeddingSuperPosition](https://github.com/bijmuj/EmbeddingSuperPosition)
found that soft-prior (embedding superposition) inputs to DFlash and EAGLE-3 drafters did not raise
acceptance length.

Target: Qwen2.5-1.5B-Instruct. Drafter: Qwen2.5-0.5B-Instruct (same tokenizer). One RTX 4070 Ti (12 GB), Windows 11.

## Files

| File | What it does |
|---|---|
| `model.py` | Self-written Qwen2 forward pass loaded from the HF safetensors. Position-gated LoRA (`gate=g` adds the LoRA delta only to chunk rows ≥ g), soft inputs `Eᵀq`, a static KV cache, and CUDA-graph replay of decode chunks. Every arm runs through it, so they share kernels. |
| `decode.py` | Decoding loops at batch size 1: AR, classic SD, block-drafter SD (Arm D), and the soft-tail family: Jacobi, training-free soft Jacobi, Arm C, C-hard and C+D. All share one `verify` (greedy match or standard rejection sampling). |
| `test_exactness.py` | Exactness harness: parity with HF, same-layout bitwise check of the gated LoRA (hard-row logits and kept KV), cross-layout drift (informational), and a chi-square test of the rejection sampler. |
| `gen_data.py` | Self-distilled training data: the target's own responses (half greedy, half T=1). |
| `train.py` | Arm C (`--arm C --inp soft/hard`) and Arm D (`--arm D --label d1..d4 --inp soft/hard`) training with packed blocks over a no-grad base context KV. |
| `arm0.py` | Arm 0 diagnostic: teacher-forced vs free-running per-offset accuracy. |
| `eval.py` | Runs method specs on prompt sets and writes `results/<name>.json` (τ, per-position acceptance, wall-clock speedup, exactness). |
| `common.py` | Train, dev and eval prompt sets (disjoint). |

## Reproduce

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -m pip install transformers datasets peft accelerate numpy
```

Then, in order (see RESULTS.md for the exact commands and settings behind each table):

```powershell
.\.venv\Scripts\python test_exactness.py --hf
.\.venv\Scripts\python gen_data.py --n_chat 4000 --n_math 2500
.\.venv\Scripts\python arm0.py --sets gsm8k:40 mtbench:40 humaneval:40 --drafters base --out arm0_base
.\.venv\Scripts\python train.py --arm D --label d3 --inp soft --steps 1500 --name d3_soft
.\.venv\Scripts\python train.py --arm C --inp soft --steps 1500 --name c_soft
.\.venv\Scripts\python eval.py --sets mtbench:40 gsm8k:40 humaneval:40 --methods "sd:g=5" "jacobi:H=8" "C:ckpt=c_soft,H=4,S=8" --out example
```

Models and datasets download to the default Hugging Face cache (`~/.cache/huggingface`). Generated data, checkpoints and logs go to `data/`, `ckpt/` and `logs/`, all git-ignored. Only `results/*.json` and the code are committed.

## Notes

- Run the scripts with the venv's python from the repo root (they import each other).
- On Windows without Developer Mode, the HF cache copies files instead of symlinking them. That only costs disk space.
- At exit, Python 3.12.0 prints a harmless `ResourceTracker ... _recursion_count` traceback from the `multiprocess` package.
