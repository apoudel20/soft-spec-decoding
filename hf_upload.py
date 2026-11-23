"""Publish the generated answers and data stats as a public Hugging Face dataset.

Uploads tokens/*.jsonl (from the GitHub repo checkout) plus results/data_stats.json and results/state_stats.json,
with a dataset card. Re-running updates the same repo.
Run: python hf_upload.py [--repo ablastvacant/soft-spec-decoding-qwen3-8b-answers]
"""
import argparse
import glob
import json
import os

from huggingface_hub import HfApi

from common import RES

REPO_DIR = r"C:\Users\aayus\Projects\soft-spec-decoding"

CARD = """---
license: other
license_name: mixed-sources
language:
- en
tags:
- speculative-decoding
- qwen3
- distillation
pretty_name: Qwen3-8B greedy answers for drafter training
size_categories:
- 1K<n<10K
configs:
{configs}
---

# Qwen3-8B greedy answers for drafter training

Qwen3-8B's own greedy answers (temperature 0, thinking off, chat template) to chat, math and code prompts, stored
as token IDs from the `Qwen/Qwen3-8B` tokenizer. Generated with the 4-bit Ollama build `qwen3:8b` for speed and
re-tokenised with the HF tokenizer. Used to train and test drafters for lossless speculative decoding in
https://github.com/apoudel20/soft-spec-decoding.

Each line: `{{"prompt": [ids], "resp": [ids]}}`. `prompt` includes the chat template. Decode with
`AutoTokenizer.from_pretrained("Qwen/Qwen3-8B").decode(ids)`.

| File | Answers | What |
|---|---|---|
{rows}

Answers were generated with a 320-token cap. Files ending in `_long` (when present) continue the capped answers
greedily to 1,024 tokens. About 87% of chat answers, 25% of math and 13% of code hit the 320 cap.

## Sources and licenses

Prompts: UltraChat 200k (MIT), GSM8K (MIT), MBPP (CC BY 4.0). Train and dev prompts come from disjoint splits
(UltraChat train_sft vs test_sft, GSM8K train minus its last 500 vs those 500, MBPP train+test vs validation).
Answers are Qwen3-8B outputs (Apache 2.0). Use follows the most restrictive source, CC BY 4.0 with attribution to
MBPP's authors.

## Caveats

The 4-bit HF build of Qwen3-8B picks a different next token than this Ollama build at about 4% (math), 6% (code)
and 13% (chat) of answer positions, nearly all near-ties. Score drafts against the target's own top-1, not this
text, when exactness matters. `data_stats.json` and `state_stats.json` hold length breakdowns and per-position
margin and flip-radius statistics of the 4-bit HF target.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="ablastvacant/soft-spec-decoding-qwen3-8b-answers")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(REPO_DIR, "tokens", "*.jsonl")))
    rows, configs = [], []
    for f in files:
        n = os.path.basename(f)[:-6]
        cnt = sum(1 for _ in open(f, encoding="utf-8"))
        rows.append(f"| `{n}.jsonl` | {cnt} | {'dev' if 'dev' in n else 'train'} |")
        configs.append(f"- config_name: {n}\n  data_files: {n}.jsonl")
    card = CARD.replace("{rows}", "\n".join(rows)).replace("{configs}", "\n".join(configs)).replace("{{", "{").replace("}}", "}")
    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True, private=False)
    ops_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_stage")
    os.makedirs(ops_dir, exist_ok=True)
    open(os.path.join(ops_dir, "README.md"), "w", encoding="utf-8").write(card)
    api.upload_file(path_or_fileobj=os.path.join(ops_dir, "README.md"), path_in_repo="README.md", repo_id=args.repo, repo_type="dataset")
    for f in files + [os.path.join(RES, "data_stats.json"), os.path.join(RES, "state_stats.json")]:
        if os.path.exists(f):
            api.upload_file(path_or_fileobj=f, path_in_repo=os.path.basename(f), repo_id=args.repo, repo_type="dataset")
            print("uploaded", os.path.basename(f), flush=True)
    listed = api.list_repo_files(args.repo, repo_type="dataset")
    print("REPO FILES", json.dumps(listed))
    print(f"https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
