"""Export the generated token data to the public repo as JSON lines (tokens/<name>.jsonl), one answer per line.

Run:    python export_tokens.py
"""
import json
import os

import torch

from common import DATA

REPO = r"C:\Users\aayus\Projects\soft-spec-decoding"
NAMES = ["q3_train", "q3_train_more", "q3_dev_math_40", "q3_dev_code_40", "q3_dev_chat_40"]

README = """# Generated token data

Qwen3-8B's own greedy answers (temperature 0, thinking off, chat template, 4-bit Ollama build `qwen3:8b`),
re-tokenised with the HF `Qwen/Qwen3-8B` tokenizer. One answer per line: `{"prompt": [ids], "resp": [ids]}`.
`prompt` includes the chat template. Answers were generated with a 320-token cap.

| File | Answers | What |
|---|---|---|
{rows}

Prompt sources: UltraChat 200k (MIT), GSM8K (MIT), MBPP (CC BY 4.0). Train and dev prompts come from disjoint
splits (see `common.py`). Answer text is model output from Qwen3-8B (Apache 2.0).

Scripts read these through `common.load_items(name)`, which uses `data/<name>.pt` if present and these files
otherwise. To build the state cache from them: `python cache_states.py`.
"""
WHAT = {"q3_train": "training, first 600 (1/2 chat, 1/4 math, 1/4 code)",
        "q3_train_more": "training, 1,400 more, same mix, disjoint prompts",
        "q3_dev_math_40": "dev, GSM8K held-out", "q3_dev_code_40": "dev, MBPP validation",
        "q3_dev_chat_40": "dev, UltraChat test"}


def main():
    out = os.path.join(REPO, "tokens")
    os.makedirs(out, exist_ok=True)
    rows = []
    for n in NAMES:
        p = os.path.join(DATA, n + ".pt")
        if not os.path.exists(p):
            print("MISSING", n)
            continue
        items = torch.load(p)
        with open(os.path.join(out, n + ".jsonl"), "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps({"prompt": list(it["prompt"]), "resp": list(it["resp"])}) + "\n")
        rows.append(f"| `{n}.jsonl` | {len(items)} | {WHAT[n]} |")
        print(n, len(items))
    open(os.path.join(out, "README.md"), "w", encoding="utf-8").write(README.replace("{rows}", "\n".join(rows)))


if __name__ == "__main__":
    main()
