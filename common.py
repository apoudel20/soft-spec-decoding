"""Prompt sets. Train, dev and eval prompts come from disjoint sources or splits."""
import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CKPT = os.path.join(HERE, "ckpt")
RES = os.path.join(HERE, "results")
for d in (DATA, CKPT, RES):
    os.makedirs(d, exist_ok=True)

GSM_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
HE_TMPL = "Complete the following Python function. Reply with the full function in a Python code block.\n\n```python\n{}```"
MBPP_TMPL = "{}\nWrite a Python function for this. Reply with the code in a Python code block."
GSM_DEV_HOLDOUT = 500  # last 500 GSM8K train questions are dev-only, never trained on


def chat_ids(tok, text):
    out = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True)
    if isinstance(out, dict) or hasattr(out, "keys"):
        out = out["input_ids"]
    return list(out)


def _stride(n_total, n):
    if n is None or n >= n_total:
        return list(range(n_total))
    return [round(i * n_total / n) for i in range(n)]


def eval_prompts(name, n=None):
    from datasets import load_dataset
    if name == "mtbench":
        d = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        rows = sorted(d, key=lambda r: r["prompt_id"])
        return [rows[i]["prompt"][0] for i in _stride(len(rows), n)]  # first turn, stride keeps categories balanced
    if name == "gsm8k":
        d = load_dataset("openai/gsm8k", "main", split="test")
        return [d[i]["question"] + GSM_SUFFIX for i in range(n or len(d))]
    if name == "humaneval":
        d = load_dataset("openai/openai_humaneval", split="test")
        return [HE_TMPL.format(d[i]["prompt"]) for i in range(n or len(d))]
    raise ValueError(name)


def dev_prompts(name, n=40):
    from datasets import load_dataset
    if name == "chat":
        d = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft", streaming=True)
        out = []
        for r in d:
            if len(r["prompt"]) < 600:
                out.append(r["prompt"])
            if len(out) >= n:
                break
        return out
    if name == "math":
        d = load_dataset("openai/gsm8k", "main", split="train")
        return [d[i]["question"] + GSM_SUFFIX for i in range(len(d) - GSM_DEV_HOLDOUT, len(d) - GSM_DEV_HOLDOUT + n)]
    if name == "code":
        d = load_dataset("google-research-datasets/mbpp", "full", split="validation")
        return [MBPP_TMPL.format(d[i]["text"]) for i in range(min(n, len(d)))]
    raise ValueError(name)


def train_prompts(n_chat=4000):
    from datasets import load_dataset
    out = []
    d = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    for r in d:
        if len(r["prompt"]) < 600:
            out.append(("chat", r["prompt"]))
        if len(out) >= n_chat:
            break
    g = load_dataset("openai/gsm8k", "main", split="train")
    out += [("math", g[i]["question"] + GSM_SUFFIX) for i in range(len(g) - GSM_DEV_HOLDOUT)]
    for split in ("train", "test"):  # MBPP test is not an eval set here
        mb = load_dataset("google-research-datasets/mbpp", "full", split=split)
        out += [("code", MBPP_TMPL.format(r["text"])) for r in mb]
    return out


TOKENS = os.path.join(HERE, "tokens")


def load_items(name):
    """Token data {"prompt": ids, "resp": ids} from data/<name>.pt, else from the repo's tokens/<name>.jsonl."""
    import json
    p = os.path.join(DATA, name + ".pt")
    if os.path.exists(p):
        import torch
        return torch.load(p)
    j = os.path.join(TOKENS, name + ".jsonl")
    if os.path.exists(j):
        with open(j, encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    return None
