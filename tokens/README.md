# Generated token data

Qwen3-8B's own greedy answers (temperature 0, thinking off, chat template, 4-bit Ollama build `qwen3:8b`),
re-tokenised with the HF `Qwen/Qwen3-8B` tokenizer. One answer per line: `{"prompt": [ids], "resp": [ids]}`.
`prompt` includes the chat template. Answers were generated with a 320-token cap.

| File | Answers | What |
|---|---|---|
| `q3_train.jsonl` | 600 | training, first 600 (1/2 chat, 1/4 math, 1/4 code) |
| `q3_train_more.jsonl` | 1400 | training, 1,400 more, same mix, disjoint prompts |
| `q3_dev_math_40.jsonl` | 40 | dev, GSM8K held-out |
| `q3_dev_code_40.jsonl` | 40 | dev, MBPP validation |
| `q3_dev_chat_40.jsonl` | 40 | dev, UltraChat test |
| `q3_train_long.jsonl` | 600 | training, first 600 (1/2 chat, 1/4 math, 1/4 code), capped answers continued to 1,024 tokens |
| `q3_train_more_long.jsonl` | 1400 | training, 1,400 more, same mix, disjoint prompts, capped answers continued to 1,024 tokens |
| `q3_dev_math_40_long.jsonl` | 40 | dev, GSM8K held-out, capped answers continued to 1,024 tokens |
| `q3_dev_code_40_long.jsonl` | 40 | dev, MBPP validation, capped answers continued to 1,024 tokens |
| `q3_dev_chat_40_long.jsonl` | 40 | dev, UltraChat test, capped answers continued to 1,024 tokens |

Prompt sources: UltraChat 200k (MIT), GSM8K (MIT), MBPP (CC BY 4.0). Train and dev prompts come from disjoint
splits (see `common.py`). Answer text is model output from Qwen3-8B (Apache 2.0).

Scripts read these through `common.load_items(name)`, which uses `data/<name>.pt` if present and these files
otherwise. To build the state cache from them: `python cache_states.py`.
