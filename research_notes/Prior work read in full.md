# Prior work read in full (23 Nov 2026)

Read from the arXiv HTML. Where a fetch was truncated, the parts not read verbatim are marked. Numbers are
from the papers' own tables.

## The question each paper answers

| Paper | Plan signal from | Trained how | Horizon | Used for drafting? | Target changed? |
|---|---|---|---|---|---|
| ConFu (2603.08899) | Target, one last-layer vector from an extra "contemplate" token | Only the drafter's KL loss (no auxiliary future loss) | Undefined (next L draft tokens) | Yes, EAGLE-3 head, autoregressive | Frozen, plus learned soft prompt |
| SD² (2511.09844) | Target, steering vector from 3 layers | Only KL, with a random offset of 1 to 8 tokens | Implicit, k = 8 | Yes, separate small LM, autoregressive chain | Frozen |
| FSP (2510.14751) | Model's own hidden state | Explicit future-summary loss: bag-of-words, or reverse-LM hidden state up to end of sequence | Up to whole sequence | No, the head is discarded at inference | Pretrained with it |
| LLM-JEPA (2509.14252) | Last-token embedding of one view | Cosine loss to another view's embedding (needs paired data) | None, not about the future | No | Fine-tuned or pretrained with it |
| NextLat (2511.05963) | Model's own final-layer hidden state | MLP predicts next hidden state from (h_t, next token), stop-grad target, plus KL | One step, rolled forward recursively | Yes, self-speculative, rolls the latent forward | Pretrained from scratch with it |

## Notes per paper

**ConFu** (Qualcomm and UCLA, Mar 2026). Frozen target. Learnable soft prompts sit in the KV cache, and
a "contemplate" token (an MoE-mixed embedding) is appended after each draft node. Its last-layer hidden state
is the future vector, fed as one extra input token to an EAGLE-3 head. Verification processes 2T tokens. No
future-prediction loss, only EAGLE-3's KL. Gains over EAGLE-3 on SpecBench are +7% to +13% accepted length
and +4% to +15% speedup on Llama 3.2 3B and 3.1 8B. The Qwen3-4B results and the ablations were not read
verbatim.

**SD²** (ETH Zurich, AAAI 2026). Frozen target. The verifier's layer 3, L/2 and L-2 states are mixed
linearly into a steering vector that is added as a bias inside every MLP of an independent small drafter
(160M to 1B). Trained with KL and a random offset of 1 to 8, so the vector must encode the next block.
Accepted length improves by 2% to 57% over a pretrained drafter (largest on Vicuna with a 160M drafter, smallest
on Llama 3.1 8B with a 1B drafter), and about 16% to 20% average speedup. No numeric EAGLE comparison.

**FSP** (Meta FAIR and Mila, ICLR 2026). A pretraining objective. One auxiliary head predicts a summary of
the future. The handcrafted version is a multi-hot bag of future tokens (horizons 4 to 100). The learned version
is the hidden state of a separately trained reverse LM, covering the rest of the sequence. 3B and 8B models.
Gains are modest at 8B (ARC-E 0.766 vs 0.718, MBPP 0.683 vs 0.657, GSM8K slightly worse) and DS-MTP is
competitive at 3B. About 2x training compute. Not used for drafting. The 8B table was not read verbatim.

**LLM-JEPA** (Huang, LeCun, Balestriero, arXiv 2025). Adds a cosine loss between the last-token
embedding of one view (for example a description) and a predicted embedding of a paired view (for example
the code), with tied weights and optional [PRED] tokens. No EMA or stop-grad. Needs paired data. About 2x training
compute. Gains on small fine-tuning tasks with 1B models (NL-RX 57 to 71, GSM8K 44.3 to 45.0 on Qwen3 1.7B).
Nothing about the future, planning or decoding.

**NextLat** (Microsoft Research, NeurIPS 2026). A 3-layer MLP predicts the next final-layer hidden state from the
current one plus the next token's embedding (residual). Smooth L1 to a stop-gradient target, plus a KL term on the
decoded distribution. Pretrained into 1.3B models on 100B FineWeb-Edu tokens. At inference the MLP rolls the
latent forward to draft tokens with the model's own LM head, verified with Leviathan speculative sampling.
Up to 3.3x speedup and 4.9 accepted tokens, against 1.7x to 1.9x for MTP and JTP baselines. No EAGLE
comparison. Post-hoc fine-tuning of a pretrained model is listed as future work. Long-horizon planning is not
tested. The paper never uses the word "lossless".

## Connection that matters

NextLat's dynamics model (predict the next hidden state from the current one plus the next token, then decode
it with the LM head) is the same mechanism EAGLE-1 (2401.15077) uses post hoc on a frozen target, with
second-to-top-layer features. So "a latent world model of the target, used to draft" exists in two forms.
EAGLE trains it post hoc and NextLat trains it into pretraining.

## What is left of the latent-planner idea

1. An explicit long-horizon plan objective (FSP-style summary or JEPA-style latent), trained post hoc on a
   frozen target, used to condition a drafter. FSP has the objective but never drafts. ConFu and SD² draft
   but use only the KL loss with a short or undefined horizon.
2. Conditioning a one-pass block drafter (DFlash) on such a plan. ConFu and SD² use autoregressive drafters.

Both are combinations of existing parts. ConFu's +7% to +13% from a KL-only future vector is the realistic
prior for what a target-side future signal adds. A plan objective would have to beat that clearly to matter.
