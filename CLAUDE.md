# Project instructions

## Adversarial review with neon-auditor

After a major code change, a benchmark run, or an edit to a write-up (RESULTS.md, AGENT_NOTES.md, the website or a
deck), call the `neon-auditor` MCP tool `audit_with_neon` with `thinking_level="high"` as an adversarial reviewer.
- Pass only the relevant scripts, `results/*.json` files and diffs as `file_paths` (repo-relative). Never weights,
  caches or large logs.
- Ask it to check claims against the code and JSON, look for bugs, leakage, confounds and overclaiming.
- Address every critical or high finding (fix it, or disclose it in the write-up) before calling the task done.
- Check `neon_budget_status` first. The spend cap is NEON_MAX_USD (default $50). Calls are slow (minutes).
