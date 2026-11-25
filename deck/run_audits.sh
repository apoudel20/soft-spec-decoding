#!/usr/bin/env bash
# Option 2: three focused Neon audits, run one after another (one ledger writer at a time).
cd "$(dirname "$0")/.."
PY=tools/.venv/Scripts/python.exe
N="$PY tools/neon_auditor_mcp.py --cli --thinking-level high"
while powershell.exe -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'neon_auditor_mcp' }) { exit 0 } else { exit 1 }"; do sleep 20; done
B1=(); for f in *.py *.sh tools/neon_auditor_mcp.py; do B1+=(--file "$f"); done
$N --prompt-file deck/neon_prompt_b1_code.md "${B1[@]}" > deck/neon_audit_b1_code.md 2> deck/neon_audit_b1_code.err
echo "B1 done $(date +%H:%M)"
B2=(--file RESULTS.md --file AGENT_NOTES.md --file TWO_GPU_PLAN.md --file deck/outline.md)
while IFS= read -r f; do B2+=(--file "$f"); done < <(find results -name "*.json" -size -100k | sort)
$N --prompt-file deck/neon_prompt_b2_claims.md "${B2[@]}" > deck/neon_audit_b2_claims.md 2> deck/neon_audit_b2_claims.err
echo "B2 done $(date +%H:%M)"
B3=(--file RESULTS.md --file AGENT_NOTES.md --file deck/outline.md)
while IFS= read -r f; do B3+=(--file "$f"); done < <(find reports research_notes -name "*.md" | grep -v "Mixture of specialized drafters" | grep -v "Experiment handoff" | sort)
$N --prompt-file deck/neon_prompt_b3_novelty.md "${B3[@]}" > deck/neon_audit_b3_novelty.md 2> deck/neon_audit_b3_novelty.err
echo "B3 done $(date +%H:%M)"
$PY tools/neon_auditor_mcp.py --cli --budget
