"""neon-auditor: a local, read-only MCP server that sends repo files plus a prompt to gemini-4-neon on Vertex AI.

Auth: Application Default Credentials (gcloud auth application-default login), location "global", project from
GOOGLE_CLOUD_PROJECT. Every call is checked against a local spend ledger (tools/.neon_spend.json) and refused if the
worst-case cost of the call would push estimated spend past NEON_MAX_USD (default $50).

Model rules for gemini-4-neon: thinking_level only (low / medium / high / xhigh), no temperature / top_p / top_k,
max_output_tokens 65536, HTTP timeout 15 minutes. The Interactions API is tried first; if it is not available on
Vertex in the installed google-genai, calls fall back to models.generate_content with ThinkingConfig.

stdout belongs to the MCP stdio transport, so all logging goes to stderr.

Run as a server:  python tools/neon_auditor_mcp.py
Smoke test:       python tools/neon_auditor_mcp.py --smoke
"""
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s neon-auditor %(levelname)s %(message)s")
log = logging.getLogger("neon-auditor")

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent                                    # files are read relative to the repo root, and only inside it
LEDGER = TOOLS / ".neon_spend.json"
MODEL = "gemini-4-neon"
def _adc_project():                                    # default: the quota project saved in the local ADC file
    p = Path(os.environ.get("APPDATA", Path.home() / ".config")) / "gcloud" / "application_default_credentials.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("quota_project_id")
    except (OSError, ValueError):
        return None


PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT") or _adc_project()
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
MAX_USD = float(os.environ.get("NEON_MAX_USD", "50"))
LEVELS = ("low", "medium", "high", "xhigh")
MAX_OUT = 65536
PRICE_IN, PRICE_CACHED, PRICE_OUT = 3.00e-6, 0.15e-6, 15.00e-6     # USD per token
MAX_FILE = 500_000
BLOCKED_EXT = {".pt", ".npy", ".npz", ".safetensors", ".bin", ".pkl", ".pth", ".ckpt", ".parquet", ".zip", ".gz",
               ".png", ".jpg", ".jpeg", ".pdf", ".exe", ".dll"}
_lock = threading.Lock()
_client = None
_api = os.environ.get("NEON_API", "auto")              # "auto", "interactions" or "generate_content"


def _ledger():
    if LEDGER.exists():
        return json.loads(LEDGER.read_text(encoding="utf-8"))
    return {"calls": 0, "input_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "usd": 0.0, "history": []}


def _save(led):
    tmp = LEDGER.with_suffix(".tmp")
    tmp.write_text(json.dumps(led, indent=1), encoding="utf-8")
    os.replace(tmp, LEDGER)


def _cost(inp, cached, out):
    return (inp - cached) * PRICE_IN + cached * PRICE_CACHED + out * PRICE_OUT


def _client_():
    global _client
    if _client is None:
        from google import genai
        from google.genai import types
        _client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION,
                               http_options=types.HttpOptions(timeout=900_000))
    return _client


def _read_files(paths):
    parts, notes = [], []
    for rel in paths or []:
        p = (ROOT / rel).resolve()
        if ROOT not in p.parents and p != ROOT:
            notes.append(f"refused {rel}: outside the repo")
            continue
        if not p.is_file():
            notes.append(f"refused {rel}: not a file")
            continue
        if p.suffix.lower() in BLOCKED_EXT:
            notes.append(f"refused {rel}: binary or weight file type {p.suffix}")
            continue
        raw = p.read_bytes()
        if b"\x00" in raw[:8192]:
            notes.append(f"refused {rel}: looks binary")
            continue
        text = raw.decode("utf-8", errors="replace")
        if len(raw) > MAX_FILE:
            text = text[:MAX_FILE]
            notes.append(f"WARNING {rel}: {len(raw):,} bytes, truncated to the first {MAX_FILE:,}")
        parts.append(f"===== FILE: {rel} =====\n{text}\n===== END FILE: {rel} =====")
    return "\n\n".join(parts), notes


def _call(text, level, prev_id):
    """One model call. Returns (reply, input_tokens, cached_tokens, output_tokens, interaction_id, api_used)."""
    global _api
    client = _client_()
    if _api in ("auto", "interactions"):
        try:
            kw = {"model": MODEL, "input": text,
                  "generation_config": {"thinking_level": level, "max_output_tokens": MAX_OUT}}
            if prev_id:
                kw["previous_interaction_id"] = prev_id
            r = client.interactions.create(**kw)
            u = r.usage
            inp = (getattr(u, "total_input_tokens", 0) or 0)
            cached = (getattr(u, "total_cached_tokens", 0) or 0)
            out = (getattr(u, "total_output_tokens", 0) or 0) + (getattr(u, "total_thought_tokens", 0) or 0)
            reply = getattr(r, "output_text", None) or ""
            _api = "interactions"
            return reply, inp, cached, out, getattr(r, "id", None), "interactions"
        except Exception as e:                          # not available on Vertex here, or rejected: fall back once
            if _api == "interactions":
                raise
            log.warning("interactions API failed (%s: %s); using models.generate_content", type(e).__name__, e)
            _api = "generate_content"
    from google.genai import types
    if prev_id:
        log.warning("previous_interaction_id ignored: generate_content has no server-side conversation state")
    cfg = types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_level=level),
                                      max_output_tokens=MAX_OUT)
    r = client.models.generate_content(model=MODEL, contents=text, config=cfg)
    u = r.usage_metadata
    inp = u.prompt_token_count or 0
    cached = u.cached_content_token_count or 0
    out = (u.candidates_token_count or 0) + (u.thoughts_token_count or 0)
    return (r.text or ""), inp, cached, out, None, "generate_content"


def audit(prompt, file_paths=None, thinking_level="high", previous_interaction_id=None):
    if thinking_level not in LEVELS:
        return f"refused: thinking_level must be one of {LEVELS}"
    files, notes = _read_files(file_paths)
    text = prompt if not files else f"{prompt}\n\n{files}"
    est_in = len(text) / 3.0                           # conservative token estimate
    worst = est_in * PRICE_IN + MAX_OUT * PRICE_OUT
    with _lock:
        led = _ledger()
        if led["usd"] + worst > MAX_USD:
            return (f"refused: spend cap. Spent ${led['usd']:.4f} of ${MAX_USD:.2f}; this call could cost up to "
                    f"${worst:.4f}. Raise NEON_MAX_USD to continue.")
    t0 = time.time()
    reply, inp, cached, out, iid, api = _call(text, thinking_level, previous_interaction_id)
    c = _cost(inp, cached, out)
    with _lock:
        led = _ledger()
        led["calls"] += 1
        led["input_tokens"] += inp
        led["cached_tokens"] += cached
        led["output_tokens"] += out
        led["usd"] += c
        led["history"].append({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "api": api, "level": thinking_level,
                               "files": list(file_paths or []), "in": inp, "cached": cached, "out": out,
                               "usd": round(c, 6), "seconds": round(time.time() - t0, 1)})
        _save(led)
    head = [f"[gemini-4-neon via {api}, thinking_level={thinking_level}, {time.time() - t0:.0f}s]",
            f"[tokens: input {inp:,} (cached {cached:,}), output+thinking {out:,}; call ${c:.4f}; "
            f"total ${led['usd']:.4f} of ${MAX_USD:.2f}]"]
    if iid:
        head.append(f"[interaction_id: {iid}]")
    head += [f"[{n}]" for n in notes]
    return "\n".join(head) + "\n\n" + reply


def budget():
    led = _ledger()
    return (f"calls {led['calls']}, input tokens {led['input_tokens']:,} (cached {led['cached_tokens']:,}), "
            f"output+thinking tokens {led['output_tokens']:,}, estimated spend ${led['usd']:.4f}, "
            f"remaining ${max(0.0, MAX_USD - led['usd']):.4f} of ${MAX_USD:.2f}")


def main():
    if "--smoke" in sys.argv:
        out = audit("Reply with OK only.", None, "low")
        sys.stderr.write(out + "\n\n" + budget() + "\n")
        return
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("neon-auditor")

    @mcp.tool()
    def audit_with_neon(prompt: str, file_paths: list[str] | None = None, thinking_level: str = "high",
                        previous_interaction_id: str | None = None) -> str:
        """Send a prompt plus repo-relative text files to gemini-4-neon (Vertex AI) as an adversarial reviewer.
        Read-only: files are read from the repo only; binary and weight files are refused, files over 500 KB are
        truncated with a warning. thinking_level: low, medium, high (default) or xhigh. Returns the review plus token
        counts, this call's cost, running spend and the interaction id when available. Refuses calls that could
        exceed the NEON_MAX_USD spend cap."""
        try:
            return audit(prompt, file_paths, thinking_level, previous_interaction_id)
        except Exception as e:
            log.exception("audit failed")
            return f"error: {type(e).__name__}: {e}"

    @mcp.tool()
    def neon_budget_status() -> str:
        """Total calls, input / cached / output tokens, estimated USD spent and remaining budget under NEON_MAX_USD."""
        return budget()

    mcp.run()                                           # stdio


if __name__ == "__main__":
    main()
