"""Print markdown tables from results/eval_*.json.

Usage: python make_tables.py results/eval_greedy_base.json [more.json ...]
Each cell is tau / wall-clock speedup vs AR, plus greedy exact matches when recorded.
"""
import json
import sys

SET_ORDER = ["gsm8k", "mtbench", "humaneval"]


def cell(m):
    s = f"{m['tau']:.2f} / {m['speedup']:.2f}x"
    if "exact_match" in m:
        em = m["exact_match"]
        if isinstance(em, (list, tuple)):
            s += f" ({em[0]}/{em[1]})"
        else:
            s += f" ({em})"
    return s


def table(path):
    d = json.load(open(path))
    sets = [s for s in SET_ORDER if s in d["sets"]]
    methods = []
    for s in sets:
        for k in d["sets"][s]["methods"]:
            if k not in methods:
                methods.append(k)
    temp = d["args"].get("temp", 0)
    print(f"\n#### {path} (T={temp}, n per set: " + ", ".join(f"{s} {d['sets'][s]['n']}" for s in sets) + ")\n")
    print("| Method | " + " | ".join(sets) + " |")
    print("|---|" + "---|" * len(sets))
    for k in methods:
        row = []
        for s in sets:
            m = d["sets"][s]["methods"].get(k)
            row.append(cell(m) if m else "")
        print(f"| `{k}` | " + " | ".join(row) + " |")
    ar = [d["sets"][s]["methods"].get("ar", {}).get("tok_per_s") for s in sets]
    print("\nAR tok/s: " + ", ".join(f"{s} {v:.1f}" for s, v in zip(sets, ar) if v))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        table(p)
