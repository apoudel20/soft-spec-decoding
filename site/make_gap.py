"""Section 04: chart of the selection-gap results (drafter selection vs best possible, by node budget)."""
import json
import math

d = json.load(open(r"C:\Users\aayus\Projects\movement-softspec\research\soft-spec-decoding\results\selection_gap.json"))
B = [8, 16, 32, 64, 128]
COL = {"chat": "#6a4fc0", "code": "#c25a17", "math": "#008a72"}
NAME = {"chat": "Chat", "code": "Code", "math": "Math"}
W, H = 760, 330
x0, x1, y0, y1 = 70, 600, 280, 30          # plot box (y0 bottom)


def X(b):
    return x0 + (math.log2(b) - 3) / 4 * (x1 - x0)


def Y(v):
    return y0 - v / 8 * (y0 - y1)


o = []
A = o.append
A(f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Accepted tokens out of 8 against the number of nodes the target checks. '
  'For each of math, code and chat, a solid line shows the drafter choosing its own most confident nodes and a dashed line shows the best any '
  'choice could do from the same forest. The gap is largest for chat and for small budgets." style="width:100%;height:auto;color:var(--fg);font-family:var(--sans)">')
for v in range(0, 9, 2):
    A(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x1}" y2="{Y(v):.1f}" style="stroke:var(--line);stroke-width:1"/>')
    A(f'<text x="{x0 - 10}" y="{Y(v) + 4:.1f}" text-anchor="end" font-size="11.5" style="fill:var(--muted)">{v}</text>')
for b in B:
    A(f'<text x="{X(b):.1f}" y="{y0 + 20}" text-anchor="middle" font-size="11.5" style="fill:var(--muted)">{b}</text>')
A(f'<text x="{(x0 + x1) / 2}" y="{y0 + 42}" text-anchor="middle" font-size="12" style="fill:var(--muted)">nodes the target checks per pass</text>')
A(f'<text x="{x0 - 46}" y="{(y0 + y1) / 2}" text-anchor="middle" font-size="12" transform="rotate(-90 {x0 - 46} {(y0 + y1) / 2})" style="fill:var(--muted)">accepted tokens (of 8)</text>')
A(f'<rect x="{X(16):.1f}" y="{y1}" width="{X(64) - X(16):.1f}" height="{y0 - y1}" style="fill:var(--accent-bg);opacity:.55"/>')
A(f'<text x="{(X(16) + X(64)) / 2:.1f}" y="{y0 - 10}" text-anchor="middle" font-size="11" style="fill:var(--accent)">typical tree budgets</text>')
lab_y = {}
prev = None
for s in sorted(['math', 'code', 'chat'], key=lambda k: -d['sets'][k]['oracle_mean']):
    y = Y(d['sets'][s]['oracle_mean'])
    if prev is not None and y - prev < 32:
        y = prev + 32
    lab_y[s] = y
    prev = y
for s in ["math", "code", "chat"]:
    r = d["sets"][s]
    c = COL[s]
    orc = r["oracle_mean"]
    A(f'<line x1="{x0}" y1="{Y(orc):.1f}" x2="{x1}" y2="{Y(orc):.1f}" style="stroke:{c};stroke-width:1.6;stroke-dasharray:5 4"/>')
    pts = " ".join(f"{X(b):.1f},{Y(r[f'drafter_B{b}_mean']):.1f}" for b in B)
    A(f'<polyline points="{pts}" style="fill:none;stroke:{c};stroke-width:2.4;stroke-linejoin:round"/>')
    for b in B:
        v = r[f"drafter_B{b}_mean"]
        g = r[f"gap_B{b}"]
        lo, hi = r[f"gap_B{b}_ci95"]
        A(f'<circle cx="{X(b):.1f}" cy="{Y(v):.1f}" r="5" style="fill:{c};stroke:var(--panel);stroke-width:2">'
          f'<title>{NAME[s]}, {b} nodes: drafter choice {v:.2f}, best possible {orc:.2f} (+{g * 100:.0f}%, 95% CI {lo * 100:.0f} to {hi * 100:.0f}%)</title></circle>')
    A(f'<text x="{x1 + 10}" y="{lab_y[s] + 4:.1f}" font-size="12" font-weight="600" style="fill:{c}">{NAME[s]}</text>')
    A(f'<text x="{x1 + 10}" y="{lab_y[s] + 18:.1f}" font-size="10.5" style="fill:var(--muted)">best possible {orc:.1f}</text>')
A('</svg>')
svg = "\n".join(o)

rows = ""
for s in ["chat", "code", "math"]:
    r = d["sets"][s]
    cells = "".join(f"<td>+{r[f'gap_B{b}'] * 100:.0f}%</td>" for b in B)
    rows += f'<tr><td><span class="dotk" style="background:{COL[s]}"></span>{NAME[s]}</td><td>{r["positions"]}</td>{cells}</tr>'

html = f'''<div class="live" id="gapResults">
      <div class="state">Done · Qwen3-8B (4-bit) target · Qwen3-0.6B drafter · 30 held-out prompts each · {sum(r["positions"] for r in d["sets"].values())} tree positions</div>
      <figure class="fig1" style="margin:0"><div class="figscroll" style="border:0;padding:0">{svg}</div>
      <figcaption><b>Figure 4.</b> At each position the drafter grows about 460 candidate nodes. Solid lines: the target checks the nodes the drafter is most confident in. Dashed lines: the best any choice of nodes could do from the same forest. The space between them is what a better selector, such as one guided by the target&#8217;s gist, could gain at most. Hover a point for its numbers.</figcaption></figure>
      <div class="scroll"><table class="gaptable"><thead><tr><th>Set</th><th>Positions</th><th>8 nodes</th><th>16</th><th>32</th><th>64</th><th>128</th></tr></thead><tbody>{rows}</tbody></table></div>
      <p>Better selection is worth most where the drafter is weakest and the budget is small. At 32 nodes the ceiling is +36% on chat, +20% on code and +13% on math. That clears our go line of 20% on chat and code. These are upper bounds, and a real selector would recover only part of each gap.</p>
      <p class="muted">Two caveats. On chat the forest often misses the target&#8217;s path entirely, so even perfect selection reaches 5.2 of 8: how the forest is grown matters there too. And on this 4-bit setup, checking 128 nodes cost the same as checking one (about 100 ms), because the 4-bit kernels dominate. Real inference kernels have a knee, usually in the tens of nodes, and that knee decides which column above matters.</p>
    </div>'''
open("gap.html", "w", encoding="utf-8").write(html)
print("ok")
