"""Layer-sweep chart: where Qwen3-8B keeps future information, against Qwen3-0.6B, by relative depth."""
import json

d = json.load(open(r"C:\Users\aayus\Projects\movement-softspec\research\soft-spec-decoding\results\probes_q3.json"))
COL = {"chat": "#6a4fc0", "code": "#c25a17", "math": "#008a72"}
MODELS = {"Qwen/Qwen3-8B": ("8B", "none"), "Qwen/Qwen3-0.6B": ("0.6B", "5 4")}


def series(model, metric, s):
    out = []
    for L, sets in d["models"][model].items():
        a, b = L.replace("layer", "").split("of")
        out.append((int(a) / int(b), sets[s][metric]))
    return out


def panel(metric, title, ymin, ymax, ticks, ox):
    W, H = 340, 250
    x0, x1, y0, y1 = ox + 46, ox + W - 10, 210, 30
    X = lambda f: x0 + f * (x1 - x0)
    Y = lambda v: y0 - (max(min(v, ymax), ymin) - ymin) / (ymax - ymin) * (y0 - y1)
    o = [f'<text x="{ox + 46}" y="18" font-size="12.5" font-weight="600" style="fill:currentColor">{title}</text>']
    for t in ticks:
        o.append(f'<line x1="{x0}" y1="{Y(t):.1f}" x2="{x1}" y2="{Y(t):.1f}" style="stroke:var(--line);stroke-width:1"/>')
        o.append(f'<text x="{x0 - 8}" y="{Y(t) + 4:.1f}" text-anchor="end" font-size="11" style="fill:var(--muted)">{t:g}</text>')
    for f in (0, .25, .5, .75, 1):
        o.append(f'<text x="{X(f):.1f}" y="{y0 + 18}" text-anchor="middle" font-size="11" style="fill:var(--muted)">{int(f * 100)}%</text>')
    o.append(f'<text x="{(x0 + x1) / 2}" y="{y0 + 36}" text-anchor="middle" font-size="11.5" style="fill:var(--muted)">layer depth</text>')
    for s in ("math", "code", "chat"):
        for m, (nm, dash) in MODELS.items():
            pts = series(m, metric, s)
            p = " ".join(f"{X(f):.1f},{Y(v):.1f}" for f, v in pts)
            o.append(f'<polyline points="{p}" style="fill:none;stroke:{COL[s]};stroke-width:{2.2 if nm == "8B" else 1.5};'
                     f'{"" if dash == "none" else "stroke-dasharray:" + dash + ";"}stroke-linejoin:round"><title>{s}, {nm}</title></polyline>')
    return "\n".join(o)


svg = ('<svg viewBox="0 0 720 270" role="img" aria-label="Probe accuracy by layer depth for Qwen3-8B (solid) and Qwen3-0.6B (dashed). '
       'Left: predicting a summary of the next 64 tokens, where the 8B is no better than the 0.6B. Right: predicting remaining length, '
       'where the 8B is somewhat better and peaks near the middle of the network." style="width:100%;height:auto;color:var(--fg);font-family:var(--sans)">'
       + panel("bag+64_r2", "Summary of the next 64 tokens (R²)", -0.1, 0.4, [0, 0.1, 0.2, 0.3, 0.4], 0)
       + panel("remaining_r2", "Remaining length (R²)", 0, 0.8, [0, 0.2, 0.4, 0.6, 0.8], 370)
       + '</svg>')
legend = ''.join(f'<span class="lg"><i style="background:{COL[s]}"></i>{s.capitalize()}</span>' for s in ("math", "code", "chat"))
legend += '<span class="lg"><i class="ln"></i>8B</span><span class="lg"><i class="ln dash"></i>0.6B</span>'
html = f'''<div class="live" id="layerResults">
      <div class="state">Done · layer sweep · Qwen3-8B (4-bit) and Qwen3-0.6B · 600 training answers, 40 held-out dev answers per set · every second layer</div>
      <figure class="fig1" style="margin:0"><div class="figscroll" style="border:0;padding:0">{svg}</div>
      <div class="legendrow">{legend}</div>
      <figcaption><b>Figure 5.</b> Linear probes on each layer&#8217;s hidden state, trained on the 8B&#8217;s own answers and tested on held-out dev prompts. Solid lines are the 8B, dashed the 0.6B. Higher is better, and 0 means no better than predicting the average.</figcaption></figure>
      <p>The 8B&#8217;s middle layers (18 to 20 of 36) hold the most about how long and how structured the rest of the answer will be, and there it beats the 0.6B on every set. For the content of the next 64 tokens it does not: its best layers match the 0.6B on math (R² 0.32 vs 0.33) and trail it on code (0.16 vs 0.20).</p>
      <p class="muted">On chat the 8B probe falls below zero at most layers while the 0.6B reaches 0.18. With a 4,096-wide state and about 14,000 training rows the probe likely overfits, so chat is inconclusive here. A fairer comparison needs more data or both models reduced to the same width. Linear probes also only see what is linearly readable.</p>
    </div>'''
open("layers.html", "w", encoding="utf-8").write(html)
print("ok")
