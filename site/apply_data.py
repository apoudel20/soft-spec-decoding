import html
import json

D = json.load(open(r"C:\Users\aayus\Projects\movement-softspec\research\soft-spec-decoding\results\data_stats.json"))
tr = D["sets"]["train"]
dev = {k: D["sets"][f"dev_{k}"][k] for k in ("math", "code", "chat")}
COL = {"math": "#008a72", "code": "#c25a17", "chat": "#6a4fc0"}
NAME = {"math": "Math", "code": "Code", "chat": "Chat"}
SRC = {"math": ("GSM8K", "grade-school word problems, answered step by step"),
       "code": ("MBPP", "short Python function tasks"),
       "chat": ("UltraChat", "open requests: writing, explaining, advice")}
ORDER = ["chat", "math", "code"]
bins = D["bins"]

ntr = sum(tr[k]["count"] for k in ORDER)
tok_tr = sum(tr[k]["total_answer_tokens"] for k in ORDER)
ndev = sum(dev[k]["count"] for k in ORDER)
tok_dev = sum(dev[k]["total_answer_tokens"] for k in ORDER)


def tile(big, small):
    return f'<div class="dtile"><div class="dbig">{big}</div><div class="dsmall">{small}</div></div>'


tiles = (tile(f"{ntr:,}", "training answers now, 2,000 when generation finishes")
         + tile(f"{tok_tr / 1000:.0f}k", "answer tokens in training so far")
         + tile(f"{ndev}", f"held-out dev prompts, {tok_dev / 1000:.1f}k answer tokens")
         + tile(f"{tr['chat']['frac_at_cap'] * 100:.0f}%", "of chat answers stop at the 320-token cap"))

rows = ""
for k in ORDER:
    a, d = tr[k], dev[k]
    at = a["answer_tokens"]
    rows += (f'<tr><td><span class="dotk" style="background:{COL[k]}"></span>{NAME[k]}</td><td>{SRC[k][0]}</td>'
             f'<td class="dl">{SRC[k][1]}</td><td>{a["count"]}</td><td>{d["count"]}</td>'
             f'<td>{a["prompt_tokens"]["median"]:.0f}</td><td>{at["median"]:.0f} <span class="drng">({at["p10"]:.0f} to {at["p90"]:.0f})</span></td>'
             f'<td>{a["frac_at_cap"] * 100:.0f}%</td></tr>')
table = ('<div class="scroll"><table class="gaptable dtable"><thead><tr><th>Set</th><th>Source</th><th class="dl">What it is</th>'
         '<th>Train</th><th>Dev</th><th>Prompt</th><th>Answer (middle 80%)</th><th>At cap</th></tr></thead>'
         f'<tbody>{rows}</tbody></table></div>')

# small multiples: answer length histograms
W, H, PL, PB, PT = 230, 150, 30, 26, 22
svgs = ""
for k in ORDER:
    h = tr[k]["hist"]
    n = tr[k]["count"]
    mx = max(h)
    bw = (W - PL - 6) / len(h)
    bars = ""
    for i, c in enumerate(h):
        bh = (H - PB - PT) * c / mx
        x = PL + i * bw
        y = H - PB - bh
        last = i == len(h) - 1
        lab = f"{bins[i]} to {bins[i + 1] - 1} tokens" if not last else "hit the 320-token cap"
        op = "1" if last else "0.55"
        bars += (f'<rect x="{x + 1:.1f}" y="{y:.1f}" width="{bw - 2:.1f}" height="{max(bh, 0.5):.1f}" rx="2" '
                 f'fill="{COL[k]}" fill-opacity="{op}"><title>{NAME[k]}: {c} of {n} answers {lab}</title></rect>')
    capn = h[-1]
    axis = (f'<line x1="{PL}" y1="{H - PB}" x2="{W - 6}" y2="{H - PB}" style="stroke:var(--line)"/>'
            f'<text x="{PL}" y="{H - PB + 15}" font-size="11" style="fill:var(--muted)">0</text>'
            f'<text x="{PL + (W - PL - 6) / 2:.0f}" y="{H - PB + 15}" font-size="11" text-anchor="middle" style="fill:var(--muted)">160</text>'
            f'<text x="{W - 6}" y="{H - PB + 15}" font-size="11" text-anchor="end" style="fill:var(--muted)">320</text>'
            f'<text x="{PL}" y="13" font-size="12.5" font-weight="600" style="fill:var(--fg)">{NAME[k]}</text>'
            f'<text x="{W - 6}" y="13" font-size="11" text-anchor="end" style="fill:var(--muted)">{capn} at cap</text>')
    svgs += (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{NAME[k]} answer lengths: median '
             f'{tr[k]["answer_tokens"]["median"]:.0f} tokens, {capn} of {n} at the cap">{axis}{bars}</svg>')
fig = ('<figure class="fig1" style="margin:0"><div class="dmult">' + svgs + '</div>'
       '<figcaption>Answer length in tokens for the 600 training answers, in bins of 20. The solid last bar is answers '
       'cut off at the 320-token cap. Chat answers almost always run longer than the cap, so the chat data is mostly '
       'the opening 320 tokens of long answers. Code is bimodal: one-line functions and long explained ones. Hover a '
       'bar for its count.</figcaption></figure>')

# chat request types
ty = tr["chat"]["types"]
tot = sum(ty.values())
tb = ""
for name, c in sorted(ty.items(), key=lambda x: -x[1]):
    tb += (f'<div class="drow"><div class="dlab">{html.escape(name)}</div><div class="dtrack">'
           f'<div class="dfill" style="width:{100 * c / max(ty.values()):.1f}%"></div></div>'
           f'<div class="dval">{c} <span>({100 * c / tot:.0f}%)</span></div></div>')
types = ('<div class="dtypes"><div class="dhead">What the chat prompts ask for</div>' + tb +
         '<p class="muted" style="font-size:13px;margin:6px 0 0">Sorted by the request\'s opening words, so this is a rough '
         'split. "Other" prompts open with context or a passage and put the request after it.</p></div>')

# examples
ex = ""
for k in ORDER:
    e = tr[k]["examples"][1 if k == "math" else 0]
    ex += (f'<div class="dex"><div class="dexh"><span class="dotk" style="background:{COL[k]}"></span>{NAME[k]} · '
           f'{e["answer_tokens"]} answer tokens</div><div class="dexp">{html.escape(e["prompt"])}</div>'
           f'<div class="dexa">{html.escape(e["answer"])}&hellip;</div></div>')
examples = '<div class="dhead">One example from each set</div><div class="dexs">' + ex + '</div>'

blk = f'''<div class="live" id="dataBlock">
      <div class="state">Training data · Qwen3-8B&#8217;s own greedy answers · thinking off · answers capped at 320 tokens</div>
      <div class="dtiles">{tiles}</div>
      {table}
      {fig}
      <div class="dsplit">{types}<div>{examples}</div></div>
      <p class="muted" style="font-size:13.5px">Every answer is what Qwen3-8B itself writes at temperature 0, generated with Ollama (4-bit) for speed and re-tokenised for the HF model. The 1,400 answers being added use the same mix (half chat, a quarter each math and code) from prompts not used before. Dev prompts come from separate splits and are only used to check results. The two 4-bit builds of Qwen3-8B pick a different next token at 4% of math, 6% of code and 13% of chat positions, almost all near-ties, so offline scores are measured against the HF model&#8217;s own choice.</p>
    </div>
'''

CSS = '''
.dtiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
.dtile { border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; background: var(--bg); }
.dbig { font: 600 26px var(--display); color: var(--fg); }
.dsmall { font-size: 12.5px; color: var(--muted); line-height: 1.4; margin-top: 2px; }
.dtable td, .dtable th { white-space: nowrap; }
.dtable td:nth-child(2), .dtable th:nth-child(2) { text-align: left; }
.dtable .dl { text-align: left !important; color: var(--muted); white-space: normal; min-width: 140px; }
.drng { color: var(--muted); font-size: 12px; }
.dmult { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; }
.dmult svg { width: 100%; height: auto; }
.dmult rect:hover { fill-opacity: 1; }
.dsplit { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 22px; align-items: start; }
.dsplit > * { min-width: 0; }
.dhead { font-weight: 600; font-size: 14px; margin-bottom: 8px; }
.drow { display: grid; grid-template-columns: minmax(0, 1.3fr) minmax(0, 1fr) auto; gap: 10px; align-items: center; font-size: 13px; padding: 4px 0; }
.dtrack { background: var(--line); border-radius: 4px; height: 10px; }
.dfill { background: #6a4fc0; border-radius: 4px; height: 10px; }
.dval { font: 12.5px var(--mono); }
.dval span { color: var(--muted); }
.dexs { display: grid; gap: 10px; }
.dex { border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px; font-size: 13px; }
.dexh { font: 12px var(--mono); color: var(--muted); margin-bottom: 4px; }
.dexp { font-weight: 500; margin-bottom: 4px; }
.dexa { color: var(--muted); white-space: pre-wrap; overflow-wrap: anywhere; max-height: 7.5em; overflow: hidden; }
'''

t = open("template.html", encoding="utf-8").read()
if "id=\"dataBlock\"" in t:
    a = t.index('<div class="live" id="dataBlock">')
    b = t.index('<div class="live" id="gapResults">')
    t = t[:a] + blk + "    " + t[b:]
else:
    a = t.index('<div class="live" id="gapResults">')
    t = t[:a] + blk + "    " + t[a:]
if ".dtiles {" not in t:
    i = t.index("</style>")
    t = t[:i] + CSS + t[i:]
t = t.replace('<a href="#live">Experiments</a>', '<a href="#dataBlock">Data</a><a href="#live">Experiments</a>') \
    if 'href="#dataBlock"' not in t else t
open("template.html", "w", encoding="utf-8").write(t)
open("index.html", "w", encoding="utf-8").write(t.replace("__SUMMARY__", open("summary.json", encoding="utf-8").read()))
print("ok", t.count('id="dataBlock"'))
