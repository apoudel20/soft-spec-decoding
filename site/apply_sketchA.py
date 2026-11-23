import json
d = json.load(open(r"C:\Users\aayus\Projects\movement-softspec\research\soft-spec-decoding\results\dflash_plan.json"))
COL = {"math": "#008a72", "code": "#c25a17", "chat": "#6a4fc0"}
rows = ""
for s in ["math", "code", "chat"]:
    r = d["sets"][s]
    base = r["none"]["mean_accepted"]
    def cell(p):
        v = r[p]["mean_accepted"]
        return f"{v:.2f}" if p == "none" else f"{v:.2f} ({(v / base - 1) * 100:+.1f}%)"
    rows += (f'<tr><td><span class="dotk" style="background:{COL[s]}"></span>{s.capitalize()}</td><td>{r["none"]["blocks"]}</td>'
             f'<td>{cell("none")}</td><td>{cell("mid")}</td><td>{cell("oracle")}</td></tr>')
html = f'''<div class="live" id="sketchAResults">
      <div class="state">Done · Sketch A on DFlash · Qwen3-8B (4-bit) target · z-lab DFlash drafter, frozen · held-out dev answers</div>
      <div class="scroll"><table class="gaptable"><thead><tr><th>Set</th><th>Blocks</th><th>Plain DFlash</th><th>+ plan from layers 18, 20</th><th>+ oracle plan (true future)</th></tr></thead><tbody>{rows}</tbody></table></div>
      <p>Accepted tokens per 16-token block. A learned plan vector added to the block&#8217;s input embeddings changes almost nothing. Code&#8217;s +3% is within about one standard error. Even the oracle, built from the target&#8217;s actual next 64 positions, does not help, which points at the injection rather than the idea: DFlash was trained to read the target through its context keys and values, not through its input embeddings.</p>
      <p class="muted">Next: a no-training test that appends the target&#8217;s true future states as one extra context position, DFlash&#8217;s own channel. If DFlash can use future information there, the follow-up is learning to predict that position.</p>
    </div>'''
t = open('template.html', encoding='utf-8').read()
if 'id="sketchAResults"' in t:
    a = t.index('<div class="live" id="sketchAResults">'); b = t.index('</p>\n    </div>', a) + len('</p>\n    </div>')
    t = t[:a] + html + t[b:]
else:
    a = t.index('<div class="live" id="layerResults">')
    end = t.index('</section>', a)
    t = t[:end] + '  ' + html + '\n  ' + t[end:]
t = t.replace('<tr><td>Better tree selection, ceiling</td>', '<tr><td>Plan vector added to DFlash inputs</td><td><span class="pill no">No effect</span>±3%, oracle included</td><td>ConFu (different pathway)</td></tr>\n        <tr><td>Better tree selection, ceiling</td>')
open('template.html', 'w', encoding='utf-8').write(t)
open('index.html', 'w', encoding='utf-8').write(t.replace('__SUMMARY__', open('summary.json', encoding='utf-8').read()))
print('page ok', t.count('id="sketchAResults"'))
