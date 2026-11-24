"""Experiments section newest first: Next, latest result, data, then earlier experiments folded (newest first).
Also drops stale paragraphs that earlier Next-block rewrites left behind after the Next block."""
t = open("template.html", encoding="utf-8").read()
s0 = t.index('<section class="sec" id="live"')
s1 = t.index("  </section>", s0)
sec = t[s0:s1]
order_orig = ["dataBlock", "gapResults", "layerResults", "dflashBase", "sketchAResults", "futureTests", "treeGap", "aeGate", "nextBuild"]
starts = {}
for k in order_orig:
    i = sec.find(f'<div class="live" id="{k}">')
    assert i >= 0, k
    starts[k] = sec.rfind("\n", 0, i) + 1
head = sec[:starts["dataBlock"]]
assert "earlier" not in head
blocks = {}
for a, b in zip(order_orig, order_orig[1:]):
    blocks[a] = sec[starts[a]:starts[b]]
nb = sec[starts["nextBuild"]:]
end = nb.index("\n    </div>\n") + len("\n    </div>\n")
blocks["nextBuild"] = nb[:end]
dropped = nb[end:]
for k, v in blocks.items():
    assert v.count("<div") == v.count("</div>"), (k, v.count("<div"), v.count("</div>"))
earlier = ["treeGap", "futureTests", "sketchAResults", "dflashBase", "layerResults", "gapResults"]
new = (head + blocks["nextBuild"] + blocks["aeGate"] + blocks["dataBlock"]
       + '    <details class="earlier">\n      <summary>Earlier experiments, newest first <span>tree headroom, future-information tests, '
         'Sketch A, DFlash baseline, layer sweep, selection gap</span></summary>\n      <div class="earlierBody">\n'
       + "".join(blocks[k] for k in earlier) + "      </div>\n    </details>\n")
CSS = """
details.earlier { border: 1px solid var(--line); border-radius: 12px; background: var(--panel); }
details.earlier > summary { cursor: pointer; padding: 16px 22px; font-weight: 600; font-size: 15px; list-style-position: inside; }
details.earlier > summary span { display: block; font-weight: 400; font-size: 13px; color: var(--muted); margin-top: 4px; }
details.earlier .earlierBody { display: grid; gap: 16px; padding: 0 14px 14px; }
"""
t = t[:s0] + new + t[s1:]
if "details.earlier" not in t:
    i = t.index("</style>")
    t = t[:i] + CSS + t[i:]
open("template.html", "w", encoding="utf-8").write(t)
open("index.html", "w", encoding="utf-8").write(t.replace("__SUMMARY__", open("summary.json", encoding="utf-8").read()))
print("ok; dropped stale chars:", len(dropped.strip()), "| paragraphs dropped:", dropped.count("<p"))
