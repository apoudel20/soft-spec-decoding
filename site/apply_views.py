"""Lead with the current design and an honest status, move the original animation into The idea, and give the
Experiments section two views (newest first, what worked first) with an outcome tag on every block."""
t = open("template.html", encoding="utf-8").read()

# 1. honest summary
a = t.index('<p class="sum">')
b = t.index("</p>", a) + 4
t = t[:a] + ('<p class="sum">A small model drafts tokens and a large model checks them, so the output stays exactly the '
             'large model&#8217;s. This project tests whether the large model&#8217;s own internal states can make drafting '
             'better. So far they have not. The target does carry a small amount of extra information about its upcoming '
             'text, but drafters do not use it when it is handed to them, and the current design, a compressed latent of the '
             'target&#8217;s future states, fails its first gate. What does work, trees of drafts, is known prior work.</p>') + t[b:]

# 2. move the original animation into The idea, put the current design diagram up top
fa = t.index('<figure class="fig1 scrub" id="scrub">')
fb = t.index("</figure>", fa) + len("</figure>")
scrub = t[fa:fb]
t = t[:fa] + "__TOPFIG__" + t[fb:]
ga = t.index('<figure class="fig1" style="margin:0" id="aeFig">')
gb = t.index("</figure>", ga) + len("</figure>")
aefig = t[ga:gb]
t = t[:ga] + t[gb:]
top = ('<section class="sec" id="now" aria-labelledby="nowT" style="margin-top:8px">\n    <div class="eyebrow">Now</div>\n'
       '    <h2 id="nowT">Current design and status</h2>\n    '
       + aefig.replace('style="margin:0" id="aeFig"', 'style="margin:0" id="aeFig"')
       + '\n    <p class="muted" style="font-size:13.5px">Status: part A fails its gate (top table in Experiments). Two '
         'diagnostic runs are testing whether that is the training recipe or a real limit. Part B waits on them.</p>\n  </section>\n')
t = t.replace("__TOPFIG__", top)
ia = t.index('<h2 id="ideaT">')
ib = t.index("</h2>", ia) + len("</h2>")
t = t[:ib] + ('\n    <p class="muted" style="font-size:14px">Where the project started: the target&#8217;s view of its own '
              'future (its gist, read from middle layers) steers which drafted branches get checked. The experiments below '
              'tested the pieces of this, and so far none of them helped.</p>\n    ' + scrub) + t[ib:]

# 3. outcome tags and two views of the experiments
TAG = {"nextBuild": ("Next", "pend"), "aeGate": ("Fails gate", "no"), "dataBlock": ("Data", "pend"),
       "treeGap": ("Works, known", "ok"), "futureTests": ("No effect", "no"), "sketchAResults": ("No effect", "no"),
       "dflashBase": ("Reproduced", "ok"), "layerResults": ("Small signal", "ok"), "gapResults": ("Room to gain", "ok")}
for k, (lab, cls) in TAG.items():
    a = t.index(f'<div class="live" id="{k}">')
    s = t.index('<div class="state">', a) + len('<div class="state">')
    if t[s:s + 30].startswith('<span class="otag'):
        continue
    t = t[:s] + f'<span class="otag {cls}">{lab}</span>' + t[s:]

toggle = ('<div class="viewtog" role="group" aria-label="Order of experiments">'
          '<button type="button" data-view="newest" aria-pressed="true">Newest first</button>'
          '<button type="button" data-view="worked" aria-pressed="false">What worked first</button></div>\n    ')
if 'class="viewtog"' not in t:
    a = t.index('<div class="live" id="nextBuild">')
    a = t.rfind("\n", 0, a) + 1
    t = t[:a] + "    " + toggle + '<div id="expList">\n' + t[a:]
    d = t.index('<details class="earlier">')
    e = t.index("</details>", d) + len("</details>")
    t = t[:e] + "\n    </div>" + t[e:]
    s = t.index("<summary>", d)
    se = t.index("</summary>", s)
    t = t[:s] + '<summary><span class="earlierTitle">Earlier experiments, newest first</span> <span class="earlierSub">tree headroom, future-information tests, Sketch A, DFlash baseline, layer sweep, selection gap</span>' + t[se:]

JS = """
<script>
(function () {
  var NEWEST = { list: ["nextBuild", "aeGate", "dataBlock"], fold: ["treeGap", "futureTests", "sketchAResults", "dflashBase", "layerResults", "gapResults"],
    title: "Earlier experiments, newest first", sub: "tree headroom, future-information tests, Sketch A, DFlash baseline, layer sweep, selection gap" };
  var WORKED = { list: ["nextBuild", "treeGap", "gapResults", "layerResults", "dflashBase", "dataBlock"], fold: ["aeGate", "futureTests", "sketchAResults"],
    title: "What did not work", sub: "chunk autoencoder gate, true future fed to DFlash, plan vector on DFlash" };
  function apply(v) {
    var cfg = v === "worked" ? WORKED : NEWEST;
    var list = document.getElementById("expList"); if (!list) return;
    var det = list.querySelector("details.earlier"), body = det.querySelector(".earlierBody");
    cfg.list.forEach(function (id) { var el = document.getElementById(id); if (el) list.insertBefore(el, det); });
    cfg.fold.forEach(function (id) { var el = document.getElementById(id); if (el) body.appendChild(el); });
    det.querySelector(".earlierTitle").textContent = cfg.title;
    det.querySelector(".earlierSub").textContent = cfg.sub;
    document.querySelectorAll(".viewtog button").forEach(function (b) { b.setAttribute("aria-pressed", b.dataset.view === v ? "true" : "false"); });
    try { localStorage.setItem("expView", v); } catch (e) {}
  }
  document.querySelectorAll(".viewtog button").forEach(function (b) { b.addEventListener("click", function () { apply(b.dataset.view); }); });
  var v = "newest"; try { v = localStorage.getItem("expView") || "newest"; } catch (e) {}
  apply(v);
})();
</script>
"""
CSS = """
.otag { display: inline-block; font: 600 11px var(--sans); padding: 2px 8px; border-radius: 999px; margin-right: 8px; vertical-align: 1px; }
.otag.ok { background: var(--accept-bg); color: var(--accept); }
.otag.no { background: var(--reject-bg); color: var(--reject); }
.otag.pend { background: var(--accent-bg); color: var(--accent); }
.viewtog { display: inline-flex; gap: 6px; flex-wrap: wrap; }
.viewtog button { font: 500 13.5px var(--sans); padding: 7px 14px; border-radius: 8px; border: 1px solid var(--line); background: var(--panel); color: var(--fg); cursor: pointer; }
.viewtog button[aria-pressed="true"] { background: var(--btn); color: var(--btn-fg); border-color: var(--btn); }
#expList { display: grid; gap: 18px; }
details.earlier > summary .earlierSub { display: block; font-weight: 400; font-size: 13px; color: var(--muted); margin-top: 4px; }
"""
if ".otag {" not in t:
    i = t.index("</style>")
    t = t[:i] + CSS + t[i:]
if 'getItem("expView")' not in t:
    i = t.rindex("</body>") if "</body>" in t else len(t)
    t = t[:i] + JS + t[i:]
open("template.html", "w", encoding="utf-8").write(t)
open("index.html", "w", encoding="utf-8").write(t.replace("__SUMMARY__", open("summary.json", encoding="utf-8").read()))
import re
print("ok", "div balance", len(re.findall(r"<div\b", t)) - t.count("</div>"), "| now sec", t.count('id="now"'), "| scrub in idea", t.index('id="scrub"') > t.index('id="ideaT"'))
