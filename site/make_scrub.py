"""Interactive figure: scrub through one round of target-guided tree drafting."""
out = []
A = out.append
W, H = 960, 360


def ph(lo, hi=99):
    return f'class="ph" data-lo="{lo}" data-hi="{hi}"'


A(f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="One round of target-guided drafting, in phases: the drafter grows a tree, '
  'the target reads a gist from middle layers, the gist picks branches, the target checks them in every layer at once, samples one path, '
  'and the drafter re-roots on it." style="width:100%;height:auto;color:var(--fg);font-family:var(--sans)">')

# ---- drafter tree geometry (tidy layout) ----
fan = [3, 3, 2, 2]
cols = [150, 235, 320, 405, 490]
top, bottom = 62, 300
kids = {(): []}
frontier = [()]
for f in fan:
    nxt = []
    for p in frontier:
        kids[p] = [p + (i,) for i in range(f)]
        for c in kids[p]:
            kids[c] = []
        nxt += kids[p]
    frontier = nxt
pos = {}
step = (bottom - top) / (len(frontier) - 1)
for i, lf in enumerate(frontier):
    pos[lf] = (cols[len(lf)], top + i * step)


def place(n):
    if n in pos:
        return pos[n][1]
    ys = [place(c) for c in kids[n]]
    pos[n] = (cols[len(n)], sum(ys) / len(ys))
    return pos[n][1]


place(())
acc = {(), (0,), (0, 1), (0, 1, 0), (0, 1, 0, 1)}
kv = acc | {(1,), (0, 0), (0, 2), (1, 0), (0, 0, 0), (0, 1, 1), (1, 0, 0), (0, 1, 0, 0), (0, 0, 0, 0), (0, 1, 1, 0)}
ry = pos[()][1]


def curve(n):
    x, y = pos[n]
    px, py = pos[n[:-1]]
    mx = (px + x) / 2
    return f'M{px:.1f} {py:.1f} C{mx:.1f} {py:.1f},{mx:.1f} {y:.1f},{x:.1f} {y:.1f}'


# phase 0: verified prefix
A(f'<g {ph(0)}>')
A(f'<line x1="34" y1="{ry:.1f}" x2="{cols[0]}" y2="{ry:.1f}" style="stroke:var(--ours);stroke-width:2.6"/>')
for i in range(4):
    A(f'<rect x="{30 + i * 26}" y="{ry - 6:.1f}" width="14" height="12" rx="2" style="fill:var(--ours)"/>')
A(f'<text x="30" y="{ry - 15:.1f}" font-size="11" style="fill:var(--ours)">verified prefix</text>')
A('</g>')

# phase 1: drafter tree (fades after accept)
A(f'<g {ph(1, 5)}>')
for n in pos:
    if n != ():
        A(f'<path d="{curve(n)}" style="fill:none;stroke:var(--muted);stroke-width:1;opacity:.5"/>')
for n, (x, y) in pos.items():
    if n != ():
        A(f'<circle cx="{x}" cy="{y:.1f}" r="{3 if len(n) < 4 else 2.4}" style="fill:var(--muted);opacity:.65"/>')
A('<text x="150" y="46" font-size="12" font-weight="600" style="fill:currentColor">Drafter tree</text>')
A('</g>')

# ---- target stack ----
NL, px0, pw, skew, phh, ptop, gap = 6, 640, 190, 44, 30, 96, 36
hot = {2, 3}
ymin = min(pos[n][1] for n in kv)
ymax = max(pos[n][1] for n in kv)


def to_plane(n, y0):
    x, y = pos[n]
    u = (x - cols[0]) / (cols[-1] - cols[0])
    v = (y - ymin) / (ymax - ymin)
    return (px0 + 14 + u * (pw - 28) + (1 - v) * skew, y0 - 4 - (1 - v) * (phh - 8))


A(f'<g {ph(0)}>')
A(f'<text x="{px0}" y="{ptop - phh - 14}" font-size="12" font-weight="600" style="fill:currentColor">9B target</text>')
for L in range(NL - 1, -1, -1):
    y0 = ptop + L * gap
    A(f'<polygon points="{px0},{y0} {px0 + pw},{y0} {px0 + pw + skew},{y0 - phh} {px0 + skew},{y0 - phh}" style="fill:var(--panel);stroke:var(--line);stroke-width:1"/>')
A('</g>')
# phase 2: middle layers glow, gist forms
A(f'<g {ph(2)}>')
for L in sorted(hot, reverse=True):
    y0 = ptop + L * gap
    A(f'<polygon points="{px0},{y0} {px0 + pw},{y0} {px0 + pw + skew},{y0 - phh} {px0 + skew},{y0 - phh}" style="fill:var(--self-bg);stroke:var(--self);stroke-width:1.1;opacity:.9"/>')
A(f'<text x="{px0 + pw + skew + 8}" y="{ptop + 2.5 * gap - phh / 2 + 4:.0f}" font-size="11" style="fill:var(--self)">middle layers</text>')
hs = [10, 22, 7, 16, 25, 9, 19, 13]
for i, h in enumerate(hs):
    A(f'<rect x="{560 + i * 7}" y="{330 - h}" width="5" height="{h}" rx="1" style="fill:var(--self)"/>')
A('<text x="588" y="346" text-anchor="middle" font-size="11" font-style="italic" style="fill:var(--self)">gist</text>')
A(f'<path d="M{px0 + 20} {ptop + 3 * gap - 4} C{px0 - 10} {ptop + 3 * gap + 40},610 300,596 304" style="fill:none;stroke:var(--self);stroke-width:1.2"/>')
A('</g>')

# phase 3: tendrils + chosen branches become cache entries
A(f'<g {ph(3, 4)}>')
for n in kv:
    if n != ():
        A(f'<path d="{curve(n)}" style="fill:none;stroke:var(--self);stroke-width:11;stroke-linecap:round;opacity:.13"/>')
        A(f'<path d="{curve(n)}" style="fill:none;stroke:var(--self);stroke-width:1.6"/>')
for n in kv:
    x, y = pos[n]
    A(f'<rect x="{x - 5.5}" y="{y - 4.5:.1f}" width="11" height="9" rx="2" style="fill:var(--self-bg);stroke:var(--self);stroke-width:1.4"/>')
for n in [(0,), (1,), (0, 1), (0, 0), (0, 1, 0)]:
    x, y = pos[n]
    A(f'<path d="M560 300 C520 {y + 90:.0f},{x + 70:.0f} {y + 50:.0f},{x + 7:.1f} {y + 5:.1f}" style="fill:none;stroke:var(--self);stroke-width:1.1;stroke-dasharray:3 3"/>')
A('</g>')

# phase 4: the KV-cache tree appears in every layer
A(f'<g {ph(4)}>')
for L in range(NL - 1, -1, -1):
    y0 = ptop + L * gap
    pp = {n: to_plane(n, y0) for n in kv}
    for n in kv:
        if n != ():
            a, b = pp[n[:-1]], pp[n]
            A(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" style="stroke:var(--self);stroke-width:0.9"/>')
    for n in kv:
        p = pp[n]
        A(f'<rect x="{p[0] - 2.6:.1f}" y="{p[1] - 2.2:.1f}" width="5.2" height="4.4" rx="1" style="fill:var(--self)"/>')
A('</g>')
A(f'<g {ph(4, 4)}>')
A(f'<path d="M500 120 C560 100,600 {ptop - 8},{px0 - 4} {ptop - 8}" style="fill:none;stroke:var(--self);stroke-width:1.4"/>')
A('<text x="560" y="92" text-anchor="middle" font-size="11" style="fill:var(--self)">all layers, one pass</text>')
A('</g>')

# phase 5: accepted path (in the drafter tree and in every layer)
A(f'<g {ph(5)}>')
for n in acc:
    if n != ():
        A(f'<path d="{curve(n)}" style="fill:none;stroke:var(--ours);stroke-width:2.8"/>')
for n in acc:
    x, y = pos[n]
    A(f'<rect x="{x - 6}" y="{y - 5:.1f}" width="12" height="10" rx="2" style="fill:var(--ours)"/>')
for L in range(NL - 1, -1, -1):
    y0 = ptop + L * gap
    pp = {n: to_plane(n, y0) for n in acc}
    for n in acc:
        if n != ():
            a, b = pp[n[:-1]], pp[n]
            A(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" style="stroke:var(--ours);stroke-width:1.8"/>')
A('<text x="760" y="330" text-anchor="middle" font-size="11" style="fill:var(--ours)">target samples one path, output unchanged</text>')
A('</g>')

# phase 6: re-root and regrow from the end of the accepted path
ex, ey = pos[(0, 1, 0, 1)]
A(f'<g {ph(6)}>')
new = [(ex + 40, ey - 26), (ex + 40, ey + 4), (ex + 40, ey + 30)]
for (x, y) in new:
    A(f'<path d="M{ex} {ey:.1f} C{ex + 20} {ey:.1f},{ex + 20} {y:.1f},{x} {y:.1f}" style="fill:none;stroke:var(--muted);stroke-width:1"/>')
    for dy in (-9, 9):
        A(f'<path d="M{x} {y:.1f} C{x + 14} {y:.1f},{x + 14} {y + dy:.1f},{x + 26} {y + dy:.1f}" style="fill:none;stroke:var(--muted);stroke-width:0.9;opacity:.7"/>')
        A(f'<circle cx="{x + 26}" cy="{y + dy:.1f}" r="2.2" style="fill:var(--muted)"/>')
    A(f'<circle cx="{x}" cy="{y:.1f}" r="2.8" style="fill:var(--muted)"/>')
A(f'<text x="{ex + 4}" y="{ey + 54:.0f}" font-size="11" style="fill:var(--muted)">new tree, already growing</text>')
A('</g>')
A('</svg>')
svg = "\n".join(out)

PHASES = [
    ("Context", "The verified text so far is already in the target's KV cache."),
    ("Draft", "The small drafter grows a wide tree of possible next tokens. This is cheap."),
    ("Read the gist", "The target's middle layers carry a rough sense of where the answer is going. A small head reads it out as a gist vector."),
    ("Choose", "The gist reaches into the drafter's tree and picks the branches worth checking. Those become the KV-cache tree."),
    ("Verify", "The target checks every chosen node in one forward pass. Each of its layers stores a key and value for each node."),
    ("Accept", "The target samples one path from its own distribution, so the output is exactly what it would have written alone."),
    ("Re-root", "The drafter keeps the subtree under the accepted path, which it has already grown, and the next round begins."),
]
steps_html = "".join(f'<li data-i="{i}"><span>{i + 1}</span>{t}</li>' for i, (t, _) in enumerate(PHASES))
import json
html = f'''<figure class="fig1 scrub" id="scrub">
<div class="figscroll">{svg}</div>
<div class="scrubbar">
  <button type="button" id="scrubPrev" aria-label="Previous phase">‹</button>
  <button type="button" id="scrubPlay" aria-label="Play">Play</button>
  <button type="button" id="scrubNext" aria-label="Next phase">›</button>
  <input type="range" id="scrubRange" min="0" max="{len(PHASES) - 1}" step="1" value="{len(PHASES) - 1}" aria-label="Phase">
</div>
<ol class="phases" id="scrubSteps">{steps_html}</ol>
<p class="phasetext" id="scrubText" aria-live="polite"></p>
<figcaption><b>Figure 1.</b> One round of target-guided drafting. Drag the slider or press play to step through it. This illustrates the design. It is not a measured result.</figcaption>
</figure>
<script>
(function () {{
  const P = {json.dumps(PHASES)};
  const fig = document.getElementById("scrub");
  const r = document.getElementById("scrubRange"), txt = document.getElementById("scrubText");
  const items = fig.querySelectorAll(".ph"), steps = fig.querySelectorAll("#scrubSteps li");
  let timer = null;
  function show(p) {{
    p = Math.max(0, Math.min(P.length - 1, p)); r.value = p;
    items.forEach(g => {{ const on = p >= +g.dataset.lo && p <= +g.dataset.hi; g.style.opacity = on ? 1 : 0; }});
    steps.forEach((li, i) => li.classList.toggle("on", i === p));
    txt.textContent = P[p][1];
  }}
  function stop() {{ clearInterval(timer); timer = null; document.getElementById("scrubPlay").textContent = "Play"; }}
  r.addEventListener("input", () => {{ stop(); show(+r.value); }});
  document.getElementById("scrubPrev").onclick = () => {{ stop(); show(+r.value - 1); }};
  document.getElementById("scrubNext").onclick = () => {{ stop(); show(+r.value + 1); }};
  steps.forEach(li => li.addEventListener("click", () => {{ stop(); show(+li.dataset.i); }}));
  document.getElementById("scrubPlay").onclick = () => {{
    if (timer) return stop();
    document.getElementById("scrubPlay").textContent = "Pause";
    if (+r.value >= P.length - 1) show(0);
    timer = setInterval(() => {{ if (+r.value >= P.length - 1) return stop(); show(+r.value + 1); }}, 1600);
  }};
  show(P.length - 1);
}})();
</script>'''
open("scrub.html", "w", encoding="utf-8").write(html)
print("ok", len(html))
