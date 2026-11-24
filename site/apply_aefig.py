"""Add the autoencoder-drafter diagram (stage 2 tested, stage 3 not built) to the aeGate block."""
T = 'style="fill:var(--fg)"'
M = 'style="fill:var(--muted)"'
PURPLE = "#6a4fc0"


def arrow(x1, y1, x2, y2, dash=False):
    d = ' stroke-dasharray="4 3"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" style="stroke:var(--muted);stroke-width:1.4"{d} '
            f'marker-end="url(#aeArrow)"/>')


def grid(x0, y0, hi_final=False, faded=False):
    s = ""
    for r in range(4):
        for c in range(8):
            op = 0.25 + 0.18 * r if not faded else 0.18 + 0.1 * r
            fill = "var(--ours)"
            if hi_final and r == 3:
                op = 0.9
            s += f'<rect x="{x0 + c * 12}" y="{y0 + r * 12}" width="10" height="10" rx="2" style="fill:{fill}" fill-opacity="{op:.2f}"/>'
    return s


def chip(x, y, w, txt, col="var(--ours)"):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="20" rx="5" style="fill:var(--panel);stroke:{col}"/>'
            f'<text x="{x + w / 2}" y="{y + 14}" text-anchor="middle" font-size="11.5" {T}>{txt}</text>')


svg = ['<svg viewBox="0 0 820 400" role="img" aria-label="Two-part diagram. A, tested: Qwen3-8B hidden states at layers 18, 24, 30 and the final layer for 8 upcoming positions form a grid, an encoder squeezes the grid into one latent, a decoder rebuilds it, and the frozen output layer must pick the target&#8217;s own next tokens. It reaches 94% math, 93% code and 78% chat, below the 99% needed. B, not built: a search network samples several latents for the next 8 tokens, each decodes into a whole continuation, and the continuations are merged into one draft tree that the target checks in a single pass.">',
       '<defs><marker id="aeArrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
       '<path d="M0,0 L10,5 L0,10 z" style="fill:var(--muted)"/></marker></defs>']
# panel A
svg.append(f'<text x="20" y="18" font-size="13" font-weight="600" {T}>A · Tested (stage 2): an autoencoder over the target&#8217;s future states</text>')
svg.append('<rect x="20" y="34" width="122" height="156" rx="10" style="fill:var(--accent-bg);stroke:var(--accent)"/>')
svg.append(f'<text x="81" y="54" text-anchor="middle" font-size="12.5" font-weight="600" {T}>Qwen3-8B</text>')
svg.append(f'<text x="81" y="69" text-anchor="middle" font-size="11" {M}>target, frozen</text>')
for i, lab in enumerate(["layer 18", "layer 24", "layer 30", "final layer"]):
    y = 80 + i * 26
    svg.append(f'<rect x="32" y="{y}" width="98" height="20" rx="4" style="fill:var(--panel);stroke:var(--line)"/>'
               f'<text x="81" y="{y + 14}" text-anchor="middle" font-size="11" {T}>{lab}</text>')
svg.append(arrow(144, 118, 176, 118))
svg.append(f'<text x="160" y="110" text-anchor="middle" font-size="10.5" {M}>cache</text>')
svg.append(grid(182, 94))
svg.append(f'<text x="229" y="86" text-anchor="middle" font-size="11" {M}>next 8 positions</text>')
svg.append(f'<text x="229" y="157" text-anchor="middle" font-size="11" {M}>4 layers &#215; 8</text>')
svg.append(arrow(282, 118, 298, 118))
svg.append('<path d="M300,86 L356,104 L356,132 L300,150 Z" style="fill:var(--ours-bg);stroke:var(--ours)"/>')
svg.append(f'<text x="328" y="168" text-anchor="middle" font-size="11" {T}>encoder</text>')
svg.append(f'<rect x="366" y="106" width="26" height="24" rx="5" style="fill:{PURPLE}"/>'
           '<text x="379" y="122" text-anchor="middle" font-size="12" font-weight="600" style="fill:#ffffff">z</text>')
svg.append(f'<text x="379" y="148" text-anchor="middle" font-size="11" {T}>one latent</text>')
svg.append(f'<text x="379" y="162" text-anchor="middle" font-size="10.5" {M}>512 to 2048</text>')
svg.append('<path d="M402,104 L458,86 L458,150 L402,132 Z" style="fill:var(--ours-bg);stroke:var(--ours)"/>')
svg.append(f'<text x="430" y="168" text-anchor="middle" font-size="11" {T}>decoder</text>')
svg.append(arrow(460, 118, 476, 118))
svg.append(grid(480, 94, hi_final=True))
svg.append(f'<text x="527" y="86" text-anchor="middle" font-size="11" {M}>rebuilt grid</text>')
svg.append(arrow(578, 136, 606, 136))
svg.append('<rect x="610" y="122" width="96" height="28" rx="6" style="fill:var(--accent-bg);stroke:var(--accent)"/>')
svg.append(f'<text x="658" y="140" text-anchor="middle" font-size="11.5" {T}>output layer</text>')
svg.append(f'<text x="658" y="112" text-anchor="middle" font-size="10.5" {M}>frozen, shared</text>')
svg.append(arrow(708, 136, 726, 136))
for i in range(4):
    svg.append(f'<rect x="{730 + i * 21}" y="126" width="17" height="20" rx="4" style="fill:var(--panel);stroke:var(--ours)"/>')
svg.append(f'<text x="771" y="162" text-anchor="middle" font-size="10.5" {M}>8 tokens</text>')
svg.append('<rect x="560" y="174" width="250" height="38" rx="8" style="fill:var(--reject-bg);stroke:var(--reject)"/>')
svg.append(f'<text x="685" y="190" text-anchor="middle" font-size="11.5" font-weight="600" style="fill:var(--reject)">Same token as the target? Fails gate</text>')
svg.append(f'<text x="685" y="205" text-anchor="middle" font-size="11" {T}>94% math · 93% code · 78% chat, needs 99%+</text>')
# divider
svg.append('<line x1="20" y1="228" x2="800" y2="228" style="stroke:var(--line)"/>')
# panel B
svg.append(f'<text x="20" y="252" font-size="13" font-weight="600" {T}>B · Not built (stage 3): sample several futures, check them as one tree</text>')
svg.append('<rect x="20" y="292" width="110" height="46" rx="8" style="fill:var(--accent-bg);stroke:var(--accent)"/>')
svg.append(f'<text x="75" y="311" text-anchor="middle" font-size="11.5" {T}>target state</text>'
           f'<text x="75" y="326" text-anchor="middle" font-size="10.5" {M}>at the last token</text>')
svg.append(arrow(132, 315, 160, 315))
svg.append('<rect x="164" y="292" width="110" height="46" rx="8" style="fill:var(--ours-bg);stroke:var(--ours)"/>')
svg.append(f'<text x="219" y="311" text-anchor="middle" font-size="11.5" {T}>search network</text>'
           f'<text x="219" y="326" text-anchor="middle" font-size="10.5" {M}>predicts z, samples</text>')
branches = [("so", "x", "=", "4"), ("so", "we", "get", "4"), ("thus", "x", "=", "4")]
for i, br in enumerate(branches):
    y = 272 + i * 40
    svg.append(arrow(276, 315, 300, y + 10))
    svg.append(f'<rect x="304" y="{y}" width="22" height="20" rx="4" style="fill:{PURPLE}" fill-opacity="{0.95 - 0.2 * i:.2f}"/>'
               f'<text x="315" y="{y + 14}" text-anchor="middle" font-size="10.5" style="fill:#ffffff">z{i + 1}</text>')
    svg.append(arrow(328, y + 10, 346, y + 10))
    svg.append(f'<path d="M350,{y + 4} L368,{y - 2} L368,{y + 22} L350,{y + 16} Z" style="fill:var(--ours-bg);stroke:var(--ours)"/>')
    x = 378
    for w in br:
        ww = 18 + 7 * len(w)
        svg.append(chip(x, y, ww, w))
        x += ww + 4
svg.append(f'<text x="440" y="392" text-anchor="middle" font-size="10.5" {M}>each latent decodes into a whole continuation</text>')
svg.append(arrow(548, 312, 592, 312))
# merged tree
nodes = {"root": (612, 312), "so": (668, 288), "thus": (668, 344), "x": (722, 270), "we": (722, 306), "x2": (722, 344)}
edges = [("root", "so"), ("root", "thus"), ("so", "x"), ("so", "we"), ("thus", "x2")]
for a, b in edges:
    (x1, y1), (x2, y2) = nodes[a], nodes[b]
    svg.append(f'<line x1="{x1 + 10}" y1="{y1}" x2="{x2 - 18}" y2="{y2}" style="stroke:var(--line);stroke-width:1.5"/>')
svg.append(f'<circle cx="612" cy="312" r="6" style="fill:var(--accent)"/>')
for k, lab in [("so", "so"), ("thus", "thus"), ("x", "x = 4"), ("we", "we get 4"), ("x2", "x = 4")]:
    x, y = nodes[k]
    w = 18 + 7 * len(lab)
    svg.append(chip(x - 18, y - 10, w, lab))
svg.append(f'<text x="700" y="378" text-anchor="middle" font-size="11" {T}>one draft tree, checked in one pass</text>')
svg.append(f'<text x="700" y="392" text-anchor="middle" font-size="10.5" {M}>output stays exactly the target&#8217;s</text>')
svg.append("</svg>")
SVG = "".join(svg)

fig = ('<figure class="fig1" style="margin:0"><div class="figscroll" style="border:0;padding:0">' + SVG + '</div>'
       '<figcaption>The design under test. A: the target&#8217;s hidden states for the next 8 positions are squeezed into '
       'one latent and rebuilt, and the rebuilt final-layer states must make the target&#8217;s own output layer pick the '
       'same tokens. This is the part that failed. B: the part it was meant to enable, not built yet. Sampling several '
       'latents gives several whole continuations, and shared beginnings merge into one tree, unlike per-slot trees that '
       'mix pieces of different continuations.</figcaption></figure>')

t = open("template.html", encoding="utf-8").read()
a = t.index('<div class="live" id="aeGate">')
st = t.index("</div>", t.index('<div class="state">', a)) + len("</div>")
if 'id="aeFig"' in t:
    i = t.index('<figure class="fig1" style="margin:0" id="aeFig">')
    j = t.index("</figure>", i) + len("</figure>")
    t = t[:i] + t[j:]
fig = fig.replace('<figure class="fig1" style="margin:0">', '<figure class="fig1" style="margin:0" id="aeFig">', 1)
t = t[:st] + "\n      " + fig + t[st:]
open("template.html", "w", encoding="utf-8").write(t)
open("index.html", "w", encoding="utf-8").write(t.replace("__SUMMARY__", open("summary.json", encoding="utf-8").read()))
print("ok", t.count('id="aeFig"'))
