"""Builds Figure 1 (standard tree drafting vs target-guided branched cache) as inline SVG."""
W, H = 900, 560
COLS = [430, 545, 660, 775]  # x of draft-tree depth columns
CW, CH = 90, 24
out = []
A = out.append


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def chip(x, y, label, kind, w=CW):
    styles = {
        "acc": "fill:var(--ours-bg);stroke:var(--ours);stroke-width:1.6",
        "rej": "fill:none;stroke:var(--muted);stroke-width:1;stroke-dasharray:3 3;opacity:.75",
        "keep": "fill:var(--accent-bg);stroke:var(--accent);stroke-width:1.4;stroke-dasharray:1.5 2.5",
        "prune": "fill:none;stroke:var(--muted);stroke-width:1;stroke-dasharray:2 4;opacity:.55",
        "ctx": "fill:var(--panel);stroke:currentColor;stroke-width:1",
        "plain": "fill:var(--panel);stroke:var(--muted);stroke-width:1",
    }
    tcol = {"acc": "var(--ours)", "rej": "var(--muted)", "keep": "var(--accent)", "prune": "var(--muted)",
            "ctx": "currentColor", "plain": "currentColor"}[kind]
    op = ' opacity=".6"' if kind == "prune" else ""
    A(f'<rect x="{x}" y="{y - CH / 2}" width="{w}" height="{CH}" rx="6" style="{styles[kind]}"/>')
    A(f'<text x="{x + w / 2}" y="{y + 4.5}" text-anchor="middle" font-family="var(--mono)" font-size="12.5" '
      f'style="fill:{tcol}"{op}>{esc(label)}</text>')
    if kind == "rej":
        A(f'<text x="{x + w - 7}" y="{y - 4}" text-anchor="middle" font-size="11" style="fill:var(--reject)">×</text>')


def edge(x1, y1, x2, y2, kind):
    st = {"acc": "stroke:var(--ours);stroke-width:2.4",
          "rej": "stroke:var(--muted);stroke-width:1;stroke-dasharray:3 3;opacity:.7",
          "keep": "stroke:var(--accent);stroke-width:1.4;stroke-dasharray:1.5 2.5",
          "prune": "stroke:var(--muted);stroke-width:1;stroke-dasharray:2 4;opacity:.5",
          "plain": "stroke:var(--muted);stroke-width:1.2"}[kind]
    A(f'<path d="M{x1} {y1} C{x1 + 22} {y1},{x2 - 22} {y2},{x2} {y2}" style="fill:none;{st}"/>')


def tree(nodes, root):
    """nodes: list of (id, parent_id, depth, y, label, kind). root: (x_right, y)."""
    pos = {"root": root}
    for nid, par, d, y, label, kind in nodes:
        x = COLS[d]
        px, py = pos[par]
        edge(px, py, x, y, kind if kind != "acc" or par == "root" or nodes_kind[par] == "acc" else "plain")
        pos[nid] = (x + CW, y)
    for nid, par, d, y, label, kind in nodes:
        chip(COLS[d], y, label, kind)


A(f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Two draft trees with the same nine-node budget. '
  'In standard drafting the drafter grows branches by its own confidence and discards everything off the verified path. '
  'In the proposed version a gist read from the target\'s middle layers prunes an unlikely branch before drafting, '
  'spends the budget deeper on the likely one, and keeps an unchosen branch in the cache." '
  'style="width:100%;height:auto;color:var(--fg);font-family:var(--sans)">')
A('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
  '<path d="M1 1L9 5L1 9" style="fill:none;stroke:currentColor;stroke-width:1.6"/></marker>'
  '<marker id="ahg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
  '<path d="M1 1L9 5L1 9" style="fill:none;stroke:var(--self);stroke-width:1.6"/></marker></defs>')

# ---------- panel (a) ----------
A('<text x="20" y="20" font-size="14" font-weight="600" style="fill:currentColor">(a) Standard tree drafting</text>')
A('<rect x="30" y="100" width="150" height="50" rx="8" style="fill:var(--panel);stroke:currentColor;stroke-width:1"/>')
A('<text x="105" y="121" text-anchor="middle" font-size="13" font-weight="600" style="fill:currentColor">0.5B drafter</text>')
A('<text x="105" y="139" text-anchor="middle" font-size="11.5" style="fill:var(--muted)">its own confidence</text>')
A('<line x1="182" y1="125" x2="266" y2="125" stroke="currentColor" stroke-width="1.3" marker-end="url(#ah)"/>')
A('<text x="224" y="116" text-anchor="middle" font-size="11" style="fill:var(--muted)">grows tree</text>')
chip(270, 125, "…in May she sold", "ctx", w=130)
nodes_kind = {}
na = [("a24", "root", 0, 70, "24", "acc"), ("a48", "root", 0, 125, "48", "rej"), ("ahalf", "root", 0, 180, "half", "rej"),
      ("aclips", "a24", 1, 52, " clips", "acc"), ("adot", "a24", 1, 92, " .", "rej"), ("aslash", "a48", 1, 130, " /", "rej"),
      ("aof", "ahalf", 1, 180, " of", "rej"), ("adot2", "aclips", 2, 40, " .", "acc"), ("ain", "aclips", 2, 78, " in", "rej")]
nodes_kind = {n[0]: n[5] for n in na}
tree(na, (400, 125))
A('<text x="660" y="140" font-size="11.5" style="fill:var(--muted)">off-path branches are</text>')
A('<text x="660" y="155" font-size="11.5" style="fill:var(--muted)">thrown away after each round</text>')
A('<text x="430" y="222" font-size="11.5" style="fill:var(--muted)">9 drafted nodes · depth stops at 3</text>')

A('<line x1="20" y1="245" x2="880" y2="245" style="stroke:var(--line);stroke-width:1"/>')

# ---------- panel (b) ----------
A('<text x="20" y="272" font-size="14" font-weight="600" style="fill:currentColor">(b) Target-guided tree with a branched cache</text>')
A('<text x="90" y="300" text-anchor="middle" font-size="12" font-weight="600" style="fill:currentColor">9B target</text>')
for i in range(6):
    y = 310 + i * 24
    hot = i in (2, 3)
    st = "fill:var(--self-bg);stroke:var(--self);stroke-width:1.4" if hot else "fill:var(--panel);stroke:var(--muted);stroke-width:1"
    A(f'<rect x="30" y="{y}" width="120" height="18" rx="4" style="{st}"/>')
A('<text x="90" y="471" text-anchor="middle" font-size="11" style="fill:var(--self)">middle layers carry</text>')
A('<text x="90" y="485" text-anchor="middle" font-size="11" style="fill:var(--self)">the answer\'s gist</text>')
heights = [14, 30, 9, 22, 34, 12, 26, 18]
for i, h in enumerate(heights):
    A(f'<rect x="{178 + i * 8}" y="{398 - h}" width="6" height="{h}" rx="1" style="fill:var(--self)"/>')
A('<text x="208" y="414" text-anchor="middle" font-size="11.5" font-style="italic" style="fill:var(--self)">gist z</text>')
A('<line x1="152" y1="367" x2="172" y2="378" style="stroke:var(--self);stroke-width:1.4" marker-end="url(#ahg)"/>')
A('<line x1="244" y1="382" x2="266" y2="382" style="stroke:var(--self);stroke-width:1.4" marker-end="url(#ahg)"/>')
A('<text x="255" y="352" text-anchor="middle" font-size="11" style="fill:var(--self)">steers</text>')
A('<text x="255" y="366" text-anchor="middle" font-size="11" style="fill:var(--self)">growth</text>')
chip(270, 382, "…in May she sold", "ctx", w=130)
nb = [("b24", "root", 0, 352, "24", "acc"), ("b48", "root", 0, 425, "48", "keep"), ("bhalf", "root", 0, 470, "half", "prune"),
      ("bclips", "b24", 1, 332, " clips", "acc"), ("bdot", "b24", 1, 375, " .", "rej"), ("bslash", "b48", 1, 425, " /", "keep"),
      ("bdot2", "bclips", 2, 312, " .", "acc"), ("bin", "bclips", 2, 348, " in", "rej"),
      ("balt", "bdot2", 3, 296, " Altogether", "acc"), ("bso", "bdot2", 3, 330, " So", "rej")]
nodes_kind = {n[0]: n[5] for n in nb}
tree(nb, (400, 382))
A('<text x="660" y="420" font-size="11.5" style="fill:var(--accent)">kept in the KV cache,</text>')
A('<text x="660" y="435" font-size="11.5" style="fill:var(--accent)">reused if a later round needs it</text>')
A('<text x="530" y="474" font-size="11.5" style="fill:var(--muted)">never drafted: the gist says a number comes next</text>')
A('<text x="430" y="505" font-size="11.5" style="fill:var(--muted)">same 9 drafted nodes · the budget goes one step deeper on the likely branch</text>')

# ---------- legend ----------
ly = 538
A(f'<line x1="30" y1="{ly}" x2="54" y2="{ly}" style="stroke:var(--ours);stroke-width:2.4"/>')
A(f'<text x="60" y="{ly + 4}" font-size="11.5" style="fill:currentColor">verified path</text>')
A(f'<line x1="160" y1="{ly}" x2="184" y2="{ly}" style="stroke:var(--muted);stroke-width:1;stroke-dasharray:3 3"/>')
A(f'<text x="190" y="{ly + 4}" font-size="11.5" style="fill:currentColor">rejected</text>')
A(f'<line x1="265" y1="{ly}" x2="289" y2="{ly}" style="stroke:var(--muted);stroke-width:1;stroke-dasharray:2 4;opacity:.55"/>')
A(f'<text x="295" y="{ly + 4}" font-size="11.5" style="fill:currentColor">pruned before drafting</text>')
A(f'<line x1="440" y1="{ly}" x2="464" y2="{ly}" style="stroke:var(--accent);stroke-width:1.4;stroke-dasharray:1.5 2.5"/>')
A(f'<text x="470" y="{ly + 4}" font-size="11.5" style="fill:currentColor">kept in cache across rounds</text>')
A(f'<rect x="640" y="{ly - 6}" width="12" height="12" rx="2" style="fill:var(--self)"/>')
A(f'<text x="658" y="{ly + 4}" font-size="11.5" style="fill:currentColor">read from the target</text>')
A('</svg>')

svg = "\n".join(out)
fig = ('<figure class="fig1">\n<div class="figscroll">' + svg + '</div>\n<figcaption><b>Figure 1.</b> The proposal in one picture. '
       'Both trees spend the same 9-node draft budget on the GSM8K-style prefix "…in May she sold". In (a) the drafter grows '
       'branches by its own confidence, and everything off the verified path is discarded. In (b) a gist read from the target\'s '
       'middle layers rules out a branch before any drafting, spends the saved budget one step deeper on the likely branch, and '
       'keeps an unchosen branch in the KV cache for later rounds. This is an illustration of the idea, not a measured result.'
       '</figcaption>\n</figure>')
open("fig1.html", "w", encoding="utf-8").write(fig)
print(len(fig))
