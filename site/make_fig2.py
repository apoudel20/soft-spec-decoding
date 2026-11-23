"""Figure 2: cheap drafter forest, gist-selected verified subset, and the drafting/verification timeline."""
W, H = 960, 560
out = []
A = out.append

A(f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="A drafter grows a wide, cheap tree of candidate tokens. '
  'The target\'s gist ranks its paths and sends only about thirty nodes to the target, which checks them in one forward pass and accepts one path. '
  'On a timeline, the drafter keeps growing while each verification runs, and re-roots on the accepted path when the result returns." '
  'style="width:100%;height:auto;color:var(--fg);font-family:var(--sans)">')
A('<defs><marker id="f2a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
  '<path d="M1 1L9 5L1 9" style="fill:none;stroke:currentColor;stroke-width:1.6"/></marker>'
  '<marker id="f2g" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
  '<path d="M1 1L9 5L1 9" style="fill:none;stroke:var(--self);stroke-width:1.6"/></marker>'
  '<marker id="f2t" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
  '<path d="M1 1L9 5L1 9" style="fill:none;stroke:var(--ours);stroke-width:1.6"/></marker></defs>')

exec(open('panel_a.py', encoding='utf-8').read())

A('<line x1="20" y1="330" x2="940" y2="330" style="stroke:var(--line);stroke-width:1"/>')

# ---------- timeline ----------
A('<text x="20" y="358" font-size="14" font-weight="600" style="fill:currentColor">(b) Over time: drafting never waits</text>')
lx, x0, x1 = 30, 130, 930
yd, yt = 410, 480
A(f'<text x="{lx}" y="{yd + 4}" font-size="12" font-weight="600" style="fill:currentColor">Drafter</text>')
A(f'<text x="{lx}" y="{yt + 4}" font-size="12" font-weight="600" style="fill:currentColor">Target</text>')
A(f'<rect x="{x0}" y="{yd - 12}" width="{x1 - x0}" height="24" rx="6" style="fill:var(--ours-bg);stroke:var(--ours);stroke-width:1"/>')
A(f'<text x="{(x0 + x1) / 2}" y="{yd - 18}" text-anchor="middle" font-size="11" style="fill:var(--muted)">keeps growing the forest from the newest accepted prefix</text>')
vs = [(160, 330), (380, 550), (600, 770)]
for k, (a, b) in enumerate(vs, 1):
    A(f'<rect x="{a}" y="{yt - 13}" width="{b - a}" height="26" rx="6" style="fill:var(--panel);stroke:currentColor;stroke-width:1.1"/>')
    A(f'<text x="{(a + b) / 2}" y="{yt + 4}" text-anchor="middle" font-size="11.5" style="fill:currentColor">verify round {k}</text>')
    A(f'<line x1="{a + 6}" y1="{yd + 13}" x2="{a + 6}" y2="{yt - 15}" style="stroke:var(--self);stroke-width:1.3" marker-end="url(#f2g)"/>')
    A(f'<line x1="{b - 6}" y1="{yt - 15}" x2="{b - 6}" y2="{yd + 13}" style="stroke:var(--ours);stroke-width:1.3" marker-end="url(#f2t)"/>')
    A(f'<line x1="{b - 6}" y1="{yd - 12}" x2="{b - 6}" y2="{yd + 12}" style="stroke:var(--ours);stroke-width:2.4"/>')
A(f'<text x="{vs[0][0] + 12}" y="{(yd + yt) / 2 + 4}" font-size="11" style="fill:var(--self)">top ~30</text>')
A(f'<text x="{vs[0][1] - 12}" y="{(yd + yt) / 2 + 4}" text-anchor="end" font-size="11" style="fill:var(--ours)">accepted path</text>')
A(f'<text x="{x0}" y="{yt + 36}" font-size="11" style="fill:var(--muted)">Each return re-roots the forest on the accepted path, which the drafter has already grown.</text>')
A(f'<text x="{x0}" y="{yt + 51}" font-size="11" style="fill:var(--muted)">The target drops cache entries for branches it didn&#8217;t take. On one GPU the two lanes take turns, on two devices they overlap.</text>')

# legend
ly = 548
A(f'<line x1="130" y1="{ly}" x2="154" y2="{ly}" style="stroke:var(--ours);stroke-width:2.4"/>')
A(f'<text x="160" y="{ly + 4}" font-size="11.5" style="fill:currentColor">accepted path</text>')
A(f'<rect x="284" y="{ly - 4.5}" width="11" height="9" rx="2" style="fill:var(--self-bg);stroke:var(--self);stroke-width:1.4"/>')
A(f'<text x="300" y="{ly + 4}" font-size="11.5" style="fill:currentColor">KV-cache entry, chosen by the gist</text>')
A(f'<circle cx="530" cy="{ly}" r="2.4" style="fill:var(--muted);opacity:.55"/>')
A(f'<text x="540" y="{ly + 4}" font-size="11.5" style="fill:currentColor">drafted, never checked</text>')
A(f'<line x1="690" y1="{ly}" x2="714" y2="{ly}" style="stroke:var(--self);stroke-width:1.2;stroke-dasharray:3 3"/>')
A(f'<text x="720" y="{ly + 4}" font-size="11.5" style="fill:currentColor">gist, read from middle layers</text>')
A('</svg>')

fig = ('<figure class="fig1">\n<div class="figscroll">' + "\n".join(out) + '</div>\n<figcaption><b>Figure 2.</b> Two trees working together. '
       '(a) The KV-cache tree is the part of the drafter tree that the target checks. It is not one structure but one per layer: '
       'every layer of the target stores a key and value for each checked node, all filled in one pass at about the cost of one decode step. '
       'A gist read from the middle layers reaches back into the drafter tree and picks which branches get checked. '
       'The target then samples one path, so its output is unchanged. '
       '(b) The drafter never waits. Each verification result re-roots its forest on the accepted path, which it has already grown. '
       'The selection step is the part no published method does with target information. This is an illustration of the design, not a measured result.'
       '</figcaption>\n</figure>')
open("fig2.html", "w", encoding="utf-8").write(fig)
print("ok", len(fig))
