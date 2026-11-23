# Panel (a) of Figure 2: the drafter tree, the KV-cache tree inside it, and the target as a stack of layers
# whose KV caches each hold a copy of that tree. Executed inside make_fig2.py, which provides A().
A('<text x="20" y="22" font-size="14" font-weight="600" style="fill:currentColor">(a) One round: the KV-cache tree grows inside the drafter tree, and lives in every layer of the target</text>')

# ---- drafter tree with a tidy layout ----
fan = [3, 3, 2, 2]
cols = [150, 235, 320, 405, 490]
top, bottom = 70, 300
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
leaves = frontier
pos = {}
step = (bottom - top) / (len(leaves) - 1)
for i, lf in enumerate(leaves):
    pos[lf] = (cols[len(lf)], top + i * step)


def place(n):
    if n in pos:
        return pos[n][1]
    ys = [place(c) for c in kids[n]]
    pos[n] = (cols[len(n)], sum(ys) / len(ys))
    return pos[n][1]


place(())
accepted = {(), (0,), (0, 1), (0, 1, 0), (0, 1, 0, 1)}
kvtree = accepted | {(1,), (0, 0), (0, 2), (1, 0), (0, 0, 0), (0, 1, 1), (1, 0, 0), (0, 1, 0, 0), (0, 0, 0, 0), (0, 1, 1, 0)}
ry = pos[()][1]


def curve(p, n):
    x, y = p[n]
    px, py = p[n[:-1]]
    mx = (px + x) / 2
    return f'M{px:.1f} {py:.1f} C{mx:.1f} {py:.1f},{mx:.1f} {y:.1f},{x:.1f} {y:.1f}'


# verified prefix, already cached
A(f'<line x1="34" y1="{ry:.1f}" x2="{cols[0]}" y2="{ry:.1f}" style="stroke:var(--ours);stroke-width:2.6"/>')
for i in range(4):
    A(f'<rect x="{30 + i * 26}" y="{ry - 6:.1f}" width="14" height="12" rx="2" style="fill:var(--ours)"/>')
A(f'<text x="30" y="{ry - 15:.1f}" font-size="11" style="fill:var(--ours)">verified prefix</text>')
A('<text x="150" y="50" font-size="12" font-weight="600" style="fill:currentColor">Drafter tree</text>')
A('<text x="150" y="64" font-size="11" style="fill:var(--muted)">0.5–1B, cheap, mostly never checked</text>')

for n in pos:
    if n != () and n not in kvtree:
        A(f'<path d="{curve(pos, n)}" style="fill:none;stroke:var(--muted);stroke-width:1;opacity:.42"/>')
for n in kvtree:
    if n != ():
        band = "var(--ours)" if n in accepted else "var(--self)"
        A(f'<path d="{curve(pos, n)}" style="fill:none;stroke:{band};stroke-width:11;stroke-linecap:round;opacity:.14"/>')
for n in kvtree:
    if n != ():
        st = "stroke:var(--ours);stroke-width:2.6" if n in accepted else "stroke:var(--self);stroke-width:1.5"
        A(f'<path d="{curve(pos, n)}" style="fill:none;{st}"/>')
for n, (x, y) in pos.items():
    if n in accepted:
        A(f'<rect x="{x - 6}" y="{y - 5:.1f}" width="12" height="10" rx="2" style="fill:var(--ours)"/>')
    elif n in kvtree:
        A(f'<rect x="{x - 5.5}" y="{y - 4.5:.1f}" width="11" height="9" rx="2" style="fill:var(--self-bg);stroke:var(--self);stroke-width:1.4"/>')
    else:
        A(f'<circle cx="{x}" cy="{y:.1f}" r="{3 if len(n) < 4 else 2.4}" style="fill:var(--muted);opacity:.6"/>')

# ---- the target: a stack of layer planes, each holding the KV-cache tree ----
NL = 6
px0, pw, skew, ph = 600, 190, 44, 30          # plane origin x, width, horizontal skew, depth height
py_top, gap = 92, 36
hot = {2, 3}                                  # middle layers, where the gist is read
sub_nodes = [n for n in kvtree]
# map tree coordinates (depth, height) into a plane: u = depth fraction, v = height fraction
ymin = min(pos[n][1] for n in kvtree)
ymax = max(pos[n][1] for n in kvtree)


def to_plane(n, y0):
    x, y = pos[n]
    u = (x - cols[0]) / (cols[-1] - cols[0])
    v = (y - ymin) / (ymax - ymin)
    # u runs along the plane's width, v runs back into its depth
    return (px0 + 14 + u * (pw - 28) + (1 - v) * skew, y0 - 4 - (1 - v) * (ph - 8))


plane_pos = {}
for L in range(NL - 1, -1, -1):              # draw back to front
    y0 = py_top + L * gap
    fill = "var(--self-bg)" if L in hot else "var(--panel)"
    stroke = "var(--self)" if L in hot else "var(--muted)"
    A(f'<polygon points="{px0},{y0} {px0 + pw},{y0} {px0 + pw + skew},{y0 - ph} {px0 + skew},{y0 - ph}" '
      f'style="fill:{fill};stroke:{stroke};stroke-width:1;opacity:.96"/>')
    pp = {n: to_plane(n, y0) for n in kvtree}
    plane_pos[L] = pp
    for n in kvtree:
        if n != ():
            st = "stroke:var(--ours);stroke-width:1.6" if n in accepted else "stroke:var(--self);stroke-width:0.9"
            x, y = pp[n]
            qx, qy = pp[n[:-1]]
            A(f'<line x1="{qx:.1f}" y1="{qy:.1f}" x2="{x:.1f}" y2="{y:.1f}" style="{st}"/>')
    for n in kvtree:
        x, y = pp[n]
        f = "var(--ours)" if n in accepted else "var(--self)"
        A(f'<rect x="{x - 2.6:.1f}" y="{y - 2.2:.1f}" width="5.2" height="4.4" rx="1" style="fill:{f}"/>')
    label = {0: "layer 1", 2: "middle layers", 5: f"layer L"}.get(L)
    if label:
        A(f'<text x="{px0 + pw + skew + 10}" y="{y0 - ph / 2 + 4:.0f}" font-size="11" style="fill:{stroke if L in hot else "var(--muted)"}">{label}</text>')
A(f'<text x="{px0}" y="{py_top - ph - 14}" font-size="12" font-weight="600" style="fill:currentColor">9B target: K and V cached in every layer</text>')

# ---- links between the two halves ----
# the checked nodes are copied into the target's layers
A(f'<path d="M{cols[-1] + 14} 140 C 540 112, 560 {py_top - 8}, {px0 - 4} {py_top - 8}" '
  'style="fill:none;stroke:var(--self);stroke-width:1.4" marker-end="url(#f2g)"/>')
A(f'<text x="508" y="72" font-size="11" style="fill:var(--self)">checked in one pass</text>')
# the gist, read from the middle layers, reaches back into the drafter tree
gy = py_top + 2.5 * gap - ph / 2
for n in [(0,), (1,), (0, 1), (0, 0), (0, 1, 0)]:
    x, y = pos[n]
    A(f'<path d="M{px0 - 2} {gy:.0f} C{px0 - 90} {gy + 40:.0f},{x + 80:.0f} {y + 60:.0f},{x + 7:.1f} {y + 5:.1f}" '
      'style="fill:none;stroke:var(--self);stroke-width:1.1;stroke-dasharray:3 3;opacity:.85"/>')
    A(f'<circle cx="{x + 7:.1f}" cy="{y + 5:.1f}" r="1.8" style="fill:var(--self)"/>')
A(f'<text x="{px0 - 12}" y="{bottom - 6:.0f}" text-anchor="end" font-size="11" style="fill:var(--self)">gist picks which</text>')
A(f'<text x="{px0 - 12}" y="{bottom + 8:.0f}" text-anchor="end" font-size="11" style="fill:var(--self)">branches get checked</text>')
A(f'<text x="{px0 + 40}" y="{py_top + NL * gap + 4}" font-size="11" style="fill:var(--ours)">accepts one path by sampling, so the output stays exact</text>')
