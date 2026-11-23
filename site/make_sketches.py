"""Three design sketches: tapped layers -> plan latent -> block-diffusion drafter -> draft tree."""
import math


def svg_open(A, W, H, label):
    A(f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{label}" style="width:100%;height:auto;color:var(--fg);font-family:var(--sans)">')
    A('<defs><marker id="sk_o" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
      '<path d="M1 1L9 5L1 9" style="fill:none;stroke:var(--self);stroke-width:1.6"/></marker>'
      '<marker id="sk_t" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
      '<path d="M1 1L9 5L1 9" style="fill:none;stroke:var(--ours);stroke-width:1.6"/></marker>'
      '<marker id="sk_k" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
      '<path d="M1 1L9 5L1 9" style="fill:none;stroke:currentColor;stroke-width:1.6"/></marker></defs>')


MINI = {(): (0.0, 0.5), (0,): (0.33, 0.25), (1,): (0.33, 0.75), (0, 0): (0.66, 0.1), (0, 1): (0.66, 0.4),
        (1, 0): (0.66, 0.65), (1, 1): (0.66, 0.9), (0, 1, 0): (1.0, 0.35), (0, 0, 0): (1.0, 0.05)}
MINI_ACC = {(), (0,), (0, 1), (0, 1, 0)}


def stack(A, x0, ytop, NL, tapped, gap=22, pw=150, skew=36, ph=20, colors=None):
    """Layer planes back to front. Tapped planes hold a KV tree. Returns {layer: (x_right, y_mid)}."""
    out = {}
    for L in range(NL - 1, -1, -1):
        y0 = ytop + L * gap
        tap = L in tapped
        col = (colors or {}).get(L, "var(--self)")
        fill = "var(--self-bg)" if tap else "var(--panel)"
        A(f'<polygon points="{x0},{y0} {x0 + pw},{y0} {x0 + pw + skew},{y0 - ph} {x0 + skew},{y0 - ph}" '
          f'style="fill:{fill};stroke:{col if tap else "var(--line)"};stroke-width:{1.1 if tap else 1}"/>')
        if tap:
            def tp(u, v):
                return (x0 + 12 + u * (pw - 24) + (1 - v) * skew * 0.9, y0 - 3 - (1 - v) * (ph - 6))
            for n, (u, v) in MINI.items():
                if n != ():
                    a, b = tp(*MINI[n[:-1]]), tp(u, v)
                    st = "var(--ours)" if n in MINI_ACC else col
                    A(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" style="stroke:{st};stroke-width:{1.3 if n in MINI_ACC else 0.8}"/>')
            for n, (u, v) in MINI.items():
                p = tp(u, v)
                f = "var(--ours)" if n in MINI_ACC else col
                A(f'<rect x="{p[0] - 2.2:.1f}" y="{p[1] - 1.8:.1f}" width="4.4" height="3.6" rx="0.8" style="fill:{f}"/>')
        out[L] = (x0 + pw + skew * 0.5, y0 - ph / 2)
    return out


def latent(A, x, y, heights, color="var(--self)", label="plan z", w=6):
    for i, h in enumerate(heights):
        A(f'<rect x="{x + i * (w + 2)}" y="{y - h}" width="{w}" height="{h}" rx="1" style="fill:{color}"/>')
    A(f'<text x="{x + len(heights) * (w + 2) / 2 - 1}" y="{y + 15}" text-anchor="middle" font-size="11.5" font-style="italic" style="fill:{color}">{label}</text>')


def diffusion(A, x, y, slots=12, rows=3, s=15, labels=True, accent="var(--ours)"):
    """Block-diffusion drafter: masked block -> partially denoised -> draft block."""
    filled = [[0] * slots, [1 if (i * 7) % 3 == 0 else 0 for i in range(slots)], [1] * slots]
    for r in range(rows):
        yy = y + r * (s + 8)
        for i in range(slots):
            xx = x + i * (s + 2)
            if filled[r][i]:
                A(f'<rect x="{xx}" y="{yy}" width="{s}" height="{s}" rx="3" style="fill:{accent};opacity:{0.55 + 0.45 * (r == rows - 1)}"/>')
            else:
                A(f'<rect x="{xx}" y="{yy}" width="{s}" height="{s}" rx="3" style="fill:none;stroke:var(--muted);stroke-width:1;stroke-dasharray:2 2"/>')
        if labels:
            lab = ["masked block", "denoising", "draft block"][r]
            A(f'<text x="{x + slots * (s + 2) + 8}" y="{yy + s - 3}" font-size="10.5" style="fill:var(--muted)">{lab}</text>')
    return (x + slots * (s + 2) / 2, y + rows * (s + 8))


def token_tree(A, x, y, w=110, h=70, color="var(--ours)"):
    pts = {(): (x, y), (0,): (x + w * .33, y - h * .3), (1,): (x + w * .33, y + h * .3),
           (0, 0): (x + w * .66, y - h * .45), (0, 1): (x + w * .66, y - h * .1), (1, 0): (x + w * .66, y + h * .3),
           (0, 1, 0): (x + w, y - h * .15), (0, 0, 0): (x + w, y - h * .5), (1, 0, 0): (x + w, y + h * .35)}
    acc = {(), (0,), (0, 1), (0, 1, 0)}
    for n, (px, py) in pts.items():
        if n != ():
            qx, qy = pts[n[:-1]]
            mx = (qx + px) / 2
            st = f"stroke:{color};stroke-width:2.2" if n in acc else "stroke:var(--muted);stroke-width:1;opacity:.6"
            A(f'<path d="M{qx:.1f} {qy:.1f} C{mx:.1f} {qy:.1f},{mx:.1f} {py:.1f},{px:.1f} {py:.1f}" style="fill:none;{st}"/>')
    for n, (px, py) in pts.items():
        A(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{3.6 if n in acc else 2.6}" style="fill:{color if n in acc else "var(--muted)"}"/>')


def caption_fig(svg, num, text):
    return (f'<figure class="fig1">\n<div class="figscroll">{svg}</div>\n<figcaption><b>Sketch {num}.</b> {text}</figcaption>\n</figure>')


figs = []

# ---------------- Sketch A: tapped layers merge into one plan latent ----------------
out = []; A = out.append
svg_open(A, 960, 330, "Three tapped layers of the target, each with its KV-cache tree, merge into one plan latent that conditions a block-diffusion drafter, which produces a draft tree.")
A('<text x="20" y="24" font-size="13.5" font-weight="600" style="fill:currentColor">A. Tapped layers merge into one plan latent</text>')
A('<text x="40" y="58" font-size="11.5" font-weight="600" style="fill:currentColor">9B target, 8 of 32 layers shown</text>')
taps = stack(A, 40, 84, 8, {1, 4, 6})
for L in (1, 4, 6):
    A(f'<text x="{taps[L][0] + 30}" y="{taps[L][1] + 4:.0f}" font-size="10.5" style="fill:var(--self)">tap</text>')
lx, ly = 390, 190
for L, (x, y) in taps.items():
    if L in (1, 4, 6):
        A(f'<path d="M{x + 52} {y:.1f} C{x + 100} {y:.1f},{lx - 60} {ly - 20},{lx - 6} {ly - 20}" style="fill:none;stroke:var(--self);stroke-width:1.3;stroke-dasharray:3 3"/>')
latent(A, lx, ly - 2, [12, 26, 8, 20, 30, 11, 22, 16])
A(f'<line x1="{lx + 70}" y1="{ly - 20}" x2="{lx + 112}" y2="{ly - 20}" style="stroke:var(--self);stroke-width:1.4" marker-end="url(#sk_o)"/>')
A(f'<text x="{lx + 91}" y="{ly - 30}" text-anchor="middle" font-size="10.5" style="fill:var(--self)">conditions</text>')
A('<text x="512" y="110" font-size="11.5" font-weight="600" style="fill:currentColor">Block-diffusion drafter</text>')
cx, cy = diffusion(A, 512, 124)
A(f'<line x1="{cx}" y1="{cy + 2}" x2="{cx}" y2="{cy + 26}" style="stroke:var(--ours);stroke-width:1.4" marker-end="url(#sk_t)"/>')
token_tree(A, cx - 50, cy + 70)
A(f'<text x="{cx + 80}" y="{cy + 74}" font-size="10.5" style="fill:var(--muted)">draft tree, checked by the target</text>')
figs.append(caption_fig("\n".join(out), "A",
    "A few layers of the target are tapped, as DFlash does with 8 of 32 layers and EAGLE-3 with three. Each tapped layer holds the KV-cache "
    "tree of the checked drafts. Their states are pooled into one plan latent, which conditions a block-diffusion drafter that denoises a whole "
    "block of tokens in one or two steps. The finished block seeds the next draft tree."))

# ---------------- Sketch B: each tapped layer steers a different part of the block ----------------
out = []; A = out.append
svg_open(A, 960, 330, "Each tapped layer conditions a different stretch of the diffusion block: early layers the next tokens, later layers tokens further ahead.")
A('<text x="20" y="24" font-size="13.5" font-weight="600" style="fill:currentColor">B. Each tapped layer steers a different stretch of the block</text>')
cols3 = {1: "var(--accent)", 4: "var(--self)", 6: "var(--reject)"}
taps = stack(A, 40, 84, 8, {1, 4, 6}, colors=cols3)
bx, by, s = 470, 150, 22
spans = {1: (0, 4), 4: (4, 9), 6: (9, 14)}
for i in range(14):
    A(f'<rect x="{bx + i * (s + 3)}" y="{by}" width="{s}" height="{s}" rx="4" style="fill:var(--ours-bg);stroke:var(--ours);stroke-width:1"/>')
for L, (a, b) in spans.items():
    x, y = taps[L]
    tx = bx + ((a + b) / 2) * (s + 3) - 2
    col = cols3[L]
    A(f'<path d="M{x + 52} {y:.1f} C{x + 170} {y:.1f},{x + 170} {by - 46},{bx - 20} {by - 46} '
      f'S{tx} {by - 46},{tx} {by - 6}" style="fill:none;stroke:{col};stroke-width:1.4;stroke-dasharray:3 3" marker-end="url(#sk_k)"/>')
    A(f'<rect x="{bx + a * (s + 3) - 2}" y="{by + s + 6}" width="{(b - a) * (s + 3) - 1}" height="4" rx="2" style="fill:{col}"/>')
A(f'<text x="{bx + 2 * (s + 3)}" y="{by + s + 26}" text-anchor="middle" font-size="10.5" style="fill:var(--accent)">next few tokens</text>')
A(f'<text x="{bx + 6.5 * (s + 3)}" y="{by + s + 26}" text-anchor="middle" font-size="10.5" style="fill:var(--self)">mid-block</text>')
A(f'<text x="{bx + 11.5 * (s + 3)}" y="{by + s + 26}" text-anchor="middle" font-size="10.5" style="fill:var(--reject)">far end of the block</text>')
A(f'<text x="{bx}" y="{by - 84}" font-size="11.5" font-weight="600" style="fill:currentColor">Diffusion block, 14 positions</text>')
A(f'<text x="{bx}" y="{by - 70}" font-size="10.5" style="fill:var(--muted)">each tapped layer is cross-attended by one stretch</text>')
A(f'<text x="{bx}" y="{by + 80}" font-size="10.5" style="fill:var(--muted)">Our probes found exact tokens fade after about 4 positions while the gist lasts longer,</text>')
A(f'<text x="{bx}" y="{by + 95}" font-size="10.5" style="fill:var(--muted)">which is where block drafters lose acceptance. This variant tests whether deeper layers help there.</text>')
figs.append(caption_fig("\n".join(out), "B",
    "Instead of one pooled latent, each tapped layer conditions its own stretch of the diffusion block. Lower layers, which carry the next "
    "tokens, steer the start of the block. Middle and late layers, which carry more of the gist, steer the far end, where our block drafters "
    "lost most of their acceptance (0.90 at position 1, 0.53 at position 8)."))

# ---------------- Sketch C: the latent itself branches ----------------
out = []; A = out.append
svg_open(A, 960, 380, "The plan latent branches in proportion to the target's uncertainty. Each latent branch runs its own diffusion block and token subtree, giving a tree on the target side and a tree on the drafter side.")
A('<text x="20" y="24" font-size="13.5" font-weight="600" style="fill:currentColor">C. The latent branches too: a tree on both sides</text>')
taps = stack(A, 40, 132, 6, {2, 3})
A('<text x="40" y="62" font-size="11.5" font-weight="600" style="fill:currentColor">9B target</text>')
z0x, z0y = 330, 232
x2, y2 = taps[3]
A(f'<path d="M{x2 + 52} {y2:.1f} C{x2 + 90} {y2:.1f},{z0x - 40} {z0y - 14},{z0x - 6} {z0y - 14}" style="fill:none;stroke:var(--self);stroke-width:1.3;stroke-dasharray:3 3"/>')
latent(A, z0x, z0y, [10, 22, 7, 18, 26, 9, 19, 13], label="gist", w=5)
branches = [(0.8, 160, [12, 24, 9, 20, 28, 10, 21, 15], "about 80%: a number comes next", 4),
            (0.2, 315, [20, 8, 26, 12, 7, 22, 10, 18], "about 20%: an explanation first", 2)]
for p, by_, hs, txt, nslots in branches:
    A(f'<path d="M{z0x + 62} {z0y - 14} C{z0x + 110} {z0y - 14},{z0x + 100} {by_ - 14},{z0x + 140} {by_ - 14}" '
      f'style="fill:none;stroke:var(--self);stroke-width:{1 + 3 * p:.1f};opacity:.85"/>')
    latent(A, z0x + 150, by_, hs, label=f"z, p≈{p}", w=5)
    A(f'<line x1="{z0x + 212}" y1="{by_ - 14}" x2="{z0x + 248}" y2="{by_ - 14}" style="stroke:var(--self);stroke-width:1.3" marker-end="url(#sk_o)"/>')
    slots = 12 if p > 0.5 else 6
    cx, cy = diffusion(A, z0x + 256, by_ - 54, slots=slots, rows=3, s=12, labels=False)
    token_tree(A, z0x + 256 + slots * 14 + 22, by_ - 22, w=90 if p > 0.5 else 55, h=56 if p > 0.5 else 34)
    A(f'<text x="{z0x + 150}" y="{by_ + 34}" font-size="10.5" style="fill:var(--muted)">{txt}</text>')
A('<text x="610" y="62" font-size="10.5" style="fill:var(--muted)">branch width follows the target&#8217;s own probabilities:</text>')
A('<text x="610" y="76" font-size="10.5" style="fill:var(--muted)">the likely plan gets a bigger block and a wider token tree</text>')
figs.append(caption_fig("\n".join(out), "C",
    "When the target is unsure where it is going, the plan latent itself splits, with each branch weighted by how likely the target thinks it is. "
    "Each latent branch runs its own diffusion block and token subtree, so the likely plan gets more of the draft budget. This is the "
    "&ldquo;tree on both sides&rdquo; idea: a small tree of plans on the target side and token trees on the drafter side. It matters most when sampling, "
    "since at temperature 0 the target has only one future."))

# ---------------- Sketch D: latent diffusion, then decode ----------------
out = []; A = out.append
svg_open(A, 960, 340, "A plan latent starts as noise and is denoised in a few steps, conditioned on the target's tapped layers. The clean latent is decoded into a block of tokens and a draft tree.")
A('<text x="20" y="24" font-size="13.5" font-weight="600" style="fill:currentColor">D. Latent diffusion: denoise the plan, then decode it</text>')
A('<text x="40" y="62" font-size="11.5" font-weight="600" style="fill:currentColor">9B target</text>')
taps = stack(A, 40, 108, 6, {2, 3}, gap=28, pw=130, skew=32, ph=20)
noisy = [17, 6, 24, 11, 20, 4, 15, 22]
mid = [13, 18, 12, 17, 24, 8, 18, 16]
clean = [12, 24, 9, 20, 28, 10, 21, 15]
ly = 212
xs = [330, 430, 530]
def bars(x, hs, mode):
    for i, h in enumerate(hs):
        if mode == "noise":
            st = "fill:var(--muted);opacity:.55"
        elif mode == "mid":
            st = "fill:var(--self);opacity:.55"
        else:
            st = "fill:var(--self)"
        A(f'<rect x="{x + i * 8}" y="{ly - h}" width="6" height="{h}" rx="1" style="{st}"/>')
bars(xs[0], noisy, "noise"); bars(xs[1], mid, "mid"); bars(xs[2], clean, "clean")
for x, lab in zip(xs, ["noise", "denoising", "clean plan z"]):
    A(f'<text x="{x + 31}" y="{ly + 16}" text-anchor="middle" font-size="11" style="fill:{"var(--self)" if lab != "noise" else "var(--muted)"}">{lab}</text>')
for a, b in ((xs[0] + 66, xs[1] - 6), (xs[1] + 66, xs[2] - 6)):
    A(f'<line x1="{a}" y1="{ly - 14}" x2="{b}" y2="{ly - 14}" style="stroke:var(--self);stroke-width:1.3" marker-end="url(#sk_o)"/>')
A(f'<text x="{xs[1] + 31}" y="{ly + 44}" text-anchor="middle" font-size="10.5" style="fill:var(--muted)">1–2 denoising steps, in a space far smaller than tokens</text>')
x3, y3 = taps[3]
for tx in (xs[0] + 31, xs[1] + 31, xs[2] + 31):
    A(f'<path d="M{x3 + 40} {y3:.0f} C{x3 + 90} {y3 - 70:.0f},{tx} {ly - 100},{tx} {ly - 36}" style="fill:none;stroke:var(--self);stroke-width:1;stroke-dasharray:3 3;opacity:.8" marker-end="url(#sk_o)"/>')
A(f'<text x="{xs[1] + 31}" y="{ly - 108}" text-anchor="middle" font-size="10.5" style="fill:var(--self)">every step is conditioned on the tapped layers</text>')
# decoder
dx = 640
A(f'<polygon points="{dx},{ly - 44} {dx + 70},{ly - 58} {dx + 70},{ly + 30} {dx},{ly + 16}" style="fill:var(--panel);stroke:currentColor;stroke-width:1.1"/>')
A(f'<text x="{dx + 35}" y="{ly - 10}" text-anchor="middle" font-size="11.5" font-weight="600" style="fill:currentColor">decoder</text>')
A(f'<line x1="{xs[2] + 68}" y1="{ly - 14}" x2="{dx - 6}" y2="{ly - 14}" style="stroke:var(--self);stroke-width:1.3" marker-end="url(#sk_o)"/>')
for i in range(8):
    A(f'<rect x="{dx + 92 + i * 16}" y="{ly - 70}" width="13" height="13" rx="3" style="fill:var(--ours)"/>')
A(f'<text x="{dx + 92}" y="{ly - 78}" font-size="10.5" style="fill:var(--muted)">a block of tokens</text>')
A(f'<line x1="{dx + 72}" y1="{ly - 30}" x2="{dx + 88}" y2="{ly - 58}" style="stroke:var(--ours);stroke-width:1.3" marker-end="url(#sk_t)"/>')
A(f'<line x1="{dx + 72}" y1="{ly + 4}" x2="{dx + 88}" y2="{ly + 26}" style="stroke:var(--ours);stroke-width:1.3" marker-end="url(#sk_t)"/>')
token_tree(A, dx + 98, ly + 46, w=120, h=60)
A(f'<text x="{dx + 98}" y="{ly + 96}" font-size="10.5" style="fill:var(--muted)">or a tree, if the decoder keeps</text>')
A(f'<text x="{dx + 98}" y="{ly + 110}" font-size="10.5" style="fill:var(--muted)">several candidates per position</text>')
figs.append(caption_fig("\n".join(out), "D",
    "Latent diffusion in the strict sense. The plan starts as noise and is denoised in a continuous latent space, guided at every step by the "
    "target's tapped layers. Only the clean latent is decoded into tokens, either as one block or as a tree of candidates. ChainFlow (2611.08669) "
    "does a flow-matching version of this, one compressed latent per future token, decoded through the target's output layer. The open variant "
    "is one latent standing for a much longer stretch, the gist, decoded into a draft tree."))

# ---------------- Sketch E: coarse-to-fine decoder, each level steered by a different layer ----------------
out = []; A = out.append
svg_open(A, 960, 350, "The clean plan latent is decoded coarse to fine: one code for the whole block, then four segment codes, then sixteen tokens. "
         "Each level is steered by a different tapped layer of the target. A button swaps which layer steers which level.")
A('<text x="20" y="24" font-size="13.5" font-weight="600" style="fill:currentColor">E. Coarse-to-fine decoding, one tapped layer per scale</text>')
A('<text x="40" y="62" font-size="11.5" font-weight="600" style="fill:currentColor">9B target</text>')
ecol = {1: "var(--accent)", 3: "var(--self)", 5: "var(--reject)"}
taps = stack(A, 40, 104, 6, {1, 3, 5}, gap=34, pw=130, skew=32, ph=22, colors=ecol)
names = {1: "shallow", 3: "middle", 5: "deep"}
for L, (x, y) in taps.items():
    if L in names:
        A(f'<text x="{x + 28}" y="{y + 4:.0f}" font-size="10.5" style="fill:{ecol[L]}">{names[L]}</text>')
zx, zy = 300, 214
latent(A, zx, zy, [12, 24, 9, 20, 28, 10, 21, 15], label="clean plan z", w=6)
lx0, lw = 520, 330
levels = {"coarse": 104, "mid": 170, "fine": 236}
A(f'<rect x="{lx0}" y="{levels["coarse"]}" width="{lw}" height="18" rx="5" style="fill:var(--ours);opacity:.35"/>')
for k in range(4):
    A(f'<rect x="{lx0 + k * (lw / 4) + 2:.0f}" y="{levels["mid"]}" width="{lw / 4 - 4:.0f}" height="18" rx="5" style="fill:var(--ours);opacity:.6"/>')
for k in range(16):
    A(f'<rect x="{lx0 + k * (lw / 16) + 1.5:.1f}" y="{levels["fine"]}" width="{lw / 16 - 3:.1f}" height="18" rx="3" style="fill:var(--ours)"/>')
for lab, y in levels.items():
    A(f'<text x="{lx0 + lw + 10}" y="{y + 13}" font-size="10.5" style="fill:var(--muted)">{ {"coarse": "whole block", "mid": "4 segments", "fine": "16 tokens"}[lab] }</text>')
for a, b in (("coarse", "mid"), ("mid", "fine")):
    A(f'<line x1="{lx0 + lw / 2}" y1="{levels[a] + 20}" x2="{lx0 + lw / 2}" y2="{levels[b] - 4}" style="stroke:var(--ours);stroke-width:1.3" marker-end="url(#sk_t)"/>')
A(f'<path d="M{zx + 66} {zy - 18} C{zx + 140} {zy - 18},{lx0 - 90} {levels["coarse"] + 9},{lx0 - 6} {levels["coarse"] + 9}" style="fill:none;stroke:var(--self);stroke-width:1.4" marker-end="url(#sk_o)"/>')
A(f'<text x="{lx0}" y="{levels["coarse"] - 12}" font-size="11.5" font-weight="600" style="fill:currentColor">Decoder, coarse to fine</text>')
def links(mapping, cls, hidden):
    A(f'<g class="{cls}"{" style=&quot;display:none&quot;" if False else ""}{" hidden" if hidden else ""}>')
    for L, lvl in mapping.items():
        x, y = taps[L]
        ty = levels[lvl] + 9
        A(f'<path d="M{x + 60} {y:.0f} C{x + 200} {y:.0f},{lx0 - 140} {ty},{lx0 - 6} {ty}" style="fill:none;stroke:{ecol[L]};stroke-width:1.3;stroke-dasharray:3 3" marker-end="url(#sk_k)"/>')
    A('</g>')
links({1: "coarse", 3: "mid", 5: "fine"}, "e-cnn", False)
links({5: "coarse", 3: "mid", 1: "fine"}, "e-probe", True)
A(f'<text x="{lx0}" y="{levels["fine"] + 50}" font-size="10.5" style="fill:var(--muted)" class="e-cnn">CNN-style: shallow layers steer the whole block, deep layers the details</text>')
A(f'<text x="{lx0}" y="{levels["fine"] + 50}" font-size="10.5" style="fill:var(--muted)" class="e-probe" hidden>Probe-style: deep layers carry the gist, shallow layers the next tokens</text>')
svgE = "\n".join(out)
figE = ('<figure class="fig1" id="sketchE">\n<div class="figscroll">' + svgE + '</div>\n'
        '<div class="scrubbar"><button type="button" id="eSwap">Swap to probe-style</button></div>\n'
        '<figcaption><b>Sketch E.</b> The clean latent is decoded coarse to fine, like the receptive fields of a CNN or the upsampling path of a U-Net: '
        'one code for the whole block, then four segment codes, then sixteen tokens. Each scale cross-attends a different tapped layer. '
        'Which layer should steer which scale is an open question. The CNN intuition puts shallow layers on the whole block. Probing work on '
        'language models points the other way, with paragraph-level gist strongest at 60 to 80% depth and exact next tokens clearest earlier. '
        'The layer sweep would settle it.</figcaption>\n</figure>\n'
        '<script>(function(){const b=document.getElementById("eSwap"),f=document.getElementById("sketchE");let probe=false;'
        'b.addEventListener("click",()=>{probe=!probe;f.querySelectorAll(".e-cnn").forEach(e=>e.toggleAttribute("hidden",probe));'
        'f.querySelectorAll(".e-probe").forEach(e=>e.toggleAttribute("hidden",!probe));b.textContent=probe?"Swap to CNN-style":"Swap to probe-style";});})();</script>')
figs.append(figE)

open("sketches.html", "w", encoding="utf-8").write("\n".join(figs))
print("ok", sum(len(f) for f in figs))
