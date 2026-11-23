"""Design sketches A-E as one tabbed viewer. Each sketch is split into phases that a shared scrubber steps through.
Elements of a phase sit in <g class="ph" data-lo=.. data-hi=..>; the viewer shows a group when lo <= phase <= hi."""
import json

FIGS = []  # (key, title, phases, svg, caption)


class Svg:
    def __init__(self, W, H, label):
        self.o = []
        self.A(f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{label}" style="width:100%;height:auto;color:var(--fg);font-family:var(--sans)">')
        self.A('<defs><marker id="sk_o" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
               '<path d="M1 1L9 5L1 9" style="fill:none;stroke:var(--self);stroke-width:1.6"/></marker>'
               '<marker id="sk_t" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
               '<path d="M1 1L9 5L1 9" style="fill:none;stroke:var(--ours);stroke-width:1.6"/></marker>'
               '<marker id="sk_k" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
               '<path d="M1 1L9 5L1 9" style="fill:none;stroke:currentColor;stroke-width:1.6"/></marker></defs>')

    def A(self, s):
        self.o.append(s)

    def ph(self, lo, hi=99, extra=""):
        self.A(f'<g class="ph{(" " + extra) if extra else ""}" data-lo="{lo}" data-hi="{hi}">')

    def end(self):
        self.A('</g>')

    def done(self):
        self.A('</svg>')
        return "\n".join(self.o)


MINI = {(): (0.0, 0.5), (0,): (0.33, 0.25), (1,): (0.33, 0.75), (0, 0): (0.66, 0.1), (0, 1): (0.66, 0.4),
        (1, 0): (0.66, 0.65), (1, 1): (0.66, 0.9), (0, 1, 0): (1.0, 0.35), (0, 0, 0): (1.0, 0.05)}
MINI_ACC = {(), (0,), (0, 1), (0, 1, 0)}


def planes(S, x0, ytop, NL, gap=22, pw=150, skew=36, ph=20):
    for L in range(NL - 1, -1, -1):
        y0 = ytop + L * gap
        S.A(f'<polygon points="{x0},{y0} {x0 + pw},{y0} {x0 + pw + skew},{y0 - ph} {x0 + skew},{y0 - ph}" style="fill:var(--panel);stroke:var(--line);stroke-width:1"/>')


def tapped(S, x0, ytop, taps, gap=22, pw=150, skew=36, ph=20, colors=None):
    """Draw tapped planes (with their KV tree) over the plain ones. Returns {layer: (x_right, y_mid)}."""
    out = {}
    for L in sorted(taps, reverse=True):
        y0 = ytop + L * gap
        col = (colors or {}).get(L, "var(--self)")
        S.A(f'<polygon points="{x0},{y0} {x0 + pw},{y0} {x0 + pw + skew},{y0 - ph} {x0 + skew},{y0 - ph}" style="fill:var(--self-bg);stroke:{col};stroke-width:1.1"/>')

        def tp(u, v):
            return (x0 + 12 + u * (pw - 24) + (1 - v) * skew * 0.9, y0 - 3 - (1 - v) * (ph - 6))
        for n, (u, v) in MINI.items():
            if n != ():
                a, b = tp(*MINI[n[:-1]]), tp(u, v)
                st = "var(--ours)" if n in MINI_ACC else col
                S.A(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" style="stroke:{st};stroke-width:{1.3 if n in MINI_ACC else 0.8}"/>')
        for n, (u, v) in MINI.items():
            p = tp(u, v)
            f = "var(--ours)" if n in MINI_ACC else col
            S.A(f'<rect x="{p[0] - 2.2:.1f}" y="{p[1] - 1.8:.1f}" width="4.4" height="3.6" rx="0.8" style="fill:{f}"/>')
        out[L] = (x0 + pw + skew * 0.5, y0 - ph / 2)
    return out


def latent(S, x, y, heights, color="var(--self)", label="plan z", w=6, op=1.0):
    for i, h in enumerate(heights):
        S.A(f'<rect x="{x + i * (w + 2)}" y="{y - h}" width="{w}" height="{h}" rx="1" style="fill:{color};opacity:{op}"/>')
    if label:
        S.A(f'<text x="{x + len(heights) * (w + 2) / 2 - 1}" y="{y + 15}" text-anchor="middle" font-size="11.5" font-style="italic" style="fill:{color}">{label}</text>')


def block_row(S, x, y, slots, s, kind, accent="var(--ours)"):
    for i in range(slots):
        xx = x + i * (s + 2)
        filled = kind == "clean" or (kind == "mid" and (i * 7) % 3 == 0)
        if filled:
            S.A(f'<rect x="{xx}" y="{y}" width="{s}" height="{s}" rx="3" style="fill:{accent};opacity:{1 if kind == "clean" else 0.55}"/>')
        else:
            S.A(f'<rect x="{xx}" y="{y}" width="{s}" height="{s}" rx="3" style="fill:none;stroke:var(--muted);stroke-width:1;stroke-dasharray:2 2"/>')


def token_tree(S, x, y, w=110, h=70, color="var(--ours)"):
    pts = {(): (x, y), (0,): (x + w * .33, y - h * .3), (1,): (x + w * .33, y + h * .3),
           (0, 0): (x + w * .66, y - h * .45), (0, 1): (x + w * .66, y - h * .1), (1, 0): (x + w * .66, y + h * .3),
           (0, 1, 0): (x + w, y - h * .15), (0, 0, 0): (x + w, y - h * .5), (1, 0, 0): (x + w, y + h * .35)}
    acc = {(), (0,), (0, 1), (0, 1, 0)}
    for n, (px, py) in pts.items():
        if n != ():
            qx, qy = pts[n[:-1]]
            mx = (qx + px) / 2
            st = f"stroke:{color};stroke-width:2.2" if n in acc else "stroke:var(--muted);stroke-width:1;opacity:.6"
            S.A(f'<path d="M{qx:.1f} {qy:.1f} C{mx:.1f} {qy:.1f},{mx:.1f} {py:.1f},{px:.1f} {py:.1f}" style="fill:none;{st}"/>')
    for n, (px, py) in pts.items():
        S.A(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{3.6 if n in acc else 2.6}" style="fill:{color if n in acc else "var(--muted)"}"/>')


def link(S, x1, y1, x2, y2, col="var(--self)", arrow="sk_o", dash=True, w=1.3):
    d = ' stroke-dasharray:3 3;' if dash else ''
    S.A(f'<path d="M{x1:.1f} {y1:.1f} C{(x1 + x2) / 2:.1f} {y1:.1f},{(x1 + x2) / 2:.1f} {y2:.1f},{x2:.1f} {y2:.1f}" '
        f'style="fill:none;stroke:{col};stroke-width:{w};{d}" marker-end="url(#{arrow})"/>')


# ================= A =================
S = Svg(960, 330, "Three tapped layers of the target merge into one plan latent that conditions a block-diffusion drafter, which produces a draft tree.")
S.A('<text x="40" y="58" font-size="11.5" font-weight="600" style="fill:currentColor">9B target, 8 of 32 layers shown</text>')
planes(S, 40, 84, 8)
S.ph(1); tA = tapped(S, 40, 84, {1, 4, 6})
for L in (1, 4, 6):
    S.A(f'<text x="{tA[L][0] + 30}" y="{tA[L][1] + 4:.0f}" font-size="10.5" style="fill:var(--self)">tap</text>')
S.end()
lx, ly = 390, 190
S.ph(2)
for L in (1, 4, 6):
    x, y = tA[L]
    link(S, x + 52, y, lx - 6, ly - 20, arrow="sk_o")
latent(S, lx, ly - 2, [12, 26, 8, 20, 30, 11, 22, 16])
S.end()
S.ph(3)
S.A(f'<line x1="{lx + 70}" y1="{ly - 20}" x2="{lx + 112}" y2="{ly - 20}" style="stroke:var(--self);stroke-width:1.4" marker-end="url(#sk_o)"/>')
S.A(f'<text x="{lx + 91}" y="{ly - 30}" text-anchor="middle" font-size="10.5" style="fill:var(--self)">conditions</text>')
S.A('<text x="512" y="110" font-size="11.5" font-weight="600" style="fill:currentColor">Block-diffusion drafter</text>')
block_row(S, 512, 124, 12, 15, "mask"); S.A('<text x="724" y="136" font-size="10.5" style="fill:var(--muted)">masked block</text>')
S.end()
S.ph(4); block_row(S, 512, 147, 12, 15, "mid"); S.A('<text x="724" y="159" font-size="10.5" style="fill:var(--muted)">denoising</text>'); S.end()
S.ph(5); block_row(S, 512, 170, 12, 15, "clean"); S.A('<text x="724" y="182" font-size="10.5" style="fill:var(--muted)">draft block</text>')
S.A('<line x1="614" y1="192" x2="614" y2="214" style="stroke:var(--ours);stroke-width:1.4" marker-end="url(#sk_t)"/>')
token_tree(S, 562, 258); S.A('<text x="692" y="262" font-size="10.5" style="fill:var(--muted)">draft tree, checked by the target</text>'); S.end()
FIGS.append(("A", "Pooled latent", [
    ("Target", "The target model. Only a few of its layers will be read."),
    ("Tap", "Three layers are tapped, as DFlash reads 8 of 32 layers and EAGLE-3 reads three. Each holds the KV-cache tree of the checked drafts."),
    ("Pool", "Their states are pooled into one plan latent."),
    ("Condition", "The latent conditions a block-diffusion drafter, which starts from a fully masked block."),
    ("Denoise", "The drafter fills in the block over one or two denoising steps."),
    ("Draft", "The finished block seeds the next draft tree, which the target checks."),
], S.done(), "A few tapped layers are pooled into one plan latent that conditions a block-diffusion drafter."))

# ================= B =================
S = Svg(960, 330, "Each tapped layer conditions a different stretch of the diffusion block.")
cols3 = {1: "var(--accent)", 4: "var(--self)", 6: "var(--reject)"}
planes(S, 40, 84, 8)
bx, by, s = 470, 160, 22
spans = {1: (0, 4), 4: (4, 9), 6: (9, 14)}
S.ph(1)
for i in range(14):
    S.A(f'<rect x="{bx + i * (s + 3)}" y="{by}" width="{s}" height="{s}" rx="4" style="fill:var(--ours-bg);stroke:var(--ours);stroke-width:1"/>')
S.A(f'<text x="{bx}" y="{by - 84}" font-size="11.5" font-weight="600" style="fill:currentColor">Diffusion block, 14 positions</text>')
S.A(f'<text x="{bx}" y="{by - 70}" font-size="10.5" style="fill:var(--muted)">each tapped layer is cross-attended by one stretch</text>')
S.end()
tB = {}
for k, (L, lab) in enumerate([(1, "next few tokens"), (4, "mid-block"), (6, "far end of the block")]):
    S.ph(2 + k)
    tB.update(tapped(S, 40, 84, {L}, colors=cols3))
    x, y = tB[L]
    a, b = spans[L]
    tx = bx + ((a + b) / 2) * (s + 3) - 2
    col = cols3[L]
    S.A(f'<path d="M{x + 52} {y:.1f} C{x + 170} {y:.1f},{x + 170} {by - 46},{bx - 20} {by - 46} S{tx} {by - 46},{tx} {by - 6}" '
        f'style="fill:none;stroke:{col};stroke-width:1.4;stroke-dasharray:3 3" marker-end="url(#sk_k)"/>')
    S.A(f'<rect x="{bx + a * (s + 3) - 2}" y="{by + s + 6}" width="{(b - a) * (s + 3) - 1}" height="4" rx="2" style="fill:{col}"/>')
    S.A(f'<text x="{bx + (a + b) / 2 * (s + 3)}" y="{by + s + 26}" text-anchor="middle" font-size="10.5" style="fill:{col}">{lab}</text>')
    S.end()
S.ph(5)
S.A(f'<text x="{bx}" y="{by + 80}" font-size="10.5" style="fill:var(--muted)">Exact tokens fade after about 4 positions while the gist lasts longer,</text>')
S.A(f'<text x="{bx}" y="{by + 95}" font-size="10.5" style="fill:var(--muted)">which is where block drafters lose acceptance (0.90 at position 1, 0.53 at 8).</text>')
S.end()
FIGS.append(("B", "Per-stretch layers", [
    ("Target", "The target model, before any layer is read."),
    ("Block", "The drafter works on a block of 14 positions at once."),
    ("Near", "A lower layer, which carries the next tokens, steers the start of the block."),
    ("Middle", "A middle layer steers the middle of the block."),
    ("Far", "A deeper layer, which carries more of the gist, steers the far end."),
    ("Why", "The far end is where block drafters lose most of their acceptance, so that is where extra target information should help."),
], S.done(), "Each tapped layer conditions its own stretch of the diffusion block, near to far."))

# ================= C =================
S = Svg(960, 380, "The plan latent branches in proportion to the target's uncertainty, each branch running its own diffusion block and token subtree.")
S.A('<text x="40" y="62" font-size="11.5" font-weight="600" style="fill:currentColor">9B target</text>')
planes(S, 40, 132, 6)
z0x, z0y = 330, 232
S.ph(1)
tC = tapped(S, 40, 132, {2, 3})
x2, y2 = tC[3]
link(S, x2 + 52, y2, z0x - 6, z0y - 14)
latent(S, z0x, z0y, [10, 22, 7, 18, 26, 9, 19, 13], label="gist", w=5)
S.end()
branches = [(0.8, 160, [12, 24, 9, 20, 28, 10, 21, 15], "about 80%: a number comes next", 12),
            (0.2, 315, [20, 8, 26, 12, 7, 22, 10, 18], "about 20%: an explanation first", 6)]
for k, (p, by_, hs, txt, slots) in enumerate(branches):
    S.ph(2)
    S.A(f'<path d="M{z0x + 62} {z0y - 14} C{z0x + 110} {z0y - 14},{z0x + 100} {by_ - 14},{z0x + 140} {by_ - 14}" style="fill:none;stroke:var(--self);stroke-width:{1 + 3 * p:.1f};opacity:.85"/>')
    latent(S, z0x + 150, by_, hs, label=f"z, p≈{p}", w=5)
    S.A(f'<text x="{z0x + 150}" y="{by_ + 34}" font-size="10.5" style="fill:var(--muted)">{txt}</text>')
    S.end()
    S.ph(3)
    S.A(f'<line x1="{z0x + 212}" y1="{by_ - 14}" x2="{z0x + 248}" y2="{by_ - 14}" style="stroke:var(--self);stroke-width:1.3" marker-end="url(#sk_o)"/>')
    for r, kind in enumerate(["mask", "mid", "clean"]):
        block_row(S, z0x + 256, by_ - 54 + r * 20, slots, 12, kind)
    S.end()
    S.ph(4)
    token_tree(S, z0x + 256 + slots * 14 + 22, by_ - 22, w=90 if p > 0.5 else 55, h=56 if p > 0.5 else 34)
    S.end()
S.ph(2)
S.A('<text x="610" y="62" font-size="10.5" style="fill:var(--muted)">branch width follows the target&#8217;s own probabilities:</text>')
S.A('<text x="610" y="76" font-size="10.5" style="fill:var(--muted)">the likely plan gets a bigger block and a wider token tree</text>')
S.end()
FIGS.append(("C", "Branching latent", [
    ("Target", "The target model."),
    ("Gist", "Its middle layers yield a gist of where the answer is going."),
    ("Split", "When the target is unsure, the gist splits, weighted by how likely the target thinks each plan is."),
    ("Denoise", "Each latent branch runs its own diffusion block. The likely plan gets the bigger block."),
    ("Trees", "Each block becomes a token subtree: a tree of plans on the target side, token trees on the drafter side."),
], S.done(), "The plan latent splits by the target's own probabilities. This matters most when sampling."))

# ================= D =================
S = Svg(960, 340, "A plan latent is denoised from noise, conditioned on the target's tapped layers, then decoded into tokens.")
S.A('<text x="40" y="62" font-size="11.5" font-weight="600" style="fill:currentColor">9B target</text>')
planes(S, 40, 108, 6, gap=28, pw=130, skew=32, ph=20)
ly = 212
xs = [330, 430, 530]
S.ph(1)
latent(S, xs[0], ly, [17, 6, 24, 11, 20, 4, 15, 22], color="var(--muted)", label="noise", op=.55)
S.end()
S.ph(2)
tD = tapped(S, 40, 108, {2, 3}, gap=28, pw=130, skew=32, ph=20)
x3, y3 = tD[3]
for tx in (xs[0] + 31, xs[1] + 31, xs[2] + 31):
    S.A(f'<path d="M{x3 + 40} {y3:.0f} C{x3 + 90} {y3 - 70:.0f},{tx} {ly - 100},{tx} {ly - 36}" style="fill:none;stroke:var(--self);stroke-width:1;stroke-dasharray:3 3;opacity:.8" marker-end="url(#sk_o)"/>')
S.A(f'<text x="{xs[1] + 31}" y="{ly - 108}" text-anchor="middle" font-size="10.5" style="fill:var(--self)">every step is conditioned on the tapped layers</text>')
S.A(f'<line x1="{xs[0] + 66}" y1="{ly - 14}" x2="{xs[1] - 6}" y2="{ly - 14}" style="stroke:var(--self);stroke-width:1.3" marker-end="url(#sk_o)"/>')
latent(S, xs[1], ly, [13, 18, 12, 17, 24, 8, 18, 16], label="denoising", op=.55)
S.end()
S.ph(3)
S.A(f'<line x1="{xs[1] + 66}" y1="{ly - 14}" x2="{xs[2] - 6}" y2="{ly - 14}" style="stroke:var(--self);stroke-width:1.3" marker-end="url(#sk_o)"/>')
latent(S, xs[2], ly, [12, 24, 9, 20, 28, 10, 21, 15], label="clean plan z")
S.A(f'<text x="{xs[1] + 31}" y="{ly + 44}" text-anchor="middle" font-size="10.5" style="fill:var(--muted)">1–2 denoising steps, in a space far smaller than tokens</text>')
S.end()
dx = 640
S.ph(4)
S.A(f'<polygon points="{dx},{ly - 44} {dx + 70},{ly - 58} {dx + 70},{ly + 30} {dx},{ly + 16}" style="fill:var(--panel);stroke:currentColor;stroke-width:1.1"/>')
S.A(f'<text x="{dx + 35}" y="{ly - 10}" text-anchor="middle" font-size="11.5" font-weight="600" style="fill:currentColor">decoder</text>')
S.A(f'<line x1="{xs[2] + 68}" y1="{ly - 14}" x2="{dx - 6}" y2="{ly - 14}" style="stroke:var(--self);stroke-width:1.3" marker-end="url(#sk_o)"/>')
for i in range(8):
    S.A(f'<rect x="{dx + 92 + i * 16}" y="{ly - 70}" width="13" height="13" rx="3" style="fill:var(--ours)"/>')
S.A(f'<text x="{dx + 92}" y="{ly - 78}" font-size="10.5" style="fill:var(--muted)">a block of tokens</text>')
S.A(f'<line x1="{dx + 72}" y1="{ly - 30}" x2="{dx + 88}" y2="{ly - 58}" style="stroke:var(--ours);stroke-width:1.3" marker-end="url(#sk_t)"/>')
S.end()
S.ph(5)
S.A(f'<line x1="{dx + 72}" y1="{ly + 4}" x2="{dx + 88}" y2="{ly + 26}" style="stroke:var(--ours);stroke-width:1.3" marker-end="url(#sk_t)"/>')
token_tree(S, dx + 98, ly + 46, w=120, h=60)
S.A(f'<text x="{dx + 98}" y="{ly + 96}" font-size="10.5" style="fill:var(--muted)">or a tree, if the decoder keeps</text>')
S.A(f'<text x="{dx + 98}" y="{ly + 110}" font-size="10.5" style="fill:var(--muted)">several candidates per position</text>')
S.end()
FIGS.append(("D", "Latent diffusion", [
    ("Target", "The target model."),
    ("Noise", "The plan latent starts as pure noise."),
    ("Denoise", "It is denoised in a continuous space, with every step conditioned on the target's tapped middle layers."),
    ("Clean plan", "After one or two steps the latent is a clean plan."),
    ("Decode", "Only the clean latent is decoded, here into a block of tokens."),
    ("Tree", "Or into a tree of candidates. ChainFlow (2611.08669) does a flow-matching version with one latent per token."),
], S.done(), "Denoise a plan latent, then decode it. The open variant is one latent for a long stretch, decoded into a tree."))

# ================= E =================
S = Svg(960, 350, "The clean plan latent is decoded coarse to fine, each scale steered by a different tapped layer.")
S.A('<text x="40" y="62" font-size="11.5" font-weight="600" style="fill:currentColor">9B target</text>')
ecol = {1: "var(--accent)", 3: "var(--self)", 5: "var(--reject)"}
planes(S, 40, 104, 6, gap=34, pw=130, skew=32, ph=22)
S.ph(0)
tE = tapped(S, 40, 104, {1, 3, 5}, gap=34, pw=130, skew=32, ph=22, colors=ecol)
for L, nm in ((1, "shallow"), (3, "middle"), (5, "deep")):
    S.A(f'<text x="{tE[L][0] + 28}" y="{tE[L][1] + 4:.0f}" font-size="10.5" style="fill:{ecol[L]}">{nm}</text>')
S.end()
zx, zy = 300, 214
S.ph(1); latent(S, zx, zy, [12, 24, 9, 20, 28, 10, 21, 15], label="clean plan z"); S.end()
lx0, lw = 520, 330
lev = {"coarse": 104, "mid": 170, "fine": 236}
names = {"coarse": "whole block", "mid": "4 segments", "fine": "16 tokens"}
order = ["coarse", "mid", "fine"]
cnn = {"coarse": 1, "mid": 3, "fine": 5}
probe = {"coarse": 5, "mid": 3, "fine": 1}
for k, lv in enumerate(order):
    S.ph(2 + k)
    y = lev[lv]
    if lv == "coarse":
        S.A(f'<rect x="{lx0}" y="{y}" width="{lw}" height="18" rx="5" style="fill:var(--ours);opacity:.35"/>')
        S.A(f'<path d="M{zx + 66} {zy - 18} C{zx + 140} {zy - 18},{lx0 - 90} {y + 9},{lx0 - 6} {y + 9}" style="fill:none;stroke:var(--self);stroke-width:1.4" marker-end="url(#sk_o)"/>')
        S.A(f'<text x="{lx0}" y="{y - 12}" font-size="11.5" font-weight="600" style="fill:currentColor">Decoder, coarse to fine</text>')
    elif lv == "mid":
        for j in range(4):
            S.A(f'<rect x="{lx0 + j * (lw / 4) + 2:.0f}" y="{y}" width="{lw / 4 - 4:.0f}" height="18" rx="5" style="fill:var(--ours);opacity:.6"/>')
    else:
        for j in range(16):
            S.A(f'<rect x="{lx0 + j * (lw / 16) + 1.5:.1f}" y="{y}" width="{lw / 16 - 3:.1f}" height="18" rx="3" style="fill:var(--ours)"/>')
    if k > 0:
        S.A(f'<line x1="{lx0 + lw / 2}" y1="{lev[order[k - 1]] + 20}" x2="{lx0 + lw / 2}" y2="{y - 4}" style="stroke:var(--ours);stroke-width:1.3" marker-end="url(#sk_t)"/>')
    S.A(f'<text x="{lx0 + lw + 10}" y="{y + 13}" font-size="10.5" style="fill:var(--muted)">{names[lv]}</text>')
    for mapping, cls, hid in ((cnn, "e-cnn", False), (probe, "e-probe", True)):
        L = mapping[lv]
        x, yy = tE[L]
        S.A(f'<path class="{cls}"{" hidden" if hid else ""} d="M{x + 60} {yy:.0f} C{x + 200} {yy:.0f},{lx0 - 140} {y + 9},{lx0 - 6} {y + 9}" '
            f'style="fill:none;stroke:{ecol[L]};stroke-width:1.3;stroke-dasharray:3 3" marker-end="url(#sk_k)"/>')
    S.end()
S.A(f'<text x="{lx0}" y="{lev["fine"] + 50}" font-size="10.5" style="fill:var(--muted)" class="e-cnn">CNN-style: shallow layers steer the whole block, deep layers the details</text>')
S.A(f'<text x="{lx0}" y="{lev["fine"] + 50}" font-size="10.5" style="fill:var(--muted)" class="e-probe" hidden>Probe-style: deep layers carry the gist, shallow layers the next tokens</text>')
FIGS.append(("E", "Coarse to fine", [
    ("Taps", "Three layers are tapped: shallow, middle and deep."),
    ("Plan", "The clean plan latent from Sketch D."),
    ("Whole block", "The decoder first produces one code for the whole block, steered by one tapped layer."),
    ("Segments", "Then four segment codes, steered by another."),
    ("Tokens", "Then sixteen tokens, steered by the third. Use the swap button to flip which layer steers which scale."),
], S.done(), "Coarse-to-fine decoding, like a CNN's receptive fields. Which layer steers which scale is what the layer sweep will decide."))

# ================= viewer =================
tabs = "".join(f'<button type="button" role="tab" class="sktab" data-k="{k}" aria-selected="{"true" if i == 0 else "false"}">'
               f'<span>{k}</span>{t}</button>' for i, (k, t, _, _, _) in enumerate(FIGS))
panes = "".join(f'<div class="skpane" data-k="{k}"{"" if i == 0 else " hidden"}><div class="figscroll">{svg}</div>'
                f'<p class="skcap"><b>Sketch {k}.</b> {cap}</p></div>' for i, (k, t, ph, svg, cap) in enumerate(FIGS))
phases = {k: ph for (k, t, ph, svg, cap) in FIGS}
html = f'''<div class="skviewer" id="skv">
  <div class="sktabs" role="tablist" aria-label="Design sketches">{tabs}</div>
  {panes}
  <div class="scrubbar">
    <button type="button" id="skPrev" aria-label="Previous phase">‹</button>
    <button type="button" id="skPlay">Play</button>
    <button type="button" id="skNext" aria-label="Next phase">›</button>
    <input type="range" id="skRange" min="0" max="5" step="1" value="5" aria-label="Phase">
    <button type="button" id="eSwap" hidden>Swap to probe-style</button>
  </div>
  <ol class="phases" id="skSteps"></ol>
  <p class="phasetext" id="skText" aria-live="polite"></p>
</div>
<script>
(function () {{
  const PH = {json.dumps(phases)};
  const root = document.getElementById("skv"), r = document.getElementById("skRange");
  const stepsEl = document.getElementById("skSteps"), txt = document.getElementById("skText"), swap = document.getElementById("eSwap");
  let cur = "A", timer = null, probe = false;
  function pane() {{ return root.querySelector('.skpane[data-k="' + cur + '"]'); }}
  function show(p) {{
    const n = PH[cur].length; p = Math.max(0, Math.min(n - 1, p)); r.value = p;
    pane().querySelectorAll(".ph").forEach(g => {{ g.style.opacity = (p >= +g.dataset.lo && p <= +g.dataset.hi) ? 1 : 0; }});
    stepsEl.querySelectorAll("li").forEach((li, i) => li.classList.toggle("on", i === p));
    txt.textContent = PH[cur][p][1];
  }}
  function stop() {{ clearInterval(timer); timer = null; document.getElementById("skPlay").textContent = "Play"; }}
  function select(k) {{
    stop(); cur = k;
    root.querySelectorAll(".sktab").forEach(b => b.setAttribute("aria-selected", b.dataset.k === k ? "true" : "false"));
    root.querySelectorAll(".skpane").forEach(p => {{ p.hidden = p.dataset.k !== k; }});
    stepsEl.innerHTML = PH[k].map((s, i) => '<li data-i="' + i + '"><span>' + (i + 1) + '</span>' + s[0] + '</li>').join("");
    stepsEl.querySelectorAll("li").forEach(li => li.addEventListener("click", () => {{ stop(); show(+li.dataset.i); }}));
    r.max = PH[k].length - 1; swap.hidden = k !== "E";
    show(PH[k].length - 1);
  }}
  root.querySelectorAll(".sktab").forEach(b => b.addEventListener("click", () => select(b.dataset.k)));
  r.addEventListener("input", () => {{ stop(); show(+r.value); }});
  document.getElementById("skPrev").onclick = () => {{ stop(); show(+r.value - 1); }};
  document.getElementById("skNext").onclick = () => {{ stop(); show(+r.value + 1); }};
  document.getElementById("skPlay").onclick = () => {{
    if (timer) return stop();
    document.getElementById("skPlay").textContent = "Pause";
    if (+r.value >= PH[cur].length - 1) show(0);
    timer = setInterval(() => {{ if (+r.value >= PH[cur].length - 1) return stop(); show(+r.value + 1); }}, 1600);
  }};
  swap.addEventListener("click", () => {{
    probe = !probe;
    root.querySelectorAll(".e-cnn").forEach(e => e.toggleAttribute("hidden", probe));
    root.querySelectorAll(".e-probe").forEach(e => e.toggleAttribute("hidden", !probe));
    swap.textContent = probe ? "Swap to CNN-style" : "Swap to probe-style";
  }});
  select("A");
}})();
</script>'''
open("sketches.html", "w", encoding="utf-8").write(html)
print("ok", len(html))
