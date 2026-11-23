exec(open('make_sketches.py', encoding='utf-8').read())
t = open('template.html', encoding='utf-8').read()
css = '''
.skviewer { display: grid; gap: 12px; }
.sktabs { display: flex; flex-wrap: wrap; gap: 8px; }
.sktab { font: 500 13.5px var(--sans); color: var(--muted); background: var(--panel); border: 1px solid var(--line); border-radius: 999px; padding: 6px 14px 6px 8px; cursor: pointer; display: inline-flex; align-items: center; gap: 8px; }
.sktab span { font: 600 11.5px var(--mono); width: 22px; height: 22px; border-radius: 50%; display: inline-grid; place-items: center; background: var(--bg); border: 1px solid var(--line); }
.sktab[aria-selected="true"] { color: var(--fg); border-color: var(--fg); }
.sktab[aria-selected="true"] span { background: var(--fg); color: var(--panel); border-color: var(--fg); }
.sktab:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.skpane { display: grid; gap: 10px; }
.skpane .ph { transition: opacity .45s ease; }
@media (prefers-reduced-motion: reduce) { .skpane .ph { transition: none; } }
.skcap { margin: 0; font-size: 13.5px; color: var(--muted); max-width: 76ch; }
.skcap b { color: var(--fg); }
'''
if '.skviewer {' not in t:
    t = t.replace('</style>', css + '</style>', 1)
block = ('\n    <div class="eyebrow" style="margin-top:18px">Design sketches</div>\n'
         '    <p class="muted">Five ways the target&#8217;s tapped layers could feed a diffusion drafter. A to C denoise tokens. D denoises a latent and then decodes it, and E decodes that latent coarse to fine. Pick a sketch, then drag the slider or press play to step through it. None is built yet.</p>\n    '
         + open('sketches.html', encoding='utf-8').read() + '\n')
a = t.index('\n    <div class="eyebrow" style="margin-top:18px">Design sketches</div>')
b = t.index('</section>', a)
t = t[:a] + block + '  ' + t[b:]
open('template.html', 'w', encoding='utf-8').write(t)
open('index.html', 'w', encoding='utf-8').write(t.replace('__SUMMARY__', open('summary.json', encoding='utf-8').read()))
i = t.index('id="skv"')
s = t[t.index('<script>', i) + 8:t.index('</script>', i)]
open('_sk.js', 'w', encoding='utf-8').write(s)
print('page ok', t.count('class="skpane"'), t.count('Sketch A.'), 'old E fig:', t.count('id="sketchE"'))
