exec(open('make_sketches.py', encoding='utf-8').read())
t = open('template.html', encoding='utf-8').read()

# light-only theme: drop the two dark blocks, keep the light tokens
import re
if '@media (prefers-color-scheme: dark)' in t:
    i = t.index('@media (prefers-color-scheme: dark)')
    j = t.index(':root[data-theme="dark"] {')
    k = t.index('}', j) + 1
    t = t[:i] + t[k:]
if 'color-scheme: light;' not in t:
    t = t.replace('  --btn: #1b1f24; --btn-fg: #ffffff;', '  --btn: #1b1f24; --btn-fg: #ffffff; color-scheme: light;', 1)
t = t.replace('--bg: #f7f7f4; --panel: #ffffff;', '--bg: #fbfbf9; --panel: #ffffff;', 1)

block = ('\n    <div class="eyebrow" style="margin-top:18px">Design sketches</div>\n'
         '    <p class="muted">Five ways the target&#8217;s tapped layers could feed a diffusion drafter. A to C denoise tokens. D denoises a latent and then decodes it, and E decodes that latent coarse to fine. None is built yet. They frame what the experiments would compare.</p>\n    '
         + open('sketches.html', encoding='utf-8').read() + '\n')
if 'Design sketches' in t:
    a = t.index('\n    <div class="eyebrow" style="margin-top:18px">Design sketches</div>')
    b = t.index('</section>', a)
    t = t[:a] + block + '  ' + t[b:]
else:
    a = t.index('<h2 id="treesT">Two trees</h2>')
    b = t.index('</section>', a)
    t = t[:b] + block + '  ' + t[b:]
open('template.html', 'w', encoding='utf-8').write(t)
open('index.html', 'w', encoding='utf-8').write(t.replace('__SUMMARY__', open('summary.json', encoding='utf-8').read()))
print('page ok', t.count('Sketch '), 'dark blocks left:', t.count('prefers-color-scheme: dark'))
