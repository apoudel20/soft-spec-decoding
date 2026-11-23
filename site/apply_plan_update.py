t = open('template.html', encoding='utf-8').read()
old_intro_start = t.index('    <p class="muted">Five ways the target')
old_intro_end = t.index('</p>', old_intro_start) + len('</p>')
intro = ('    <p class="muted">Five ways the target&#8217;s tapped layers could feed a diffusion drafter. A to C denoise tokens, D denoises a latent and '
         'then decodes it, and E decodes that latent coarse to fine. Pick a sketch, then drag the slider or press play to step through it.</p>\n'
         '    <div class="scroll"><table class="gaptable sktable"><thead><tr><th>Sketch</th><th>Status</th><th>Where it goes next</th></tr></thead><tbody>'
         '<tr><td>A. Pooled latent</td><td><span class="pill no">Tested, no effect</span></td><td>Added to DFlash&#8217;s inputs, even an oracle plan did nothing (section 04)</td></tr>'
         '<tr><td>B. Per-stretch layers</td><td><span class="pill run">Folded in</span></td><td>How the new decoder&#8217;s levels read the target</td></tr>'
         '<tr><td>C. Branching latent</td><td><span class="pill run">Folded in</span></td><td>How the new drafter builds a tree</td></tr>'
         '<tr><td>D. Latent diffusion</td><td><span class="pill run">Folded in</span></td><td>The core of the new drafter</td></tr>'
         '<tr><td>E. Coarse to fine</td><td><span class="pill run">Folded in</span></td><td>The new drafter&#8217;s decoder, deep-first or shallow-first</td></tr>'
         '</tbody></table></div>')
t = t[:old_intro_start] + intro + t[old_intro_end:]

nxt = '''  <div class="live" id="nextBuild">
      <div class="state">Next · a latent-plan drafter</div>
      <p>Bolting a plan onto DFlash does not help, so the next step is a drafter built around the plan: denoise a latent that stands for the next stretch of text (Sketch D), decode it coarse to fine (Sketch E), with each level reading a different tapped layer of the target (Sketch B) and several latent branches becoming a draft tree (Sketch C). It would compete with DFlash rather than ride on it. ChainFlow is the closest prior work, with one latent per token rather than one per stretch.</p>
      <p class="muted">Pending: a no-training test that feeds the target&#8217;s true future states through DFlash&#8217;s own context channel. Its result decides whether the new drafter reads the target through context attention or through the latent alone.</p>
    </div>
'''
if 'id="nextBuild"' not in t:
    a = t.index('<div class="live" id="sketchAResults">')
    end = t.index('</section>', a)
    t = t[:end] + nxt + '  ' + t[end:]
t = t.replace('<tr><td>Target gist chooses the tree</td><td><span class="pill run">Testing</span>selection gap, below</td><td>None found</td></tr>', '')
open('template.html', 'w', encoding='utf-8').write(t)
open('index.html', 'w', encoding='utf-8').write(t.replace('__SUMMARY__', open('summary.json', encoding='utf-8').read()))
print('ok', t.count('id="nextBuild"'), t.count('sktable'))
