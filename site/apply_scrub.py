exec(open('make_scrub.py', encoding='utf-8').read())
t = open('template.html', encoding='utf-8').read()
scrub = open('scrub.html', encoding='utf-8').read()
css = '''
.scrub .ph { transition: opacity .45s ease; }
@media (prefers-reduced-motion: reduce) { .scrub .ph { transition: none; } }
.scrubbar { display: flex; align-items: center; gap: 8px; }
.scrubbar button { font: 500 13.5px var(--sans); color: var(--fg); background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 6px 12px; cursor: pointer; min-width: 38px; }
.scrubbar button:hover { border-color: var(--muted); }
.scrubbar button:focus-visible, .scrubbar input:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.scrubbar input[type=range] { flex: 1; accent-color: var(--ours); min-width: 0; }
.phases { list-style: none; margin: 0; padding: 0; display: flex; flex-wrap: wrap; gap: 6px 14px; font-size: 13px; color: var(--muted); }
.phases li { cursor: pointer; display: inline-flex; align-items: center; gap: 6px; }
.phases li span { font: 11px var(--mono); width: 18px; height: 18px; border-radius: 50%; border: 1px solid var(--line); display: inline-grid; place-items: center; }
.phases li.on { color: var(--fg); font-weight: 500; }
.phases li.on span { background: var(--ours); border-color: var(--ours); color: #fff; }
.phasetext { margin: 0; font-size: 15px; min-height: 2.6em; max-width: 70ch; }
'''
if '.scrub .ph' not in t:
    t = t.replace('</style>', css + '</style>', 1)
# the teaser slot holds Figure 2 now: move that static figure into section 02, put the scrubber in the teaser slot
i = t.index('<figure class="fig1">')
j = t.index('</figure>', i) + len('</figure>')
static2 = t[i:j]
assert 'Figure 2.' in static2
if 'id="scrub"' in t:
    a = t.index('<figure class="fig1 scrub" id="scrub">'); b = t.index('</script>', a) + len('</script>')
    t = t[:a] + scrub + t[b:]
else:
    t = t[:i] + scrub + t[j:]
    anchor = t.index('<h2 id="treesT">Two trees</h2>')
    pend = t.index('</p>', anchor) + len('</p>')
    t = t[:pend] + '\n    ' + static2 + t[pend:]
# the idea section's static figure becomes Figure 2 and the two-trees figure becomes Figure 3
t = t.replace('<b>Figure 1.</b> The proposal in one picture.', '<b>Figure 2.</b> The proposal in one picture.')
t = t.replace('<b>Figure 2.</b> Two trees working together.', '<b>Figure 3.</b> Two trees working together.')
t = t.replace('(Figure 2, top)', '(Figure 3, top)')
open('template.html', 'w', encoding='utf-8').write(t)
open('index.html', 'w', encoding='utf-8').write(t.replace('__SUMMARY__', open('summary.json', encoding='utf-8').read()))
s = t[t.rindex('<script>') + 8:t.rindex('</script>')]
open('_chk1.js', 'w', encoding='utf-8').write(s)
s2 = scrub[scrub.index('<script>') + 8:scrub.index('</script>')]
open('_chk2.js', 'w', encoding='utf-8').write(s2)
print('ok', t.count('id="scrub"'), t.count('Figure 1.'), t.count('Figure 2.'), t.count('Figure 3.'))
