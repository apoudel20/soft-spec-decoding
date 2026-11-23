exec(open('make_layers.py', encoding='utf-8').read())
t = open('template.html', encoding='utf-8').read()
css = '''
.legendrow { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 12.5px; color: var(--muted); }
.legendrow .lg { display: inline-flex; align-items: center; gap: 6px; }
.legendrow i { display: inline-block; width: 12px; height: 12px; border-radius: 3px; }
.legendrow i.ln { width: 22px; height: 0; border-top: 2.2px solid var(--fg); border-radius: 0; }
.legendrow i.ln.dash { border-top: 1.5px dashed var(--fg); }
'''
if '.legendrow' not in t:
    t = t.replace('</style>', css + '</style>', 1)
blk = open('layers.html', encoding='utf-8').read()
if 'id="layerResults"' in t:
    a = t.index('<div class="live" id="layerResults">')
    b = t.index('</p>\n    </div>', a) + len('</p>\n    </div>')
    t = t[:a] + blk + t[b:]
else:
    a = t.index('<div class="live" id="gapResults">')
    sec_end = t.index('</section>', a)
    t = t[:sec_end] + '  ' + blk + '\n  ' + t[sec_end:]
t = t.replace('<h2 id="liveT">Selection test</h2>', '<h2 id="liveT">Experiments</h2>')
t = t.replace('<a href="#live">Selection test</a>', '<a href="#live">Experiments</a>')
open('template.html', 'w', encoding='utf-8').write(t)
open('index.html', 'w', encoding='utf-8').write(t.replace('__SUMMARY__', open('summary.json', encoding='utf-8').read()))
print('page ok', t.count('id="layerResults"'))
