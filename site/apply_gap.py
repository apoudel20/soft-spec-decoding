exec(open('make_gap.py', encoding='utf-8').read())
t = open('template.html', encoding='utf-8').read()
a = t.index('<div class="live" id="gapResults">')
# the live block ends at the first </div> that closes it: find matching by counting
i, depth = a, 0
while True:
    o = t.find('<div', i)
    c = t.find('</div>', i)
    if o != -1 and o < c:
        depth += 1
        i = o + 4
    else:
        depth -= 1
        i = c + 6
        if depth == 0:
            break
t = t[:a] + open('gap.html', encoding='utf-8').read() + t[i:]
css = '''
.gaptable { min-width: 480px; }
.gaptable td, .gaptable th { padding: 8px 10px; text-align: right; }
.gaptable td:first-child, .gaptable th:first-child { text-align: left; }
.dotk { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; vertical-align: 0; }
#gapResults .state { color: var(--ours); }
'''
if '.gaptable' not in t:
    t = t.replace('</style>', css + '</style>', 1)
t = t.replace('<tr><td>Target gist chooses the tree</td><td><span class="pill run">Testing</span>selection gap, below</td><td>None found</td></tr>',
              '<tr><td>Better tree selection, ceiling</td><td><span class="pill ok">Room to gain</span>up to +36% at 32 nodes (chat)</td><td>None found that uses the target</td></tr>')
open('template.html', 'w', encoding='utf-8').write(t)
open('index.html', 'w', encoding='utf-8').write(t.replace('__SUMMARY__', open('summary.json', encoding='utf-8').read()))
print('page ok', t.count('id="gapResults"'), 'Figure 4' in t)
