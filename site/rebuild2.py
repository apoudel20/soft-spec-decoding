exec(open('make_fig2.py', encoding='utf-8').read())
f2 = open('fig2.html', encoding='utf-8').read().replace('<b>Figure 2.</b> Two trees working together.', '<b>Figure 3.</b> Two trees working together.')
t = open('template.html', encoding='utf-8').read()
k = t.index('Two trees working together.')
i = t.rindex('<figure class="fig1">', 0, k); j = t.index('</figure>', k) + len('</figure>')
t = t[:i] + f2 + t[j:]
open('template.html', 'w', encoding='utf-8').write(t)
open('index.html', 'w', encoding='utf-8').write(t.replace('__SUMMARY__', open('summary.json', encoding='utf-8').read()))
print('page ok', t.count('Figure 3.'))
