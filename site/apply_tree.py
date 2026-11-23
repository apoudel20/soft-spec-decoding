t = open('template.html', encoding='utf-8').read()
blk = '''<div class="live" id="treeGap">
      <div class="state">Done · tree headroom on DFlash · top 4 tokens per slot · held-out dev answers · accepted tokens per pass</div>
      <div class="scroll"><table class="gaptable"><thead><tr><th>Set</th><th>Single chain</th><th>Tree, 32 nodes</th><th>64</th><th>128</th><th>256</th><th>Best possible</th></tr></thead><tbody>
        <tr><td><span class="dotk" style="background:#008a72"></span>Math</td><td>7.46</td><td>8.63</td><td>9.03</td><td>9.31</td><td>9.52</td><td>10.49</td></tr>
        <tr><td><span class="dotk" style="background:#c25a17"></span>Code</td><td>6.77</td><td>8.21</td><td>8.68</td><td>9.10</td><td>9.37</td><td>10.72</td></tr>
        <tr><td><span class="dotk" style="background:#6a4fc0"></span>Chat</td><td>2.96</td><td>3.85</td><td>4.00</td><td>4.13</td><td>4.20</td><td>4.43</td></tr>
      </tbody></table></div>
      <p>Checking a tree of DFlash&#8217;s alternatives instead of its single chain adds 16 to 35% accepted tokens at 32 to 64 nodes. That part is known (DDTree builds trees on DFlash). What is left for a smarter selector is the gap to the best possible choice from the same candidates: at 64 nodes, +16% on math, +24% on code and +11% on chat. On chat the candidates themselves are the limit, since even perfect selection from the top 4 per slot reaches only 4.4.</p>
      <p class="muted">Trees cost more verification compute per pass than a chain. The real speedup depends on where the target&#8217;s cost starts rising with the number of nodes, which this 4-bit setup cannot measure. Prior work to check before building a selector: GRAFT (2608.20375) reshapes trees with a scorer distilled from the target.</p>
    </div>
'''
if 'id="treeGap"' not in t:
    a = t.index('<div class="live" id="nextBuild">')
    t = t[:a] + blk + '  ' + t[a:]
a = t.index('<div class="live" id="nextBuild">'); b = t.index('</div>\n', a) + len('</div>\n')
t = t[:a] + '''<div class="live" id="nextBuild">
      <div class="state">Next · a selector for DFlash trees</div>
      <p>The plan route is closed for now: DFlash does not use future information however it is fed in. The tree route has room: a perfect selector would add 11 to 24% over the standard most-probable-first tree at 64 nodes. The next step is a small learned scorer that ranks DFlash&#8217;s candidate branches using the target&#8217;s state at the last verified token, compared against most-probable-first at the same budget. First, a full read of GRAFT, the closest prior work.</p>
    </div>
''' + t[b:]
t = t.replace('<tr><td>True future fed to DFlash, frozen or fine-tuned</td>', '<tr><td>Tree on DFlash (64 nodes)</td><td><span class="pill ok">Works</span>+21 to +35% over chain</td><td>Known: DDTree</td></tr>\n        <tr><td>True future fed to DFlash, frozen or fine-tuned</td>')
open('template.html', 'w', encoding='utf-8').write(t)
open('index.html', 'w', encoding='utf-8').write(t.replace('__SUMMARY__', open('summary.json', encoding='utf-8').read()))
print('ok', t.count('id="treeGap"'))
