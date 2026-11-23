t = open('template.html', encoding='utf-8').read()
blk = '''<div class="live" id="futureTests">
      <div class="state">Done · can DFlash use the future? · Qwen3-8B (4-bit) · held-out dev answers · accepted tokens per 16-token block</div>
      <div class="scroll"><table class="gaptable"><thead><tr><th>What DFlash is given</th><th>Drafter</th><th>Math</th><th>Code</th><th>Chat</th></tr></thead><tbody>
        <tr><td>Nothing extra</td><td>as released</td><td>7.46</td><td>6.71</td><td>2.94</td></tr>
        <tr><td>True future (next 16), raw, one extra context position</td><td>frozen</td><td>5.44</td><td>4.11</td><td>2.04</td></tr>
        <tr><td>True future (next 16), through a trained adapter</td><td>frozen</td><td>7.43</td><td>6.71</td><td>2.94</td></tr>
        <tr><td>Nothing extra, LoRA fine-tune (control)</td><td>fine-tuned</td><td>7.44</td><td>6.87</td><td>2.97</td></tr>
        <tr><td>True future (next 16), LoRA fine-tune</td><td>fine-tuned</td><td>7.48</td><td>6.81</td><td>2.95</td></tr>
      </tbody></table></div>
      <p>Even a summary of the target&#8217;s true states for the very block being drafted does not help DFlash, whether the drafter is frozen or fine-tuned alongside it. Against the matched fine-tune, the difference is +0.04, &minus;0.06 and &minus;0.02 tokens, with 95% intervals that straddle zero. Fed raw and untrained, it hurts, because DFlash never saw such an input.</p>
      <p class="muted">With 300 training answers and a single averaged summary, knowing the future does not make this drafter better. Much more training might still find a use for it, but three ways of feeding a plan in have now been flat, oracle included. The remaining route is choosing better which drafted tokens get checked, which does not need the drafter to absorb a plan.</p>
    </div>
'''
if 'id="futureTests"' not in t:
    a = t.index('<div class="live" id="nextBuild">')
    t = t[:a] + blk + '  ' + t[a:]
old_next = t[t.index('<div class="live" id="nextBuild">'):]
old_next = old_next[:old_next.index('</div>\n') + len('</div>\n')]
new_next = '''<div class="live" id="nextBuild">
      <div class="state">Next · measuring tree headroom on DFlash</div>
      <p>A drafter built around a plan would need the plan to carry usable information, and the tests above say a strong block drafter does not use it. What is left is the tree: DFlash gives a distribution for every slot, so many alternative continuations can be checked in one pass. A tree version of DFlash already exists (DDTree); choosing which branches to check using the target&#8217;s own signal does not. The next measurement asks how much a tree on DFlash adds over its single chain, and how much room a perfect selector would still have.</p>
    </div>
'''
t = t.replace(old_next, new_next)
t = t.replace('<tr><td>Plan vector added to DFlash inputs</td><td><span class="pill no">No effect</span>±3%, oracle included</td><td>ConFu (different pathway)</td></tr>',
              '<tr><td>Plan vector added to DFlash inputs</td><td><span class="pill no">No effect</span>±3%, oracle included</td><td>ConFu (different pathway)</td></tr>\n        <tr><td>True future fed to DFlash, frozen or fine-tuned</td><td><span class="pill no">No effect</span>within ±0.06 tokens</td><td>None found</td></tr>')
open('template.html', 'w', encoding='utf-8').write(t)
open('index.html', 'w', encoding='utf-8').write(t.replace('__SUMMARY__', open('summary.json', encoding='utf-8').read()))
print('ok', t.count('id="futureTests"'))
