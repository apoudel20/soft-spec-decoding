import json
import os

R = r"C:\Users\aayus\Projects\movement-softspec\research\soft-spec-decoding\results"
runs = [("z512_full", "512", "Layers 18, 24, 30, final"), ("z1024_full", "1024", "Layers 18, 24, 30, final"),
        ("z2048_full", "2048", "Layers 18, 24, 30, final"), ("z1024_final", "1024", "Final layer only")]
rows = ""
for name, z, rws in runs:
    r = json.load(open(os.path.join(R, f"ae_gate_{name}.json")))
    s = r["sets"]
    rows += (f"<tr><td>{z}</td><td>{rws}</td><td>{s['dev_math']['agree'] * 100:.1f}%</td><td>{s['dev_code']['agree'] * 100:.1f}%</td>"
             f"<td>{s['dev_chat']['agree'] * 100:.1f}%</td><td><span class=\"pill no\">Fail</span></td></tr>\n        ")
blk = f'''<div class="live" id="aeGate">
      <div class="state">Done · stage-2 gate · can 8 future Qwen3-8B states be squeezed into one latent and decoded back? · 2,000 training answers · held-out dev</div>
      <div class="scroll"><table class="gaptable"><thead><tr><th>Latent size</th><th>Rows of the grid</th><th>Math</th><th>Code</th><th>Chat</th><th>Gate</th></tr></thead><tbody>
        {rows}<tr><td colspan="2" style="text-align:left;color:var(--muted)">Needed (fixed before training)</td><td>99.5%</td><td>99.5%</td><td>99%</td><td></td></tr>
      </tbody></table></div>
      <p>Each number is how often the decoded state, sent through Qwen3-8B&#8217;s own output layer, picks the same next token as the real state. None of the four autoencoders gets close to the bar. A drafter built on this latent would lose about a quarter of its acceptance on chat before predicting anything, which is more than coherent branches could win back.</p>
      <p>Three things point at the recipe more than at the idea. Latent size barely matters (512 to 2048 adds under a point), the middle layers do not help, and the models are as good on their training data as on held-out data, so they are underfit rather than overfit. The decoders also shrug off latent noise completely, which suggests the noise added during training to make sampling safe may be capping exact reconstruction.</p>
      <p class="muted">Two diagnostic runs are queued: a plain autoencoder with no training noise, and a five times larger decoder trained three times longer. They say whether this is a ceiling of the method or of a 20-minute run. They do not reopen the gate.</p>
    </div>
'''
t = open("template.html", encoding="utf-8").read()
if 'id="aeGate"' in t:
    a = t.index('<div class="live" id="aeGate">')
    b = t.index('<div class="live" id="nextBuild">')
    t = t[:a] + blk + "    " + t[b:]
else:
    a = t.index('<div class="live" id="nextBuild">')
    t = t[:a] + blk + "    " + t[a:]
a = t.index('<div class="live" id="nextBuild">')
b = t.index("</div>\n", a) + len("</div>\n")
t = t[:a] + '''<div class="live" id="nextBuild">
      <div class="state">Next · a latent the target&#8217;s output layer can read back</div>
      <p>The open idea from the literature is narrow: sample several latents of the target&#8217;s own future states and turn each into a whole continuation, so a draft tree holds coherent branches instead of mixed-up per-slot picks. That only pays if the latent decodes back almost perfectly. The diagnostic runs decide whether to keep pushing on the autoencoder or drop it and inject sampled noise straight into a block drafter, which prior work (Multi-Future Drafting) suggests works better.</p>
    </div>
''' + t[b:]
old = '<tr><td>Better tree selection, ceiling</td>'
if "Chunk autoencoder over target states" not in t:
    t = t.replace(old, '<tr><td>Chunk autoencoder over target states</td><td><span class="pill no">Fails gate</span>94 / 93 / 78% vs 99%+</td><td>ChainFlow, CALM</td></tr>\n        ' + old)
open("template.html", "w", encoding="utf-8").write(t)
open("index.html", "w", encoding="utf-8").write(t.replace("__SUMMARY__", open("summary.json", encoding="utf-8").read()))
print("ok", t.count('id="aeGate"'))
