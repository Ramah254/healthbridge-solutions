"""Procedural soundtrack for the long (≈56 s) HealthBridge explainer (src/long.html).
Scene times are read from the HTML so music/SFX cues stay in sync with the timeline.
Writes audio/long_music.wav, long_sfx.wav, long_mix.wav (48 kHz stereo).
"""
import re, os
from audiolib import *

html = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'long.html')).read()
def grab(line_pat):
    m = re.search(line_pat, html); return {k: float(v) for k, v in re.findall(r'(\w+)\s*=\s*([\d.]+)', m.group(0))}
sc = grab(r'const A = [^;]+;'); st = grab(r'const S1 = [^;]+;')
A, B, C, D, E, F, G, END = (sc[k] for k in ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'END'))
S1, S2, S3, S3END = (st[k] for k in ('S1', 'S2', 'S3', 'S3END'))
N = int(SR * END)
BAR, BEAT = 2.4, .6
IMPACT = C + .3                      # light background fully covers the screen -> resolution

M = np.zeros((2, N)); S = np.zeros((2, N))

# ================= MUSIC =================
# --- tension (0 → IMPACT): drone that grinds, heartbeat, thin tone, riser
td = T(IMPACT + .7)
drone = np.sin(2 * np.pi * 55 * td) + .6 * np.sin(2 * np.pi * 82.4 * td * (1 + .002 * np.sin(2 * np.pi * .3 * td)))
drone += np.linspace(0, 1, len(td)) ** 1.6 * .7 * np.sin(2 * np.pi * 116.5 * td)
drone = signal.sosfilt(signal.butter(2, 380, 'low', fs=SR, output='sos'), drone)
drone *= np.minimum(1, td / 1.5) * np.clip((IMPACT + .7 - td) / .6, 0, 1) * (.06 + .05 * np.linspace(0, 1, len(td)))
add(M, drone, 0, 1.0)
t = 0.0
while t < IMPACT - .6:
    g = .12 + .22 * (t / IMPACT)
    add(M, thump(75, 38, .35, .1) * g, t, 1.0); add(M, thump(70, 36, .3, .08) * g * .55, t + .19, 1.0)
    t += BEAT
thin = np.sin(2 * np.pi * 1760 * T(10.6)) * (1 + .35 * np.sin(2 * np.pi * 5.5 * T(10.6)))
thin = thin * np.linspace(0, 1, len(thin)) ** 2 * .016
add(M, thin, 4.2, 1.0, .35); add(M, thin * .8, 4.7, 1.0, -.35)
add(M, riser(1.8, 400, 9000, 2.2) * .5, IMPACT - 1.8, 1.0)

# --- resolution: 100 bpm, 2.4 s bars from IMPACT
prog = [('C', 0), ('Am', 0), ('F', 0), ('G', 0)]
CH = {
  'C':  (['C3','G3','D4','E4'],      'C2', ['C4','E4','G4','C5','E5']),
  'Am': (['A2','E3','C4','A3'],      'A1', ['A3','C4','E4','A4','C5']),
  'F':  (['F2','C3','A3','E4'],      'F2', ['F3','A3','C4','F4','A4']),
  'G':  (['G2','D3','G3','B3','D4'], 'G1', ['G3','B3','D4','G4','B4']),
}
order = ['C','Am','F','G'] * 4 + ['C']        # 16 bars + final C (END sets how many are used)
nb = len(order)
for b, name in enumerate(order):
    t0 = IMPACT + b * BAR
    if t0 >= END: break
    voicing, root, arp = CH[name]
    final = (b == nb - 1)
    d = (END - t0 + .5) if final else BAR + .15
    add(M, pad([nf(n) for n in voicing], d, att=.05 if b == 0 else (.6 if final else .35), rel=1.0, cut=1500) * (1.1 if final else 1.0), t0, 1.0)
    bn = nf(root)
    for off, ln, gg in ([(0, 1.1, .26), (1.5, .6, .18)] if not final else [(0, 3.0, .26)]):
        tt = T(ln); bx = (np.sin(2*np.pi*bn*tt) + .55*np.sin(2*np.pi*bn*2*tt) + .25*np.sin(2*np.pi*bn*3*tt)) * np.minimum(1, tt/.01) * np.clip((ln-tt)/.15, 0, 1) * gg
        add(M, bx, t0 + off, 1.0)
    dr_off = (46.5 <= t0 < 49.7)                 # breath before the call to action
    # arps: from bar 1; louder through the impact + why sections
    if b >= 1 and not (final and True):
        pat = [0, 2, 1, 3, 2, 4, 3, 2]
        lift = 1.15 if (E - 1 <= t0 < G) else 1.0
        for i, pi_ in enumerate(pat):
            g = (.2 if b < 8 else .24) * lift * (1.0 if i % 2 == 0 else .75) * (.7 if dr_off else 1.0)
            add(M, pluck(nf(arp[pi_]), 1.1) * g, t0 + i * .3, 1.0, -.3 if i % 2 else .3)
            if b >= 8 and i % 4 == 0 and not dr_off:    # octave sparkle
                add(M, pluck(nf(arp[pi_]) * 2, .9, .995, .6) * g * .35, t0 + i * .3, 1.0, .4 if i % 2 else -.4)
    if final:
        for i, pi_ in enumerate([0, 2, 1]):
            add(M, pluck(nf(arp[pi_]), 1.8) * .16, t0 + i * .3, 1.0, -.3 if i % 2 else .3)
    # drums: kick on 1 & 3 from bar 2, claps from bar 4, shaker from bar 3, thinned before the CTA
    if 2 <= b and not final and not dr_off:
        add(M, thump(120, 46, .45, .12) * .3, t0, 1.0)
        add(M, thump(110, 46, .4, .1) * .22, t0 + 2 * BEAT, 1.0)
    if b >= 4 and not final and not dr_off:
        for bt in (1, 3): add(M, noise_hit(.16, 1200, 6500, .045) * .2, t0 + bt * BEAT, 1.0, .1)
    if b >= 3 and not final:
        for i in range(8):
            add(M, noise_hit(.05, 6000, 14000, .012) * (.12 if i % 2 else .07), t0 + i * .3, 1.0, .25 if i % 2 else -.25)
# impact + final landing
add(M, thump(105, 33, 1.6, .5, 1.3) * .4, IMPACT, 1.0)
add(M, noise_hit(.9, 3000, 14000, .3) * .16, IMPACT, 1.0)
add(M, bell(nf('C6'), 3.0, 1.2) * .22, IMPACT, 1.0, .2)
add(M, riser(1.0, 700, 9000, 2.0) * .35, G - 1.0, 1.0)                 # lift into the CTA
add(M, thump(105, 34, 1.4, .45, 1.2) * .34, G, 1.0)
add(M, bell(nf('G6'), 2.4, 1.0) * .14, G + .1, 1.0, .25)
FIN = IMPACT + (nb - 1) * BAR
add(M, thump(100, 34, 1.5, .45, 1.2) * .3, FIN, 1.0)
add(M, bell(nf('C6'), 3.2, 1.4) * .26, FIN, 1.0, -.15); add(M, bell(nf('E6'), 2.6, 1.1) * .13, FIN + .12, 1.0, .25)

# ================= SFX =================
penta = [nf(n) for n in ['C5','D5','E5','G5','A5','C6','D6','E6']]
def P(f, t, g=.09, d=.12, pan=0.0, tau=.03, glide=1.35): add(S, pop(f, d, glide, tau) * g, t, 1.0, pan)
def W(t, d, f0, f1, g, pan=0.0, q=1.0, curve=1.3): add(S, whoosh(d, f0, f1, q, curve) * g, t, 1.0, pan)
def TH(t, g, f0=110, f1=48, d=.5, tau=.14): add(S, thump(f0, f1, d, tau) * g, t, 1.0)
def BL(f, t, g, d=1.4, tau=.55, pan=0.0): add(S, bell(f, d, tau) * g, t, 1.0, pan)
rp = lambda lo, hi: rng.uniform(lo, hi)

# --- A  the leak
for i in range(10): P(520 + 38 * i, .1 + .12 + i * .08, .09, .12, rp(-.6, .6))
TH(1.3, .45, 95, 40); add(S, noise_hit(.4, 800, 5000, .12) * .12, 1.3, 1.0)
P(880, 1.8 + .25, .09)
for i, tt in enumerate([2.4, 2.5, 2.6, 2.7]): P(300 - 30 * i, tt, .1, .2, [-.5, .5, -.3, .3][i], .06, .8)
W(3.0, 1.5, 2600, 300, .28, -.2, curve=1.3); W(3.05, 1.5, 2200, 400, .22, .35)
W(4.95, .6, 600, 3800, .3, curve=1.5)
# --- B  the cost
W(5.3, .8, 500, 2500, .18, curve=1.5)
for i in range(20): P(penta[(i * 3) % 8] * .5, 5.75 + i * .03 + .1, .05, .1, rp(-.7, .7), .03, 1.1)
TH(5.9, .45, 110, 50, .45, .12)
for k in range(8):
    tt = 6.4 + k * .15 + .15
    add(S, thump(200, 90, .22, .05, 1.4) * .32, tt, 1.0, rp(-.5, .5)); P(1244, tt + .01, .035, .09, rp(-.5, .5), .02, 1.0)
TH(9.2, .5, 110, 50, .45, .12)
W(11.45, .6, 3000, 600, .16, curve=1.2)
for i in range(6): P(nf('E4') * (1 + i * .12), 11.95 + i * .1 + .1, .07, .12, -.5 + i * .2)
TH(12.0, .55, 110, 50, .6, .16)
add(S, noise_hit(.5, 1500, 14000, .09) * .5, 13.6, 1.0)
add(S, ssum(pop(1244, .5, 1.0, .25) * .09, pop(1319, .5, 1.0, .25) * .09), 13.6, 1.0)
TH(13.75, .6, 120, 40, 1.0, .35)
W(C - .7, 1.0, 400, 5200, .4, curve=1.0)
# --- C  the bridge
W(C + .5, .8, 700, 3800, .22, curve=1.4)
for i, f in enumerate([nf(n) for n in ['C5','D5','E5','G5','A5','C6','D6','E6','G6','A6']]):
    add(S, pluck(f, .9, .995, .6) * .09, C + .9 + i * .11, 1.0, -.5 + i * .11)
for i in range(11): P(1200 + 60 * abs(5 - i), C + 1.8 + abs(5 - i) * .035 + .1, .035, .05, (i - 5) * .12, .015, 1.0)
BL(nf('C6'), C + 2.2, .3, 2.2, 1.0); BL(nf('E6'), C + 2.26, .14, 1.8, .8, .3); TH(C + 2.2, .4, 95, 42, .6, .16)
W(C + 2.5, .9, 1500, 5200, .1, curve=1.3)
W(C + 4.3, .9, 800, 3000, .24, curve=1.2)
# --- D  how it works
P(1000, D + .2 + .15, .08)
W(S1, .9, 1200, 3500, .08, curve=1.4)
W(S1 + .3, .7, 500, 2200, .12, curve=1.3); TH(S1 + .4, .22, 100, 46, .35, .1)
for i in range(3): P(700 + 90 * i, S1 + 1.0 + i * .22 + .1, .08, .1, -.4 + i * .4)
BL(nf('G5'), S1 + 1.5, .16, 1.6, .7); add(S, ssum(pop(900, .1, 1.2, .03) * .06), S1 + 1.55, 1.0)
P(900, S1 + 2.0 + .1, .08, pan=-.3); P(1100, S1 + 2.25 + .1, .08, pan=.3)
W(S2 - .6, .6, 2500, 900, .12, curve=1.2)
P(1100, S2 + .1 + .2, .08); P(1100, S3 + .1 + .2, .08)
W(S2 + .2, 1.1, 400, 2600, .3, .5, curve=1.0)
P(620, S2 + 1.2 + .08, .1, pan=.5)
for i in range(4): P(520, S2 + 1.3 + i * .1, .05, .07, .5, .02, 1.0)
add(S, ssum(pop(880, .09, 1.2, .03) * .16, pop(1320, .12, 1.2, .045) * .12), S2 + 1.85, 1.0, .55)
W(S2 + 3.7, .35, 2500, 900, .12, .5, curve=1.2); P(1040, S2 + 3.75, .12, pan=.1)
add(S, ssum(pop(1100, .07, 1.0, .03) * .12, pop(740, .09, 1.0, .03) * .1), S2 + 4.2, 1.0, .5)
P(1500, S2 + 4.7, .08, .06, .5, .02, 1.0)
BL(nf('E6'), S2 + 4.65, .2, 1.2, .5, -.1); BL(nf('A6'), S2 + 4.75, .16, 1.2, .5, .1)
W(S3 - .7, .7, 2500, 500, .22, .3, curve=1.0)
W(S3 + .3, .9, 500, 2800, .2, curve=1.4)
for i in range(4):
    P(780 + 80 * i, S3 + 1.0 + i * .35 + .1, .08, .1, -.3 + i * .2)
    for j in range(6): P(penta[j + (i % 2)], S3 + 1.2 + i * .35 + j * .16, .035, .07, -.4 + i * .25, .02, 1.0)
    BL(nf('G6'), S3 + 2.15 + i * .35, .09, .9, .35, -.4 + i * .25)
W(S3END, .6, 2500, 500, .2, curve=1.0)
# --- E  impact
TH(E + .2, .35, 100, 42)
nt = [E + 1.3 + f * 1.9 for f in (0, .2, .4, .6, .8, 1)]
for i, tt in enumerate(nt): BL([nf(n) for n in ['C5','D5','E5','G5','A5','C6']][i], tt, .22, 1.4, .55, -.5 + i * .2)
W(nt[2] + .05, .55, 1200, 4800, .2, curve=1.0)
for i in range(3):
    W(E + 2.0 + i * .85, .8, 500, 2600, .16, curve=1.3); P(880 + 140 * i, E + 2.0 + i * .85 + .3, .08)
W(F - .7, .6, 2500, 500, .2, curve=1.0)
# --- F  why owners choose us
TH(F + .35, .3, 100, 44)
for i in range(4):
    tt = F + 1.1 + i * 1.5
    W(tt, .9, 500, 2400, .13, -.3 + i * .2, curve=1.3); BL([nf(n) for n in ['E6','G6','A6','C7']][i], tt + .35, .12, 1.0, .4, -.3 + i * .2)
P(1100, F + 6.6 + .15, .07)
add(S, riser(1.0, 600, 9000, 1.8) * .3, G - 1.0, 1.0); W(G - 1.0, 1.1, 3200, 500, .3, curve=1.0)
# --- G  call to action
W(G + .2, .9, 800, 3500, .14, curve=1.2)
TH(G + .3, .28, 95, 44, .4, .1); TH(G + .55, .28, 95, 44, .4, .1)
P(700, G + 1.3 + .3, .12, .12, 0, .04, 1.3)
P(1000, G + 1.8 + .25, .07)
W(G + 2.0, .8, 1800, 600, .1, .4, curve=1.2)
P(1100, G + 2.5 + .1, .06); P(1100, G + 2.9 + .1, .06)
cl = np.zeros(int(.06 * SR)); cl[:int(.0015 * SR)] = rng.uniform(-1, 1, int(.0015 * SR))
add(S, ssum(cl * .5, pop(1500, .05, 1.0, .015) * .1), G + 3.0, 1.0, .1)
BL(nf('C7'), G + 3.05, .14, 1.5, .6); W(G + 3.05, .9, 1200, 6000, .12, curve=1.0)

# ================= MASTER =================
_hp = signal.butter(2, 38, 'high', fs=SR, output='sos')
def lim(x): return np.tanh(signal.sosfilt(_hp, x, axis=1) * 1.1) / np.tanh(1.1)
def fade_out(x, t0):
    i = int(t0 * SR); x[:, i:] *= np.linspace(1, 0, x.shape[1] - i) ** 1.5; return x
M = fade_out(M, END - 1.6); S = fade_out(S, END - 1.0)
M, S = lim(M), lim(S)
def norm(x, peak): return x / (np.abs(x).max() + 1e-9) * peak
music, sfx = norm(M, .6), norm(S, .5)
mix = norm(lim(M * .85 + S * .8), .89)
for name, x in (('long_music', music), ('long_sfx', sfx), ('long_mix', mix)):
    wavfile.write(os.path.join(OUT, name + '.wav'), SR, (x.T * 32767).astype(np.int16))
print('wrote long audio: %.1fs peak %.3f rms %.3f' % (END, float(np.abs(mix).max()), float(np.sqrt((mix ** 2).mean()))))
