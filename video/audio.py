"""Procedural soundtrack for the 18 s HealthBridge motion piece (no samples, no licensing).
Cue times mirror the GSAP timeline in src/index.html. Writes music.wav, sfx.wav, mix.wav (48 kHz stereo).
"""
import numpy as np
from scipy import signal
from scipy.io import wavfile
import os

SR, DUR = 48000, 18.0
N = int(SR * DUR)
rng = np.random.default_rng(11)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audio')
os.makedirs(OUT, exist_ok=True)

def T(d): return np.arange(int(d * SR)) / SR
def mid2f(m): return 440.0 * 2 ** ((m - 69) / 12)
NOTE = {'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}
def nf(name):  # 'C4' -> Hz
    return mid2f(12 * (int(name[-1]) + 1) + NOTE[name[0]] + (1 if '#' in name else 0))

def add(buf, x, t, gain=1.0, pan=0.0):
    i = int(round(t * SR))
    if i >= buf.shape[1] or i < 0: return
    n = min(len(x), buf.shape[1] - i)
    a = (pan + 1) * np.pi / 4
    buf[0, i:i + n] += x[:n] * gain * np.cos(a)
    buf[1, i:i + n] += x[:n] * gain * np.sin(a)

def env_ad(n, att=.005, tau=.1):
    t = np.arange(n) / SR
    return np.minimum(1, t / max(att, 1e-4)) * np.exp(-t / tau)

# ---------------- primitives ----------------
def pop(f=700, d=.14, glide=1.35, tau=.03):
    t = T(d); fr = f * (1 + (glide - 1) * np.exp(-t / .02))
    return np.sin(2 * np.pi * np.cumsum(fr) / SR) * env_ad(len(t), .002, tau)

def thump(f0=130, f1=42, d=.8, tau=.22, drive=1.0):
    t = T(d); fr = f1 + (f0 - f1) * np.exp(-t / .045)
    x = np.sin(2 * np.pi * np.cumsum(fr) / SR) * env_ad(len(t), .002, tau)
    return np.tanh(x * drive * 1.6) / np.tanh(1.6)

def noise_hit(d=.3, hp=2000, lp=12000, tau=.08):
    x = rng.standard_normal(int(d * SR))
    sos = signal.butter(2, [hp, min(lp, SR / 2 * .95)], 'band', fs=SR, output='sos')
    return signal.sosfilt(sos, x) * env_ad(len(x), .001, tau)

def bell(f, d=2.0, tau=.9, bright=1.0):
    t = T(d); out = np.zeros_like(t)
    for r, a, k in [(1, 1, 1), (2.76, .45, .55), (5.4, .22 * bright, .3), (8.93, .1 * bright, .18)]:
        out += a * np.sin(2 * np.pi * f * r * t) * np.exp(-t / (tau * k))
    return out * np.minimum(1, t / .003) * .5

def pluck(f, d=1.4, g=.996, bright=.5):
    D = int(round(SR / f)); n = int(d * SR)
    x = np.zeros(n); x[:D] = rng.uniform(-1, 1, D)
    x[:D] = signal.lfilter([1 - bright, bright], [1], x[:D])
    a = np.zeros(D + 2); a[0] = 1; a[D] = -g / 2; a[D + 1] = -g / 2
    y = signal.lfilter([1], a, x)
    return y * np.minimum(1, np.arange(n) / (SR * .004)) * np.exp(-np.arange(n) / SR / (d * .55))

def whoosh(d, f0, f1, q=1.1, curve=2.0, blk=256):
    n = int(d * SR); x = rng.standard_normal(n); out = np.zeros(n); zi = None
    for s in range(0, n, blk):
        f = f0 * (f1 / f0) ** (s / n); bw = f / q
        lo, hi = max(80, f - bw / 2), min(SR * .45, f + bw / 2)
        sos = signal.butter(2, [lo, hi], 'band', fs=SR, output='sos')
        if zi is None: zi = np.zeros((sos.shape[0], 2))
        y, zi = signal.sosfilt(sos, x[s:s + blk], zi=zi); out[s:s + blk] = y
    out /= (np.sqrt(np.mean(out ** 2)) + 1e-9)
    return out * np.sin(np.pi * np.linspace(0, 1, n)) ** curve * .5

def riser(d, f0=300, f1=9000, curve=2.4):
    n = int(d * SR); x = rng.standard_normal(n); out = np.zeros(n); zi = None; blk = 512
    for s in range(0, n, blk):
        f = f0 * (f1 / f0) ** (s / n)
        sos = signal.butter(2, min(f, SR * .45), 'high', fs=SR, output='sos')
        if zi is None: zi = np.zeros((sos.shape[0], 2))
        y, zi = signal.sosfilt(sos, x[s:s + blk], zi=zi); out[s:s + blk] = y
    out /= (np.sqrt(np.mean(out ** 2)) + 1e-9)
    return out * np.linspace(0, 1, n) ** curve * .5

def pad(freqs, d, att=.4, rel=.6, cut=1400, harm=14):
    t = T(d + rel); y = np.zeros_like(t)
    for f in freqs:
        for det in (-.0045, 0, .0045):
            ff = f * (1 + det)
            for k in range(1, harm + 1):
                if ff * k > cut * 2.5: break
                y += np.sin(2 * np.pi * ff * k * t + rng.uniform(0, 6.28)) / k ** 1.25
    sos = signal.butter(2, cut, 'low', fs=SR, output='sos'); y = signal.sosfilt(sos, y)
    e = np.minimum(1, t / att) * np.minimum(1, np.clip((d + rel - t) / rel, 0, 1))
    return y * e / (len(freqs) * 3)


def ssum(*xs):
    n = max(len(x) for x in xs); out = np.zeros(n)
    for x in xs: out[:len(x)] += x
    return out

# ---------------- music bus ----------------
M = np.zeros((2, N))

# --- tension: 0–6 s (drone + heartbeat + thin tone + riser)
td = T(6.4)
drone = (np.sin(2 * np.pi * 55 * td) + .6 * np.sin(2 * np.pi * 82.4 * td * (1 + .002 * np.sin(2 * np.pi * .3 * td))))
drone += np.linspace(0, 1, len(td)) ** 1.6 * .7 * np.sin(2 * np.pi * 116.5 * td)   # b-flat grinds against A
drone = signal.sosfilt(signal.butter(2, 380, 'low', fs=SR, output='sos'), drone)
drone *= np.minimum(1, td / 1.0) * np.clip((6.4 - td) / .6, 0, 1) * (.07 + .05 * np.linspace(0, 1, len(td)))
add(M, drone, 0, 1.0)
t = 0.0; k = 0
while t < 5.7:
    g = .14 + .2 * (t / 5.7)
    add(M, thump(75, 38, .35, .1) * g, t, 1.0); add(M, thump(70, 36, .3, .08) * g * .55, t + .19, 1.0)
    t += .6; k += 1
thin = np.sin(2 * np.pi * 1760 * T(4.2)) * (1 + .35 * np.sin(2 * np.pi * 5.5 * T(4.2)))
thin = thin * np.linspace(0, 1, len(thin)) ** 2 * .018
add(M, thin, 1.6, 1.0, .35); add(M, thin * .8, 1.8, 1.0, -.35)
add(M, riser(1.0, 500, 9000, 2.2) * .5, 5.0, 1.0)

# --- resolution: 6.0–18 s, 100 bpm, bar = 2.4 s
BAR, BEAT = 2.4, .6
chords = [
  (['C3','G3','D4','E4'], 'C', ['C4','E4','G4','C5','E5']),
  (['A2','E3','C4','A3'], 'A', ['A3','C4','E4','A4','C5']),
  (['F2','C3','A3','E4'], 'F', ['F3','A3','C4','F4','A4']),
  (['G2','D3','G3','B3','D4'], 'G', ['G3','B3','D4','G4','B4']),
  (['C3','G3','D4','E4','C5'], 'C', ['C4','E4','G4','C5','E5']),
]
roots = {'C': 'C2', 'A': 'A1', 'F': 'F2', 'G': 'G1'}
for b, (voicing, rt, arp) in enumerate(chords):
    t0 = 6.0 + b * BAR
    d = BAR + (1.2 if b == 4 else .15)
    add(M, pad([nf(n) for n in voicing], d, att=.35 if b else .05, rel=.9, cut=1500) * (1.0 if b < 4 else 1.1), t0, 1.0)
    # bass: root on beat 1 and the "and" of 3
    bn = nf(roots[rt])
    if b >= 1 or True:
        for off, ln, gg in ([(0, 1.1, .26), (1.5, .6, .18)] if b < 4 else [(0, 2.4, .26)]):
            tt = T(ln); bx = (np.sin(2 * np.pi * bn * tt) + .55 * np.sin(2 * np.pi * bn * 2 * tt) + .25 * np.sin(2 * np.pi * bn * 3 * tt)) * np.minimum(1, tt / .01) * np.clip((ln - tt) / .15, 0, 1) * gg
            add(M, bx, t0 + off, 1.0)
    # arpeggio (pluck), 8ths, starts with the logo (bar 1 second half) so bar 0 breathes
    if b >= 1:
        pat = [0, 2, 1, 3, 2, 4, 3, 2]
        for i, pi_ in enumerate(pat):
            tt = t0 + i * .3
            if b == 4 and i > 2: break
            g = (.24 if b < 4 else .18) * (1.0 if i % 2 == 0 else .75)
            add(M, pluck(nf(arp[pi_]), 1.5 if b == 4 else 1.1) * g, tt, 1.0, (-.3 if i % 2 else .3))
    # kick on 1 and 3, clap on 2 & 4 (from bar 2 onward)
    if 1 <= b <= 3:
        add(M, thump(120, 46, .45, .12) * .3, t0, 1.0)
        add(M, thump(110, 46, .4, .1) * .22, t0 + 2 * BEAT, 1.0)
        if b >= 2:
            for bt in (1, 3):
                add(M, noise_hit(.16, 1200, 6500, .045) * .22, t0 + bt * BEAT, 1.0, .1)
    # shaker 8ths from 9.0
    if b >= 1:
        for i in range(8):
            tt = t0 + i * .3
            if tt < 9.0: continue
            add(M, noise_hit(.05, 6000, 14000, .012) * (.12 if i % 2 else .07), tt, 1.0, .25 if i % 2 else -.25)
# impact on the bridge reveal
add(M, thump(105, 33, 1.6, .5, 1.3) * .4, 6.0, 1.0)
add(M, noise_hit(.9, 3000, 14000, .3) * .16, 6.0, 1.0)
add(M, bell(nf('C6'), 3.0, 1.2) * .22, 6.0, 1.0, .2)
# final resolve
add(M, bell(nf('C6'), 2.8, 1.3) * .26, 15.6, 1.0, -.15)
add(M, bell(nf('G6'), 2.4, 1.0) * .14, 15.72, 1.0, .25)
add(M, thump(100, 34, 1.4, .45, 1.2) * .32, 15.5, 1.0)

# ---------------- sfx bus ----------------
S = np.zeros((2, N))
penta = [nf(n) for n in ['C5','D5','E5','G5','A5','C6','D6','E6']]

# S1 : persons pop in, one by one
for i in range(10):
    add(S, pop(520 + 38 * i, .12) * .09, .05 + i * .055 + .12, 1.0, rng.uniform(-.6, .6))
add(S, thump(95, 40, .5, .14) * .45, 1.0, 1.0)                       # "…and never come back." hit
add(S, noise_hit(.4, 800, 5000, .12) * .12, 1.0, 1.0)
for i, tt in enumerate([1.3, 1.37, 1.44, 1.51]):                       # four patients grey out
    add(S, pop(300 - 30 * i, .2, .8, .06) * .1, tt, 1.0, [-.5, .5, -.3, .3][i])
add(S, whoosh(1.0, 2600, 300, 1.0, 1.3) * .28, 1.78, 1.0, -.2)        # they walk away
add(S, whoosh(1.0, 2200, 400, 1.0, 1.3) * .22, 1.82, 1.0, .35)
add(S, pop(880, .12) * .09, 1.95 + .18, 1.0)                           # stat chip
add(S, whoosh(.45, 600, 3800, 1.2, 1.5) * .3, 2.68, 1.0)               # → calendar
# S2 : calendar
add(S, whoosh(.5, 500, 2500, 1.2, 1.5) * .18, 2.85, 1.0)
for i in range(20):
    add(S, pop(penta[(i * 3) % 8] * .5, .1, 1.1, .03) * .05, 3.1 + i * .024 + .1, 1.0, rng.uniform(-.7, .7))
add(S, thump(110, 50, .45, .12) * .45, 3.08, 1.0)
for k in range(8):                                                     # no-shows: dull "tok" + dissonant tick
    tt = 3.5 + k * .1 + .12
    add(S, thump(200, 90, .22, .05, 1.4) * .32, tt, 1.0, rng.uniform(-.5, .5))
    add(S, pop(1244, .09, 1.0, .02) * .035, tt + .01, 1.0, rng.uniform(-.5, .5))
add(S, thump(110, 50, .45, .12) * .5, 3.98, 1.0)
add(S, thump(110, 50, .6, .16) * .55, 4.98, 1.0)
add(S, whoosh(.3, 3000, 600, 1.0, 1.2) * .16, 4.84, 1.0)
for i in range(6):                                                     # timeline nodes
    add(S, pop(nf('E4') * (1 + i * .12), .12) * .07, 5.08 + i * .06 + .1, 1.0, -.5 + i * .2)
crack = noise_hit(.5, 1500, 14000, .09) * .5 + .0
add(S, crack, 5.5, 1.0)
add(S, ssum(pop(1244, .5, 1.0, .25) * .09, pop(1319, .5, 1.0, .25) * .09), 5.5, 1.0)  # dissonant crack tone
add(S, thump(120, 40, 1.0, .35) * .6, 5.56, 1.0)                       # "!" badge boom
add(S, whoosh(.7, 400, 5200, 1.0, 1.0) * .4, 5.74, 1.0)                # iris
# S3 : bridge
add(S, whoosh(.55, 700, 3800, 1.1, 1.4) * .22, 6.3, 1.0)               # deck
glide = [nf(n) for n in ['C5','D5','E5','G5','A5','C6','D6','E6','G6','A6']]
for i, f in enumerate(glide):                                          # harp glissando as the arch draws
    add(S, pluck(f, .9, .995, .6) * .09, 6.5 + i * .065, 1.0, -.5 + i * .11)
for i in range(11):                                                    # hangers
    add(S, pop(1200 + 60 * abs(5 - i), .05, 1.0, .015) * .035, 6.85 + abs(5 - i) * .025 + .1, 1.0, (i - 5) * .12)
add(S, bell(nf('C6'), 2.2, 1.0) * .3, 7.1, 1.0)                        # logo lands
add(S, bell(nf('E6'), 1.8, .8) * .14, 7.16, 1.0, .3)
add(S, thump(95, 42, .6, .16) * .4, 7.1, 1.0)
add(S, whoosh(.5, 1500, 5200, 1.0, 1.3) * .1, 7.3, 1.0)                # wordmark
add(S, whoosh(.55, 800, 3000, 1.0, 1.2) * .24, 8.5, 1.0)               # logo → corner
# S4 : phone, chat
add(S, whoosh(.7, 400, 2600, 1.0, 1.0) * .3, 8.95, 1.0, .5)
for tt in (9.1, 9.25):
    add(S, whoosh(.35, 1200, 3500, 1.2, 1.4) * .08, tt, 1.0)
add(S, pop(620, .1) * .1, 9.6 + .08, 1.0, .5)                          # typing
for i in range(4):
    add(S, pop(520, .07, 1.0, .02) * .05, 9.7 + i * .1, 1.0, .5)
add(S, ssum(pop(880, .09, 1.2, .03) * .16, pop(1320, .12, 1.2, .045) * .12), 10.02, 1.0, .55)  # message in
for i in range(3):
    add(S, pop(700 + 120 * i, .1) * .09, 10.15 + i * .42 + .1, 1.0, -.3 + i * .15)
add(S, whoosh(.3, 2500, 900, 1.0, 1.2) * .12, 10.75, 1.0, .5)          # language flip
add(S, pop(1040, .1, 1.0, .03) * .12, 10.8, 1.0, .1)
add(S, ssum(pop(1100, .07, 1.0, .03) * .12, pop(740, .09, 1.0, .03) * .1), 11.47, 1.0, .5)      # reply sent
add(S, pop(1500, .06, 1.0, .02) * .08, 11.86, 1.0, .5)
add(S, bell(nf('E6'), 1.2, .5) * .2, 11.82, 1.0, -.1); add(S, bell(nf('A6'), 1.2, .5) * .16, 11.92, 1.0, .1)  # visit confirmed
add(S, whoosh(.5, 2500, 500, 1.0, 1.0) * .22, 12.3, 1.0, .3)
# S5 : impact
add(S, thump(100, 42, .5, .14) * .35, 12.55, 1.0)
nt = [12.98 + f * .95 for f in (0, .2, .4, .6, .8, 1)]
for i, tt in enumerate(nt):                                            # checkmarks climb a scale
    add(S, bell([nf(n) for n in ['C5','D5','E5','G5','A5','C6']][i], 1.4, .55) * .22, tt, 1.0, -.5 + i * .2)
add(S, whoosh(.34, 1200, 4800, 1.0, 1.0) * .2, 13.38, 1.0)             # bridge arc closes the gap
add(S, whoosh(.6, 500, 2800, 1.0, 1.4) * .2, 13.7, 1.0)                # report card
for i in range(3):
    for j in range(8):
        add(S, pop(penta[j] , .07, 1.0, .02) * .04, 14.1 + i * .12 + j * .1, 1.0, -.4 + i * .4)
    add(S, bell(nf('G6'), .9, .35) * .1, 14.85 + i * .12, 1.0, -.4 + i * .4)
# transition to CTA
add(S, riser(.6, 600, 9000, 1.8) * .3, 14.95, 1.0)
add(S, whoosh(.6, 3200, 500, 1.0, 1.0) * .3, 15.1, 1.0)
# S6 : CTA
add(S, whoosh(.5, 800, 3500, 1.1, 1.2) * .14, 15.62, 1.0)
for tt in (15.68, 15.84):
    add(S, thump(95, 44, .4, .1) * .28, tt, 1.0)
add(S, pop(700, .12, 1.3, .04) * .12, 16.1 + .22, 1.0)                 # button
add(S, whoosh(.45, 1800, 600, 1.0, 1.2) * .1, 16.15, 1.0, .4)          # cursor glide
cl = np.zeros(int(.06 * SR)); cl[:int(.0015 * SR)] = rng.uniform(-1, 1, int(.0015 * SR))
add(S, ssum(cl * .5, pop(1500, .05, 1.0, .015) * .1), 16.62, 1.0, .1)       # click
add(S, bell(nf('C7'), 1.5, .6) * .14, 16.66, 1.0); add(S, whoosh(.6, 1200, 6000, 1.0, 1.0) * .12, 16.66, 1.0)
add(S, pop(1320, .1, 1.0, .03) * .07, 16.85 + .2, 1.0)

# ---------------- master ----------------
def fade_out(x, t0):
    i = int(t0 * SR); x[:, i:] *= np.linspace(1, 0, x.shape[1] - i) ** 1.5
    return x
_hp = signal.butter(2, 38, 'high', fs=SR, output='sos')
def lim(x): return np.tanh(signal.sosfilt(_hp, x, axis=1) * 1.1) / np.tanh(1.1)
M = fade_out(M, 16.9); S = fade_out(S, 17.3)
def norm(x, peak): return x / (np.abs(x).max() + 1e-9) * peak
M, S = lim(M), lim(S)
music = norm(M, .6); sfx = norm(S, .5)
mix = lim(M * .85 + S * .8)
mix = norm(mix, .89)
for name, x in (('music', music), ('sfx', sfx), ('mix', mix)):
    wavfile.write(os.path.join(OUT, name + '.wav'), SR, (x.T * 32767).astype(np.int16))
print('wrote audio: peak', float(np.abs(mix).max()), 'rms', float(np.sqrt((mix ** 2).mean())))
