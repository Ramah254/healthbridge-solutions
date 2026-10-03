"""Shared synthesis primitives for the HealthBridge soundtrack scripts (audio.py, audio_long.py)."""
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

