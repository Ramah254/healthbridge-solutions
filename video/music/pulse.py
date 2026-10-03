#!/usr/bin/env python3
"""pulse.py -- "Bridge" score for the HealthBridge Solutions videos (direction: pulse).

Modern, warm, confident product-explainer pulse: Bb major (the analyzer spells it "A# major"), 110.5 BPM, 4/4.
Procedural synthesis only (numpy + scipy). Everything is composed symbolically -- a chord list, a theme note list,
groove patterns -- and rendered through instrument functions, so key, harmony and timing are right by construction.

    python3 pulse.py [--out-dir DIR] [--debug]

writes  <out>/pulse_long_music.wav (57.6 s)  and  <out>/pulse_short_music.wav (18.0 s); 48 kHz, stereo, 16-bit.
Deterministic: one seeded numpy Generator per video, no clocks, no threads. Renders both files in about 15 s.

Pitch frame.  All MIDI numbers in the tables are written in G major (G = 67) and sound TRANSPOSE = +3 semitones higher,
i.e. in Bb major.  (Written G-major pitch classes therefore map to Bb: G->Bb, A->C, B->D, C->Eb, D->F, E->G, F#->A.)
Theme, sounding pitches (Bb major), 4 bars, every note on the 8th-note grid:
   bar1  I     F5 (1.5) G5 .5  F5 1  D5 .5  C5 .5       "question": neighbour-note arch that falls to the 9th
   bar2  vi7   D5 1  F5 1  Bb5 1.5  A5 .5 (or 1.5)       the "bridge" leap D5-F5-Bb5 (a 4th up) = peak; A (#11 of Eb) hangs
   bar3  IV    G5 1.5  F5 .5  D5 1  Eb5 .5  F5 .5        "answer": falls by step, passing Eb-F pushes into V
   bar4  V     A5 1.5  G5 .5  F5 2                        open half-close; the next I statement resolves it
Tension uses the same figure darkened (D->Db, G->Gb, A->Ab: parallel minor); IMPACT is the moment the third goes
from Db back to D -- the "light" arrives.  The bass is a TONIC PEDAL (Bb) through every chord, dark or bright: the one
constant that bridges the two worlds (and it keeps the low end tonally unambiguous).

Time frame.  One global beat grid anchored on IMPACT (beat 0, bars every 4 beats), so a strong downbeat lands on IMPACT in
both videos.  Groove and melody sit on the 8th-note grid; each section's first chord/stinger sits on the exact cue time
(the final chord is quantised to the nearest 8th, ~30 ms before the button click).

Voiceover room (revision 2).  The bed is built so that the 300-3400 Hz band is moderate AND phrased:
  * warm layer: near-sine low triads (150-300 Hz, 'warm' bus) + a bass 2nd partial carry the body, so the thin top can rest;
  * melody in phrases: the glass lead sings bars 1+3 (F) / bars 1-2 (E) and rests otherwise, a marimba answers one octave
    down at low level; third/octave layers only on the phrase peak; phrase dynamics (soft question -> crescendo to the leap);
  * arps are thinned after IMPACT (3 onsets per bar in D/F, 5 elsewhere, bar 4 of every 4 rests in D/F);
  * the shaker is a quiet tick at 3.3-6 kHz (nothing above ~6.5 kHz): each octave from 2 kHz up sits >= 3 dB under the
    previous one (the old 4.5-7.8 kHz shaker made an inverted tilt); the lead is brighter in its 2nd/3rd partial so the
    presence octave is carried by the melody instead of by hiss.
"""
import os
import sys
import math
import argparse
import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import minimum_filter1d, uniform_filter1d

SR = 48000
BPM = 110.5
BEAT = 60.0 / BPM
KEY = 'Bb major'                # analyzer spelling: 'A# major'
VID = {
    'long':  dict(dur=57.6, imp=15.4, seed=1101),
    'short': dict(dur=18.0, imp=6.0, seed=4417),
}
TARGET_RMS_DB = -21.0          # mono RMS of the body (impact .. start of final fade); sets the loudness

# ----------------------------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------------------------
TL = 4096


def _tab(amps):
    ph = np.arange(TL) / TL
    y = np.zeros(TL)
    for k, a in enumerate(amps, 1):
        y += a * np.sin(2 * np.pi * k * ph)
    return y / np.max(np.abs(y))


SAW = _tab([1.0 / k for k in range(1, 25)])
SAWS = _tab([1.0 / k ** 1.5 for k in range(1, 25)])                      # softer saw (pads, drone, bass)
TRI = _tab([((-1) ** ((k - 1) // 2)) / k ** 2 if k % 2 else 0.0 for k in range(1, 25)])


def osc(tab, f, n, ph0=0.0):
    """Wavetable oscillator. f scalar or array (Hz)."""
    if np.ndim(f) == 0:
        ph = ph0 + (f / SR) * np.arange(n)
    else:
        ph = ph0 + np.cumsum(f) / SR
    x = (ph % 1.0) * TL
    i = x.astype(np.int64) % TL
    fr = x - np.floor(x)
    return tab[i] * (1 - fr) + tab[(i + 1) % TL] * fr


TRANSPOSE = 3     # semitones; the score is written in a G-major frame and sounds TRANSPOSE semitones higher


def mtof(m):
    return 440.0 * 2.0 ** ((np.asarray(m, dtype=float) + TRANSPOSE - 69.0) / 12.0)


def env_note(n_hold, n_rel, att, dec=None, sus=1.0):
    n = n_hold + n_rel
    t = np.arange(n) / SR
    a = np.sin(np.clip(t / max(att, 1e-4), 0, 1) * np.pi / 2)
    if dec is not None:
        a = a * (sus + (1 - sus) * np.exp(-np.maximum(t - att, 0) / dec))
    if n_rel > 0:
        r = np.ones(n)
        r[n_hold:] = 0.5 * (1 + np.cos(np.pi * np.arange(n_rel) / n_rel))
        a = a * r
    return a


_SOS = {}


def sos_of(kind, *fc, order=2):
    key = (kind, fc, order)
    if key not in _SOS:
        _SOS[key] = signal.butter(order, list(fc) if len(fc) > 1 else fc[0], kind, fs=SR, output='sos')
    return _SOS[key]


def lp(x, fc, order=2):
    return signal.sosfilt(sos_of('low', fc, order=order), x)


def hp(x, fc, order=2):
    return signal.sosfilt(sos_of('high', fc, order=order), x)


def bp(x, lo, hi, order=2):
    return signal.sosfilt(sos_of('band', lo, hi, order=order), x)


def _bank(lo, hi, seed, dur=4.0, order=2):
    r = np.random.default_rng(seed)
    y = bp(r.standard_normal(int(SR * dur)), lo, hi, order)
    return y / np.sqrt(np.mean(y ** 2))


SHAKN = lp(_bank(3300, 6000, 7, order=3), 6500)       # shaker / hats: 3.3-6 kHz, nothing above ~6.5 kHz (no hiss shelf)
SHAKN /= np.sqrt(np.mean(SHAKN ** 2))
CLAPN = _bank(950, 3300, 8)        # clap body
SNAPN = _bank(1700, 3900, 9)       # finger snap
TICKN = _bank(2600, 6200, 10)      # clock ticks
CRASHN = _bank(2200, 7600, 11)     # soft crash
MALN = lp(np.random.default_rng(12).standard_normal(SR * 2), 2600)
MALN /= np.sqrt(np.mean(MALN ** 2))


def nseg(bank, n, r):
    off = int(r.integers(0, len(bank) - n - 1))
    return bank[off:off + n]


class Bus:
    def __init__(self, n):
        self.L = np.zeros(n, np.float32)
        self.R = np.zeros(n, np.float32)
        self.n = n

    def add(self, x, t, g=1.0, pan=0.0):
        i = int(round(t * SR))
        m = len(x)
        s0 = max(0, -i)
        e0 = min(m, self.n - i)
        if e0 <= s0:
            return
        a = (pan + 1.0) * math.pi / 4
        seg = x[s0:e0]
        self.L[i + s0:i + e0] += seg * (g * math.cos(a))
        self.R[i + s0:i + e0] += seg * (g * math.sin(a))

    def add2(self, xl, xr, t, g=1.0):
        i = int(round(t * SR))
        m = len(xl)
        s0 = max(0, -i)
        e0 = min(m, self.n - i)
        if e0 <= s0:
            return
        self.L[i + s0:i + e0] += xl[s0:e0] * g
        self.R[i + s0:i + e0] += xr[s0:e0] * g

    def mul(self, curve):
        self.L *= curve
        self.R *= curve


# ----------------------------------------------------------------------------------------------------------------
# instruments
# ----------------------------------------------------------------------------------------------------------------
def marimba(f, vel, r):
    """Modal marimba/pizzicato-mallet: partials 1 : 4 : 9.9, upper modes decay fast, short wooden knock."""
    tau0 = float(np.clip(0.50 * (261.6 / f) ** 0.5, 0.13, 0.9))
    dur = min(6 * tau0, 1.5)
    n = int(dur * SR)
    t = np.arange(n) / SR
    y = np.sin(2 * np.pi * f * t) * np.exp(-t / tau0)
    if f * 4.0 < 9000:
        y += 0.50 * vel * np.sin(2 * np.pi * 4.0 * f * t + 0.3) * np.exp(-t / (tau0 * 0.30))
    if f * 9.9 < 9000:
        y += 0.08 * vel * np.sin(2 * np.pi * 9.9 * f * t + 1.1) * np.exp(-t / (tau0 * 0.09))
    k = int(0.006 * SR)
    y[:k] += nseg(MALN, k, r) * np.exp(-t[:k] / 0.0016) * 0.20 * vel
    y *= np.minimum(1.0, t / 0.0008) * np.minimum(1.0, (dur - t) / 0.05)
    return y * (0.30 + 0.70 * vel)


def glass(f, hold, vel, r, bright=1.6, vib=0.0035, rel=0.34):
    """Clean glassy lead: soft FM onset 'ting' fading into a near-sine/additive tone, delayed vibrato."""
    n_hold = max(int(hold * SR), 64)
    n_rel = int(rel * SR)
    n = n_hold + n_rel
    t = np.arange(n) / SR
    vdep = vib * np.clip((t - 0.20) / 0.35, 0, 1)
    ph = 2 * np.pi * np.cumsum(f * (1 + vdep * np.sin(2 * np.pi * 5.1 * t + r.uniform(0, 6.28)))) / SR
    idx = 1.2 * bright * np.exp(-t / 0.05)
    y = np.sin(ph + idx * np.sin(2 * ph))
    y += (0.26 + 0.20 * bright) * np.sin(2 * ph + 0.4) + (0.08 + 0.14 * bright) * np.sin(3 * ph + 1.1) + (0.03 + 0.06 * bright) * np.sin(4 * ph)
    if f * 2.756 < 7500:
        y += 0.07 * bright * np.sin(2 * np.pi * f * 2.756 * t) * np.exp(-t / 0.10)
    return 0.89 * y * env_note(n_hold, n_rel, 0.012, 0.35, 0.78) * vel


def dark_pluck(f, vel, r, hold=0.4):
    """Muted pizzicato-synth: detuned saws through a closing low-pass."""
    n = int((hold + 0.3) * SR)
    t = np.arange(n) / SR
    x = osc(SAW, f, n, r.uniform()) + 0.55 * osc(SAW, f * 1.006, n, r.uniform())
    a = np.exp(-t / 0.07)
    y = lp(x, 650) * (1 - a) + lp(x, 3000) * a
    return y * np.exp(-t / 0.22) * (1 - np.exp(-t / 0.003)) * np.minimum(1.0, (n / SR - t) / 0.06) * vel * 0.55


def bass_note(f, hold, vel):
    n_hold = int(hold * SR)
    n_rel = int(0.10 * SR)
    n = n_hold + n_rel
    x = 0.55 * osc(TRI, f, n) + 0.55 * osc(SAWS, f, n) + 0.25 * osc(SAW, f * 1.003, n)
    if f < 150.0:                                           # root notes: add a clean 2nd partial (the 150-300 Hz 'warm' octave)
        x = x + 0.60 * osc(TRI, 2.0 * f, n)
    x = lp(x, 850)
    x = np.tanh(1.6 * x) / np.tanh(1.6)
    return x * env_note(n_hold, n_rel, 0.006, 0.30, 0.55) * vel


def kick(vel=1.0, f0=190.0, f1=54.0, tau_f=0.026, tau_a=0.16, d=0.55, drive=1.3, click=0.10, r=None):
    n = int(d * SR)
    t = np.arange(n) / SR
    f = f1 + (f0 - f1) * np.exp(-t / tau_f)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * (1 - np.exp(-t / 0.0012)) * np.exp(-t / tau_a)
    if r is not None and click > 0:
        k = int(0.004 * SR)
        x[:k] += nseg(MALN, k, r) * np.exp(-t[:k] / 0.0012) * click
    x = np.tanh(drive * x) / np.tanh(drive)
    return x * np.minimum(1.0, (d - t) / 0.05) * vel


def clap(vel, r, snap=0.0):
    """Hand clap / finger snap: three bursts within ~11 ms fuse into one attack, then a smooth short tail."""
    n = int(0.30 * SR)
    t = np.arange(n) / SR
    body = nseg(CLAPN, n, r)
    env = np.zeros(n)
    for o, w, tau in ((0.0, 0.55, 0.004), (0.005, 0.75, 0.004), (0.011, 1.0, 0.030)):
        k = int(o * SR)
        env[k:] += w * np.exp(-(t[:n - k]) / tau)
    y = body * env
    if snap > 0:
        m = int(0.05 * SR)
        s_ = nseg(SNAPN, m, r) * np.exp(-t[:m] / 0.010) * 0.9 + np.sin(2 * np.pi * 1750 * t[:m]) * np.exp(-t[:m] / 0.006) * 0.35
        y[:m] += snap * s_
    y = lp(y, 3800)
    return y * (1 - np.exp(-t / 0.0007)) * np.minimum(1.0, (0.30 - t) / 0.04) * vel * 0.5


def shaker(vel, accent, r):
    n = int((0.16 if accent else 0.11) * SR)
    t = np.arange(n) / SR
    env = (1 - np.exp(-t / 0.007)) * np.exp(-t / (0.028 if accent else 0.020))
    return nseg(SHAKN, n, r) * env * vel * np.minimum(1.0, (n / SR - t) / 0.02)


def tick(vel, kind, r):
    n = int(0.06 * SR)
    t = np.arange(n) / SR
    f = 2350.0 if kind == 0 else 1550.0
    y = 0.55 * nseg(TICKN, n, r) * np.exp(-t / 0.004) + 0.9 * np.sin(2 * np.pi * f * t) * np.exp(-t / 0.011)
    return y * (1 - np.exp(-t / 0.0007)) * np.minimum(1.0, (0.06 - t) / 0.01) * vel


def thud(vel, r):
    return kick(vel, f0=105.0, f1=62.0, tau_f=0.025, tau_a=0.11, d=0.40, drive=1.0, click=0.0, r=r)


def crash(d, vel, r):
    n = int(d * SR)
    t = np.arange(n) / SR
    y = nseg(CRASHN, n, r) * np.exp(-t / (d / 4.2)) * (1 - np.exp(-t / 0.004))
    y = lp(y, 7600)
    return y * np.minimum(1.0, (d - t) / 0.08) * vel


def riser(d, f0, f1, r, curve=2.0):
    n = int(d * SR)
    x = r.standard_normal(n)
    out = np.zeros(n)
    zi = None
    blk = 480
    for s in range(0, n, blk):
        fc = f0 * (f1 / f0) ** (s / n)
        sos = signal.butter(2, [fc * 0.55, min(fc * 1.7, SR * 0.45)], 'band', fs=SR, output='sos')
        if zi is None:
            zi = np.zeros((sos.shape[0], 2))
        y, zi = signal.sosfilt(sos, x[s:s + blk], zi=zi)
        out[s:s + blk] = y
    out /= (np.sqrt(np.mean(out ** 2)) + 1e-9)
    env = (np.arange(n) / n) ** curve
    env *= np.minimum(1.0, (n - np.arange(n)) / (0.03 * SR))
    return out * env * 0.6


def uplift(d, f0, f1, r):
    """Tonal riser: detuned saw stack gliding up with an opening low-pass (no noise, so it adds no random flux)."""
    n = int(d * SR)
    t = np.arange(n) / SR
    u = t / d
    f = f0 * (f1 / f0) ** u
    y = np.zeros(n)
    for det in (-10.0, -3.5, 3.5, 10.0):
        y += osc(SAWS, f * 2.0 ** (det / 1200.0), n, r.uniform())
    y += 0.55 * osc(SAWS, f * 1.5, n, r.uniform())
    y = lp(y, 650) * (1 - u ** 1.5) + lp(y, 2400) * u ** 1.5
    env = u ** 1.7 * np.minimum(1.0, (d - t) / 0.03)
    return y * env / 3.0


def sweep_down(d, f0, f1, r):
    """Tonal 'light wipe': a stack of sines gliding down (no noise, so no random spectral flux)."""
    n = int(d * SR)
    t = np.arange(n) / SR
    u = t / d
    f = f0 * (f1 / f0) ** u
    y = np.zeros(n)
    for ratio, a in ((1.0, 1.0), (1.5, 0.55), (2.0, 0.35), (0.5, 0.5)):
        y += a * np.sin(2 * np.pi * np.cumsum(f * ratio) / SR + r.uniform(0, 6.28))
    return y * (1 - np.exp(-t / 0.006)) * np.exp(-t / (d / 2.6)) * np.minimum(1.0, (d - t) / 0.05) * 0.35


def bell(f, d, vel):
    """Glass chime: near-harmonic partials with fast-decaying upper modes."""
    n = int(d * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for ratio, amp, k in ((1.0, 1.0, 0.50), (2.0, 0.42, 0.30), (3.0, 0.16, 0.20), (4.07, 0.12, 0.13), (5.43, 0.05, 0.08)):
        if f * ratio < 8000:
            y += amp * np.sin(2 * np.pi * f * ratio * t + 0.2 * ratio) * np.exp(-t / (d * k))
    return y * np.minimum(1.0, t / 0.002) * np.minimum(1.0, (d - t) / 0.1) * vel * 0.5


def pad_chord(tones, dur, att, rel, cut, r, det=2.5, width=0.75, tab=None):
    """Wide clean pad: 2 detuned wavetable voices per tone, voices panned apart, low-passed. Returns (L, R)."""
    tab = SAWS if tab is None else tab
    n = int((dur + rel) * SR)
    L = np.zeros(n)
    R = np.zeros(n)
    for j, m in enumerate(tones):
        f = float(mtof(m))
        for s, sign in enumerate((-1.0, 1.0)):
            y = osc(tab, f * 2.0 ** (sign * det / 1200.0), n, r.uniform())
            pan = sign * width * (1.0 if (j % 2 == 0) else 0.6)
            a = (pan + 1.0) * math.pi / 4
            L += y * math.cos(a)
            R += y * math.sin(a)
    sos = sos_of('low', cut, order=2)
    L = signal.sosfilt(sos, L)
    R = signal.sosfilt(sos, R)
    n_hold = int(dur * SR)
    e = env_note(n_hold, n - n_hold, att)
    g = 1.0 / (len(tones) * 1.4)
    return L * e * g, R * e * g


def rev_chord(tones, d, r, cut=2600):
    """Reverse swell: a decaying pad chord played backwards (rises to its peak at the end)."""
    n = int(d * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for m in tones:
        f = float(mtof(m))
        y += osc(SAWS, f * 0.997, n, r.uniform()) + osc(SAWS, f * 1.003, n, r.uniform())
    y = lp(y, cut) * np.exp(-t / (d / 3.2)) / (len(tones) * 1.6)
    y = y[::-1].copy()
    y *= np.minimum(1.0, np.arange(n) / (0.12 * SR))
    y[-int(0.02 * SR):] *= np.linspace(1, 0, int(0.02 * SR))
    return y


def drone(n, f_root, r, T_total):
    """Low drone that tightens: detune spread narrows, tremolo speeds up, filter opens."""
    t = np.arange(n) / SR
    u = np.clip(t / T_total, 0, 1) ** 1.4
    spread = 28.0 * (1 - u) + 1.5 * u                      # cents
    outs = []
    for root, amp in ((f_root, 1.0), (f_root * 1.5, 0.55), (f_root * 2.0, 0.35)):
        y = np.zeros(n)
        for m in (-1.0, -0.5, 0.0, 0.5, 1.0):
            y += osc(SAWS, root * 2.0 ** (spread * m / 1200.0), n, r.uniform())
        outs.append(amp * y / 5.0)
    y = sum(outs)
    lo = lp(y, 260)
    hi = lp(y, 1000)
    y = lo * (1 - u) + hi * u
    rate = 0.35 + 4.2 * u
    trem = 1 - (0.12 + 0.28 * u) * (0.5 + 0.5 * np.sin(2 * np.pi * np.cumsum(rate) / SR))
    return y * trem


# ----------------------------------------------------------------------------------------------------------------
# reverb / master
# ----------------------------------------------------------------------------------------------------------------
def make_ir(rt60, seed, pre=0.015, damp=5200):
    r = np.random.default_rng(seed)
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    outs = []
    for ch in range(2):
        z = r.standard_normal(n)
        lo = lp(z, 2200)
        hi = z - lo
        ir = lo * np.exp(-6.91 * t / rt60) + 0.55 * hi * np.exp(-6.91 * t / (rt60 * 0.45))
        ir = lp(ir, damp)
        ir *= (1 - np.exp(-t / 0.010))
        for k in range(9):                                   # sparse early reflections
            i = int((0.007 + 0.075 * r.uniform() ** 0.8) * SR)
            if i < n:
                ir[i] += (0.9 - 0.07 * k) * (1 if r.uniform() > 0.5 else -1) * np.std(ir[:int(0.15 * SR)]) * 3.0
        ir = np.concatenate([np.zeros(int(pre * SR)), ir])
        outs.append(ir / np.sqrt(np.sum(ir ** 2)))
    return outs


def conv_send(sendL, sendR, irs):
    n = len(sendL)
    mono = (sendL.astype(np.float64) + sendR.astype(np.float64)) * 0.7071
    wl = signal.fftconvolve(mono, irs[0])[:n]
    wr = signal.fftconvolve(mono, irs[1])[:n]
    return wl, wr


def limiter(x, ceil, look=int(0.008 * SR)):
    pk = np.max(np.abs(x), axis=0)
    g = np.minimum(1.0, ceil / np.maximum(pk, 1e-9))
    gm = minimum_filter1d(g, size=2 * look + 1, mode='nearest')
    gs = uniform_filter1d(gm, size=2 * look + 1, mode='nearest')
    return x * gs


# ----------------------------------------------------------------------------------------------------------------
# harmony / theme data
# ----------------------------------------------------------------------------------------------------------------
CH = {
    # chords by roman numeral (written in the G-major frame; sounding pitch = written + TRANSPOSE):
    # pad voicing, arp tone set (ascending), bell tones. The bass is a tonic pedal (see TONIC_BASS).
    'I':    dict(pad=[62, 67, 71, 74, 79], arp=[62, 67, 71, 74, 79, 83], bell=[79, 83, 86]),
    'vi7':  dict(pad=[64, 67, 71, 74, 76], arp=[64, 67, 71, 74, 76, 79], bell=[76, 79, 83]),
    'IV':   dict(pad=[60, 67, 71, 76, 79], arp=[64, 67, 71, 72, 76, 79], bell=[79, 76, 83]),
    'V':    dict(pad=[62, 66, 69, 74, 78], arp=[62, 66, 69, 74, 78, 81], bell=[78, 81, 74]),
    'Vsus': dict(pad=[62, 67, 69, 74, 79], arp=[62, 67, 69, 74, 79, 81], bell=[79, 81, 74]),
    'i':    dict(pad=[55, 62, 67, 70, 74], arp=[62, 67, 70, 74, 79, 82], bell=[79, 82, 86]),
    'bVI':  dict(pad=[55, 63, 67, 70, 74], arp=[63, 67, 70, 74, 75, 79], bell=[75, 79, 82]),
    'iv':   dict(pad=[55, 63, 67, 72, 74], arp=[63, 67, 72, 74, 75, 79], bell=[75, 79, 84]),
    'V7sus': dict(pad=[57, 62, 67, 69, 72], arp=[62, 67, 69, 72, 74, 79], bell=[79, 81, 72]),
}

WARM = {                              # low dyads/triads, sounding Eb3..C4 (155-262 Hz): warmth that stays out of the voice band
    'I': [50, 55], 'vi7': [52, 55], 'IV': [48, 55], 'V': [50, 54, 57], 'Vsus': [50, 55, 57],
}
TONIC_BASS = 43                       # written G2 (sounds Bb2 with TRANSPOSE = 3)
G_SCALE = [7, 9, 11, 0, 2, 4, 6]       # G A B C D E F#


def scale_shift(m, steps):
    pc = m % 12
    idx = G_SCALE.index(pc)
    deg = (m // 12) * 7 + idx + steps
    return (deg // 7) * 12 + G_SCALE[deg % 7]


def dark(m):
    return m - 1 if (m % 12) in (11, 4, 6) else m          # B->Bb, E->Eb, F#->F


T1 = [(0.0, 74, 1.5), (1.5, 76, 0.5), (2.0, 74, 1.0), (3.0, 71, 0.5), (3.5, 69, 0.5),
      (4.0, 71, 1.0), (5.0, 74, 1.0), (6.0, 79, 1.5), (7.5, 78, None)]
T2 = [(0.0, 76, 1.5), (1.5, 74, 0.5), (2.0, 71, 1.0), (3.0, 72, 0.5), (3.5, 74, 0.5),
      (4.0, 78, 1.5), (5.5, 76, 0.5), (6.0, 74, 2.0)]


def theme(which, b0, last=0.5, upto=None):
    src = T1 if which == 1 else T2
    out = []
    for o, m, d in src:
        if upto is not None and o >= upto:
            continue
        out.append((b0 + o, m, last if d is None else d))
    return out


def grid(b0, b1, step=0.5):
    k0 = math.ceil(b0 / step - 1e-6)
    k1 = math.ceil(b1 / step - 1e-6)
    return [k * step for k in range(k0, k1)]


def keys(video):
    v = VID[video]
    imp, dur = v['imp'], v['dur']
    c = lambda t: (t - imp) / BEAT
    K = dict(S=c(0.0), END=c(dur), C0=0.0, cut=-0.25)
    if video == 'long':
        K.update(B0=-17.5, crack=c(13.6), D0=c(20.1), Dg=9.0, E0=c(36.3), Eg=39.0, F0=c(43.0), G0=67.0, Gf=72.5)         # final chord on the 8th grid: ~30 ms before the click
    else:
        K.update(B0=-5.5, crack=c(5.5), D0=c(9.0), Dg=6.0, E0=c(12.5), Eg=12.0, G0=17.5, Gf=19.5)            # final chord on the 8th grid: ~31 ms before the click
    return K


def chord_list(video, K):
    if video == 'long':
        return [('i', K['B0'], -12), ('bVI', -12, -8), ('iv', -8, -4), ('V7sus', -4, K['cut']),
                ('I', 0, 4), ('vi7', 4, K['D0']),
                ('IV', K['D0'], 13), ('V', 13, 17), ('I', 17, 21), ('vi7', 21, 25), ('IV', 25, 29), ('V', 29, 33),
                ('I', 33, 37), ('Vsus', 37, K['E0']),
                ('I', K['E0'], 43), ('vi7', 43, 47), ('IV', 47, 49), ('V', 49, K['F0']),
                ('I', K['F0'], 55), ('vi7', 55, 59), ('IV', 59, 63), ('V', 63, K['G0']),
                ('I', K['G0'], 69), ('IV', 69, 70.5), ('V', 70.5, K['Gf']), ('I', K['Gf'], K['END'] + 1)]
    return [('i', K['B0'], -3.5), ('bVI', -3.5, -2), ('V7sus', -2, K['cut']),
            ('I', 0, 4), ('vi7', 4, K['D0']),
            ('IV', K['D0'], 10), ('V', 10, K['E0']),
            ('I', K['E0'], 16), ('vi7', 16, 17.5),
            ('IV', 17.5, 18.5), ('V', 18.5, K['Gf']), ('I', K['Gf'], K['END'] + 1)]


# ----------------------------------------------------------------------------------------------------------------
# section level table (multiplies each bus; sections are cut from the cue sheet)
# ----------------------------------------------------------------------------------------------------------------
LV = {
    #           A          B          Bd         C     D     E     F     G           R
    'drone': {'A': (1.3, 1.9), 'B': (1.9, 2.2), 'Bd': (2.2, 1.7), 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'R': 0},
    'pad':   {'A': 0, 'B': (0.4, 1.2), 'Bd': (1.2, 1.6), 'C': 0.80, 'D': 0.40, 'E': 0.45, 'F': 0.30, 'G': 0.80, 'R': 1.0},
    'arp':   {'A': 0, 'B': 1.05, 'Bd': 0.0, 'C': 0.85, 'D': 0.64, 'E': 0.60, 'F': 0.42, 'G': 0.85, 'R': 0.9},
    'lead':  {'A': 1.5, 'B': 1.3, 'Bd': 1.3, 'C': 1.0, 'D': 0.8, 'E': 1.0, 'F': 0.85, 'G': 1.0, 'R': 1.0},
    'bass':  {'A': 0, 'B': 0.95, 'Bd': 0, 'C': 1.0, 'D': 0.85, 'E': 1.1, 'F': 0.9, 'G': 1.0, 'R': 1.0},
    'kick':  {'A': 1, 'B': 1, 'Bd': 1, 'C': 0.9, 'D': 0.8, 'E': 1.1, 'F': 0.85, 'G': 1.0, 'R': 1.0},
    'clap':  {'A': 1, 'B': 1, 'Bd': 1, 'C': 1.4, 'D': 0.55, 'E': 0.65, 'F': 0.55, 'G': 1.4, 'R': 1.0},
    'shk':   {'A': 1, 'B': 1, 'Bd': 1, 'C': 1.0, 'D': 0.8, 'E': 1.0, 'F': 0.8, 'G': 1.0, 'R': 1.0},
    'tick':  {'A': 2.2, 'B': 1.9, 'Bd': 1.9, 'C': 1, 'D': 1, 'E': 1, 'F': 1, 'G': 1, 'R': 1},
    'fx':    {'A': 1, 'B': 1, 'Bd': 1, 'C': 1, 'D': 1, 'E': 1, 'F': 1, 'G': 0.8, 'R': 1},
    'warm':  {'A': 0, 'B': 0, 'Bd': 0, 'C': 0.75, 'D': 0.70, 'E': 1.15, 'F': 0.80, 'G': 1.0, 'R': 1.0},
}
BG = dict(drone=0.132, pad=0.25, arp=0.335, lead=0.222, bass=0.244, kick=0.44, clap=1.022, shk=0.28, tick=0.813, fx=0.51, warm=0.42)
HALL = dict(drone=0.30, pad=0.35, arp=0.30, lead=0.42, bass=0.0, kick=0.0, clap=0.18, shk=0.10, tick=0.22, fx=0.45, warm=0.20)
ROOM = dict(drone=0.0, pad=0.0, arp=0.0, lead=0.0, bass=0.0, kick=0.04, clap=0.0, shk=0.0, tick=0.0, fx=0.0, warm=0.0)
HALL_RET = 0.30
ROOM_RET = 0.30
PUMP = dict(pad=0.42, arp=0.14, bass=0.22, warm=0.25)


PAD_CUT = {'C': 1500, 'D': 1000, 'E': 1500, 'F': 1000, 'G': 1700}      # pad low-pass (Hz) by section

TRIM = {
    'long':  {'G': 1.30},
    'short': {'E': 1.0, 'G': 1.55},
}


def section_list(video, K):
    """[(name, first beat, last beat)] -- the cue-sheet sections on the global beat grid (beat 0 = IMPACT)."""
    secs = [('A', K['S'], K['B0']), ('B', K['B0'], K['crack']), ('Bd', K['crack'], K['cut']), ('C', K['cut'], K['D0']),
            ('D', K['D0'], K['E0'])]
    if video == 'long':
        secs += [('E', K['E0'], K['F0']), ('F', K['F0'], K['G0']), ('G', K['G0'], K['Gf'])]
    else:
        secs += [('E', K['E0'], K['G0']), ('G', K['G0'], K['Gf'])]
    secs += [('R', K['Gf'], K['END'] + 5)]
    return secs


def level_curve(video, K, bus, n):
    """Per-sample gain curve for a bus from the section table."""
    imp = VID[video]['imp']
    secs = section_list(video, K)
    cur = np.ones(n)
    for name, b0, b1 in secs:
        g = LV[bus].get(name, 1.0)
        g0, g1 = (g, g) if np.ndim(g) == 0 else g
        if bus in ('pad', 'arp', 'lead', 'bass', 'shk', 'fx', 'warm'):          # sustained layers only: keeps transient peaks in check
            g0, g1 = g0 * TRIM[video].get(name, 1.0), g1 * TRIM[video].get(name, 1.0)
        i0 = max(0, int(round((imp + b0 * BEAT) * SR)))
        i1 = min(n, int(round((imp + b1 * BEAT) * SR)))
        if i1 > i0:
            cur[i0:i1] = np.linspace(g0, g1, i1 - i0)
    w = int(0.02 * SR)
    return uniform_filter1d(cur, size=w, mode='nearest')


# ----------------------------------------------------------------------------------------------------------------
# build one video
# ----------------------------------------------------------------------------------------------------------------
ARP_PATS = {
    'skip': ([0, 2, 1, 3, 2, 4, 3, 5], [1.0, 0.55, 0.80, 0.55, 0.90, 0.55, 0.80, 0.60]),
    'up':   ([0, 1, 2, 3, 4, 5, 4, 3], [1.0, 0.55, 0.75, 0.60, 0.90, 0.60, 0.80, 0.55]),
    'dark': ([0, None, 2, None, 3, None, 2, None], [1.0, 0, 0.70, 0, 0.85, 0, 0.70, 0]),
    'dark8': ([0, 2, 3, 2, 4, 3, 2, 3], [1.0, 0.5, 0.75, 0.5, 0.9, 0.55, 0.8, 0.6]),
    # post-IMPACT patterns with rests: 3 onsets per bar (8th slots 0, 3, 5 = 1 . . 2 . 3 . .) and 5 onsets per bar
    'skip3': ([0, None, None, 2, None, 4, None, None], [1.0, 0, 0, 0.78, 0, 0.92, 0, 0]),
    'up5':   ([0, None, 2, 3, None, 5, None, 3], [1.0, 0, 0.72, 0.60, 0, 0.92, 0, 0.58]),
}
BASS_PATS = {
    'drive':  [(0.0, 0, 0.90, 0.9), (0.5, 12, 0.40, 0.75), (1.0, 12, 0.40, 0.85), (1.5, 12, 0.40, 0.75),
               (2.0, 12, 0.40, 0.95), (2.5, 12, 0.40, 0.75), (3.0, 12, 0.40, 0.85), (3.5, 12, 0.40, 0.80)],
    'sparse': [(0.0, 0, 0.95, 0.9), (1.0, 12, 0.40, 0.80), (1.5, 12, 0.40, 0.75), (2.0, 12, 0.45, 0.95),
               (3.0, 12, 0.40, 0.80), (3.5, 12, 0.40, 0.75)],
    'pulse8': [(i * 0.5, 0, 0.42, 0.55 + (0.2 if i % 2 == 0 else 0)) for i in range(8)],
}


def render_buses(video):
    cfg = VID[video]
    imp, dur = cfg['imp'], cfg['dur']
    n = int(round(dur * SR))
    r = np.random.default_rng(cfg['seed'])
    K = keys(video)
    CHL = chord_list(video, K)
    T = lambda b: imp + b * BEAT
    jit = lambda ms: r.uniform(-ms, ms) / 1000.0
    names = list(BG.keys())
    bus = {k: Bus(n) for k in names}
    kick_times = []

    def chord_at(b):
        for nm, b0, b1 in CHL:
            if b0 - 1e-6 <= b < b1 - 1e-6:
                return nm, b0
        return None, None

    def seg_ref(b0):
        return math.ceil(b0 * 2 - 0.25) / 2.0

    # ---------------------------------------------------------------- drone (A, B, breakdown)
    d_b0, d_b1 = K['S'], K['cut']
    nd = int((T(d_b1) - T(d_b0)) * SR)
    dr = drone(nd, float(mtof(43)), r, T(d_b1) - T(d_b0))
    dr = dr * np.minimum(1.0, np.arange(nd) / (3.0 * SR)) * np.minimum(1.0, (nd - np.arange(nd)) / (0.04 * SR))
    drR = np.roll(dr, int(0.011 * SR)) * 0.9
    bus['drone'].add2(dr * 0.95, drR, T(d_b0), 1.0)

    # ---------------------------------------------------------------- ticking pulse (A / B)
    if video == 'long':
        tick_q = (-26.0, K['B0'])
    else:
        tick_q = (-9.0, K['B0'])
    for b in grid(tick_q[0], tick_q[1], 1.0):
        u = (b - tick_q[0]) / (tick_q[1] - tick_q[0])
        bus['tick'].add(tick(0.12 + 0.20 * u, int(round(b)) % 2, r), T(b) + jit(3), 1.0, -0.4 if int(round(b)) % 2 == 0 else 0.4)
        bus['tick'].add(tick(0.07 + 0.12 * u, 1, r), T(b + 0.5) + jit(3), 1.0, 0.25 if int(round(b)) % 2 == 0 else -0.25)     # soft off-beat tick: the clock gets restless
    for b in grid(K['B0'], K['crack'] - 0.1, 0.5):
        u = (b - K['B0']) / (K['crack'] - K['B0'])
        kind = int(round(b * 2)) % 2
        bus['tick'].add(tick(0.20 + 0.50 * u ** 1.3, kind, r), T(b) + jit(3), 1.0, -0.45 if kind == 0 else 0.45)
    for b in grid(K['crack'] + 0.2, K['cut'] - 0.4, 1.0):
        bus['tick'].add(tick(0.45, int(round(b)) % 2, r), T(b), 1.0, 0.0)

    # ---------------------------------------------------------------- tension "cost" thuds + crack
    if video == 'long':
        thuds = [5.9, 9.2, 12.0]
    else:
        thuds = [3.08, 4.98]
    for k, t in enumerate(thuds):
        bus['fx'].add(thud(0.50 + 0.12 * k, r), t, 1.0, 0.0)
        bus['fx'].add(bell(float(mtof(81 if k % 2 else 79)), 0.9, 0.10 + 0.04 * k), t, 1.0, 0.25 * (1 if k % 2 else -1))
    bus['fx'].add(bell(float(mtof(90)), 1.6, 0.30), T(K['crack']), 1.0, 0.3)           # glass crack
    bus['fx'].add(bell(float(mtof(91)), 1.2, 0.22), T(K['crack']) + 0.006, 1.0, -0.3)
    bus['fx'].add(crash(0.3, 0.03, r), T(K['crack']), 1.0, 0.0)
    bus['fx'].add(thud(0.8, r), T(K['crack']), 1.0, 0.0)
    if video == 'long':
        bus['fx'].add(bell(float(mtof(86)), 1.0, 0.12), 1.3, 1.0, 0.4)                 # headline 2
    else:
        bus['fx'].add(bell(float(mtof(86)), 1.0, 0.12), 1.0, 1.0, 0.4)

    # ---------------------------------------------------------------- pad + warm low triad (per chord)
    SEC = section_list(video, K)
    sec_at = lambda b: next((nm for nm, a, z in SEC if a - 1e-6 <= b < z - 1e-6), 'R')
    for nm, b0, b1 in CHL:
        ch = CH[nm]
        tones = ch['pad']
        darkc = nm in ('i', 'bVI', 'iv', 'V7sus')
        is_final = (b0 == K['Gf'])
        sc = sec_at(b0)
        d_b = (b1 - b0) * BEAT
        if darkc:
            att, rel, cut = 1.0, 0.8, 1000
            tn = tones
        elif is_final:
            att, rel, cut = 0.05, 1.8 if video == 'short' else 3.2, 1700
            d_b = 1.2 if video == 'short' else 2.6
            tn = tones + [81]
        else:
            att = 0.07 if b0 in (0.0, K['D0'], K['E0'], K.get('F0'), K['G0']) else 0.20
            rel, cut = 0.9, PAD_CUT.get(sc, 1700)
            tn = tones[:4] if sc in ('D', 'F') else tones        # D/F: no top voice, darker: a cushion, not a second melody
        L, Rr = pad_chord(tn, d_b + 0.15, att, rel, cut, r)
        bus['pad'].add2(L, Rr, T(b0), 1.0)
        if not darkc and sc == 'G' and not is_final:               # CTA: one held tonic dyad (the bass is a tonic pedal anyway)
            if b0 == K['G0']:
                Lw, Rw = pad_chord(WARM['I'], (K['Gf'] - K['G0']) * BEAT + 0.15, att, 0.9, 520, r, det=1.0, width=0.5, tab=TRI)
                bus['warm'].add2(Lw, Rw, T(b0), 1.0)
        elif not darkc:                                          # warm low triad: 150-300 Hz body under the thin top
            Lw, Rw = pad_chord(WARM[nm], d_b + 0.15, att, rel, 520, r, det=1.0, width=0.5, tab=TRI)
            bus['warm'].add2(Lw, Rw, T(b0), 1.0)

    # ---------------------------------------------------------------- arps (marimba / dark pizz)
    def arps(b0, b1, pat, vel, dark_voice=False, octave=0, drop=None):
        """drop=(period, index): rest the whole bar number `index` of every `period` bars (counted from b0)."""
        idxs, vels = ARP_PATS[pat]
        for b in grid(b0, b1, 0.5):
            if drop is not None and int((b - b0 + 1e-6) // 4.0) % drop[0] == drop[1]:
                continue
            nm, cs = chord_at(b)
            if nm is None:
                continue
            k = int(round((b - seg_ref(cs)) * 2)) % 8
            ix = idxs[k]
            if ix is None:
                continue
            m = CH[nm]['arp'][ix] + 12 * octave
            v = vel * vels[k] * r.uniform(0.88, 1.08)
            pan = -0.55 + 0.22 * ix + r.uniform(-0.05, 0.05)
            t = T(b) + jit(7)
            if dark_voice:
                bus['arp'].add(dark_pluck(float(mtof(m)), v, r), t, 1.0, pan)
            else:
                bus['arp'].add(marimba(float(mtof(m)), v, r), t, 1.0, pan)

    # ---------------------------------------------------------------- bass
    def bass_line(b0, b1, pat, vel=1.0):
        for nm, cb0, cb1 in CHL:
            lo, hi = max(b0, cb0), min(b1, cb1)
            if hi <= lo:
                continue
            root = TONIC_BASS                       # tonic pedal: harmony moves above, the bass stays on the home note
            for o, iv, d, v in BASS_PATS[pat]:
                b = (seg_ref(cb0) if cb0 >= b0 - 1e-6 else b0) + o
                if b < lo - 1e-6 or b >= hi - 1e-6:
                    continue
                dd = min(d, hi - b) * BEAT * 0.97
                bus['bass'].add(bass_note(float(mtof(root + iv)), dd, v * vel * r.uniform(0.92, 1.05)), T(b) + jit(4), 1.0, 0.0)

    # ---------------------------------------------------------------- drums
    def kicks(b0, b1, vel, step=1.0, accent_first=True):
        for b in grid(b0, b1, step):
            v = vel * (1.0 if (int(round(b)) % 4 == (int(round(b0)) % 4)) and accent_first else 0.85)
            t = T(b) + jit(2)
            bus['kick'].add(kick(v, r=r), t, 1.0, 0.0)
            kick_times.append(t)

    def claps(b0, b1, vel, snap=0.35, step=2.0):
        for b in grid(b0, b1, step):
            t = T(b) + jit(4)
            pan = 0.12 * (1 if int(round(b)) % 4 == 1 or int(round(b)) % 4 == 2 else -1)
            bus['clap'].add(clap(vel * r.uniform(0.9, 1.05), r, snap), t, 1.0, pan)

    def shakers(b0, b1, vel, ghosts=0.0):
        for b in grid(b0, b1, 0.25 if ghosts > 0 else 0.5):
            on8 = abs(b * 2 - round(b * 2)) < 1e-6
            if on8:
                off = abs(b - round(b)) > 1e-6
                v = vel * (0.95 if off else 0.85)
                acc = off
            else:
                v = ghosts
                acc = False
            if v <= 0:
                continue
            side = -1 if int(math.floor(b * 4 + 1e-6)) % 2 == 0 else 1
            bus['shk'].add(shaker(v * r.uniform(0.85, 1.1), acc, r), T(b) + jit(3), 1.0, 0.35 * side)

    # ---------------------------------------------------------------- stingers / transitions
    def stinger(b, strength, tones, boom=0.0, wipe=False, tcue=None):
        t = T(b) if tcue is None else tcue
        bus['fx'].add(crash(0.5 + 0.7 * strength, 0.09 * strength, r), t, 1.0, 0.0)
        for j, m in enumerate(tones):
            bus['fx'].add(bell(float(mtof(m)), 1.1 + 0.9 * strength, 0.26 * strength), t + 0.012 * j, 1.0, (-0.5, 0.0, 0.5)[j % 3])
        if boom > 0:
            bus['kick'].add(kick(boom, f0=150.0, f1=48.0, tau_f=0.045, tau_a=0.34, d=0.9, drive=1.2, r=r), t, 1.0, 0.0)
            kick_times.append(t)
        if wipe:
            bus['fx'].add(sweep_down(1.3, 2800, 520, r), t, 0.45, 0.0)

    def rise(b0, b1, f0=500, f1=7500, gain=0.5, curve=2.0, lo=294.0, hi=1175.0, noise=0.0):
        # tonal riser only (noise=0): a noise riser adds hiss above 4 kHz and random spectral flux that smears the 8th-note grid
        d = (T(b1) - T(b0))
        bus['fx'].add(uplift(d, lo, hi, r), T(b0), gain * 1.1, 0.0)
        if noise > 0:
            x = riser(d, f0, f1, r, curve)
            bus['fx'].add(x, T(b0), gain * noise, 0.0)

    def lead_notes(notes, voice='glass', vel=0.8, shift=0, pan=0.0, legato=0.94, hum=6, bright=1.6):
        for b, m, d in notes:
            f = float(mtof(m + shift))
            hold = max(d * BEAT * legato, 0.06)
            t = T(b) + jit(hum)
            v = vel * r.uniform(0.92, 1.05)
            if voice == 'glass':
                bus['lead'].add(glass(f * 0.9985, hold, v, r, bright=bright), t, 0.55, pan - 0.22)
                bus['lead'].add(glass(f * 1.0015, hold, v, r, bright=bright), t, 0.55, pan + 0.22)
            elif voice == 'soft':
                bus['lead'].add(glass(f, hold, v, r, bright=0.5, vib=0.004), t, 0.8, pan)
            elif voice == 'dark':
                bus['lead'].add(dark_pluck(f, v, r, hold=min(hold, 0.6)), t, 1.0, pan)
                bus['lead'].add(glass(f, hold, v * 0.35, r, bright=0.2, vib=0.006), t, 0.8, pan)
            elif voice == 'mar':
                bus['arp'].add(marimba(f, v, r), t, 1.0, pan)

    def lead_arc(notes, vels, cuts, **kw):
        """Phrase dynamics for the glass lead: velocity vels[i] for notes before cuts[i] (beats), vels[-1] after the last cut."""
        lo = -1e9
        for v, hi in zip(vels, list(cuts) + [1e9]):
            lead_notes([n for n in notes if lo <= n[0] < hi], 'glass', v, **kw)
            lo = hi

    def harm_below(notes, steps=-2):
        return [(b, scale_shift(m, steps), d) for b, m, d in notes]

    def darken(notes):
        return [(b, dark(m), d) for b, m, d in notes]

    def peaks(notes, lo=78):
        """Phrase-peak notes only (written pitch >= lo: the Bb5/A5 'bridge' leap and its landing) -- the third / octave
        layers are reserved for these so the rest of a phrase stays a single thin line."""
        return [n for n in notes if n[1] >= lo]

    def bars(notes, b0, keep):
        """Notes whose onset lies in bar numbers `keep` (0-based, bars of 4 beats counted from beat b0)."""
        return [n for n in notes if int((n[0] - b0 + 1e-6) // 4.0) in keep]

    # ================================================================ arrangement (long / short)
    if video == 'long':
        # ---- A: hint of the theme in the dark, then B tension
        lead_notes(darken([(-21.0, 74, 1.5), (-19.5, 76, 0.5), (-19.0, 74, 1.0), (-18.0, 71, 0.5)]), 'dark', 0.55, pan=0.2)
        # B: dark arps start (quarters, then 8ths), dark bass pulse, dark theme fragments
        arps(-16.0, -8.0, 'dark', 0.55, True)
        arps(-8.0, K['crack'] - 0.25, 'dark8', 0.60, True)
        bass_line(-12.0, K['crack'] - 0.25, 'pulse8', 0.65)
        lead_notes(darken([(-16.0, 74, 1.5), (-14.5, 76, 0.5), (-14.0, 74, 1.0), (-13.0, 71, 0.5), (-12.5, 69, 0.5)]), 'dark', 0.62, pan=-0.15)
        lead_notes(darken([(-8.0, 71, 1.0), (-7.0, 74, 1.0), (-6.0, 79, 1.5), (-4.5, 78, 0.5)]), 'dark', 0.70, pan=0.15)
        lead_notes([(-4.0, 67, 0.62)], 'soft', 0.35)
        rise(K['crack'] + 0.1, -0.30, 700, 6500, 0.55, 2.2)
        bus['fx'].add(rev_chord(CH['V7sus']['pad'] + [74], T(0) - T(K['crack']) - 0.04, r), T(K['crack']), 0.55, 0.0)
        # ---- C: IMPACT
        stinger(0.0, 1.0, [79, 86, 83, 91], boom=1.0, wipe=True)
        stinger(3.5, 0.45, [86, 91], tcue=17.3)                 # logo lands
        lead_arc(theme(1, 0.0, last=1.5), (0.62, 0.74, 0.92), (4.0, 6.0), bright=2.0)       # question soft -> leap peak
        lead_notes(peaks(theme(1, 0.0, last=1.5)), 'soft', 0.32, shift=12)          # octave lift on the phrase peak only
        arps(0.0, K['D0'] - 0.3, 'up5', 0.62)
        bass_line(0.0, K['D0'], 'drive')
        kicks(0.0, 8.0, 0.9, 2.0)
        kicks(3.5, 8.0, 0.45, 4.0, False)
        claps(1.0, 8.0, 0.8)
        shakers(0.5, K['D0'], 0.55, 0.0)
        bus['fx'].add(rev_chord(CH['vi7']['pad'], 1.0, r), T(K['D0']) - 1.0, 0.35, 0.0)
        # ---- D: how it works
        stinger(K['D0'], 0.55, [79, 83], tcue=T(K['D0']))
        lead_notes(theme(2, K['Dg']), 'glass', 0.58)
        arps(K['Dg'], K['E0'] - 0.5, 'skip3', 0.52, drop=(4, 3))                     # 3 onsets/bar, bar 4 of every 4 rests
        bass_line(K['D0'] + 0.01, K['E0'] - 0.4, 'sparse')
        kicks(K['Dg'], 37.0, 0.72)
        claps(K['Dg'] + 1.0, 36.5, 0.66, 0.5)
        shakers(K['Dg'], 36.5, 0.50, 0.0)
        lead_notes([(21.0, 74, 1.5), (22.5, 76, 0.5), (23.0, 74, 1.0)], 'soft', 0.40)               # motif x echo at step 2
        lead_notes([(33.0, 71, 1.0), (34.0, 74, 1.0), (35.0, 79, 1.5)], 'soft', 0.42)               # leap figure at step 3
        bus['fx'].add(bell(float(mtof(86)), 1.2, 0.10), 25.1, 1.0, 0.3)
        bus['fx'].add(bell(float(mtof(83)), 1.2, 0.10), 28.8, 1.0, -0.3)
        bus['fx'].add(bell(float(mtof(86)), 1.2, 0.10), 31.5, 1.0, 0.3)
        for b in grid(37.0, 38.3, 0.5):
            u = (b - 37.0) / 1.3
            bus['clap'].add(clap(0.45 + 0.4 * u, r, 0.3), T(b), 1.0, 0.0)
        rise(36.2, K['E0'] - 0.1, 600, 7500, 0.55, 2.0)
        bus['fx'].add(rev_chord(CH['I']['pad'], 1.1, r), T(K['E0']) - 1.1 - 0.02, 0.5, 0.0)
        # ---- E: the impact (mothers & babies)
        stinger(K['E0'], 0.9, [79, 86, 83], boom=0.8, wipe=True)
        e_t1 = theme(1, K['Eg'], last=0.5)                                    # bars 1-2: question + leap (the lead sings)
        e_ans = theme(2, K['Eg'] + 8.0, upto=4.0)                             # bar 3: answer -- the lead rests, marimba (octave down) answers
        lead_notes(e_t1, 'glass', 0.80)
        lead_notes(harm_below(peaks(e_t1)), 'soft', 0.46, pan=-0.1)           # third + octave only on the phrase peak
        lead_notes(peaks(e_t1), 'soft', 0.30, shift=12, pan=0.1)
        lead_notes(e_ans, 'mar', 0.62, shift=-12)
        arps(K['E0'] + 0.3, K['Eg'] + 8.0, 'up5', 0.62)
        arps(K['Eg'] + 8.0, K['F0'] - 0.3, 'skip3', 0.50)
        bass_line(K['E0'], K['F0'] - 0.1, 'drive', 1.05)
        kicks(K['Eg'], 50.5, 0.90)
        claps(K['Eg'] + 1.0, 50.5, 0.88, 0.6)
        shakers(K['Eg'], 50.5, 0.62, 0.0)
        for t_ck in (37.6, 38.2, 38.8, 39.5):
            pass
        # ---- F: why owners choose us
        stinger(K['F0'], 0.6, [79, 83], boom=0.5, tcue=T(K['F0']))
        f_mel = theme(1, 51.0, last=0.5) + theme(2, 59.0)
        lead_notes(bars(f_mel, 51.0, {0, 2}), 'glass', 0.66)                  # bars 1 + 3 (question, answer): the lead sings
        lead_notes(bars(f_mel, 51.0, {1, 3}), 'mar', 0.45, shift=-12)         # bars 2 + 4: the lead rests, marimba (octave down) answers
        arps(51.0, K['G0'] - 0.5, 'skip3', 0.58, drop=(4, 3))
        bass_line(K['F0'], K['G0'], 'drive', 0.95)
        kicks(51.0, 66.5, 0.82)
        claps(52.0, 66.5, 0.70, 0.45)
        shakers(51.0, 66.5, 0.50, 0.0)
        rise(64.8, K['G0'] - 0.05, 700, 7500, 0.45, 2.0)
        # ---- G: call to action -> final tonic
        stinger(K['G0'], 0.7, [79, 83, 86], boom=0.7, tcue=T(K['G0']))
        g_mel = [(67.0, 74, 1.5), (68.5, 76, 0.5), (69.0, 79, 1.5), (70.5, 81, 1.0), (71.5, 78, 1.0)]
        lead_notes(g_mel, 'glass', 0.92, bright=2.0)
        lead_notes(harm_below(peaks(g_mel, 79)), 'soft', 0.48, pan=-0.1)       # third + octave only on the phrase peak
        lead_notes(peaks(g_mel, 79), 'soft', 0.32, shift=12, pan=0.1)
        arps(67.0, K['Gf'] - 0.2, 'up5', 0.70)
        bass_line(K['G0'], K['Gf'], 'drive', 1.05)
        kicks(67.0, 72.5, 0.95)
        claps(68.0, 70.0, 0.9, 0.6)
        for b in grid(70.0, 72.5, 0.5):
            u = (b - 70.0) / 2.5
            bus['clap'].add(clap(0.55 + 0.4 * u, r, 0.5), T(b) + jit(3), 1.0, 0.0)
        shakers(67.0, 72.5, 0.70, 0.0)
        rise(69.0, K['Gf'] - 0.08, 500, 7800, 0.62, 1.8)
        bus['fx'].add(rev_chord(CH['V']['pad'] + [74, 78], 1.3, r), T(K['Gf']) - 1.3 - 0.02, 0.5, 0.0)
        final_t = T(K['Gf'])
    else:
        # ---- short: A, B
        lead_notes(darken([(-8.0, 74, 1.4), (-6.5, 76, 0.5), (-6.0, 74, 0.5)]), 'dark', 0.55, pan=0.2)
        arps(-5.5, -3.0, 'dark', 0.55, True)
        arps(-3.0, K['crack'] - 0.25, 'dark8', 0.62, True)
        bass_line(-4.0, K['crack'] - 0.25, 'pulse8', 0.65)
        lead_notes(darken([(-3.5, 71, 0.5), (-3.0, 74, 0.5), (-2.5, 79, 0.5)]), 'dark', 0.70, pan=0.15)
        lead_notes([(-2.0, 67, 0.7)], 'soft', 0.35)
        rise(K['crack'] + 0.05, -0.30, 700, 6500, 0.55, 2.0)
        bus['fx'].add(rev_chord(CH['V7sus']['pad'] + [74], T(0) - T(K['crack']) - 0.04, r), T(K['crack']), 0.55, 0.0)
        # ---- C: IMPACT
        stinger(0.0, 1.0, [79, 86, 83, 91], boom=1.0, wipe=True)
        stinger(2.03, 0.45, [86, 91], tcue=7.1)
        c_mel = theme(1, 0.0)[:5] + [(4.0, 71, 0.5), (4.5, 74, 0.5), (5.0, 79, 0.5), (5.5, 78, 0.45)]
        lead_arc(c_mel, (0.60, 0.76, 0.94), (4.0, 5.0), bright=2.0)
        lead_notes(peaks(c_mel), 'soft', 0.32, shift=12)                       # octave lift on the phrase peak only
        arps(0.0, K['D0'] - 0.3, 'up5', 0.62)
        bass_line(0.0, K['D0'], 'drive')
        kicks(0.0, 5.5, 0.9, 2.0)
        kicks(3.5, 5.5, 0.45, 4.0, False)
        claps(1.0, 5.5, 0.8)
        shakers(0.5, K['D0'], 0.55, 0.0)
        # ---- D
        stinger(K['D0'], 0.55, [79, 83], tcue=T(K['D0']))
        lead_notes([(6.0, 76, 1.5), (7.5, 74, 0.5), (8.0, 71, 1.0), (10.0, 78, 1.5), (11.5, 76, 0.4)], 'glass', 0.54)      # one beat of rest at 9-10
        arps(K['Dg'], K['E0'] - 0.5, 'skip3', 0.46, drop=(2, 1))                     # 3 onsets/bar, the second bar rests
        bass_line(K['D0'] + 0.01, K['E0'] - 0.4, 'sparse')
        kicks(6.0, 11.0, 0.72)
        claps(7.0, 11.0, 0.66, 0.5)
        shakers(6.0, 11.5, 0.50, 0.0)
        rise(10.2, K['E0'] - 0.08, 600, 7500, 0.50, 2.0)
        bus['fx'].add(rev_chord(CH['I']['pad'], 1.0, r), T(K['E0']) - 1.0 - 0.02, 0.5, 0.0)
        # ---- E
        stinger(K['E0'], 0.9, [79, 86, 83], boom=0.8, wipe=True)
        e_mel = theme(1, 12.0, last=0.5)[:5] + [(16.0, 71, 0.5), (16.5, 74, 0.5), (17.0, 79, 0.45)]
        e_peak = [n for n in e_mel if n[0] >= 16.0]                            # the rising leap figure
        lead_arc(e_mel, (0.44, 0.95), (16.0,))
        lead_notes(harm_below(e_peak), 'soft', 0.46, pan=-0.1)                 # third + octave only on the phrase peak
        lead_notes(e_peak, 'soft', 0.30, shift=12, pan=0.1)
        arps(K['E0'] + 0.3, K['G0'] - 0.1, 'skip3', 0.52)
        bass_line(K['E0'], K['G0'], 'drive', 1.05)
        kicks(12.0, 17.0, 0.90)
        claps(13.0, 17.0, 0.88, 0.6)
        shakers(12.0, 17.5, 0.62, 0.0)
        # ---- G: CTA -> final
        stinger(K['G0'], 0.7, [79, 83, 86], boom=0.7, tcue=T(K['G0']))
        g_mel = [(17.5, 79, 0.95), (18.5, 81, 0.5), (19.0, 78, 0.5)]
        lead_notes(g_mel, 'glass', 0.92, bright=2.0)
        lead_notes(harm_below(g_mel), 'soft', 0.48, pan=-0.1)
        arps(17.5, K['Gf'] - 0.2, 'up5', 0.70)
        bass_line(K['G0'], K['Gf'], 'drive', 1.05)
        kicks(18.0, 19.5, 0.95)
        shakers(17.5, 19.5, 0.70, 0.0)
        for b in grid(18.0, 19.5, 0.5):
            u = (b - 18.0) / 1.5
            bus['clap'].add(clap(0.55 + 0.4 * u, r, 0.5), T(b) + jit(3), 1.0, 0.0)
        rise(17.6, K['Gf'] - 0.08, 500, 7800, 0.62, 1.8)
        final_t = T(K['Gf'])

    # ---------------------------------------------------------------- final tonic statement + ring-out (both videos)
    stinger(K['Gf'], 1.0, [79, 86, 83, 91], boom=1.0, tcue=final_t)
    bus['lead'].add(glass(float(mtof(79)) * 0.9985, 2.4, 0.95, r), final_t, 0.55, -0.22)
    bus['lead'].add(glass(float(mtof(79)) * 1.0015, 2.4, 0.95, r), final_t, 0.55, 0.22)
    bus['lead'].add(glass(float(mtof(67)), 2.0, 0.55, r, bright=0.4), final_t, 0.6, 0.0)
    bus['bass'].add(bass_note(float(mtof(43)), 1.6, 1.0), final_t, 1.0, 0.0)
    bus['bass'].add(bass_note(float(mtof(31 + 12)), 1.6, 0.0), final_t, 1.0, 0.0)
    b_tw = math.ceil((K['Gf'] + 0.3) * 2) / 2.0                           # twinkling ring-out, on the 8th grid
    for j, m in enumerate([79, 83, 86, 91, 95, 98]):
        bus['arp'].add(marimba(float(mtof(m)), 0.45 - 0.05 * j, r), T(b_tw + 0.5 * j) + jit(4), 1.0, -0.6 + 0.24 * j)

    # ---------------------------------------------------------------- bus processing: automation, pumping
    for k in names:
        bus[k].mul(level_curve(video, K, k, n))
    kt = sorted(kick_times)
    for k, depth in PUMP.items():
        g = np.ones(n, np.float32)
        tau = 0.19
        for tk in kt:
            i = int(tk * SR)
            m = int(0.62 * SR)
            e = min(n, i + m)
            if e <= i:
                continue
            tt = np.arange(e - i) / SR
            duck = (1 - depth * np.minimum(1.0, tt / 0.004) * np.exp(-tt / tau)).astype(np.float32)
            g[i:e] = np.minimum(g[i:e], duck)
        bus[k].mul(g)

    # ---------------------------------------------------------------- tame each bus's rare outlier peaks (look-ahead, smooth)
    for k in names:
        x = np.stack([bus[k].L, bus[k].R]).astype(np.float64)
        pk = np.max(np.abs(x), axis=0)
        act = pk > 1e-4
        if act.sum() > 100:
            x = limiter(x, float(np.percentile(pk[act], 99.9)))
            bus[k].L[:] = x[0]
            bus[k].R[:] = x[1]
    return dict(bus=bus, K=K, n=n, names=names)


def mixdown(video, R, debug=False):
    cfg = VID[video]
    imp, dur = cfg['imp'], cfg['dur']
    bus, K, n, names = R['bus'], R['K'], R['n'], R['names']
    T = lambda b: imp + b * BEAT
    # ---------------------------------------------------------------- sends / reverb / sum
    hall_ir = make_ir(2.3, 31, pre=0.020, damp=5000)
    room_ir = make_ir(0.55, 32, pre=0.004, damp=6500)
    dryL = np.zeros(n)
    dryR = np.zeros(n)
    sendHL = np.zeros(n, np.float32)
    sendHR = np.zeros(n, np.float32)
    sendRL = np.zeros(n, np.float32)
    sendRR = np.zeros(n, np.float32)
    stems = {}
    for k in names:
        b = bus[k]
        dryL += b.L * BG[k]
        dryR += b.R * BG[k]
        sendHL += b.L * (BG[k] * HALL[k])
        sendHR += b.R * (BG[k] * HALL[k])
        sendRL += b.L * (BG[k] * ROOM[k])
        sendRR += b.R * (BG[k] * ROOM[k])
        stems[k] = 0.5 * (b.L * BG[k] + b.R * BG[k])
    hwl, hwr = conv_send(sendHL, sendHR, hall_ir)
    rwl, rwr = conv_send(sendRL, sendRR, room_ir)
    outL = dryL + HALL_RET * hwl + ROOM_RET * rwl
    outR = dryR + HALL_RET * hwr + ROOM_RET * rwr
    x = np.stack([outL, outR])

    # ---------------------------------------------------------------- master: tone, fades, limiter
    x = np.stack([signal.sosfilt(sos_of('high', 32, order=2), x[0]), signal.sosfilt(sos_of('high', 32, order=2), x[1])])
    x = np.stack([signal.sosfilt(sos_of('low', 9500, order=2), x[0]), signal.sosfilt(sos_of('low', 9500, order=2), x[1])])
    fin = np.minimum(1.0, np.arange(n) / (0.35 * SR)) ** 2
    fade_len = 2.6 if video == 'long' else 1.45
    fo = np.ones(n)
    i0 = n - int(fade_len * SR)
    fo[i0:] = 0.5 * (1 + np.cos(np.pi * np.arange(n - i0) / (n - i0 - 1)))
    x = x * (fin * fo)
    body = x.mean(0)[int(T(0.0) * SR):n - int(fade_len * SR)]
    cur_db = 20 * np.log10(np.sqrt(np.mean(body ** 2)) + 1e-9)
    x *= 10 ** ((TARGET_RMS_DB - cur_db) / 20.0)
    pre_peak = np.max(np.abs(x))
    x = limiter(x, 0.86)
    gr = 20 * np.log10(max(pre_peak, 1e-9) / max(np.max(np.abs(x)), 1e-9))
    x[:, -1] = 0.0
    if debug:
        wet = np.sqrt(np.mean((HALL_RET * hwl) ** 2 + (ROOM_RET * rwl) ** 2))
        dry = np.sqrt(np.mean(dryL ** 2))
        print(f'  [{video}] wet/dry rms ratio {wet / dry:.2f}  pre-limiter peak {20 * np.log10(pre_peak):.1f} dBFS (limiter max GR {gr:.1f} dB)  gain {TARGET_RMS_DB - cur_db:.1f} dB')
        secs = section_list(video, K)[:-1] + [('R', K['Gf'], K['END'])]
        hdr = 'sec   ' + ' '.join(f'{k:>6s}' for k in names)
        print(hdr)
        for nm, b0, b1 in secs:
            i0, i1 = int(max(0, T(b0)) * SR), int(min(dur, T(b1)) * SR)
            row = ' '.join(f'{20 * np.log10(np.sqrt(np.mean(stems[k][i0:i1] ** 2)) + 1e-9):6.1f}' for k in names)
            print(f'{nm:4s}  {row}')
    return x


def build(video, debug=False):
    return mixdown(video, render_buses(video), debug)

def write_wav(path, x):
    y = np.clip(x.T, -1.0, 1.0)
    wavfile.write(path, SR, np.round(y * 32767.0).astype(np.int16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out'))
    ap.add_argument('--debug', action='store_true')
    ap.add_argument('--only', choices=['long', 'short'])
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    for video in ('long', 'short'):
        if a.only and a.only != video:
            continue
        x = build(video, a.debug)
        p = os.path.join(a.out_dir, f'pulse_{video}_music.wav')
        write_wav(p, x)
        print(f'wrote {p}  ({x.shape[1] / SR:.3f} s)')


if __name__ == '__main__':
    main()
