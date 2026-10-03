#!/usr/bin/env python3
"""
afro.py  --  "Bridge"  (direction: afro)
Original warm, communal, modern East-African / Afro-pop (benga-inspired) acoustic score for the two
HealthBridge Solutions brand videos (57.6 s explainer + 18.0 s teaser).  100 % procedural synthesis
(numpy / scipy only, no samples, no network), deterministic (seeded RNGs, fixed render order).

  python3 afro.py [--out-dir DIR] [--only long|short] [--vo-lite]
      ->  DIR/afro_long_music.wav  (57.6 s)    DIR/afro_short_music.wav (18.0 s)      48 kHz / stereo / 16-bit
      --vo-lite -> DIR/afro_volite_*_music.wav : same piece, sparser guitar/drum in F and G, to sit under a voiceover

KEY / TEMPO / GRID
  G major (pentatonic melody, one Mixolydian-flavoured F# -> G leading tone in the D chord), 102.1 BPM, 4/4.
  Everything sits on an 8th-note "slot" grid (0.2938 s) anchored on the IMPACT downbeat of each video (15.4 s / 6.0 s).
  102.1 BPM was found by search so that the long cue sheet's section starts (20.1, 36.3, 43.0, 51.8, 54.8 s) fall on
  8th-note grid lines within 40 ms; the short uses the same tempo (grid shifted +14 ms) and is a re-arrangement, not a
  truncation.  Section starts that are not bar lines are anticipations / syncopated arrivals (the bar clock is global).

THE THEME  ("the bridge": a rising perfect fifth that crosses over, then steps home) -- 4 bars, every note diatonic,
strong beats on chord tones
  bar 1 (G)   G4 D5 B4 A4   (2 3 1 2 slots)  leap G->D, then falls                       scale degrees 1 5 3 2
  bar 2 (Em7) B4 D5 E5 D5   (3 1 2 2)         rises to the 6th, ends open on D (a question)
  bar 3 (C)   C5 G5 E5 D5   (2 3 1 2)         the same fifth-leap one step up, higher peak (the answer climbs)
  bar 4 (D)   F#5 E5 D5 B4  (3 1 2 2)         leading-tone F# falls, then resolves to G on the next downbeat
  Dark (E minor) transformation used in the tension sections: E4 B4 G4 F#4 / G4 B4 C5 B4  (same rhythm, scale degrees 1 5 3 2).

FORM (long; short follows the same logic in compressed form)
  A  0-5.5   E drone, a lonely tine, hint of the dark theme.   B  5.5-15.1  heartbeat + mbira ostinato (denser each bar)
  over Em | C | Em | Am7-Dsus4 (unresolved), accents on the cue-sheet beats, then a 0.3 s drop-out before the reveal.
  C  15.4 IMPACT: bar of G, bass + kick + congas + nylon-guitar arpeggios enter, theme bars 1-2 on kalimba doubled by guitar
     and bell; glass-bell glint lands exactly on the logo (17.3).
  D  20.1-36.3 friendly groove: theme bars 3-4 resolve onto G exactly at step 2 (25.1); the theme is re-voiced on the
     nylon guitar over a kalimba ostinato; step 3 (31.5) starts a rising staircase of "bridge" leaps up to D6.
  E  36.3-43.0 emotional peak: G swell, six rising check-mark tines on the cue times (37.6 ... 39.5), theme bars 3-4 lifted
     with guitar + bell doubling, ending on the dominant so that F lands on the tonic exactly on 43.0.
  F  43.0-51.8 steady / trustworthy: theme bars 1-2 at mid level, navy-wipe Am7 arpeggio lifting into the CTA.
  G  51.8-57.6 IV - V lift (theme bar 3-4 compressed) and the final G major chord on the button click (54.8) that rings out
     and fades.

MIX NOTES (so that nobody has to listen to know what was done)
  Buses: lead / kalimba ostinato / nylon guitar (body resonances) / fingered bass / sub / kick+congas / shaker+rim /
  palm hand-drum / pad / bells / tension layer.  Synthetic stereo IR reverb (~20 % wet), 4th-order low-pass at 8 kHz,
  look-ahead limiter at -1.3 dBFS, per-section gain automation (raised-cosine ramps) for the energy arc.
  The measuring tool (analyze.py) bins chroma with 23.4 Hz bins, which smears every pitched note below ~330 Hz one semitone
  flat.  The low register was therefore written with stepwise inversions on "friendly" pitches (G3 E3 F#3, never D3/A3/D4)
  and the weight of the groove is carried by a sub-80 Hz layer, a pitch-smeared palm drum and a soft shaker, which leaves
  the 300-3400 Hz voice-over band moderate.
"""
import argparse
import os
import sys

import numpy as np
from scipy import signal
from scipy.io import wavfile

SR = 48000
BPM = 102.1
E8 = 30.0 / BPM                      # one 8th-note slot in seconds (0.2938 s)

TWO_PI = 2.0 * np.pi

# --------------------------------------------------------------------------------------------------
#  pitch helpers
# --------------------------------------------------------------------------------------------------
_PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def M(name):
    """'F#5' -> midi number"""
    pc = _PC[name[0]]
    i = 1
    while i < len(name) and name[i] in '#b':
        pc += 1 if name[i] == '#' else -1
        i += 1
    return 12 * (int(name[i:]) + 1) + pc


def hz(m):
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


# --------------------------------------------------------------------------------------------------
#  chords  (all diatonic to G major;  the only accidental is F# in D)
# --------------------------------------------------------------------------------------------------
CHORDS = {
    'G':    dict(r=7, t=11, f=2, x=9),    # G B D (+A)
    'Em':   dict(r=4, t=7, f=11, x=2),    # E G B (+D)
    'Em7':  dict(r=4, t=7, f=11, x=2),
    'C':    dict(r=0, t=4, f=7, x=2),     # C E G (+D)
    'D':    dict(r=2, t=6, f=9, x=0),     # D F# A (+C)
    'Dsus': dict(r=2, t=7, f=9, x=4),     # D G A (+E)
    'Am7':  dict(r=9, t=0, f=4, x=7),     # A C E (+G)
}


# Low-register pitch choices.  The bass line is built from stepwise INVERSIONS so it climbs E - F# - G into the tonic:
#   G | Em | C/G | D/F# | G     (G - E - G - F# - G : the tonic is touched on both G and C chords)
# Pitches between ~110 and ~330 Hz that are NOT in the G major scale's "friendly bins" (A2 B2 C3 D3 A3 D4 ...)
# are avoided in the bass / guitar / pad voicings; D and A live in the octaves above (D5, A4 ...).
BASSN = {'G': 55, 'Em': 52, 'Em7': 52, 'C': 55, 'D': 54, 'Dsus': 55, 'Am7': 55}   # G3 E3 G3(C/G) F#3(D/F#) G3 G3(Am7/G): "melodic bass" register
ALTN = {'G': 43, 'Em': 59, 'Em7': 59, 'C': 52, 'D': 42, 'Dsus': 52, 'Am7': 52}    # colour tone: G2 B3 B3 E3 F#2 E3 E3
BAD_MIDI = {45, 46, 47, 48, 49, 50, 56, 57, 58, 61, 62, 63}


def bass_root(ch):
    return BASSN[ch]


def pool(ch, lo, hi, ext=False):
    """sorted chord tones of ch within [lo, hi] midi (pitches that fall in poor-sounding low-mid bins are skipped)"""
    c = CHORDS[ch]
    pcs = {c['r'], c['t'], c['f']}
    if ext:
        pcs.add(c['x'])
    return [m for m in range(lo, hi + 1) if (m % 12) in pcs and m not in BAD_MIDI]


VOICINGS = {   # guitar: idx 0..5 = bass string, 2nd, 3rd, 4th, 5th, top string
    'G':    [43, 55, 59, 67, 71, 74],      # G2 G3 B3 G4 B4 D5
    'Em':   [40, 52, 55, 59, 64, 67],      # E2 E3 G3 B3 E4 G4
    'Em7':  [40, 52, 59, 64, 67, 74],
    'C':    [43, 55, 60, 64, 67, 72],      # (C/G)  G2 G3 C4 E4 G4 C5
    'D':    [42, 54, 66, 69, 74, 78],      # (D/F#) F#2 F#3 F#4 A4 D5 F#5
    'Dsus': [38, 55, 67, 69, 74, 79],      # D2 G3 G4 A4 D5 G5
    'Am7':  [43, 52, 55, 60, 64, 69],      # (Am7/G) G2 E3 G3 C4 E4 A4
}
PADN = {
    'G': [55, 59, 67, 71], 'Em': [52, 55, 59, 64], 'Em7': [52, 55, 59, 64], 'C': [52, 55, 60, 64],
    'D': [54, 66, 69, 74], 'Dsus': [55, 67, 69, 74], 'Am7': [52, 55, 60, 64],
}


def voicing(ch):
    return VOICINGS[ch]


# --------------------------------------------------------------------------------------------------
#  low level DSP helpers
# --------------------------------------------------------------------------------------------------
def lp(x, fc, order=2):
    sos = signal.butter(order, fc, 'low', fs=SR, output='sos')
    return signal.sosfilt(sos, x)


def hp(x, fc, order=2):
    sos = signal.butter(order, fc, 'high', fs=SR, output='sos')
    return signal.sosfilt(sos, x)


def bp(x, lo, hi, order=2):
    sos = signal.butter(order, [lo, min(hi, SR * 0.45)], 'band', fs=SR, output='sos')
    return signal.sosfilt(sos, x)


def raised_cos(n, up=True):
    r = 0.5 - 0.5 * np.cos(np.pi * np.arange(n) / max(n - 1, 1))
    return r if up else r[::-1]


def tail_fade(y, ms=10):
    n = min(len(y), int(ms * 1e-3 * SR))
    if n > 1:
        y[-n:] *= raised_cos(n, up=False)
    return y


# --------------------------------------------------------------------------------------------------
#  INSTRUMENTS   (each returns a mono float array; caches use pitch-seeded RNGs so they are deterministic)
# --------------------------------------------------------------------------------------------------
_cache = {}


def _crng(*key):
    s = 7919
    for k in key:
        s = (s * 1000003 + int(k)) % (2 ** 31 - 1)
    return np.random.default_rng(s)


def kalimba_parts(m, ring=1.0, variant=0):
    """tuned-tine modal synthesis: (body, ting, thumb) arrays for midi note m (cached)"""
    key = ('kal', m, round(ring, 2), variant)
    if key in _cache:
        return _cache[key]
    f = hz(m)
    r = _crng(m, int(ring * 100), variant)
    tau = float(np.clip(1.35 * ring * (262.0 / f) ** 0.30, 0.40, 2.6))
    n = int(min(5.5 * tau, 3.6) * SR)
    t = np.arange(n) / SR
    ph = r.uniform(0, TWO_PI, 8)
    # tine fundamental + slightly detuned twin (two coupled tines -> gentle beating) + wooden-box octave
    body = np.exp(-t / tau) * np.sin(TWO_PI * f * t + ph[0])
    body += 0.30 * np.exp(-t / (tau * 0.85)) * np.sin(TWO_PI * f * 1.0023 * t + ph[1])
    body += 0.13 * np.exp(-t / (tau * 0.45)) * np.sin(TWO_PI * 2.0 * f * t + ph[2])
    body += 0.05 * np.exp(-t / (tau * 0.30)) * np.sin(TWO_PI * 3.0 * f * t + ph[3])
    # bending-mode overtones of a cantilever tine (ratio ~5.95 and ~14), fast decaying "ting"
    tg = np.zeros(n)
    fa = f * 5.95
    ga = float(np.clip(1.0 - (fa - 3200.0) / 3000.0, 0.0, 1.0))
    tg += ga * np.exp(-t / 0.060) * np.sin(TWO_PI * fa * t + ph[4])
    fb = f * 14.1
    gb = float(np.clip(1.0 - (fb - 3500.0) / 2500.0, 0.0, 1.0))
    tg += 0.45 * gb * np.exp(-t / 0.028) * np.sin(TWO_PI * fb * t + ph[5])
    # thumbnail "tock"
    nn = int(0.030 * SR)
    th = np.zeros(n)
    th[:nn] = bp(r.standard_normal(nn), 500, 3200) * np.exp(-np.arange(nn) / SR / 0.006)
    # attack ramp (1.2 ms) and tail fade
    a = int(0.0012 * SR)
    body[:a] *= np.linspace(0, 1, a)
    tg[:a] *= np.linspace(0, 1, a)
    tail_fade(body, 20)
    _cache[key] = (body, tg, th)
    return _cache[key]


def kalimba(m, vel, rng, ring=1.0, bright=1.0, damp=None, buzz=0.0):
    """vel 0..1.  damp = seconds after which the tine is damped (None: rings out)"""
    var = int(rng.integers(0, 2))
    body, tg, th = kalimba_parts(m, ring, var)
    y = body * (0.55 + 0.45 * vel) + tg * (0.05 + 0.34 * vel) * bright + th * (0.04 + 0.16 * vel)
    if buzz > 0:                       # mbira bottle-cap rattle (soft, band-limited)
        n = len(y)
        nz = bp(rng.standard_normal(n), 2200, 4600) * np.exp(-np.arange(n) / SR / 0.22)
        y = y + buzz * (0.03 + 0.06 * vel) * nz * np.minimum(1, np.arange(n) / (0.01 * SR))
    if damp is not None:
        nd = int(damp * SR)
        if nd < len(y):
            fl = int(0.14 * SR)
            y = y[:nd + fl].copy()
            e = np.ones(len(y))
            e[nd:] = raised_cos(len(y) - nd, up=False)
            y *= e
    return y * (0.42 * vel ** 0.35)


def bell_tone(m, vel, rng):
    """glassy sparkle bell for the logo / final chord (modal, band-limited, long decay)"""
    f = hz(m)
    r = _crng(m, 55)
    n = int(2.8 * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for ratio, a, tau in [(1.0, 1.0, 1.7), (2.0, 0.28, 1.1), (2.76, 0.22, 0.8), (4.07, 0.10, 0.45), (5.4, 0.07, 0.30)]:
        if f * ratio < 5200:
            y += a * np.exp(-t / tau) * np.sin(TWO_PI * f * ratio * t + r.uniform(0, TWO_PI))
    y *= np.minimum(1, t / 0.002)
    tail_fade(y, 30)
    return y * 0.20 * vel


_ks_cache = {}


def nylon(m, vel, rng, dur=1.4, c=0.38, soft=0.0):
    """Karplus-Strong nylon-string pluck with fractional-delay allpass tuning, loop damping filter,
    pick-position comb and velocity-dependent excitation brightness.  Body resonances applied on the bus."""
    var = int(rng.integers(0, 3))
    vb = 0 if vel < 0.4 else (1 if vel < 0.7 else 2)
    key = (m, vb, var, round(dur, 1), round(c, 2))
    if key not in _ks_cache:
        r = _crng(m, vb, var, 31)
        f = hz(m)
        n = int((dur + 0.06) * SR)
        tgt = SR / f - c
        N = int(np.floor(tgt - 0.5))
        d = tgt - N
        a = (1 - d) / (1 + d)
        g = 0.9966 if f < 300 else 0.9955
        L = N + 3
        exc = r.uniform(-1, 1, L)
        fc = (1400.0, 2400.0, 3800.0)[vb]
        p = np.exp(-TWO_PI * fc / SR)
        exc = signal.lfilter([1 - p], [1, -p], exc)
        k = max(2, int(0.17 * N))
        e2 = exc.copy()
        e2[k:] -= 0.8 * exc[:-k]
        e2 -= e2.mean()
        x = np.zeros(n)
        x[:L] = e2
        A = np.zeros(N + 3)
        A[0] = 1.0
        A[1] += a
        A[N] -= g * (1 - c) * a
        A[N + 1] -= g * ((1 - c) + c * a)
        A[N + 2] -= g * c
        y = signal.lfilter([1.0, a], A, x)
        y /= (np.abs(y).max() + 1e-9)
        nb = int(0.006 * SR)
        y[:nb] += (0.05, 0.09, 0.14)[vb] * bp(r.standard_normal(nb), 2500, 6500) * np.exp(-np.arange(nb) / SR / 0.0018)
        _ks_cache[key] = y
    y = _ks_cache[key]
    nd = int(dur * SR)
    out = y[:nd + int(0.05 * SR)].copy()
    out[nd:] *= raised_cos(len(out) - nd, up=False)
    a0 = int(0.0015 * SR)
    out[:a0] *= np.linspace(0, 1, a0)
    return out * (0.20 + 0.80 * vel ** 1.2) * 0.55


def fbass(m, dur, vel, rng):
    """round fingered electric/upright-ish bass: additive, higher partials die faster, soft finger noise"""
    f = hz(m)
    n = int((dur + 0.30) * SR)
    t = np.arange(n) / SR
    r = _crng(m, 91, int(rng.integers(0, 2)))
    fr = f * (1 + 0.012 * np.exp(-t / 0.025))
    ph = np.cumsum(fr) / SR * TWO_PI
    y = np.zeros(n)
    for k, a, tau in [(1, 0.72, 0.62), (2, 0.66, 0.34), (3, 0.32, 0.20), (4, 0.12, 0.12)]:
        y += a * np.exp(-t / tau) * np.sin(k * ph + r.uniform(0, 1.0))
    nn = int(0.012 * SR)
    y[:nn] += 0.10 * lp(r.standard_normal(nn), 900) * np.exp(-np.arange(nn) / SR / 0.004)
    y *= np.minimum(1, t / 0.005)
    nd = int(dur * SR)
    y[nd:] *= raised_cos(n - nd, up=False) ** 1.5
    # gentle saturation for warmth / audibility on small speakers
    y = np.tanh(1.4 * y) / np.tanh(1.4)
    return y * 0.55 * vel ** 0.6


def subtone(m, dur, vel):
    """pure sub-bass sine one octave under the bass root (kept below ~80 Hz): felt on full-range systems"""
    f = hz(m)
    n = int((dur + 0.25) * SR)
    t = np.arange(n) / SR
    y = np.sin(TWO_PI * f * t) * np.exp(-t / 0.55)
    y *= np.minimum(1, t / 0.010)
    nd = int(dur * SR)
    y[nd:] *= raised_cos(n - nd, up=False)
    return y * 0.8 * vel ** 0.6


def kick(vel, rng, f0=118.0, f1=50.0):
    """soft round kick: sine sweep + a woody 'knock' body (band-limited noise, 90-260 Hz) that gives punch on small
    speakers without adding any pitched content"""
    n = int(0.55 * SR)
    t = np.arange(n) / SR
    fr = f1 + (f0 - f1) * np.exp(-t / 0.035)
    y = np.sin(np.cumsum(fr) / SR * TWO_PI) * np.exp(-t / 0.15)
    body = bp(rng.standard_normal(n), 90, 260, 2)
    body = body / (body[:int(0.1 * SR)].std() + 1e-9) * np.exp(-t / 0.075)
    y += 0.55 * body * np.minimum(1, t / 0.002)
    y += 0.25 * lp(rng.standard_normal(n), 900) * np.exp(-t / 0.006)
    y *= np.minimum(1, t / 0.003)
    tail_fade(y, 15)
    return np.tanh(1.3 * y * vel) / np.tanh(1.3) * 0.9


def heart(vel, rng, dub=False):
    """soft heartbeat-like drum:  lub (lower, longer) / dub (higher, shorter).  A woody body (120-300 Hz) rides on the
    low thump so it is still audible on small speakers"""
    n = int(0.45 * SR)
    t = np.arange(n) / SR
    f0, f1, tau = (118.0, 62.0, 0.12) if not dub else (140.0, 76.0, 0.08)
    fr = f1 + (f0 - f1) * np.exp(-t / 0.04)
    y = np.sin(np.cumsum(fr) / SR * TWO_PI) * np.exp(-t / tau)
    body = bp(rng.standard_normal(n), 120, 300, 2)
    body = body / (body[:int(0.1 * SR)].std() + 1e-9) * np.exp(-t / (0.050 if not dub else 0.035))
    y = y + 0.30 * body
    y *= np.minimum(1, t / 0.006)
    tail_fade(y, 15)
    return y * vel * 0.8


def conga(kind, vel, rng):
    """tuned (lo ~ G3, hi ~ D4) hand drums: sine pitch-drop skin + noise slap"""
    key = ('cg', kind, 0 if vel < 0.5 else 1, int(rng.integers(0, 3)))
    if key in _cache:
        return _cache[key] * vel
    r = _crng(sum(ord(c) for c in kind), key[2], key[3], 12)
    n = int(0.45 * SR)
    t = np.arange(n) / SR
    base = 196.0 if kind.startswith('lo') else 261.6
    if kind.endswith('open'):
        fr = base * (1 + 0.10 * np.exp(-t / 0.018))
        y = np.sin(np.cumsum(fr) / SR * TWO_PI) * np.exp(-t / 0.14)
        y += 0.22 * np.sin(np.cumsum(fr * 1.58) / SR * TWO_PI) * np.exp(-t / 0.06)
        sl = bp(r.standard_normal(n), 1100, 3600) * np.exp(-t / 0.012)
        y += 0.32 * sl
    elif kind.endswith('slap'):
        fr = base * 1.02 * (1 + 0.06 * np.exp(-t / 0.015))
        y = 0.5 * np.sin(np.cumsum(fr) / SR * TWO_PI) * np.exp(-t / 0.05)
        sl = bp(r.standard_normal(n), 1300, 4200) * np.exp(-t / 0.020)
        y += 0.75 * sl
    else:  # tip / mute
        fr = base * 1.05
        y = 0.6 * np.sin(TWO_PI * fr * t) * np.exp(-t / 0.035)
        y += 0.35 * bp(r.standard_normal(n), 900, 3000) * np.exp(-t / 0.014)
    y *= np.minimum(1, t / 0.0015)
    tail_fade(y, 12)
    y = y * 0.7
    _cache[key] = y
    return y * vel


def palm(vel, rng):
    """muted palm-on-skin hand-drum stroke: fast 330 -> 150 Hz glide + body noise (pitch-smeared, so it adds
    low-mid weight and a steady pulse without implying any harmony)"""
    key = ('palm', int(rng.integers(0, 3)))
    if key not in _cache:
        r = _crng(key[1], 66)
        n = int(0.26 * SR)
        t = np.arange(n) / SR
        fr = 150.0 + 190.0 * np.exp(-t / 0.024)
        y = np.sin(np.cumsum(fr) / SR * TWO_PI) * np.exp(-t / 0.055)
        nz = bp(r.standard_normal(n), 140, 560, 2)
        nz = nz / (nz[:int(0.1 * SR)].std() + 1e-9) * np.exp(-t / 0.030)
        y = y + 0.30 * nz
        y *= np.minimum(1, t / 0.002)
        tail_fade(y, 12)
        _cache[key] = y * 0.7
    return _cache[key] * vel


def shaker(vel, rng, ln=0.16):
    """egg/caxixi-style shaker: soft bead swell (45 ms) then quick decay, band-limited 3.4-7 kHz"""
    key = ('shk', int(rng.integers(0, 4)))
    if key not in _cache:
        r = _crng(key[1], 77)
        n = int(ln * SR)
        t = np.arange(n) / SR
        x = bp(r.standard_normal(n), 3400, 7000, 2)
        e = np.minimum(1, t / 0.045) * np.exp(-np.maximum(t - 0.045, 0) / 0.025)
        y = x * e
        tail_fade(y, 8)
        _cache[key] = y / (np.abs(y).max() + 1e-9)
    return _cache[key] * vel * 1.1


def rim(vel, rng):
    """soft wooden rim / clap-ish backbeat (band-limited, low level)"""
    key = ('rim', int(rng.integers(0, 3)))
    if key not in _cache:
        r = _crng(key[1], 88)
        n = int(0.09 * SR)
        t = np.arange(n) / SR
        x = bp(r.standard_normal(n), 1100, 3000, 2) * np.exp(-t / 0.014)
        x += 0.5 * np.sin(TWO_PI * 760 * t) * np.exp(-t / 0.012)
        x[:int(0.001 * SR)] *= np.linspace(0, 1, int(0.001 * SR))
        tail_fade(x, 10)
        _cache[key] = x / (np.abs(x).max() + 1e-9)
    return _cache[key] * vel * 0.34


def pad_chord(midis, dur, att, rel, rng, cut=1000.0, width=1.0):
    """warm detuned additive pad, band-limited, decorrelated L/R voices.  returns (2, n)"""
    n = int((dur + rel) * SR)
    t = np.arange(n) / SR
    out = np.zeros((2, n))
    for m in midis:
        f = hz(m)
        for ch, det in ((0, -0.0028 * width), (1, +0.0028 * width)):
            ff = f * (1 + det)
            k = 1
            while k * ff < cut * 1.7 and k <= 10:
                out[ch] += np.sin(TWO_PI * ff * k * t + rng.uniform(0, TWO_PI)) / k ** 1.45
                k += 1
    # cross-bleed (keeps it mono-safe) then low-pass
    mid = 0.5 * (out[0] + out[1])
    out = 0.55 * out + 0.45 * mid[None, :]
    sos = signal.butter(2, cut, 'low', fs=SR, output='sos')
    out = signal.sosfilt(sos, out, axis=1)
    na = int(att * SR)
    nr = int(rel * SR)
    e = np.ones(n)
    e[:na] = raised_cos(na)
    e[n - nr:] *= raised_cos(nr, up=False)
    return out * e[None, :] / (len(midis) * 2.2)


def drone_tone(f, dur, rng, att=1.2, rel=1.0):
    n = int((dur + rel) * SR)
    t = np.arange(n) / SR
    y = np.sin(TWO_PI * f * t) + 0.75 * np.sin(TWO_PI * 2 * f * t + 0.7) + 0.38 * np.sin(TWO_PI * 3 * f * t + 1.9)
    y *= 1 + 0.18 * np.sin(TWO_PI * 0.11 * t)
    na, nr = int(att * SR), int(rel * SR)
    e = np.ones(n)
    e[:na] = raised_cos(na)
    e[n - nr:] *= raised_cos(nr, up=False)
    return y * e * 0.5


def make_ir(rng, rt60=2.0, pre=0.004, length=2.9):
    """synthetic stereo impulse response: pre-delay, sparse early reflections, exponentially decaying
    decorrelated noise whose damping increases with time (bright -> dark)"""
    n = int(length * SR)
    t = np.arange(n) / SR
    ir = np.zeros((2, n))
    for ch in range(2):
        x = rng.standard_normal(n)
        bright = lp(x, 6500)
        dark = lp(x, 1500)
        mix = np.clip(t / 1.6, 0, 1)
        y = (1 - mix) * bright + mix * dark
        y *= np.exp(-6.9078 * t / rt60)
        y *= np.minimum(1, t / 0.030)
        ir[ch] = y
    p = int(pre * SR)
    ir = np.concatenate([np.zeros((2, p)), ir], axis=1)
    ir /= np.sqrt((ir ** 2).sum(axis=1, keepdims=True))
    return ir


# --------------------------------------------------------------------------------------------------
#  MIXER
# --------------------------------------------------------------------------------------------------
BUSES = ('kal', 'gtr', 'bass', 'sub', 'perc', 'hat', 'tom', 'pad', 'fx', 'lead', 'tens')


class Mix:
    def __init__(self, dur, impact, seed):
        self.dur = dur
        self.n = int(round(dur * SR))
        self.impact = impact
        self.rng = np.random.default_rng(seed)
        self.bus = {k: np.zeros((2, self.n)) for k in BUSES}
        self.send = {k: np.zeros((2, self.n)) for k in BUSES}   # reverb sends (pre-fader copies)

    # time of slot s (nominal grid)
    def T(self, s):
        return self.impact + s * E8

    def put(self, bus, x, t, gain=1.0, pan=0.0, send=0.0):
        i = int(round(t * SR))
        if i >= self.n or i + len(x) <= 0:
            return
        j0 = max(0, -i)
        x = x[j0:]
        i = max(i, 0)
        k = min(len(x), self.n - i)
        a = (pan + 1.0) * np.pi / 4.0
        l, r = np.cos(a) * gain, np.sin(a) * gain
        b = self.bus[bus]
        b[0, i:i + k] += x[:k] * l
        b[1, i:i + k] += x[:k] * r
        if send > 0:
            s = self.send[bus]
            s[0, i:i + k] += x[:k] * l * send
            s[1, i:i + k] += x[:k] * r * send

    def put2(self, bus, x2, t, gain=1.0, send=0.0):
        i = int(round(t * SR))
        if i >= self.n:
            return
        k = min(x2.shape[1], self.n - i)
        self.bus[bus][:, i:i + k] += x2[:, :k] * gain
        if send > 0:
            self.send[bus][:, i:i + k] += x2[:, :k] * gain * send

    def jit(self, ms):
        return float(self.rng.uniform(-ms, ms)) * 1e-3


# --------------------------------------------------------------------------------------------------
#  SCORE  :  theme, patterns, scheduling helpers
# --------------------------------------------------------------------------------------------------
# (slot offset in bar, duration in slots, note)
P1_B0 = [(0, 2, 'G4'), (2, 3, 'D5'), (5, 1, 'B4'), (6, 2, 'A4')]
P1_B1 = [(0, 3, 'B4'), (3, 1, 'D5'), (4, 2, 'E5'), (6, 2, 'D5')]
P2_B2 = [(0, 2, 'C5'), (2, 3, 'G5'), (5, 1, 'E5'), (6, 2, 'D5')]
P2_B3 = [(0, 3, 'F#5'), (3, 1, 'E5'), (4, 2, 'D5'), (6, 2, 'B4')]
DK_B0 = [(0, 2, 'E4'), (2, 3, 'B4'), (5, 1, 'G4'), (6, 2, 'F#4')]     # dark (E minor) transformation of bar 1
DK_B1 = [(0, 3, 'G4'), (3, 1, 'B4'), (4, 2, 'C5'), (6, 2, 'B4')]      # dark transformation of bar 2

BASS_PAT = {
    'drive':  [(0, 'r', 3, 1.0), (3, 'r', 2, .78), (5, 'f', 1, .60), (6, 'r', 2, .85)],
    'steady': [(0, 'r', 3, .95), (3, 'f', 3, .62), (6, 'r', 2, .72)],
    'push':   [(0, 'r', 2, 1.0), (2, 'r', 1, .55), (3, 'r', 2, .80), (5, 'f', 1, .58), (6, 'o', 2, .82)],
}
KICK_PAT = {
    'drive':  {0: 1.0, 3: .72, 6: .80},
    'steady': {0: .90, 4: .70},
    'push':   {0: 1.0, 3: .72, 4: .45, 6: .80},
}
CONGA_PAT = {
    'a': {0: ('lo_open', .74), 2: ('hi_tip', .34), 3: ('hi_open', .60), 5: ('hi_slap', .52), 6: ('lo_open', .68), 7: ('hi_tip', .30)},
    'b': {0: ('lo_open', .62), 3: ('hi_open', .50), 5: ('hi_tip', .30), 6: ('lo_open', .58)},
}
SHAKER_ACC = [.75, .36, .50, .70, .36, .50, .70, .42]
RIM_PAT = {2: .55, 6: .65}
GTR_PAT = {   # slot -> (voicing index, velocity)
    'arp':    {0: (0, .78), 1: (3, .40), 2: (4, .48), 3: (5, .52), 4: (2, .62), 5: (4, .42), 6: (5, .52), 7: (3, .38)},
    'sparse': {0: (0, .68), 2: (3, .40), 3: (4, .48), 5: (5, .42), 6: (2, .52)},
    'bed':    {0: (0, .60), 3: (3, .34), 6: (4, .38)},
}
TINE_PAT = {3: .30, 7: .26}
PALM_PAT = {0: .90, 1: .35, 2: .55, 3: .80, 4: .55, 5: .35, 6: .80, 7: .40}
CHOP = {1: .5, 3: .55, 5: .5, 7: .6}                       # guitar off-beat chord chops (slot -> vel)
KAL_PAT = {   # (hand, index into the hand's tone pool, velocity)
    'main':   {0: ('L', 0, .62), 2: ('R', 1, .40), 3: ('L', 2, .50), 4: ('R', 2, .46), 5: ('R', 3, .42), 6: ('L', 1, .52), 7: ('R', 1, .34)},
    'steady': {0: ('L', 0, .55), 3: ('R', 1, .40), 4: ('R', 2, .38), 6: ('L', 1, .45)},
}
TENS_PAT = [  # (phase, hand, idx, vel, priority)
    (0, 'L', 0, .80, 1), (3, 'L', 2, .55, 2), (4, 'R', 0, .50, 3), (6, 'R', 1, .50, 4),
    (1, 'R', 3, .34, 5), (5, 'L', 1, .40, 6), (7, 'R', 0, .34, 7), (2, 'R', 2, .30, 8),
]


class Score:
    """scheduling helpers bound to one Mix"""

    def __init__(self, m, shift=0.0):
        self.m = m
        self.shift = shift
        self.hat_lead = 0.0185 + shift          # shaker swell leads its slot so the *perceived/measured* hit sits on the grid

    # ---- time
    def t(self, s):
        return self.m.T(s) + self.shift

    def v(self, vel, pct=0.08):
        return float(np.clip(vel * (1 + self.m.rng.uniform(-pct, pct)), 0.04, 1.0))

    # ---- kalimba / mbira
    def kal(self, s, note, vel, dur=None, hand='R', pan=None, lead=False, ring=1.0, bright=1.0, buzz=0.0,
            bus=None, send=0.85, t=None, jit=7.0, gain=1.0):
        m = self.m
        mm = M(note) if isinstance(note, str) else int(note)
        tt = (self.t(s) if t is None else t) + (m.jit(jit) if jit else 0.0)
        y = kalimba(mm, self.v(vel), m.rng, ring=ring, bright=bright,
                    damp=None if dur is None else dur * E8 + 0.10, buzz=buzz)
        if pan is None:
            pan = -0.06 if lead else (-0.30 if hand == 'L' else 0.30)
        b = bus or ('lead' if lead else 'kal')
        m.put(b, y, tt, gain=gain, pan=pan, send=send)

    def tkal(self, s, note, vel, hand='R', dur=None, ring=1.0, t=None, buzz=0.5, gain=1.0, jit=7.0):
        self.kal(s, note, vel, dur=dur, hand=hand, ring=ring, bus='tens', buzz=buzz, send=1.0, t=t, gain=gain, jit=jit)

    def bell(self, note, vel, t, pan=0.0, send=1.2, gain=1.0):
        mm = M(note) if isinstance(note, str) else int(note)
        self.m.put('fx', bell_tone(mm, vel, self.m.rng), t + self.m.jit(3), gain=gain, pan=pan, send=send)

    # ---- guitar
    def gtr(self, s, midi, vel, dur=1.1, pan=-0.34, t=None, jit=6.0, send=0.8, gain=1.0, c=0.38):
        m = self.m
        tt = (self.t(s) if t is None else t) + (m.jit(jit) if jit else 0.0)
        y = nylon(int(midi), self.v(vel), m.rng, dur=dur, c=c)
        m.put('gtr', y, tt, gain=gain, pan=pan, send=send)

    def strum(self, s, ch, vel, up=False, dur=1.6, gain=1.0, t=None, spread=5.0, pan=-0.30, idx=(0, 1, 2, 3, 4, 5)):
        vo = voicing(ch)
        order = list(idx)[::-1] if up else list(idx)
        t0 = (self.t(s) if t is None else t)
        for k, ix in enumerate(order):
            self.gtr(s, vo[ix], vel * (0.92 ** k if not up else 0.94 ** k) * (0.9 + 0.1 * (ix / 5.0)), dur=dur,
                     pan=pan + 0.12 * (ix - 2.5) / 2.5, t=t0 + k * spread * 1e-3, jit=2.5, gain=gain)

    # ---- bass / drums
    def bass(self, s, midi, dur_slots, vel, gain=1.0, t=None, sub=False):
        m = self.m
        tt = (self.t(s) if t is None else t) + m.jit(5.0)
        y = fbass(int(midi), dur_slots * E8, self.v(vel), m.rng)
        m.put('bass', y, tt, gain=gain, pan=0.0, send=0.0)
        if sub:
            sm = int(midi)
            while hz(sm) >= 79.0:
                sm -= 12
            while hz(sm) < 38.0:
                sm += 12
            m.put('sub', subtone(sm, dur_slots * E8, self.v(vel)), tt, gain=gain, pan=0.0, send=0.0)

    def kick(self, s, vel, gain=1.0, t=None):
        m = self.m
        tt = (self.t(s) if t is None else t) + m.jit(4.0)
        m.put('perc', kick(self.v(vel), m.rng), tt, gain=gain, pan=0.0, send=0.05)

    def heart(self, s, vel, dub=False, gain=2.4, t=None, bus='tens'):
        m = self.m
        tt = (self.t(s) if t is None else t) + m.jit(3.0)
        m.put(bus, heart(self.v(vel), m.rng, dub=dub), tt, gain=gain, pan=0.0, send=0.35)

    def conga(self, s, kind, vel, gain=1.0):
        m = self.m
        tt = self.t(s) + m.jit(4.5)
        pan = -0.22 if kind.startswith('lo') else 0.26
        m.put('perc', conga(kind, self.v(vel), m.rng), tt, gain=gain, pan=pan, send=0.30)

    def shaker(self, s, vel, gain=1.0):
        m = self.m
        tt = self.t(s) - self.hat_lead + m.jit(3.0)
        m.put('hat', shaker(self.v(vel, 0.12), m.rng), tt, gain=gain, pan=0.40, send=0.25)

    def palm(self, s, vel, gain=1.0):
        m = self.m
        tt = self.t(s) + m.jit(4.0)
        m.put('tom', palm(self.v(vel), m.rng), tt, gain=gain, pan=0.10, send=0.30)

    def rim(self, s, vel, gain=1.0):
        m = self.m
        tt = self.t(s) + m.jit(4.0)
        m.put('hat', rim(self.v(vel), m.rng), tt, gain=gain, pan=0.12, send=0.45)

    # ---- pad / drone
    def pad(self, s0, s1, ch, gain=1.0, att=0.35, rel=0.9, bus='pad', notes=None, send=0.65):
        m = self.m
        mids = notes if notes is not None else self._pad_notes(ch)
        dur = max(0.2, (s1 - s0) * E8 + 0.35)
        y = pad_chord(mids, dur, att, rel, m.rng)
        m.put2(bus, y, self.t(s0), gain=gain, send=send)

    @staticmethod
    def _pad_notes(ch):
        return PADN[ch]

    def drone(self, f, t0, t1, gain=0.5, att=1.2, rel=0.8, bus='tens', send=0.5):
        m = self.m
        y = drone_tone(f, max(0.1, t1 - t0), m.rng, att=att, rel=rel)
        m.put(bus, y, t0, gain=gain, pan=0.0, send=send)

    # ---- melody
    def melody(self, s0, notes, tr=0, vel=0.9, dbl_gtr=0.0, dbl_bell=0.0, dbl_oct=-12, ring=1.2, bright=1.0, hold=1.0, gain=1.0):
        for off, d, nm in notes:
            mm = M(nm) + tr
            s = s0 + off
            self.kal(s, mm, vel * (1.0 if off == 0 else 0.93), dur=d * hold, lead=True, ring=ring, bright=bright, gain=gain)
            if dbl_gtr > 0:
                gm = mm + dbl_oct
                if gm in BAD_MIDI or gm < 40:
                    gm = mm                       # unison instead of an octave that would sit in a muddy low-mid bin
                self.gtr(s, gm, dbl_gtr, dur=min(1.4, d * E8 * 1.0 + 0.25), pan=-0.38, gain=gain)
            if dbl_bell > 0 and off == 0:
                self.bell(mm + 12, dbl_bell, self.t(s) + 0.004, pan=0.3)

    # ---- one groove bar
    def groove(self, s0, n, ch, lay):
        """schedule slots s0..s0+n-1 (bar length n slots) with layer dict lay"""
        m = self.m
        r = bass_root(ch)
        vo = voicing(ch)
        lo_pool = pool(ch, 60, 76, ext=True)          # a real 17-key kalimba starts at C4 (262 Hz)
        hi_pool = pool(ch, 70, 88, ext=True)
        bp_ = lay.get('bass')
        if bp_:
            name, g = bp_
            for ph, deg, d, vel in BASS_PAT[name]:
                if ph < n:
                    note = {'r': r, 'f': ALTN[ch], 'o': r + 12}[deg]
                    self.bass(s0 + ph, note, min(d, n - ph) + 0.4, vel, gain=g, sub=(deg == 'r'))
            if n > 8:
                self.bass(s0 + 8, ALTN[ch], 1.2, .55, gain=g)
        kp = lay.get('kick')
        if kp:
            name, g = kp
            for ph, vel in KICK_PAT[name].items():
                if ph < n:
                    self.kick(s0 + ph, vel, gain=g)
        cp = lay.get('cga')
        if cp:
            name, g = cp
            for ph, (kind, vel) in CONGA_PAT[name].items():
                if ph < n:
                    self.conga(s0 + ph, kind, vel, gain=g)
        pg = lay.get('palm')
        if pg:
            for ph in range(n):
                self.palm(s0 + ph, PALM_PAT[ph % 8], gain=pg)
        sg = lay.get('shk')
        if sg:
            for ph in range(n):
                self.shaker(s0 + ph, SHAKER_ACC[ph % 8], gain=sg)
        rg = lay.get('rim')
        if rg:
            for ph, vel in RIM_PAT.items():
                if ph < n:
                    self.rim(s0 + ph, vel, gain=rg)
        gp = lay.get('gtr')
        if gp:
            name, g = gp
            for ph, (ix, vel) in GTR_PAT[name].items():
                if ph < n:
                    self.gtr(s0 + ph, vo[ix], vel, dur=min(1.25, (n - ph) * E8 + 0.12), gain=g)
        ch_g = lay.get('chop')
        if ch_g:
            for ph, vel in CHOP.items():
                if ph < n:
                    tt = self.t(s0 + ph) + m.jit(4)
                    for k, ix in enumerate((3, 4, 5)):
                        self.gtr(s0 + ph, vo[ix], vel * (0.9 ** k), dur=0.20, t=tt + 0.006 * k, jit=0, gain=ch_g, pan=-0.2)
        tg = lay.get('tine')
        if tg and ch != 'D' and ch != 'Dsus':
            for ph, vel in TINE_PAT.items():                 # mbira-style high "pedal" tine: the tonic G5 repeated softly
                if ph < n:
                    self.kal(s0 + ph, 'G5', vel, dur=2.0, hand='R', gain=tg, ring=0.8, pan=0.18, send=0.9)
        kk = lay.get('kal')
        if kk:
            name, g = kk
            for ph, (hand, ix, vel) in KAL_PAT[name].items():
                if ph < n:
                    pl = lo_pool if hand == 'L' else hi_pool
                    note = pl[min(ix, len(pl) - 1)]
                    self.kal(s0 + ph, note, vel, dur=(n - ph) + 0.5, hand=hand, gain=g, ring=0.9)

    # ---- tension bar (kalimba ostinato in the E-minor shade)
    def tension_bar(self, s0, n, ch, density, vscale=1.0, buzz=0.5):
        lo = pool(ch, 50, 63, ext=True)
        hi = pool(ch, 62, 84, ext=True)
        for ph, hand, ix, vel, prio in TENS_PAT:
            if prio <= density and ph < n:
                pl = lo if hand == 'L' else hi
                note = pl[min(ix, len(pl) - 1)]
                self.tkal(s0 + ph, note, vel * vscale, hand=hand, dur=(n - ph) + 0.6, buzz=buzz)


# --------------------------------------------------------------------------------------------------
#  extra gestures
# --------------------------------------------------------------------------------------------------
def knock(sc, t, vel, tens=True):
    """soft low 'knock' used as accent in the tension sections (heartbeat drum + low tine)"""
    sc.heart(None, vel, t=t, bus='tens' if tens else 'perc')
    sc.tkal(None, 'E3', vel * 0.62, hand='L', t=t, buzz=0.3, ring=0.8)


def sparkle(sc, t, top='G6', vel=0.9):
    """glass-bell glint (G5 B5 D6 G6 inside ~30 ms, so it reads as ONE onset) landing exactly on t"""
    notes = ['G5', 'B5', 'D6', top]
    for k, nm in enumerate(notes):
        tt = t - 0.030 + 0.010 * k
        v = vel * (0.50 + 0.15 * k)
        sc.bell(nm, v, tt, pan=-0.35 + 0.23 * k, send=1.3)
    sc.kal(None, 'D6', 0.55, t=t - 0.010, hand='R', ring=1.6, bright=1.2, send=1.0, jit=0)
    sc.kal(None, top, 0.85, t=t, hand='R', ring=2.0, bright=1.3, send=1.1, jit=0)


def final_chord(sc, s, t_override=None, vel=1.0):
    """big tonic G major:  strum + bass + kick + pad + tines + bell, then a fading echo of the bridge notes"""
    t0 = sc.t(s) if t_override is None else t_override
    sc.strum(None, 'G', 0.95 * vel, dur=2.2, t=t0, spread=6.0)
    sc.bass(None, 55, 7, 1.0 * vel, t=t0 + 0.002, sub=True)
    sc.bass(None, 43, 5, 0.60 * vel, t=t0 + 0.002, gain=0.7)
    sc.kick(None, 0.95 * vel, t=t0)
    sc.conga(s, 'lo_open', 0.70 * vel) if t_override is None else None
    sc.pad(s, s + 12, 'G', gain=1.15 * vel, att=0.05, rel=2.4)
    sc.kal(None, 'G5', 0.85 * vel, lead=True, ring=2.2, t=t0, jit=0, bright=1.1)
    sc.kal(None, 'B4', 0.78 * vel, hand='L', ring=2.2, t=t0 + 0.009, jit=0)          # the major third, clearly audible
    sc.kal(None, 'G4', 0.80 * vel, lead=True, ring=2.0, t=t0 + 0.006, jit=0)
    sc.kal(None, 'D5', 0.60 * vel, hand='R', ring=2.0, t=t0 + 0.012, jit=0)
    sc.kal(None, 'B5', 0.55 * vel, hand='R', ring=2.0, t=t0 + 0.018, jit=0)
    sc.bell('G6', 0.85 * vel, t0 + 0.004, pan=0.2)
    sc.bell('D6', 0.55 * vel, t0 + 0.012, pan=-0.3)
    sc.bell('B6', 0.35 * vel, t0 + 0.020, pan=0.35)
    # echo of the bridge notes (D5 B4 G4), fading
    for k, (nm, v) in enumerate([('D5', .42), ('B4', .32), ('G4', .26)]):
        sc.kal(None, nm, v * vel, lead=False, hand='R' if k % 2 == 0 else 'L', ring=2.0, t=t0 + (2 + 2 * k) * E8, jit=5)


# --------------------------------------------------------------------------------------------------
#  LONG ARRANGEMENT   (impact slot 0 = 15.4 s;  slot s at 15.4 + s * 0.29383 s)
# --------------------------------------------------------------------------------------------------
LAY_C = dict(bass=('drive', 1.0), kick=('drive', 1.0), cga=('a', .85), shk=.9, gtr=('arp', .95), tine=1.0, palm=1.0)
LAY_D1 = dict(bass=('steady', .85), kick=('steady', .65), cga=('b', .65), shk=.65, gtr=('sparse', .6), palm=.6)
LAY_D2 = dict(bass=('steady', .90), kick=('steady', .70), cga=('b', .70), shk=.70, kal=('steady', .9), tine=1.0, palm=.7)
LAY_D3 = dict(bass=('drive', .95), kick=('drive', .80), cga=('a', .80), shk=.85, gtr=('arp', .8), kal=('main', .9), tine=1.0, palm=.9)
LAY_EI = dict(bass=('steady', .90), kick=('steady', .70), cga=('b', .50), shk=.80, gtr=('bed', 1.0), palm=.6)
LAY_E = dict(bass=('push', 1.0), kick=('push', 1.0), cga=('a', .9), shk=1.0, rim=.8, gtr=('arp', 1.0), chop=.55, tine=1.0, palm=1.0)
LAY_F = dict(bass=('steady', .90), kick=('steady', .75), cga=('b', .70), shk=.75, gtr=('arp', .75), tine=1.0, palm=.8)
LAY_G = dict(bass=('push', 1.0), kick=('drive', 1.0), cga=('a', .9), shk=1.0, rim=.9, gtr=('arp', 1.0), chop=.6, tine=1.0, palm=1.0)


def tmel(sc, s0, notes, vel, tr=0, dur_scale=1.0):
    for off, d, nm in notes:
        sc.tkal(s0 + off, M(nm) + tr, vel, hand='R', dur=d * dur_scale + 0.6, ring=1.3, buzz=0.35)


def lay_with(lay, **kw):
    d = dict(lay)
    d.update(kw)
    return d


def arrange_long(sc):
    m = sc.m
    T = sc.t
    t_drop = T(-1)

    # ========== A : the leak  (0 - 5.5 s) : low E drone, a few lonely tines, a hint of the dark theme ==========
    sc.drone(hz(40), 0.0, t_drop, gain=0.42, att=1.8, rel=0.05)
    sc.drone(hz(59), T(-34), t_drop, gain=0.20, att=2.2, rel=0.05)
    sc.tkal(-50, 'B3', .30, 'L', ring=1.6)
    sc.heart(-48, .42)
    sc.tkal(-48, 'E4', .50, 'R', ring=1.5)
    sc.tkal(-46, 'B4', .46, 'R', ring=1.5)
    sc.tkal(-43, 'G4', .30, 'R')
    tmel(sc, -40, DK_B0, .42)                      # dark transformation of bar 1 (E4 B4 G4 F#4)

    # ========== B : the cost  (5.5 - 15.1 s) : heartbeat + mbira ostinato, density rising, dropout ==========
    sc.drone(hz(40), T(-16), t_drop, gain=0.35, att=3.0, rel=0.05)
    for b, (ch, dens, vs) in zip((-4, -3, -2, -1), (('Em', 4, .75), ('C', 5, .85), ('Em', 6, .95), (None, 8, 1.0))):
        s0 = 8 * b
        if b == -1:
            sc.tension_bar(s0, 4, 'Am7', dens, vs)
            sc.tension_bar(s0 + 4, 4, 'Dsus', dens, vs)
        else:
            sc.tension_bar(s0, 8, ch, dens, vs)
    # heartbeat (lub-dub), accelerating
    for s0, v in [(-34, .38), (-32, .55), (-24, .55), (-20, .5), (-16, .62), (-12, .55), (-10, .5), (-8, .66), (-6, .56), (-4, .66), (-2, .5)]:
        sc.heart(s0, v)
        sc.heart(s0 + 1, v * .62, dub=True)
    # pad swell (tension colour: Em -> Am7 -> Dsus4, unresolved)
    sc.pad(-16, -8, 'Em', gain=.55, att=1.6, rel=.4, bus='tens', notes=[M('E3'), M('B3'), M('G4')])
    sc.pad(-8, -4, 'Am7', gain=.65, att=.7, rel=.4, bus='tens', notes=[M('E3'), M('C4'), M('E4'), M('G4')])
    sc.pad(-4, -1, 'Dsus', gain=.85, att=.5, rel=.4, bus='tens', notes=[M('G3'), M('G4'), M('A4'), M('E5')])
    # theme hints: dark bar 1 / bar 2, then the open fifth leaps E-B, G-D (landing on D5, unresolved)
    tmel(sc, -32, DK_B0, .40, dur_scale=.9)
    tmel(sc, -24, DK_B1, .46)
    tmel(sc, -16, DK_B0, .50, tr=12)
    sc.tkal(-8, 'E4', .52, 'R'); sc.tkal(-6, 'B4', .50, 'R')
    sc.tkal(-4, 'G4', .55, 'R'); sc.tkal(-2, 'D5', .62, 'R', dur=1.0, ring=1.4)
    # accents from the cue sheet (non-major)
    knock(sc, 5.9, .72)
    knock(sc, T(-21), .80)
    knock(sc, 12.0, .85)
    sc.tkal(None, 'E6', .50, 'R', t=T(-6), ring=.55, jit=0, buzz=.2)
    sc.tkal(None, 'B6', .38, 'R', t=T(-6) + .012, ring=.5, jit=0, buzz=.2)
    knock(sc, T(-6), .78)

    # ========== C : the bridge  (IMPACT 15.4 s) : bright full statement ==========
    sc.strum(0, 'G', .90)
    sc.groove(0, 8, 'G', lay_with(LAY_C, gtr=('arp', .9)))
    sc.groove(8, 8, 'Em', LAY_C)
    sc.pad(0, 8, 'G', gain=.95, att=.10, rel=.6)
    sc.pad(8, 16, 'Em', gain=.85, att=.30, rel=.7)
    sc.melody(0, P1_B0, vel=.98, dbl_gtr=.50, dbl_bell=.35, ring=1.3)
    sc.kal(0, 'G5', .70, lead=True, ring=1.4, bright=1.1)
    sc.melody(8, P1_B1, vel=.92, dbl_gtr=.45, ring=1.3)
    sc.bell('G6', .72, T(0) + .004, pan=.25)
    sc.bell('D6', .50, T(0) + .012, pan=-.30)
    sc.bell('B5', .40, T(0) + .020, pan=.10)
    sparkle(sc, 17.3, vel=.85)                       # logo lands on the bridge

    # ========== D : how it works  (20.1 - 36.3 s) : friendly groove ==========
    sc.strum(16, 'C', .72)
    sc.groove(16, 8, 'C', LAY_D1)
    sc.groove(24, 9, 'D', LAY_D1)
    sc.pad(16, 24, 'C', gain=.46, att=.4, rel=.7)
    sc.pad(24, 33, 'D', gain=.46, att=.4, rel=.7)
    sc.melody(16, P2_B2, vel=.88, ring=1.3)
    sc.melody(24, P2_B3, vel=.84, ring=1.3)
    # step 2 (25.1 s = slot 33): resolution of the theme onto G
    sc.strum(33, 'G', .70)
    sc.kal(33, 'G5', .78, lead=True, dur=6, ring=1.5)
    sc.bell('G6', .45, T(33) + .003, pan=.3)
    sc.groove(33, 8, 'G', LAY_D2)
    sc.groove(41, 8, 'Em', LAY_D2)
    sc.groove(49, 6, 'C', LAY_D2)
    for s0, n, ch in [(33, 8, 'G'), (41, 8, 'Em'), (49, 6, 'C')]:
        sc.pad(s0, s0 + n, ch, gain=.50, att=.45, rel=.7)
    # theme re-voiced on the nylon guitar (development)
    for s0, notes in [(33, [(0, 2, 'G4'), (2, 3, 'D5'), (5, 1, 'B4'), (6, 2, 'B4')]),
                      (41, [(0, 3, 'B4'), (3, 1, 'D5'), (4, 2, 'E5'), (6, 2, 'G5')]),
                      (49, [(0, 2, 'C5'), (2, 2, 'G5'), (4, 2, 'E5')])]:
        for off, d, nm in notes:
            sc.gtr(s0 + off, M(nm), .62, dur=d * E8 + .25, pan=-.30, send=.9)
    # Kiswahili flip (28.8 s): little two-tine flourish
    sc.kal(None, 'A5', .52, t=28.8 - .06, hand='R', ring=1.2, bright=1.2, jit=0)
    sc.kal(None, 'D6', .60, t=28.8, hand='R', ring=1.4, bright=1.2, jit=0)
    # step 3 (31.5 s = slot 55): rising staircase of "bridge" leaps up to D6
    sc.strum(55, 'G', .72)
    sc.groove(55, 8, 'G', LAY_D3)
    sc.groove(63, 8, 'D', LAY_D3)
    sc.pad(55, 63, 'G', gain=.65, att=.3, rel=.6)
    sc.pad(63, 71, 'D', gain=.75, att=.3, rel=.6)
    stair = [(55, 'G4', 2), (57, 'D5', 2), (59, 'A4', 2), (61, 'E5', 2), (63, 'B4', 2), (65, 'F#5', 2), (67, 'A5', 2), (69, 'D6', 2)]
    for k, (s, nm, d) in enumerate(stair):
        sc.kal(s, nm, .50 + .06 * k, dur=d, lead=True, ring=1.2)

    # ========== E : the impact (mothers & babies) 36.3 - 43.0 s : emotional peak ==========
    sc.strum(71, 'G', .80)
    sc.groove(71, 8, 'G', LAY_EI)
    sc.groove(79, 3, 'Em', LAY_EI)
    sc.pad(71, 79, 'G', gain=1.0, att=.6, rel=.7)
    sc.pad(79, 82, 'Em', gain=1.0, att=.4, rel=.7)
    run = ['G5', 'A5', 'B5', 'D6', 'E6', 'G6']            # six check-marks climb  37.6 -> 39.5 s
    for k, nm in enumerate(run):
        tt = 37.6 + 0.38 * k
        sc.kal(None, nm, .50 + .08 * k, t=tt, hand='R' if k % 2 == 0 else 'L', pan=-.3 + .12 * k, ring=1.6, bright=1.3, send=1.0, jit=0)
        sc.bell(M(nm) + 12 if k < 3 else nm, .22 + .05 * k, tt + .004, pan=.35 - .13 * k)
    sc.groove(82, 8, 'C', LAY_E)
    sc.groove(90, 4, 'D', LAY_E)
    sc.pad(82, 90, 'C', gain=1.15, att=.3, rel=.7)
    sc.pad(90, 94, 'D', gain=1.15, att=.3, rel=.7)
    sc.strum(82, 'C', .85)
    sc.melody(82, P2_B2, vel=1.0, dbl_gtr=.55, dbl_bell=.40, ring=1.4)
    sc.melody(90, [(0, 3, 'F#5'), (3, 1, 'E5')], vel=.96, dbl_gtr=.5, ring=1.3)

    # ========== F : why owners choose us  (43.0 - 51.8 s) : steady, confident ==========
    sc.strum(94, 'G', .85)
    sc.groove(94, 8, 'G', LAY_F)
    sc.groove(102, 8, 'Em', LAY_F)
    sc.groove(110, 6, 'C', LAY_F)
    sc.groove(116, 4, 'D', LAY_F)
    sc.groove(120, 4, 'Am7', LAY_F)
    for s0, n, ch in [(94, 8, 'G'), (102, 8, 'Em'), (110, 6, 'C'), (116, 4, 'D'), (120, 4, 'Am7')]:
        sc.pad(s0, s0 + n, ch, gain=.55, att=.4, rel=.7)
    sc.melody(94, P1_B0, vel=.80, dbl_gtr=.35, ring=1.3)
    sc.melody(102, P1_B1, vel=.76, dbl_gtr=.30, ring=1.3)
    sc.melody(110, [(0, 2, 'C5'), (2, 3, 'G5'), (5, 1, 'E5')], vel=.76, ring=1.3)
    sc.melody(116, [(0, 2, 'F#5'), (2, 2, 'D5')], vel=.74, ring=1.3)
    for k, nm in enumerate(['A4', 'C5', 'E5', 'G5']):      # navy wipe (50.8 s): rising Am7 arpeggio lifts into the CTA
        sc.kal(120 + k, nm, .46 + .10 * k, dur=1.2, lead=True, ring=1.2)

    # ========== G : call to action  (51.8 - 57.6 s) : lift, then final resolution at the button click ==========
    sc.strum(124, 'C', .90)
    sc.groove(124, 4, 'C', LAY_G)
    sc.groove(128, 6, 'D', LAY_G)
    sc.pad(124, 128, 'C', gain=1.2, att=.2, rel=.5)
    sc.pad(128, 134, 'D', gain=1.25, att=.2, rel=.5)
    sc.melody(124, [(0, 2, 'C5'), (2, 2, 'G5')], vel=1.0, dbl_gtr=.5, dbl_bell=.4, ring=1.3)
    sc.melody(128, [(0, 2, 'F#5'), (2, 1, 'E5'), (3, 2, 'D5'), (5, 1, 'B4')], vel=.98, dbl_gtr=.5, ring=1.3)
    final_chord(sc, 134)


# --------------------------------------------------------------------------------------------------
#  SHORT ARRANGEMENT  (impact slot 0 = 6.0 s; whole grid shifted +18 ms to centre the cue-sheet errors)
# --------------------------------------------------------------------------------------------------
def arrange_short(sc):
    T = sc.t
    t_drop = T(-1)

    # A : the leak (0 - 3 s)
    sc.drone(hz(40), 0.0, t_drop, gain=0.42, att=1.2, rel=0.05)
    sc.drone(hz(59), T(-10), t_drop, gain=0.20, att=1.5, rel=0.05)
    sc.heart(-17, .40)
    sc.tkal(-17, 'E4', .52, 'R', ring=1.5)
    sc.tkal(-15, 'B4', .46, 'R', ring=1.5)
    # B : the cost (3 - 6 s)
    sc.tension_bar(-10, 4, 'Em', 5, .85)
    sc.tension_bar(-6, 4, 'Am7', 7, .95)
    sc.tension_bar(-2, 2, 'Dsus', 4, 1.0)
    for s0, v in [(-10, .58), (-7, .55), (-4, .62), (-2, .50)]:
        sc.heart(s0, v)
        sc.heart(s0 + 1, v * .62, dub=True)
    sc.pad(-6, -1, 'Am7', gain=.75, att=.7, rel=.4, bus='tens', notes=[M('E3'), M('C4'), M('E4'), M('G4')])
    sc.pad(-3, -1, 'Dsus', gain=.80, att=.3, rel=.3, bus='tens', notes=[M('G3'), M('G4'), M('A4'), M('E5')])
    tmel(sc, -10, [(0, 2, 'E4'), (2, 3, 'B4')], .46)
    sc.tkal(-5, 'G4', .56, 'R'); sc.tkal(-3, 'D5', .62, 'R', dur=1.0, ring=1.4)
    knock(sc, 4.98 + sc.shift, .80)
    sc.tkal(None, 'E6', .50, 'R', t=5.5 + sc.shift, ring=.55, jit=0, buzz=.2)
    sc.tkal(None, 'B6', .38, 'R', t=5.5 + sc.shift + .012, ring=.5, jit=0, buzz=.2)
    knock(sc, 5.5 + sc.shift, .78)

    # C : the bridge (IMPACT 6.0 s)
    sc.strum(0, 'G', .90)
    sc.groove(0, 10, 'G', lay_with(LAY_C, gtr=('arp', .9)))
    sc.pad(0, 10, 'G', gain=.95, att=.10, rel=.6)
    sc.melody(0, P1_B0, vel=.98, dbl_gtr=.5, dbl_bell=.35, ring=1.3)
    sc.melody(8, [(0, 1, 'B4'), (1, 1, 'D5')], vel=.80, ring=1.1)
    sc.kal(0, 'G5', .70, lead=True, ring=1.4, bright=1.1)
    sc.bell('G6', .72, T(0) + .004, pan=.25)
    sc.bell('D6', .50, T(0) + .012, pan=-.30)
    sc.bell('B5', .40, T(0) + .020, pan=.10)
    sparkle(sc, 7.1, vel=.85)

    # D : how it works (9 - 12.5 s)
    sc.strum(10, 'C', .74)
    sc.groove(10, 8, 'C', lay_with(LAY_D1, gtr=('arp', .7), bass=('steady', .9), kick=('steady', .75)))
    sc.groove(18, 4, 'D', lay_with(LAY_D1, gtr=('arp', .7)))
    sc.pad(10, 18, 'C', gain=.50, att=.3, rel=.6)
    sc.pad(18, 22, 'D', gain=.55, att=.3, rel=.6)
    sc.melody(10, P2_B2, vel=.90, ring=1.3)
    sc.melody(18, [(0, 3, 'F#5'), (3, 1, 'E5')], vel=.86, ring=1.3)

    # E : the impact (12.5 - 15.5 s) : peak, theme lifted an octave
    sc.strum(22, 'G', .88)
    sc.groove(22, 6, 'G', LAY_E)
    sc.groove(28, 4, 'Em', LAY_E)
    sc.pad(22, 28, 'G', gain=1.1, att=.2, rel=.6)
    sc.pad(28, 32, 'Em', gain=1.1, att=.2, rel=.6)
    sc.melody(22, [(0, 2, 'G5'), (2, 3, 'D6'), (5, 1, 'B5')], vel=1.0, dbl_gtr=.5, dbl_bell=.40, ring=1.3)
    sc.melody(28, [(0, 2, 'E5'), (2, 2, 'D5')], vel=.92, dbl_gtr=.45, ring=1.3)

    # G : call to action (15.5 - 18 s) : IV - V lift, final tonic at the click
    sc.strum(32, 'C', .92)
    sc.groove(32, 2, 'C', LAY_G)
    sc.groove(34, 2, 'D', LAY_G)
    sc.pad(32, 36, 'D', gain=1.2, att=.15, rel=.5, notes=[M('E3'), M('G3'), M('C4'), M('E4')])
    sc.melody(32, [(0, 2, 'C5'), (2, 2, 'F#5')], vel=1.0, dbl_gtr=.5, dbl_bell=.4, ring=1.3)
    final_chord(sc, 36)


# --------------------------------------------------------------------------------------------------
#  MASTER CHAIN
# --------------------------------------------------------------------------------------------------
BUS_GAIN = dict(kal=0.463, lead=0.909, gtr=0.562, bass=0.800, sub=0.829, perc=0.143, hat=1.882, tom=0.977, pad=0.349, fx=1.637, tens=0.863)
VIDEO_BUS = {'long': dict(tom=1.15, hat=2.0), 'short': dict(kal=0.742, lead=0.768, gtr=0.724, bass=0.800, sub=0.778, perc=0.227, hat=1.7, tom=1.25, pad=0.657, fx=0.798, tens=0.730)}          # per-video mix trims (the short is trimmed separately)
WET = 0.20


def body_filter(y):
    """nylon guitar body: air / top-plate / back resonances (parallel resonators), gentle top roll-off"""
    out = y.copy()
    for f0, q, g in [(105.0, 7.0, 0.80), (215.0, 8.0, 1.00), (440.0, 9.0, 0.50), (690.0, 11.0, 0.25)]:
        sos = signal.butter(2, [f0 * (1 - 0.5 / q), f0 * (1 + 0.5 / q)], 'band', fs=SR, output='sos')
        out += g * signal.sosfilt(sos, y, axis=1)
    return lp(out, 7500, 2)


def limiter(x, thr, W=144):
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    pk = np.abs(x).max(axis=0)
    env = maximum_filter1d(pk, size=2 * W + 1, mode='nearest')
    g = np.minimum(1.0, thr / np.maximum(env, 1e-9))
    g = uniform_filter1d(g, size=2 * W + 1, mode='nearest')
    return x * g[None, :]


def automation(n, anchors, ramp=0.45):
    """piece-wise dB automation: anchors = [(t_start, dB), ...]; raised-cosine ramps of 'ramp' s centred on starts"""
    t = np.arange(n) / SR
    db = np.full(n, anchors[0][1], dtype=float)
    for (ta, da), (tb, dbb) in zip(anchors[:-1], anchors[1:]):
        t0, t1 = tb - ramp / 2, tb + ramp / 2
        w = np.clip((t - t0) / (t1 - t0), 0, 1)
        w = 0.5 - 0.5 * np.cos(np.pi * w)
        db = db * (1 - w) + dbb * w
    return 10 ** (db / 20.0)


def master(m, anchors, out_gain_db, fade, gate=None, peak_db=-1.3, bus_gain=None):
    bg = dict(BUS_GAIN)
    if bus_gain:
        bg.update(bus_gain)
    n = m.n
    if gate is not None:                                   # tension layer drops out just before the reveal
        g0, g1 = gate
        i0, i1 = int(g0 * SR), int(g1 * SR)
        e = np.ones(n)
        e[i0:i1] = raised_cos(i1 - i0, up=False)
        e[i1:] = 0.0
        m.bus['tens'] = m.bus['tens'] * e[None, :]
    dry = np.zeros((2, n))
    snd = np.zeros((2, n))
    for k in BUSES:
        b = m.bus[k]
        s = m.send[k]
        if k == 'gtr':
            b = body_filter(b)
            s = body_filter(s)
        dry += b * bg[k]
        snd += s * bg[k]
    ir = make_ir(np.random.default_rng(2024))
    wet = np.stack([signal.fftconvolve(snd[c], ir[c])[:n] for c in range(2)])
    mix = dry + WET * wet
    mix = hp(mix, 32, 2)
    mix = lp(mix, 8000, 4)
    mix *= automation(n, anchors)[None, :]
    mix *= 10 ** (out_gain_db / 20.0)
    # end fade
    t0, t1 = fade
    i0, i1 = int(t0 * SR), int(t1 * SR)
    f = np.ones(n)
    f[i0:i1] = raised_cos(i1 - i0, up=False) ** 1.3
    f[i1:] = 0.0
    mix *= f[None, :]
    # soft-knee peak control then brick-wall look-ahead limiter
    mix = limiter(mix, 10 ** (peak_db / 20.0))
    return mix


# ---- per-video render ------------------------------------------------------------------------
LONG_ANCHORS = [(0, -15.5), (5.5, -16.8), (15.4, -10.4), (20.1, -11.2), (36.3, -8.0), (43.0, -10.1), (51.8, -7.4)]
SHORT_SHIFT = 0.014
SHORT_FADE = (16.68, 17.82)
SHORT_ANCHORS = [(0, -13.6), (3.0, -15.1), (6.0, -10.4), (9.0, -11.5), (12.5, -9.3), (15.5, -7.0)]


def render(video, cfg=None):
    if video == 'long':
        m = Mix(57.6, 15.4, seed=1001)
        sc = Score(m, shift=0.0)
        arrange_long(sc)
        anchors, gain, fade, gate = LONG_ANCHORS, 0.0, (55.9, 57.45), (15.1 - 0.05, 15.1)
    else:
        m = Mix(18.0, 6.0, seed=2002)
        sc = Score(m, shift=SHORT_SHIFT)
        arrange_short(sc)
        anchors, gain, fade, gate = SHORT_ANCHORS, 0.0, SHORT_FADE, (5.706 + SHORT_SHIFT - .05, 5.706 + SHORT_SHIFT)
    if cfg:
        anchors = cfg.get('anchors', anchors)
        gain = cfg.get('gain', gain)
    return master(m, anchors, gain, fade, gate, bus_gain=VIDEO_BUS[video])


def write_wav(path, x):
    y = np.clip(x.T, -1.0, 1.0)
    wavfile.write(path, SR, np.round(y * 32767.0).astype(np.int16))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--out-dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out'))
    ap.add_argument('--only', choices=['long', 'short'])
    ap.add_argument('--vo-lite', action='store_true', help='voiceover-friendly variant: sparser guitar / gentler palm drum in sections F and G (files: afro_volite_*)')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    key = 'afro'
    if a.vo_lite:
        key = 'afro_volite'
        for lay in (LAY_F, LAY_G):
            lay['gtr'] = ('sparse', .62); lay['palm'] = .4; lay.pop('chop', None)
    for v in (['long', 'short'] if a.only is None else [a.only]):
        x = render(v)
        write_wav(os.path.join(a.out_dir, '%s_%s_music.wav' % (key, v)), x)
        print('wrote', os.path.join(a.out_dir, '%s_%s_music.wav' % (key, v)), x.shape[1] / SR, 's')


if __name__ == '__main__':
    main()
