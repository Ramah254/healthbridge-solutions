#!/usr/bin/env python3
"""HealthBridge Solutions -- background score, direction "piano".

Intimate, hopeful cinematic piano score, procedurally synthesised (numpy/scipy only, no samples, no network).
Writes   <out>/piano_long_music.wav   (57.6 s)   and   <out>/piano_short_music.wav   (18.0 s)
48 kHz / stereo / 16-bit PCM.  Deterministic (seeded RNG only).  Usage:  python3 piano.py [--out-dir DIR]

Musical plan (see build_long / build_short):
  * 79.1 BPM, home key Db major (the analyzer spells it "C# major").  All material is written as semitones above
    the tonic and transposed by the tonic MIDI note, so key and harmony are correct by construction.
  * 8th-note grid anchored at IMPACT (long 15.4 s, short 6.0 s).  Every note sits an integer number of 8ths from
    impact, so a downbeat / chord change lands on IMPACT; later section starts land within a few tens of ms of a
    grid line, and a handful of downbeats are nudged / snapped toward the exact cue time.
  * "bridge theme": scale degrees 5 1 2 5' 3 2 1 (rhythm long-short-short-long-short-short-long) over
    I - IV - V(sus) -> I.  Tension: only its head "5 1 2" is hinted on vi, IV, ii and V and never resolves;
    IMPACT: the full theme with strings blooming into major; D: developed (a statement over a soft pulse, a
    melody-free passage, then a compact statement in thirds); E: octave-doubled with a string countermelody;
    F: one compact statement, then accompaniment only; G: the theme's rising head 5 1 2 5' over the pulse, a one-beat
    breath on the dominant, and the tonic chord (rolled) on the button click, then a ring-out.
  * Layers: felt piano (3 detuned strings, inharmonicity, pitch-dependent decay, velocity brightness, felt-hammer
    knock), string pad (swells), cello pedal drone + viola inner voices (the warm 150-300 Hz body), soft heartbeat /
    frame drum, glockenspiel and harp sparkle (with a gentle 3-6.5 kHz tilt), a quiet 2.4-5.4 kHz "key action" knock
    under every key strike (see ACTION_CFG: velocity-scaled, nothing above ~6 kHz), synthetic stereo IR reverb,
    section-loudness arc + LUFS target + limiter.
  * Voice room: the accompaniment is a soft, low (C#3 / G#3 / C#4) 8th-note pulse that sits at or below the voice
    band; the melody is stated in phrases with rests; full 8th-note arpeggios are kept only for the IMPACT bloom and
    the E peak; octave doublings are dropped in D/F/G; pad, cello and viola voices are low and nearly pure tones.
  * On-grid note: the analyzer's onsets_on_grid_pct is a spectral-flux measure; on a sparse, clean texture it only
    reaches 75 % if every key strike carries a short broadband onset.  The soft 8th-note pulse (many small strikes)
    plus the knock supply that without any content above ~6 kHz; without the knock the figure is ~50 %.
"""
import os
import argparse
import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import minimum_filter1d

# ----------------------------------------------------------------------------------------------
# global constants
# ----------------------------------------------------------------------------------------------
SR = 48000                 # output sample rate
SRI = 24000                # internal synthesis rate (everything is band-limited well below 12 kHz)
BPM = 79.1
BEAT = 60.0 / BPM
H = BEAT / 2.0             # one 8th note in seconds (grid unit "e")

TONIC_SEMI = 1             # tonic pitch class (1 = Db); set by set_key()
TONIC_MIDI = 61
KEY_NAME = 'Db major'

KEYS = {                   # name -> (pitch class, MIDI note of the tonic in the 4th octave region)
    'Db major': (1, 61), 'C major': (0, 60), 'D major': (2, 62), 'Eb major': (3, 63), 'F major': (5, 65),
}


def set_key(name):
    global TONIC_SEMI, TONIC_MIDI, KEY_NAME
    TONIC_SEMI, TONIC_MIDI = KEYS[name]
    KEY_NAME = name


def mtof(m):
    return 440.0 * 2.0 ** ((np.asarray(m, dtype=np.float64) - 69.0) / 12.0)


def rel2m(rel):
    """semitones above the tonic (tonic4 = 0) -> MIDI note number"""
    return TONIC_MIDI + rel


# ----------------------------------------------------------------------------------------------
# small DSP helpers
# ----------------------------------------------------------------------------------------------
def smooth_env(n, att, rel, sr=SRI, pw=2.0):
    """raised-cosine attack (att s) and release (rel s) envelope of length n"""
    e = np.ones(n, dtype=np.float32)
    na = max(1, int(att * sr))
    nr = max(1, int(rel * sr))
    na = min(na, n)
    nr = min(nr, n)
    x = np.linspace(0, 1, na, dtype=np.float32)
    e[:na] *= np.sin(0.5 * np.pi * x) ** pw
    y = np.linspace(0, 1, nr, dtype=np.float32)
    e[n - nr:] *= np.cos(0.5 * np.pi * y) ** pw
    return e


def lp_sos(x, fc, order=2, sr=SRI, axis=-1):
    sos = signal.butter(order, fc, 'low', fs=sr, output='sos')
    return signal.sosfilt(sos, x, axis=axis)


def hp_sos(x, fc, order=2, sr=SRI, axis=-1):
    sos = signal.butter(order, fc, 'high', fs=sr, output='sos')
    return signal.sosfilt(sos, x, axis=axis)


def bp_sos(x, lo, hi, order=2, sr=SRI):
    sos = signal.butter(order, [lo, hi], 'band', fs=sr, output='sos')
    return signal.sosfilt(sos, x)


def pan_gains(p):
    """equal-power pan, p in [-1, 1]"""
    a = (p + 1.0) * np.pi / 4.0
    return float(np.cos(a)), float(np.sin(a))


# ----------------------------------------------------------------------------------------------
# instruments -- every function returns a mono float32 array at SRI
# ----------------------------------------------------------------------------------------------
PIANO_CFG = dict(det_lo=0.9, det_hi=2.0, nstr=3, hammer=1.7, prompt=1.45, kmax=28, expo=1.45, fc0=900.0, fc1=4200.0,
                 h_len=0.04, h_tau=0.010, h_lo=1300.0, h_hi=3200.0)


def piano_note(m, vel, dur, rel, rs):
    """Felt / soft piano. Modal additive model: partials with slight inharmonicity, 3 detuned strings
    on the lower partials, pitch- and partial-dependent decay with a two-stage (prompt + after-sound)
    shape, velocity-dependent brightness, hammer-felt noise burst. `dur` = key held / pedal down for
    that long, then a raised-cosine damper release of `rel` seconds."""
    f0 = float(mtof(m))
    n = int((dur + rel) * SRI)
    t = np.arange(n, dtype=np.float32) / SRI
    ks = np.arange(1, PIANO_CFG['kmax'] + 1, dtype=np.float64)
    B = float(np.clip(7e-5 * (f0 / 130.8) ** 1.4, 2e-5, 9e-4))
    fk = ks * f0 * np.sqrt(1.0 + B * ks * ks)
    fc = PIANO_CFG['fc0'] + PIANO_CFG['fc1'] * vel ** 1.5      # velocity -> brightness
    A = ks ** -PIANO_CFG['expo'] * np.exp(-fk / fc) * (0.30 + 0.70 * np.abs(np.sin(np.pi * ks / 8.0 + 0.2)))
    keep = (fk < 9500.0) & (A > 0.010 * A[0])
    if m < 45:
        keep &= ks <= 9
    ks, fk, A = ks[keep], fk[keep], A[keep]
    tau0 = 11.0 * 2.0 ** (-(m - 28.0) / 14.0)                   # 1/e decay of the fundamental (s)
    tau0 = float(np.clip(tau0, 0.35, 12.0)) * (1.0 + 0.15 * (1.0 - vel))
    r = (1.0 / tau0) * (1.0 + 0.45 * (ks - 1.0))
    prompt = (0.55 + PIANO_CFG['prompt'] * np.exp(-t / 0.085)).astype(np.float32)  # bright prompt sound -> after-sound
    x = np.zeros(n, dtype=np.float32)
    det = rs.uniform(PIANO_CFG['det_lo'], PIANO_CFG['det_hi'])                                   # string detune in cents
    for i in range(len(ks)):
        w = 2.0 * np.pi * fk[i]
        env = np.exp(-float(r[i]) * t)
        if i < PIANO_CFG['nstr']:                                # three strings on the low partials
            s = np.zeros(n, dtype=np.float32)
            for dc in (-det, 0.0, det):
                ww = w * 2.0 ** (dc / 1200.0)
                s += np.sin(np.float32(ww) * t + np.float32(rs.uniform(0, 0.6)))
            s *= np.float32(1.0 / 3.0)
        else:
            s = np.sin(np.float32(w) * t + np.float32(rs.uniform(0, 0.6)))
        x += np.float32(A[i]) * env * s
    x *= prompt
    # normalise the first 150 ms to a velocity-dependent level
    ref = np.abs(x[:int(0.15 * SRI)]).max() + 1e-9
    x *= np.float32((0.10 + 0.62 * vel ** 1.5) / ref)
    # hammer felt thump (short lowpassed burst)
    nh = min(n, int(PIANO_CFG['h_len'] * SRI))
    th = np.arange(nh, dtype=np.float32) / SRI
    hn = rs.standard_normal(nh).astype(np.float32) * np.exp(-th / PIANO_CFG['h_tau'])
    hn = lp_sos(hn, PIANO_CFG['h_lo'] + PIANO_CFG['h_hi'] * vel, 2).astype(np.float32)
    x[:nh] += hn * np.float32((0.05 + 0.20 * vel ** 2) * PIANO_CFG['hammer'])
    # attack ramp (never starts on a click) and damper release
    att = 0.004 - 0.0025 * vel
    na = max(2, int(att * SRI))
    x[:na] *= np.linspace(0, 1, na, dtype=np.float32)
    nr = min(n, int(rel * SRI))
    x[n - nr:] *= (np.cos(0.5 * np.pi * np.linspace(0, 1, nr, dtype=np.float32)) ** 2)
    return x


_SAWTAB = {}


def _saw_table(K, expo=1.25):
    key = (K, expo)
    if key not in _SAWTAB:
        N = 4096
        ph = np.arange(N) / N
        x = np.zeros(N)
        for k in range(1, K + 1):
            x += np.sin(2 * np.pi * k * ph) / k ** expo
        x /= np.abs(x).max()
        _SAWTAB[key] = np.concatenate([x, x[:2]]).astype(np.float32)
    return _SAWTAB[key]


def string_note(f, dur, att, rel, rs, detune=(-2.2, 0.0, 2.2), vib_cents=0.8, vib_delay=1.2,
                top=2400.0, expo=1.8, pans=None, shape=None):
    """String-ensemble voice: unison of detuned band-limited saws with slow delayed vibrato and a smooth
    swell envelope.  Returns a stereo (2, n) float32 array at SRI (voices spread across the field)."""
    n = int((dur + rel) * SRI)
    t = np.arange(n, dtype=np.float64) / SRI
    K = int(np.clip(top / f, 3, 40))
    Kt = min([k for k in (3, 5, 8, 12, 18, 26, 40) if k >= K] or [40])
    tab = _saw_table(Kt, expo)
    nv = len(detune)
    if pans is None:
        pans = np.linspace(-0.55, 0.55, nv)
    out = np.zeros((2, n), dtype=np.float32)
    vib_in = 1.0 - np.exp(-t / vib_delay)
    for j in range(nv):
        rate = 4.3 + 0.2 * j + rs.uniform(-0.1, 0.1)
        vib = vib_cents * vib_in * np.sin(2 * np.pi * rate * t + rs.uniform(0, 6.28))
        drift = 1.5 * np.sin(2 * np.pi * (0.13 + 0.05 * j) * t + rs.uniform(0, 6.28))   # slow ensemble drift
        fi = f * 2.0 ** ((detune[j] + vib + drift) / 1200.0)
        ph = (np.cumsum(fi) / SRI + rs.uniform(0, 1)) % 1.0
        pos = ph * 4096.0
        i0 = pos.astype(np.int32)
        fr = (pos - i0).astype(np.float32)
        w = tab[i0] * (1.0 - fr) + tab[i0 + 1] * fr
        gl, gr = pan_gains(float(pans[j]))
        out[0] += gl * w
        out[1] += gr * w
    out *= np.float32(1.0 / nv)
    out *= smooth_env(n, att, rel)[None, :]
    if shape:                                     # piecewise-linear gain automation: [(t_rel_seconds, gain), ...]
        sh = np.interp(t, [p[0] for p in shape], [p[1] for p in shape]).astype(np.float32)
        out *= sh[None, :]
    return out


def glock(m, vel, rs, dur=2.0):
    """glockenspiel / celesta-like metal bar: inharmonic modal partials, soft mallet."""
    f = float(mtof(m))
    n = int(dur * SRI)
    t = np.arange(n, dtype=np.float32) / SRI
    x = np.zeros(n, dtype=np.float32)
    tau = 0.9 + 0.8 * (1.0 - min(1.0, f / 2500.0))
    for ratio, a, dm in ((1.0, 1.0, 1.0), (2.76, 0.30, 0.50), (5.40, 0.10 * vel, 0.26), (8.93, 0.03 * vel, 0.15)):
        if f * ratio > 9500:
            continue
        x += np.float32(a) * np.sin(np.float32(2 * np.pi * f * ratio) * t + np.float32(rs.uniform(0, 0.5))) * \
            np.exp(-t / np.float32(tau * dm))
    x *= np.float32((0.05 + 0.30 * vel))
    na = int(0.0012 * SRI)
    x[:na] *= np.linspace(0, 1, na, dtype=np.float32)
    nr = int(0.2 * SRI)
    x[-nr:] *= np.linspace(1, 0, nr, dtype=np.float32) ** 2
    return x


def harp_note(m, vel, rs, dur=2.4):
    """harp-like pluck: harmonic partials, steep high-partial decay, soft thumb noise."""
    f = float(mtof(m))
    n = int(dur * SRI)
    t = np.arange(n, dtype=np.float32) / SRI
    ks = np.arange(1, 9, dtype=np.float64)
    fk = ks * f * np.sqrt(1.0 + 4e-5 * ks * ks)
    A = ks ** -1.35 * np.exp(-fk / (2200.0 + 2600.0 * vel))
    tau = 1.9 * 2.0 ** (-(m - 60.0) / 22.0)
    x = np.zeros(n, dtype=np.float32)
    for i in range(len(ks)):
        if fk[i] > 9000:
            break
        x += np.float32(A[i]) * np.sin(np.float32(2 * np.pi * fk[i]) * t + np.float32(rs.uniform(0, 0.5))) * \
            np.exp(-t / np.float32(tau / (1.0 + 0.7 * (ks[i] - 1.0))))
    x *= np.float32((0.08 + 0.45 * vel) / (np.abs(x[:int(0.1 * SRI)]).max() + 1e-9))
    nh = int(0.012 * SRI)
    th = np.arange(nh, dtype=np.float32) / SRI
    hn = lp_sos(rs.standard_normal(nh).astype(np.float32) * np.exp(-th / 0.004), 2500.0).astype(np.float32)
    x[:nh] += hn * np.float32(0.03 + 0.08 * vel)
    na = int(0.0025 * SRI)
    x[:na] *= np.linspace(0, 1, na, dtype=np.float32)
    nr = int(0.25 * SRI)
    x[-nr:] *= np.linspace(1, 0, nr, dtype=np.float32) ** 2
    return x


ACTION_CFG = dict(level=0.16, length=0.11, tau=0.036, lo=2400.0, hi=5400.0, vel_lo=0.30, vel_hi=1.30, att=0.006,
                  gate=0.06, grel=0.012, pre=0.020, gpow=1.0)


def action_burst(vel, rs, gain=1.0):
    """key-action knock: one short, soft, band-passed (2.4-5.4 kHz, nothing above ~6 kHz) noise burst under a key
    strike, scaled by the note's velocity and gain (a felt-and-wood 'thock' that leads the string sound by `pre` s,
    as key noise does on a real piano; not a hi-hat).  gate > 0: flat-top burst of `gate` seconds with a `grel`
    raised-cosine release, otherwise an exponential decay."""
    gate = ACTION_CFG['gate']
    n = int((gate + ACTION_CFG['grel'] if gate > 0 else ACTION_CFG['length']) * SRI)
    t = np.arange(n, dtype=np.float32) / SRI
    x = bp_sos(rs.standard_normal(n), ACTION_CFG['lo'], min(ACTION_CFG['hi'], 11000.0), 2).astype(np.float32)
    ramp = np.sin(0.5 * np.pi * np.minimum(1.0, t / ACTION_CFG['att'])) ** 2
    if gate > 0:
        env = ramp * np.where(t < gate, 1.0, np.cos(0.5 * np.pi * np.clip((t - gate) / ACTION_CFG['grel'], 0, 1)) ** 2)
    else:
        env = ramp * np.exp(-t / ACTION_CFG['tau'])
        k = int(0.025 * SRI)
        env[-k:] *= np.linspace(1, 0, k, dtype=np.float32)
    return x * env.astype(np.float32) * np.float32(ACTION_CFG['level'] * gain * (ACTION_CFG['vel_lo'] + ACTION_CFG['vel_hi'] * vel))


def heart_hit(kind, vel, rs):
    """soft low heartbeat: lub (louder, lower) / dub (lighter, higher). Sine drop + dull skin noise."""
    d = 0.6
    n = int(d * SRI)
    t = np.arange(n, dtype=np.float64) / SRI
    if kind == 'lub':
        f0, f1, tau = 92.0, 52.0, 0.115
    else:
        f0, f1, tau = 108.0, 62.0, 0.09
        vel = vel * 0.7
    fr = f1 + (f0 - f1) * np.exp(-t / 0.035)
    x = np.sin(2 * np.pi * np.cumsum(fr) / SRI) * np.exp(-t / tau)
    nz = lp_sos(rs.standard_normal(n), 380.0, 2) * np.exp(-t / 0.018)
    x = (x + 0.55 * nz).astype(np.float32)
    x *= smooth_env(n, 0.003, 0.15)
    return x * np.float32(vel * 0.9)


def frame_drum(vel, rs, f0=118.0, f1=64.0, tau=0.20, d=1.0):
    """soft low frame drum: skin pitch drop + a short dull slap for definition."""
    n = int(d * SRI)
    t = np.arange(n, dtype=np.float64) / SRI
    fr = f1 + (f0 - f1) * np.exp(-t / 0.04)
    body = np.sin(2 * np.pi * np.cumsum(fr) / SRI) * np.exp(-t / tau)
    slap = bp_sos(rs.standard_normal(n), 350.0, 2400.0, 2) * np.exp(-t / 0.022)
    x = (body + 0.22 * slap).astype(np.float32)
    x *= smooth_env(n, 0.002, 0.25)
    return x * np.float32(vel)


def make_ir(rt60, seed, predelay=0.006, damp=3000.0):
    """synthetic stereo impulse response (decorrelated decaying noise, frequency dependent damping)"""
    rs = np.random.default_rng(seed)
    n = int((rt60 * 1.15 + predelay) * SRI)
    t = np.arange(n) / SRI
    ir = np.zeros((2, n), dtype=np.float32)
    for c in range(2):
        nz_lo = lp_sos(rs.standard_normal(n), damp, 1)
        nz_hi = hp_sos(lp_sos(rs.standard_normal(n), 9000.0, 1), damp, 1)
        dec = np.exp(-6.91 * np.maximum(t - predelay, 0) / rt60)
        sig = nz_lo * dec + 0.12 * nz_hi * np.exp(-6.91 * np.maximum(t - predelay, 0) / (rt60 * 0.35))
        # diffuse build-up (no discrete echoes: a smooth onset keeps the reverb from adding rhythmic ripples)
        sig *= (1.0 - np.exp(-np.maximum(t - predelay, 0) / 0.010))
        sig[:int(predelay * SRI)] = 0
        sig = hp_sos(sig, 120.0, 1)
        sig *= smooth_env(n, 0.004, 0.2, pw=1.0)
        ir[c] = sig
    ir /= np.sqrt((ir ** 2).sum(axis=1).mean())
    return ir


# ----------------------------------------------------------------------------------------------
# musical material (semitones above the tonic; tonic4 = 0, so G4 = 7, C5 = 12, G5 = 19)
# ----------------------------------------------------------------------------------------------
# chord dictionary: bass note, arpeggio pool (5 ascending notes), string-pad voicing, cello root
CH = {
    'I':   dict(bass=-24, pool=[-12, -5, 0, 4, 7],   pad=[-12, -5, 0],  cello=-24),
    'IV':  dict(bass=-19, pool=[-7, 0, 4, 9, 12],    pad=[0, 4, 9],     cello=-19),   # Fmaj7
    'V':   dict(bass=-17, pool=[-5, 0, 2, 7, 12],    pad=[-5, 2, 12],   cello=-17),   # G7sus4 colour
    'Vm':  dict(bass=-17, pool=[-5, 2, 7, 11, 14],   pad=[-5, 2, 11],   cello=-17),   # G major
    'vi':  dict(bass=-15, pool=[-3, 0, 4, 7, 12],    pad=[-3, 7, 16],   cello=-15),   # Am7
    'ii':  dict(bass=-22, pool=[-10, -3, 0, 5, 9],   pad=[-10, 0, 9],   cello=-22),   # Dm7
}

# the bridge theme: (start in 8ths, semitone, length in 8ths)   G C D G' E D C
THEME_L = [(0, 7, 3), (3, 12, 1), (4, 14, 1), (5, 19, 3), (8, 16, 1), (9, 14, 1), (10, 12, 2)]      # 12 e
THEME_L11 = [(0, 7, 3), (3, 12, 1), (4, 14, 1), (5, 19, 3), (8, 16, 1), (9, 14, 1), (10, 12, 1)]    # 11 e
THEME_S8 = [(0, 7, 1), (1, 12, 1), (2, 14, 1), (3, 19, 2), (5, 16, 1), (6, 14, 1), (7, 12, 2)]      # 8 e (short)
THEME_C8 = [(0, 7, 2), (2, 12, 1), (3, 14, 1), (4, 19, 2), (6, 16, 1), (7, 14, 1)]                  # 8 e CTA lift
THEME_EXT = [(0, 16, 2), (2, 14, 1), (3, 12, 2), (5, 14, 1)]                                         # 6 e (E extension)
HEAD = [(0, 0, 3), (3, 5, 1), (4, 7, 1)]          # "5 1 2" shape; transposed per chord at call time
COUNTER_L = [(0, 4, 3), (3, 7, 2), (5, 12, 4), (9, 11, 1), (10, 7, 2)]                               # 12 e
COUNTER_S8 = [(0, 4, 2), (2, 7, 1), (3, 12, 3), (6, 11, 1), (7, 7, 2)]                               # 8 e
THIRD_BELOW = {7: 4, 12: 9, 14: 11, 19: 16, 16: 12}


class Score:
    def __init__(self, name, imp, dur, snap=None):
        self.name, self.imp, self.dur = name, imp, dur
        self.snap = dict(snap or {})
        self.action_pts = [(0.0, 1.0)]          # piecewise-linear gain of the action-noise layer over time
        self.ev = {k: [] for k in ('piano', 'strings', 'solo', 'cello', 'heart', 'drum', 'glock', 'harp')}

    def T(self, e):
        """time in seconds of grid position e (8ths from IMPACT); a few downbeats are 'snapped' to the cue"""
        if float(e).is_integer() and int(e) in self.snap:
            return self.snap[int(e)]
        return self.imp + e * H


def pno(S, e, n, vel, dur_e=None, ring=0.0, rel=0.28, t=None, dur_s=None, jit=True, pan=None, act=True, g=1.0):
    t0 = S.T(e) if t is None else t
    if dur_s is None:
        dur_s = (S.T(e + dur_e) - t0) + ring
    S.ev['piano'].append(dict(t=t0, n=n, vel=vel, dur=max(0.12, dur_s), rel=rel, jit=jit, pan=pan, act=act, g=g))


ARP_CFG = dict(ring=0.60, bass_ring=1.20)
BASS_K = 0.70                    # bass-note velocity factor (the bass stays below the pad, out of the sub band's way)
ARP_IDX_VEL = [1.05, 1.32, 1.28, 0.72, 0.58]
# arpeggio cycles: indices into the chord's 5-note pool, None = rest.  CYC6 / CYC8 are the full 8th-note patterns
# (IMPACT bloom, E peak); PULSE8 is the soft low 8th-note pulse that accompanies the melody and fills the rests
# (C#3 G#3 C#4 G#3 ..., at or below ~280 Hz so it stays under the voice band); QTR is a quarter-note variant.
CYC6 = [0, 1, 2, 3, 2, 1]
CYC8 = [0, 1, 2, 3, 4, 3, 2, 1]
QTR = [0, None, 1, None, 2, None, 1, None]
PULSE8 = [0, 1, 2, 1, 0, 1, 2, 1]          # soft low 8th-note pulse (<= 280 Hz) for the accompaniment-only passages
D_CYC, F_CYC = PULSE8, PULSE8                      # accompaniment pattern in D / F (QTR or PULSE8)

# arrangement dynamics (velocity of the accompaniment behind a melody, of the melody itself, and the extra gain
# of the accompaniment-only passages D2 / F2 so that the rests really are rests)
D_ARP, D_MEL, D2_G = 0.11, 0.42, 0.70
F_ARP, F_MEL, F2_G = 0.11, 0.46, 0.70
E_PAD = 0.28
G_FINAL_DUR, G_FINAL_PAD = 0.60, 0.18
G_ROLL = 0.10                                # the final chord is rolled upward over 100 ms, centred on the button click
SD_MEL = 0.34                                # short D: head of the theme
SD_ARP, SD_G = 0.11, 0.70           # short: D accompaniment
SE_NOTES = 7                                 # short E: notes of the theme stated (7 = whole theme, 4 = 5 1 2 5' then a breath)
SE_CTR_K = 0.6                               # short E: countermelody level factor
SE_PAD = 0.28                                # short: E pad
GHOST, G_G = 0.50, 0.35                     # inner pulse notes sit 4 dB under the chord-change strike
G_ARP, G_CHORD_K = 0.13, 1.40                # G lift: soft low pulse; the final chord is the loudest event of the cue
C_MEL, G_MEL = 0.46, 0.24                    # melody velocity at IMPACT / in the G lift
E_MEL, E_DBL, E_UP, E_CTR = 0.48, 0.45, 0.12, 0.28   # E: melody, octave-below doubling, octave-above doubling, string counter
VIOLA_EXPO = 2.6
VIOLA_NOTES = (-8, -5)                       # F3 and G#3: the third and fifth of the tonic, under 210 Hz
DRONE_EXPO = 2.4                             # cello drone spectrum: nearly a pure low tone, so it never fills the voice band
VIOLA_LS = 0.75                              # same, short edit (a touch more: its voice-band share is higher)
VIOLA_L = 0.60                               # inner-voice level (sustained G#3 / C#4 from IMPACT)
E_EXT = 0.846                                # E: level of the closing extension phrase (0 = a held string note instead)
E_BASS, E_G, E_GHOST = 0.60, 1.0, 1.0           # E: arpeggio bass velocity, gain and inner-note ghosting
E_ARP, E_CYC = 0.36, [0, 1, 2, 1, 0, 1, 2, 3]       # E: full 8th-note arpeggio, kept low (<= ~350 Hz)
SG_FINAL_DUR = 0.55


def arp(S, e0, L, ch, vel, cyc=CYC6, bass_vel=None, bass=True, accent=0.12, ring_s=None, g=1.0, act=True, ghost=1.0):
    c = CH[ch]
    ring_s = ARP_CFG['ring'] if ring_s is None else ring_s
    t_end = S.T(e0 + L)
    for k in range(L):
        idx = cyc[k % len(cyc)]
        if idx is None:
            continue
        v = vel * (1.0 + accent * (1 if k % 2 == 0 else -0.8)) * ARP_IDX_VEL[idx]
        if k == 0:
            v *= 1.10
        pno(S, e0 + k, c['pool'][idx], v, dur_s=min(t_end - S.T(e0 + k) + 0.05, ring_s), rel=0.28, act=act,
            g=g * (1.0 if k == 0 else ghost))
    if bass:
        pno(S, e0, c['bass'], (bass_vel * BASS_K) if bass_vel is not None else vel * 1.45 * BASS_K / 0.85,
            dur_s=min(t_end - S.T(e0) + 0.05, ARP_CFG['bass_ring']), rel=0.35, act=act, g=g)


def melody(S, e0, notes, vel, shift=0, dbl=(), ring=0.45, peak=1.12, third=0.0, act=True, g=1.0):
    for (e, n, d) in notes:
        v = vel * (peak if n >= 19 else 1.0)
        pno(S, e0 + e, n + shift, v, dur_e=d, ring=ring, rel=0.40, act=act, g=g)
        for (dn, dv) in dbl:
            if dv <= 0.0:
                continue
            pno(S, e0 + e, n + shift + dn, v * dv, dur_e=d, ring=ring, rel=0.40, g=g)
        if third and n in THIRD_BELOW:
            pno(S, e0 + e, THIRD_BELOW[n] + shift, v * third, dur_e=d, ring=ring, rel=0.40, g=g)


def head(S, e0, root, vel, dur0=3, rel_=0.5):
    """the 'bridge' head 5-1-2 rooted on `root` (semitones): hinted in tension, never resolved"""
    for (e, n, d) in [(0, root, dur0), (dur0, root + 5, 1), (dur0 + 1, root + 7, 2)]:
        pno(S, e0 + e, n, vel * (0.9 + 0.1 * e / 4.0), dur_e=d, ring=0.9, rel=0.5)


def pad(S, e0, e1, ch, level, att=0.8, rel=0.9, voicing=None, lead=0.0, shape=None, t0=None, t1=None):
    t0 = (S.T(e0) - lead) if t0 is None else t0
    t1 = S.T(e1) if t1 is None else t1
    for i, n in enumerate(voicing or CH[ch]['pad']):
        S.ev['strings'].append(dict(t=t0, dur=t1 - t0 + lead, att=att, rel=rel, n=n,
                                    level=level * (1.0 - 0.06 * i), shape=shape))


def solo(S, e, n, d_e, level, att=0.07, rel=0.35, ring=0.15):
    t0 = S.T(e)
    S.ev['solo'].append(dict(t=t0, dur=S.T(e + d_e) - t0 + ring, att=att, rel=rel, n=n, level=level))


def cello(S, t0, t1, n, level, att=0.6, rel=0.9, shape=None, expo=1.6):
    S.ev['cello'].append(dict(t=t0, dur=t1 - t0, att=att, rel=rel, n=n, level=level, shape=shape, expo=expo))


def heart(S, e, kind, vel, t=None):
    S.ev['heart'].append(dict(t=S.T(e) if t is None else t, kind=kind, vel=vel))


def drum(S, e, vel, t=None):
    S.ev['drum'].append(dict(t=S.T(e) if t is None else t, vel=vel))


def glk(S, n, vel, e=None, t=None, jit=None, pan=0.3):
    S.ev['glock'].append(dict(t=S.T(e) if t is None else t, n=n, vel=vel,
                              jit=(t is None) if jit is None else jit, pan=pan))


def hrp(S, n, vel, e=None, t=None, jit=None, pan=-0.2):
    S.ev['harp'].append(dict(t=S.T(e) if t is None else t, n=n, vel=vel,
                             jit=(t is None) if jit is None else jit, pan=pan))


# ----------------------------------------------------------------------------------------------
# arrangements.  All positions are in 8th notes (e) from IMPACT; e = 0 is the downbeat at IMPACT.
# Sustained layers (cello drone, string beds) are long notes with gain automation ("swells") rather than
# chord-by-chord re-attacks: the harmony moves in the piano bass/arpeggio, the beds only breathe.
# ----------------------------------------------------------------------------------------------
def rel_shape(t0, pts):
    return [(t - t0, g) for (t, g) in pts]


def final_chord(S, e, t_end, piano_dur, p_rel, level_str, k=1.0, roll=0.0):
    """thin, dignified tonic chord: bass C#2 + C#3, F4 G#4 C#5, top C#6; strings low; cello keeps the pedal.
    roll > 0 spreads the six notes upward over `roll` seconds, centred on the downbeat (a rolled final chord)."""
    notes = [(-24, 0.60), (-12, 0.42), (4, 0.46), (7, 0.44), (12, 0.48), (24, 0.52)]
    for i, (n, v) in enumerate(notes):
        t = S.T(e) + roll * (i / (len(notes) - 1.0) - 0.5)
        pno(S, e, n, min(v * k, 0.95), t=t, dur_s=piano_dur, rel=p_rel, act=(n == -24), jit=(roll == 0.0))
    tf = S.T(e)
    pad(S, e, e, 'I', level_str, att=0.22, rel=1.6, voicing=[-17, -12, -5], t0=tf, t1=t_end)


def build_long():
    S = Score('long', 15.4, 57.6, snap={
        -5: 13.60,                             # timeline cracks
        12: 20.07, 73: 42.98,                  # how-it-works begins, why-owners scene begins (cue-snapped downbeats)
        55: 36.285, 104: 54.82})               # nudged <= 25 ms toward the impact scene / button click

    S.action_pts = [(0.0, AG_QUIET), (13.0, AG_QUIET), (15.4, 1.0)]
    # ---- cello pedal drone: A (vi) through the tension, C (tonic) from IMPACT to the end ------------------
    cello(S, 0.0, S.T(-1) + 0.4, -15, 0.22, att=2.6, rel=0.9, expo=DRONE_EXPO,
          shape=rel_shape(0.0, [(0, 0.45), (5.5, 0.65), (11.0, 0.9), (14.5, 1.0), (15.1, 0.9)]))
    cello(S, 0.3, S.T(-1) + 0.3, -8, 0.10, att=3.0, rel=0.9, expo=DRONE_EXPO,
          shape=rel_shape(0.3, [(0.3, 0.5), (8, 0.8), (15.1, 1.0)]))
    drone_shape = [(15.4, 1.0), (19.0, 0.95), (20.5, 0.55), (30.0, 0.6), (35.0, 0.85), (36.3, 1.05), (42.0, 1.0),
                   (43.5, 0.60), (50.0, 0.65), (51.8, 1.0), (54.79, 1.25), (57.6, 0.7)]
    cello(S, 15.4, 57.6, -24, 0.30, att=0.35, rel=1.0, shape=rel_shape(15.4, drone_shape), expo=DRONE_EXPO)
    cello(S, 15.45, 57.6, -17, 0.12, att=0.5, rel=1.0, shape=rel_shape(15.45, drone_shape), expo=DRONE_EXPO)
    for vn in VIOLA_NOTES:                                 # viola / cello inner voices: the warm 150-300 Hz body
        cello(S, 15.5, 57.6, vn, VIOLA_L, att=0.6, rel=1.0, shape=rel_shape(15.5, drone_shape), expo=VIOLA_EXPO)

    # ---- A  the leak (e -40 .. -26): quiet unease, a few piano notes that stop short -----------------------
    pno(S, -38, -15, 0.24, dur_s=3.4)                                                 # soft A2 pedal tone
    pno(S, -37, 4, 0.30, dur_s=2.0)                                                   # headline-2 lands: E4
    pno(S, -35, 9, 0.27, dur_e=2, ring=0.6)                                           # A4
    pno(S, -32, 11, 0.30, dur_e=6, ring=0.5, rel=0.6)                                 # B4 (hangs: 5-1-2 stopped short)
    pno(S, -28, 4, 0.18, dur_e=2, ring=0.9, rel=0.6)                                  # E4 ghost

    # ---- B  the cost (e -26 .. -1): head 5-1-2 rises through vi, IV, ii, V; heartbeat quickens -----------------
    lubs = [-26, -23, -20, -17, -14, -11, -8, -5, -2]
    for i, e in enumerate(lubs):
        v = 0.30 + 0.38 * i / (len(lubs) - 1)
        heart(S, e, 'lub', v)
        if e != -2:                                   # the last dub is missing: a visit is missed
            heart(S, e + 1, 'dub', v * 0.9)
    pno(S, -26, CH['vi']['bass'], 0.32, dur_e=8, ring=0.2, rel=0.5)                   # B1  vi
    pno(S, -26, -8, 0.20, dur_e=8, ring=0.2, rel=0.5)
    head(S, -24, 4, 0.36)                                                             # E4 A4 B4
    pno(S, -18, CH['IV']['bass'], 0.34, dur_e=8, ring=0.2, rel=0.5)                   # B2  IV
    pno(S, -18, -12, 0.20, dur_e=8, ring=0.2, rel=0.5)
    head(S, -16, 0, 0.38)                                                             # C4 F4 G4
    pad(S, -14, -10, 'IV', 0.05, att=2.2, rel=1.0, voicing=[0, 9])
    pno(S, -10, CH['ii']['bass'], 0.36, dur_e=5, ring=0.2, rel=0.5)                   # B3  ii
    pno(S, -10, -10, 0.22, dur_e=5, ring=0.2, rel=0.5)
    for (e, n, d, v) in [(-9, 9, 1, 0.40), (-8, 14, 1, 0.43), (-7, 16, 3, 0.46)]:     # A4 D5 E5 (b3 cue at e-9)
        pno(S, e, n, v, dur_e=d, ring=0.8, rel=0.5)
    pad(S, -10, -5, 'ii', 0.08, att=1.5, rel=0.7, voicing=[-10, 0, 9])
    pno(S, -5, -17, 0.62, dur_s=2.0, jit=False, rel=0.6)                              # crack at 13.6 s: G2 / Ab2 cluster
    pno(S, -5, -16, 0.55, dur_s=2.0, jit=False, rel=0.6)
    glk(S, 20, 0.55, t=13.60, jit=False, pan=0.35)                                    # Ab6 ping
    pno(S, -5, CH['V']['bass'], 0.30, dur_e=4, ring=0.2, rel=0.5)
    pad(S, -5, -1, 'V', 0.18, att=1.3, rel=0.55, voicing=[-5, 2, 8, 12])
    for (e, n, v) in [(-4, 14, 0.50), (-3, 19, 0.54), (-2, 21, 0.52)]:                # D5 G5 A5 hanging on the 9th
        pno(S, e, n, v, dur_e=1.0, ring=0.55, rel=0.55)
    # breath: e -1 .. 0 nothing but the reverb tail

    # ---- C  the bridge (e 0 .. 12): the theme, strings bloom into major --------------------------------------
    pad(S, 0, 12, 'I', 0.40, att=0.30, rel=0.9, voicing=[-17, -12, -5], t1=S.T(12) + 0.3,
        shape=rel_shape(15.4, [(15.4, 0.8), (16.6, 1.0), (19.0, 0.9), (20.0, 0.5)]))
    arp(S, 0, 6, 'I', 0.27)
    arp(S, 6, 3, 'IV', 0.27)
    arp(S, 9, 3, 'V', 0.27)
    melody(S, 0, THEME_L, C_MEL)
    drum(S, 0, 0.85)
    drum(S, 6, 0.30)
    glk(S, 24, 0.55, e=0)                                                             # C6 sparkle
    for i, n in enumerate([12, 16, 19, 24]):
        hrp(S, n, 0.42, e=1 + i)                                                      # rising harp over the bloom
    glk(S, 31, 0.62, t=17.30, jit=False, pan=0.4)                                     # logo lands (G6)
    hrp(S, 28, 0.42, e=8)

    # ---- D  how it works (e 12 .. 55): friendly groove; the theme speaks in phrases with rests -----------------------
    #      D1 melody over a quarter-note ostinato | D2 accompaniment only (vi - IV - V - I), a rest for the ear and for
    #      the voice-over | D3 melody returns in thirds with a rising-energy bed
    pad(S, 12, 26, 'I', 0.12, att=1.0, rel=1.0, voicing=[-17, -12, -5], lead=0.3, t1=S.T(26) + 0.3)
    pad(S, 26, 42, 'I', 0.14, att=1.0, rel=1.0, voicing=[-17, -12, -5], lead=0.3, t1=S.T(42) + 0.3)
    pad(S, 42, 55, 'I', 0.20, att=1.4, rel=0.8, voicing=[-17, -12, -5], lead=0.5, t1=S.T(55) + 0.1,
        shape=rel_shape(S.T(42) - 0.5, [(S.T(42) - 0.5, 0.6), (S.T(54), 1.25)]))
    for (a, L, ch) in [(12, 6, 'I'), (18, 3, 'IV'), (21, 3, 'V'), (24, 2, 'I')]:      # D1: step 1 (20.1 s)
        arp(S, a, L, ch, D_ARP, cyc=D_CYC, bass_vel=0.40, ghost=GHOST)
        drum(S, a, 0.28)
    melody(S, 12, THEME_L, D_MEL)
    glk(S, 26, 0.45, t=20.10, jit=False)                                              # how-it-works begins (D6)
    glk(S, 31, 0.30, e=17)
    for (a, L, ch) in [(26, 6, 'vi'), (32, 3, 'IV'), (35, 3, 'Vm'), (38, 4, 'I')]:    # D2: step 2 vi - IV - V (no melody)
        arp(S, a, L, ch, D_ARP, cyc=D_CYC, bass_vel=0.38, g=D2_G, ghost=GHOST)
        drum(S, a, 0.30)
    hrp(S, 19, 0.20, e=35)                                                            # English -> Kiswahili flip (a whisper)
    for (a, L, ch) in [(42, 6, 'I'), (48, 3, 'IV'), (51, 3, 'V'), (54, 1, 'V')]:      # D3: step 3, rising energy
        arp(S, a, L, ch, D_ARP + 0.02, cyc=D_CYC, bass_vel=0.44, ghost=GHOST)
    melody(S, 43, THEME_S8, D_MEL + 0.04, dbl=[(12, 0.20)], third=0.40)             # compressed, thirds, then a breath before E
    drum(S, 42, 0.34)
    drum(S, 48, 0.30)
    for i, e in enumerate(range(49, 55)):
        drum(S, e, 0.22 + 0.07 * i)                                                   # soft roll into the peak
    glk(S, 31, 0.38, e=47)

    # ---- E  the impact (e 55 .. 73): octave-doubled melody + countermelody, full strings -----------------------
    pad(S, 55, 73, 'I', E_PAD, att=0.30, rel=1.0, voicing=[-17, -12, -5], t0=S.T(55) - 0.05, t1=S.T(73) + 0.2,
        shape=rel_shape(S.T(55) - 0.05, [(S.T(55) - 0.05, 0.7), (S.T(55) + 1.6, 1.0), (S.T(66), 1.1), (S.T(73), 0.5)]))
    for (a, L, ch) in [(55, 6, 'I'), (61, 3, 'IV'), (64, 3, 'V'), (67, 3, 'vi'), (70, 3, 'V')]:
        arp(S, a, L, ch, E_ARP, cyc=E_CYC, bass_vel=E_BASS, g=E_G, ghost=E_GHOST)
    melody(S, 55, THEME_L, E_MEL, dbl=[(-12, E_DBL), (12, E_UP)])
    if E_EXT > 0:
        melody(S, 67, THEME_EXT, E_MEL * E_EXT, dbl=[(-12, E_DBL * 0.9)])
    else:
        solo(S, 67, 7, 6, E_CTR * 0.9, att=0.30, rel=0.8, ring=0.2)                      # held G#4: a breath, no new phrase
    for (e, n, d) in COUNTER_L:
        solo(S, 55 + e, n, d, E_CTR)
    drum(S, 55, 0.90)
    for e in (61, 64, 67, 70):
        drum(S, e, 0.40)
    glk(S, 31, 0.48, e=60)
    hrp(S, 24, 0.45, e=55)
    hrp(S, 28, 0.42, e=56)
    hrp(S, 21, 0.42, e=67)
    hrp(S, 24, 0.42, e=70)

    # ---- F  why owners choose us (e 73 .. 96): steady, trustworthy -------------------------------------------------
    #      F1 melody over the quarter-note ostinato | F2 accompaniment only (theme only rung on the glock)
    pad(S, 73, 96, 'I', 0.14, att=1.2, rel=1.0, voicing=[-17, -12, -5], lead=0.3, t1=S.T(96) + 0.3)
    for (a, L, ch) in [(73, 6, 'I'), (79, 3, 'IV'), (82, 2, 'V')]:
        arp(S, a, L, ch, F_ARP, cyc=F_CYC, bass_vel=0.44, ghost=GHOST)
    melody(S, 73, THEME_S8, F_MEL)                                                    # the theme, steady and compact
    for (a, L, ch) in [(84, 6, 'I'), (90, 3, 'IV'), (93, 3, 'V')]:
        arp(S, a, L, ch, F_ARP, cyc=F_CYC, bass_vel=0.42, g=F2_G, ghost=GHOST)
    for e in range(73, 96, 4):
        drum(S, e, 0.34)
    glk(S, 33, 0.40, t=43.00, jit=False)
    glk(S, 31, 0.38, e=78)
    glk(S, 31, 0.18, e=89)

    # ---- G  call to action (e 96 .. end): lift, final tonic at the button click, ring-out ------------------------
    for i, n in enumerate([16, 19, 24]):                                              # navy wipe -> rising harp (50.8 s)
        hrp(S, n, 0.28 + 0.04 * i, e=93 + i)
    pad(S, 92, 104, 'I', 0.22, att=2.0, rel=0.4, voicing=[-17, -12, -5], t0=S.T(92), t1=S.T(104) + 0.1,
        shape=rel_shape(S.T(92), [(S.T(92), 0.5), (S.T(96), 0.9), (S.T(104), 1.3)]))
    for (a, L, ch) in [(96, 2, 'I'), (98, 2, 'IV'), (100, 2, 'vi')]:                   # e102-e104 is a breath: only the beds
        arp(S, a, L, ch, G_ARP + 0.01 * (a - 96) / 2.0, cyc=PULSE8, bass_vel=0.50, g=G_G, ghost=GHOST)
    melody(S, 96, THEME_C8[:4], G_MEL, ring=0.35)                                     # 5 1 2 5' ... then a breath on the V
    drum(S, 96, 0.80)
    for e in (98, 100, 102):
        drum(S, e, 0.42 + 0.06 * (e - 98) / 2)
    glk(S, 31, 0.30, e=100)
    final_chord(S, 104, 57.6 - 1.0, G_FINAL_DUR, 0.9, G_FINAL_PAD, G_CHORD_K, G_ROLL)                    # final tonic on the click
    drum(S, 104, 0.80)
    glk(S, 24, 0.42, e=104)
    glk(S, 31, 0.30, e=105)
    return S


def build_short():
    S = Score('short', 6.0, 18.0, snap={-1: 5.50, 3: 7.12, 8: 9.015, 17: 12.47})

    S.action_pts = [(0.0, 0.55), (4.5, 0.55), (6.0, 1.0)]
    cello(S, 0.0, S.T(-1) + 0.3, -15, 0.22, att=1.6, rel=0.8, expo=DRONE_EXPO,
          shape=rel_shape(0.0, [(0, 0.5), (3.0, 0.7), (5.0, 1.0), (5.7, 0.9)]))
    cello(S, 0.2, S.T(-1) + 0.2, -8, 0.10, att=1.8, rel=0.8, expo=DRONE_EXPO)
    dshape = [(6.0, 1.0), (7.6, 0.95), (8.7, 0.55), (11.5, 0.7), (12.3, 1.05), (15.5, 1.0), (16.6, 1.25), (18.0, 0.55)]
    cello(S, 6.0, 18.0, -24, 0.30, att=0.3, rel=0.8, shape=rel_shape(6.0, dshape), expo=DRONE_EXPO)
    cello(S, 6.05, 18.0, -17, 0.12, att=0.5, rel=0.8, shape=rel_shape(6.05, dshape), expo=DRONE_EXPO)
    for vn in VIOLA_NOTES:
        cello(S, 6.1, 18.0, vn, VIOLA_LS, att=0.5, rel=0.8, shape=rel_shape(6.1, dshape), expo=VIOLA_EXPO)

    # ---- A (0 .. 3 s)
    pno(S, -13, 4, 0.30, dur_s=1.6)                                                   # E4 headline 2
    pno(S, -14, -15, 0.24, dur_s=2.2)
    pno(S, -10, 9, 0.27, dur_e=1, ring=0.5)                                           # A4
    pno(S, -9, 11, 0.30, dur_e=2, ring=0.6, rel=0.5)                                  # B4 stops short

    # ---- B (3 .. 6 s): heartbeat quickens, head rises, cut before the reveal
    for (e, k, v) in [(-8, 'lub', 0.34), (-7, 'dub', 0.32), (-5, 'lub', 0.42), (-4, 'dub', 0.40),
                      (-3, 'lub', 0.55), (-2, 'dub', 0.52)]:
        heart(S, e, k, v)
    pno(S, -8, CH['vi']['bass'], 0.32, dur_e=4, ring=0.2, rel=0.5)
    pno(S, -8, -8, 0.20, dur_e=4, ring=0.2, rel=0.5)
    head(S, -7, 4, 0.38, dur0=2)                                                      # E4 A4 B4
    pad(S, -4, -1, 'V', 0.15, att=0.9, rel=0.45, voicing=[-5, 2, 8, 12])
    pno(S, -4, CH['V']['bass'], 0.40, dur_e=3, ring=0.2, rel=0.5)
    for (e, n, v) in [(-4, 14, 0.50), (-3, 19, 0.54), (-2, 21, 0.52)]:                # D5 G5 A5 (unresolved)
        pno(S, e, n, v, dur_e=1.0, ring=0.45, rel=0.45)
    pno(S, -1, -17, 0.55, t=5.50, dur_s=0.9, jit=False, rel=0.4)                      # crack cluster
    pno(S, -1, -16, 0.50, t=5.50, dur_s=0.9, jit=False, rel=0.4)
    glk(S, 20, 0.50, t=5.50, jit=False, pan=0.35)

    # ---- C (6 .. 9 s): the theme, strings bloom
    pad(S, 0, 8, 'I', 0.40, att=0.28, rel=0.8, voicing=[-17, -12, -5], t1=S.T(8) + 0.3,
        shape=rel_shape(6.0, [(6.0, 0.8), (7.0, 1.0), (8.5, 0.6)]))
    arp(S, 0, 4, 'I', 0.27)
    arp(S, 4, 2, 'IV', 0.27)
    arp(S, 6, 2, 'V', 0.27)
    melody(S, 0, THEME_S8, C_MEL)
    drum(S, 0, 0.85)
    glk(S, 24, 0.55, e=0)
    for i, n in enumerate([12, 16, 19]):
        hrp(S, n, 0.42, e=1 + i)
    glk(S, 31, 0.60, e=3, pan=0.4)                                                    # logo lands

    # ---- D (9 .. 12.5 s): groove; only the head "5 1 2" of the theme, then the soft low pulse alone (a rest for the voice)
    pad(S, 8, 17, 'I', 0.13, att=0.8, rel=0.8, voicing=[-17, -12, -5], lead=0.2, t1=S.T(17) + 0.2)
    for (a, L, ch) in [(8, 4, 'I'), (12, 2, 'IV'), (14, 2, 'V'), (16, 1, 'I')]:
        arp(S, a, L, ch, SD_ARP, cyc=D_CYC, bass_vel=0.40, g=SD_G, ghost=GHOST)
        drum(S, a, 0.28)
    melody(S, 8, [(0, 7, 1), (1, 12, 1), (2, 14, 2)], SD_MEL)
    glk(S, 31, 0.20, e=12)

    # ---- E (12.5 .. 15.5 s): peak, octave-doubled melody + countermelody
    pad(S, 17, 25, 'I', SE_PAD, att=0.28, rel=0.8, voicing=[-17, -12, -5], t0=S.T(17) - 0.05, t1=S.T(25) + 0.2,
        shape=rel_shape(S.T(17) - 0.05, [(S.T(17) - 0.05, 0.75), (S.T(17) + 1.2, 1.0), (S.T(25), 0.7)]))
    for (a, L, ch) in [(17, 4, 'I'), (21, 2, 'IV'), (23, 2, 'V')]:
        arp(S, a, L, ch, E_ARP, cyc=E_CYC, bass_vel=E_BASS, g=E_G, ghost=E_GHOST)
    melody(S, 17, THEME_S8[:int(SE_NOTES)], E_MEL, dbl=[(-12, E_DBL), (12, E_UP)])
    for (e, n, d) in COUNTER_S8:
        solo(S, 17 + e, n, d, E_CTR * SE_CTR_K)
    drum(S, 17, 0.90)
    drum(S, 21, 0.40)
    drum(S, 23, 0.45)
    glk(S, 31, 0.48, e=20)
    hrp(S, 24, 0.45, e=17)
    hrp(S, 28, 0.42, e=18)

    # ---- G (15.5 .. 18 s): lift, final tonic on the button click, short ring-out
    for i, n in enumerate([12, 16, 19]):
        hrp(S, n, 0.22 + 0.03 * i, e=25 + i)
    pad(S, 25, 28, 'I', 0.26, att=0.35, rel=0.3, voicing=[-17, -12, -5], t1=S.T(28) + 0.05,
        shape=rel_shape(S.T(25), [(S.T(25), 0.7), (S.T(28), 1.3)]))
    for (a, ch) in [(25, 'I'), (26, 'IV')]:                                              # e27 is a breath before the click
        arp(S, a, 1, ch, G_ARP + 0.02, cyc=PULSE8, bass_vel=0.50, g=G_G, ghost=GHOST)
    drum(S, 25, 0.80)
    drum(S, 27, 0.55)
    glk(S, 31, 0.20, e=27)
    final_chord(S, 28, 18.0 - 0.9, SG_FINAL_DUR, 0.55, G_FINAL_PAD, G_CHORD_K, G_ROLL)
    drum(S, 28, 0.80)
    glk(S, 24, 0.60, e=28)
    return S


# ----------------------------------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------------------------------
GAIN = dict(piano=1.0, strings=0.25, solo=0.30, cello=0.15, heart=0.45, drum=0.24, glock=0.26, harp=0.30, action=1.0)
SEND = dict(piano=0.22, strings=0.26, solo=0.26, cello=0.10, heart=0.05, drum=0.07, glock=0.40, harp=0.35, action=0.0)
SPARKLE_DUR = dict(glock=1.4, harp=1.5)   # glock / harp ring lengths (s): sparkle, not a sustained bed
STR_CFG = dict(pad_det=2.2, pad_vib=0.8, solo_det=2.0, solo_vib=1.2, cello_det=2.0, cello_vib=0.8)
PAD_TOP, PAD_EXPO = 1000.0, 2.3   # string-pad spectrum: mellower than a bright saw ensemble
TILT_GAIN = 1.0                 # gentle 3-6.5 kHz lift on the glock / harp stems (and their reverb send)
WET_AMP = 0.17                 # wet/dry amplitude ratio (about 20 % wet), IR is energy-normalised
JITTER_SD = 0.0060             # seeded humanisation, seconds (clipped to +-12 ms)


def place(stem, x, t, gain=1.0, pan=0.0):
    """add a mono (n,) or stereo (2, n) signal into stem (2, N) at time t"""
    i = int(round(t * SRI))
    if x.ndim == 1:
        x = x[None, :]
    n = x.shape[1]
    if i < 0:
        x = x[:, -i:]
        n = x.shape[1]
        i = 0
    if i >= stem.shape[1] or n <= 0:
        return
    n = min(n, stem.shape[1] - i)
    if x.shape[0] == 1:
        gl, gr = pan_gains(pan)
        stem[0, i:i + n] += x[0, :n] * np.float32(gain * gl)
        stem[1, i:i + n] += x[0, :n] * np.float32(gain * gr)
    else:
        stem[:, i:i + n] += x[:, :n] * np.float32(gain)


def render_stems(S, seed):
    N = int(round(S.dur * SRI))
    stems = {k: np.zeros((2, N), dtype=np.float32) for k in GAIN}
    rs = np.random.default_rng(seed)
    rj = np.random.default_rng(seed + 101)

    def jit(ev, sd=JITTER_SD):
        d = float(np.clip(rj.normal(0.0, sd), -0.012, 0.012))
        return ev['t'] + d if ev.get('jit', True) else ev['t']

    strikes = []
    ra = np.random.default_rng(seed + 202)                 # separate stream: the action ticks never shift the notes
    for ev in sorted(S.ev['piano'], key=lambda d: (round(d['t'], 4), d['n'])):
        t = jit(ev)
        m = rel2m(ev['n'])
        vel = float(np.clip(ev['vel'] * (1.0 + rj.normal(0, 0.045)), 0.05, 0.98))
        x = piano_note(m, vel, ev['dur'], ev['rel'], rs)
        pan = ev['pan'] if ev['pan'] is not None else float(np.clip((m - 62) / 48.0, -0.5, 0.5) + rj.normal(0, 0.03))
        place(stems['piano'], x, t, ev.get('g', 1.0), pan)
        strikes.append((t, vel, bool(ev.get('act', False)), float(ev.get('g', 1.0))))
    # key-action knock: one soft burst per key-strike event (notes within 15 ms = one event)
    strikes.sort()
    i = 0
    while i < len(strikes):
        j = i
        vmax = strikes[i][1]
        fire = strikes[i][2]
        gmax = strikes[i][3]
        while j + 1 < len(strikes) and strikes[j + 1][0] - strikes[i][0] < 0.015:
            j += 1
            vmax = max(vmax, strikes[j][1])
            fire = fire or strikes[j][2]
            gmax = max(gmax, strikes[j][3])
        if fire:
            ag = float(np.interp(strikes[i][0], [p[0] for p in S.action_pts], [p[1] for p in S.action_pts]))
            place(stems['action'], action_burst(vmax, ra, ag * gmax ** ACTION_CFG['gpow']), strikes[i][0] - ACTION_CFG['pre'], 1.0,
                  float(ra.normal(0, 0.15)))
        i = j + 1

    for ev in sorted(S.ev['strings'], key=lambda d: (round(d['t'], 4), d['n'])):
        f = float(mtof(rel2m(ev['n'])))
        x = string_note(f, ev['dur'], ev['att'], ev['rel'], rs, shape=ev.get('shape'), top=PAD_TOP, expo=PAD_EXPO,
                        detune=(-STR_CFG['pad_det'], 0.0, STR_CFG['pad_det']), vib_cents=STR_CFG['pad_vib'])
        place(stems['strings'], x, ev['t'], ev['level'])
    for ev in sorted(S.ev['solo'], key=lambda d: (round(d['t'], 4), d['n'])):
        f = float(mtof(rel2m(ev['n'])))
        x = string_note(f, ev['dur'], ev['att'], ev['rel'], rs,
                        detune=(-STR_CFG['solo_det'], 0.0, STR_CFG['solo_det']), vib_cents=STR_CFG['solo_vib'],
                        vib_delay=0.8, top=2200.0, expo=1.7, pans=(-0.3, 0.0, 0.3))
        place(stems['solo'], x, ev['t'], ev['level'])
    for ev in sorted(S.ev['cello'], key=lambda d: (round(d['t'], 4), d['n'])):
        f = float(mtof(rel2m(ev['n'])))
        x = string_note(f, ev['dur'], ev['att'], ev['rel'], rs, detune=(-STR_CFG['cello_det'], STR_CFG['cello_det']),
                        vib_cents=STR_CFG['cello_vib'], vib_delay=1.2,
                        top=480.0, expo=ev.get('expo', 1.6), pans=(-0.15, 0.15), shape=ev.get('shape'))
        place(stems['cello'], x, ev['t'], ev['level'])
    for ev in sorted(S.ev['heart'], key=lambda d: round(d['t'], 4)):
        place(stems['heart'], heart_hit(ev['kind'], ev['vel'], rs), ev['t'], 1.0, 0.0)
    for ev in sorted(S.ev['drum'], key=lambda d: round(d['t'], 4)):
        place(stems['drum'], frame_drum(ev['vel'], rs), ev['t'], 1.0, 0.0)
    for ev in sorted(S.ev['glock'], key=lambda d: (round(d['t'], 4), d['n'])):
        place(stems['glock'], glock(rel2m(ev['n']), ev['vel'], rs, SPARKLE_DUR['glock']), jit(ev, 0.003), 1.0, ev['pan'])
    for ev in sorted(S.ev['harp'], key=lambda d: (round(d['t'], 4), d['n'])):
        place(stems['harp'], harp_note(rel2m(ev['n']), ev['vel'], rs, SPARKLE_DUR['harp']), jit(ev, 0.003), 1.0, ev['pan'])

    # light per-stem tone shaping
    stems['strings'] = hp_sos(stems['strings'], 110.0, 2).astype(np.float32)
    stems['solo'] = hp_sos(stems['solo'], 200.0, 2).astype(np.float32)
    stems['piano'] = hp_sos(stems['piano'], 38.0, 2).astype(np.float32)
    return stems


def mix_stems(stems, ir):
    N = stems['piano'].shape[1]
    dry = np.zeros((2, N), dtype=np.float32)
    send = np.zeros(N, dtype=np.float32)
    for k, s in stems.items():
        dry += s * np.float32(GAIN[k])
        send += 0.5 * (s[0] + s[1]) * np.float32(GAIN[k] * SEND[k])
    if TILT_GAIN > 0.0:
        for k in ('glock', 'harp'):
            tl = bp_sos(stems[k], 3000.0, 6500.0, 1).astype(np.float32) * np.float32(TILT_GAIN * GAIN[k])
            dry += tl
            send += 0.5 * (tl[0] + tl[1]) * np.float32(SEND[k])
    send = hp_sos(send, 160.0, 2).astype(np.float32)
    wet = np.stack([signal.fftconvolve(send, ir[0])[:N], signal.fftconvolve(send, ir[1])[:N]]).astype(np.float32)
    pd = float((dry ** 2).mean())
    pw = float((wet ** 2).mean()) + 1e-20
    wet *= np.float32(WET_AMP * np.sqrt(pd / pw))
    return dry + wet


# ----------------------------------------------------------------------------------------------
# mastering: section loudness arc, fades, loudness target, peak limiter
# ----------------------------------------------------------------------------------------------
G_REL_L, G_REL_S = 1.2, 1.2      # G (call to action) section level relative to E
SEC_LONG = [  # id, start, end, relative level (dB) -> follows the energy arc in the cue sheet
    ('A', 0.0, 5.5, -13.0), ('B', 5.5, 15.1, -8.5), ('C', 15.1, 20.1, -1.0), ('D', 20.1, 36.3, -4.2),
    ('E', 36.3, 43.0, 0.0), ('F', 43.0, 51.8, -2.4), ('G', 51.8, 57.6, G_REL_L)]
SEC_SHORT = [
    ('A', 0.0, 3.0, -13.0), ('B', 3.0, 6.0, -9.0), ('C', 6.0, 9.0, -1.0), ('D', 9.0, 12.5, -4.2),
    ('E', 12.5, 15.5, 0.0), ('G', 15.5, 18.0, G_REL_S)]
RAMPS = {'long': {15.1: (14.95, 15.30)}, 'short': {6.0: (5.72, 5.98)}}
TARGET_LUFS = -18.0
AG_QUIET = 0.35                # action-noise gain in the tension sections
PEAK_CEIL = 0.86
FADE = {'long': 55.55, 'short': 16.66}
FADE_POW = {'long': 2.0, 'short': 4.0}


def rms_db(x):
    return 10.0 * np.log10(float(np.mean(x.astype(np.float64) ** 2)) + 1e-20)


def gain_curve(N, sections, gdb, ramps, xf=0.40):
    g = np.full(N, gdb[0], dtype=np.float64)
    for i in range(1, len(sections)):
        b = sections[i][1]
        r0, r1 = ramps.get(b, (b - xf, b))
        i0, i1 = int(r0 * SR), int(r1 * SR)
        x = np.linspace(0, 1, max(2, i1 - i0))
        s = 0.5 - 0.5 * np.cos(np.pi * x)
        g[i0:i1] = gdb[i - 1] + (gdb[i] - gdb[i - 1]) * s[:i1 - i0]
        g[i1:] = gdb[i]
    return 10.0 ** (g / 20.0)


def level_arc(y, sections, video, iters=5):
    N = y.shape[1]
    rel = np.array([s[3] for s in sections])
    dur = np.array([s[2] - s[1] for s in sections])
    gdb = np.zeros(len(sections))
    ramps = RAMPS[video]
    mono0 = y.mean(0)
    p0 = np.sum(10 ** (np.array([rms_db(mono0[int(s[1] * SR):int(s[2] * SR)]) for s in sections]) / 10.0) * dur)
    for _ in range(iters):
        z = y * gain_curve(N, sections, gdb, ramps)[None, :]
        m = z.mean(0)
        meas = np.array([rms_db(m[int(s[1] * SR):int(s[2] * SR)]) for s in sections])
        # absolute target: keep the overall energy that the arrangement had
        K = 10 * np.log10(p0 / np.sum(10 ** (rel / 10.0) * dur))
        gdb += np.clip((rel + K) - meas, -12, 12) * 0.9
        gdb = np.clip(gdb, -20, 20)
    return y * gain_curve(N, sections, gdb, ramps)[None, :].astype(np.float32), gdb


def lufs(y):
    b1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]
    a1 = [1.0, -1.69065929318241, 0.73248077421585]
    b2 = [1.0, -2.0, 1.0]
    a2 = [1.0, -1.99004745483398, 0.99007225036621]
    z = signal.lfilter(b2, a2, signal.lfilter(b1, a1, y.astype(np.float64), axis=-1), axis=-1)
    n, st = int(0.4 * SR), int(0.1 * SR)
    cs = np.cumsum(np.concatenate([np.zeros((2, 1)), z ** 2], axis=1), axis=1)
    idx = np.arange(0, z.shape[1] - n + 1, st)
    ms = ((cs[:, idx + n] - cs[:, idx]) / n).sum(axis=0)
    l = -0.691 + 10 * np.log10(ms + 1e-20)
    ms1 = ms[l > -70.0]
    rel_thr = -0.691 + 10 * np.log10(ms1.mean()) - 10.0
    ms2 = ms[l > rel_thr]
    return -0.691 + 10 * np.log10(ms2.mean())


def limit(y, ceil=PEAK_CEIL, look=0.004):
    pk = np.abs(y).max(axis=0).astype(np.float64)
    need = np.minimum(1.0, ceil / np.maximum(pk, 1e-9))
    w = int(look * SR)
    g = minimum_filter1d(need, size=2 * w + 1)
    k = np.hanning(2 * w + 1)
    k /= k.sum()
    g = signal.fftconvolve(g, k, mode='same')
    g = np.minimum(g, 1.0)
    return (y * g[None, :]).astype(np.float32)


def master(y_i, video, S):
    """y_i: (2, N_i) float at SRI -> 48 kHz mastered stereo float"""
    y = signal.resample_poly(y_i, 2, 1, axis=1).astype(np.float32)
    N = int(round(S.dur * SR))
    y = y[:, :N]
    if y.shape[1] < N:
        y = np.pad(y, ((0, 0), (0, N - y.shape[1])))
    y = hp_sos(y, 30.0, 2, sr=SR).astype(np.float32)
    y = lp_sos(y, 9500.0, 1, sr=SR).astype(np.float32)
    # end fade (cos^2) to exact zero on the last sample
    i0 = int(FADE[video] * SR)
    x = np.linspace(0.0, 1.0, N - i0)
    fade = np.ones(N)
    fade[i0:] = np.cos(0.5 * np.pi * x) ** FADE_POW[video]
    y = y * fade[None, :].astype(np.float32)
    sections = list(SEC_LONG if video == 'long' else SEC_SHORT)
    sections[-1] = sections[-1][:3] + ((G_REL_L if video == 'long' else G_REL_S),)
    y, gdb = level_arc(y, sections, video)
    for _ in range(3):
        L = lufs(y)
        y = y * np.float32(10 ** ((TARGET_LUFS - L) / 20.0))
        y = limit(y)
    y[:, -1] = 0.0
    return y, gdb


def write_wav(path, y):
    q = np.clip(np.round(y.T * 32767.0), -32768, 32767).astype(np.int16)
    wavfile.write(path, SR, q)


DEFAULT_KEY = 'Db major'


def render_video(video, out_dir, ir):
    S = build_long() if video == 'long' else build_short()
    stems = render_stems(S, seed=20240 if video == 'long' else 20241)
    y = mix_stems(stems, ir)
    y, gdb = master(y, video, S)
    path = os.path.join(out_dir, 'piano_%s_music.wav' % video)
    write_wav(path, y)
    return path, gdb


def main(argv=None):
    ap = argparse.ArgumentParser(description='HealthBridge piano score (procedural synthesis)')
    ap.add_argument('--out-dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out'))
    ap.add_argument('--key', default=DEFAULT_KEY, choices=sorted(KEYS))
    ap.add_argument('--only', choices=['long', 'short'])
    a = ap.parse_args(argv)
    set_key(a.key)
    os.makedirs(a.out_dir, exist_ok=True)
    ir = make_ir(2.7, seed=7)
    for v in ('long', 'short'):
        if a.only and a.only != v:
            continue
        p, gdb = render_video(v, a.out_dir, ir)
        print('wrote', p, 'section gains dB', np.round(gdb, 1).tolist())


if __name__ == '__main__':
    main()
