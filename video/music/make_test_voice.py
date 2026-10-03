#!/usr/bin/env python3
"""Synthesise a SPEECH-LIKE TEST SIGNAL (not intelligible speech!) timed to a VO cue sheet, for exercising add_voiceover.py
without a real recording: syllable-rate amplitude pulses, a gliding 110-150 Hz glottal source shaped by moving vowel formants, short
consonant noise bursts, phrase pauses.  Use only to test levels / ducking / timing.

  python3 music/make_test_voice.py --video long|short --out test_voice.wav [--seed 3] [--lines-dir DIR]
"""
import argparse, os, re
import numpy as np
from scipy import signal
from scipy.io import wavfile
import mixlib as M

ap = argparse.ArgumentParser()
ap.add_argument('--video', choices=['long', 'short'], required=True); ap.add_argument('--out', required=True)
ap.add_argument('--seed', type=int, default=3); ap.add_argument('--lines-dir', help='also write one file per cue (01.wav, 02.wav, ...)')
ap.add_argument('--peak', type=float, default=-6.0)
a = ap.parse_args()
rng = np.random.default_rng(a.seed); SR = M.SR
VOWELS = [(730, 1090, 2440), (270, 2290, 3010), (300, 870, 2240), (570, 840, 2410), (530, 1840, 2480), (660, 1720, 2410)]   # a i u o e ɛ-ish


def syllable(f0a, f0b, formants, dur, vel):
    n = int(dur * SR); t = np.arange(n) / SR
    f0 = np.linspace(f0a, f0b, n) * (1 + 0.012 * np.sin(2 * np.pi * 5.2 * t))
    ph = 2 * np.pi * np.cumsum(f0) / SR
    out = np.zeros(n)
    for k in range(1, int(4200 / min(f0a, f0b)) + 1):
        fk = k * f0
        amp = sum(1.0 / (1 + ((fk - fm) / (bw / 2)) ** 2) for fm, bw in zip(formants, (90, 110, 160))) / k ** 0.35
        out += amp * np.sin(k * ph)
    env = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 0.8 * np.clip(t / 0.012, 0, 1)
    burst = rng.standard_normal(int(0.035 * SR)); burst = signal.sosfilt(signal.butter(2, [2500, 7000], 'band', fs=SR, output='sos'), burst) * np.hanning(len(burst))
    out = out / (np.abs(out).max() + 1e-9) * env * vel; out[:len(burst)] += burst * 0.18 * vel
    return out


def speak(text, dur):
    phrases = [p for p in re.split(r'(?<=[.,—…])\s+', text) if p.strip()]
    wc = [max(1, len(p.split())) for p in phrases]; pause = 0.16 if len(phrases) > 1 else 0.0
    avail = dur - pause * (len(phrases) - 1); y = np.zeros(int(dur * SR)); t0 = 0.0
    for p, w in zip(phrases, wc):
        pd = avail * w / sum(wc); syl = max(1, int(round(w * 1.45))); sd = pd / syl
        f0 = rng.uniform(125, 150)
        for s in range(syl):
            seg = syllable(f0 - 6 * s, f0 - 6 * s - 4, VOWELS[rng.integers(0, len(VOWELS))], sd * 0.9, rng.uniform(0.6, 1.0) * (1.0 if s % 3 else 1.15))
            i = int((t0 + s * sd) * SR); y[i:i + len(seg)] += seg[:len(y) - i]
        t0 += pd + pause
    return y


cfg = M.VIDEOS[a.video]; cues = M.read_srt(os.path.join(M.ROOT, cfg['srt']))
mix = np.zeros(int(cfg['dur'] * SR))
if a.lines_dir: os.makedirs(a.lines_dir, exist_ok=True)
for i, (s, e, text) in enumerate(cues):
    d = max(0.5, (e - s) * rng.uniform(0.9, 0.98)); y = speak(text, d)
    j = int(s * SR); mix[j:j + len(y)] += y[:len(mix) - j]
    if a.lines_dir:
        yy = y / (np.abs(y).max() + 1e-9) * 10 ** (a.peak / 20); wavfile.write(os.path.join(a.lines_dir, f'{i + 1:02d}.wav'), SR, (yy * 32767).astype(np.int16))
mix = mix / (np.abs(mix).max() + 1e-9) * 10 ** (a.peak / 20)
wavfile.write(a.out, SR, (mix * 32767).astype(np.int16)); print('wrote', a.out, len(mix) / SR, 's', len(cues), 'cues')
