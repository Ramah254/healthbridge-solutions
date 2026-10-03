#!/usr/bin/env python3
"""Mix a composed music bed with the existing sound-effect stems and attach it to the finished videos.

  python3 music/finalize.py <key> [--music-db 0] [--sfx-db 2] [--tension-sfx-db -9] [--target-lufs -15.5] [--no-mux]

For each video (long/short):  music/out/<key>_<video>_music.wav  +  audio/{long_sfx,sfx}.wav  ->  audio/final_<key>_<video>.wav
then (unless --no-mux) the existing H.264 picture is stream-copied and the new audio muxed in:
  long  -> healthbridge-solutions-explainer-58s[ -<key>].mp4        short -> healthbridge-solutions-promo-18s[ -<key>].mp4
The picture is never re-encoded, so this takes seconds. Loudness is set with ONE constant gain (default -15.5 LUFS integrated) followed by a peak limiter — not dynamic loudnorm,
which would lift the quiet opening and flatten the composed energy arc.
"""
import argparse, os, subprocess, sys
import numpy as np
from scipy import signal
from scipy.io import wavfile

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
ap = argparse.ArgumentParser()
ap.add_argument('key'); ap.add_argument('--music-db', type=float, default=0.0); ap.add_argument('--sfx-db', type=float, default=2.0)
ap.add_argument('--target-lufs', type=float, default=-15.5, help='integrated loudness of the final mix (constant gain; no dynamic loudnorm, so the composed arc survives)')
ap.add_argument('--tension-sfx-db', type=float, default=-9.0, help='SFX trim before the reveal (the score already carries the tension and the cue-time knocks)')
ap.add_argument('--no-mux', action='store_true'); ap.add_argument('--suffix', default='', help="output name suffix, e.g. '-afro' (default: replace the main file)")
a = ap.parse_args()

VIDEOS = {
    'long':  dict(sfx='audio/long_sfx.wav', video='healthbridge-solutions-explainer-58s.mp4', dur=57.6, reveal=14.3),
    'short': dict(sfx='audio/sfx.wav',      video='healthbridge-solutions-promo-18s.mp4',      dur=18.0, reveal=5.4),
}

def load(path):
    sr, x = wavfile.read(path); x = x.astype(np.float64)
    if x.dtype != np.float64 or np.abs(x).max() > 2: x /= 32768.0
    if x.ndim == 1: x = np.stack([x, x], 1)
    assert sr == 48000, (path, sr)
    return x.T                                    # (2, N)

def lufs(path):
    import re
    p = subprocess.run(['ffmpeg', '-hide_banner', '-nostats', '-i', path, '-af', 'ebur128=peak=true', '-f', 'null', '-'], capture_output=True, text=True)
    m = re.search(r'I:\s+(-?[\d.]+) LUFS', p.stderr.split('Summary:')[-1]); return float(m.group(1)) if m else float('nan')

for name, cfg in VIDEOS.items():
    mpath = os.path.join(HERE, 'out', f'{a.key}_{name}_music.wav')
    if not os.path.exists(mpath): sys.exit(f'missing {mpath}')
    mus, sfx = load(mpath), load(os.path.join(ROOT, cfg['sfx']))
    n = int(round(cfg['dur'] * 48000)); mus, sfx = mus[:, :n], sfx[:, :n]
    mus = mus / (np.abs(mus).max() + 1e-9) * 0.6; sfx = sfx / (np.abs(sfx).max() + 1e-9) * 0.5      # same staging as the original mix
    tt = np.arange(n) / 48000.0                                      # SFX trimmed before the reveal, ramping to full over 0.5 s
    env = 10 ** (a.tension_sfx_db / 20) + (1 - 10 ** (a.tension_sfx_db / 20)) * np.clip((tt - cfg['reveal']) / 0.5, 0, 1)
    mix = mus * 0.85 * 10 ** (a.music_db / 20) + sfx * env * 0.8 * 10 ** (a.sfx_db / 20)
    mix = np.tanh(mix * 1.05) / np.tanh(1.05)      # soft ceiling, no hard clipping
    mix = mix / (np.abs(mix).max() + 1e-9) * 0.89
    out = os.path.join(ROOT, 'audio', f'final_{a.key}_{name}.wav')
    wavfile.write(out, 48000, (mix.T * 32767).astype(np.int16))
    mix_lufs = lufs(out); gain_db = a.target_lufs - mix_lufs
    over = 20 * np.log10(np.abs(mix).max()) + gain_db
    print(f'{name}: music {lufs(mpath):.1f} LUFS, sfx {lufs(os.path.join(ROOT, cfg["sfx"])):.1f} LUFS, mix {mix_lufs:.1f} LUFS -> gain {gain_db:+.1f} dB (peaks {over:+.1f} dBFS before the limiter) -> {os.path.relpath(out, ROOT)}')
    if a.no_mux: continue
    src = os.path.join(ROOT, cfg['video']); dst = src.replace('.mp4', f'{a.suffix}.mp4')
    tmp = dst + '.tmp.mp4'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', src, '-i', out, '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy',
                    '-af', f'volume={gain_db:.2f}dB,alimiter=limit=0.80:attack=3:release=40:level=disabled', '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-movflags', '+faststart',
                    '-t', str(cfg['dur']), tmp], check=True)
    os.replace(tmp, dst); print('   wrote', os.path.relpath(dst, ROOT))
