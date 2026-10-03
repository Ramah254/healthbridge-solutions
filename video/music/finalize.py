#!/usr/bin/env python3
"""Mix a composed music bed with the existing sound-effect stems and attach it to the finished videos.

  python3 music/finalize.py <key> [--music-db 0] [--sfx-db 2] [--tension-sfx-db -9] [--target-lufs -15.5] [--no-mux] [--suffix -name]

For each video (long/short):  music/out/<key>_<video>_music.wav  +  audio/{long_sfx,sfx}.wav  ->  audio/final_<key>_<video>.wav
then (unless --no-mux) the existing H.264 picture is stream-copied and the new audio muxed in:
  long  -> healthbridge-solutions-explainer-58s[<suffix>].mp4        short -> healthbridge-solutions-promo-18s[<suffix>].mp4
The picture is never re-encoded, so this takes seconds. Loudness is set with ONE constant gain (default -15.5 LUFS integrated) followed by a
peak limiter -- not dynamic loudnorm, which would lift the quiet opening and flatten the composed energy arc.
(For the voiceover-ready versions see voiceover_ready.py / add_voiceover.py.)
"""
import argparse, os, sys
import numpy as np
import mixlib as M

ap = argparse.ArgumentParser()
ap.add_argument('key'); ap.add_argument('--music-db', type=float, default=0.0); ap.add_argument('--sfx-db', type=float, default=2.0)
ap.add_argument('--target-lufs', type=float, default=-15.5, help='integrated loudness of the final mix (constant gain; no dynamic loudnorm, so the composed arc survives)')
ap.add_argument('--tension-sfx-db', type=float, default=-9.0, help='SFX trim before the reveal (the score already carries the tension and the cue-time knocks)')
ap.add_argument('--no-mux', action='store_true'); ap.add_argument('--suffix', default='', help="output name suffix, e.g. '-afro' (default: replace the main file)")
a = ap.parse_args()

for name, cfg in M.VIDEOS.items():
    if not os.path.exists(os.path.join(M.HERE, 'out', f'{a.key}_{name}_music.wav')): sys.exit(f'missing music/out/{a.key}_{name}_music.wav (run its composer script first)')
    mus, sfx, n = M.stage(a.key, name, a.music_db, a.sfx_db, a.tension_sfx_db)
    mix = M.soft_ceiling(mus + sfx)
    out = os.path.join(M.ROOT, 'audio', f'final_{a.key}_{name}.wav'); M.save(out, mix)
    mix_lufs = M.lufs_file(out); gain_db = a.target_lufs - mix_lufs
    over = 20 * np.log10(np.abs(mix).max()) + gain_db
    print(f'{name}: mix {mix_lufs:.1f} LUFS -> gain {gain_db:+.1f} dB (peaks {over:+.1f} dBFS before the limiter) -> {os.path.relpath(out, M.ROOT)}')
    if a.no_mux: continue
    src = os.path.join(M.ROOT, cfg['video']); dst = src.replace('.mp4', f'{a.suffix}.mp4')
    M.mux(src, out, dst, cfg['dur'], gain_db, limit=0.80); print('   wrote', os.path.relpath(dst, M.ROOT))
