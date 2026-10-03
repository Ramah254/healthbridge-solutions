#!/usr/bin/env python3
"""Mix a recorded voiceover over the HealthBridge music + effects and attach it to the finished video.
The music/SFX are ducked FROM THE ACTUAL RECORDING (sidechain-style, computed offline with look-ahead), so any timing works.

  python3 music/add_voiceover.py --video long|short  (--voice take.wav | --lines-dir my_lines/)  [options]

Voice input
  --voice FILE       one file for the whole video (any format ffmpeg reads: wav/mp3/m4a/ogg/...; stereo is mixed to mono). It is placed at
                     t = --offset seconds (default 0), so for a whole-take file the silence between lines must be in the file.
  --lines-dir DIR    one file per cue, named by number (01.wav, 02.m4a, 3.mp3, ...), each placed at that cue's start time from the
                     video's vo-script*.srt (+ --offset). See voiceover/script-*.md for the lines and their windows.
Options
  --bed dynamic|vo-ready   dynamic (default): un-ducked bed + ducking from your recording.  vo-ready: use the pre-dipped bed as-is
                           (what you get when you drop a voice onto the *-vo-ready.mp4 in an editor; no extra ducking)
  --voice-lufs -16   voice loudness (integrated, constant gain)      --bed-lufs -23   bed loudness before ducking
  --duck-db 9 --sfx-duck-db 5 --pocket-db 4   ducking depth for music / effects and the speech-pocket EQ on the music
  --target-lufs -15  loudness of the final mix (constant gain + peak limiter)
  --music-key afro_volite   which score (music/out/<key>_<video>_music.wav)
  --out FILE.mp4  --wav FILE.wav  --report FILE.json
"""
import argparse, json, os, re, sys
import numpy as np
from scipy import signal
import mixlib as M

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--video', choices=['long', 'short'], required=True)
ap.add_argument('--voice'); ap.add_argument('--lines-dir'); ap.add_argument('--offset', type=float, default=0.0)
ap.add_argument('--bed', choices=['dynamic', 'vo-ready'], default='dynamic')
ap.add_argument('--voice-lufs', type=float, default=-16.0); ap.add_argument('--bed-lufs', type=float, default=-23.0)
ap.add_argument('--duck-db', type=float, default=9.0); ap.add_argument('--sfx-duck-db', type=float, default=5.0); ap.add_argument('--pocket-db', type=float, default=4.0)
ap.add_argument('--target-lufs', type=float, default=-15.0); ap.add_argument('--music-key', default='afro_volite')
ap.add_argument('--out'); ap.add_argument('--wav'); ap.add_argument('--report')
a = ap.parse_args()
if bool(a.voice) == bool(a.lines_dir): sys.exit('give exactly one of --voice FILE or --lines-dir DIR')
SR = M.SR; cfg = M.VIDEOS[a.video]; n = int(round(cfg['dur'] * SR)); cues = M.read_srt(os.path.join(M.ROOT, cfg['srt']))
warn = lambda m: print('WARNING:', m, file=sys.stderr)


def place(v, at):
    """put mono array v on a silent timeline of length n starting at `at` seconds (negative = trim the start)"""
    out = np.zeros(n); i = int(round(at * SR))
    if i < 0: v, i = v[-i:], 0
    if i >= n: warn(f'audio starts after the video ends ({at:.2f}s)'); return out
    m = min(len(v), n - i)
    if len(v) > m: warn(f'audio runs {len(v) / SR - m / SR:.2f}s past the end of the video and is trimmed'); v = v[:m].copy(); k = min(len(v), int(0.1 * SR)); v[-k:] *= np.linspace(1, 0, k)
    out[i:i + m] = v[:m]; return out


# ---- 1. the voice timeline
if a.voice:
    v = place(M.decode_audio(a.voice), a.offset)
else:
    files = {}
    for f in sorted(os.listdir(a.lines_dir)):
        m = re.match(r'(?:line[_-]?)?0*(\d+)(?=[._ -]|$)', f, re.I)
        if m and os.path.isfile(os.path.join(a.lines_dir, f)): files.setdefault(int(m.group(1)), os.path.join(a.lines_dir, f))
    if not files: sys.exit(f'no numbered audio files (01.wav, 02.wav, ...) in {a.lines_dir}')
    v = np.zeros(n)
    for idx, path in sorted(files.items()):
        if not 1 <= idx <= len(cues): warn(f'{os.path.basename(path)}: there is no cue #{idx} (1..{len(cues)}) - skipped'); continue
        s, e, _ = cues[idx - 1]; y = M.decode_audio(path); d = len(y) / SR
        if d > (e - s) + 0.8: warn(f'cue {idx}: your line is {d:.1f}s but its window is {e - s:.1f}s')
        if idx < len(cues) and s + a.offset + d > cues[idx][0] + a.offset + 0.3: warn(f'cue {idx} runs into cue {idx + 1}')
        v += place(y, s + a.offset)
    missing = [i for i in range(1, len(cues) + 1) if i not in files]
    if missing: warn(f'no file for cue(s) {missing}')
if np.abs(v).max() >= 0.999: warn('the recording is clipped (peaks at full scale) - re-record at a lower input level if you can')
v = signal.sosfilt(signal.butter(2, 80, 'high', fs=SR, output='sos'), v)                       # rumble / plosive thumps
vl = M.lufs(v)
if not np.isfinite(vl) or vl < -60: sys.exit('the voice track is silent (or almost) - nothing to mix')
v = v * 10 ** (min(30.0, a.voice_lufs - vl) / 20)
k = 0.6; av = np.abs(v); v = np.where(av < k, v, np.sign(v) * (k + (1 - k) * np.tanh((av - k) / (1 - k))))   # soft peak control

# ---- 2. the bed, ducked from the recording
act = M.voice_activity(v)
if a.bed == 'dynamic':
    mus, sfx, _ = M.bed_base(a.music_key, a.video, a.bed_lufs)
    mus, sfx = M.duck_bed(mus, sfx, act, a.duck_db, a.sfx_duck_db, a.pocket_db)
else:
    path = os.path.join(M.ROOT, 'audio', f'vo_ready_{a.video}.wav')
    if not os.path.exists(path): sys.exit(f'{path} missing - run music/voiceover_ready.py first')
    mus, sfx = M.load(path)[:, :n], np.zeros((2, n))
bed = mus + sfx

# ---- 3. sum, set the final loudness with one constant gain, limit peaks, attach to the picture
mix = np.stack([v, v]) + bed
mix_l = M.lufs(mix); gain_db = a.target_lufs - mix_l
wav = a.wav or os.path.join(M.ROOT, 'audio', f'with_voiceover_{a.video}.wav'); M.save(wav, mix)
dst = a.out or os.path.join(M.ROOT, f'healthbridge-solutions-{cfg["tag"]}-with-voiceover.mp4')
M.mux(os.path.join(M.ROOT, cfg['video']), wav, dst, cfg['dur'], gain_db, limit=0.89)

# ---- 4. report: is the voice clearly above the bed, line by line (300-3400 Hz, only while the voice is actually present)?
sos = signal.butter(4, [300, 3400], 'band', fs=SR, output='sos')
vb, bb = signal.sosfilt(sos, v), signal.sosfilt(sos, bed.mean(0))
rows = []
for i, (s, e, text) in enumerate(cues):
    i0, i1 = int((s + a.offset) * SR), min(n, int((e + a.offset + 0.3) * SR))
    if i0 >= n: continue
    m = np.zeros(n, bool); m[i0:i1] = True; m &= act > 0.5
    if m.sum() < SR * 0.2: rows.append(dict(cue=i + 1, text=text, note='no voice detected in this window')); continue
    vdb, bdb = 10 * np.log10((vb[m] ** 2).mean() + 1e-12), 10 * np.log10((bb[m] ** 2).mean() + 1e-12)
    rows.append(dict(cue=i + 1, text=text, voice_dbfs=round(float(vdb), 1), bed_dbfs=round(float(bdb), 1), voice_over_bed_db=round(float(vdb - bdb), 1)))
snr = [r['voice_over_bed_db'] for r in rows if 'voice_over_bed_db' in r]
rep = dict(video=a.video, bed=a.bed, voice_lufs_before_gain=round(vl, 1), voice_lufs=round(M.lufs(v), 1), bed_lufs=round(M.lufs(bed), 1), mix_lufs_before_final_gain=round(mix_l, 1),
           final_gain_db=round(gain_db, 2), min_voice_over_bed_db=min(snr) if snr else None, lines=rows, out=os.path.relpath(dst, M.ROOT))
if a.report: json.dump(rep, open(a.report, 'w'), indent=1)
print(f'voice {rep["voice_lufs"]} LUFS, bed {rep["bed_lufs"]} LUFS, final gain {gain_db:+.1f} dB;  voice over bed (speech band): min {rep["min_voice_over_bed_db"]} dB, median {np.median(snr) if snr else None:.1f} dB')
for r in rows: print(f'  cue {r["cue"]:2d}: ' + (f'{r["voice_over_bed_db"]:+5.1f} dB' if 'voice_over_bed_db' in r else r['note']) + f'  {r["text"][:60]}')
if snr and min(snr) < 10: warn('some lines sit less than 10 dB above the bed - raise --duck-db or lower --bed-lufs')
print('wrote', os.path.relpath(dst, M.ROOT))
