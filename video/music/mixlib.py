"""Shared mixing helpers for finalize.py (music + SFX), voiceover_ready.py and add_voiceover.py."""
import os, re, subprocess, tempfile
import numpy as np
from scipy import signal
from scipy.io import wavfile

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
SR = 48000

# one place that knows each video: file names, length, where the reveal starts (SFX are trimmed before it), the VO cue sheet
VIDEOS = {
    'long':  dict(sfx='audio/long_sfx.wav', video='healthbridge-solutions-explainer-58s.mp4', dur=57.6, reveal=14.3, srt='vo-script-explainer.srt', tag='explainer-58s'),
    'short': dict(sfx='audio/sfx.wav',      video='healthbridge-solutions-promo-18s.mp4',      dur=18.0, reveal=5.4,  srt='vo-script.srt',           tag='promo-18s'),
}


def load(path):
    """-> float64 array (2, N); sample rate must be 48 kHz"""
    sr, x = wavfile.read(path); x = x.astype(np.float64)
    if np.abs(x).max() > 2: x /= 32768.0
    if x.ndim == 1: x = np.stack([x, x], 1)
    assert sr == SR, (path, sr)
    return x.T


def save(path, x):
    """write (2, N) float array as 16-bit stereo WAV"""
    wavfile.write(path, SR, (np.clip(x, -1, 1).T * 32767).astype(np.int16))


def lufs_file(path):
    p = subprocess.run(['ffmpeg', '-hide_banner', '-nostats', '-i', path, '-af', 'ebur128=peak=true', '-f', 'null', '-'], capture_output=True, text=True)
    m = re.search(r'I:\s+(-?[\d.]+) LUFS', p.stderr.split('Summary:')[-1]); return float(m.group(1)) if m else float('nan')


def lufs(x):
    """integrated loudness of a (2, N) / (N,) float array"""
    if x.ndim == 1: x = np.stack([x, x])
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'x.wav'); save(p, x); return lufs_file(p)


def stage(key, name, music_db=0.0, sfx_db=2.0, tension_sfx_db=-9.0):
    """music bed (afro score) and SFX bed, staged exactly as finalize.py always has.
    -> (mus, sfx, n)  with  mix = mus + sfx   (both (2, n))"""
    cfg = VIDEOS[name]
    mus = load(os.path.join(HERE, 'out', f'{key}_{name}_music.wav')); sfx = load(os.path.join(ROOT, cfg['sfx']))
    n = int(round(cfg['dur'] * SR)); mus, sfx = mus[:, :n], sfx[:, :n]
    mus = mus / (np.abs(mus).max() + 1e-9) * 0.6; sfx = sfx / (np.abs(sfx).max() + 1e-9) * 0.5
    tt = np.arange(n) / float(SR)
    env = 10 ** (tension_sfx_db / 20) + (1 - 10 ** (tension_sfx_db / 20)) * np.clip((tt - cfg['reveal']) / 0.5, 0, 1)
    return mus * 0.85 * 10 ** (music_db / 20), sfx * env * 0.8 * 10 ** (sfx_db / 20), n


def soft_ceiling(mix):
    mix = np.tanh(mix * 1.05) / np.tanh(1.05)
    return mix / (np.abs(mix).max() + 1e-9) * 0.89


def read_srt(path):
    """-> [(start_s, end_s, text)]"""
    txt = open(path, encoding='utf-8').read().replace('\r', '')
    out = []
    for blk in re.split(r'\n\s*\n', txt.strip()):
        ln = [l for l in blk.split('\n') if l.strip()]
        m = re.match(r'(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)', ln[1]) if len(ln) >= 3 else None
        if not m: continue
        g = [int(v) for v in m.groups()]
        out.append((g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000.0, g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000.0, ' '.join(ln[2:])))
    return out


def peaking(x, f0, q, gain_db, sr=SR):
    """RBJ peaking EQ on (2, N)"""
    A = 10 ** (gain_db / 40); w = 2 * np.pi * f0 / sr; al = np.sin(w) / (2 * q); c = np.cos(w)
    b = np.array([1 + al * A, -2 * c, 1 - al * A]); a = np.array([1 + al / A, -2 * c, 1 - al / A])
    return signal.lfilter(b / a[0], a / a[0], x, axis=-1)


def smooth_curve(n, windows, pre, post, ramp_in, ramp_out):
    """0..1 curve that is 1 inside every (start-pre, end+post) window with raised-cosine ramps (ramp_in before, ramp_out after)"""
    t = np.arange(n) / float(SR); c = np.zeros(n)
    for a, b, *_ in windows:
        lo, hi = a - pre, b + post
        up = np.clip((t - (lo - ramp_in)) / max(ramp_in, 1e-6), 0, 1); dn = np.clip(((hi + ramp_out) - t) / max(ramp_out, 1e-6), 0, 1)
        c = np.maximum(c, 0.5 - 0.5 * np.cos(np.pi * np.minimum(up, dn)))
    return c


def mux(video_in, wav, dst, dur, gain_db=0.0, limit=0.89):
    """stream-copy the picture, encode `wav` (optionally with a constant gain and a peak limiter) as AAC"""
    tmp = dst + '.tmp.mp4'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', video_in, '-i', wav, '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy',
                    '-af', f'volume={gain_db:.2f}dB,alimiter=limit={limit}:attack=3:release=40:level=disabled', '-c:a', 'aac', '-b:a', '192k', '-ar', '48000',
                    '-movflags', '+faststart', '-t', str(dur), tmp], check=True)
    os.replace(tmp, dst)


# ----------------------------------------------------------------------------------------------------------------------
#  voiceover helpers
# ----------------------------------------------------------------------------------------------------------------------
from scipy.ndimage import maximum_filter1d, uniform_filter1d

CONTROL_HZ = 200          # ducking / VAD control rate


def bed_base(key, name, bed_lufs=-23.0, sfx_db=0.0, tension_sfx_db=-9.0):
    """un-ducked music + SFX beds scaled so that (music + SFX) measures `bed_lufs` LUFS -> (mus, sfx, n)"""
    mus, sfx, n = stage(key, name, 0.0, sfx_db, tension_sfx_db)
    g = 10 ** ((bed_lufs - lufs(mus + sfx)) / 20)
    return mus * g, sfx * g, n


def decode_audio(path, mono=True):
    """any audio file ffmpeg can read -> float32 (N,) at 48 kHz (mono mixdown)"""
    p = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-vn', '-ar', str(SR), '-ac', '1', '-f', 'f32le', '-'], capture_output=True)
    if p.returncode != 0 or not p.stdout: raise SystemExit(f'cannot decode {path}: {p.stderr.decode(errors="ignore")[:300]}')
    return np.frombuffer(p.stdout, dtype='<f4').astype(np.float64)


def voice_activity(v, hold=0.20, attack=0.04, release=0.35, lookahead=0.06, floor_db=-55.0):
    """0..1 'is someone speaking' control signal at the sample rate, for sidechain-style ducking.
    Threshold adapts to the recording (noise floor + 12 dB, but never above peak - 30 dB); words are bridged by `hold`;
    the gain is smoothed with separate attack/release and anticipates the voice by `lookahead` (we render offline)."""
    n = len(v); hop = SR // CONTROL_HZ; win = int(0.02 * SR)
    pw = uniform_filter1d(v * v, win, mode='constant')[::hop]
    L = 10 * np.log10(pw + 1e-12)
    noise, peak = np.percentile(L, 10), np.percentile(L, 98)
    thr = max(floor_db, min(noise + 12, peak - 30))
    a = np.clip((L - thr) / 6.0, 0, 1)
    a = maximum_filter1d(a, size=max(1, int(hold * CONTROL_HZ)))
    sh = int(lookahead * CONTROL_HZ); a = np.concatenate([a[sh:], np.zeros(sh)])
    g = np.zeros_like(a); dt = 1.0 / CONTROL_HZ; ka, kr = 1 - np.exp(-dt / attack), 1 - np.exp(-dt / release); cur = 0.0
    for i, t in enumerate(a):
        cur += (t - cur) * (ka if t > cur else kr); g[i] = cur
    return np.interp(np.arange(n) / float(SR), np.arange(len(g)) / float(CONTROL_HZ), g)


def duck_bed(mus, sfx, act, duck_db=7.0, sfx_duck_db=4.0, pocket_db=3.0):
    """apply ducking + a speech 'pocket' (broad dip around 1.5 kHz on the music) in proportion to the 0..1 control signal `act`"""
    m = mus * (1 - act) + peaking(mus, 1500, 0.7, -pocket_db) * act if pocket_db else mus
    return m * 10 ** (-duck_db * act / 20), sfx * 10 ** (-sfx_duck_db * act / 20)


def band_rms_db(x, lo=300, hi=3400):
    sos = signal.butter(4, [lo, hi], 'band', fs=SR, output='sos')
    y = signal.sosfilt(sos, x, axis=-1); return 10 * np.log10((y ** 2).mean() + 1e-12)
