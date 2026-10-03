#!/usr/bin/env python3
"""Objective checks for a music-only candidate against the video cue sheet.
  python3 analyze.py <file.wav> --video long|short [--bpm N] [--key "C major"] [--anchor SEC] [--spec out.png]
Prints a JSON report. Numpy/scipy + ffmpeg only.
"""
import sys, os, json, re, subprocess, argparse
import numpy as np
from scipy.io import wavfile
from scipy import signal

ap = argparse.ArgumentParser()
ap.add_argument('wav'); ap.add_argument('--video', choices=['long', 'short'], required=True)
ap.add_argument('--bpm', type=float); ap.add_argument('--key'); ap.add_argument('--anchor', type=float)
ap.add_argument('--spec')
a = ap.parse_args()
cues = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cues.json')))[a.video]

sr, raw = wavfile.read(a.wav)
x = raw.astype(np.float64)
if raw.dtype == np.int16: x /= 32768.0
elif raw.dtype == np.int32: x /= 2147483648.0
if x.ndim == 1: x = np.stack([x, x], 1)
mono = x.mean(1); N = len(mono)
db = lambda v: float(20 * np.log10(max(v, 1e-9)))
R = {'file': os.path.basename(a.wav), 'sr': int(sr), 'channels': int(x.shape[1]), 'duration_s': round(N / sr, 3),
     'expected_duration_s': cues['duration']}
R['duration_ok'] = abs(N / sr - cues['duration']) < 0.02 and sr == 48000

# --- level / clipping / loudness
R['peak_dbfs'] = round(db(np.abs(x).max()), 2)
R['clipped_samples'] = int((np.abs(x) >= 0.999).sum())
R['nan_or_inf'] = bool(not np.isfinite(x).all())
R['dc_offset'] = round(float(abs(mono.mean())), 6)
p = subprocess.run(['ffmpeg', '-hide_banner', '-nostats', '-i', a.wav, '-af', 'ebur128=peak=true', '-f', 'null', '-'], capture_output=True, text=True)
tail = p.stderr.split('Summary:')[-1]
g = lambda pat: (re.search(pat, tail) or [None, None])[1]
R['lufs_integrated'] = float(g(r'I:\s+(-?[\d.]+) LUFS') or 'nan'); R['lra_lu'] = float(g(r'LRA:\s+([\d.]+) LU') or 'nan')
R['true_peak_dbfs'] = float(g(r'Peak:\s+(-?[\d.]+) dBFS') or 'nan')

# --- sections: RMS vs intended energy arc
secs = []
for s in cues['sections']:
    seg = mono[int(s['start'] * sr):int(s['end'] * sr)]
    secs.append({'id': s['id'], 'rms_dbfs': round(db(np.sqrt((seg ** 2).mean())), 2), 'target_energy': s['energy']})
R['sections'] = secs
rm = np.array([s['rms_dbfs'] for s in secs]); te = np.array([s['target_energy'] for s in secs])
rk = lambda v: np.argsort(np.argsort(v)).astype(float)
R['arc_spearman'] = round(float(np.corrcoef(rk(rm), rk(te))[0, 1]), 3)     # +1 = loudness follows the intended arc
R['dynamic_range_db_between_sections'] = round(float(rm.max() - rm.min()), 2)

# --- spectrum
f, P = signal.welch(mono, sr, nperseg=8192); tot = P.sum() + 1e-20
band = lambda lo, hi: float(P[(f >= lo) & (f < hi)].sum() / tot)
R['band_share'] = {'sub_20_150': round(band(20, 150), 3), 'low_150_800': round(band(150, 800), 3), 'mid_800_4k': round(band(800, 4000), 3), 'high_4k_16k': round(band(4000, 16000), 3)}
R['voice_band_share_300_3400'] = round(band(300, 3400), 3)                  # competes with a voiceover
R['spectral_centroid_hz'] = round(float((f * P).sum() / tot), 1)
R['harshness_share_2k_5k'] = round(band(2000, 5000), 4)

# --- stereo
mid, side = (x[:, 0] + x[:, 1]) / 2, (x[:, 0] - x[:, 1]) / 2
R['stereo_lr_corr'] = round(float(np.corrcoef(x[:, 0], x[:, 1])[0, 1]), 3) if x[:, 0].std() > 0 and x[:, 1].std() > 0 else 1.0
R['side_to_mid_db'] = round(db(np.sqrt((side ** 2).mean())) - db(np.sqrt((mid ** 2).mean())), 2)

# --- discontinuities (clicks)
d = np.abs(np.diff(mono)); thr = max(0.25, 8 * np.percentile(d, 99.9))
bad = np.where(d > thr)[0]
R['click_events'] = int(len(bad)); R['click_times_s'] = [round(float(i / sr), 3) for i in bad[:10]]

# --- ending
tail1 = mono[-int(sr * 1.0):]; before = mono[-int(sr * 4.0):-int(sr * 2.0)]
R['final_sample_abs'] = round(float(abs(mono[-1])), 6)
R['end_fade_db'] = round(db(np.sqrt((tail1 ** 2).mean())) - db(np.sqrt((before ** 2).mean())), 2)

# --- onset envelope, tempo, accents
nfft, hop = 2048, 512
f2, t2, Z = signal.stft(mono, sr, nperseg=nfft, noverlap=nfft - hop)
Mg = np.log1p(np.abs(Z) * 30)
flux = np.maximum(0, np.diff(Mg, axis=1)).sum(0); flux = np.concatenate([[0], flux]); flux /= (flux.max() + 1e-9)
fr = sr / hop
ac = signal.correlate(flux - flux.mean(), flux - flux.mean(), 'full')[len(flux) - 1:]; ac /= ac[0] + 1e-9
lags = np.arange(len(ac)); bpm_of = lambda l: 60.0 * fr / np.maximum(l, 1)
mask = (bpm_of(lags) >= 60) & (bpm_of(lags) <= 190)
cand = np.where(mask)[0]; order = cand[np.argsort(ac[cand])[::-1]]
top = []
for l in order:
    b = round(float(bpm_of(l)), 1)
    if all(abs(b - t[0]) > 4 for t in top): top.append((b, round(float(ac[l]), 3)))
    if len(top) == 3: break
R['tempo_candidates_bpm'] = top
if a.bpm:
    ests = [t[0] for t in top]
    R['tempo_matches_declared'] = any(abs(e - a.bpm) / a.bpm < .04 or abs(e - a.bpm * 2) / (a.bpm * 2) < .04 or abs(e * 2 - a.bpm) / a.bpm < .04 for e in ests)
    anchor = a.anchor if a.anchor is not None else cues['impact']
    beat = 60.0 / a.bpm
    pk, _ = signal.find_peaks(flux, height=np.percentile(flux, 85), distance=int(.06 * fr))
    times = t2[pk]
    off = np.abs(((times - anchor) / (beat / 2) + .5) % 1 - .5) * (beat / 2)   # distance to nearest 8th-note grid line
    R['onsets_on_grid_pct'] = round(float((off < .035).mean() * 100), 1) if len(off) else 0.0
tidx = lambda t: int(round(t * fr))
med = np.median(flux) + 1e-6
acc = []
for c in cues['accents']:
    i = tidx(c['t']); w = flux[max(0, i - 5):i + 6]
    acc.append({'t': c['t'], 'major': c['major'], 'label': c['label'], 'onset_ratio': round(float(w.max() / med), 1) if len(w) else 0})
R['accent_hits'] = acc
mj = [r for r in acc if r['major']]
R['major_accents_with_onset_pct'] = round(100 * sum(r['onset_ratio'] >= 4 for r in mj) / max(1, len(mj)), 1)
# resolution at impact: loudness/brightness 1 s after vs 1 s before
im = cues['impact']
def seg_rms(t0, t1): s = mono[int(t0 * sr):int(t1 * sr)]; return db(np.sqrt((s ** 2).mean()))
R['impact_release_db'] = round(seg_rms(im, im + 1.5) - seg_rms(im - 1.5, im - .1), 2)

# --- key estimate (Krumhansl-Schmuckler on chroma)
sel = (f2 >= 80) & (f2 <= 2000); pc = (np.round(12 * np.log2(f2[sel] / 440.0)) + 9).astype(int) % 12
pw = (np.abs(Z[sel]) ** 2).sum(1); chroma = np.zeros(12)
for k in range(12): chroma[k] = pw[pc == k].sum()
maj = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]); mnr = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
sc = []
for k in range(12):
    sc.append((float(np.corrcoef(chroma, np.roll(maj, k))[0, 1]), names[k] + ' major'))
    sc.append((float(np.corrcoef(chroma, np.roll(mnr, k))[0, 1]), names[k] + ' minor'))
sc.sort(reverse=True); R['key_estimates'] = [{'key': k, 'corr': round(c, 3)} for c, k in sc[:3]]
if a.key: R['key_matches_declared'] = any(k['key'].lower() == a.key.lower() for k in R['key_estimates'][:2])
# whole-file chroma only tells the home key; also report the chroma of the final 3 s (should sit on the tonic chord)
fin = (t2 >= (N / sr - 3.5)) & (t2 <= (N / sr - .5)); pwf = (np.abs(Z[sel][:, fin]) ** 2).sum(1); cf = np.zeros(12)
for k in range(12): cf[k] = pwf[pc == k].sum()
R['final_chord_top_pitch_classes'] = [names[i] for i in np.argsort(cf)[::-1][:3]]

if a.spec:
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', a.wav, '-lavfi', 'showspectrumpic=s=1800x420:legend=1:scale=log:fscale=log', a.spec])
    R['spectrogram_png'] = a.spec
print(json.dumps(R, indent=1))
