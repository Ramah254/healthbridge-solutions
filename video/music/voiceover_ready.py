#!/usr/bin/env python3
"""Build the VOICEOVER-READY mix of both videos.

  python3 music/voiceover_ready.py [--key afro_volite] [--bed-lufs -23] [--duck-db 8] [--sfx-duck-db 4] [--pocket-db 4] [--no-mux]

The score (voiceover-lite variant) + the SFX stems are mixed to sit UNDER a voice:
  * a base level of --bed-lufs (so a voice at -16 LUFS has ~10 dB of headroom over the whole bed),
  * a pre-dip while the voice is expected (the cue times in vo-script*.srt): music -duck-db (8), SFX -sfx-duck-db (4), plus a broad
    1.5 kHz "speech pocket" on the music -- 0.15 s early, 0.25 s late, with soft ramps,
  * no limiter-driven loudness games: the bed is deliberately quieter than the normal mix.
Outputs
  healthbridge-solutions-<tag>-vo-ready.mp4   picture (stream-copied) + the VO-ready bed, for dropping a voice on in any editor
  voiceover/<tag>-vo-ready-bed.m4a            the same bed as audio only
  voiceover/script-<tag>.md                   cue-timed recording script (durations, words, pace, on-screen text)
If your voice is timed differently, use add_voiceover.py instead: it ducks the music from the ACTUAL recording.
"""
import argparse, json, os
import numpy as np
import mixlib as M

ap = argparse.ArgumentParser()
ap.add_argument('--key', default='afro_volite', help='music key (music/out/<key>_<video>_music.wav)')
ap.add_argument('--bed-lufs', type=float, default=-23.0); ap.add_argument('--duck-db', type=float, default=8.0)
ap.add_argument('--sfx-duck-db', type=float, default=4.0); ap.add_argument('--pocket-db', type=float, default=4.0)
ap.add_argument('--no-mux', action='store_true')
a = ap.parse_args()

ON_SCREEN = {
    'long': ["Patients walk out. …and never come back.",
             "Missed visits — reminders by hand don't scale past a few dozen patients.",
             "Empty slots — every no-show is a paid-for slot lost.",
             "Mothers off schedule — a missed ANC visit today is a high-risk delivery tomorrow.",
             "HealthBridge Solutions — patient follow-up on WhatsApp.",
             "Step 1: we connect with your facility.",
             "Step 2: we reach your patients on WhatsApp (English / Kiswahili).",
             "Step 3: you see results in your monthly report.",
             "Every mother on schedule. Every baby on time.",
             "Why hospital owners choose us: built in Kenya · live and running · you see the numbers · a local team that picks up the phone.",
             "Patients who come back. Book a 20-minute Demo — healthbridgesolutions.net"],
    'short': ["Patients walk out. …and never come back.", "Missed visits.", "Empty slots.", "Mothers off schedule.",
              "HealthBridge Solutions — patient follow-up on WhatsApp.", "Automatic WhatsApp follow-up (English / Kiswahili).",
              "Every mother on schedule. Every baby on time.", "Patients who come back. Book a 20-minute Demo."],
}


def write_script(name, cues):
    cfg = M.VIDEOS[name]; path = os.path.join(M.ROOT, 'voiceover', f'script-{cfg["tag"]}.md'); os.makedirs(os.path.dirname(path), exist_ok=True)
    rows, words_total = [], 0
    for i, (s, e, t) in enumerate(cues):
        nw = len(t.replace('—', ' ').replace('…', ' ').split()); words_total += nw; d = e - s
        rows.append(f'| {i + 1} | {s:5.1f} – {e:5.1f} | {d:4.1f} s | {nw} | {nw / d:3.1f} | {t} | {ON_SCREEN[name][i]} |')
    md = f"""# Voiceover script — {cfg['tag']} ({cfg['dur']:.1f} s)

Record each line to fit its window (start on the cue, finish by the end time). Pace is the words-per-second you will need — anything above ~3.0
is tight, trim or speed up slightly; ~2.0–2.6 sounds natural.

| # | Window | Budget | Words | Words/s | Say | On screen |
|---|---|---|---|---|---|---|
""" + '\n'.join(rows) + f"""

**{words_total} words total.**

## Delivery
* Warm, calm, confident — a nurse explaining to a colleague, not an advert. Smile on "Patients who come back." East African English is welcome.
* Say it plainly; no hype, no up-talk. Natural breaths between lines are fine — the music has room.
* Pronunciation: **HealthBridge** (HELTH-brij) · **Kiswahili** (kee-swah-HEE-lee) · **WhatsApp** (WAHTS-app).
* Record dry (no music), 48 kHz / 24-bit WAV if you can, 15–20 cm from the mic with a pop filter, a quiet room. One file per line, or one file for the whole video.

## Dropping it in
* **Any editor:** import `healthbridge-solutions-{cfg['tag']}-vo-ready.mp4` (its music and effects already dip under these windows) and place each line at its start time.
* **Automatic mix (recommended):** `python3 music/add_voiceover.py --video {name} --lines-dir my_lines/` (files `01.wav`, `02.wav`, … placed at the start times above)
  or `--voice whole_take.wav`. It ducks the music from your actual recording, so slightly different timing is fine.
"""
    open(path, 'w', encoding='utf-8').write(md); return path


report = {}
for name, cfg in M.VIDEOS.items():
    cues = M.read_srt(os.path.join(M.ROOT, cfg['srt']))
    assert len(cues) == len(ON_SCREEN[name]), (name, len(cues), len(ON_SCREEN[name]))
    mus, sfx, n = M.bed_base(a.key, name, a.bed_lufs)
    curve = M.smooth_curve(n, cues, pre=0.15, post=0.25, ramp_in=0.25, ramp_out=0.5)
    mus, sfx = M.duck_bed(mus, sfx, curve, a.duck_db, a.sfx_duck_db, a.pocket_db)
    bed = mus + sfx
    out = os.path.join(M.ROOT, 'audio', f'vo_ready_{name}.wav'); M.save(out, bed)
    inwin = np.zeros(n, bool)
    for s, e, _ in cues: inwin[int(s * M.SR):int(e * M.SR)] = True
    report[name] = dict(bed_lufs=M.lufs_file(out), peak_dbfs=round(float(20 * np.log10(np.abs(bed).max())), 2),
                        voice_band_300_3400_dbfs_in_vo_windows=round(float(M.band_rms_db(bed[:, inwin])), 1),
                        voice_band_300_3400_dbfs_between_lines=round(float(M.band_rms_db(bed[:, ~inwin])), 1))
    print(name, report[name])
    m4a = os.path.join(M.ROOT, 'voiceover', f'{cfg["tag"]}-vo-ready-bed.m4a'); os.makedirs(os.path.dirname(m4a), exist_ok=True)
    import subprocess; subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', out, '-c:a', 'aac', '-b:a', '256k', m4a], check=True)
    print('  wrote', os.path.relpath(write_script(name, cues), M.ROOT), os.path.relpath(m4a, M.ROOT))
    if not a.no_mux:
        dst = os.path.join(M.ROOT, f'healthbridge-solutions-{cfg["tag"]}-vo-ready.mp4')
        M.mux(os.path.join(M.ROOT, cfg['video']), out, dst, cfg['dur'], 0.0, limit=0.89); print('  wrote', os.path.relpath(dst, M.ROOT))
json.dump(report, open(os.path.join(M.HERE, 'out', 'vo_ready_report.json'), 'w'), indent=1)
