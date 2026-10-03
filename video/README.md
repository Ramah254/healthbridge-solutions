# HealthBridge Solutions — motion videos

Two cuts for healthbridgesolutions.net. Both 1920×1080, 30 fps, H.264 + AAC, with an original score + sound effects (no voiceover yet).

| File | Length | Use |
|---|---|---|
| `healthbridge-solutions-explainer-58s.mp4` | 57.6 s | Full explainer — site walkthrough, paced so every line can be read |
| `healthbridge-solutions-promo-18s.mp4` | 18.0 s | Short teaser / social cut |

## Explainer (57.6 s)
| Time | Scene | What the viewer sees |
|---|---|---|
| 0.0–5.5 | The leak | Patients walk out… and never come back. (3–5 of every 10 lost after visit one) |
| 5.5–15.1 | The cost | Missed visits (reminders by hand don't scale) · Empty slots (every no-show is a paid-for slot lost) · Mothers off schedule (a missed ANC visit is a high-risk delivery) |
| 15.1–20.1 | The bridge | HealthBridge Solutions — patient follow-up on WhatsApp |
| 20.1–36.3 | How it works | Step 1 Connect with your facility · Step 2 Reach patients on WhatsApp (English ⇄ Kiswahili) · Step 3 Monthly owner report |
| 36.3–43.0 | The impact | Every mother on schedule, every baby on time — ANC, KEPI immunisations, PNC |
| 43.0–51.8 | Why owners choose us | Built in Kenya · Live and running · You see the numbers · A local team that picks up the phone |
| 51.8–57.6 | Call to action | Patients who come back. Book a 20-minute Demo · healthbridgesolutions.net · WhatsApp +254 795 822 310 |

Pacing rule used: every text block stays on screen long enough to read (≈ 3–4 words/second), transitions are 0.5–1.0 s, and nothing leaves while its neighbour is still arriving.

## Music
An original, procedurally composed score ("Bridge", `music/afro.py`): warm Afro-acoustic — kalimba/mbira theme, nylon guitar,
round bass, soft congas/shaker — in G major at 102 BPM. A four-bar "bridge" motif (a rising fifth that steps home) is hinted
in a darker form during the problem scenes, stated in full at the reveal (the light wipe + logo), developed through
*how it works*, lifted for the mothers-and-babies scene, and resolves on a G major chord at the button click.
No samples, no licensing: everything is synthesised by the script and is byte-reproducible.

How it was chosen: three composers wrote competing scores (`afro.py`, `piano.py`, `pulse.py`), each audited by an independent
technical and musical verifier, repaired where needed, and ranked by a judge on measurements (`music/analyze.py`: loudness, spectral
balance, intended energy arc, tempo/key, alignment of the reveal and other cues, clicks, voiceover headroom). **Nobody in that pipeline can
listen** — please audition it and tell me if the mood is off. Alternates: `piano.py` (calm piano & strings; its repair was not independently
re-audited), `pulse.py` (modern tech; ranked last, not recommended).

Mix: score + the SFX stems (SFX trimmed 9 dB before the reveal so the score carries the tension), constant-gain to ≈ -15.5 LUFS and a
peak limiter (no dynamic loudnorm, so the quiet-to-loud arc survives). The picture is stream-copied, never re-encoded.

## Voiceover
Ready for a voice: `healthbridge-solutions-explainer-58s-vo-ready.mp4` / `...-promo-18s-vo-ready.mp4` (music + effects already sit under a voice), the
cue-timed recording scripts, and an automatic mixer that ducks the music from your recording. **See `voiceover/README.md`.**

    python3 music/add_voiceover.py --video long --lines-dir my_lines/      # or --voice whole_take.wav

Cue sheets: `vo-script-explainer.srt` (≈100 words) and `vo-script.srt` (18 s cut). Stems for a DAW: `audio/explainer-music-stem.m4a` / `audio/music-stem.m4a` (score),
`audio/*-music-volite-stem.m4a` (sparser score for under a voice), `audio/explainer-sfx-stem.m4a` / `audio/sfx-stem.m4a` (effects).

## Rebuild
```bash
# needs: node 18+, playwright (chromium), ffmpeg, python3 + numpy + scipy
cd video
# explainer picture
SRC=long.html node render.js frames 30 4        # -> frames/ (duration read from the page)
ffmpeg -framerate 30 -i frames/%04d.png -c:v libx264 -preset slow -crf 15 -tune animation -pix_fmt yuv420p -movflags +faststart -t 57.6 picture.mp4
# then add audio with music/finalize.py (see below), or use the SFX-only bed: python3 audio_long.py
# 18 s cut: node render.js frames 30 4  (uses src/index.html; picture only)
# music (both videos):
python3 music/afro.py                 # -> music/out/afro_{long,short}_music.wav   (add --vo-lite for the voiceover-friendly variant)
python3 music/finalize.py afro        # mix with SFX stems + re-attach audio to the finished MP4s (no re-render)
python3 music/voiceover_ready.py      # voiceover-ready beds + MP4s + recording scripts (needs music/out/afro_volite_*: run music/afro.py --vo-lite)
python3 music/analyze.py music/out/afro_long_music.wav --video long --bpm 102.1 --key "G major"   # objective report
# the previous procedural bed: audio.py / audio_long.py (still in the repo, now superseded)
# preview a frame: [SRC=long.html] node render.js still 12.5
```
Each video is one seekable GSAP timeline (`src/long.html`, `src/index.html`); fonts are Plus Jakarta Sans + Inter; brand colours come from `styles.css`.
Make sure all frames exist before encoding (`ls frames | wc -l` = duration × 30).

## Content notes
- The "3–5 of every 10" stat is the homepage's own claim — confirm you can substantiate it before paid use.
- The report card is labelled *Illustrative*; no client results are shown (results.html says there are no case studies yet).
- Only the live MCH system is shown; OPD ("coming soon") is deliberately not promoted.
- The Kiswahili chat line should be checked by a native speaker.
