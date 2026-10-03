# HealthBridge Solutions — motion videos

Two cuts for healthbridgesolutions.net. Both 1920×1080, 30 fps, H.264 + AAC, music + sound effects (no voiceover yet).

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

## Voiceover
`vo-script-explainer.srt` (≈100 words, timed to the explainer) and `vo-script.srt` (18 s cut).
`audio/explainer-music-stem.m4a` / `explainer-sfx-stem.m4a` (and `music-stem` / `sfx-stem` for the 18 s cut) let a recorded voice be mixed over the music.

## Rebuild
```bash
# needs: node 18+, playwright (chromium), ffmpeg, python3 + numpy + scipy
cd video
# explainer
SRC=long.html node render.js frames 30 4        # -> frames/ (duration read from the page)
python3 audio_long.py                           # -> audio/long_*.wav (scene times read from src/long.html)
ffmpeg -framerate 30 -i frames/%04d.png -i audio/long_mix.wav \
  -af "loudnorm=I=-14:TP=-1.5:LRA=9,volume=-1.0dB" \
  -c:v libx264 -preset slow -crf 15 -tune animation -pix_fmt yuv420p \
  -c:a aac -b:a 192k -movflags +faststart -t 57.6 out.mp4
# 18 s cut: node render.js frames 30 4 && python3 audio.py  (uses src/index.html)
# preview a frame: [SRC=long.html] node render.js still 12.5
```
Each video is one seekable GSAP timeline (`src/long.html`, `src/index.html`); fonts are Plus Jakarta Sans + Inter; brand colours come from `styles.css`.
Make sure all frames exist before encoding (`ls frames | wc -l` = duration × 30).

## Content notes
- The "3–5 of every 10" stat is the homepage's own claim — confirm you can substantiate it before paid use.
- The report card is labelled *Illustrative*; no client results are shown (results.html says there are no case studies yet).
- Only the live MCH system is shown; OPD ("coming soon") is deliberately not promoted.
- The Kiswahili chat line should be checked by a native speaker.
