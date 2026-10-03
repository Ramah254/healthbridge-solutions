# HealthBridge Solutions — "Patients Who Come Back" (18 s)

Motion piece for healthbridgesolutions.net. 1920×1080, 30 fps, H.264 + AAC, 18.0 s.

**Final file:** `healthbridge-solutions-promo-18s.mp4` (music + sound effects, no voiceover yet)

| Time | Scene | On-screen |
|---|---|---|
| 0.0–3.0 | The leak | Patients walk out. …and never come back. |
| 3.0–6.0 | The cost | Missed visits. / Empty slots. / Mothers off schedule. |
| 6.0–9.0 | The bridge | HealthBridge Solutions — Patient follow-up on WhatsApp. |
| 9.0–12.5 | How it works | Automatic WhatsApp follow-up. English ⇄ Kiswahili |
| 12.5–15.5 | The impact | Every mother on schedule. Every baby on time. |
| 15.5–18.0 | Call to action | Patients who come back. Book a 20-minute Demo · healthbridgesolutions.net |

## Voiceover
`vo-script.srt` has the line-by-line VO with timecodes (≈38 words, calm pace).
`audio/music-stem.m4a` and `audio/sfx-stem.m4a` are separate stems so a VO track can be mixed over the music.

## Rebuild
```bash
# needs: node 18+, playwright (chromium), ffmpeg, python3 + numpy + scipy
cd video
node render.js still 7.9            # preview a frame -> stills/
node render.js frames 30 4          # render 540 frames -> frames/
python3 audio.py                    # synthesize soundtrack -> audio/*.wav
ffmpeg -framerate 30 -i frames/%04d.png -i audio/mix.wav \
  -af "loudnorm=I=-14:TP=-1.5:LRA=9,volume=-1.0dB" \
  -c:v libx264 -preset slow -crf 15 -tune animation -pix_fmt yuv420p \
  -c:a aac -b:a 192k -movflags +faststart -t 18 out.mp4
```
The whole piece is one seekable GSAP timeline in `src/index.html` (fonts: Plus Jakarta Sans + Inter, brand colours from `styles.css`).

## Content notes
- The "3–5 of every 10" stat is the homepage's own claim — confirm you can substantiate it before paid use.
- The report card is labelled *Illustrative*; no client results are shown (results.html says there are no case studies yet).
- Only the live MCH system is shown; OPD ("coming soon") is deliberately not promoted.
- The Kiswahili chat line should be checked by a native speaker.
