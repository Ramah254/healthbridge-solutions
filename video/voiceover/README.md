# Adding the voiceover

Everything here is built from the cue sheets `../vo-script-explainer.srt` (57.6 s) and `../vo-script.srt` (18 s).

| File | What it is |
|---|---|
| `script-explainer-58s.md`, `script-promo-18s.md` | The lines to record: window per line (start → end), word count, words/second, what is on screen, delivery notes |
| `explainer-58s-vo-ready-bed.m4a`, `promo-18s-vo-ready-bed.m4a` | The music + effects bed, audio only, already sitting under a voice |
| `../healthbridge-solutions-explainer-58s-vo-ready.mp4`, `../healthbridge-solutions-promo-18s-vo-ready.mp4` | The finished picture with that bed (no voice yet) — drop your voice on top in any editor |

## 1. Record
Follow `script-*.md`. Dry (no music), quiet room, pop filter, 48 kHz / 24-bit WAV if possible. Either one file per line or one file for the whole video.
Target pace is ~2.0–2.6 words/second; the Pace column shows what each line needs. Warm, calm, plain — no hype.

## 2a. Mix it automatically (recommended)
```bash
# one file per line, named by cue number (01.wav, 02.wav, ... any format ffmpeg reads):
python3 music/add_voiceover.py --video long  --lines-dir my_lines/
# or one file covering the whole video (silences between lines included):
python3 music/add_voiceover.py --video long  --voice my_whole_take.wav
python3 music/add_voiceover.py --video short --voice my_teaser_take.wav
```
Output: `healthbridge-solutions-<cut>-with-voiceover.mp4` (picture untouched, audio re-mixed). The music and effects are ducked **from your actual
recording** (9 dB for the music, 5 dB for effects, plus a speech-pocket dip around 1.5 kHz), with look-ahead and a short hold so they don't pump
between words. The voice is levelled to -16 LUFS and the final mix to -15 LUFS with a peak limiter.
The command prints, per line, how far your voice sits above the music in the speech band (300–3400 Hz). **Aim for ≥ 12 dB**; it warns below 10.
Useful flags: `--offset SEC` (shift everything), `--duck-db`, `--bed-lufs`, `--voice-lufs`, `--target-lufs`, `--report out.json`, `--wav out.wav`,
`--music-key afro` (the fuller score instead of the sparser voiceover variant). `python3 music/add_voiceover.py -h` lists them all.

## 2b. Mix it in an editor
Import the `*-vo-ready.mp4` (or the `.m4a` bed) and place each line at its start time. The bed is deliberately quiet (-27 LUFS) and dips under each
line's window (music -8 dB, effects -4 dB, plus a speech-pocket EQ) so a voice at about -16 LUFS sits ~13–16 dB above it. If your timing differs
from the cue sheet, use 2a instead — it follows your recording rather than the cue sheet.

## Rebuild from a clean checkout
The score and the effects are generated, not stored:
```bash
python3 audio.py && python3 audio_long.py          # effects stems -> audio/sfx.wav, audio/long_sfx.wav
python3 music/afro.py && python3 music/afro.py --vo-lite   # score + voiceover-lite score -> music/out/
python3 music/finalize.py afro                      # the normal (no-voice) mix; needs the picture MP4s
python3 music/voiceover_ready.py                    # the VO-ready beds, MP4s, and these scripts
```

## How it was checked
With a speech-like **test signal** (`music/make_test_voice.py` — syllable-rate pulses through moving vowel formants, timed to the cue sheets; it is
*not* intelligible speech), because no real recording exists yet: voice -16 LUFS, minimum voice-over-bed margin 13.5 dB (explainer) / 13.9 dB (teaser)
in automatic mode and 12.8 dB / 13.2 dB for the pre-dipped bed. A real voice will differ — trust the per-line report the script prints for *your* take.
