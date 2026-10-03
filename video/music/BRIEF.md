# Background-music brief (shared by every composer / verifier)

## Product & story
HealthBridge Solutions — a nurse-founded Kenyan company that follows up with patients over WhatsApp so
private-hospital patients (especially mothers and babies: ANC, PNC, immunisations) keep coming back.
Two videos need ONE relevant background score each (same piece, two edits):
* `long`  — 57.6 s explainer   * `short` — 18.0 s teaser
Story arc (see `cues.json` for exact times): quiet unease about patients lost → tension as missed visits
pile up → a light "reveal" (IMPACT) where the HealthBridge bridge appears → calm, friendly "how it works"
→ emotional peak (every mother on schedule, every baby on time) → confident "why us" → call-to-action
lift and a clean final resolution that rings out and fades.
Feeling to land: warm, hopeful, human, trustworthy, modern. Healthcare + Kenya + technology. Not sad-cinematic
for its own sake, not cold corporate, not stereotyped "tribal" music.

## Hard constraints
* Procedural synthesis only: Python 3.11, numpy 2.x, scipy. No samples, no downloads, no network, no new pip
  installs, no copyrighted melodies (everything original). `ffmpeg` exists for analysis only.
* Compose SYMBOLICALLY (note lists / chord progressions / rhythm patterns) and render through instrument
  functions, so key, harmony and timing are correct by construction.
* Deterministic: seeded RNG only; running the script twice must give byte-identical WAVs.
* Script `music/<key>.py` runs with no arguments, writes BOTH outputs in < 90 s total, accepts optional
  `--out-dir DIR` (default `music/out`). It must only write inside the out dir.
  Outputs (48 kHz, stereo, 16-bit PCM WAV, MUSIC ONLY — no sound effects, no voice):
  `<out>/<key>_long_music.wav` (exactly 57.6 s) and `<out>/<key>_short_music.wav` (exactly 18.0 s).
  The short is NOT a truncation: re-arrange to the short cue sheet (impact at 6.0 s, end 18.0 s) but keep the
  same key, motif and sound world.
* Do not modify any existing file. Read-only: `audiolib.py` (primitives: SR, T, add, nf, mid2f, env_ad, pop,
  thump, noise_hit, bell, pluck, whoosh, riser, pad, ssum, rng), `audio.py`, `audio_long.py`, `src/*`.
  Write only: `music/<key>.py` and `music/out/<key>_*` (wavs, json, png). You may import audiolib or write your
  own instruments (better ones are expected — `pluck`/`pad` there are basic).

## What "good" means here
1. A real, singable MELODY: a 4–8 bar motif (the "bridge theme") that is HINTED in fragments/in a darker
   transformation during tension, STATED fully at IMPACT, developed in the how-it-works section, lifted
   (octave / harmony / countermelody) in the impact scene, steady in "why us", and given a final statement
   on the tonic in the CTA.
2. Harmony that does the dramatic work: unresolved / suspended / minor colour before IMPACT, a decisive
   resolution to a warm major tonic at IMPACT, and a final tonic chord at the end.
3. Arrangement that evolves (layers enter/leave per section), humanised timing (±5–12 ms seeded jitter),
   velocity variation, stereo placement, gentle synthetic-IR reverb (fftconvolve with decaying stereo
   noise, 15–25 % wet), tone shaping (no harsh content above ~6 kHz), no clicks (every note has attack and
   release; fade/crossfade at edits).
4. Bar grid aligned to the video: a strong downbeat / chord change exactly at IMPACT (±30 ms), the section
   starts in `cues.json` on or near bar lines, final chord lands near the CTA click, natural ring-out then fade.
5. Leaves room for a voiceover and sound effects that will be mixed on top later: keep the 300–3400 Hz band
   moderate and sparse (melody in phrases with rests, not wall-to-wall), no sustained loud mid-range pads.

## Objective pass criteria (measured by `python3 music/analyze.py <wav> --video long|short --bpm N --key "X major"`)
* duration_ok true; nan_or_inf false; clipped_samples 0; peak_dbfs <= -1.0
* lufs_integrated between -21 and -15; click_events 0
* arc_spearman >= 0.80 (section loudness follows the intended arc in cues.json)
* section RMS: C >= B + 4 dB; E >= D + 1.5 dB; A <= B - 2 dB; G >= F (before the fade)
* band_share.sub_20_150 <= 0.30 and (band_share.mid_800_4k + band_share.high_4k_16k) >= 0.12 (not boomy/dull)
* voice_band_share_300_3400 between 0.15 and 0.45; harshness_share_2k_5k <= 0.12
* stereo: side_to_mid_db between -22 and -6 and stereo_lr_corr >= 0.5 (some width, still mono-safe)
* tempo_matches_declared true; onsets_on_grid_pct >= 75; key_matches_declared true
* major_accents_with_onset_pct >= 70; impact_release_db >= 3
* end_fade_db <= -8 and final_sample_abs <= 0.001
Baseline of the OLD bed (what we are replacing; do much better): long arc_spearman 0.68, sub+low 95 % of energy,
mid+high 4 %, centroid 363 Hz, side_to_mid -32 dB (almost mono), no melody, sections C–G all equal loudness.
(See music/out/baseline_long.json and baseline_short.json.)

## Honest limits
Nobody in this pipeline can LISTEN. Do not claim subjective beauty; justify choices by construction
(theory, structure) and by the measurements above. View spectrograms (`--spec file.png`, then Read the PNG).

## Tools
* `python3 music/analyze.py FILE --video long|short [--bpm N] [--key "G major"] [--anchor SEC] [--spec out.png]`
  prints a JSON report (anchor defaults to the IMPACT time).
* `ffmpeg` for loudness/spectrograms.
