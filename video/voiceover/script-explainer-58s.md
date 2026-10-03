# Voiceover script — explainer-58s (57.6 s)

Record each line to fit its window (start on the cue, finish by the end time). Pace is the words-per-second you will need — anything above ~3.0
is tight, trim or speed up slightly; ~2.0–2.6 sounds natural.

| # | Window | Budget | Words | Words/s | Say | On screen |
|---|---|---|---|---|---|---|
| 1 |   0.5 –   3.3 |  2.8 s | 7 | 2.5 | Patients walk out… and never come back. | Patients walk out. …and never come back. |
| 2 |   6.1 –   9.0 |  2.9 s | 8 | 2.8 | Missed visits. Reminders by hand just don’t scale. | Missed visits — reminders by hand don't scale past a few dozen patients. |
| 3 |   9.4 –  11.4 |  2.0 s | 7 | 3.5 | Every no-show is a paid-for slot, lost. | Empty slots — every no-show is a paid-for slot lost. |
| 4 |  12.2 –  15.0 |  2.8 s | 7 | 2.5 | And mothers and babies fall off schedule. | Mothers off schedule — a missed ANC visit today is a high-risk delivery tomorrow. |
| 5 |  17.4 –  19.8 |  2.4 s | 6 | 2.5 | HealthBridge brings them back — on WhatsApp. | HealthBridge Solutions — patient follow-up on WhatsApp. |
| 6 |  20.7 –  24.4 |  3.7 s | 10 | 2.7 | First, we connect with your facility and fit your workflow. | Step 1: we connect with your facility. |
| 7 |  25.4 –  29.1 |  3.7 s | 11 | 3.0 | Then we reach your patients on WhatsApp — in English or Kiswahili. | Step 2: we reach your patients on WhatsApp (English / Kiswahili). |
| 8 |  31.8 –  35.4 |  3.6 s | 8 | 2.2 | And every month, you see who came back. | Step 3: you see results in your monthly report. |
| 9 |  36.7 –  40.5 |  3.8 s | 8 | 2.1 | Every mother on schedule. Every baby on time. | Every mother on schedule. Every baby on time. |
| 10 |  44.2 –  50.4 |  6.2 s | 15 | 2.4 | Built in Kenya. Live today. Backed by a local team that picks up the phone. | Why hospital owners choose us: built in Kenya · live and running · you see the numbers · a local team that picks up the phone. |
| 11 |  52.2 –  56.2 |  4.0 s | 9 | 2.2 | HealthBridge Solutions. Patients who come back. Book your demo. | Patients who come back. Book a 20-minute Demo — healthbridgesolutions.net |

**96 words total.**

## Delivery
* Warm, calm, confident — a nurse explaining to a colleague, not an advert. Smile on "Patients who come back." East African English is welcome.
* Say it plainly; no hype, no up-talk. Natural breaths between lines are fine — the music has room.
* Pronunciation: **HealthBridge** (HELTH-brij) · **Kiswahili** (kee-swah-HEE-lee) · **WhatsApp** (WAHTS-app).
* Record dry (no music), 48 kHz / 24-bit WAV if you can, 15–20 cm from the mic with a pop filter, a quiet room. One file per line, or one file for the whole video.

## Dropping it in
* **Any editor:** import `healthbridge-solutions-explainer-58s-vo-ready.mp4` (its music and effects already dip under these windows) and place each line at its start time.
* **Automatic mix (recommended):** `python3 music/add_voiceover.py --video long --lines-dir my_lines/` (files `01.wav`, `02.wav`, … placed at the start times above)
  or `--voice whole_take.wav`. It ducks the music from your actual recording, so slightly different timing is fine.
