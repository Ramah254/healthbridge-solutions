# Voiceover script — promo-18s (18.0 s)

Record each line to fit its window (start on the cue, finish by the end time). Pace is the words-per-second you will need — anything above ~3.0
is tight, trim or speed up slightly; ~2.0–2.6 sounds natural.

| # | Window | Budget | Words | Words/s | Say | On screen |
|---|---|---|---|---|---|---|
| 1 |   0.3 –   2.8 |  2.5 s | 7 | 2.8 | Patients walk out… and never come back. | Patients walk out. …and never come back. |
| 2 |   3.1 –   4.0 |  0.9 s | 2 | 2.4 | Missed visits. | Missed visits. |
| 3 |   4.0 –   4.9 |  0.9 s | 2 | 2.2 | Empty slots. | Empty slots. |
| 4 |   5.0 –   5.9 |  0.9 s | 3 | 3.3 | Mothers off schedule. | Mothers off schedule. |
| 5 |   6.3 –   8.6 |  2.3 s | 4 | 1.7 | HealthBridge brings them back. | HealthBridge Solutions — patient follow-up on WhatsApp. |
| 6 |   9.2 –  12.2 |  3.0 s | 7 | 2.3 | Automatic WhatsApp follow-up — in English or Kiswahili. | Automatic WhatsApp follow-up (English / Kiswahili). |
| 7 |  12.7 –  15.1 |  2.4 s | 8 | 3.3 | Every mother on schedule. Every baby on time. | Every mother on schedule. Every baby on time. |
| 8 |  15.7 –  17.7 |  2.0 s | 7 | 3.5 | Patients who come back. Book your demo. | Patients who come back. Book a 20-minute Demo. |

**40 words total.**

## Delivery
* Warm, calm, confident — a nurse explaining to a colleague, not an advert. Smile on "Patients who come back." East African English is welcome.
* Say it plainly; no hype, no up-talk. Natural breaths between lines are fine — the music has room.
* Pronunciation: **HealthBridge** (HELTH-brij) · **Kiswahili** (kee-swah-HEE-lee) · **WhatsApp** (WAHTS-app).
* Record dry (no music), 48 kHz / 24-bit WAV if you can, 15–20 cm from the mic with a pop filter, a quiet room. One file per line, or one file for the whole video.

## Dropping it in
* **Any editor:** import `healthbridge-solutions-promo-18s-vo-ready.mp4` (its music and effects already dip under these windows) and place each line at its start time.
* **Automatic mix (recommended):** `python3 music/add_voiceover.py --video short --lines-dir my_lines/` (files `01.wav`, `02.wav`, … placed at the start times above)
  or `--voice whole_take.wav`. It ducks the music from your actual recording, so slightly different timing is fine.
