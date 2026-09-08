# Sessions

Every session — from `cyclops` or `cyclops-kiosk` — gets **one folder of its own**,
holding everything it produced. Pull the SD card into a laptop and it reads like a GoPro's:

```
sessions/
  2026-08-26_14-32-05_lego-falcon/
    session.md        the conversation, with the photos in it, for you
    session.jsonl     the same events, one JSON object per line, for a program
    summary.md        one sentence and one paragraph: what this session was
    project.md        which project this got filed under, or why it didn't
    video.mp4         the recording - the panel, or the camera (kiosk only)
    clips/
      plan.json       what was decided about this session, and when
      1.mp4  1.ass    the moments worth watching, best first - see "Clips" below
    photos/
      14-33-12_you.jpg
      14-33-05_edit.jpg      one redrawn with a change - see tools.md
      14-35-01_drawn.jpg     a diagram it drew - see tools.md
```

`session.md` opens in any markdown viewer with the images inline and every line stamped as an
offset from the start, so you can scrub straight to it in `video.mp4`. `session.jsonl` is the
same events for a machine: who said what and when, which photos were taken and by whom, what was
searched, and which turns you interrupted. The session's UUID lives inside both files — never in
a name, because nobody can read a UUID.

## The name and the summary

When a session ends, `gpt-5.4-nano` reads the transcript once and answers three things at a
stroke: two to four words to rename the folder with (`2026-08-26_14-32-05` becomes
`2026-08-26_14-32-05_lego-falcon`), one sentence saying what the session was, and one paragraph
of what happened — what was decided, what was measured, what was left open. The sentence and the
paragraph become `summary.md`. One call for all three, so the name and the summary can never
disagree about the same conversation. The three answers are read out of the reply independently,
so a mangled one costs you one of them rather than all three.

With no network, no key, or `CYCLOPS_SLUG=0`, the folder keeps its date-time name and gets no
summary.

**None of it happens while you wait.** The tap that ends a session ends it: teardown writes
`session.md` from records it already has and hands the folder to a detached `python -m
cyclops.after`, which names it, notes anything it said about you, and files it (see
[memory.md](memory.md)). The kiosk is asleep before the first round trip comes back. Each job
sweeps the whole card rather than visiting one folder, so a child that never started, or ran with
the wifi down, costs nothing but time: the next session to end tries again, and so does every
boot.

## Continuity

Every new session starts knowing where the last one got to. Cyclops's instructions gain a short
*Where you left off* section built from the cards already written: **the previous session's
paragraph in full, plus the one-sentence titles of the last five sessions**, oldest first. It is
told not to recite it back at you, not to assume today is a continuation — you say what today
is — and that this is *all* it remembers, so it says so plainly rather than inventing something
when you ask about a month ago.

Only folders that already have a `summary.md` count, which keeps a session out of its own recap.
The line printed when a session connects says what was handed over:

```
· continuity: last session yesterday (22m 14s) + 4 earlier headlines
```

## The video

H.264 with a **stereo** audio track, **you on the left channel and Cyclops on the right**. The
two voices are never mixed, so you can listen to either side alone. Only the kiosk records:
`cyclops` opens the camera per photo rather than holding it open, so there is no continuous video
for it to record, and it has no panel either.

**The left channel is an open microphone, cut only while Cyclops is actually speaking, and
levelled.** Open means the room goes down with you: the bench, the fan, someone in the doorway.
The one thing it must not hold is Cyclops, because a mic in a room with a loudspeaker records a
delayed, room-coloured second copy of a voice already on the other channel — so the track is
silenced for exactly as long as sound is leaving the speaker, plus a quarter-second for the
room's tail. That is a narrower cut than [speaker mode](configuration.md#macos) puts on the mic,
which stays shut across a whole utterance.

The levelling is a compressor on the way in (`cyclops.audio.Compressor`) — a lavalier on a shirt
is 20–30 dB down on a mic at the mouth. Quiet speech is lifted about 20 dB, loud speech pulled
back, nothing clips. It sits ahead of everything, so what the model hears and what the recording
keeps are the same audio. `CYCLOPS_MIC_COMPRESS=0` turns it off; `CYCLOPS_MIC_GAIN_DB` adds a
fixed gain in front of it.

**What it records is a switch on the settings screen** — tap the eye, under INTERRUPT. It settles
what the *next* session records: an encoder is opened once, at one frame size, so a recording
already running cannot be handed something else halfway through.

| | what lands in `video.mp4` |
| --- | --- |
| **SCREEN** *(the default)* | The panel, 1:1 at 800×480 — the sharpened preview with the halo, the timer, the caption, the REC tag and the tab row composited on it. Watching it back is watching the session happen: when Cyclops was thinking, when you cut in, where the shutter went off. |
| **CAMERA** | The sensor alone, at its own resolution and shape, nothing drawn over it and nothing trimmed off the sides. The one to reach for when the recording is evidence rather than a memory — a part number, a wiring colour, a serial you will squint at later. |

`CYCLOPS_RECORD_SOURCE` decides it on a box where nobody has touched the switch. Photos are
unaffected either way: SNAP, the look tool and everything in `photos/` come off the raw camera.
Nothing on the box is mirrored, so text held up to the lens reads the right way round wherever it
turns up later.

Two consequences of recording the screen. A session with **no camera** still produces a video —
the panel saying `No camera found`, chrome and full stereo audio. And while a diagram, a
scratchpad or the admin page covers the panel the recording goes **black**, because the screen is
no longer the kiosk's to hand over; the audio carries on throughout.

Needs `ffmpeg` on `PATH`. Without it the session runs exactly as before and says once that it
isn't recording — nothing here ever blocks a conversation.

Measured on a Pi 5 at 15 fps, against the ~93% of one core the panel already spends drawing
itself at 25 fps whether anything is recording or not:

| | frame | over drawing alone | on the card |
| --- | --- | --- | --- |
| **SCREEN** | 800×480 | **+18%** of one core | **2.5 MB/min** |
| **CAMERA** | the sensor's own — 1280×720 on a C920 | **+35%** of one core | **8.5 MB/min** |

Two thirds of each figure is the kiosk handing frames over rather than ffmpeg taking them.
`CYCLOPS_RECORD_WIDTH` caps it if the card matters more than the detail does. Nothing is ever
pruned; delete what you don't want.

## Clips

A recording is an archive, not something anybody watches. So the index service looks at each
finished session once and asks what — if anything — in it is worth showing to somebody who was
not there, and writes what it finds to `clips/`: between **zero and three** clips of about
fifteen seconds each, every one a single moment. They are what the **VIDEOS** screen plays.

**Zero is the ordinary answer.** Two filters stand between a session and an encode, and both are
meant to say no:

1. **The log, free.** A session with fewer than two turns each way, under thirty seconds long, or
   with barely any words in it is never asked about at all — no `ffprobe`, no model, no network.
   On the card this was built against that answered for **55 of 98 sessions**, which is what a
   drawer full of "does the microphone work" looks like.
2. **The model, once.** The other 43 get one call, and the prompt spends most of its length
   making `NOTHING` an easy answer. A session that was genuinely just a torque figure being read
   out gets an empty plan and is never asked about again.

**What makes a clip tight is the audio, not the transcript.** `session.jsonl` knows when each
*turn* started, but a thirteen-second answer is really eight bursts of speech with 2.7 seconds of
pause inside it. So `silencedetect` is run over each channel — 0.55 s of work for a six-minute
recording, because `-vn` means the video is never decoded — and the moments the model named are
trimmed down onto the spans where somebody was actually audible. Across this card that removes
**65% of the running time**. Filler words are *not* removed and cannot be: the transcription
model normalises them out, so 220 real turns hold one "uh" and no "um" at all. What reads as the
"um"s going is the holes around them closing.

`clips/plan.json` being there is the whole "this one has been considered" marker, which is what
makes a render killed mid-way — by a deploy, say — cost the encode and never the model call. To
make it look again, press **Find clips** on the session screen; to fix a bad cut by hand, edit
the ranges in `plan.json`, delete that clip's `.mp4`, and the next sweep renders what you wrote.

## If a session is cut off

Power loss, a killed process — nothing is lost and nothing has to be done by hand. The Pi
finishes it on the next boot, and two rules make that work.

*Every file lands whole or not at all.* Everything written into a session folder goes through
`cyclops.card`: scratch name, fsync, rename into place. `Path.write_text` does not give you this,
and the failure it leaves is a **zero-byte `session.md`** wearing the name that means "finished".

*Presence is not existence.* "Finished" therefore means `session.md` **has bytes in it**, so a
zero-byte page is something to repair rather than something to skip.

An interrupted session keeps a `parts/` with a playable `video-raw.mp4` and the two WAVs, and a
`session.jsonl` that ends wherever the power did — `read_log` drops the half-written last line
and the page says so. `cyclops-sessions` lists it as `UNFINISHED`, or `EMPTY` if nothing at all
survived.

```bash
uv run cyclops-sessions                  # what's on the card
uv run cyclops-sessions --fix            # finish anything half-done (offline, no key)
uv run cyclops-sessions --fix --name     # ...and name/summarise what the live call missed
uv run cyclops-sessions --recover        # all of the above over the whole card, then file it
uv run cyclops-sessions --recover --dry-run   # ...say what that would do, touch nothing
uv run python -m cyclops.after PATH      # what a session end spawns: name, remember, file
```

`--recover` is what `cyclops-recover.service` runs at boot: the two verbs above, plus a folder
**nothing** survived in (no records, no video, no photos, no `parts/`) being deleted and whatever
was repaired being handed to the projects sweep. Anything with any salvage at all is kept. A
folder holding a file cyclops did not write is never deleted. A folder a session is writing into
right now is skipped entirely — a live session holds a `flock` on its own log, which is how a
boot sweep can tell, and renaming one out from under a conversation is the one mistake here that
would actually cost you something.

Install it once, on the Pi:

```bash
sudo deploy/install-recover.sh
journalctl -u cyclops-recover -b     # what it found and what it did
```

Photos taken with no session running land in `captures/`, which keeps only the last 20 alongside
`latest.jpg`. Nothing sees those — with no session there is no model to show them to — and the
session page says so. A session's own photos are never pruned.
