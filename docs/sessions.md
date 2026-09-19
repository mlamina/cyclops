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
    video.mp4         the recording - the whole screen, or the camera (kiosk only)
    clips/
      plan.json       what was decided about this session, and when
      1.mp4           the whole session, quiet stretches played fast - see "Clips" below
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

**What it records is the whole screen, exactly as it was on the glass**, at 30 fps: the camera
with its halo, timer, caption and tab row, and then whatever covered it — a diagram, a photo, a
scratchpad, a sketch, a manual page scrolling under your finger, the settings screen. Nothing is
rebuilt afterwards and nothing goes black. `wf-recorder` takes each frame off the compositor as
it is drawn (`cyclops.screen`) and the recorder samples the newest one on the wall clock, so the
sound lines up with the picture the same way it always has.

If the screen cannot be captured — `wf-recorder` is not installed, or no frame arrives within
the first second — the session records **the camera** instead, and `session.jsonl` says why in a
`recording` line:

```
{"type": "recording", "source": "camera", "why": "wf-recorder is not installed"}
```

`deploy/install-screen-capture.sh` installs it, once per box. Photos are unaffected either way:
SNAP, the look tool and everything in `photos/` come off the raw camera. Nothing on the box is
mirrored, so text held up to the lens reads the right way round wherever it turns up later.

A session with **no camera** still produces a video — the panel saying `No camera found`, and
full stereo audio.

Needs `ffmpeg` on `PATH`. Without it the session runs exactly as before and says once that it
isn't recording — nothing here ever blocks a conversation.

What it costs, measured on the Pi 5 before it was built (2026-09-18), on top of the ~93% of one
core the panel already spends drawing itself:

| | over not recording |
| --- | --- |
| `wf-recorder` capturing the screen, ~29 fps, no encode | ~15% of one core |
| encoding 800×480 at 30 fps | ~18% of one core |
| **together** | **~33% of one core** |

At 15 fps the old stitched recording cost about 18%. What it costs on the card at 30 fps has not
been measured yet. `CYCLOPS_RECORD_WIDTH` caps the size if the card matters more than the
detail does. Nothing is ever pruned; delete what you don't want.

## Clips

A recording is an archive, not something anybody watches: most of its running time is somebody
working with nobody talking. So the index service turns each finished session into **one video**,
`clips/1.mp4` — everything from the first word to the last, with every stretch where nobody is
talking played at **8×**. Nothing is chosen and nothing is cut out, no model is asked anything,
and it needs no key and no network, so the same recording always comes out the same way. It is
what the **HIGHLIGHTS** screen plays, titled with the first line of `summary.md`.

**Who gets one.** The log is read first, for free. A session with no turns from you, fewer than
two from Cyclops, under thirty seconds, or barely any words in it gets no video and costs no
`ffprobe` and no `ffmpeg` — that is a button pressed to check the microphone.

**Where the talking is.** `silencedetect` runs over each channel at its own floor — the mic on the
left at −38 dB, Cyclops on the right at −50 dB — and either one being audible counts. That is
about half a second for a six-minute recording, because `-vn` means the video is never decoded.
Every audible span is widened by 0.15 s in front and 0.25 s behind so no word is clipped, and the
empty head and tail are trimmed off. A quiet stretch under 1.2 s is a breath and stays at normal
speed; anything longer plays at 8×. The sound is sped up with it rather than dropped, so picture
and sound cannot drift apart.

**What it costs.** One `ffmpeg` pass that decodes the whole recording, so a long session takes
longer than a short one. It waits for a live conversation to end and for the board to be under
80 °C, and stands down if either changes mid-encode; one video per wake of the index service.

`clips/plan.json` being there is the whole "this one has been considered" marker, so a render
killed mid-way — by a deploy, say — is simply rendered again on the next sweep. To make it look
again, press **Find clips** on the session screen. To fix a cut by hand, edit the segments in
`plan.json` — `[start, end, speed]`, in seconds of the recording — delete `clips/1.mp4`, and the
next sweep renders what you wrote.

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
and the page says so. `cyclops-sessions` lists it as `UNFINISHED`, or `NOTHING IN IT` if nobody
spoke and nothing was made.

```bash
uv run cyclops-sessions                  # what's on the card
uv run cyclops-sessions --fix            # finish anything half-done (offline, no key)
uv run cyclops-sessions --fix --name     # ...and name/summarise what the live call missed
uv run cyclops-sessions --recover        # all of the above over the whole card, then file it
uv run cyclops-sessions --recover --dry-run   # ...say what that would do, touch nothing
uv run python -m cyclops.after PATH      # what a session end spawns: name, remember, file
```

`--recover` is what `cyclops-recover.service` runs at boot: the two verbs above, plus a folder
**nothing happened in** being deleted and whatever was repaired being handed to the projects
sweep. Nothing happened in it means its log holds no turn anybody spoke and nothing that was
made — a picture, a drawing, a project, a number. A `video.mp4` is not salvage on its own: ten
seconds of an empty room is what a button pressed by accident leaves behind. Where there is no
log to judge by, whatever the folder has is kept. A folder holding a file cyclops did not write
is never deleted.

`--tidy` is the other half, and the half you run by hand. Some sessions are dialogue and still
nothing — "hey, can you hear me?" — and no rule cheaper than reading them can tell those from a
question answered in twenty seconds. So it reads each one back and asks, skipping anything that
made something. It needs a key, costs one small call per session, and takes `--dry-run`. A folder a session is writing into
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
