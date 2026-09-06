# What it remembers

[Continuity](sessions.md#continuity) is what *happened*. This page is the other two kinds: what
Cyclops knows about **you**, and what it keeps about the **projects** you build.

## About you

Neither continuity nor projects is about the person on the other side of the bench, and without
that a machine that has talked to you for a month still opens on a stranger — no name, no idea
what you do, no memory that you said last week you wanted to get better at dovetails.

So there is a third kind of memory, and it is one file at the root of the card:

```markdown
# About you

- Goes by Priya.
- Restores furniture at weekends; ten years at it.
- Has a small garage shop: bandsaw, thicknesser, no lathe.
- Wants to get better at hand-cut dovetails.
- Deaf in the left ear, so say things once, clearly.
```

That is the whole format, and it is a file rather than a database for the same reason
`Project Data.xlsx` is a spreadsheet: **if a line is wrong you open it and fix it.** Delete a
line and it stays deleted unless Cyclops hears the same thing again; delete the file and it
starts over knowing nobody. The prose at the top is written by the code and never read back, so
what you edit is only ever the list.

**A line earns its place by changing how Cyclops talks to you** — your name, what you do, how
practised you are, what tools you have, what you want to get better at, and how you want to be
spoken to: language, units, anything about hearing or sight or handedness that changes what help
looks like. What does not belong: anything about *today* rather than about *you* (the bore is
12.7 mm, and that goes in the sheet), anything you did not say about yourself, and anything
private you did not offer as a standing fact — health, money, who you live with, where you live.
Held up against a session that mentioned chemotherapy, an address and a tight month, it wrote
down two lines: *works as a nurse, not an experienced builder* and *has a hammer drill and a
spirit level*.

**The list is rewritten, not added to.** When a session ends, a model gets the list as it stands
plus the transcript and answers with the whole list back, so *uses Fusion 360* becomes *uses
OnShape* in place rather than sitting on the card next to its own contradiction. It goes out in
the same detached process as the naming, once the session is already over, so it costs a shutdown
nothing.

It is `gpt-5.6-terra`, not the `gpt-5.4-nano` that names the session, because naming is a
transcription and this is a *reconciliation*: work out which sentences about a person a new
conversation has made untrue. Handed *uses Fusion 360* and a move to OnShape five times over, the
small model got it clean four times and once left a spare line about sketching being quicker — a
sentence about a week, in a list meant to be about a person. It is not the cheaper trade either:
over 32 real sessions the small model ran a median 1.4 s against 1.0 s, worst 12.3 s against 5.3 s.

Rewriting has one sharp edge, and the whole of `cyclops.about` is built around it: **an empty
answer never empties the file.** No key, no network, a refusal, a timeout, a reply in the wrong
shape, or a model that decided to say nothing — all of them mean *leave the list alone*, never
*forget everything*. The only things that can shorten it are a well-formed shorter reply and you,
with an editor. Most conversations teach it nothing about you, and an identical answer is not
written at all.

**What it does when it knows nothing.** An empty list is not silence. Cyclops is told that
nothing has been written down about you, and to ask — *one* open question, once, never as the
opening line and never while you are mid-cut or waiting on an answer. Brush it off and it drops
the subject for the day. Same "offer once" rule as the project question, and for the same reason:
the fastest way to ruin this is a box that interviews someone who came in to fix a tap.

What was handed over is printed when a session connects, beside the continuity line:

```
· about you: 6 thing(s) known
· about you: nothing yet - it will ask once
```

The list is deliberately not on the admin page — the screen you open to find a recording is not
the screen you open to audit a memory. The file is the interface: `about-you.md`, in any editor,
which is also where you correct it.

`CYCLOPS_REMEMBER=0` turns the whole thing off: no call at the end of a session, and not a word
about you in the instructions at the start of one.

## Projects

A session is one conversation. A **project** is the thing you keep coming back to, and it gets a
folder of its own that a person can read without knowing anything about any of this:

```
projects/
  Pelican Display Mount/
    README.md            what it is, where it stands, what's still open, what was decided
    Log.md               one dated entry per session, oldest first, never rewritten
    Project Data.xlsx    the numbers: torques, sizes, part numbers, codes
    Photos/
      2026-08-26_16-48-48_cyclops.jpg
```

**A project starts because you said so.** Cyclops is told the names of the projects it keeps at
the start of every session. Working on something that isn't one of them, and looks like a thing
you'll come back to, it asks — once, in one sentence — whether to keep notes. Say yes and the
folder appears there and then. Nothing else can create one, which is what stops the card filling
up with "Pelican Case", "Pelican Display" and "Pelican Mount" as a model changes its mind.

It's told the names and nothing else. When you come back to something it calls `open_project` to
read what was decided last time — so twenty projects cost twenty lines of its attention, not
twenty pages.

**The filing happens afterwards, on its own.** It is the last of the three jobs `cyclops.after`
does once a session has ended, and the kiosk forgets about all of them: an orchestrator on
[Pydantic AI](https://ai.pydantic.dev) reads the session and delegates to four smaller agents —
one reads the transcript, one decides which project it advanced, one writes the log entry and the
new page, one picks a few photos worth keeping. It never invents a project: it either files under
one that exists or writes down that it filed nothing.

`Log.md` is the truth and `README.md` is a picture built from it, so you can delete a README and
the next sweep makes it again. Nothing on the card is a database; the machine-readable part is
the YAML block at the top of each README, and one invisible HTML comment per log entry holding
the session's UUID — which is what stops an entry being written twice.

Nothing is filed twice, and nothing is lost. `project.md` in the session folder is the receipt: if
it's there the session has been read, and if it isn't, the next sweep reads it. So a power cut
costs a re-read and nothing else, and it's safe to run over a whole card as often as you like.

```bash
uv run cyclops-projects              # what's on the card, and how many sessions are waiting
uv run cyclops-projects --check      # offline: pages that drifted from their log, torn entries
uv run cyclops-projects --sweep      # file everything unfiled, oldest first (needs a key)
uv run cyclops-projects --sweep --dry-run   # say what it would write, write nothing
uv run cyclops-projects --sweep --again  # re-read everything; the log ledger prevents duplicates
```

Oldest first is not cosmetic: a project's page is rewritten against what the session before it
left behind. `cyclops-sessions` flags anything still `unfiled`.

### The numbers

The prose says what was decided. **`Project Data.xlsx`** says the bore is 12.7 mm. Numbers want
different storage from sentences — a torque that goes through a summariser comes back as "around
25" — so they live in a sheet, one row each, under tabs Cyclops picks: *Torque specs*,
*Dimensions*, *Paint*. Three columns, **Key | Value | Note**, and the value is stored exactly as
it was spoken or read off the plate, unit included. If a value is wrong you open it and fix it.

Hold a spec plate up, press SNAP, and every number on it goes down in one write. Ask
*"what was the caliper torque?"* three weeks later and it looks the value up rather than
remembering it — the difference being that looking it up can come back empty, and it will tell
you so instead of producing a plausible number.

**Values are never loaded into the conversation.** Opening a project tells Cyclops the tab names
and how many rows are in each — one line — and every value after that is a tool call. So two
hundred numbers cost the same attention as two, which is the entire reason this is a file and not
part of the notes. Retrieval is a plain word-overlap score with a spelling fallback, so "caliper
torq" through a microphone still finds *Caliper bolt torque*; no index to rebuild, no embedding
to go stale.

It only ever writes into a project you already track, and it will ask which one rather than
guess. Deleting is the one thing it refuses to do on a guess: if more than one value could be the
one you meant, it deletes nothing and asks.

Every session end costs one model call even when nothing gets filed, because deciding that *is*
the call. At these model sizes it rounds to nothing, but `CYCLOPS_PROJECTS=0` turns the whole
feature off — no sweep, and the two project tools aren't offered to the voice agent at all.
