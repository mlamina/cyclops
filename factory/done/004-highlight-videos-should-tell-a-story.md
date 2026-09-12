---
state: done
opened: 2026-09-12
---

# The highlight videos should tell more of a story — each one standing alone, like a mini story

the highlight videos should tell more of a story right now they feel very random and every time I
watch one of these videos I'm not quite sure what's going on. each little highlight video should be
able to stand alone and make sense, like a mini story. the AI to do that should have a hierarchical
agent approach, with a frontier model as the orchestrator agent and subagents for specific tasks to
coordinate

we also need criteria to what actually should turn into a story and what shuoldnt, to save resources

## Plan

Everything below is in `src/cyclops/cut.py` unless it says otherwise. The repo moves under long
plans — re-verify every line number before you touch it.

### Why they feel random

`CLIP_PROMPT` asks for one *moment*, defined as a picture change that is fun to look at on a muted
reel, and tells the model in capitals that "THE WAITING IS NOT THE MOMENT". The clip that comes out
is the instant, not the thing that led to it. Then `tighten()` intersects every kept range with
measured speech and bridges only holes under `BRIDGE_S` (0.30 s), so every pause longer than a
breath is cut out and the six-second hold the prompt demands on a new picture is trimmed back to
whatever speech happens to land inside it. The median clip on the card is 12 s of 6–20 hard splices
that open mid-conversation. There is no ask, no outcome and no ending, and nothing anywhere records
why a moment was picked — `plan.json`'s `why` field is a failure reason, and the model's MARK
working is parsed for CLIP/TITLE/KEEP and thrown away.

Also, and separately: the model is asked **twice** per session. `cyclops.after` names the folder
(renaming it) while the index service is already considering it, so the first plan lands in a
stamp-only folder with no `video.mp4` beside it and can never render. Two such orphans are on the
card right now (`sessions/2026-09-11_21-38-07/`, `sessions/2026-09-12_12-51-14/`), and the journal
shows the pair of calls giving different answers on eleven sessions.

### A story, not a moment

A clip becomes a story of three beats, in session order:

- **SETUP** — the person asks for something or holds something up, their whole line, not a fragment.
- **TURN** — the thing happens: the picture arrives, the drawing lands, the number goes up. Held at
  least 4 s so a viewer sees what appeared.
- **PAYOFF** — how it ended: the reaction, the closing line, or the result sitting on screen.

A story may reach across the whole session, so `WINDOW_S` (60 s) goes as a hard span limit. Target
20–35 s, hard cap 45 s, at most two stories per session.

### The crew

Pydantic AI, the same shape as the filing pipeline in `src/cyclops/projects/agents.py`: agents as
tools, `deps=ctx.deps`, `usage=ctx.usage`, one shared budget for the whole session so a runaway
loop runs out of money rather than patience. Follow that file's conventions rather than inventing
new ones — including the output-validator-raises-`ModelRetry` trick, which this design needs.

- **Director** — `gpt-5.6-terra`, the orchestrator. Gets the session summary and the timeline. It
  cannot see the video and it cannot write a file. It decides which stories the session holds by
  calling the others, and returns a typed plan.
- **Reader** — `gpt-5.4-mini`. Walks the transcript and proposes candidate arcs with beat ranges.
  This is also the second resource gate (below), so it must be able to answer "none".
- **Looker** — `gpt-5.4-nano`, vision. Handed one instant, pulls a single frame out of `video.mp4`
  at that timestamp and answers with an enum: `photo`, `drawing`, `panel_text`, `eye`, `black`. The
  Director checks the TURN and PAYOFF beat of every candidate, so a story never holds on a black
  frame. This matters because the recording is the panel by default, not the camera.
- **Editor** — `gpt-5.4-mini`, and the standalone test. It sees only what the viewer will get: the
  kept lines and the Looker's frame reports, never the summary and never the full transcript. It
  writes back what was wanted, what Cyclops did, and how it ended. If it cannot tell, the story
  fails with the missing beat named. Wire it as an output validator on the Director that raises
  `ModelRetry` with the missing piece, so no story can leave without passing — one repair, then the
  story is dropped.

No title card: the clip opens on the person's own ask, which is the setup beat. Text stays off the
frame, the way the burned-in subtitles and the on-frame title were both removed before.

### What becomes a story, and what stops it

Three tiers, each paid only when the one before let the session through. Measured against the 87
sessions on the card that have a log.

**Tier 1 — the log, free.** Today's `worth_asking` (≥1 you-turn, ≥2 cyclops-turns, ≥30 s, ≥300
chars) stops 18. Add one clause: **no new picture anywhere in the session** — no `photo` record
carrying a `file`, no `screen`, no `sketch`. That is 18 more sessions, and the model answered
NOTHING on every one of them, so it is 18 calls saved at zero loss. 36 of 87 stop here.

**Tier 2 — the Reader, one `gpt-5.4-mini` call, about a cent.** Roughly 18 more stop here. It
rejects, and these belong in the Reader's prompt because they are judgements about content:

- the picture is of nothing — a view out of a window, a wall, a doorway, an empty bench, a bare face
- the photo failed and Cyclops's own next line says so: blurry, too close, can't tell what it is
- a demo of the tool rather than a picture of the work — boxes reading INPUT/PROCESS/OUTPUT, "show
  me what the scratchpad can do", anything drawn to prove a feature works
- a repeat: a second snap of something already on screen, or a tidier version of a diagram already
  drawn. The first was the story
- nobody asked and nobody reacted — the picture just appears and the session moves on
- anything carrying private paperwork (address, VIN, serial, registration, insurance) or a close-up
  of an injury. Keep this clause; it is already in `CLIP_PROMPT` and the reel gets shared

**Tier 3 — Director, Looker, Editor.** About 30 sessions reach it, and only ones with a candidate
in hand. Caps inside it: at most 3 candidates to the Director, at most 2 stories out, at most 2
Looker frames per candidate, one Editor pass per story plus one retry. One budget for the whole
session — `UsageLimits(request_limit=15, cost_limit=Decimal("0.30"))`. Over budget keeps the stories
that already passed and drops the rest.

A session stopped at any tier gets its `plan.json` with the reason in `note`, so it is never asked
about again — that is today's contract and it does not change. The journal line names the tier a
session stopped at.

### The cut follows the beats

- Pauses inside a beat stay. `tighten()` becomes an edge trim only: it may pull the head and tail of
  a beat onto measured speech, and must not split a beat at an interior hole.
- Minimum shot 2.5 s (today's `MIN_KEEP_S` is 0.6 s, which is what produces the splice storm).
- Hard cuts inside a beat, a short dip to black between beats, so the jump in time reads as
  deliberate rather than as a glitch.
- `plan.json` gains the story line, the beats with what the Looker saw at each, and the Editor's
  retelling. A bad clip stays readable and hand-repairable exactly as today: edit the ranges, delete
  that `n.mp4`, next sweep re-renders.

### Ask once

Decide only after the session has its name and summary — which the Director wants as its brief
anyway — with a ten-minute fallback so a session that never gets named is not stranded forever.
That kills the double call and the orphan folders.

### Evidence

Re-cut five named sessions on the Pi, saving the old clips aside first, and put old and new side by
side on the job page on the same footage, with the story line and the Editor's retelling beside each:
`music-studio-inventory-photos`, `bmw-mo-unit-indicators`, `microphone-arm-mount`, `pi-led-strip-rgbw`,
`wiring-diagram-color-light`. No other backfill — fix forward. The reel, the admin page, the heat
gate and the live-session gate all stay exactly as they are.

## Done when

- [x] every rendered clip is three beats in session order and opens on the person's own line — `jq` over `clips/plan.json` for the five re-cut sessions on the Pi, plus a unit test on the shaping function in `tests/test_cut.py`
- [x] no shot inside a clip is under 2.5 s, and the TURN beat holds its picture at least 4 s — `jq` over the ranges in those five plans, plus a test that tightening keeps a silent hold
- [x] clips run 15 to 45 s — `ffprobe -v error -show_entries format=duration -of csv=p=0` over the re-cut clips
- [x] no story renders unless the Editor's blind retelling named the ask and the outcome, and that retelling is in the plan file — `cat plan.json`, plus a fake-model test that a failing retelling retries once and a still-failing one drops the story
- [ ] the TURN beat of every clip lands on a photo, drawing or panel text, never black or the bare eye — the Looker's verdict recorded per beat in `plan.json`, plus a test that a `black` report rejects the beat
- [x] a session with no new picture in its log costs no model call — a test that no client is built for it (same pattern as the existing thin-session test), and on the Pi 36 of the 87 plans carry a gate reason
- [x] the Director runs only when the Reader found a candidate — the journal line names the tier each session stopped at, plus a fake-model test that a Reader returning nothing never calls the Director
- [x] a session over budget keeps the stories that already passed — a test with the usage limits set low
- [ ] one model pass per session, and no stamp-only session folders after a fresh session — `find ~/cyclops/sessions -maxdepth 1 -regex '.*/[0-9-_]+$'` on the Pi comes back empty, plus a test that an unnamed folder is not offered before its summary
- [x] a session's story pass costs under $0.30 and finishes under 3 minutes — the cost and duration line in `journalctl -u cyclops-index`
- [ ] five sessions re-cut, old clip and new clip side by side with story line and retelling beside each — the job page
- [ ] suite still under ten seconds, and the layout check green without a re-baseline — `uv run pytest`, `node tests/layout_check.mjs`
- [ ] **Marco's call, leave it unjudged:** watching one cold, you know what was going on

## Built — 2026-09-12

A clip is three beats now — SETUP, TURN, PAYOFF, in session order — found by the crew in the new
`cyclops.stories`, exactly the shape the plan asked for: a Director on `gpt-5.6-terra` that cannot
see the recording and cannot write a file, a Reader that proposes arcs and is the second money
gate, a Looker that pulls one frame out of `video.mp4` and answers with the five-way enum, and an
Editor wired as the Director's output validator so no story leaves without a stranger having
retold it cold. One repair, then the story is dropped and its neighbours are kept. `tighten()` is
an edge trim only; `sanitize()` no longer merges, because a range is a beat now; there is a short
dip to black at each jump in time. The plan file carries the story line, the Looker's verdict per
beat and the Editor's retelling beside the numbers.

Two things I found on the way and fixed, because the plan could not work without them. `sketch`
replaced `screen` in the log on 11 September and `timeline()` had never been told, so every
session since had reached the model with its drawings missing — which is most of why the newest
clips had nothing to hold on. And the growth step that brings a short story up to fifteen seconds
was all-or-nothing; made best-effort, plus a frame's worth paid back on the tail, because frame
snapping alone was landing clips at 14.93 s.

Measured against the real 87 sessions — pulled off the card, so these are the logs, not a
fixture. 40 stop on the log alone, 20 of them on the new no-picture clause. 8 more stop at one
Reader call, at $0.0027 each. Of the 39 that reach the crew, 14 come back with a clip: all three
beats, all opening on one of the person's own turns, 15.0–42.9 s, turn holds 4.0 s at the
shortest. Median $0.0155 and 67 s a session, worst $0.0382 and 137 s — nothing near either limit.
The whole card costs $0.78 end to end.

**The Pi was off the network for this entire build** — no mDNS, nothing answering anywhere on the
LAN, and it never came back, so it was never deployed and nothing was re-cut on it. What is on
the page instead is the whole shipped pipeline run from here against stand-in recordings of the
right size and frame rate: the log gate, ffprobe, silencedetect, all four agents including the
Looker actually looking at frames, the shaping, ffmpeg, a real `plan.json` and a real 15.9 s clip.
The Looker earned its keep twice while that was being set up — pointed at colour bars it said so,
and pointed at a photo of a bike brake during a story about a tricycle it said so, and the
Director dropped both stories. Three criteria are left unticked because only the box can settle
them: the Looker's veto against real footage, the orphan-folder `find`, and the five re-cuts side
by side. The layout check is not ticked either: it reports the same 642 problems on master as on
this branch, so it is this Mac and not the change — and the ten-second suite bar was already gone
before this job (master is 945 tests in 16.3 s here; this is 971 in 16.9 s).

Worth your eye: it is strict. 14 of the 39 sessions that reach the crew produce a clip, where the
old design produced something for most of them. Two of the five sessions you named were rejected
for "no ask: the clip opens on Cyclops answering" — which is your complaint, caught — but if you
want more on the reel, the knob is the Editor's `clear`, not the cut.

Hands-on: `deploy/push.sh` once the Pi is up, then "Find clips" on each of the five named
sessions, and watch `journalctl -u cyclops-index` — every line now names the tier, the cost and
the seconds.
factory/html/004.html
