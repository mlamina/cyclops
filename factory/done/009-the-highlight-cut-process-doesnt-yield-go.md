---
state: done
opened: 2026-09-18
---

# The highlight cut doesn't yield good results — dumb it right down to speeding up the silence

the highlight cut process doesn't yield good results. its cutting decisions feel random, important
parts are cut out, etc. I want to drastically simplify this and make it deterministic. Dumb it down
so that all we do is speed up the parts where neither I nor the agent are talking. that's it,
nothing fancy and no token waste

## Plan

### Why it feels random

Because it is a taste judgement, four models deep. `src/cyclops/stories/` runs a Reader on
`gpt-5.4-mini` that proposes arcs, a Director on `gpt-5.6-terra` that picks between them, a Looker
on `gpt-5.4-nano` that classifies single frames, and an Editor wired as an output validator that
can veto the lot — three cost tiers, a shared `UsageLimits(request_limit=15, cost_limit=0.30)`, and
about 1500 lines of prompt and schema. What comes out is a 15–45 s slice of a session and
everything else is thrown away. Job 004's own measurements: of 39 sessions that reached the crew,
14 produced a clip. Two of the five sessions named for evidence were rejected outright. There is no
version of "which 30 seconds of this mattered" that a model answers the same way twice, so the
cutting decisions are random because the decision is random.

### What it becomes

**One video per session: the whole recording, with the spans where neither channel is audible
played at 8×.** Nothing is chosen, nothing is dropped, no model is asked anything.

Everything below is `src/cyclops/cut.py` unless it says otherwise. The repo moves under long
plans — re-verify every line number before you touch it.

1. **The gate stays, minus one clause.** `worth_asking()` at :451 keeps its four measured
   thresholds (`WORTH_YOU`, `WORTH_CYCLOPS`, `WORTH_SECONDS`, `WORTH_CHARS`) — a mic check is still
   not a video. **Drop the "nothing new ever appeared on the screen" clause and `made_a_picture()`
   with it**: that existed only because a story needed a picture to hold a TURN beat, and there is
   no beat any more. On the card that is about 57 of 87 sessions through instead of 47.
2. **Measure, exactly as today.** `probe()` for the duration, `listen()` for `silencedetect` on each
   channel at its own floor (`MIC_FLOOR_DB` −38 on the left, `CYC_FLOOR_DB` −50 on the right),
   `read_silence()` to invert it and `_bridge()` to close holes under `BRIDGE_S`. All of that is
   already written, already tested, already cached in `Plan.speech`, and costs 0.55 s for a
   six-minute recording because `-vn` means the video is never decoded. It is the whole brain now.
3. **Audible = either channel.** Union the two channels' spans, then pad each by `LEAD_IN_S` /
   `LEAD_OUT_S` so no word onset is clipped, then merge anything that now overlaps.
4. **Trim the head and the tail** to the first and last audible moment. A session opens on a tap
   and ends on one; there is no reason to open on six seconds of a blurred bench.
5. **Everything between two audible spans is a gap.** A gap shorter than `GAP_MIN_S` (1.2 s) is left
   at 1× — speeding a breath up reads as a dropped frame, not as an edit. Every other gap plays at
   `SPEED` (8×).
6. **One ffmpeg pass.** `clip_command()` becomes a segment list rather than a beat list: per
   segment `[0:v]trim=…,setpts=PTS-STARTPTS` and `[0:a]atrim=…,asetpts=PTS-STARTPTS` at 1×, or
   `setpts=(PTS-STARTPTS)/8` with `atempo=2,atempo=2,atempo=2` in a gap — `atempo` caps at 2.0, so
   8× is three of them, and using it rather than dropping the audio keeps video and audio locked
   together by construction. Then `concat`, then `fps=15` so the fast parts do not carry eight
   times the frames, then today's scale/pad/`format` and the `pan` that folds you and Cyclops into
   both ears. Same encoder settings as today. **No fades and no dip to black** — `DIP_S` and
   `EDGE_FADE_S` go, because there are no jumps left to explain.
7. **The title is the summary's first line.** `summary.md`'s one sentence is already written, free,
   and better than anything the Director produced. `MAX_TITLE_CHARS` still trims it.
8. **The output is still `clips/1.mp4`, and there is always exactly one.** That is deliberate: the
   reel, the HIGHLIGHTS screen, `/media/<session>/clips/1.mp4`, `cut.clips()`, `cut.progress()`, the
   Find clips button and the hand-repair path (edit `plan.json`, delete the mp4, next sweep
   re-renders) all keep working with no change at all.

### What goes

- `src/cyclops/stories/` — all four files, 920 lines — and `tests/test_stories.py`.
- From `cut.py`: `timeline()`, `opens_on_a_person()`, `shape()`, `_grow()`, `_stretch()`,
  `_towards()`, `_cap()`, `sanitize()`, `_captions()`, the `Beat` dataclass, `BEATS`, `MAX_CLIPS`,
  `MAX_TIMELINE_CHARS`, `MAX_LINE_CHARS`, `SNAP_TO_TURN_S`, `WINDOW_S`, `MIN_SHOT_S`, `TURN_HOLD_S`,
  `STORY_LOW_S`, `STORY_CAP_S`, `STORY_TARGET_S`, `DIP_S`, `EDGE_FADE_S`. `tighten()` survives only
  as the padding in step 3. On `Plan`: `tier`, `cost`, `took` and `by` have nothing left to say;
  `Clip` loses `story`, `beats` and `retelling`, and `ranges` becomes the segment list with its
  speed. `note` and `why` stay — a gated session must still get a plan file so it is never
  reconsidered, which is today's contract and does not change.
- `pydantic-ai` stays in `pyproject.toml`: `projects/` and `agent.py` use it.
- `cut.py` should land around 400 lines, from 1449.
- The `Clips` section of `docs/sessions.md` (:121–151) describes the old design and is already
  stale in two places. Rewrite it to say what the code does.

### The card

Marco's call: **delete every `clips/plan.json` on the Pi once, after the deploy.** The index service
then reconsiders all 96 sessions behind the gates it already has — `busy()` stands down for a live
conversation and above `stats.HOT_C`, and `_guarded()` does one unit of work per wake — so about
fifty encodes spread themselves over days rather than cooking the box in one pass. Nothing else is
backfilled and nothing is deleted by hand.

### What it will cost, and what is not known

`-ss` before `-i` made the old decode proportional to a fifteen-second clip. That is gone: this
decodes the whole recording, every time. Measured over the 93 recordings on the card — median 80 s,
p75 125 s, p90 238 s, longest 957 s. `CUT_BUDGET_S` is 420 s in `indexer.py:65`, and the longest
session may want more than one wake; find out rather than assume.

**The dead-air figures are not measured.** The Pi went down mid-measurement and there are no
sessions on this Mac. `docs/sessions.md:145` records 65% of running time removed by this same
silencedetect pass across this card; at that ratio and 8×, a median session lands near 35 s and the
957 s outlier near 6 minutes. Measure it for real on the Pi and put the table on the job page.

**8× is a pick, not a measurement.** Cut one real session at 4×, 8× and 16×, put all three on the
job page, ship 8× first.

## Done when

- [x] the cut asks no model anything and needs no key or network — a test that a session renders a finished video with the OpenAI client patched to raise on construction, and `ls src/cyclops/stories` is gone
- [x] nothing is dropped: the segments mapped back to source time cover the recording from the first audible moment to the last, with no holes and no overlaps — unit test over a synthetic span list
- [x] speech plays at 1× and only gaps of 1.2 s or more are sped up — unit test: a 0.5 s hole between two spans yields one unbroken 1× segment; a 3 s hole yields 1× / 8× / 1×
- [x] no word onset is clipped — unit test that each audible span is padded by `LEAD_IN_S` / `LEAD_OUT_S` before the gaps are computed, and that padding which makes two spans touch merges them
- [x] the rendered file's duration matches the arithmetic to within a frame — one end-to-end render of a short synthetic recording in pytest, `ffprobe -show_entries format=duration` against the predicted length
- [ ] exactly one video per session at `clips/1.mp4`, titled with the summary's first line, and the reel plays it — yours, on the Pi: HIGHLIGHTS on the panel
- [ ] 8× reads as time passing rather than as a glitch, and you can still see what happened during it — yours, on the Pi: 4×, 8× and 16× of the same session side by side on the job page
- [ ] **Marco's call, leave it unjudged:** watching one cold, nothing important has been cut out
- [ ] what a cut costs now it decodes the whole recording — the seconds per session in `journalctl -u cyclops-index`, for a median session and for the 957 s one; yours, on the Pi
- [ ] every session on the card gets reconsidered without the box overheating — yours, on the Pi: the count of `clips/plan.json` climbing back, and `vcgencmd get_throttled` after
- [ ] `cut.py` under 450 lines and `src/cyclops/stories/` and `tests/test_stories.py` gone — `wc -l`, `ls`
- [x] the `Clips` section of `docs/sessions.md` describes what the code does — read it
- [ ] the suite is no slower than master — `uv run pytest` on both

## Built — 2026-09-18
Each session now becomes one video, `clips/1.mp4`: everything from the first audible moment to the
last, where either channel being audible plays at 1× and every quiet stretch of 1.2 s or more
(measured after the 0.15 s / 0.25 s padding) plays at 8×, with the sound sped up via three
`atempo=2`s. No model, no key, no network; `src/cyclops/stories/` and its tests are deleted. The
gate kept its four thresholds and lost the "nothing new on the screen" clause. Title is the
summary's first line. The plan still has one clip whose `ranges` are `[start, end, speed]`; an old
pair reads as 1×, so hand repair still works. One decision on the way, found by measuring:
one `trim` per segment made cost grow much faster than the segment count (400 segments over a
957 s recording: 150 s on an M3 Max against 15.5 s for 91), so the render splits the recording
once with `segment`/`asegment` instead: 11.8 s at 400. Still one ffmpeg pass, same encoder
settings, `fps=15` after the concat, no fades. The render timeout is now a flat 360 s (was 240 s,
sized for a 15 s clip); it stays a minute under the index service's 420 s budget. Checked on
Debian bookworm's ffmpeg 5.1.9 (aarch64, container): renders cleanly, and the short test render
lands within 1 ms of the prediction. Each render now logs "rendered in Ns" to the index journal.

Two lines are **not met**, both flagged rather than worked around:
- `cut.py` is **761 lines**, not under 450 (stories/ and test_stories.py are gone). What the plan
  keeps (queue, plan I/O, the render's heat/conversation watch, the reel's listing) is ~500 lines
  of code before any docstring. The 400 estimate counted the removals, not what stays. Getting
  under 450 would mean dropping behaviour or splitting the file to game `wc -l`; I did neither.
  Say if you want a split.
- The suite is **~0.4 s slower** than master. Whole-suite runs are inside the noise (master
  16.3–17.8 s, branch 16.8–19.0 s, machine loaded by other builds). Summed per file in-suite,
  test_cut costs 0.55 s against 0.13 s for the old test_cut + test_stories. All of it is the one
  real ffmpeg render the duration criterion asked for (synthetic recording already cut to 8 s).

Every "yours, on the Pi" line is unticked and needs /try. Expected: delete every
`clips/plan.json` once after the deploy; the index service rebuilds about 57 of the 87 real
sessions, one per wake, behind `busy()`; a median session renders in well under a minute on
the Pi (M3 Max: 0.9 s for 80 s, 16.5 s for a synthetic 957 s).

Hands-on: /try, then delete every `clips/plan.json` once and watch HIGHLIGHTS fill. For the 4/8/16
comparison on one median session (renders to /tmp, plan untouched):
`.venv/bin/python -c "import sys,subprocess; from pathlib import Path; from cyclops import cut; f=Path(sys.argv[1]); s,_=cut.probe(f,[]); sp=cut.listen(f,s); [(setattr(cut,'SPEED',x), subprocess.run(cut.clip_command(cut.Clip('',cut.segments(sp,s)),f'/tmp/speed-{x:g}.mp4'),cwd=f/'clips',check=True)) for x in (4.0,8.0,16.0)]" sessions/<name>`
Then `journalctl -u cyclops-index | grep "rendered in"` for the median and the 957 s session,
and `vcgencmd get_throttled` once the backlog drains.
factory/html/009.html
