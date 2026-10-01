---
state: done
opened: 2026-09-18
---

# A manual answer flows like talk: no naming the page, and keep looking when a page misses

when cyclops reads info from a manual, it always shows me the page and explicitely names the
manual name and page. that feels too mechanical and breaks the conversation flow. also, when it
finds a page and it turns out that page didn't have the info we were looking for, it should keep
looking instead of saying the page it found wasn't helpful. look at the transcript of my OXI
session from yesterday to see what i mean

Asked when a page should go on the panel (only when seeing it helps / always, silently / only
when asked): **only when asked**.

## Plan

**What yesterday's session shows** (Pi: `sessions/2026-09-17_21-05-56_oxi-one-sequencer-baseline`)
- **Every manual answer opens with "OXI ONE manual, page 18."**, including at 10:10 and 15:09,
  when it had looked nothing up. Three places tell it to cite:
  - `LOOKING THINGS UP`: "just answer and say where it came from in a few words" (`agent.py:~1037`)
  - `RECALL_TOOL`: "Say which manual and page in a few words, and stop" (`agent.py:~803`)
  - `add_found`: "Say which manual and page it is, in a few words, so they know where the answer
    came from." (`agent.py:~3142`). This is a user-role message that stays in the history under
    every page it has read, so the instruction keeps firing on later turns.
- **Every page goes on the panel automatically.** `_recall` calls `_show_found` for any showable
  best hit (`agent.py:~3023`), and the model has no say.
- **It gives up instead of looking again.**
  - Nothing tells it to search again. The wording says "offer to look another way" / "offer one
    of them", and that turns into "If you want, I can look for that page next" (7:24, 10:44).
  - When it did search again, the search couldn't help. Three rewordings of the pitch question
    all returned page 18 (7:09, 7:23, 7:53). Ranking is deterministic embeddings with no session
    memory, and page 12 had the answer.

**Changes**
1. **Answer like it knows.** Remove the citation from all three places. It names the manual and
   page only when they ask where the answer came from. The transcript still records every page
   read (`*Looked for* … → **title**`), so nothing is lost. Don't tell it to "stop" after
   answering from a manual. Keep the "don't volunteer what the manual says when it doesn't answer
   the question" rule.
2. **Reading a page is separate from showing it.**
   - `recall` always hands the full page render to the model, as today.
   - A manual page goes on the panel only when they asked to see it. `recall` gets a `show` flag,
     false by default. Its description says to set it only when they asked to see the page, or
     a picture from it.
   - Photos, drawings and other non-page pictures keep going up every time, unchanged.
   - When a page is not on the panel, the `add_found` message and the tool's `note` say only
     Cyclops can see it. It must never say "as you can see" or "it's on your screen".
3. **Keep looking, quietly.**
   - If the page it read doesn't answer what was asked, it calls `recall` again straight away
     with a differently worded query, and never tells them about the miss. It reads up to three
     pages for one question.
   - If none answers, it says in a few words that the manual doesn't cover it, then goes to the
     web as `LOOKING THINGS UP` already says.
   - Rewrite every "offer to look" / "offer one of them" in the recall wording (description, the
     no-hit note, the others note) so it *does* rather than offers.
   - The when/when-not detail goes in the tool description, not the system prompt.
4. **The search stops repeating itself.** Any page already handed to the model since they last
   spoke is skipped, so the next page looked at is a different one. The list clears on
   `_on_user_speech_started`. Rank enough hits that three skips still leave candidates above
   `MIN_SCORE`. When nothing new is left, say so in the result.

**Probe without the Pi.** The OXI manual is copied to
`../cyclops-fixtures/manuals/OXI One User Manual` (pages.json, the 161 page
renders and the PDF, outside the repo).
- Add a `--script manual` to `tools/talk_probe.py` that replays yesterday's questions against the
  real realtime model:
  - how to add a slide
  - how to reach the notes below the ones lit on the grid
  - which one is the Div button
  - how to copy a pattern
  - "show me that page"
  - "where's that from?"
- Point `CYCLOPS_MANUALS_DIR` at the fixtures and put the recall index somewhere scratch, never
  the Mac's `~/.cache/cyclops`.
- The tally reports, per turn: the recall calls, the page each one read, whether `show` was set,
  seconds from question to answer, and any line naming the manual or a page number.

## Done when
- [ ] A question that goes to the manual is announced once, in a few words like "Let me look at the manual", before the lookup — same probe: the line spoken before each question's first recall call
- [x] No Cyclops turn names the manual's title or a page number, except the answer to "where's that from?", which names both — `talk_probe.py --script manual --runs 5`: the tally, plus the flagged lines read by eye
- [ ] Asked how to reach the notes below the grid, Cyclops answers in the same turn (a wrong answer is OK for now), and no turn says a page didn't have it or offers to look further — same probe: recall calls per turn and the page each one read, plus the lines
- [x] That chain of lookups gets to the answer in 8 s or less from the question — same probe, seconds per turn
- [x] `show` is set only on the "show me that page" turn, and never on the slide, lower-notes, Div button or copy-pattern turns — same probe, `show` per recall call
- [x] A page already read since they last spoke is never handed back again, and their next words clear that — `uv run pytest`, new test
- [x] A page read without `show` reaches the model and not the panel, a page read with `show` reaches both, and a recalled photo still goes on the panel — `uv run pytest`
- [ ] The whole suite is green and runs in under 10 s — `uv run pytest`
- [ ] In a real session with a manual, answers sound like it just knows, and nobody hears about a page that missed — yours, on the Pi
- [ ] The page stays off the panel unless you ask, and "show me that page" puts it up full width — yours, on the Pi

## Built — 2026-09-18 (left at `ready`: the page-12 criterion can't be met by this plan)
Cyclops no longer names the manual or the page unless it's asked where the answer came from, and
then it names both (5 of 5). A manual page goes to the model alone. It goes on the panel only
when `recall` is called with `show`, which the model set on 10 of 10 "show me that page" turns and
on none of 64 other lookups. Photos still go up every time. Anything handed back since they last
spoke is skipped, and their next words (or a typed turn) clear that. No turn with the page off
the screen talked as if they could see it (0 of 70).

One decision on the way: the three-page limit is enforced in code as well as in the description.
In the first probe run the model read six pages for "which one is the Div button?", talking
through each miss, and took 13 s. A fourth lookup for one question is now refused, with an
instruction to say the manual doesn't cover it and use the web. I also added
`talk_probe.py --script manual-cold`, which asks the lower-notes question before anything has been
read. In the full script, the slide question usually reads page 12 first.

**Why it stays at `ready`:** the lower-notes criterion fails, and the plan can't fix it. The
skip list works as designed: a rewording now always gets a different page, never page 18 three
times. But page 12 ranks only 2nd to 15th (usually 5th to 7th) for the words the model searches
with, with scores bunched between 0.64 and 0.75, so three reads rarely get there. Cold, page 12
was read in 0 of 5 runs. Across the 10 lower-notes turns, 5 answers were right (from page 12
already in hand, or from page 51), 4 were "SHIFT + Page" from page 28, which transposes the notes
instead of scrolling the view, and 1 was vague. With no citation, that wrong answer now sounds
certain. Getting page 12 means indexing manual pages at a finer grain, for example by paragraph,
so the one sentence that answers ranks and not the whole page. That's a new plan. The Div-button
turn also gave up in 3 of 5 runs with "the manual pages we found don't show the layout", which
this line counts as saying a page didn't have it.

Suite: green, 968 passed, 16.3–16.8 s. Master is 16.6–17.0 s on this Mac, so the 10 s bar was
already missed before this job.

Hands-on: `/try` it. Ask the OXI One a few things: nothing should be cited, and no page should
come up until you say "show me that page", which should put it up full width. Then ask "where's
that from?". Expect "which one is the Div button?" to often ask to look at the device, and expect
some lower-notes answers to be wrong (see the page).
factory/html/011.html

## Feedback — 2026-09-18
It should still let me know that it's looking at the manual (not hide the fact that it's doing that). it just shouldn't be as verbose about it as before. a simple "Let me look at the manual" or soemthing is fine. I udnerstand that it makes mistakes, that's OK for now and we can fix that later

## Built — 2026-09-18 (round 2)
Before the first look in a manual for a question, Cyclops now says "Let me check the manual." and
then looks. It never says the title, and it says nothing more before a second or third look. It
still names the manual and page only when asked where the answer came from (10 of 10), and no
other turn named either (0 of 60). Over 10 runs of `--script manual` and 5 of `manual-cold`, 27 of
28 lookups were announced before the first page, 20 of them in exactly those five words. Round
one was 2 of 21, mostly "let me think through…". The copy-pattern question is the one that runs
long: 7 of 10 added a tail, 4 of them 12–22 words ("…so we don't duplicate the wrong thing"). The
line is left unticked for that. The lower-notes question was answered in the same turn every
time (10 of 10, and 5 of 5 cold, all right). Two of 70 turns still talked about a page that
missed, both "which one is the Div button?", where no page shows the button's position. In 7 of
10 runs that question asks to see the device instead, which on the Pi means it points.
Decided on the way: when three pages miss it says "The manual doesn't say." and goes to the web,
and the third page now tells it that it's the last. A page put up because they asked is said to
be up and nothing more. "If you want, I can…" on that turn went from 3 of 5 to 0 of 10. The `show`
description now rules out "which button" and "where", after one run set it there; since then it
was set on 0 of 75 lookups outside "show me that page". Suite: green, 968 passed, 16.7 s, the
same as master.
Hands-on: `/try` it and ask the OXI One a couple of things. Each should start "Let me check the
manual." and then answer with no title and no page. "Show me that page" puts it up; "where's that
from?" names both.
factory/html/011.html
