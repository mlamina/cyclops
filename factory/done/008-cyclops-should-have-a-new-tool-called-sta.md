---
state: done
opened: 2026-09-18
---

# Cyclops should have a start_tutorial tool, and the status line should show where you are

cyclops should have a new tool called "start_tutorial". when called, the top row of the status
line should show a segmented completion bar of the tutorial steps and the second line should show
which step we're currently on. it should have a second tool "advance_tutorial", which moves the
status indicator forward. cyclops should use this whenever i ask it to walk me through something
step by step. instead of jamming all instructions into the tool descriptions and base prompt, make
smart use of agent instructions in tool return text, for a natural, progressive-disclosure like
mechanism

Asked how a tutorial ends when you say "forget it, stop" halfway through: **a third tool,
`end_tutorial`**. Asked what happens to job 005's tutorial mode, which is in review and built on a
branch: **standalone — build against master as if 005 does not exist, and decide 005 later.**

## Plan

**The tutorial becomes real state in Python — a list of steps and an index — and the status line
grows a second job: the top row is a segmented bar, the bottom row is the step you are on. Nothing
about it goes in the system prompt. The rules arrive in the tool returns, one step at a time.**

### The status line has to be split first

Today it is a two-row monospace terminal, `Rect(264, 404, 352, 72)` (`overlay.py:3733`), glass
334×54, two rows of 24 px, ~35 chars each. The two rows are **one wrapped paragraph, not two
slots**: `_wrap` (`overlay.py:5910`) greedily fills row 1 and spills into row 2, `_elide`
(`:5939`) truncates row 2 with `…`, and there is no way to address row 2 on its own.

So: while a tutorial is running, **row 1 belongs to the bar and captions are capped to row 1's
worth of text on row 2**, elided rather than wrapped. `CAPTION_LINES = 2` (`overlay.py:622`) and
the ink ceiling over those rows (`:7203-7211`) both assume two text rows; re-solve them for the
tutorial case rather than changing the case height. The glass is cut to exactly two rows and must
stay that way — the bar goes *in* row 1, not above the terminal.

### The bar is drawn, not typed

The panel already has a settled idiom for "how much of something", and the code says so out loud
in `_draw_slider`'s docstring (`overlay.py:8357`): *"a panel with two ways of drawing 'how much of
something' has one too many."* There are four of them today — the 8-segment signal meter
(`METER_SEGMENTS`, `overlay.py:2500`, `_meter` `:6471`), the 20-rung volume column (`:8352`), the
hold arc (`:8503`), the dial graduations (`:7763`).

**Follow the signal meter.** One segment per step, lit up to and including the current one, unlit
after, phosphor green on the same glass, reusing `_meter`'s tile-and-cache pattern (keyed on
`(lit, accent)`) so it costs one composite a frame. Composite it — never `ImageDraw` onto the
layer directly, or it punches a hole through the glass (`overlay.py:7068-7072`).

`_print` (`overlay.py:7054`) is the only thing that runs per frame over the baked bezel; the bar
goes through the same path.

**Two to ten steps.** The glass gives ~322 px, so ten segments is about 29 px each with gaps and
still reads; twenty is mush. `start_tutorial` refuses more than ten and the refusal note tells it
to group them — which is itself a good step, because a twenty-step list read out of a manual is
not a tutorial anybody can follow either.

### Row 2 keeps the precedence chain it already has

Arbitration today is a plain `or` chain at `kiosk.py:2162` plus one at `overlay.py:6964` — first
non-empty string wins the whole caption: kiosk notice (4 s) → session error → agent activity →
teardown phase → background task → resting caption. **Add the current step's label as a new
fallback just above the resting caption**, and change nothing else. So "searching for the M8
torque…" still shows mid-tutorial and the step comes back after it, and the bar never moves —
which is the whole point of giving it its own row.

### Three tools, and the rules live in what they return

The `"note"` key is already how this codebase steers the model after a call — some thirty sites in
`agent.py` (`:2146` "do not guess what is there", `:2326` "wait for them to speak", `:2543` "take
a look first, then point", `:3346` "tell them so, briefly, once"). This job's one new idea is that
**the note is composed from the step index**, so the model reads the waiting rule at the moment
waiting is what comes next, and reads nothing about tutorials on any turn that is not one.

- `start_tutorial(steps)` — `steps` is an array of 2–10 short labels, in order, each a few words
  as it will read on the glass. The description says only *when*: they asked to be walked through
  something step by step.
- `advance_tutorial()` — no parameters. The description is one line: they said the current step is
  done.
- `end_tutorial()` — no parameters. They want to stop before the end.

What comes back, and only this, carries the behaviour:

| when | the note |
|---|---|
| step 1 is up | the whole brief — say only this step, one or two sentences, and only what is not on the glass; then stop talking and wait; they will say when it is done; do not read the list out and do not say how many steps there are unless asked |
| a middle step | terse — *"Step 4 of 7 is up. Say it, then wait."* |
| the last step is up | that it is the last one |
| advanced past the end | *"That was the last step. Say so in a few words. The tutorial is over."* |
| more than ten steps | refused, with the note to group them into ten or fewer |
| advance/end with none running | *"No tutorial is running. If they want one, call `start_tutorial` with the steps."* |

**The base prompt gains nothing at all**, and the three schemas stay small. That is both what was
asked for and the budget argument: standing prompt is 9,631 tok today — instructions 4,299, tools
5,332, against a 16,384 cap (`tools/prompt_audit.py`). A block that describes tutorials is read
every session whether or not anybody asks for one.

Note for the prompt audit's "say each rule once": the waiting rule genuinely is repeated on every
advance. That is not duplication — it is one function composing one note, and the most recently
read text is what the model acts on. The rule has one place to live; it is read more than once on
purpose.

### Wiring

Follow the path `detail` already takes, which is a direct in-process call on the render thread —
no websocket, no `panel.offer_*`, none of which can reach the status line.

1. `VoiceAgent._tutorial: Tutorial | None` — a frozen dataclass (steps tuple, index), beside
   `_on_panel` and `_sketch_*` in `__init__` (`agent.py:1295-1386`). Per-session, dies with the
   session, never persisted: a tutorial you started on Tuesday must not be on the glass on
   Thursday.
2. An `agent.tutorial` property → `SessionController.status()` (`ui.py:236`) → `kiosk.py:2162` →
   a new `Overlay.render(tutorial=…)` argument.
3. `_activity_line` (`agent.py:3794`) needs a branch for each of the three — every offered tool
   must have one, and `tests/test_caption.py:79` hard-codes the offered count, which moves.
4. The record takes **three** edits or it vanishes without error: `session.note("tutorial", …)`, a
   branch in `_render_record` (`session.py:971-1012`), and an entry in `library.SPOKEN`
   (`library.py:58-72`). This is a live bug for `point` today, which renders to `""` and is
   silently dropped — do not add the fourth instance of it.
5. `_dispatch_tool` (`agent.py:2071`), the tools list (`agent.py:1546`), and `_send_tool_output` +
   `_request_response` at the end of each handler, like every other tool.

### What this does not touch

The scratchpad stays a normal tool — freed from being the step counter, it is there when a step
genuinely needs a picture or a list. No new file formats, no change to the filing agents, no
setting, no control on the panel: the bar is drawn by the kiosk, so it is in the recording and it
is not something you can press.

### The 005 collision, written down on purpose

Job 005 is in `review` with a tutorial mode built on `job/005-cyclops-should-be-able-to-switch-into-dif`:
steps on the scratchpad, a `TUTORIAL MODE IS ON` block in the prompt, a `STEP` tag in the status
pod, and `point_at` marks held 45 s. Marco's call is to build this one standalone against master
and decide 005 afterwards. **Build nothing here that depends on 005 and nothing that pre-emptively
removes it.** Whoever ships both reconciles them then — most likely by dropping the prompt block
and the scratchpad counter and keeping this bar, but that is not this job's decision to make.

## Done when

- [x] A running tutorial puts a segmented bar on the top row and the current step on the second —
      `panel_shot.py` with the new tutorial flags, at 3 of 7, looked at
- [x] The bar has one segment per step and lights up to the current one — pytest, using
      `_ink(frame, ov, 0)` (`tests/test_caption.py:218`) on row 1: 7 steps at step 3 gives 3 lit
      segments and 4 unlit, and at step 7 gives 7 lit
- [x] `advance_tutorial` moves the bar and the label together — pytest across a 7-step tutorial,
      plus renders at 1 of 7 and 7 of 7
- [x] `end_tutorial` and running off the end both clear it, and the caption goes back to wrapping
      across both rows — pytest
- [x] More than ten steps is refused with a note to group them, and nothing appears on the glass —
      pytest
- [x] Nothing about tutorials is in the base prompt, and the three schemas are small —
      `uv run python tools/prompt_audit.py`: instructions stays at today's 4,299 tok, tools rises
      by under 200
- [x] The returns do the teaching: it puts one step up and stops, never reads the list out, and
      never advances on a turn that did not say the step was done — `tools/talk_probe.py` with a
      tutorial script, 10 runs, `tools` tallied per turn and the lines read
- [x] It never starts a tutorial unasked — `start_tutorial` never called across the existing
      6-turn `SCRIPT` × 10 runs — `talk_probe.py --tally`
- [x] A long activity line still shows mid-tutorial, on the second row only, elided rather than
      eating the bar — pytest plus a render with a tutorial up and
      `--detail "editing the picture to paint the doors matt black…"`
- [x] A tutorial is in the record — it appears in `session.jsonl` **and** renders a line in
      `session.md` — pytest (the three-place trap; one place out of three is silent)
- [x] The bar costs nothing — `panel_shot.py --bench`, listening and speaking ms with a tutorial up
      and without, against master's ~9.3 ms
- [ ] `uv run pytest` green and under ten seconds
- [ ] From a pace away with your hands busy, the bar says how far through you are and the row
      under it says what you are doing, without asking and without touching anything — **yours, on
      the Pi** (try it at ten steps too, which is the readability limit)
- [ ] A real job walked through end to end: it waits after each step, "done" moves it on, "stop
      this" clears the bar, and you never reach for the glass to find out where you are — **yours,
      on the Pi**

## Built — 2026-09-18
While a walkthrough is running, the top row of the terminal is a bar with one cell per step, lit
up to and including the current step, and the row under it shows that step's label. Anything the
controller has to say still wins that row (for example "editing the picture…"), cut to one line,
and the step label comes back afterwards. The bar never moves. With no walkthrough, the panel is
byte-identical to master. `start_tutorial`, `advance_tutorial` and `end_tutorial` hold a frozen
`Tutorial` on the agent that dies with the session. The walkthrough is recorded in session.jsonl,
session.md and the companion's transcript (four places, counting app.js, not three). The base
prompt is untouched. The rules arrive in the tool replies: step 1 carries the whole brief,
including "the steps stay on screen until you call end_tutorial", and later steps get one line.
Two things were decided along the way:
- `start_tutorial`'s description names the scratchpad ("use this and not the scratchpad"),
  because the scratchpad's description and HOW YOU TALK both claim "steps of a job", and in one
  of two early runs it picked the scratchpad.
- The notes went through three rounds. Round 1 never ended the walkthrough on "stop this" (0/4).
  Round 2, which is what ships, ended it 8/10. Round 3 put the stop rule on every step: it made
  no difference, and it once skipped a step, so it was reverted.
Leftovers:
- Nearly every step opens with a line about moving on ("Nice, let me move you to the next
  step"), whatever the reply says.
- "Stop this" leaves the bar up about 1 time in 5.
- The test suite is 17–20 s, but master was already 18 s on this machine before the job started.
- The plan's 4,299-token base-prompt figure predates the prompt-audit cuts. Today's is 4,158,
  and it hasn't changed.
Hands-on: `/try 008`. Ask it to walk you through a real job step by step, say "done" a few times,
ask a question mid-step, then say "forget it, stop this". Check the bar from a pace away, and try a
job long enough to get ten steps.
factory/html/008.html
