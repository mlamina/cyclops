---
state: dropped
opened: 2026-09-14
---

# Cyclops should switch into operating modes — tutorial, documentation, on command

Cyclops should be able to switch into different operating modes. the default one is fairly
generic, but i want specific modes for different use case scenarios. one mode is "tutorial mode",
where cyclops guides me through something i'm working on step by step using on-screen instructions
and its "pointer" feature. another mode would be "documentation mode", where it helps me document
the work that I do. it should be able to switch between modes on command.

## Plan

**A mode is who Cyclops is being for a while. Nothing else about the box changes.**

The reason this works at all is one line: `HOW YOU TALK` opens with *"They set the agenda, always.
Go where they go. Never steer them somewhere else, and never hand them a plan they did not ask
for."* (`agent.py:879`). That line is why the default feels generic, and it was earned. A mode is
the person handing the agenda over on purpose — so tutorial mode may hand you a plan, and it may
only because you asked for one. Build the modes as a suspension of that rule, not as a pile of
new capabilities.

### The mechanism

New module `src/cyclops/modes.py` — one job: name the modes and hold what each adds to the prompt.
A mode is a name, the words that mean it, and a block of instructions. **Default is the absence of
a mode**: no block, exactly today's prompt, byte for byte. Hard-code the two in Python; no YAML, no
user-definable modes, nothing to tend.

- `build_instructions(settings, mode)` (`agent.py:1195`) appends the mode's block **last**, after
  `WHEN IT IS`. The end of the prompt is what it has just read — that is the lesson `002` paid four
  probes for — and a mode is the strongest override in the file.
- `_mode` is per-session state on `VoiceAgent` (beside `_on_panel` et al, `agent.py:1294-1386`),
  **not** a `Settings` field: `Settings` is frozen and settled once per session (`ui.py:373`), and a
  mode changes mid-sentence.
- The switch is **`session.update` carrying `instructions` only** — never the whole
  `session_config()`. A voice cannot be changed once audio has been emitted (`voice.py:15-20`), and
  re-sending the full config puts `audio.output.voice` back on the wire. There is exactly one
  `session.update` in the tree today (`agent.py:1588`); this is the second.
- **The tool list does not change with the mode.** All 13 tools stay in every mode. A mode that
  quietly took a capability away is the bad kind of surprise, and there is no budget reason for it:
  instructions are 4,615 of the 16,384-token cap today, tools another 5,353.
- One new tool, `set_mode`. The when/when-not lives in the **schema**, not the system prompt.
  Leaving a mode is the same call. It must never be called unless they asked — see *Decided* below.
- `_activity_line` (`agent.py:3699`) needs a line for it: `tests/test_caption.py:79` asserts every
  offered tool has one, and hard-codes `len(offered) == 9`. That count moves.
- `session.note("mode", …)` on every switch — **plus** a `_render_record` branch
  (`session.py:971-1011`) **plus** an entry in `library.SPOKEN` (`library.py:58-72`). There is a
  live trap: `point` writes a note that renders to `""` and is silently dropped
  (`session.py:948-950`), and `sketch` has the mirror-image bug. A note kind added in one place out
  of three vanishes without error.

### What you see

**A mode tag in the status pod, built like the `REC` tag** (`_rec_tile` `overlay.py:6635`,
`_draw_readouts` `:6883`). On for the duration of the mode, nothing there when there is none — the
same grammar REC already uses. Kiosk-drawn, so it is in the recording, it reads from a pace away,
and it is not a control.

**Not the controls column.** It is full and the failure is silent: `space-between` has no negative
space to give back and a fifth row is drawn under the close bar with nothing said
(`static/system.css:29-51`, `dashboard.html:180-186`; `tests/layout_check.mjs` guards the screen).
And not a control in any case — the mode is switched by voice.

### Tutorial mode

- One step at a time, and then it stops talking. It does not read the list out. Where you are —
  step 3 of 7 — is on the glass, so you never have to ask.
- The step goes on the **scratchpad**, which is already built for exactly this down to the
  typography: the tool says *"Steps are an `<ol>` of short `<li>`"* (`agent.py:259-312`) and the
  sheet forces numbered lists left because *"ragged-centre numbering is unreadable at arm's
  length"* (`admin/static/panel.js:88-119`).
- It waits. After a step is up it holds position until they say it is done — that is the whole
  behavioural difference from the default, which answers and moves on.
- **`point_at` has to actually work, and today it silently does not.** While a page owns the panel
  the render loop draws nothing — *"nothing we draw now can be seen by anyone"* (`kiosk.py:1997`) —
  and `_run_point` never withdraws it. So a mark placed while a step is up is drawn under a window
  nobody can see, the gesture expires unseen (`kiosk.py:2097` sits below that `continue`), and
  Cyclops says *"that one"* over a screen showing something else. A scratchpad holds the glass for
  up to `ADMIN_MAX_S` = 15 minutes, so in a tutorial that is **every mark**. Fix: `_run_point`
  withdraws the panel before showing a gesture. **This is a bug on master, not a mode feature** —
  fix it as one, and it is worth fixing whatever happens to the rest of this job.
- **A mark holds as long as the step does.** `Gesture.hold_s` is already a parameter
  (`point.py:83`, default `HOLD_S = 3.0`). Three seconds is a gesture made while talking; a
  tutorial mark is a thing you work next to.
- Manuals come free: the default already says look in a manual first, with `recall`, unasked
  (`agent.py:977-981`). A tutorial for a thing we hold the manual for should come off the manual.

### Documentation mode

The record already gets written, and the pipeline behind it is large: one dated entry per session
in `Log.md`, a project README rendered **by code** under `## Where it stands` / `## Still open` /
`## Decided` / `## Details` (`projects/store.py:798-834`, fields at `projects/models.py:59-72`), and
every value in `Project Data.xlsx`. What none of it can do is **invent what was never said**. A
torque you never spoke is in no summary. (Worth knowing: `SessionDigest`
(`projects/models.py:15-32`) already carries `facts`, `decisions`, `open_threads` for every filed
session and is deliberately thrown away, because the filing agents cannot write — `agents.py:16-19`.
Do not go and change that.)

So documentation mode is **Cyclops making the record complete at the source**, while your hands are
in it: the number, the reason, the shot, the thing that was tried and abandoned. It asks for what
the record needs and cannot get afterwards, it writes values down with `save_data` as they land,
and when asked where things stand it reads back what it has.

This is the one real risk in the job and the criteria are written to catch it. A droid asking for a
part number while you are holding the part is precisely the nag *"It never becomes the task"*
forbids. Every question must be one only they can answer and only now. Note that this mode has to
license asking at all, against a base that says *"Ask at most once a session"* and *"never nag"*
(`agent.py:1027-1029`, `:966-967`) — expect that seam to be where it misbehaves, and measure it
rather than trusting the prose.

### Decided with Marco, not to be relitigated

- **A mode resets every wake.** No note file, no persistence. You cannot be surprised by a mode you
  set on Tuesday and forgot; the button is the reset. Do not copy the voice/record-source note-file
  pattern here.
- **It only ever enters a mode when asked.** It does not switch on its own and it does not offer.
  Entering tutorial mode unasked is literally handing them a plan they did not ask for, and an
  offer would reopen the offer-shaped ending that `002` spent nine probes driving to zero.
- **Documentation mode writes no new file.** It completes the existing record. `summary.md`'s shape
  is one heading plus one paragraph and is parsed in three places (`slug.py:118`, `session.py:608`,
  `projects/deps.py:88`); changing it is a different job.

### What this does not touch

Default mode's voice — `002` tuned that across five rounds and nothing here reopens it. No new file
formats. No change to the filing agents. No fifth row in the controls column.

## Done when

- [x] "Walk me through this" switches it and "back to normal" switches back — `set_mode` fires on
      the asking turn and the mode's block is in the prompt afterwards — `tools/talk_probe.py` with
      a tutorial script, 10 runs (`tools` is already recorded per turn)
- [x] It never enters a mode unasked — `set_mode` is never called across the existing 6-turn
      `SCRIPT` × 10 runs — `talk_probe.py --tally`
- [x] A switch changes the instructions and nothing else — the update payload carries `instructions`
      and no `audio` key — pytest
- [x] A switch is in the record — it appears in `session.jsonl` **and** renders a line in
      `session.md` — pytest (this is the `point`-note trap; one place out of three is silent)
- [x] Every mode's standing prompt stays inside the cap, with the number printed for each —
      `tools/prompt_audit.py` per mode; 4,615 tok is today's baseline
- [x] A mark placed while a step is on the glass is visible — the panel is withdrawn first and the
      gesture is still live when the render loop next draws — pytest, plus a
      `panel_shot.py --point` render in the evidence folder
- [x] A tutorial mark outlives a step — at `--point-age 8` it still draws, where a default gesture
      is gone by 3.6 s — `panel_shot.py`
- [x] The mode tag is on the panel and costs nothing — a render with it up, and `--bench` before and
      after: today is render 11.9 ms / composite 2.5 ms listening
- [x] Tutorial mode holds position — across 10 runs of a tutorial script it puts one step up and
      stops, and never advances on a turn that did not say the step was done — `talk_probe.py`
- [x] Documentation mode gets more of the work written down — on one script carrying four spoken
      values and a decision, `save_data` calls rise against default mode on the identical script —
      `talk_probe.py`
- [ ] …and it does not become a form — questions asked per turn stay at or below default mode's on
      that same script, and no turn asks for something it was just told — `talk_probe.py`, lines read
- [ ] A real job walked through end to end: the step reads from a pace away, the mark lands on the
      thing, and you never touch the glass to find out where you are — **yours, on the Pi**
- [ ] Documentation mode through a real session: it asks for what you would have wanted written
      down and nothing else, and next week's `Log.md` entry is better for it — **yours, on the Pi**

## Built — 2026-09-14

It has two modes now and it only ever enters one when you ask for it. "Walk me through this"
puts one step on the scratchpad — "Step 1 of 7" and a short list — says the one thing that is
not on the glass, and then stops talking until you say the step is done; "help me write this up"
has it asking for what only you can tell it and writing values down with `save_data` as they
land; "back to normal" is the same call, and so is pressing the button. A mode is one block on
the end of the system prompt and nothing else: every tool stays offered, nothing new is written,
and no mode survives a wake. The panel carries a tag — STEP or DOCS, green, beside REC — for as
long as one is on, and nothing when there is none.

Measured against the live model, not argued from the code. Ten runs of a tutorial script: it
switched on the asking turn and back on "back to normal" in all ten, put up three steps a run,
and never advanced on "hang on" or on a question. Ten runs of the old six-turn script: `set_mode`
never called once. Five runs of a documentation script with and without the mode, identical
either way: 5.0 values a run against 3.4, every one carrying where it came from, and the
decision-and-its-reason written down in five runs of five where normally it was written down in
none.

Two things decided on the way. The tag is four letters (STEP, DOCS) because every tag on the pod
is drawn to one width and REC is three — a longer word would have widened the record light and
the heat lamp with it. And it announced a switch twice in ten runs of ten, once before the call
and once after, because a tool call makes no sound of its own; the half after the call is now
told to get on with the turn instead. What is left is a short clause on the leaving turn.

The one criterion I have left unticked that could be worked here is the form one: documentation
mode asked 2 questions in 35 turns where the default asked none, so on the letter of the line it
is over. Both are in one run and both are about units — the first fair ("is that mm, or did you
mean thou by 'mil'?"), the second asking again one turn later for something it had just been
told. That second one is the failure the line was written to catch, at one turn in 35. Your call
whether a mode that exists to ask is allowed to be over a ceiling of zero.

Also fixed, as a bug rather than a feature: `point_at` drew its mark behind whatever page owned
the panel, where the kiosk draws nothing and the gesture expired unseen. It asks for the glass
back first now — only when something of its own is on it, so a bare panel is left alone — and in
tutorial mode a mark holds 45 seconds rather than 3.

Hands-on: `/try 005`. Ask it to walk you through something real. Watch whether the step reads
from a pace away with your hands busy, whether the silence after a step feels like waiting or
like it stopped working, and whether a mark still looks right after the panel comes back from the
scratchpad to show it — expect about half a second of handover first. Then a session in
documentation mode, and see next week whether the `Log.md` entry is better for it.

factory/html/005.html
