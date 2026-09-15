---
state: done
opened: 2026-09-15
---

# OpenAI just released new image models — use those instead of the old ones

openai just released new image models. we should use those in cyclops instead of the old ones

(From the launch mail: GPT-Image-2.5 ships as two models — **Flare**, "higher-quality images than
GPT-Image-2 at 50% lower latency", and **Sunburst**, "built for premium visual workflows that
benefit from tighter control across edits". Both also do transparent backgrounds, which Cyclops
has no use for: everything it makes is a JPEG on a lit panel.)

## Plan

**One constant changes. The work is everything downstream of the constant being three times
faster.**

`IMAGE_MODEL = "gpt-image-2"` (`imagine.py:67`) is the only image model id in the repo, and both
call sites read it — `draw()` at `quality="high"` for a picture of a physical thing, `edit()` at
`quality="low"` for a photo redrawn. It becomes **`gpt-image-2.5-sunburst`**.

### Measured, 2026-09-15, from the Mac, through `imagine.draw` and `imagine.edit` themselves

Production settings, the real `DRAW_PROMPT` and `DEFAULT_STYLE`, same request each time:

| | drawing, 1200x720 `high` | edit, 1152x640 `low` | output tokens |
|---|---|---|---|
| `gpt-image-2` — today | 76–88 s *(recorded 2026-09-04)* | 13.1 / 13.2 s *(recorded 2026-09-02)* | 7024 |
| `gpt-image-2.5-flare` | 20.5 s | 10.8 s | 1756 |
| **`gpt-image-2.5-sunburst`** | **26.0 s** | **13.1 s** | **1756** |

A drawing lands in **a third of the time and a quarter of the price**. Edits were already `low`
and already quick; they do not move, and they did not need to.

Sunburst over Flare on the picture, not the clock: asked for the same diagram, Flare padded it
with a "Colour Key (Wires)" legend that nothing had asked for, and Sunburst drew what was asked.
Both edits were indistinguishable — the tank painted, every label untouched.

**The quality ladder is not the win, so do not spend the saved time on it.** 2.5 adds `xhigh` and
`max`. Sunburst at `xhigh` took 34.4 s and at `max` 62.3 s; `xhigh` drew crisper lines and then
invented a colour key *and* misspelled a label. `DRAW_QUALITY` stays `"high"`.

### What changes

1. **`IMAGE_MODEL`** → `"gpt-image-2.5-sunburst"`, and its comment carries the table above with
   today's date, the way every constant in that file carries its own measurement.

2. **Every duration Cyclops says out loud comes down.** This is the real work and it is value 4:
   a drawing that arrives in half a minute must not be announced as a minute and a half, or the
   model sends you off to talk about something else for three times the actual wait. The six
   model-facing ones are `DRAW_DIAGRAM_TOOL`'s description (`agent.py:199`), the system prompt
   (`agent.py:928`, `agent.py:1002`), the sentence `_run_draw_diagram` hands back
   (`agent.py:2514`), and the edit's pair (`agent.py:329`, `agent.py:3066`). A drawing becomes
   *about half a minute*; an edit's "about half a minute" is still true and stays.

3. **The stale prose goes with it.** About thirty comments and test docstrings across eleven files
   argue from "eighty seconds" / "ninety seconds" / "a minute and a half" — `imagine.py:40,124,400`,
   `agent.py:2483,2487,2522,2965,2990,3005,1274,1499,368`, `panel.py:92,153,159`, `overlay.py:172`,
   `tasks.py:4`, `smoke.py:32,179`, `docs/tools.md:40,49,89,120,163`, and the docstrings in
   `test_imagine.py:123,133,430`, `test_eye.py:257`, `test_panel_swap.py:101`,
   `test_scratchpad.py:76`. Fix all of them. **Not** `agent.py:61`, `agent.py:1767` or
   `search.py:17` — those are about a fault appearing, a video ending and a web search, and have
   nothing to do with an image call. `panel.py:153` also names `gpt-image-2` by hand.

4. **`_draw` gets the duration line `_edit` already has.** `agent.py:3093` logs
   `[tool] edit: N KB in X.Xs`; `_draw` (`agent.py:2521-2577`) times nothing, so the one number
   that says whether this job worked is invisible on the box. Mirror it.

### What this does not touch

`DRAW_QUALITY` and `QUALITY`. `PANEL_SIZE` — 1200x720 is still 16-divisible, still 5:3, and still
clears 2.5's documented 655,360-pixel floor. `PIXEL_BUDGET`, `BUDGET_GROWTH` and the one retry in
`edit()` — the floor is documented now but the retry exists because OpenAI can move it, and that
is still true. `OUTPUT_FORMAT = "jpeg"`. `EDIT_TIMEOUT_S` / `DRAW_TIMEOUT_S` / `DRAW_DEADLINE_S` —
generous ceilings cost nothing. The tool rules: `draw_diagram` is still only for a picture of a
PHYSICAL thing, `sketch` still has the structural work, and a drawing is still **drawn, not
checked** — a faster model is not a checked one. And `input_fidelity` is still **not** passed:
verified today against Sunburst, which returns *"does not support the 'input_fidelity'
parameter"*, so `imagine.py:302-303` stands as written.

## Done when

- [x] `IMAGE_MODEL` is `gpt-image-2.5-sunburst` and its comment carries the 2026-09-15 numbers —
      the constant block in the diff
- [x] Nothing Cyclops reads or says still promises a minute and a half —
      `grep -rn "minute and a half\|ninety seconds\|eighty seconds" src/ docs/ tests/`, and every
      line left is about a fault, a video or a web search
- [x] No `gpt-image-2` left outside `Plans/` — `grep -rn "gpt-image-2\b" src/ docs/ tests/`
- [x] Suite green and still under ten seconds — `uv run pytest`
- [ ] The journal says how long a drawing took, the way it already does for an edit — **yours, on
      the Pi**
- [ ] A drawing of a physical thing lands in about half a minute, not a minute and a half — **yours,
      on the Pi** (the new log line gives the number)
- [ ] What it tells you the wait will be is what you actually wait — **yours, on the Pi**
- [ ] The picture is at least as good as today's, from a pace away — **yours, on the Pi**
- [ ] A redrawn photo still changes what you asked and nothing else — **yours, on the Pi**

## Built — 2026-09-15

`IMAGE_MODEL` is `gpt-image-2.5-sunburst`, and the constant carries the comparison table dated
today. Everything downstream came with it: six model-facing sentences that promised a minute and
a half now say half a minute — `draw_diagram`, two system-prompt lines, the note `_run_draw_diagram`
hands back, and the two places where `scratchpad` and `sketch` price a drawing against themselves.
The edit's "about half a minute" was already true and is untouched. The stale prose in `imagine`,
`agent`, `panel`, `overlay`, `tasks`, `smoke`, `docs/tools.md` and five test docstrings is gone;
`DRAW_QUALITY` keeps its 2026-09-04 measurement, now dated to the model it was taken against, with
`xhigh` and `max` named and refused beside it. `_draw` now logs `[tool] draw: N KB in X.Xs`.

Two decisions on the way. The plan named six model-facing durations; I changed eight, because the
`scratchpad` and `sketch` descriptions both price `draw_diagram` at "a minute" to argue you should
use them instead, and leaving those would have been the same lie in the same place. And I ran the
real thing rather than trusting the plan's table: one `imagine.draw` and one `imagine.edit` from
the Mac at production settings, plus the same two through `gpt-image-2` for a side-by-side.
23.3 s against 70.9 s on the same relay diagram, and the new one numbers every pin on the header
while the old one left a wire running off the bottom of the page. That also proves the id is a
live model, which was the one way this job could have failed silently and only shown up on the
glass. Pictures are on the page.

Two notes on the criteria. "No `gpt-image-2` left" — three mentions survive, all three inside the
measurement comments that record what it used to cost, which is what the plan asked those comments
to carry; nothing reaches the API by that name. (The criterion's grep as written also matches
`gpt-image-2.5-sunburst`, so it could never have come back empty.) And "still under ten seconds" —
the suite is 982 passed in 16.9 s, and it was 18.0 s on the commit before this one, measured the
same way in a scratch worktree. The budget was already spent; this job did not move it.

Hands-on: `/try 006`, then ask for a wiring diagram. Three things to watch — that it tells you
about half a minute rather than a minute and a half, that the wait matches what it said, and that
`[tool] draw: N KB in X.Xs` shows up in the journal beside the `edit:` line with a 2 at the front.
Then take a photo and change one thing in it, which should be as it was.

factory/html/006.html
