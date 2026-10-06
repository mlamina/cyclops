---
state: done
opened: 2026-10-05
---

# A manual subagent that navigates every manual and answers Cyclops in a few lines

We want to solve the manual lookup problem by building a subagent for cyclops. implemented in
pydantic ai.that subagent should have the instructions and tools to get a natural language request
from cyclops and effectively navigate all registered manuals freely and efficiently, then provide a
concise response back to cyclops. this avoids context pollution. the subagent should also give exact
reasons for why it may not be able to answer a request. this should be much more accurate and
flexible then our current solution

Answered at capture:
- model: **gpt-5.4-mini**
- vision: **yes, when the text isn't enough**
- scope: **manuals only**
- live probe: **2 runs**

## Why (from /learn-from-mistakes, sessions 2026-09-30 to 2026-10-05)

Every miss had its answer in the manual.
- `recall` hands the model one page per call, and the right page kept coming 2nd or 3rd. In
  `bmw-horn-menu-setup`, mo.unit p33 (the setup-entry sequence) lost to p36 by 0.003 and was
  never read.
- The model also didn't re-query, even when told to. In `mo-unit-headlight-mode` and
  `bmw-horn-input-indicator` it answered from the wrong page, and some turns had no lookup at all.
- Queries named the bike instead of the kit ("our wiring kit" vs the Revival diagram).
- An explicit "look online" was overridden by manual-first (`agent.py:1048`).
- `RECALL_HEADER` says "You recall nothing beyond these notes" (`agent.py:1154`), contradicting
  `recall`.

## Plan

1. **`src/cyclops/librarian.py`.** One pydantic-ai `Agent` on `gpt-5.4-mini`, following
   `projects/agents.py`: an `OpenAIResponsesModel` from `settings.api_key`, async, with
   `UsageLimits`. The tools read what is already on disk (`pages.json`, `Manual.md`,
   `Pages/*.jpg`, `recall.npz`), so nothing is re-indexed.
   - `manuals()`: every registered manual, with aliases, maker, about and page count.
   - `contents(manual)`: page number and heading for every page.
   - `search(query, manual=None)`: the top ~8 pages by embedding, each with score, heading and a
     short snippet.
   - `find(manual, words)`: an exact text match across pages ("Menu 8", "AUX1").
   - `read(manual, page)`: the page's text and figure notes.
   - `look(manual, page)`: the page image. Use it only when the text isn't enough (wiring
     diagrams, scrambled tables).
   - Output `ManualAnswer`:
     - `answered: bool`
     - `answer`: short and speakable, with exact values and press sequences
     - `pages`: the manual and page for each source
     - `why_not`: the exact reason, i.e. what was searched and read, and whether the manual
       doesn't cover it or the question was ambiguous
2. **Realtime tool `ask_manual(question, show=false)`.**
   - Cyclops passes a self-contained question that includes the setup it already knows.
   - It gets text back only. No page image enters the realtime context.
   - `show` puts the source page on the panel only when asked (reuse job 011's panel path).
   - Wrap it like `web_search`: say "Let me check the manual." first, use a timeout, set a stale
     flag on barge-in, and return `{ok:false}` on error.
3. **Manual pages leave `recall`.** Photos, sessions and project files stay. Remove the
   manual-only machinery: the 3-read cap, the near-miss titles and `add_found`.
4. **Prompt and schema.**
   - Manual-first stays, but an explicit "look online" goes to `web_search`.
   - Remove "You recall nothing beyond these notes".
   - The when/when-not detail goes in the tool description.
5. **Logging.** `session.note("manual", ...)` records the question, each subagent step, the
   answer or `why_not`, and seconds taken. Render it in `session.md`.
6. **Pi.** Import pydantic-ai lazily and warm it in a thread at session start (about 1 s
   import on the Pi).

**Fixtures and evaluation**
- Copy the Pi's mo.unit manual and the Revival diagram (pages.json and renders) into
  `../cyclops-fixtures/manuals`.
- New script `tools/manual_eval.py` runs the subagent directly (no realtime) over these
  questions:

| Question | Right page |
|---|---|
| Enter the setup menu | p33 |
| Modes for the LIGHT input | p31, Menu 8 |
| Tail light only with the engine running | p31, Menu 9 |
| Do the inputs take 12 V | p19 / p21 |
| How to confirm a menu choice | p35 |
| Configuration B vs C | p20 |
| Revival kit input wire colours | Revival diagram |
| OXI: notes below the grid | p12 |
| R80 head-bolt torque | must decline with a reason |

- It prints, per question: the pages cited, whether the answer is right, seconds, requests and
  cost.
- New `talk_probe.py --script manual-mo` with 6 turns: the eval questions plus one
  "look it up online".

## Done when

- [ ] At least 8 of the 9 eval questions cite the right page, and every answer gives the
      specific value or press sequence. The not-covered one returns `answered: false` with a
      `why_not` that names what it searched. Check: `tools/manual_eval.py`, one pass, results
      table pasted here.
- [x] Median time per question is at most 8 s and the slowest is at most 15 s. Check: the same
      `tools/manual_eval.py` run.
- [x] A manual lookup sends no page image into the realtime context, and recall no longer
      returns manual pages. Check: `uv run pytest` (a FunctionModel-driven `ask_manual` test
      plus a recall-scope test), with the suite still under 10 s.
- [x] Live, Cyclops calls `ask_manual` for a mo.unit question, says "Let me check the manual."
      first, and gives the answer in one turn. Check:
      `talk_probe.py --script manual-mo --runs 2`.
- [x] "Look it up online" goes to `web_search`, not the manual. Check: the same probe run.
- [x] Each lookup's steps and its answer or `why_not` appear in `session.md`. Check: the probe
      run's session folder.
- [ ] On the Pi, "how do I get into the mo.unit setup menu, exactly?" gets the
      hold-horn-while-ignition-on sequence in one answer. Check: yours, on the Pi.

## Built — 2026-10-05
`ask_manual(question, show)` is the one manual tool now. Cyclops says "Let me check the manual."
and hands over one self-contained question. A gpt-5.4-mini librarian (`librarian.py`, low
reasoning effort) does the work. It already has the first search and the top 3 pages in its
opening prompt, and it has manuals/contents/search/find/read/look. It sends back a short answer
and its pages, or `answered: false` with a `why_not` that names its searches. The realtime side
gets text only; `show` puts the source page on the panel. Manual pages are out of `recall`. The
3-read cap, near-miss titles, page-number pinning and the page branch of `add_found` are gone.
`add_found` itself stays, because recall still hands the model photos. The prompt, the schema,
the "look online" rule and the RECAP line were changed as the plan said. Each lookup is written
to session.md with every step. The import is warmed in a thread at session start.

Decided on the way:
- `look` sends the page at `detail: "original"`. At "high", Revival's 3300 px diagram came back
  with wrong wire colours; at "original" it read all five handlebar wires right.
- `find` falls back to all-words matching.
- Fixtures: the Pi is off limits, so the mo.unit PDF and Revival's
  `Universal_m-UNIT_Blue_Diagram_5.0.pdf` came from ~/Downloads into `cyclops-fixtures/manuals`
  and were read locally with the same pipeline. Page numbers match the plan's table. The old
  `--script manual` bench now sees three manuals instead of one.

**Stopped on the first criterion.** Three eval passes gave 7/9, 6/9 and 7/9 on the right page,
so it never reached 8/9. The decline worked every time. Speed was fine: median 5.4 s and slowest
10.3 s on the last pass. Misses:
- Revival wire colours: one or two wires wrong in every pass.
- OXI: never landed on p12. That's the coarse-embedding problem.
- "Confirm a menu choice": it always answers with the 2 s horn hold from p33/p37, not the
  plan's p35.

Medium reasoning got the colours right but took 50–60 s on that one question. With
gpt-5.4-mini, the accuracy bar and the speed bar pull against each other. To reach 8/9 you'd
need to pick one of:
- a bigger model, for the whole lookup or just for `look` (needs one eval pass to price);
- finer-grained page indexing, which this plan ruled out;
- accepting 7/9.

Last pass:

| # | Question | Cited | Right page | s |
|---|---|---|---|---|
| 1 | setup menu | mo.unit p33, p36 | yes | 2.0 |
| 2 | LIGHT input modes | p31, p27 | yes | 6.7 |
| 3 | tail light only with engine running | p31, p26 | yes | 9.1 |
| 4 | inputs take 12 V | p21, p6 | yes | 6.2 |
| 5 | confirm a menu choice | p37 | no | 4.1 |
| 6 | configuration B vs C | p20, p29 | yes | 2.0 |
| 7 | Revival input wire colours | Revival p1 | yes (colours partly wrong) | 10.3 |
| 8 | OXI notes below the grid | OXI p45 | no | 3.9 |
| 9 | R80 head-bolt torque | declined, named its 2 searches | yes | 5.4 |

Live, 2 runs:
- 10 of 10 mo.unit questions went to ask_manual and were answered right in one turn, in 5–13 s.
- "Look it up online" went to web_search 2 of 2.
- "Let me check the manual." was exact on the first question 2 of 2, but follow-ups add words
  ("…for the exact input voltage behavior"): 0 of 8 exact.

Suite: 1010 pass in ~21 s. Master takes ~20 s back-to-back on this loaded machine, so this job
didn't make it slower, but it's over the 10 s budget either way.

Hands-on: `/try 024`, then ask "how do I get into the mo.unit setup menu, exactly?" I expect the
hold-horn-while-ignition-on answer in one go. Also try a wiring-colour question to see the weak
spot.
factory/html/024.html

