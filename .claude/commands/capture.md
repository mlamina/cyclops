---
description: Capture an idea, a bug or a feature request — explore it, plan it, agree what "done" means, and put it on the factory board
argument-hint: <the idea, bug or feature request>
---

`$ARGUMENTS` is the thing. This is the only place in the factory where anything is *decided* —
the loop that follows only builds — so the plan and the criteria you agree here are the whole
contract. Take the time; Marco is holding the context that made him capture it, and that is the
cheapest moment this will ever be discussed.

## Explore first, as much as it warrants

Use Explore subagents, in parallel, in one message. How many, and on what, is yours to judge:
none for a one-line fix in a file he named, several for anything touching the face, the session
loop or the card. You cannot ask a good question about code you have not read — exploring is what
turns *"how should this work"* into *"these two specific things are a coin-flip."*

## Then propose, and stop

Three parts, in this order:

**The plan.** What changes and what it will look like when it works. Read `README.md` for the
values it has to keep.

**The questions**, if you have any. Only things Marco has an opinion about and you would otherwise
be guessing at. Each with your own recommendation, so answering is one word. None is a fine number.

**Done when** — the acceptance criteria, and **every line names its own check**:

- `the eye's position differs across all 8 frames` → `panel_shot.py --strip 8`
- `render stays under 12 ms a frame` → `panel_shot.py --bench`, the number before and after
- `a wedged camera recovers without an ssh` → yours, on the Pi

**A line whose check needs the Pi is written as *"yours, on the Pi"*, every time.** The builder
has no Pi — it cannot deploy, photograph or touch the glass — so a criterion that needs one and does
not say so is a line nobody can ever tick: the builder either leaves it hanging or, worse, argues it
from the code. Anything involving a photo of the panel, a `grim` burst, a finger on it, the camera,
the mic, the amp or "from a pace away" is his.

If nothing can check a line, say so now — while he can still change what he asked for — rather
than leaving the builder to invent something that merely looks convincing. Vague is the failure
mode: *"the eye looks better"* is not a criterion.

**Write nothing yet.** An abandoned capture must leave no file, no branch and nothing for the loop
to find.

## On his word

`"you decide"` is a real answer, not a shrug: take your own recommendation on every open question
and file it.

Write `factory/NNN-slug.md`, where NNN is the next free number across `factory/` **and**
`factory/done/` — numbers are never reused — and the slug is a few words from the request:

```markdown
---
state: draft
opened: <YYYY-MM-DD>
---

# <one line, in his words>

<what he asked for, verbatim>

## Plan
<the plan as agreed>

## Done when
- [ ] <criterion> — <its check>
```

**`draft`, not `ready`.** A captured job sits until he says go, and that is the whole difference
between a board and a queue. The loop only ever takes `ready`, so nothing happens to a draft — he
can read it again, change the plan, change what *done* means, or leave it a week.

The cost of getting this wrong is measured: job 015 was captured at `ready`, the loop took it inside
thirty seconds, and the `draft` he set two and a half minutes later did nothing at all — the build
was already eighteen minutes into a plan he was still writing.

Then one line: the number, the title, and that it is a draft — **`state: ready` in the job file is
what sets it building**. Nothing else — no recap of the plan he just read.
