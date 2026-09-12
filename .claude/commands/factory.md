---
description: Run the factory — show the board, plan what's new, build what's ready, and say what's waiting on Marco
---

Read `factory/README.md` first; it's the contract and it's short.

## What `$ARGUMENTS` means

| Arguments | Do |
|---|---|
| *(none)* | Print the board, then move everything that can move |
| `7` | Only job 007 |
| `done 7` | Marco accepts 007 — `git mv` it to `factory/done/`, set `state: done`, stop |

## The board

Every `factory/*.md`, one line each: number, state, title. Put **what's waiting on Marco** first
and what's waiting on the factory second. If nothing is waiting on him, say so in one line.

## Moving a job

Work the queue lowest number first, and **stop building the moment one job sits in `review`** —
plan on, but don't build a second. That limit is the point of the whole thing.

### `new` → plan it

Read enough of the repo to write a plan Marco could hand to anyone. Append:

```markdown
## Plan
<what changes, which files, what it looks like when it works — a dozen lines, not a design doc>
```

Then decide, honestly: **would you be guessing at something Marco has an opinion about?** Taste,
naming, a behaviour with two defensible defaults, anything that touches the face or the voice.

- If yes, append a `## Questions` section — numbered, each followed by a blank line for his
  answer, each with your own recommendation so answering can be one word. Ask at most three; if
  you have more than three, the job is too big — say so and propose splitting it. Set `asking`.
- If no, set `ready`. Don't invent a question to be polite.

### `asking` → check for answers

Every question answered → fold the answers into the plan, clear the `## Questions` section down to
the questions and their answers, set `ready`. Any question still blank → leave it, and list it on
the board. Prose Marco added anywhere else in the file is an instruction: treat it as an answer.

### `ready` → build it

Normal work, under `CLAUDE.md`: surgical changes, follow the patterns already there, commit
straight to master, `deploy/push.sh`, tests with `uv run pytest`. If it changed anything visible,
screenshot the panel and copy the image to `~/Downloads`. Then append:

```markdown
## Built
<commit sha> — <what actually changed, and anything you decided along the way>
<screenshot path, if any>
```

Set `review`. Say the sha and the screenshot in chat too — that's Marco's cue to look.

### `review` → hands off

Untouched unless there is prose under `## Feedback`. If there is: that's another round. Fold it in,
set `ready`, and build again — appending a second `## Built`, never editing the first. The history
of a job is the job.

## When you stop

One short block: what you built, what you're waiting on, and the exact question if there is one.
Nothing else — no summary of the board you just printed.
