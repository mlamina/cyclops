---
description: Run the factory — plan what's new, build ready jobs in parallel worktrees, and show Marco what came out
---

Read `factory/README.md` first; it's the contract and it's short.

## What `$ARGUMENTS` means

| Arguments | Do |
|---|---|
| *(none)* | Show the board, plan what's `new`, fan out what's `ready` |
| `7` | Only job 007 |
| `pi 7` | Put 007's branch on the Pi so Marco can use it |
| `done 7` | Merge 007 to master, deploy master, tear its worktree down |

## Show the board first, always

Every `factory/*.md`: number, state, title. **What's waiting on Marco goes first.**

For anything in `review`, **Read its evidence images so they appear in his terminal.** That is the
review — a gallery he scrolls, not a diff he reads. Don't describe a picture you're about to show
him. If nothing is waiting on him, one line saying so.

## `new` → plan it

Read enough of the repo to write a plan anyone could hand off. Append to the job file:

```markdown
## Plan
<what changes, which files, and — first line — what it will look like when it works>
```

Then be honest about whether you'd be **guessing at something Marco has an opinion about**: taste,
naming, anything touching the face, the voice, or a value in README.md.

- Yes → append `## Questions`. Numbered, at most three, each with your own recommendation so
  answering is one word, each followed by a blank line. Set `asking`. More than three questions
  means the job is too big — say so and propose the split.
- No → set `ready`. Don't invent a question to be polite.

When answers appear, fold them into the plan and set `ready`. Prose Marco leaves anywhere in the
file is an instruction, wherever he put it.

## `ready` → build it, in its own worktree

Set up each job from the main checkout:

```sh
git worktree add -b job/007-slug ../cyclops-jobs/007 master
```

Then **launch one agent per ready job, all in a single message so they run at once.** Each agent
gets: the job file's plan verbatim, its worktree path, and this brief —

> Work only inside `<worktree>`. Never touch `factory/` — the board lives on master.
> Follow `CLAUDE.md` and `.claude/rules/`. Vibe it; Marco reviews outcomes, not code.
> Commit to your branch. `uv run pytest` must pass.
> **Then produce the evidence** (below) and report back its paths in one paragraph.

Set those jobs `building` while they run. Fan out as wide as there is work — the screenshot is
cheap to judge, so there is no reason to build one at a time.

## Evidence is the deliverable

A job isn't built until there's something to *look at*. Pick what actually fits:

| The change is | Evidence |
|---|---|
| Anything on the panel | `uv run python tools/panel_shot.py` before **and** after, same `--bg`, same `--state` |
| Motion — the eye, a transition | `--strip 8 --seconds 4`, which shows movement in a still |
| Speed, heat, CPU | The number before and the number after. Two numbers, one line |
| Voice or behaviour | The transcript of a real exchange, or say plainly it needs his hands |
| Plumbing with nothing to see | Say so, and give the check that now passes. Don't dress it up |

Images go to `~/Downloads/factory/007/` (CLAUDE.md — never leave one in a scratchpad). Then the
factory appends to the job file and sets `review`:

```markdown
## Built
<one paragraph: what it does now, and anything you decided along the way>
~/Downloads/factory/007/after.png
Hands-on: <what to try on the Pi, or "not needed">
```

Never mention files changed, lines, or approach. He doesn't want it.

## `pi 7` → his hands on it

One Pi, one job at a time — `factory/ON-THE-PI` says who has it.

```sh
cd ../cyclops-jobs/007 && deploy/push.sh && deploy/start-kiosk.sh
```

Write `007` to `factory/ON-THE-PI`, and tell him in one line what to try. When a job leaves the
Pi, push master back so the box is never left on a branch.

## `review` → hands off

Untouched unless there's prose under `## Feedback`. That's another round: fold it in, set `ready`,
build again on the same branch, append a second `## Built`. Never edit the first — the history of
a job is the job.

## `done 7` → ship it

`git merge --no-ff job/007-slug` on master, deploy master to the Pi, `git worktree remove` and
delete the branch, move the file to `factory/done/`, clear `ON-THE-PI` if it was 007.

## When you stop

What's waiting on him, and the exact question if there is one. Nothing else.
