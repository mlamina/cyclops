# The factory

You say what you want. It gets built while you do something else. You look at the result.

```
/capture ──► ready ──► (the loop builds it) ──► review ──► /ship ──► done
                ▲                                 │
                └──────────── /rework ────────────┘
```

Start `factory/loop.sh` in a terminal and leave it. It is the only thing that runs unattended,
and all it does is build. It never merges, never touches the Pi, and never decides anything.

## What you type

| | |
|---|---|
| `/capture <anything>` | idea, bug, feature request — one way in for all three |
| `/board` | what's building, what's waiting on you |
| `/review N` | look at what came out |
| `/try N` | put it on the Pi and use it |
| `/ship N` | merge to master and deploy |
| `/rework N <what's wrong>` | another round |
| `/drop N` | bin it |

Every command ends by naming the next one, so the process is something you read rather than
something you remember. Every command works at any state — you never need to know what stage
something is in to act on it.

## Capture is where the thinking happens

It explores the code, plans, asks you what it would otherwise be guessing at, and agrees **what
done means** before anything is built. That is the only conversation; the loop afterwards just
executes it. `"you decide"` is a real answer to any question it asks.

**Nothing is written until you approve it.** An abandoned capture leaves no file and no branch.

## Review is an outcome, never a diff

A picture of the panel, a number, or the Pi under your hands. No code, no file lists, no PRs —
if a job page shows you any of that, the page is wrong.

Every job's plan ends in a **Done when** list where each line names its own check: a
`panel_shot.py` strip, a `--bench` number, a test, or *"yours, on the Pi"*. The builder works them
and reports each one. Lines that were only ever yours to judge stay unjudged.

## Branches, worktrees, and the one Pi

Each job gets `job/NNN-slug` and its own worktree at `../cyclops-jobs/NNN/`, so several build at
once and dropping one costs nothing — master never knew. **The board itself lives only on master**,
in the main checkout; worktrees never touch `factory/`.

Three at a time, because judging a screenshot takes five seconds. **One on the Pi at a time**,
because there is one Pi — `factory/ON-THE-PI` says which.

## If something stops

A build that dies leaves `factory/logs/NNN.log` and the job sitting at `ready`. The loop won't
try again — that's deliberate, so a broken job can't burn an afternoon in a loop nobody is
watching. `/board` shows it as stalled and `/rework N` sends it round again.
