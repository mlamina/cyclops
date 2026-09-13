---
description: Build one factory job in its worktree and leave an outcome to look at — the loop's command, not usually yours
argument-hint: <job number>
---

Build job `$ARGUMENTS`. You are almost certainly running headless, detached, with nobody watching,
in that job's worktree. Work like it.

## Where things are

You are **in the worktree**. Build here, commit here, on this branch.

The board is not here. It lives only in the main checkout:

```sh
MAIN=$(dirname "$(git rev-parse --git-common-dir)")   # the real repo, from any worktree
```

The job file is `$MAIN/factory/NNN-*.md` and the page you write is `$MAIN/factory/html/NNN.html`.
**Never edit the worktree's copy of `factory/`, and never commit anything under it** — the board
belongs to master, and a page written in here dies with the worktree.

## Build it

The plan and the criteria are in the job file. They were agreed with Marco. **Follow them; don't
re-explore and don't redesign.** If the plan turns out to be wrong — the code doesn't look the way
it was described, a criterion can't be met — **stop, say so in the job file, and leave the state
at `ready`**. A wrong plan quietly replaced with a better one, in a worktree nobody is watching,
is the worst thing this command can do.

Otherwise it's normal work under `CLAUDE.md` and `.claude/rules/`: surgical changes, follow the
patterns already there, `uv run pytest` passes. Vibe the code — nobody is going to read it.

## The Pi is not yours

**Build as though there is no Pi, because there may not be.** Never deploy, never `ssh`, never
`rsync`, never `ping` it, and above all **never wait for it**. `deploy/push.sh` refuses outright
while you are a build, and that is deliberate rather than a fault to work around.

Three reasons, all of them already paid for: there is one Pi and up to three of you, so two builds
pushing overwrite each other's panel mid-measurement; a Pi asleep, on battery or off the network
turns finished work into half an hour of pinging; and the glass is where Marco looks, so what is on
it should be what he last asked for, not whatever a background job pushed while he was elsewhere.

So every criterion that needs the real glass — a photo of the panel, a `grim` burst, a finger on it,
*"can I see it from a pace away"* — **is not yours to tick.** Leave it unticked, say it needs `/try`,
and say what you expect to happen. Do not substitute an argument from the code for a measurement on
the glass, and do not tick a line because the reasoning is sound: an unticked line with a sentence
under it is worth more than a tick he has to distrust.

## Then prove it

Work every criterion in **Done when** that can be worked here, and get a real result for each. The
evidence lives in `$MAIN/factory/html/NNN/`, and everything you have is headless:
`tools/panel_shot.py` (`--strip N` for motion, `--bench` for render cost, `--point` for gestures),
`tools/eye_sheet.py`, `tools/iris_strip.py`, `tools/corner_sheet.py`, and `captures/` for a real
photo behind the chrome. Before **and** after, same background, same state — that is the whole
reason `panel_shot.py` exists.

A criterion that was only ever Marco's to judge stays unjudged. Say so; don't grade it yourself.

**Your turn ending is the build ending.** You are running under `claude -p`, which exits the moment
you end your turn and takes everything you started with it — a background probe, a measurement
half-run, a test still going. There is no checking back, because there is no later. A turn that
ends *"I'll tally it the moment it completes"* is a build that completed nothing, and that has
already happened here: a sixty-turn measurement died at forty-nine because the turn that started it
ended to say how it was going. So never end a turn while work you started is still running. Wait
for it.

Waiting is free, but silence is not, and it is no longer private: every five minutes the loop
writes down whether you are working or waiting, and when you are quiet it names the command you are
sitting on. At ninety minutes of writing nothing it kills you. So if you are waiting on something
long, write as it goes — results into `$MAIN/factory/html/NNN/` line by line rather than all at the
end — and the wait takes care of itself. And never wait on something that may never arrive: a
retry loop around a machine that is not there is not work, it is the build failing slowly.

If the work genuinely cannot fit, that is a wrong plan: say so in the job file and leave the state
at `ready`.

## Leave two things

**`$MAIN/factory/html/NNN.html`** — one page, written by you, shaped to this job. A picture and a
checklist, not a document. Marco opens it, looks, and decides. **No code, no diff, no file list, no
line counts** — he does not review code and putting it in front of him is a failure of the page.
Plain `<img src="NNN/after.png">` beside it is fine; it opens off disk.

**The job file**, `$MAIN/factory/NNN-*.md` — **append**, never rewrite. He may have typed in it
while you were working and rewriting it whole would eat that. Tick the criteria you met, add:

```markdown
## Built — <date>
<a paragraph: what it does now, and anything you decided on the way>
Hands-on: <what to try on the Pi, or "not needed">
factory/html/NNN.html
```

Then change the one line `state: ready` to `state: review`. That is the signal Marco is waiting
for, and it goes last — after the page exists, after the criteria are ticked.
