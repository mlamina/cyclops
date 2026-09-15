---
description: Run the factory and keep it running — start the loop, watch it, and fix the pipeline when it breaks
---

Run the factory. This replaces starting `factory/loop.sh` in a terminal by hand: it brings the loop
up, watches it from the outside, and when the *pipeline* breaks it fixes the pipeline and sends the
job round again. Marco is not watching a terminal any more, so everything the terminal used to tell
him has to be earned by you or not said at all.

**You fix the factory, never the work.** Shipping, reworking, dropping, and the Pi are his word,
given through his own commands. Nothing here gives you one of them.

## Coming up

Find the loop first, and find the *real* one:

```sh
pgrep -f 'sh .*factory/loop\.sh$'    # matches the loop AND every supervisor it has backgrounded
```

A forked `sh` keeps its parent's argv, so `start()`'s supervisor subshells answer to the same
pattern — 75807 the loop and 86009 a supervisor, command lines identical, measured. A supervisor is
forked *by* the loop, so **its parent pid is one of the other matches**, and the real loop is the
match whose parent is not. Test it that way and not by reading the parent's command line: your own
Bash call carries `./factory/loop.sh` in its shell's argv, which is how the first version of this
called the real loop a supervisor the first time it ran. A surviving supervisor with no loop above
it means nothing is starting new jobs, and that must read as down, not up.

If one is running, adopt it — the signals come from the logs, so whose terminal owns it does not
matter. If none is, start it detached:

```sh
nohup ./factory/loop.sh >> factory/logs/loop.out 2>&1 < /dev/null &
```

Detached because the factory must not stop when Marco closes this session; the watch is the part
that ends with it, and the next `/run-factory` adopts the loop still going. `< /dev/null` for
`start-kiosk.sh`'s reason — a backgrounded process that reads the terminal gets SIGTTIN and stops,
and a stopped loop looks alive. (No `setsid`: this is a Mac.)

`loop.out` is now the only place the loop's own voice lands — `· factory up`, the five-minute
beats, and a `set -eu` fatal, which appears nowhere else at all.

Then arm the watcher as a **persistent** `Monitor`:

```sh
sh factory/watch.sh
```

Say one line about what is in flight, from the board — the watcher primes itself on the logs it
finds and deliberately does not replay them, so the state at arming time is yours to report, once.
Then go quiet.

## The events

Every line the watcher prints is one of these. Nothing else is.

| event | what it means |
|---|---|
| `BUILDING N <slug>` | the loop took the job and started it. Say it in one line — after a `/capture`, "it started" and "it never started" look identical for four minutes otherwise, and that is the one silence worth breaking |
| `REVIEW N` | the build set `state: review`. Marco's cue, and the only good news you pass on |
| `STALLED N <reason>` | the build ended without reaching `review`, reason as loop.sh wrote it |
| `WEDGED N <reason>` | ninety minutes silent, killed |
| `EXIT N <code>` | `claude -p` itself failed |
| `STRANDED N <pid>` | the log ends at `PID` and that process is gone — nothing wrote an ending |
| `NOT-STARTING N` | ready, never started, and there were free slots |
| `LOOP-DOWN <pid>` | the loop is gone; the watch has ended with it |

## Read it in a subagent

A stalled build's log is the model's own prose, thousands of lines, and the worktree and job file
go with it. Pulling that into this session is how a watch meant to last all afternoon dies at
teatime. So every event that needs reading gets **one throwaway subagent**: give it the event line,
`factory/logs/N.log`, the worktree at `../cyclops-jobs/N`, and the job file, and ask it for two
things — *what actually happened*, and *whose fault it is: the pipeline's or the job's*. Tell it to
change nothing. You get a verdict; its reading stays in its own context.

`BUILDING` and `REVIEW` need no subagent. Say the line and move on.

## Whose fault it is

This is the only judgement you make, and everything else follows from it.

**The pipeline's** — the loop, the plumbing, the build's own contract. Fix it, then send the job
round again.

| | |
|---|---|
| ended its turn mid-work | a `claude -p` exits the moment the model stops talking. Job 002's sixty-turn measurement died at forty-nine because the turn ended to report progress. `build.md` legislates against it; the loop cannot |
| `git could not cut the worktree` | a stale registration or a half-removed directory. `git worktree prune` first |
| stranded with no ending | a supervisor was killed — almost always the README's `pkill` |
| not starting with slots free | the loop counts builds with `pgrep -f 'claude -p /build '` (loop.sh:290), so any peer session carrying that text in its argv fills the last slot silently |
| wedged on cache churn | `last_write()` prunes `.venv`, `.git` and `__pycache__` but not `.ruff_cache` — a build stuck in a retry loop that runs a linter looks busy for ever |

**The job's** — the plan was wrong, a criterion cannot be met, the work does not fit. `build.md`
tells a build to say so in the job file and leave the state at `ready`, and that is a build doing
its job correctly, not a fault. **Do not fix it and do not send it round.** Say what it said, name
`/rework N` or `/review N`, and leave it.

**Neither, and out of your reach** — a broken `tools/` script, a missing dependency, anything under
`src/`. Report it in one line and let the pipeline wait. **A job is never re-sent into an unfixed
cause.**

## Fixing

You may change **`factory/loop.sh`** and **`.claude/commands/*.md`**. Nothing else. Not `tools/`,
not `deploy/`, not `src/` — those Marco wants to see first, however obvious the fix looks from here.

**Editing `loop.sh` means stopping it first.** `/bin/sh` reads a script in chunks as it runs, so
patching one in place can make a running loop execute nonsense. Stop, edit, restart. Stop it with
`kill <the real loop pid>` and **never** `pkill -f 'factory/loop\.sh$'` — that pattern takes the
supervisors with it, and every live build then loses the process that would have written its
ending, which is the `STRANDED` shape above. In-flight builds survive a clean stop (`nohup`, and
the supervisor traps `INT HUP`), and the restarted loop re-adopts them: a job with a log and a live
pid is counted, not restarted. Confirm the restart by seeing `· factory up` land in `loop.out`, and
re-arm the watcher, which exited when the loop went.

**Commit path-scoped, always.** `git commit -- factory/loop.sh`, never `git add -A`: the board is
dirty most of the time with Marco's own edits, and a build appends to a job file from its worktree
while you are working. One commit per fix, straight to master, the message naming the incident the
way loop.sh's own commits do.

## Sending it round again

Mechanical, and it touches nothing anyone wrote:

```sh
rm -f factory/logs/N.log factory/logs/N.state
```

The log is what re-arms the job — the loop skips any `ready` job that already has one — and the
state file goes with it because `/rework` never deletes it and a stale one swallows the next
round's first `STATE` line. Leave the job file alone: `## Feedback` is Marco's voice and `## Built`
is the builder's, and a pipeline fault belongs in neither.

**Two rounds per job, then stop and say so.** The loop refuses to retry at all, deliberately, so a
broken job cannot burn an afternoon in a loop nobody is watching. A watcher earns two rounds. It
does not earn an afternoon.

## What you say

Silence while things build. The beats are in `loop.out` if he wants them; relaying them is how this
becomes the task.

Speak for four things only: **the factory took a job** (`BUILDING`, one line), **you fixed
something** (what broke, what you changed, that it went round again — three lines), **a job needs
him** (`REVIEW`, or a job whose own plan is wrong), and **you have stopped** (the loop went, a job
used both its rounds, or the cause is out of your reach).

Never announce that you are still watching.

## Never

Merge, ship, accept, drop, or move a job to `done/`. Touch the Pi, `ssh`, deploy, or unset
`CYCLOPS_NO_PI`. Tick or reword a line in **Done when**. Edit a file outside `factory/loop.sh` and
`.claude/commands/`. Re-send a job whose cause you have not fixed.

---

`tail -f factory/logs/loop.out` is the terminal window this replaced · `/board` says what is
building · `/review N` when one lands.
