---
description: What the factory is doing and what's waiting on you
---

Read the frontmatter of every `factory/[0-9]*.md`, and for each one the **markers** in
`factory/logs/NNN.log` — the lines the loop writes about that build: `START`, `PID`, `WORKING`,
`WAITING`, `STATE`, `EXIT`, `DONE`, `STOPPED`, `STALLED`, `WEDGED`. The loop writes them as they happen — a
status line every five minutes, and every state change — so the log is what knows how a build is
going. Nothing here has to be inferred.

**Never `pgrep` for a build.** `pgrep -f "claude -p /build NNN"` matches the shell you are running
it from, because Claude Code puts your command text in its own argv — it will tell you a build is
alive when you are the only thing alive. It reported a dead job as building for twenty minutes.
Liveness is `kill -0 <the pid in the log>`, which cannot match the process asking.

**Waiting on Marco first** — that's the only part he has to act on.

| what you find | what it is |
|---|---|
| `state: review` | **waiting on him** — the number, the title, and `/review N` |
| `state: stopped` | **waiting on him** — the build says this plan cannot meet a criterion. The number, the title, which criterion, and `/review N` |
| no log | queued; the loop takes it within half a minute |
| last marker `PID n`, and `kill -0 n` succeeds | **building** — say how long since `START` |
| the last `WORKING` line | building normally; it says what was written last and when |
| the last `WAITING` line | **stuck or waiting** — say how long it has been quiet and, if the line names it, what on. Killed at 90m |
| `STATE a -> b` | the board moved under it — when, and from what to what |
| `DONE` | finished; the state line says where it went |
| `STOPPED` | the build stopped on a wrong plan — deliberate, and **not** a stall. The reason is in the job file's last section, not on the marker |
| `STALLED` or `WEDGED` | **stalled** — say the reason written on that line |
| last marker `PID n`, and the pid is gone | **stalled** — killed hard, or the machine restarted |
| a log with no markers at all | **stalled** — it predates them |
| any other state in the frontmatter | say it as it stands; the loop ignores it, so it waits on nobody |

Stalled is `/rework N`, and say so on the line. Stopped is not — it is a decision waiting on him,
so it goes up with `review` and the exit is `/review N`.

**`STALLED` and `STOPPED` are different endings and must never be reported as the same one.** A
stall is a build that died somewhere in the middle; a stop is a build that finished what it could
and refused to pass itself. Jobs 011 and 012 both stopped correctly on the same afternoon and the
loop called both of them stalled, which is what the `STOPPED` marker exists to end.

If the loop isn't running nothing will move — say that first. Check it with the anchored pattern
`pgrep -f 'sh .*factory/loop\.sh$'`; written loose it matches your own shell, which is the same
trap as above.

One line per job, newest first. If nothing is waiting on him, say that in one line and stop.
