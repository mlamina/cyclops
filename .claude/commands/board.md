---
description: What the factory is doing and what's waiting on you
---

Read the frontmatter of every `factory/[0-9]*.md` and say where things stand. **Waiting on Marco
first** — that's the only part he has to act on.

- `state: review` → waiting on him. Give the number, the title, and `/review N`.
- `state: ready` with a build running (`pgrep -f "claude -p /build NNN"`) → building.
- `state: ready` with no process but a `factory/logs/NNN.log` → **stalled**: the loop took its
  swing and the build died. Say `/rework N` to send it round again.
- `state: ready`, no process, no log → queued, the loop will take it within half a minute.
- anything else → say the state as it stands. The loop ignores it, so it is waiting on nobody
  until he does something with it.

If `factory/loop.sh` isn't running (`pgrep -f factory/loop.sh`), say so first — nothing will move
until it is.

One line per job, newest first. If nothing is waiting on him, say that in one line and stop.
