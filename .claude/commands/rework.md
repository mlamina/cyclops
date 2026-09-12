---
description: Send a job back round with what's wrong
argument-hint: <job number> <what's wrong>
---

Another round on job `$ARGUMENTS` — the number, then what's wrong in Marco's words.

Append his words to the job file under `## Feedback — <date>`, verbatim. **Never edit the earlier
rounds** — the history of a job is the job, and he should be able to see it converging, or not.

Then set `state: ready` and **delete `factory/logs/NNN.log`**. That log is the loop's record that
it already took a swing; removing it is what makes the job eligible again. The branch and the
worktree stay, so the next build continues rather than starting over.

If what he said changes what *done* means, update the **Done when** list and say which line you
changed. Criteria that quietly drift to match whatever got built are worse than none.

One line: it's back in the queue, the loop has it.
