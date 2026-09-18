---
description: Look at what came out of a job — the outcome, never the code
argument-hint: <job number>
---

Job `$ARGUMENTS` is built. Show Marco the result.

Read the job file's **Done when** and **Built** sections, and **read the PNGs under
`factory/html/NNN/` so they appear here** — that saves him a switch. Then open the full page
in his browser without asking: `open factory/html/NNN.html`.

Say, in this order:
1. What it does now — one or two lines, in outcome terms.
2. The checklist: what's met, what isn't, and which lines were only ever his to judge.
3. That the page is open in the browser, and its path.

**Never show or summarise the code, the diff, the files touched or the approach.** If a criterion
is unmet, say so plainly rather than talking around it.

End with the exits, on one line: `/try N` to put it on the Pi · `/ship N` · `/rework N <what's
wrong>` · `/drop N`.
