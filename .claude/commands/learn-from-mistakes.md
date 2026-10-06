---
description: Find where Cyclops fell short in recent sessions, then dig into the causes with evidence
argument-hint: "[number of sessions, default 10]"
---

Two passes: a quick triage of every session, then an investigation of what the triage found.
Read-only from start to finish. Change nothing on the Pi or in the repo; fixing is a separate step.

## 1. Find the sessions

List the newest session folders on the Pi yourself, without delegating:

```
ssh cyclops@cyclops.local 'cd ~/cyclops 2>/dev/null || cd ~; ls -dt sessions/*/ | head -${ARGUMENTS:-10}'
```

Each folder has `session.md` (the transcript) and `session.jsonl` (tool calls and results).

## 2. Triage: one quick sub-agent per session, all in parallel

Spawn every sub-agent in a single message, with `model: sonnet`, in the background. Each one reads
one `session.md`, skimming `session.jsonl` only if it has to. They must not run scripts, look at
images or video, or verify anything. This pass is a quick read.

Each sub-agent flags:
- Marco asking the same thing again, rephrasing, "what?", "no, the other one", pushing back
- Cyclops not finding, or answering vaguely about, something that plausibly exists (a manual
  page, a past session, project data, a diagram)
- guessing before looking up, contradicting itself, tool errors, unannounced waits, Marco
  having to manage Cyclops itself

Each sub-agent replies in under 150 words: a verdict (fine / minor / bad), then bullets, each
with a short quote and a one-line guess at the cause.

Combine the reports into a short summary for Marco, with the misses grouped by pattern and
counted ("4 of 10 sessions"). **Stop there and ask which patterns to dig into.**

## 3. Investigate: one sub-agent per pattern, in parallel

When Marco picks, spawn the sub-agents in a single message:
- one per pattern (or per bad session), given the specific sessions and quotes from the triage
- one for the code side: which lookup tools the model has, what their descriptions and the
  system prompt say about looking up and searching again, how manuals and project data are
  indexed, and what context goes in at session start. Cite file:line.

Every session sub-agent establishes, for each miss:
1. The exact tool calls from `session.jsonl` (query → results), or the fact that none were made.
2. Where the answer actually lives (manual page, project file or tab, session), with a short
   quote. Text search only, no images. Re-ranking the logged queries against the Pi's recall
   index is allowed if it writes nothing.
3. The cause, marked VERIFIED or GUESS: no lookup / bad query / right result ranked low or not
   returned / returned but misread / not searchable / not in the sources at all.

Each sub-agent replies in under 250 words.

## 4. Report

Write one concise problem description that is built on evidence, not a list of fixes. Rank the
problems by how many misses they explain. For each one, give the mechanism, the quotes and pages
or paths that prove it, and the file:line where it lives in the code. Say which parts are
guesses. Close with the one or two problems that explain most of the repeated questions.
