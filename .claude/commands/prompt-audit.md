---
description: Audit what Cyclops reads before it hears a word — length, repetition, and the realtime prompting best practices
---

Everything in the standing prompt is re-read on every turn of every session, and every line
competes with the line next to it for the model's attention. This checks it against how OpenAI
says to write for a realtime voice model, and comes back with what to cut.

## Measure first

```sh
uv run python tools/prompt_audit.py --ceiling
```

That prints the totals, `BASE_INSTRUCTIONS` by section with bullet/prose line counts, each tool
split description vs JSON schema, and the passages that appear in more than one block. Read
`src/cyclops/agent.py` for anything the numbers point at. `--json` if you want to diff two runs.

## The rules it is being judged against

From OpenAI's [realtime prompting guide](https://developers.openai.com/cookbook/examples/realtime_prompting_guide).
These are for speech-to-speech models specifically and do not all hold for text ones.

| | What it says | What to look for here |
|---|---|---|
| **Don't over-prompt** | Start minimal, add a rule only when a behaviour actually fails | A rule nobody can name a session that needed it |
| **Bullets, not paragraphs** | "Clear, short bullets outperform long paragraphs" | Sections with a 0 in the bullets column |
| **Be precise** | "Ambiguity or conflicting instructions = degraded performance" | Everything in the repetition scan |
| **Guide with examples** | "The model strongly closely follows sample phrases" | Banned-phrase lists — still phrases, still in context |
| **Reduce repetition** | A Variety rule beats robotic phrasing | Whether variety is one rule or an argument |
| **Capitals for emphasis** | Capitalising a key rule makes it stand out | Whether SHOUTING has spread far enough to mean nothing |
| **Preambles** | Say something before a slow tool so silence has context | Every tool that takes more than ~2s |

Its skeleton is Role & Objective · Personality & Tone · Context · Reference Pronunciations ·
Tools · Instructions · Conversation Flow · Safety. Say which of ours map onto it and which
don't. Pronunciations is the one we keep not having, and this thing reads M6, 1/4-20, 8.8 Nm
and R80RT out loud over a compressor.

## What Cyclops adds to that

- **Tool rules belong in the tool description, not the system prompt.** The schema is read when
  the model is already considering the tool; the system prompt is read every session whether the
  tool comes up or not. A rule in both places is paid twice and obeyed no harder.
- **Rules that fire once should not be read every turn.** The greeting rules are the standing
  example: they govern one line at the top of a session and are re-read on all of them.
- **A rule that caps a behaviour reads as a licence to do it once.** If a cap is there, check the
  transcripts before defending it.
- **Length is a proxy, not the goal.** Cutting 400 tokens off a section that earns its keep is a
  loss. The wins are duplication, once-per-session rules, and prose that should be bullets.

## What to come back with

1. The totals, and what moved since last time if you can tell.
2. The three or four worst findings, each as: what it is, where it lives (`file:line`), and
   roughly what cutting it saves.
3. A ranked list of cuts, most valuable first.
4. Anything the guide wants that we don't have at all.

**Analysis only — change nothing.** These lines are Marco's ear, not a style rule: say what you
would cut and let him pick. Where a section is long because it is doing real work, say so
instead of finding something to trim in it.
