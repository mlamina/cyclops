#!/usr/bin/env python3
"""Measure the standing prompt: what the model reads before it has heard a word.

Length is the thing to watch. Every token here is re-read on every turn of every session, it
competes with the rules next to it for the model's attention, and OpenAI caps the instructions
at 16,384. A rule stated twice is not stated twice as firmly - the realtime prompting guide is
blunt that ambiguous or conflicting instructions degrade behaviour - so the repetition scan
below matters as much as the totals.

    uv run python tools/prompt_audit.py
    uv run python tools/prompt_audit.py --ceiling   # what a full card would send

Measurement only, and no verdict: this prints sizes and overlaps, and what to cut is a
judgement about what Cyclops should sound like. `/prompt-audit` is the command that makes it.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import tiktoken  # noqa: E402

from cyclops import agent, session  # noqa: E402
from cyclops.config import Settings  # noqa: E402

ENC = tiktoken.get_encoding("o200k_base")  # what gpt-realtime counts in
LIMIT = 16_384  # the API's cap on `instructions`
GRAM = 5  # shortest word run worth calling a repetition
# Words that carry no meaning on their own. A run made only of these says nothing about whether
# two blocks are repeating each other - "and it is not the" is grammar, not a rule said twice.
DULL = frozenset(
    "a an and are as at be but by do for from get has have in is it its not of on one or "
    "out so that the them then they this to up was what when which who will with you your "
    "- it is do not they are you are".split()
)


def toks(text: str) -> int:
    return len(ENC.encode(text))


def blocks() -> dict[str, str]:
    """Every distinct piece of standing text, named. Sections of BASE_INSTRUCTIONS are split on
    their own ALL-CAPS headings rather than a hardcoded list, so a section renamed or added
    shows up here without this file being touched."""
    found: dict[str, str] = {}
    name, buf = "(preamble)", []
    for line in agent.BASE_INSTRUCTIONS.splitlines():
        if re.fullmatch(r"[A-Z][A-Z ']{4,}", line.strip()) and not line.startswith(" "):
            found[name] = "\n".join(buf)
            name, buf = line.strip(), []
        else:
            buf.append(line)
    found[name] = "\n".join(buf)
    return found


def payload(settings: Settings) -> tuple[str, list[dict]]:
    """The instructions and tools exactly as the session would send them.

    Built once: ``session_config`` reads the card and narrates what it found, and a second call
    would print the lot again under the numbers.
    """
    agent_ = agent.VoiceAgent.__new__(agent.VoiceAgent)
    agent_.settings = settings
    config = agent.VoiceAgent.session_config(agent_)
    made = [t if isinstance(t, dict) else t.model_dump(exclude_none=True) for t in config["tools"]]
    return config["instructions"], made


def bullets(text: str) -> tuple[int, int]:
    """Lines that are bullets, and lines that are prose. The guide prefers the first."""
    lines = [one for one in text.splitlines() if one.strip()]
    marked = sum(1 for one in lines if one.lstrip().startswith(("-", "*", "•")))
    return marked, len(lines) - marked


def repeats(named: dict[str, str]) -> list[tuple[str, list[str]]]:
    """Whole passages that turn up in more than one block - the same rule, said again elsewhere.

    Pairwise and maximal, rather than counting fixed-length shingles: one repeated sentence is
    one finding here, where shingles report it a dozen times as a dozen near-copies of itself
    and bury the other rules. What comes back is the run of words, and everywhere it was found.
    """
    words = {name: re.findall(r"[a-z]+", text.lower()) for name, text in named.items()}
    where: defaultdict[str, set[str]] = defaultdict(set)
    names = sorted(words)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            matcher = difflib.SequenceMatcher(None, words[first], words[second], autojunk=False)
            for start, _, size in matcher.get_matching_blocks():
                run = words[first][start : start + size]
                if size < GRAM or all(word in DULL for word in run):
                    continue
                where[" ".join(run)].update((first, second))
    # A passage in three blocks is found as three pairs, and a slightly shorter version of it
    # may be found too. Fold a shorter run into the longer one that contains it - but only the
    # blocks that carry the whole of the longer run, or a block sharing one clause of a long
    # passage is reported as carrying all of it, which reads as a verbatim copy that is not there.
    kept: list[tuple[str, list[str]]] = []
    for run in sorted(where, key=len, reverse=True):
        if any(run in seen for seen, _ in kept):
            continue
        kept.append((run, sorted(n for n in named if run in " ".join(words[n]))))
    return kept


def ceiling() -> int:
    """What a full card would send: every context block at the cap it is allowed."""
    letters = ("the quick brown fox jumps over a lazy dog " * 200)[:20_000]
    return sum(
        [
            toks(agent.BASE_INSTRUCTIONS),
            toks(agent.ABOUT_HEADER) + toks(letters[: agent.ABOUT_MAX_CHARS]),
            toks(agent.RECAP_HEADER) + toks(letters[: session.RECAP_MAX_CHARS]),
            toks(agent.PROJECTS_HEADER)
            + toks(letters[: agent.PROJECTS_LISTED * agent.PROJECT_LINE_CHARS]),
            toks(agent.MANUALS_HEADER) + toks(letters[: agent.MANUALS_LISTED * 90]),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ceiling", action="store_true", help="also show a full card's worst case")
    parser.add_argument("--json", action="store_true", help="machine-readable, for a diff")
    args = parser.parse_args()

    settings = Settings(api_key="audit")
    instructions, schemas = payload(settings)
    sections = blocks()
    described = {t["name"]: t.get("description", "") for t in schemas}

    instr_t = toks(instructions)
    tools_t = sum(toks(json.dumps(t)) for t in schemas)

    if args.json:
        print(
            json.dumps(
                {
                    "instructions": instr_t,
                    "tools": tools_t,
                    "sections": {k: toks(v) for k, v in sections.items()},
                    "tool_total": {t["name"]: toks(json.dumps(t)) for t in schemas},
                    "tool_desc": {k: toks(v) for k, v in described.items()},
                    "repeats": [[g, ns] for g, ns in repeats({**sections, **described})],
                },
                indent=1,
            )
        )
        return

    print(f"STANDING PROMPT   {instr_t + tools_t:>6} tok")
    print(f"  instructions    {instr_t:>6} tok   ({instr_t / LIMIT:.0%} of the {LIMIT} cap)")
    print(f"  tools           {tools_t:>6} tok   ({len(schemas)} tools)")
    if args.ceiling:
        top = ceiling()
        print(f"  full card would be {top} tok of instructions ({top / LIMIT:.0%} of the cap)")

    print("\nBASE_INSTRUCTIONS by section        bullets/prose lines")
    for name, text in sorted(sections.items(), key=lambda kv: -toks(kv[1])):
        marked, prose = bullets(text)
        print(f"  {toks(text):>5} tok  {name:<26} {marked:>3} / {prose:<3}")

    print("\ntools                      description / schema")
    for t in sorted(schemas, key=lambda t: -toks(json.dumps(t))):
        whole, desc = toks(json.dumps(t)), toks(t.get("description", ""))
        print(f"  {whole:>5} tok  {t['name']:<22} {desc:>4} / {whole - desc:<4}")

    shared = repeats({**sections, **described})
    print(f"\nrepeated across blocks ({len(shared)} runs of {GRAM}+ words in 2+ places)")
    for gram, names in shared[:20]:
        print(f'  "{gram}"\n      {", ".join(names)}')
    if len(shared) > 20:
        print(f"  ... and {len(shared) - 20} more (--json for all)")

    spread = Counter(n for _, names in shared for n in names)
    if spread:
        print("\nblocks that most often repeat something said elsewhere")
        for name, count in spread.most_common(8):
            print(f"  {count:>3}  {name}")


if __name__ == "__main__":
    main()
