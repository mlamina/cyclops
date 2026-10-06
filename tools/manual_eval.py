#!/usr/bin/env python3
"""Ask the librarian the questions that went wrong on the Pi, and grade every answer.

    uv run python tools/manual_eval.py
    uv run python tools/manual_eval.py --out eval.jsonl --only 1,7

No realtime session: this calls :func:`cyclops.librarian.ask` directly, against the manuals in
``../cyclops-fixtures/manuals`` (the mo.unit blue manual, Revival's m-unit wiring diagram and the
OXI One manual). Their recall rows are embedded into a scratch index on the first run and kept,
so nothing is written under ``~/.cache/cyclops``.

Per question it prints the pages cited, whether that is the right page, whether the answer holds
the specific thing that was asked for, seconds, requests and cost. "Specific" is a word check -
a menu number, a wire colour - and a pair of eyes on the printed answers is the real grade.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from openai import AsyncOpenAI

from cyclops import card, librarian, recall
from cyclops.config import load_settings

FIXTURE_MANUALS = Path.home() / "code" / "cyclops-fixtures" / "manuals"
BENCH = Path(tempfile.gettempdir()) / "cyclops-manual-eval"
COLOURS = r"\b(red|black|white|green|blue|yellow|orange|brown|purple|violet|pink|grey|gray)\b"


@dataclass(frozen=True)
class Case:
    question: str
    manual: str  # a fragment of the right manual's key; "" for a question that must be declined
    pages: frozenset[int]
    specific: str  # a regex the answer must match to count as giving the specific thing


CASES = (
    Case("mo.unit blue: how do I get into the setup menu, exactly?", "mo unit", frozenset({33}),
         r"(?s)horn.*ignition|ignition.*horn"),
    Case("mo.unit blue: what modes can I set for the LIGHT input?", "mo unit", frozenset({31}),
         r"menu 8"),
    Case("mo.unit blue: how do I make the tail light come on only when the engine is running?",
         "mo unit", frozenset({31}), r"menu 9"),
    Case("Do the mo.unit blue inputs take 12 volts?", "mo unit", frozenset({19, 21}),
         r"earth|ground"),
    Case("mo.unit blue setup: how do I confirm a menu choice?", "mo unit", frozenset({35}),
         r"horn"),
    Case("mo.unit blue: what's the difference between handlebar configuration B and C?",
         "mo unit", frozenset({20}), r"(?s)(harley|bmw).*(japanese|european)|(japanese|european).*(harley|bmw)"),
    Case("Revival m-unit wiring diagram: what colour is each input wire - kill, aux, brake, "
         "turn, horn, start?", "revival", frozenset({1}), rf"(?s){COLOURS}.*{COLOURS}.*{COLOURS}"),
    Case("OXI One: how do I get to the notes below the ones lit on the grid?", "oxi",
         frozenset({12}), r"\bdiv"),
    Case("BMW R80: what is the cylinder head bolt torque?", "", frozenset(), r"search"),
)


async def _bench() -> None:
    """The fixture manuals and nothing else, with a scratch index of their pages."""
    BENCH.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CYCLOPS_MANUALS_DIR", str(FIXTURE_MANUALS))
    for name in ("SESSIONS", "PROJECTS", "CAPTURES"):
        os.environ[f"CYCLOPS_{name}_DIR"] = str(BENCH / name.lower())
    recall.RECALL_FILE = BENCH / "recall.npz"
    settings = load_settings()
    items = recall.corpus(settings)
    if [i.key for i in recall.load(recall.RECALL_FILE).items] == [i.key for i in items]:
        return
    client = AsyncOpenAI(api_key=settings.api_key)
    try:
        vectors = await recall.embed([item.text for item in items], client)
    finally:
        await client.close()
    card.write_bytes(recall.RECALL_FILE, recall.dump(recall.Index(items=items, vectors=vectors)))
    print(f"· indexed {len(items)} items into {recall.RECALL_FILE}", flush=True)


def grade(case: Case, found: librarian.Lookup) -> dict:
    answer = found.answer
    cited = [(s.manual, s.page) for s in answer.pages]
    if not case.manual:
        right = not answer.answered
        specific = bool(re.search(case.specific, answer.why_not, re.IGNORECASE))
    else:
        right = answer.answered and any(
            case.manual in m and p in case.pages for m, p in cited
        )
        specific = bool(re.search(case.specific, answer.answer, re.IGNORECASE))
    return {
        "question": case.question,
        "answered": answer.answered,
        "answer": answer.answer,
        "why_not": answer.why_not,
        "cited": [f"{m} p{p}" for m, p in cited],
        "right_page": right,
        "specific": specific,
        "seconds": found.seconds,
        "requests": found.requests,
        "cost": found.cost,
        "steps": found.steps,
    }


async def run(only: set[int], out: Path | None) -> int:
    await _bench()
    settings = load_settings()
    rows = []
    for n, case in enumerate(CASES, 1):
        if only and n not in only:
            continue
        try:
            found = await librarian.ask(case.question, settings)
        except Exception as exc:  # one failed question is a row, not a failed run
            print(f"{n}. FAILED {type(exc).__name__}: {exc}", flush=True)
            rows.append({"question": case.question, "error": str(exc), "right_page": False,
                         "specific": False, "seconds": 0.0, "requests": 0, "cost": 0.0})
            continue
        row = grade(case, found)
        rows.append(row)
        said = row["answer"] if row["answered"] else f"[why_not] {row['why_not']}"
        print(f"\n{n}. {case.question}\n   {said}\n   cited {', '.join(row['cited']) or '-'}"
              f" | right page {row['right_page']} | specific {row['specific']}"
              f" | {row['seconds']:.1f}s, {row['requests']} req, ${row['cost']:.4f}"
              f"\n   steps: {' / '.join(row['steps'])}", flush=True)
        if out:
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
    table(rows)
    return 0


def table(rows: list[dict]) -> None:
    print("\n| # | Question | Cited | Right page | Specific | s | req | $ |")
    print("|---|---|---|---|---|---|---|---|")
    for n, row in enumerate(rows, 1):
        print(f"| {n} | {row['question']} | {', '.join(row.get('cited', [])) or '-'} | "
              f"{'yes' if row['right_page'] else 'NO'} | {'yes' if row['specific'] else 'NO'} | "
              f"{row['seconds']:.1f} | {row['requests']} | {row['cost']:.4f} |")
    times = [row["seconds"] for row in rows if row["seconds"]]
    if times:
        print(f"\nright page {sum(r['right_page'] for r in rows)}/{len(rows)}, specific "
              f"{sum(r['specific'] for r in rows)}/{len(rows)}, median {statistics.median(times):.1f}s, "
              f"slowest {max(times):.1f}s, total ${sum(r['cost'] for r in rows):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--only", default="", help="comma-separated question numbers")
    args = parser.parse_args()
    only = {int(n) for n in args.only.split(",") if n.strip()}
    sys.exit(asyncio.run(run(only, args.out)))


if __name__ == "__main__":
    main()
