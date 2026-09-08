#!/usr/bin/env python3
"""Render every mood of the eye across a strip of time, so a change can be looked at.

The eye is nine numbers and a colour per state (``cyclops.overlay.MOODS``), and none of those
numbers can be argued about usefully - a spin of 40 versus 70 is not a discussion, it is a look.
So: edit the table, run this, open the sheet.

    uv run python tools/eye_sheet.py                 # every mood, three seconds of each
    uv run python tools/eye_sheet.py --seconds 8     # ...over a longer window, to catch a blink
    uv run python tools/eye_sheet.py --level 1.0     # ...with the signal meter pinned
    uv run python tools/eye_sheet.py --state listening --frames 16 --seconds 6

One row per mood, one column per frame, and the aperture printed underneath so a mood that looks
wrong can be traced back to the number that made it. Writes a PNG and says where.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from cyclops import overlay
from cyclops.eye import EyeEngine

RADIUS = 60  # what it is on the 800x480 panel, so the sheet is life-size and not a mock-up
PAD = 10
LABEL = 92  # room down the left for the state's name


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("out", nargs="?", default="eye-sheet.png", type=Path)
    ap.add_argument("--state", action="append", help="only these (repeatable); default is all")
    ap.add_argument("--seconds", type=float, default=3.0, help="how much time a row covers")
    ap.add_argument("--frames", type=int, default=8, help="how many frames that is sampled into")
    ap.add_argument("--level", type=float, default=0.4,
                    help="the level fed to the eye, 0..1. Only speaking opens to it")
    ap.add_argument("--scale", type=int, default=1, help="blow the sheet up by this much")
    args = ap.parse_args()

    moods = {k: v for k, v in overlay.MOODS.items() if not args.state or k in args.state}
    if not moods:
        raise SystemExit(f"no such state. have: {', '.join(overlay.MOODS)}")

    cell = 2 * RADIUS + 2 * PAD
    sheet = Image.new("RGB", (LABEL + cell * args.frames, cell * len(moods)), (0, 0, 0))
    d = ImageDraw.Draw(sheet)
    for row, (state, mood) in enumerate(moods.items()):
        engine = EyeEngine(RADIUS, 2, overlay.SCREEN, mood)
        top = row * cell
        d.text((8, top + cell // 2 - 4), state.upper(), fill=overlay.GREEN_MID)
        d.text((8, top + cell // 2 + 8), f"{state}", fill=overlay.GREEN_DIM)
        for col in range(args.frames):
            phase = col * args.seconds / max(1, args.frames)
            tile = Image.new("RGBA", (cell, cell), (*overlay.SCREEN, 255))
            engine.paint(tile, cell // 2, cell // 2, mood, phase, args.level)
            sheet.paste(tile.convert("RGB"), (LABEL + col * cell, top))
            ImageDraw.Draw(sheet).text(
                (LABEL + col * cell + 4, top + cell - 12),
                f"{engine.aperture(mood, phase, args.level):.2f}",
                fill=overlay.GREEN_DIM,
            )
    if args.scale > 1:
        sheet = sheet.resize((sheet.width * args.scale, sheet.height * args.scale), Image.NEAREST)
    sheet.save(args.out)
    print(f"· {args.out} — {len(moods)} moods x {args.frames} frames over {args.seconds:g}s "
          f"at level {args.level:g}")


if __name__ == "__main__":
    main()
