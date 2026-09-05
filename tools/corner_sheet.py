#!/usr/bin/env python3
"""Render the bottom-right corner - the knob and the gauge - so a change can be looked at.

The two instruments are a handful of radii and three colours, and none of those numbers can be
argued about usefully: a pointer at 0.52 of the disc against one at 0.60 is not a discussion, it
is a look. So: edit the constants in ``cyclops.overlay``, run this, open the sheet.

    uv run python tools/corner_sheet.py                # both rows, life-size and at 3x
    uv run python tools/corner_sheet.py --scale 6      # ...closer, to argue about the bezel

Top row is the volume knob across its travel, with the last cell turning under a finger; bottom
row is the heat gauge across the board's own range, from a cold Pi to one that is being throttled.
Life-size on the left of each pair and magnified on the right, because the first is the only
honest one and the second is the only one you can see the stairs in.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from cyclops import overlay
from cyclops.overlay import Overlay

PANEL = (800, 480)
CROP = 200  # of the corner, which holds both dials and the rail they are bolted to
PAD = 12
LABEL = 16

VOLUMES = (0, 25, 60, 100, None)
TEMPS = (32.0, 55.0, 74.0, 84.0, None)


def corner(ov: Overlay, **kwargs: object) -> Image.Image:
    """One frame of the panel, cropped to the bracket the two instruments are bolted to."""
    frame = Image.fromarray(ov.render(state=overlay.IDLE, level=0.0, **kwargs))
    return frame.crop((PANEL[0] - CROP, PANEL[1] - CROP, PANEL[0], PANEL[1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=3, help="how far the magnified copy is blown up")
    ap.add_argument("--out", type=Path, default=Path("corner-sheet.png"))
    args = ap.parse_args()

    ov = Overlay(*PANEL)
    rows = [
        [corner(ov, volume=v, temp_c=58.0, pressed=None if v != 100 else overlay.VOLUME)
         for v in VOLUMES],
        [corner(ov, volume=60, temp_c=t) for t in TEMPS],
    ]

    big = CROP * args.scale
    cell_w, cell_h = CROP + PAD + big, max(CROP, big)
    sheet = Image.new("RGB", (cell_w * len(VOLUMES) + PAD, (cell_h + PAD) * len(rows) + PAD),
                      (12, 12, 12))
    for r, row in enumerate(rows):
        for c, tile in enumerate(row):
            x, y = PAD + c * cell_w, PAD + r * (cell_h + PAD)
            sheet.paste(tile.convert("RGB"), (x, y))
            sheet.paste(tile.convert("RGB").resize((big, big), Image.NEAREST), (x + CROP + PAD, y))
    sheet.save(args.out)
    print(f"wrote {args.out} ({sheet.width}x{sheet.height})")
    print(f"  top: volume {VOLUMES} (the last one turning under a finger)")
    print(f"  bottom: {TEMPS} °C")


if __name__ == "__main__":
    main()
