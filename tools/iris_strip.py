#!/usr/bin/env python3
"""Render the steel cover winding across him as a filmstrip, so the close can be looked at.

The close is six rigid blades on a timing curve, and neither of those is a thing to argue about
in a comment - a blade that stretches and a close that reads as a wipe both look wrong at a
glance and are invisible in the source. So: run this, open the frames, and look.

    uv run python tools/iris_strip.py --frames 8 --out /tmp/iris
    uv run python tools/iris_strip.py --frames 8 --out /tmp/iris --open   # ...and the other way

One PNG per frame, at the size he actually is on the panel, on the panel's own background. The
frames are sampled evenly in TIME across the real animation and not evenly across the aperture,
which is the whole point of the strip: what it shows is the curve `EyeEngine.shut` is on, so a
close that spends a quarter of its window on the first sixth of the travel looks like one here.

Nothing is faked. It drives the real engine through the real state change - awake to asleep, or
back - so the mood crossfade underneath the cover is the panel's own too.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from cyclops import overlay
from cyclops.eye import COVER_OPEN_S, COVER_SHUT_S, EyeEngine

SIZE = 132  # what he is on the 800x480 panel: eye_r of 66, edge to edge and no bigger
AWAKE = overlay.LISTENING  # the state on the other side of the line from asleep
PHASE = 100.0  # far enough in that nothing here is watching the panel's first second
LAG = 0.04  # one frame at the kiosk's rate, and the beat the panel really does lose here: the
# eye is handed the old mood on the frame a state changes (the crossfade's clock starts there),
# so the cover learns where it is going on the frame after. Frame zero is that one - the first
# frame of the animation rather than the last frame before it.


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--frames", type=int, default=8, help="how many frames the window is cut into")
    ap.add_argument("--out", type=Path, required=True, help="the directory to write them into")
    ap.add_argument("--open", dest="opening", action="store_true",
                    help="render the cover clearing rather than closing")
    args = ap.parse_args()

    was, now = (overlay.IDLE, AWAKE) if args.opening else (AWAKE, overlay.IDLE)
    span = COVER_OPEN_S if args.opening else COVER_SHUT_S
    radius = SIZE // 2
    engine = EyeEngine(radius, overlay.Overlay(800, 480).line, overlay.SCREEN, overlay.MOODS[was])
    engine.look(now, overlay.MOODS[now], PHASE)  # the state changes here, and the strip follows
    args.out.mkdir(parents=True, exist_ok=True)
    for i in range(args.frames):
        # Evenly in time, and the last frame lands exactly on the end of the window rather than
        # one step short of it - so frame zero is the cover clear and the last one is the cover
        # arrived, whatever `--frames` says.
        phase = PHASE + LAG + span * i / max(1, args.frames - 1)
        mood = engine.look(now, overlay.MOODS[now], phase)
        # His own tile is 2r+1 across, which is one more than he is on the panel: painted into a
        # square of his real size, the bottom and right pixel of that tile are off the end of it.
        tile = Image.new("RGBA", (SIZE + 1, SIZE + 1), (*overlay.SCREEN, 255))
        engine.paint(tile, radius, radius, mood, phase, 0.0)
        tile.crop((0, 0, SIZE, SIZE)).convert("RGB").save(args.out / f"{i:02d}.png")
    print(f"· {args.out} — {args.frames} frames of the "
          f"{'open' if args.opening else 'close'} over {span:g}s, {SIZE}x{SIZE}")


if __name__ == "__main__":
    main()
