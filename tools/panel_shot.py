#!/usr/bin/env python3
"""Render the kiosk panel over a real camera frame, on a Mac, so a change can be looked at.

The chrome is pure PIL and numpy, so one frame of the panel can be produced with no Pi, no camera
and no session - only the picture behind it has to be borrowed, and a photo Cyclops took is the
honest choice for that. This is the harness the gauntlet loop judges with: the same command, the
same photo, the same state, and the only thing that differs between two shots is the code.

    uv run python tools/panel_shot.py --bg photo.jpg --state listening --out shot.png
    uv run python tools/panel_shot.py --bg photo.jpg --state speaking --level 0.6 --out shot.png
    uv run python tools/panel_shot.py --bg photo.jpg --state listening --strip 8 --seconds 4 --out eye.png
    uv run python tools/panel_shot.py --bench

``--strip`` crops the eye and lays N frames of it across a row, which is how motion is looked at
in a still: a face that is alive differs from frame to frame, and one that is a decoration does
not. ``--bench`` prints the per-frame render and composite cost, which is the number the Pi is
short of. Writes a PNG and says where.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

from cyclops import eye, overlay

WIDTH, HEIGHT = 800, 480
STRIP_PAD = 12  # picture kept around the eye in each strip cell, so the bezel's shadow is in it
AWAKE_ELAPSED = 12.0  # a session twelve seconds old, which is what the clock readouts show


def shown(args: argparse.Namespace) -> dict:
    """The render() kwargs for the state asked for. Asleep has no session, so no elapsed."""
    asleep = args.state == overlay.IDLE
    return dict(
        state=args.state,
        level=args.level,
        elapsed=None if asleep else AWAKE_ELAPSED,
        recording=args.recording,
        detail=args.detail,
        volume=args.volume,
        temp_c=args.temp,
    )


def background(path: Path | None) -> np.ndarray:
    """The picture behind the chrome as a BGR frame the panel's size, or a dark plate if none."""
    if path is None:
        return np.full((HEIGHT, WIDTH, 3), 24, np.uint8)
    rgb = np.asarray(Image.open(path).convert("RGB"))
    return overlay.fit_to_window(np.ascontiguousarray(rgb[..., ::-1]), WIDTH, HEIGHT)


def settle(ov: overlay.Overlay, kw: dict) -> None:
    """Run the mood crossfade and the caption typing out, so a shot is the state and not the journey."""
    ov.render(phase=0.0, **kw)
    ov.render(phase=eye.MOOD_EASE_S * 4, **kw)


def shot(ov: overlay.Overlay, frame: np.ndarray, kw: dict, phase: float) -> Image.Image:
    rgba = ov.render(phase=phase, **kw)
    bgr = overlay.composite(frame.copy(), rgba)
    return Image.fromarray(np.ascontiguousarray(bgr[..., ::-1]))


def eye_box(ov: overlay.Overlay) -> tuple[int, int, int, int]:
    cx, cy = ov.eye
    r = ov.eye_r + STRIP_PAD
    return (int(cx - r), int(cy - r), int(cx + r), int(cy + r))


def strip(ov: overlay.Overlay, frame: np.ndarray, kw: dict, frames: int, seconds: float,
          start: float) -> Image.Image:
    box = eye_box(ov)
    cell = (box[2] - box[0], box[3] - box[1])
    out = Image.new("RGB", (cell[0] * frames, cell[1]), (0, 0, 0))
    for i in range(frames):
        phase = start + i * seconds / max(1, frames)
        out.paste(shot(ov, frame, kw, phase).crop(box), (i * cell[0], 0))
    return out


def bench(ov: overlay.Overlay, frame: np.ndarray, kw: dict) -> tuple[float, float]:
    settle(ov, kw)
    t = time.perf_counter()
    for i in range(30):
        rgba = ov.render(phase=1.0 + i * 0.066, **kw)
    render_ms = (time.perf_counter() - t) / 30 * 1000
    t = time.perf_counter()
    for _ in range(30):
        overlay.composite(frame, rgba)
    return render_ms, (time.perf_counter() - t) / 30 * 1000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bg", type=Path, help="photo behind the chrome; a dark plate if omitted")
    ap.add_argument("--state", default=overlay.LISTENING, choices=list(overlay.MOODS))
    ap.add_argument("--level", type=float, default=0.3, help="mic/voice level 0..1")
    ap.add_argument("--volume", type=int, default=60)
    ap.add_argument("--temp", type=float, default=60.0)
    ap.add_argument("--recording", action="store_true")
    ap.add_argument("--detail", default="", help="the controller's own line on the terminal")
    ap.add_argument("--phase", type=float, default=12.0, help="the clock, in seconds")
    ap.add_argument("--strip", type=int, default=0, help="crop the eye and lay N frames in a row")
    ap.add_argument("--seconds", type=float, default=4.0, help="how much time a strip covers")
    ap.add_argument("--bench", action="store_true", help="print ms per frame and exit")
    ap.add_argument("--out", type=Path, default=Path("panel.png"))
    args = ap.parse_args()

    ov = overlay.Overlay(WIDTH, HEIGHT)
    frame = background(args.bg)
    kw = shown(args)
    if args.bench:
        for state in (overlay.IDLE, overlay.LISTENING, overlay.SPEAKING):
            ov = overlay.Overlay(WIDTH, HEIGHT)
            kw = {**kw, "state": state, "elapsed": None if state == overlay.IDLE else AWAKE_ELAPSED}
            r, c = bench(ov, frame, kw)
            print(f"{state:10s} render {r:5.1f} ms  composite {c:5.1f} ms")
        return
    settle(ov, kw)
    if args.strip:
        image = strip(ov, frame, kw, args.strip, args.seconds, args.phase)
    else:
        image = shot(ov, frame, kw, args.phase)
    image.save(args.out)
    print(f"· {args.out} — {args.state} at phase {args.phase:g}"
          + (f", eye x{args.strip} over {args.seconds:g}s" if args.strip else ""))


if __name__ == "__main__":
    main()
