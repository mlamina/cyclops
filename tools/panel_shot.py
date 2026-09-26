#!/usr/bin/env python3
"""Render the kiosk panel over a real camera frame, on a Mac, so a change can be looked at.

The chrome is pure PIL and numpy, so one frame of the panel can be produced with no Pi, no camera
and no session - only the picture behind it has to be borrowed, and a photo Cyclops took is the
honest choice for that. This is the harness the gauntlet loop judges with: the same command, the
same photo, the same state, and the only thing that differs between two shots is the code.

    uv run python tools/panel_shot.py --bg photo.jpg --state listening --out shot.png
    uv run python tools/panel_shot.py --bg photo.jpg --state speaking --level 0.6 --out shot.png
    uv run python tools/panel_shot.py --bg photo.jpg --state listening --strip 8 --seconds 4 --out eye.png
    uv run python tools/panel_shot.py --handed-over --out away.png
    uv run python tools/panel_shot.py --tutorial 7 --step 3 --out steps.png
    uv run python tools/panel_shot.py --usb camera:Endoscope,music:MiniLab 3 --out usb.png
    uv run python tools/panel_shot.py --menu --out menu.png
    uv run python tools/panel_shot.py --wifi --page 2 --out list.png
    uv run python tools/panel_shot.py --keyboard --typed hunter22 --status "wrong" --out kb.png
    uv run python tools/panel_shot.py --offline --out offline.png
    uv run python tools/panel_shot.py --bench

``--strip`` crops the eye and lays N frames of it across a row, which is how motion is looked at
in a still: a face that is alive differs from frame to frame, and one that is a decoration does
not. ``--bench`` prints the per-frame render and composite cost, which is the number the Pi is
short of. Writes a PNG and says where.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

from cyclops import devices, eye, overlay, point, tutorial, wifi

WIDTH, HEIGHT = 800, 480
STRIP_PAD = 12  # picture kept around the eye in each strip cell, so the bezel's shadow is in it
# The scan the picker was designed from: 18 rows off the Pi on 2026-09-26, one of them saved.
SCAN = Path(__file__).resolve().parent.parent / "tests" / "wifi_scan.txt"
SAVED = {"Noise Upstairs 5g"}
AWAKE_ELAPSED = 12.0  # a session twelve seconds old, which is what the clock readouts show
# A real job's steps, as the model would set them out, for --tutorial to take the first N of.
SAMPLE_STEPS = (
    "Turn the water off under the sink",
    "Open the tap to drain the line",
    "Prise off the red and blue cap",
    "Undo the screw under the cap",
    "Pull the handle off",
    "Unscrew the cartridge nut, 17 mm",
    "Swap in the new cartridge",
    "Nut back on, hand tight plus a quarter",
    "Handle, screw and cap back on",
    "Water on, check for drips",
)


def gesture(args: argparse.Namespace) -> point.Gesture | None:
    """The marks asked for on the command line, born at zero. None if none were."""
    if not args.point:
        return None
    return point.Gesture(tuple(point.parse(args.point.encode().decode("unicode_escape"))),
                         None, born=0.0)


def shown(args: argparse.Namespace, src: tuple[int, int] = (WIDTH, HEIGHT)) -> dict:
    """The render() kwargs for the state asked for. Asleep has no session, so no elapsed.

    ``--point`` goes through :func:`cyclops.point.frame_args`, which is the same call the kiosk's
    loop makes - so a gesture rendered here is the one the panel would draw, and not a second
    implementation of it that happens to look similar. *src* is the background photo's own size,
    because marks are fractions of the picture the model was shown rather than of the panel.
    """
    asleep = args.state == overlay.IDLE
    kw = dict(
        state=args.state,
        level=args.level,
        elapsed=None if asleep else AWAKE_ELAPSED,
        recording=args.recording,
        detail=args.detail,
        volume=args.volume,
        temp_c=args.temp,
        handed_over=args.handed_over,
        plugged_in=plugged_in(args.usb),
        menu=args.menu,
        wifi=picker(args),
        offline=args.offline,
    )
    shape = gesture(args)
    if shape is not None:
        kw.update(point.frame_args(shape, args.point_age, *src, WIDTH, HEIGHT))
    if args.tutorial:
        kw["tutorial"] = tutorial.Tutorial(SAMPLE_STEPS[: args.tutorial], args.step - 1)
    return kw


def picker(args: argparse.Namespace) -> wifi.Picker | None:
    """The Wi-Fi picker as --wifi or --keyboard ask for it, off the real scan. None if neither."""
    if not (args.wifi or args.keyboard):
        return None
    nets = wifi.parse_scan(SCAN.read_text(), SAVED)
    made = wifi.Picker(networks=nets, page=max(0, args.page - 1), status=args.status)
    if args.keyboard:
        made.choose(next(n for n in nets if n.ssid == "MidnightFlour"))
        made.typed, made.status = args.typed, args.status
        made.shift, made.symbols, made.show = args.shift, args.symbols, args.show
    return made


def plugged_in(spec: str) -> tuple[devices.Device, ...]:
    """``--usb camera:Endoscope,music:MiniLab 3`` as the list devices.connected() would hand back.

    There is no sysfs on a Mac, so the only way to look at the rail with something on it is to
    say what is on it. The IDs are made up and nothing reads them - what a label draws from is
    the category and the name.
    """
    made = []
    for one in (part.strip() for part in spec.split(",") if part.strip()):
        category, _, name = one.partition(":")
        made.append(devices.Device("0000", f"{len(made):04d}", name.strip() or category,
                                   category.strip()))
    return tuple(made)


def background(path: Path | None) -> tuple[np.ndarray, tuple[int, int]]:
    """The picture behind the chrome, panel-sized, and the size it came in at.

    The second half is what a mark is measured against: the model is shown a 1024-wide photo and
    the panel is 800 wide, so a fraction only lands in the right place if the crop is undone
    against the source's own shape.
    """
    if path is None:
        return np.full((HEIGHT, WIDTH, 3), 24, np.uint8), (WIDTH, HEIGHT)
    rgb = np.asarray(Image.open(path).convert("RGB"))
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    return overlay.fit_to_window(bgr, WIDTH, HEIGHT), (bgr.shape[1], bgr.shape[0])


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
          start: float, marks_at: Callable[[float], dict] | None = None) -> Image.Image:
    """N frames of the eye across a row, so motion can be looked at in a still.

    *marks_at* gives the gesture's kwargs that far into the strip. Without it a pointed-at strip
    would be six copies of one instant; with it the row is the eye actually turning to what was
    marked and letting it go again, which is the whole of what there is to judge.
    """
    box = eye_box(ov)
    cell = (box[2] - box[0], box[3] - box[1])
    out = Image.new("RGB", (cell[0] * frames, cell[1]), (0, 0, 0))
    for i in range(frames):
        step = i * seconds / max(1, frames)
        cell_kw = kw if marks_at is None else {**kw, **marks_at(step)}
        out.paste(shot(ov, frame, cell_kw, start + step).crop(box), (i * cell[0], 0))
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
    ap.add_argument("--handed-over", action="store_true",
                    help="a companion on the LAN is holding his voice")
    ap.add_argument("--detail", default="", help="the controller's own line on the terminal")
    ap.add_argument("--phase", type=float, default=12.0, help="the clock, in seconds")
    ap.add_argument("--strip", type=int, default=0, help="crop the eye and lay N frames in a row")
    ap.add_argument("--seconds", type=float, default=4.0, help="how much time a strip covers")
    ap.add_argument("--bench", action="store_true", help="print ms per frame and exit")
    ap.add_argument("--point", default="", help="marks to point with, one per line: "
                    r"'ring X Y label\nn N X Y\ntag X Y label\narrow X1 Y1 X2 Y2 label'")
    ap.add_argument("--point-age", type=float, default=1.0,
                    help="seconds since the gesture was made; past the hold it is fading")
    ap.add_argument("--tutorial", type=int, default=0,
                    help="a walkthrough of N steps (2-10) on the terminal's top row")
    ap.add_argument("--step", type=int, default=1, help="which of its steps is up, from 1")
    ap.add_argument("--usb", default="", help="what is plugged in, as "
                    "'camera:Endoscope,music:MiniLab 3' - category is camera/music/storage/other")
    ap.add_argument("--menu", action="store_true", help="the long-press menu is up")
    ap.add_argument("--wifi", action="store_true", help="the Wi-Fi list, off the real scan")
    ap.add_argument("--page", type=int, default=1, help="which page of the list, from 1")
    ap.add_argument("--keyboard", action="store_true", help="the password keyboard")
    ap.add_argument("--typed", default="", help="what has been typed on it")
    ap.add_argument("--status", default="", help="the picker's line of news")
    ap.add_argument("--shift", action="store_true")
    ap.add_argument("--symbols", action="store_true")
    ap.add_argument("--show", action="store_true", help="the password in the clear")
    ap.add_argument("--offline", action="store_true", help="the box has no network")
    ap.add_argument("--out", type=Path, default=Path("panel.png"))
    args = ap.parse_args()

    ov = overlay.Overlay(WIDTH, HEIGHT)
    frame, src = background(args.bg)
    kw = shown(args, src)
    if args.bench:
        for state in (overlay.IDLE, overlay.LISTENING, overlay.SPEAKING):
            ov = overlay.Overlay(WIDTH, HEIGHT)
            kw = {**kw, "state": state, "elapsed": None if state == overlay.IDLE else AWAKE_ELAPSED}
            r, c = bench(ov, frame, kw)
            print(f"{state:10s} render {r:5.1f} ms  composite {c:5.1f} ms")
        return
    settle(ov, kw)
    if args.strip:
        shape = gesture(args)
        marks_at = None if shape is None else (
            lambda step: point.frame_args(shape, args.point_age + step, *src, WIDTH, HEIGHT)
        )
        image = strip(ov, frame, kw, args.strip, args.seconds, args.phase, marks_at)
    else:
        image = shot(ov, frame, kw, args.phase)
    image.save(args.out)
    print(f"· {args.out} — {args.state} at phase {args.phase:g}"
          + (f", eye x{args.strip} over {args.seconds:g}s" if args.strip else ""))


if __name__ == "__main__":
    main()
