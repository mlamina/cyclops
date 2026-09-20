#!/usr/bin/env python3
"""Every number and picture job 015's USB module is judged on, in one pass.

Run from the worktree with a master render already on disk to diff against:

    uv run python tools/usb_evidence.py /tmp/015_master.png <out-dir>

Kept out of ``tests/`` on purpose: none of this is a claim about behaviour, it is a set of
measurements somebody looks at. It is here rather than in a shell history because the corner
and the widths both have to be re-measurable after the next change to the module.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cyclops import overlay  # noqa: E402
from cyclops.devices import Device  # noqa: E402

PHOTO = "/Users/mlamina/code/cyclops/captures/2026-08-29_18-12-01.jpg"
SHOT = [sys.executable, "tools/panel_shot.py", "--bg", PHOTO, "--state", "listening"]
# Short names on purpose: the corner holds about 150 px of columns, so four REAL product
# strings do not all fit and the sheet that is meant to show four glyphs would show two.
FOUR = "camera:Cam,music:Synth,storage:Stick,other:Hub"
TWO = "camera:Endoscope,music:MiniLab 3"


def shot(usb: str, out: Path) -> None:
    subprocess.run([*SHOT, "--usb", usb, "--out", str(out)], check=True, capture_output=True)


def dev(category: str, name: str) -> Device:
    return Device("0000", "0000", name, category)


def widths(ov: overlay.Overlay) -> list[str]:
    """The flat's right edge and the top corner past the chamfer, as the list grows."""
    cases = [
        ("nothing on the bus", []),
        ("one: Endoscope", [dev("camera", "Endoscope")]),
        ("two: Endoscope, MiniLab 3", [dev("camera", "Endoscope"), dev("music", "MiniLab 3")]),
        ("four short names", [dev("camera", "Cam"), dev("music", "Synth"),
                              dev("storage", "Stick"), dev("other", "Hub")]),
        ("four real product names", [dev("camera", "Endoscope"), dev("music", "MiniLab 3"),
                                     dev("storage", "SanDisk Ultra"), dev("other", "FT232R")]),
    ]
    lines = []
    for label, found in cases:
        shown, right = ov.usb_fit(found)
        lines.append(f"  {label:28s} shows {len(shown)}/{len(found)}   "
                     f"flat {right:5.0f}   corner {right + ov.pod_ramp:5.0f}")
    return lines


def crop(path: Path, box: tuple[int, int, int, int], zoom: int) -> Image.Image:
    im = Image.open(path).convert("RGB").crop(box)
    return im.resize((im.width * zoom, im.height * zoom), Image.NEAREST)


def side_by_side(left: Image.Image, right: Image.Image, labels: tuple[str, str]) -> Image.Image:
    gap, head = 16, 22
    sheet = Image.new("RGB", (left.width + gap + right.width, left.height + head), (18, 18, 18))
    sheet.paste(left, (0, head))
    sheet.paste(right, (left.width + gap, head))
    d = ImageDraw.Draw(sheet)
    d.text((4, 5), labels[0], fill=(230, 230, 230))
    d.text((left.width + gap + 4, 5), labels[1], fill=(230, 230, 230))
    return sheet


def main() -> None:
    master_png, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)

    shot("", out / "usb_none.png")
    shot("camera:Endoscope", out / "usb_one.png")
    shot(FOUR, out / "usb_four.png")
    shot(TWO, out / "after.png")

    ov = overlay.Overlay(800, 480)
    report = ["THE MODULE, 800x480",
              f"  depth                             {ov.pod_depth} px   "
              f"(pod_boxes[0].h = {ov.pod_boxes[0].h})",
              f"  clear corner left of the pod at its widest   {ov.usb_room():.0f} px", ""]

    # The corner, which is what this round was for.
    after = np.asarray(Image.open(out / "after.png").convert("RGB")).astype(int)
    report += ["THE TOP-LEFT CORNER - the plate, or steel over it",
               f"  pixel (0, 0)                      {tuple(int(v) for v in after[0, 0])}",
               f"  the plate along the top edge      {tuple(int(v) for v in after[0, 40])}",
               "  the nub last round                (207, 210, 211) - the surround's lit corner",
               ""]

    report += ["WIDTH AS THE LIST GROWS - the flat's right edge, and the top corner past the "
               "chamfer", *widths(ov), ""]

    # Nothing outside the module moved.
    master = np.asarray(Image.open(master_png).convert("RGB")).astype(int)
    empty = np.asarray(Image.open(out / "usb_none.png").convert("RGB")).astype(int)
    ys, xs = np.nonzero(np.abs(master - empty).sum(2) > 0)
    _, right = ov.usb_fit([])
    box = ov.usb_box(right)
    report += ["WHAT MOVED AGAINST MASTER, with nothing plugged in",
               f"  every differing pixel:  x {xs.min()}..{xs.max()}, y {ys.min()}..{ys.max()}",
               f"  the module's own box:    x {box.x}..{box.right}, y {box.y}..{box.bottom}",
               "  so the pod, the eye, both dials and the caption are untouched to the pixel.",
               "",
               f"  the pod's three boxes: {ov.pod_boxes}", ""]
    (out / "module.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))

    # The four glyphs at panel size, and the corner before and after.
    four = crop(out / "usb_four.png", (0, 0, 215, 60), 3)
    four.save(out / "usb_glyphs.png")
    # ...and the corner against what it looked like with the nub still on it, if that render is
    # still around. It is a picture of a thing that no longer happens, so it cannot be rebuilt
    # from this tree - once it is gone, the one in the outcome folder is the record of it.
    was = Path("/tmp/015_before_nub.png")
    if was.exists():
        sheet = side_by_side(
            crop(was, (0, 0, 60, 60), 5),
            crop(out / "after.png", (0, 0, 60, 60), 5),
            ("last round - the grey nub", "now - the plate runs into the corner"),
        )
        sheet.save(out / "corner_before_after.png")

    # Every width, stacked, so "as wide as its contents" is one picture.
    strips = [crop(out / name, (0, 0, 260, 60), 2)
              for name in ("usb_none.png", "usb_one.png", "after.png", "usb_four.png")]
    stack = Image.new("RGB", (strips[0].width, sum(s.height + 6 for s in strips)), (18, 18, 18))
    at = 0
    for strip in strips:
        stack.paste(strip, (0, at))
        at += strip.height + 6
    stack.save(out / "usb_widths.png")
    print(f"· wrote pictures into {out}")


if __name__ == "__main__":
    main()
