#!/usr/bin/env python3
"""How much a panel recording shakes, in panel pixels - the before/after number for cyclops.steady.

    uv run python tools/shake.py sessions/.../video.mp4 [more.mp4 ...]
    uv run python tools/shake.py --bench
    uv run python tools/shake.py --synthetic out/ --photo captures/latest.jpg

Jitter is the picture's path minus its own one-second moving average, 90th percentile, measured
by tracking corners in the camera area of an 800x480 panel recording with the chrome masked out.
Tracked features and not phase correlation: phase correlation over the panel reads ~0 on every
session, because the static focus brackets dominate it.

``--bench`` is the stabilizer's own cost per 1280x720 frame, on top of the grey thumbnail the
focus score already pays for. ``--synthetic`` is the check for a builder with no Pi: a photo cut
along a seeded handheld path, rendered through the real panel path without and with the
stabilizer, both written out as mp4 and both scored.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from cyclops import camera, overlay, steady
from cyclops.webcam import FRAME_HEIGHT, FRAME_RATE, FRAME_WIDTH

PANEL = (800, 480)
WINDOW = FRAME_RATE  # the moving average: one second of frames
ENLARGE = 1.6  # the photo is blown up this much so a 1280x720 window has room to wander
TARGET_JITTER = 30.0  # the stapler session's score, which the synthetic clip is sized to
SWAY_HZ, TREMOR_HZ = (0.5, 2.0), (4.0, 8.0)
TREMOR_SHARE = 0.25  # of the sway's amplitude
BENCH_FRAMES = 150


def panel_mask() -> np.ndarray:
    """The camera area of the panel: chrome, focus brackets, eye, gauges and caption bar out."""
    mask = np.zeros(PANEL[::-1], np.uint8)
    mask[60:400, 60:740] = 255
    mask[175:305, 335:465] = 0  # focus brackets
    mask[260:, :260] = 0  # eye
    mask[280:, 610:] = 0  # gauges
    mask[395:, :] = 0  # caption bar
    return mask


def steps(frames) -> np.ndarray:
    """Per-frame translation of the picture, panel px, for each consecutive pair it could track."""
    mask, prev, out = panel_mask(), None, []
    for frame in frames:
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if grey.shape != mask.shape:
            grey = cv2.resize(grey, PANEL)
        if prev is not None:
            corners = cv2.goodFeaturesToTrack(prev, 200, 0.01, 8, mask=mask)
            if corners is not None and len(corners) >= 10:
                moved, found, _ = cv2.calcOpticalFlowPyrLK(prev, grey, corners, None)
                ok = found.ravel() == 1
                if ok.sum() >= 10:
                    fit, _ = cv2.estimateAffinePartial2D(corners[ok], moved[ok])
                    if fit is not None:
                        out.append(fit[:, 2])
        prev = grey
    return np.array(out).reshape(-1, 2)


def jitter(path: np.ndarray) -> np.ndarray:
    """Distance of a path from its own centred one-second average, ends trimmed."""
    box = np.ones(WINDOW) / WINDOW
    smooth = np.stack([np.convolve(path[:, i], box, mode="same") for i in (0, 1)], 1)
    return np.hypot(*(path - smooth)[WINDOW:-WINDOW].T)


def score(frames) -> float:
    return float(np.percentile(jitter(np.cumsum(steps(frames), axis=0)), 90))


def read(path: Path):
    cap = cv2.VideoCapture(str(path))
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        yield frame


# ---------------------------------------------------------------- the synthetic handheld clip


def handheld(frames: int, seed: int) -> np.ndarray:
    """A seeded hand: slow sway plus tremor, camera px per frame, before it is sized."""
    rng = np.random.default_rng(seed)
    t = np.arange(frames) / FRAME_RATE
    path = np.zeros((frames, 2))
    for band, share, count in ((SWAY_HZ, 1.0, 3), (TREMOR_HZ, TREMOR_SHARE, 2)):
        for axis in (0, 1):
            for _ in range(count):
                hz, phase = rng.uniform(*band), rng.uniform(0, 2 * np.pi)
                path[:, axis] += share * rng.uniform(0.5, 1.0) * np.sin(2 * np.pi * hz * t + phase)
    return path


def sized(path: np.ndarray) -> np.ndarray:
    """The path scaled so the panel would see TARGET_JITTER of it (1 panel px = 1.5 camera px)."""
    per_panel = FRAME_HEIGHT / PANEL[1]
    return path * TARGET_JITTER / np.percentile(jitter(path / per_panel), 90)


def camera_frames(photo: np.ndarray, path: np.ndarray):
    """What the camera would have delivered: a 1280x720 window of the photo, moved by the hand."""
    h, w = photo.shape[:2]
    for dx, dy in path:
        yield cv2.getRectSubPix(photo, (FRAME_WIDTH, FRAME_HEIGHT), (w / 2 + dx, h / 2 + dy))


def panel(frames, steadied: bool):
    """The frames through the real panel path, with the stabilizer in front of it or not."""
    ov = overlay.Overlay(*PANEL)
    held = steady.Steady()
    kw = dict(state=overlay.LISTENING, level=0.3, elapsed=12.0)
    for i, frame in enumerate(frames):
        if steadied:
            shift = held.update(camera.thumbnail(frame), frame.shape[1], frame.shape[0])
            frame = steady.live(frame, shift)
        chrome = ov.render(phase=12.0 + i / FRAME_RATE, **kw)
        yield overlay.composite(overlay.fit_to_window(frame, *PANEL), chrome)


def write(frames, out: Path):
    """Encode as the recorder does, and pass the frames through so they can be scored too."""
    enc = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
         "-pix_fmt", "bgr24", "-s", f"{PANEL[0]}x{PANEL[1]}", "-framerate", str(FRAME_RATE),
         "-i", "-", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26",
         "-pix_fmt", "yuv420p", str(out)],
        stdin=subprocess.PIPE,
    )
    for frame in frames:
        enc.stdin.write(frame.tobytes())
        yield frame
    enc.stdin.close()
    enc.wait()


def synthetic(out: Path, photo_path: Path, seconds: float, seed: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    photo = cv2.imread(str(photo_path))
    photo = cv2.resize(photo, None, fx=ENLARGE, fy=ENLARGE, interpolation=cv2.INTER_CUBIC)
    path = sized(handheld(round(seconds * FRAME_RATE), seed))
    np.save(out / "path.npy", path)
    for name, steadied in (("before", False), ("after", True)):
        video = out / f"{name}.mp4"
        list(write(panel(camera_frames(photo, path), steadied), video))
        print(f"{name:7s} {score(read(video)):5.1f} px   {video}")


# ---------------------------------------------------------------- bench


def bench(photo_path: Path | None) -> None:
    """The stabilizer's own ms per 1280x720 frame, on a moving picture so it has work to do."""
    if photo_path is not None:
        photo = cv2.resize(cv2.imread(str(photo_path)), None, fx=ENLARGE, fy=ENLARGE)
    else:
        noise = np.random.default_rng(1).integers(0, 255, (1200, 2000, 3), np.uint8)
        photo = cv2.GaussianBlur(noise, (0, 0), 3)
    frames = list(camera_frames(photo, sized(handheld(BENCH_FRAMES, 5))))
    thumbs = [camera.thumbnail(f) for f in frames]
    held, spent = steady.Steady(), 0.0
    for frame, small in zip(frames, thumbs, strict=True):
        began = time.perf_counter()
        shift = held.update(small, frame.shape[1], frame.shape[0])
        steady.live(frame, shift)
        spent += time.perf_counter() - began
    began = time.perf_counter()
    for frame in frames:
        camera.thumbnail(frame)
    shared = (time.perf_counter() - began) / len(frames) * 1000
    print(f"stabilizer {spent / len(frames) * 1000:5.2f} ms a frame"
          f"   (the shared thumbnail, already paid by the focus score: {shared:4.2f} ms)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="*", type=Path)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--synthetic", type=Path, metavar="DIR")
    ap.add_argument("--photo", type=Path, help="the picture the synthetic clip is cut from")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=12)
    args = ap.parse_args()
    if args.bench:
        bench(args.photo)
    if args.synthetic:
        if args.photo is None:
            ap.error("--synthetic needs --photo")
        synthetic(args.synthetic, args.photo, args.seconds, args.seed)
    for video in args.videos:
        print(f"{score(read(video)):5.1f} px   {video}")


if __name__ == "__main__":
    main()
