"""Device probe for the time-lapse (arc-eng/hq#669): the real camera and recorder, nothing else.

Writes only into the folder it's given: no session, no filing, no conversation. The kiosk must be
stopped first, since it owns the camera. Prints one JSON line every ``--every`` seconds.

    python tools/lapse_probe.py OUT --minutes 12            # a run, stopped and saved
    python tools/lapse_probe.py OUT --minutes 1 --camera-only   # the camera alone at 720p, for a baseline
    python tools/lapse_probe.py OUT --minutes 99 --no-stop  # left running, for the power-cut test
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cyclops import camera, lapse  # noqa: E402

TICK = os.sysconf("SC_CLK_TCK")


def _tree_ticks() -> int:
    """CPU ticks used by this process and everything it started (ffmpeg, rpicam-vid)."""
    total = 0
    me = os.getpid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
        except OSError:
            continue
        pid, ppid = int(entry.name), int(fields[1])
        if pid == me or ppid == me:
            total += int(fields[11]) + int(fields[12])
    return total


def _temp() -> float | None:
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
    except (OSError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--minutes", type=float, default=12)
    ap.add_argument("--every", type=float, default=30)
    ap.add_argument("--camera-only", action="store_true")
    ap.add_argument("--no-stop", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cam = camera.CameraSource(None)
    cam.start()
    cam.wait_for_frame(10)
    rec = lapse.TimeLapse(args.out, cam.full_frame, hold=cam.hold_full)
    if not args.camera_only:
        print(json.dumps({"start": rec.start() or "started", "camera": cam.index}), flush=True)
    t0, ticks0 = time.monotonic(), _tree_ticks()
    last_t, last_ticks = t0, ticks0
    end = t0 + args.minutes * 60
    while time.monotonic() < end:
        time.sleep(args.every)
        now, ticks = time.monotonic(), _tree_ticks()
        full = cam.full_frame()
        print(json.dumps({
            "t_s": round(now - t0), "frames": rec.frames, "panel_s": round(rec.seconds(), 2),
            "cpu_pct_of_one_core": round(100 * (ticks - last_ticks) / TICK / (now - last_t), 1),
            "temp_c": _temp(), "frame": None if full is None else list(full.shape[1::-1]),
        }), flush=True)  # fmt: skip
        last_t, last_ticks = now, ticks
    if args.no_stop:
        return
    spent = time.monotonic() - t0
    print(json.dumps({"mean_cpu_pct_of_one_core": round(100 * (_tree_ticks() - ticks0) / TICK / spent, 1)}))
    if not args.camera_only:
        shown = rec.seconds()
        last = rec.stop()
        print(json.dumps({"panel_last_s": round(shown, 2), **last}), flush=True)
        video = args.out / last["folder"] / "video.mp4"
        print(json.dumps({"video_bytes": video.stat().st_size if video.exists() else 0}), flush=True)
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                "stream=width,height,r_frame_rate,nb_frames", "-of", "json", str(video)],
                               capture_output=True, text=True)  # fmt: skip
        print(probe.stdout.replace("\n", " "), flush=True)
    cam.stop()


if __name__ == "__main__":
    main()
