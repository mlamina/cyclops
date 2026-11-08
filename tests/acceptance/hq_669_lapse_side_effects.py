"""qa checks for arc-eng/hq#669 that need no device: real ffmpeg, a scratch folder, a tiny real disk.

Usage: uv run python tests/acceptance/hq_669_lapse_side_effects.py SMALL_VOLUME_PATH WORK_DIR
SMALL_VOLUME_PATH is a mounted, nearly empty, small volume (qa used a 64 MB disk image).
Touches nothing outside those two folders.
"""

from __future__ import annotations

import fcntl
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cyclops import card, cut, lapse, library, session  # noqa: E402


def encode(raw: Path, seconds: int, size: str = "1280x720") -> None:
    raw.parent.mkdir(parents=True, exist_ok=True)
    width, height = size.split("x")
    argv = lapse.encoder_argv(int(width), int(height), raw)
    gen = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
         f"nullsrc=s={size}:r=30:d={seconds},geq=random(1)*255:128:128",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE,
    )  # fmt: skip
    subprocess.run(argv, stdin=gen.stdout, check=True)
    gen.wait()


def lapse_folder(root: Path, name: str, mark: str) -> Path:
    folder = root / name
    (folder / card.PARTS).mkdir(parents=True)
    (folder / card.LOG_NAME).write_text('{"t":0,"type":"session","started":"x"}\n')
    (folder / lapse.MARK).write_text(mark)
    return folder


def check_no_room_to_remux(volume: Path) -> None:
    """The 1 GB stop, scaled down: the run's file is bigger than the free space left on the card."""
    folder = lapse_folder(volume, "2026-11-08_10-00-00_time-lapse", '{"stopped":"space"}')
    raw = folder / card.PARTS / lapse.RAW
    encode(raw, 4)
    free = shutil.disk_usage(volume).free
    print(f"[space] raw={raw.stat().st_size} free={free} raw_bigger_than_free={raw.stat().st_size > free}")
    print(f"[space] finish -> {lapse.finish(folder)}")
    print(f"[space] video.mp4 exists: {(folder / card.VIDEO).exists()}; raw kept: {raw.exists()}")


def check_recovered_summary(work: Path) -> None:
    """A run that was unplugged: lapse.json as _open wrote it, nothing added at stop."""
    folder = lapse_folder(work, "2026-11-08_11-00-00_time-lapse", '{"started":"x","size":[1280,720]}')
    encode(folder / card.PARTS / lapse.RAW, 2)
    print(f"[recover] finish -> {lapse.finish(folder)}")
    print(f"[recover] summary: {(folder / card.SUMMARY_NAME).read_text()!r}")


def check_recap(work: Path) -> None:
    """What the next conversation is told about the last session, once a time-lapse has been saved."""
    root = work / "recap"
    root.mkdir()
    talk = root / "2026-11-08_09-00-00_fixing-the-amp"
    talk.mkdir()
    body = "# Fixing the amp\n\nReflowed C12, still hums at idle; next is the ground lift.\n"
    (talk / card.SUMMARY_NAME).write_text(body)
    (talk / card.PAGE_NAME).write_text(body)
    settings = SimpleNamespace(sessions_dir=root)
    print(f"[recap] before: {session.recent_context(settings).text!r}")
    folder = lapse_folder(root, "2026-11-08_09-30-00_time-lapse", '{"ran_s":720,"stopped":"asked"}')
    encode(folder / card.PARTS / lapse.RAW, 1)
    lapse.finish(folder)
    print(f"[recap] after:  {session.recent_context(settings).text!r}")


def check_live(work: Path) -> None:
    """A run in progress, seen by the two things that treat a flocked folder as a conversation."""
    root = work / "live"
    root.mkdir()
    folder = lapse_folder(root, "2026-11-08_12-00-00_time-lapse", "{}")
    with (folder / card.LOG_NAME).open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        print(f"[live] cut.busy -> {cut.busy(root)!r}")
        print(f"[live] library.live -> {library.live(root)!r}")


if __name__ == "__main__":
    volume_arg, work_arg = Path(sys.argv[1]), Path(sys.argv[2])
    work_arg.mkdir(parents=True, exist_ok=True)
    check_no_room_to_remux(volume_arg)
    check_recovered_summary(work_arg)
    check_recap(work_arg)
    check_live(work_arg)
