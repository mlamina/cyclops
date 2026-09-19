"""The whole screen as a session's video: the capture, the camera it falls back to, and the end.

No compositor here, so ``wf-recorder`` is played by a few lines of Python that take the same
arguments and write the same thing - raw frames, red byte first as the Pi's writes them, into the file named after ``-f``, after
reading the answer to "overwrite?" off stdin. Everything on this side of the pipe is the real
thing: the FIFO, the reader, the recorder, ffmpeg and the mux.

The recorder under the session log is a stand-in in the tests that are about which source a
session films and when the capture ends, because encoding is not what they are about.
"""

from __future__ import annotations

import subprocess
import sys
import time
from types import SimpleNamespace

import cv2
import pytest

from cyclops import card, record, session
from cyclops.config import Settings
from cyclops.screen import ScreenSource

SIZE = (160, 96)
PIXEL = (10, 200, 30)  # BGR; the fake's screen is one flat colour, so its frames are recognisable

CAPTURE = """
import signal, sys, time
signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
path = sys.argv[sys.argv.index("-f") + 1]
sys.stdin.readline()
frame = bytes([{r}, {g}, {b}, 0]) * ({w} * {h})
with open(path, "wb", buffering=0) as out:
    while True:
        out.write(frame)
        time.sleep(1 / 30)
"""

BROKEN = """
import sys
sys.stderr.write("compositor does not support wlr-screencopy\\n")
sys.exit(1)
"""


def fake(tmp_path, name: str, body: str) -> str:
    """An executable that stands in for wf-recorder."""
    path = tmp_path / name
    path.write_text(f"#!{sys.executable} -S\n" + body)
    path.chmod(0o755)
    return str(path)


def capturing(tmp_path) -> ScreenSource:
    b, g, r = PIXEL
    body = CAPTURE.format(b=b, g=g, r=r, w=SIZE[0], h=SIZE[1])
    return ScreenSource(SIZE, program=fake(tmp_path, "capture", body))


# ---------------------------------------------------------------- into the recorder


def test_a_capture_of_the_screen_becomes_a_30_fps_video_of_the_right_length(tmp_path) -> None:
    screen = capturing(tmp_path)
    try:
        assert screen.start() == "", "the capture's first frame is in"
        recorder = record.SessionRecorder(screen, tmp_path / "session", fps=record.DEFAULT_FPS)
        assert recorder.start()
        time.sleep(0.3)
        # One block on each track, now: a track that starts late is padded from the start, so
        # both are as long as the recording and -shortest cuts nothing that matters.
        recorder.on_mic_block(bytes(960))
        recorder.on_speaker_block(bytes(960))
        ran = time.monotonic() - recorder._t0
        video = recorder.stop()
    finally:
        screen.stop()

    assert video is not None, recorder.failed
    probe = subprocess.run(  # noqa: S603
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=r_frame_rate,duration", "-of", "csv=p=0", str(video)],
        capture_output=True, check=True, timeout=30, text=True,
    )
    rate, seconds = probe.stdout.strip().split(",")
    assert rate == "30/1"
    assert abs(float(seconds) - ran) <= 1 / 30, f"{seconds} s of video for {ran:.3f} s recorded"
    ok, first = cv2.VideoCapture(str(video)).read()
    assert ok and all(abs(float(first[..., c].mean()) - PIXEL[c]) < 12 for c in range(3)), (
        "the frames in the video are the capture's"
    )


# ---------------------------------------------------------------- which source a session films


class Recorder:
    """The session's recorder, minus the encoding: it only remembers what it was pointed at."""

    def __init__(self, frames, folder, **kw) -> None:
        self.frames = frames
        self.failed = ""
        Recorder.made.append(self)

    def start(self) -> bool:
        return True

    def on_mic_block(self, block: bytes) -> None: ...

    def on_speaker_block(self, block: bytes) -> None: ...

    def stop(self):
        return None


class Agent:
    on_event = None


@pytest.fixture
def filmed(tmp_path, monkeypatch):
    """Run a kiosk session around ``screen``; say what it filmed and what the log says."""
    monkeypatch.setattr(session, "SessionRecorder", Recorder)
    Recorder.made = []
    camera = SimpleNamespace(frame=lambda: None)
    settings = Settings(api_key="", sessions_dir=tmp_path / "sessions")

    def run(screen: ScreenSource, *, during=lambda: None):
        log = session.SessionLog(
            settings, Agent(), entrypoint="kiosk",
            mic=SimpleNamespace(on_block=None), speaker=SimpleNamespace(on_block=None),
            frames=camera, screen=screen,
        )
        with log:
            log.event("you", text="is this being recorded?")  # so the folder is kept
            during()
        records, _ = card.read_log(log.dir / card.LOG_NAME)
        said = next(r for r in records if r["type"] == "recording")
        return Recorder.made[0].frames, camera, said

    return run


@pytest.mark.parametrize("program", ["missing", "broken"])
def test_a_screen_that_cannot_be_captured_films_the_camera_and_says_why(
    tmp_path, filmed, program
) -> None:
    if program == "missing":
        screen = ScreenSource(SIZE, program=str(tmp_path / "wf-recorder"))
    else:
        screen = ScreenSource(SIZE, program=fake(tmp_path, "broken", BROKEN))
    frames, camera, said = filmed(screen)
    assert frames is camera
    assert said["source"] == "camera" and said["why"]
    if program == "broken":
        assert "wlr-screencopy" in said["why"], "the capture's own last word is the reason"


def test_the_capture_ends_with_the_session_even_when_the_session_fell_over(
    tmp_path, filmed
) -> None:
    screen = capturing(tmp_path)
    running = []

    def fall_over() -> None:
        running.append(screen._proc)
        raise RuntimeError("the session fell over")

    with pytest.raises(RuntimeError):
        filmed(screen, during=fall_over)
    (proc,) = running
    assert proc is not None, "the session was filming the screen"
    assert proc.poll() is not None, "and the capture did not outlive it"
