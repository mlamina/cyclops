"""The time-lapse recorder alone: a run, its folder, its stops, what an unplug leaves (hq#669).

No ffmpeg, camera or Pi. ffmpeg is a stand-in that counts the bytes it was piped and, asked to
remux, writes a file where the real one would. Frames come every 0 s instead of every second.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cyclops import card, lapse

FRAME = np.zeros((1296, 2304, 3), dtype=np.uint8)
PLENTY = 50 * 1024**3


class Pipe:
    def __init__(self, broken_after: int | None = None) -> None:
        self.frames, self.closed, self._broken_after = 0, False, broken_after

    def write(self, data: bytes) -> None:
        if self._broken_after is not None and self.frames >= self._broken_after:
            raise BrokenPipeError("ffmpeg died")
        self.frames += 1

    def close(self) -> None:
        self.closed = True


class FakeFFmpeg:
    """Popen for the encoder (writes the raw file as it starts) and run for remux and ffprobe."""

    def __init__(self, broken_after: int | None = None, remux_fails: bool = False) -> None:
        self.pipe = Pipe(broken_after)
        self.remux_fails = remux_fails
        self.argv: list[list[str]] = []

    def popen(self, argv, **_):
        self.argv.append(argv)
        Path(argv[-1]).write_bytes(b"fragmented")
        return SimpleNamespace(stdin=self.pipe, wait=lambda timeout=None: 0)

    def run(self, argv, **_):
        self.argv.append(argv)
        if argv[0] == "ffprobe":
            return SimpleNamespace(
                returncode=0, stdout=f"{self.pipe.frames / 30:.3f}\n", stderr=b""
            )
        if self.remux_fails:
            return SimpleNamespace(returncode=1, stdout=b"", stderr=b"moov atom not found\n")
        Path(argv[-1]).write_bytes(b"playable")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")


@pytest.fixture(autouse=True)
def fast(monkeypatch) -> None:
    monkeypatch.setattr(lapse, "CAPTURE_EVERY_S", 0.0)
    monkeypatch.setattr(lapse, "SPACE_EVERY_S", 0.0)


def recorder(tmp_path: Path, ff: FakeFFmpeg, free=lambda _p: PLENTY, frames=lambda: FRAME, **more):
    return lapse.TimeLapse(tmp_path, frames, popen=ff.popen, run=ff.run, free_bytes=free, **more)


def wait_for(condition, timeout: float = 2.0) -> None:
    end = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.001)


def test_a_run_is_live_while_it_records_and_a_finished_session_after(tmp_path) -> None:
    ff = FakeFFmpeg()
    lapser = recorder(tmp_path, ff)
    assert lapser.start() == ""
    wait_for(lambda: lapser.frames >= 5)
    folder = lapser.folder
    assert folder.name.endswith("_time-lapse") and lapse.is_lapse(folder)
    assert card.triage(folder).verdict == "live"  # recovery leaves it alone mid-run

    last = lapser.stop()
    state = card.triage(folder)
    assert (
        state.verdict == "finished"
        and state.video
        and not state.parts
        and state.summary
        and state.named
    )
    assert not lapser.running() and last["saved"] and last["stopped"] == lapse.ASKED
    assert last["seconds"] == pytest.approx(ff.pipe.frames / 30, abs=0.01)


def test_the_panel_length_is_frames_over_thirty(tmp_path) -> None:
    lapser = recorder(tmp_path, FakeFFmpeg())
    lapser.frames = 360
    assert lapser.seconds() == 12.0


def test_the_encoder_writes_fragments_and_no_sound(tmp_path) -> None:
    ff = FakeFFmpeg()
    lapser = recorder(tmp_path, ff)
    lapser.start()
    lapser.stop()
    argv = ff.argv[0]
    assert "-an" in argv and "2304x1296" in argv and argv[argv.index("-framerate") + 1] == "30"
    assert (
        "frag_keyframe" in argv[argv.index("-movflags") + 1] and argv[argv.index("-g") + 1] == "60"
    )


def test_a_second_run_is_refused_while_one_records(tmp_path) -> None:
    lapser = recorder(tmp_path, FakeFFmpeg())
    lapser.start()
    try:
        assert lapser.start() != "" and len(list(tmp_path.iterdir())) == 1
    finally:
        lapser.stop()


@pytest.mark.parametrize(
    "free, frames",
    [(lambda _p: 1024**3, lambda: FRAME), (lambda _p: PLENTY, lambda: None)],
    ids=["no space", "no camera picture"],
)
def test_a_refused_start_leaves_nothing_on_the_card(tmp_path, free, frames) -> None:
    lapser = recorder(tmp_path, FakeFFmpeg(), free=free, frames=frames)
    assert lapser.start() != "" and not lapser.running() and list(tmp_path.iterdir()) == []


def test_it_stops_itself_at_the_space_line_saves_and_says_why(tmp_path) -> None:
    left = iter([PLENTY] + [PLENTY] * 3 + [1024**3] * 100)
    told: list[str] = []
    lapser = recorder(tmp_path, FakeFFmpeg(), free=lambda _p: next(left), on_stop=told.append)
    lapser.start()
    wait_for(lambda: not lapser.running())
    assert (
        told == [lapse.NO_SPACE]
        and lapser.last["saved"]
        and lapser.last["stopped"] == lapse.NO_SPACE
    )
    assert "nearly full" in (lapser.folder / card.SUMMARY_NAME).read_text()


def test_a_broken_encoder_stops_the_run_and_keeps_what_was_written(tmp_path) -> None:
    told: list[str] = []
    lapser = recorder(tmp_path, FakeFFmpeg(broken_after=3), on_stop=told.append)
    lapser.start()
    wait_for(lambda: not lapser.running())
    assert (
        told == [lapse.BROKE] and lapser.last["saved"] and card.written(lapser.folder / card.VIDEO)
    )


def test_a_stalled_camera_repeats_the_last_picture_instead_of_leaving_a_gap(tmp_path) -> None:
    pictures = iter([FRAME, FRAME] + [None] * 1000)
    ff = FakeFFmpeg()
    lapser = recorder(tmp_path, ff, frames=lambda: next(pictures, None))
    lapser.start()
    wait_for(lambda: lapser.frames >= 20)
    lapser.stop()
    assert ff.pipe.frames == lapser.frames >= 20


# ---- after an unplug ----


def unplugged(tmp_path: Path) -> Path:
    """What a power cut mid-run leaves: the mark, a log nobody holds, and a raw file in parts/."""
    folder = tmp_path / "2026-11-08_01-00-00_time-lapse"
    (folder / card.PARTS).mkdir(parents=True)
    (folder / card.PARTS / lapse.RAW).write_bytes(b"fragments up to the unplug")
    (folder / lapse.MARK).write_text('{"started": "2026-11-08_01-00-00"}')
    (folder / card.LOG_NAME).write_text(
        '{"t": 0, "type": "session"}\n{"t": 0, "type": "timelapse"}\n'
    )
    return folder


def test_recovery_turns_an_unplugged_run_into_a_playable_video(tmp_path) -> None:
    folder = unplugged(tmp_path)
    assert card.triage(folder).verdict == "unfinished"
    ok, _ = lapse.finish(folder, FakeFFmpeg().run)
    state = card.triage(folder)
    assert ok and state.verdict == "finished" and state.video and not state.parts


def test_a_remux_that_fails_keeps_the_footage_for_the_next_boot(tmp_path) -> None:
    folder = unplugged(tmp_path)
    ok, why = lapse.finish(folder, FakeFFmpeg(remux_fails=True).run)
    assert not ok and "moov" in why and card.written(folder / card.PARTS / lapse.RAW)
    assert not (folder / card.VIDEO).exists()


def test_real_ffmpeg_reads_a_fragmented_file_cut_off_mid_run(tmp_path) -> None:
    """The one claim the whole unplug story rests on, with the real ffmpeg when there is one."""
    if subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0:
        pytest.skip("no ffmpeg here")
    folder = unplugged(tmp_path)
    raw = folder / card.PARTS / lapse.RAW
    frame = np.zeros((96, 128, 3), dtype=np.uint8)
    enc = subprocess.Popen(
        lapse.encoder_argv(128, 96, raw), stdin=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    for n in range(150):
        frame[:] = n
        enc.stdin.write(frame.tobytes())
    enc.stdin.flush()
    time.sleep(0.2)
    enc.kill()  # no clean close: the last fragment is torn, as an unplug leaves it
    enc.wait()
    ok, why = lapse.finish(folder)
    assert ok, why
    assert lapse.probe_seconds(folder / card.VIDEO) >= 2.0  # at least the first two full fragments


def test_boot_recovery_finishes_a_time_lapse_without_the_sound_a_session_mux_needs(
    tmp_path, monkeypatch
) -> None:
    from cyclops import session

    folder = unplugged(tmp_path)
    monkeypatch.setattr(subprocess, "run", FakeFFmpeg().run)
    session._fix(folder)
    state = card.triage(folder)
    assert state.verdict == "finished" and state.video and state.summary and state.named


def finished(tmp_path: Path, name: str, lapsed: bool) -> Path:
    folder = tmp_path / name
    folder.mkdir()
    for file in (card.VIDEO, card.PAGE_NAME, card.SUMMARY_NAME):
        (folder / file).write_text("x")
    (folder / card.LOG_NAME).write_text('{"t": 0, "type": "recording_asked"}\n')
    if lapsed:
        (folder / lapse.MARK).write_text("{}")
    return folder


def test_filing_and_the_highlight_cut_leave_a_time_lapse_alone(tmp_path) -> None:
    from cyclops import cut, projects

    talk = finished(tmp_path, "2026-11-08_01-00-00_bench", lapsed=False)
    finished(tmp_path, "2026-11-08_02-00-00_time-lapse", lapsed=True)
    assert projects.unfiled(SimpleNamespace(sessions_dir=tmp_path)) == [talk]
    assert cut.work(tmp_path) == cut.Work(talk, -1)
