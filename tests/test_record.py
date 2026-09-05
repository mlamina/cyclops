"""What a session's video is of, at what size, and what reaches its microphone track.

None of it needs ffmpeg, a camera or a panel. The note is a file, the panel source is one
attribute, the size the encoder is opened at is arithmetic on a frame's shape, and the mic hook
is handed a stand-in track that only remembers what it was given. What is being guarded here is
a set of silent failures: a recording that is quietly of the wrong thing, one that is quietly
resampled to mush, a frame the render loop goes on writing into after the recorder has been
handed it, and a microphone track with holes cut in it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cyclops import filming
from cyclops.config import ConfigError, Settings, _record_source
from cyclops.overlay import composite
from cyclops.record import PanelSource, SessionRecorder

PANEL = (800, 480)  # the official 7" panel, and what a screen recording is
CAMERA = (640, 480)  # the endoscope, and what a camera recording is

ENV_SCREEN = Settings(api_key="", record_source="screen")
ENV_CAMERA = Settings(api_key="", record_source="camera")


def frame(width: int, height: int):
    return np.zeros((height, width, 3), dtype=np.uint8)


def size_for(frame_size: tuple[int, int], width: int) -> tuple[int, int]:
    """What the encoder would be opened at for this source and this ``CYCLOPS_RECORD_WIDTH``."""
    recorder = SessionRecorder(PanelSource(), Path("unused"), width=width)
    return recorder._output_size(frame(*frame_size))


# ---- the note ----


def test_no_note_leaves_the_environment_to_decide(tmp_path: Path) -> None:
    note = tmp_path / "record-source"
    assert filming.requested(note) is None
    assert filming.chosen(ENV_SCREEN, note) == filming.SCREEN
    assert filming.chosen(ENV_CAMERA, note) == filming.CAMERA


def test_the_switch_survives_a_round_trip(tmp_path: Path) -> None:
    note = tmp_path / "cache" / "record-source"  # the directory does not exist yet, as at boot
    filming.request(filming.CAMERA, note)
    assert filming.requested(note) == filming.CAMERA
    filming.request(filming.SCREEN, note)
    assert filming.requested(note) == filming.SCREEN


def test_the_switch_beats_the_environment(tmp_path: Path) -> None:
    """The whole point: it is the one of the two you can reach without an ssh session."""
    note = tmp_path / "record-source"
    filming.request(filming.CAMERA, note)
    assert filming.chosen(ENV_SCREEN, note) == filming.CAMERA
    assert filming.on_screen(ENV_SCREEN, note) is False
    filming.request(filming.SCREEN, note)
    assert filming.chosen(ENV_CAMERA, note) == filming.SCREEN
    assert filming.on_screen(ENV_CAMERA, note) is True


def test_an_unreadable_note_is_not_an_answer(tmp_path: Path) -> None:
    """A preference must never cost a recording; anything unrecognised means "nobody said"."""
    note = tmp_path / "record-source"
    note.write_text("the ceiling\n")
    assert filming.requested(note) is None
    assert filming.chosen(ENV_SCREEN, note) == filming.SCREEN


def test_a_typo_in_the_environment_is_caught_at_startup(monkeypatch) -> None:
    """Not at the tap. A box that will not record what you meant should say so while you watch."""
    monkeypatch.setenv("CYCLOPS_RECORD_SOURCE", "sideways")
    with pytest.raises(ConfigError):
        _record_source("CYCLOPS_RECORD_SOURCE")
    monkeypatch.setenv("CYCLOPS_RECORD_SOURCE", "SCREEN")  # case is not a typo
    assert _record_source("CYCLOPS_RECORD_SOURCE") == filming.SCREEN


# ---- the size the encoder is opened at ----


def test_the_panel_is_recorded_pixel_for_pixel() -> None:
    """The default records the source's own width, so nothing resamples the chrome to mush."""
    assert size_for(PANEL, width=0) == PANEL
    assert size_for(CAMERA, width=0) == CAMERA


def test_a_width_is_a_ceiling_and_never_an_upscale() -> None:
    assert size_for((1280, 720), width=800) == (800, 450)
    assert size_for(CAMERA, width=800) == CAMERA, "800 must not blow a 640 frame up"
    assert size_for(PANEL, width=640) == (640, 384), "what setting a width still costs you"


def test_both_dimensions_come_back_even_for_yuv420p() -> None:
    """An odd dimension is rejected by the pixel format, not by us, and only at encode time."""
    for width in (0, 1281):
        out_w, out_h = size_for((1281, 721), width=width)
        assert out_w % 2 == 0 and out_h % 2 == 0


# ---- the panel as a source ----


def test_the_panel_hands_back_the_last_frame_painted() -> None:
    panel = PanelSource()
    assert panel.frame() is None, "before the first paint, which is what makes start() decline"
    first, second = frame(*PANEL), frame(*PANEL)
    panel.publish(first)
    assert panel.frame() is first
    panel.publish(second)
    assert panel.frame() is second


def test_a_composited_frame_is_the_recorder_s_to_keep() -> None:
    """The one real hazard, and the reason PanelSource needs no lock.

    The render loop hands a frame over and immediately builds the next one. That is only safe
    while `composite` allocates - its docstring says "in place-ish", and the day it becomes
    literally in-place the recorder starts encoding a buffer being rewritten underneath it, on
    the Pi, in the video, where nothing else would notice.
    """
    canvas = np.zeros((480, 800, 3), dtype=np.uint8)
    chrome = np.zeros((480, 800, 4), dtype=np.uint8)
    published = composite(canvas, chrome)
    assert published is not canvas
    published[:] = 255
    assert not canvas.any(), "composite handed back a view of the canvas, not a frame of its own"


def test_the_recorder_takes_a_panel_wherever_it_takes_a_camera() -> None:
    """Protocol conformance, which is the whole of the seam the switch flips."""
    recorder = SessionRecorder(PanelSource(), Path("unused"))
    assert recorder._frames.frame() is None, "and start() declines rather than raising"


# ---- the tap, which is where the two are actually chosen between ----


class CameraStandIn:
    """A FrameSource that is plainly not the panel."""

    def frame(self):
        return None


class FakeController:
    """Just enough SessionController for _toggle_session: what it asks, and what it is told."""

    def __init__(self, settings) -> None:
        self.settings = settings
        self.frames = "never set"
        self.started = False

    def status(self) -> dict:
        return {"state": "idle"}  # so the tap reads as "start", the branch that chooses a source

    def set_record_source(self, frames) -> None:
        self.frames = frames

    def start(self) -> None:
        self.started = True


def tap(settings: Settings, note: Path, monkeypatch):
    """Press WAKE UP on a kiosk whose two sources are told apart, and see which it hands over."""
    from cyclops import kiosk as kiosk_module

    kiosk = object.__new__(kiosk_module.Kiosk)  # the render loop's fields are not under test
    kiosk.controller = FakeController(settings)
    kiosk.panel, kiosk.camera = PanelSource(), CameraStandIn()
    kiosk._pending = kiosk._pending_at = None
    # The kiosk calls `chosen` with the settings alone and takes the default for the path, so
    # to point it at this test's note rather than at ~/.cache the whole function is replaced.
    def chosen(settings, path=note):
        return filming.requested(path) or settings.record_source

    monkeypatch.setattr(filming, "chosen", chosen)
    kiosk._toggle_session()
    assert kiosk.controller.started, "choosing a source must not cost the tap its session"
    return kiosk.controller.frames, kiosk


def test_the_tap_hands_over_the_panel_when_the_switch_says_screen(tmp_path, monkeypatch) -> None:
    note = tmp_path / "record-source"
    filming.request(filming.SCREEN, note)
    frames, kiosk = tap(ENV_CAMERA, note, monkeypatch)  # the switch beats .env here too
    assert frames is kiosk.panel


def test_the_tap_hands_over_the_camera_when_the_switch_says_camera(tmp_path, monkeypatch) -> None:
    note = tmp_path / "record-source"
    filming.request(filming.CAMERA, note)
    frames, kiosk = tap(ENV_SCREEN, note, monkeypatch)
    assert frames is kiosk.camera


def test_with_no_note_the_tap_follows_the_environment(tmp_path, monkeypatch) -> None:
    note = tmp_path / "record-source"
    frames, kiosk = tap(ENV_SCREEN, note, monkeypatch)
    assert frames is kiosk.panel
    frames, kiosk = tap(ENV_CAMERA, note, monkeypatch)
    assert frames is kiosk.camera


# ---- the microphone track ----


class Track:
    """A stand-in for the recorder's ``_Track``: it remembers, and it does no I/O."""

    def __init__(self) -> None:
        self.written: list[bytes] = []

    def append(self, pcm: bytes, at: float) -> None:
        self.written.append(pcm)


def recorder_with_a_track() -> tuple[SessionRecorder, Track]:
    recorder = SessionRecorder(PanelSource(), Path("unused"))
    track = Track()
    recorder._user = track  # what start() would have opened, minus the wave file
    return recorder, track


def test_every_block_the_microphone_heard_reaches_the_track() -> None:
    """Whatever the echo guard did with a block, the recording keeps it - see cyclops.record.

    The guard is the model's business: it stops Cyclops answering his own voice. It used to be
    the recording's too, and three quarters of a session's mic track came out digital silence.
    """
    recorder, track = recorder_with_a_track()
    blocks = [bytes([n]) * 960 for n in range(1, 6)]
    for block in blocks:
        recorder.on_mic_block(block)
    assert track.written == blocks, "no block dropped, none replaced with silence, none reordered"


def test_a_block_is_written_on_the_callback_that_brought_it() -> None:
    """There is no delay line any more: nothing is held back waiting to be un-muted."""
    recorder, track = recorder_with_a_track()
    recorder.on_mic_block(b"\x01\x02" * 480)
    assert len(track.written) == 1


def test_a_broken_track_disables_the_recording_rather_than_the_session() -> None:
    """An audio thread may not raise: it sets the flag, and the writer thread reports it."""
    recorder, _ = recorder_with_a_track()
    recorder._user = object()  # no append(); the next block is going to hurt
    recorder.on_mic_block(b"\x00" * 960)
    assert recorder.failed
    recorder.on_mic_block(b"\x00" * 960)  # and it stays quiet about it afterwards
