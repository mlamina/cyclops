"""What the panel does when the box, or its camera, is the thing that is wrong.

Three failures met on 2026-08-31, in one session, all of which the panel reported as "listening
— talk to me" on a picture of the room:

* a C920 stalled, and the preview went on showing the frame it stalled on for the rest of the
  session, because the reader needed thirty failed reads to give up and a stalled V4L2 read
  takes ten seconds to fail - five minutes of a frozen panel;
* the shutter could not photograph a stale frame, refused, and said so only to a log file;
* the Pi was at 85 C and capping its own clock, which is why the first two happened, and the
  panel had no way at all to mention it.

And a fourth, met on 2026-09-02: taking the panel back from a page left the window decorated
rather than fullscreen, so every later frame was fitted into an 800x418 client area on an
800x480 panel and nothing ever noticed.

Everything here is pure: no camera, no key, no window.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from cyclops import camera, overlay, stats

AWAKE = dict(state=overlay.LISTENING, level=0.4, elapsed=12.0, recording=False, phase=10.0)
TOLERANCE = 24  # how far a drawn pixel may sit from the colour it was asked for


# ---------------------------------------------------------------- how hot is too hot


@pytest.mark.parametrize(
    ("temp_c", "want"),
    [
        (None, ""),  # not Linux, or no thermal zone: no lamp, not a broken one
        (20.0, ""),
        (stats.WARN_C, ""),  # the admin tile goes amber here; the panel is not interrupting yet
        (stats.HOT_C - 0.1, ""),
        (stats.HOT_C, "hot"),  # the board starts capping its clock
        (stats.THROTTLE_C - 0.1, "hot"),
        (stats.THROTTLE_C, "throttled"),
        (91.0, "throttled"),
    ],
)
def test_the_lamp_lights_where_the_board_starts_taking_something_away(
    temp_c: float | None, want: str
) -> None:
    assert stats.heat_alarm(temp_c) == want


def test_the_lamp_holds_its_step_while_the_board_sits_on_the_line() -> None:
    """A throttled Pi is held at its limit, not carried past it, and the reading wanders.

    Measured on this one while it was being throttled: 84.2, 85.3, 84.8, 84.8, 84.2, 85.3 - six
    readings, straddling THROTTLE_C, ten seconds apart. Compared plainly that is a lamp changing
    colour every time the kiosk looks at it, which reads as a broken lamp rather than as a hot
    board. It goes up on the line and comes down HYSTERESIS_C under it.
    """
    wandering = [84.2, 85.3, 84.8, 84.8, 84.8, 84.2, 84.8, 84.2, 84.8, 85.3, 85.3, 84.8]
    shown, alarm = [], ""
    for reading in wandering:
        alarm = stats.heat_alarm(reading, alarm)
        shown.append(alarm)
    changes = [b for a, b in zip(shown, shown[1:], strict=False) if a != b]
    assert changes == ["throttled"], f"the lamp did not settle: {shown}"
    assert shown[-1] == "throttled", "and it must still be on at the end of the wander"


def test_the_lamp_does_come_back_down_when_the_board_really_cools() -> None:
    """Sticky is not stuck. A fan, or an idle hour, has to be able to put it out."""
    alarm = stats.heat_alarm(86.0)
    assert alarm == "throttled"
    alarm = stats.heat_alarm(stats.THROTTLE_C - stats.HYSTERESIS_C - 0.1, alarm)
    assert alarm == "hot", "it should step down one rung, not go dark"
    alarm = stats.heat_alarm(stats.HOT_C - stats.HYSTERESIS_C - 0.1, alarm)
    assert alarm == "", "and out once the board is properly cool again"


def test_the_panel_is_stricter_than_the_page() -> None:
    """A gauge someone went to look at may cry wolf; a lamp that interrupts them may not.

    WARN_C is a temperature a Pi 5 reaches doing ordinary work, so a panel lamp that started
    there would be lit most of the time and would mean nothing by the time it mattered.
    """
    assert stats.temp_band(stats.WARN_C) == "warn", "the page still warns early, on purpose"
    assert stats.heat_alarm(stats.WARN_C) == "", "and the panel deliberately does not"


# ---------------------------------------------------------------- the lamp on the strip


def _strip(frame: np.ndarray, ov: overlay.Overlay) -> np.ndarray:
    return frame[: ov.header.bottom]


def _painted(band: np.ndarray, colour: tuple[int, int, int]) -> int:
    """Opaque pixels of the strip drawn in (near enough) *colour*."""
    rgb, alpha = band[..., :3].astype(int), band[..., 3]
    near = np.abs(rgb - np.array(colour)).max(axis=2) <= TOLERANCE
    return int((near & (alpha > 200)).sum())


def test_a_cool_box_gets_no_lamp() -> None:
    ov = overlay.Overlay(800, 480)
    band = _strip(ov.render(heat="", **AWAKE), ov)
    assert _painted(band, overlay.AMBER) == 0
    assert _painted(band, overlay.RED) == 0, "and nothing red either - red here means FAULT"


@pytest.mark.parametrize(
    ("heat", "colour", "other"),
    [("hot", overlay.AMBER, overlay.RED), ("throttled", overlay.RED, overlay.AMBER)],
)
def test_the_lamp_says_which_kind_of_hot_in_colour(
    heat: str, colour: tuple[int, int, int], other: tuple[int, int, int]
) -> None:
    """Amber where the clock is being capped, red where it is being capped in earnest.

    LISTENING is the state under test precisely because neither colour belongs to it: its halo
    is the tube's white and it is not recording, so there is no REC tag. Anything amber or red
    on this strip is the lamp.
    """
    ov = overlay.Overlay(800, 480)
    band = _strip(ov.render(heat=heat, **AWAKE), ov)
    lit = _painted(band, colour)
    assert lit > 100, f"the {heat} lamp is not on the strip ({lit} px of it)"
    assert _painted(band, other) == 0, "and it is wearing the wrong colour"


def test_the_lamp_never_shoves_the_readouts_along() -> None:
    """SIG, REC and the session clock are laid out from the frame edge inwards.

    Putting the lamp in that group would mean the clock moved whenever the board got warm, which
    is the sort of thing that makes a panel feel unreliable while it is telling you the truth.
    It goes after the mode word instead, and this is what says it stayed there.
    """
    ov = overlay.Overlay(800, 480)
    cool = _strip(ov.render(heat="", **AWAKE), ov)
    hot = _strip(ov.render(heat="throttled", **AWAKE), ov)
    right = slice(cool.shape[1] // 2, None)
    assert np.array_equal(cool[:, right], hot[:, right]), "the right-hand readouts moved"
    assert not np.array_equal(cool, hot), "...and nothing was drawn on the left"


def test_the_lamp_costs_one_baked_layer_per_temperature() -> None:
    """The strip is baked and cached, so the lamp has to be part of the cache key.

    Left out of it, the first frame after the box warmed up would keep the layer that was baked
    while it was cool and the lamp would never appear at all.
    """
    ov = overlay.Overlay(800, 480)
    for heat in ("", "hot", "throttled", "hot", ""):
        ov.render(heat=heat, **AWAKE)
    keys = {key[2] for key in ov._bases}
    assert keys == {"", "hot", "throttled"}


# ---------------------------------------------------------------- a camera that stops


class _Stalling:
    """One frame, and then the way a USB camera fails: slowly.

    The detail this whole test rests on is that a stalled V4L2 read does not return an error
    promptly. It sits in ``select()`` for the driver's timeout - ten seconds on the Pi, which is
    what ``cap_v4l.cpp: select() timeout`` in the kiosk log is - and only then says no.
    """

    def __init__(self, block_s: float) -> None:
        self.block_s = block_s
        self.reads = 0

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.reads += 1
        if self.reads == 1:
            return True, np.zeros((48, 64, 3), dtype=np.uint8)
        time.sleep(self.block_s)
        return False, None

    def release(self) -> None:
        pass


class _Hiccup:
    """A device that is still there and drops a run of frames, failing at once each time."""

    def __init__(self, stop: threading.Event, bad: int) -> None:
        self._stop = stop
        self._bad = bad
        self.reads = 0

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.reads += 1
        if self.reads <= self._bad:
            return False, None
        self._stop.set()  # recovered - end the loop so the test finishes
        return True, np.zeros((48, 64, 3), dtype=np.uint8)

    def release(self) -> None:
        pass


def test_one_blocked_read_is_enough_to_stop_believing_the_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This used to be thirty failed reads, which is five minutes at ten seconds each.

    Five minutes is not a recovery, it is an outage: the preview holds one frame the whole time
    and the shutter refuses every tap. One read that spent longer than the panel's own staleness
    limit getting nowhere is all the evidence there is to be had.
    """
    monkeypatch.setattr(camera, "STALE_AFTER_S", 0.2)
    cap = _Stalling(block_s=0.35)
    source = camera.CameraSource()

    source._read_loop(cap, source._generation)

    assert cap.reads == 2, "a good frame and one stalled read should have settled it"
    assert "stopped delivering" in source.error


def test_the_clock_starts_when_the_read_did_not_when_it_gave_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ten seconds a stalled read spends waiting are ten seconds with no frame.

    Timing the grace from when the read *returned* would throw those away and cost a second
    full timeout to notice the first - which on the Pi is another ten seconds of frozen panel.
    """
    monkeypatch.setattr(camera, "STALE_AFTER_S", 0.2)
    cap = _Stalling(block_s=0.35)
    source = camera.CameraSource()

    started = time.monotonic()
    source._read_loop(cap, source._generation)

    assert time.monotonic() - started < 1.0, "it waited out a second timeout to be sure"


def test_a_run_of_dropped_frames_is_not_an_unplugged_camera() -> None:
    """Failing fast is the device saying "not that one", not the device going away.

    The grace is a duration exactly so these two are told apart: twenty instant failures cost a
    fraction of a second and must not throw away a camera that is about to hand over a frame.
    """
    source = camera.CameraSource()
    cap = _Hiccup(source._stop, bad=20)

    source._read_loop(cap, source._generation)

    assert source.error == "", "a hiccup cost us the device"
    assert source.latest() is not None, "and the frame that arrived after it was dropped"


# ---------------------------------------------------------------- the window that shrank


@pytest.mark.parametrize(
    ("shown", "screen", "lost"),
    [
        ((800, 480), (800, 480), False),  # fullscreen, and the picture fills it
        ((696, 418), (800, 480), True),  # measured: a decorated window, the picture fitted into it
        ((800, 450), (800, 480), True),  # a 16:9 frame in a 5:3 window, which is the same fault
        ((0, 0), (800, 480), False),  # nothing has landed yet; there is nothing to conclude
        ((696, 418), None, False),  # --windowed: no panel to fill, so nothing to be wrong about
    ],
)
def test_a_window_that_stopped_filling_the_panel_is_noticed(shown, screen, lost) -> None:
    """The rule behind Kiosk._keep_fullscreen.

    Fullscreen is a request to a compositor, and under XWayland one made a moment too early is
    dropped without a word. The render size goes on coming from the screen, so the panel stays
    wrong until somebody restarts the kiosk - which is why this is checked every half second
    rather than asked for once and believed.
    """
    from cyclops import kiosk

    assert kiosk.lost_fullscreen(shown, screen) is lost


# ---------------------------------------------------------------- how hot, and how busy, on a bar


def _proc_stat(tmp_path: Path, name: str, busy: int, idle: int, iowait: int) -> Path:
    """A /proc/stat whose aggregate line is known. user nice system idle iowait irq softirq steal."""
    path = tmp_path / name
    path.write_text(f"cpu  {busy} 0 0 {idle} {iowait} 0 0 0 0 0\ncpu0 1 2 3 4 5 6 7 8\nintr 0\n")
    return path


def test_the_temperature_bar_turns_amber_where_the_tile_did() -> None:
    """73% is written into the page's own gradient by hand. This is what stops the two drifting.

    See ``.mtrack`` in cyclops/admin/static/system.css: the amber stop sits at 73%
    because that is where WARN_C lands on this scale, so the bar changes colour at the same
    reading temp_band() does. Move COOL_C or THROTTLE_C without moving the stop and the bar goes
    on looking right while quietly warning at the wrong temperature.
    """
    assert stats.temp_percent(stats.COOL_C) == 0
    assert stats.temp_percent(stats.THROTTLE_C) == 100
    assert stats.temp_percent(stats.WARN_C) == 73, "the .mtrack gradient's amber stop"
    assert stats.temp_percent(stats.HOT_C) == 91, "well into the red by the time the clock is capped"
    assert stats.temp_percent(4.0) == 0, "a bar cannot be less than empty"
    assert stats.temp_percent(120.0) == 100, "or more than full"
    assert stats.temp_percent(None) is None


def test_the_first_cpu_reading_is_a_dash_and_not_a_guess(monkeypatch, tmp_path) -> None:
    """One read of a counter climbing since boot is not a rate. Say nothing rather than divide."""
    monkeypatch.setattr(stats, "_cpu_sample", None)  # monkeypatch puts the global back afterwards
    monkeypatch.setattr(stats, "STAT", _proc_stat(tmp_path, "a", 100, 900, 0))
    assert stats.cpu_percent() is None


def test_cpu_is_the_work_done_between_two_polls(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(stats, "_cpu_sample", None)
    monkeypatch.setattr(stats, "STAT", _proc_stat(tmp_path, "a", 100, 900, 0))
    stats.cpu_percent()
    monkeypatch.setattr(stats, "STAT", _proc_stat(tmp_path, "b", 130, 970, 0))
    assert stats.cpu_percent() == 30, "30 busy jiffies out of the 100 that passed"


def test_waiting_on_the_card_is_not_the_cpu_working(monkeypatch, tmp_path) -> None:
    """iowait counted as busy would show this box at 100% every time it wrote a frame."""
    monkeypatch.setattr(stats, "_cpu_sample", None)
    monkeypatch.setattr(stats, "STAT", _proc_stat(tmp_path, "a", 100, 900, 0))
    stats.cpu_percent()
    monkeypatch.setattr(stats, "STAT", _proc_stat(tmp_path, "b", 100, 900, 100))
    assert stats.cpu_percent() == 0


def test_a_sample_from_last_time_the_page_was_open_is_thrown_away(monkeypatch, tmp_path) -> None:
    """Otherwise the first bar after a quiet hour is the average over that hour, labelled "now"."""
    monkeypatch.setattr(stats, "_cpu_sample", (time.monotonic() - stats.CPU_SAMPLE_MAX_AGE_S - 1, 0, 0))
    monkeypatch.setattr(stats, "STAT", _proc_stat(tmp_path, "a", 5000, 5000, 0))
    assert stats.cpu_percent() is None, "stale: nothing to subtract from"
    monkeypatch.setattr(stats, "STAT", _proc_stat(tmp_path, "b", 5050, 5050, 0))
    assert stats.cpu_percent() == 50, "and the poll after it is a real reading again"
