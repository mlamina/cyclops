"""Putting the camera down: who is allowed to release the device, and when.

One method, three branches, and the reason it is worth a file of its own is what the wrong answer
costs. A ``VideoCapture`` released from one thread while another is inside ``cap.read()`` on it
leaves a UVC device mid-stream, and a C920 left mid-stream does not come back on its own - it
stays on the bus, answers ``lsusb``, and gives ``uvcvideo ... Failed to set UVC probe control :
-110`` for ever with no ``/dev/video`` node at all. Clearing that needs a USB re-enumerate and
somebody logged in to do it.

It is not a narrow race either, which is what makes it worth pinning rather than commenting.
``_read_loop``'s docstring measures a stalled V4L2 read at ten seconds, and ``stop()`` waits three
- so any camera already unhappy enough to be worth restarting for hit this every single time. It
happened on 2026-09-05 after three kiosk restarts in a row.

No camera here, and none wanted: this is a stand-in with the real method borrowed onto it, the
same trick ``test_panel_swap.py`` plays on ``Kiosk.show_picture``.
"""

from __future__ import annotations

import threading

from cyclops import camera
from cyclops.camera import CameraSource


class Cap:
    """A VideoCapture that only remembers whether anybody released it."""

    def __init__(self) -> None:
        self.released = 0

    def release(self) -> None:
        self.released += 1


class Reader:
    """A reader thread that either came back when asked, or did not."""

    def __init__(self, *, comes_back: bool) -> None:
        self._comes_back = comes_back
        self.joined_for: float | None = None

    def join(self, timeout: float | None = None) -> None:
        self.joined_for = timeout

    def is_alive(self) -> bool:
        return not self._comes_back


class Source:
    """As much of a CameraSource as ``stop`` touches."""

    def __init__(self, thread: Reader | None, cap: Cap | None) -> None:
        self._stop = threading.Event()
        self._thread = thread
        self._cap = cap
        self.forgotten = 0

    def _forget(self) -> None:
        self.forgotten += 1

    stop = CameraSource.stop


def test_a_reader_still_in_a_read_keeps_the_device(capsys) -> None:
    """The whole point. This is the branch that used to wedge the camera.

    Still alive means still inside ``cap.read()``. Releasing from here would put a second thread
    in the same VideoCapture, and the device does not survive it. The process is on its way out
    and the kernel closes the fd on exit, which stops the stream from the one place that cannot
    race a reader.
    """
    cap = Cap()
    source = Source(Reader(comes_back=False), cap)

    source.stop()

    assert cap.released == 0, "a device released out from under a live reader is a wedged device"
    assert source._stop.is_set(), "the reader must still be told to stop"
    assert source.forgotten == 1, "and what it was showing is no longer a picture of the room"
    assert "still in a read" in capsys.readouterr().out, "say so; it is why the panel goes dark"


def test_a_reader_that_came_back_has_already_released_it() -> None:
    """The ordinary exit. ``_supervise`` clears ``_cap`` in its own finally, before releasing.

    So on a clean stop there is nothing left here to release - which is exactly why the line that
    used to do it could only ever fire in the dangerous case.
    """
    source = Source(Reader(comes_back=True), None)

    source.stop()

    assert source.forgotten == 1


def test_a_source_that_never_started_a_reader_releases_its_own_handle() -> None:
    """Nobody else can be holding it, so this is the one caller entitled to let it go."""
    cap = Cap()
    source = Source(None, cap)

    source.stop()

    assert cap.released == 1


def test_the_wait_is_shorter_than_the_kiosk_gets_before_a_sigkill() -> None:
    """``deploy/start-kiosk.sh`` waits ten seconds for the old kiosk and then sends SIGKILL.

    A teardown that waited out a stalled ten-second read would be killed halfway through it, and
    a SIGKILL is the unclean exit this whole file exists to avoid. Pinned rather than commented
    because the two numbers live in different languages, in different directories.
    """
    assert camera.CLOSE_WAIT_S < 10.0

    cap = Cap()
    reader = Reader(comes_back=True)
    Source(reader, cap).stop()
    assert reader.joined_for == camera.CLOSE_WAIT_S
