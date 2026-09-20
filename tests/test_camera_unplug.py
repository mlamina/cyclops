"""Pulling the endoscope out while it is the camera on screen must not take the kiosk with it.

Found on 2026-09-20, the first time there was a tag on the panel worth pulling a cable to watch
change. The endoscope was the live camera; it was unplugged; the whole process died:

    · camera useeplus stopped delivering - looking for it again
    python3: ../../libusb/os/threads_posix.h:58:
             usbi_mutex_destroy: Assertion `pthread_mutex_destroy(mutex) == 0' failed.

The supervisor gave up on the dead handle and released it on its way back round to look for a
camera, and destroying a libusb handle whose device has gone is an ``assert()`` in C. That is a
SIGABRT, not an exception: no ``except`` catches it, no ``finally`` runs after it, and the panel,
the face and the session go with it.

So there are two separate claims here and they need different medicine. The abort can only be
headed off by never making the call - that is :class:`_UseeplusCapture`'s. An ordinary raising
release is catchable, and must not stop the supervisor going round again - that is
:func:`camera._let_go`'s.

No device and no libusb: the driver underneath is a stand-in that records, raises, or "aborts"
by raising something no handler in the tree is allowed to catch.
"""

from __future__ import annotations

import threading

import pytest

from cyclops import camera, webcam


class Aborted(BaseException):
    """What libusb's assertion would be, if a process death could be caught.

    A :class:`BaseException` on purpose. Nothing in the camera path catches one, so a test that
    survives this really did avoid the call rather than swallow it - which is the whole
    difference between the fix and a ``try`` that would not have helped.
    """


class Driver:
    """The supercamera ``Camera`` underneath the wrapper: reads, and a release that aborts."""

    def __init__(self) -> None:
        self.releases = 0

    def read(self) -> tuple[bool, object]:
        return False, None  # the device is gone; every read from here fails

    def release(self) -> None:
        self.releases += 1
        raise Aborted("usbi_mutex_destroy: Assertion `pthread_mutex_destroy(mutex) == 0' failed")


def test_an_endoscope_off_the_bus_is_never_handed_back() -> None:
    """The one that killed the kiosk. Nothing may reach the driver once the device has gone."""
    driver = Driver()
    cap = webcam._UseeplusCapture(driver, lambda: False)

    assert cap.read() == (False, None), "a device off the bus has no frame to give"
    cap.release()

    assert driver.releases == 0, "releasing a yanked libusb handle aborts the whole process"


def test_a_bus_probe_that_raises_counts_as_gone() -> None:
    """libusb has just had a device pulled out of it and is entitled to raise about it.

    Answering "still there" to a question that could not be asked is how the handle stays alive
    long enough to be released, which is the abort again by a longer road.
    """
    driver = Driver()

    def probe() -> bool:
        raise OSError("libusb: no such device")

    cap = webcam._UseeplusCapture(driver, probe)
    cap.read()
    cap.release()

    assert driver.releases == 0


def test_an_endoscope_still_on_the_bus_is_handed_back_properly() -> None:
    """The other side of it: a camera being put down normally is still put down normally."""
    driver = Driver()
    cap = webcam._UseeplusCapture(driver, lambda: True)

    with pytest.raises(Aborted):  # the stand-in always raises; a real one would just close
        cap.release()

    assert driver.releases == 1, "a present device must still be released, or it stays wedged"


class Raiser:
    """A capture whose close does not go well - a USBError, not an abort."""

    def __init__(self) -> None:
        self.releases = 0

    def release(self) -> None:
        self.releases += 1
        raise OSError("[Errno 19] No such device")


class Source:
    """As much of a CameraSource as ``_supervise`` touches."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._generation = 1
        self._framing = camera.DEFAULT_FRAMING
        self._preferred = None
        self._error = ""
        self._cap = None
        self._index = None
        self._switching = False  # set when the loop stands down for a better camera, not a dead one
        self.opens = 0
        self.caps: list[Raiser] = []

    def _forget(self) -> None:
        pass

    def _read_loop(self, cap, token: int, framing: str) -> None:
        if self.opens >= 2:  # it came back round, which is all this needs to show
            self._stop.set()

    _supervise = camera.CameraSource._supervise


def test_a_release_that_raises_does_not_end_the_hunt_for_a_camera(monkeypatch, capsys) -> None:
    """The supervisor's ``finally``. It runs on the way back to looking, so it cannot throw.

    An exception here leaves the panel on "no camera" until somebody restarts the kiosk, for a
    device that had already gone - which is the same outage as the abort, just quieter.
    """
    source = Source()

    def open_camera(preferred, framing):
        source.opens += 1
        cap = Raiser()
        source.caps.append(cap)
        return cap, webcam.USEEPLUS

    monkeypatch.setattr(camera, "open_camera", open_camera)

    source._supervise(1)

    assert source.opens >= 2, "a raising close must not stop it opening the camera again"
    assert all(cap.releases == 1 for cap in source.caps)
    assert "would not close cleanly" in capsys.readouterr().out
