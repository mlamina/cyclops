"""The panel's own light, switched straight through sysfs.

The kiosk goes dark after a minute untouched, and on a battery the backlight is the expensive
half of that: the official 7" panel draws about as much as the Pi does idling, so a black picture
on a lit panel saves almost nothing. Sleeping therefore turns the light off as well, and a tap
brings it back to exactly the brightness it had rather than to full.

Raspberry Pi OS ships ``/sys/class/backlight/*/brightness`` as ``root:video`` 664 and puts the
desktop user in ``video``, so none of this needs root or a udev rule. Where the file is missing
or not writable - a Mac, a windowed dev run, an HDMI monitor - every call is a no-op and the
kiosk simply leaves alone the light it cannot reach.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

BACKLIGHT_DIR = Path("/sys/class/backlight")


class Backlight:
    """The first backlight device that will let us write to it, or nothing at all."""

    def __init__(self, root: Path = BACKLIGHT_DIR) -> None:
        self.path: Path | None = None
        self.maximum = 0
        self._restore = 0  # the level to come back to; never 0, or waking would be invisible
        self.note = "none writable; the panel's light is left alone"
        for device in sorted(root.glob("*")):
            brightness = device / "brightness"
            if not os.access(brightness, os.W_OK):
                continue
            try:
                self.maximum = int((device / "max_brightness").read_text())
                self._restore = int(brightness.read_text()) or self.maximum
            except (OSError, ValueError):
                continue
            self.path = brightness
            self.note = f"{device.name} (0-{self.maximum})"
            break

    @property
    def available(self) -> bool:
        return self.path is not None

    def off(self) -> None:
        """Kill the light, remembering the level to come back to."""
        if self.path is None:
            return
        try:
            current = int(self.path.read_text())
        except (OSError, ValueError):
            current = 0
        if current > 0:  # someone may have dimmed the panel; wake to *that*, not to full
            self._restore = current
        self._write(0)

    def on(self) -> None:
        """Back to the brightness it had before it went out."""
        self._write(self._restore)

    def _write(self, level: int) -> None:
        """Set the level, and give up on the device for good if it ever refuses.

        A backlight that cannot be written once will not be writable a second later either, and
        the kiosk must not shout about it several times a minute for the rest of the day.
        """
        if self.path is None:
            return
        try:
            self.path.write_text(f"{level}\n")
        except OSError as exc:
            print(f"· backlight not writable, leaving it alone: {exc}", file=sys.stderr, flush=True)
            self.path = None
            self.note = f"unwritable ({exc.strerror})"
