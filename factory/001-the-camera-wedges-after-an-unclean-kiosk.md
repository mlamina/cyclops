---
state: asking
opened: 2026-09-12
---

# the camera wedges after an unclean kiosk kill - lsusb sees it, /dev/video0 opens, no frames until you unbind/bind 1-1 by hand

the camera wedges after an unclean kiosk kill - lsusb sees it, /dev/video0 opens, no frames until you unbind/bind 1-1 by hand

## Plan

`camera.py:195-230` already documents this exactly: two threads in one `VideoCapture` leaves a
C920 mid-stream, and a C920 left mid-stream "stays on the bus, answers `lsusb`, and gives
`uvcvideo ... Failed to set UVC probe control : -110` for ever with no `/dev/video` node at all.
Only a USB re-enumerate clears it." `stop()` was already changed to stop *causing* it. Nothing
yet *clears* it, so one bad teardown still costs a hand ssh.

Two halves, and they're independent:

**Stop making it happen.** `deploy/start-kiosk.sh:44` sends `pkill -9` after 10 s of asking
nicely. That is the unclean exit, on the path we walk five times an afternoon. Raise the grace
window past `CLOSE_WAIT_S` + a stalled read (~13 s, measured in `_read_loop`) so a kiosk that is
merely slow to put the camera down is never SIGKILLed for it.

**Clear it when it happens anyway.** The supervisor already retries open forever
(`RECONNECT_EVERY_S`). Give it one more move: when open fails repeatedly *and* the device is
still on the bus, re-enumerate it once, then carry on retrying. Two ways to do that:

- `USBDEVFS_RESET` ioctl on `/dev/bus/usb/<bus>/<dev>` — no root, but needs a udev rule for the
  C920's id the way `99-useeplus-camera.rules` already grants the endoscope's. One `ioctl` call,
  no shelling out.
- unbind/bind on `/sys/bus/usb/drivers/usb/1-1` — the move that works by hand, but those paths
  are root-only, so it needs a sudoers line or a tiny setuid helper.

Recommend the first: same shape as a rule we already ship, and no new privilege on the box.

**Working means:** kill the kiosk with `-9` mid-session, start it again, and the panel shows a
picture without anybody ssh-ing in. `camera.error` says what it's doing in the meantime.

**Needs checking on the box** (it wasn't reachable when this was planned): the C920's
`idVendor:idProduct`, and that the reset actually clears the `-110` state rather than needing a
full unbind.

## Questions

**1.** Self-heal, or heal on restart? Resetting inside the running supervisor fixes it without
anyone noticing; doing it once in `start-kiosk.sh` before launch is three lines of shell and
touches no Python. Recommend self-heal — the wedge can also happen to a box nobody restarted.

*(answer here)*

**2.** Should the panel say anything while it re-enumerates? It's a ~2 s gap where `connected` is
False, so today it would flash the "no camera" card. Recommend saying nothing new and just
suppressing that card for a couple of seconds — the face shouldn't narrate plumbing.

*(answer here)*

**3.** Do you want the `pkill -9` grace window raised at the same time, or kept as its own job?
Recommend same job: it's a one-number change and it's the half that stops the bug rather than
mopping it up.

*(answer here)*
