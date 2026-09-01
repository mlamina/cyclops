---
description: Diagnose a Cyclops kiosk fault on the Pi — where the evidence is and what usually caused it
---

Something went wrong on the panel. Work through this before exploring from scratch.

## Getting on the box

`ssh cyclops@cyclops.local` — **`.local` is required**, plain `cyclops` does not resolve.

## Where the evidence is

| What | Where |
|---|---|
| Kiosk log (stdout **and** stderr) | `/tmp/kiosk_live.log` if started by `deploy/start-kiosk.sh`, else `~/.xsession-errors` |
| Same log, previous X session | `~/.xsession-errors.old` — rotated on X restart, so this is where a crashed run's log went |
| Sessions | `~/cyclops/sessions/<stamp>_<slug>/` → `session.jsonl`, `session.md`, `video.mp4`, `photos/` |
| Real boot boundaries | `journalctl --list-boots` |

`session.jsonl` is the fastest read: a gap in it is the session going deaf.

## The usual suspects, in order

1. **Heat.** `vcgencmd measure_temp` and `vcgencmd get_throttled`. This box runs at 85 °C and
   throttles, which drops camera frames and misses audio deadlines. Decode the bits before
   blaming power: 0/16 are undervoltage, 1/2/17/18/19 are thermal. `0xe0006` is heat.
2. **Camera stalled.** `select() timeout` repeating every 10 s in the kiosk log. `lsusb` and
   `/dev/video0` still look healthy — the device is open and delivering nothing. The panel
   should say "Camera stopped responding"; if it is showing a frozen picture instead, that is a
   regression in `camera.py:_read_loop` or the staleness check in the kiosk render loop.
3. **Realtime socket black-holed.** No `You:` lines in the log and no transcript entries, but
   the recording's left channel has your voice on it. Pings should now surface this as FAULT
   within ~30 s (`agent.py:KEEPALIVE_S`); if it hung silently instead, the pings are not on.
4. **Unclean reboot.** No shutdown sequence in `journalctl -b -1` means it died rather than
   restarted.

## Reading the session video

`video.mp4` is a recording of the **panel**, so it shows exactly what you were looking at and
when. Audio is stereo: **left = you, right = Cyclops**.

```sh
cd ~/cyclops/sessions/<session>
# when did the picture stop moving?
ffmpeg -i video.mp4 -an -vf mpdecimate,showinfo -f null - 2>&1 | grep -oE 'pts_time:[0-9.]+'
# was anyone talking? (c0 = you, c1 = Cyclops)
ffmpeg -i video.mp4 -af "pan=mono|c0=c0,silencedetect=n=-45dB:d=1" -f null -
```

Gotcha: a freeze at the *end* of the file leaves no gap between kept frames. Compare the last
`pts_time` against the duration, don't just look for gaps.

## Gotchas that will waste your time

- Early dmesg timestamps come from fake-hwclock and are wrong until NTP corrects them mid-boot.
  `last -x reboot` is nonsense for the same reason. Trust `journalctl --list-boots`.
- A short `ffprobe` duration is not evidence of a freeze — the recorder rewrites the previous
  frame on a fixed clock, so a frozen camera still produces a full-length file.
- Deploy with `deploy/push.sh`; it restarts the kiosk, which is the only long-lived process and
  the one that otherwise keeps running the old code.
- Screenshot the panel: `XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0 grim /tmp/p.png`
