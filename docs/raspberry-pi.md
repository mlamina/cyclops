# Raspberry Pi

cyclops runs headless-free on a Raspberry Pi (tested on a Pi 5, 64-bit Bookworm) with a USB
webcam (a Logitech C920 gives camera + mic), an [I2S amplifier](audio.md) on the GPIO header, and
the official 7" touch display.

**Install** (OpenCV is the `-headless` build, so no desktop GL libs are needed):

```bash
sudo apt install -y libportaudio2            # runtime for the audio library
curl -LsSf https://astral.sh/uv/install.sh | sh   # if you don't have uv
cd ~/cyclops && cp .env.example .env         # paste your OPENAI_API_KEY
uv sync                                       # fetches Python 3.12 + aarch64 wheels
uv run cyclops-smoke                          # camera + API check, no audio needed
```

Then set the audio devices — see [audio.md](audio.md#picking-devices), which a Pi always needs.

## Starting the panel

`cyclops-kiosk` must run **in your desktop session** — it needs PipeWire for audio and a Wayland
socket for the window, neither of which a bare systemd unit has. On the Wayland (labwc) desktop,
put this in `~/.config/labwc/autostart`:

```sh
#!/bin/sh
cd /home/<you>/cyclops && .venv/bin/cyclops-kiosk >/tmp/kiosk_live.log 2>&1 &
```

**What a boot looks like, and why nothing flickers.** The panel is one OpenCV window and the
admin page is a Chromium kept warm behind it, and a compositor puts whichever mapped last on
top — so the browser is started *before* the kiosk has a window at all, with the panel's light
switched off. Chromium's white first frame, the dashboard painting and its habit of raising its
own window a second time all happen in the dark, and the window opened afterwards maps last and
stays. The light, the first camera frame and the "ready" cue then arrive together, which is what
makes that cue worth anything: when you hear it, the box is up and the eye is instant.

The kiosk prints a timed line per phase to its log, so a slow boot can be read off it:

```
· +  1.8s  settings loaded; opening the camera
· +  2.8s  camera settled
· +  2.9s  panel dark; warming the browser behind it
· +  6.7s  browser has the page; waiting for it to run
· +  8.7s  browser warm; taking the panel
· +  8.9s  panel up
```

Everything the panel shows before that last line is the desktop underneath, which for those
seconds *is* the interface — so `deploy/install-panel-look.sh` makes it the same boot splash
plymouth is already showing and takes the LXDE taskbar out of the session. Run it once per Pi.

## Deploying

`deploy/push.sh [user@host]` rsyncs the working tree to the Pi, runs `uv sync`, restarts
`cyclops-admin`, and then restarts the kiosk via `deploy/start-kiosk.sh`. It never copies your
local `.env`.

That last step is not a convenience. The kiosk is the only long-lived process here — it runs
whatever code it loaded at startup — so a deploy that does not restart it leaves the panel on the
old build while `cyclops-smoke`, the CLI and every other check happily report the new one. The
restart is verified: `start-kiosk.sh` waits for the process to reappear and exits non-zero if it
does not, which aborts the push rather than printing a success line over a dead panel. Pass
`SKIP_KIOSK=1` to opt out — a docs-only push, or when someone is mid-conversation with it.

```bash
deploy/push.sh                       # deploy, restart both, verify the kiosk came back
SKIP_KIOSK=1 deploy/push.sh          # leave the running kiosk alone
ssh cyclops@cyclops.local cyclops/deploy/start-kiosk.sh   # just restart it
```
