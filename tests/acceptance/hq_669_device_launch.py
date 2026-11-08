"""qa device run for arc-eng/hq#669: the branch kiosk, a real time-lapse, real conversations.

Starts a run with a typed turn in one real conversation, ends that conversation, watches the idle
kiosk past its sleep time, then stops the run from a second real conversation. Then a short run
with the flip note read as on. Refuses to start unless every write is in /tmp/qa669.
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, "/tmp/qa669/src")

from cyclops import after, flip, kiosk, lapse, ui  # noqa: E402
from cyclops.config import load_settings  # noqa: E402

OUT = "/tmp/qa669"
if not kiosk.__file__.startswith("/tmp/qa669/src/"):
    sys.exit(f"refusing: {kiosk.__file__}")
s = load_settings()
for name in ("sessions_dir", "captures_dir", "projects_dir", "manuals_dir"):
    if not str(getattr(s, name)).startswith("/tmp/qa669/"):
        sys.exit(f"refusing: {name}={getattr(s, name)}")
if after.wanted(s) or s.worker:
    sys.exit("refusing: filing or the worker is on")
print("· guards ok: dirs in scratch, nothing to file, worker off", flush=True)

held = {}


class Watched(ui.SessionController):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        held["ctl"] = self


kiosk.SessionController = Watched


class WatchedKiosk(kiosk.Kiosk):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        held["kiosk"] = self


kiosk.Kiosk = WatchedKiosk


def note(**kw):
    kw["at"] = time.strftime("%H:%M:%S")
    with open(f"{OUT}/phases.jsonl", "a") as f:
        f.write(json.dumps(kw, default=str) + "\n")
    print(f"== {kw}", flush=True)


def voice_links():
    me = str(os.getpid())
    out = subprocess.run(["ss", "-tnp", "state", "established"], capture_output=True, text=True).stdout
    return sum(1 for line in out.splitlines() if f"pid={me}," in line and ":443" in line)


def backlight():
    return [p.read_text().strip() for p in Path("/sys/class/backlight").glob("*/brightness")]


def wait(pred, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.25)
    return False


def say(text):
    ctl = held["ctl"]
    asyncio.run_coroutine_threadsafe(ctl._agent.send_text(text), ctl._loop).result(15)
    note(phase="said", text=text)


def conversation():
    ctl = held["ctl"]
    ctl.start()
    ok = wait(lambda: ctl._agent is not None and ctl._agent.ready.is_set(), 40)
    note(phase="conversation-open", ready=ok, links=voice_links())
    return ok


def hang_up():
    held["ctl"].stop()
    ok = wait(lambda: held["ctl"]._agent is None, 30)
    time.sleep(3)
    note(phase="conversation-ended", closed=ok, links=voice_links())


def shot(name):
    subprocess.run(["grim", f"{OUT}/{name}.png"], check=False)


def drive():
    try:
        for _ in range(15):  # warm-up with the panel kept lit, as a voice wake or a tap would have it
            time.sleep(5)
            if "kiosk" in held:
                held["kiosk"]._touched_at = time.monotonic()
        rec = lapse.current()
        if conversation():
            say("Start a time-lapse.")
            note(phase="start", running=wait(rec.running, 30))
            time.sleep(12)
            hang_up()
        for i in range(10):  # 150 s: longer than the kiosk's 60 s sleep time
            time.sleep(15)
            k = held.get("kiosk")
            note(phase="idle-watch", running=rec.running(), frames=rec.frames, shown=round(rec.seconds(), 2),
                 links=voice_links(), backlight=backlight(), state=held["ctl"].status().get("state"))
            if i == 7:
                shot("idle-lapse")
        last_frames = rec.frames
        if conversation():
            say("Is the time-lapse still running?")
            time.sleep(14)
            shown = rec.seconds()
            say("Stop the time-lapse.")
            note(phase="stop", stopped=wait(lambda: not rec.running(), 40), panel_last=round(shown, 2),
                 last_before=last_frames, result=rec.last)
            time.sleep(12)
            hang_up()
        time.sleep(4)
        real = flip.enabled
        flip.enabled = lambda path=None: True
        note(phase="flip-start", why=rec.start())
        time.sleep(35)
        note(phase="flip-stop", result=rec.stop())
        flip.enabled = real
    except Exception as exc:  # noqa: BLE001
        note(phase="driver-error", error=repr(exc))
    finally:
        os.kill(os.getpid(), signal.SIGTERM)


threading.Thread(target=drive, daemon=True, name="qa669").start()
kiosk.main()
