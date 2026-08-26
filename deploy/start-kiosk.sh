#!/bin/sh
# Restart the cyclops kiosk in the Pi's desktop session, detached from this shell, and then
# check that it actually came back.
#
# Why this exists twice over:
#
# * The kiosk is a *long-lived* process. It runs whatever code it loaded at startup, so an
#   rsync-and-uv-sync deploy changes nothing on the panel until it is restarted. That used to be
#   a line of advice printed by push.sh, which is exactly as reliable as it sounds.
# * The pkill pattern lives in a file rather than in an ssh command line, so it can never match
#   the invoking shell's own command line and kill the session issuing it.
#
# It must run in the desktop session's environment: the kiosk needs PipeWire for audio and a
# Wayland socket for the window, neither of which a bare systemd unit has.
set -eu

export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
export WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-wayland-0}
export DISPLAY=${DISPLAY:-:0}
export XAUTHORITY=${XAUTHORITY:-$HOME/.Xauthority}

PATTERN='bin/cyclops-kiosk'
ROOT=$(cd "$(dirname "$0")/.." && pwd)  # the repo root, whichever copy this script was run from
LOG=/tmp/kiosk_live.log

pkill -f "$PATTERN" || true  # nothing running is a fine starting point, not an error
sleep 2
cd "$ROOT" || exit 1
setsid nohup .venv/bin/cyclops-kiosk > "$LOG" 2>&1 < /dev/null &

# Insist on seeing it. A restart that failed - no desktop session, a syntax error in a module it
# imports - leaves a dark panel and, far worse, the impression that the new code is now running.
n=0
while [ "$n" -lt 15 ]; do
  sleep 1
  pid=$(pgrep -f "$PATTERN" | head -1 || true)
  if [ -n "$pid" ]; then
    echo "· kiosk restarted from $ROOT (pid $pid) — log: $LOG"
    exit 0
  fi
  n=$((n + 1))
done

echo "· KIOSK DID NOT COME BACK — the panel is dead. Last of $LOG:" >&2
tail -n 8 "$LOG" >&2 2>/dev/null || true
exit 1
