#!/bin/sh
# Install what a session's video is taken with: wf-recorder, which captures the whole screen.
# Idempotent; run on the Pi, once per box.
#   deploy/install-screen-capture.sh
#
# A session records the screen exactly as it was on the glass - the camera and its chrome, and
# every picture, scratchpad and manual page that covered it (cyclops.screen). wf-recorder is how
# it gets there: it asks the compositor for each frame over wlr-screencopy and hands it down a
# pipe, without encoding anything. Without it nothing breaks - every session records the camera
# instead, and its session.jsonl says why - so this is the difference between a video of the
# panel and a video of the bench.
set -eu

# apt needs root, and over ssh there is nobody at the panel to ask.
SUDO=""
[ "$(id -u)" -eq 0 ] || SUDO="sudo"

if dpkg -s wf-recorder >/dev/null 2>&1; then
  echo "· wf-recorder already installed"
else
  # Retried once after an update: a Pi that has not run apt for a while has lists pointing at
  # package versions the mirror no longer carries.
  $SUDO apt-get install -y -qq wf-recorder >/dev/null \
    || { $SUDO apt-get update -qq && $SUDO apt-get install -y -qq wf-recorder >/dev/null; }
  echo "· wf-recorder installed"
fi

# Prove it can see the screen, from the session the kiosk runs in, with the arguments the
# kiosk uses. Two seconds of frames into a file that does not exist yet (so there is no
# "overwrite?" to answer); anything at all in it means the compositor said yes.
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
export WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-wayland-0}
OUT=$(mktemp -u /tmp/cyclops-screen-check.XXXXXX)
timeout -s INT 2 wf-recorder -c rawvideo -m rawvideo -x bgr0 -f "$OUT" </dev/null >/dev/null 2>&1 || true
BYTES=$(stat -c %s "$OUT" 2>/dev/null || echo 0)
rm -f "$OUT"
if [ "$BYTES" -gt 0 ]; then
  echo "· the screen can be captured: $BYTES bytes of frames in 2 s"
else
  echo "· NO frames from the screen - is the desktop session up on $WAYLAND_DISPLAY?"
fi
