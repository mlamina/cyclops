#!/bin/bash
# arc-eng/hq#669 qa run, ON the device, detached. His kiosk always comes back (EXIT plus signals).
S=/tmp/qa669
PY=/home/cyclops/cyclops/.venv/bin/python
sums() {
  find /home/cyclops/cyclops/sessions /home/cyclops/cyclops/projects /home/cyclops/cyclops/captures \
    /home/cyclops/cyclops/recordings /home/cyclops/cyclops/manuals /home/cyclops/cyclops/about-you.md \
    /home/cyclops/.cache/cyclops/jobs.json /home/cyclops/.cache/cyclops/recall.npz /home/cyclops/.cache/cyclops/flip \
    /var/lib/cyclops-worker -type f 2>/dev/null | LC_ALL=C sort | xargs -d '\n' sha256sum 2>/dev/null > "$1"
  printf '%s files, %s\n' "$(wc -l < "$1")" "$(sha256sum < "$1" | cut -c1-16)"
}
if [ -n "$(find /home/cyclops/cyclops/sessions -newermt '-5 minutes' -print -quit)" ]; then
  echo "a session wrote in the last 5 minutes; not starting"; exit 1
fi
if grep -q "connected" /tmp/kiosk_live.log; then echo "a conversation opened on his kiosk; not starting"; exit 1; fi
mkdir -p "$S/worker" "$S/sessions"
cp /home/cyclops/cyclops/.env "$S/.env"
echo "before: $(sums $S/before.txt)"
pkill -f 'bin/cyclops-kiosk'
n=0; while pgrep -f 'bin/cyclops-kiosk' >/dev/null && [ $n -lt 20 ]; do sleep 0.5; n=$((n+1)); done
restore() {
  trap - EXIT HUP INT TERM
  rm -f "$S/.env"
  /home/cyclops/cyclops/deploy/start-kiosk.sh >/dev/null 2>&1
  echo "after:  $(sums $S/after.txt)"
  diff "$S/before.txt" "$S/after.txt" | head
  echo "kiosk back at $(date +%H:%M:%S)"
  touch "$S/finished"
}
trap restore EXIT
trap 'exit 1' HUP INT TERM
export XDG_RUNTIME_DIR=/run/user/$(id -u) WAYLAND_DISPLAY=wayland-0 DISPLAY=:0 XAUTHORITY=/home/cyclops/.Xauthority
cd "$S" || exit 1
CYCLOPS_SESSIONS_DIR="$S/sessions" CYCLOPS_CAPTURES_DIR="$S/captures" \
CYCLOPS_PROJECTS_DIR="$S/projects" CYCLOPS_MANUALS_DIR="$S/manuals" CYCLOPS_ABOUT_FILE="$S/about-you.md" \
CYCLOPS_WORKER_DIR="$S/worker" CYCLOPS_WORKER=0 CYCLOPS_SLUG=0 CYCLOPS_REMEMBER=0 CYCLOPS_PROJECTS=0 \
  timeout 720 $PY "$S/hq_669_device_launch.py" > "$S/kiosk.log" 2>&1
echo "kiosk exit $?"
