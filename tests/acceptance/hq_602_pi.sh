#!/bin/bash
# hq#602, on the Pi: real-model checks from a throwaway copy, his records compared before and after.
# Expects the copy in /tmp/qa-602 (src, tests/acceptance). Refuses to start during a real session.
Q=/tmp/qa-602
cd ~/cyclops || exit 1
live=$(curl -s -m 3 localhost/api/live)
case "$live" in
  *'"name": null'*) ;;
  *) echo "ABORT: not idle: $live"; exit 2 ;;
esac
mkdir -p $Q/out
panel=$(head -1 /sys/class/drm/card*-DSI-*/modes /sys/class/drm/card*-HDMI-A-1/modes 2>/dev/null | grep -m1 -E '^[0-9]+x[0-9]+')
echo "panel: ${panel:-unknown} · sleep setting: $(grep -E '^CYCLOPS_SLEEP_AFTER_S' .env)"
bash $Q/tests/acceptance/hq_572_records_sum.sh $Q/before.txt
touch $Q/stamp
XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0 QA_OUT=$Q/out QA_PANEL=${panel:-800x480} \
  PYTHONPATH=$Q/src timeout 1500 .venv/bin/python -u $Q/tests/acceptance/hq_602_real_sessions.py "$@" 2>&1 \
  | grep -vE '^(Cyclops|You): ' | cut -c1-600
echo "run exit=${PIPESTATUS[0]}"
bash $Q/tests/acceptance/hq_572_records_sum.sh $Q/after.txt
diff $Q/before.txt $Q/after.txt && echo RECORDS-IDENTICAL
echo "--- files changed outside /tmp since the start"
find ~ -xdev -type f -newer $Q/stamp -not -path "*/.config/chromium/*" -not -path "*/admin-profile/*" 2>/dev/null | cut -c1-120 | head -20
echo "--- left running"
pgrep -a -f "wf-recorder|hq_602" | cut -c1-120
