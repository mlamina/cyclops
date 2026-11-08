#!/bin/sh
# One device run for arc-eng/hq#669: kiosk stopped, a camera-only baseline, a 12-minute time-lapse,
# and the kiosk always put back. Everything is written under ~/swe-669/scratch.
OUT=/home/cyclops/swe-669/scratch
PY=/home/cyclops/cyclops/.venv/bin/python
mkdir -p "$OUT"
pkill -f 'bin/cyclops-kiosk'
sleep 4
trap '/home/cyclops/cyclops/deploy/start-kiosk.sh >/dev/null 2>&1; echo "kiosk restarted"' EXIT
echo "== baseline: camera alone at 1280x720"
$PY /home/cyclops/swe-669/tools/lapse_probe.py "$OUT/baseline" --minutes 1 --every 20 --camera-only 2>&1 | grep -v "^\["
echo "== time-lapse, 12 minutes"
$PY /home/cyclops/swe-669/tools/lapse_probe.py "$OUT/run" --minutes 12 --every 60 2>&1 | grep -v "^\["
echo "== done"
