#!/bin/sh
# The power-cut test for arc-eng/hq#669: a run left going, then an immediate reboot with no sync,
# which is what an unplug does to the card. Writes only under ~/swe-669/cut.
OUT=/home/cyclops/swe-669/cut
PY=/home/cyclops/cyclops/.venv/bin/python
mkdir -p "$OUT"
pkill -f 'bin/cyclops-kiosk'
sleep 4
$PY /home/cyclops/swe-669/tools/lapse_probe.py "$OUT" --minutes 3.5 --every 30 --no-stop 2>&1 | grep --line-buffered -v "^\[" > "$OUT/probe.log" &
sleep 200  # 3 min 20 s of recording, mid-fragment
grep frames "$OUT/probe.log" | tail -1 > /home/cyclops/swe-669/cut-last.txt
date +%s > /home/cyclops/swe-669/cut-at.txt
sync /home/cyclops/swe-669/cut-last.txt  # only the note: the video gets no sync, as with an unplug
sudo sh -c 'echo 1 > /proc/sys/kernel/sysrq; echo b > /proc/sysrq-trigger'
