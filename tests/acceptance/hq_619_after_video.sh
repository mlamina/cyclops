#!/bin/bash
# hq#619: wrapper for after.py on the Pi, with the packet counter and the records fingerprint around it.
S=/tmp/qa-619
case "$(curl -s -m 3 localhost/api/live)" in *'"name": null'*) ;; *) echo "ABORT: not idle"; exit 2 ;; esac
bash "$S/hq_572_records_sum.sh" "$S/before3.txt"
echo start > "$S/phase"
sudo -n python3 "$S/hq_619_packets.py" wlan0 "$S/phase" 720 "$S/packets3.txt" &
sleep 1
cd "$S/vid" || exit 1
PYTHONPATH="$S/src" timeout -k 10 700 /home/cyclops/cyclops/.venv/bin/python -u "$S/hq_619_after_video.py"
echo "after.py exit $?"
echo end > "$S/phase"
wait
bash "$S/hq_572_records_sum.sh" "$S/after3.txt"
diff "$S/before3.txt" "$S/after3.txt" && echo RECORDS-IDENTICAL
echo "FINISHED"
