#!/bin/bash
# hq#619 round 2, on the Pi: the real panel's browser on the PR's flags, idle. No session, no sound.
S=/tmp/qa-619
RESTART=/home/cyclops/cyclops/deploy/start-kiosk.sh

idle() { case "$(curl -s -m 3 localhost/api/live)" in *'"name": null'*) return 0 ;; *) return 1 ;; esac; }
stamp() { date -u +%H:%M:%S; }

sample() {  # <seconds> <label>: every 5 s, the browser's non-loopback sockets, and whether a session opened
  end=$(( $(date +%s) + $1 ))
  while [ "$(date +%s)" -lt "$end" ]; do
    idle || echo "$(stamp) $2 SESSION OPEN: this read is spoiled" >> "$S/ss.log"
    ss -tunap 2>/dev/null | grep -i chrom | grep -vE '127\.0\.0\.1|\[::1\]|\[::ffff:127' | awk -v p="$2" -v t="$(stamp)" '{print t, p, $1, $2, $6}' >> "$S/ss.log"
    sleep 5
  done
}

idle || { echo "ABORT: not idle"; exit 2; }
: > "$S/ss.log"
bash "$S/hq_572_records_sum.sh" "$S/before.txt"
touch "$S/stamp"
echo baseline > "$S/phase"
sudo -n python3 "$S/hq_619_packets.py" wlan0 "$S/phase" 465 "$S/packets.txt" &
sleep 1

echo "$(stamp) baseline, today's settings, 120 s"
sample 120 baseline

idle || { echo "ABORT before restart: session open"; exit 3; }
echo restart > "$S/phase"
echo "$(stamp) restart on the PR's kiosk.py"
PYTHONPATH="$S/src" "$RESTART"

NS=""
for _ in $(seq 1 40); do
  NS=$(pgrep -nf 'utility-sub-type=network.mojom.NetworkService')
  [ -n "$NS" ] && break
  sleep 0.5
done
echo "$(stamp) network process: $NS"
sudo -n timeout 300 strace -f -qq -s 0 -e trace=socket,connect,sendto,sendmsg,sendmmsg,write -p "$NS" -o "$S/strace.log" &
TR=$!
echo after > "$S/phase"
MAIN=$(pgrep -f '^/usr/lib/chromium/chromium .*--user-data-dir=' | head -1)
sleep 20
echo "--- real browser's command line, network flags"
tr '\0' '\n' < "/proc/$MAIN/cmdline" | grep -E 'proxy|host-resolver|background-networking|^http'
echo "--- kiosk's code"
KP=$(pgrep -f 'bin/cyclops-kiosk' | head -1)
tr '\0' '\n' < "/proc/$KP/environ" | grep '^PYTHONPATH'
sample 280 after
echo "$(stamp) waiting for the trace to end"
wait "$TR"
sudo -n chown cyclops "$S/strace.log"
echo "--- browser's links to this box"
ss -tnp 2>/dev/null | grep -i chrom | grep -E '127\.0\.0\.1:(80|8081)' | awk '{print $1, $5}' | sort | uniq -c
echo "--- kiosk log"
cut -c1-120 /tmp/kiosk_live.log | tail -n 8
echo "--- browser log lines"
wc -l < ~/.cache/cyclops/chromium.log
grep -c 'google_apis/gcm' ~/.cache/cyclops/chromium.log
grep -ciE 'proxy|ERR_' ~/.cache/cyclops/chromium.log
grep -iE 'ERR_|proxy' ~/.cache/cyclops/chromium.log | cut -c1-160 | sort | uniq -c | sort -rn | head -n 6

echo restore > "$S/phase"
until idle; do echo "$(stamp) session open, waiting to restore"; sleep 30; done
echo "$(stamp) restart back on the installed code"
env -u PYTHONPATH "$RESTART"
sleep 15
MAIN=$(pgrep -f '^/usr/lib/chromium/chromium .*--user-data-dir=' | head -1)
echo "--- restored browser's network flags (expect none but the page)"
tr '\0' '\n' < "/proc/$MAIN/cmdline" | grep -E 'proxy|host-resolver|background-networking|^http'
KP=$(pgrep -f 'bin/cyclops-kiosk' | head -1)
echo "kiosk PYTHONPATH: $(tr '\0' '\n' < "/proc/$KP/environ" | grep -c '^PYTHONPATH')"
wait
bash "$S/hq_572_records_sum.sh" "$S/after.txt"
diff "$S/before.txt" "$S/after.txt" && echo RECORDS-IDENTICAL
echo "--- files changed outside /tmp since the start"
find ~ -xdev -type f -newer "$S/stamp" -not -path '*/.config/chromium/*' -not -path '*/admin-profile/*' 2>/dev/null | cut -c1-120 | head -n 20
echo "DONE $(stamp)"
