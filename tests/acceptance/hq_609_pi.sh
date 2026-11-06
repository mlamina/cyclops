#!/bin/bash
# hq#609, on the Pi: everything a fake sound card can show, on the PR's code, in a throwaway copy.
# Expects the copy in /tmp/qa-609 (src, tests) and swe's build venv and clips in /tmp/swe-609.
# Nothing is played aloud and the real mic isn't opened. Refuses to start during a session.
Q=/tmp/qa-609
PY=/tmp/swe-609/build2/.venv/bin/python
CLIPS=/tmp/swe-609/clips
SUM=$Q/tests/acceptance/hq_572_records_sum.sh
cd ~/cyclops || exit 1

idle() {
  case "$(curl -s -m 3 localhost/api/live)" in
    *'"name": null'*) return 0 ;;
    *) echo "ABORT: not idle"; exit 2 ;;
  esac
}
records() {
  bash "$SUM" $Q/after.txt
  diff $Q/before.txt $Q/after.txt && echo RECORDS-IDENTICAL
  echo "--- files changed outside /tmp since the start"
  find ~ -xdev -type f -newer $Q/stamp -not -path "*/.config/chromium/*" -not -path "*/admin-profile/*" 2>/dev/null | cut -c1-120 | head -12
}

idle
bash "$SUM" $Q/before.txt
touch $Q/stamp

echo "=== the check alone: anything native on the network?"
bash "$Q/tests/acceptance/hq_609_native_net.sh" 300 2>&1 | cut -c1-200
echo "internet sockets: $(grep -cE 'AF_INET6?[,)]' $Q/strace.log)  connections to 443 or 53: $(grep -cE 'htons\((443|53)\)' $Q/strace.log)"
records

echo "=== the real Ear with no session, noise looping: anything native on the network?"
idle
PYTHONPATH=$Q/src strace -f -qq -e trace=network -o $Q/strace-ear.log timeout 500 "$PY" -u "$Q/tests/acceptance/hq_609_idle_cost.py" 300 "$CLIPS"/keyboard_typing/*.wav "$CLIPS"/door_wood_knock/*.wav "$CLIPS"/music/*.wav 2>&1 | tail -4 | cut -c1-160
echo "network system calls: $(wc -l < $Q/strace-ear.log)  internet sockets: $(grep -cE 'AF_INET6?[,)]' $Q/strace-ear.log)  connections to 443 or 53: $(grep -cE 'htons\((443|53)\)' $Q/strace-ear.log)"
grep -E 'AF_INET6?[,)]' $Q/strace-ear.log | cut -c1-160 | head -10
records

echo "=== a fresh detector per attempt, and a new worker per Ear: does it give back?"
idle
PYTHONPATH=$Q/src timeout 2400 "$PY" -u "$Q/tests/acceptance/hq_609_churn.py" 2000 200 2>&1 | grep -v -E '^(whisper|ggml)_' | tail -20 | cut -c1-160
records

echo "=== swe's replay of 592 clips through the real Ear, repeated"
idle
PYTHONPATH=$Q/src timeout 2400 "$PY" -u "$Q/tests/acceptance/hq_609_noise.py" "$CLIPS" 2>&1 | tail -30 | cut -c1-160
records

echo "=== real model, real Ear, kiosk's start path"
idle
PYTHONPATH=$Q/src timeout 2400 "$PY" -u "$Q/tests/acceptance/hq_609_real_sessions.py" "$CLIPS" 2 2>&1 \
  | grep -vE '^· (mic|continuity|right now|projects|connected|listening)' | cut -c1-500
echo "run exit=${PIPESTATUS[0]}"
records

echo "=== regression: #572's talk during a close, goodbye, silent start, on this Ear"
idle
PYTHONPATH=$Q/src timeout 900 "$PY" -u "$Q/tests/acceptance/hq_572_real_sessions.py" 2>&1 \
  | grep -vE '^· (mic|continuity|right now|projects|connected|listening)' | tail -40 | cut -c1-300
echo "run exit=${PIPESTATUS[0]}"
records

echo "=== a noisy room for 5 minutes beside the live kiosk"
idle
PYTHONPATH=$Q/src timeout 500 "$PY" -u "$Q/tests/acceptance/hq_609_idle_cost.py" 300 "$CLIPS"/keyboard_typing/*.wav "$CLIPS"/door_wood_knock/*.wav "$CLIPS"/music/*.wav 2>&1 | tail -12 | cut -c1-160
records
echo 609-ALL-DONE
