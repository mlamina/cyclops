#!/bin/bash
# hq#609, on the Pi: every network system call the second check makes, native code included.
# Runs swe's scratch venv against the PR's code in /tmp/qa-609. No session, no mic, no sound.
# Usage: bash hq_609_native_net.sh [seconds]
SECONDS_ALIVE=${1:-150}
Q=/tmp/qa-609
PY=/tmp/swe-609/build2/.venv/bin/python
OUT=$Q/strace.log
"$PY" -c 'import importlib.metadata as m, importlib.util as u; print("pysilero-vad", m.version("pysilero-vad"), "onnxruntime installed:", u.find_spec("onnxruntime") is not None)'
PYTHONPATH=$Q/src strace -f -qq -e trace=network -o "$OUT" \
  "$PY" -u "$Q/tests/acceptance/hq_609_native_net.py" "$SECONDS_ALIVE"
echo "exit=$? network system calls: $(wc -l < "$OUT")"
echo "--- by call"
sed -E 's/^[0-9]+ +//; s/\(.*//' "$OUT" | sort | uniq -c
echo "--- every connect and send"
grep -E 'connect\(|sendto\(|sendmsg\(' "$OUT" | cut -c1-200 | head -30
echo "--- non-local socket families"
grep -E 'socket\(' "$OUT" | grep -v AF_UNIX | cut -c1-160 | head -20
echo NET-DONE
