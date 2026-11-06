#!/bin/bash
# hq#602, on the Pi: the scenarios that need clean input, one process each, so no two ears share a room.
# Each pass goes through hq_602_pi.sh, which refuses to start during a real session and compares his records.
Q=/tmp/qa-602
for which in nospace refusal refusal refusal claim claim status status; do
  echo "=== $which"
  bash $Q/tests/acceptance/hq_602_pi.sh --one "$which"
  sleep 5
done
echo ALL-DONE
