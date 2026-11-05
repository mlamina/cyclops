#!/bin/bash
# hq#572, on the Pi: second round of real-model checks, then swe's probe as shipped, with his
# records compared before and after. Expects the copy in /tmp/qa-572.
Q=/tmp/qa-572
cd ~/cyclops || exit 1
if ! ls -t sessions | head -1 | grep -q 2026-11-05_13-52-11; then
  echo "a new real session exists since the check began; compare by hand"
fi
bash $Q/tests/acceptance/hq_572_records_sum.sh $Q/before2.txt
touch $Q/stamp2
timeout 330 .venv/bin/python -u $Q/tests/acceptance/hq_572_real_sessions.py --second > $Q/run2.log 2>&1
echo "round2 exit=$?"
timeout 150 .venv/bin/python -u $Q/tools/start_latency.py --runs 4 > $Q/probe_long.log 2>&1
echo "probe long exit=$?"
timeout 150 .venv/bin/python -u $Q/tools/start_latency.py --runs 4 --short > $Q/probe_short.log 2>&1
echo "probe short exit=$?"
bash $Q/tests/acceptance/hq_572_records_sum.sh $Q/after2.txt
diff $Q/before2.txt $Q/after2.txt && echo RECORDS-IDENTICAL
find ~ -xdev -type f -newer $Q/stamp2 -not -path "*/.config/chromium/*" -not -path "*/admin-profile/*" 2>/dev/null | cut -c1-120 | head -20
echo "--- round 2"
grep -vE '^\[|^· |^$' $Q/run2.log | cut -c1-420
echo "--- probe"
grep -E '^(run|median)' $Q/probe_long.log | cut -c1-170
grep -E '^(run|median)' $Q/probe_short.log | cut -c1-170
