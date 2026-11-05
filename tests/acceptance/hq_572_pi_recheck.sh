#!/bin/bash
# hq#572, on the Pi: the real-model checks again on the build under review, plus smoke and swe's
# probe, with his records compared before and after. Expects the copy in /tmp/qa-572.
# Nothing here plays aloud: every session's sound card is fake. Isolation: see hq_572_real_sessions.py.
Q=/tmp/qa-572
cd ~/cyclops || exit 1
ls -t sessions | head -1 > $Q/last_session.txt
bash $Q/tests/acceptance/hq_572_records_sum.sh $Q/before3.txt
touch $Q/stamp3
for part in "" --recheck --second; do
  name=run3${part#--}
  timeout 420 .venv/bin/python -u $Q/tests/acceptance/hq_572_real_sessions.py $part > $Q/$name.log 2>&1
  echo "$name exit=$?"
done
timeout 150 .venv/bin/python -u $Q/tools/start_latency.py --runs 3 --short > $Q/probe3.log 2>&1
echo "probe exit=$?"
bash $Q/tests/acceptance/hq_572_records_sum.sh $Q/after3.txt
diff $Q/before3.txt $Q/after3.txt && echo RECORDS-IDENTICAL
find ~ -xdev -type f -newer $Q/stamp3 -not -path "*/.config/chromium/*" -not -path "*/admin-profile/*" 2>/dev/null | cut -c1-120 | head -20
for f in run3 run3recheck run3second; do
  echo "--- $f"
  grep -vE '^\[|^· |^$' $Q/$f.log | cut -c1-420
done
echo "--- probe"
grep -E '^(run|median)' $Q/probe3.log | cut -c1-170
