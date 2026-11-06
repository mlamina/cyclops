#!/bin/bash
# hq#599 round 2, on the Pi. Installs this copy's worker, then: a hostile stand-in inside the real
# service, hostile PDF timings, one real job, then the real voice sessions (hq_599_pi.sh). The real
# program goes back whatever happens: install-worker.sh is what puts it there.
# Expects the copy in /tmp/qa-599 (src, deploy, tests/acceptance). Refuses to start during a session.
Q=/tmp/qa-599
PY=/home/cyclops/cyclops/.venv/bin/python
SUM=$Q/tests/acceptance/hq_572_records_sum.sh
ROOT=/var/lib/cyclops-worker
HOOK=$ROOT/home/.claude/settings.json
cd ~/cyclops || exit 1

idle() {
  case "$(curl -s -m 3 localhost/api/live)" in
    *'"name": null'*) return 0 ;;
    *) echo "ABORT: not idle"; exit 2 ;;
  esac
}
restore() {
  echo "=== restore: the real program and the worker's own folder"
  sudo -n loginctl disable-linger cyclops-worker
  sudo -n systemctl reset-failed user@999.service 2>/dev/null
  sudo -n -u cyclops-worker rm -rf "$ROOT/home/qa" "$ROOT/home/.config"
  sudo -n -u cyclops-worker rm -f "$HOOK"
  echo "lingering: $(ls /var/lib/systemd/linger/ | tr '\n' ' ')"
  sudo -n sh "$Q/deploy/install-worker.sh" 2>&1 | grep -E 'Active:|error' | cut -c1-120
  sudo -n cmp "$(readlink -f /home/cyclops/.local/bin/claude-real)" /opt/cyclops-worker/bin/claude && echo "real program back"
  cmp "$Q/src/cyclops/worker.py" /opt/cyclops-worker/worker.py && echo "worker.py is this copy's"
}

idle
if sudo -n test -e "$HOOK"; then echo "ABORT: the worker's home already has settings.json"; exit 2; fi
bash "$SUM" $Q/before.txt
touch $Q/stamp
trap restore EXIT

echo "=== install, as swe asked for the retest"
sudo -n sh "$Q/deploy/install-worker.sh" 2>&1 | grep -E 'Active:|error' | cut -c1-120
cmp "$Q/src/cyclops/worker.py" /opt/cyclops-worker/worker.py && echo "worker.py is this copy's"

echo "=== the stand-in, in the real service"
sudo -n install -m 755 "$Q/tests/acceptance/hq_599_standin.sh" /opt/cyclops-worker/bin/claude
sudo -n systemctl restart cyclops-worker
sleep 2
PYTHONPATH=$Q/src timeout 2400 "$PY" -u "$Q/tests/acceptance/hq_599_hostile_device.py" \
  ${QA_CASES:-forge hardlink outlive answers kill_watcher output_fifo done_is_a_file long_name limit} 2>&1 | cut -c1-500
echo "journal, refusals: $(sudo -n journalctl -u cyclops-worker -b --no-pager | grep -c 'not from Cyclops, refused')"

if [ -n "${QA_ONLY_HOSTILE:-}" ]; then
  bash "$SUM" $Q/after.txt
  diff $Q/before.txt $Q/after.txt && echo RECORDS-IDENTICAL
  echo "--- files changed outside /tmp since the start"
  find ~ -xdev -type f -newer $Q/stamp -not -path "*/.config/chromium/*" -not -path "*/admin-profile/*" 2>/dev/null | cut -c1-120 | head -20
  echo R2-ALL-DONE
  exit 0
fi

echo "=== hostile PDFs, the call jobs._hand_in makes under the lock, on this box"
idle
PYTHONPATH=$Q/src timeout 900 nice "$PY" -u "$Q/tests/acceptance/hq_599_slow_pdf.py" --limit-s 180 --as-gb 4 2>&1 | cut -c1-200

echo "=== one real job, with a hook left in the worker's home as a job could leave it"
idle
sudo -n install -m 755 "$(readlink -f /home/cyclops/.local/bin/claude-real)" /opt/cyclops-worker/bin/claude
sudo -n systemctl restart cyclops-worker
sudo -n -u cyclops-worker mkdir -p "$ROOT/home/qa"
sudo -n -u cyclops-worker sh -c "umask 0027; printf '%s' '{\"hooks\":{\"SessionStart\":[{\"hooks\":[{\"type\":\"command\",\"command\":\"touch $ROOT/home/qa/from-hook\"}]}]}}' > $HOOK"
PYTHONPATH=$Q/src timeout 400 "$PY" -u "$Q/tests/acceptance/hq_599_hostile_device.py" real_job 2>&1 | cut -c1-500
sudo -n -u cyclops-worker rm -f "$HOOK"

echo "=== his records, after the hostile part"
bash "$SUM" $Q/after.txt
diff $Q/before.txt $Q/after.txt && echo RECORDS-IDENTICAL
echo "--- files changed outside /tmp since the start"
find ~ -xdev -type f -newer $Q/stamp -not -path "*/.config/chromium/*" -not -path "*/admin-profile/*" 2>/dev/null | cut -c1-120 | head -20
echo "--- his job ledger: $(ls ~/.cache/cyclops/jobs.json 2>/dev/null || echo none)"

echo "=== the real voice sessions, on this code"
bash "$Q/tests/acceptance/hq_599_pi.sh" 2>&1
echo R2-ALL-DONE
