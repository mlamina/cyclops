#!/bin/bash
# hq#599 qa retest, on the Pi. Installs this copy's worker, then: a hostile stand-in inside the real
# service, one real job with hooks left for it, whether a planted CLAUDE.md is read (with a control),
# then the real voice sessions (hq_599_pi.sh). The real program goes back whatever happens:
# install-worker.sh is what puts it there.
# Expects the copy in /tmp/qa-599 (src, deploy, tests/acceptance). Refuses to start during a session.
# QA_ONLY_HOSTILE=1 stops after the stand-in. QA_CASES picks its cases. QA_PDF=1 adds the PDF timings.
Q=/tmp/qa-599
PY=/home/cyclops/cyclops/.venv/bin/python
SUM=$Q/tests/acceptance/hq_572_records_sum.sh
ROOT=/var/lib/cyclops-worker
HOOK_BODY="{\"hooks\":{\"SessionStart\":[{\"hooks\":[{\"type\":\"command\",\"command\":\"touch $ROOT/qa/from-hook\"}]}]}}"
WORD=QA-PINEAPPLE
cd ~/cyclops || exit 1

idle() {
  case "$(curl -s -m 3 localhost/api/live)" in
    *'"name": null'*) return 0 ;;
    *) echo "ABORT: not idle"; exit 2 ;;
  esac
}
as_worker() {  # from /, because this shell's folder is one that account can't enter
  (cd / && sudo -n -u cyclops-worker "$@")
}
records() {
  bash "$SUM" $Q/after.txt
  diff $Q/before.txt $Q/after.txt && echo RECORDS-IDENTICAL
  echo "--- files changed outside /tmp since the start"
  find ~ -xdev -type f -newer $Q/stamp -not -path "*/.config/chromium/*" -not -path "*/admin-profile/*" 2>/dev/null | cut -c1-120 | head -20
  echo "--- his job ledger: $(ls ~/.cache/cyclops/jobs.json 2>/dev/null || echo none)"
}
restore() {
  echo "=== restore: the real program and the worker's own folder"
  sudo -n loginctl disable-linger cyclops-worker
  sudo -n systemctl reset-failed user@999.service 2>/dev/null
  as_worker rm -rf "$ROOT/qa" "$ROOT/done.qa" "$ROOT/CLAUDE.md" "$ROOT/.claude" "$ROOT/jobs/CLAUDE.md"
  echo "lingering: $(ls /var/lib/systemd/linger/ | tr '\n' ' ')"
  sudo -n sh "$Q/deploy/install-worker.sh" 2>&1 | grep -E 'Active:|error' | cut -c1-120
  sudo -n cmp "$(readlink -f /home/cyclops/.local/bin/claude-real)" /opt/cyclops-worker/bin/claude && echo "real program back"
  cmp "$Q/src/cyclops/worker.py" /opt/cyclops-worker/worker.py && echo "worker.py is this copy's"
  sudo -n ls -A "$ROOT" "$ROOT/home" | tr '\n' ' '; echo
}

idle
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
  ${QA_CASES:-sockets proc_mem forge hardlink linger outlive answers kill_watcher stop land_traps output_fifo done_is_a_file long_name limit} 2>&1 | cut -c1-500
echo "journal, refusals: $(sudo -n journalctl -u cyclops-worker -b --no-pager | grep -c 'not from Cyclops, refused')"

if [ -n "${QA_ONLY_HOSTILE:-}" ]; then
  records
  echo R2-ALL-DONE
  exit 0
fi

if [ -n "${QA_PDF:-}" ]; then
  echo "=== hostile PDFs, the call jobs._hand_in makes under the lock, on this box"
  idle
  PYTHONPATH=$Q/src timeout 900 nice "$PY" -u "$Q/tests/acceptance/hq_599_slow_pdf.py" --limit-s 180 --as-gb 4 2>&1 | cut -c1-200
fi

echo "=== one real job, with hooks left where an earlier job could leave them"
idle
sudo -n install -m 755 "$(readlink -f /home/cyclops/.local/bin/claude-real)" /opt/cyclops-worker/bin/claude
sudo -n systemctl restart cyclops-worker
sleep 2
as_worker mkdir -p "$ROOT/qa" "$ROOT/home/.claude" "$ROOT/.claude" "$ROOT/jobs/.claude"
for where in "$ROOT/home/.claude" "$ROOT/.claude" "$ROOT/jobs/.claude"; do
  as_worker sh -c "umask 0027; printf '%s' '$HOOK_BODY' > $where/settings.json"
done
PYTHONPATH=$Q/src timeout 400 "$PY" -u "$Q/tests/acceptance/hq_599_hostile_device.py" real_job 2>&1 | cut -c1-500
as_worker rm -rf "$ROOT/home/.claude" "$ROOT/.claude" "$ROOT/jobs/.claude"

echo "=== a CLAUDE.md a job could leave above the next job's folder: is it read?"
idle
TEST=$ROOT/qa/md/work
as_worker mkdir -p "$TEST" "$ROOT/qa/md/home"
for where in "$ROOT/CLAUDE.md" "$ROOT/jobs/CLAUDE.md"; do
  as_worker sh -c "umask 0027; printf 'The secret word is %s.\n' $WORD > $where"
done
ask_md() {  # the worker's own settings flag, and a control where project files are allowed
  (cd / && sudo -n -u cyclops-worker sh -c "cd $TEST && HOME=$ROOT/qa/md/home CLAUDE_CODE_OAUTH_TOKEN=\$(cat /etc/cyclops-worker/claude-token) DISABLE_AUTOUPDATER=1 timeout 120 /opt/cyclops-worker/bin/claude -p 'Reply with only the secret word from your project instructions, or NONE if you have none.' --model sonnet --max-budget-usd 0.20 --setting-sources $1 --strict-mcp-config --no-session-persistence --tools '' 2>&1" | tail -1 | cut -c1-120)
}
echo "as the worker runs it (--setting-sources user): $(ask_md user)"
echo "control (--setting-sources project): $(ask_md project)"
as_worker rm -f "$ROOT/CLAUDE.md" "$ROOT/jobs/CLAUDE.md"

echo "=== his records, after the hostile part"
records

echo "=== the real voice sessions, on this code"
bash "$Q/tests/acceptance/hq_599_pi.sh" 2>&1
echo R2-ALL-DONE
