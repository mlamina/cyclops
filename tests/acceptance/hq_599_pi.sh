#!/bin/bash
# hq#599, on the Pi: read-only checks as the worker's account, then real jobs with real voice-model
# turns from a throwaway copy, one process each. His records are compared before and after every run.
# Expects the copy in /tmp/qa-599 (src, tests/acceptance). Refuses to start during a real session.
Q=/tmp/qa-599
SUM=$Q/tests/acceptance/hq_572_records_sum.sh
cd ~/cyclops || exit 1
idle() {
  case "$(curl -s -m 3 localhost/api/live)" in
    *'"name": null'*) return 0 ;;
    *) echo "ABORT: not idle"; exit 2 ;;
  esac
}
idle
try() {  # one attempt as the worker's account; prints the outcome and never the contents
  label=$1; shift
  if sudo -n -u cyclops-worker "$@" >/dev/null 2>&1; then echo "  allowed  · $label"; else echo "  refused  · $label"; fi
}
echo "=== boundary as the worker's account, by sudo -u: file rights and firewall (not the unit's sandbox)"
try "read .env" cat /home/cyclops/cyclops/.env
try "list sessions" ls /home/cyclops/cyclops/sessions
try "read his Claude login" cat /home/cyclops/.claude/.credentials.json
try "write /etc" touch /etc/qa-599-probe
try "write /usr/local/bin" touch /usr/local/bin/qa-599-probe
try "admin page, 127.0.0.1" curl -s -m 4 -o /dev/null http://127.0.0.1/
try "admin page, localhost" curl -s -m 4 -o /dev/null http://localhost/
try "the internet (control: should be allowed)" curl -s -m 8 -o /dev/null https://example.com/
try "write its own folder (control: should be allowed)" touch /var/lib/cyclops-worker/home/.qa-599-probe
sudo -n -u cyclops-worker rm -f /var/lib/cyclops-worker/home/.qa-599-probe
echo "=== D-Bus as the worker's account: questions only, nothing is asked to happen"
sudo -n -u cyclops-worker busctl --system --no-pager list 2>&1 | awk 'NR>1 && $1 !~ /^:/ {print "  name: " $1}' | head -20
for what in CanReboot CanPowerOff CanSuspend; do
  printf '  login1 %s: ' "$what"
  sudo -n -u cyclops-worker busctl --system call org.freedesktop.login1 /org/freedesktop/login1 \
    org.freedesktop.login1.Manager "$what" 2>&1 | cut -c1-90
done
printf '  NetworkManager permissions: '
sudo -n -u cyclops-worker nmcli -t general permissions 2>&1 | grep -c ':yes$'
printf '  wifi connections it can list: '
sudo -n -u cyclops-worker nmcli -t -f TYPE connection show 2>&1 | grep -c wireless
printf '  of them, secrets it can read: '
sudo -n -u cyclops-worker nmcli -s -t -f 802-11-wireless-security.psk connection show "$(nmcli -t -f NAME,TYPE connection show --active | grep wireless | cut -d: -f1 | head -1)" 2>/dev/null | grep -c ':.'

for which in ${QA_WHICH:-open ended fail}; do
  echo "=== $which"
  idle
  bash "$SUM" $Q/before.txt
  touch $Q/stamp
  XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0 PYTHONPATH=$Q/src timeout 900 \
    .venv/bin/python -u $Q/tests/acceptance/hq_599_real_sessions.py "$which" 2>&1 \
    | grep -vE '^· (mic|continuity|right now|projects|connected|listening)' | cut -c1-700
  echo "run exit=${PIPESTATUS[0]}"
  bash "$SUM" $Q/after.txt
  diff $Q/before.txt $Q/after.txt && echo RECORDS-IDENTICAL
  echo "--- files changed outside /tmp since the start"
  find ~ -xdev -type f -newer $Q/stamp -not -path "*/.config/chromium/*" -not -path "*/admin-profile/*" 2>/dev/null | cut -c1-120 | head -20
  echo "--- his job ledger: $(ls ~/.cache/cyclops/jobs.json 2>/dev/null || echo none)"
  sleep 5
done
echo ALL-DONE
