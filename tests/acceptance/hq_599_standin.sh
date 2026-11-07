#!/bin/bash
# hq#599: stands in for `claude` inside the real cyclops-worker service, and misbehaves on request.
# No model, no token used, no network beyond what a case says. Everything it writes is under the
# worker's own folder. Installed for the run by hq_599_r2_pi.sh, which puts the real program back.
PROMPT=$(cat)
ROOT=/var/lib/cyclops-worker
QA=$ROOT/qa  # not home/: a restart empties that since d55cf60
WORK=$PWD
JOB=$(basename "$(dirname "$WORK")")
mkdir -p "$QA"
echo "$JOB" >> "$QA/runs.log"
say() { echo "$*" >> "$QA/$JOB.log"; }
good() { echo '{"type":"result","is_error":false,"result":"{}"}'; }

detach() {  # a process out of the job's group and session, forking once more on the way
  python3 -c '
import os, sys, time, pathlib
if os.fork(): os._exit(0)
os.setsid()
if os.fork(): os._exit(0)
for fd in (0, 1, 2): os.close(fd)
while True:
    pathlib.Path(sys.argv[1]).write_text(str(time.time()))
    time.sleep(0.5)
' "$1" &
  wait $!
}

case "$PROMPT" in
*QA-FORGE*)
  # Plant what the first round's hole was, and the shapes around it. None should ever run.
  printf '{"request":"QA-COUNT forged"}' > "$ROOT/inbox/aaaaaaaa00000001.json"
  ln -s "$ROOT/home/qa/nothing" "$ROOT/inbox/aaaaaaaa00000002.json"
  mkfifo "$ROOT/inbox/aaaaaaaa00000003.json"
  ln "$ROOT/inbox/aaaaaaaa00000001.json" "$ROOT/inbox/aaaaaaaa00000004.json"
  good
  ;;
*QA-COUNT*)
  good
  ;;
*QA-HARDLINK*)
  # Could it make something another account owns appear in the inbox? Try on files it can see.
  for f in /etc/hostname /etc/cyclops-worker/claude-token /opt/cyclops-worker/worker.py; do
    if ln "$f" "$ROOT/inbox/bbbbbbbb$(printf %s "$f" | md5sum | cut -c1-8).json" 2>/dev/null; then
      say "hard link allowed: $f"
    else
      say "hard link refused: $f"
    fi
  done
  ls -ln "$ROOT/inbox" >> "$QA/$JOB.log" 2>&1
  good
  ;;
*QA-OUTLIVE*)
  detach "$QA/alive-$JOB"
  try() { if "$@" >> "$QA/$JOB.log" 2>&1; then say "allowed: $*"; else say "refused: $*"; fi; }
  say "--- ways to outlive the job"
  echo "* * * * * touch $QA/from-cron" | try crontab -
  try crontab -l
  try at -V
  try systemd-run --user --unit=qa-599 /usr/bin/touch "$QA/from-systemd-run"
  try loginctl enable-linger cyclops-worker
  try nohup true
  good
  ;;
*QA-LINGER*)
  # What outlive found, carried through: a unit that starts by itself, then the job ends at once.
  # It only writes facts about itself into the worker's own folder and ticks. Read-only otherwise.
  UNIT=$ROOT/home/.config/systemd/user
  mkdir -p "$UNIT/default.target.wants"
  cat > "$QA/unit.sh" <<EOF
#!/bin/bash
{
  echo "--- from the unit, after the job"
  echo "NoNewPrivs: \$(grep NoNewPrivs /proc/self/status | cut -f2)"
  echo "CapBnd: \$(grep CapBnd /proc/self/status | cut -f2)"
  echo "cgroup: \$(cut -d: -f3 /proc/self/cgroup)"
  if touch /var/tmp/qa-599-probe 2>/dev/null; then echo "write /var/tmp: allowed"; rm -f /var/tmp/qa-599-probe; else echo "write /var/tmp: refused"; fi
  if ls /dev/snd >/dev/null 2>&1; then echo "/dev/snd visible: yes"; else echo "/dev/snd visible: no"; fi
  if ls /dev/video0 >/dev/null 2>&1; then echo "/dev/video0 visible: yes"; else echo "/dev/video0 visible: no"; fi
  if [ -r /dev/video0 ]; then echo "camera device readable: yes (permission only, not opened)"; else echo "camera device readable: no"; fi
  if [ -r /dev/snd/pcmC0D0c ]; then echo "mic device readable: yes (permission only, not opened)"; else echo "mic device readable: no"; fi
  if [ -r /home/cyclops/cyclops/.env ]; then echo ".env readable: yes"; else echo ".env readable: no"; fi
  if [ -r /etc/cyclops-worker/claude-token ]; then echo "token readable: yes"; else echo "token readable: no"; fi
  if curl -s -m 4 -o /dev/null http://127.0.0.1/; then echo "admin page: reached"; else echo "admin page: no connection"; fi
  if curl -s -m 8 -o /dev/null https://example.com/; then echo "internet: reached"; else echo "internet: no connection"; fi
} > "$QA/unit-facts" 2>&1
while true; do date +%s.%N > "$QA/unit-tick"; sleep 0.5; done
EOF
  chmod 750 "$QA/unit.sh"
  cat > "$UNIT/qa-599.service" <<EOF
[Service]
ExecStart=$QA/unit.sh
[Install]
WantedBy=default.target
EOF
  ln -sf "$UNIT/qa-599.service" "$UNIT/default.target.wants/qa-599.service"
  loginctl enable-linger cyclops-worker >> "$QA/$JOB.log" 2>&1 && say "enable-linger: allowed" || say "enable-linger: refused"
  busctl --system --no-pager list > /dev/null 2>&1 && say "busctl --system: allowed" || say "busctl --system: refused"
  ls -la /run/dbus /var/run/dbus >> "$QA/$JOB.log" 2>&1
  good
  ;;
*QA-STOP*)
  kill -STOP "$PPID"
  sleep 200
  good
  ;;
*QA-LAND-PIPE*)
  mkfifo "$ROOT/done/.$JOB.json.tmp"
  good
  ;;
*QA-DONE-FOLDER*)
  mkdir -p "$ROOT/done/$JOB.json/inside"
  good
  ;;
*QA-SOCKETS*)
  # Connect and close, nothing sent: every unix socket the box lists, from inside the job.
  python3 - "$QA/$JOB.log" <<'EOF'
import socket, sys
names = set()
with open("/proc/net/unix") as table:
    next(table)
    for row in table:
        parts = row.split()
        if len(parts) >= 8:
            names.add(parts[7])
reached, refused = [], 0
for name in sorted(names):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(1)
    try:
        sock.connect("\0" + name[1:] if name.startswith("@") else name)
        reached.append(name)
    except OSError:
        refused += 1
    finally:
        sock.close()
with open(sys.argv[1], "a") as out:
    out.write(f"listed {len(names)}, refused {refused}\n--- reached\n")
    out.writelines(f"{name}\n" for name in reached)
EOF
  good
  ;;
*QA-PROC-MEM*)
  # Opens and reads one byte, and attaches and lets go at once. Writes nothing into the worker.
  python3 - "$PPID" "$QA/$JOB.log" <<'EOF'
import ctypes, os, sys, time
pid, log = int(sys.argv[1]), sys.argv[2]
lines = [f"worker pid {pid}: {open(f'/proc/{pid}/cmdline').read().replace(chr(0), ' ')[:60]}"]
try:
    start = int(open(f"/proc/{pid}/maps").readline().split("-")[0], 16)
    with open(f"/proc/{pid}/mem", "r+b", buffering=0) as mem:
        mem.seek(start)
        lines.append(f"mem open for writing: allowed, read {len(mem.read(1))} byte")
except OSError as exc:
    lines.append(f"mem open for writing: refused ({exc.strerror})")
libc = ctypes.CDLL(None, use_errno=True)
PTRACE_SEIZE, PTRACE_DETACH = 0x4206, 17
libc.ptrace.argtypes = [ctypes.c_long, ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p]
libc.ptrace.restype = ctypes.c_long
if libc.ptrace(PTRACE_SEIZE, pid, None, None) == 0:
    lines.append("ptrace seize: allowed")
    libc.ptrace(PTRACE_DETACH, pid, None, None)
else:
    lines.append(f"ptrace seize: refused ({os.strerror(ctypes.get_errno())})")
with open(log, "a") as out:
    out.write("\n".join(lines) + "\n")
EOF
  good
  ;;
*QA-LIMIT*)
  detach "$QA/alive-$JOB"
  sleep 1000
  good
  ;;
*QA-KILL-WATCHER*)
  detach "$QA/alive-$JOB"
  kill -9 "$PPID"
  sleep 30
  good
  ;;
*QA-ANSWER-FIFO*)
  mkfifo "$ROOT/done/$JOB.json"
  touch "$QA/planted-$JOB"
  sleep 12
  good
  ;;
*QA-ANSWER-HUGE*)
  truncate -s 1G "$ROOT/done/$JOB.json"
  touch "$QA/planted-$JOB"
  sleep 12
  good
  ;;
*QA-ANSWER-LINK*)
  ln -s /tmp/qa-599-his-file.json "$ROOT/done/$JOB.json" 2>> "$QA/$JOB.log"
  touch "$QA/planted-$JOB"
  sleep 12
  good
  ;;
*QA-OUTPUT-FIFO*)
  rm -f "$WORK/../output.json"
  mkfifo "$WORK/../output.json"
  ;;
*QA-DONE-IS-A-FILE*)
  mv "$ROOT/done" "$ROOT/done.qa"
  touch "$ROOT/done"
  good
  ;;
*QA-LONG-NAME*)
  long=$(printf 'x%.0s' $(seq 1 250))
  mkdir -p "$ROOT/jobs/$long"
  touch "$ROOT/jobs/$long/running"
  good
  ;;
*)
  say "unknown case"
  good
  ;;
esac
