#!/bin/bash
# hq#602, on the Pi: the real naming, summary, memory and filing on a no-video folder and a recorded
# one, against scratch. Expects /tmp/qa-602 with the code and tonight's folders in out/.
# HOME and every record folder are scratch; a preflight refuses to go on if any setting isn't.
Q=/tmp/qa-602
A=$Q/after-scratch
cd ~/cyclops || exit 1
case "$(curl -s -m 3 localhost/api/live)" in
  *'"name": null'*) ;;
  *) echo "ABORT: not idle"; exit 2 ;;
esac
mkdir -p $A/home/.cache/cyclops $A/sessions $A/projects $A/manuals $A/captures
cp -a $Q/out/s1-no-video/2026-* $Q/out/s2-recorded/2026-* $A/sessions/
export HOME=$A/home CYCLOPS_SESSIONS_DIR=$A/sessions CYCLOPS_PROJECTS_DIR=$A/projects \
  CYCLOPS_MANUALS_DIR=$A/manuals CYCLOPS_CAPTURES_DIR=$A/captures CYCLOPS_ABOUT_FILE=$A/about-you.md \
  CYCLOPS_SLUG=1 CYCLOPS_REMEMBER=1 CYCLOPS_PROJECTS=1 PYTHONPATH=$Q/src
PY=/home/cyclops/cyclops/.venv/bin/python
$PY - "$A" <<'CHECK' || exit 3
import sys
from dataclasses import fields
from pathlib import Path

from cyclops import tasks
from cyclops.config import load_settings

root = Path(sys.argv[1]).resolve()
s = load_settings()
for f in fields(s):
    if f.name.endswith(("_dir", "_file")) and f.name != "worker_dir":
        v = getattr(s, f.name)
        if v and not Path(v).resolve().is_relative_to(root):
            raise SystemExit(f"refusing: {f.name}={v}")
if not Path(tasks.TASKS_FILE).resolve().is_relative_to(root):
    raise SystemExit(f"refusing: {tasks.TASKS_FILE}")
print("preflight: every record path is scratch")
CHECK
HOME=/home/cyclops bash $Q/tests/acceptance/hq_572_records_sum.sh $Q/before.txt
touch $Q/stamp
for folder in "$A"/sessions/2026-*; do
  echo "=== after $(basename "$folder")"
  timeout 240 $PY -m cyclops.after "$folder" 2>&1 | cut -c1-200 | tail -12
  echo "exit=${PIPESTATUS[0]}"
done
echo "--- sessions now"
for folder in "$A"/sessions/*; do
  echo "$(basename "$folder"): $(ls "$folder" | tr '\n' ' ')"
  [ -f "$folder/summary.md" ] && head -c 220 "$folder/summary.md" && echo
done
echo "--- projects: $(ls $A/projects | tr '\n' ' ')"
$PY - <<'LIST'
from cyclops import library
from cyclops.config import load_settings

s = load_settings()
for p in sorted(s.sessions_dir.iterdir()):
    e = library._read(p)
    print(f"{p.name}: verdict={e.verdict} video={e.video}")
LIST
HOME=/home/cyclops bash $Q/tests/acceptance/hq_572_records_sum.sh $Q/after.txt
diff $Q/before.txt $Q/after.txt && echo RECORDS-IDENTICAL
echo "--- files changed outside /tmp"
find /home/cyclops -xdev -type f -newer $Q/stamp -not -path "*/.config/chromium/*" -not -path "*/admin-profile/*" 2>/dev/null | cut -c1-120 | head
echo ALL-DONE
