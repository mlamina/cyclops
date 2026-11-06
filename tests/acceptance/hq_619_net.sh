#!/bin/bash
# hq#619, on the Pi: what a second headless browser with the real network flags connects to, idle and playing one video.
# Needs this script, hq_619_read.py and hq_572_records_sum.sh copied to /tmp/swe-619, named as below. Never add
# --default-background-color: it makes headless do one job and quit. A short run says TOO SHORT and proves little.
S=/tmp/swe-619
SUM=$S/hq_572_records_sum.sh
idle() {
  case "$(curl -s -m 3 localhost/api/live)" in
    *'"name": null'*) return 0 ;;
    *) echo "ABORT: not idle"; exit 2 ;;
  esac
}
idle
bash "$SUM" $S/before.txt
touch $S/stamp
BASE="--headless=new --ozone-platform=headless --noerrdialogs --disable-infobars --no-first-run --no-default-browser-check --disable-session-crashed-bubble --disable-component-update --password-store=basic --force-device-scale-factor=1 --autoplay-policy=no-user-gesture-required --disable-features=Translate --mute-audio"

mkdir -p $S/www $S/yt/home
echo '<!doctype html><title>blank</title><body bgcolor=black>' > $S/www/index.html
: > $S/www.log
(cd $S/www && exec python3 -m http.server 18619 --bind 127.0.0.1 >>$S/www.log 2>&1) &
WWW=$!
trap 'kill $WWW 2>/dev/null' EXIT
sleep 1

traced() {
  name=$1; secs=$2; url=$3; shift 3
  idle
  h=$S/r6-$name
  rm -rf "$h"; mkdir -p "$h/home"
  cp -a ~/.cache/cyclops/admin-profile "$h/profile" 2>/dev/null
  rm -f "$h"/profile/Singleton*
  echo "=== $name (limit $secs s)"
  t0=$(date +%s)
  HOME="$h/home" XDG_CONFIG_HOME="$h/home/.config" XDG_CACHE_HOME="$h/home/.cache" \
    timeout -k 5 "$secs" strace -f -qq -e trace=socket,connect,sendto,sendmsg,sendmmsg,write -o "$h/strace.log" \
    chromium $BASE --disable-background-networking --proxy-server=127.0.0.1:9 --proxy-bypass-list=*.googlevideo.com "--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE *.googlevideo.com, EXCLUDE localhost, EXCLUDE 127.0.0.1" --user-data-dir="$h/profile" "$url" >"$h/out.log" 2>&1
  code=$?
  ran=$(( $(date +%s) - t0 ))
  echo "  really ran $ran s, exit $code$([ "$ran" -lt $((secs - 2)) ] && echo '  TOO SHORT: this read proves little')"
  python3 "$S/hq_619_read.py" "$h/strace.log"
  echo "  browser log, push service lines: $(grep -c 'google_apis/gcm' "$h/out.log")"
  rm -rf "$h/profile" "$h/home" "$h/strace.log"
}

traced new 180 http://127.0.0.1:18619/

echo "=== the video's address"
idle
cd $S/yt || exit 1
STREAM=$(HOME="$S/yt/home" XDG_CACHE_HOME="$S/yt/home/.cache" OPENAI_API_KEY=scratch-not-a-key ~/cyclops/.venv/bin/python - <<'PY' 2>/dev/null
from cyclops import youtube
v = youtube.details("jNQXAC9IVRw")
print(v.stream if v else "")
PY
)
cd $S
if [ -z "$STREAM" ]; then
  echo "NOT RUN: no stream"
else
  HOST=$(python3 -c 'import sys,urllib.parse as u; print(u.urlparse(sys.argv[1]).hostname)' "$STREAM")
  echo "stream host: $HOST   resolves to: $(getent ahosts "$HOST" | awk '{print $1}' | sort -u | tr '\n' ' ')"
  python3 - "$STREAM" > $S/www/video.html <<'PY'
import html, sys
src = html.escape(sys.argv[1], quote=True)
print(f"""<!doctype html><title>video</title><body bgcolor=black>
<video id=v muted autoplay src="{src}"></video>
<script>
const v = document.getElementById('v');
const keep = [];
const report = (why) => {{ const i = new Image(); keep.push(i); i.src = '/state?' + why + '&ready=' + v.readyState
  + '&t=' + Math.round(v.currentTime) + '&dur=' + Math.round(v.duration || 0) + '&paused=' + v.paused
  + '&err=' + (v.error ? v.error.code : 'none'); }};
setInterval(() => report('tick'), 5000);
v.addEventListener('ended', () => report('ended'));
v.addEventListener('error', () => report('error'));
</script>""")
PY
  traced video 60 http://127.0.0.1:18619/video.html
  echo "  what the page reported:"
  grep -oE 'GET /state\?[^ ]+' $S/www.log | sed -n '1p;3p;5p;$p' | sed 's/^/    /'
  grep -oE 'GET /state\?ended[^ ]+' $S/www.log | head -n 1 | sed 's/^/    /'
fi

bash "$SUM" $S/after.txt
diff $S/before.txt $S/after.txt && echo RECORDS-IDENTICAL
echo "--- files changed outside /tmp since the start"
find ~ -xdev -type f -newer $S/stamp -not -path "*/.config/chromium/*" -not -path "*/admin-profile/*" 2>/dev/null | cut -c1-120 | head -12
rm -f $S/www/video.html
echo DONE
