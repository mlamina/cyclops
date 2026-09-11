#!/bin/sh
# Make the desktop underneath the kiosk look like the boot splash, because for a few seconds of
# every boot it *is* what the panel is showing.
#
# The kiosk is the last thing to start on this box, and between plymouth letting go and the
# kiosk's window mapping there is a stretch - forty seconds on a cold, contended boot - where
# labwc's LXDE session owns the panel. What it drew there was a near-white fill (#d6d3de) until
# the wallpaper decoded, a grey band under a wallpaper that does not reach the bottom, and a
# taskbar with a Raspberry menu and a wifi icon. That is what "a white screen appears" was.
#
# Hanging the same splash.png plymouth shows on the desktop, making the fill black and taking
# the furniture off is the whole of it: the splash simply stays up from ten seconds in until the
# camera preview replaces it. Run it once per Pi, like the other install-*.sh. The
# desktop half is user config; the taskbar half edits one line of a session file under /etc and
# keeps the original beside it, so both are reversible by hand.
set -eu

# --- the desktop: the splash, black behind it, and nothing on top -------------------------
# pcmanfm keeps one config file per output, named after the DRM connector it found:
# desktop-items-DSI-1.conf. That name is not stable. Fitting the Camera Module renumbered the
# panel from DSI-1 to DSI-2, pcmanfm looked for a file that was not there, and the desktop went
# back to the stock Pi wallpaper for the whole stretch before the kiosk maps. So the file is
# written for whatever output is actually connected right now, rather than patched at a name
# baked in here - and written whole, so the wallpaper is part of what this script guarantees
# instead of something someone once set by hand.
#
# desktop_bg is what pcmanfm paints before it has decoded the wallpaper, and what shows wherever
# a cropped wallpaper does not reach. Black is the only value that cannot be seen against
# splash.png, which is black to its edges.
SPLASH="$HOME/cyclops/src/cyclops/assets/splash.png"
DESKTOP_DIR="$HOME/.config/pcmanfm/LXDE-pi"
mkdir -p "$DESKTOP_DIR"

[ -f "$SPLASH" ] || echo "· no $SPLASH - push the repo first" >&2

wrote=
for conn in /sys/class/drm/card*-*; do
  [ "$(cat "$conn/status" 2>/dev/null)" = connected ] || continue
  name=${conn##*/}
  name=${name#card*-}
  cat > "$DESKTOP_DIR/desktop-items-$name.conf" <<EOF
[*]
wallpaper_mode=crop
wallpaper_common=1
wallpaper=$SPLASH
desktop_bg=#000000
desktop_fg=#e8e8e8
desktop_shadow=#000000
desktop_font=PibotoLt 12
show_wm_menu=0
sort=mtime;ascending;
show_documents=0
show_trash=0
show_mounts=0
EOF
  echo "· desktop: splash on $name, black behind it, no icons"
  wrote=1
done
[ -n "$wrote" ] || echo "· no connected DRM output - is the panel plugged in?" >&2

# pcmanfm reads the file at start and not again. lwrespawn puts it straight back, so killing it
# is how the new desktop appears without a reboot. -x matches the process name, not the command
# line, so this cannot match the shell running the script.
pkill -x pcmanfm 2>/dev/null || true

# --- the taskbar: not launched at all ------------------------------------------------------
# wf-panel-pi draws the Raspberry menu, the launchers, the wifi and update icons and their
# notification bubbles across the top eighth of the panel. It is started by the session, not by
# us, and it cannot be talked out of appearing: its `layer` and `monitor` options were both
# tried and neither hides it (it falls back to the first output), and killing it does not stick
# because lwrespawn restarts it for as long as labwc lives. So it is taken out of the session.
#
# The original is kept beside it. A distribution upgrade that ships a new autostart will restore
# the taskbar, and re-running this script is the fix.
AUTOSTART=/etc/xdg/labwc/autostart
NOTE="# wf-panel-pi: the LXDE taskbar. Not launched on this box - the panel is a Cyclops kiosk"
if [ -f "$AUTOSTART" ] && grep -q '^/usr/bin/lwrespawn /usr/bin/wf-panel-pi' "$AUTOSTART"; then
  [ -f "$AUTOSTART.before-cyclops" ] || sudo cp "$AUTOSTART" "$AUTOSTART.before-cyclops"
  sudo sed -i "s|^/usr/bin/lwrespawn /usr/bin/wf-panel-pi &|$NOTE|" "$AUTOSTART"
  echo "· taskbar: no longer started by the session ($AUTOSTART)"
elif [ -f "$AUTOSTART" ]; then
  echo "· taskbar: already out of $AUTOSTART"
else
  echo "· no $AUTOSTART - leaving the taskbar alone" >&2
fi

# --- the pointer: a cursor with nothing in it ------------------------------------------------
# There is no mouse. Every touch arrives as one anyway (rc.xml has mouseEmulation="yes", and the
# kiosk reads taps as mouse events), so an arrow gets left sitting on the panel wherever the last
# finger came off - and at boot labwc parks one on the splash before anything has been touched
# at all.
#
# labwc has no switch for hiding it. XCURSOR_SIZE=1 and <theme><cursor size="1"/> were both
# tried and neither takes: the size is clamped somewhere below us and a 12-pixel arrow comes
# back. What does work is giving it a cursor that is genuinely empty - a one-pixel, fully
# transparent Xcursor - and pointing both labwc and its clients at that theme. Nothing is
# hidden, so nothing has to be un-hidden: a tap still lands exactly where the finger is.
THEME="$HOME/.icons/cyclops-blank"
mkdir -p "$THEME/cursors"
cat > "$THEME/index.theme" <<'EOF'
[Icon Theme]
Name=cyclops-blank
Comment=A cursor with nothing in it, for a panel that is touched rather than pointed at
EOF

# The Xcursor container by hand, because xcursorgen is not on a Pi OS image: a file header, a
# one-entry table of contents, and one 1x1 image whose single pixel is transparent ARGB.
python3 - "$THEME/cursors/left_ptr" <<'EOF'
import struct
import sys

NOMINAL = 24  # what a caller asking for "a normal cursor" is matched against; the only image
IMAGE_TYPE = 0xFFFD0002

header = struct.pack("<4sIII", b"Xcur", 16, 0x00010000, 1)
toc = struct.pack("<III", IMAGE_TYPE, NOMINAL, len(header) + 12)
image = struct.pack(
    "<IIIIIIIII", 36, IMAGE_TYPE, NOMINAL, 1, 1, 1, 0, 0, 0
) + struct.pack("<I", 0)  # width 1, height 1, hotspot 0,0, no delay, one transparent pixel
with open(sys.argv[1], "wb") as out:
    out.write(header + toc + image)
EOF

# Every name anything here might ask for. A name the theme does not carry falls back to a theme
# that does, which would put the arrow straight back.
for name in default arrow top_left_arrow left_ptr_watch watch wait progress pointer \
            hand hand1 hand2 xterm text ibeam crosshair cross fleur move grab grabbing \
            not-allowed no-drop help question_arrow sb_h_double_arrow sb_v_double_arrow \
            n-resize s-resize e-resize w-resize ne-resize nw-resize se-resize sw-resize \
            col-resize row-resize all-scroll; do
  [ "$name" = "left_ptr" ] || ln -sf left_ptr "$THEME/cursors/$name"
done
echo "· pointer: a blank cursor theme in $THEME"

# ...and point labwc at it, both ways it will be asked: its own cursor comes from rc.xml, and
# every client under it - the kiosk's Qt window included - reads the environment file.
ENVIRONMENT="$HOME/.config/labwc/environment"
mkdir -p "$(dirname "$ENVIRONMENT")"
touch "$ENVIRONMENT"
sed -i '/^XCURSOR_SIZE=/d' "$ENVIRONMENT"
if grep -q '^XCURSOR_THEME=' "$ENVIRONMENT"; then
  sed -i 's/^XCURSOR_THEME=.*/XCURSOR_THEME=cyclops-blank/' "$ENVIRONMENT"
else
  printf 'XCURSOR_THEME=cyclops-blank\n' >> "$ENVIRONMENT"
fi

RC="$HOME/.config/labwc/rc.xml"
if [ -f "$RC" ]; then
  if grep -q '<cursor ' "$RC"; then
    sed -i 's|<cursor [^/]*/>|<cursor theme="cyclops-blank"/>|' "$RC"
  else
    sed -i 's|</openbox_config>|\t<theme>\n\t\t<cursor theme="cyclops-blank"/>\n\t</theme>\n</openbox_config>|' "$RC"
  fi
  echo "· pointer: labwc and its clients pointed at it ($RC, $ENVIRONMENT)"
fi

echo "· log out and back in (or reboot) for these to take effect"
