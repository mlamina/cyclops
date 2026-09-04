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
# The wallpaper is already the same splash.png plymouth shows, so making the fill black and
# taking the furniture off is the whole of it: the splash simply stays up from ten seconds in
# until the camera preview replaces it. Run it once per Pi, like the other install-*.sh. The
# desktop half is user config; the taskbar half edits one line of a session file under /etc and
# keeps the original beside it, so both are reversible by hand.
set -eu

DESKTOP="$HOME/.config/pcmanfm/LXDE-pi/desktop-items-DSI-1.conf"

# --- the desktop: black behind the splash, and nothing on top of it -----------------------
# desktop_bg is what pcmanfm paints before it has decoded the wallpaper, and what shows wherever
# a cropped wallpaper does not reach. Black is the only value that cannot be seen against
# splash.png, which is black to its edges.
if [ -f "$DESKTOP" ]; then
  sed -i \
    -e 's/^desktop_bg=.*/desktop_bg=#000000/' \
    -e 's/^desktop_shadow=.*/desktop_shadow=#000000/' \
    -e 's/^show_documents=.*/show_documents=0/' \
    -e 's/^show_trash=.*/show_trash=0/' \
    -e 's/^show_mounts=.*/show_mounts=0/' \
    "$DESKTOP"
  echo "· desktop: black behind the splash, no icons ($DESKTOP)"
else
  echo "· no $DESKTOP - is this the LXDE-pi labwc session on the DSI panel?" >&2
fi

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
