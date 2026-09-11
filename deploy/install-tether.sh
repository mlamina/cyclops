#!/bin/sh
# Teach the Pi to fall back to a phone hotspot when the home network isn't there.
# Idempotent; run on the Pi, once per box (and again to change the hotspot).
#   deploy/install-tether.sh [SSID]
#
# This is all NetworkManager, no cyclops code: a second saved wifi profile with a lower
# autoconnect priority than home. NM only reaches for it when the higher-priority network
# is out of range, so the box is on the hotspot exactly when it has no other way out.
#
# The switch back is not symmetric, and that is deliberate. NM reconsiders priorities when a
# device goes down, not while a connection is working - so coming home does not pull the box
# off the phone. Turning the hotspot off does: the link drops, NM rescans, home wins. That is
# the gesture to remember, and it is also the one that stops cellular data being spent by
# accident.
#
# The password is prompted for rather than taken as an argument, so it stays out of shell
# history and out of the scrollback of whoever runs this. It is still handed to nmcli in the
# usual way, so it is briefly visible in `ps` - on a single-user box that is a fair trade for
# not having a second secret store to keep.
set -eu

# nmcli reads fine as the user, but every write here goes through polkit - and over ssh there
# is no local session for it to consent on behalf of, so writes are refused. sudo is not
# belt-and-braces; without it this script only works sitting at the panel.
SUDO=""
[ "$(id -u)" -eq 0 ] || SUDO="sudo"

HOME_PRIORITY=100     # anything already saved: the networks with no data plan behind them
PHONE_PRIORITY=10     # the hotspot: wanted, but only when nothing better is on the air

# ---- the phone, over the air ----
SSID=${1:-}
if [ -z "$SSID" ]; then
  echo "· scanning (turn Personal Hotspot on now - iOS only broadcasts while it is on)"
  nmcli -t -f SSID device wifi list --rescan yes | grep . | sort -u | sed 's/^/    /'
  printf 'SSID: '
  read -r SSID
fi
[ -n "$SSID" ] || { echo "no SSID; nothing to do" >&2; exit 1; }

# Hidden at a terminal; plain on a pipe, so this can also be driven over ssh without a tty.
if [ -t 0 ]; then
  printf 'Password for "%s" (not echoed): ' "$SSID"
  stty -echo; read -r PSK; stty echo; echo
else
  read -r PSK
fi

# Every wifi profile that is not the phone is a network we would rather be on. Raising them
# all - rather than naming the home one - means this keeps working after a move, or on a
# second Pi, without editing the script.
nmcli -t -f NAME,TYPE connection show | while IFS=: read -r name type; do
  [ "$type" = "802-11-wireless" ] || continue
  [ "$name" != "$SSID" ] || continue
  $SUDO nmcli connection modify "$name" connection.autoconnect-priority "$HOME_PRIORITY"
  echo "· $name: priority $HOME_PRIORITY"
done

$SUDO nmcli connection delete "$SSID" >/dev/null 2>&1 || true
$SUDO nmcli connection add type wifi con-name "$SSID" ssid "$SSID" \
  wifi-sec.key-mgmt wpa-psk wifi-sec.psk "$PSK" \
  connection.autoconnect yes \
  connection.autoconnect-priority "$PHONE_PRIORITY" \
  connection.autoconnect-retries 0 >/dev/null
# A hotspot is absent far more often than it is present, and the stock four retries would
# give up and leave the profile blocked. 0 is unlimited: keep watching for it, forever.
echo "· $SSID: priority $PHONE_PRIORITY, retries unlimited"

# ---- the phone, over a cable ----
# ipheth is already in the Pi kernel, so the interface itself needs nothing installed. What is
# missing is usbmuxd, which is what answers the phone's "Trust this computer?" and holds the
# pairing afterwards. Without it the device enumerates and never carries traffic. NM picks the
# interface up as plain ethernet and DHCPs it with no profile of ours.
if dpkg -s usbmuxd >/dev/null 2>&1; then
  echo "· usbmuxd already installed"
else
  $SUDO apt-get install -y -qq usbmuxd >/dev/null && echo "· usbmuxd installed"
fi

echo
nmcli -t -f NAME,TYPE,AUTOCONNECT-PRIORITY connection show \
  | awk -F: '$2=="802-11-wireless"{printf "  %-28s %s\n", $1, $3}'
echo "· on: $(nmcli -t -f NAME,TYPE connection show --active | awk -F: '$2!="loopback"{print $1; exit}')"
