#!/bin/sh
# Install (or reinstall) the I2S speaker: the DAC overlay, and the mono sink in front of it.
# Idempotent; run on the Pi, once per box.
#
# Two halves, because they take effect at different moments. The overlay is a boot-time device
# tree change and needs a reboot before a sound card exists at all; the PipeWire config is a
# user-session file that the next `systemctl --user restart pipewire` picks up. Both are claims
# worth proving rather than assuming, which is why this ends by reporting what it can see.
set -eu

CONFIG=/boot/firmware/config.txt
CONF_DIR="$HOME/.config/pipewire/pipewire.conf.d"
HERE=$(cd "$(dirname "$0")" && pwd)
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}

# ---- the card ----
if grep -q '^dtoverlay=hifiberry-dac' "$CONFIG"; then
  echo "· overlay already in $CONFIG"
else
  sudo cp "$CONFIG" "$CONFIG.bak-$(date +%Y%m%d)"
  printf '\n# NS4168 I2S DAC amp on GPIO18/19/21\ndtparam=i2s=on\ndtoverlay=hifiberry-dac\n' \
    | sudo tee -a "$CONFIG" >/dev/null
  echo "· overlay added to $CONFIG — REBOOT before the card appears"
fi

# ---- the fold ----
mkdir -p "$CONF_DIR"
install -m 644 "$HERE/51-mono-speaker.conf" "$CONF_DIR/"
systemctl --user restart pipewire pipewire-pulse wireplumber
sleep 3
pactl set-default-sink mono_speaker    # wireplumber remembers this across reboots

echo "· default sink: $(pactl get-default-sink)"
aplay -l | grep -q sndrpihifiberry \
  && echo "· card present: sndrpihifiberry" \
  || echo "· NO sound card yet — reboot, then run this again"
