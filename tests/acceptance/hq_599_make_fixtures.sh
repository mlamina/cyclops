#!/bin/bash
# hq#599: speech for the scripted room, 24 kHz mono, with the volume printed so a silent `say` shows.
# Needs macOS `say` (with -v: without it, it can write a silent file) and ffmpeg.
d="$(cd "$(dirname "$0")" && pwd)/fixtures_599"
mkdir -p "$d"
cp "$(dirname "$0")"/fixtures_572/ask.wav "$d"/
make() {
  say -v Samantha -o "$d/$1.aiff" "$2"
  ffmpeg -y -hide_banner -loglevel error -i "$d/$1.aiff" -ar 24000 -ac 1 -c:a pcm_s16le "$d/$1.wav"
  rm "$d/$1.aiff"
  printf '%s: ' "$1"
  ffmpeg -hide_banner -i "$d/$1.wav" -af volumedetect -f null - 2>&1 | grep -o 'max_volume.*'
}
make job "Please find the manual for the Raspberry Pi Pico 2 and download it."
make job2 "Also find the manual for the Raspberry Pi Zero 2 W and download it."
make nothing "Find the manual for the Zorblax Q9 quantum toaster and download it."
make question "According to my manuals, how many GPIO pins does the Pico 2 have?"
