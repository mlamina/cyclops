#!/bin/bash
# hq#602: speech for the scripted room, 24 kHz mono, with the volume printed so a silent `say` shows.
# Needs macOS `say` (with -v: without it, it can write a silent file) and ffmpeg.
d="$(cd "$(dirname "$0")" && pwd)/fixtures_602"
mkdir -p "$d"
cp "$(dirname "$0")"/fixtures_572/ask.wav "$(dirname "$0")"/fixtures_572/bye1.wav "$d"/
make() {
  say -v Samantha -o "$d/$1.aiff" "$2"
  ffmpeg -y -hide_banner -loglevel error -i "$d/$1.aiff" -ar 24000 -ac 1 -c:a pcm_s16le "$d/$1.wav"
  rm "$d/$1.aiff"
  printf '%s: ' "$1"
  ffmpeg -hide_banner -i "$d/$1.wav" -af volumedetect -f null - 2>&1 | grep -o 'max_volume.*'
}
make record "Hey, record this please."
make record2 "Start recording."
make stop "Okay, stop recording now."
make status "Are you recording right now?"
make mention"I made a recording of the engine last week. How long does an oil change usually take?"
