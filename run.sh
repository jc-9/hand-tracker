#!/usr/bin/env bash
# Run hand tracker (HD USB camera default). Pass extra args to app.py.
# Single-instance: menu relaunches just notify instead of grabbing the camera twice.
set -euo pipefail
cd "$(dirname "$0")"
exec 9>/tmp/hand_tracker.lock
if ! flock -n 9; then
  notify-send "Hand Tracker" "Already running - see the Menu window." 2>/dev/null || true
  exit 0
fi
exec ./venv/bin/python app.py --camera 2 --width 1280 --height 720 "$@"
