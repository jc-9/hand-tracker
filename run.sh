#!/usr/bin/env bash
# Run hand tracker (OAK-D default). Pass extra args to app.py.
# Single-instance: relaunch focuses the open window + notifies
# instead of grabbing the camera twice.
set -euo pipefail
cd "$(dirname "$0")"
exec 9>/tmp/hand_tracker.lock
if ! flock -n 9; then
  hyprctl eval 'for _,w in ipairs(hl.get_windows()) do if w.title and w.title:find("^Hand") then hl.dispatch(hl.dsp.focus({ window = w })) break end end' 2>/dev/null || true
  notify-send "Hand Tracker" "Already running - focused the open window." 2>/dev/null || true
  exit 0
fi
exec ./venv/bin/python app.py --camera oak --width 1280 --height 720 "$@"
