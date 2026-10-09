# Hand Tracker

Live hand tracking with motion trails, plus a one-time camera setup view for
adjusting a mechanical lens. Built with OpenCV + MediaPipe Tasks.

## Run

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
./run.sh            # opens the Menu: S = setup, T = tracker, Q = quit
```

Direct modes: `venv/bin/python app.py --mode {menu,setup,tracker}`
Headless self-test: `venv/bin/python app.py --no-gui --mode tracker --test-frames 30`

## How it works

- **Menu** (`Menu` window): `[S]` Camera setup, `[T]` Hand tracker, `[Q]` Quit.
  Camera / resolution / detector / trail picks carry over between modes.
- **Camera Setup**: live view, FOCUS sharpness score, magnified center inset,
  crosshair. Turn the physical lens ring to maximize the score.
  `C` camera, `R` resolution, `S` snapshot, `ESC` menu.
- **Hand Tracker**: MediaPipe hand boxes + 21 landmarks (auto-falls back to a
  skin/contour detector), fading motion trails of the last N seconds
  (`TrailSecs` slider, default 3s). `H` help, `S` snapshot, `D` detector,
  `ESC` menu.

## Notes

- `requirements.txt` pins `mediapipe==0.10.35`: the 1.x native lib is
  SIGKilled on load on some machines (see PROJECT.md). Do not upgrade
  without re-running the headless self-test.
- The hand-landmark model (~8MB) auto-downloads to `models/` on first run.
- `assets/make_icon.py` regenerates the app icon (hand in a detector box).
- Desktop entry (`~/.local/share/applications/hand-tracker.desktop`) is
  machine-local and intentionally not committed.
