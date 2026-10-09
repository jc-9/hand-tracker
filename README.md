# Hand Tracker

Live hand tracking with motion trails, plus a one-time camera setup view.
Sources: HD USB cam, integrated cam, or Luxonis OAK-D (on-device inference).

## Run

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
./run.sh            # opens the Menu: S = setup, T = tracker, Q = quit
```

Direct modes: `venv/bin/python app.py --mode {menu,setup,tracker} --camera {2,0,oak}`
Headless self-test: `venv/bin/python app.py --no-gui --mode tracker --test-frames 30`

## How it works

- **Menu** (`Menu` window): `[S]` Camera setup, `[T]` Hand tracker, `[Q]` Quit.
  `C` cycles source (USB 2 / integrated 0 / OAK-D); picks carry across modes.
- **Camera Setup**: live view, FOCUS sharpness score, magnified center inset,
  crosshair. `C` source, `R` resolution, `S` snapshot, `ESC` menu.
- **Hand Tracker**: hand boxes + 21 landmarks, fading motion trails of the
  last N seconds (`TrailSecs` slider, default 3s). `H` help, `S` snapshot,
  `D` detector (USB only), `ESC` menu.
- **OAK-D** (`--camera oak` or `C`, the default): palm-detection +
  hand-landmark NNs run on-device (Myriad X, depthai v2 edge pipeline,
  duo mode — up to 2 hands tracked simultaneously); the host only
  renders and trails. Needs the udev rule for `03e7` (see Notes).

## Notes

- `requirements.txt` pins `mediapipe==0.10.35`: the 1.x native lib is
  SIGKilled on load on some machines (see PROJECT.md). Do not upgrade
  without re-running the headless self-test.
- `depthai==2.33.0.0` (v2 classic API, not v3).
- The host-side hand-landmark model (~8MB) auto-downloads to `models/`.
- OAK-D blobs in `oak_models/` + `oak_template_script_duo.py` + pipeline
  logic in `oak_hands.py` adapted from geaxgx/depthai_hand_tracker
  (MIT License, Copyright (c) 2021 geax). OAK-D udev rule:
  `SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"` in
  `/etc/udev/rules.d/80-movidius.rules`, then reload udev.
- `assets/make_icon.py` regenerates the app icon (hand in a detector box).
- Desktop entry (`~/.local/share/applications/hand-tracker.desktop`) is
  machine-local and intentionally not committed.
