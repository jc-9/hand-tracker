#!/usr/bin/env python3
"""Hand tracker app with menu: camera setup vs hand tracking in separate windows.

  Menu (small app window):
    [S] Camera setup  - live view + focus zoom, pick camera/resolution.
                        Use once to adjust the mechanical lens ring.
    [T] Hand tracker  - detector boxes + 21 landmarks + motion trails.
    [Q] Quit.

  Camera Setup window: live view, big FOCUS sharpness score, magnified
  center inset + crosshair. Turn the physical lens ring to maximize the
  score. C = switch camera, R = cycle resolution, S = snapshot, ESC = menu.

  Hand Tracker window: MediaPipe (or skin fallback) boxes + trails of the
  last N seconds. H = help, S = snapshot, D = detector, +/- = trail,
  ESC = menu. TrailSecs slider sits on this window.

Detectors (--detector): auto | mediapipe | skin.
"""

import argparse
import collections
import os
import time
import urllib.request
from datetime import datetime

import cv2
import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "models", "hand_landmarker.task")

CAMERA_CANDIDATES = [2, 0]  # default HD USB first, then integrated
RESOLUTIONS = [(640, 480), (960, 540), (1280, 720)]
TRAIL_COLORS = [(0, 255, 0), (255, 200, 0), (0, 200, 255), (255, 0, 255)]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


class Settings:
    def __init__(self, args):
        self.cams = [args.camera] + [c for c in CAMERA_CANDIDATES if c != args.camera]
        self.cam_pos = 0
        self.res_pos = 2
        for i, (w, h) in enumerate(RESOLUTIONS):
            if w == args.width and h == args.height:
                self.res_pos = i
        self.detector = args.detector  # auto|mediapipe|skin (resolved on tracker entry)
        self.trail_secs = float(np.clip(args.trail, 1, 10))
        self.mp_backend = None

    @property
    def cam(self):
        return self.cams[self.cam_pos]

    @property
    def res(self):
        return RESOLUTIONS[self.res_pos]

    def summary(self):
        w, h = self.res
        return f"cam {self.cam} {w}x{h} | det {self.detector} | trail {self.trail_secs:.0f}s"


def ensure_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        return MODEL_PATH
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print(f"[info] downloading hand model (~8MB) to {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("[info] model downloaded.")
    return MODEL_PATH


def sharpness_score(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def open_camera(index, width, height):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def put_text(img, text, org, scale=0.6, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)


def draw_hand_skeleton(frame, pts, color):
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), 3, color, 1)


def skin_detect(frame, min_area=4000):
    """YCrCb skin segmentation -> list of (x1,y1,x2,y2,cx,cy,contour)."""
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = frame.shape[:2]
    frame_area = H * W
    boxes = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:4]:
        area = cv2.contourArea(c)
        if area < min_area or area > 0.5 * frame_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < 40 or h < 40:
            continue
        M = cv2.moments(c)
        cx, cy = (x + w // 2, y + h // 2) if M["m00"] == 0 else (
            int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        boxes.append((x, y, x + w, y + h, cx, cy, c))
    return boxes, mask


class MediapipeBackend:
    """Lazy MediaPipe Tasks wrapper. Raises on failure so caller can fall back."""

    def __init__(self, num_hands=4):
        import mediapipe as mp
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
            VisionTaskRunningMode,
        )
        self.mp = mp
        ensure_model()
        options = mp_vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.lm = mp_vision.HandLandmarker.create_from_options(options)

    def detect(self, frame, now):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        res = self.lm.detect_for_video(mp_image, int(now * 1000))
        H, W = frame.shape[:2]
        out = []  # (x1,y1,x2,y2,cx,cy,pts,handed)
        for slot, lms in enumerate(res.hand_landmarks or []):
            pts = [(int(p.x * W), int(p.y * H)) for p in lms]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            handed = ""
            try:
                if res.handedness and slot < len(res.handedness):
                    handed = res.handedness[slot][0].category_name
            except Exception:
                pass
            out.append((max(0, min(xs)), max(0, min(ys)),
                        min(W - 1, max(xs)), min(H - 1, max(ys)),
                        pts[9][0], pts[9][1], pts, handed))
        return out

    def close(self):
        try:
            self.lm.close()
        except Exception:
            pass


def ensure_tracker_backend(st, headless=False):
    """Resolve auto->mediapipe|skin once; reuse backend across tracker entries."""
    if st.detector == "auto":
        try:
            st.mp_backend = st.mp_backend or MediapipeBackend()
            st.detector = "mediapipe"
            print("[info] MediaPipe backend ready.")
        except Exception as e:
            print(f"[warn] MediaPipe unavailable ({e}); using skin detector.")
            st.detector = "skin"
    elif st.detector == "mediapipe" and st.mp_backend is None:
        try:
            st.mp_backend = MediapipeBackend()
            print("[info] MediaPipe backend ready.")
        except Exception as e:
            if headless:
                raise
            print(f"[warn] mediapipe init failed: {e}")
            return False
    return True


# ---------------------------------------------------------------- menu ---

def menu_loop(st):
    """Small menu window. Returns 'setup', 'tracker', or 'quit'."""
    cv2.namedWindow("Menu", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Menu", 560, 340)
    while True:
        img = np.zeros((340, 560, 3), dtype=np.uint8)
        put_text(img, "HAND TRACKER", (20, 40), 1.0, (0, 255, 0))
        put_text(img, "[S] Camera setup  - focus lens, pick camera", (20, 95))
        put_text(img, "[T] Hand tracker  - boxes + motion trails", (20, 130))
        put_text(img, "[C] camera  [R] resolution  [D] detector  [+/-] trail", (20, 175), 0.5)
        put_text(img, "[Q] Quit", (20, 210))
        put_text(img, st.summary(), (20, 265), 0.55, (0, 255, 255))
        put_text(img, "Setup once, then run the tracker.", (20, 300), 0.5, (180, 180, 180))
        cv2.imshow("Menu", img)
        key = cv2.waitKey(100) & 0xFF
        if key in (ord("s"), ord("S")):
            return "setup"
        if key in (ord("t"), ord("T")):
            return "tracker"
        if key in (27, ord("q"), ord("Q")):
            return "quit"
        if key in (ord("c"), ord("C")):
            st.cam_pos = (st.cam_pos + 1) % len(st.cams)
        elif key in (ord("r"), ord("R")):
            st.res_pos = (st.res_pos + 1) % len(RESOLUTIONS)
        elif key in (ord("d"), ord("D")):
            order = ["auto", "mediapipe", "skin"]
            st.detector = order[(order.index(st.detector) + 1) % len(order)]
        elif key in (ord("+"), ord("=")):
            st.trail_secs = min(10, st.trail_secs + 1)
        elif key in (ord("-"), ord("_")):
            st.trail_secs = max(1, st.trail_secs - 1)


# --------------------------------------------------------------- setup ---

def draw_setup_overlay(frame):
    """Focus score + big center zoom inset + crosshair. Returns center score."""
    H, W = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    s_all = sharpness_score(gray)
    ch, cw = H // 3, W // 3
    crop = gray[H // 2 - ch // 2:H // 2 + ch // 2,
                W // 2 - cw // 2:W // 2 + cw // 2]
    s_c = sharpness_score(crop)
    col = (0, 255, 0) if s_c > 80 else (0, 255, 255) if s_c > 30 else (0, 0, 255)
    put_text(frame, f"FOCUS {s_c:.0f} (full {s_all:.0f})", (10, 40), 1.0, col)
    put_text(frame, "Turn the lens ring to maximize. ESC = back to menu.",
             (10, 75), 0.55)
    # crosshair at frame center
    cx, cy = W // 2, H // 2
    cv2.drawMarker(frame, (cx, cy), col, cv2.MARKER_CROSS, 30, 2)
    # magnified inset, bottom-right
    inset = cv2.resize(crop, (cw * 2, ch * 2), interpolation=cv2.INTER_LINEAR)
    inset = cv2.cvtColor(inset, cv2.COLOR_GRAY2BGR)
    ih, iw = inset.shape[:2]
    frame[H - ih - 10:H - 10, W - iw - 10:W - 10] = inset
    cv2.rectangle(frame, (W - iw - 10, H - ih - 10), (W - 10, H - 10), col, 2)
    put_text(frame, "focus zoom", (W - iw - 10, H - ih - 16), 0.5, col)
    return s_c


def setup_loop(st, headless=False, test_frames=20):
    w, h = st.res
    cap = open_camera(st.cam, w, h)
    if cap is None:
        print(f"[error] cannot open camera {st.cam}.")
        return False
    print(f"[setup] camera {st.cam} @ {w}x{h} - ESC returns to menu")
    if not headless:
        cv2.namedWindow("Camera Setup", cv2.WINDOW_NORMAL)
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue
        n += 1
        s_c = draw_setup_overlay(frame)
        put_text(frame, f"cam {st.cam} {frame.shape[1]}x{frame.shape[0]} "
                        f"[C]amera [R]es [S]nap",
                 (10, frame.shape[0] - 12), 0.55)
        if headless:
            if n == 1 or n % 10 == 0:
                print(f"[selftest-setup] frame {n}: focus={s_c:.0f}")
            if n == test_frames:
                cv2.imwrite("verify_setup.jpg", frame)
                print(f"[selftest-setup] OK: {n} frames, saved verify_setup.jpg")
                break
            continue
        cv2.imshow("Camera Setup", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break
        elif key in (ord("c"), ord("C")):
            st.cam_pos = (st.cam_pos + 1) % len(st.cams)
            cap.release()
            w, h = st.res
            cap = open_camera(st.cam, w, h)
            print(f"[setup] camera -> {st.cam}")
        elif key in (ord("r"), ord("R")):
            st.res_pos = (st.res_pos + 1) % len(RESOLUTIONS)
            w, h = st.res
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            print(f"[setup] resolution -> {w}x{h}")
        elif key in (ord("s"), ord("S")):
            fn = f"snapshots/setup_{datetime.now():%Y%m%d_%H%M%S}_cam{st.cam}.jpg"
            cv2.imwrite(fn, frame)
            print(f"[info] saved {fn}")
    cap.release()
    if not headless:
        cv2.destroyWindow("Camera Setup")
    return True


# -------------------------------------------------------------- tracker ---

def tracker_loop(st, headless=False, test_frames=30):
    if not ensure_tracker_backend(st, headless=headless):
        return False
    w, h = st.res
    cap = open_camera(st.cam, w, h)
    if cap is None:
        print(f"[error] cannot open camera {st.cam}.")
        return False
    print(f"[tracker] camera {st.cam} @ {w}x{h} [{st.detector}] - ESC returns to menu")
    trails = collections.defaultdict(collections.deque)
    show_help = True
    if not headless:
        cv2.namedWindow("Hand Tracker", cv2.WINDOW_NORMAL)

        def on_trail(v):
            st.trail_secs = float(max(1, v))
        cv2.createTrackbar("TrailSecs", "Hand Tracker",
                           int(st.trail_secs), 10, on_trail)
    n_frames = 0
    hands_now = 0
    t0 = time.time()
    fps = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue
        n_frames += 1
        now = time.time()
        if n_frames % 10 == 0:
            fps = 10.0 / max(1e-6, now - t0)
            t0 = now
        H, W = frame.shape[:2]
        dets = []
        if st.detector == "mediapipe" and st.mp_backend is not None:
            try:
                dets = st.mp_backend.detect(frame, now)
            except Exception as e:
                print(f"[warn] mediapipe detect failed ({e}); falling back to skin.")
                st.detector = "skin"
        if st.detector == "skin":
            boxes, _ = skin_detect(frame)
            dets = [(x1, y1, x2, y2, cx, cy, None, "")
                    for x1, y1, x2, y2, cx, cy, _c in boxes]
        hands_now = len(dets)

        for slot, d in enumerate(dets):
            x1, y1, x2, y2, cx, cy, pts, handed = d
            color = TRAIL_COLORS[slot % len(TRAIL_COLORS)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"Hand {slot+1} {handed}".strip() + f" [{st.detector}]"
            cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            if pts is not None:
                draw_hand_skeleton(frame, pts, color)
            trails[slot].append((cx, cy, now))
            cv2.circle(frame, (cx, cy), 5, color, -1)

        cutoff = now - st.trail_secs
        for slot in list(trails.keys()):
            dq = trails[slot]
            while dq and dq[0][2] < cutoff:
                dq.popleft()
            if slot >= max(1, hands_now) and not dets:
                continue
            color = TRAIL_COLORS[slot % len(TRAIL_COLORS)]
            pts_t = list(dq)
            for i in range(1, len(pts_t)):
                age = now - pts_t[i][2]
                a = max(0.15, 1.0 - age / st.trail_secs)
                cv2.line(frame, pts_t[i - 1][:2], pts_t[i][:2],
                         color, max(1, int(4 * a)), cv2.LINE_AA)
            if pts_t:
                cv2.putText(frame, f"trail {len(pts_t)}pts/{st.trail_secs:.0f}s",
                            (pts_t[-1][0] + 10, pts_t[-1][1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        hud = (f"cam {st.cam} {W}x{H} fps {fps:.1f} "
               f"trail {st.trail_secs:.0f}s [{st.detector}]  ESC=menu")
        put_text(frame, hud, (10, H - 12), 0.55)
        if show_help and not headless:
            put_text(frame, "H help | S snapshot | D detector | +/- trail | ESC menu",
                     (10, 24), 0.55)

        if headless:
            if n_frames == 1 or n_frames % 10 == 0:
                print(f"[selftest] frame {n_frames}: hands={hands_now} "
                      f"det={st.detector} fps~{fps:.1f}")
            if n_frames >= test_frames:
                print(f"[selftest] OK: {n_frames} frames, last_hands={hands_now}, "
                      f"det={st.detector}, fps~{fps:.1f}")
                break
            continue

        cv2.imshow("Hand Tracker", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break
        elif key in (ord("h"), ord("H")):
            show_help = not show_help
        elif key in (ord("d"), ord("D")):
            if st.detector == "skin":
                try:
                    st.mp_backend = st.mp_backend or MediapipeBackend()
                    st.detector = "mediapipe"
                except Exception as e:
                    print(f"[warn] mediapipe still unavailable: {e}")
            else:
                st.detector = "skin"
            trails.clear()
            print(f"[info] detector -> {st.detector}")
        elif key in (ord("s"), ord("S")):
            fn = f"snapshots/snap_{datetime.now():%Y%m%d_%H%M%S}_cam{st.cam}.jpg"
            cv2.imwrite(fn, frame)
            print(f"[info] saved {fn}")
        elif key in (ord("+"), ord("=")):
            st.trail_secs = min(10, st.trail_secs + 1)
            try:
                cv2.setTrackbarPos("TrailSecs", "Hand Tracker", int(st.trail_secs))
            except Exception:
                pass
        elif key in (ord("-"), ord("_")):
            st.trail_secs = max(1, st.trail_secs - 1)
            try:
                cv2.setTrackbarPos("TrailSecs", "Hand Tracker", int(st.trail_secs))
            except Exception:
                pass
    cap.release()
    if not headless:
        cv2.destroyWindow("Hand Tracker")
    return True


# ----------------------------------------------------------------- main ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=2)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--trail", type=float, default=3.0)
    ap.add_argument("--detector", choices=["auto", "mediapipe", "skin"],
                    default="auto")
    ap.add_argument("--mode", choices=["menu", "setup", "tracker"],
                    default="menu", help="start directly in a mode (default: menu)")
    ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--test-frames", type=int, default=30)
    args = ap.parse_args()
    os.makedirs("snapshots", exist_ok=True)

    st = Settings(args)

    if args.no_gui:
        if args.mode == "setup":
            ok = setup_loop(st, headless=True, test_frames=args.test_frames)
        else:
            ok = tracker_loop(st, headless=True, test_frames=args.test_frames)
        if st.mp_backend is not None:
            st.mp_backend.close()
        raise SystemExit(0 if ok else 1)

    try:
        if args.mode == "setup":
            setup_loop(st)
        elif args.mode == "tracker":
            tracker_loop(st)
        else:
            while True:
                choice = menu_loop(st)
                if choice == "setup":
                    cv2.destroyWindow("Menu")
                    setup_loop(st)
                elif choice == "tracker":
                    cv2.destroyWindow("Menu")
                    tracker_loop(st)
                else:
                    break
    finally:
        if st.mp_backend is not None:
            st.mp_backend.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
