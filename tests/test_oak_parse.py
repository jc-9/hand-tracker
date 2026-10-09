#!/usr/bin/env python3
"""Offline tests for oak_hands result parsing (no OAK device needed).

Run: ./venv/bin/python tests/test_oak_parse.py
"""
import marshal
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import oak_hands


def fake_result(n_hands=1):
    # Mimics the dict built by send_result_hand() in the device script.
    rrn = []
    sqn = []
    for h in range(n_hands):
        rrn += [[0.05 * ((i + h) % 10), 0.05 * ((i + 2 * h) % 10), 0.01 * i]
                for i in range(21)]
        sqn += [0.3 + 0.01 * h + 0.005 * (i % 7) for i in range(42)]
    return {
        "pd_inf": False, "nb_lm_inf": n_hands,
        "lm_score": [0.9] * n_hands,
        "handedness": [0.8 if h % 2 == 0 else 0.2 for h in range(n_hands)],
        "rotation": [0.1 * h for h in range(n_hands)],
        "rect_center_x": [0.4 + 0.1 * h for h in range(n_hands)],
        "rect_center_y": [0.5 for _ in range(n_hands)],
        "rect_size": [0.3 for _ in range(n_hands)],
        "rrn_lms": [rrn[i * 63:(i + 1) * 63] for i in range(n_hands)],
        "sqn_lms": [sqn[i * 42:(i + 1) * 42] for i in range(n_hands)],
    }


def test_marshal_roundtrip():
    res = fake_result(2)
    blob = marshal.dumps(res)
    hands = oak_hands.hands_from_result(blob, frame_size=1152, pad_h=200)
    assert len(hands) == 2, f"expected 2 hands, got {len(hands)}"
    assert hands[0].label == "right", hands[0].label
    assert hands[1].label == "left", hands[1].label
    assert hands[0].landmarks.shape == (21, 2)
    print("[ok] marshal roundtrip: 2 hands, labels right/left, 21x2 landmarks")


def test_det_tuple():
    res = fake_result(1)
    hands = oak_hands.hands_from_result(res, frame_size=1152, pad_h=200)
    d = oak_hands.hand_to_det(hands[0], 1152, 752)
    x1, y1, x2, y2, cx, cy, pts, handed = d
    assert len(pts) == 21 and all(len(p) == 2 for p in pts)
    assert handed == "right"
    assert 0 <= x1 <= x2 < 1152 and 0 <= y1 <= y2 < 752, (x1, y1, x2, y2)
    assert (cx, cy) == pts[9], "centroid must be landmark 9"
    print(f"[ok] det tuple: box=({x1},{y1})-({x2},{y2}) palm=({cx},{cy})")


def test_no_hands():
    res = {"pd_inf": True, "nb_lm_inf": 0}
    assert oak_hands.hands_from_result(res, 1152) == []
    print("[ok] empty result -> no hands")


def test_rect_points():
    pts = oak_hands.mpu.rotated_rect_to_points(100, 100, 40, 40, 0.0)
    assert len(pts) == 4 and all(len(p) == 2 for p in pts)
    print(f"[ok] rotated rect points: {pts}")


if __name__ == "__main__":
    test_marshal_roundtrip()
    test_det_tuple()
    test_no_hands()
    test_rect_points()
    print("ALL OAK PARSE TESTS PASSED")
