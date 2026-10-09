"""Trimmed helpers vendored from geaxgx/depthai_hand_tracker (MIT License,
Copyright (c) 2021 geax) - mediapipe_utils.py: HandRegion container,
rotated_rect_to_points(), find_isp_scale_params(). Only what the duo edge
pipeline needs; gesture/depth/body helpers omitted."""

import numpy as np
from math import sin, cos, gcd


class HandRegion:
    """One tracked hand. landmarks: 21x2 int pixels in the source frame."""

    def __init__(self, pd_score=None, pd_box=None, pd_kps=None):
        self.pd_score = pd_score
        self.pd_box = pd_box
        self.pd_kps = pd_kps


def rotated_rect_to_points(cx, cy, w, h, rotation):
    b = cos(rotation) * 0.5
    a = sin(rotation) * 0.5
    points = []
    p0x = cx - a * h - b * w
    p0y = cy + b * h - a * w
    p1x = cx + a * h - b * w
    p1y = cy - b * h - a * w
    p2x = int(2 * cx - p0x)
    p2y = int(2 * cy - p0y)
    p3x = int(2 * cx - p1x)
    p3y = int(2 * cy - p1y)
    p0x, p0y, p1x, p1y = int(p0x), int(p0y), int(p1x), int(p1y)
    return [[p0x, p0y], [p1x, p1y], [p2x, p2y], [p3x, p3y]]


def find_isp_scale_params(size, resolution, is_height=True):
    """
    Find closest valid ISP size near 'size' + setIspScale() params.
    Works around a depthai bug where ImageManip scrambles invalid sizes.
    Returns: valid size, (numerator, denominator)
    """
    if size < 288:
        size = 288
    width, height = resolution
    if is_height:
        reference = height
        other = width
    else:
        reference = width
        other = height
    size_candidates = {}
    for s in range(288, reference, 16):
        f = gcd(reference, s)
        n = s // f
        d = reference // f
        if n <= 16 and d <= 63 and int(round(other * n / d) % 2 == 0):
            size_candidates[s] = (n, d)
    min_dist = -1
    for s in size_candidates:
        dist = abs(size - s)
        if min_dist == -1:
            min_dist = dist
            candidate = s
        else:
            if dist > min_dist:
                break
            candidate = s
            min_dist = dist
    return candidate, size_candidates[candidate]
