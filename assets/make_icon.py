#!/usr/bin/env python3
"""Generate the Hand Tracker icon: white hand pictogram inside a green
detector bounding box, on a dark rounded square.

Outputs (into assets/ next to this script):
  hand-tracker.svg ......... master (64x64 viewBox)
  hand-tracker-128.png, -64.png, -48.png ... raster fallbacks
Regenerate: ./venv/bin/python assets/make_icon.py
Install:    copies into ~/.local/share/icons/hicolor/... (see below)
"""
import os

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))

BG = (30, 34, 46, 255)
GREEN = (46, 255, 140, 255)
HAND = (238, 240, 244, 255)

# Hand strokes as (x1, y1, x2, y2, width) on a 256 canvas.
STROKES = [
    (128, 192, 128, 124, 26),   # palm
    (102, 124, 90, 56, 17),     # index
    (119, 120, 116, 44, 17),    # middle
    (137, 120, 140, 44, 17),    # ring
    (154, 124, 166, 56, 17),    # pinky
    (128, 162, 92, 138, 18),    # thumb
]
BOX = (46, 34, 210, 222)


def draw_png(size=256):
    s = size / 256.0

    def sc(v):
        return int(round(v * s))

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([sc(8), sc(8), sc(248), sc(248)],
                        radius=sc(52), fill=BG)
    d.rectangle([sc(BOX[0]), sc(BOX[1]), sc(BOX[2]), sc(BOX[3])],
                outline=GREEN, width=max(1, sc(7)))
    # corner brackets: thicker L accents
    arm, cw = sc(42), max(2, sc(13))
    x1, y1, x2, y2 = (sc(v) for v in BOX)
    for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                           (x1, y2, 1, -1), (x2, y2, -1, -1)):
        d.line([cx, cy, cx + dx * arm, cy], fill=GREEN, width=cw)
        d.line([cx, cy, cx, cy + dy * arm], fill=GREEN, width=cw)
    # hand: thick lines + round caps
    for x1, y1, x2, y2, w in STROKES:
        w = max(2, sc(w))
        d.line([sc(x1), sc(y1), sc(x2), sc(y2)], fill=HAND, width=w)
        for px, py in ((x1, y1), (x2, y2)):
            r = w / 2
            d.ellipse([sc(px) - r, sc(py) - r, sc(px) + r, sc(py) + r], fill=HAND)
    return img


SVG_TEMPLATE = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <rect x="2" y="2" width="60" height="60" rx="13" fill="#1e222e"/>
  <rect x="11.5" y="8.5" width="41" height="47" fill="none" stroke="#2eff8c" stroke-width="1.8"/>
  <g stroke="#2eff8c" stroke-width="3.2" stroke-linecap="butt">
    <path d="M11.5 19 V8.5 H22 M42 8.5 H52.5 V19 M52.5 45 V55.5 H42 M22 55.5 H11.5 V45" fill="none"/>
  </g>
  <g stroke="#eef0f4" stroke-width="4.4" stroke-linecap="round" fill="none">
    <path d="M32 48 V31" stroke-width="6.4"/>
    <path d="M25.5 31 L22.5 14 M29.75 30 L29 11 M34.25 30 L35 11 M38.5 31 L41.5 14"/>
    <path d="M32 40.5 L23 34.5"/>
  </g>
</svg>
"""


def main():
    os.makedirs(HERE, exist_ok=True)
    with open(os.path.join(HERE, "hand-tracker.svg"), "w") as f:
        f.write(SVG_TEMPLATE)
    for size in (128, 64, 48):
        img = draw_png(256).resize((size, size), Image.LANCZOS)
        img.save(os.path.join(HERE, f"hand-tracker-{size}.png"))
    draw_png(256).save(os.path.join(HERE, "hand-tracker-256.png"))
    print("[icon] wrote hand-tracker.svg + PNGs (48/64/128/256) to", HERE)


if __name__ == "__main__":
    main()
