"""Crisp on-screen text whose colour follows the picture behind it.

OpenCV's built-in fonts are thin strokes without proper anti-aliasing, which is why text drawn
with cv2.putText looks rough. This module draws with the bundled Inter font through Pillow instead.
For every piece of text it looks at the pixels underneath: over a dark area the text is light, over a
light area it is dark, and a soft halo of the opposite tone keeps it readable on busy backgrounds.
If Pillow or the font cannot be loaded, it falls back to OpenCV so the app still runs.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont, features
except ImportError:  # pragma: no cover - exercised through the fallback test
    Image = None

BGR = Tuple[int, int, int]
FONT_FILE = Path(__file__).with_name("assets") / "Inter.ttf"
_SYSTEM_FONTS = (
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)
MIN_CONTRAST = 4.5


@lru_cache(maxsize=64)
def _font(px: int, weight: int):
    """Inter at the given pixel size and weight, or None if no usable font is found."""
    if Image is None:
        return None
    try:
        font = ImageFont.truetype(str(FONT_FILE), px)
        try:
            font.set_variation_by_axes([14 if px < 20 else min(32, px), weight])
        except (OSError, AttributeError):
            pass
        return font
    except OSError:
        pass
    for path in _SYSTEM_FONTS:
        try:
            return ImageFont.truetype(path, px)
        except OSError:
            continue
    return None


@lru_cache(maxsize=1)
def _can_use_features() -> bool:
    return Image is not None and bool(features.check("raqm"))


def _kw() -> dict:
    # Tabular figures keep digits the same width, so live numbers do not jitter.
    return {"features": ["tnum"]} if _can_use_features() else {}


def _linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def luminance_bgr(bgr) -> float:
    """Relative luminance (0 black .. 1 white) of a BGR colour."""
    b, g, r = _linear(np.asarray(bgr, dtype=np.float64))
    return float(0.2126 * r + 0.7152 * g + 0.0722 * b)


def contrast_ratio(l1: float, l2: float) -> float:
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def region_luminance(img: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    """Average luminance of a rectangle of the picture."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    mean_bgr = img[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)
    return luminance_bgr(mean_bgr)


def _readable(color: BGR, bg_lum: float) -> BGR:
    """Keep the hue of a coloured label (an alert level, say) but make sure it can be read."""
    if contrast_ratio(luminance_bgr(color), bg_lum) >= MIN_CONTRAST:
        return color
    toward = (255, 255, 255) if bg_lum < 0.18 else (0, 0, 0)
    out = np.asarray(color, dtype=np.float64)
    for step in range(1, 11):
        out = np.asarray(color, dtype=np.float64) * (1 - step / 10) + np.asarray(toward) * (step / 10)
        if contrast_ratio(luminance_bgr(out), bg_lum) >= MIN_CONTRAST:
            break
    return tuple(int(v) for v in out)


def _auto_color(bg_lum: float) -> BGR:
    """Light text on a dark background, dark text on a light one (picked by contrast)."""
    white, black = contrast_ratio(1.0, bg_lum), contrast_ratio(0.0, bg_lum)
    return (250, 250, 250) if white >= black else (18, 18, 18)


def text_size(s: str, size: int, weight: int = 600) -> Tuple[int, int]:
    """Width and height in pixels."""
    font = _font(size, weight)
    if font is None:
        (w, h), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, size / 30.0, 2)
        return w, h
    left, top, right, bottom = font.getbbox(s, anchor="ls", **_kw())
    return right - left, bottom - top


@lru_cache(maxsize=512)
def _masks(s: str, size: int, weight: int):
    """Glyph and halo coverage masks for a string (cached, so repeated text costs almost nothing).
    Returns (glyph, halo, dx, dy, mid): ``dx, dy`` is where the baseline origin sits inside the masks
    and ``mid`` is the horizontal centre of the text relative to that origin."""
    font = _font(size, weight)
    left, top, right, bottom = font.getbbox(s, anchor="ls", **_kw())
    pad = max(4, size // 2)
    canvas = Image.new("L", (right - left + 2 * pad, bottom - top + 2 * pad), 0)
    dx, dy = pad - left, pad - top
    ImageDraw.Draw(canvas).text((dx, dy), s, font=font, fill=255, anchor="ls", **_kw())
    glyph = np.asarray(canvas, dtype=np.uint8)
    halo = cv2.GaussianBlur(glyph, (0, 0), max(1.2, size / 11))
    halo = np.clip(halo.astype(np.float32) * 1.6, 0, 255).astype(np.uint8)
    return glyph, halo, dx, dy, (left + right) / 2.0


def draw_text(img: np.ndarray, s: str, org: Tuple[int, int], size: int = 20,
              color: Optional[BGR] = None, weight: int = 600, anchor: str = "l") -> None:
    """Draw ``s`` onto ``img`` in place. ``org`` is the left (or centre, with anchor "m") end of
    the baseline. ``color`` None means: pick light or dark to suit the background."""
    if not s:
        return
    x, y = int(org[0]), int(org[1])
    if _font(size, weight) is None:
        _cv_fallback(img, s, (x, y), size, color, anchor)
        return

    glyph, halo, dx, dy, mid = _masks(s, size, weight)
    mh, mw = glyph.shape
    origin_x = int(round(x - mid)) if anchor == "m" else x
    ox, oy = origin_x - dx, y - dy                            # top-left of the masks in the picture

    h, w = img.shape[:2]
    x0, y0 = max(0, ox), max(0, oy)
    x1, y1 = min(w, ox + mw), min(h, oy + mh)
    if x1 <= x0 or y1 <= y0:
        return
    gm = glyph[y0 - oy:y1 - oy, x0 - ox:x1 - ox].astype(np.float32) / 255.0
    hm = halo[y0 - oy:y1 - oy, x0 - ox:x1 - ox].astype(np.float32) / 255.0 * 0.6

    crop = img[y0:y1, x0:x1]
    bg_lum = region_luminance(img, x0, y0, x1, y1)
    ink = _auto_color(bg_lum) if color is None else _readable(tuple(color), bg_lum)
    # The halo is the opposite tone of the text: a soft shadow under light text, a glow under dark text.
    halo_bgr = (255, 255, 255) if luminance_bgr(ink) < bg_lum else (0, 0, 0)

    out = crop.astype(np.float32)
    out = out * (1 - hm[..., None]) + np.asarray(halo_bgr, np.float32) * hm[..., None]
    out = out * (1 - gm[..., None]) + np.asarray(ink, np.float32) * gm[..., None]
    img[y0:y1, x0:x1] = out.astype(np.uint8)


def _cv_fallback(img, s, org, size, color, anchor) -> None:
    scale = size / 30.0
    (tw, _), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x = org[0] - tw // 2 if anchor == "m" else org[0]
    h, w = img.shape[:2]
    bg = region_luminance(img, x, org[1] - size, x + tw, org[1] + 4)
    ink = _auto_color(bg) if color is None else _readable(tuple(color), bg)
    edge = (255, 255, 255) if luminance_bgr(ink) < bg else (0, 0, 0)
    cv2.putText(img, s, (x, org[1]), cv2.FONT_HERSHEY_SIMPLEX, scale, edge, 4, cv2.LINE_AA)
    cv2.putText(img, s, (x, org[1]), cv2.FONT_HERSHEY_SIMPLEX, scale, ink, 2, cv2.LINE_AA)
