"""Image quality checks: is the picture good enough to trust the eye measurements?"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import Config
from .features import LEFT_EYE, RIGHT_EYE

NAN = float("nan")


@dataclass
class FrameStats:
    brightness: float        # mean gray of the face box (whole frame if no face), 0..255
    sharpness: float         # variance of the Laplacian on the face crop (NaN if no face)
    glare: float             # fraction of saturated pixels around the eyes, 0..1 (NaN if no face)
    face_width_ratio: float  # face width / frame width (NaN if no face)


def scene_brightness(frame_bgr: np.ndarray) -> float:
    """Cheap whole-frame brightness (downscaled)."""
    gray = cv2.cvtColor(cv2.resize(frame_bgr, (80, 60)), cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def brighten(frame_bgr: np.ndarray, brightness: float, target: float = 100.0) -> np.ndarray:
    """Gamma-lift a dark frame toward ``target`` mean brightness. Helps face detection in dim light."""
    b = max(brightness, 1.0)
    gamma = float(np.clip(np.log(target / 255.0) / np.log(b / 255.0), 0.35, 1.0))
    if gamma >= 0.999:
        return frame_bgr
    lut = (np.linspace(0, 1, 256) ** gamma * 255).astype(np.uint8)
    return cv2.LUT(frame_bgr, lut)


def _box(points: np.ndarray, w: int, h: int, pad: float = 0.0):
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    return (int(max(0, x0 - px)), int(max(0, y0 - py)),
            int(min(w, x1 + px)), int(min(h, y1 + py)))


def measure(frame_bgr: np.ndarray, landmarks: Optional[np.ndarray]) -> FrameStats:
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if landmarks is None:
        return FrameStats(float(gray.mean()), NAN, NAN, NAN)

    x0, y0, x1, y1 = _box(landmarks[:, :2], w, h)
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0 or min(crop.shape) < 8:
        return FrameStats(float(gray.mean()), NAN, NAN, NAN)

    brightness = float(crop.mean())
    sharpness = float(cv2.Laplacian(cv2.resize(crop, (128, 128)), cv2.CV_64F).var())

    sat, total = 0, 0
    for idx in (RIGHT_EYE, LEFT_EYE):
        ex0, ey0, ex1, ey1 = _box(landmarks[list(idx), :2], w, h, pad=0.25)
        eye = gray[ey0:ey1, ex0:ex1]
        sat += int((eye >= 245).sum())
        total += eye.size
    glare = sat / total if total else NAN
    return FrameStats(brightness, sharpness, glare, (x1 - x0) / float(w))


def stats_unreliable(stats: Optional[FrameStats], config: Config) -> bool:
    """Instant check (no debounce): True when eye measurements can't be trusted."""
    if stats is None or np.isnan(stats.sharpness):
        return False
    return (stats.glare > config.glare_fraction
            or stats.sharpness < config.blur_sharpness_min
            or stats.face_width_ratio < config.face_min_width_ratio)
