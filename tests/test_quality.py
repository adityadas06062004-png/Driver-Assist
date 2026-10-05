import math

import cv2
import numpy as np
import pytest

from drowsiness import Config, FrameStats
from drowsiness import features as F
from drowsiness import quality as Q


def box_landmarks():
    """478 landmarks inside a 240x240 face box; eye landmarks around (300,200) and (380,200)."""
    rng = np.random.default_rng(0)
    pts = np.zeros((478, 3))
    pts[:, 0] = rng.uniform(200, 440, 478)
    pts[:, 1] = rng.uniform(120, 360, 478)
    for idx, cx in ((F.RIGHT_EYE, 300), (F.LEFT_EYE, 380)):
        for k, i in enumerate(idx):
            pts[i, :2] = (cx + (k - 2.5) * 6, 200 + (k % 2) * 6)
    return pts


def noise_frame(mean=120, std=30, seed=1):
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(mean, std, (480, 640, 3)), 0, 255).astype(np.uint8)


def test_no_face_gives_scene_brightness_only():
    s = Q.measure(noise_frame(mean=40), None)
    assert s.brightness == pytest.approx(40, abs=3)
    assert math.isnan(s.sharpness) and math.isnan(s.glare) and math.isnan(s.face_width_ratio)


def test_sharp_vs_blurred_sharpness():
    sharp = noise_frame()
    blurred = cv2.GaussianBlur(sharp, (0, 0), 6)
    lm = box_landmarks()
    assert Q.measure(sharp, lm).sharpness > 200
    assert Q.measure(blurred, lm).sharpness < Config().blur_sharpness_min


def test_glare_detected_around_eyes():
    frame = noise_frame()
    lm = box_landmarks()
    assert Q.measure(frame, lm).glare < 0.02
    for cx in (300, 380):
        cv2.circle(frame, (cx, 203), 22, (255, 255, 255), -1)
    assert Q.measure(frame, lm).glare > Config().glare_fraction


def test_face_width_ratio():
    assert Q.measure(noise_frame(), box_landmarks()).face_width_ratio == pytest.approx(240 / 640, abs=0.03)


def test_brighten_lifts_dark_frames_and_leaves_bright_ones():
    dark = (noise_frame(mean=100, std=20) * 0.2).astype(np.uint8)
    b = Q.scene_brightness(dark)
    lifted = Q.brighten(dark, b)
    assert Q.scene_brightness(lifted) > b + 30
    bright = noise_frame(mean=140)
    assert Q.brighten(bright, Q.scene_brightness(bright)) is bright


def test_stats_unreliable_rules():
    cfg = Config()
    good = FrameStats(120, 300, 0.0, 0.3)
    assert not Q.stats_unreliable(good, cfg)
    assert not Q.stats_unreliable(None, cfg)
    assert Q.stats_unreliable(FrameStats(120, 300, 0.5, 0.3), cfg)      # glare
    assert Q.stats_unreliable(FrameStats(120, 5, 0.0, 0.3), cfg)        # blur
    assert Q.stats_unreliable(FrameStats(120, 300, 0.0, 0.05), cfg)     # too far
    assert not Q.stats_unreliable(FrameStats(20, 300, 0.0, 0.3), cfg)   # dark alone does not gate
    assert not Q.stats_unreliable(FrameStats(50, float("nan"), float("nan"), float("nan")), cfg)
