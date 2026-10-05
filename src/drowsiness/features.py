"""Per-frame geometric features from Face Mesh landmarks (pixel coordinates)."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

# MediaPipe Face Mesh indices. Eye order: [corner, top1, top2, other corner,
# bottom2, bottom1] -> the classic p1..p6 of the EAR formula.
RIGHT_EYE = (33, 160, 158, 133, 153, 144)   # subject's right eye
LEFT_EYE = (362, 385, 387, 263, 373, 380)   # subject's left eye

# Inner-lip vertical pairs plus mouth corners for MAR.
MOUTH_VERTICAL = ((81, 178), (13, 14), (311, 402))
MOUTH_CORNERS = (78, 308)

# solvePnP points: nose tip, chin, image-left eye corner, image-right eye
# corner, image-left mouth corner, image-right mouth corner.
POSE_IDX = (1, 152, 33, 263, 61, 291)

# Generic face model in mm. Frame: x right, y DOWN, z away from the camera,
# origin at the nose tip. Facing the camera straight on => rotation = identity.
_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),          # nose tip
        (0.0, 330.0, 65.0),       # chin
        (-225.0, -170.0, 135.0),  # image-left eye outer corner
        (225.0, -170.0, 135.0),   # image-right eye outer corner
        (-150.0, 150.0, 125.0),   # image-left mouth corner
        (150.0, 150.0, 125.0),    # image-right mouth corner
    ],
    dtype=np.float64,
)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(pts: np.ndarray, idx: Sequence[int]) -> float:
    """EAR = (|p2-p6| + |p3-p5|) / (2 |p1-p4|). About 0.3 open, near 0 closed."""
    p = pts[list(idx), :2]
    horizontal = max(_dist(p[0], p[3]), 1e-6)
    return (_dist(p[1], p[5]) + _dist(p[2], p[4])) / (2.0 * horizontal)


def mean_ear(pts: np.ndarray) -> float:
    return 0.5 * (eye_aspect_ratio(pts, RIGHT_EYE) + eye_aspect_ratio(pts, LEFT_EYE))


def mouth_aspect_ratio(pts: np.ndarray) -> float:
    """MAR = mean inner-lip opening / mouth width. Roughly 0 closed, >0.6 yawn."""
    p = pts[:, :2]
    vertical = sum(_dist(p[a], p[b]) for a, b in MOUTH_VERTICAL) / len(MOUTH_VERTICAL)
    width = max(_dist(p[MOUTH_CORNERS[0]], p[MOUTH_CORNERS[1]]), 1e-6)
    return vertical / width


def head_pose(pts: np.ndarray, frame_size: Tuple[int, int]) -> Tuple[float, float, float]:
    """Return (pitch, yaw, roll) in degrees.

    pitch > 0: head tilted DOWN (nod). yaw > 0: turned toward image right.
    roll > 0: tilted clockwise on screen. Returns NaNs if solvePnP fails.
    """
    import cv2

    w, h = frame_size
    image_points = pts[list(POSE_IDX), :2].astype(np.float64)
    focal = float(w)
    camera = np.array([[focal, 0, w / 2.0], [0, focal, h / 2.0], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(
        _MODEL_POINTS, image_points, camera, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return float("nan"), float("nan"), float("nan")
    rot, _ = cv2.Rodrigues(rvec)
    sy = float(np.hypot(rot[0, 0], rot[1, 0]))
    rx = float(np.degrees(np.arctan2(rot[2, 1], rot[2, 2])))
    yaw = float(np.degrees(np.arctan2(-rot[2, 0], sy)))
    roll = float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])))
    # rx > 0 would mean chin moving away (looking up), so pitch = -rx.
    # Positive Ry brings the image-right side closer, i.e. face turned LEFT, so yaw = -ry.
    return -rx, -yaw, roll
