"""Synthetic face landmarks so tests need no camera and no MediaPipe."""

import cv2
import numpy as np

from drowsiness import features as F

W, H = 640, 480
FRAME = (W, H)


def project(pitch=0.0, yaw=0.0, roll=0.0):
    """Project the generic face model with a known pose (degrees, user convention)."""
    rx, ry, rz = np.radians([-pitch, -yaw, roll])
    rot_x = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    rot_y = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    rot_z = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    rot = rot_z @ rot_y @ rot_x
    rvec, _ = cv2.Rodrigues(rot)
    tvec = np.array([0.0, 0.0, 600.0])
    cam = np.array([[W, 0, W / 2], [0, W, H / 2], [0, 0, 1]], dtype=np.float64)
    img, _ = cv2.projectPoints(F._MODEL_POINTS, rvec, tvec, cam, np.zeros((4, 1)))
    return img.reshape(-1, 2)


def make_face(ear=0.30, mar=0.05, pitch=0.0, yaw=0.0, roll=0.0):
    """Landmark array (478, 3) whose EAR, MAR and head pose match the arguments."""
    pts = np.zeros((478, 3))
    pts[list(F.POSE_IDX), :2] = project(pitch, yaw, roll)

    eye_w = 30.0
    eye_h = ear * eye_w

    def place_eye(idx, corner_a, corner_b):
        # corner_a / corner_b are landmark indices already placed or to place.
        p1, p2, p3, p4, p5, p6 = idx
        c1, c4 = pts[p1, :2], pts[p4, :2]
        cx, cy = (c1[0] + c4[0]) / 2, (c1[1] + c4[1]) / 2
        pts[p2, :2] = (cx - eye_w / 6, cy - eye_h / 2)
        pts[p3, :2] = (cx + eye_w / 6, cy - eye_h / 2)
        pts[p6, :2] = (cx - eye_w / 6, cy + eye_h / 2)
        pts[p5, :2] = (cx + eye_w / 6, cy + eye_h / 2)

    # Right eye: corner 33 comes from the pose; put the other corner 30 px away.
    pts[133, :2] = pts[33, :2] + (eye_w, 0)
    place_eye(F.RIGHT_EYE, 33, 133)
    # Left eye: corner 263 comes from the pose; put the other corner 30 px away.
    pts[362, :2] = pts[263, :2] - (eye_w, 0)
    place_eye(F.LEFT_EYE, 362, 263)

    mouth_w = 60.0
    x0, y0 = 300.0, 350.0
    pts[78, :2] = (x0, y0)
    pts[308, :2] = (x0 + mouth_w, y0)
    opening = mar * mouth_w
    for (top, bottom), dx in zip(F.MOUTH_VERTICAL, (15.0, 30.0, 45.0)):
        pts[top, :2] = (x0 + dx, y0 - opening / 2)
        pts[bottom, :2] = (x0 + dx, y0 + opening / 2)
    return pts


FPS = 30


def run_face(feed, t, seconds, fps=FPS, **face):
    """Feed ``seconds`` of an unchanging synthetic face. ``feed(pts, t)`` -> result.
    Returns (new_t, list_of_results)."""
    pts = make_face(**face)
    out = []
    for _ in range(int(round(seconds * fps))):
        out.append(feed(pts, t))
        t += 1.0 / fps
    return t, out


def add_head_landmarks(pts):
    """Give a synthetic face the contour points the phone detector uses (ears, forehead).
    The chin (152) already exists because the head pose uses it, so it is left alone."""
    cx, ny = pts[1, 0], pts[1, 1]
    pts[234, :2], pts[454, :2], pts[10, :2] = (cx - 120.0, ny + 10.0), (cx + 120.0, ny + 10.0), (cx, ny - 200.0)
    return pts


def make_hand(cx, cy, size=40.0):
    """21 hand landmarks bunched around a palm centre."""
    hand = np.zeros((21, 3))
    rng = np.random.default_rng(1)
    hand[:, :2] = np.array([cx, cy]) + rng.uniform(-size / 2, size / 2, (21, 2))
    for i in (0, 5, 9, 13, 17):                      # palm points sit exactly on the centre
        hand[i, :2] = (cx, cy)
    return hand
