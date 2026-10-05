"""On-screen display: metrics, risk bar, calibration progress, alert banner."""

from __future__ import annotations

import cv2

from .alerts import Level
from .monitor import MonitorState

_COLORS = {  # BGR
    Level.OK: (0, 200, 0),
    Level.CAUTION: (0, 215, 255),
    Level.WARNING: (0, 140, 255),
    Level.CRITICAL: (0, 0, 255),
}
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, s, org, scale=0.65, color=(255, 255, 255), thick=2):
    cv2.putText(img, s, org, _FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, _FONT, scale, color, thick, cv2.LINE_AA)


def _centered(img, s, y, scale, color):
    (tw, _), _ = cv2.getTextSize(s, _FONT, scale, 3)
    _text(img, s, ((img.shape[1] - tw) // 2, y), scale, color, 3)


def draw_hud(view, state: MonitorState, t: float, fps: float = 0.0) -> None:
    """Draw on ``view`` (already mirrored for display) in place."""
    h, w = view.shape[:2]
    res = state.result

    if state.phase == "calibrating":
        _centered(view, "CALIBRATING", 50, 1.2, (0, 215, 255))
        _centered(view, state.message, 90, 0.6, (255, 255, 255))
        cv2.rectangle(view, (40, h - 50), (w - 40, h - 30), (80, 80, 80), -1)
        cv2.rectangle(view, (40, h - 50), (40 + int((w - 80) * state.progress), h - 30),
                      (0, 215, 255), -1)
        return

    level = state.alert.level if state.alert else Level.OK
    color = _COLORS[level]
    risk = state.risk.score if state.risk else 0.0

    if res.face_found:
        lines = [f"EAR {res.ear:.2f}  MAR {res.mar:.2f}",
                 f"Pitch {res.pitch:+.0f}  PERCLOS {res.perclos * 100:.0f}%",
                 f"Blinks/min {res.blink_rate:.0f}  Yawns {res.yawn_count}  Nods {res.nod_count}"]
        if res.eyes_closed:
            lines.append(f"Eyes closed {res.eye_closed_s:.1f}s")
        if not res.trusted:
            lines.append("Measurements paused (poor picture)")
    else:
        lines = ["No face detected"]
    for i, s in enumerate(lines):
        _text(view, s, (10, 26 + 24 * i), 0.55)
    if fps:
        _text(view, f"{fps:.0f} FPS", (w - 90, 26), 0.55)

    # Risk bar along the bottom.
    cv2.rectangle(view, (20, h - 44), (w - 20, h - 24), (60, 60, 60), -1)
    cv2.rectangle(view, (20, h - 44), (20 + int((w - 40) * risk / 100.0), h - 24), color, -1)
    _text(view, f"Risk {risk:.0f}  {level.name}", (20, h - 52), 0.6, color)

    if state.health and state.health.message:
        _centered(view, state.health.message, h - 80, 0.6, (0, 215, 255))

    if level >= Level.WARNING:
        cv2.rectangle(view, (0, 0), (w - 1, h - 1), color, 14)
    if level == Level.CRITICAL and int(t * 2) % 2 == 0:
        _centered(view, "WAKE UP!", h // 2, 2.0, color)
    elif level in (Level.WARNING, Level.CAUTION):
        _centered(view, state.alert.message, h // 2, 0.9, color)
