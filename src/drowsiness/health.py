"""Signal health: can we see the driver well enough to trust the numbers?

Bad data should not cause false alarms, and a driver who vanishes from view
(for example a head dropped below the camera) should not go unnoticed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Set, Tuple

from .alerts import Level
from .config import Config
from .quality import FrameStats


class Issue(str, Enum):
    CAMERA_LOST = "camera_lost"
    NO_FACE = "no_face"
    LOW_LIGHT = "low_light"
    GLARE = "glare"
    BLURRY = "blurry"
    FACE_TOO_SMALL = "face_too_small"
    LOOKING_AWAY = "looking_away"
    LOW_FPS = "low_fps"


# Shown in this order; the first active issue supplies the on-screen hint.
HINTS: Tuple[Tuple[Issue, str], ...] = (
    (Issue.CAMERA_LOST, "Camera lost: reconnecting"),
    (Issue.NO_FACE, "Face not visible: look at the camera"),
    (Issue.GLARE, "Glare on eyes/glasses: eyes can't be measured"),
    (Issue.BLURRY, "Image blurry: clean the lens or add light"),
    (Issue.FACE_TOO_SMALL, "Face too far: move closer to the camera"),
    (Issue.LOOKING_AWAY, "Looking away: eyes not measured"),
    (Issue.LOW_LIGHT, "Low light: accuracy reduced, auto-brightening on"),
    (Issue.LOW_FPS, "Low frame rate: detection may be slower"),
)

# While any of these is active, eye/mouth/head numbers are not fed into the metrics.
UNRELIABLE = {Issue.CAMERA_LOST, Issue.GLARE, Issue.BLURRY, Issue.FACE_TOO_SMALL, Issue.LOOKING_AWAY}


@dataclass
class Health:
    issues: Set[Issue]
    reliable: bool            # False: measurements should be ignored
    level: Level              # OK / WARNING / CRITICAL raised by health alone
    message: str = ""


class HealthMonitor:
    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()
        self.reset()

    def reset(self) -> None:
        self._on_since: Dict[Issue, float] = {}
        self._off_since: Dict[Issue, float] = {}
        self._active: Set[Issue] = set()

    def _persist_s(self, issue: Issue) -> float:
        c = self.config
        if issue in (Issue.NO_FACE, Issue.CAMERA_LOST):
            return 0.5
        if issue == Issue.LOOKING_AWAY:
            return c.looking_away_s
        return c.issue_persist_s

    def update(self, t: float, *, face_present: bool, camera_ok: bool = True,
               yaw: float = float("nan"), stats: Optional[FrameStats] = None,
               fps: Optional[float] = None) -> Health:
        c = self.config
        raw = {i: False for i in Issue}
        raw[Issue.CAMERA_LOST] = not camera_ok
        raw[Issue.NO_FACE] = camera_ok and not face_present
        if stats is not None:
            raw[Issue.LOW_LIGHT] = stats.brightness < c.low_light_brightness
            if face_present and not math.isnan(stats.sharpness):
                raw[Issue.GLARE] = stats.glare > c.glare_fraction
                raw[Issue.BLURRY] = stats.sharpness < c.blur_sharpness_min
                raw[Issue.FACE_TOO_SMALL] = stats.face_width_ratio < c.face_min_width_ratio
        raw[Issue.LOOKING_AWAY] = (face_present and not math.isnan(yaw)
                                   and abs(yaw) > c.looking_away_yaw_deg)
        raw[Issue.LOW_FPS] = fps is not None and 0 < fps < c.min_fps

        for issue, flag in raw.items():
            if flag:
                self._off_since.pop(issue, None)
                first = self._on_since.setdefault(issue, t)
                if t - first >= self._persist_s(issue):
                    self._active.add(issue)
            else:
                self._on_since.pop(issue, None)
                if issue in self._active:
                    off = self._off_since.setdefault(issue, t)
                    if t - off >= c.issue_clear_s:
                        self._active.discard(issue)
                        self._off_since.pop(issue, None)

        level = Level.OK
        if Issue.NO_FACE in self._on_since:
            lost = t - self._on_since[Issue.NO_FACE]
            if lost >= c.no_face_critical_s:
                level = Level.CRITICAL
            elif lost >= c.no_face_warn_s:
                level = Level.WARNING
        if Issue.CAMERA_LOST in self._on_since and \
                t - self._on_since[Issue.CAMERA_LOST] >= c.camera_lost_critical_s:
            level = Level.CRITICAL

        message = next((text for issue, text in HINTS if issue in self._active), "")
        return Health(set(self._active), not (self._active & UNRELIABLE), level, message)
