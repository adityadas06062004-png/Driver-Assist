"""Core engine: landmarks in, drowsiness metrics out. No UI, no camera."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from .config import Config
from .features import head_pose, mean_ear, mouth_aspect_ratio
from .metrics import BlinkCounter, PerclosTracker, SustainedEventDetector

if TYPE_CHECKING:  # avoid a circular import at runtime
    from .calibration import Calibration

NAN = float("nan")


@dataclass
class FrameResult:
    t: float
    face_found: bool
    ear: float = NAN
    mar: float = NAN
    pitch: float = NAN        # degrees, > 0 = head down (after baseline offset)
    yaw: float = NAN
    roll: float = NAN
    eyes_closed: bool = False
    eye_closed_s: float = 0.0   # how long the eyes have been closed right now
    perclos: float = 0.0        # 0..1 over the rolling window
    window_span_s: float = 0.0  # seconds of data behind the PERCLOS value
    blink_rate: float = 0.0     # blinks per minute (0 until enough data)
    blinked: bool = False       # a blink finished on this frame
    yawning: bool = False       # mouth open long enough to count as a yawn
    yawn_count: int = 0         # yawns in the yawn window
    nodding: bool = False       # head down long enough to count as a nod
    nod_count: int = 0
    no_face_s: float = 0.0      # how long the face has been missing
    trusted: bool = True        # False: picture too poor, numbers shown but not counted


class DrowsinessEngine:
    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()
        self.reset_calibration()

    # ---- calibration hooks -------------------------------------------------
    def reset_calibration(self) -> None:
        """Back to generic thresholds and a clean slate."""
        c = self.config
        self.ear_threshold = c.ear_closed_threshold
        self.mar_threshold = c.mar_yawn_threshold
        self.pitch_threshold = c.pitch_nod_threshold_deg
        self.pitch_offset = 0.0
        self.reset_windows()

    def apply_calibration(self, cal: "Calibration") -> bool:
        """Adopt a driver's calibration. Returns False (and changes nothing) if cal.ok is False."""
        if not cal.ok:
            return False
        self.ear_threshold = cal.ear_threshold
        self.mar_threshold = cal.mar_threshold
        self.pitch_offset = cal.pitch_baseline
        self.reset_windows()
        return True

    def reset_windows(self) -> None:
        """Clear all rolling windows and in-progress episodes."""
        c = self.config
        self._perclos = PerclosTracker(c.perclos_window_s, c.perclos_max_gap_s)
        self._blinks = BlinkCounter(c.min_blink_s, c.max_blink_s, c.blink_window_s,
                                    c.blink_rate_min_span_s)
        self._yawns = SustainedEventDetector(c.yawn_min_s, c.yawn_window_s)
        self._nods = SustainedEventDetector(c.nod_min_s, c.nod_window_s)
        self._no_face_since: Optional[float] = None

    # ---- per frame ---------------------------------------------------------
    def update(self, landmarks: Optional[np.ndarray], t: float,
               frame_size: Tuple[int, int], trusted: bool = True) -> FrameResult:
        """Process one frame. ``landmarks`` is None when no face was found.

        ``trusted=False`` (glare, blur, face turned away...) still computes the numbers for
        display but keeps them out of PERCLOS, blinks, yawns and nods, and ends any open episode.
        """
        if landmarks is None:
            return self._on_no_face(t)
        self._no_face_since = None

        ear = mean_ear(landmarks)
        mar = mouth_aspect_ratio(landmarks)
        pitch, yaw, roll = head_pose(landmarks, frame_size)
        pitch_adj = pitch - self.pitch_offset

        # A strongly turned head foreshortens the eyes and fakes "closed" readings.
        if not math.isnan(yaw) and abs(yaw) > self.config.looking_away_yaw_deg:
            trusted = False

        if not trusted:
            self._blinks.interrupt()
            self._yawns.interrupt()
            self._nods.interrupt()
            return FrameResult(
                t=t, face_found=True, ear=ear, mar=mar, pitch=pitch_adj, yaw=yaw, roll=roll,
                perclos=self._perclos.value, window_span_s=self._perclos.span_s,
                blink_rate=self._blinks.rate_per_min(t),
                yawn_count=self._yawns.count(t), nod_count=self._nods.count(t),
                trusted=False,
            )

        closed = ear < self.ear_threshold
        mouth_open = mar > self.mar_threshold
        head_down = (not math.isnan(pitch_adj)) and pitch_adj > self.pitch_threshold

        perclos = self._perclos.update(t, closed)
        blinked = self._blinks.update(t, closed)
        self._yawns.update(t, mouth_open)
        self._nods.update(t, head_down)

        return FrameResult(
            t=t, face_found=True, ear=ear, mar=mar, pitch=pitch_adj, yaw=yaw, roll=roll,
            eyes_closed=closed,
            eye_closed_s=self._blinks.closure_duration_s(t),
            perclos=perclos,
            window_span_s=self._perclos.span_s,
            blink_rate=self._blinks.rate_per_min(t),
            blinked=blinked,
            yawning=self._yawns.is_sustained,
            yawn_count=self._yawns.count(t),
            nodding=self._nods.is_sustained,
            nod_count=self._nods.count(t),
        )

    def _on_no_face(self, t: float) -> FrameResult:
        c = self.config
        if self._no_face_since is None:
            self._no_face_since = t
        gone = t - self._no_face_since
        if gone >= c.face_lost_reset_s:
            # Don't let a half-finished closure/yawn/nod carry over a gap.
            self._blinks.interrupt()
            self._yawns.interrupt()
            self._nods.interrupt()
        return FrameResult(
            t=t, face_found=False, perclos=self._perclos.value,
            window_span_s=self._perclos.span_s,
            blink_rate=self._blinks.rate_per_min(t),
            yawn_count=self._yawns.count(t), nod_count=self._nods.count(t),
            no_face_s=gone,
        )
