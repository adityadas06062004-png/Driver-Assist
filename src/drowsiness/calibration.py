"""Per-driver calibration: learn what "alert" looks like for THIS face and camera.

Eye shape, glasses, camera angle and lighting all shift EAR, MAR and head pitch.
For the first ``calibration_s`` seconds the driver looks at the road, eyes open,
mouth closed. We then set the closed-eye, yawn and nod thresholds relative to
that baseline instead of using generic numbers.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Union

from .config import Config
from .engine import FrameResult


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass(frozen=True)
class Calibration:
    ok: bool
    reason: str                 # why it failed, or "ok"
    ear_baseline: float
    ear_threshold: float        # eyes counted closed below this
    mar_baseline: float
    mar_threshold: float        # mouth counted open (yawn) above this
    pitch_baseline: float       # resting head pitch; subtracted from live pitch
    blink_rate: Optional[float]  # blinks/min while alert, None if too few blinks seen
    n_frames: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Calibration":
        return cls(**data)

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Calibration":
        return cls.from_dict(json.loads(Path(path).expanduser().read_text()))


class Calibrator:
    """Feed it FrameResults (from an engine with default thresholds) until ``done``."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()
        self._t0: Optional[float] = None
        self._last_t = 0.0
        self._face_now = False
        self._frames = 0
        self._frames_with_face = 0
        self._untrusted = 0
        self._ears: List[float] = []
        self._mars: List[float] = []
        self._pitches: List[float] = []
        self._blinks = 0

    def update(self, res: FrameResult) -> None:
        c = self.config
        if self._t0 is None:
            self._t0 = res.t
        self._last_t = res.t
        self._face_now = res.face_found
        if res.t - self._t0 < c.calibration_warmup_s:
            return
        self._frames += 1
        if not res.face_found:
            return
        if not res.trusted:
            self._untrusted += 1
            return
        self._frames_with_face += 1
        self._ears.append(res.ear)
        self._mars.append(res.mar)
        if not math.isnan(res.pitch):
            self._pitches.append(res.pitch)
        if res.blinked:
            self._blinks += 1

    @property
    def progress(self) -> float:
        if self._t0 is None:
            return 0.0
        return _clamp((self._last_t - self._t0) / self.config.calibration_s, 0.0, 1.0)

    @property
    def done(self) -> bool:
        return self.progress >= 1.0

    @property
    def message(self) -> str:
        left = self.config.calibration_s * (1.0 - self.progress)
        if self._t0 is not None and not self._face_now:
            return "Face not found: look at the camera"
        return f"Calibrating {left:.0f}s: look at the road, eyes open, mouth closed"

    def finish(self) -> Calibration:
        c = self.config

        def fail(reason: str) -> Calibration:
            return Calibration(False, reason, 0.0, c.ear_closed_threshold, 0.0,
                               c.mar_yawn_threshold, 0.0, None, len(self._ears))

        if self._frames == 0 or self._frames_with_face / self._frames < c.calibration_min_face_ratio:
            if self._untrusted > self._frames_with_face:
                return fail("Picture quality too poor: fix glare, blur or distance (see hint)")
            return fail("Face not visible enough: face the camera in good light")
        # Short calibrations (a video clip, say) cannot collect as many frames, so ask for about 4 per second.
        needed = min(c.calibration_min_samples, int(4 * (c.calibration_s - c.calibration_warmup_s)))
        if len(self._ears) < needed:
            return fail("Not enough frames collected")

        ear = statistics.median(self._ears)
        mar = statistics.median(self._mars)
        pitch = statistics.median(self._pitches) if self._pitches else 0.0
        pitch_std = statistics.pstdev(self._pitches) if len(self._pitches) > 1 else 0.0

        if ear < c.calibration_min_ear:
            return fail("Eyes look closed: keep them open (glasses glare can also cause this)")
        if ear > c.calibration_max_ear:
            return fail("Eye measurement looks wrong: check camera angle and lighting")
        if mar > c.calibration_max_mar:
            return fail("Mouth looks open: keep it closed during calibration")
        if pitch_std > c.calibration_max_pitch_std:
            return fail("Head moved too much: keep still and look at the road")

        sample_s = self._last_t - (self._t0 or 0.0) - c.calibration_warmup_s
        blink_rate = None
        if self._blinks >= 3 and sample_s >= 10.0:
            blink_rate = self._blinks / (sample_s / 60.0)

        return Calibration(
            ok=True, reason="ok",
            ear_baseline=ear,
            ear_threshold=_clamp(c.ear_closed_ratio * ear, c.ear_threshold_min, c.ear_threshold_max),
            mar_baseline=mar,
            mar_threshold=_clamp(mar + c.mar_margin, c.mar_threshold_min, c.mar_threshold_max),
            pitch_baseline=pitch,
            blink_rate=blink_rate,
            n_frames=len(self._ears),
        )
