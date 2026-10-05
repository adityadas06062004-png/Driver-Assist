"""Keep detection real-time on slow machines by processing every Nth frame."""

from __future__ import annotations

import math
from typing import Optional


class FrameGovernor:
    """If processing one frame takes longer than the camera's frame interval, process only
    every ``stride``-th frame (up to ``max_stride``). The engine is time-based, so a lower
    processing rate still gives correct PERCLOS, blink and yawn timing.
    """

    def __init__(self, camera_fps: float = 30.0, max_stride: int = 3, smoothing: float = 0.2) -> None:
        self.budget_s = 1.0 / camera_fps
        self.camera_fps = camera_fps
        self.max_stride = max_stride
        self.smoothing = smoothing
        self.stride = 1
        self._ema: Optional[float] = None
        self._i = 0

    def should_process(self) -> bool:
        go = self._i % self.stride == 0
        self._i += 1
        return go

    def record(self, seconds: float) -> None:
        """Report how long one processed frame took."""
        a = self.smoothing
        self._ema = seconds if self._ema is None else (1 - a) * self._ema + a * seconds
        if self._ema > self.budget_s:                                   # too slow: skip more
            self.stride = min(self.max_stride, max(self.stride, math.ceil(self._ema / self.budget_s)))
        elif self.stride > 1 and self._ema < 0.7 * self.budget_s:       # clearly fast again
            self.stride -= 1

    @property
    def effective_fps(self) -> float:
        return self.camera_fps / self.stride
