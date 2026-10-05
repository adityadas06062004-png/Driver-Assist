"""Camera access that survives unplugging, sleep/wake and failed reads."""

from __future__ import annotations

import time
from typing import Callable, Optional, Tuple

import cv2
import numpy as np


class Camera:
    """Wraps cv2.VideoCapture. A failed read never raises: it returns (False, None),
    drops the device, and reopens it every ``reopen_interval_s`` until it works again.
    """

    def __init__(self, index: int = 0, width: int = 640, height: int = 480, fps: int = 30,
                 reopen_interval_s: float = 1.0,
                 capture_factory: Optional[Callable] = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.index, self.width, self.height, self.fps = index, width, height, fps
        self.reopen_interval_s = reopen_interval_s
        self._factory = capture_factory or cv2.VideoCapture
        self._clock = clock
        self._cap = None
        self._last_try = float("-inf")
        self.failures = 0          # consecutive failed reads

    @property
    def is_open(self) -> bool:
        return self._cap is not None

    def open(self) -> bool:
        self._release()
        cap = self._factory(self.index)
        if cap is not None and cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            self._cap = cap
            return True
        if cap is not None:
            cap.release()
        return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        now = self._clock()
        if self._cap is None:
            if now - self._last_try >= self.reopen_interval_s:
                self._last_try = now
                self.open()
            if self._cap is None:
                self.failures += 1
                return False, None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self.failures += 1
            self._release()
            self._last_try = now          # wait one interval before reopening
            return False, None
        self.failures = 0
        return True, frame

    def release(self) -> None:
        self._release()

    def _release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None
