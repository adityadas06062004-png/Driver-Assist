"""Camera -> landmarks -> Monitor, with frame skipping and low-light help.

Shared by the live app and the QA tool. Takes its camera, extractor and clock as
arguments, so it can be tested with fakes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

from .camera import Camera
from .config import Config
from .monitor import Monitor, MonitorState
from .perf import FrameGovernor
from .quality import FrameStats, NightEnhancer, measure, prepare_for_detection


@dataclass
class Step:
    view: np.ndarray               # mirrored frame to show (black if the camera is lost)
    state: MonitorState
    new: bool                      # state was computed on this step (not a skipped frame)
    stats: Optional[FrameStats]
    camera_ok: bool


class Pipeline:
    def __init__(self, monitor: Monitor, camera: Camera, extractor, config: Optional[Config] = None,
                 hands=None,
                 governor: Optional[FrameGovernor] = None,
                 clock: Callable[[], float] = time.monotonic,
                 perf_clock: Callable[[], float] = time.perf_counter) -> None:
        self.monitor = monitor
        self.camera = camera
        self.extractor = extractor
        self.hands = hands
        self._hand_calls = 0
        self._last_hands: list = []
        self.config = config or monitor.config
        self.governor = governor or FrameGovernor(camera_fps=camera.fps)
        self._clock, self._perf = clock, perf_clock
        self._t0 = clock()
        self._size = (camera.width, camera.height)
        self._state: Optional[MonitorState] = None
        self._stats: Optional[FrameStats] = None
        self._last_proc: Optional[float] = None
        self.proc_fps: Optional[float] = None
        c = self.config
        self._night = NightEnhancer(c.night_clahe_clip, c.night_blend)
        self.night_active = False

    def step(self) -> Step:
        ok, frame = self.camera.read()
        now = self._clock() - self._t0

        if not ok:
            self._stats = None
            self._state = self.monitor.update(None, now, self._size, camera_ok=False)
            blank = np.zeros((self._size[1], self._size[0], 3), np.uint8)
            return Step(blank, self._state, True, None, False)

        h, w = frame.shape[:2]
        self._size = (w, h)
        new = False
        if self._state is None or self.governor.should_process():
            started = self._perf()
            seen, self.night_active = prepare_for_detection(frame, self.config, self._night)
            landmarks = self.extractor.extract(seen)
            if self.hands is not None and self.config.detect_phone:
                if self._hand_calls % max(1, self.config.hand_every_n) == 0:
                    self._last_hands = self.hands.extract(seen)
                self._hand_calls += 1
            self._stats = measure(frame, landmarks)        # honest brightness from the raw frame
            self._state = self.monitor.update(landmarks, now, (w, h), stats=self._stats,
                                              fps=self.proc_fps, hands=self._last_hands)
            self.governor.record(self._perf() - started)
            if self._last_proc is not None:
                inst = 1.0 / max(now - self._last_proc, 1e-6)
                self.proc_fps = inst if self.proc_fps is None else 0.9 * self.proc_fps + 0.1 * inst
            self._last_proc = now
            new = True
        return Step(cv2.flip(frame, 1), self._state, new, self._stats, True)
