"""Time-based rolling metrics. Pure Python, no camera or OpenCV needed."""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Tuple


class PerclosTracker:
    """PERCLOS = share of time (0..1) the eyes were closed within a rolling window.

    Time-weighted, so it does not depend on the camera frame rate. Gaps longer
    than ``max_gap_s`` (face lost, app stalled) are ignored instead of counted.
    """

    def __init__(self, window_s: float, max_gap_s: float) -> None:
        self.window_s = window_s
        self.max_gap_s = max_gap_s
        self._samples: Deque[Tuple[float, bool]] = deque()

    def update(self, t: float, closed: bool) -> float:
        self._samples.append((t, closed))
        while self._samples and t - self._samples[0][0] > self.window_s:
            self._samples.popleft()
        return self.value

    @property
    def value(self) -> float:
        closed_time = total = 0.0
        prev_t: Optional[float] = None
        prev_closed = False
        for t, closed in self._samples:
            if prev_t is not None:
                dt = t - prev_t
                if 0 < dt <= self.max_gap_s:
                    total += dt
                    if prev_closed:
                        closed_time += dt
            prev_t, prev_closed = t, closed
        return closed_time / total if total > 0 else 0.0

    @property
    def span_s(self) -> float:
        """How many seconds of data the window currently holds."""
        return self._samples[-1][0] - self._samples[0][0] if len(self._samples) > 1 else 0.0

    def reset(self) -> None:
        self._samples.clear()


class BlinkCounter:
    """Counts blinks (eye closures of min_blink_s..max_blink_s) in a rolling window.

    Also exposes how long the eyes have currently been closed, which is what the
    "eyes closed for 1.5 s" alarm in Phase 2 will use.
    """

    def __init__(self, min_blink_s: float, max_blink_s: float, window_s: float,
                 min_span_s: float) -> None:
        self.min_blink_s = min_blink_s
        self.max_blink_s = max_blink_s
        self.window_s = window_s
        self.min_span_s = min_span_s
        self._closed_since: Optional[float] = None
        self._events: Deque[float] = deque()
        self._t0: Optional[float] = None

    def update(self, t: float, closed: bool) -> bool:
        """Feed one frame. Returns True on the frame a blink finishes."""
        if self._t0 is None:
            self._t0 = t
        blink = False
        if closed:
            if self._closed_since is None:
                self._closed_since = t
        elif self._closed_since is not None:
            if self.min_blink_s <= t - self._closed_since <= self.max_blink_s:
                self._events.append(t)
                blink = True
            self._closed_since = None
        self._trim(t)
        return blink

    def closure_duration_s(self, t: float) -> float:
        return 0.0 if self._closed_since is None else t - self._closed_since

    def rate_per_min(self, t: float) -> float:
        self._trim(t)
        if self._t0 is None:
            return 0.0
        span = min(t - self._t0, self.window_s)
        if span < self.min_span_s:
            return 0.0
        return len(self._events) / (span / 60.0)

    def interrupt(self) -> None:
        """Face lost: drop the in-progress closure without counting it."""
        self._closed_since = None

    def _trim(self, t: float) -> None:
        while self._events and t - self._events[0] > self.window_s:
            self._events.popleft()


class SustainedEventDetector:
    """Fires once when a condition has stayed true for ``min_s`` seconds.

    Used for yawns (mouth open) and head nods (head down). Keeps the times of
    past events so it can report how many happened in the last ``window_s``.
    """

    def __init__(self, min_s: float, window_s: float) -> None:
        self.min_s = min_s
        self.window_s = window_s
        self._since: Optional[float] = None
        self._fired = False
        self._events: Deque[float] = deque()

    def update(self, t: float, active: bool) -> bool:
        """Feed one frame. Returns True on the frame the event fires."""
        fired = False
        if active:
            if self._since is None:
                self._since = t
                self._fired = False
            if not self._fired and t - self._since >= self.min_s:
                self._fired = True
                self._events.append(t)
                fired = True
        else:
            self._since = None
            self._fired = False
        self._trim(t)
        return fired

    def duration_s(self, t: float) -> float:
        return 0.0 if self._since is None else t - self._since

    @property
    def is_sustained(self) -> bool:
        return self._fired

    def count(self, t: float) -> int:
        self._trim(t)
        return len(self._events)

    def interrupt(self) -> None:
        self._since = None
        self._fired = False

    def _trim(self, t: float) -> None:
        while self._events and t - self._events[0] > self.window_s:
            self._events.popleft()
