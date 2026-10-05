"""Turn per-frame metrics into one smoothed 0..100 drowsiness risk score."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

from .config import Config
from .engine import FrameResult


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _ramp(x: float, lo: float, hi: float) -> float:
    """0 at or below lo, 1 at or above hi, linear in between."""
    return _clamp01((x - lo) / (hi - lo)) if hi > lo else (1.0 if x >= hi else 0.0)


@dataclass
class RiskState:
    score: float                 # smoothed, 0..100
    raw: float                   # before smoothing
    components: Dict[str, float] = field(default_factory=dict)  # each 0..1
    face_lost: bool = False
    held: bool = False           # no usable data this frame, score carried over


class RiskScorer:
    """Two ingredients, combined with max():

    * trend: weighted PERCLOS, yawns, nods and blink-rate change over the last minutes
    * acute: eyes closed right now, or head sustained down (these can't wait for a trend)

    The result is smoothed with a fast rise and slow fall, so alarms come quickly
    but don't flicker off the moment the eyes reopen.
    """

    def __init__(self, config: Optional[Config] = None,
                 baseline_blink_rate: Optional[float] = None) -> None:
        self.config = config or Config()
        self.baseline_blink_rate = baseline_blink_rate
        self._score = 0.0
        self._last_t: Optional[float] = None
        self._last_components: Dict[str, float] = {}

    def reset(self) -> None:
        self._score = 0.0
        self._last_t = None
        self._last_components = {}

    def update(self, res: FrameResult) -> RiskState:
        if not res.face_found or not res.trusted:
            # No usable evidence: hold the score (health.py raises its own alert for this).
            self._last_t = res.t
            return RiskState(self._score, self._score, dict(self._last_components),
                             face_lost=not res.face_found, held=True)

        c = self.config
        comp = {
            "closure": _ramp(res.eye_closed_s, c.closure_start_s, c.closure_full_s),
            "perclos": (_ramp(res.perclos, c.perclos_low, c.perclos_high)
                        if res.window_span_s >= c.perclos_min_span_s else 0.0),
            "yawn": _clamp01(res.yawn_count / c.yawns_full),
            "nod": max(_clamp01(res.nod_count / c.nods_full),
                       c.nod_acute_level if res.nodding else 0.0),
            "blink": self._blink_component(res.blink_rate),
        }
        trend = 100.0 * (c.w_perclos * comp["perclos"] + c.w_yawn * comp["yawn"]
                         + c.w_nod * comp["nod"] + c.w_blink * comp["blink"])
        acute = 100.0 * max(comp["closure"], c.nod_acute_level if res.nodding else 0.0)
        raw = max(trend, acute)

        if self._last_t is None:
            self._score = raw
        else:
            dt = min(max(res.t - self._last_t, 0.0), 1.0)
            tau = c.risk_tau_up_s if raw > self._score else c.risk_tau_down_s
            self._score += (raw - self._score) * (1.0 - math.exp(-dt / tau))
        self._last_t = res.t
        self._last_components = comp
        return RiskState(self._score, raw, comp)

    def _blink_component(self, rate: float) -> float:
        base = self.baseline_blink_rate
        if base is None or rate <= 0.0:
            return 0.0
        deviation = abs(rate - base) / max(base, 5.0)
        c = self.config
        return _ramp(deviation, c.blink_dev_start, c.blink_dev_full)
