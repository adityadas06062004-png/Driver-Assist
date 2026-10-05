"""Whole pipeline in one object: calibrate, then monitor. No camera, no UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .alerts import AlertEvent, AlertStateMachine, Level
from .calibration import Calibration, Calibrator
from .config import Config
from .engine import DrowsinessEngine, FrameResult
from .health import Health, HealthMonitor
from .quality import FrameStats, stats_unreliable
from .risk import RiskScorer, RiskState


@dataclass
class MonitorState:
    phase: str                          # "calibrating" or "monitoring"
    result: FrameResult
    progress: float = 0.0               # calibration progress 0..1
    message: str = ""
    risk: Optional[RiskState] = None
    alert: Optional[AlertEvent] = None
    calibration: Optional[Calibration] = None
    just_calibrated: bool = False       # calibration succeeded on this frame
    health: Optional[Health] = None


class Monitor:
    def __init__(self, config: Optional[Config] = None,
                 calibration: Optional[Calibration] = None) -> None:
        self.config = config or Config()
        self.engine = DrowsinessEngine(self.config)
        self.scorer = RiskScorer(self.config)
        self.alerts = AlertStateMachine(self.config)
        self.health_monitor = HealthMonitor(self.config)
        self._health: Optional[Health] = None
        self.calibration: Optional[Calibration] = None
        self._calibrator: Optional[Calibrator] = Calibrator(self.config)
        self._failure = ""
        if calibration is not None and calibration.ok:
            self._adopt(calibration)

    @property
    def phase(self) -> str:
        return "monitoring" if self.calibration is not None else "calibrating"

    def recalibrate(self) -> None:
        self.engine.reset_calibration()
        self.scorer.reset()
        self.scorer.baseline_blink_rate = None
        self.alerts.reset()
        self.health_monitor.reset()
        self._health = None
        self.calibration = None
        self._failure = ""
        self._calibrator = Calibrator(self.config)

    def _adopt(self, cal: Calibration) -> None:
        self.engine.apply_calibration(cal)
        self.scorer.reset()
        self.scorer.baseline_blink_rate = cal.blink_rate
        self.alerts.reset()
        self.calibration = cal
        self._calibrator = None

    def update(self, landmarks: Optional[np.ndarray], t: float, frame_size: Tuple[int, int],
               stats: Optional[FrameStats] = None, camera_ok: bool = True,
               fps: Optional[float] = None) -> MonitorState:
        """Process one frame.

        ``stats`` (from quality.measure) lets the monitor notice glare, blur, low light and
        distance. ``camera_ok=False`` reports a frame that could not be read.
        """
        if not camera_ok:
            landmarks = None
        gate = (self._health is not None and not self._health.reliable) \
            or stats_unreliable(stats, self.config)
        res = self.engine.update(landmarks, t, frame_size, trusted=not gate)
        health = self.health_monitor.update(
            t, face_present=landmarks is not None, camera_ok=camera_ok,
            yaw=res.yaw, stats=stats, fps=fps)
        self._health = health

        if self._calibrator is not None:
            self._calibrator.update(res)
            if self._calibrator.done:
                cal = self._calibrator.finish()
                if cal.ok:
                    self._adopt(cal)
                    return MonitorState("monitoring", res, 1.0, "Calibrated: monitoring",
                                        RiskState(0.0, 0.0), self.alerts.update(t, 0.0),
                                        cal, just_calibrated=True, health=health)
                # Failed: say why and try again from scratch.
                self._failure = cal.reason
                self._calibrator = Calibrator(self.config)
            msg = self._failure if (self._failure and self._calibrator.progress == 0.0) \
                else self._calibrator.message
            if health.message and health.issues:
                msg = health.message
            return MonitorState("calibrating", res, self._calibrator.progress, msg, health=health)

        risk = self.scorer.update(res)
        effective = risk.score
        if health.level == Level.CRITICAL:
            effective = max(effective, self.config.critical_at + 1.0)
        elif health.level == Level.WARNING:
            effective = max(effective, self.config.warning_at + 1.0)
        alert = self.alerts.update(t, effective)
        if health.message and health.level >= Level.WARNING and health.level >= alert.level:
            alert.message = health.message
        return MonitorState("monitoring", res, 1.0, alert.message, risk, alert,
                            self.calibration, health=health)
