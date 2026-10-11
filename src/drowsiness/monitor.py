"""Whole pipeline in one object: calibrate, then monitor. No camera, no UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .activity import ActivityDetector, ActivityState
from .alerts import AlertEvent, AlertStateMachine, Level
from .calibration import Calibration, Calibrator
from .config import Config
from .engine import DrowsinessEngine, FrameResult
from .health import Health, HealthMonitor
from .mlmodel import LogisticModel, Smoother, feature_vector
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
    activity: Optional[ActivityState] = None   # driving, talking or on a phone call


class Monitor:
    def __init__(self, config: Optional[Config] = None,
                 calibration: Optional[Calibration] = None,
                 model: Optional[LogisticModel] = None, risk_mode: str = "rules") -> None:
        """``risk_mode``: "rules" (hand-tuned weights), "ml" (learned model) or "blend" (average of
        both). The "eyes closed right now" alarm always stays on as a safety net in every mode."""
        if risk_mode not in ("rules", "ml", "blend"):
            raise ValueError("risk_mode must be rules, ml or blend")
        if risk_mode != "rules" and model is None:
            raise ValueError(f"risk_mode {risk_mode!r} needs a trained model")
        self.model, self.risk_mode = model, risk_mode
        self._ml_smooth = Smoother(Config().risk_tau_up_s if config is None else config.risk_tau_up_s,
                                   Config().risk_tau_down_s if config is None else config.risk_tau_down_s)
        self._ml_score = 0.0
        self.config = config or Config()
        self.activity = ActivityDetector(self.config)
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
        self._ml_smooth.reset()
        self._ml_score = 0.0
        self.activity.reset()
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
               fps: Optional[float] = None,
               hands: Optional[List[np.ndarray]] = None) -> MonitorState:
        """Process one frame.

        ``stats`` (from quality.measure) lets the monitor notice glare, blur, low light and
        distance. ``camera_ok=False`` reports a frame that could not be read. ``hands`` are the
        hand landmarks found in the same frame (see HandLandmarkExtractor), used to spot a phone call.
        """
        if not camera_ok:
            landmarks = None
        gate = (self._health is not None and not self._health.reliable) \
            or stats_unreliable(stats, self.config)
        res = self.engine.update(landmarks, t, frame_size, trusted=not gate)
        activity = None
        if self.config.detect_phone:
            activity = self.activity.update(t, landmarks, hands, res.mar, usable=res.trusted)
        health = self.health_monitor.update(
            t, face_present=landmarks is not None, camera_ok=camera_ok,
            yaw=res.yaw, stats=stats, fps=fps,
            face_hidden_by_hand=bool(activity is not None and activity.hand_hides_face))
        self._health = health

        if self._calibrator is not None:
            self._calibrator.update(res)
            if self._calibrator.done:
                cal = self._calibrator.finish()
                if cal.ok:
                    self._adopt(cal)
                    return MonitorState("monitoring", res, 1.0, "Calibrated: monitoring",
                                        RiskState(0.0, 0.0), self.alerts.update(t, 0.0),
                                        cal, just_calibrated=True, health=health, activity=activity)
                # Failed: say why and try again from scratch.
                self._failure = cal.reason
                self._calibrator = Calibrator(self.config)
            msg = self._failure if (self._failure and self._calibrator.progress == 0.0) \
                else self._calibrator.message
            if health.message and health.issues:
                msg = health.message
            return MonitorState("calibrating", res, self._calibrator.progress, msg, health=health, activity=activity)

        risk = self.scorer.update(res)
        if self.model is not None and self.risk_mode != "rules":
            risk = self._apply_model(res, risk)
        effective = risk.score
        if health.level == Level.CRITICAL:
            effective = max(effective, self.config.critical_at + 1.0)
        elif health.level == Level.WARNING:
            effective = max(effective, self.config.warning_at + 1.0)
        c = self.config
        phone_warn = bool(activity is not None and c.phone_alert_s > 0 and activity.phone_s >= c.phone_alert_s)
        if phone_warn:
            effective = max(effective, c.warning_at + 1.0)
        alert = self.alerts.update(t, effective)
        if health.message and health.level >= Level.WARNING and health.level >= alert.level:
            alert.message = health.message
        elif phone_warn and risk.score < c.warning_at and alert.level >= Level.WARNING:
            alert.message = "Phone in use: keep both hands on the wheel"
        return MonitorState("monitoring", res, 1.0, alert.message, risk, alert,
                            self.calibration, health=health, activity=activity)

    def _apply_model(self, res: FrameResult, rules: RiskState) -> RiskState:
        vec = feature_vector(res, self.calibration.ear_baseline if self.calibration else None)
        if vec is not None:
            self._ml_score = self._ml_smooth.update(res.t, 100.0 * float(self.model.predict_proba(vec)))
        safety = 100.0 * rules.components.get("closure", 0.0)
        if self.risk_mode == "ml":
            score = max(self._ml_score, safety)
        else:
            score = max(0.5 * (rules.score + self._ml_score), safety)
        comps = dict(rules.components)
        comps["ml"] = self._ml_score / 100.0
        return RiskState(score, rules.raw, comps, rules.face_lost, rules.held)
