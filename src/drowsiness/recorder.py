"""Turns the monitor's per-frame state into stored events and a risk timeline.

Kept separate from Monitor so the detection code never touches a database.
"""

from __future__ import annotations

from typing import Optional, Set

from .alerts import Level
from .monitor import MonitorState
from .storage import Storage


class SessionRecorder:
    """Call ``update`` with each new MonitorState. It writes:

    an "alert" event every time the alert level changes, a "yawn" or "nod" event when
    their counts go up, an "issue" event when a picture or camera problem appears,
    a "calibrated" event once, and one timeline sample per ``sample_every_s`` seconds.
    """

    def __init__(self, storage: Storage, camera: int = 0, sample_every_s: float = 1.0,
                 note: str = "") -> None:
        self.storage = storage
        self.sample_every_s = sample_every_s
        self.session_id = storage.start_session(camera=camera, note=note)
        self._last_sample_t: Optional[float] = None
        self._yawns = 0
        self._nods = 0
        self._issues: Set[str] = set()
        self._closed = False
        self._phone_since = None

    def update(self, state: MonitorState) -> None:
        if self._closed:
            return
        res, t, sid, db = state.result, state.result.t, self.session_id, self.storage

        if state.just_calibrated and state.calibration is not None:
            cal = state.calibration
            db.set_calibration(sid, cal.ear_baseline, cal.ear_threshold)
            db.log_event(sid, t, "calibrated", detail=f"EAR baseline {cal.ear_baseline:.2f}")

        if state.phase != "monitoring":
            return

        risk = state.risk.score if state.risk else 0.0
        level = int(state.alert.level) if state.alert else 0

        if state.alert is not None and state.alert.changed:
            db.log_event(sid, t, "alert", level=level, risk=risk, detail=Level(level).name)

        if res.yawn_count > self._yawns:
            db.log_event(sid, t, "yawn", level=level, risk=risk, detail=f"{res.yawn_count} in window")
        if res.nod_count > self._nods:
            db.log_event(sid, t, "nod", level=level, risk=risk, detail=f"{res.nod_count} in window")
        self._yawns, self._nods = res.yawn_count, res.nod_count

        issues = {i.value for i in state.health.issues} if state.health else set()
        for name in sorted(issues - self._issues):
            db.log_event(sid, t, "issue", level=level, risk=risk, detail=name)
        self._issues = issues

        act = state.activity
        on_phone = act is not None and act.activity.value == "phone_call"
        if on_phone and self._phone_since is None:
            self._phone_since = t
            db.log_event(sid, t, "phone_start", level=level, risk=risk, detail="hand at ear")
        elif not on_phone and self._phone_since is not None:
            db.log_event(sid, t, "phone_end", level=level, risk=risk, detail=f"{t - self._phone_since:.0f}s")
            self._phone_since = None

        if self._last_sample_t is None or t - self._last_sample_t >= self.sample_every_s:
            self._last_sample_t = t
            if res.face_found:
                db.log_sample(sid, t, risk, level, res.ear, res.mar, res.perclos, res.blink_rate)
            else:
                db.log_sample(sid, t, risk, level)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.storage.end_session(self.session_id)
