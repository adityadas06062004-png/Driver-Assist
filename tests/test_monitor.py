import numpy as np
import pytest

from drowsiness import Calibration, Level, Monitor
from drowsiness.overlay import draw_hud
from helpers import FRAME, run_face


def feed_for(mon):
    return lambda pts, t: mon.update(pts, t, FRAME)


def calibrated_monitor(**face):
    mon = Monitor()
    face = face or dict(ear=0.26, mar=0.02)
    t, out = run_face(feed_for(mon), 0, 21, **face)
    assert mon.phase == "monitoring", out[-1].message
    return mon, t


def test_starts_calibrating_then_monitors():
    mon = Monitor()
    assert mon.phase == "calibrating"
    t, out = run_face(feed_for(mon), 0, 10, ear=0.26)
    assert out[-1].phase == "calibrating" and 0.4 < out[-1].progress < 0.6
    t, out = run_face(feed_for(mon), t, 11, ear=0.26)
    assert mon.phase == "monitoring" and mon.calibration.ok
    assert sum(s.just_calibrated for s in out) == 1


def test_failed_calibration_explains_and_retries():
    mon = Monitor()
    t, out = run_face(feed_for(mon), 0, 22, ear=0.05)      # eyes shut the whole time
    assert mon.phase == "calibrating"
    assert any("Eyes look closed" in s.message for s in out)
    t, out = run_face(feed_for(mon), t, 22, ear=0.26)      # second attempt is fine
    assert mon.phase == "monitoring"


def test_eyes_closed_triggers_critical_with_sound_then_recovers():
    mon, t = calibrated_monitor()
    t, closed = run_face(feed_for(mon), t, 2.5, ear=0.05)
    assert any(s.alert.level == Level.CRITICAL for s in closed)
    assert any(s.alert.sound == "critical" for s in closed)
    first_critical = next(s.result.t for s in closed if s.alert.level == Level.CRITICAL)
    assert first_critical - closed[0].result.t < 2.0       # alarm within 2 s of eyes closing
    t, after = run_face(feed_for(mon), t, 12, ear=0.26)
    assert after[-1].alert.level == Level.OK


def test_calibration_prevents_false_alarm_for_naturally_narrow_eyes():
    # Eyes naturally open to EAR 0.20, which a generic 0.21 threshold calls "closed".
    mon, t = calibrated_monitor(ear=0.20, mar=0.02)
    t, out = run_face(feed_for(mon), t, 30, ear=0.20, mar=0.02)
    assert not any(s.result.eyes_closed for s in out)
    assert out[-1].risk.score < 5 and out[-1].alert.level == Level.OK

    from drowsiness import DrowsinessEngine
    uncalibrated = DrowsinessEngine()
    _, res = run_face(lambda p, tt: uncalibrated.update(p, tt, FRAME), 0, 1, ear=0.20)
    assert res[-1].eyes_closed                              # the problem calibration solves


def test_pitch_baseline_stops_false_nod():
    # Camera angle makes this driver read pitch ~ +20 at rest; that must not count as a nod.
    mon, t = calibrated_monitor(pitch=20, ear=0.26, mar=0.02)
    t, out = run_face(feed_for(mon), t, 10, pitch=20, ear=0.26, mar=0.02)
    assert out[-1].result.nod_count == 0 and out[-1].alert.level == Level.OK


def test_preloaded_calibration_skips_calibrating():
    c = Calibration(True, "ok", 0.26, 0.195, 0.0, 0.5, 0.0, None, 500)
    mon = Monitor(calibration=c)
    assert mon.phase == "monitoring"


def test_recalibrate_goes_back():
    mon, t = calibrated_monitor()
    mon.recalibrate()
    assert mon.phase == "calibrating" and mon.alerts.level == Level.OK


def test_overlay_draws_in_every_phase():
    img = np.zeros((480, 640, 3), np.uint8)
    mon = Monitor()
    t, out = run_face(feed_for(mon), 0, 5, ear=0.26)
    draw_hud(img, out[-1], t)                               # calibrating
    t, out = run_face(feed_for(mon), t, 20, ear=0.26)
    draw_hud(img, out[-1], t)                               # monitoring OK
    t, out = run_face(feed_for(mon), t, 3, ear=0.05)
    draw_hud(img, out[-1], t)                               # critical
    assert img.any()
