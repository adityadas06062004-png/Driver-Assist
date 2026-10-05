import numpy as np
import pytest

from drowsiness import Calibration, FrameStats, Level, Monitor
from helpers import FRAME, make_face

GOOD = FrameStats(120, 300, 0.0, 0.3)
GLARE = FrameStats(120, 300, 0.5, 0.3)
DARK = FrameStats(20, 300, 0.0, 0.3)
CAL = Calibration(True, "ok", 0.26, 0.195, 0.0, 0.5, 0.0, None, 500)


def monitor():
    return Monitor(calibration=CAL)


def run(mon, t, seconds, face=None, fps=30, **kw):
    pts = make_face(**face) if face is not None else None
    out = []
    for _ in range(int(round(seconds * fps))):
        out.append(mon.update(pts, t, FRAME, **kw))
        t += 1.0 / fps
    return t, out


def test_driver_vanishing_raises_warning_then_critical_with_sound():
    mon = monitor()
    t, _ = run(mon, 0, 3, dict(ear=0.26), stats=GOOD)
    t, out = run(mon, t, 7, None, stats=GOOD)
    levels = [s.alert.level for s in out]
    assert Level.WARNING in levels and levels[-1] == Level.CRITICAL
    assert any(s.alert.sound == "critical" for s in out)
    assert "Face not visible" in out[-1].message
    t, out = run(mon, t, 12, dict(ear=0.26), stats=GOOD)
    assert out[-1].alert.level == Level.OK


def test_glare_pauses_measuring_so_no_false_alarm():
    mon = monitor()
    t, _ = run(mon, 0, 3, dict(ear=0.26), stats=GOOD)
    t, out = run(mon, t, 6, dict(ear=0.05), stats=GLARE)       # "eyes closed" readings under glare
    assert not any(s.alert.level >= Level.WARNING for s in out)
    assert all(not s.result.trusted for s in out[5:])
    assert not any(s.result.eyes_closed for s in out)
    assert "Glare" in out[-1].health.message


def test_same_closed_eyes_with_a_good_picture_do_alarm():
    mon = monitor()
    t, _ = run(mon, 0, 3, dict(ear=0.26), stats=GOOD)
    t, out = run(mon, t, 3, dict(ear=0.05), stats=GOOD)
    assert any(s.alert.level == Level.CRITICAL for s in out)


def test_low_light_alone_does_not_stop_detection():
    mon = monitor()
    t, _ = run(mon, 0, 3, dict(ear=0.26), stats=DARK)
    t, out = run(mon, t, 3, dict(ear=0.05), stats=DARK)
    assert any(s.alert.level == Level.CRITICAL for s in out)
    assert "Low light" in out[-1].health.message


def test_blink_episode_is_dropped_when_picture_goes_bad():
    mon = monitor()
    t, _ = run(mon, 0, 2, dict(ear=0.26), stats=GOOD)
    t, _ = run(mon, t, 0.3, dict(ear=0.05), stats=GOOD)        # eyes closing...
    t, out = run(mon, t, 1.0, dict(ear=0.05), stats=GLARE)     # ...picture goes bad
    t, out = run(mon, t, 2.0, dict(ear=0.26), stats=GOOD)
    assert out[-1].result.eye_closed_s == 0 and not any(s.result.blinked for s in out)


def test_camera_lost_is_critical_and_recovers():
    mon = monitor()
    t, _ = run(mon, 0, 2, dict(ear=0.26), stats=GOOD)
    t, out = run(mon, t, 3, None, camera_ok=False)
    assert out[-1].alert.level == Level.CRITICAL and "Camera lost" in out[-1].message
    t, out = run(mon, t, 12, dict(ear=0.26), stats=GOOD)
    assert out[-1].alert.level == Level.OK


def test_looking_away_stops_counting_but_is_not_an_alarm():
    mon = monitor()
    t, _ = run(mon, 0, 2, dict(ear=0.26), stats=GOOD)
    t, out = run(mon, t, 6, dict(ear=0.05, yaw=45), stats=GOOD)  # head turned, eyes read "closed"
    assert not any(s.alert.level >= Level.WARNING for s in out)      # not even in the first 2 s
    assert not any(s.result.eyes_closed for s in out)
    assert "Looking away" in out[-1].health.message


def test_calibration_blames_picture_quality_not_the_driver():
    mon = Monitor()
    t, out = run(mon, 0, 22, dict(ear=0.26), stats=GLARE)
    assert mon.phase == "calibrating"
    assert any("Glare" in s.message or "Picture quality" in s.message for s in out)
    t, out = run(mon, t, 22, dict(ear=0.26), stats=GOOD)       # fixed the glare, tries again
    assert mon.phase == "monitoring"
