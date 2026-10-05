import pytest

from drowsiness import Config, DrowsinessEngine
from helpers import FRAME, make_face

DT = 1 / 30


def feed(engine, seconds, t0=0.0, **face):
    pts = make_face(**face) if face is not None else None
    t, res = t0, None
    for _ in range(int(seconds / DT)):
        res = engine.update(pts, t, FRAME)
        t += DT
    return t, res


def test_awake_driver_is_quiet():
    eng = DrowsinessEngine()
    _, res = feed(eng, 5, ear=0.30, mar=0.05)
    assert res.face_found and not res.eyes_closed
    assert res.perclos == 0 and res.yawn_count == 0 and res.nod_count == 0
    assert res.ear == pytest.approx(0.30, abs=1e-3)


def test_eyes_closed_two_seconds():
    eng = DrowsinessEngine()
    t, _ = feed(eng, 2, ear=0.30)
    _, res = feed(eng, 2, t0=t, ear=0.05)
    assert res.eyes_closed
    assert res.eye_closed_s == pytest.approx(2.0, abs=0.2)
    assert res.perclos > 0.3


def test_yawn_detected_after_sustained_open_mouth():
    eng = DrowsinessEngine()
    t, _ = feed(eng, 1, mar=0.05)
    _, res = feed(eng, 2.5, t0=t, mar=0.9)
    assert res.yawning and res.yawn_count == 1


def test_head_nod_detected():
    eng = DrowsinessEngine()
    t, _ = feed(eng, 1, pitch=0)
    _, res = feed(eng, 2, t0=t, pitch=25)
    assert res.nodding and res.nod_count == 1
    assert res.pitch == pytest.approx(25, abs=2)


def test_pitch_offset_baseline_is_subtracted():
    eng = DrowsinessEngine()
    eng.pitch_offset = 10.0
    _, res = feed(eng, 1, pitch=10)
    assert res.pitch == pytest.approx(0, abs=1.5)


def test_no_face_reports_and_resets_episode():
    eng = DrowsinessEngine(Config(face_lost_reset_s=0.5))
    t, _ = feed(eng, 1, ear=0.30)
    t, _ = feed(eng, 0.3, t0=t, ear=0.05)           # eyes closing...
    res = None
    for _ in range(60):                              # ...then 2 s with no face
        res = eng.update(None, t, FRAME)
        t += DT
    assert not res.face_found and res.no_face_s > 1.5
    _, res = feed(eng, 0.5, t0=t, ear=0.30)          # face back, eyes open
    assert res.eye_closed_s == 0 and not res.blinked
