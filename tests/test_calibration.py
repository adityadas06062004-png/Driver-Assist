import json

import pytest

from drowsiness import Calibration, Calibrator, Config, DrowsinessEngine
from helpers import FRAME, make_face, run_face


def calibrate(cfg=None, plan=None):
    """plan: list of (seconds, face-kwargs or None for no face)."""
    cfg = cfg or Config()
    eng, cal = DrowsinessEngine(cfg), Calibrator(cfg)
    t = 0.0
    for seconds, face in plan:
        if face is None:
            for _ in range(int(seconds * 30)):
                cal.update(eng.update(None, t, FRAME))
                t += 1 / 30
        else:
            t, _ = run_face(lambda pts, tt: (cal.update(r := eng.update(pts, tt, FRAME)), r)[1],
                            t, seconds, **face)
    return cal


def test_good_calibration_sets_thresholds():
    cal = calibrate(plan=[(21, dict(ear=0.26, mar=0.02, pitch=-18))])
    assert cal.done
    c = cal.finish()
    assert c.ok, c.reason
    assert c.ear_baseline == pytest.approx(0.26, abs=0.01)
    assert c.ear_threshold == pytest.approx(0.75 * 0.26, abs=0.01)
    assert c.mar_threshold == pytest.approx(0.50, abs=0.01)   # floor, since resting MAR ~ 0
    assert c.pitch_baseline == pytest.approx(-18, abs=2)


def test_blink_rate_measured_when_enough_blinks():
    cfg = Config()
    eng, cal = DrowsinessEngine(cfg), Calibrator(cfg)
    t = 0.0
    for sec in range(21):  # one 0.2 s blink every 3 s
        for frame in range(30):
            closed = (sec % 3 == 0) and frame < 6
            res = eng.update(make_face(ear=0.05 if closed else 0.26), t, FRAME)
            cal.update(res)
            t += 1 / 30
    c = cal.finish()
    assert c.ok and c.blink_rate == pytest.approx(20, abs=5)


@pytest.mark.parametrize("plan,fragment", [
    ([(21, dict(ear=0.05))], "Eyes look closed"),
    ([(21, dict(ear=0.26, mar=0.8))], "Mouth looks open"),
    ([(21, None)], "Face not visible"),
])
def test_bad_calibration_is_rejected_with_reason(plan, fragment):
    c = calibrate(plan=plan).finish()
    assert not c.ok and fragment in c.reason


def test_head_moving_is_rejected():
    plan = []
    for _ in range(10):
        plan += [(1, dict(pitch=-25)), (1, dict(pitch=25))]
    c = calibrate(plan=plan).finish()
    assert not c.ok and "Head moved" in c.reason


def test_warmup_frames_are_ignored():
    # First 2 s the driver is still settling with eyes shut; should not poison the baseline.
    c = calibrate(plan=[(2, dict(ear=0.05)), (19, dict(ear=0.28))]).finish()
    assert c.ok and c.ear_baseline == pytest.approx(0.28, abs=0.01)


def test_apply_calibration_changes_engine_and_rejects_failed():
    eng = DrowsinessEngine()
    good = calibrate(plan=[(21, dict(ear=0.20, pitch=-15))]).finish()
    assert eng.apply_calibration(good)
    assert eng.ear_threshold == pytest.approx(0.15, abs=0.01)
    assert eng.pitch_offset == pytest.approx(-15, abs=2)
    before = eng.ear_threshold
    bad = calibrate(plan=[(21, dict(ear=0.05))]).finish()
    assert not eng.apply_calibration(bad) and eng.ear_threshold == before


def test_save_and_load_round_trip(tmp_path):
    c = calibrate(plan=[(21, dict(ear=0.26))]).finish()
    path = tmp_path / "sub" / "cal.json"
    c.save(path)
    assert Calibration.load(path) == c
    assert json.loads(path.read_text())["ok"] is True
