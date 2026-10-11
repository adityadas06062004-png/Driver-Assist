import csv

import pytest

from drowsiness import Calibration, Monitor, SessionRecorder, Storage
from helpers import FRAME, make_face

CAL = Calibration(True, "ok", 0.26, 0.195, 0.0, 0.5, 0.0, None, 500)


class Tick:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        self.t += 1.0
        return self.t


def test_session_lifecycle_and_summary():
    db = Storage(":memory:", clock=Tick())
    sid = db.start_session(camera=1, note="glasses")
    db.set_calibration(sid, 0.26, 0.195)
    db.log_event(sid, 5.0, "alert", level=2, risk=60.0, detail="WARNING")
    db.log_event(sid, 8.0, "alert", level=3, risk=90.0, detail="CRITICAL")
    db.log_event(sid, 9.0, "yawn", level=3, risk=90.0)
    for i, (risk, lvl) in enumerate([(5, 0), (60, 2), (90, 3), (90, 3)]):
        db.log_sample(sid, float(i), risk, lvl, ear=0.2, mar=float("nan"))
    db.end_session(sid)
    s = db.summary(sid)
    assert s["max_risk"] == 90 and s["avg_risk"] == pytest.approx(61.25)
    assert s["alerts"] == {"CAUTION": 0, "WARNING": 1, "CRITICAL": 1}
    assert s["seconds_at_level"]["CRITICAL"] == 2 and s["yawns"] == 1 and s["duration_s"] > 0
    assert db.sessions()[0]["ear_threshold"] == pytest.approx(0.195)
    assert db.samples(sid)[0]["mar"] is None          # NaN stored as NULL


def test_unknown_session_and_delete_cascades():
    db = Storage(":memory:")
    with pytest.raises(KeyError):
        db.summary(99)
    sid = db.start_session()
    db.log_event(sid, 1.0, "nod")
    db.log_sample(sid, 1.0, 10, 0)
    db.delete_session(sid)
    assert db.events() == [] and db.samples() == []


def test_csv_export(tmp_path):
    db = Storage(":memory:")
    a, b = db.start_session(), db.start_session()
    db.log_event(a, 1.0, "alert", level=2, risk=55.5, detail="WARNING")
    db.log_event(b, 2.0, "nod")
    out = tmp_path / "x" / "events.csv"
    assert db.export_csv(out, "events", session_id=a) == 1
    rows = list(csv.DictReader(open(out)))
    assert rows[0]["kind"] == "alert" and rows[0]["risk"] == "55.5"
    assert db.export_csv(tmp_path / "all.csv", "events") == 2
    assert db.export_csv(tmp_path / "s.csv", "sessions") == 2
    with pytest.raises(ValueError):
        db.export_csv(tmp_path / "bad.csv", "nothing")


def test_database_file_is_created_and_reopened(tmp_path):
    path = tmp_path / "deep" / "da.db"
    with Storage(path) as db:
        db.start_session()
    with Storage(path) as db:
        assert len(db.sessions()) == 1


def run(mon, rec, t, seconds, fps=30, **face):
    pts = make_face(**face)
    for _ in range(int(round(seconds * fps))):
        rec.update(mon.update(pts, t, FRAME))
        t += 1 / fps
    return t


def test_recorder_logs_alert_changes_calibration_and_timeline():
    db = Storage(":memory:")
    mon = Monitor()
    rec = SessionRecorder(db)
    t = run(mon, rec, 0.0, 21, ear=0.26, mar=0.02)           # calibrate
    assert [e["kind"] for e in db.events(rec.session_id)] == ["calibrated"]
    assert db.sessions()[0]["ear_baseline"] == pytest.approx(0.26, abs=0.01)
    t = run(mon, rec, t, 3, ear=0.05)                         # eyes closed
    t = run(mon, rec, t, 12, ear=0.26)                        # recover
    rec.close()
    alerts = [e["detail"] for e in db.events(rec.session_id) if e["kind"] == "alert"]
    assert "CRITICAL" in alerts and alerts[-1] == "OK"
    samples = db.samples(rec.session_id)
    assert 12 <= len(samples) <= 18 and max(s["risk"] for s in samples) > 75
    assert db.sessions()[0]["ended_at"] is not None
    rec.update(mon.update(make_face(), t, FRAME))              # ignored after close
    assert len(db.samples(rec.session_id)) == len(samples)


def test_recorder_logs_yawn_nod_and_issue_once():
    db = Storage(":memory:")
    rec = SessionRecorder(db)
    mon = Monitor(calibration=CAL)
    t = run(mon, rec, 0.0, 1, ear=0.26)
    t = run(mon, rec, t, 3, ear=0.26, mar=0.9)                # yawn
    t = run(mon, rec, t, 1, ear=0.26)
    t = run(mon, rec, t, 3, ear=0.26, pitch=25)               # nod
    kinds = [e["kind"] for e in db.events(rec.session_id)]
    assert kinds.count("yawn") == 1 and kinds.count("nod") == 1
    # face disappears long enough to count as an issue
    pts_none = None
    for i in range(120):
        rec.update(mon.update(pts_none, t + i / 30, FRAME))
    issues = [e["detail"] for e in db.events(rec.session_id) if e["kind"] == "issue"]
    assert issues.count("no_face") == 1
