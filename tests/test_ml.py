import csv
import sys

import numpy as np
import pytest

from drowsiness import Calibration, Level, LogisticModel, Monitor
from drowsiness import train as train_mod
from drowsiness.collect import COLUMNS, append_rows, make_row
from drowsiness.engine import FrameResult
from drowsiness.mlmodel import FEATURES, Smoother, binary_metrics, feature_vector, leave_one_session_out
from helpers import FRAME, make_face, run_face

CAL = Calibration(True, "ok", 0.26, 0.195, 0.0, 0.5, 0.0, None, 500)


def toy_data(n=200, seed=0):
    """Alert rows: eyes open, no closure. Drowsy rows: eyes drooping, closures, high PERCLOS."""
    rng = np.random.default_rng(seed)
    alert = np.column_stack([rng.normal(1.0, 0.08, n), rng.normal(0.05, 0.03, n), rng.normal(3, 2, n),
                             rng.normal(0.03, 0.02, n), np.zeros(n), rng.normal(18, 4, n),
                             np.zeros(n), np.zeros(n)])
    drowsy = np.column_stack([rng.normal(0.55, 0.2, n), rng.normal(0.3, 0.3, n), rng.normal(14, 6, n),
                              rng.normal(0.35, 0.1, n), rng.uniform(0, 3, n), rng.normal(30, 8, n),
                              rng.integers(0, 4, n), rng.integers(0, 3, n)])
    X = np.vstack([alert, drowsy])
    y = np.array([0] * n + [1] * n)
    return X, y


def test_model_learns_a_separable_problem_and_round_trips(tmp_path):
    X, y = toy_data()
    model = LogisticModel.fit(X, y)
    acc = binary_metrics(y, (model.predict_proba(X) >= 0.5).astype(int))["accuracy"]
    assert acc > 0.95
    path = tmp_path / "m" / "model.json"
    model.save(path)
    again = LogisticModel.load(path)
    assert again.predict_proba(X[0]) == pytest.approx(model.predict_proba(X[0]))
    assert model.importance()[0][0] in FEATURES


def test_training_needs_both_classes():
    X, y = toy_data(20)
    with pytest.raises(ValueError):
        LogisticModel.fit(X, np.zeros(len(y)))


def test_metrics():
    m = binary_metrics([1, 1, 0, 0], [1, 0, 0, 1])
    assert m["accuracy"] == 0.5 and m["recall"] == 0.5 and m["precision"] == 0.5
    assert m["false_alarm_rate"] == 0.5


def test_leave_one_session_out_never_trains_on_the_held_out_session():
    X, y = toy_data(60)
    groups = np.array([f"a{i // 20}" for i in range(60)] + [f"d{i // 20}" for i in range(60)])
    pred = leave_one_session_out(X, y, groups)
    assert binary_metrics(y, pred)["accuracy"] > 0.9
    with pytest.raises(ValueError):
        leave_one_session_out(X, y, np.array(["a"] * 60 + ["d"] * 60))   # one session per class


def test_smoother_rises_fast_and_falls_slowly():
    s = Smoother(0.25, 3.0)
    s.update(0.0, 0)
    up = s.update(0.5, 100)
    down = s.update(1.0, 0)
    assert up > 80 and down > 0.7 * up


def res(**kw):
    base = dict(t=1.0, face_found=True, ear=0.26, mar=0.05, pitch=2.0, perclos=0.1, eye_closed_s=0.0,
                blink_rate=15.0, yawn_count=1, nod_count=0, trusted=True)
    base.update(kw)
    return FrameResult(**base)


def test_feature_vector_scales_ear_by_the_drivers_baseline_and_skips_bad_frames():
    v = feature_vector(res(), 0.26)
    assert len(v) == len(FEATURES) and v[0] == pytest.approx(1.0)
    assert feature_vector(res(trusted=False), 0.26) is None
    assert feature_vector(res(face_found=False), 0.26) is None
    assert feature_vector(res(ear=float("nan")), 0.26) is None
    assert feature_vector(res(pitch=float("nan")), 0.26)[2] == 0.0


def trained_model():
    X, y = toy_data()
    return LogisticModel.fit(X, y)


def feed(mon, t, seconds, **face):
    pts = make_face(**face)
    out = []
    for _ in range(int(seconds * 30)):
        out.append(mon.update(pts, t, FRAME))
        t += 1 / 30
    return t, out


def test_ml_mode_needs_a_model_and_a_valid_name():
    with pytest.raises(ValueError):
        Monitor(risk_mode="ml")
    with pytest.raises(ValueError):
        Monitor(risk_mode="magic", model=trained_model())


def test_ml_mode_stays_calm_when_alert_and_raises_risk_when_drooping():
    mon = Monitor(calibration=CAL, model=trained_model(), risk_mode="ml")
    t, out = feed(mon, 0, 8, ear=0.26)
    assert out[-1].alert.level == Level.OK and out[-1].risk.score < 25
    assert "ml" in out[-1].risk.components
    t, out = feed(mon, t, 4, ear=0.05)
    assert any(s.alert.level >= Level.WARNING for s in out)


def test_safety_net_keeps_the_eyes_closed_alarm_even_if_the_model_says_alert():
    class AlwaysAlert:
        def predict_proba(self, x):
            return 0.0

    mon = Monitor(calibration=CAL, model=AlwaysAlert(), risk_mode="ml")
    t, _ = feed(mon, 0, 2, ear=0.26)
    t, out = feed(mon, t, 3, ear=0.05)
    assert any(s.alert.level == Level.CRITICAL for s in out)


def test_blend_mode_runs():
    mon = Monitor(calibration=CAL, model=trained_model(), risk_mode="blend")
    t, out = feed(mon, 0, 3, ear=0.26)
    assert 0 <= out[-1].risk.score <= 100


def test_rules_mode_is_unchanged_by_a_loaded_model():
    a = Monitor(calibration=CAL)
    b = Monitor(calibration=CAL, model=trained_model(), risk_mode="rules")
    _, oa = feed(a, 0, 3, ear=0.05)
    _, ob = feed(b, 0, 3, ear=0.05)
    assert oa[-1].risk.score == pytest.approx(ob[-1].risk.score)


def test_make_row_and_append_rows_write_a_readable_csv(tmp_path):
    mon = Monitor(calibration=CAL)
    _, out = run_face(lambda p, t: mon.update(p, t, FRAME), 0, 2, ear=0.26)
    row = make_row("s1", "alert", out[-1])
    assert row is not None and len(row) == len(COLUMNS)
    path = tmp_path / "d" / "samples.csv"
    append_rows(path, [row])
    append_rows(path, [row])
    rows = list(csv.DictReader(open(path)))
    assert len(rows) == 2 and rows[0]["label"] == "alert" and float(rows[0]["ear_ratio"]) > 0.9


def test_train_command_end_to_end(tmp_path, monkeypatch, capsys):
    X, y = toy_data(90, seed=3)
    rows = []
    for i, (x, lab) in enumerate(zip(X, y)):
        sess = f"{'drowsy' if lab else 'alert'}-{(i % 90) // 30}"
        rows.append([sess, "drowsy" if lab else "alert", float(i), 60.0 if lab else 5.0, *x.tolist()])
    data = tmp_path / "samples.csv"
    append_rows(data, rows)
    out, report = tmp_path / "model.json", tmp_path / "report.md"
    monkeypatch.setattr(sys, "argv", ["train", "--data", str(data), "--out", str(out), "--report", str(report)])
    train_mod.main()
    assert out.exists() and LogisticModel.load(out)
    text = report.read_text()
    assert "Hand-tuned rules" in text and "unseen sessions" in text and "acted" in text
    assert "ROC AUC" in text and "Per session" in text
    assert "accuracy" in capsys.readouterr().out


def test_train_without_data_explains_what_to_do(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train", "--data", str(tmp_path / "none.csv")])
    with pytest.raises(SystemExit) as e:
        train_mod.main()
    assert "collect" in str(e.value)


def test_roc_auc_and_recall_at_false_alarms():
    from drowsiness.mlmodel import recall_at_false_alarms, roc_auc
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    assert roc_auc(y, [1, 2, 3, 4, 5, 6, 7, 8]) == pytest.approx(1.0)
    assert roc_auc(y, [8, 7, 6, 5, 4, 3, 2, 1]) == pytest.approx(0.0)
    assert roc_auc(y, [1] * 8) == pytest.approx(0.5)               # ties count as guessing
    assert roc_auc([1, 1], [1, 2]) != roc_auc([1, 1], [1, 2])      # one class only: NaN
    assert recall_at_false_alarms(y, [1, 2, 3, 4, 5, 6, 7, 8], 0.05) == 1.0
    assert recall_at_false_alarms(y, [1, 2, 3, 6, 5, 4, 7, 8], 0.05) == pytest.approx(0.5)


def test_leave_one_out_can_return_probabilities():
    X, y = toy_data(60)
    groups = np.array([f"a{i // 20}" for i in range(60)] + [f"d{i // 20}" for i in range(60)])
    p = leave_one_session_out(X, y, groups, proba=True)
    assert p.dtype == float and 0 <= p.min() and p.max() <= 1
