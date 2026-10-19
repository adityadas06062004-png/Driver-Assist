import pytest

from drowsiness.insights import alertness_label, alertness_score, insights, longest_stretch, mmss, phone_seconds


def S(t, risk):
    return {"t": float(t), "risk": float(risk)}


def summary(**kw):
    base = {"max_risk": 10.0, "avg_risk": 5.0, "yawns": 0, "nods": 0, "phone_calls": 0, "issues": 0}
    base.update(kw)
    return base


def test_time_and_score_helpers():
    assert mmss(0) == "0:00" and mmss(65.9) == "1:05" and mmss(-3) == "0:00"
    assert alertness_score(0) == 100 and alertness_score(44.6) == 55 and alertness_score(150) == 0
    assert alertness_label(90)[1] == "good" and alertness_label(50) == ("Signs of fatigue", "warn")
    assert alertness_label(20) == ("High fatigue risk", "bad")


def test_longest_stretch_finds_the_longest_run_and_handles_open_ends():
    samples = [S(0, 10), S(1, 60), S(2, 70), S(3, 20), S(4, 55), S(5, 56), S(6, 57), S(7, 58)]
    assert longest_stretch(samples) == (4.0, 3.0)
    assert longest_stretch([S(0, 10), S(1, 20)]) is None
    assert longest_stretch([S(0, 60), S(2, 60)]) == (0.0, 2.0)


def test_phone_seconds_pairs_start_and_end():
    ev = [{"kind": "phone_start", "t": 10}, {"kind": "phone_end", "t": 25}, {"kind": "phone_start", "t": 40}]
    assert phone_seconds(ev) == pytest.approx(15)


def test_a_calm_session_says_so():
    out = insights(summary(), [S(0, 3), S(1, 8)], [])
    assert out[0].tone == "good" and "No signs of fatigue" in out[0].title


def test_a_rough_session_gets_specific_highlights():
    samples = [S(i, 90 if 10 <= i < 20 else 10) for i in range(40)]
    events = [{"kind": "alert", "t": 10.0, "level": 2}, {"kind": "yawn", "t": 11.0},
              {"kind": "phone_start", "t": 25.0}, {"kind": "phone_end", "t": 35.0},
              {"kind": "issue", "t": 5.0, "detail": "low_light"}]
    out = insights(summary(max_risk=90, yawns=4, nods=2, phone_calls=1), samples, events)
    titles = [i.title for i in out]
    assert "Highest risk" in titles and "Repeated yawning" in titles and "Head nodding" in titles
    assert any("Phone use" == t for t in titles) and any("low light" in t for t in titles)
    text = {i.title: i.text for i in out}
    assert "yawn" in text["Highest risk"] or "peaked at 90" in text["Highest risk"]
    assert "10 seconds" in text["Phone use"]
    assert "starting at 0:10" in text["Longest stretch at warning level or above"]
    assert text["First alarm"].endswith("0:10.")


def test_no_samples_is_handled():
    assert insights(summary(), [], [])[0].title == "Not enough data"
