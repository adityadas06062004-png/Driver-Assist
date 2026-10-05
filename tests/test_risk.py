import pytest

from drowsiness import Config, DrowsinessEngine, RiskScorer
from helpers import FRAME, run_face


def make_pipeline(baseline_blink=None):
    eng, scorer = DrowsinessEngine(), RiskScorer(baseline_blink_rate=baseline_blink)

    def feed(pts, t):
        return scorer.update(eng.update(pts, t, FRAME))

    return feed


def test_alert_driver_scores_near_zero():
    feed = make_pipeline()
    _, out = run_face(feed, 0, 30, ear=0.30, mar=0.05)
    assert max(r.score for r in out) < 5


def test_eyes_closed_rises_fast_to_critical():
    feed = make_pipeline()
    t, _ = run_face(feed, 0, 2, ear=0.30)
    _, out = run_face(feed, t, 2.5, ear=0.05)
    scores = [r.score for r in out]
    assert scores[int(0.5 * 30)] < 25          # 0.5 s of closure is still not alarming
    assert scores[int(1.0 * 30)] > 30          # 1 s: clearly rising
    assert scores[-1] > 90                     # 2.5 s: maximum


def test_score_falls_slowly_after_eyes_reopen():
    feed = make_pipeline()
    t, _ = run_face(feed, 0, 2, ear=0.30)
    t, closed = run_face(feed, t, 2.5, ear=0.05)
    peak = closed[-1].score
    t, after = run_face(feed, t, 1.0, ear=0.30)
    assert after[-1].score > 0.6 * peak        # still elevated after 1 s
    t, later = run_face(feed, t, 10, ear=0.30)
    # It comes back down, leaving only the PERCLOS memory of the closure, below CAUTION (25).
    assert later[-1].score < 25
    assert later[-1].score == pytest.approx(later[-1].raw, abs=2)


def test_repeated_yawns_raise_risk():
    feed = make_pipeline()
    t = 0.0
    for _ in range(3):
        t, _ = run_face(feed, t, 3, mar=0.05)
        t, out = run_face(feed, t, 2.5, mar=0.9)
    assert out[-1].components["yawn"] == pytest.approx(1.0)
    assert out[-1].raw == pytest.approx(100 * Config().w_yawn, abs=1)


def test_sustained_nod_is_acute_warning():
    feed = make_pipeline()
    t, _ = run_face(feed, 0, 2, pitch=0)
    _, out = run_face(feed, t, 4, pitch=25)
    assert out[-1].raw >= 80 - 1e-6


def test_face_lost_holds_score_and_flags_it():
    eng, scorer = DrowsinessEngine(), RiskScorer()
    t, _ = run_face(lambda p, tt: scorer.update(eng.update(p, tt, FRAME)), 0, 2, ear=0.05)
    last = scorer.update(eng.update(None, t, FRAME))
    held = scorer.update(eng.update(None, t + 1, FRAME))
    assert held.face_lost and held.score == pytest.approx(last.score)


def test_blink_rate_change_from_baseline_adds_risk():
    scorer = RiskScorer(baseline_blink_rate=15.0)
    assert scorer._blink_component(15.0) == 0.0
    assert scorer._blink_component(40.0) > 0.5
    assert RiskScorer()._blink_component(40.0) == 0.0   # no baseline, no opinion
