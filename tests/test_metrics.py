import pytest

from drowsiness.metrics import BlinkCounter, PerclosTracker, SustainedEventDetector

DT = 0.1


def run(tracker, seconds, closed, t0=0.0):
    t = t0
    end = t0 + seconds
    while t < end - 1e-9:
        tracker.update(t, closed)
        t += DT
    return t


def test_perclos_half_closed():
    tr = PerclosTracker(window_s=60, max_gap_s=0.5)
    t = run(tr, 15, True)
    run(tr, 15, False, t0=t)
    assert tr.value == pytest.approx(0.5, abs=0.05)


def test_perclos_forgets_old_data():
    tr = PerclosTracker(window_s=10, max_gap_s=0.5)
    t = run(tr, 5, True)
    run(tr, 20, False, t0=t)
    assert tr.value == pytest.approx(0.0, abs=1e-9)


def test_perclos_ignores_long_gaps():
    tr = PerclosTracker(window_s=60, max_gap_s=0.5)
    tr.update(0.0, True)
    tr.update(10.0, False)  # 10 s gap should not count as 10 s closed
    tr.update(10.1, False)
    assert tr.value == pytest.approx(0.0, abs=1e-9)


def make_blinker():
    return BlinkCounter(min_blink_s=0.05, max_blink_s=0.5, window_s=60, min_span_s=10)


def test_short_closure_is_a_blink():
    b = make_blinker()
    b.update(0.0, False)
    b.update(1.0, True)
    b.update(1.1, True)
    assert b.update(1.2, False) is True


def test_long_closure_is_not_a_blink():
    b = make_blinker()
    b.update(0.0, False)
    b.update(1.0, True)
    assert b.closure_duration_s(2.0) == pytest.approx(1.0)
    assert b.update(2.0, False) is False


def test_blink_rate_about_20_per_minute():
    b = make_blinker()
    t = 0.0
    while t < 60:
        closed = (t % 3.0) < 0.2
        b.update(t, closed)
        t += 0.05
    assert b.rate_per_min(60.0) == pytest.approx(20, abs=2)


def test_blink_rate_zero_until_enough_data():
    b = make_blinker()
    b.update(0.0, False)
    b.update(0.2, True)
    b.update(0.4, False)
    assert b.rate_per_min(0.5) == 0.0


def test_interrupt_drops_in_progress_closure():
    b = make_blinker()
    b.update(0.0, True)
    b.interrupt()
    assert b.update(0.2, False) is False


def test_sustained_fires_once_after_min_duration():
    d = SustainedEventDetector(min_s=1.5, window_s=300)
    fired = [d.update(i * DT, True) for i in range(40)]  # 4 s of mouth open
    assert fired.count(True) == 1
    assert d.is_sustained and d.count(4.0) == 1


def test_sustained_short_activation_does_not_fire():
    d = SustainedEventDetector(min_s=1.5, window_s=300)
    for i in range(10):
        assert d.update(i * DT, True) is False
    d.update(1.0, False)
    assert d.count(1.0) == 0 and not d.is_sustained


def test_sustained_counts_separate_events_and_expires():
    d = SustainedEventDetector(min_s=1.0, window_s=100)
    t = 0.0
    for _ in range(2):
        for _ in range(15):
            d.update(t, True)
            t += DT
        for _ in range(5):
            d.update(t, False)
            t += DT
    assert d.count(t) == 2
    assert d.count(t + 200) == 0
