import math

from drowsiness.qa import SCENARIOS, Sample, summarize, to_markdown


def samples(levels, yawns=None, face=True, issues=()):
    return [Sample(t=i * 0.1, face_found=face, trusted=True, ear=0.25, brightness=120, sharpness=300,
                   glare=0.0, level=lvl, yawns=(yawns or [0] * len(levels))[i], issues=frozenset(issues))
            for i, lvl in enumerate(levels)]


def test_quiet_scenario_passes_without_warning_and_fails_with_one():
    assert summarize("Baseline", "quiet", samples([0, 1, 1, 0])).passed
    r = summarize("Baseline", "quiet", samples([0, 0, 2, 0]))
    assert not r.passed and "FALSE ALARM" in r.detail


def test_alarm_scenario_reports_latency():
    r = summarize("Eyes closed", "alarm", samples([0, 0, 0, 2, 3]))
    assert r.passed and "0.3s" in r.detail and r.max_level == "CRITICAL"
    assert not summarize("Eyes closed", "alarm", samples([0, 1, 1])).passed


def test_yawn_scenario_counts_yawns():
    assert summarize("Yawn", "yawn", samples([0, 0, 0], yawns=[0, 1, 1])).passed
    assert not summarize("Yawn", "yawn", samples([0, 0])).passed


def test_empty_and_missing_values_are_handled():
    r = summarize("x", "quiet", [])
    assert not r.passed and r.frames == 0
    nan_samples = [Sample(0, False, True, math.nan, math.nan, math.nan, math.nan, 0, 0)]
    assert summarize("x", "quiet", nan_samples).detect_rate == 0


def test_markdown_report_lists_every_scenario():
    results = [summarize(n, e, samples([0, 0])) for n, _, _, e in SCENARIOS[:2]]
    md = to_markdown(results, "glasses, daylight", "2026-10-04 10:00")
    assert "glasses, daylight" in md and "Baseline" in md and "Talking" in md and "/2 scenarios passed" in md
